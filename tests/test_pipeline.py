"""Сквозной тест индексации: PDF -> страницы -> чанки -> Qdrant.

Эмбеддер подставной и детерминированный: проверяется склейка конвейера, а не
качество векторов, поэтому тест не тянет из сети 2 ГБ весов.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gost_rag.config import Settings
from gost_rag.ingest.chunk import ApproxTokenizer
from gost_rag.ingest.embed import Embedding
from gost_rag.ingest.index import (
    count_points,
    ensure_collection,
    get_client,
    list_documents,
    prune_missing_sources,
)
from gost_rag.ingest.pipeline import (
    append_ledger,
    discover_files,
    index_fingerprint,
    ingest_file,
    read_ledger,
)
from gost_rag.retrieval.store import hybrid_search

fitz = pytest.importorskip("fitz")

DIM = 8


class FakeEmbedder:
    """Вектор выводится из хэша текста: одинаковый текст -> одинаковый вектор."""

    def encode(self, texts: list[str], **kwargs) -> list[Embedding]:
        return [self._one(text) for text in texts]

    def encode_one(self, text: str) -> Embedding:
        return self._one(text)

    @staticmethod
    def _one(text: str) -> Embedding:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        dense = [digest[i] / 255.0 for i in range(DIM)]
        sparse = {int(digest[i]): 1.0 for i in range(DIM, DIM + 4)}
        return Embedding(dense=dense, sparse=sparse)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_path=tmp_path / "qdrant",
        collection_name="pipeline_test",
        dense_vector_size=DIM,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        chunk_tokens=120,
        chunk_overlap_tokens=12,
        min_chunk_tokens=5,
        ocr_enabled=False,
    )


#: Колонтитулы вырезаются только на документах от 4 страниц — на более коротких
#: повтор строки слишком часто оказывается совпадением. Фикстура должна быть
#: длиннее этого порога, иначе тест проверял бы отключённую логику.
PAGES = 6


@pytest.fixture
def corpus(settings: Settings) -> Path:
    """Многостраничный PDF с колонтитулами, пунктами и таблицей."""
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = settings.raw_dir / "ГОСТ 14634-93.pdf"

    doc = fitz.open()
    for page_no in range(1, PAGES + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 50), "GOST 14634-93", fontname="helv", fontsize=9)
        page.insert_text(
            (72, 110), f"3.{page_no} Razdel nomer {page_no}", fontname="helv", fontsize=11
        )
        for line in range(12):
            page.insert_text(
                (72, 140 + line * 16),
                f"stranica {page_no} stroka {line} tekst pro radius gibki listovoy stali",
                fontname="helv",
                fontsize=10,
            )
        page.insert_text((72, 800), f"Str. {page_no}", fontname="helv", fontsize=9)

    # Таблица на последней странице.
    page = doc[PAGES - 1]
    top, left, row_h, col_w = 380, 72, 24, 130
    for r in range(3):
        y = top + r * row_h
        page.draw_line(fitz.Point(left, y), fitz.Point(left + 2 * col_w, y))
    for c in range(3):
        x = left + c * col_w
        page.draw_line(fitz.Point(x, top), fitz.Point(x, top + 2 * row_h))
    for r, row in enumerate([["Tolshina", "Radius"], ["3", "6"]]):
        for c, value in enumerate(row):
            page.insert_text(
                (left + c * col_w + 5, top + r * row_h + 16), value, fontname="helv", fontsize=10
            )

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def client(settings: Settings):
    client = get_client(settings)
    ensure_collection(client, settings)
    yield client
    client.close()


def _ingest(path: Path, client, settings: Settings, registry=None):
    return ingest_file(
        path,
        client=client,
        embedder=FakeEmbedder(),
        tokenizer=ApproxTokenizer(),
        registry=registry or {},
        settings=settings,
    )


# --------------------------------------------------------------------------- #


def test_discover_files_finds_pdfs_recursively(settings, corpus):
    nested = settings.raw_dir / "sub"
    nested.mkdir()
    (nested / "note.txt").write_text("не документ", encoding="utf-8")
    assert discover_files(settings.raw_dir) == [corpus]


def test_ingest_indexes_chunks(corpus, client, settings):
    report = _ingest(corpus, client, settings)
    assert report.error is None
    assert report.pages == PAGES
    assert report.chunks > 0
    assert report.indexed == report.chunks
    assert count_points(client, settings) == report.chunks


def test_designation_extracted_from_filename(corpus, client, settings):
    report = _ingest(corpus, client, settings)
    assert report.designation == "ГОСТ 14634-93"
    assert report.doc_id == "гост-14634-93"


def test_registry_status_reaches_payload(corpus, client, settings):
    registry = {
        "ГОСТ 14634-93": {
            "designation": "ГОСТ 14634-93",
            "title": "Ленты стальные",
            "year": "1993",
            "status": "заменён",
            "source_url": "https://example.org/x",
            "replaced_by": "ГОСТ 14634-2020",
        }
    }
    _ingest(corpus, client, settings, registry)
    hits = hybrid_search(client, FakeEmbedder().encode_one("радиус"), settings)
    assert hits
    assert hits[0].payload["status"] == "заменён"
    assert hits[0].payload["replaced_by"] == "ГОСТ 14634-2020"
    assert hits[0].citation_label().startswith("ГОСТ 14634-93 (заменён)")


def test_running_header_is_not_indexed(corpus, client, settings):
    """Колонтитул «GOST 14634-93 / Str. N» не должен занимать место в чанках."""
    _ingest(corpus, client, settings)
    points, _ = client.scroll(settings.collection_name, limit=100, with_payload=True)
    body = "\n".join((p.payload or {}).get("text", "") for p in points)
    for page_no in range(1, PAGES + 1):
        assert f"Str. {page_no}" not in body
    # Содержательный текст при этом обязан уцелеть.
    assert "radius gibki" in body


def test_table_survives_into_chunk(corpus, client, settings):
    _ingest(corpus, client, settings)
    points, _ = client.scroll(settings.collection_name, limit=100, with_payload=True)
    payloads = [p.payload or {} for p in points]
    assert any(p.get("contains_table") for p in payloads)
    assert any("| Tolshina | Radius |" in p.get("text", "") for p in payloads)


def test_section_numbers_recorded(corpus, client, settings):
    _ingest(corpus, client, settings)
    points, _ = client.scroll(settings.collection_name, limit=100, with_payload=True)
    sections = {(p.payload or {}).get("section") for p in points}
    assert sections & {f"3.{n}" for n in range(1, PAGES + 1)}


def test_pages_are_tracked_for_citations(corpus, client, settings):
    _ingest(corpus, client, settings)
    points, _ = client.scroll(settings.collection_name, limit=100, with_payload=True)
    pages = {(p.payload or {}).get("page_start") for p in points}
    assert pages <= set(range(1, PAGES + 1))
    assert len(pages) > 1


def test_reingest_does_not_duplicate(corpus, client, settings):
    first = _ingest(corpus, client, settings)
    _ingest(corpus, client, settings)
    assert count_points(client, settings) == first.chunks


def test_edited_document_replaces_old_chunks(corpus, client, settings):
    _ingest(corpus, client, settings)

    doc = fitz.open(str(corpus))
    doc[0].insert_text((72, 600), "novyy abzac pro dopuski", fontname="helv", fontsize=10)
    doc.saveIncr()
    doc.close()

    report = _ingest(corpus, client, settings)
    # Старые чанки удалены, а не оставлены рядом с новыми.
    assert count_points(client, settings) == report.chunks


def test_ledger_roundtrip(settings, corpus, client):
    report = _ingest(corpus, client, settings)
    append_ledger(settings, report, "sha-123", "fp-1")
    assert read_ledger(settings) == {str(corpus): {"sha256": "sha-123", "fingerprint": "fp-1"}}


def test_ledger_ignores_failed_entries(settings, corpus, client):
    report = _ingest(corpus, client, settings)
    report.error = "сломалось"
    append_ledger(settings, report, "sha-bad")
    assert read_ledger(settings) == {}


def test_document_without_text_reports_error(settings, client, tmp_path):
    doc = fitz.open()
    doc.new_page()
    empty = settings.raw_dir / "ГОСТ 1-11.pdf"
    empty.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(empty))
    doc.close()

    report = _ingest(empty, client, settings)
    assert report.chunks == 0
    assert report.error is not None
    assert "OCR" in report.error


# --------------------------------------------------------------------------- #
# Журнал и целостность индекса
# --------------------------------------------------------------------------- #


def test_fingerprint_changes_with_pipeline_settings(settings):
    """Смена размера чанка должна переиндексировать файл, даже если он не менялся."""
    base = index_fingerprint(settings)
    assert index_fingerprint(settings) == base
    assert index_fingerprint(settings.model_copy(update={"chunk_tokens": 400})) != base
    assert index_fingerprint(settings, approx_tokens=True) != base


def test_legacy_ledger_entry_has_no_fingerprint(settings, corpus, client):
    report = _ingest(corpus, client, settings)
    append_ledger(settings, report, "sha-old")
    assert read_ledger(settings)[str(corpus)]["fingerprint"] == ""


def test_deleted_file_is_pruned_from_index(corpus, client, settings):
    _ingest(corpus, client, settings)
    assert count_points(client, settings) > 0
    corpus.unlink()

    removed = prune_missing_sources(client, settings.raw_dir, settings)
    assert removed == ["гост-14634-93"]
    assert count_points(client, settings) == 0


def test_prune_leaves_files_outside_scanned_root_alone(corpus, client, settings, tmp_path):
    _ingest(corpus, client, settings)
    corpus.unlink()
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    assert prune_missing_sources(client, elsewhere, settings) == []
    assert count_points(client, settings) > 0


def test_second_file_with_same_designation_does_not_wipe_first(corpus, client, settings):
    first = _ingest(corpus, client, settings)
    duplicate = corpus.with_name("ГОСТ 14634-93 копия.pdf")
    duplicate.write_bytes(corpus.read_bytes())

    report = _ingest(duplicate, client, settings)
    assert report.error is not None
    assert "уже проиндексирован" in report.error
    # Чанки первого файла на месте.
    assert count_points(client, settings) == first.chunks
    [doc] = list_documents(client, settings)
    assert doc["source_path"] == str(corpus)
