"""Интеграционные тесты индекса и гибридного поиска на встроенном Qdrant.

Эмбеддинги синтетические — проверяется схема коллекции, RRF-слияние и фильтры,
а не качество модели. Поэтому тесты быстрые и не тянут веса из сети.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gost_rag.config import Settings
from gost_rag.ingest.embed import Embedding
from gost_rag.ingest.index import (
    build_points,
    count_points,
    delete_document,
    ensure_collection,
    existing_point_ids,
    get_client,
    list_documents,
    upsert_points,
)
from gost_rag.models import Chunk, DocumentMeta
from gost_rag.retrieval.filters import build_filter, extract_designations, known_designations
from gost_rag.retrieval.store import hybrid_search

DIM = 4


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_path=tmp_path / "qdrant",
        collection_name="test_chunks",
        dense_vector_size=DIM,
        prefetch_limit=10,
        fusion_limit=10,
    )


@pytest.fixture
def client(settings: Settings):
    client = get_client(settings)
    ensure_collection(client, settings)
    yield client
    client.close()


def _meta(designation: str, doc_id: str, status: str = "действующий") -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        designation=designation,
        title=f"Стандарт {designation}",
        year=1993,
        status=status,  # type: ignore[arg-type]
        doc_type="ГОСТ",
        source_path=f"/data/{doc_id}.pdf",
    )


def _chunk(doc_id: str, index: int, text: str) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_index=index,
        text=text,
        page_start=index + 1,
        page_end=index + 1,
        token_count=len(text.split()),
        section=f"3.{index}",
    )


def _embedding(dense: list[float], sparse: dict[int, float]) -> Embedding:
    return Embedding(dense=dense, sparse=sparse)


def _seed(client, settings) -> None:
    """Два документа: один «плотно» похож на запрос, другой — «лексически»."""
    meta_a = _meta("ГОСТ 14634-93", "gost-14634-93")
    meta_b = _meta("ГОСТ 1050-88", "gost-1050-88", status="заменён")

    points = build_points(
        [
            _chunk("gost-14634-93", 0, "радиус гибки листовой стали"),
            _chunk("gost-14634-93", 1, "толщина проката"),
        ],
        [
            _embedding([1.0, 0.0, 0.0, 0.0], {10: 0.9, 11: 0.5}),
            _embedding([0.0, 1.0, 0.0, 0.0], {20: 0.7}),
        ],
        meta_a,
        settings,
    )
    points += build_points(
        [_chunk("gost-1050-88", 0, "сортовой прокат из стали")],
        [_embedding([0.0, 0.0, 1.0, 0.0], {10: 0.95, 12: 0.4})],
        meta_b,
        settings,
    )
    upsert_points(client, points, settings)


# --------------------------------------------------------------------------- #
# Индекс
# --------------------------------------------------------------------------- #


def test_collection_created_with_both_vector_kinds(client, settings):
    info = client.get_collection(settings.collection_name)
    assert settings.dense_vector_name in info.config.params.vectors
    assert settings.sparse_vector_name in info.config.params.sparse_vectors


def test_upsert_and_count(client, settings):
    _seed(client, settings)
    assert count_points(client, settings) == 3


def test_reingest_is_idempotent(client, settings):
    _seed(client, settings)
    _seed(client, settings)
    assert count_points(client, settings) == 3


def test_existing_point_ids_detects_known_chunks(client, settings):
    _seed(client, settings)
    chunk = _chunk("gost-14634-93", 0, "радиус гибки листовой стали")
    assert existing_point_ids(client, [chunk.point_id], settings) == {chunk.point_id}


def test_changed_text_produces_new_point(client, settings):
    _seed(client, settings)
    changed = _chunk("gost-14634-93", 0, "радиус гибки листовой стали (ред. 2)")
    assert existing_point_ids(client, [changed.point_id], settings) == set()


def test_build_points_rejects_length_mismatch(settings):
    with pytest.raises(ValueError, match="совпадать"):
        build_points([_chunk("d", 0, "т")], [], _meta("ГОСТ 1-11", "d"), settings)


def test_delete_document_removes_only_its_chunks(client, settings):
    _seed(client, settings)
    delete_document(client, "gost-14634-93", settings)
    assert count_points(client, settings) == 1


def test_list_documents_summarises_corpus(client, settings):
    _seed(client, settings)
    docs = list_documents(client, settings)
    assert [d["designation"] for d in docs] == ["ГОСТ 1050-88", "ГОСТ 14634-93"]
    assert {d["chunks"] for d in docs} == {1, 2}
    assert docs[0]["status"] == "заменён"


# --------------------------------------------------------------------------- #
# Гибридный поиск
# --------------------------------------------------------------------------- #


def test_hybrid_search_returns_results_with_payload(client, settings):
    _seed(client, settings)
    hits = hybrid_search(client, _embedding([1.0, 0.0, 0.0, 0.0], {10: 1.0}), settings)
    assert hits
    assert hits[0].payload["designation"]
    assert hits[0].fusion_score is not None
    assert hits[0].text


def test_rrf_merges_dense_and_sparse_winners(client, settings):
    """Плотный лидер и лексический лидер — разные документы; RRF обязан вернуть оба."""
    _seed(client, settings)
    query = _embedding([0.0, 1.0, 0.0, 0.0], {10: 1.0})  # плотно ~ чанк 2, лексически ~ чанк 1 и 3
    hits = hybrid_search(client, query, settings)
    doc_ids = {hit.payload["doc_id"] for hit in hits}
    assert doc_ids == {"gost-14634-93", "gost-1050-88"}


def test_search_on_missing_collection_returns_empty(settings, tmp_path):
    other = Settings(
        qdrant_path=tmp_path / "empty", collection_name="absent", dense_vector_size=DIM
    )
    client = get_client(other)
    try:
        assert hybrid_search(client, _embedding([1.0, 0, 0, 0], {1: 1.0}), other) == []
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# Фильтр по обозначению
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("какой радиус гибки по ГОСТ 14634", ["ГОСТ 14634"]),
        ("по ГОСТ 14634-93", ["ГОСТ 14634-93"]),
        ("см. ГОСТ Р 1.2 и ГОСТ 1050", ["ГОСТ Р 1.2", "ГОСТ 1050"]),
        ("какой радиус гибки листовой стали", []),
    ],
)
def test_extract_designations(query, expected):
    assert extract_designations(query) == expected


def test_filter_matches_edition_when_year_omitted(client, settings):
    _seed(client, settings)
    available = known_designations(client, settings)
    query_filter = build_filter("радиус гибки по ГОСТ 14634", available)
    assert query_filter is not None

    hits = hybrid_search(
        client, _embedding([0.0, 0.0, 1.0, 0.0], {10: 1.0}), settings, query_filter=query_filter
    )
    assert hits
    assert {hit.payload["doc_id"] for hit in hits} == {"gost-14634-93"}


def test_filter_is_skipped_when_designation_absent_from_corpus(client, settings):
    """Фильтр по отсутствующему документу обнулил бы выдачу — его быть не должно."""
    _seed(client, settings)
    available = known_designations(client, settings)
    assert build_filter("что там в ГОСТ 99999-11", available) is None


def test_no_filter_without_designation(client, settings):
    _seed(client, settings)
    available = known_designations(client, settings)
    assert build_filter("радиус гибки листовой стали", available) is None


def test_known_designations_lists_corpus(client, settings):
    _seed(client, settings)
    assert known_designations(client, settings) == {"ГОСТ 14634-93", "ГОСТ 1050-88"}
