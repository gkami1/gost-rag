"""Тесты чанкинга: границы, перекрытие, неделимость таблиц."""

from __future__ import annotations

import pytest

from gost_rag.ingest.chunk import ApproxTokenizer, chunk_blocks, prepare_blocks
from gost_rag.models import Block


class WordTokenizer:
    """1 слово = 1 токен — делает арифметику окон наглядной в тестах."""

    def count(self, text: str) -> int:
        return max(1, len(text.split()))

    def split_text(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)] or [
            text
        ]


@pytest.fixture
def tok() -> WordTokenizer:
    return WordTokenizer()


def _para(n_words: int, page: int = 1, marker: str = "сл") -> Block:
    return Block(text=" ".join(f"{marker}{i}" for i in range(n_words)), page_no=page)


def test_empty_input_gives_no_chunks(tok):
    assert chunk_blocks([], "doc", tok) == []
    assert chunk_blocks([Block(text="   ", page_no=1)], "doc", tok) == []


def test_single_small_block_is_one_chunk(tok):
    chunks = chunk_blocks([_para(10)], "doc", tok, chunk_tokens=100, overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_start == chunks[0].page_end == 1


def test_no_chunk_exceeds_budget(tok):
    blocks = [_para(30, page=i + 1, marker=f"p{i}") for i in range(20)]
    chunks = chunk_blocks(blocks, "doc", tok, chunk_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert tok.count(chunk.text) <= 100


def test_consecutive_chunks_overlap(tok):
    blocks = [_para(20, page=1, marker=f"b{i}") for i in range(10)]
    chunks = chunk_blocks(
        blocks, "doc", tok, chunk_tokens=100, overlap_tokens=25, min_chunk_tokens=1
    )
    assert len(chunks) >= 2
    # Хвост предыдущего чанка должен встречаться в начале следующего.
    first_tail = chunks[0].text.split("\n\n")[-1]
    assert first_tail in chunks[1].text


def test_overlap_must_be_smaller_than_window(tok):
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_blocks([_para(10)], "doc", tok, chunk_tokens=100, overlap_tokens=100)


def test_chunk_indices_are_sequential(tok):
    blocks = [_para(30, page=1, marker=f"x{i}") for i in range(15)]
    chunks = chunk_blocks(blocks, "doc", tok, chunk_tokens=100, overlap_tokens=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_pages_span_is_tracked(tok):
    blocks = [_para(30, page=p, marker=f"p{p}") for p in (4, 5, 6)]
    chunks = chunk_blocks(blocks, "doc", tok, chunk_tokens=200, overlap_tokens=10)
    assert chunks[0].page_start == 4
    assert chunks[0].page_end == 6


def test_section_is_carried_from_first_block(tok):
    blocks = [
        Block(text="текст пункта три два", page_no=2, section="3.2"),
        Block(text="ещё текст того же пункта", page_no=2),
    ]
    chunks = chunk_blocks(blocks, "doc", tok, chunk_tokens=100, overlap_tokens=10)
    assert chunks[0].section == "3.2"


def _clause(section: str, n_words: int = 20) -> Block:
    return Block(
        text=" ".join(f"п{section}_{i}" for i in range(n_words)), page_no=1, section=section
    )


def test_section_ignores_overlap_tail(tok):
    """Регрессия: чанк подписывался пунктом из хвоста перекрытия.

    Хвост предыдущего чанка (п. 3.2) стоит в начале следующего, но собственное
    содержимое следующего чанка — пп. 3.3 и 3.4. Подпись «п. 3.2» указала бы
    инженеру не туда.
    """
    blocks = [_clause("3.1", 60), _clause("3.2", 20), _clause("3.3", 40), _clause("3.4", 30)]
    chunks = chunk_blocks(
        blocks, "doc", tok, chunk_tokens=100, overlap_tokens=25, min_chunk_tokens=1
    )
    assert len(chunks) == 2
    second = chunks[1]
    assert second.text.startswith("п3.2_0")  # перекрытие действительно есть
    assert second.section == "3.3"
    assert second.sections == ["3.2", "3.3", "3.4"]


def test_sections_list_every_clause_in_order_without_repeats(tok):
    blocks = [
        _clause("3.1", 10),
        Block(text="продолжение", page_no=1, section="3.1"),
        _clause("3.2"),
    ]
    chunks = chunk_blocks(blocks, "doc", tok, chunk_tokens=100, overlap_tokens=10)
    assert chunks[0].sections == ["3.1", "3.2"]
    assert chunks[0].section == "3.1"


def test_short_tail_clauses_are_added_to_previous_chunk(tok):
    blocks = [_clause("3.1", 60), _clause("3.2", 38), _clause("3.3", 5)]
    chunks = chunk_blocks(
        blocks, "doc", tok, chunk_tokens=100, overlap_tokens=0, min_chunk_tokens=10
    )
    assert len(chunks) == 1
    assert chunks[0].sections == ["3.1", "3.2", "3.3"]


def test_sections_reach_payload(tok):
    from gost_rag.models import DocumentMeta

    chunk = chunk_blocks([_clause("3.1"), _clause("3.2")], "doc", tok, chunk_tokens=100)[0]
    meta = DocumentMeta("doc", "ГОСТ 1-11", None, None, "действующий", None, "x.pdf")
    assert chunk.as_payload(meta)["sections"] == ["3.1", "3.2"]


# ---------- таблицы ----------

TABLE = "\n".join(
    [
        "| Толщина, мм | Радиус, мм |",
        "| --- | --- |",
        *[f"| {i} | {i * 2} |" for i in range(1, 21)],
    ]
)


def test_oversized_table_splits_between_rows_and_repeats_header(tok):
    block = Block(text=TABLE, page_no=3, kind="table")
    parts = prepare_blocks([block], tok, max_tokens=30)
    assert len(parts) > 1
    for part in parts:
        assert part.kind == "table"
        assert part.text.startswith("| Толщина, мм | Радиус, мм |")
        # Разделитель markdown должен идти сразу за шапкой в каждой части.
        assert part.text.split("\n")[1].strip() == "| --- | --- |"
        # Ни одна строка не должна оказаться разрезанной посередине.
        for line in part.text.split("\n"):
            assert line.startswith("|") and line.endswith("|")


def test_table_rows_are_not_lost_when_split(tok):
    block = Block(text=TABLE, page_no=3, kind="table")
    parts = prepare_blocks([block], tok, max_tokens=30)
    rows = [ln for part in parts for ln in part.text.split("\n") if ln.startswith("| ")]
    data_rows = [r for r in rows if "Толщина" not in r and "---" not in r]
    assert len(data_rows) == 20
    assert "| 20 | 40 |" in data_rows


def test_small_table_stays_atomic(tok):
    small = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    block = Block(text=small, page_no=1, kind="table")
    parts = prepare_blocks([block], tok, max_tokens=100)
    assert len(parts) == 1
    assert parts[0].text == small


def test_chunk_flags_table_content(tok):
    blocks = [_para(10), Block(text="| a | b |\n| --- | --- |\n| 1 | 2 |", page_no=1, kind="table")]
    chunks = chunk_blocks(blocks, "doc", tok, chunk_tokens=200, overlap_tokens=10)
    assert chunks[0].contains_table is True


# ---------- OCR-метаданные ----------


def test_ocr_flag_and_confidence_propagate(tok):
    blocks = [_para(10, page=5)]
    chunks = chunk_blocks(
        blocks,
        "doc",
        tok,
        chunk_tokens=100,
        overlap_tokens=10,
        ocr_pages={5},
        ocr_confidence={5: 87.5},
    )
    assert chunks[0].from_ocr is True
    assert chunks[0].ocr_confidence == 87.5


def test_non_ocr_chunk_has_no_confidence(tok):
    chunks = chunk_blocks([_para(10, page=1)], "doc", tok, chunk_tokens=100, overlap_tokens=10)
    assert chunks[0].from_ocr is False
    assert chunks[0].ocr_confidence is None


# ---------- идентификаторы ----------


def test_point_id_is_deterministic_and_content_addressed(tok):
    a = chunk_blocks([_para(10)], "doc", tok, chunk_tokens=100, overlap_tokens=10)[0]
    b = chunk_blocks([_para(10)], "doc", tok, chunk_tokens=100, overlap_tokens=10)[0]
    assert a.point_id == b.point_id

    c = chunk_blocks([_para(11)], "doc", tok, chunk_tokens=100, overlap_tokens=10)[0]
    assert c.point_id != a.point_id


def test_approx_tokenizer_roundtrip():
    tok = ApproxTokenizer()
    text = "поверхность" * 100
    pieces = tok.split_text(text, 10)
    assert "".join(pieces) == text
    assert all(tok.count(p) <= 11 for p in pieces)
