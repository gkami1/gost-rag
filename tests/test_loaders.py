"""Тесты загрузчиков на синтетическом PDF: порядок чтения, таблицы, пункты."""

from __future__ import annotations

from pathlib import Path

import pytest

from gost_rag.ingest.loaders import blocks_from_text, load_document, load_pdf, table_to_markdown

fitz = pytest.importorskip("fitz")


# --------------------------------------------------------------------------- #
# table_to_markdown
# --------------------------------------------------------------------------- #


def test_table_to_markdown_basic():
    md = table_to_markdown([["Толщина", "Радиус"], ["3", "6"]])
    assert md.split("\n") == [
        "| Толщина | Радиус |",
        "| --- | --- |",
        "| 3 | 6 |",
    ]


def test_table_to_markdown_handles_none_and_ragged_rows():
    md = table_to_markdown([["a", None, "c"], ["1"], [None, None, None]])
    lines = md.split("\n")
    assert lines[0] == "| a |  | c |"
    # Пустая строка выброшена, короткая дополнена до ширины таблицы.
    assert lines[-1] == "| 1 |  |  |"


def test_table_to_markdown_flattens_newlines_in_cells():
    md = table_to_markdown([["шерохова\nтость"], ["Ra 3,2"]])
    assert "шерохова тость" in md
    assert len(md.split("\n")) == 3


def test_empty_table_returns_empty_string():
    assert table_to_markdown([]) == ""
    assert table_to_markdown([[None, None]]) == ""


# --------------------------------------------------------------------------- #
# blocks_from_text
# --------------------------------------------------------------------------- #


def test_blocks_split_on_blank_lines():
    blocks, _ = blocks_from_text("первый абзац\n\nвторой абзац", 1)
    assert [b.text for b in blocks] == ["первый абзац", "второй абзац"]


def test_clause_number_starts_new_block_and_sets_section():
    text = "вводный текст\n3.2 Радиус гибки\nсоставляет 2S\n3.3 Следующий пункт"
    blocks, section = blocks_from_text(text, 1)
    assert [b.section for b in blocks] == [None, "3.2", "3.3"]
    assert section == "3.3"


def test_section_carries_into_next_page():
    _, section = blocks_from_text("4.1 Первый пункт", 1)
    blocks, _ = blocks_from_text("продолжение того же пункта", 2, section)
    assert blocks[0].section == "4.1"


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def _make_pdf(path: Path) -> Path:
    """Собрать PDF: заголовок, пункт, обведённая таблица, текст под таблицей."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 80), "ГОСТ 14634-93", fontname="helv", fontsize=12)
    page.insert_text((72, 110), "3.2 Radius gibki", fontname="helv", fontsize=11)
    page.insert_text((72, 130), "text nad tablicey", fontname="helv", fontsize=11)

    # Таблица 3x2 с настоящими линиями — иначе pdfplumber её не найдёт.
    top, left, row_h, col_w = 200, 72, 26, 120
    for r in range(4):
        y = top + r * row_h
        page.draw_line(fitz.Point(left, y), fitz.Point(left + 2 * col_w, y))
    for c in range(3):
        x = left + c * col_w
        page.draw_line(fitz.Point(x, top), fitz.Point(x, top + 3 * row_h))

    cells = [["Tolshina", "Radius"], ["3", "6"], ["4", "8"]]
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            page.insert_text(
                (left + c * col_w + 6, top + r * row_h + 18), value, fontname="helv", fontsize=10
            )

    page.insert_text((72, 320), "text pod tablicey", fontname="helv", fontsize=11)

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path / "ГОСТ 14634-93.pdf")


def test_pdf_loads_single_page(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert len(pages) == 1
    assert pages[0].page_no == 1


def test_pdf_table_becomes_markdown_block(sample_pdf):
    blocks = load_pdf(sample_pdf)[0].blocks
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    assert "| Tolshina | Radius |" in tables[0].text
    assert "| 3 | 6 |" in tables[0].text


def test_reading_order_is_preserved_around_table(sample_pdf):
    blocks = load_pdf(sample_pdf)[0].blocks
    kinds = [b.kind for b in blocks]
    table_at = kinds.index("table")
    before = " ".join(b.text for b in blocks[:table_at])
    after = " ".join(b.text for b in blocks[table_at + 1 :])
    assert "nad tablicey" in before
    assert "pod tablicey" in after


def test_table_cells_are_not_duplicated_as_loose_text(sample_pdf):
    """Ячейки должны попасть в чанк один раз — таблицей, а не россыпью чисел."""
    blocks = load_pdf(sample_pdf)[0].blocks
    text_only = " ".join(b.text for b in blocks if b.kind != "table")
    assert "Tolshina" not in text_only


def test_clause_section_detected_in_pdf(sample_pdf):
    blocks = load_pdf(sample_pdf)[0].blocks
    assert any(b.section == "3.2" for b in blocks)


def test_page_without_text_layer_is_flagged_for_ocr(tmp_path):
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # пустая страница = скан без текста
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    doc.close()

    pages = load_pdf(path, ocr_min_chars=100)
    assert pages[0].text == ""
    assert pages[0].blocks == []


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("текст", encoding="utf-8")
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        load_document(path)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def test_docx_paragraphs_and_tables_keep_document_order(tmp_path):
    docx = pytest.importorskip("docx")

    path = tmp_path / "руководство.docx"
    document = docx.Document()
    document.add_paragraph("3.1 Общие требования")
    document.add_paragraph("текст перед таблицей")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Толщина"
    table.cell(0, 1).text = "Радиус"
    table.cell(1, 0).text = "3"
    table.cell(1, 1).text = "6"
    document.add_paragraph("текст после таблицы")
    document.save(str(path))

    blocks = load_document(path)[0].blocks
    kinds = [b.kind for b in blocks]
    table_at = kinds.index("table")
    assert "перед таблицей" in " ".join(b.text for b in blocks[:table_at])
    assert "после таблицы" in " ".join(b.text for b in blocks[table_at + 1 :])
    assert "| Толщина | Радиус |" in blocks[table_at].text
    assert blocks[0].section == "3.1"
