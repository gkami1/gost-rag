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


def test_table_belongs_to_clause_that_references_it():
    """ГОСТ 24705-2004: табл. 1 вводит п. 4.1, а напечатана после п. 4.2."""
    from gost_rag.ingest.loaders import assign_sections
    from gost_rag.models import Block

    blocks = [
        Block(text="4.1 Номинальные значения диаметров приведены в таблице 1.", page_no=5),
        Block(text="4.2 Значения вычисляют по формулам.", page_no=5),
        Block(text="Таблица 1\n| d | P |\n| --- | --- |\n| 10 | 1,5 |", page_no=6, kind="table"),
        Block(
            text="Продолжение таблицы 1\n| d | P |\n| --- | --- |\n| 12 | 1,75 |",
            page_no=7,
            kind="table",
        ),
        Block(text="Таблица 2\n| a | b |\n| --- | --- |\n| 1 | 2 |", page_no=8, kind="table"),
    ]
    assign_sections(blocks)
    assert [b.section for b in blocks] == ["4.1", "4.2", "4.1", "4.1", "4.2"]


def test_text_only_vector_table_stays_a_table(tmp_path):
    """Отбраковка «сеток без данных» — только для линовки со скана."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Obychnyy tekst stranitsy dlya obyoma " * 4, fontsize=10)
    top, left, row_h, col_w = 200, 72, 26, 160
    for r in range(3):
        page.draw_line(
            fitz.Point(left, top + r * row_h), fitz.Point(left + 2 * col_w, top + r * row_h)
        )
    for c in range(3):
        page.draw_line(
            fitz.Point(left + c * col_w, top), fitz.Point(left + c * col_w, top + 2 * row_h)
        )
    for r, row in enumerate([["Strana", "Organ"], ["Armeniya", "Armstandart"]]):
        for c, value in enumerate(row):
            page.insert_text((left + c * col_w + 6, top + r * row_h + 18), value, fontsize=10)
    path = tmp_path / "text_table.pdf"
    doc.save(str(path))
    doc.close()

    tables = [b for b in load_pdf(path)[0].blocks if b.kind == "table"]
    assert len(tables) == 1
    assert "| Armeniya | Armstandart |" in tables[0].text


def _tbl(caption: str, page: int):
    from gost_rag.models import Block

    body = "| f | R |\n| --- | --- |\n| 4,0 | 1,0 |"
    return Block(text=f"{caption}\n{body}", page_no=page, kind="table")


def test_numbered_table_notes_are_not_clauses():
    """ГОСТ 10549-80: примечания 3 и 4 под табл. 1 забирали номер у настоящего
    п. 3, и табл. 2 про внутреннюю резьбу подписывалась «п. 4»."""
    from gost_rag.ingest.loaders import assign_sections
    from gost_rag.models import Block

    blocks = [
        Block(text="2. Размеры для наружной резьбы — в табл. 1.", page_no=2),
        _tbl("Таблица 1", 2),
        Block(text="Примечания:", page_no=3),
        Block(text="1. Проточки типа 2 снижают концентрацию напряжений.", page_no=3),
        Block(text="2. Размеры проточек допускается устанавливать по шагу.", page_no=3),
        Block(text="3. Для деталей из высокопрочных материалов допускается иное.", page_no=3),
        Block(text="4. Допускается применять размеры по ГОСТ 27148.", page_no=3),
        Block(text="3. Размеры для внутренней метрической резьбы — в табл. 2.", page_no=3),
        _tbl("Т а б л и и а 2", 3),
        Block(text="* Ширина дана для диаметров 6 мм.\nП р и м с ч а н и я:", page_no=4),
        Block(text="1. Проточки типа 2 снижают концентрацию напряжений.", page_no=4),
        Block(text="4. Размеры для трубной цилиндрической резьбы — в габл. 3. 4.", page_no=4),
    ]
    assign_sections(blocks)
    assert [b.section for b in blocks] == [
        "2",
        "2",
        "2",
        "2",
        "2",
        "2",
        "2",
        "3",
        "3",
        "3",
        "3",
        "4",
    ]


def test_table_number_recovered_from_ocr_caption_and_order():
    """«Таблица I», «Таблииа4», «Таблицаб», «Таблица?» — номера 1, 4, 6, 7."""
    from gost_rag.ingest.loaders import assign_sections

    blocks = [
        _tbl("Таблица I Размеры в миллиметрах", 1),
        _tbl("Продолжение табл. 1", 2),
        _tbl("Таблица 2", 3),
        _tbl("Т а б л и и а 3", 4),
        _tbl("Таблииа4 Размеры в миллиметрах", 5),
        _tbl("Таблица 5", 6),
        _tbl("Таблицаб Размеры в миллиметрах", 7),
        _tbl("Таблица? В миллиметрах", 8),
        _tbl("Окончание таблицы 7 В миллиметрах", 9),
        _tbl("Т абли ца)", 10),
    ]
    assign_sections(blocks)
    assert [b.text.split("\n", 1)[0] for b in blocks] == [
        "Таблица 1 Размеры в миллиметрах",
        "Продолжение табл. 1",
        "Таблица 2",
        "Таблица 3",
        "Таблица 4 Размеры в миллиметрах",
        "Таблица 5",
        "Таблица 6 Размеры в миллиметрах",
        "Таблица 7 В миллиметрах",
        "Окончание табл. 7 В миллиметрах",
        "Таблица 8",
    ]


def test_continued_table_carries_sentence_that_introduces_it():
    """На странице «Продолжения табл. 2» нет слов «внутренняя метрическая резьба» —
    чанк со строкой для шага 1 не находился по вопросу о ней."""
    from gost_rag.ingest.loaders import assign_sections
    from gost_rag.models import Block

    blocks = [
        Block(
            text=(
                "3. Размеры сбегов для внутренней метрической резьбы — на черт. 7 и в табл. 2.\n"
                "Форма и размеры проточек для внутренней метрической резьбы — на черт. 8 и в\n"
                "табл. 2. Шаг выбирают по ГОСТ 8724."
            ),
            page_no=3,
        ),
        _tbl("Таблица 2 В миллиметрах", 3),
        _tbl("Продолжение табл. 2 В миллиметрах", 4),
    ]
    assign_sections(blocks)
    for table in blocks[1:]:
        _caption, context, header, *_ = table.text.split("\n")
        assert context.startswith("К п. 3: Размеры сбегов для внутренней метрической резьбы")
        assert "Форма и размеры проточек" in context
        # Предложение без ссылки на таблицу в контекст не попадает.
        assert "ГОСТ 8724" not in context
        assert header == "| f | R |"


def test_table_reference_list_and_range_cover_each_table():
    from gost_rag.ingest.loaders import _ref_numbers

    assert _ref_numbers("3, 4") == [3, 4]
    assert _ref_numbers("3. 4") == [3, 4]
    assert _ref_numbers("3—7") == [3, 4, 5, 6, 7]
    assert _ref_numbers("2") == [2]


def test_split_table_repeats_caption_and_context_lines():
    """Строка контекста — часть заголовка: при разрезании таблицы она повторяется."""
    from gost_rag.ingest.chunk import ApproxTokenizer, prepare_blocks
    from gost_rag.models import Block

    rows = "\n".join(f"| {i} | {i},5 |" for i in range(60))
    block = Block(
        text=f"Таблица 2 В мм\nК п. 3: для внутренней резьбы.\n| P | R |\n| --- | --- |\n{rows}",
        page_no=3,
        kind="table",
    )
    parts = prepare_blocks([block], ApproxTokenizer(), max_tokens=60)
    assert len(parts) > 1
    for part in parts:
        assert part.text.startswith("Таблица 2 В мм\nК п. 3: для внутренней резьбы.\n| P | R |")
