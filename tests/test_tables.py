"""Таблицы со сканов: линовка из растра, объединённые ячейки, шапка, подпись.

Синтетический «скан» повторяет устройство реального корпуса: страница целиком —
картинка с линиями таблицы, текст лежит поверх невидимым слоем. Векторных линий
нет, и штатный ``find_tables()`` на такой странице не находит ничего.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gost_rag.ingest.loaders import _split_caption, load_pdf
from gost_rag.ingest.tables import (
    Rule,
    close_frames,
    detect_rules,
    extend_open_columns,
    flatten_header,
    is_data_row,
    is_real_table,
    split_header,
)

fitz = pytest.importorskip("fitz")
cv2 = pytest.importorskip("cv2")

#: Масштаб картинки относительно страницы: 2 px на пункт = 144 dpi.
_PX = 2


# --------------------------------------------------------------------------- #
# Чистые функции
# --------------------------------------------------------------------------- #


def test_detect_rules_finds_grid_but_not_text_strokes():
    image = np.full((400, 600), 255, np.uint8)
    for y in (50, 150, 250):
        cv2.line(image, (40, y), (560, y), 0, 2)
    for x in (40, 300, 560):
        cv2.line(image, (x, 50), (x, 250), 0, 2)
    # Короткий «штрих буквы» линией таблицы считаться не должен.
    cv2.line(image, (100, 320), (110, 320), 0, 2)

    horizontal, vertical = detect_rules(image, scale=0.5)
    assert len(horizontal) == 3
    assert len(vertical) == 3
    # Координаты пересчитаны в пункты PDF.
    assert sorted(round(r.top) for r in horizontal) == [25, 75, 125]


def test_close_frames_adds_missing_borders_of_open_table():
    """Табл. 1 ГОСТ 12.2.007.0-75: две горизонтали шапки, внутренние вертикали."""
    horizontal = [Rule(46, 448, 450, 448), Rule(46, 492, 450, 492)]
    vertical = [Rule(147, 447, 147, 573), Rule(248, 447, 248, 573)]
    h_out, v_out = close_frames(horizontal, vertical)
    assert any(abs(r.top - 573) < 1 and r.x0 <= 46 and r.x1 >= 450 for r in h_out)
    assert any(abs(r.x0 - 46) < 1 for r in v_out)
    assert any(abs(r.x0 - 450) < 1 for r in v_out)


def test_close_frames_ignores_lone_horizontal_rules():
    """Черта над «Издание официальное» — не таблица, рамку ей не строим."""
    h_out, v_out = close_frames([Rule(40, 800, 300, 800)], [])
    assert len(h_out) == 1
    assert v_out == []


def test_extend_open_columns_stops_at_rule_of_same_width():
    horizontal = [
        Rule(40, 100, 400, 100),  # верх шапки
        Rule(40, 130, 400, 130),  # низ шапки
        Rule(40, 300, 400, 300),  # низ таблицы
        Rule(40, 700, 150, 700),  # чужая короткая черта ниже
    ]
    vertical = [Rule(200, 100, 200, 130)]
    [extended] = extend_open_columns(horizontal, vertical)
    assert extended.bottom == 300


def test_extend_open_columns_does_not_reach_rule_of_other_width():
    horizontal = [Rule(40, 100, 400, 100), Rule(40, 130, 400, 130), Rule(40, 700, 150, 700)]
    vertical = [Rule(100, 100, 100, 130)]
    [extended] = extend_open_columns(horizontal, vertical)
    assert extended.bottom == 130


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (["10", "1,5", "9,026"], True),
        (["Св. 4 до 6", "М 3", "10", "7"], True),
        # Шапка ГОСТ 5264-80: диапазоны толщин — подписи, а не данные.
        (["Предел текучести", "От 3 до 4", "Св. 4 до 5", "Св. 10 до 16"], False),
        # Одна ошибка распознавания («и» вместо 11) не делает столбец текстом.
        (["М 3.5", "10 и 12 14", "7 8 9"], True),
        (["Номинальный диаметр", "Шаг", "d2"], False),
    ],
)
def test_is_data_row(row, expected):
    assert is_data_row(row) is expected


def test_split_header_and_continuation_without_header():
    grid = [["Диаметр", "Шаг"], ["Диаметр", "P"], ["10", "1,5"]]
    assert split_header(grid) == 2
    # Продолжение таблицы начинается сразу с данных.
    assert split_header([["10", "1,25"], ["12", "1,75"]]) == 0


def test_flatten_header_joins_levels_without_repeats():
    rows = [
        ["Шаг", "Проточка", "Проточка"],
        ["Шаг", "Тип 1", "Тип 1"],
        ["Шаг", "f", "R"],
    ]
    assert flatten_header(rows) == ["Шаг", "Проточка / Тип 1 / f", "Проточка / Тип 1 / R"]


def test_is_real_table_rejects_empty_drawing_frames():
    assert not is_real_table([["", ""], ["", ""]])
    assert not is_real_table([["только одна строка", "x"]])
    assert is_real_table([["Шаг", "d2"], ["1,5", "9,026"]])


def test_split_caption_takes_spaced_caption_and_units():
    band, caption = _split_caption("Текст пункта.\nТ а б л и ц а 1\nРазмеры в миллиметрах")
    assert band == "Текст пункта."
    assert caption == "Т а б л и ц а 1 Размеры в миллиметрах"


def test_split_caption_leaves_plain_text_alone():
    band, caption = _split_caption("Обычный текст\nбез подписи")
    assert caption is None
    assert band == "Обычный текст\nбез подписи"


# --------------------------------------------------------------------------- #
# Синтетический скан целиком
# --------------------------------------------------------------------------- #


def _scan_pdf(path: Path, lines: list[tuple[int, int, int, int]], texts) -> Path:
    """PDF, где страница — картинка с линиями, а текст — невидимый слой поверх."""
    width, height = 595, 842
    image = np.full((height * _PX, width * _PX), 255, np.uint8)
    for x0, y0, x1, y1 in lines:
        cv2.line(image, (x0 * _PX, y0 * _PX), (x1 * _PX, y1 * _PX), 0, 2)
    ok, png = cv2.imencode(".png", image)
    assert ok

    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=png.tobytes())
    for x, y, value in texts:
        # render_mode=3 — невидимый текст, как у слоя распознавания издателя.
        # Встроенный CJK-шрифт — единственный встроенный с кириллицей.
        page.insert_text((x, y), value, fontname="china-s", fontsize=10, render_mode=3)
    doc.save(str(path))
    doc.close()
    return path


def test_scanned_table_with_merged_cell_gives_self_contained_rows(tmp_path):
    """ГОСТ 24705-2004: номинальный диаметр «10» — одна объединённая ячейка на
    несколько шагов. Каждая строка должна нести его сама."""
    xs, ys = (72, 192, 312, 432), (200, 226, 252, 278)
    lines = [(xs[0], y, xs[-1], y) for y in (ys[0], ys[1], ys[3])]
    # Между строками тела линия есть только в столбцах 2–3: столбец 1 объединён.
    lines.append((xs[1], ys[2], xs[-1], ys[2]))
    lines += [(x, ys[0], x, ys[-1]) for x in xs]
    texts = [
        # Страница короче ocr_min_chars ушла бы в Tesseract целиком.
        (72, 120, "Размеры метрической резьбы должны соответствовать таблице 1."),
        (72, 180, "Таблица 1"),
        (78, 218, "Diametr"),
        (198, 218, "Shag"),
        (318, 218, "d2"),
        (78, 257, "10"),
        (198, 244, "1,5"),
        (318, 244, "9,026"),
        (198, 270, "1,25"),
        (318, 270, "9,188"),
    ]
    pages = load_pdf(_scan_pdf(tmp_path / "scan.pdf", lines, texts))

    tables = [b for b in pages[0].blocks if b.kind == "table"]
    assert len(tables) == 1
    text = tables[0].text
    assert text.startswith("Таблица 1\n")
    assert "| Diametr | Shag | d2 |" in text
    assert "| 10 | 1,5 | 9,026 |" in text
    assert "| 10 | 1,25 | 9,188 |" in text
    # Текстовый слой скана — чужое распознавание, ему нельзя доверять как набору.
    assert pages[0].from_ocr


def test_open_table_body_is_split_into_rows(tmp_path):
    """ГОСТ 12.2.007.0-75, табл. 1: нет ни боковых стенок, ни нижней линейки,
    тело — одна «ячейка» на столбец с несколькими строками значений."""
    lines = [
        (72, 200, 432, 200),
        (72, 226, 432, 226),
        (192, 200, 192, 300),
        (312, 200, 312, 300),
    ]
    texts = [
        (78, 218, "Tok"),
        (198, 218, "Bolt"),
        (318, 218, "Ploshadka"),
        (78, 244, "4-6"),
        (198, 244, "3"),
        (318, 244, "10"),
        (78, 264, "6-16"),
        (198, 264, "3,5"),
        (318, 264, "11"),
        (78, 284, "16-40"),
        (198, 284, "4"),
        (318, 284, "12"),
    ]
    pages = load_pdf(_scan_pdf(tmp_path / "open.pdf", lines, texts))

    [table] = [b for b in pages[0].blocks if b.kind == "table"]
    assert "| Tok | Bolt | Ploshadka |" in table.text
    assert "| 4-6 | 3 | 10 |" in table.text
    assert "| 6-16 | 3,5 | 11 |" in table.text
    assert "| 16-40 | 4 | 12 |" in table.text


def test_born_digital_page_is_not_marked_as_ocr(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Obychnyy nabrannyy tekst " * 10, fontsize=10)
    path = tmp_path / "digital.pdf"
    doc.save(str(path))
    doc.close()
    assert not load_pdf(path)[0].from_ocr
