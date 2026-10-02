"""Таблицы: поиск сетки и сборка строк, пригодных для индекса.

Весь текущий корпус — сканы со скрытым текстовым слоем: страница целиком
картинка, поверх неё невидимый текст от распознавания издателя. Векторных линий
нет ни одной, поэтому штатный ``find_tables()`` pdfplumber не находил ни одной
таблицы, и решения №3 и №4 на реальных документах не работали вовсе: таблицы
доезжали до индекса россыпью чисел, где объединённая ячейка «10» висела между
чужими строками.

Линовка при этом есть — на картинке. Здесь она выделяется морфологией OpenCV и
передаётся pdfplumber как явные линии; текст ячеек по-прежнему берётся из
текстового слоя.

Вторая половина модуля превращает сетку в строки, каждая из которых понятна
без соседних: объединённые ячейки размножаются по всем строкам и столбцам, что
они покрывают, многоуровневая шапка склеивается в одну строку, а «открытые»
таблицы ГОСТов (линовка только в шапке) режутся на строки по строкам текста.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

#: Разрешение растра для поиска линий: 150 dpi хватает, чтобы линия в 0,3 мм
#: была толще пикселя, и вчетверо дешевле, чем 300 dpi для OCR.
RULE_DPI = 150

#: Допуски pdfplumber при сборке ячеек из линий, в пунктах. Линии со скана
#: слегка гуляют из-за перекоса, и нулевой допуск рвал бы сетку.
_TABLE_TOLERANCE = 4


@dataclass(frozen=True, slots=True)
class Rule:
    """Отрезок линовки в координатах страницы PDF (пункты, y вниз)."""

    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def is_horizontal(self) -> bool:
        return self.bottom - self.top < self.x1 - self.x0

    def as_line(self) -> dict:
        """Объект линии в том виде, в каком pdfplumber принимает явные линии."""
        return {
            "object_type": "line",
            "x0": self.x0,
            "x1": self.x1,
            "top": self.top,
            "bottom": self.bottom,
            "width": self.x1 - self.x0,
            "height": self.bottom - self.top,
        }


# --------------------------------------------------------------------------- #
# Линии из растра
# --------------------------------------------------------------------------- #


def detect_rules(gray: np.ndarray, scale: float) -> tuple[list[Rule], list[Rule]]:
    """Горизонтальные и вертикальные линии скана, пересчитанные в пункты PDF.

    Открытие длинным тонким ядром оставляет только то, что длиннее любого
    штриха буквы: горизонтальное — от 1/25 ширины страницы, вертикальное — от
    1/60 высоты (ячейки в одну строку низкие, их стенки короткие).
    """
    import cv2

    binary = cv2.adaptiveThreshold(
        255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    height, width = binary.shape
    kernels = {
        "h": (max(20, width // 25), 1),
        "v": (1, max(20, height // 60)),
    }
    found: dict[str, list[Rule]] = {}
    for name, size in kernels.items():
        mask = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size)
        )
        # Склеить разрывы в пару пикселей — следы перекоса и неровной печати.
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        rules: list[Rule] = []
        for x, y, w, h, _area in stats[1:]:
            if name == "h":
                mid = (y + h / 2) * scale
                rules.append(Rule(x * scale, mid, (x + w) * scale, mid))
            else:
                mid = (x + w / 2) * scale
                rules.append(Rule(mid, y * scale, mid, (y + h) * scale))
        found[name] = rules
    return found["h"], found["v"]


def extend_open_columns(
    horizontal: list[Rule], vertical: list[Rule], *, tol: float = 3.0
) -> list[Rule]:
    """Продлить вертикали шапки до нижней линейки «открытой» таблицы.

    В ГОСТах тело таблицы часто без вертикалей: столбцы разделены только в
    шапке, а тело замыкает одна горизонтальная линейка. Без продления
    pdfplumber видит одну шапку и ни одной ячейки тела (табл. 1 в
    ГОСТ 12.2.007.0-75). Вертикаль продлевается, только если упирается в
    линейку, под которой есть другая линейка той же ширины — то есть разделитель
    шапки и низ таблицы. Линейка другой ширины ниже — уже не эта таблица
    (например, черта над «Издание официальное»), и туда тянуть нельзя.
    """
    extended: list[Rule] = []
    for rule in vertical:
        x = (rule.x0 + rule.x1) / 2
        spanning = [h for h in horizontal if h.x0 - tol <= x <= h.x1 + tol]
        stop = next((h for h in spanning if abs(h.top - rule.bottom) <= tol), None)
        if stop is None:
            extended.append(rule)
            continue
        width = stop.x1 - stop.x0
        below = [
            h
            for h in spanning
            if h.top > stop.top + tol
            and abs(h.x0 - stop.x0) <= 0.05 * width
            and abs(h.x1 - stop.x1) <= 0.05 * width
        ]
        if not below:
            extended.append(rule)
            continue
        floor = min(below, key=lambda h: h.top)
        extended.append(Rule(rule.x0, rule.top, rule.x1, floor.top))
    return extended


def _touches(a: Rule, b: Rule, tol: float) -> bool:
    return (
        a.x0 - tol <= b.x1
        and b.x0 - tol <= a.x1
        and a.top - tol <= b.bottom
        and b.top - tol <= a.bottom
    )


def close_frames(
    horizontal: list[Rule], vertical: list[Rule], *, tol: float = 3.0
) -> tuple[list[Rule], list[Rule]]:
    """Достроить внешнюю рамку каждой группы связанных линий.

    Таблицы ГОСТов часто печатаются без боковых стенок и без нижней линейки:
    три внутренние вертикали и две горизонтали шапки (табл. 1 в
    ГОСТ 12.2.007.0-75). pdfplumber собирает ячейку только из четырёх сторон,
    и такое тело для него не существует. Рамка по охвату группы замыкает крайние
    столбцы и низ; у уже замкнутой таблицы она совпадает с настоящей и ничего
    не меняет. Рамки чертежей тоже замкнутся, но их отсеет ``is_real_table`` —
    текста в них нет.
    """
    rules = horizontal + vertical
    parent = list(range(len(rules)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            if _touches(rules[i], rules[j], tol):
                parent[find(i)] = find(j)

    groups: dict[int, list[Rule]] = {}
    for index, rule in enumerate(rules):
        groups.setdefault(find(index), []).append(rule)

    h_out, v_out = list(horizontal), list(vertical)
    for members in groups.values():
        has_h = any(r.is_horizontal for r in members)
        v_count = sum(1 for r in members if not r.is_horizontal)
        if not has_h or v_count < 1:
            continue
        x0 = min(r.x0 for r in members)
        x1 = max(r.x1 for r in members)
        top = min(r.top for r in members)
        bottom = max(r.bottom for r in members)
        h_out += [Rule(x0, top, x1, top), Rule(x0, bottom, x1, bottom)]
        v_out += [Rule(x0, top, x0, bottom), Rule(x1, top, x1, bottom)]
    return h_out, v_out


def raster_table_settings(horizontal: list[Rule], vertical: list[Rule]) -> dict | None:
    """Настройки ``find_tables`` по линиям скана; None — линий на таблицу не хватает."""
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    return {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": [r.as_line() for r in vertical],
        "explicit_horizontal_lines": [r.as_line() for r in horizontal],
        "snap_tolerance": _TABLE_TOLERANCE,
        "join_tolerance": _TABLE_TOLERANCE,
        "intersection_tolerance": _TABLE_TOLERANCE,
    }


# --------------------------------------------------------------------------- #
# Сетка -> строки
# --------------------------------------------------------------------------- #


def _clean_cell(text: str | None) -> str:
    # Мягкий перенос в конце строки ячейки: «Односторон\xad\nний» -> «Односторонний».
    text = (text or "").replace("\xad\n", "").replace("\xad", "")
    return text.strip()


def _index(edges: list[float], value: float) -> int:
    """Номер ближайшей границы сетки."""
    return min(range(len(edges)), key=lambda i: abs(edges[i] - value)) if edges else 0


def grid_cells(table) -> list[list[str | None]]:
    """Матрица текста таблицы, где объединённая ячейка занимает все свои клетки.

    Без размножения строка «1,5 | 9,026 | 8,376» теряла номинальный диаметр 10:
    он стоит один раз, посередине объединённой ячейки, и строке с шагом 1,5 не
    принадлежит ни по какому признаку, кроме геометрии.
    """
    texts = table.extract()
    col_edges = sorted({round(c[0], 1) for c in table.cells} | {round(table.bbox[2], 1)})
    row_edges = sorted({round(c[1], 1) for c in table.cells} | {round(table.bbox[3], 1)})
    n_rows, n_cols = len(row_edges) - 1, len(col_edges) - 1
    grid: list[list[str | None]] = [[None] * n_cols for _ in range(n_rows)]

    for row, row_texts in zip(table.rows, texts, strict=False):
        for bbox, text in zip(row.cells, row_texts, strict=False):
            if bbox is None:
                continue
            c0, c1 = _index(col_edges, bbox[0]), _index(col_edges, bbox[2])
            r0, r1 = _index(row_edges, bbox[1]), _index(row_edges, bbox[3])
            value = _clean_cell(text)
            for r in range(r0, max(r1, r0 + 1)):
                for c in range(c0, max(c1, c0 + 1)):
                    if r < n_rows and c < n_cols:
                        grid[r][c] = value
    return grid


_NUMERIC_CHARS = set("0123456789,.±+-−–—×/%°")

#: Доля числовых знаков, с которой ячейка считается числовой. Не 100 %: слой
#: распознавания издателя пишет «и» вместо «11», и одна такая буква не должна
#: превращать столбец значений в текст. Не 50 %: у «Св. 10 до 16» цифр и точек
#: больше половины, а это подпись диапазона в шапке.
_NUMERIC_SHARE = 0.8


def _is_numeric(cell: str) -> bool:
    chars = [ch for ch in cell if not ch.isspace()]
    return bool(chars) and sum(ch in _NUMERIC_CHARS for ch in chars) >= _NUMERIC_SHARE * len(chars)


def is_data_row(row: list[str | None]) -> bool:
    """Строка данных: чисто числовых ячеек не меньше половины непустых.

    Шапка ГОСТа тоже полна цифр — «Св. 10 до 16», «20°», — но это диапазоны
    со словами или одиночные подписи среди размноженных названий столбцов.
    Доля цифр в ячейке тут не годится: у «Св. 10 до 16» она выше половины.
    """
    cells = [c for c in row if c]
    return bool(cells) and sum(_is_numeric(c) for c in cells) * 2 >= len(cells)


#: Предел высоты шапки в строках сетки. У ГОСТ 10549-80 шапка в пять-шесть
#: строк сетки (повёрнутые подписи режут её мелко); дальше — уже тело.
_MAX_HEADER_ROWS = 8


def split_header(grid: list[list[str | None]]) -> int:
    """Число строк шапки: ведущие строки до первой строки данных.

    Ноль — таблица начинается сразу с данных: это продолжение таблицы с прошлой
    страницы, и шапку ей даёт вызывающая сторона. Если данных не видно вовсе —
    таблица текстовая, шапкой считается первая строка.
    """
    for index, row in enumerate(grid[:_MAX_HEADER_ROWS]):
        if is_data_row(row):
            return index
    return 1 if grid else 0


def flatten_header(rows: list[list[str | None]]) -> list[str]:
    """Склеить многоуровневую шапку в одну строку: «Проточка / Тип 1 / f».

    Повтор заголовка при разрезании таблицы (решение №4) повторяет ровно одну
    строку — поэтому уровни шапки должны жить в ней, а не в строках тела.
    """
    if not rows:
        return []
    header: list[str] = []
    for col in range(len(rows[0])):
        parts: list[str] = []
        for row in rows:
            value = (row[col] or "").replace("\n", " ").strip()
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        header.append(" / ".join(parts))
    return header


def explode_row(page, table, row_index: int, grid_row: list[str | None]) -> list[list[str]]:
    """Разрезать строку-блок открытой таблицы на строки текста.

    В теле без горизонталей вся колонка значений — одна ячейка: «10\\n11\\n12».
    Слова раскладываются по строкам по вертикали и по столбцам по горизонтали —
    так пропуск значения в одной колонке не сдвигает соседние. Строка режется,
    только если многострочных ячеек хотя бы две: одна многострочная ячейка —
    это перенос длинного текста, а не несколько строк таблицы.
    """
    multiline = sum(1 for cell in grid_row if cell and "\n" in cell)
    if multiline < 2:
        return [[(c or "").replace("\n", " ") for c in grid_row]]

    row_bbox = table.rows[row_index].bbox
    col_edges = sorted({c[0] for c in table.cells} | {table.bbox[2]})
    words = page.crop(row_bbox).extract_words(keep_blank_chars=False, use_text_flow=False)
    if not words:
        return [[(c or "").replace("\n", " ") for c in grid_row]]

    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0]["top"] - word["top"]) <= 3:
            lines[-1].append(word)
        else:
            lines.append([word])

    exploded: list[list[str]] = []
    for line in lines:
        cells = [""] * len(grid_row)
        for word in sorted(line, key=lambda w: w["x0"]):
            center = (word["x0"] + word["x1"]) / 2
            col = max(0, min(len(grid_row) - 1, sum(1 for e in col_edges[1:] if e <= center)))
            cells[col] = f"{cells[col]} {word['text']}".strip()
        exploded.append(cells)
    return exploded


def is_real_table(grid: list[list[str | None]]) -> bool:
    """Отсечь «таблицы» из чертежей: рамки без текста или почти без него."""
    if len(grid) < 2 or not grid[0] or len(grid[0]) < 2:
        return False
    cells = [c for row in grid for c in row]
    filled = sum(1 for c in cells if c)
    return filled >= 4 and filled * 2 >= len(cells)


@dataclass(slots=True)
class TableRows:
    header: list[str]
    body: list[list[str]]
    #: Шапка найдена в самой таблице, а не унаследована с прошлой страницы.
    own_header: bool = True


def table_rows(page, table, *, require_data: bool = False) -> TableRows | None:
    """Строки таблицы: плоская шапка + тело с размноженными объединёнными ячейками.

    ``require_data`` — для сеток, найденных по растру: без единой строки данных
    это почти всегда штриховка чертежа («У///ЛМ» в каждой клетке), и её текст
    вызывающая сторона оставит в потоке. Векторной линовке можно верить и на
    чисто текстовой таблице.
    """
    grid = grid_cells(table)
    if not is_real_table(grid):
        return None
    header_count = split_header(grid)
    if require_data and not any(is_data_row(row) for row in grid[header_count:]):
        return None
    header = flatten_header(grid[:header_count])
    body: list[list[str]] = []
    for index in range(header_count, len(grid)):
        row = grid[index]
        # Резать по строкам текста можно только данные: многострочная ячейка
        # в текстовой строке — это перенос, а не несколько строк таблицы.
        if index < len(table.rows) and is_data_row(row):
            body.extend(explode_row(page, table, index, row))
        else:
            body.append([(c or "").replace("\n", " ") for c in row])
    body = [row for row in body if any(cell for cell in row)]
    return TableRows(header=header, body=body, own_header=header_count > 0)


def uses_raster_rules(page, gray: np.ndarray | None) -> bool:
    """Линовку придётся искать на картинке: векторной нет, а растр есть."""
    return gray is not None and not (page.lines or page.rects)


def find_tables(page, gray: np.ndarray | None = None) -> list:
    """Таблицы страницы: по векторной линовке, а для сканов — по линиям растра.

    ``gray`` — растр страницы в ``RULE_DPI``; без него работает только
    векторный путь (DOCX-конвертации, «родные» PDF).
    """
    if not uses_raster_rules(page, gray):
        return page.find_tables()
    scale = page.width / gray.shape[1]
    horizontal, vertical = detect_rules(gray, scale)
    vertical = extend_open_columns(horizontal, vertical)
    horizontal, vertical = close_frames(horizontal, vertical)
    settings = raster_table_settings(horizontal, vertical)
    return page.find_tables(settings) if settings else []


#: Подпись таблицы над ней: «Таблица 1», «Т а б л и ц а 1», «Продолжение табл. 1».
#: Слой распознавания пишет «ц» как «и» — «Т а б л и и а 2», «Таблииа4»; без
#: допуска табл. 2 ГОСТ 10549-80 попадала в индекс вовсе без подписи.
#: Номер — тоже как его прочёл OCR: «Таблица I», «Таблицаб», «Таблица?»;
#: привести его к числу — забота ``caption_number``.
CAPTION_RE = re.compile(
    r"^\s*(?:(?P<new>Т\s*а\s*б\s*л\s*и\s*[цил]\s*а)"
    r"|(?P<cont>(?:Продолжение|Окончание)\s+табл\w*\.?))"
    r"\s*(?P<num>[0-9IlбЗзОO|?]{1,3})?",
    re.IGNORECASE,
)

#: Буквы, которыми слой распознавания подменяет цифры в номере таблицы.
_OCR_DIGITS = str.maketrans(
    {"I": "1", "l": "1", "|": "1", "б": "6", "З": "3", "з": "3", "О": "0", "O": "0"}
)


def caption_number(caption: str) -> tuple[int | None, bool] | None:
    """Номер таблицы из подписи и признак продолжения; None — это не подпись.

    Номер None — подпись есть, но цифра нечитаема («Таблица?»): номер тогда
    восстанавливают по порядку таблиц в документе.
    """
    match = CAPTION_RE.match(caption)
    if match is None:
        return None
    raw = (match.group("num") or "").translate(_OCR_DIGITS)
    return (int(raw) if raw.isdigit() else None), match.group("cont") is not None
