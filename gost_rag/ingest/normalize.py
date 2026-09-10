"""Нормализация текста, извлечённого из PDF/DOCX.

Задачи: склеить переносы по дефису, убрать повторяющиеся колонтитулы,
схлопнуть пробелы — но сохранить нумерацию пунктов (строки вида "3.2.1 ..."),
потому что именно она даёт ссылку на пункт стандарта в цитате.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

#: Строка, начинающаяся с номера пункта: "3", "3.2", "3.2.1".
CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+(?=\S)")

#: Перенос слова: "поверх-\nность" -> "поверхность".
_HYPHEN_WRAP_RE = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")

#: Мягкие переносы и неразрывные пробелы, которыми полны PDF ГОСТов.
_SOFT_HYPHEN_RE = re.compile(r"[­​]")
_NBSP_RE = re.compile(r"[   ]")

#: Разрядка заголовков в старых ГОСТах: "О Б Щ И Е  Т Р Е Б О В А Н И Я".
_SPACED_CAPS_RE = re.compile(r"\b(?:[А-ЯЁ]\s){2,}[А-ЯЁ]\b")

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_page_text(text: str) -> str:
    """Базовая чистка одной страницы (без учёта соседних страниц)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = _NBSP_RE.sub(" ", text)
    text = _HYPHEN_WRAP_RE.sub(r"\1\2", text)
    text = _SPACED_CAPS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def find_repeated_lines(
    pages: Iterable[str],
    *,
    edge_lines: int = 3,
    min_ratio: float = 0.5,
    min_pages: int = 4,
) -> set[str]:
    """Найти колонтитулы: короткие строки у края страницы, повторяющиеся часто.

    Порог по доле страниц, а не по абсолютному числу, чтобы одинаково работать
    и на 10-страничном, и на 200-страничном стандарте. На коротких документах
    (< ``min_pages``) не срабатывает вовсе — там повтор чаще случаен.
    """
    page_list = list(pages)
    if len(page_list) < min_pages:
        return set()

    counter: Counter[str] = Counter()
    for page in page_list:
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        window = _edge_window(len(lines), edge_lines)
        edges = lines[:window] + lines[-window:]
        # Одна и та же строка на одной странице считается один раз.
        for line in set(edges):
            if len(line) <= 80 and not CLAUSE_RE.match(line):
                counter[_normalize_for_compare(line)] += 1

    threshold = max(min_pages // 2, int(len(page_list) * min_ratio))
    return {line for line, count in counter.items() if count >= threshold}


#: Максимальная длина «буквенной» части строки, при которой цифры считаются
#: номером страницы и маскируются при сравнении.
_LOCATOR_ALPHA_LIMIT = 15


def _normalize_for_compare(line: str) -> str:
    """Сравниваем колонтитулы без номеров страниц: «стр. 7» и «стр. 8» — одно.

    Маскируем цифры только у коротких «локаторов» (``Стр. 7``, ``ГОСТ 14634-93``).
    У содержательных строк цифры значимы: без этого ограничения строки вида
    «... на странице 1» и «... на странице 2» слились бы в одну и весь текст
    документа был бы принят за колонтитул и вырезан.
    """
    collapsed = line.casefold().strip()
    alpha_only = re.sub(r"[\d\W_]+", "", collapsed)
    if len(alpha_only) <= _LOCATOR_ALPHA_LIMIT:
        return re.sub(r"\d+", "#", collapsed)
    return collapsed


def _edge_window(line_count: int, edge_lines: int) -> int:
    """На коротких страницах «край» не должен покрывать страницу целиком."""
    return max(1, min(edge_lines, line_count // 3))


def strip_repeated_lines(text: str, repeated: set[str], *, edge_lines: int = 3) -> str:
    if not repeated:
        return text
    lines = text.split("\n")
    window = _edge_window(len(lines), edge_lines)
    keep: list[str] = []
    for idx, line in enumerate(lines):
        near_edge = idx < window or idx >= len(lines) - window
        if near_edge and line.strip() and _normalize_for_compare(line.strip()) in repeated:
            continue
        keep.append(line)
    return _MULTI_NEWLINE_RE.sub("\n\n", "\n".join(keep)).strip()


def extract_clause(line: str) -> str | None:
    """Вернуть номер пункта, если строка его начинает."""
    match = CLAUSE_RE.match(line)
    return match.group(1) if match else None
