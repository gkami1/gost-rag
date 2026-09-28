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
#: Уровень ограничен тремя цифрами: пункта «1777771» в стандартах не бывает, а
#: такие строки в изобилии даёт искажённый текстовый слой таблиц.
CLAUSE_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3}){0,4})\.?\s+(?=\S)")

#: Любая буква (кириллица, латиница) — цифры, знаки и пунктуация не считаются.
_LETTER_RE = re.compile(r"[^\W\d_]")

#: Сколько букв должно стоять за номером, чтобы строка считалась пунктом.
#: Двух мало: «6 68,103 мм» — это строка таблицы с единицей измерения, а не пункт.
_CLAUSE_MIN_LETTERS = 3

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
        # Порядок слов тоже не важен: в книжной вёрстке чётная страница несёт
        # «С. 3 ГОСТ 12.2.007.0-75», нечётная — «ГОСТ 12.2.007.0-75 С. 2». Каждый
        # вариант встречается лишь на половине страниц и порога не набирает.
        return " ".join(sorted(re.sub(r"\d+", "#", collapsed).split()))
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


#: На сколько номеров вперёд может «прыгнуть» пункт того же уровня. Больше единицы —
#: чтобы строка пункта, потерянная OCR или разрывом страницы, не рвала цепочку.
_CLAUSE_MAX_STEP = 2


def clause_follows(prev: str, new: str) -> bool:
    """Может ли пункт ``new`` идти в документе сразу за пунктом ``prev``.

    Допустимы три хода: вглубь к первому подпункту (3.4 -> 3.4.1), к следующему
    пункту того же уровня (3.4.7 -> 3.4.8) и наверх к следующему пункту одного из
    предков (3.4.15 -> 3.5, 3.9.5 -> 4) — в последних двух случаях можно сразу
    спуститься к первым подпунктам (3.9.5 -> 4.1). Всё остальное — номер строки
    таблицы, примечания или мусор распознавания, а не пункт.
    """
    p = [int(x) for x in prev.split(".")]
    q = [int(x) for x in new.split(".")]
    if q == [*p, 1]:
        return True
    for level in range(len(p)):
        head, value = p[:level], p[level]
        if q[:level] != head or len(q) <= level:
            continue
        step = q[level] - value
        if 1 <= step <= _CLAUSE_MAX_STEP and all(x == 1 for x in q[level + 1 :]):
            return True
    return False


def clause_chain(candidates: list[str]) -> set[int]:
    """Индексы кандидатов, образующих самую длинную правдоподобную цепочку пунктов.

    Решать по одному соседу нельзя: современный ГОСТ начинается с предисловия,
    пронумерованного 1–6, а потом нумерация честно начинается заново с
    «1 Область применения». Строгое «следующий должен продолжать предыдущий»
    застряло бы на «6» до конца документа. Самая длинная цепочка по всему
    документу выбирает основную нумерацию, а предисловие, примечания и строки
    таблиц остаются вне её.
    """
    n = len(candidates)
    if n == 0:
        return set()
    length = [1] * n
    parent = [-1] * n
    for i in range(n):
        for j in range(i):
            if length[j] + 1 > length[i] and clause_follows(candidates[j], candidates[i]):
                length[i] = length[j] + 1
                parent[i] = j
    # При равной длине — более ранний конец: он не тянет за собой хвостовой мусор.
    end = max(range(n), key=lambda i: (length[i], -i))
    chain: set[int] = set()
    while end != -1:
        chain.add(end)
        end = parent[end]
    return chain


def extract_clause(line: str) -> str | None:
    """Вернуть номер пункта, если строка его начинает.

    Одного числа в начале строки мало. Строка таблицы «6 68,103 65,505 64,639»
    начинается точно так же, как пункт «6 Технические требования», и без
    дополнительной проверки номер строки таблицы становится номером пункта у
    всего чанка. Дальше он уходит в цитату — «п. 6, стр. 11» на содержимое,
    которого в шестом пункте нет. Выдуманная ссылка на пункт хуже, чем её
    отсутствие, поэтому за номером обязаны идти буквы, а не только числа.
    """
    match = CLAUSE_RE.match(line)
    if match is None:
        return None
    if len(_LETTER_RE.findall(line[match.end() :])) < _CLAUSE_MIN_LETTERS:
        return None
    return match.group(1)
