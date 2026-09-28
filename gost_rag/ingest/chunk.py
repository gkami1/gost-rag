"""Чанкинг: фиксированное окно 800 токенов с перекрытием 10 %.

Это осознанно простой baseline — никакого семантического разбиения. Единственная
уступка предметной области: таблица не режется посередине строки, потому что
именно в таблицах ГОСТов лежат ответы вида «радиус гибки для толщины 3 мм».
Если таблица не влезает в окно, она делится по строкам с повтором заголовка.

Токенайзер внедряется снаружи: в проде это токенайзер BGE-M3 (чтобы бюджет чанка
совпадал с бюджетом энкодера), в тестах — дешёвая замена.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from gost_rag.models import Block, Chunk


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...

    def split_text(self, text: str, max_tokens: int) -> list[str]: ...


class ApproxTokenizer:
    """Грубая оценка без загрузки моделей: ~1 токен на 3 символа кириллицы.

    Нужна для тестов и как аварийный вариант, если transformers недоступны.
    """

    chars_per_token = 3

    def count(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)

    def split_text(self, text: str, max_tokens: int) -> list[str]:
        size = max_tokens * self.chars_per_token
        return [text[i : i + size] for i in range(0, len(text), size)] or [text]


class HFTokenizer:
    """Токенайзер BGE-M3 (XLM-RoBERTa). Загружается лениво."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def split_text(self, text: str, max_tokens: int) -> list[str]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return [text]
        return [
            self._tok.decode(ids[i : i + max_tokens], skip_special_tokens=True)
            for i in range(0, len(ids), max_tokens)
        ]


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|-: ")


def _split_table_block(block: Block, tokenizer: Tokenizer, max_tokens: int) -> list[Block]:
    """Разрезать markdown-таблицу по строкам, повторяя заголовок в каждой части.

    Заголовок — всё до строки-разделителя включительно: подпись («Таблица 1
    Размеры в мм») и шапка. Без подписи продолжение таблицы в соседнем чанке
    теряет и номер таблицы, и единицы измерения.
    """
    lines = block.text.split("\n")
    separator = next((i for i, line in enumerate(lines[:4]) if _is_separator_row(line)), None)
    has_header = separator is not None and separator >= 1 and len(lines) > separator + 1
    header = lines[: separator + 1] if has_header else []
    body = lines[len(header) :]
    header_text = "\n".join(header)
    header_cost = tokenizer.count(header_text) if header else 0

    parts: list[Block] = []
    current: list[str] = []
    current_cost = header_cost

    for line in body:
        line_cost = tokenizer.count(line)
        if current and current_cost + line_cost > max_tokens:
            parts.append(_table_part(block, header_text, current))
            current = []
            current_cost = header_cost
        current.append(line)
        current_cost += line_cost

    if current:
        parts.append(_table_part(block, header_text, current))
    return parts or [block]


def _table_part(block: Block, header_text: str, rows: list[str]) -> Block:
    text = "\n".join(([header_text] if header_text else []) + rows)
    return Block(
        text=text,
        page_no=block.page_no,
        kind="table",
        section=block.section,
        table_header=header_text or None,
    )


def prepare_blocks(blocks: list[Block], tokenizer: Tokenizer, max_tokens: int) -> list[Block]:
    """Привести блоки к размеру, гарантированно влезающему в окно."""
    prepared: list[Block] = []
    for block in blocks:
        if not block.text.strip():
            continue
        if tokenizer.count(block.text) <= max_tokens:
            prepared.append(block)
            continue
        if block.kind == "table":
            prepared.extend(_split_table_block(block, tokenizer, max_tokens))
        else:
            prepared.extend(
                Block(text=piece, page_no=block.page_no, kind=block.kind, section=block.section)
                for piece in tokenizer.split_text(block.text, max_tokens)
            )
    return prepared


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    tokenizer: Tokenizer,
    *,
    chunk_tokens: int = 800,
    overlap_tokens: int = 80,
    min_chunk_tokens: int = 50,
    ocr_pages: set[int] | None = None,
    ocr_confidence: dict[int, float] | None = None,
) -> list[Chunk]:
    """Собрать блоки в чанки по ``chunk_tokens`` с перекрытием ``overlap_tokens``.

    Перекрытие реализуется переносом хвостовых блоков предыдущего чанка в начало
    следующего — так граница никогда не рассекает строку таблицы или предложение.
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens должен быть меньше chunk_tokens")

    prepared = prepare_blocks(blocks, tokenizer, chunk_tokens)
    if not prepared:
        return []

    ocr_pages = ocr_pages or set()
    ocr_confidence = ocr_confidence or {}

    chunks: list[Chunk] = []
    current: list[Block] = []
    costs: list[int] = []
    current_cost = 0
    # Сколько блоков в начале ``current`` перенесено перекрытием из предыдущего чанка.
    carried = 0

    for block in prepared:
        cost = tokenizer.count(block.text)
        if current and current_cost + cost > chunk_tokens:
            chunks.append(
                _build_chunk(
                    current, doc_id, len(chunks), current_cost, carried, ocr_pages, ocr_confidence
                )
            )
            # Перекрытие не должно выталкивать чанк за бюджет: иначе хвост чанка
            # уходит за окно энкодера и не попадает ни в вектор, ни к реранкеру.
            carry = _overlap_tail(current, costs, min(overlap_tokens, chunk_tokens - cost))
            current = list(carry)
            carried = len(carry)
            costs = [tokenizer.count(b.text) for b in carry]
            current_cost = sum(costs)
        current.append(block)
        costs.append(cost)
        current_cost += cost

    if current:
        # Хвост короче минимума приклеиваем к предыдущему чанку, чтобы не плодить огрызки.
        if chunks and current_cost < min_chunk_tokens:
            # Перенесённые перекрытием блоки уже лежат в предыдущем чанке.
            fresh = current[carried:]
            _append_tail(chunks[-1], fresh, sum(costs[carried:]))
        else:
            chunks.append(
                _build_chunk(
                    current, doc_id, len(chunks), current_cost, carried, ocr_pages, ocr_confidence
                )
            )

    return chunks


def _append_tail(last: Chunk, blocks: list[Block], cost: int) -> None:
    tail_text = "\n\n".join(b.text for b in blocks)
    if tail_text in last.text:
        return
    last.text = f"{last.text}\n\n{tail_text}"
    last.token_count += cost
    last.page_end = max(last.page_end, max(b.page_no for b in blocks))
    last.sections = _unique_sections([*last.sections, *(b.section for b in blocks)])
    last.contains_table = last.contains_table or any(b.kind == "table" for b in blocks)


def _unique_sections(sections: Iterable[str | None]) -> list[str]:
    """Пункты в порядке появления, без повторов и пустых значений."""
    return list(dict.fromkeys(s for s in sections if s))


def _overlap_tail(blocks: list[Block], costs: list[int], overlap_tokens: int) -> list[Block]:
    """Хвостовые блоки, укладывающиеся в бюджет перекрытия.

    Возвращаем строго меньше, чем весь чанк: иначе следующий чанк начался бы
    с полной копии предыдущего и упаковка зациклилась бы.
    """
    if overlap_tokens <= 0 or len(blocks) < 2:
        return []
    tail: list[Block] = []
    total = 0
    for block, cost in zip(reversed(blocks), reversed(costs), strict=True):
        if total + cost > overlap_tokens:
            break
        tail.append(block)
        total += cost
    tail.reverse()
    return tail if len(tail) < len(blocks) else tail[1:]


def _build_chunk(
    blocks: list[Block],
    doc_id: str,
    index: int,
    token_count: int,
    carried: int,
    ocr_pages: set[int],
    ocr_confidence: dict[int, float],
) -> Chunk:
    pages = [b.page_no for b in blocks]
    # Все пункты, чей текст реально лежит в чанке, — для цитаты диапазоном.
    sections = _unique_sections(b.section for b in blocks)
    # Основной пункт — первый из собственного содержимого чанка. Раньше брался
    # первый вообще, то есть пункт хвоста перекрытия: чанк с пп. 3.3.13–3.4.7
    # подписывался как «п. 3.3.12», и цитата ссылалась на чужой пункт.
    own = _unique_sections(b.section for b in blocks[carried:])
    used_ocr = [p for p in pages if p in ocr_pages]
    confs = [ocr_confidence[p] for p in used_ocr if p in ocr_confidence]
    return Chunk(
        doc_id=doc_id,
        chunk_index=index,
        text="\n\n".join(b.text for b in blocks).strip(),
        page_start=min(pages),
        page_end=max(pages),
        token_count=token_count,
        section=(own or sections or [None])[0],
        sections=sections,
        contains_table=any(b.kind == "table" for b in blocks),
        from_ocr=bool(used_ocr),
        ocr_confidence=round(sum(confs) / len(confs), 2) if confs else None,
    )
