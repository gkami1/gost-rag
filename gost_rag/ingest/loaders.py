"""Извлечение текста и таблиц из PDF и DOCX.

PDF читается pdfplumber'ом: он один раз находит таблицы, отдаёт их структурно и
позволяет вырезать их области из текстового слоя. Это важно — если брать текст
и таблицы независимо, ячейки попадают в чанк дважды: один раз связной таблицей,
второй раз россыпью чисел. Порядок чтения восстанавливается по вертикали:
текст над таблицей, таблица, текст под ней.

Отрисовка страниц для OCR делается PyMuPDF — он быстрее и не требует внешних
зависимостей для растеризации.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from gost_rag.ingest.normalize import (
    clean_page_text,
    extract_clause,
    find_repeated_lines,
    strip_repeated_lines,
)
from gost_rag.logging import get_logger
from gost_rag.models import Block, PageDoc

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Таблицы
# --------------------------------------------------------------------------- #


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Преобразовать таблицу в GitHub-markdown.

    Пустые ячейки — обычное дело для ГОСТов с объединёнными ячейками; заменяем
    их пробелом, чтобы не поехала разметка столбцов.
    """
    cleaned = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Разбиение текста на блоки
# --------------------------------------------------------------------------- #


def blocks_from_text(
    text: str, page_no: int, section: str | None = None
) -> tuple[list[Block], str | None]:
    """Разбить текст на абзацы-блоки, отслеживая текущий номер пункта.

    Возвращает блоки и номер пункта, действующий в конце страницы, — он
    переносится на следующую страницу, если та начинается с продолжения абзаца.
    """
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        body = "\n".join(paragraph).strip()
        if body:
            blocks.append(Block(text=body, page_no=page_no, kind="text", section=section))
        paragraph = []

    for line in text.split("\n"):
        if not line.strip():
            flush()
            continue
        clause = extract_clause(line)
        if clause is not None:
            flush()
            section = clause
        paragraph.append(line)

    flush()
    return blocks, section


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def _page_blocks_pdf(
    page, page_no: int, section: str | None, repeated: set[str]
) -> tuple[list[Block], str | None, str]:
    """Блоки одной PDF-страницы в порядке чтения + сырой текст страницы."""
    tables = page.find_tables()
    blocks: list[Block] = []
    raw_parts: list[str] = []

    if not tables:
        text = strip_repeated_lines(clean_page_text(page.extract_text() or ""), repeated)
        raw_parts.append(text)
        page_blocks, section = blocks_from_text(text, page_no, section)
        return page_blocks, section, "\n".join(raw_parts)

    ordered = sorted(tables, key=lambda t: t.bbox[1])
    cursor_top = 0.0

    for table in ordered:
        _, top, _, bottom = table.bbox
        band = strip_repeated_lines(_crop_text(page, cursor_top, top), repeated)
        if band:
            raw_parts.append(band)
            band_blocks, section = blocks_from_text(band, page_no, section)
            blocks.extend(band_blocks)

        markdown = table_to_markdown(table.extract())
        if markdown:
            raw_parts.append(markdown)
            blocks.append(Block(text=markdown, page_no=page_no, kind="table", section=section))
        cursor_top = bottom

    trailing = strip_repeated_lines(_crop_text(page, cursor_top, page.height), repeated)
    if trailing:
        raw_parts.append(trailing)
        tail_blocks, section = blocks_from_text(trailing, page_no, section)
        blocks.extend(tail_blocks)

    return blocks, section, "\n".join(raw_parts)


def _crop_text(page, top: float, bottom: float) -> str:
    """Текст горизонтальной полосы страницы; пустая или вырожденная — пропуск."""
    if bottom - top < 1:
        return ""
    try:
        band = page.crop((0, max(top, 0), page.width, min(bottom, page.height)))
    except ValueError:
        return ""
    return clean_page_text(band.extract_text() or "")


def load_pdf(path: Path, *, ocr_min_chars: int = 100) -> list[PageDoc]:
    """Прочитать PDF постранично. Страницы без текстового слоя помечаются под OCR.

    Проход первый — собрать текст всех страниц и найти колонтитулы: понять, что
    строка повторяется, можно только увидев документ целиком. Проход второй —
    разобрать страницы на блоки, вырезая найденные колонтитулы до того, как
    текст попадёт в чанк.
    """
    import pdfplumber

    pages: list[PageDoc] = []
    section: str | None = None

    with pdfplumber.open(str(path)) as pdf:
        page_texts: list[str] = []
        for page in pdf.pages:
            try:
                page_texts.append(clean_page_text(page.extract_text() or ""))
            except Exception:
                page_texts.append("")
        repeated = find_repeated_lines(page_texts)
        if repeated:
            log.info("running_headers_found", path=str(path), count=len(repeated))

        for index, page in enumerate(pdf.pages, start=1):
            try:
                blocks, section, raw = _page_blocks_pdf(page, index, section, repeated)
            except Exception as exc:
                log.warning("pdf_page_failed", path=str(path), page=index, error=str(exc))
                blocks, raw = [], ""
            needs_ocr = len(raw.strip()) < ocr_min_chars
            pages.append(
                PageDoc(
                    page_no=index,
                    text="" if needs_ocr else raw,
                    blocks=[] if needs_ocr else blocks,
                    needs_ocr=needs_ocr,
                )
            )
    return pages


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def load_docx(path: Path) -> list[PageDoc]:
    """Прочитать DOCX. Страниц в docx нет, поэтому весь файл — «страница 1».

    Абзацы и таблицы обходятся в порядке документа, чтобы таблица не оторвалась
    от вводящего её абзаца.
    """
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks: list[Block] = []
    section: str | None = None
    buffer: list[str] = []
    raw_parts: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer, section
        if not buffer:
            return
        text = clean_page_text("\n".join(buffer))
        if text:
            raw_parts.append(text)
            new_blocks, section = blocks_from_text(text, 1, section)
            blocks.extend(new_blocks)
        buffer = []

    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            buffer.append(Paragraph(child, document).text)
        elif tag == "tbl":
            flush_buffer()
            rows = [[cell.text for cell in row.cells] for row in Table(child, document).rows]
            markdown = table_to_markdown(rows)
            if markdown:
                raw_parts.append(markdown)
                blocks.append(Block(text=markdown, page_no=1, kind="table", section=section))

    flush_buffer()
    return [PageDoc(page_no=1, text="\n".join(raw_parts), blocks=blocks)]


def load_document(path: Path, *, ocr_min_chars: int = 100) -> list[PageDoc]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, ocr_min_chars=ocr_min_chars)
    if suffix == ".docx":
        return load_docx(path)
    raise ValueError(
        f"Неподдерживаемый формат: {suffix} (поддерживаются {sorted(SUPPORTED_SUFFIXES)})"
    )
