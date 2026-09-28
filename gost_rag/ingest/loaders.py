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
import re
from dataclasses import dataclass
from pathlib import Path

from gost_rag.ingest.normalize import (
    clause_chain,
    clean_page_text,
    extract_clause,
    find_repeated_lines,
    strip_repeated_lines,
)
from gost_rag.ingest.tables import (
    CAPTION_RE,
    RULE_DPI,
    TableRows,
    find_tables,
    table_rows,
    uses_raster_rules,
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


def assign_sections(blocks: list[Block]) -> None:
    """Проставить пункты заново, по всему документу сразу.

    Постраничная разметка в ``blocks_from_text`` видит только соседние строки и
    принимает за пункт любую строку с номером: примечание «2. Для источников…»
    посреди п. 3.3, строку таблицы «1.0 …», номер пункта предисловия. Сканы
    размечаются отдельно от текстового слоя, и нумерации двух проходов
    перемежаются. Здесь блоки идут в порядке документа, и пунктом считается
    только номер из самой длинной правдоподобной цепочки (``clause_chain``);
    остальные блоки наследуют пункт, действующий на момент их появления.
    """
    starts: list[tuple[int, str]] = []
    for index, block in enumerate(blocks):
        if block.kind == "text":
            clause = extract_clause(block.text.split("\n", 1)[0])
            if clause is not None:
                starts.append((index, clause))

    chain = clause_chain([clause for _, clause in starts])
    opening = {starts[i][0]: starts[i][1] for i in chain}

    section: str | None = None
    for index, block in enumerate(blocks):
        section = opening.get(index, section)
        block.section = section

    _assign_table_sections(blocks)


#: Номер таблицы в подписи: «Таблица 1», «Т а б л и ц а 18», «Продолжение таблицы 1».
_CAPTION_NUMBER_RE = re.compile(
    r"^\s*(?:Т\s*а\s*б\s*л\s*и\s*ц\s*а|(?:Продолжение|Окончание)\s+табл\w*\.?)\s*(\d+)",
    re.IGNORECASE,
)
#: Ссылка на таблицу в тексте: «в таблице 1», «см. табл. 3», «по таблицам 2».
_TABLE_REF_RE = re.compile(r"\bтабл(?:\.|\w*)\s*(\d+)", re.IGNORECASE)


def _assign_table_sections(blocks: list[Block]) -> None:
    """Отдать таблицу пункту, который на неё ссылается, а не тому, где она напечатана.

    Таблица стоит там, где ей нашлось место на странице. В ГОСТ 24705-2004
    табл. 1 вводит п. 4.1 («приведены в таблице 1»), но печатается после
    п. 4.2 — и по месту получала «п. 4.2». Модель честно переписывала эту
    подпись в ответ: «таблица 1 (п. 4.2)» — правдоподобный, но чужой пункт.
    Берётся первая ссылка до таблицы; нет ссылки — остаётся пункт по месту.
    """
    first_ref: dict[str, str] = {}
    for block in blocks:
        if block.kind == "table":
            match = _CAPTION_NUMBER_RE.match(block.text)
            if match and match.group(1) in first_ref:
                block.section = first_ref[match.group(1)]
            continue
        if block.section is None:
            continue
        for ref in _TABLE_REF_RE.finditer(block.text):
            first_ref.setdefault(ref.group(1), block.section)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _TableContext:
    """Что таблица на следующей странице может унаследовать от предыдущей."""

    #: Плоская шапка последней таблицы — для «Продолжения табл.» без своей шапки.
    header: list[str] | None = None


def _split_caption(band: str) -> tuple[str, str | None]:
    """Отделить от текста над таблицей её подпись: «Таблица 1 / Размеры в мм».

    Без подписи таблица в чанке — голая сетка чисел: ни номера, по которому
    её называет текст («см. табл. 1»), ни единиц измерения.
    """
    lines = band.split("\n")
    for index in range(len(lines) - 1, max(-1, len(lines) - 4), -1):
        if CAPTION_RE.match(lines[index]):
            caption = " ".join(line.strip() for line in lines[index:] if line.strip())
            return "\n".join(lines[:index]).strip(), caption
    return band, None


def _table_block(
    rows: TableRows, page_no: int, section: str | None, caption: str | None, ctx: _TableContext
) -> Block | None:
    header = rows.header
    width = len(rows.body[0]) if rows.body else len(header)
    if not rows.own_header and ctx.header and len(ctx.header) == width:
        header = ctx.header
    if header:
        ctx.header = header
    markdown = table_to_markdown([header, *rows.body] if header else rows.body)
    if not markdown:
        return None
    text = f"{caption}\n{markdown}" if caption else markdown
    return Block(text=text, page_no=page_no, kind="table", section=section)


def is_scanned_page(page, *, min_cover: float = 0.8) -> bool:
    """Страница — картинка во всю площадь (скан), текст поверх неё невидимый.

    Такой текстовый слой — чужое распознавание, а не набранный текст: в нём
    «размешен па изделии» и «128,505» вместо «118,505». Доверять ему нельзя
    так же, как собственному OCR.
    """
    area = float(page.width * page.height) or 1.0
    return any(
        (img["x1"] - img["x0"]) * (img["bottom"] - img["top"]) / area >= min_cover
        for img in page.images
    )


def _page_blocks_pdf(
    page,
    page_no: int,
    section: str | None,
    repeated: set[str],
    gray=None,
    ctx: _TableContext | None = None,
) -> tuple[list[Block], str | None, str]:
    """Блоки одной PDF-страницы в порядке чтения + сырой текст страницы.

    ``gray`` — растр страницы: у сканов линовка таблиц есть только в нём.
    """
    ctx = ctx if ctx is not None else _TableContext()
    # Отбраковка — до разрезания страницы на полосы: текст сетки, которая
    # оказалась чертежом, должен остаться в текстовом потоке, а не пропасть.
    accepted = []
    raster = uses_raster_rules(page, gray)
    for table in sorted(find_tables(page, gray), key=lambda t: t.bbox[1]):
        if accepted and table.bbox[1] < accepted[-1][0].bbox[3]:
            continue  # вложенная или перекрывающаяся сетка — рамка внутри таблицы
        rows = table_rows(page, table, require_data=raster)
        if rows is not None:
            accepted.append((table, rows))

    blocks: list[Block] = []
    raw_parts: list[str] = []

    if not accepted:
        text = strip_repeated_lines(clean_page_text(page.extract_text() or ""), repeated)
        raw_parts.append(text)
        page_blocks, section = blocks_from_text(text, page_no, section)
        return page_blocks, section, "\n".join(raw_parts)

    cursor_top = 0.0

    for table, rows in accepted:
        _, top, _, bottom = table.bbox
        band = strip_repeated_lines(_crop_text(page, cursor_top, top), repeated)
        band, caption = _split_caption(band)
        if band:
            raw_parts.append(band)
            band_blocks, section = blocks_from_text(band, page_no, section)
            blocks.extend(band_blocks)

        block = _table_block(rows, page_no, section, caption, ctx)
        if block is not None:
            raw_parts.append(block.text)
            blocks.append(block)
        elif caption:
            raw_parts.append(caption)
            caption_blocks, section = blocks_from_text(caption, page_no, section)
            blocks.extend(caption_blocks)
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
    ctx = _TableContext()

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
            scanned = is_scanned_page(page)
            try:
                gray = _render_gray(path, index) if scanned else None
                blocks, section, raw = _page_blocks_pdf(page, index, section, repeated, gray, ctx)
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
                    # Текстовый слой скана — распознавание издателя: помечаем как
                    # OCR, чтобы предупреждение о возможных ошибках дошло до модели.
                    from_ocr=scanned and not needs_ocr,
                )
            )
    return pages


def _render_gray(path: Path, page_no: int):
    """Растр страницы для поиска линовки таблиц; None — если отрисовать не вышло."""
    try:
        from gost_rag.ingest.ocr import render_page

        return render_page(path, page_no, dpi=RULE_DPI)
    except Exception as exc:
        log.warning("render_failed", path=str(path), page=page_no, error=str(exc))
        return None


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
