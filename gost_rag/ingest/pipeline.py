"""Конвейер индексации: файл -> страницы -> блоки -> чанки -> Qdrant.

Запуск повторно безопасен: ID точки выводится из хэша текста, поэтому повторная
запись того же чанка ничего не дублирует. Журнал в ``data/interim/ingest_ledger.jsonl``
позволяет продолжить прерванный прогон, не перечитывая уже разобранные файлы.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer

from gost_rag.config import Settings, get_settings
from gost_rag.ingest import ocr as ocr_module
from gost_rag.ingest.chunk import ApproxTokenizer, HFTokenizer, Tokenizer, chunk_blocks
from gost_rag.ingest.embed import BGEM3Embedder
from gost_rag.ingest.index import (
    build_points,
    count_points,
    delete_document,
    document_sources,
    ensure_collection,
    get_client,
    prune_missing_sources,
    upsert_points,
)
from gost_rag.ingest.loaders import (
    SUPPORTED_SUFFIXES,
    assign_sections,
    blocks_from_text,
    file_sha256,
    load_document,
)
from gost_rag.ingest.metadata import build_metadata, load_registry
from gost_rag.ingest.normalize import (
    clean_page_text,
    find_repeated_lines,
    strip_repeated_lines,
)
from gost_rag.logging import configure_logging, get_logger
from gost_rag.models import Block, Chunk, DocumentMeta, PageDoc

log = get_logger(__name__)

LEDGER_NAME = "ingest_ledger.jsonl"

#: Версия конвейера разбора. Поднимать при любом изменении, после которого старый
#: индекс перестаёт соответствовать коду: разбор таблиц, чанкинг, поля payload.
#: Журнал сравнивает её вместе с настройками и переиндексирует файл, даже если
#: сам файл не менялся.
PIPELINE_VERSION = "2026-09-28"


def index_fingerprint(settings: Settings, *, approx_tokens: bool = False) -> str:
    """Отпечаток всего, от чего зависят чанки и векторы, кроме самого файла.

    Раньше журнал помнил только путь и хэш файла: после смены размера чанка,
    модели или разбора таблиц повторная индексация молча пропускала всё, и
    индекс оставался собранным старым кодом.
    """
    fields = {
        "version": PIPELINE_VERSION,
        "tokenizer": "approx" if approx_tokens else settings.embedding_model,
        "embedding_model": settings.embedding_model,
        "embed_max_length": settings.embed_max_length,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "min_chunk_tokens": settings.min_chunk_tokens,
        "ocr_enabled": settings.ocr_enabled,
        "ocr_lang": settings.ocr_lang,
        "ocr_dpi": settings.ocr_dpi,
        "ocr_min_chars": settings.ocr_min_chars,
    }
    raw = json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


@dataclass(slots=True)
class IngestReport:
    path: str
    doc_id: str
    designation: str | None
    status: str
    pages: int
    ocr_pages: int
    chunks: int
    indexed: int
    skipped: bool = False
    error: str | None = None


# --------------------------------------------------------------------------- #
# Журнал
# --------------------------------------------------------------------------- #


def _ledger_path(settings: Settings) -> Path:
    settings.interim_dir.mkdir(parents=True, exist_ok=True)
    return settings.interim_dir / LEDGER_NAME


def read_ledger(settings: Settings) -> dict[str, dict[str, str]]:
    """Путь -> {sha256, fingerprint} последней успешной индексации файла.

    Записи без отпечатка (журнал до его появления) отпечатка не совпадут ни с
    каким текущим — такие файлы переиндексируются, и это правильно.
    """
    path = _ledger_path(settings)
    if not path.exists():
        return {}
    done: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("sha256") and not record.get("error"):
                done[record["path"]] = {
                    "sha256": record["sha256"],
                    "fingerprint": record.get("fingerprint", ""),
                }
    return done


def append_ledger(
    settings: Settings, report: IngestReport, sha256: str, fingerprint: str = ""
) -> None:
    record = asdict(report) | {
        "sha256": sha256,
        "fingerprint": fingerprint,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with _ledger_path(settings).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #


def apply_ocr(path: Path, pages: list[PageDoc], settings: Settings) -> int:
    """Распознать страницы без текстового слоя. Возвращает число распознанных."""
    pending = [page for page in pages if page.needs_ocr]
    if not pending:
        return 0

    if path.suffix.lower() != ".pdf":
        return 0

    if not settings.ocr_enabled:
        log.warning("ocr_disabled", path=str(path), pages=len(pending))
        return 0

    ocr_module.configure_tesseract(settings.tesseract_cmd)
    if not ocr_module.ocr_available(settings.ocr_lang):
        log.warning(
            "ocr_skipped_no_tesseract",
            path=str(path),
            pages=len(pending),
            hint="установите Tesseract и языковой пакет rus — иначе сканы не попадут в индекс",
        )
        return 0

    recognised_pages: list[tuple[PageDoc, str, float | None]] = []
    for page in pages:
        if not page.needs_ocr:
            continue
        result = ocr_module.ocr_page(
            path, page.page_no, lang=settings.ocr_lang, dpi=settings.ocr_dpi
        )
        # Та же чистка, что у текстового слоя: без неё переносы «поверх-ность» и
        # разрядка «Т А Б Л И Ц А» доходили до индекса только со сканов.
        text = clean_page_text(result.text)
        if text:
            recognised_pages.append((page, text, result.confidence))

    # Колонтитулы сканов ищутся по самим сканам: у страниц текстового слоя они
    # уже вырезаны загрузчиком, и вместе те и другие порога не наберут. Без этого
    # «С. 2 ГОСТ 5264-80» оставался в тексте каждой распознанной страницы.
    repeated = find_repeated_lines(text for _, text, _ in recognised_pages)
    section: str | None = None
    for page, text, confidence in recognised_pages:
        text = strip_repeated_lines(text, repeated)
        blocks, section = blocks_from_text(text, page.page_no, section)
        page.text = text
        page.blocks = blocks
        page.from_ocr = True
        page.ocr_confidence = confidence
    recognised = len(recognised_pages)

    log.info("ocr_done", path=str(path), pages=recognised)
    return recognised


# --------------------------------------------------------------------------- #
# Один документ
# --------------------------------------------------------------------------- #


def collect_blocks(pages: list[PageDoc]) -> list[Block]:
    """Блоки документа в порядке страниц с пунктами, размеченными по всему документу.

    Разметка делается здесь, после OCR, а не в загрузчике: только теперь блоки
    текстового слоя и сканов стоят в одном ряду, и нумерация у них общая.
    """
    blocks = [block for page in pages for block in page.blocks]
    assign_sections(blocks)
    return blocks


def chunk_document(
    pages: list[PageDoc], meta: DocumentMeta, tokenizer: Tokenizer, settings: Settings
) -> list[Chunk]:
    return chunk_blocks(
        collect_blocks(pages),
        meta.doc_id,
        tokenizer,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_chunk_tokens=settings.min_chunk_tokens,
        ocr_pages={p.page_no for p in pages if p.from_ocr},
        ocr_confidence={
            p.page_no: p.ocr_confidence
            for p in pages
            if p.from_ocr and p.ocr_confidence is not None
        },
    )


def ingest_file(
    path: Path,
    *,
    client,
    embedder: BGEM3Embedder,
    tokenizer: Tokenizer,
    registry: dict[str, dict[str, str]],
    settings: Settings,
) -> IngestReport:
    sha = file_sha256(path)
    pages = load_document(path, ocr_min_chars=settings.ocr_min_chars)
    ocr_pages = apply_ocr(path, pages, settings)

    first_text = next((p.text for p in pages if p.text.strip()), "")
    meta = build_metadata(path, first_text, registry, sha256=sha)
    chunks = chunk_document(pages, meta, tokenizer, settings)

    if not chunks:
        return IngestReport(
            path=str(path),
            doc_id=meta.doc_id,
            designation=meta.designation,
            status=meta.status,
            pages=len(pages),
            ocr_pages=ocr_pages,
            chunks=0,
            indexed=0,
            error="не удалось извлечь текст (нужен OCR?)",
        )

    # Два файла с одним обозначением («ГОСТ 5264-80.pdf» и «ГОСТ 5264-80 Изм. 1.pdf»)
    # дают один doc_id, и удаление старых чанков второго стирало первый. Занятый
    # другим живым файлом doc_id — ошибка, которую должен разрешить человек.
    owners = {
        source
        for source in document_sources(client, meta.doc_id, settings)
        if Path(source).resolve() != path.resolve() and Path(source).exists()
    }
    if owners:
        return IngestReport(
            path=str(path),
            doc_id=meta.doc_id,
            designation=meta.designation,
            status=meta.status,
            pages=len(pages),
            ocr_pages=ocr_pages,
            chunks=len(chunks),
            indexed=0,
            error=(
                f"{meta.designation or meta.doc_id} уже проиндексирован из "
                f"{sorted(owners)[0]} — переименуйте файл или уберите дубликат"
            ),
        )

    # Сначала эмбеддинги — самый долгий и самый хрупкий шаг (память, Ctrl+C).
    # Если он упадёт после удаления, документ исчезнет из индекса до следующего
    # прогона; до удаления — останется старая версия.
    embeddings = embedder.encode([c.text for c in chunks])

    # Документ мог измениться: старые чанки с другим текстом остались бы мусором,
    # потому что их ID зависит от содержимого и перезаписью они не затрутся.
    delete_document(client, meta.doc_id, settings)
    indexed = upsert_points(client, build_points(chunks, embeddings, meta, settings), settings)

    return IngestReport(
        path=str(path),
        doc_id=meta.doc_id,
        designation=meta.designation,
        status=meta.status,
        pages=len(pages),
        ocr_pages=ocr_pages,
        chunks=len(chunks),
        indexed=indexed,
    )


def discover_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

app = typer.Typer(add_completion=False, help="Индексация ГОСТов и документации в Qdrant.")


@app.command()
def run(
    target: Path = typer.Argument(None, help="Файл или папка (по умолчанию data/raw)"),
    recreate: bool = typer.Option(False, "--recreate", help="Пересоздать коллекцию с нуля"),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Пропускать уже разобранные файлы"
    ),
    approx_tokens: bool = typer.Option(
        False, "--approx-tokens", help="Считать токены грубо, не загружая токенайзер BGE-M3"
    ),
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    target = target or settings.raw_dir
    # Абсолютные пути: по ним журнал узнаёт файл при запуске из другого каталога,
    # а /api/source и чистка индекса находят исходник.
    files = [path.resolve() for path in discover_files(target)]
    if not files:
        typer.secho(f"В {target} нет PDF/DOCX для индексации.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    registry = load_registry(settings.registry_path)
    done = read_ledger(settings) if resume else {}

    client = get_client(settings)
    ensure_collection(client, settings, recreate=recreate)
    if recreate:
        done = {}

    tokenizer: Tokenizer = (
        ApproxTokenizer() if approx_tokens else HFTokenizer(settings.embedding_model)
    )
    embedder = BGEM3Embedder(settings)
    fingerprint = index_fingerprint(settings, approx_tokens=approx_tokens)

    reports: list[IngestReport] = []
    for path in files:
        sha = file_sha256(path)
        entry = done.get(str(path))
        if entry and entry["sha256"] == sha and entry["fingerprint"] == fingerprint:
            typer.echo(f"= пропуск (не изменился): {path.name}")
            reports.append(IngestReport(str(path), "", None, "", 0, 0, 0, 0, skipped=True))
            continue
        if entry and entry["sha256"] == sha:
            typer.echo(f"↻ {path.name}: изменились настройки или версия конвейера")

        typer.echo(f"→ {path.name}")
        try:
            report = ingest_file(
                path,
                client=client,
                embedder=embedder,
                tokenizer=tokenizer,
                registry=registry,
                settings=settings,
            )
        except Exception as exc:
            log.exception("ingest_failed", path=str(path))
            report = IngestReport(str(path), "", None, "", 0, 0, 0, 0, error=str(exc))

        reports.append(report)
        append_ledger(settings, report, sha, fingerprint)
        _echo_report(report)

    if target.is_dir():
        for doc_id in prune_missing_sources(client, target, settings):
            typer.secho(f"- удалён из индекса (файла больше нет): {doc_id}", fg=typer.colors.YELLOW)

    _echo_summary(reports, count_points(client, settings))
    client.close()


def _echo_report(report: IngestReport) -> None:
    if report.error:
        typer.secho(f"  ! {report.error}", fg=typer.colors.RED)
        return
    warn = " ⚠ статус: " + report.status if report.status in {"отменён", "заменён"} else ""
    ocr = f", OCR стр.: {report.ocr_pages}" if report.ocr_pages else ""
    typer.secho(
        f"  {report.designation or report.doc_id}: стр. {report.pages}{ocr}, "
        f"чанков {report.chunks}{warn}",
        fg=typer.colors.GREEN,
    )


def _echo_summary(reports: list[IngestReport], total_points: int) -> None:
    ok = [r for r in reports if not r.error and not r.skipped]
    failed = [r for r in reports if r.error]
    skipped = [r for r in reports if r.skipped]
    typer.echo("")
    typer.secho(
        f"Готово: документов {len(ok)}, пропущено {len(skipped)}, ошибок {len(failed)}; "
        f"чанков в индексе {total_points}.",
        fg=typer.colors.CYAN,
        bold=True,
    )
    for report in failed:
        typer.secho(f"  ошибка: {Path(report.path).name} — {report.error}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
