"""Паспорт документа: обозначение, год, тип, статус.

Обозначение ищется в имени файла и на первой странице. Статус («действующий»,
«отменён», «заменён») не выводится из текста — он берётся из реестра
``data/registry/documents.csv``, потому что ответ по отменённому стандарту хуже,
чем отсутствие ответа, и полагаться тут на догадки нельзя.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from gost_rag.logging import get_logger
from gost_rag.models import DocStatus, DocumentMeta

log = get_logger(__name__)

#: Префиксы российских НТД. Порядок важен: «ГОСТ Р» должен проверяться до «ГОСТ».
_PREFIXES = ["ГОСТ Р", "ГОСТ", "ОСТ", "СТО", "СТП", "СНиП", "СП", "РД", "ТУ", "ЕСКД"]
_PREFIX_ALT = "|".join(p.replace(" ", r"\s+") for p in _PREFIXES)

#: Международные обозначения внутри российских: ГОСТ Р ИСО 9001-2015.
_INTL = r"(?:ISO|ИСО|IEC|МЭК|EN|ЕН)"

DESIGNATION_RE = re.compile(
    rf"\b(?P<prefix>{_PREFIX_ALT})\s*(?P<intl>{_INTL}(?:/{_INTL})?)?\s*"
    rf"(?P<number>\d+(?:\.\d+)*)\s*[-–—]\s*(?P<year>\d{{2,4}})\b",
    re.IGNORECASE,
)

_VALID_STATUSES: set[str] = {"действующий", "отменён", "заменён", "неизвестно"}


def normalize_designation(match: re.Match[str]) -> str:
    """Привести обозначение к каноничному виду: «ГОСТ Р ИСО 9001-2015».

    Год в обозначении остаётся ровно таким, как в документе: «ГОСТ 14634-93» —
    это и есть каноничная запись, а не сокращение от «-1993». Разворачивать её
    нельзя, иначе обозначение перестанет совпадать с реестром и с тем, как его
    пишет пользователь в запросе. Четырёхзначный год живёт в поле ``year``.
    """
    prefix = re.sub(r"\s+", " ", match.group("prefix")).upper()
    parts = [prefix]
    if intl := match.group("intl"):
        parts.append(intl.upper().replace("ISO", "ИСО").replace("IEC", "МЭК"))
    parts.append(f"{match.group('number')}-{match.group('year')}")
    return " ".join(parts)


def expand_year(token: str) -> int:
    """Двузначный год приводим к четырём: 93 -> 1993, 15 -> 2015."""
    if len(token) == 4:
        return int(token)
    value = int(token)
    return 1900 + value if value > 30 else 2000 + value


def find_designation(text: str) -> str | None:
    match = DESIGNATION_RE.search(text)
    return normalize_designation(match) if match else None


def parse_year(designation: str | None) -> int | None:
    if not designation:
        return None
    match = re.search(r"-(\d{2,4})$", designation)
    return expand_year(match.group(1)) if match else None


def parse_doc_type(designation: str | None) -> str | None:
    if not designation:
        return None
    for prefix in _PREFIXES:
        if designation.upper().startswith(prefix.upper()):
            return prefix
    return None


def make_doc_id(designation: str | None, path: Path) -> str:
    """Стабильный идентификатор: по обозначению, иначе по имени файла."""
    source = designation or path.stem
    slug = re.sub(r"[^\w]+", "-", source, flags=re.UNICODE).strip("-").lower()
    return slug or "document"


# --------------------------------------------------------------------------- #
# Реестр
# --------------------------------------------------------------------------- #

REGISTRY_FIELDS = ["designation", "title", "year", "status", "source_url", "replaced_by"]


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    """Прочитать реестр документов. Отсутствие файла — не ошибка."""
    if not path.exists():
        log.info("registry_missing", path=str(path))
        return {}

    registry: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            designation = (row.get("designation") or "").strip()
            if not designation:
                continue
            registry[_registry_key(designation)] = {k: (v or "").strip() for k, v in row.items()}
    log.info("registry_loaded", path=str(path), documents=len(registry))
    return registry


def _registry_key(designation: str) -> str:
    return re.sub(r"\s+", " ", designation).strip().upper()


#: Написания статуса, встречающиеся в реестрах, -> каноничное значение.
#: Ключи хранятся без «ё»: в CSV его пишут через раз.
_STATUS_ALIASES: dict[str, DocStatus] = {
    "деиствующии": "действующий",
    "деиствует": "действующий",
    "деиствующая": "действующий",
    "актуальныи": "действующий",
    "отменен": "отменён",
    "отмененныи": "отменён",
    "отменена": "отменён",
    "заменен": "заменён",
    "замененныи": "заменён",
    "заменена": "заменён",
    "неизвестно": "неизвестно",
}


def _fold_status(value: str) -> str:
    """Убрать «ё» и «й» из различий написания, чтобы сравнивать варианты."""
    return value.strip().casefold().replace("ё", "е").replace("й", "и")


def _coerce_status(raw: str) -> DocStatus:
    return _STATUS_ALIASES.get(_fold_status(raw), "неизвестно")


def build_metadata(
    path: Path,
    first_page_text: str,
    registry: dict[str, dict[str, str]] | None = None,
    *,
    sha256: str | None = None,
) -> DocumentMeta:
    """Собрать паспорт документа: имя файла имеет приоритет над текстом страницы.

    Имя файла обычно задаётся человеком и надёжнее, чем первая страница скана,
    где обозначение может быть распознано с ошибками.
    """
    designation = find_designation(path.name) or find_designation(first_page_text[:3000])
    registry = registry or {}
    row = registry.get(_registry_key(designation)) if designation else None

    status: DocStatus = _coerce_status(row["status"]) if row and row.get("status") else "неизвестно"
    title = (row.get("title") if row else None) or _guess_title(first_page_text, designation)
    # Реестр надёжнее обозначения: в нём год редакции, а не год из шифра.
    year = int(row["year"]) if row and row.get("year", "").isdigit() else parse_year(designation)

    return DocumentMeta(
        doc_id=make_doc_id(designation, path),
        designation=designation,
        title=title,
        year=year,
        status=status,
        doc_type=parse_doc_type(designation),
        source_path=str(path),
        source_url=(row.get("source_url") or None) if row else None,
        replaced_by=(row.get("replaced_by") or None) if row else None,
        sha256=sha256,
    )


def _guess_title(text: str, designation: str | None) -> str | None:
    """Заголовок — первая содержательная строка после обозначения на титуле."""
    if not text:
        return None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    start = 0
    if designation:
        for i, line in enumerate(lines):
            if find_designation(line):
                start = i + 1
                break
    for line in lines[start : start + 6]:
        letters = re.sub(r"[^А-Яа-яЁё]", "", line)
        if len(letters) >= 10:
            return line[:300]
    return None
