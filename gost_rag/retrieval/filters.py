"""Фильтры к поиску, выведенные из текста запроса.

Если инженер пишет «по ГОСТ 14634», он называет документ явно — заставлять
векторный поиск угадывать его было бы расточительно. Обозначение вынимается
регуляркой и превращается в точный фильтр, но только если такой документ
действительно есть в индексе: иначе фильтр обнулил бы выдачу.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from gost_rag.config import Settings, get_settings
from gost_rag.ingest.metadata import _INTL, _PREFIX_ALT
from gost_rag.logging import get_logger

log = get_logger(__name__)

#: Обозначение без года: «ГОСТ 14634», «ГОСТ Р 1.2». Год необязателен, потому что
#: в запросах его почти никогда не пишут. Падежное окончание — «по ГОСТу 5264»,
#: «требования ГОСТа» — самая частая форма в живом вопросе, без него обозначение
#: не находилось вовсе.
LOOSE_DESIGNATION_RE = re.compile(
    rf"\b(?P<prefix>{_PREFIX_ALT})(?:ом|ов|у|а|е)?\s*"
    rf"(?:(?P<intl>{_INTL}(?:/{_INTL})?)\s*)?"
    rf"(?P<number>\d+(?:\.\d+)*)(?:\s*[-–—]\s*(?P<year>\d{{2,4}}))?\b",
    re.IGNORECASE,
)


def extract_designations(query: str) -> list[str]:
    """Найти в запросе обозначения стандартов (без года, если он не указан)."""
    found: list[str] = []
    for match in LOOSE_DESIGNATION_RE.finditer(query):
        parts = [re.sub(r"\s+", " ", match.group("prefix")).upper()]
        if intl := match.group("intl"):
            parts.append(intl.upper().replace("ISO", "ИСО").replace("IEC", "МЭК"))
        number = match.group("number")
        year = match.group("year")
        parts.append(f"{number}-{year}" if year else number)
        value = " ".join(parts)
        if value not in found:
            found.append(value)
    return found


def _same_edition(value: str, designation: str) -> bool:
    """«ГОСТ 14634» — любая редакция, «ГОСТ 14634-93» — только она."""
    value, designation = value.upper(), designation.upper()
    return designation == value or designation.startswith(f"{value}-")


def _base_number(value: str) -> str:
    """Обозначение без года: «ГОСТ 5264-80» -> «ГОСТ 5264»."""
    return re.sub(r"-\d{2,4}$", "", value.upper())


@dataclass(slots=True)
class DesignationMatch:
    """Что вопрос говорит о документах корпуса.

    ``missing`` — названные в вопросе документы, которых в корпусе нет. Раньше
    такой вопрос молча уходил в поиск по всему корпусу и получал уверенный ответ
    из другого стандарта: «что ГОСТ 16037-80 говорит о швах трубопроводов» —
    ответ по ГОСТ 5264-80, который трубопроводы как раз исключает.
    """

    mentioned: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    #: Для отсутствующего документа — другие редакции того же номера в корпусе.
    alternatives: dict[str, list[str]] = field(default_factory=dict)

    @property
    def only_missing(self) -> bool:
        """Вопрос называет документы, и ни одного из них в корпусе нет."""
        return bool(self.missing) and not self.matched

    @property
    def filter(self) -> models.Filter | None:
        if not self.matched:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="designation", match=models.MatchAny(any=sorted(set(self.matched)))
                )
            ]
        )


def resolve_designations(query: str, available: set[str]) -> DesignationMatch:
    """Сопоставить обозначения из вопроса с корпусом."""
    result = DesignationMatch(mentioned=extract_designations(query))
    for value in result.mentioned:
        found = sorted(d for d in available if _same_edition(value, d))
        if found:
            result.matched.extend(d for d in found if d not in result.matched)
            continue
        result.missing.append(value)
        base = _base_number(value)
        other = sorted(d for d in available if _base_number(d) == base)
        if other:
            result.alternatives[value] = other
    if result.missing:
        log.info("designation_not_in_corpus", missing=result.missing, matched=result.matched)
    if result.matched:
        log.info("designation_filter", designations=result.matched)
    return result


def known_designations(
    client: QdrantClient, settings: Settings | None = None, *, limit: int = 10_000
) -> set[str]:
    """Обозначения, реально присутствующие в индексе."""
    settings = settings or get_settings()
    if not client.collection_exists(settings.collection_name):
        return set()

    values: set[str] = set()
    offset = None
    scanned = 0
    while scanned < limit:
        points, offset = client.scroll(
            collection_name=settings.collection_name,
            limit=512,
            offset=offset,
            with_payload=["designation"],
            with_vectors=False,
        )
        for point in points:
            designation = (point.payload or {}).get("designation")
            if designation:
                values.add(designation)
        scanned += len(points)
        if offset is None:
            break
    return values


def build_filter(query: str, available: set[str]) -> models.Filter | None:
    """Собрать фильтр по обозначению, если оно найдено и есть в корпусе.

    Обозначение без года («ГОСТ 14634») сопоставляется с любой редакцией
    («ГОСТ 14634-93»): пользователь редко помнит год, а редакция в корпусе одна.
    Отсутствующий в корпусе документ фильтра не даёт — что делать с таким
    вопросом, решает граф по ``resolve_designations``.
    """
    return resolve_designations(query, available).filter
