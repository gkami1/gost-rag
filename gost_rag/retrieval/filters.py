"""Фильтры к поиску, выведенные из текста запроса.

Если инженер пишет «по ГОСТ 14634», он называет документ явно — заставлять
векторный поиск угадывать его было бы расточительно. Обозначение вынимается
регуляркой и превращается в точный фильтр, но только если такой документ
действительно есть в индексе: иначе фильтр обнулил бы выдачу.
"""

from __future__ import annotations

import re

from qdrant_client import QdrantClient, models

from gost_rag.config import Settings, get_settings
from gost_rag.ingest.metadata import _PREFIX_ALT
from gost_rag.logging import get_logger

log = get_logger(__name__)

#: Обозначение без года: «ГОСТ 14634», «ГОСТ Р 1.2». Год необязателен, потому что
#: в запросах его почти никогда не пишут.
LOOSE_DESIGNATION_RE = re.compile(
    rf"\b(?P<prefix>{_PREFIX_ALT})\s*(?P<number>\d+(?:\.\d+)*)(?:\s*[-–—]\s*(?P<year>\d{{2,4}}))?\b",
    re.IGNORECASE,
)


def extract_designations(query: str) -> list[str]:
    """Найти в запросе обозначения стандартов (без года, если он не указан)."""
    found: list[str] = []
    for match in LOOSE_DESIGNATION_RE.finditer(query):
        prefix = re.sub(r"\s+", " ", match.group("prefix")).upper()
        number = match.group("number")
        year = match.group("year")
        value = f"{prefix} {number}-{year}" if year else f"{prefix} {number}"
        if value not in found:
            found.append(value)
    return found


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
    """
    mentioned = extract_designations(query)
    if not mentioned or not available:
        return None

    matched: list[str] = []
    for value in mentioned:
        exact = [d for d in available if d.upper() == value.upper()]
        prefixed = [d for d in available if d.upper().startswith(f"{value.upper()}-")]
        matched.extend(exact or prefixed)

    if not matched:
        log.info("designation_not_in_corpus", mentioned=mentioned)
        return None

    log.info("designation_filter", designations=matched)
    return models.Filter(
        must=[
            models.FieldCondition(
                key="designation", match=models.MatchAny(any=sorted(set(matched)))
            )
        ]
    )
