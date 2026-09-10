"""Состояние графа и проверка цитат."""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from gost_rag.models import RetrievedChunk

#: Маркер источника в ответе модели: [S1], [S12].
CITATION_RE = re.compile(r"\[S(\d+)\]")


class GraphState(TypedDict, total=False):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    answer: str
    citations: list[dict[str, Any]]
    insufficient: bool
    best_score: float


def used_indices(answer: str, available: int) -> list[int]:
    """Номера источников, на которые модель реально сослалась (1-based).

    Ссылки за пределами списка отбрасываются: [S7] при шести фрагментах — это
    выдумка, и подставлять под неё случайный чанк нельзя.
    """
    seen: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= available and index not in seen:
            seen.append(index)
    return seen


def strip_invalid_citations(answer: str, available: int) -> str:
    """Убрать из текста ссылки на несуществующие фрагменты."""

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= available else ""

    cleaned = CITATION_RE.sub(replace, answer)
    # Подчистить пробелы, оставшиеся от удалённых маркеров.
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return re.sub(r" +([.,;:])", r"\1", cleaned).strip()


def build_citations(answer: str, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Собрать структурированные цитаты для тех источников, что реально упомянуты."""
    citations: list[dict[str, Any]] = []
    for index in used_indices(answer, len(chunks)):
        chunk = chunks[index - 1]
        citations.append(
            {
                "marker": f"S{index}",
                "point_id": chunk.point_id,
                "label": chunk.citation_label(),
                "designation": chunk.designation,
                "title": chunk.payload.get("title"),
                "status": chunk.status,
                "replaced_by": chunk.payload.get("replaced_by"),
                "section": chunk.payload.get("section"),
                "page_start": chunk.payload.get("page_start"),
                "page_end": chunk.payload.get("page_end"),
                "doc_id": chunk.payload.get("doc_id"),
                "source_path": chunk.payload.get("source_path"),
                "source_url": chunk.payload.get("source_url"),
                "ocr": bool(chunk.payload.get("ocr")),
                "rerank_score": chunk.rerank_score,
                "text": chunk.text,
            }
        )
    return citations
