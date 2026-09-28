"""Состояние графа и проверка цитат."""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from gost_rag.models import RetrievedChunk
from gost_rag.retrieval.filters import LOOSE_DESIGNATION_RE

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
    #: Документы, названные в вопросе, которых нет в корпусе.
    missing_designations: list[str]
    #: Другие редакции отсутствующих документов, которые в корпусе есть.
    designation_alternatives: dict[str, list[str]]
    #: Предупреждения проверки ответа — показываются пользователю вместе с ним.
    warnings: list[str]


#: Число в тексте: «9,026», «0.5», «40», «±2». Буква вплотную перед числом
#: («M10», «d2», «S1») делает его частью обозначения, а не значением.
NUMBER_RE = re.compile(r"(?<![\w,.])[-+±]?\d+(?:[.,]\d+)?")


def _canonical_numbers(text: str) -> set[str]:
    """Числа текста в едином виде: десятичная запятая, без знака.

    Текстовый слой пишет то «0,5», то «0.5», модель — как ей удобнее; сравнивать
    надо значения, а не написание.
    """
    return {m.group(0).lstrip("-+±").replace(".", ",") for m in NUMBER_RE.finditer(text)}


def ungrounded_numbers(answer: str, sources: list[RetrievedChunk], question: str = "") -> list[str]:
    """Числа ответа, которых нет ни в одном из процитированных фрагментов.

    Самая дешёвая и самая ценная проверка для этой предметной области: ответ —
    это почти всегда размер, допуск или температура, и правдоподобная, но
    выдуманная цифра со ссылкой [S1] выглядит так же убедительно, как настоящая.
    Число разрешено, если оно есть в тексте фрагмента, в его подписи (обозначение,
    пункт, страница) или в самом вопросе. Обозначения стандартов в ответе
    («ГОСТ 10549-80») не проверяются: это не значения, а имена документов, и
    ссылка на соседний стандарт из текста фрагмента — не выдумка размера.
    """
    allowed = _canonical_numbers(question)
    for chunk in sources:
        allowed |= _canonical_numbers(chunk.text)
        allowed |= _canonical_numbers(chunk.citation_label())
    body = LOOSE_DESIGNATION_RE.sub(" ", CITATION_RE.sub(" ", answer))
    missing: list[str] = []
    for match in NUMBER_RE.finditer(body):
        value = match.group(0).lstrip("-+±")
        if value.replace(".", ",") not in allowed and value not in missing:
            missing.append(value)
    return missing


#: Модель прямо говорит, что ответа во фрагментах нет. Проверяется только
#: начало ответа: дальше она обычно поясняет, что во фрагментах всё-таки есть.
_NO_ANSWER_RE = re.compile(
    r"ответа\s+нет|нет\s+ответа|нет\s+(?:сведений|данных)|(?:сведени|данны)\w*\s[^.]{0,80}?"
    r"(?:нет|отсутству)|не\s+содерж|отсутству|не\s+указан|не\s+привод|невозможно",
    re.IGNORECASE,
)


def says_no_answer(answer: str) -> bool:
    """Ответ по существу — отказ, даже если модель сослалась на фрагменты.

    На вопросах вне корпуса модель почти всегда отвечает «во фрагментах ответа
    нет» и тут же цитирует фрагмент, объясняя, что в нём есть вместо этого.
    По ссылкам такой ответ неотличим от настоящего, и пользователь, и оценка
    считали его ответом.
    """
    first = CITATION_RE.sub("", answer.strip().split("\n\n", 1)[0])[:400]
    return _NO_ANSWER_RE.search(first) is not None


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


def invalid_citations(answer: str, available: int) -> list[int]:
    """Номера ссылок за пределами списка фрагментов, в порядке появления."""
    seen: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if not 1 <= index <= available and index not in seen:
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


def strip_citation_markers(answer: str) -> str:
    """Убрать все маркеры [S#] — для ответов прошлых ходов, где они уже ничего не значат."""
    return strip_invalid_citations(answer, 0)


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
                "sections": chunk.payload.get("sections") or [],
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
