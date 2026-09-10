"""Тесты графа: формат контекста, проверка цитат, отказ при слабых источниках."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from gost_rag.config import Settings
from gost_rag.graph import nodes as nodes_module
from gost_rag.graph.build import build_graph
from gost_rag.graph.nodes import Retrieval, build_prompt_messages, guard
from gost_rag.graph.prompts import NO_CONTEXT_MESSAGE, format_context
from gost_rag.graph.state import build_citations, strip_invalid_citations, used_indices
from gost_rag.models import RetrievedChunk


def _chunk(marker: str, designation: str, status: str = "действующий", **payload) -> RetrievedChunk:
    base = {
        "designation": designation,
        "status": status,
        "section": "3.2",
        "page_start": 7,
        "page_end": 7,
        "doc_id": designation.lower().replace(" ", "-"),
        "title": "Тестовый стандарт",
        "source_path": "/data/raw/x.pdf",
    }
    base.update(payload)
    return RetrievedChunk(
        point_id=f"id-{marker}", text=f"текст {marker}", payload=base, rerank_score=0.9
    )


# --------------------------------------------------------------------------- #
# Формат контекста
# --------------------------------------------------------------------------- #


def test_context_blocks_are_numbered_from_one():
    context = format_context([_chunk("a", "ГОСТ 14634-93"), _chunk("b", "ГОСТ 1050-88")])
    assert "[S1] ГОСТ 14634-93" in context
    assert "[S2] ГОСТ 1050-88" in context


def test_citation_label_carries_status_clause_and_page():
    label = _chunk("a", "ГОСТ 14634-93").citation_label()
    assert label == "ГОСТ 14634-93 (действующий), п. 3.2, стр. 7"


def test_page_range_rendered_when_chunk_spans_pages():
    label = _chunk("a", "ГОСТ 1-11", page_start=7, page_end=9).citation_label()
    assert "стр. 7–9" in label


def test_ocr_chunks_are_marked_in_context():
    context = format_context([_chunk("a", "ГОСТ 1-11", ocr=True)])
    assert "распознано OCR" in context


def test_withdrawn_status_visible_to_model():
    context = format_context([_chunk("a", "ГОСТ 1050-88", status="заменён")])
    assert "(заменён)" in context


# --------------------------------------------------------------------------- #
# Проверка цитат
# --------------------------------------------------------------------------- #


def test_used_indices_are_deduplicated_and_ordered():
    assert used_indices("а [S2] б [S1] в [S2]", available=3) == [2, 1]


def test_out_of_range_citation_is_ignored():
    assert used_indices("ответ [S9]", available=3) == []


def test_invalid_citation_marker_is_stripped_from_answer():
    answer = strip_invalid_citations("Радиус 6 мм [S1], допуск ±0,2 [S9].", available=1)
    assert "[S1]" in answer
    assert "[S9]" not in answer
    # После удаления маркера не должно остаться пробела перед точкой.
    assert "±0,2." in answer


def test_valid_citations_survive_untouched():
    answer = "Радиус 6 мм [S1][S2]."
    assert strip_invalid_citations(answer, available=2) == answer


def test_build_citations_returns_only_referenced_sources():
    chunks = [_chunk("a", "ГОСТ 14634-93"), _chunk("b", "ГОСТ 1050-88")]
    citations = build_citations("вывод [S2]", chunks)
    assert len(citations) == 1
    assert citations[0]["marker"] == "S2"
    assert citations[0]["designation"] == "ГОСТ 1050-88"


def test_citation_carries_fields_needed_by_ui():
    citations = build_citations("[S1]", [_chunk("a", "ГОСТ 14634-93", replaced_by="ГОСТ 1-2020")])
    citation = citations[0]
    for field in ("label", "status", "page_start", "source_path", "text", "doc_id", "replaced_by"):
        assert field in citation


def test_answer_without_citations_yields_none():
    assert build_citations("ответ без ссылок", [_chunk("a", "ГОСТ 1-11")]) == []


# --------------------------------------------------------------------------- #
# Guard
# --------------------------------------------------------------------------- #


def test_guard_routes_to_generate_when_sources_exist():
    assert guard({"reranked": [_chunk("a", "ГОСТ 1-11")]}) == "generate"


def test_guard_routes_to_refuse_when_nothing_passed_threshold():
    assert guard({"reranked": []}) == "refuse"
    assert guard({}) == "refuse"


# --------------------------------------------------------------------------- #
# История диалога
# --------------------------------------------------------------------------- #


def test_history_is_trimmed_to_configured_turns():
    settings = Settings(history_turns=1)
    state = {
        "question": "новый вопрос",
        "reranked": [_chunk("a", "ГОСТ 1-11")],
        "messages": [
            HumanMessage(content="старый вопрос"),
            AIMessage(content="старый ответ"),
            HumanMessage(content="предыдущий вопрос"),
            AIMessage(content="предыдущий ответ"),
        ],
    }
    messages = build_prompt_messages(state, settings)
    rendered = [m.content for m in messages]
    assert "предыдущий вопрос" in rendered
    assert "старый вопрос" not in rendered


def test_question_and_context_are_in_final_message():
    settings = Settings(history_turns=0)
    state = {"question": "какой радиус гибки", "reranked": [_chunk("a", "ГОСТ 14634-93")]}
    last = build_prompt_messages(state, settings)[-1]
    assert "какой радиус гибки" in last.content
    assert "[S1] ГОСТ 14634-93" in last.content


# --------------------------------------------------------------------------- #
# Граф целиком
# --------------------------------------------------------------------------- #


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.answer)


class FakeReranker:
    def __init__(self, keep: list[RetrievedChunk]) -> None:
        self.keep = keep

    def rerank(self, query, candidates, **kwargs):
        return self.keep


@pytest.fixture
def patched_retrieval(monkeypatch):
    """Подменяем поиск: граф проверяется без Qdrant и без моделей."""
    found: list[RetrievedChunk] = []

    monkeypatch.setattr(nodes_module, "hybrid_search", lambda *a, **k: list(found))
    monkeypatch.setattr(nodes_module, "build_filter", lambda *a, **k: None)
    monkeypatch.setattr(nodes_module, "known_designations", lambda *a, **k: set())

    class FakeEmbedder:
        def encode_one(self, text):
            return object()

    def make(keep: list[RetrievedChunk]):
        found.clear()
        found.extend(keep)
        return Retrieval(
            client=object(),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(keep),
            settings=Settings(history_turns=0),
        )

    return make


def _invoke(graph, question: str, thread: str = "t1"):
    return graph.invoke({"question": question}, config={"configurable": {"thread_id": thread}})


def test_graph_answers_and_returns_citations(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 14634-93")]
    llm = FakeLLM("Радиус гибки — 2S [S1].")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=0))

    result = _invoke(graph, "какой радиус гибки")
    assert result["answer"] == "Радиус гибки — 2S [S1]."
    assert result["insufficient"] is False
    assert [c["designation"] for c in result["citations"]] == ["ГОСТ 14634-93"]


def test_graph_refuses_without_sources_and_never_calls_llm(patched_retrieval):
    llm = FakeLLM("этого мы не должны увидеть")
    graph = build_graph(patched_retrieval([]), llm, Settings(history_turns=0))

    result = _invoke(graph, "чего нет в корпусе")
    assert result["answer"] == NO_CONTEXT_MESSAGE
    assert result["insufficient"] is True
    assert result["citations"] == []
    assert llm.calls == 0


def test_graph_strips_hallucinated_citation_before_returning(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 14634-93")]
    llm = FakeLLM("Радиус 6 мм [S1], а допуск ±0,2 [S7].")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=0))

    result = _invoke(graph, "радиус и допуск")
    assert "[S7]" not in result["answer"]
    assert "[S1]" in result["answer"]
    assert len(result["citations"]) == 1


def test_thread_keeps_history_between_turns(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 14634-93")]
    llm = FakeLLM("Ответ [S1].")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=4))

    _invoke(graph, "первый вопрос", thread="t-shared")
    result = _invoke(graph, "второй вопрос", thread="t-shared")
    assert len(result["messages"]) >= 2
