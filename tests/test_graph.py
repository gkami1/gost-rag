"""Тесты графа: формат контекста, проверка цитат, отказ при слабых источниках."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from gost_rag.config import Settings
from gost_rag.graph import nodes as nodes_module
from gost_rag.graph.build import build_graph
from gost_rag.graph.nodes import Retrieval, build_prompt_messages, guard
from gost_rag.graph.prompts import NO_CONTEXT_MESSAGE, format_context
from gost_rag.graph.state import (
    build_citations,
    strip_invalid_citations,
    ungrounded_numbers,
    used_indices,
)
from gost_rag.models import RetrievedChunk


def _chunk(
    marker: str,
    designation: str,
    status: str = "действующий",
    *,
    text: str | None = None,
    **payload,
) -> RetrievedChunk:
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
        point_id=f"id-{marker}",
        text=text if text is not None else f"текст {marker}",
        payload=base,
        rerank_score=0.9,
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


def test_clause_range_rendered_when_chunk_spans_clauses():
    chunk = _chunk("a", "ГОСТ 1-11")
    chunk.payload["sections"] = ["3.3.13", "3.4.1", "3.4.7"]
    assert "пп. 3.3.13–3.4.7" in chunk.citation_label()


def test_single_clause_in_sections_rendered_as_one():
    chunk = _chunk("a", "ГОСТ 1-11")
    chunk.payload["sections"] = ["3.2"]
    assert "п. 3.2," in chunk.citation_label()


def test_legacy_payload_without_sections_uses_section():
    """Индекс, собранный до появления ``sections``, не должен терять номер пункта."""
    chunk = _chunk("a", "ГОСТ 1-11")
    chunk.payload.pop("sections", None)
    assert "п. 3.2" in chunk.citation_label()


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
    searches: list[object] = []

    def fake_search(*args, **kwargs):
        searches.append(kwargs.get("query_filter"))
        return list(found)

    monkeypatch.setattr(nodes_module, "hybrid_search", fake_search)

    class FakeEmbedder:
        def encode_one(self, text):
            return object()

    def make(keep: list[RetrievedChunk], corpus: set[str] | None = None):
        found.clear()
        found.extend(keep)
        monkeypatch.setattr(nodes_module, "known_designations", lambda *a, **k: set(corpus or ()))
        make.searches = searches
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
    chunks = [_chunk("a", "ГОСТ 14634-93", text="Радиус гибки принимают равным 2S.")]
    llm = FakeLLM("Радиус гибки — 2S [S1].")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=0))

    result = _invoke(graph, "какой радиус гибки")
    assert result["answer"] == "Радиус гибки — 2S [S1]."
    assert result["insufficient"] is False
    assert result["warnings"] == []
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
    chunks = [_chunk("a", "ГОСТ 14634-93", text="Радиус 6 мм, допуск ±0,2 мм.")]
    llm = FakeLLM("Радиус 6 мм [S1], а допуск ±0,2 [S7].")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=0))

    result = _invoke(graph, "радиус и допуск")
    assert "[S7]" not in result["answer"]
    assert "[S1]" in result["answer"]
    assert len(result["citations"]) == 1
    # Утверждение осталось без ссылки — пользователь должен об этом узнать.
    assert any("S7" in w for w in result["warnings"])
    assert "несуществующие фрагменты" in result["answer"]


def test_thread_keeps_history_between_turns(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 14634-93")]
    llm = FakeLLM("Ответ [S1].")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=4))

    _invoke(graph, "первый вопрос", thread="t-shared")
    result = _invoke(graph, "второй вопрос", thread="t-shared")
    assert len(result["messages"]) >= 2


def test_graph_refuses_when_named_standard_is_not_in_corpus(patched_retrieval):
    """«Что ГОСТ 16037-80 говорит о швах трубопроводов» не должен получать ответ
    по ГОСТ 5264-80, найденный поиском по всему корпусу."""
    chunks = [_chunk("a", "ГОСТ 5264-80", text="Конструктивные элементы швов 5 мм.")]
    llm = FakeLLM("Ответ по чужому стандарту [S1].")
    deps = patched_retrieval(chunks, corpus={"ГОСТ 5264-80"})
    graph = build_graph(deps, llm, Settings(history_turns=0))

    result = _invoke(graph, "какие швы по ГОСТ 16037-80 для трубопроводов?")
    assert result["insufficient"] is True
    assert "ГОСТ 16037-80 нет в базе" in result["answer"]
    assert llm.calls == 0
    assert patched_retrieval.searches == []


def test_refusal_suggests_other_edition_of_same_standard(patched_retrieval):
    llm = FakeLLM("не должно быть вызвано")
    deps = patched_retrieval([], corpus={"ГОСТ 5264-80"})
    graph = build_graph(deps, llm, Settings(history_turns=0))

    result = _invoke(graph, "что изменилось в ГОСТ 5264-69?")
    assert "есть другая редакция: ГОСТ 5264-80" in result["answer"]


def test_named_standard_in_corpus_filters_search(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 5264-80", text="Смещение кромок 0,5 мм.")]
    llm = FakeLLM("Смещение — 0,5 мм [S1].")
    deps = patched_retrieval(chunks, corpus={"ГОСТ 5264-80", "ГОСТ 10549-80"})
    graph = build_graph(deps, llm, Settings(history_turns=0))

    result = _invoke(graph, "смещение кромок по ГОСТу 5264")
    assert result["insufficient"] is False
    [query_filter] = patched_retrieval.searches
    assert query_filter is not None


def test_partly_missing_standards_answer_with_warning(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 5264-80", text="Смещение кромок 0,5 мм.")]
    llm = FakeLLM("Смещение — 0,5 мм [S1].")
    deps = patched_retrieval(chunks, corpus={"ГОСТ 5264-80"})
    graph = build_graph(deps, llm, Settings(history_turns=0))

    result = _invoke(graph, "сравни ГОСТ 5264-80 и ГОСТ 16037-80 по смещению кромок")
    assert result["insufficient"] is False
    assert any("ГОСТ 16037-80 нет в базе" in w for w in result["warnings"])


def test_number_absent_from_cited_source_is_flagged(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 24705-2004", text="| 10 | 1,5 | 9,026 | 8,376 |")]
    llm = FakeLLM("Средний диаметр М10×1,5 — 9,025 мм [S1].")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=0))

    result = _invoke(graph, "средний диаметр М10 с шагом 1,5")
    assert result["warnings"]
    assert "9,025" in result["warnings"][-1]
    assert "9,025" in result["answer"]  # сам ответ не переписываем — предупреждаем


def test_answer_without_any_citation_counts_as_insufficient(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 1-11")]
    llm = FakeLLM("Во фрагментах ответа нет.")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=0))

    result = _invoke(graph, "вопрос")
    assert result["insufficient"] is True
    assert result["citations"] == []


# --------------------------------------------------------------------------- #
# Проверка чисел
# --------------------------------------------------------------------------- #


def test_grounded_numbers_pass_regardless_of_decimal_separator():
    chunks = [_chunk("a", "ГОСТ 1-11", text="допуск 0.5 мм при 40 °C")]
    assert ungrounded_numbers("допуск 0,5 мм, температура 40 °C [S1]", chunks) == []


def test_number_matching_is_whole_value_not_substring():
    chunks = [_chunk("a", "ГОСТ 1-11", text="средний диаметр 9,026")]
    assert ungrounded_numbers("9,02 [S1]", chunks) == ["9,02"]


def test_numbers_from_question_and_label_are_allowed():
    chunks = [_chunk("a", "ГОСТ 24705-2004", text="значение 9,026")]
    answer = "Для М10 с шагом 1,5 по ГОСТ 24705-2004 (стр. 7) — 9,026 мм [S1]."
    assert ungrounded_numbers(answer, chunks, question="М10 с шагом 1,5") == []


def test_numbers_inside_standard_designations_are_not_checked():
    chunks = [_chunk("a", "ГОСТ 12.2.007.0-75", text="болт М 10 для тока до 630 А")]
    answer = "Болт М10 [S1]; размеры знака — по ГОСТ 21130-75, резьба — по ГОСТ 24705."
    assert ungrounded_numbers(answer, chunks) == []


@pytest.mark.parametrize(
    "answer",
    [
        "Во приведённых фрагментах ответа нет [S1]. ГОСТ 5264-80 трубопроводы исключает.",
        "**Краткий ответ:** конструктивные элементы швов во фрагментах отсутствуют [S1].",
        "Конкретные значения в приведённых фрагментах не указаны [S1].",
    ],
)
def test_polite_refusal_with_citation_is_insufficient(patched_retrieval, answer):
    """Отказ «во фрагментах ответа нет» со ссылкой на фрагмент — всё равно отказ."""
    chunks = [_chunk("a", "ГОСТ 5264-80", text="Стандарт не распространяется на трубопроводы.")]
    graph = build_graph(patched_retrieval(chunks), FakeLLM(answer), Settings(history_turns=0))
    assert _invoke(graph, "швы трубопроводов")["insufficient"] is True


def test_answer_mentioning_limits_is_not_mistaken_for_refusal(patched_retrieval):
    chunks = [_chunk("a", "ГОСТ 10549-80", text="сбег не более 2,8 мм")]
    llm = FakeLLM("Сбег — не более 2,8 мм [S1].\n\nДругих ограничений во фрагментах нет.")
    graph = build_graph(patched_retrieval(chunks), llm, Settings(history_turns=0))
    assert _invoke(graph, "сбег")["insufficient"] is False
