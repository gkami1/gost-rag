"""Узлы графа: поиск -> переранжирование -> проверка достаточности -> ответ."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from gost_rag.config import Settings, get_settings
from gost_rag.graph.prompts import NO_CONTEXT_MESSAGE, SYSTEM_PROMPT, build_user_message
from gost_rag.graph.state import GraphState, build_citations, strip_invalid_citations
from gost_rag.logging import get_logger
from gost_rag.retrieval.filters import build_filter, known_designations
from gost_rag.retrieval.store import hybrid_search

log = get_logger(__name__)


class Retrieval:
    """Связка «клиент Qdrant + эмбеддер + реранкер», разделяемая узлами графа.

    Список обозначений корпуса кэшируется: он нужен на каждый запрос, а меняется
    только при переиндексации.
    """

    def __init__(self, client, embedder, reranker, settings: Settings | None = None) -> None:
        self.client = client
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or get_settings()
        self._designations: set[str] | None = None

    @property
    def designations(self) -> set[str]:
        if self._designations is None:
            self._designations = known_designations(self.client, self.settings)
        return self._designations

    def refresh_designations(self) -> None:
        self._designations = None


def make_retrieve_node(deps: Retrieval):
    def retrieve(state: GraphState) -> GraphState:
        question = state["question"]
        query_filter = build_filter(question, deps.designations)
        embedding = deps.embedder.encode_one(question)
        candidates = hybrid_search(deps.client, embedding, deps.settings, query_filter=query_filter)
        log.info("retrieved", question=question[:80], candidates=len(candidates))
        return {"candidates": candidates}

    return retrieve


def make_rerank_node(deps: Retrieval):
    def rerank(state: GraphState) -> GraphState:
        candidates = state.get("candidates") or []
        reranked = deps.reranker.rerank(state["question"], candidates)
        best = max((c.rerank_score or 0.0 for c in reranked), default=0.0)
        return {"reranked": reranked, "best_score": best}

    return rerank


def guard(state: GraphState) -> str:
    """Решить, есть ли на чём строить ответ.

    Если после переранжирования не осталось ничего выше порога, генерацию не
    запускаем вовсе: модели, которой нечего цитировать, свойственно отвечать по
    памяти — а неверная ссылка на ГОСТ хуже честного «не нашёл».
    """
    return "generate" if state.get("reranked") else "refuse"


def refuse(state: GraphState) -> GraphState:
    log.info("refused", question=state["question"][:80], best=state.get("best_score", 0.0))
    return {
        "answer": NO_CONTEXT_MESSAGE,
        "citations": [],
        "insufficient": True,
        "messages": [AIMessage(content=NO_CONTEXT_MESSAGE)],
    }


def build_prompt_messages(state: GraphState, settings: Settings) -> list:
    """Системный промпт + история диалога + вопрос с контекстом."""
    history = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    # 0 означает «без истории»: срез [-0:] вернул бы весь список, а не пустой.
    history = history[-settings.history_turns * 2 :] if settings.history_turns else []
    user = build_user_message(state["question"], state["reranked"])
    return [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=user)]


def make_generate_node(llm, settings: Settings | None = None):
    settings = settings or get_settings()

    def generate(state: GraphState) -> GraphState:
        response = llm.invoke(build_prompt_messages(state, settings))
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {"answer": answer}

    return generate


def verify_citations(state: GraphState) -> GraphState:
    """Отбросить ссылки на несуществующие фрагменты и собрать список источников."""
    chunks = state.get("reranked") or []
    answer = strip_invalid_citations(state.get("answer", ""), len(chunks))
    citations = build_citations(answer, chunks)
    log.info("answered", citations=len(citations), sources=len(chunks))
    return {
        "answer": answer,
        "citations": citations,
        "insufficient": False,
        "messages": [AIMessage(content=answer)],
    }
