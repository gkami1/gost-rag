"""Сборка графа LangGraph.

    retrieve -> rerank -> guard -+-> generate -> verify_citations -> END
                                 +-> refuse -> END

Узлы разделены намеренно: чтобы позже добавить multi-query или переписывание
вопроса, достаточно вставить узел перед retrieve, не трогая остальное.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from gost_rag.config import Settings, get_settings
from gost_rag.graph.nodes import (
    Retrieval,
    guard,
    make_generate_node,
    make_refuse_node,
    make_rerank_node,
    make_retrieve_node,
    verify_citations,
)
from gost_rag.graph.state import GraphState


def build_graph(deps: Retrieval, llm, settings: Settings | None = None, *, checkpointer=None):
    settings = settings or get_settings()

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", make_retrieve_node(deps))
    graph.add_node("rerank", make_rerank_node(deps))
    graph.add_node("generate", make_generate_node(llm, settings))
    graph.add_node("verify_citations", verify_citations)
    graph.add_node("refuse", make_refuse_node(deps))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", guard, {"generate": "generate", "refuse": "refuse"})
    graph.add_edge("generate", "verify_citations")
    graph.add_edge("verify_citations", END)
    graph.add_edge("refuse", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


@lru_cache(maxsize=1)
def get_runtime():
    """Ленивая сборка боевого графа: клиент, модели и LLM — по одному экземпляру."""
    from gost_rag.ingest.embed import get_embedder
    from gost_rag.ingest.index import get_client
    from gost_rag.llm import get_chat_model
    from gost_rag.retrieval.rerank import get_reranker

    settings = get_settings()
    deps = Retrieval(get_client(settings), get_embedder(), get_reranker(), settings)
    return deps, build_graph(deps, get_chat_model(streaming=True), settings)
