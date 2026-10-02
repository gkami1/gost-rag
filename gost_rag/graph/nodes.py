"""Узлы графа: поиск -> переранжирование -> проверка достаточности -> ответ."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from gost_rag.config import Settings, get_settings
from gost_rag.graph.prompts import NO_CONTEXT_MESSAGE, SYSTEM_PROMPT, build_user_message
from gost_rag.graph.state import (
    GraphState,
    build_citations,
    invalid_citations,
    says_no_answer,
    strip_citation_markers,
    strip_invalid_citations,
    ungrounded_numbers,
    used_indices,
)
from gost_rag.ingest.metadata import registry_key
from gost_rag.logging import get_logger
from gost_rag.retrieval.filters import known_designations, resolve_designations
from gost_rag.retrieval.registry import LiveRegistry
from gost_rag.retrieval.rerank import passes_dense_gate, select_context
from gost_rag.retrieval.store import dense_only_search, hybrid_search

log = get_logger(__name__)


class Retrieval:
    """Связка «клиент Qdrant + эмбеддер + реранкер», разделяемая узлами графа.

    Список обозначений корпуса кэшируется: он нужен на каждый запрос, а меняется
    только при переиндексации. Реестр, наоборот, читается живым: статус
    стандарта меняется правкой CSV, без переиндексации.
    """

    def __init__(
        self,
        client,
        embedder,
        reranker,
        settings: Settings | None = None,
        registry: LiveRegistry | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or get_settings()
        self.registry = registry or LiveRegistry(self.settings.registry_path)
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
        # Вопрос пишется в историю здесь, а не вызывающей стороной: иначе в треде
        # копились одни ответы, и уточняющий вопрос видел ответы без вопросов к ним.
        update: GraphState = {
            "messages": [HumanMessage(content=question)],
            "missing_designations": [],
            "designation_alternatives": {},
            "warnings": [],
        }
        match = resolve_designations(question, deps.designations)
        update["missing_designations"] = match.missing
        update["designation_alternatives"] = match.alternatives
        if match.only_missing:
            # Вопрос прямо про документ, которого нет. Поиск по остальному корпусу
            # нашёл бы похожую тему в чужом стандарте, и ответ выглядел бы как
            # ответ по названному — это хуже честного отказа.
            log.info("retrieve_skipped_missing_designation", missing=match.missing)
            return {**update, "candidates": []}

        embedding = deps.embedder.encode_one(question)
        dense = dense_only_search(
            deps.client, embedding, deps.settings, query_filter=match.filter, limit=1
        )
        dense_top = dense[0].fusion_score if dense else None
        if not passes_dense_gate(dense_top, deps.settings):
            # Решение №6: даже ближайший фрагмент далёк от вопроса — отвечать не
            # по чему. Ни реранкер, ни LLM не вызываются.
            log.info("retrieve_weak_sources", question=question[:80], dense_top=dense_top)
            return {**update, "candidates": [], "dense_top": dense_top}
        candidates = hybrid_search(deps.client, embedding, deps.settings, query_filter=match.filter)
        # Статус и замена — из реестра сейчас, а не из индекса на момент загрузки.
        deps.registry.apply(candidates)
        log.info(
            "retrieved", question=question[:80], candidates=len(candidates), dense_top=dense_top
        )
        return {**update, "candidates": candidates, "dense_top": dense_top}

    return retrieve


def make_rerank_node(deps: Retrieval):
    def rerank(state: GraphState) -> GraphState:
        candidates = state.get("candidates") or []
        if not candidates:
            # Отказ не должен зависеть от того, как реранкер обходится с пустым входом.
            return {"reranked": [], "best_score": 0.0}
        reranked = select_context(state["question"], candidates, deps.reranker, deps.settings)
        # Лучший скор — по всем оценённым кандидатам, а не по прошедшим порог: иначе
        # при отказе он всегда 0.0 и не видно, насколько вопрос не дотянул до порога.
        best = max((c.rerank_score or 0.0 for c in candidates), default=0.0)
        return {"reranked": reranked, "best_score": best}

    return rerank


def guard(state: GraphState) -> str:
    """Решить, есть ли на чём строить ответ.

    Если после переранжирования не осталось ничего выше порога, генерацию не
    запускаем вовсе: модели, которой нечего цитировать, свойственно отвечать по
    памяти — а неверная ссылка на ГОСТ хуже честного «не нашёл».
    """
    return "generate" if state.get("reranked") else "refuse"


def make_refuse_node(deps: Retrieval):
    def refuse(state: GraphState) -> GraphState:
        missing = state.get("missing_designations") or []
        log.info(
            "refused",
            question=state["question"][:80],
            reason=_refusal_reason(state, deps.settings),
            dense_top=state.get("dense_top"),
            best=round(state.get("best_score", 0.0), 4),
        )
        answer = (
            missing_documents_message(missing, state.get("designation_alternatives") or {})
            if missing
            else NO_CONTEXT_MESSAGE
        )
        # Без списка корпуса отказ неотличим от сбоя поиска: пользователь не знает,
        # что тема просто не покрыта, и ищет ошибку там, где её нет.
        corpus = corpus_summary(deps.designations, deps.registry)
        if corpus:
            answer = f"{answer}\n\n{corpus}"
        return _refusal(answer)

    return refuse


def _refusal_reason(state: GraphState, settings: Settings) -> str:
    """Какая из проверок остановила вопрос — для журнала, не для пользователя."""
    if state.get("missing_designations") and not state.get("candidates"):
        return "missing_designation"
    if not passes_dense_gate(state.get("dense_top"), settings):
        return "weak_dense_match"
    if not state.get("candidates"):
        return "no_candidates"
    return "below_rerank_threshold"


def corpus_summary(designations: set[str], registry: LiveRegistry, limit: int = 20) -> str:
    """«Сейчас в базе: …» — обозначения из индекса, названия из реестра."""
    if not designations:
        return ""
    rows = registry.rows()
    ordered = sorted(designations)
    lines = []
    for designation in ordered[:limit]:
        title = (rows.get(registry_key(designation)) or {}).get("title")
        lines.append(f"— {designation}" + (f" «{title}»" if title else ""))
    if len(ordered) > limit:
        lines.append(f"…и ещё {len(ordered) - limit}")
    return "Сейчас в базе:\n" + "\n".join(lines)


def _refusal(answer: str) -> GraphState:
    return {
        "answer": answer,
        "citations": [],
        "insufficient": True,
        "warnings": [],
        "messages": [AIMessage(content=answer)],
    }


def build_prompt_messages(state: GraphState, settings: Settings) -> list:
    """Системный промпт + история диалога + вопрос с контекстом."""
    history = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    # Текущий вопрос уже лежит в истории (его добавил retrieve), но в промпт он
    # уходит ниже — вместе с контекстом. Дважды его передавать незачем.
    last = history[-1] if history else None
    if isinstance(last, HumanMessage) and last.content == state["question"]:
        history = history[:-1]
    # 0 означает «без истории»: срез [-0:] вернул бы весь список, а не пустой.
    history = history[-settings.history_turns * 2 :] if settings.history_turns else []
    # [S1] в старом ответе указывал на фрагмент прошлого хода, а в новом контексте
    # под тем же номером лежит другой чанк — модель скопировала бы чужую ссылку.
    history = [
        AIMessage(content=strip_citation_markers(m.content)) if isinstance(m, AIMessage) else m
        for m in history
    ]
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
    """Проверить ответ по источникам и сказать пользователю, что не подтвердилось.

    Молча вырезать ``[S7]`` при шести фрагментах мало: утверждение остаётся в
    ответе, только теперь без ссылки, и единственный видимый признак выдумки
    исчезает. Поэтому каждая проблема оставляет предупреждение рядом с ответом.
    """
    chunks = state.get("reranked") or []
    raw = state.get("answer", "")
    invalid = invalid_citations(raw, len(chunks))
    answer = strip_invalid_citations(raw, len(chunks))
    citations = build_citations(answer, chunks)
    cited = [chunks[i - 1] for i in used_indices(answer, len(chunks))]

    warnings: list[str] = []
    missing = state.get("missing_designations") or []
    if missing:
        warnings.append(
            missing_documents_message(missing, state.get("designation_alternatives") or {})
            + " Ответ дан только по имеющимся документам."
        )
    if invalid:
        warnings.append(
            # Без квадратных скобок: иначе предупреждение само разбиралось бы
            # как ссылка — в истории диалога и в подсветке источников.
            "Модель сослалась на несуществующие фрагменты ("
            + ", ".join(f"S{i}" for i in invalid)
            + "): относящиеся к ним утверждения источником не подтверждены."
        )
    if not citations:
        warnings.append("Ответ не подкреплён ни одной ссылкой на фрагменты документов.")
    elif numbers := ungrounded_numbers(answer, cited, state.get("question", "")):
        warnings.append(
            "Этих чисел нет в процитированных фрагментах: "
            + ", ".join(numbers)
            + ". Сверьте их с документом."
        )

    log.info(
        "answered",
        citations=len(citations),
        sources=len(chunks),
        invalid=len(invalid),
        warnings=len(warnings),
    )
    shown = answer if not warnings else answer + "\n\n" + "\n".join(f"⚠ {w}" for w in warnings)
    return {
        "answer": shown,
        "citations": citations,
        # Без единой ссылки ответ не на чем проверить; «во фрагментах ответа нет»
        # со ссылкой на фрагмент — тоже отказ, только вежливый.
        "insufficient": not citations or says_no_answer(answer),
        "warnings": warnings,
        "messages": [AIMessage(content=answer)],
    }


def missing_documents_message(missing: list[str], alternatives: dict[str, list[str]]) -> str:
    """«ГОСТ 16037-80 нет в базе» — с подсказкой, если есть другая редакция."""
    parts: list[str] = []
    for value in missing:
        other = alternatives.get(value)
        if other:
            parts.append(f"{value} нет в базе (есть другая редакция: {', '.join(other)}).")
        else:
            parts.append(f"{value} нет в базе.")
    return " ".join(parts)
