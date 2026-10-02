"""Переранжирование кросс-энкодером bge-reranker-v2-m3.

RRF расставляет кандидатов, не видя пары «запрос-чанк» целиком; кросс-энкодер
видит и потому заметно точнее на технических формулировках, где решает одно
слово («наружный» против «внутреннего» радиуса). Он дорог, поэтому применяется
к 30 кандидатам, а не ко всему корпусу.
"""

from __future__ import annotations

from functools import lru_cache

from gost_rag.config import Settings, get_settings
from gost_rag.ingest.embed import _detect_device, configure_torch
from gost_rag.logging import get_logger
from gost_rag.models import RetrievedChunk

log = get_logger(__name__)


class Reranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._device = _detect_device(self._settings.embedding_device)
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            configure_torch(self._settings)

            log.info("loading_reranker", model=self._settings.reranker_model, device=self._device)
            self._model = FlagReranker(
                self._settings.reranker_model,
                use_fp16=self._settings.use_fp16 and self._device != "cpu",
                devices=self._device,
                max_length=self._settings.rerank_max_length,
            )
        return self._model

    def score(self, query: str, candidates: list[RetrievedChunk]) -> list[float]:
        if not candidates:
            return []
        model = self._load()
        scores = model.compute_score(
            [[query, candidate.text] for candidate in candidates], normalize=True
        )
        # compute_score возвращает число, а не список, если пара всего одна.
        return [float(scores)] if isinstance(scores, float) else [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_n: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Отсортировать кандидатов и отсечь слабые."""
        if not candidates:
            return []

        top_n = top_n if top_n is not None else self._settings.rerank_top_n
        threshold = threshold if threshold is not None else self._settings.rerank_threshold

        for candidate, score in zip(candidates, self.score(query, candidates), strict=True):
            candidate.rerank_score = score

        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        kept = [c for c in ranked if (c.rerank_score or 0.0) >= threshold][:top_n]
        log.info(
            "reranked",
            candidates=len(candidates),
            kept=len(kept),
            best=round(ranked[0].rerank_score or 0.0, 4),
        )
        return kept


def select_context(
    question: str,
    candidates: list[RetrievedChunk],
    reranker,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Фрагменты, которые уйдут в контекст LLM, — одна функция для графа и оценки.

    При ``rerank_candidates = 0`` кросс-энкодер не вызывается: в контекст идут
    первые ``rerank_top_n`` по RRF. На оценке это почти ничего не меняет в том,
    что видит модель (нужный документ в топ-6: 25 из 25 по RRF против 24 из 25
    после реранкера), а на CPU экономит минуты на каждом вопросе.
    """
    if not candidates:
        return []
    if settings.rerank_candidates <= 0:
        return candidates[: settings.rerank_top_n]
    return reranker.rerank(question, candidates[: settings.rerank_candidates])


def passes_dense_gate(dense_top: float | None, settings: Settings) -> bool:
    """Достаточно ли близок лучший плотный фрагмент, чтобы вообще отвечать."""
    if settings.min_dense_score <= 0:
        return True
    return dense_top is not None and dense_top >= settings.min_dense_score


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
