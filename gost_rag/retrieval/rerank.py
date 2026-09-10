"""Переранжирование кросс-энкодером bge-reranker-v2-m3.

RRF расставляет кандидатов, не видя пары «запрос-чанк» целиком; кросс-энкодер
видит и потому заметно точнее на технических формулировках, где решает одно
слово («наружный» против «внутреннего» радиуса). Он дорог, поэтому применяется
к 30 кандидатам, а не ко всему корпусу.
"""

from __future__ import annotations

from functools import lru_cache

from gost_rag.config import Settings, get_settings
from gost_rag.ingest.embed import _detect_device
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

            log.info("loading_reranker", model=self._settings.reranker_model, device=self._device)
            self._model = FlagReranker(
                self._settings.reranker_model,
                use_fp16=self._settings.use_fp16 and self._device != "cpu",
                devices=self._device,
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


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
