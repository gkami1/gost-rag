"""Векторизация чанков моделью BGE-M3.

BGE-M3 за один проход выдаёт и плотный вектор, и обученные разреженные веса —
поэтому гибридный поиск не требует второй модели и второго прохода по корпусу.
Модель тяжёлая (~2 ГБ), поэтому загружается лениво и переиспользуется процессом.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from gost_rag.config import Settings, get_settings
from gost_rag.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Embedding:
    dense: list[float]
    #: token_id -> вес. Пустой словарь — валидный случай (текст из одних стоп-слов).
    sparse: dict[int, float]

    @property
    def sparse_indices(self) -> list[int]:
        return list(self.sparse.keys())

    @property
    def sparse_values(self) -> list[float]:
        return list(self.sparse.values())


def configure_torch(settings: Settings) -> None:
    """Число потоков torch на CPU — до загрузки модели, один раз на процесс."""
    import os

    import torch

    threads = settings.torch_threads or os.cpu_count() or 1
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
        log.info("torch_threads", threads=threads)


def _detect_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover - torch всегда есть в проде
        pass
    return "cpu"


class BGEM3Embedder:
    """Обёртка над BGEM3FlagModel с единым форматом результата."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._device = _detect_device(self._settings.embedding_device)
        self._model = None

    @property
    def device(self) -> str:
        return self._device

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            configure_torch(self._settings)

            log.info(
                "loading_embedder",
                model=self._settings.embedding_model,
                device=self._device,
            )
            self._model = BGEM3FlagModel(
                self._settings.embedding_model,
                # fp16 на CPU даёт только замедление.
                use_fp16=self._settings.use_fp16 and self._device != "cpu",
                devices=self._device,
                passage_max_length=self._settings.embed_max_length,
            )
        return self._model

    def encode(self, texts: list[str], *, batch_size: int | None = None) -> list[Embedding]:
        if not texts:
            return []
        model = self._load()
        output = model.encode(
            texts,
            batch_size=batch_size or self._settings.embed_batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        sparse = output["lexical_weights"]
        return [
            Embedding(dense=[float(x) for x in dense[i]], sparse=_to_int_keys(sparse[i]))
            for i in range(len(texts))
        ]

    def encode_one(self, text: str) -> Embedding:
        return self.encode([text])[0]


def _to_int_keys(weights: dict) -> dict[int, float]:
    """FlagEmbedding отдаёт id токенов строками — Qdrant ждёт целые."""
    return {int(token_id): float(weight) for token_id, weight in weights.items() if weight > 0}


@lru_cache(maxsize=1)
def get_embedder() -> BGEM3Embedder:
    return BGEM3Embedder()
