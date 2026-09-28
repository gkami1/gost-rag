"""Конфигурация приложения.

Все настройки читаются из переменных окружения / .env. Провайдер LLM задаётся
одним пресетом (см. LLM_PRESETS) — любой OpenAI-совместимый эндпоинт подходит,
поэтому смена провайдера не требует изменений в коде.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ProviderName = Literal["deepseek", "qwen", "yandex", "custom"]

#: Пресеты OpenAI-совместимых провайдеров: base_url и модель по умолчанию.
#: Для yandex модель имеет вид gpt://<folder_id>/yandexgpt-5.1/latest —
#: folder_id подставляется из LLM_FOLDER_ID.
LLM_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
    },
    "qwen": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "yandex": {
        "base_url": "https://llm.api.cloud.yandex.net/v1",
        "model": "gpt://{folder_id}/yandexgpt-5.1/latest",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM ----------
    llm_provider: ProviderName = "deepseek"
    llm_api_key: str = ""
    #: Переопределяют пресет, если заданы явно.
    llm_base_url: str | None = None
    llm_model: str | None = None
    #: Только для Yandex: подставляется в model URI.
    llm_folder_id: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3

    # ---------- Модели поиска ----------
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_device: str | None = None  # None -> авто (cuda, если доступна)
    use_fp16: bool = True
    embed_batch_size: int = 4
    #: Окна энкодера и реранкера в токенах. У FlagEmbedding оба по умолчанию 512,
    #: а чанк — 800: хвост почти каждого чанка молча отрезался и не попадал ни в
    #: вектор, ни к кросс-энкодеру (23 % токенов корпуса). Держать с запасом над
    #: ``chunk_tokens`` — хвост короче ``min_chunk_tokens`` приклеивается сверху.
    embed_max_length: int = 1024
    #: Пара «вопрос + чанк» целиком; bge-reranker-v2-m3 обучен на 1024.
    rerank_max_length: int = 1024

    # ---------- Чанкинг ----------
    chunk_tokens: int = 800
    chunk_overlap_tokens: int = 80
    min_chunk_tokens: int = 50

    # ---------- Индекс ----------
    qdrant_path: Path = PROJECT_ROOT / "data" / "processed" / "qdrant"
    qdrant_url: str | None = None  # если задан — используется сервер, не embedded
    qdrant_api_key: str | None = None
    collection_name: str = "gost_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_vector_size: int = 1024

    # ---------- Поиск ----------
    prefetch_limit: int = 50
    fusion_limit: int = 30
    rerank_top_n: int = 6
    #: Порог отсечения после кросс-энкодера. Ноль здесь равносилен выключенному
    #: решению №6: score() вызывается с normalize=True, то есть выдаёт сигмоиду
    #: строго больше нуля, и при 0.0 не отсекается вообще ничего — guard никогда
    #: не уходит в refuse. Значение подобрано на eval/questions.yaml: 0.2 отсекает
    #: 5 из 6 вопросов вне корпуса ценой 3 из 22 вопросов внутри корпуса.
    rerank_threshold: float = 0.2
    history_turns: int = 4

    # ---------- Ingestion ----------
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    registry_path: Path = PROJECT_ROOT / "data" / "registry" / "documents.csv"
    ocr_enabled: bool = True
    ocr_lang: str = "rus"
    ocr_dpi: int = 300
    #: Меньше этого числа символов на странице -> считаем страницу сканом.
    ocr_min_chars: int = 100
    tesseract_cmd: str | None = None  # путь к бинарю, если не в PATH

    # ---------- API ----------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _apply_preset(self) -> Settings:
        preset = LLM_PRESETS.get(self.llm_provider)
        if preset is not None:
            if self.llm_base_url is None:
                self.llm_base_url = preset["base_url"]
            if self.llm_model is None:
                self.llm_model = preset["model"]
        if self.llm_model and "{folder_id}" in self.llm_model:
            if not self.llm_folder_id:
                raise ValueError(
                    "llm_model содержит {folder_id}, но LLM_FOLDER_ID не задан "
                    "(требуется для провайдера yandex)"
                )
            self.llm_model = self.llm_model.format(folder_id=self.llm_folder_id)
        return self

    @property
    def qdrant_location(self) -> dict[str, object]:
        """Аргументы для QdrantClient: сервер, если задан URL, иначе embedded."""
        if self.qdrant_url:
            return {"url": self.qdrant_url, "api_key": self.qdrant_api_key}
        self.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        return {"path": str(self.qdrant_path)}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
