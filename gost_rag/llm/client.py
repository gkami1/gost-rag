"""Фабрика чат-модели.

Провайдер задаётся пресетом в настройках (deepseek / qwen / yandex / custom).
Все они предоставляют OpenAI-совместимый API, поэтому клиент один — ChatOpenAI
с подменённым base_url. Смена провайдера — это правка .env, а не кода.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from gost_rag.config import Settings, get_settings


def build_chat_model(settings: Settings, *, streaming: bool = True) -> ChatOpenAI:
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY не задан. Скопируйте .env.example в .env и укажите ключ "
            f"провайдера {settings.llm_provider!r}."
        )
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        streaming=streaming,
    )


@lru_cache(maxsize=2)
def get_chat_model(*, streaming: bool = True) -> ChatOpenAI:
    return build_chat_model(get_settings(), streaming=streaming)
