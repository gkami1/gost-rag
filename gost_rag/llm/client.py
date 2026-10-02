"""Фабрика чат-модели.

Провайдер задаётся пресетом в настройках (deepseek / qwen / yandex / custom).
Все они предоставляют OpenAI-совместимый API, поэтому клиент один — ChatOpenAI
с подменённым base_url. Смена провайдера — это правка .env, а не кода.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI

from gost_rag.config import Settings, get_settings


class LLMDeadlineExceeded(httpx.ReadTimeout):
    """Ответ LLM не уложился в ``llm_deadline_s`` целиком.

    Наследник ReadTimeout: для openai SDK и вызывающего кода это обычный тайм-аут.
    """


def _expired(deadline: float, limit_s: float, request: httpx.Request) -> None:
    if time.monotonic() > deadline:
        raise LLMDeadlineExceeded(
            f"LLM не ответила за {limit_s:.0f} с — провайдер перегружен или завис",
            request=request,
        )


class _DeadlineStream(httpx.SyncByteStream):
    def __init__(self, inner, deadline: float, limit_s: float, request: httpx.Request) -> None:
        self._inner, self._deadline, self._limit_s, self._request = (
            inner,
            deadline,
            limit_s,
            request,
        )

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            _expired(self._deadline, self._limit_s, self._request)
            yield chunk

    def close(self) -> None:
        self._inner.close()


class _AsyncDeadlineStream(httpx.AsyncByteStream):
    def __init__(self, inner, deadline: float, limit_s: float, request: httpx.Request) -> None:
        self._inner, self._deadline, self._limit_s, self._request = (
            inner,
            deadline,
            limit_s,
            request,
        )

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            _expired(self._deadline, self._limit_s, self._request)
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()


class DeadlineTransport(httpx.BaseTransport):
    """Предел на весь HTTP-ответ, а не на паузу между байтами.

    Тайм-аут httpx — на чтение: он сбрасывается каждым пришедшим байтом. Перегруженный
    DeepSeek держит соединение пустыми ``: keep-alive`` каждые 12 с и не шлёт ни
    одного токена, и стриминговый ответ в интерфейсе висел бесконечно. Здесь
    отсчёт идёт от отправки запроса, и поток обрывается, как только предел вышел.
    """

    def __init__(self, limit_s: float, inner: httpx.BaseTransport | None = None) -> None:
        self._limit_s = limit_s
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        deadline = time.monotonic() + self._limit_s
        response = self._inner.handle_request(request)
        response.stream = _DeadlineStream(response.stream, deadline, self._limit_s, request)
        return response

    def close(self) -> None:
        self._inner.close()


class AsyncDeadlineTransport(httpx.AsyncBaseTransport):
    """То же для асинхронного клиента."""

    def __init__(self, limit_s: float, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._limit_s = limit_s
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        deadline = time.monotonic() + self._limit_s
        response = await self._inner.handle_async_request(request)
        response.stream = _AsyncDeadlineStream(response.stream, deadline, self._limit_s, request)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def build_chat_model(settings: Settings, *, streaming: bool = True) -> ChatOpenAI:
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY не задан. Скопируйте .env.example в .env и укажите ключ "
            f"провайдера {settings.llm_provider!r}."
        )
    clients: dict[str, object] = {}
    if settings.llm_deadline_s > 0:
        clients = {
            "http_client": httpx.Client(
                transport=DeadlineTransport(settings.llm_deadline_s), follow_redirects=True
            ),
            "http_async_client": httpx.AsyncClient(
                transport=AsyncDeadlineTransport(settings.llm_deadline_s), follow_redirects=True
            ),
        }
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        streaming=streaming,
        **clients,
    )


@lru_cache(maxsize=2)
def get_chat_model(*, streaming: bool = True) -> ChatOpenAI:
    return build_chat_model(get_settings(), streaming=streaming)
