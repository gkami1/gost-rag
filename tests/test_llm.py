"""Предел на весь ответ LLM. Сеть не нужна: провайдер подменён MockTransport."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from gost_rag.config import Settings
from gost_rag.llm.client import (
    AsyncDeadlineTransport,
    DeadlineTransport,
    LLMDeadlineExceeded,
    build_chat_model,
)


def _keepalive_forever():
    # Так выглядит перегруженный DeepSeek: байты идут, токенов нет.
    while True:
        time.sleep(0.01)
        yield b": keep-alive\n\n"


async def _akeepalive_forever():
    while True:
        await asyncio.sleep(0.01)
        yield b": keep-alive\n\n"


def test_keepalive_stream_is_cut_at_deadline():
    inner = httpx.MockTransport(lambda request: httpx.Response(200, content=_keepalive_forever()))
    started = time.monotonic()
    with (
        httpx.Client(transport=DeadlineTransport(0.1, inner)) as client,
        client.stream("POST", "https://llm.test/v1/chat/completions") as response,
        pytest.raises(LLMDeadlineExceeded),
    ):
        for _ in response.iter_bytes():
            pass
    assert time.monotonic() - started < 2


def test_deadline_is_a_timeout_for_callers():
    # openai SDK и вызывающий код ловят тайм-ауты httpx — новый тип обязан быть одним из них.
    assert issubclass(LLMDeadlineExceeded, httpx.TimeoutException)


def test_fast_response_passes_untouched():
    inner = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with httpx.Client(transport=DeadlineTransport(5, inner)) as client:
        assert client.post("https://llm.test/v1/chat/completions").json() == {"ok": True}


def test_async_keepalive_stream_is_cut_at_deadline():
    async def run() -> None:
        inner = httpx.MockTransport(
            lambda request: httpx.Response(200, content=_akeepalive_forever())
        )
        async with (
            httpx.AsyncClient(transport=AsyncDeadlineTransport(0.1, inner)) as client,
            client.stream("POST", "https://llm.test/v1/chat/completions") as response,
        ):
            async for _ in response.aiter_bytes():
                pass

    with pytest.raises(LLMDeadlineExceeded):
        asyncio.run(run())


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, llm_api_key="sk-test", **overrides)


def test_chat_model_gets_deadline_clients():
    llm = build_chat_model(_settings(llm_deadline_s=30))
    assert isinstance(llm.http_client._transport, DeadlineTransport)
    assert isinstance(llm.http_async_client._transport, AsyncDeadlineTransport)


def test_zero_deadline_keeps_default_clients():
    llm = build_chat_model(_settings(llm_deadline_s=0))
    assert llm.http_client is None
    assert llm.http_async_client is None
