"""Тесты HTTP-слоя. Модели и LLM не поднимаются — граф подменяется заглушкой."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gost_rag.api import main as api_main
from gost_rag.config import get_settings
from gost_rag.models import RetrievedChunk


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    # Встроенный Qdrant держит эксклюзивный лок на каталоге: пока идёт индексация
    # или eval, тесты на боевом индексе падали с «already accessed by another
    # instance». Пустой временный индекс изолирует их от чужих процессов.
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    get_settings.cache_clear()
    yield TestClient(api_main.app)
    get_settings.cache_clear()


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ГОСТ-RAG" in response.text
    # Предупреждение о том, что ответ не заменяет стандарт, обязано быть на странице.
    assert "не заменяет официальный текст стандарта" in response.text


def test_static_css_served(client):
    assert client.get("/static/app.css").status_code == 200


def test_healthz_reports_configuration(client):
    body = client.get("/healthz").json()
    assert body["llm_provider"] == "deepseek"
    assert "collection" in body
    # Без ключа в .env приложение обязано честно сообщать, что не настроено.
    assert body["llm_configured"] in (True, False)


def test_documents_endpoint_returns_list(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_unknown_document_source_is_404(client):
    assert client.get("/api/source/нет-такого").status_code == 404


def test_chat_requires_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# SSE
# --------------------------------------------------------------------------- #


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        point_id="p1",
        text="Радиус гибки составляет 2S.",
        payload={
            "designation": "ГОСТ 14634-93",
            "status": "действующий",
            "section": "3.2",
            "page_start": 7,
            "doc_id": "гост-14634-93",
            "title": "Ленты стальные",
        },
        rerank_score=0.91,
    )


class FakeGraph:
    """Воспроизводит поток событий LangGraph, который разбирает эндпоинт."""

    def __init__(self, events):
        self._events = events

    async def astream_events(self, inputs, config=None, version=None):
        for event in self._events:
            yield event


def _events():
    chunk = _chunk()
    return [
        {"event": "on_chain_end", "name": "rerank", "data": {"output": {"reranked": [chunk]}}},
        {"event": "on_chat_model_stream", "name": "llm", "data": {"chunk": _Token("Радиус ")}},
        {"event": "on_chat_model_stream", "name": "llm", "data": {"chunk": _Token("2S [S1][S9].")}},
        {
            "event": "on_chain_end",
            "name": "verify_citations",
            "data": {
                "output": {
                    "answer": "Радиус 2S [S1].",
                    "citations": [{"marker": "S1", "label": "ГОСТ 14634-93 (действующий), п. 3.2"}],
                    "insufficient": False,
                }
            },
        },
    ]


class _Token:
    def __init__(self, content: str) -> None:
        self.content = content


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    # sse-starlette разделяет строки CRLF — без нормализации разбор молча даёт пусто.
    text = text.replace("\r\n", "\n")
    events = []
    for block in text.split("\n\n"):
        name, payload = None, None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
        if name and payload is not None:
            events.append((name, payload))
    return events


@pytest.fixture
def stubbed_runtime(monkeypatch):
    monkeypatch.setattr(api_main, "_runtime", lambda: (object(), FakeGraph(_events())))


def test_chat_streams_sources_then_tokens_then_final(client, stubbed_runtime):
    response = client.post("/api/chat", json={"question": "какой радиус гибки"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    # Источники должны прийти раньше первого токена — иначе цитаты появятся
    # в интерфейсе позже ответа, который на них ссылается.
    assert names.index("sources") < names.index("token")
    assert names.index("token") < names.index("final")


def test_sources_event_carries_citation_fields(client, stubbed_runtime):
    events = dict(_parse_sse(client.post("/api/chat", json={"question": "q"}).text))
    source = events["sources"]["sources"][0]
    assert source["marker"] == "S1"
    assert source["label"].startswith("ГОСТ 14634-93 (действующий)")
    assert source["page_start"] == 7
    assert source["text"]


def test_final_event_overrides_streamed_text(client, stubbed_runtime):
    """Потоковый текст содержал выдуманную ссылку [S9]; финальный — уже без неё."""
    events = _parse_sse(client.post("/api/chat", json={"question": "q"}).text)
    streamed = "".join(p["text"] for n, p in events if n == "token")
    final = next(p for n, p in events if n == "final")

    assert "[S9]" in streamed
    assert "[S9]" not in final["answer"]
    assert final["insufficient"] is False


def test_thread_id_is_returned_and_reusable(client, stubbed_runtime):
    events = dict(_parse_sse(client.post("/api/chat", json={"question": "q"}).text))
    thread_id = events["start"]["thread_id"]
    assert thread_id

    again = dict(
        _parse_sse(client.post("/api/chat", json={"question": "q2", "thread_id": thread_id}).text)
    )
    assert again["start"]["thread_id"] == thread_id


def test_graph_failure_is_reported_as_error_event(client, monkeypatch):
    class Boom:
        async def astream_events(self, *a, **k):
            raise RuntimeError("провайдер недоступен")
            yield  # pragma: no cover

    monkeypatch.setattr(api_main, "_runtime", lambda: (object(), Boom()))
    events = dict(_parse_sse(client.post("/api/chat", json={"question": "q"}).text))
    assert "провайдер недоступен" in events["error"]["message"]
    assert "done" in events


# --------------------------------------------------------------------------- #
# Разбор потока в браузере
# --------------------------------------------------------------------------- #

NODE = shutil.which("node")
SSE_JS = Path(api_main.__file__).parent / "static" / "sse.js"

_NODE_DRIVER = """
const { takeEvents } = require(process.argv[1]);
const pieces = JSON.parse(require("fs").readFileSync(0, "utf8"));
let buffer = "";
const names = [];
for (const piece of pieces) {
  const { events, rest } = takeEvents(buffer + piece);
  buffer = rest;
  for (const e of events) names.push([e.name, e.data]);
}
process.stdout.write(JSON.stringify(names));
"""


def _browser_parse(text: str, piece: int) -> list[tuple[str, dict]]:
    """Прогнать поток через static/sse.js, нарезав его кусками по ``piece`` символов."""
    pieces = [text[i : i + piece] for i in range(0, len(text), piece)]
    out = subprocess.run(
        [NODE, "-e", _NODE_DRIVER, str(SSE_JS)],
        input=json.dumps(pieces),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [tuple(e) for e in json.loads(out.stdout)]


@pytest.mark.skipif(NODE is None, reason="node не установлен")
@pytest.mark.parametrize("piece", [1, 7, 100_000])
def test_browser_parser_sees_final_event_of_refusal(client, monkeypatch, piece):
    """Страница искала «\n\n», а sse-starlette шлёт «\r\n\r\n»: ни одно событие не
    разбиралось, и интерфейс навсегда оставался на «Ищу в документах…». Разбираем
    настоящий ответ сервера тем же кодом, что и браузер; кусок в 1 символ рвёт
    «\r\n» посередине."""
    refusal = [
        {"event": "on_chain_end", "name": "rerank", "data": {"output": {"reranked": []}}},
        {
            "event": "on_chain_end",
            "name": "refuse",
            "data": {"output": {"answer": "Не нашлось.\n\nСейчас в базе:", "insufficient": True}},
        },
    ]
    monkeypatch.setattr(api_main, "_runtime", lambda: (object(), FakeGraph(refusal)))
    raw = client.post("/api/chat", json={"question": "q"}).text
    assert "\r\n" in raw  # предпосылка теста: сервер действительно шлёт CRLF

    events = _browser_parse(raw, piece)
    names = [name for name, _ in events]
    assert names == ["start", "sources", "final", "done"]
    final = dict(events)["final"]
    assert final["insufficient"] is True
    assert final["answer"] == "Не нашлось.\n\nСейчас в базе:"
