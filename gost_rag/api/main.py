"""FastAPI-приложение: страница чата и SSE-поток ответа.

Источники отправляются отдельным событием сразу после переранжирования — до
первого токена. Инженер видит, на какие документы опирается ответ, ещё до того,
как ответ дописан, и может отбросить его, если стандарт не тот.

Финальный текст приходит отдельным событием ``final``: узел проверки цитат может
вырезать из ответа ссылку на несуществующий фрагмент уже после того, как токен
ушёл в поток, и клиент заменяет показанный текст на проверенный.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from gost_rag.api.schemas import ChatRequest, DocumentSummary, HealthResponse
from gost_rag.config import get_settings
from gost_rag.ingest.index import count_points, list_documents
from gost_rag.logging import configure_logging, get_logger

log = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ГОСТ-RAG", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def _startup() -> None:
    configure_logging(get_settings().log_level)


def _runtime():
    """Отложенная инициализация: модели грузятся при первом запросе, не при импорте."""
    from gost_rag.graph.build import get_runtime

    return get_runtime()


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


# --------------------------------------------------------------------------- #
# Страница
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    settings = get_settings()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model": settings.llm_model, "provider": settings.llm_provider},
    )


# --------------------------------------------------------------------------- #
# Чат
# --------------------------------------------------------------------------- #


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    _deps, graph = _runtime()
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def stream():
        yield _sse("start", {"thread_id": thread_id})
        sources_sent = False
        try:
            async for event in graph.astream_events(
                {"question": payload.question}, config=config, version="v2"
            ):
                if await request.is_disconnected():
                    break

                kind = event["event"]
                name = event.get("name")

                if kind == "on_chain_end" and name == "rerank" and not sources_sent:
                    chunks = (event["data"].get("output") or {}).get("reranked") or []
                    yield _sse(
                        "sources",
                        {"sources": [_source_dict(c, i) for i, c in enumerate(chunks, 1)]},
                    )
                    sources_sent = True

                elif kind == "on_chat_model_stream":
                    token = getattr(event["data"]["chunk"], "content", "")
                    if token:
                        yield _sse("token", {"text": token})

                elif kind == "on_chain_end" and name in {"verify_citations", "refuse"}:
                    output = event["data"].get("output") or {}
                    yield _sse(
                        "final",
                        {
                            "answer": output.get("answer", ""),
                            "citations": output.get("citations", []),
                            "insufficient": bool(output.get("insufficient")),
                        },
                    )
        except Exception as exc:
            log.exception("chat_failed", thread_id=thread_id)
            yield _sse("error", {"message": f"Ошибка обработки запроса: {exc}"})
        finally:
            yield _sse("done", {"thread_id": thread_id})

    return EventSourceResponse(stream())


def _source_dict(chunk, index: int) -> dict[str, Any]:
    payload = chunk.payload or {}
    return {
        "marker": f"S{index}",
        "label": chunk.citation_label(),
        "designation": chunk.designation,
        "title": payload.get("title"),
        "status": chunk.status,
        "replaced_by": payload.get("replaced_by"),
        "section": payload.get("section"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "doc_id": payload.get("doc_id"),
        "source_url": payload.get("source_url"),
        "ocr": bool(payload.get("ocr")),
        "rerank_score": chunk.rerank_score,
        "text": chunk.text,
    }


# --------------------------------------------------------------------------- #
# Корпус
# --------------------------------------------------------------------------- #


@app.get("/api/documents", response_model=list[DocumentSummary])
def documents():
    from gost_rag.ingest.index import get_client

    settings = get_settings()
    client = get_client(settings)
    try:
        return [DocumentSummary(**doc) for doc in list_documents(client, settings)]
    finally:
        client.close()


@app.get("/api/source/{doc_id}")
def source(doc_id: str, page: int = 1):
    """Отдать исходный PDF, чтобы цитату можно было проверить глазами."""
    from gost_rag.ingest.index import get_client

    settings = get_settings()
    client = get_client(settings)
    try:
        match = next((d for d in list_documents(client, settings) if d["doc_id"] == doc_id), None)
    finally:
        client.close()

    if not match or not match.get("source_path"):
        raise HTTPException(status_code=404, detail="Документ не найден")

    path = Path(match["source_path"])
    # Отдаём только то, что лежит в каталоге корпуса.
    try:
        path.resolve().relative_to(settings.raw_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Файл вне каталога корпуса") from None
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл исходника отсутствует на диске")

    return FileResponse(path, media_type="application/pdf", filename=path.name)


# --------------------------------------------------------------------------- #
# Здоровье
# --------------------------------------------------------------------------- #


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    from gost_rag.ingest.index import get_client
    from gost_rag.ingest.ocr import configure_tesseract, ocr_available

    settings = get_settings()
    client = get_client(settings)
    try:
        chunks = count_points(client, settings)
    except Exception as exc:
        chunks = 0
        log.warning("healthz_index_error", error=str(exc))
    finally:
        client.close()

    configure_tesseract(settings.tesseract_cmd)
    return HealthResponse(
        status="ok" if chunks else "empty_index",
        collection=settings.collection_name,
        chunks=chunks,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_configured=bool(settings.llm_api_key),
        ocr_available=ocr_available(settings.ocr_lang),
        details={"qdrant": settings.qdrant_url or str(settings.qdrant_path)},
    )
