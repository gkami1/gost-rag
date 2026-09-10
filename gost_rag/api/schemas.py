"""Схемы запросов и ответов API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    #: Идентификатор диалога; новый создаётся автоматически, если не задан.
    thread_id: str | None = None


class Citation(BaseModel):
    marker: str
    label: str
    designation: str | None = None
    title: str | None = None
    status: str | None = None
    replaced_by: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    doc_id: str | None = None
    source_url: str | None = None
    ocr: bool = False
    rerank_score: float | None = None
    text: str = ""


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    insufficient: bool = False
    thread_id: str


class DocumentSummary(BaseModel):
    doc_id: str
    designation: str | None = None
    title: str | None = None
    status: str | None = None
    chunks: int = 0


class HealthResponse(BaseModel):
    status: str
    collection: str
    chunks: int
    llm_provider: str
    llm_model: str | None = None
    llm_configured: bool
    ocr_available: bool
    details: dict[str, Any] = {}
