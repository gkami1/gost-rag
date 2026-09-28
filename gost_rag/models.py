"""Общие структуры данных: документ -> страницы -> блоки -> чанки -> цитаты."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

#: Пространство имён для детерминированных ID точек в Qdrant.
_NAMESPACE = uuid.UUID("6f0b9d1e-3a4c-4f8b-9c2a-1d5e7f0a2b31")

BlockKind = Literal["text", "table", "heading"]
DocStatus = Literal["действующий", "отменён", "заменён", "неизвестно"]


@dataclass(slots=True)
class Block:
    """Минимальная неделимая единица текста документа."""

    text: str
    page_no: int
    kind: BlockKind = "text"
    #: Ближайший вышестоящий пункт, например "3.2.1".
    section: str | None = None
    #: Для таблиц: строка заголовка в Markdown, повторяется при разрезании.
    table_header: str | None = None


@dataclass(slots=True)
class PageDoc:
    """Одна страница исходного файла после извлечения текста."""

    page_no: int
    text: str
    blocks: list[Block] = field(default_factory=list)
    #: Текстового слоя нет — страница уйдёт в OCR на следующем шаге.
    needs_ocr: bool = False
    from_ocr: bool = False
    ocr_confidence: float | None = None


@dataclass(slots=True)
class DocumentMeta:
    """Паспорт документа: обозначение, статус, происхождение."""

    doc_id: str
    designation: str | None
    title: str | None
    year: int | None
    status: DocStatus
    doc_type: str | None
    source_path: str
    source_url: str | None = None
    replaced_by: str | None = None
    sha256: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "designation": self.designation,
            "title": self.title,
            "year": self.year,
            "status": self.status,
            "doc_type": self.doc_type,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "replaced_by": self.replaced_by,
        }


@dataclass(slots=True)
class Chunk:
    """Единица индексации и цитирования."""

    doc_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    token_count: int
    #: Первый пункт собственного содержимого чанка (без хвоста перекрытия).
    section: str | None = None
    #: Все пункты, чей текст попал в чанк, в порядке появления.
    sections: list[str] = field(default_factory=list)
    contains_table: bool = False
    from_ocr: bool = False
    ocr_confidence: float | None = None

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def point_id(self) -> str:
        """Детерминированный ID: повторный ingest того же текста — no-op."""
        return str(uuid.uuid5(_NAMESPACE, f"{self.doc_id}:{self.chunk_index}:{self.text_hash}"))

    def as_payload(self, meta: DocumentMeta) -> dict[str, Any]:
        payload = meta.as_payload()
        payload.update(
            {
                "chunk_index": self.chunk_index,
                "text": self.text,
                "page_start": self.page_start,
                "page_end": self.page_end,
                "token_count": self.token_count,
                "section": self.section,
                "sections": self.sections,
                "contains_table": self.contains_table,
                "ocr": self.from_ocr,
                "ocr_conf": self.ocr_confidence,
                "text_hash": self.text_hash,
                "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        return payload


@dataclass(slots=True)
class RetrievedChunk:
    """Чанк, поднятый поиском, вместе со скорами каждой стадии."""

    point_id: str
    text: str
    payload: dict[str, Any]
    fusion_score: float | None = None
    rerank_score: float | None = None

    @property
    def designation(self) -> str:
        return self.payload.get("designation") or self.payload.get("doc_id", "документ")

    @property
    def status(self) -> str:
        return self.payload.get("status") or "неизвестно"

    def citation_label(self) -> str:
        """Человекочитаемая ссылка: «ГОСТ 14634-93 (действующий), п. 3.2, стр. 7»."""
        parts = [self.designation]
        if self.status and self.status != "неизвестно":
            parts[0] = f"{parts[0]} ({self.status})"
        if clauses := self.clause_label():
            parts.append(clauses)
        page_start = self.payload.get("page_start")
        page_end = self.payload.get("page_end")
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                parts.append(f"стр. {page_start}–{page_end}")
            else:
                parts.append(f"стр. {page_start}")
        return ", ".join(parts)

    def clause_label(self) -> str | None:
        """«п. 3.2» или «пп. 3.3.13–3.4.7» — диапазоном, как и страницы.

        Чанк в 800 токенов обычно охватывает несколько пунктов; назвать один из
        них значило бы выдать правдоподобный, но неточный номер пункта.
        """
        sections = self.payload.get("sections") or []
        if len(sections) > 1:
            return f"пп. {sections[0]}–{sections[-1]}"
        # Индекс, собранный до появления поля ``sections``, хранит только ``section``.
        section = sections[0] if sections else self.payload.get("section")
        return f"п. {section}" if section else None
