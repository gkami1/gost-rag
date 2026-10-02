"""Qdrant: схема коллекции и запись чанков.

Коллекция держит два именованных вектора — плотный и разреженный, — чтобы RRF
считался на стороне Qdrant одним запросом. Встроенный режим (path=...) и сервер
(url=...) отличаются только аргументами клиента, поэтому переезд на сервер не
требует правок кода.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient, models

from gost_rag.config import Settings, get_settings
from gost_rag.ingest.embed import Embedding
from gost_rag.logging import get_logger
from gost_rag.models import Chunk, DocumentMeta

if TYPE_CHECKING:
    from collections.abc import Sequence

log = get_logger(__name__)

#: Поля, по которым делаются точные фильтры (обозначение стандарта, статус, тип).
_INDEXED_PAYLOAD_FIELDS = ("designation", "status", "doc_type", "doc_id")


def get_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(**settings.qdrant_location)


def ensure_collection(
    client: QdrantClient, settings: Settings | None = None, *, recreate: bool = False
) -> None:
    settings = settings or get_settings()
    exists = client.collection_exists(settings.collection_name)

    if exists and recreate:
        log.warning("recreating_collection", collection=settings.collection_name)
        client.delete_collection(settings.collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                settings.dense_vector_name: models.VectorParams(
                    size=settings.dense_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                settings.sparse_vector_name: models.SparseVectorParams(
                    # Веса BGE-M3 уже взвешены обучением — второй раз IDF не нужен.
                    modifier=models.Modifier.NONE,
                )
            },
        )
        log.info("collection_created", collection=settings.collection_name)

    # Во встроенном режиме payload-индексы игнорируются (фильтрация всё равно
    # идёт полным перебором), поэтому создаём их только на сервере.
    if not settings.qdrant_url:
        return

    for field in _INDEXED_PAYLOAD_FIELDS:
        try:
            client.create_payload_index(
                collection_name=settings.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            log.debug("payload_index_skipped", field=field, error=str(exc))


def build_points(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Embedding],
    meta: DocumentMeta,
    settings: Settings | None = None,
) -> list[models.PointStruct]:
    settings = settings or get_settings()
    if len(chunks) != len(embeddings):
        raise ValueError("Число чанков и эмбеддингов должно совпадать")

    return [
        models.PointStruct(
            id=chunk.point_id,
            vector={
                settings.dense_vector_name: embedding.dense,
                settings.sparse_vector_name: models.SparseVector(
                    indices=embedding.sparse_indices,
                    values=embedding.sparse_values,
                ),
            },
            payload=chunk.as_payload(meta),
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def _batched(items: Sequence[models.PointStruct], size: int) -> Iterator[list[models.PointStruct]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def upsert_points(
    client: QdrantClient,
    points: Sequence[models.PointStruct],
    settings: Settings | None = None,
    *,
    batch_size: int = 64,
) -> int:
    settings = settings or get_settings()
    for batch in _batched(points, batch_size):
        client.upsert(collection_name=settings.collection_name, points=batch, wait=True)
    return len(points)


def existing_point_ids(
    client: QdrantClient, ids: Iterable[str], settings: Settings | None = None
) -> set[str]:
    """Какие точки уже в индексе — чтобы не пересчитывать эмбеддинги зря."""
    settings = settings or get_settings()
    id_list = list(ids)
    if not id_list:
        return set()
    found = client.retrieve(
        collection_name=settings.collection_name,
        ids=id_list,
        with_payload=False,
        with_vectors=False,
    )
    return {str(point.id) for point in found}


def delete_document(client: QdrantClient, doc_id: str, settings: Settings | None = None) -> None:
    """Удалить все чанки документа — нужно при переиндексации изменившегося файла."""
    settings = settings or get_settings()
    client.delete(
        collection_name=settings.collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        ),
        wait=True,
    )


def document_sources(
    client: QdrantClient, doc_id: str, settings: Settings | None = None
) -> set[str]:
    """Из каких файлов в индексе лежат чанки документа ``doc_id``."""
    settings = settings or get_settings()
    if not client.collection_exists(settings.collection_name):
        return set()
    selector = models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )
    sources: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=selector,
            limit=256,
            offset=offset,
            with_payload=["source_path"],
            with_vectors=False,
        )
        sources.update(str((p.payload or {}).get("source_path") or "") for p in points)
        if offset is None:
            break
    sources.discard("")
    return sources


def prune_missing_sources(
    client: QdrantClient, root: Path, settings: Settings | None = None
) -> list[str]:
    """Удалить из индекса документы, чьих файлов под ``root`` больше нет.

    Без этого удалённый или переименованный стандарт жил в индексе вечно и
    продолжал попадать в ответы. Трогаются только файлы внутри ``root``:
    индексация одного файла из другой папки не должна чистить весь корпус.
    """
    settings = settings or get_settings()
    root = root.resolve()
    removed: list[str] = []
    for doc in list_documents(client, settings):
        source = doc.get("source_path")
        if not source:
            continue
        path = Path(source).resolve()
        if not path.is_relative_to(root) or path.exists():
            continue
        delete_document(client, doc["doc_id"], settings)
        removed.append(doc["doc_id"])
        log.info("pruned_missing_source", doc_id=doc["doc_id"], source=source)
    return removed


def count_points(client: QdrantClient, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    if not client.collection_exists(settings.collection_name):
        return 0
    return client.count(settings.collection_name, exact=True).count


def list_documents(client: QdrantClient, settings: Settings | None = None) -> list[dict]:
    """Сводка по корпусу: документ, обозначение, статус, число чанков."""
    settings = settings or get_settings()
    if not client.collection_exists(settings.collection_name):
        return []

    docs: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.collection_name,
            limit=512,
            offset=offset,
            with_payload=["doc_id", "designation", "title", "status", "source_path"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            entry = docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "designation": payload.get("designation"),
                    "title": payload.get("title"),
                    "status": payload.get("status"),
                    "source_path": payload.get("source_path"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        if offset is None:
            break

    return sorted(docs.values(), key=lambda d: (d["designation"] or "", d["doc_id"]))
