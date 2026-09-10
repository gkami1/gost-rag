"""Гибридный поиск: плотный + разреженный префетч и RRF на стороне Qdrant.

Один вызов ``query_points`` выполняет оба поиска и сливает их Reciprocal Rank
Fusion. Слияние на стороне базы — не только экономия round-trip: ранги считаются
по полным спискам кандидатов, а не по усечённым выдачам, склеенным в приложении.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from gost_rag.config import Settings, get_settings
from gost_rag.ingest.embed import Embedding
from gost_rag.logging import get_logger
from gost_rag.models import RetrievedChunk

log = get_logger(__name__)


def hybrid_search(
    client: QdrantClient,
    query_embedding: Embedding,
    settings: Settings | None = None,
    *,
    query_filter: models.Filter | None = None,
    limit: int | None = None,
    prefetch_limit: int | None = None,
) -> list[RetrievedChunk]:
    """Вернуть кандидатов после RRF-слияния плотного и разреженного поиска."""
    settings = settings or get_settings()
    limit = limit or settings.fusion_limit
    prefetch_limit = prefetch_limit or settings.prefetch_limit

    if not client.collection_exists(settings.collection_name):
        log.warning("collection_missing", collection=settings.collection_name)
        return []

    prefetch = [
        models.Prefetch(
            query=query_embedding.dense,
            using=settings.dense_vector_name,
            limit=prefetch_limit,
            filter=query_filter,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=query_embedding.sparse_indices,
                values=query_embedding.sparse_values,
            ),
            using=settings.sparse_vector_name,
            limit=prefetch_limit,
            filter=query_filter,
        ),
    ]

    response = client.query_points(
        collection_name=settings.collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return [
        RetrievedChunk(
            point_id=str(point.id),
            text=(point.payload or {}).get("text", ""),
            payload=point.payload or {},
            fusion_score=point.score,
        )
        for point in response.points
    ]


def dense_only_search(
    client: QdrantClient,
    query_embedding: Embedding,
    settings: Settings | None = None,
    *,
    query_filter: models.Filter | None = None,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Только плотный поиск — нужен оценке, чтобы измерить вклад гибрида."""
    settings = settings or get_settings()
    response = client.query_points(
        collection_name=settings.collection_name,
        query=query_embedding.dense,
        using=settings.dense_vector_name,
        query_filter=query_filter,
        limit=limit or settings.fusion_limit,
        with_payload=True,
    )
    return [
        RetrievedChunk(
            point_id=str(point.id),
            text=(point.payload or {}).get("text", ""),
            payload=point.payload or {},
            fusion_score=point.score,
        )
        for point in response.points
    ]
