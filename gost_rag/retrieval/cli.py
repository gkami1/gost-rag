"""CLI для отладки поиска: показать, что нашёл RRF и как это переставил реранкер.

Отдельная команда нужна, чтобы отлаживать поиск без LLM: если ответ плохой,
сначала надо понять, дошёл ли нужный фрагмент до контекста вообще.
"""

from __future__ import annotations

import typer

from gost_rag.config import get_settings
from gost_rag.logging import configure_logging

app = typer.Typer(add_completion=False, help="Отладка гибридного поиска.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Вопрос на русском"),
    top: int = typer.Option(6, "--top", help="Сколько фрагментов показать после реранкера"),
    chars: int = typer.Option(300, "--chars", help="Сколько символов текста печатать"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Показать выдачу сразу после RRF"),
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    from gost_rag.ingest.embed import get_embedder
    from gost_rag.ingest.index import get_client
    from gost_rag.retrieval.filters import build_filter, known_designations
    from gost_rag.retrieval.rerank import get_reranker
    from gost_rag.retrieval.store import hybrid_search

    client = get_client(settings)
    try:
        query_filter = build_filter(query, known_designations(client, settings))
        if query_filter is not None:
            typer.secho("Применён фильтр по обозначению из запроса.", fg=typer.colors.BLUE)

        candidates = hybrid_search(
            client, get_embedder().encode_one(query), settings, query_filter=query_filter
        )
        if not candidates:
            typer.secho("Ничего не найдено. Индекс пуст?", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        typer.secho(f"\nПосле RRF: {len(candidates)} кандидатов", fg=typer.colors.CYAN, bold=True)
        if no_rerank:
            _print(candidates[:top], chars, score_attr="fusion_score")
            return

        ranked = get_reranker().rerank(query, candidates, top_n=top)
        typer.secho(f"После реранкера: {len(ranked)}\n", fg=typer.colors.CYAN, bold=True)
        _print(ranked, chars, score_attr="rerank_score")

        rrf_order = [c.point_id for c in candidates[:top]]
        moved = [c.point_id for c in ranked if c.point_id not in rrf_order]
        if moved:
            typer.secho(
                f"Реранкер поднял в топ-{top} фрагментов, которых там не было: {len(moved)}",
                fg=typer.colors.MAGENTA,
            )
    finally:
        client.close()


def _print(chunks, chars: int, *, score_attr: str) -> None:
    for index, chunk in enumerate(chunks, start=1):
        score = getattr(chunk, score_attr) or 0.0
        typer.secho(f"[{index}] {chunk.citation_label()}", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"    score={score:.4f}")
        snippet = chunk.text[:chars].replace("\n", "\n    ")
        typer.echo(f"    {snippet}{'…' if len(chunk.text) > chars else ''}\n")


if __name__ == "__main__":
    app()
