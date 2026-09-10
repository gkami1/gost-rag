"""Обёртка над CLI индексации: python scripts/ingest.py data/raw"""

from gost_rag.ingest.pipeline import app

if __name__ == "__main__":
    app()
