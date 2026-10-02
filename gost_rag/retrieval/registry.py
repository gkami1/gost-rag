"""Статус документа — из реестра в момент ответа, а не из индекса.

Раньше статус копировался в payload чанка при индексации, а журнал индексации
пропускает неизменённые файлы. Правка «действующий -> отменён» в
``data/registry/documents.csv`` не доходила до ответа, пока индекс не
пересобирали с ``--recreate``: пользователь видел рядом с цитатой статус,
который реестр уже опроверг. Решение №8 («статус — из реестра») на деле не
выполнялось.

Теперь реестр читается при ответе и перечитывается, как только файл изменился.
Статус из индекса остаётся запасным — для документов, которых в реестре нет.
"""

from __future__ import annotations

from pathlib import Path

from gost_rag.ingest.metadata import coerce_status, load_registry, registry_key
from gost_rag.logging import get_logger
from gost_rag.models import RetrievedChunk

log = get_logger(__name__)


class LiveRegistry:
    """Реестр документов, перечитываемый при изменении файла."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stamp: tuple[float, int] | None = None
        self._rows: dict[str, dict[str, str]] = {}

    def rows(self) -> dict[str, dict[str, str]]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._stamp, self._rows = None, {}
            return self._rows
        # Размер вместе со временем: правка в ту же секунду не должна потеряться.
        stamp = (stat.st_mtime, stat.st_size)
        if stamp != self._stamp:
            self._rows = load_registry(self.path)
            self._stamp = stamp
        return self._rows

    def overlay(self, payload: dict) -> dict:
        """Поля паспорта документа из реестра поверх тех, что лежат в индексе."""
        designation = payload.get("designation")
        row = self.rows().get(registry_key(designation)) if designation else None
        if not row:
            return payload
        if row.get("status"):
            payload["status"] = coerce_status(row["status"])
        # Пустая клетка в реестре — тоже ответ: замены нет.
        payload["replaced_by"] = row.get("replaced_by") or None
        for field in ("title", "source_url"):
            if row.get(field):
                payload[field] = row[field]
        return payload

    def apply(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        for chunk in chunks:
            self.overlay(chunk.payload)
        return chunks
