"""Запуск веб-интерфейса."""

from __future__ import annotations

import uvicorn

from gost_rag.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "gost_rag.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
