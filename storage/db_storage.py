"""
DbStorageBackend — stub for future DB persistence.

Set DB_URL env var to activate. Chain falls through to FileStorage if unset.
"""

from __future__ import annotations

import os

from storage.base import SaveContext, StorageBackend, StorageHandler


class DbStorageBackend(StorageBackend):
    """Placeholder: persist to a relational database."""

    def __init__(self, db_url: str):
        self.db_url = db_url

    def can_handle(self, ctx: SaveContext) -> bool:
        return bool(self.db_url)

    def save(self, ctx: SaveContext) -> bool:
        # TODO: implement SQLAlchemy / psycopg2 insert
        print(f"[DbStorage] (stub) would write {len(ctx.rows)} rows to {self.db_url}")
        return True


class DbStorageHandler(StorageHandler):
    """Chain node for DB storage — skipped when DB_URL is not configured."""

    def __init__(self):
        super().__init__()
        self._db_url = os.getenv("DB_URL", "")
        self._backend = DbStorageBackend(self._db_url) if self._db_url else None

    def handle(self, ctx: SaveContext) -> bool:
        if self._backend and self._backend.can_handle(ctx):
            return self._backend.save(ctx)
        return self._forward(ctx)


def build_storage_chain():
    """
    Factory: builds the default storage chain.
      DbStorageHandler → FileStorageHandler

    DB is tried first. If DB_URL is not set, falls through to file.
    """
    from storage.file_storage import FileStorageHandler
    db   = DbStorageHandler()
    file = FileStorageHandler()
    db.set_next(file)
    return db   # head of chain
