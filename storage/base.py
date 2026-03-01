"""
StorageBackend (Strategy Pattern)
----------------------------------
Abstract interface for persisting option chain results.
Concrete implementations: FileStorageBackend, DbStorageBackend.

StorageChain (Chain of Responsibility)
---------------------------------------
Walks a list of handlers; the first handler that can_handle() wins.
This lets you switch between file / DB / both without touching strategy code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class SaveContext:
    """All data needed to persist one batch of results."""
    rows:        list[dict]
    spot_price:  float
    date_str:    str
    time_str:    Optional[str]
    expiry_dt:   Optional[datetime]
    is_expired:  bool
    prefix:      str          # 'option' | 'mcx'
    symbol:      str          # 'NIFTY', 'BANKNIFTY', etc.
    is_fallback: bool         = False



# ── Strategy: StorageBackend ─────────────────────────────────────────────────

class StorageBackend(ABC):
    """Abstract storage strategy."""

    @abstractmethod
    def save(self, ctx: SaveContext) -> bool:
        """Persist data; return True on success."""

    @abstractmethod
    def can_handle(self, ctx: SaveContext) -> bool:
        """Return True if this backend is available/configured."""


# ── Chain of Responsibility: StorageHandler ───────────────────────────────────

class StorageHandler(ABC):
    """One node in the storage chain."""

    def __init__(self):
        self._next: Optional[StorageHandler] = None

    def set_next(self, handler: StorageHandler) -> StorageHandler:
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, ctx: SaveContext) -> bool:
        """Handle or forward."""

    def _forward(self, ctx: SaveContext) -> bool:
        if self._next:
            return self._next.handle(ctx)
        print("[StorageChain] No handler could persist the data.")
        return False


class StorageChain:
    """Builds and executes a chain of StorageHandlers."""

    def __init__(self, handlers: list[StorageHandler]):
        if not handlers:
            raise ValueError("StorageChain requires at least one handler.")
        for i in range(len(handlers) - 1):
            handlers[i].set_next(handlers[i + 1])
        self._head = handlers[0]

    def save(self, ctx: SaveContext) -> bool:
        return self._head.handle(ctx)
