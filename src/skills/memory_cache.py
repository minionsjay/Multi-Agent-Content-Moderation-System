"""
L0 In-Memory Cache — exact text hash lookup in < 0.01ms.

Sits BEFORE ChromaDB (L1). Catches exact duplicate texts instantly.
Uses TTLCache with 1-hour TTL, max 100K entries (~50MB memory).
"""

import hashlib
import logging
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class MemoryCache:
    def __init__(self, maxsize: int = 100_000, ttl: int = 3600):
        self._store: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> dict | None:
        key = self._key(text)
        entry = self._store.get(key)
        if entry is not None:
            self.hits += 1
            return entry
        self.misses += 1
        return None

    def set(self, text: str, decision: str, confidence: float, reason: str):
        key = self._key(text)
        self._store[key] = {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
        }

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)


# Singleton
memory_cache = MemoryCache()
