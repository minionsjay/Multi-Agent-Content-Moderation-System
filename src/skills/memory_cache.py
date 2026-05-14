"""
L0 In-Memory Cache — exact text hash lookup in < 0.01ms.

Sits BEFORE ChromaDB (L1). Catches exact duplicate texts instantly.
Uses TTLCache with 1-hour TTL, max 100K entries (~50MB memory).

Persistence: saves to ./data/memory_cache.json on each write. Survives restarts.
"""

import hashlib
import json
import logging
import os
from cachetools import TTLCache

logger = logging.getLogger(__name__)

MEMORY_CACHE_PATH = os.getenv("MEMORY_CACHE_PATH", "./data/memory_cache.json")


class MemoryCache:
    def __init__(self, maxsize: int = 100_000, ttl: int = 3600):
        self._store: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.hits = 0
        self.misses = 0
        self._path = MEMORY_CACHE_PATH
        self._load()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Restore cache from disk on startup."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            loaded = 0
            for key, entry in data.items():
                if len(self._store) >= self._store.maxsize:
                    break
                self._store[key] = entry
                loaded += 1
            logger.info("Memory cache loaded from disk: %d entries", loaded)
        except Exception as e:
            logger.warning("Failed to load memory cache from %s: %s", self._path, e)

    def _save(self):
        """Dump cache to disk as JSON."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            data = dict(self._store)
            with open(self._path, "w") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save memory cache: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> dict | None:
        key = self._key(text)
        entry = self._store.get(key)
        if entry is not None:
            self.hits += 1
            return entry
        self.misses += 1
        return None

    def set(self, text: str, decision: str, confidence: float, reason: str,
            tier: str = ""):
        key = self._key(text)
        self._store[key] = {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "tier": tier,
        }
        self._save()

    def clear(self):
        """Clear all cached entries and delete the disk file."""
        self._store.clear()
        self.hits = 0
        self.misses = 0
        try:
            if os.path.exists(self._path):
                os.remove(self._path)
        except Exception as e:
            logger.warning("Failed to remove cache file: %s", e)
        logger.info("Memory cache cleared (%d entries purged)", self.size)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)


# Singleton
memory_cache = MemoryCache()
