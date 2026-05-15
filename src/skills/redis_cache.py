"""
L0b Redis Shared Cache — persistent, cross-worker exact-match cache.

Extends L0a (local TTLCache) with Redis for:
  - Persistence across process restarts
  - Sharing across multiple API workers
  - Higher total capacity (limited by Redis memory, not process memory)

Graceful degradation: if Redis is unavailable, the system falls back
to local TTLCache-only operation with zero impact on availability.

Usage:
    from src.skills.redis_cache import redis_cache
    redis_cache.get(text)   # → dict or None
    redis_cache.set(text, decision, confidence, reason)
"""

import json
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL = int(os.getenv("REDIS_CACHE_TTL", "7200"))  # 2 hours (vs local 1h)
REDIS_PREFIX = "mod:v1:"  # versioned prefix for easy cache invalidation


class RedisCache:
    """Redis-backed exact-match cache with graceful fallback."""

    def __init__(self, redis_url: str | None = None, ttl: int | None = None):
        self.redis_url = redis_url or REDIS_URL
        self.ttl = ttl or REDIS_TTL
        self._client = None
        self._available = None  # None=untested, True=ok, False=down
        self.hits = 0
        self.misses = 0

    # -- lazy init with health check --

    def _ensure_client(self) -> bool:
        """Lazy-init Redis client. Returns True if connected."""
        if self._available is False:
            return False
        if self._client is not None:
            return True
        try:
            import redis
            self._client = redis.Redis.from_url(
                self.redis_url,
                socket_timeout=0.1,           # 100ms
                socket_connect_timeout=0.05,  # 50ms connect timeout
                decode_responses=True,
            )
            self._client.ping()
            self._available = True
            logger.info("Redis connected: %s (TTL=%ds)", self.redis_url, self.ttl)
            return True
        except Exception as e:
            self._available = False
            self._client = None
            logger.warning("Redis unavailable (%s) — using local cache only", e)
            return False

    # -- public API --

    def get(self, text: str) -> dict | None:
        """Lookup exact text in Redis. Returns cached decision or None."""
        if not self._ensure_client():
            return None
        try:
            key = self._make_key(text)
            raw = self._client.get(key)
            if raw is None:
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(raw)
        except Exception as e:
            logger.debug("Redis GET failed: %s", e)
            self._available = False  # mark down, will retry next call
            return None

    def set(self, text: str, decision: str, confidence: float, reason: str,
            tier: str = ""):
        """Store moderation result in Redis with TTL."""
        if not self._ensure_client():
            return
        try:
            key = self._make_key(text)
            value = json.dumps({
                "decision": decision,
                "confidence": confidence,
                "reason": reason,
                "tier": tier,
            }, ensure_ascii=False)
            self._client.setex(key, self.ttl, value)
        except Exception as e:
            logger.debug("Redis SET failed: %s", e)

    def invalidate(self, text: str):
        """Remove a cached entry (called when human review overrides)."""
        if not self._ensure_client():
            return
        try:
            self._client.delete(self._make_key(text))
        except Exception as e:
            logger.debug("Redis DEL failed: %s", e)

    # -- helpers --

    def _make_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{REDIS_PREFIX}{digest}"

    # -- stats --

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def status(self) -> str:
        if self._available is None:
            return "untested"
        return "connected" if self._available else "unavailable"


# Singleton
redis_cache = RedisCache()
