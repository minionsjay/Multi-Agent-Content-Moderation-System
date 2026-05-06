import logging
import hashlib
import numpy as np
from cachetools import TTLCache
from src.config import EMBED_MODEL

logger = logging.getLogger(__name__)

# Per-process embedding cache: avoids recomputing identical vectors.
# 500K entries × (64B key + 512×4B float32) ≈ 1 GB at capacity.
_EMB_CACHE_SIZE = 500_000
_EMB_CACHE_TTL = 3600  # 1 hour


class Embedder:
    """Sentence-transformers embedding using BGE-small-zh (POC).

    BGE-small-zh-v1.5: 512-dim, 95 MB, optimized for Chinese.
    Includes an in-memory LRU cache (SHA256 keyed) to eliminate
    redundant BGE inference for duplicate/spam texts.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or EMBED_MODEL
        self._model = None
        self._dim = None
        self._emb_cache: TTLCache | None = None
        self._cache_hits = 0
        self._cache_misses = 0

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_embedding_dimension()
        self._emb_cache = TTLCache(maxsize=_EMB_CACHE_SIZE, ttl=_EMB_CACHE_TTL)
        logger.info("Embedder loaded: %s (dim=%d, emb_cache=%d)", self._model_name, self._dim, _EMB_CACHE_SIZE)

    @property
    def DIM(self) -> int:
        self._load()
        return self._dim

    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            self._load()
            return [0.0] * self._dim
        self._load()

        # Check embedding cache
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self._emb_cache is not None:
            cached = self._emb_cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                return cached
        self._cache_misses += 1

        vec = self._model.encode(text[:8191], normalize_embeddings=True).tolist()

        if self._emb_cache is not None:
            self._emb_cache[key] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load()
        # Check cache for each text; only encode misses
        results = []
        miss_indices = []
        miss_texts = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                results.append([0.0] * self._dim)
                continue
            key = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if self._emb_cache is not None:
                cached = self._emb_cache.get(key)
                if cached is not None:
                    self._cache_hits += 1
                    results.append(cached)
                    continue
            self._cache_misses += 1
            miss_indices.append(i)
            miss_texts.append(t[:8191])
            results.append(None)  # placeholder

        if miss_texts:
            vecs = self._model.encode(
                miss_texts, normalize_embeddings=True, show_progress_bar=False,
            ).tolist()
            for idx, vec in zip(miss_indices, vecs):
                results[idx] = vec
                if self._emb_cache is not None:
                    key = hashlib.sha256(texts[idx].encode("utf-8")).hexdigest()
                    self._emb_cache[key] = vec

        return results  # type: ignore[return-value]


# Singleton
embedder = Embedder()
