"""
Gateway — hot-path pre-filter. Handles 80%+ of traffic without touching LangGraph.

Flow:
  1. Image URL hash lookup (< 0.01ms) → return cached decision (image dedup)
  2. L0 Memory cache (SHA256, < 0.01ms) → return cached decision (exact duplicates)
  3. Keyword filter (AC automaton, < 0.5ms) → block immediately or escalate
  4. L1 ChromaDB cache (embedding, < 5ms) → return cached decision (semantic dup)
  5. If all miss → return decision=None + full traces (escalate to LangGraph)

Always returns a dict: {"decision": dict|None, "traces": [...], "keyword_prefiltered": bool, ...}
"""

import time
import hashlib
import logging
from src.skills.keyword_filter import keyword_filter
from src.skills.vector_cache import vector_cache
from src.skills.embedder import embedder
from src.skills.memory_cache import memory_cache
from src.skills.redis_cache import redis_cache
from src.skills.image_phash import image_phash

logger = logging.getLogger("gateway")


# ---------------------------------------------------------------------------
# Gateway class
# ---------------------------------------------------------------------------

class Gateway:
    """Hot-path pre-filter with multi-tier caching and keyword blocking."""

    def __init__(self):
        self.stats = {
            "total": 0,
            "image_cache_hit": 0,
            "memory_cache_hit": 0,
            "redis_cache_hit": 0,
            "keyword_hit": 0,
            "whitelist_hit": 0,
            "chroma_cache_hit": 0,
            "escalated": 0,
        }

    # -- public API --

    def check(self, text: str, image_url: str = "",
              image_base64: str = "") -> dict:
        """Fast pre-check. Always returns a dict:

        {
          "decision": dict | None,   # None = escalate to cold path
          "traces": [...],           # Always populated
          "keyword_prefiltered": bool,
          "keyword_label": str|None,
          "keyword_confidence": float,
        }
        """
        t0 = time.perf_counter()
        traces = []
        self.stats["total"] += 1

        # -- image present → multi-tier image hot path --
        has_image = bool(
            (image_url and image_url.strip()) or
            (image_base64 and image_base64.strip())
        )
        if has_image:
            # Only apply pHash if we have actual image bytes (base64)
            if image_base64 and image_base64.strip():
                image_bytes = None
                try:
                    import base64
                    raw = image_base64
                    if "," in raw:
                        raw = raw.split(",", 1)[1]
                    image_bytes = base64.b64decode(raw)
                except Exception:
                    pass

                if image_bytes:
                    # Step I1: Perceptual hash → check known harmful DB
                    t_ph = time.perf_counter()
                    img_hash = image_phash.dhash(image_bytes)
                    ph_ms = (time.perf_counter() - t_ph) * 1000

                    known = image_phash.check_known(img_hash)
                    if known is not None:
                        self.stats["keyword_hit"] += 1  # reuse keyword counter for "hard block"
                        traces.append(_t("gateway", "image_phash_block",
                            f"dHash · Hamming≤{image_phash._known_hashes and '10' or 'N/A'}",
                            image_url or "base64",
                            {"phash": img_hash, "category": known["category"],
                             "action": "block", "source": known.get("source", "?")},
                            ph_ms, "zero"))
                        return self._resolve(
                            {"decision": "block", "confidence": 1.0,
                             "reason": f"Perceptual hash match: {known['category']}",
                             "tier": "L0_phash"},
                            traces, keyword_prefiltered=False)

                    # Step I2: Check if this pHash was seen before (semantic image cache)
                    ph_mem = memory_cache.get(f"[PH]{img_hash}")
                    if ph_mem is not None:
                        self.stats["image_cache_hit"] += 1
                        traces.append(_t("gateway", "image_phash_cache",
                            f"dHash · cached", image_url or "base64",
                            {**ph_mem, "phash": img_hash},
                            ph_ms, "zero"))
                        return self._resolve(ph_mem, traces,
                                           keyword_prefiltered=False)

                    traces.append(_t("gateway", "image_phash_ok",
                        f"dHash · {img_hash[:16]}", image_url or "base64",
                        {"phash": img_hash, "action": "escalate"},
                        ph_ms, "zero"))
                else:
                    traces.append(_t("gateway", "image_decode_fail",
                        "none", image_url or "base64",
                        {"error": "Cannot decode base64 image"},
                        0.0, "zero"))
            else:
                traces.append(_t("gateway", "image_no_bytes",
                    "none", image_url or "base64",
                    {"reason": "URL-only image — cannot compute pHash without bytes"},
                    0.0, "zero"))

            # Step I3: Exact URL hash cache (fallback for URL-only images)
            img_ref_hash = self._hash_image_ref(image_url, image_base64)
            ref_result = memory_cache.get(f"[IMG]{img_ref_hash}")
            if ref_result is not None:
                self.stats["image_cache_hit"] += 1
                traces.append(_t("gateway", "image_url_cache_hit",
                    "SHA256 · image ref", image_url or "base64",
                    ref_result, 0.0, "zero"))
                return self._resolve(ref_result, traces,
                                     keyword_prefiltered=False)

            # Image not resolved in hot path → escalate
            self.stats["escalated"] += 1
            traces.append(_t("gateway", "image_escalate",
                "none", image_url or "base64",
                {"reason": "Image not in any hot-path cache → cold path"},
                (time.perf_counter() - t0) * 1000, "zero"))
            return self._escalate(traces, keyword_prefiltered=False)

        # -- empty content --
        if not text or not text.strip():
            return self._resolve(
                {"decision": "pass", "confidence": 1.0,
                 "reason": "empty content", "tier": "L0_empty"},
                traces, keyword_prefiltered=False)

        # -- Step 0: L0 memory cache (SHA256 exact match) --
        t_mem = time.perf_counter()
        mem_result = memory_cache.get(text)
        mem_ms = (time.perf_counter() - t_mem) * 1000
        if mem_result is not None:
            self.stats["memory_cache_hit"] += 1
            traces.append(_t("gateway", "L0_memory_hit",
                "SHA256 · TTLCache(1h)", text, mem_result, mem_ms, "zero"))
            return self._resolve(mem_result, traces, keyword_prefiltered=False)
        traces.append(_t("gateway", "L0_memory_miss",
            "SHA256 · TTLCache(1h)", text, {"miss": True}, mem_ms, "zero"))

        # -- Step 0b: L0b Redis cache (shared across workers, survives restart) --
        t_redis = time.perf_counter()
        redis_result = redis_cache.get(text)
        redis_ms = (time.perf_counter() - t_redis) * 1000
        if redis_result is not None:
            self.stats["redis_cache_hit"] += 1
            # Also populate local cache so next lookup on this worker is instant
            memory_cache.set(text, redis_result["decision"],
                           redis_result.get("confidence", 1.0),
                           redis_result.get("reason", ""))
            traces.append(_t("gateway", "L0b_redis_hit",
                "Redis · shared L0", text, redis_result, redis_ms, "zero"))
            return self._resolve(redis_result, traces, keyword_prefiltered=False)
        traces.append(_t("gateway", "L0b_redis_miss",
            "Redis · shared L0", text, {"miss": True}, redis_ms, "zero"))

        # -- Step 1: Keyword filter (AC automaton) --
        t1 = time.perf_counter()
        kw = keyword_filter.match(text)
        kw_ms = (time.perf_counter() - t1) * 1000

        # Whitelist hit → pass immediately
        if kw.get("whitelist_hit") and kw["confidence"] == 0.0:
            self.stats["whitelist_hit"] += 1
            traces.append(_t("gateway", "whitelist_pass",
                "AC自动机 · 白名单", text,
                {"whitelist": True,
                 "suppressed": kw.get("suppressed_matches", [])},
                kw_ms, "zero"))
            return self._resolve(
                {"decision": "pass", "confidence": 1.0,
                 "reason": "Whitelist match — known false positive phrase",
                 "tier": "L0_whitelist"},
                traces, keyword_prefiltered=True,
                kw_label=None, kw_confidence=0.0)

        # Standalone keyword (conf=1.0) → block immediately
        if kw["confidence"] > 0.99:
            self.stats["keyword_hit"] += 1
            matches = kw.get("matches", [])
            matched_keywords = [{"word": m["word"], "category": m["category"], "context": m["context"]} for m in matches]

            # Include combo words if present
            combo = kw.get("combo_hit")
            if combo:
                for w in combo.get("matched_words", []):
                    if not any(m["word"] == w for m in matched_keywords):
                        matched_keywords.append({
                            "word": w, "category": combo["label"],
                            "context": "combo",
                        })
                matched_keywords.append({
                    "word": f"[{combo.get('note', '')}]",
                    "category": "combo_rule",
                    "context": "combo_trigger",
                })

            trace_output = {
                "label": kw["label"],
                "confidence": 1.0,
                "matched_keywords": matched_keywords,
            }
            traces.append(_t("gateway", "keyword_block",
                "AC自动机 · 独立词", text, trace_output, kw_ms, "zero"))
            return self._resolve(
                {"decision": "block", "confidence": 1.0,
                 "reason": f"Hot-path keyword match: {kw['label']}",
                 "tier": "L1_keyword"},
                traces, keyword_prefiltered=True,
                kw_label=kw["label"], kw_confidence=1.0)

        # Ambiguous keyword (conf=0.6, embedded in larger word) → escalate
        if kw["confidence"] >= 0.4:
            matches = kw.get("matches", [])
            traces.append(_t("gateway", "keyword_ambiguous",
                "AC自动机 · 嵌入词", text,
                {"label": kw["label"], "confidence": kw["confidence"],
                 "matched_keywords": [{"word": m["word"], "category": m["category"], "context": m["context"]} for m in matches],
                 "reason": "Keyword embedded in larger word → escalate to BERT"},
                kw_ms, "zero"))
            return self._escalate(traces, keyword_prefiltered=True,
                                  kw_label=kw["label"],
                                  kw_confidence=kw["confidence"])

        traces.append(_t("gateway", "keyword_miss",
            "AC自动机", text, {"label": None}, kw_ms, "zero"))

        # -- Step 2: L1 ChromaDB semantic cache --
        t2 = time.perf_counter()
        embedding = embedder.embed(text)
        cached = vector_cache.lookup(embedding)
        cache_ms = (time.perf_counter() - t2) * 1000

        if cached:
            self.stats["chroma_cache_hit"] += 1
            traces.append(_t("gateway", "L1_chroma_hit",
                "ChromaDB · cosine≥0.95", text, cached, cache_ms, "zero"))
            return self._resolve(cached, traces, keyword_prefiltered=True,
                                 kw_label=None, kw_confidence=0.0)

        # -- All missed → escalate to LangGraph --
        self.stats["escalated"] += 1
        traces.append(_t("gateway", "L1_chroma_miss",
            "ChromaDB", text,
            {"similarity_below_threshold": True}, cache_ms, "zero"))
        return self._escalate(traces, keyword_prefiltered=True,
                              kw_label=None, kw_confidence=0.0)

    # -- helpers --

    def _resolve(self, decision: dict, traces: list,
                 keyword_prefiltered: bool,
                 kw_label: str | None = None,
                 kw_confidence: float = 0.0) -> dict:
        # Write to L0 memory cache so next identical request hits instantly
        text = traces[-1].get("input", "") if traces else ""
        if text and text.strip():
            memory_cache.set(
                text,
                decision.get("decision", "pass"),
                decision.get("confidence", 1.0),
                decision.get("reason", ""),
                decision.get("tier", ""),
            )
            # Fire-and-forget: also write to L1 ChromaDB so semantically
            # similar content hits in future batches (runs in background)
            self._cache_chroma_async(text, decision)
        return {
            "decision": decision,
            "traces": traces,
            "keyword_prefiltered": keyword_prefiltered,
            "keyword_label": kw_label,
            "keyword_confidence": kw_confidence,
        }

    @staticmethod
    def _cache_chroma_async(text: str, decision: dict):
        """Write hot-path result to ChromaDB in a background thread."""
        import threading

        def _store():
            try:
                vec = embedder.embed(text)
                vector_cache.store(
                    embedding=vec,
                    text=text,
                    decision=decision.get("decision", "pass"),
                    confidence=decision.get("confidence", 1.0),
                    reason=decision.get("reason", ""),
                    tier=decision.get("tier", ""),
                )
            except Exception:
                pass  # ChromaDB write failure shouldn't affect the hot path

        t = threading.Thread(target=_store, daemon=True)
        t.start()

    def _escalate(self, traces: list,
                  keyword_prefiltered: bool,
                  kw_label: str | None = None,
                  kw_confidence: float = 0.0) -> dict:
        return {
            "decision": None,
            "traces": traces,
            "keyword_prefiltered": keyword_prefiltered,
            "keyword_label": kw_label,
            "keyword_confidence": kw_confidence,
        }

    @staticmethod
    def _hash_image_ref(image_url: str, image_base64: str) -> str:
        """Deterministic hash for image reference (URL or base64 prefix)."""
        raw = image_url or image_base64[:128]
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -- stats --

    def get_stats(self) -> dict:
        s = self.stats
        total = max(s["total"], 1)
        return {
            "total_requests": s["total"],
            "image_cache_hit_rate": round(s["image_cache_hit"] / total, 4),
            "memory_cache_hit_rate": round(s["memory_cache_hit"] / total, 4),
            "redis_cache_hit_rate": round(s["redis_cache_hit"] / total, 4),
            "keyword_hit_rate": round(s["keyword_hit"] / total, 4),
            "whitelist_hit_rate": round(s["whitelist_hit"] / total, 4),
            "chroma_cache_hit_rate": round(s["chroma_cache_hit"] / total, 4),
            "hot_path_rate": round(
                (s["image_cache_hit"] + s["memory_cache_hit"] +
                 s["redis_cache_hit"] +
                 s["keyword_hit"] + s["whitelist_hit"] +
                 s["chroma_cache_hit"]) / total, 4),
            "escalated_rate": round(s["escalated"] / total, 4),
            "langgraph_calls_saved": (
                s["image_cache_hit"] + s["memory_cache_hit"] +
                s["redis_cache_hit"] +
                s["keyword_hit"] + s["whitelist_hit"] +
                s["chroma_cache_hit"]),
        }


# ---------------------------------------------------------------------------
# Trace helper
# ---------------------------------------------------------------------------

def _t(node: str, step: str, model: str, text: str, output: dict,
       latency_ms: float, cost: str) -> dict:
    return {
        "node": node, "step": step, "model": model,
        "input": text[:200], "output": output,
        "latency_ms": round(latency_ms, 2), "cost": cost,
        "ts": int(time.time() * 1000),
    }


# Singleton
gateway = Gateway()
