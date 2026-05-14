import time
import logging
from src.state import ModerationState
from src.skills.vector_cache import vector_cache
from src.skills.embedder import embedder
from src.skills.memory_cache import memory_cache
from src.skills.redis_cache import redis_cache
from src.skills.review_queue import review_queue
from src.feedback.event_collector import event_collector

logger = logging.getLogger(__name__)


async def action_executor(state: ModerationState) -> dict:
    t0 = time.perf_counter()
    traces = []
    decision = state.get("decision", "pass")
    text = state.get("text", "")

    # Determine tier for cache storage
    text_result = state.get("text_result") or {}
    image_result = state.get("image_result") or {}
    tier = text_result.get("tier", "") if text_result else ""

    # Write result to caches
    if not state.get("cache_hit") and text.strip():
        # L0a: Local memory cache (instant exact-match lookup)
        memory_cache.set(text, decision, state.get("confidence", 0.0),
                        state.get("reason", ""), tier)
        traces.append(_trace("action", "cache_L0a_store", "Python dict · SHA256",
                            f"Storing exact match: {decision}", {"cached": True},
                            0.0, "zero"))

        # L0b: Redis shared cache (persistent, cross-worker)
        redis_cache.set(text, decision, state.get("confidence", 0.0),
                       state.get("reason", ""), tier)
        traces.append(_trace("action", "cache_L0b_store", "Redis · shared",
                            f"Storing in Redis: {decision}", {"cached": True},
                            0.0, "zero"))

        # L1: ChromaDB semantic cache
        t1 = time.perf_counter()
        try:
            embedding = embedder.embed(text)
            vector_cache.store(
                embedding=embedding, text=text, decision=decision,
                confidence=state.get("confidence", 0.0),
                reason=state.get("reason", ""),
                tier=tier,
            )
            traces.append(_trace("action", "cache_L1_store", "TF-IDF · ChromaDB",
                                f"Storing semantic: {decision}", {"cached": True},
                                (time.perf_counter() - t1) * 1000, "zero"))
        except Exception as e:
            logger.warning("Failed to write ChromaDB cache: %s", e)

    # Human review queue: grey zone → human moderator
    if decision == "review":
        review_id = review_queue.enqueue({
            "content_id": state.get("content_id", "?"),
            "text": state.get("text", ""),
            "image_url": state.get("image_url", ""),
            "ai_decision": decision,
            "ai_confidence": state.get("confidence", 0.0),
            "ai_reason": state.get("reason", ""),
            "signals": {
                "text_result": state.get("text_result"),
                "image_result": state.get("image_result"),
            },
            "priority": state.get("priority_score", 0.5),
            "traces": state.get("traces", []),
        })
        traces.append(_trace("action", "human_review_queue", "JSONL · review_queue",
                            f"Queued for human review: {review_id}",
                            {"review_id": review_id, "priority": state.get("priority_score", 0.5)},
                            (time.perf_counter() - t0) * 1000, "zero"))

    traces.append(_trace("action", "execute", "none",
                        f"Action: {decision}", {"action_taken": decision},
                        (time.perf_counter() - t0) * 1000, "zero"))

    text_result = state.get("text_result") or {}
    image_result = state.get("image_result") or {}

    # Record event for offline feedback loop (non-blocking)
    try:
        event_collector.record({
            "content_id": state.get("content_id", "?"),
            "text": state.get("text", ""),
            "image_url": state.get("image_url", ""),
            "user_id": state.get("user_id", "anonymous"),
            "source": state.get("source", "api"),
            "decision": decision,
            "confidence": state.get("confidence", 0.0),
            "reason": state.get("reason", ""),
            "tier": text_result.get("tier", "cache") if text_result else "cache",
            "source_label": "ai",
            "text_result": text_result,
            "image_result": image_result,
            "traces": state.get("traces", []),
        })
    except Exception as e:
        logger.warning("Event recording failed: %s", e)

    logger.info("ACTION | id=%s | decision=%s | confidence=%.3f | tier=%s | reason=%s",
                state.get("content_id", "?"), decision,
                state.get("confidence", 0.0),
                text_result.get("tier", "cache") if text_result else "cache",
                state.get("reason", ""))

    return {"action_taken": decision, "traces": traces}


def _trace(node: str, step: str, model: str, input_summary: str,
           output: dict, latency_ms: float, cost: str) -> dict:
    return {
        "node": node, "step": step, "model": model,
        "input": input_summary[:200], "output": output,
        "latency_ms": round(latency_ms, 2), "cost": cost,
        "ts": int(time.time() * 1000),
    }
