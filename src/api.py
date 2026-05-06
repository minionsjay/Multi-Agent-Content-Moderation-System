import time
import json
import asyncio
import logging
from io import StringIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.gateway import gateway
from src.graph import graph
from src.state import ModerationState
from src.skills.review_queue import review_queue
from src.skills.memory_cache import memory_cache
from src.skills.redis_cache import redis_cache
from src.skills.vector_cache import vector_cache
from src.skills.embedder import embedder

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="Content Moderation POC", version="0.5.0")


@app.on_event("startup")
async def startup_warmup():
    """Pre-load models so first request is fast."""
    import logging
    log = logging.getLogger("startup")
    t0 = time.perf_counter()

    # Warm BERT (transformers pipeline)
    try:
        from src.skills.bert_classify import bert_classifier
        bert_classifier.warmup()
    except Exception as e:
        log.warning("BERT transformers warmup failed: %s", e)

    # Warm ONNX session
    try:
        from src.skills.bert_onnx import bert_onnx
        if bert_onnx._enabled:
            bert_onnx.classify("warmup")
            log.info("ONNX BERT warmed up")
        else:
            log.info("ONNX not available, skipped warmup")
    except Exception as e:
        log.warning("ONNX warmup failed: %s", e)

    log.info("Startup complete (%.1fs)", time.perf_counter() - t0)


class ModerationRequest(BaseModel):
    content_id: str = Field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    text: str = ""
    image_url: str = ""
    image_base64: str = ""
    # Model overrides
    bert_model: str = ""     # empty = use config default
    llm_provider: str = ""   # deepseek / openai / anthropic
    llm_model: str = ""      # deepseek-chat / gpt-4o-mini / claude-3-5-haiku-latest
    user_id: str = "anonymous"
    source: str = "api"


# ---- Static ----
@app.get("/")
async def index(): return FileResponse("static/index.html")

@app.get("/health")
async def health(): return {"status": "ok", "version": "0.5.0", "architecture": "gateway + langgraph"}


# ---- Gateway stats ----
@app.get("/gateway/stats")
async def gateway_stats(): return gateway.get_stats()


# ---- Human Review ----
@app.get("/review/pending")
async def review_pending(limit: int = 50):
    """List pending human review items (highest priority first)."""
    items = review_queue.get_pending(limit)
    return {"total": len(items), "items": items}


@app.get("/review/stats")
async def review_stats():
    """Human review queue statistics."""
    return review_queue.get_stats()


@app.post("/review/resolve")
async def review_resolve(req: dict):
    """Resolve a human review item.

    Body: {
      "review_id": "...",
      "human_decision": "pass" | "block",
      "reviewer": "reviewer_name",
      "reason": "why"
    }

    On resolve, the human decision is written back to all cache layers
    so the same content won't need review again.
    """
    review_id = req.get("review_id", "")
    human_decision = req.get("human_decision", "pass")
    reviewer = req.get("reviewer", "anonymous")
    reason = req.get("reason", "")

    if human_decision not in ("pass", "block"):
        return {"error": "human_decision must be 'pass' or 'block'"}

    # Resolve the review entry
    record = review_queue.resolve(review_id, human_decision, reviewer, reason)
    if record is None:
        return {"error": f"Review {review_id} not found"}

    # Feed back into caches so this content won't be re-reviewed
    text = record.get("text", "")
    confidence = 1.0  # human decision is authoritative
    if text and text.strip():
        # L0a: local cache
        memory_cache.set(text, human_decision, confidence, reason)
        # L0b: Redis
        redis_cache.set(text, human_decision, confidence, reason)
        # L1: ChromaDB (async — fire and forget in POC)
        try:
            embedding = embedder.embed(text)
            vector_cache.store(embedding, text, human_decision, confidence, reason)
        except Exception:
            pass

    return {
        "status": "resolved",
        "review_id": review_id,
        "human_decision": human_decision,
        "cached": bool(text and text.strip()),
    }


# ---- Core endpoint (gateway → graph split) ----
@app.post("/moderate")
async def moderate(req: ModerationRequest) -> dict:
    t0 = time.perf_counter()
    text = _resolve_text(req)

    # === Gateway pre-filter (always returns a dict now) ===
    gw = gateway.check(text, req.image_url, req.image_base64)

    if gw["decision"] is not None:
        # === HOT PATH: Gateway resolved → return immediately ===
        resp = dict(gw["decision"])
        resp["content_id"] = req.content_id
        resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        resp["traces"] = gw["traces"]
        resp["path"] = "hot"
        return resp

    # === COLD PATH: Gateway escalated → run LangGraph ===
    state = _make_state(req, text, gw)
    result = await graph.ainvoke(state)
    total_ms = (time.perf_counter() - t0) * 1000

    text_result = result.get("text_result") or {}
    tier = text_result.get("tier", "L3_llm") if text_result else "L3_llm"

    # Merge gateway traces + LangGraph traces
    all_traces = gw["traces"] + result.get("traces", [])

    return {
        "content_id": req.content_id,
        "decision": result.get("decision", "pass"),
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason", ""),
        "tier": tier,
        "latency_ms": round(total_ms, 2),
        "traces": all_traces,
        "path": "cold",
    }


# ---- Streaming (with gateway) ----
@app.post("/moderate/stream")
async def moderate_stream(req: ModerationRequest):
    text = _resolve_text(req)
    t0 = time.perf_counter()

    async def event_stream():
        # Step 1: Gateway check
        gw_t0 = time.perf_counter()
        gw = gateway.check(text, req.image_url, req.image_base64)
        gw_ms = (time.perf_counter() - gw_t0) * 1000

        if gw["decision"] is not None:
            # Hot path hit — send traces and done
            yield f"data: {json.dumps({'event': 'node_complete', 'node': 'gateway', 'node_index': 1, 'traces': gw['traces'], 'partial': {'decision': gw['decision']['decision'], 'tier': gw['decision'].get('tier', '?')}}, ensure_ascii=False)}\n\n"
            resp = dict(gw["decision"])
            resp["content_id"] = req.content_id
            resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            resp["traces"] = gw["traces"]
            resp["path"] = "hot"
            yield f"data: {json.dumps({'event': 'done', 'result': resp}, ensure_ascii=False)}\n\n"
            return

        # Step 2: Cold path — stream LangGraph nodes
        # Send gateway traces first (even on miss)
        yield f"data: {json.dumps({'event': 'node_complete', 'node': 'gateway', 'node_index': 1, 'traces': gw['traces'], 'partial': {'escalated': True}}, ensure_ascii=False)}\n\n"

        state = _make_state(req, text, gw)
        node_idx = 2
        last_fragment = {}
        async for chunk in graph.astream(state, stream_mode="updates"):
            for node_name, fragment in chunk.items():
                last_fragment.update({k: v for k, v in fragment.items() if k != "traces"})
                yield f"data: {json.dumps({'event': 'node_complete', 'node': node_name, 'node_index': node_idx, 'traces': fragment.get('traces', []), 'partial': {}}, ensure_ascii=False)}\n\n"
                node_idx += 1

        t1 = time.perf_counter()
        final = _format_response(req.content_id, last_fragment, t1 - t0, "cold")
        final["gateway_latency_ms"] = round(gw_ms, 2)
        # Merge gateway traces into final
        final["traces"] = gw["traces"] + final["traces"]
        yield f"data: {json.dumps({'event': 'done', 'result': final}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---- Batch (with gateway) ----
@app.post("/moderate/batch")
async def moderate_batch(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    content = await file.read()
    texts = _parse_upload(content, file.filename or "")

    if not texts:
        return {"error": "No valid texts found in file", "total": 0}

    sem = asyncio.Semaphore(1)  # limit LLM concurrency to avoid rate limits

    async def process_one(item):
        async with sem:
            t_item = time.perf_counter()
            # Gateway first
            gw = gateway.check(item["text"], item.get("image_url", ""))
            if gw["decision"] is not None:
                resp = dict(gw["decision"])
                resp["content_id"] = item["id"]
                resp["path"] = "hot"
                resp["traces"] = gw["traces"]
                resp["latency_ms"] = round((time.perf_counter() - t_item) * 1000, 2)
                return resp
            # Cold path
            state = _make_state(
                ModerationRequest(content_id=item["id"], text=item["text"]),
                item["text"], gw)
            result = await graph.ainvoke(state)
            resp = _format_response(item["id"], result, time.perf_counter() - t_item, "cold")
            resp["traces"] = gw["traces"] + resp["traces"]
            return resp

    results = await asyncio.gather(*[process_one(t) for t in texts])
    total_ms = (time.perf_counter() - t0) * 1000

    passed = sum(1 for r in results if r["decision"] == "pass")
    blocked = sum(1 for r in results if r["decision"] == "block")
    reviewed = sum(1 for r in results if r["decision"] == "review")
    hot = sum(1 for r in results if r.get("path") == "hot")
    cold = sum(1 for r in results if r.get("path") == "cold")
    tiers = {}
    for r in results:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

    return {
        "total": len(results),
        "passed": passed, "blocked": blocked, "reviewed": reviewed,
        "hot_path": hot, "cold_path": cold,
        "hot_path_rate": round(hot / len(results), 4) if results else 0,
        "total_latency_ms": round(total_ms, 2),
        "avg_latency_ms": round(total_ms / len(results), 2) if results else 0,
        "tier_distribution": tiers,
        "llm_call_rate": round(tiers.get("L3_llm", 0) / len(results), 4) if results else 0,
        "gateway_stats": gateway.get_stats(),
        "results": results,
    }


# ---- Helpers ----

def _resolve_text(req: ModerationRequest) -> str:
    text = req.text.strip()
    if req.image_url and not text:
        text = f"[Image URL: {req.image_url}]"
    return text


def _make_state(req: ModerationRequest, text: str, gw: dict | None = None) -> ModerationState:
    """Build LangGraph initial state. If gw (gateway result) is provided,
    carry forward keyword prefilter status so Text Agent can skip L1."""
    state: ModerationState = {
        "content_id": req.content_id,
        "text": text,
        "image_url": req.image_url,
        "image_base64": req.image_base64,
        "bert_model": req.bert_model,
        "llm_provider": req.llm_provider,
        "llm_model": req.llm_model,
        "user_id": req.user_id,
        "source": req.source,
        "content_type": "text_only",
        "cache_hit": False,
        "cached_decision": None,
        "keyword_confidence": 0.0,
        "keyword_label": None,
        "keyword_prefiltered": False,
        "priority_score": 0.3,
        "text_result": None,
        "decision": "pass",
        "confidence": 0.0,
        "reason": "",
        "traces": [],
    }
    if gw is not None:
        state["keyword_prefiltered"] = gw.get("keyword_prefiltered", False)
        state["keyword_label"] = gw.get("keyword_label")
        state["keyword_confidence"] = gw.get("keyword_confidence", 0.0)
    return state


def _format_response(content_id: str, result: dict, latency_s: float, path: str = "cold") -> dict:
    text_result = result.get("text_result") or {}
    tier = text_result.get("tier", "L3_llm") if text_result else "L3_llm"
    return {
        "content_id": content_id,
        "decision": result.get("decision", "pass"),
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason", ""),
        "tier": tier,
        "latency_ms": round(latency_s * 1000, 2),
        "traces": result.get("traces", []),
        "path": path,
    }


def _parse_upload(content: bytes, filename: str) -> list[dict]:
    texts = []
    if filename.endswith(".jsonl"):
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                texts.append({"id": obj.get("id", f"b{len(texts)}"), "text": obj.get("text", "")})
            except json.JSONDecodeError:
                continue
    else:
        import csv
        reader = csv.reader(StringIO(content.decode("utf-8")))
        for i, row in enumerate(reader):
            if i == 0 and row and row[0].lower() in ("text", "content", "内容"):
                continue
            if not row or not row[0].strip(): continue
            texts.append({"id": row[1].strip() if len(row) > 1 else f"b{i}", "text": row[0].strip()})
    return texts


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
