import time
import json
import asyncio
import logging
import os
import uuid
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

TRACES_PATH = "data/traces.jsonl"
REVIEW_RESULTS_PATH = "data/review_results.jsonl"


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
    # Prototype contract aliases. Keep content_id/text for POC compatibility.
    case_id: str = ""
    content: str = ""
    image_url: str = ""
    image_base64: str = ""
    country: str = ""
    language: str = ""
    domain: str = ""
    customer_type: str = ""
    risk_type: str = ""
    algorithm_score: float = 0.0
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


# ---- Prototype Trace Contract ----
@app.get("/traces")
async def traces(limit: int = 100):
    items = _list_traces(limit)
    return {"total": len(items), "items": items}


@app.get("/traces/{trace_id}")
async def trace_detail(trace_id: str):
    trace = _get_trace(trace_id)
    if trace is None:
        return {"error": f"Trace {trace_id} not found"}
    return trace


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
    human_risk_type = req.get("human_risk_type", "")
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

    review_result = _make_review_result(record, human_decision, human_risk_type, reviewer, reason)
    _append_jsonl(REVIEW_RESULTS_PATH, review_result)

    return {
        "status": "resolved",
        "review_id": review_id,
        "human_decision": human_decision,
        "review_result": review_result,
        "cached": bool(text and text.strip()),
    }


# ---- Core endpoint (gateway → graph split) ----
@app.post("/moderate")
async def moderate(req: ModerationRequest) -> dict:
    t0 = time.perf_counter()
    text = _resolve_text(req)
    case_id = _case_id(req)
    trace_id = _new_trace_id()

    # === Gateway pre-filter (always returns a dict now) ===
    gw = gateway.check(text, req.image_url, req.image_base64)

    if gw["decision"] is not None:
        # === HOT PATH: Gateway resolved → return immediately ===
        resp = dict(gw["decision"])
        resp["content_id"] = req.content_id
        resp["case_id"] = case_id
        resp["trace_id"] = trace_id
        resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        resp["traces"] = gw["traces"]
        resp["path"] = "hot"
        _record_trace(req, text, trace_id, resp, gw["traces"], resp["latency_ms"], "hot")
        return resp

    # === COLD PATH: Gateway escalated → run LangGraph ===
    state = _make_state(req, text, gw, trace_id)
    result = await graph.ainvoke(state)
    total_ms = (time.perf_counter() - t0) * 1000

    text_result = result.get("text_result") or {}
    tier = text_result.get("tier", "L3_llm") if text_result else "L3_llm"

    # Merge gateway traces + LangGraph traces
    all_traces = gw["traces"] + result.get("traces", [])

    resp = {
        "content_id": req.content_id,
        "case_id": case_id,
        "trace_id": trace_id,
        "decision": result.get("decision", "pass"),
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason", ""),
        "tier": tier,
        "latency_ms": round(total_ms, 2),
        "traces": all_traces,
        "path": "cold",
    }
    _record_trace(req, text, trace_id, resp, all_traces, total_ms, "cold", result)
    return resp


# ---- Streaming (with gateway) ----
@app.post("/moderate/stream")
async def moderate_stream(req: ModerationRequest):
    text = _resolve_text(req)
    case_id = _case_id(req)
    trace_id = _new_trace_id()
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
            resp["case_id"] = case_id
            resp["trace_id"] = trace_id
            resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            resp["traces"] = gw["traces"]
            resp["path"] = "hot"
            _record_trace(req, text, trace_id, resp, gw["traces"], resp["latency_ms"], "hot")
            yield f"data: {json.dumps({'event': 'done', 'result': resp}, ensure_ascii=False)}\n\n"
            return

        # Step 2: Cold path — stream LangGraph nodes
        # Send gateway traces first (even on miss)
        yield f"data: {json.dumps({'event': 'node_complete', 'node': 'gateway', 'node_index': 1, 'traces': gw['traces'], 'partial': {'escalated': True}}, ensure_ascii=False)}\n\n"

        state = _make_state(req, text, gw, trace_id)
        node_idx = 2
        last_fragment = {}
        async for chunk in graph.astream(state, stream_mode="updates"):
            for node_name, fragment in chunk.items():
                last_fragment.update({k: v for k, v in fragment.items() if k != "traces"})
                yield f"data: {json.dumps({'event': 'node_complete', 'node': node_name, 'node_index': node_idx, 'traces': fragment.get('traces', []), 'partial': {}}, ensure_ascii=False)}\n\n"
                node_idx += 1

        t1 = time.perf_counter()
        final = _format_response(req, last_fragment, t1 - t0, "cold", trace_id)
        final["gateway_latency_ms"] = round(gw_ms, 2)
        # Merge gateway traces into final
        final["traces"] = gw["traces"] + final["traces"]
        _record_trace(req, text, trace_id, final, final["traces"], (t1 - t0) * 1000, "cold", last_fragment)
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
            req = ModerationRequest(
                content_id=item["id"],
                case_id=item.get("case_id", item["id"]),
                text=item["text"],
                content=item.get("content", ""),
                image_url=item.get("image_url", ""),
                image_base64=item.get("image_base64", ""),
            )
            trace_id = _new_trace_id()
            text = _resolve_text(req)
            # Gateway first
            gw = gateway.check(text, req.image_url, req.image_base64)
            if gw["decision"] is not None:
                resp = dict(gw["decision"])
                resp["content_id"] = req.content_id
                resp["case_id"] = _case_id(req)
                resp["trace_id"] = trace_id
                resp["path"] = "hot"
                resp["traces"] = gw["traces"]
                resp["latency_ms"] = round((time.perf_counter() - t_item) * 1000, 2)
                _record_trace(req, text, trace_id, resp, gw["traces"], resp["latency_ms"], "hot")
                return resp
            # Cold path
            state = _make_state(req, text, gw, trace_id)
            result = await graph.ainvoke(state)
            elapsed_s = time.perf_counter() - t_item
            resp = _format_response(req, result, elapsed_s, "cold", trace_id)
            resp["traces"] = gw["traces"] + resp["traces"]
            _record_trace(req, text, trace_id, resp, resp["traces"], elapsed_s * 1000, "cold", result)
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
    text = (req.content or req.text or "").strip()
    if req.image_url and not text:
        text = f"[Image URL: {req.image_url}]"
    return text


def _make_state(
    req: ModerationRequest,
    text: str,
    gw: dict | None = None,
    trace_id: str = "",
) -> ModerationState:
    """Build LangGraph initial state. If gw (gateway result) is provided,
    carry forward keyword prefilter status so Text Agent can skip L1."""
    case_id = _case_id(req)
    state: ModerationState = {
        "case_id": case_id,
        "trace_id": trace_id,
        "content_id": req.content_id,
        "text": text,
        "image_url": req.image_url,
        "image_base64": req.image_base64,
        "country": req.country,
        "language": req.language,
        "domain": req.domain,
        "customer_type": req.customer_type,
        "risk_type": req.risk_type or "other",
        "algorithm_score": req.algorithm_score,
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


def _format_response(
    req: ModerationRequest,
    result: dict,
    latency_s: float,
    path: str = "cold",
    trace_id: str = "",
) -> dict:
    text_result = result.get("text_result") or {}
    tier = text_result.get("tier", "L3_llm") if text_result else "L3_llm"
    return {
        "content_id": req.content_id,
        "case_id": _case_id(req),
        "trace_id": trace_id,
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
                case_id = obj.get("case_id") or obj.get("content_id") or obj.get("id") or f"b{len(texts)}"
                texts.append({
                    "id": case_id,
                    "case_id": case_id,
                    "text": obj.get("content", obj.get("text", "")),
                    "image_url": obj.get("image_url", ""),
                    "image_base64": obj.get("image_base64", ""),
                })
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


def _case_id(req: ModerationRequest) -> str:
    return req.case_id or req.content_id


def _new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:16]}"


def _request_dict(req: ModerationRequest) -> dict:
    if hasattr(req, "model_dump"):
        data = req.model_dump()
    else:
        data = req.dict()
    if data.get("image_base64"):
        data["image_base64"] = f"<base64:{len(data['image_base64'])} chars>"
    data["case_id"] = _case_id(req)
    data["content"] = _resolve_text(req)
    return data


def _record_trace(
    req: ModerationRequest,
    text: str,
    trace_id: str,
    detection_result: dict,
    traces: list[dict],
    latency_ms: float,
    path: str,
    node_result: dict | None = None,
):
    trace_record = {
        "case_id": _case_id(req),
        "trace_id": trace_id,
        "input": _request_dict(req),
        "gateway_node": _node_summary(traces, "gateway"),
        "image_text_node": _node_summary(traces, "image_agent"),
        "language_node": {"language": req.language or "unknown"},
        "rule_node": _rule_summary(traces),
        "risk_preclassify_node": {"risk_type": req.risk_type or "other"},
        "algorithm_score_node": {"algorithm_score": req.algorithm_score},
        "llm_judge_node": _llm_summary(traces, node_result or {}),
        "decision_node": {
            "decision": detection_result.get("decision", "pass"),
            "confidence": detection_result.get("confidence", 0.0),
            "reason": detection_result.get("reason", ""),
            "path": path,
        },
        "detection_result": {
            key: value for key, value in detection_result.items()
            if key != "traces"
        },
        "runtime": {
            "latency_ms": round(latency_ms, 2),
            "llm_called": _llm_called(traces),
            "cost_estimate": _cost_estimate(traces),
        },
        "version": {
            "rule_version": "poc_v0.5.0",
            "prompt_version": "poc_current",
        },
        "raw_traces": traces,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    _append_jsonl(TRACES_PATH, trace_record)


def _node_summary(traces: list[dict], node: str) -> dict:
    node_traces = [trace for trace in traces if trace.get("node") == node]
    return {
        "steps": node_traces,
        "error": next(
            (trace.get("output", {}).get("error") for trace in node_traces
             if trace.get("output", {}).get("error")),
            None,
        ),
    }


def _rule_summary(traces: list[dict]) -> dict:
    rule_traces = [
        trace for trace in traces
        if "keyword" in trace.get("step", "") or "whitelist" in trace.get("step", "")
    ]
    matched = []
    for trace in rule_traces:
        output = trace.get("output", {})
        matched.extend(output.get("matched_keywords") or output.get("context") or [])
    return {"steps": rule_traces, "matched": matched}


def _llm_summary(traces: list[dict], node_result: dict) -> dict:
    llm_steps = [trace for trace in traces if "llm" in trace.get("step", "").lower()]
    text_result = node_result.get("text_result") or {}
    return {"steps": llm_steps, "result": text_result}


def _llm_called(traces: list[dict]) -> bool:
    return any("llm" in trace.get("step", "").lower() for trace in traces)


def _cost_estimate(traces: list[dict]) -> float:
    high_cost_calls = sum(1 for trace in traces if trace.get("cost") == "high")
    low_cost_calls = sum(1 for trace in traces if trace.get("cost") == "low")
    return round(high_cost_calls * 0.002 + low_cost_calls * 0.0001, 6)


def _make_review_result(
    record: dict,
    human_decision: str,
    human_risk_type: str,
    reviewer: str,
    reason: str,
) -> dict:
    system_decision = record.get("ai_decision") or record.get("system_decision", "review")
    system_risk_type = record.get("risk_type") or record.get("system_risk_type", "other")
    human_risk_type = human_risk_type or system_risk_type
    error_type = _review_error_type(system_decision, human_decision, system_risk_type, human_risk_type)
    return {
        "case_id": record.get("case_id") or record.get("content_id") or "",
        "trace_id": record.get("trace_id", ""),
        "system_decision": system_decision,
        "system_risk_type": system_risk_type,
        "human_decision": human_decision,
        "human_risk_type": human_risk_type,
        "is_correct": error_type in ("true_positive", "true_negative"),
        "error_type": error_type,
        "human_reason": reason,
        "reviewer": reviewer,
        "review_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def _review_error_type(
    system_decision: str,
    human_decision: str,
    system_risk_type: str,
    human_risk_type: str,
) -> str:
    if human_decision == "block" and system_risk_type != human_risk_type:
        return "category_error"
    if system_decision == "pass" and human_decision == "block":
        return "false_negative"
    if system_decision in ("block", "review") and human_decision == "pass":
        return "false_positive"
    if system_decision in ("block", "review") and human_decision == "block":
        return "true_positive"
    if system_decision == "pass" and human_decision == "pass":
        return "true_negative"
    return "unknown"


def _append_jsonl(path: str, item: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _list_traces(limit: int = 100) -> list[dict]:
    return list(reversed(_read_jsonl(TRACES_PATH)))[:limit]


def _get_trace(trace_id: str) -> dict | None:
    for trace in reversed(_read_jsonl(TRACES_PATH)):
        if trace.get("trace_id") == trace_id:
            return trace
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
