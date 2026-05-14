import time
import asyncio
import logging
from src.state import ModerationState
from src.skills.keyword_filter import keyword_filter
from src.skills.bert_classify import bert_classifier
from src.skills.llm_audit import llm_auditor
from src.config import BERT_ENABLED, BERT_MODEL, LLM_PROVIDER

logger = logging.getLogger(__name__)



async def text_specialist(state: ModerationState) -> dict:
    t0 = time.perf_counter()
    text = state.get("text", "") or ""
    traces = []

    if not text.strip():
        traces.append(_trace("text_agent", "empty", "none", text,
                            {"label": "safe", "confidence": 1.0},
                            (time.perf_counter() - t0) * 1000, "zero"))
        return {"text_result": {"label": "safe", "confidence": 1.0, "tier": "L0_empty"}, "traces": traces}

    # L1: Keyword filter (skip if Gateway already scanned)
    if state.get("keyword_prefiltered"):
        traces.append(_trace("text_agent", "L1_keyword", "skipped · Gateway prefiltered", text,
                            {"reason": "Gateway already scanned keywords → skipping duplicate scan",
                             "gateway_label": state.get("keyword_label"),
                             "gateway_confidence": state.get("keyword_confidence", 0.0)},
                            0, "zero"))
        # Carry forward Gateway's ambiguous keyword result if any
        kw_label = state.get("keyword_label")
        kw_confidence = state.get("keyword_confidence", 0.0)
        if kw_label and kw_confidence >= 0.4:
            kw_result = {"label": kw_label, "confidence": kw_confidence,
                         "matches": [], "source": "gateway"}
        else:
            kw_result = {"label": None, "confidence": 0.0, "matches": []}
    else:
        t1 = time.perf_counter()
        kw_result = keyword_filter.match(text)
        kw_ms = (time.perf_counter() - t1) * 1000

        if kw_result["confidence"] > 0.99:
            traces.append(_trace("text_agent", "L1_keyword", "AC自动机 · 关键词匹配", text,
                                {"label": kw_result["label"], "confidence": kw_result["confidence"],
                                 "matched_keywords": kw_result.get("matches", [])},
                                kw_ms, "zero"))
            return {
                "text_result": {
                    "label": kw_result["label"], "confidence": kw_result["confidence"],
                    "matches": kw_result.get("matches", []), "tier": "L1_keyword", "cost": "zero",
                },
                "traces": traces,
            }
        traces.append(_trace("text_agent", "L1_keyword", "ac_automaton", text,
                            {"label": None, "confidence": 0.0}, kw_ms, "zero"))

    # L2: BERT classification (try ONNX first, fallback to transformers)
    from src.config import BERT_ENABLED

    bert_enabled = BERT_ENABLED and state.get("bert_model", "x") != "skip"
    if state.get("bert_enabled") is False:
        bert_enabled = False  # explicit disable from frontend

    if bert_enabled:
        t2 = time.perf_counter()
        bert_model_name = state.get("bert_model") or BERT_MODEL

        # Try ONNX first for speed, fallback to HF if confidence too low
        bert_backend = "unknown"
        try:
            from src.skills.bert_onnx import bert_onnx
            bert_result = bert_onnx.classify(text)
            bert_backend = bert_result.get("backend", "onnx")

            # ONNX confidence too low to skip LLM → try HF as second opinion
            if not bert_classifier.should_skip_llm(bert_result):
                hf_check = bert_classifier.classify(text, model_override=bert_model_name)
                if bert_classifier.should_skip_llm(hf_check):
                    bert_result = hf_check
                    bert_backend = "onnx→HF(high conf)"
                # else: keep ONNX result, go to L3
        except Exception:
            bert_result = bert_classifier.classify(text, model_override=bert_model_name)
            bert_backend = "transformers"

        bert_ms = (time.perf_counter() - t2) * 1000
        traces.append(_trace("text_agent", "L2_bert",
            f"BERT · {bert_model_name.rsplit('/', 1)[-1]} ({bert_backend})", text,
            {"label": bert_result["label"], "confidence": bert_result["confidence"],
             "all_scores": bert_result.get("raw", []),
             "threshold_high": 0.95, "threshold_low": 0.4,
             "error": bert_result.get("error")},
            bert_ms, "low"))

        if bert_classifier.should_skip_llm(bert_result):
            traces.append(_trace("text_agent", "L2_decision", "决策规则 · 置信度阈值", "",
                                {"decision": bert_result["label"], "tier": "L2_bert",
                                 "reason": f"BERT conf={bert_result['confidence']:.4f} >= threshold(0.95) → 跳过LLM"},
                                (time.perf_counter() - t0) * 1000, "low"))
            return {
                "text_result": {
                    "label": bert_result["label"], "confidence": bert_result["confidence"],
                    "raw": bert_result.get("raw", []), "tier": "L2_bert", "cost": "low",
                },
                "traces": traces,
            }
    else:
        traces.append(_trace("text_agent", "L2_bert", "skipped", text,
                            {"reason": "BERT_ENABLED=false"}, 0, "zero"))
        bert_result = {"label": "unsafe", "confidence": 0.5}

    # L3: LLM deep audit (API or local, with timeout → fallback to BERT result)
    t3 = time.perf_counter()
    llm_provider = state.get("llm_provider") or LLM_PROVIDER
    llm_model = state.get("llm_model") or ""
    context = {"bert_label": bert_result["label"], "bert_confidence": bert_result["confidence"],
               "user_id": state.get("user_id")}

    # Select LLM backend: API (DeepSeek/OpenAI/Anthropic), local (llama.cpp), or transformers
    if llm_provider == "local":
        from src.skills.llm_local import local_llm
        try:
            llm_result = await asyncio.wait_for(
                local_llm.audit(text, context),
                timeout=30.0)  # local LLM can be slower on CPU
        except asyncio.TimeoutError:
            logger.warning("Local LLM timed out — falling back to BERT result")
            llm_result = {"label": bert_result["label"], "confidence": bert_result["confidence"],
                          "reason": "Local LLM timeout, using BERT result", "model": "fallback_bert",
                          "cost": "low"}
        llm_cost = "low"  # local inference = low cost
        model_label = f"Local · {llm_result.get('model', 'qwen')}"
    elif llm_provider == "transformers":
        from src.skills.llm_transformers import llm_transformers
        try:
            llm_result = await asyncio.wait_for(
                llm_transformers.audit(text, context),
                timeout=60.0)  # first call loads model, subsequent calls faster
        except asyncio.TimeoutError:
            logger.warning("Transformers LLM timed out — falling back to BERT result")
            llm_result = {"label": bert_result["label"], "confidence": bert_result["confidence"],
                          "reason": "Transformers LLM timeout, using BERT result", "model": "fallback_bert",
                          "cost": "low"}
        llm_cost = "low"
        model_label = f"Local · {llm_result.get('model', 'transformers')}"
    elif llm_provider == "sglang":
        from src.skills.llm_sglang import llm_sglang
        try:
            llm_result = await asyncio.wait_for(
                llm_sglang.audit(text, context),
                timeout=120.0)  # first call loads engine, subsequent calls <1s
        except asyncio.TimeoutError:
            logger.warning("SGLang timed out — falling back to BERT result")
            llm_result = {"label": bert_result["label"], "confidence": bert_result["confidence"],
                          "reason": "SGLang timeout, using BERT result", "model": "fallback_bert",
                          "cost": "low"}
        llm_cost = "low"
        model_label = f"Local · {llm_result.get('model', 'sglang')}"
    elif llm_provider == "qwen_guard":
        from src.skills.llm_qwen_guard import llm_qwen_guard
        try:
            llm_result = await asyncio.wait_for(
                llm_qwen_guard.audit(text, context, model=llm_model),
                timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Qwen3Guard timed out — falling back to BERT result")
            llm_result = {"label": bert_result["label"], "confidence": bert_result["confidence"],
                          "reason": "Qwen3Guard timeout, using BERT result", "model": "fallback_bert",
                          "cost": "low"}
        llm_cost = "low"
        model_label = f"Qwen3Guard · {llm_result.get('model', 'qwen_guard')}"
    else:
        try:
            llm_result = await asyncio.wait_for(
                llm_auditor.audit(text, context, provider=llm_provider, model=llm_model),
                timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning("LLM audit timed out — falling back to BERT result")
            llm_result = {"label": bert_result["label"], "confidence": bert_result["confidence"],
                          "reason": "LLM timeout, using BERT result", "model": "fallback_bert",
                          "cost": "high"}
        llm_cost = "high"  # API call = high cost
        model_label = f"{llm_provider} · {llm_result.get('model', 'llm')}"

    llm_ms = (time.perf_counter() - t3) * 1000

    traces.append(_trace("text_agent", "L3_llm", model_label, text,
                        {"label": llm_result["label"], "confidence": llm_result["confidence"],
                         "analysis": llm_result.get("reason", ""),
                         "bert_preliminary": bert_result["label"],
                         "bert_confidence": bert_result["confidence"],
                         "error": llm_result.get("error")},
                        llm_ms, llm_cost))

    return {
        "text_result": {
            "label": llm_result["label"], "confidence": llm_result["confidence"],
            "reason": llm_result.get("reason", ""),
            "bert_preliminary": bert_result["label"],
            "tier": "L3_llm", "cost": llm_cost,
            "model": llm_result.get("model", ""),
        },
        "traces": traces,
    }


def _trace(node: str, step: str, model: str, input_summary: str,
           output: dict, latency_ms: float, cost: str) -> dict:
    return {
        "node": node,
        "step": step,
        "model": model,
        "input": input_summary[:200],
        "output": output,
        "latency_ms": round(latency_ms, 2),
        "cost": cost,
        "ts": int(time.time() * 1000),
    }
