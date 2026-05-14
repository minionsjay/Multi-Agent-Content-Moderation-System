import time
import logging
from src.state import ModerationState
from src.skills.decision_config import get_config

logger = logging.getLogger(__name__)


async def decision_aggregator(state: ModerationState) -> dict:
    t0 = time.perf_counter()
    traces = []
    cfg = get_config()  # live config (adjustable via API/frontend)

    # Path 1: Cache hit
    if state.get("cache_hit") and state.get("cached_decision"):
        cached = state["cached_decision"]
        traces.append(_trace("decision", "cache_reuse", "none",
                            f"Reusing cached decision: {cached['decision']}",
                            cached, (time.perf_counter() - t0) * 1000, "zero"))
        return {
            "decision": cached["decision"],
            "confidence": cached["confidence"],
            "reason": cached.get("reason", "Cache hit"),
            "traces": traces,
        }

    # Path 2: Gateway/Triage pre-set decision (keyword hit, cache hit carried forward)
    if state.get("keyword_confidence", 0) > 0.99:
        label = state.get("keyword_label", "unsafe")
        zero_tol = cfg.get("zero_tolerance", ["politics", "violence"])
        if label in zero_tol:
            traces.append(_trace("decision", "zero_tolerance", "none",
                                f"Bypass: category={label}", {"decision": "block"},
                                (time.perf_counter() - t0) * 1000, "zero"))
            return {"decision": "block", "confidence": 1.0,
                    "reason": state.get("reason", "Zero-tolerance policy"),
                    "traces": traces}
        else:
            traces.append(_trace("decision", "keyword_high_conf", "none",
                                f"Keyword match: {label} (conf=1.0)", {"decision": "block"},
                                (time.perf_counter() - t0) * 1000, "zero"))
            return {"decision": "block", "confidence": 1.0,
                    "reason": f"High-confidence keyword match: {label}",
                    "traces": traces}

    # Path 3: Weighted aggregation of text_result + image_result
    text_result = state.get("text_result")
    image_result = state.get("image_result") or {}
    multimodal_result = state.get("multimodal_result") or {}

    # If only image result exists (no text)
    if not text_result and image_result:
        ir_label = image_result.get("label", "normal")
        ir_conf = image_result.get("confidence", 0.5)
        if ir_label == "nsfw" and ir_conf > 0.5:
            return {"decision": "block", "confidence": ir_conf,
                    "reason": f"Image NSFW detected (conf={ir_conf:.2f})",
                    "traces": traces}
        return {"decision": "pass", "confidence": ir_conf,
                "reason": "Image appears normal",
                "traces": traces}

    if not text_result:
        traces.append(_trace("decision", "fallback", "none", "No text_result",
                            {"decision": "pass"}, (time.perf_counter() - t0) * 1000, "zero"))
        return {"decision": "pass", "confidence": 0.5,
                "reason": "No text moderation result", "traces": traces}

    # --- Weighted score aggregation ---
    label, confidence, tier = _aggregate_signals(text_result, image_result, multimodal_result, cfg)

    traces.append(_trace("decision", "aggregate", "weighted_scoring",
                        f"Weighted: label={label}, confidence={confidence:.4f}, tier={tier}",
                        {"label": label, "confidence": confidence, "tier": tier,
                         "text_label": text_result.get("label"), "text_conf": text_result.get("confidence"),
                         "image_label": image_result.get("label"), "image_conf": image_result.get("confidence"),
                         "agent_weights": cfg.get("agent_weights", {}),
                         "label_severity": cfg.get("label_severity", {}),
                         "score_thresholds": cfg.get("score_thresholds", {}),
                         "grey_zone": cfg.get("grey_zone", {})},
                        (time.perf_counter() - t0) * 1000, "zero"))

    # Zero-tolerance override
    zero_tol = cfg.get("zero_tolerance", ["politics", "violence"])
    if label in zero_tol:
        return {"decision": "block", "confidence": confidence,
                "reason": text_result.get("reason", f"Zero-tolerance: {label}"),
                "traces": traces}

    # Safe → always pass (grey zone only applies to unsafe/unclear content)
    if label == "safe":
        traces.append(_trace("decision", "final", "none",
                            f"Decision: pass (safe, confidence={confidence:.4f})",
                            {"decision": "pass", "confidence": confidence},
                            (time.perf_counter() - t0) * 1000, "zero"))
        return {"decision": "pass", "confidence": confidence,
                "reason": text_result.get("reason", "Content classified as safe"),
                "traces": traces}

    gz = cfg.get("grey_zone", {"low": 0.3, "high": 0.7})
    gz_low = gz["low"]
    gz_high = gz["high"]

    # Grey zone: uncertain unsafe → human review
    if gz_low <= confidence <= gz_high:
        traces.append(_trace("decision", "grey_zone", "none",
                            f"Confidence {confidence:.2f} in [{gz_low}, {gz_high}] → review",
                            {"decision": "review"},
                            (time.perf_counter() - t0) * 1000, "zero"))
        return {"decision": "review", "confidence": confidence,
                "reason": text_result.get("reason", "Grey zone — needs human review"),
                "traces": traces}

    # Low confidence unsafe → pass
    if confidence < gz_low:
        traces.append(_trace("decision", "low_confidence_pass", "none",
                            f"Confidence {confidence:.2f} < {gz_low} → pass (avoid false positive)",
                            {"decision": "pass"},
                            (time.perf_counter() - t0) * 1000, "zero"))
        return {"decision": "pass", "confidence": confidence,
                "reason": f"Low confidence ({confidence:.2f}) — passing to avoid false positive",
                "traces": traces}

    # High confidence unsafe → block
    decision = "block"
    traces.append(_trace("decision", "final", "none",
                        f"Decision: block (confidence={confidence:.4f})",
                        {"decision": decision, "confidence": confidence},
                        (time.perf_counter() - t0) * 1000, "zero"))
    return {"decision": decision, "confidence": confidence,
            "reason": text_result.get("reason", f"Classified as: {label}"),
            "traces": traces}


def _aggregate_signals(text_result: dict | None, image_result: dict,
                       multimodal_result: dict, cfg: dict) -> tuple[str, float, str]:
    """Weighted score aggregation using live config."""
    scores = []
    weights = []
    min_conf = 1.0
    agent_weights = cfg.get("agent_weights", {"text": 0.4, "image": 0.35, "multimodal": 0.25})
    severity = cfg.get("label_severity", {})

    if text_result:
        tl = text_result.get("label", "safe")
        tc = text_result.get("confidence", 0.5)
        scores.append(severity.get(tl, 0.5) * tc)
        weights.append(agent_weights["text"])
        min_conf = min(min_conf, tc)

    if image_result:
        il = image_result.get("label", "normal")
        ic = image_result.get("confidence", 0.5)
        if il == "nsfw" and ic > 0.5:
            scores.append(severity.get("nsfw", 0.85) * ic)
        else:
            scores.append(severity.get(il, 0.0) * ic)
        weights.append(agent_weights["image"])
        min_conf = min(min_conf, ic)

    if multimodal_result:
        ml = multimodal_result.get("label", "safe")
        mc = multimodal_result.get("confidence", 0.5)
        scores.append(severity.get(ml, 0.5) * mc)
        weights.append(agent_weights["multimodal"])
        min_conf = min(min_conf, mc)

    if not scores:
        return "safe", 1.0, "no_signals"

    # Normalize weights
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    # Weighted score
    weighted_score = sum(s * w for s, w in zip(scores, norm_weights))

    # Convert score back to label + confidence
    label, confidence = _score_to_label(weighted_score, min_conf, cfg)

    # Determine tier from which agent was the final authority
    tier = text_result.get("tier", "L3_llm") if text_result else "decision"

    return label, confidence, tier


def _score_to_label(score: float, min_confidence: float, cfg: dict) -> tuple[str, float]:
    """Convert severity score to discrete label using configurable thresholds."""
    thresholds = cfg.get("score_thresholds", {})

    if score >= thresholds.get("politics", 0.85):
        return "politics" if score >= thresholds.get("politics", 0.95) else "violence", min_confidence
    elif score >= thresholds.get("porn", 0.65):
        return "porn", min_confidence
    elif score >= thresholds.get("gambling", 0.45):
        return "gambling", min_confidence
    elif score >= thresholds.get("toxic", 0.25):
        return "toxic", min_confidence
    elif score >= thresholds.get("spam", 0.10):
        return "spam", min_confidence
    else:
        return "safe", min_confidence


def _trace(node: str, step: str, model: str, input_summary: str,
           output: dict, latency_ms: float, cost: str) -> dict:
    return {
        "node": node, "step": step, "model": model,
        "input": input_summary[:200], "output": output,
        "latency_ms": round(latency_ms, 2), "cost": cost,
        "ts": int(time.time() * 1000),
    }
