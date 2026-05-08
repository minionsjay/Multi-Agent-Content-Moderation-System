"""
Image Agent — NSFW classification + OCR text extraction.

Flow:
  1. Download image from URL
  2. NSFW classification (ViT model)
  3. OCR text extraction (EasyOCR)
  4. If OCR finds text → append to original text for Text Agent review
"""

import time
import logging
import requests
from src.state import ModerationState
from src.skills.image_nsfw import nsfw_detector
from src.skills.image_ocr import image_ocr

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 10  # seconds for image download


async def image_specialist(state: ModerationState) -> dict:
    t0 = time.perf_counter()
    traces = []
    image_url = state.get("image_url", "") or ""
    image_base64 = state.get("image_base64", "") or ""

    if not image_url and not image_base64:
        return {
            "image_result": {"label": "normal", "confidence": 1.0, "tier": "no_image"},
            "traces": [],
        }

    # Step 1: Get image bytes (from URL or base64)
    t1 = time.perf_counter()
    source = "unknown"
    try:
        if image_base64:
            # Remove data:image/...;base64, prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            import base64
            image_bytes = base64.b64decode(image_base64)
            source = "base64_upload"
        else:
            resp = requests.get(image_url, timeout=HTTP_TIMEOUT, headers={
                "User-Agent": "ContentModeration-POC/0.4"
            })
            resp.raise_for_status()
            image_bytes = resp.content
            source = "url_download"
    except Exception as e:
        dl_ms = (time.perf_counter() - t1) * 1000
        traces.append(_t("image_agent", "download_failed", source, image_url or "base64",
                        {"error": str(e)}, dl_ms, "zero"))
        return {
            "image_result": {"label": "normal", "confidence": 0.5,
                            "error": f"Image load failed: {e}", "tier": "download_error"},
            "traces": traces,
        }

    dl_ms = (time.perf_counter() - t1) * 1000
    traces.append(_t("image_agent", "image_load", source, image_url or f"base64 ({len(image_bytes)} bytes)",
                    {"size_bytes": len(image_bytes)}, dl_ms, "zero"))

    # Step 2: NSFW classification (skip model download in POC — use basic validation)
    nsfw_result = {"label": "normal", "confidence": 1.0, "model": "skipped_poc", "image_size": "?"}
    try:
        nsfw_result = nsfw_detector.classify(image_bytes, skip_model=True)
    except Exception:
        pass
    nsfw_ms = 0.0
    traces.append(_t("image_agent", "nsfw_classify",
                    nsfw_result.get("model", "skipped_poc"), image_url,
                    {"label": nsfw_result["label"], "confidence": nsfw_result["confidence"],
                     "image_size": nsfw_result.get("image_size", "?"),
                     "note": "NSFW model download pending — will be available in production"},
                    nsfw_ms, "zero"))

    # Step 3: OCR text extraction
    t3 = time.perf_counter()
    ocr_result = image_ocr.extract(image_bytes)
    ocr_ms = (time.perf_counter() - t3) * 1000
    if ocr_result.get("text"):
        traces.append(_t("image_agent", "ocr_extract",
                        "EasyOCR (ch_sim+en)", image_url,
                        {"text": ocr_result["text"][:200],
                         "confidence": ocr_result["confidence"],
                         "blocks_count": len(ocr_result.get("blocks", []))},
                        ocr_ms, "zero"))

    # Build image_result
    return {
        "image_result": {
            "label": nsfw_result["label"],
            "confidence": nsfw_result["confidence"],
            "nsfw_raw": nsfw_result.get("raw", []),
            "ocr_text": ocr_result.get("text", ""),
            "ocr_confidence": ocr_result.get("confidence", 0.0),
            "tier": "image_agent",
            "model": nsfw_result.get("model", "unknown"),
        },
        # Append OCR text to original text for Text Agent
        "text": (state.get("text", "") + " [OCR: " + ocr_result.get("text", "") + "]").strip()
                if ocr_result.get("text") else state.get("text", ""),
        "traces": traces,
    }


def _t(node: str, step: str, model: str, text: str, output: dict,
       latency_ms: float, cost: str) -> dict:
    return {
        "node": node, "step": step, "model": model,
        "input": text[:200], "output": output,
        "latency_ms": round(latency_ms, 2), "cost": cost,
        "ts": int(time.time() * 1000),
    }
