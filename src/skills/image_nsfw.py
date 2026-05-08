"""
Image NSFW detection using ViT-based model.
Uses Falconsai/nsfw_image_detection (ViT, ~350MB).
Falls back gracefully if model not yet downloaded.
"""

import io
import logging
import time

logger = logging.getLogger(__name__)


class ImageNSFWDetector:
    def __init__(self):
        self._pipeline = None
        self._downloading = False

    def _load(self):
        if self._pipeline is not None:
            return True
        if self._downloading:
            return False
        self._downloading = True
        try:
            from transformers import pipeline
            from PIL import Image
            logger.info("Loading NSFW image model (Falconsai/nsfw_image_detection)...")
            t0 = time.perf_counter()
            self._pipeline = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                local_files_only=False,  # allow download
                timeout=30,  # give up after 30s
            )
            logger.info("NSFW model loaded (%.1fs)", time.perf_counter() - t0)
            return True
        except Exception as e:
            logger.warning("NSFW model unavailable (will re-download in background): %s", e)
            self._pipeline = None
            self._downloading = False  # retry next time
            return False

    def classify(self, image_bytes: bytes, skip_model: bool = False) -> dict:
        """Classify image as nsfw/normal.

        Returns: {"label": "nsfw"|"normal", "confidence": float, "raw": [...], "model": str}
        """
        if len(image_bytes) > 10 * 1024 * 1024:
            return {"label": "normal", "confidence": 0.5, "raw": [],
                    "model": "none", "error": "Image too large (>10MB)", "skip": True}

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")

            w, h = img.size
            if w < 10 or h < 10:
                return {"label": "normal", "confidence": 0.5, "raw": [],
                        "model": "none", "error": "Image too small", "skip": True}

            # POC: skip model download, use basic validation only
            if skip_model:
                return {"label": "normal", "confidence": 1.0, "raw": [],
                        "model": "poc_basic_validation", "image_size": f"{w}x{h}",
                        "note": "NSFW model pending download"}

            # Try ViT model
            if self._load():
                # Resize for the model
                if max(w, h) > 1024:
                    img.thumbnail((1024, 1024), Image.LANCZOS)
                raw = self._pipeline(img)
                nsfw_score = max(
                    (r["score"] for r in raw if r["label"].lower() == "nsfw"),
                    default=0.0,
                )
                label = "nsfw" if nsfw_score > 0.5 else "normal"
                confidence = nsfw_score if label == "nsfw" else 1.0 - nsfw_score
                return {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "raw": raw,
                    "model": "Falconsai/nsfw_image_detection",
                    "image_size": f"{w}x{h}",
                }

            # Fallback: model not available
            return {"label": "normal", "confidence": 0.5, "raw": [],
                    "model": "fallback_skip", "image_size": f"{w}x{h}",
                    "error": "NSFW model not loaded — will re-evaluate in production"}

        except Exception as e:
            logger.error("Image NSFW classification failed: %s", e)
            return {"label": "normal", "confidence": 0.5, "raw": [],
                    "model": "error", "error": str(e)}


# Singleton
nsfw_detector = ImageNSFWDetector()
