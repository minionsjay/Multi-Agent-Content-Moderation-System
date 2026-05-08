"""
OCR for extracting text from images.
Uses easyocr (supports Chinese + English).
Falls back gracefully if not available.
"""

import io
import logging
import time

logger = logging.getLogger(__name__)


class ImageOCR:
    def __init__(self):
        self._reader = None
        self._loading = False

    def _load(self):
        if self._reader is not None:
            return True
        if self._loading:
            return False
        self._loading = True
        try:
            import easyocr
            logger.info("Loading EasyOCR (ch_sim + en)...")
            t0 = time.perf_counter()
            self._reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
            logger.info("EasyOCR loaded (%.1fs)", time.perf_counter() - t0)
            return True
        except Exception as e:
            logger.warning("EasyOCR load failed: %s", e)
            self._reader = None
            return False

    def extract(self, image_bytes: bytes) -> dict:
        """Extract text from image.

        Returns: {"text": str, "confidence": float, "blocks": list}
        """
        if not self._load():
            return {"text": "", "confidence": 0.0, "blocks": [], "error": "OCR not loaded"}

        try:
            from PIL import Image
            import numpy as np
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(img)

            results = self._reader.readtext(arr)
            if not results:
                return {"text": "", "confidence": 0.0, "blocks": []}

            blocks = []
            full_text = []
            total_conf = 0.0
            for bbox, text, conf in results:
                blocks.append({"text": text, "confidence": round(conf, 4)})
                full_text.append(text)
                total_conf += conf

            combined = " ".join(full_text)
            avg_conf = total_conf / len(results) if results else 0.0

            return {
                "text": combined,
                "confidence": round(avg_conf, 4),
                "blocks": blocks,
            }
        except Exception as e:
            logger.error("OCR failed: %s", e)
            return {"text": "", "confidence": 0.0, "blocks": [], "error": str(e)}


# Singleton
image_ocr = ImageOCR()
