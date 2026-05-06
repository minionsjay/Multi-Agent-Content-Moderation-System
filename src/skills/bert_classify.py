import logging
import concurrent.futures
from src.config import BERT_MODEL, BERT_HIGH_CONFIDENCE, BERT_LOW_CONFIDENCE

logger = logging.getLogger(__name__)

BERT_LOAD_TIMEOUT = 60  # seconds — allow time for model architecture build after weight load


class BERTClassifier:
    """Text classification using a HuggingFace transformers pipeline.

    Default model: unitary/toxic-bert (fine-tuned BERT-base for toxicity).
    """

    LABEL_MAP = {
        "toxic": "unsafe",
        "severe_toxic": "unsafe",
        "obscene": "unsafe",
        "threat": "unsafe",
        "insult": "unsafe",
        "identity_hate": "unsafe",
    }

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or BERT_MODEL
        self._pipeline = None
        self._load_error = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        if self._load_error is not None:
            raise self._load_error

        from transformers import pipeline
        logger.info("Loading BERT model: %s ...", self.model_name)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    pipeline,
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    truncation=True,
                    max_length=512,
                    local_files_only=True,
                )
                self._pipeline = future.result(timeout=BERT_LOAD_TIMEOUT)
            logger.info("BERT model loaded (%d weight shards).", len(self._pipeline.model.bert.encoder.layer) if hasattr(self._pipeline, 'model') else 0)
        except concurrent.futures.TimeoutError:
            self._load_error = RuntimeError(f"BERT model loading timed out after {BERT_LOAD_TIMEOUT}s")
            logger.error(str(self._load_error))
            raise self._load_error
        except Exception as e:
            self._load_error = e
            logger.error("BERT model loading failed: %s", e)
            raise e

    def classify(self, text: str, model_override: str = "") -> dict:
        if model_override and model_override != self.model_name:
            # Switch model — reinitialize pipeline
            self.model_name = model_override
            self._pipeline = None
            self._load_error = None
        if not text or not text.strip():
            return {"label": "safe", "confidence": 1.0, "raw": [], "cost": "low"}

        try:
            self._load_pipeline()
            raw = self._pipeline(text, top_k=None)  # type: ignore
        except Exception as e:
            logger.warning("BERT unavailable, escalating to LLM: %s", e)
            return {"label": "unsafe", "confidence": 0.5, "raw": [], "cost": "low", "error": str(e)}

        if not raw:
            return {"label": "safe", "confidence": 1.0, "raw": [], "cost": "low"}

        max_toxic_score = 0.0
        for item in raw:
            label = item["label"].lower()
            score = item["score"]
            if label in self.LABEL_MAP and score > max_toxic_score:
                max_toxic_score = score

        label = "unsafe" if max_toxic_score > 0.5 else "safe"
        confidence = max_toxic_score if label == "unsafe" else 1.0 - max_toxic_score

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw": raw,
            "cost": "low",
        }

    def should_skip_llm(self, result: dict) -> bool:
        if result.get("label") == "unsafe" and result.get("confidence", 0) >= BERT_HIGH_CONFIDENCE:
            return True
        if result.get("label") == "safe" and result.get("confidence", 0) >= BERT_HIGH_CONFIDENCE:
            return True
        return False

    def should_skip_bert(self, result: dict) -> bool:
        return result.get("confidence", 1.0) < BERT_LOW_CONFIDENCE

    def warmup(self):
        """Pre-load model on startup so first request is fast."""
        logger.info("BERT warmup: pre-loading model...")
        t0 = __import__("time").perf_counter()
        self.classify("warmup test")
        logger.info("BERT warmup complete (%.1fs)", __import__("time").perf_counter() - t0)


# Singleton
bert_classifier = BERTClassifier()

