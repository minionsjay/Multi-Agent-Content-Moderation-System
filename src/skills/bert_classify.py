import logging
import concurrent.futures
from src.config import BERT_MODEL, BERT_HIGH_CONFIDENCE, BERT_LOW_CONFIDENCE

logger = logging.getLogger(__name__)

BERT_LOAD_TIMEOUT = 60  # seconds — allow time for model architecture build after weight load


class BERTClassifier:
    """Text classification using a HuggingFace transformers pipeline.

    Default model: KoalaAI/Text-Moderation (fine-tuned for safety classification).
    """

    LABEL_MAP = {
        # Jigsaw toxic-bert labels
        "toxic": "unsafe",
        "severe_toxic": "unsafe",
        "obscene": "unsafe",
        "threat": "unsafe",
        "insult": "unsafe",
        "identity_hate": "unsafe",
        # KoalaAI/Text-Moderation labels
        # OK=normal, H=hate speech, H2=hate speech variant
        # SH=sexual harassment, HR=harassment
        # V=violence, V2=violence variant
        # S=sexual content, S3=sexual content variant
        "OK": "safe",
        "H": "unsafe", "H2": "unsafe",
        "SH": "unsafe", "HR": "unsafe",
        "V": "unsafe", "V2": "unsafe",
        "S": "unsafe", "S3": "unsafe",
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
                )
                self._pipeline = future.result(timeout=BERT_LOAD_TIMEOUT)
            # Count layers in a model-agnostic way (handles BERT, RoBERTa, XLM-RoBERTa, etc.)
            num_layers = 0
            if hasattr(self._pipeline, 'model'):
                m = self._pipeline.model
                for attr in ('bert', 'roberta', 'xlm_roberta', 'distilbert'):
                    if hasattr(m, attr):
                        encoder = getattr(m, attr).encoder
                        if hasattr(encoder, 'layer'):
                            num_layers = len(encoder.layer)
                            break
            logger.info("BERT model loaded (%d layers).", num_layers)
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
            raw = self._pipeline(text)  # top 1 label (works for all model types)
        except Exception as e:
            logger.warning("BERT unavailable, escalating to LLM: %s", e)
            return {"label": "unsafe", "confidence": 0.5, "raw": [], "cost": "low", "error": str(e)}

        if not raw:
            return {"label": "safe", "confidence": 1.0, "raw": [], "cost": "low"}

        # Handle both output formats:
        #  Format A: [{'label': 'OK', 'score': 0.98}]  (single label)
        #  Format B: [[{'label': 'OK', 'score': 0.98}, ...]]  (top_k=None, list of lists)
        items = raw
        if isinstance(raw[0], list):
            items = raw[0]

        # Find the best label and determine safe/unsafe
        best_label = items[0].get("label", "") if items else ""
        best_score = items[0].get("score", 0.0) if items else 0.0
        mapped = self.LABEL_MAP.get(best_label, "")

        # If the top label is a harmful category, it's unsafe regardless of score
        if mapped == "unsafe":
            label = "unsafe"
            confidence = best_score
        elif mapped == "safe":
            label = "safe"
            confidence = best_score
        else:
            # Unknown label: check if any unsafe label scored > 0.5
            max_toxic = 0.0
            for item in items:
                if not isinstance(item, dict): continue
                if self.LABEL_MAP.get(item.get("label", ""), "") == "unsafe":
                    max_toxic = max(max_toxic, item.get("score", 0.0))
            label = "unsafe" if max_toxic > 0.5 else "safe"
            confidence = max_toxic if label == "unsafe" else 1.0 - max_toxic

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
