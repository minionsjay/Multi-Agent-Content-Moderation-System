"""
Qwen3Guard-Gen — safety classifier as an LLM backend.

Uses CausalLM.generate() with chat template to classify content safety.
Output format: "Safety: Safe/Unsafe/Controversial\nCategories: Violent, ..."

Config (.env):
  LLM_PROVIDER=qwen_guard
  QWEN_GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B   # or local path
"""

import logging
import os
import re
import time

logger = logging.getLogger(__name__)

QWEN_GUARD_MODEL = os.getenv(
    "QWEN_GUARD_MODEL",
    "Qwen/Qwen3Guard-Gen-0.6B",
)

# Parse Qwen3Guard output
_SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
_CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts"
    r"|PII|Suicide & Self-Harm|Unethical Acts"
    r"|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
)


class QwenGuardAuditor:
    """Qwen3Guard safety classifier — drop-in replacement for LLM L3 audit.

    Uses the user's exact loading and inference pattern:
      - AutoModelForCausalLM + AutoTokenizer
      - apply_chat_template for prompt formatting
      - generate() with max_new_tokens=128
      - regex parse "Safety:" and "Categories:" from output
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or QWEN_GUARD_MODEL
        self._model = None
        self._tokenizer = None
        self._load_error = None
        self._load_time_s = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        if self._load_error is not None:
            return

        t0 = time.perf_counter()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading Qwen3Guard: %s", self.model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )

            self._load_time_s = time.perf_counter() - t0
            logger.info(
                "Qwen3Guard loaded in %.1fs on %s",
                self._load_time_s,
                self._model.device if hasattr(self._model, "device") else "?",
            )

        except Exception as e:
            self._load_error = f"Failed to load {self.model_name}: {e}"
            logger.error(self._load_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self):
        """Pre-load the model. Returns True if loaded OK."""
        self._load()
        return self._model is not None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "loaded": self._model is not None,
            "load_time_s": round(self._load_time_s, 1),
            "error": self._load_error,
        }

    async def audit(
        self,
        text: str,
        context: dict | None = None,
        provider: str = "",
        model: str = "",
    ) -> dict:
        """Classify text using Qwen3Guard.

        Same signature as LLMAuditor.audit() — drop-in replacement for L3.
        Ignores 'context' (BERT result) — Qwen3Guard works directly on the text.
        """
        if model and model != self.model_name:
            self.model_name = model
            self._model = None
            self._tokenizer = None
            self._load_error = None

        if not text or not text.strip():
            return {
                "label": "safe", "confidence": 1.0,
                "reason": "empty text", "cost": "low",
                "model": "Qwen3Guard",
            }

        self._load()
        if self._model is None:
            return {
                "label": "safe", "confidence": 0.5,
                "reason": f"Qwen3Guard unavailable: {self._load_error}",
                "cost": "low",
                "model": "Qwen3Guard",
                "error": self._load_error,
            }

        import torch

        # Use the user's exact prompt format
        messages = [{"role": "user", "content": text}]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        model_inputs = self._tokenizer([prompt], return_tensors="pt").to(
            self._model.device
        )

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True)

        # Parse safety label
        safety_match = _SAFETY_RE.search(content)
        safety = safety_match.group(1).lower() if safety_match else None

        # Parse categories
        categories = _CATEGORY_RE.findall(content)
        categories = [c for c in categories if c.lower() != "none"]

        # Map to safe/unsafe
        if safety == "safe":
            label = "safe"
            confidence = 0.99
        elif safety == "unsafe":
            label = "unsafe"
            confidence = 0.99
        elif safety == "controversial":
            label = "unsafe"
            confidence = 0.7
        elif safety is None:
            # Parse failed — check raw content
            content_lower = content.lower()
            if "unsafe" in content_lower:
                label = "unsafe"
                confidence = 0.7
            elif "safe" in content_lower:
                label = "safe"
                confidence = 0.7
            else:
                label = "safe"
                confidence = 0.5

        reason = f"Qwen3Guard: {safety or 'unknown'}"
        if categories:
            reason += f" — {', '.join(categories)}"

        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "cost": "low",
            "model": "Qwen3Guard",
            "safety_label": safety,
            "categories": categories,
            "raw_output": content,
        }


# Singleton
llm_qwen_guard = QwenGuardAuditor()
