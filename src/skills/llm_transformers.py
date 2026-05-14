"""
Local LLM Auditor — HuggingFace transformers (model.generate).

Supports any CausalLM model from HuggingFace. Uses device_map="auto"
for GPU/CPU auto-detection. Supports 4-bit quantization for memory efficiency.

Activated when LLM_PROVIDER=transformers in .env.

Config (.env):
  LLM_PROVIDER=transformers
  TRANSFORMERS_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct   # or any HF model
  TRANSFORMERS_LLM_LOAD_IN_4BIT=true                    # 4-bit quantization
  TRANSFORMERS_LLM_MAX_NEW_TOKENS=256
"""

import json
import logging
import os
import time
from src.config import HF_LOCAL_FILES_ONLY

logger = logging.getLogger(__name__)

# ---- Config ----
TRANSFORMERS_LLM_MODEL = os.getenv(
    "TRANSFORMERS_LLM_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
TRANSFORMERS_LLM_LOAD_IN_4BIT = os.getenv(
    "TRANSFORMERS_LLM_LOAD_IN_4BIT", "true"
).lower() != "false"
TRANSFORMERS_LLM_MAX_NEW_TOKENS = int(os.getenv(
    "TRANSFORMERS_LLM_MAX_NEW_TOKENS", "256"
))
TRANSFORMERS_LLM_TEMPERATURE = float(os.getenv(
    "TRANSFORMERS_LLM_TEMPERATURE", "0.0"
))
TRANSFORMERS_LLM_DEVICE_MAP = os.getenv(
    "TRANSFORMERS_LLM_DEVICE_MAP", "auto"
)  # "auto", "cpu", "cuda:0"


class TransformersLLMAuditor:
    """Local LLM audit using HuggingFace transformers (model.generate).

    Drop-in replacement for LLMAuditor.audit() — same interface.
    Supports any CausalLM model: Qwen2.5, Llama-3, Mistral, etc.
    """

    def __init__(
        self,
        model_name: str | None = None,
        load_in_4bit: bool | None = None,
        device_map: str | None = None,
    ):
        self.model_name = model_name or TRANSFORMERS_LLM_MODEL
        self.load_in_4bit = (
            load_in_4bit
            if load_in_4bit is not None
            else TRANSFORMERS_LLM_LOAD_IN_4BIT
        )
        self.device_map = device_map or TRANSFORMERS_LLM_DEVICE_MAP
        self.max_new_tokens = TRANSFORMERS_LLM_MAX_NEW_TOKENS
        self.temperature = TRANSFORMERS_LLM_TEMPERATURE

        self._model = None
        self._tokenizer = None
        self._load_error = None
        self._load_time_s = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        """Lazy-load model + tokenizer on first use."""
        if self._model is not None:
            return
        if self._load_error is not None:
            return

        t0 = time.perf_counter()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading tokenizer: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=HF_LOCAL_FILES_ONLY,
            )

            # Build loading kwargs
            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": HF_LOCAL_FILES_ONLY,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }

            if self.load_in_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    load_kwargs["load_in_4bit"] = True
                    load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    load_kwargs["bnb_4bit_use_double_quant"] = True
                    logger.info("Using 4-bit quantization (bitsandbytes)")
                except ImportError:
                    logger.warning(
                        "bitsandbytes not installed — falling back to full precision. "
                        "Install: pip install bitsandbytes"
                    )
                    self.load_in_4bit = False

            # device_map after quantization settings
            load_kwargs["device_map"] = self.device_map

            logger.info(
                "Loading model: %s (4bit=%s, device=%s)",
                self.model_name,
                self.load_in_4bit,
                self.device_map,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            self._load_time_s = time.perf_counter() - t0
            logger.info(
                "Model loaded in %.1fs on %s",
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
        """Pre-load the model (call at startup). Returns True if loaded OK."""
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
            "load_in_4bit": self.load_in_4bit,
            "device_map": self.device_map,
            "error": self._load_error,
        }

    async def audit(
        self,
        text: str,
        context: dict | None = None,
        provider: str = "",
        model: str = "",
    ) -> dict:
        """Audit text using local transformers model.

        Same signature as LLMAuditor.audit() — drop-in replacement for L3.
        """
        if not text or not text.strip():
            return {
                "label": "safe", "confidence": 1.0,
                "reason": "empty text", "cost": "low",
                "model": self.model_name.rsplit("/", 1)[-1],
            }

        self._load()
        if self._model is None:
            return {
                "label": "safe", "confidence": 0.5,
                "reason": f"Model unavailable: {self._load_error}",
                "cost": "low",
                "model": "local_unavailable",
                "error": self._load_error,
            }

        # Build prompt
        from src.skills.llm_audit import SYSTEM_PROMPT

        user_prompt = f"Text to moderate:\n```\n{text}\n```"
        if context:
            user_prompt += (
                f"\n\nBERT preliminary analysis:\n"
                f"{json.dumps(context, ensure_ascii=False)}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            import torch

            # Use chat template for Qwen/Llama models
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._tokenizer(prompt, return_tensors="pt")
            # Move to same device as model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the generated part
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            content = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse JSON from response
            result = self._parse_response(content)
            result["cost"] = "low"
            result["model"] = self.model_name.rsplit("/", 1)[-1]
            return result

        except Exception as e:
            logger.error("Transformers LLM inference failed: %s", e)
            return {
                "label": "safe", "confidence": 0.5,
                "reason": f"LLM error: {e}",
                "cost": "low",
                "model": "local_error",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_response(self, content: str) -> dict:
        """Extract JSON from model output. Handles markdown code blocks."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` block
        import re
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: keyword-based heuristic
        content_lower = content.lower()
        unsafe_keywords = [
            "toxic", "unsafe", "violence", "politics", "porn", "gambling",
            "spam", "block", "harmful", "违规", "不安全",
        ]
        has_unsafe = any(kw in content_lower for kw in unsafe_keywords)
        return {
            "label": "unsafe" if has_unsafe else "safe",
            "confidence": 0.5,
            "reason": f"JSON parse failed, heuristic fallback. Raw: {content[:200]}",
        }


# Singleton
llm_transformers = TransformersLLMAuditor()
