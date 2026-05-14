"""
Local LLM Auditor — SGLang (RadixAttention + structured JSON output).

Key advantages over transformers/llama.cpp:
- RadixAttention: system prompt KV cache shared across all requests
  → prefill drops from ~2000 tok to ~200 tok after first request
- Structured output: json_schema guarantee → no more JSON parse errors
- Continuous batching: higher throughput under concurrent load

Activated when LLM_PROVIDER=sglang in .env.

Config (.env):
  LLM_PROVIDER=sglang
  SGLANG_MODEL=Qwen/Qwen2.5-1.5B-Instruct
  SGLANG_TP_SIZE=1                # tensor parallelism (multi-GPU)
  SGLANG_MEM_FRACTION=0.85        # GPU memory fraction
  SGLANG_MAX_TOKENS=256
  SGLANG_TEMPERATURE=0.0
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# ---- Config ----
SGLANG_MODEL = os.getenv("SGLANG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SGLANG_TP_SIZE = int(os.getenv("SGLANG_TP_SIZE", "1"))
SGLANG_MEM_FRACTION = float(os.getenv("SGLANG_MEM_FRACTION", "0.85"))
SGLANG_MAX_TOKENS = int(os.getenv("SGLANG_MAX_TOKENS", "256"))
SGLANG_TEMPERATURE = float(os.getenv("SGLANG_TEMPERATURE", "0.0"))

# Check sglang availability
_SGLANG_AVAILABLE = False
try:
    import sglang as sgl  # noqa: F401
    _SGLANG_AVAILABLE = True
except ImportError:
    pass

# JSON schema for structured output — ensures model always returns valid JSON
# matching our expected {"label": ..., "confidence": ..., "reason": ...} format.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["safe", "toxic", "violence", "politics", "porn", "gambling", "spam"],
            "description": "Content category",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score between 0 and 1",
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation in Chinese or English",
        },
    },
    "required": ["label", "confidence", "reason"],
}


class SGLangAuditor:
    """LLM audit using SGLang Engine — RadixAttention + structured JSON.

    Same interface as LLMAuditor.audit() — drop-in replacement for L3.
    """

    def __init__(
        self,
        model_name: str | None = None,
        tp_size: int | None = None,
        mem_fraction: float | None = None,
    ):
        self.model_name = model_name or SGLANG_MODEL
        self.tp_size = tp_size or SGLANG_TP_SIZE
        self.mem_fraction = mem_fraction or SGLANG_MEM_FRACTION
        self.max_tokens = SGLANG_MAX_TOKENS
        self.temperature = SGLANG_TEMPERATURE

        self._engine = None
        self._load_error = None
        self._load_time_s = 0.0
        self._cache_hits = 0
        self._total_requests = 0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        """Lazy-load SGLang Engine on first use."""
        if self._engine is not None:
            return
        if self._load_error is not None:
            return

        if not _SGLANG_AVAILABLE:
            self._load_error = (
                "sglang not installed. Install: pip install sglang[all]"
            )
            logger.error(self._load_error)
            return

        t0 = time.perf_counter()

        try:
            from sglang import Engine

            logger.info(
                "Loading SGLang engine: %s (tp=%d, mem_frac=%.2f)",
                self.model_name, self.tp_size, self.mem_fraction,
            )

            self._engine = Engine(
                model_path=self.model_name,
                tp_size=self.tp_size,
                mem_fraction=self.mem_fraction,
                trust_remote_code=True,
                # RadixAttention is enabled by default — no config needed
                # The engine will automatically detect and cache repeated prefixes
                log_level="error",
            )

            self._load_time_s = time.perf_counter() - t0
            logger.info(
                "SGLang engine loaded in %.1fs (RadixAttention enabled)",
                self._load_time_s,
            )

        except Exception as e:
            self._load_error = f"Failed to load SGLang engine: {e}"
            logger.error(self._load_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self) -> bool:
        """Pre-load the engine. Call at startup."""
        self._load()
        if self._engine is not None:
            # Run one warmup inference to populate system prompt cache
            try:
                self._generate("warmup")
                logger.info("SGLang warmup complete (system prompt cached)")
            except Exception as e:
                logger.warning("SGLang warmup inference failed: %s", e)
            return True
        return False

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def load_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "loaded": self._engine is not None,
            "load_time_s": round(self._load_time_s, 1),
            "tp_size": self.tp_size,
            "mem_fraction": self.mem_fraction,
            "radix_cache_hits": self._cache_hits,
            "total_requests": self._total_requests,
            "error": self._load_error,
        }

    @property
    def radix_hit_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._cache_hits / self._total_requests

    async def audit(
        self,
        text: str,
        context: dict | None = None,
        provider: str = "",
        model: str = "",
    ) -> dict:
        """Audit text using SGLang Engine.

        Same signature as LLMAuditor.audit() — drop-in replacement for L3.
        """
        if not text or not text.strip():
            return {
                "label": "safe", "confidence": 1.0,
                "reason": "empty text", "cost": "low",
                "model": self.model_name.rsplit("/", 1)[-1],
            }

        self._load()
        if self._engine is None:
            return {
                "label": "safe", "confidence": 0.5,
                "reason": f"SGLang unavailable: {self._load_error}",
                "cost": "low",
                "model": "sglang_unavailable",
                "error": self._load_error,
            }

        # Build user prompt (same format as all other providers)
        from src.skills.llm_audit import SYSTEM_PROMPT

        user_prompt = f"Text to moderate:\n```\n{text}\n```"
        if context:
            user_prompt += (
                f"\n\nBERT preliminary analysis:\n"
                f"{json.dumps(context, ensure_ascii=False)}"
            )

        # Construct the full prompt using chat template format
        # SGLang will cache the system prompt prefix via RadixAttention
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        full_prompt = self._format_chat(messages)

        try:
            result = self._generate(full_prompt)
            result["cost"] = "low"
            result["model"] = self.model_name.rsplit("/", 1)[-1]
            return result

        except Exception as e:
            logger.error("SGLang inference failed: %s", e)
            return {
                "label": "safe", "confidence": 0.5,
                "reason": f"SGLang error: {e}",
                "cost": "low",
                "model": "sglang_error",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _format_chat(self, messages: list[dict]) -> str:
        """Format messages into a prompt string. SGLang handles chat templates
        natively, but we pre-format for RadixAttention prefix detection."""
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}</s>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _generate(self, prompt: str) -> dict:
        """Run inference with structured JSON output constraint."""
        self._total_requests += 1

        sampling_params = {
            "temperature": self.temperature if self.temperature > 0 else 0.0,
            "max_new_tokens": self.max_tokens,
            # JSON schema constraint — SGLang guarantees valid JSON output
            "json_schema": json.dumps(RESPONSE_SCHEMA),
        }

        # If temperature is 0, disable sampling entirely
        if self.temperature == 0.0:
            sampling_params["top_p"] = 1.0

        output = self._engine.generate(prompt, sampling_params)
        content = output.get("text", "{}")

        # With json_schema constraint, this should always succeed
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try extracting JSON from response
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            return {
                "label": "safe",
                "confidence": 0.5,
                "reason": f"JSON parse fallback: {content[:200]}",
            }


# Singleton
llm_sglang = SGLangAuditor()
