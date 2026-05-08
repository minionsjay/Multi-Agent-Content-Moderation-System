"""
Local LLM Auditor — llama.cpp + Qwen2.5 GGUF (no API, no network).

Activated when LLM_PROVIDER=local in .env. Falls back gracefully if
llama-cpp-python is not installed yet.

Setup:
  pip install llama-cpp-python
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf -O models/qwen2.5-1.5b.gguf

Config (.env):
  LLM_PROVIDER=local
  LOCAL_LLM_MODEL=./models/qwen2.5-1.5b.gguf
  LOCAL_LLM_N_CTX=2048
  LOCAL_LLM_N_THREADS=4
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# System prompt is imported from llm_audit to keep them in sync
_LOCAL_LLM_AVAILABLE = False
try:
    from llama_cpp import Llama  # noqa: F401
    _LOCAL_LLM_AVAILABLE = True
except ImportError:
    pass

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "./models/qwen2.5-1.5b.gguf")
LOCAL_LLM_N_CTX = int(os.getenv("LOCAL_LLM_N_CTX", "2048"))
LOCAL_LLM_N_THREADS = int(os.getenv("LOCAL_LLM_N_THREADS", "4"))


class LocalLLMAuditor:
    """In-process LLM audit using llama.cpp + Qwen2.5 GGUF.

    Same interface as LLMAuditor.audit() — drop-in replacement for L3.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or LOCAL_LLM_MODEL
        self.n_ctx = LOCAL_LLM_N_CTX
        self.n_threads = LOCAL_LLM_N_THREADS
        self._llm = None
        self._load_error = None

    def _load(self):
        if self._llm is not None:
            return
        if self._load_error is not None:
            return

        if not _LOCAL_LLM_AVAILABLE:
            self._load_error = (
                "llama-cpp-python not installed. "
                "Install: pip install llama-cpp-python"
            )
            logger.error(self._load_error)
            return

        if not os.path.exists(self.model_path):
            self._load_error = (
                f"Model not found: {self.model_path}. "
                f"Download: wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/"
                f"resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf -O {self.model_path}"
            )
            logger.error(self._load_error)
            return

        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            logger.info("Local LLM loaded: %s (ctx=%d, threads=%d)",
                        self.model_path, self.n_ctx, self.n_threads)
        except Exception as e:
            self._load_error = f"Failed to load local LLM: {e}"
            logger.error(self._load_error)

    async def audit(self, text: str, context: dict | None = None,
                    provider: str = "", model: str = "") -> dict:
        """Audit text using local LLM. Same signature as LLMAuditor.audit()."""
        if not text or not text.strip():
            return {"label": "safe", "confidence": 1.0, "reason": "empty text",
                    "cost": "low", "model": "local"}

        self._load()
        if self._llm is None:
            return {"label": "safe", "confidence": 0.5,
                    "reason": f"Local LLM unavailable: {self._load_error}",
                    "cost": "low", "model": "local_unavailable", "error": self._load_error}

        # Use the shared system prompt from llm_audit
        from src.skills.llm_audit import SYSTEM_PROMPT

        user_prompt = f"Text to moderate:\n```\n{text}\n```"
        if context:
            user_prompt += (
                f"\n\nBERT preliminary analysis:\n"
                f"{json.dumps(context, ensure_ascii=False)}"
            )

        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            content = response["choices"][0]["message"]["content"] or "{}"
            result = json.loads(content)
            return {
                "label": result.get("label", "safe"),
                "confidence": float(result.get("confidence", 0.8)),
                "reason": result.get("reason", ""),
                "cost": "low",        # Local = low cost, not high (API)
                "model": "local_qwen",
            }
        except Exception as e:
            logger.error("Local LLM inference failed: %s", e)
            return {"label": "safe", "confidence": 0.5,
                    "reason": f"LLM error: {e}",
                    "cost": "low", "model": "local_error", "error": str(e)}


# Singleton
local_llm = LocalLLMAuditor()
