import os
from dotenv import load_dotenv

load_dotenv()

# Disable LangSmith tracing in POC
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# DeepSeek API (primary LLM)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# OpenAI API / 兼容端点 (设置 OPENAI_BASE_URL 可接入 vLLM 等自建服务)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Anthropic API (optional)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

# HTTP proxy control: set LLM_NO_PROXY=true to bypass HTTP_PROXY/HTTPS_PROXY env vars.
# Useful when the machine has a global proxy but LLM API should be called directly.
LLM_NO_PROXY = os.getenv("LLM_NO_PROXY", "false").lower() != "false"

# Local LLM — llama.cpp (Qwen GGUF, zero API cost)
# Set LLM_PROVIDER=local to activate. Requires: pip install llama-cpp-python
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "./models/qwen2.5-1.5b.gguf")
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "false").lower() != "false"

# Local LLM — HuggingFace transformers (model.generate, any CausalLM model)
# Set LLM_PROVIDER=transformers to activate. Requires: pip install transformers torch
TRANSFORMERS_LLM_MODEL = os.getenv(
    "TRANSFORMERS_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)
TRANSFORMERS_LLM_LOAD_IN_4BIT = os.getenv(
    "TRANSFORMERS_LLM_LOAD_IN_4BIT", "true"
).lower() != "false"

# Local LLM — SGLang (RadixAttention + structured JSON, highest throughput)
# Set LLM_PROVIDER=sglang to activate. Requires: pip install sglang[all]
SGLANG_MODEL = os.getenv("SGLANG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SGLANG_TP_SIZE = int(os.getenv("SGLANG_TP_SIZE", "1"))
SGLANG_MEM_FRACTION = float(os.getenv("SGLANG_MEM_FRACTION", "0.85"))

# LLM provider: "deepseek" | "openai" | "anthropic" | "local" | "transformers" | "sglang"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

# BERT model (local)
# Current: unitary/multilingual-toxic-xlm-roberta (80MB, supports 15+ languages)
#   - Trained on Jigsaw multilingual toxic comment dataset
#   - Languages: zh, en, ja, ko, ar, fr, de, es, it, pt, ru, tr, pl, nl, vi
# Old default: unitary/toxic-bert (English only, 1.3GB)
# Other options:
#   - cardiffnlp/twitter-xlm-roberta-base-sentiment (1.9GB, sentiment)
#   - KoalaAI/Text-Moderation (2.7GB, multilingual safety)
#   - govtech/lionguard-2 (multilingual safety classifier, 1.3B)
BERT_MODEL = os.getenv(
    "BERT_MODEL",
    "KoalaAI/Text-Moderation",
)
BERT_ENABLED = os.getenv("BERT_ENABLED", "true").lower() != "false"

# Embedding model (sentence-transformers)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
KEYWORD_DICT = os.getenv("KEYWORD_DICT", "./data/keywords.json")

# BERT confidence thresholds
BERT_HIGH_CONFIDENCE = 0.95
BERT_LOW_CONFIDENCE = 0.4

# Cache similarity threshold
CACHE_SIMILARITY_THRESHOLD = 0.95

# Grey zone
GREY_ZONE_LOW = 0.3
GREY_ZONE_HIGH = 0.7
