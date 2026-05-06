# 本地大模型接入指南

## 改动位置一览

整个 L3 LLM 层涉及的文件：

```
poc/src/skills/llm_audit.py   ← API LLM 调用代码
poc/src/skills/llm_local.py   ← 本地 LLM 调用代码（已实装，待激活）
poc/src/agents/text_agent.py  ← L3 后端选择逻辑（API or local）
poc/src/config.py              ← 配置
.env                           ← 开关
```

**当前状态**：`llm_local.py` 已实装，`text_agent.py` 已集成切换逻辑。只需 2 步激活。

## 激活本地 LLM（2 步）

```bash
# Step 1: 安装 llama.cpp Python 绑定
pip install llama-cpp-python

# Step 2: 下载 Qwen2.5-1.5B 量化模型 (~1GB)
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -O poc/models/qwen2.5-1.5b.gguf
```

```bash
# .env 改 1 行
LLM_PROVIDER=local
```

完成后重启服务，L3 自动切换为本地推理。不需要改任何 Python 代码。

## 方案 A：Ollama（最简单，改 2 行配置）

Ollama 暴露了 OpenAI 兼容的 HTTP API。当前代码不用改。

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉取模型
ollama pull qwen2.5:7b

# 3. 启动（默认监听 localhost:11434）
ollama serve
```

```bash
# .env 改 3 行
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b
```

原理：`llm_audit.py` 里的 `AsyncOpenAI(api_key=..., base_url=...)` 会向 `http://localhost:11434/v1/chat/completions` 发请求，Ollama 接收标准 OpenAI 格式并返回。

## 方案 B：vLLM（生产级，需要 GPU）

```bash
# 1. 安装
pip install vllm

# 2. 启动 OpenAI 兼容服务
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 4096
```

```bash
# .env
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## 方案 C：llama.cpp server（CPU 友好）

适合当前无 GPU 硬件（6.7GB RAM）。

```bash
# 1. 安装
pip install llama-cpp-python

# 2. 下载量化模型（选一个）
# Qwen2.5-1.5B Q4_K_M (~1GB, 效果不错)
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -O models/qwen2.5-1.5b.gguf

# 或 Qwen2.5-0.5B (~400MB, 极致轻量)
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -O models/qwen2.5-0.5b.gguf

# 3. 启动 OpenAI 兼容服务
python -m llama_cpp.server \
  --model models/qwen2.5-1.5b.gguf \
  --host 0.0.0.0 --port 8001 \
  --n_ctx 2048
```

```bash
# .env
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=models/qwen2.5-1.5b.gguf
```

## 方案 D：直接 Python 绑定（零网络开销）

如果不想启动 HTTP 服务，可以直接在 Python 进程内加载 llama.cpp 模型。需要新增 `llm_local.py`：

```python
# poc/src/skills/llm_local.py
import json, logging, os
from llama_cpp import Llama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a content moderation expert..."""  # 同 llm_audit.py

class LocalLLMAuditor:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv(
            "LOCAL_LLM_MODEL", "./models/qwen2.5-1.5b.gguf"
        )
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        logger.info("Local LLM loaded: %s", self.model_path)

    async def audit(self, text: str, context: dict | None = None,
                    provider: str = "", model: str = "") -> dict:
        self._load()

        user_prompt = f"Text to moderate:\n```\n{text}\n```"
        if context:
            user_prompt += f"\n\nBERT preliminary: {json.dumps(context, ensure_ascii=False)}"

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
            result = json.loads(response["choices"][0]["message"]["content"])
            return {
                "label": result.get("label", "safe"),
                "confidence": float(result.get("confidence", 0.8)),
                "reason": result.get("reason", ""),
                "cost": "low",  # 本地模型，成本为 low 而非 high
                "model": "local_qwen",
            }
        except Exception as e:
            logger.error("Local LLM failed: %s", e)
            return {
                "label": "safe", "confidence": 0.5,
                "reason": f"LLM error: {e}", "cost": "low",
                "error": str(e),
            }

local_llm = LocalLLMAuditor()
```

然后在 `text_agent.py` 中切换：

```python
# 原来
from src.skills.llm_audit import llm_auditor
llm_result = await llm_auditor.audit(text, context)

# 改为本地（根据配置决定）
from src.config import LLM_PROVIDER
if LLM_PROVIDER == "local":
    from src.skills.llm_local import local_llm
    llm_result = await local_llm.audit(text, context)
else:
    from src.skills.llm_audit import llm_auditor
    llm_result = await llm_auditor.audit(text, context)
```

## 换模型要改的地方

| 换什么 | 改哪里 | 示例 |
|------|------|------|
| 同后端换模型名 | `.env` 的 `*_MODEL` | `DEEPSEEK_MODEL=deepseek-chat` → `deepseek-reasoner` |
| 换 API 后端 | `.env` 的 `LLM_PROVIDER` | `deepseek` → `openai` |
| 换 API 地址 | `.env` 的 `*_BASE_URL` | `OPENAI_BASE_URL=http://localhost:11434/v1` |
| 换系统提示词 | `llm_audit.py:12` `SYSTEM_PROMPT` | 修改分类类别、语言、输出格式 |
| 换本地 GGUF 模型 | `.env` 的 `LOCAL_LLM_MODEL` | 换另一个 .gguf 文件路径 |
| 加新的 LLM 后端 | `llm_audit.py:_get_client()` | 新增 `elif self.provider == "xxx":` |

## 多国内容审核模型推荐

### 当前硬件 (6.7GB RAM, 无 GPU)

| 模型 | 大小 | 中文 | 英文 | 日韩 | 阿拉伯 | 推荐度 |
|------|------|------|------|------|------|------|
| **Qwen2.5-1.5B-Instruct (Q4)** | 1GB | ★★★★★ | ★★★ | ★★ | ★ | **首选** |
| Qwen2.5-0.5B-Instruct (Q4) | 400MB | ★★★★ | ★★★ | ★★ | ★ | 轻量备选 |
| Llama-3.2-3B-Instruct (Q4) | 2GB | ★★ | ★★★★★ | ★★★ | ★★ | 英文优先 |

### 生产级 (24GB+ GPU)

| 模型 | 中文 | 英文 | 日韩 | 阿拉伯 | 特色 |
|------|------|------|------|------|------|
| **Qwen2.5-14B-Instruct** | ★★★★★ | ★★★★ | ★★★ | ★★★ | **多语言首选** |
| Qwen2.5-7B-Instruct | ★★★★★ | ★★★★ | ★★★ | ★★ | 性价比最高 |
| Aya-23-8B | ★★★ | ★★★★ | ★★★★ | ★★★★ | 23 语言专优 |
| Llama-3.1-8B-Instruct | ★★ | ★★★★★ | ★★★ | ★★★ | 英文最强 |
| Mixtral-8x7B-Instruct | ★★★ | ★★★★★ | ★★★★ | ★★★ | 多语言均衡 |

### 安全专用模型（可作为 L2 替代 BERT）

| 模型 | 大小 | 用途 |
|------|------|------|
| **Llama Guard 3** | 8B | Meta 安全分类器，13 类危害 |
| **Qwen2.5-Guard** | 7B | 阿里安全模型，中文优化 |
| ShieldGemma | 2B/9B | Google 安全分类器 |

这些安全专用模型可以**直接替代 L2 BERT + L3 LLM**，一个模型完成分类，大幅降低成本。但需要微调适配你的分类体系。

### 推荐路线

```
POC 阶段:
  方案 C (llama.cpp server + Qwen2.5-1.5B Q4)
  → 400MB 下载, 1GB 内存, 中文效果好
  → 延迟: 200-500ms/条（比 API 快）

灰度阶段:
  方案 B (vLLM + Qwen2.5-7B-Instruct)
  → 需要 GPU, 效果接近 GPT-4o-mini
  → 延迟: 50-100ms/条

全量阶段:
  vLLM + Qwen2.5-14B-Instruct + Llama Guard 3 (辅助)
  → 多语言覆盖 + 安全专用辅助模型
```

## 系统提示词多语言适配

当前 `SYSTEM_PROMPT` 只有英文。如果需要审核多国内容，提示词应包含目标语言示例：

```python
MULTILINGUAL_SYSTEM_PROMPT = """You are a multilingual content moderation expert.

Analyze the given text and classify it. The text may be in any language.

Categories:
- toxic: insults, hate speech, harassment (any language)
- politics: politically sensitive content
- violence: threats, incitement to violence
- porn: sexually explicit content, solicitation
- gambling: gambling promotion
- spam: advertisements, scams
- safe: normal content

Examples by language:
Chinese:  "你真是个傻逼" → toxic
English: "you are an idiot" → toxic  
Japanese: "お前はバカだ" → toxic
Korean: "넌 바보야" → toxic
Arabic: "أنت غبي" → toxic

Respond in JSON format only:
{"label": "<category>", "confidence": <0.0-1.0>, "reason": "<brief explanation in English>"}"""
```
