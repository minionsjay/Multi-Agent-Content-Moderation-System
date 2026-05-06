# 离线部署 & 即插即用指南

> 目标：所有模型本地运行，零 API 依赖，完全离线可用

## 1. 当前状态：哪些已经本地化

| 组件 | 模型 | 大小 | 状态 |
|------|------|------|------|
| 文本向量化 | BAAI/bge-small-zh-v1.5 | 184 MB | 已缓存 · 本地 · 512 维 |
| BERT 毒性分类 | unitary/toxic-bert | 1.3 GB | 已缓存 · 本地 · 英文为主 |
| BERT 多语言 | unitary/multilingual-toxic-xlm-roberta | 80 MB | 已缓存 · 支持中文 |
| 关键词匹配 | AC 自动机 + jieba | < 5 MB | 已缓存 · 本地 · 零依赖 |
| 内存缓存 | TTLCache | 内存 | 本地 · 零依赖 |
| 语义缓存 | ChromaDB | < 50 MB | 本地 · 嵌入式 |

**总计已缓存：约 1.6 GB（在 ~/.cache/huggingface/hub/ 下）**

## 2. 需要补充下载的组件

### 2.1 必选：本地 L3 LLM（替代 DeepSeek API）

当前 L3 层调用 DeepSeek API，离线后需替换为本地模型。

**硬件约束**：当前机器 6.7 GB RAM，无 GPU，约 3 GB 空闲内存。

| 方案 | 模型 | 大小 | 内存占用 | 中文 | 推荐度 |
|------|------|------|----------|------|--------|
| A | 纯 BERT 终审（不用 LLM） | 0 | 已缓存 | 中 | 最简单，当前即可用 |
| B | Qwen2.5-0.5B-Instruct (GGUF Q4) | ~400 MB | ~600 MB | 好 | 轻量首选 |
| C | Qwen2.5-1.5B-Instruct (GGUF Q4_K_M) | ~1 GB | ~1.5 GB | 好 | 效果更好的平衡 |
| D | Qwen2.5-3B-Instruct (GGUF Q4_K_M) | ~2 GB | ~3 GB | 好 | 需加 swap |
| E | Qwen2.5-7B-Instruct (GGUF Q4_K_M) | ~4.5 GB | ~6 GB | 好 | 内存不足，需升级硬件 |

#### 方案 A：纯 BERT 终审（零额外下载）

最简单方式 —— BERT 直接做最终决策，不调 LLM。

代码改动：已在 `text_agent.py` 中支持 `BERT_ENABLED=false` 时跳过 LLM，只需调整置信度阈值即可让 BERT 覆盖 100% 流量。**当前代码已将 BERT 高置信阈值设为 0.95，把阈值降到 0 即可实现纯 BERT 模式。**

优点是零额外下载，缺点是对隐晦违规（谐音、反讽）的识别不如 LLM。

#### 方案 B：llama.cpp + Qwen2.5-0.5B-Instruct-GGUF（推荐）

最小可行 LLM，400MB 下载，600MB 内存即可运行，具备基础中文理解能力。

```bash
# 安装 llama.cpp Python 绑定
pip install llama-cpp-python

# 下载量化模型（选一个来源）
# HuggingFace（需先联网下载一次）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --local-dir ./models/

# 或者用 wget 直接下载（替换为实际 URL）
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -O ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

#### 方案 C：llama.cpp + Qwen2.5-1.5B-Instruct-GGUF

效果更好，但需要 1.5GB 内存。

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir ./models/
```

### 2.2 可选：图片 NSFW 检测模型

当前 POC 跳过了 NSFW 模型下载。离线使用需提前下载：

| 模型 | 大小 | 用途 |
|------|------|------|
| Falconsai/nsfw_image_detection | ~350 MB | ViT 色情/暴力检测 |

```bash
# 预下载（联网执行一次）
python3 -c "
from transformers import pipeline
pipeline('image-classification', model='Falconsai/nsfw_image_detection')
print('NSFW model downloaded.')
"
```

### 2.3 可选：EasyOCR 模型

OCR 模型在首次调用时自动下载。离线使用需提前触发：

| 模型 | 大小 | 用途 |
|------|------|------|
| EasyOCR ch_sim | ~200 MB | 中文文字提取 |
| EasyOCR en | ~100 MB | 英文文字提取 |
| EasyOCR detect (CRAFT) | ~150 MB | 文字检测 |

```bash
# 预下载（联网执行一次）
python3 -c "
import easyocr
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
print('EasyOCR models downloaded.')
"
```

EasyOCR 模型默认存放位置：
- Linux: `~/.EasyOCR/model/`
- Mac: `~/Library/Application Support/EasyOCR/`

### 2.4 可选：ONNX Runtime 模型导出

当前 `toxic-bert` 通过 ONNX Runtime 加速 CPU 推理（2-3x）。ONNX 模型文件需提前导出：

```bash
# 导出 ONNX 模型（约 400 MB）
python3 -m transformers.onnx \
  --model=unitary/toxic-bert \
  --feature=sequence-classification \
  poc/onnx_models/
```

---

## 3. 完整离线物料清单

按方案 B（llama.cpp + Qwen2.5-0.5B）计算：

| 物料 | 大小 | 用途 | 必需 |
|------|------|------|------|
| BGE-small-zh-v1.5 | 184 MB | 文本向量化 | 是 |
| unitary/toxic-bert (HF) | 1.3 GB | BERT 毒性分类 | 是 |
| unitary/toxic-bert (ONNX) | 400 MB | BERT CPU 加速 | 推荐 |
| Qwen2.5-0.5B-Instruct (GGUF Q4) | 400 MB | 本地 LLM 终审 | 是 |
| ChromaDB 持久化文件 | < 100 MB | 语义缓存 | 是 |
| jieba 词典 | ~50 MB | 分词 | 是 |
| EasyOCR (ch_sim + en) | ~450 MB | 图片 OCR | 否（无图片需求可跳过） |
| NSFW ViT 模型 | ~350 MB | 图片审核 | 否（无图片需求可跳过） |
| llama-cpp-python wheel | ~5 MB | LLM 推理 | 是 |

**纯文本离线最小包：约 2.3 GB**
**文本 + 图片离线完整包：约 3.2 GB**

### Python 包离线准备

```bash
# 在当前环境导出所有依赖为 wheel
mkdir -p wheels
pip download -r requirements.txt -d wheels/

# 额外下载 llama.cpp
pip download llama-cpp-python -d wheels/

# wheels/ 目录大小约 500 MB - 2 GB（取决于平台）
```

---

## 4. 代码改动指南

### 4.1 新增本地 LLM Skill

需要新增 `poc/src/skills/llm_local.py`，替代 `llm_audit.py` 的 API 调用：

```python
"""本地 LLM 审核 — llama.cpp + Qwen2.5 GGUF"""

import os
import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a content moderation expert. Analyze the given text and classify it.

Categories: toxic, politics, violence, porn, gambling, spam, safe

Respond in JSON format only:
{"label": "<category>", "confidence": <0.0-1.0>, "reason": "<brief explanation>"}"""


class LocalLLMAuditor:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv(
            "LOCAL_LLM_MODEL",
            "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        logger.info("Local LLM loaded: %s", self.model_path)

    def audit(self, text: str, context: dict | None = None) -> dict:
        from src.config import BERT_HIGH_CONFIDENCE

        # 高置信 BERT 结果直接采纳，不调 LLM
        if context and context.get("bert_confidence", 0) >= BERT_HIGH_CONFIDENCE:
            return {
                "label": context["bert_label"],
                "confidence": context["bert_confidence"],
                "reason": "BERT high confidence — skipping LLM",
                "model": "bert",
            }

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
                "model": "local_qwen",
            }
        except Exception as e:
            logger.error("Local LLM failed: %s", e)
            # 回退到 BERT 结果
            return {
                "label": context.get("bert_label", "safe") if context else "safe",
                "confidence": context.get("bert_confidence", 0.5) if context else 0.5,
                "reason": f"LLM error, fallback to BERT: {e}",
                "model": "fallback_bert",
            }


local_llm = LocalLLMAuditor()
```

### 4.2 修改 text_agent.py（切换 LLM 后端）

在 `poc/src/agents/text_agent.py` 的 L3 部分，将：

```python
llm_result = await asyncio.wait_for(
    llm_auditor.audit(text, context, provider=llm_provider, model=llm_model),
    timeout=8.0)
```

改为：

```python
from src.skills.llm_local import local_llm
llm_result = local_llm.audit(text, context)
```

### 4.3 调整 config.py

```python
# 新增本地 LLM 配置
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "false").lower() != "false"

# 离线模式下降低 BERT 置信度阈值（BERT 搞定更多，LLM 少干）
BERT_HIGH_CONFIDENCE = 0.85  # 原来 0.95
BERT_LOW_CONFIDENCE = 0.3    # 原来 0.4
```

### 4.4 纯 BERT 模式（零 LLM，最简单）

如果暂时不想折腾 llama.cpp，最简单的方式是让 BERT 直接当终审：

```python
# config.py
BERT_ENABLED = True
BERT_HIGH_CONFIDENCE = 0.0   # 设为 0 = 所有 BERT 结果都跳过 LLM
```

此时 `text_agent.py` 中的 L2 会覆盖 100% 流量，LLM 完全不触发。

---

## 5. 离线打包步骤

### Step 1：准备 Python 环境

```bash
# 在联网机器上执行一次
cd poc

# 下载所有 pip 包（含 llama.cpp）
mkdir -p offline/wheels
pip download -r requirements.txt -d offline/wheels/
pip download llama-cpp-python -d offline/wheels/

# 下载量化 LLM
mkdir -p offline/models
# 方式 1：huggingface-cli
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir offline/models/

# 方式 2：wget
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -O offline/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Step 2：确认 HuggingFace 缓存

```bash
# 确认以下目录存在且完整
ls ~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/
ls ~/.cache/huggingface/hub/models--unitary--toxic-bert/snapshots/
```

将整个 `~/.cache/huggingface/hub/` 打包带走（约 1.6 GB）：

```bash
tar -czf offline/huggingface_cache.tar.gz -C ~/.cache/huggingface hub/
```

### Step 3：确认 EasyOCR 缓存（如需图片审核）

```bash
# EasyOCR 模型位置
ls ~/.EasyOCR/model/

# 打包
tar -czf offline/easyocr_models.tar.gz -C ~/.EasyOCR model/
```

### Step 4：离线目标机器安装

```bash
# 1. 创建 Python 虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 离线安装 pip 包
pip install --no-index --find-links offline/wheels/ -r requirements.txt
pip install --no-index --find-links offline/wheels/ llama-cpp-python

# 3. 解压 HuggingFace 缓存
tar -xzf offline/huggingface_cache.tar.gz -C ~/.cache/huggingface/

# 4. 解压 EasyOCR（如需）
tar -xzf offline/easyocr_models.tar.gz -C ~/.EasyOCR/

# 5. 放置 GGUF 模型
cp offline/models/*.gguf poc/models/

# 6. 设置环境变量
cat > poc/.env << 'EOF'
DEEPSEEK_API_KEY=              # 清空，不再需要
LOCAL_LLM_ENABLED=true
LOCAL_LLM_MODEL=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
BERT_ENABLED=true
BERT_HIGH_CONFIDENCE=0.85
LLM_PROVIDER=local
EOF

# 7. 环境检查
cd poc
python check_env.py
# 预期：DeepSeek key 为 WARN，其他全部 OK

# 8. 启动
python -m src.api
```

---

## 6. 硬件要求参考

| 部署方案 | 最低 RAM | 推荐 RAM | GPU | 磁盘 | 适用场景 |
|----------|----------|----------|-----|------|----------|
| 纯 BERT 终审 | 2 GB | 4 GB | 不需要 | 2 GB | 关键词 + BERT，无 LLM |
| Qwen2.5-0.5B (Q4) | 3 GB | 4 GB | 不需要 | 3 GB | 轻量本地 LLM |
| Qwen2.5-1.5B (Q4) | 4 GB | 6 GB | 不需要 | 4 GB | 中等效果本地 LLM |
| Qwen2.5-3B (Q4) | 6 GB | 8 GB | 不需要 | 5 GB | 较好效果 |
| Qwen2.5-7B (Q4) | 8 GB | 12 GB | 推荐 | 8 GB | 接近 API 效果 |
| 全功能（含图片） | +2 GB | +3 GB | 推荐 | +1 GB | 图片审核 + OCR |

当前机器（6.7 GB RAM，无 GPU，3 GB 空闲）推荐 **方案 B：Qwen2.5-0.5B (Q4) + 纯文本**，启动后约 2.5 GB 内存占用，留有足够余量。

---

## 7. 注意事项

1. **首次推理慢**：HuggingFace 模型首次加载需 5-30 秒（编译计算图），后续请求毫秒级
2. **llama.cpp 编译**：`pip install llama-cpp-python` 在某些平台可能需要 C++ 编译器和 CMake，建议在联网机器上提前编译为 wheel
3. **ONNX 可选**：ONNX Runtime 加速是可选的，不导出也不影响功能，只是 BERT CPU 推理慢 2-3 倍
4. **模型授权**：Qwen2.5 系列使用 Apache 2.0 协议，llama.cpp 使用 MIT 协议，均允许商用
5. **ChromaDB 数据**：`data/chroma/` 目录是运行时生成的缓存，不需要预置，首次启动自动创建
6. **swap 建议**：6.7 GB RAM 跑 Qwen2.5-1.5B 建议开启 4 GB swap 作为安全边界
