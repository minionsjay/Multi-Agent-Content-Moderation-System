# 离线部署指南

> 目标：所有模型本地运行，零 API 依赖，完全离线可用

## 当前状态

| 组件 | 模型 | 大小 | 本地? | 说明 |
|------|------|------|------|------|
| 关键词 | AC自动机 + jieba | <5MB | ✓ | 零依赖 |
| 白名单 | 正则 | - | ✓ | 代码内置 |
| L2 BERT | KoalaAI/Text-Moderation | ~400MB | ✓ | 9标签安全分类 |
| L3 LLM | DeepSeek Chat API | 0 | ✗ | **需要联网** |
| 文本向量 | BGE-small-zh-v1.5 | ~95MB | ✓ | 512维，含Embedding缓存 |
| 语义缓存 | ChromaDB | <50MB | ✓ | 嵌入式 |
| 图片哈希 | dHash (Pillow) | 0 | ✓ | 纯Python |
| 图片OCR | EasyOCR | ~450MB | ⚠ | 需预下载 |
| 图片NSFW | Falconsai/nsfw ViT | ~350MB | ⚠ | POC跳过, 生产需下载 |
| Redis | - | - | ⚠ | 可选，不配也能跑 |

**离线缺失项只有一个：L3 LLM。**

## 离线方案：L3 LLM 本地化

当前 L3 调用 DeepSeek API。离线后需要替换为本地模型。两种方式：

### 方案 A：兼容 API 的本地服务（改 .env 即可）

启动 Ollama / vLLM / llama.cpp server，它们都暴露 OpenAI 兼容 API：

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉取模型
ollama pull qwen2.5:7b

# 3. 启动 (默认 localhost:11434)
ollama serve
```

```bash
# .env 改 3 行
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b
```

代码不用改。`llm_audit.py` 用的是 `AsyncOpenAI` 客户端，Ollama 兼容。

### 方案 B：进程内 llama.cpp（改 .env 1 行）

```bash
# 1. 安装
pip install llama-cpp-python

# 2. 下载量化模型
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -O models/qwen2.5-1.5b.gguf
```

```bash
# .env 改 1 行
LLM_PROVIDER=local
```

代码已实装（`llm_local.py`），`text_agent.py` 已集成切换逻辑。llama.cpp 未安装时会优雅降级，不影响系统运行。

## 完整离线物料清单

| 物料 | 大小 | 用途 | 必需 |
|------|------|------|------|
| BGE-small-zh-v1.5 | 95 MB | 文本向量化 | ✓ |
| KoalaAI/Text-Moderation | 400 MB | L2 BERT 安全分类 | ✓ |
| Qwen2.5-1.5B GGUF | 1 GB | L3 本地 LLM | ✓ |
| ChromaDB 持久化 | < 50 MB | 语义缓存 | ✓ |
| jieba 词典 | ~50 MB | 分词 | ✓ |
| EasyOCR (ch_sim+en) | 450 MB | OCR | 图片 |
| NSFW ViT | 350 MB | 图片 NSFW | 图片 |
| llama-cpp-python wheel | 5 MB | LLM 推理 | ✓ |

**纯文本离线包：约 1.6 GB | 文本+图片：约 2.4 GB**

## 离线打包步骤

### Step 1: 预下载所有模型

```bash
# HuggingFace 模型（一次性下载，之后复制缓存即可）
python -c "
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# L2 BERT
pipe = pipeline('text-classification', model='KoalaAI/Text-Moderation')
# BGE Embedding
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
# EasyOCR (如需图片)
import easyocr
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
# NSFW (如需图片)
pipe2 = pipeline('image-classification', model='Falconsai/nsfw_image_detection')
print('All models downloaded')
"

# LLM (选一个)
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -O models/qwen2.5-1.5b.gguf
```

### Step 2: 预下载 pip 包

```bash
# 在联网机器上
mkdir -p offline/wheels
pip download -r requirements.txt -d offline/wheels/
pip download llama-cpp-python -d offline/wheels/
```

### Step 3: 打包 HuggingFace 缓存

```bash
# HF 模型缓存位置
tar -czf offline/huggingface_cache.tar.gz -C ~/.cache/huggingface hub/
# EasyOCR 模型
tar -czf offline/easyocr_models.tar.gz -C ~/.EasyOCR model/
```

### Step 4: 离线目标机器安装

```bash
# 1. 解压模型
tar -xzf offline/huggingface_cache.tar.gz -C ~/.cache/huggingface/
tar -xzf offline/easyocr_models.tar.gz -C ~/.EasyOCR/   # 如需图片
cp offline/models/*.gguf models/

# 2. 离线安装 pip
pip install --no-index --find-links offline/wheels/ -r requirements.txt
pip install --no-index --find-links offline/wheels/ llama-cpp-python

# 3. 配置
cp .env.example .env
# 编辑 .env: LLM_PROVIDER=local

# 4. 验证
python check_env.py

# 5. 启动
python -m src.api
```

## 用 API 但断网怎么办

如果 LLM 仍用 API（不改本地），但其他组件要离线：

只需要 Step 1（下载 HF 模型）+ Step 2（下载 pip 包），不下载 GGUF 模型。保留 `LLM_PROVIDER=deepseek`。

离线时 API 调用会超时 → BERT 结果兜底。系统不会崩溃，但准确率会下降。

## 硬件最低要求

| 方案 | RAM | GPU | 磁盘 |
|------|------|------|------|
| 纯文本 + API LLM | 4 GB | 不需要 | 3 GB |
| 纯文本 + 本地 LLM 1.5B | 6 GB | 不需要 | 5 GB |
| 纯文本 + 本地 LLM 7B | 12 GB | 推荐 | 8 GB |
| 文本+图片 + 本地 LLM | 16 GB | 推荐 | 10 GB |
