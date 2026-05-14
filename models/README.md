# 模型存放目录

此目录存放本地下载的模型文件。由于模型文件较大 (几百 MB ~ 几 GB)，**不提交到 Git**。

## 下载方式

在可联网的机器上运行：

```bash
# 下载所有模型
python download_models.py

# 仅下载文本审核必需的模型
python download_models.py --text-only

# 也下载本地 LLM
python download_models.py --llm
```

## 需要哪些模型

| 用途 | 模型 | 大小 | 存放路径 |
|------|------|------|---------|
| L2 文本分类 | KoalaAI/Text-Moderation | ~400MB | `~/.cache/huggingface/hub/` |
| 语义向量 | BAAI/bge-small-zh-v1.5 | ~95MB | `~/.cache/huggingface/hub/` |
| L3 安全分类 (可选) | Qwen/Qwen3Guard-Gen-0.6B | ~1.2GB | `~/.cache/huggingface/hub/` |
| L3 通用 LLM (可选) | Qwen/Qwen2.5-1.5B-Instruct | ~3GB | `~/.cache/huggingface/hub/` |
| 图片 NSFW (可选) | Falconsai/nsfw_image_detection | ~350MB | `~/.cache/huggingface/hub/` |

## 手动下载

如果 HuggingFace 下载慢，可以用镜像站：

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在 Python 中下载后保存到本地
python -c "
from transformers import AutoModel
# 下载并保存到 models/ 目录
AutoModel.from_pretrained('KoalaAI/Text-Moderation').save_pretrained('./models/KoalaAI-Text-Moderation')
"
```

## 使用本地模型

在 `.env` 中设置路径：

```bash
# 如果模型在 HuggingFace 缓存中 (默认)
BERT_MODEL=KoalaAI/Text-Moderation

# 如果模型放在 models/ 目录下
BERT_MODEL=./models/KoalaAI-Text-Moderation
QWEN_GUARD_MODEL=./models/Qwen3Guard-Gen-0.6B
TRANSFORMERS_LLM_MODEL=./models/Qwen2.5-1.5B-Instruct

# 禁止自动下载
HF_LOCAL_FILES_ONLY=true
```
