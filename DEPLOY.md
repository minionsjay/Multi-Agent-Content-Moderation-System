# 部署指南

## 1. 部署架构概览

```
目标机器 (可以是单台服务器，也可以拆成两台)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌──────────────┐     ┌──────────────────────────┐      │
│  │ FastAPI       │     │ GPU / CPU 推理服务        │      │
│  │ (热路径)      │     │                          │      │
│  │              │     │  ┌──────────────────┐    │      │
│  │ • 关键词过滤  │     │  │ BERT 分类 (L2)    │    │      │
│  │ • L0 内存缓存 │ ───→│  │ KoalaAI/Text-Mod │    │      │
│  │ • ChromaDB   │     │  │ 或 ONNX 量化版本  │    │      │
│  │ • BGE 嵌入    │     │  └──────────────────┘    │      │
│  │              │     │                          │      │
│  └──────────────┘     │  ┌──────────────────┐    │      │
│                       │  │ LLM 推理 (L3)     │    │      │
│                       │  │ llama.cpp + Qwen  │    │      │
│                       │  │ 或 vLLM + Qwen    │    │      │
│                       │  └──────────────────┘    │      │
│                       └──────────────────────────┘      │
│                                                         │
│  依赖: Redis (可选), 无其他外部依赖                         │
└─────────────────────────────────────────────────────────┘
```

**核心原则**：
- 热路径（缓存 + 关键词）在 API 进程内完成，不依赖 GPU
- BERT 模型可以用 CPU（ONNX 量化后 ~50ms），也可以用 GPU
- LLM 模型建议用 GPU（llama.cpp 或 vLLM），也可以用 CPU + 量化（~200ms）
- ChromaDB 嵌入式运行，不需要独立部署

---

## 2. 环境要求

### 方案 A：纯 CPU（最低配置，适合 POC/测试）

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 20 GB | 50 GB (模型 ~8GB) |
| Python | 3.11+ | 3.12 |
| OS | Ubuntu 22.04 / Debian 12 | 同左 |

### 方案 B：CPU + GPU（生产推荐）

| 组件 | 推荐配置 |
|------|----------|
| GPU | NVIDIA RTX 3060+ (12GB VRAM+) 或 A10/A100 |
| CPU | 8 核+ |
| 内存 | 32 GB+ |
| 磁盘 | 100 GB SSD |

---

## 3. 快速部署（5 步）

### 步骤 1：创建 Python 虚拟环境

```bash
# 推荐使用 venv（Python 自带，不需要额外安装）
python3 -m venv venv
source venv/bin/activate

# 或者用 conda
# conda create -n moderation python=3.12 -y
# conda activate moderation
```

### 步骤 2：安装依赖

```bash
# 基础依赖（文本审核）
pip install -r requirements.txt

# 如果要本地 LLM（llama.cpp 方案）
pip install llama-cpp-python

# 如果要 GPU 加速的 llama.cpp
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 如果要用 vLLM 替代 llama.cpp（需要 GPU）
# pip install vllm
```

### 步骤 3：下载模型

```bash
# 下载所有必需的模型（需要网络）
python download_models.py --text-only

# 或者手动下载：
# BGE 嵌入模型（热路径缓存必须，~95MB）
# 自动从 HuggingFace 拉取: BAAI/bge-small-zh-v1.5

# BERT 分类模型（已配置 KoalaAI/Text-Moderation，~2.7GB）
# 自动从 HuggingFace 拉取

# 本地 LLM 模型（可选，~1GB，Qwen2.5-1.5B GGUF）
# wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf -O models/qwen2.5-1.5b.gguf

# 如果目标机器无网络：先在联网机器上下载，打包后拷贝
# 参考 OFFLINE_DEPLOY.md
```

### 步骤 4：配置 .env

```bash
cp .env.example .env
```

根据部署方案编辑 `.env`：

**方案 A：用外部 LLM API（DeepSeek，最简单，POC 推荐）**
```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-chat
BERT_MODEL=KoalaAI/Text-Moderation
BERT_ENABLED=true
EMBED_MODEL=BAAI/bge-small-zh-v1.5
```

**方案 B：本地 LLM（llama.cpp + Qwen，零 API 成本）**
```env
LLM_PROVIDER=local
LOCAL_LLM_MODEL=./models/qwen2.5-1.5b.gguf
LOCAL_LLM_ENABLED=true
BERT_MODEL=KoalaAI/Text-Moderation
BERT_ENABLED=true
EMBED_MODEL=BAAI/bge-small-zh-v1.5
```

**方案 C：混合（BERT 本地 + LLM 走 API，推荐过渡期使用）**
```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key-here
BERT_MODEL=KoalaAI/Text-Moderation
BERT_ENABLED=true
```

### 步骤 5：验证并启动

```bash
# 检查所有依赖和模型是否就绪
python check_env.py

# 启动服务
python -m src.api

# 或者指定 host/port/workers
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

启动后访问 `http://<机器IP>:8000` 可以看到测试界面，`/health` 端点返回健康状态。

---

## 4. 生产部署

### 4.1 systemd 服务（推荐）

```ini
# /etc/systemd/system/content-moderation.service
[Unit]
Description=Content Moderation API
After=network.target

[Service]
Type=simple
User=app
WorkingDirectory=/opt/content-moderation/poc
Environment=PATH=/opt/content-moderation/poc/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/opt/content-moderation/poc/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now content-moderation
sudo systemctl status content-moderation
```

### 4.2 Docker 部署

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 如果要用本地 LLM，取消注释：
# RUN pip install llama-cpp-python

COPY . .

# 预下载 BGE 嵌入模型（避免首次请求冷启动）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
docker build -t content-moderation .
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e DEEPSEEK_API_KEY=sk-xxx \
  content-moderation
```

### 4.3 Nginx 反向代理

```nginx
upstream moderation_backend {
    server 127.0.0.1:8000;
    # 多实例时添加更多 server
    # server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name moderation.example.com;

    # 请求体大小限制（图片审核需要）
    client_max_body_size 20M;

    location / {
        proxy_pass http://moderation_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 30s;  # LLM 调用可能较慢
    }
}
```

---

## 5. 模型选型指南

### BERT 分类模型（L2, 必须）

| 模型 | 大小 | 延迟(CPU) | 适用场景 |
|------|------|------|------|
| `KoalaAI/Text-Moderation` | 2.7 GB | ~150ms | **当前默认**，多语言 |
| `unitary/toxic-bert` | 1.3 GB | ~50ms | 纯英文，快但仅英文 |
| ONNX 量化 `toxic-bert` | ~400 MB | ~17ms | 生产推荐，需预先导出 |

> **推荐**：先用 `KoalaAI/Text-Moderation`（HuggingFace pipeline），稳定后导出 ONNX 量化版本（延迟降 3x，内存降 6x）。中文场景建议后续用 LoRA 微调 Qwen2.5-1.5B。

### LLM 模型（L3, 深度审核）

| 方案 | 延迟 | 成本 | 适用场景 |
|------|------|------|------|
| DeepSeek API (`deepseek-chat`) | 500-2000ms | $0.002/次 | **POC 推荐**，零运维 |
| llama.cpp + `Qwen2.5-1.5B` (Q4) | 200-500ms (CPU) | 零 | 小规模生产 |
| llama.cpp + `Qwen2.5-7B` (Q4) | 500-1000ms (GPU) | 零 | 中等规模，准确率更好 |
| vLLM + `Qwen2.5-7B` (FP16) | 50-200ms (GPU) | 零 | 大规模生产，高吞吐 |

> **推荐路径**：POC 用 DeepSeek API → 验证通过后切 `llama.cpp + Qwen2.5-1.5B` → 流量上去后升级 `vLLM + Qwen2.5-7B`。

### 嵌入模型（热路径缓存，必须）

| 模型 | 大小 | 维度 | 延迟(CPU) |
|------|------|------|------|
| `BAAI/bge-small-zh-v1.5` | 95 MB | 512 | ~5ms |
| `BAAI/bge-large-zh-v1.5` | 1.3 GB | 1024 | ~20ms |

> 保持默认 `bge-small-zh` 即可，缓存命中率够用。需要更高命中率时切 `bge-large`。

---

## 6. 离线部署（目标机器无网络）

### 6.1 在联网机器上准备离线包

```bash
# 1. 导出 pip 包
mkdir offline_packages
pip download -r requirements.txt -d offline_packages/
pip download llama-cpp-python -d offline_packages/

# 2. 导出 HuggingFace 模型
python download_models.py --text-only
tar -czf hf_cache.tar.gz -C ~/.cache/huggingface/ .

# 3. 下载 LLM 模型文件（可选）
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf -O qwen2.5-1.5b.gguf
```

### 6.2 拷贝到目标机器

```bash
# 需要拷贝的内容
rsync -av \
  poc/ \                    # 项目代码
  offline_packages/ \       # pip 离线包
  hf_cache.tar.gz \         # HF 模型
  qwen2.5-1.5b.gguf \       # LLM 模型(可选)
  user@target-host:/opt/content-moderation/
```

### 6.3 在目标机器上安装

```bash
cd /opt/content-moderation/poc
chmod +x install.sh
./install.sh          # 完整安装
# 或
./install.sh --text-only  # 仅文本审核（跳过 OCR/图片模型）
```

---

## 7. 监控与运维

### 7.1 健康检查

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.5.0", "architecture": "gateway + langgraph"}

# Gateway 统计
curl http://localhost:8000/gateway/stats
# {"memory_cache_hit_rate": 0.15, "keyword_hit_rate": 0.08, ...}

# 人工复核队列
curl http://localhost:8000/review/stats
```

### 7.2 关键指标

| 指标 | 含义 | 告警阈值 |
|------|------|------|
| `gateway.stats.hot_path_rate` | 热路径拦截率 | < 30% |
| `gateway.stats.chroma_cache_hit_rate` | 语义缓存命中率 | < 10% |
| L3 LLM 调用率 | 最贵路径占比 | > 30% |
| `/moderate` P99 延迟 | 端到端延迟 | > 3s |

### 7.3 日志

```bash
# 查看服务日志
journalctl -u content-moderation -f

# 调整日志级别（.env）
# 默认 WARNING，调试时改为 INFO
```

---

## 8. 扩容策略

| 阶段 | QPS | 方案 |
|------|------|------|
| POC | < 100 | 单机 + uvicorn --workers 4 + API LLM |
| 灰度 | 100-1000 | 单机 GPU + llama.cpp/vLLM + ONNX BERT |
| 生产 | 1000-10000 | 多机水平扩展 + Nginx LB + Redis 共享缓存 |
| 大规模 | 10000+ | vLLM 集群 + Ray 分布式调度 + Milvus 向量库 |

---

## 9. 常见问题

### Q: venv 和 conda 哪个好？

**venv** 够用。Python 自带，轻量，不需要额外安装。生产环境 Docker 里也是 venv。conda 的优势在于管理 CUDA 依赖，如果不用 GPU 就不需要。

### Q: 模型加载很慢怎么办？

首次启动时 BERT + BGE 模型加载需要 5-15 秒。代码里 `api.py` 的 `startup_warmup` 已做了预热。如果还是慢，检查 HuggingFace 缓存是否在 `~/.cache/huggingface/` 下。

### Q: ChromaDB 数据会丢吗？

ChromaDB 数据持久化在 `./data/chroma/` 目录。迁移部署时把这个目录拷贝到新机器即可保留缓存。

### Q: 如何切换 LLM 后端？

改 `.env` 里一行：
```env
# API: deepseek | openai | anthropic
LLM_PROVIDER=deepseek

# 本地: llama.cpp
LLM_PROVIDER=local
LOCAL_LLM_MODEL=./models/qwen2.5-1.5b.gguf
LOCAL_LLM_ENABLED=true
```

不需要改代码。
