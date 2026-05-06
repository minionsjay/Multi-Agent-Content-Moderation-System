# Multi-Agent Content Moderation System

基于 LangGraph 的 Multi-Agent 内容审核系统 POC。通过**热路径预过滤 + 冷路径 AI 推理 + 离线反馈飞轮**三层架构，实现高吞吐、低成本、持续优化的自动化内容审核。

## 快速开始

```bash
# 1. 环境配置
cp .env.example .env
# 编辑 .env，至少填入 DEEPSEEK_API_KEY

# 2. 安装依赖
pip install -r requirements.txt

# 3. 环境检查
python check_env.py

# 4. 启动服务
python -m src.api
# 打开 http://localhost:8000
```

## 系统架构

```
请求 → Gateway (热路径, <5ms, $0)
          ├── L0a 本地缓存 (SHA256, <0.01ms)
          ├── L0b Redis 共享缓存 (~0.5ms)
          ├── L1a AC自动机关键词 (<0.5ms)
          ├── L1b 白名单防误杀
          ├── I1  dHash 感知哈希 (<1ms)
          └── L2  ChromaDB 语义缓存 (<5ms)
                │
                │ 命中 → 直接返回 (70-80%)
                │ 未命中 ↓
                
       LangGraph 冷路径 (50-2000ms, $0-0.002)
          ├── Image Agent (dHash → NSFW ViT → EasyOCR)
          ├── Text Agent  (L1关键词 → L2 BERT → L3 LLM)
          ├── Decision     (加权聚合 + 零容忍 + 灰度区)
          └── Action       (缓存写回 + 人审入队 + 事件记录)
                │
                ▼
       离线反馈飞轮 (定时运行)
          ├── 标注数据集构建 (silver + gold)
          ├── 关键词发现
          ├── 误报模式分析
          ├── 准确率报告 + 漂移检测
          └── LoRA 微调触发
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/moderate` | POST | 单条审核 |
| `/moderate/stream` | POST | 流式审核 (SSE 实时追踪) |
| `/moderate/batch` | POST | 批量审核 (上传 CSV/JSONL) |
| `/gateway/stats` | GET | Gateway 热路径统计 |
| `/review/pending` | GET | 待人审列表 |
| `/review/resolve` | POST | 提交人审判定 |
| `/health` | GET | 健康检查 |

```bash
# 单条审核
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text":"今天天气真好，适合出去玩"}'

# 批量审核
curl -X POST http://localhost:8000/moderate/batch \
  -F "file=@data/bench_100.jsonl"
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 编排 | LangGraph StateGraph |
| API 框架 | FastAPI + Uvicorn |
| 关键词匹配 | AC 自动机 (pyahocorasick) + jieba 分词 |
| 文本分类 | XLM-RoBERTa (多语言) + ONNX Runtime |
| LLM 审核 | DeepSeek / GPT-4o / Claude API + 本地 llama.cpp 路径 |
| 文本向量化 | BGE-small-zh-v1.5 (512-dim) |
| 语义缓存 | ChromaDB (HNSW + Cosine Similarity) |
| 图片感知哈希 | dHash (64-bit 指纹) |
| 图片 OCR | EasyOCR (ch_sim + en) |
| 图片 NSFW | Falconsai/nsfw_image_detection (ViT) |
| 共享缓存 | Redis + TTLCache 双层 |
| 人审队列 | JSONL (POC) → PostgreSQL (生产) |
| 微调 | LoRA (peft + transformers) |

## 项目结构

```
src/
├── api.py              # FastAPI 入口
├── gateway.py          # 热路径预过滤器
├── graph.py            # LangGraph DAG
├── state.py            # 状态定义
├── config.py           # 全局配置
├── agents/             # 4 个 Agent
│   ├── image_agent.py  # 图片: dHash→NSFW→OCR
│   ├── text_agent.py   # 文本: L1→L2→L3 漏斗
│   ├── decision.py     # 加权聚合+零容忍+灰度区
│   └── action.py       # 缓存写回+人审+事件
├── skills/             # 13 个 Skill
│   ├── keyword_filter.py  # AC自动机+分词+白名单
│   ├── bert_classify.py   # HF pipeline BERT
│   ├── bert_onnx.py       # ONNX Runtime BERT
│   ├── llm_audit.py       # API LLM
│   ├── llm_local.py       # 本地 LLM (待激活)
│   ├── embedder.py        # BGE 向量化(含缓存)
│   ├── memory_cache.py    # L0a 本地缓存
│   ├── redis_cache.py     # L0b Redis 缓存
│   ├── vector_cache.py    # L1 ChromaDB 缓存
│   ├── image_phash.py     # dHash 感知哈希
│   ├── image_nsfw.py      # ViT NSFW
│   ├── image_ocr.py       # EasyOCR
│   └── review_queue.py    # 人审队列
└── feedback/           # 离线反馈飞轮
    ├── event_collector.py    # 事件采集
    ├── dataset_builder.py    # 数据资产构建
    ├── finetune_trigger.py   # 微调触发+漂移检测
    ├── train_lora.py         # LoRA 微调
    └── pipeline.py           # 离线管线
```

## Benchmark

```bash
# 热路径逐层测试 (11 项)
python research/hot_path_bench/01_l0_memory_cache/bench.py
python research/hot_path_bench/06_gateway_full/bench.py
python research/hot_path_bench/07_multi_service_load/load_test.py

# 冷路径逐层测试 (6 项)
python research/cold_path_bench/01_bert_onnx_vs_hf/bench.py
python research/cold_path_bench/06_cold_path_full/bench.py

# 离线反馈管线
python -m src.feedback.pipeline
```

## 离线反馈飞轮

```bash
# 日常数据资产构建
python -m src.feedback.pipeline

# 检查微调就绪状态
python -m src.feedback.train_lora --dry-run

# 触发 LoRA 微调 (需标注量 ≥5000)
python -m src.feedback.train_lora
```

## 文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构设计文档 |
| [ANALYSIS.md](ANALYSIS.md) | 逐层技术分析 + Bug 修复记录 |
| [OVERVIEW.md](OVERVIEW.md) | 技术总览 + 模块清单 |
| [OFFLINE_DEPLOY.md](OFFLINE_DEPLOY.md) | 离线部署指南 |
| [PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md) | 内容安全风控 Agent Prototype 的 Spec 执行入口 |
| [SDD_PROTOTYPE_DEVELOPMENT_PLAN.md](SDD_PROTOTYPE_DEVELOPMENT_PLAN.md) | Prototype 的 SDD 背景与详细规格说明，非直接执行入口 |
| [AGENTS.md](AGENTS.md) | Coding Agent 项目规则 |
| [research/LOCAL_LLM_GUIDE.md](research/LOCAL_LLM_GUIDE.md) | 本地大模型接入指南 |
| [research/README.md](research/README.md) | 优化方案索引 |

## License

MIT
