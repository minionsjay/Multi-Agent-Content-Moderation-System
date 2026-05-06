# 内容审核系统 · 完整技术总览

> POC v0.5.0 · 2026-05-06

## 1. 系统架构

```
                          ┌──────────────────────┐
                          │     Content Input     │
                          │   POST /moderate      │
                          │   POST /moderate/stream│
                          │   POST /moderate/batch │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Gateway (热路径)     │
                          │                      │
                          │  L0a: 本地 TTLCache   │ ← SHA256, <0.01ms
                          │  L0b: Redis 共享缓存   │ ← 跨Worker, ~0.5ms
                          │  L1a: AC自动机+分词    │ ← O(n), <0.5ms
                          │  L1b: 白名单正则       │ ← 防误杀
                          │  L2:  ChromaDB语义    │ ← HNSW, <5ms
                          │                      │
                          │  命中 → 直接返回(70-80%)│
                          │  未命中 → 升级冷路径     │
                          └──────────┬───────────┘
                                     │ ~20-30%
                          ┌──────────▼───────────┐
                          │  LangGraph (冷路径)    │
                          │                      │
                          │  Image Agent         │
                          │   ├── 图片下载/解码    │
                          │   ├── dHash 感知哈希   │ ← 已知有害DB匹配
                          │   ├── ViT NSFW 分类   │ ← POC跳过
                          │   └── EasyOCR 提取    │ ← ch_sim+en
                          │                      │
                          │  Text Agent          │
                          │   ├── L1: AC自动机    │ ← 如Gateway未扫
                          │   ├── L2: BERT/ONNX  │ ← 多语言XLM-RoBERTa
                          │   └── L3: LLM审核     │ ← DeepSeek/本地
                          │                      │
                          │  Decision Agent      │
                          │   ├── 加权聚合        │ ← Text53%+Image47%
                          │   ├── 零容忍硬覆盖     │ ← politics/violence
                          │   └── 灰度区分流       │ ← [0.3,0.7]→人审
                          │                      │
                          │  Action Agent        │
                          │   ├── 回写三层缓存     │
                          │   ├── 入人审队列      │
                          │   └── 记录事件        │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  在线缓存回写     │  │  人审队列        │  │  事件采集        │
    │  L0a+L0b+L1     │  │  review_queue   │  │  event_collector │
    └─────────────────┘  └────────┬────────┘  └────────┬────────┘
                                  │                    │
                                  ▼                    ▼
                          ┌─────────────────────────────────────┐
                          │        离线反馈飞轮 (旁路)            │
                          │                                     │
                          │  Dataset Builder                    │
                          │   ├── 标注数据集 (silver+gold)        │
                          │   ├── 关键词发现                     │
                          │   ├── 误报模式                      │
                          │   ├── Hard Cases                   │
                          │   └── 准确率报告                    │
                          │                                     │
                          │  Fine-Tune Trigger                 │
                          │   ├── 标签计数 → 触发微调            │
                          │   ├── 漂移检测 → 告警               │
                          │   └── 阈值校准 → 自动调整            │
                          │                                     │
                          │  Train LoRA                        │
                          │   └── BERT LoRA微调                 │
                          └─────────────────────────────────────┘
```

## 2. 技术栈全景

### 2.1 核心框架

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| Agent编排 | **LangGraph** | ≥0.2.0 | DAG状态机，条件分支，并行调度 |
| API服务 | **FastAPI + Uvicorn** | 0.136+ | REST + SSE流式 |
| 配置管理 | **python-dotenv** | ≥1.0.0 | .env加载 |

### 2.2 热路径技术

| 层 | 技术 | 数据结构/算法 | 延迟 | 拦截率 |
|------|------|------|------|------|
| L0a 本地缓存 | `cachetools.TTLCache` | SHA256 + HashTable | <0.01ms | 10-15% |
| L0b Redis | `redis-py` + `hiredis` | SHA256 + Redis GET | ~0.5ms | 5-10% |
| L1a 关键词 | `pyahocorasick` | AC Automaton | <0.5ms | 15-20% |
| L1a 分词 | `jieba` | Trie + 动态规划 | ~200μs | 上下文验证 |
| L1b 白名单 | `re` (标准库) | Regex Alternation | <0.1ms | 防误杀 |
| L2 语义缓存 | **ChromaDB** | HNSW + Cosine | <5ms | 10-15% |
| L2 向量化 | `BGE-small-zh-v1.5` | sentence-transformers, 512d | ~0.2ms | 语义相似 |
| 图片感知哈希 | `dHash` (Pillow) | 64-bit 指纹 | <1ms | 已知有害匹配 |
| 图片URL缓存 | `hashlib.sha256` | 64-bit 截断 | <0.01ms | 精确去重 |

### 2.3 冷路径技术

| 节点 | 步骤 | 技术 | 模型 | 延迟 | 成本 |
|------|------|------|------|------|------|
| Image Agent | NSFW分类 | HuggingFace pipeline | Falconsai/nsfw ViT | ~100ms (GPU) | low |
| Image Agent | OCR提取 | EasyOCR | ch_sim + en | ~300ms (CPU) | zero |
| Text Agent L1 | 关键词 | AC自动机 + jieba | 内置词库 | <0.5ms | zero |
| Text Agent L2 | BERT分类 | HF pipeline / ONNX | XLM-RoBERTa (多语言) | ~50ms | low |
| Text Agent L3 | LLM审核 | AsyncOpenAI | DeepSeek Chat / GPT-4o / Claude | 500-2000ms | **high** |
| Text Agent L3 | LLM本地 | llama.cpp (待激活) | Qwen2.5-1.5B GGUF | 200-500ms | low |
| Decision | 规则引擎 | 纯Python | - | <1ms | zero |
| Action | 缓存写回 | ChromaDB + Redis | - | ~15ms | zero |

### 2.4 离线反馈技术

| 组件 | 技术 | 存储 | 触发 |
|------|------|------|------|
| 事件采集 | JSONL append-only | `data/events.jsonl` | 每条审核 |
| 人审队列 | JSONL + API | `data/review_queue.jsonl` | 灰度区自动入队 |
| 数据集构建 | Python + JSONL | `data/assets/datasets/` | 定时/手动 |
| 微调触发 | 标签计数 + 阈值 | `data/finetune_state.json` | ≥5000条 |
| 漂移检测 | 滚动准确率对比 | 同上 | 7d vs 30d |
| 阈值校准 | 贝叶斯更新 | 同上 | 每次pipeline |
| LoRA微调 | peft + transformers | `data/models/lora_adapter/` | 手动触发 |

## 3. 数据流全景

### 3.1 文本审核（最常见）

```
"你真是个傻逼，什么都不懂"
  │
  ▼
Gateway:
  L0a: SHA256 → miss
  L0b: Redis → miss
  L1a: AC自动机 → 命中"傻逼" → jieba验证 → standalone → conf=1.0
  → 返回: {decision: "block", confidence: 1.0, tier: "L1_keyword"}
  → 延迟: 0.22ms, 成本: $0
```

```
"今天天气真好，适合出去玩"
  │
  ▼
Gateway:
  L0a: miss → L0b: miss → L1a: miss → L2: ChromaDB miss
  → 升级冷路径

LangGraph:
  Text Agent:
    L1: keyword_prefiltered=True → 跳过
    L2: BERT → XLM-RoBERTa → label=safe, conf=0.92
    → conf<0.95 → 继续L3
    L3: DeepSeek → "日常天气描述" → label=safe, conf=0.98
  Decision:
    加权聚合 → label=safe, conf=0.98 → pass
  Action:
    写回L0a/L0b/L1 + 记录事件
  → 延迟: ~1200ms, 成本: ~$0.002
```

### 3.2 图片审核（含已知有害检测）

```
用户上传图片 (base64)
  │
  ▼
Gateway:
  I1: dHash → b5d453d5a46d9b2b
  → check_known() → 匹配已知有害库! → category=test_illegal
  → 返回: {decision: "block", tier: "L0_phash"}
  → 延迟: 0.93ms, 成本: $0
```

### 3.3 灰度区 → 人审 → 反馈

```
"有些人真的就是欠骂，但我也不想说太难听"
  │
  ▼
LangGraph → LLM: label=toxic, conf=0.55
  │
  ▼
Decision: conf ∈ [0.3, 0.7] → review
  │
  ▼
Action:
  ├── review_queue.enqueue() → 入人审队列
  └── event_collector.record() → 记录事件

  ... 审核员看到这条 ...

POST /review/resolve {review_id: "xxx", human_decision: "block", reason: "隐晦辱骂"}
  │
  ▼
  ├── review_queue.resolve() → 标记已审
  ├── 回写L0a/L0b/L1缓存 → 相同内容不再进人审
  └── 计入反馈标签计数 → 积累够5000条触发微调
```

## 4. 关键性能指标

### 4.1 实测数据

| 路径 | 延迟 | QPS (单核) | 成本/条 | 占比 |
|------|------|------|------|------|
| 热路径 L0缓存 | <0.01ms | 739K | $0 | 10-15% |
| 热路径 关键词 | 0.22ms | 300K+ | $0 | 15-20% |
| 热路径 语义缓存 | 1.7ms | 956 | $0 | 10-15% |
| 热路径 图片dHash | 1.0ms | 984 | $0 | 5-10% |
| 冷路径 BERT L2 | 50ms (HF) | 20 | $0.0001 | 30-40% |
| 冷路径 LLM L3 | 1200ms | ~0.8 | **$0.002** | 5-15% |
| 人审 | 分钟级 | - | **$0.05** | 1-3% |

### 4.2 瓶颈

```
热路径唯一瓶颈: BGE Embedding (200 QPS/核)
  → 已实装 Embedding缓存 (91%命中) → 有效QPS ~1800
  → 待实装 ONNX导出 (3x加速)

冷路径唯一瓶颈: LLM API (1200ms, $0.002/条)
  → L2 BERT短路 (已修复) → 英文33%不走L3
  → 多语言BERT (已切换) → 中文不再100%走L3
  → 本地LLM路径 (已实装, 待激活) → 延迟降4x, 成本降为$0

隐藏瓶颈: ChromaDB写入 (36 QPS)
  → 需改异步批量写入
```

## 5. 模块清单

### 5.1 在线系统 (src/)

```
src/
├── api.py                 # FastAPI入口 (/moderate, /stream, /batch, /review)
├── gateway.py             # 热路径预过滤器 (5层递进)
├── graph.py               # LangGraph DAG (4节点)
├── state.py               # ModerationState TypedDict
├── config.py              # 全局配置
│
├── agents/
│   ├── image_agent.py     # 图片: 下载→dHash→NSFW→OCR
│   ├── text_agent.py      # 文本: L1关键词→L2BERT→L3LLM
│   ├── decision.py        # 加权聚合+零容忍+灰度区
│   └── action.py          # 缓存写回+入队+事件记录
│
├── skills/
│   ├── keyword_filter.py  # AC自动机+jieba+白名单
│   ├── bert_classify.py   # HF pipeline BERT
│   ├── bert_onnx.py       # ONNX Runtime BERT
│   ├── llm_audit.py       # API LLM (DeepSeek/OpenAI/Anthropic)
│   ├── llm_local.py       # 本地LLM (llama.cpp, 待激活)
│   ├── embedder.py        # BGE-small-zh 向量化 (含缓存)
│   ├── memory_cache.py    # L0a 本地TTLCache
│   ├── redis_cache.py     # L0b Redis共享缓存
│   ├── vector_cache.py    # L1 ChromaDB 语义缓存
│   ├── image_phash.py     # dHash感知哈希+已知有害DB
│   ├── image_nsfw.py      # ViT NSFW分类器
│   ├── image_ocr.py       # EasyOCR 文字提取
│   └── review_queue.py    # 人审队列 (JSONL)
│
└── feedback/              # 离线反馈飞轮
    ├── event_collector.py # 事件采集
    ├── dataset_builder.py # 数据资产构建 (5种产出)
    ├── finetune_trigger.py # 微调触发+漂移检测+阈值校准
    ├── train_lora.py      # LoRA微调脚本
    └── pipeline.py        # 离线管线入口
```

### 5.2 测试与研究 (research/)

```
research/
├── hot_path_bench/        # 热路径逐层benchmark (11项)
│   ├── 01_l0_memory_cache/
│   ├── 02_ac_automaton/
│   ├── 03_jieba_context/
│   ├── 04_chromadb_cache/
│   ├── 05_bge_embedding/
│   ├── 06_gateway_full/
│   ├── 07_multi_service_load/
│   ├── 08_image_dhash/    # 含测试图片
│   ├── 09_image_url_cache/
│   ├── 10_image_phash_cache/
│   └── 11_image_gateway_full/
│
├── cold_path_bench/       # 冷路径逐层benchmark (6项)
│   ├── 01_bert_onnx_vs_hf/
│   ├── 02_llm_latency/
│   ├── 03_nsfw_vit/
│   ├── 04_easyocr/
│   ├── 05_decision_rules/
│   └── 06_cold_path_full/
│
├── 01_embedding_cache/    # 优化方案: Embedding缓存
├── 02_multi_worker/       # 优化方案: 多Worker
├── 03_bge_batch/          # 优化方案: 批量推理
├── 04_onnx_export/        # 优化方案: ONNX导出
├── 05_redis_cache/        # 优化方案: Redis共享
├── 06_horizontal_scale/   # 优化方案: 水平扩展
└── LOCAL_LLM_GUIDE.md     # 本地LLM接入指南
```

## 6. 已修复的问题

| 日期 | 问题 | 修复 |
|------|------|------|
| 05-05 | Triage死代码 | 删除triage.py |
| 05-05 | 关键词重复扫描 | keyword_prefiltered标记 |
| 05-05 | Gateway冷路径trace丢失 | 改返回格式为始终返回dict |
| 05-05 | 图片一刀切升级 | 增加dHash+URL缓存 |
| 05-05 | Gateway膨胀 | section注释分离 |
| 05-06 | Decision灰度区以下不处理 | 新增<0.3分支→pass |
| 05-06 | safe被送入灰度区 | safe检查移到灰度区之前 |
| 05-06 | toxic关键词被放行 | Path2覆盖所有高置信关键词 |
| 05-06 | BERT短路失效(ONNX优先) | ONNX低置信回退HF |
| 05-06 | 中文全走L3 | 切换多语言XLM-RoBERTa |
| 05-06 | 系统提示词弱 | Few-shot+置信度锚定+严重度排序 |
| 05-06 | 加权聚合缺失 | 实现Text53%+Image47%评分 |
| 05-06 | 人审断头路 | 完整人审队列+API+回写 |

## 7. 待完成 (Phase 2)

| 项目 | 说明 |
|------|------|
| NSFW模型下载 | `Falconsai/nsfw_image_detection` ~350MB |
| 本地LLM激活 | `pip install llama-cpp-python` + 下载GGUF模型 |
| ONNX BGE导出 | 3x embedding加速 |
| ChromaDB异步写入 | 解决36 QPS写入瓶颈 |
| Multimodal Agent | 图文矛盾联合审核 |
| Redis Server部署 | 跨Worker缓存共享 |
| 水平扩展 | Nginx + K8s HPA |
