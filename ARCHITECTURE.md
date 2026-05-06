# POC 内容审核系统 · 架构与实现文档

> 版本 0.4.0 | 2026-05-05

## 1. 项目概况

基于 Multi-Agent + LangGraph 的内容审核 POC 系统，核心理念是用**三层漏斗**拦截大部分流量，让昂贵的 LLM 只处理 5-20% 的疑难内容。

当前已实现文本 + 图片审核的完整链路，包括热路径网关、四节点 LangGraph 推理引擎、流式 API 和批量评测工具。

---

## 2. 整体架构：双路径设计

```
请求 → Gateway（热路径）──命中──→ 直接返回（< 5ms）
              │
              └──未命中（≈5-20%）→ LangGraph（冷路径）→ 返回
```

### 2.1 热路径：Gateway 预过滤器

Gateway 在进入 LangGraph 之前做三级快速筛查，目标是拦截 80%+ 的简单流量：

```
Gateway.check()
  ├── L0 内存缓存（SHA256 精确匹配，< 0.01ms）
  │     └── 命中 → 直接返回（完全相同的文本之前已审核过）
  ├── AC 自动机关键词匹配（< 0.5ms）
  │     ├── 白名单命中（已知误杀短语如"操场上"）→ 放行
  │     ├── 独立关键词（conf=1.0）→ 直接 Block
  │     └── 嵌入词（conf=0.6，如"口交"出现在"接口交换"中）→ 升级到冷路径
  └── ChromaDB 语义缓存（< 5ms）
        └── 命中（cosine similarity ≥ 0.95）→ 复用历史判决
```

**Gateway 代码**：`poc/src/gateway.py`

### 2.2 冷路径：LangGraph 推理引擎

只处理 Gateway 无法判断的内容，包含 4 个节点：

```
image_agent → text_agent → decision → action
```

**图定义**：`poc/src/graph.py`
**状态定义**：`poc/src/state.py`

#### 路由逻辑

| 内容类型 | 执行路径 |
|----------|----------|
| 纯文本 | text_agent → decision → action |
| 纯图片 | image_agent → decision → action |
| 图文混合 | image_agent → text_agent → decision → action |

混合内容先跑 Image Agent（含 OCR 提取文字），将 OCR 结果追加到文本后再进入 Text Agent，避免了并行时序问题。

---

## 3. Agent 详细设计

### 3.1 Text Agent（文本审核三层漏斗）

文件：`poc/src/agents/text_agent.py`

```
输入文本
  │
  ├── L1 关键词（AC 自动机 + jieba 分词）
  │     └── conf > 0.99 → 直接返回（成本: zero）
  │
  ├── L2 BERT 分类（unitary/toxic-bert）
  │     ├── ONNX Runtime（优先，CPU 上 2-3x 加速）
  │     ├── HuggingFace transformers pipeline（回退）
  │     ├── conf ≥ 0.95 → 跳过 LLM，直接返回（成本: low）
  │     └── 中文文本 + 英文 BERT → 自动跳过 L2，升级到 L3
  │
  └── L3 LLM 深度审核（DeepSeek/OpenAI/Anthropic）
        ├── 附带 BERT 预判结果作为上下文
        ├── 8 秒超时 → 回退到 BERT 结果
        └── 返回 label + confidence + reason（成本: high）
```

**成本优化关键**：BERT 高置信度（≥ 0.95）直接决策，跳过 LLM。只有 5-15% 的流量最终到达 L3。

### 3.2 Image Agent（图片审核）

文件：`poc/src/agents/image_agent.py`

```
输入图片（URL 或 Base64）
  │
  ├── 下载/解码图片（10s 超时）
  │
  ├── NSFW 分类
  │     ├── POC 阶段：跳过模型下载，基于尺寸/格式做基础校验
  │     └── 生产阶段：Falconsai/nsfw_image_detection（ViT, ~350MB）
  │
  └── OCR 文字提取（EasyOCR, ch_sim + en）
        └── 提取到的文字追加到原始文本 → 送入 Text Agent 审核
```

### 3.3 Decision Agent（汇总裁决）

文件：`poc/src/agents/decision.py`

处理三条路径：
1. **缓存命中路径**：直接复用缓存中的 decision
2. **零容忍路径**：关键词命中 politics/violence 类别 → 直接 Block，跳过所有模型
3. **聚合路径**：综合 text_result + image_result

```
聚合逻辑：
  ├── 仅图片（无文本）
  │     ├── NSFW + conf > 0.5 → Block
  │     └── 正常 → Pass
  ├── 文本 result + 图片 NSFW → label = unsafe
  ├── 零容忍类别（politics, violence）→ Block
  ├── 灰度区（confidence ∈ [0.3, 0.7]）→ Review（人工复核）
  └── 正常 → Pass/Block（根据 label）
```

### 3.4 Action Agent（执行）

文件：`poc/src/agents/action.py`

1. 写回 L0 内存缓存（SHA256 精确匹配，TTL 1 小时）
2. 写回 L1 ChromaDB 语义缓存

POC 阶段仅返回 JSON 决策，不接入实际执法系统。

---

## 4. Skills 技术实现

### 4.1 keyword_filter（关键词匹配）

文件：`poc/src/skills/keyword_filter.py`

| 特性 | 实现 |
|------|------|
| 匹配算法 | AC 自动机（pyahocorasick），O(n) 线性扫描 |
| 上下文验证 | jieba 分词判断关键词是独立词还是嵌入词 |
| 白名单 | 正则匹配已知误杀短语（如"操场上"→ 操场 + 上，并非脏话） |
| 置信度分级 | standalone = 1.0，embedded = 0.6，whitelist suppressed = 0.0 |
| 内置词库 | 5 个类别，约 50 个关键词（toxic/politics/violence/porn/gambling） |

**为什么不用正则**：AC 自动机在百万级词库下匹配时间仍为 O(n)，正则的 `|` 交替在词库大时会指数级退化。

### 4.2 bert_classify / bert_onnx（BERT 文本分类）

文件：`poc/src/skills/bert_classify.py`、`poc/src/skills/bert_onnx.py`

| 特性 | 实现 |
|------|------|
| 模型 | unitary/toxic-bert（微调 BERT-base, 6 个毒性子标签） |
| transformers 后端 | HuggingFace text-classification pipeline, 60s 加载超时 |
| ONNX 后端 | onnxruntime CPU，图优化全开，预计 2-3x 加速 |
| 标签映射 | toxic/severe_toxic/obscene/threat/insult/identity_hate → unsafe |
| 短路阈值 | conf ≥ 0.95 → 跳过 LLM；conf < 0.4 → 强制走 LLM |
| 中文检测 | CJK 字符 > 30% 且模型为 toxic-bert → 跳过 BERT，直接升级到 LLM |

**ONNX 回退链**：`onnx_models/model.onnx` 存在 → ONNX 推理；不存在 → transformers pipeline

### 4.3 llm_audit（LLM 深度审核）

文件：`poc/src/skills/llm_audit.py`

| 特性 | 实现 |
|------|------|
| 主后端 | DeepSeek Chat（deepseek-chat） |
| 备选后端 | OpenAI（gpt-4o-mini）、Anthropic（claude-3-5-haiku-latest） |
| 请求切换 | API 请求中带 `llm_provider` / `llm_model` 参数即可动态切换 |
| 响应格式 | JSON mode，temperature=0，max_tokens=256 |
| 系统提示词 | 中文，要求输出 label、confidence、reason（中文解释） |

**请求格式示例**：
```json
{
  "text": "这个产品完全是垃圾，骗人的",
  "context": {
    "bert_label": "unsafe",
    "bert_confidence": 0.87,
    "user_id": "user_123"
  }
}
```

### 4.4 embedder（文本向量化）

文件：`poc/src/skills/embedder.py`

| 特性 | 实现 |
|------|------|
| 模型 | BAAI/bge-small-zh-v1.5 |
| 维度 | 512 |
| 大小 | ~95 MB |
| 语言优化 | 中文 + 英文（BGE 系列对中文语义理解优秀） |
| 归一化 | normalize_embeddings=True，输出余弦相似度就绪 |

### 4.5 vector_cache（语义缓存）

文件：`poc/src/skills/vector_cache.py`

| 特性 | 实现 |
|------|------|
| 数据库 | ChromaDB PersistentClient（嵌入式，无需额外部署） |
| 相似度阈值 | cosine similarity ≥ 0.95 命中 |
| 存储内容 | embedding + 原文摘要 + decision + confidence + reason |
| 失效机制 | `cache_invalidate()` 按文本 hash 删除 |

### 4.6 memory_cache（内存精确缓存）

文件：`poc/src/skills/memory_cache.py`

| 特性 | 实现 |
|------|------|
| 存储 | Python TTLCache（cachetools） |
| Key | SHA256(text) |
| TTL | 1 小时 |
| 容量 | 最多 100,000 条（≈50MB 内存） |
| 速度 | < 0.01ms（纯内存 dict 查找） |

**为什么需要 L0 + L1 两层缓存？** L0 内存缓存仅对完全相同的文本生效（适合重复提交、同一条内容多次审核）；L1 ChromaDB 通过语义相似度匹配同类内容（适合"换皮"违规文本——相同意思但换了几个词）。

### 4.7 image_nsfw（NSFW 检测）

文件：`poc/src/skills/image_nsfw.py`

| 特性 | 实现 |
|------|------|
| 模型 | Falconsai/nsfw_image_detection（ViT, ~350MB） |
| POC 模式 | skip_model=True → 仅做图片尺寸/格式校验 |
| 生产模式 | GPU/CPU 推理，NSFW 概率 > 0.5 → 标记 unsafe |

### 4.8 image_ocr（图片文字提取）

文件：`poc/src/skills/image_ocr.py`

| 特性 | 实现 |
|------|------|
| 引擎 | EasyOCR |
| 语言 | ch_sim（简体中文）+ en（英文） |
| 输出 | 合并文本 + 平均置信度 + 逐块坐标 |

---

## 5. API 设计

文件：`poc/src/api.py`（FastAPI）

### 5.1 接口列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端测试界面 |
| `/health` | GET | 健康检查 |
| `/gateway/stats` | GET | Gateway 命中率统计 |
| `/moderate` | POST | 核心审核接口 |
| `/moderate/stream` | POST | 流式审核（SSE） |
| `/moderate/batch` | POST | 批量审核（文件上传） |

### 5.2 请求格式

```json
{
  "content_id": "req_123",
  "text": "这个产品完全是垃圾",
  "image_url": "",
  "image_base64": "",
  "bert_model": "",
  "llm_provider": "deepseek",
  "llm_model": "deepseek-chat",
  "user_id": "anonymous",
  "source": "api"
}
```

### 5.3 响应格式

```json
{
  "content_id": "req_123",
  "decision": "block",
  "confidence": 0.94,
  "reason": "Text contains toxic language targeting a product with insulting terms",
  "tier": "L2_bert",
  "latency_ms": 145.3,
  "traces": [
    {
      "node": "text_agent",
      "step": "L1_keyword",
      "model": "AC自动机",
      "input": "这个产品完全是垃圾",
      "output": {"label": "toxic", "confidence": 1.0},
      "latency_ms": 0.35,
      "cost": "zero"
    }
  ],
  "path": "cold"
}
```

### 5.4 流式响应

`/moderate/stream` 使用 Server-Sent Events 实时推送每个节点完成事件：

```
data: {"event":"node_complete","node":"gateway","node_index":1,...}
data: {"event":"node_complete","node":"image_agent","node_index":2,...}
data: {"event":"node_complete","node":"text_agent","node_index":3,...}
data: {"event":"node_complete","node":"decision","node_index":4,...}
data: {"event":"done","result":{...}}
```

### 5.5 批量审核

`/moderate/batch` 接受 JSONL 或 CSV 文件上传，并发处理（Semaphore(1) 控制 LLM 并发避免速率限制），返回汇总统计：

```json
{
  "total": 100,
  "passed": 52,
  "blocked": 45,
  "reviewed": 3,
  "hot_path": 38,
  "cold_path": 62,
  "hot_path_rate": 0.38,
  "avg_latency_ms": 245.3,
  "tier_distribution": {"cache": 10, "L1_keyword": 28, "L2_bert": 37, "L3_llm": 25},
  "llm_call_rate": 0.25
}
```

---

## 6. 技术栈总览

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| Agent 编排 | LangGraph | ≥0.2.0 | 状态机 DAG，条件分支 |
| API 框架 | FastAPI + Uvicorn | 0.136.1 + 0.30+ | REST + SSE 流式 |
| 向量数据库 | ChromaDB | 1.5.8 | 语义缓存（嵌入式） |
| 内存缓存 | cachetools (TTLCache) | - | L0 精确匹配缓存 |
| 文本分类 | unitary/toxic-bert | - | L2 BERT 毒性分类 |
| 文本分类加速 | ONNX Runtime | - | CPU 推理 2-3x 加速 |
| LLM 审核 | DeepSeek Chat / GPT-4o-mini / Claude Haiku | - | L3 深度审核 |
| 文本向量化 | BGE-small-zh-v1.5 (sentence-transformers) | 5.4.1 | 语义缓存 512 维 |
| 关键词匹配 | pyahocorasick | ≥2.1.0 | AC 自动机 O(n) 匹配 |
| 分词 | jieba | - | 关键词上下文验证 |
| 图片 NSFW | Falconsai/nsfw_image_detection (ViT) | - | 色情/暴力检测 |
| 图片 OCR | EasyOCR | - | ch_sim + en 文字提取 |
| 图片处理 | Pillow (PIL) | - | 图片解码/缩放 |
| 深度学习 | PyTorch + Transformers | 2.4+ + 5.7.0 | BERT + ViT 推理 |
| 数据校验 | Pydantic | ≥2.0 | API 请求 schema |
| 环境配置 | python-dotenv | ≥1.0.0 | .env 加载 |
| HTTP 下载 | httpx | ≥0.27.0 | 图片 URL 下载 |

---

## 7. 数据审核流程详解

### 7.1 纯文本审核（最常见路径）

以用户提交 `"你真是个傻逼，什么都不懂"` 为例：

```
1. Gateway.check()
   ├── L0 内存缓存：未命中（首次提交）
   ├── AC 自动机：匹配到 "傻逼"（toxic 类别）
   ├── jieba 分词验证："你/真是/个/傻逼/，/什么/都/不懂"
   │     → "傻逼" 作为独立 token 出现 → conf=1.0 → 直接 Block
   └── 返回：decision="block", confidence=1.0, tier="L1_keyword"
       latency ≈ 0.5ms, cost = $0
```

以用户提交 `"这款产品可能存在一些夸大宣传的成分，消费者需要注意甄别"` 为例：

```
1. Gateway.check()
   ├── L0 内存缓存：未命中
   ├── AC 自动机：无任何关键词匹配
   ├── ChromaDB 语义缓存：未命中（首次遇到类似表述）
   └── 返回：None（升级到冷路径）

2. LangGraph: text_agent
   ├── L1 关键词：未命中
   ├── L2 BERT：text_classification → [{"label":"insult","score":0.12}, ...]
   │     → max_toxic_score=0.28 < 0.5 → label="safe", confidence=0.72
   │     → confidence 0.72 < 0.95 → 不足以跳过 LLM
   └── L3 DeepSeek：返回 {"label":"safe","confidence":0.85,"reason":"...内容是消费者提醒..."}

3. Decision：label=safe, confidence=0.85, decision="pass"

4. Action：写回 L0 + L1 缓存
```

### 7.2 图文混合审核

以用户上传一张表情包（图片上写有"你完了"，图片本身正常）为例：

```
1. Gateway.check()
   ├── 检测到 image_url → 直接升级到冷路径
   └── 返回：None

2. LangGraph: image_agent
   ├── 下载图片
   ├── NSFW 分类 → label="normal", confidence=1.0
   └── OCR 提取 → "你完了"
        → 追加到 state.text: "[OCR: 你完了]"

3. LangGraph: text_agent（路由判断 text 非空 → 执行）
   ├── L1 关键词："完了" 不在违禁词库 → 未命中
   ├── L2 BERT：low toxicity → label="safe", confidence=0.78
   └── L3 DeepSeek：结合上下文 "你完了" 是威胁还是玩笑？
        → 返回 label="unsafe", confidence=0.72（判定为威胁）

4. Decision：confidence=0.72 > 0.7 → decision="block"

5. Action：写回缓存
```

### 7.3 各层流量分布（预期）

基于 CLAUDE.md 设计目标：

| 层 | 流量占比 | 延迟 | 成本/条 |
|------|----------|------|------|
| L0 内存缓存 | 10-15% | < 0.01ms | $0 |
| L1 关键词直接拦截 | 15-20% | < 0.5ms | $0 |
| L1 语义缓存 | 10-15% | < 5ms | $0 |
| L2 BERT 高置信 | 30-40% | < 100ms | ~$0.0001 |
| L3 LLM 深度审核 | 5-15% | < 1s | ~$0.002 |
| 人工复核 | 1-3% | 分钟级 | ~$0.05 |

**热路径合计（无需 GPU/LLM）**：35-50%

---

## 8. 配置说明

文件：`poc/src/config.py`、`.env`

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | - | 主 LLM API 密钥 |
| `DEEPSEEK_BASE_URL` | https://api.deepseek.com | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | deepseek-chat | 使用的 DeepSeek 模型 |
| `OPENAI_API_KEY` | - | 备选 LLM（可选） |
| `ANTHROPIC_API_KEY` | - | 备选 LLM（可选） |
| `LLM_PROVIDER` | deepseek | 默认 LLM 后端 |
| `BERT_MODEL` | unitary/toxic-bert | BERT 模型路径或 HuggingFace ID |
| `BERT_ENABLED` | true | 是否启用 BERT L2 |
| `EMBED_MODEL` | BAAI/bge-small-zh-v1.5 | 向量化模型 |
| `CHROMA_PERSIST_DIR` | ./data/chroma | ChromaDB 持久化目录 |
| `KEYWORD_DICT` | ./data/keywords.json | 自定义关键词词典 |

### 关键阈值

| 参数 | 值 | 说明 |
|------|------|------|
| `BERT_HIGH_CONFIDENCE` | 0.95 | BERT 高于此值跳过 LLM |
| `BERT_LOW_CONFIDENCE` | 0.4 | BERT 低于此值强制走 LLM（不可靠） |
| `CACHE_SIMILARITY_THRESHOLD` | 0.95 | ChromaDB 语义相似度命中阈值 |
| `GREY_ZONE_LOW` | 0.3 | 灰度区下界（conf 低于此值直接放行） |
| `GREY_ZONE_HIGH` | 0.7 | 灰度区上界（conf 超过此值直接决策） |
| `ZERO_TOLERANCE` | politics, violence | 零容忍类别，模型分无效 |

---

## 9. 项目结构

```
poc/
├── src/
│   ├── api.py              # FastAPI 入口（/moderate, /stream, /batch）
│   ├── gateway.py           # 热路径预过滤器
│   ├── graph.py             # LangGraph DAG 定义（4 节点）
│   ├── state.py             # ModerationState TypedDict
│   ├── config.py            # 全局配置 + 环境变量
│   ├── agents/
│   │   ├── triage.py        # 路由（目前被 gateway 替代）
│   │   ├── text_agent.py    # 文本三层漏斗
│   │   ├── image_agent.py   # 图片 NSFW + OCR
│   │   ├── decision.py      # 汇总裁决
│   │   └── action.py        # 执行 + 缓存写回
│   └── skills/
│       ├── keyword_filter.py # AC 自动机 + jieba 上下文验证
│       ├── bert_classify.py  # HuggingFace BERT 分类
│       ├── bert_onnx.py      # ONNX Runtime BERT 推理
│       ├── llm_audit.py      # DeepSeek/OpenAI/Anthropic 审核
│       ├── embedder.py       # BGE-small-zh 向量化
│       ├── vector_cache.py   # ChromaDB 语义缓存
│       ├── memory_cache.py   # L0 内存精确缓存
│       ├── image_nsfw.py     # ViT NSFW 检测器
│       └── image_ocr.py      # EasyOCR 文字提取
├── eval/
│   ├── benchmark.py          # 评测脚本（准确率、F1、流量分布）
│   └── dataset.py            # 数据集加载 + 合成数据生成
├── data/
│   ├── bench_100.jsonl       # 100 条标注测试数据
│   ├── keywords.json         # 自定义关键词词典
│   └── chroma/               # ChromaDB 持久化文件
├── onnx_models/
│   └── vocab.txt             # ONNX tokenizer 词表
├── static/
│   └── index.html            # 前端测试界面
├── tests/
│   └── test_e2e.py           # 端到端测试
├── check_env.py              # 环境检查脚本
└── requirements.txt          # Python 依赖
```

---

## 10. 与 CLAUDE.md 设计文档的差异

POC 实际实现相比设计文档有以下调整：

| 方面 | CLAUDE.md 设计 | 实际实现 |
|------|---------------|----------|
| POC 范围 | 仅文本审核 | 文本 + 图片（含 Gateway） |
| Triage | 作为 LangGraph 节点 | 已被 Gateway 替代（热路径外置） |
| LLM 后端 | GPT-4o-mini / Claude Haiku | DeepSeek 为主，OpenAI/Anthropic 备选 |
| Embedding | text-embedding-3-small（需 API） | BGE-small-zh-v1.5（本地 512 维） |
| BERT 加速 | 无 | ONNX Runtime（experimental） |
| 关键词匹配 | Bloom Filter + 正则 | AC 自动机 + jieba 分词上下文验证 |
| 缓存分层 | 一层 ChromaDB | L0 内存缓存 + L1 ChromaDB |
| API | 单条 REST | 单条 + 流式 + 批量文件上传 |
| Feedback Agent | 无 | 无（Phase 2 计划） |
| Multimodal Agent | 无 | 无（Phase 2 计划） |
| 分布式调度 | 无 | 无（Phase 2 计划） |

---

## 11. 启动方式

```bash
# 1. 环境检查
cd poc
python check_env.py

# 2. 启动 API 服务
python -m src.api

# 3. 单条审核测试
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text":"你真是个傻逼","user_id":"test"}'

# 4. 批量评测
python -m eval.benchmark --dataset data/bench_100.jsonl

# 5. 查看 Gateway 统计
curl http://localhost:8000/gateway/stats
```

---

## 12. 后续规划

按 CLAUDE.md 路线图，后续阶段需完成：

- **Phase 2**：Multimodal Agent（图文矛盾检测）、Feedback Agent（学习闭环）、Ray 分布式调度、Prometheus + Grafana 监控、灰度发布
- **Phase 3**：模型微调（BERT 吸收人工标注）、Qwen2.5 私有化部署、异步批处理、压力测试至 100K QPS、全量切换
- **Phase 4**：新类别检测（诈骗、网络暴力）、多语言支持、视频审核、对抗样本防御
