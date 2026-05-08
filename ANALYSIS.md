# 架构逐层分析

> 从热/冷双路径开始，逐模块审查设计决策、发现问题、记录结论

---

## 1. 热路径 vs 冷路径：为什么要分？

### 1.1 当前实现

```
请求 → api.py
         │
         ├── gateway.check()  ← 热路径（LangGraph 之外）
         │     ├── L0 内存缓存命中 → 直接返回
         │     ├── 关键词命中(conf=1.0) → 直接返回
         │     ├── 白名单命中 → 直接返回
         │     ├── ChromaDB 语义缓存命中 → 直接返回
         │     └── 全部未命中 → 返回 None（升级）
         │
         └── graph.ainvoke()   ← 冷路径（LangGraph）
               ├── image_agent
               ├── text_agent
               ├── decision
               └── action
```

关键事实：

- **Gateway 完全在 LangGraph 之外**，是一个纯 Python 类，不参与状态机
- **Triage agent（triage.py）存在于代码库但未被使用**，graph.py 的入口是 `image_agent`，不是 `triage`
- API 层（api.py 第 82 行）先调 Gateway，Gateway 返回 None 才进 LangGraph
- Gateway 返回 None 时，它在冷路径期间的 trace 全部丢失（api.py 第 93-96 行有注释承认此问题）

### 1.2 为什么分？设计动机分析

分两个路径的核心原因有四个：

**原因 1：避免 LangGraph 开销**

LangGraph 的 `ainvoke()` 不是零成本的。每次调用涉及：

- State 对象构造（TypedDict + reducer 初始化）
- 图拓扑遍历（查找当前节点的出边）
- 节点间状态传递（每个节点返回 dict 合并到全局 state）
- Checkpointer 序列化（即使内存 checkpointer 也有 dict 深拷贝开销）

对于一条 L0 缓存命中的请求（完全相同文本之前审过），这个开销是纯浪费 —— 答案已知，不需要走任何图逻辑。实测 Gateway 直返 < 0.5ms，而 LangGraph `ainvoke()` 空跑（一个空节点）至少 5-10ms。

**原因 2：语义缓存查询需要 embedding**

Gateway 的 ChromaDB 缓存查询需要先把文本向量化（调用 BGE-small-zh）。如果把这个放在 LangGraph 的 Triage 节点里，那每次缓存查询都要进入图、执行节点、再可能退出。放在 Gateway 里，缓存命中后直接返回，图都不用进。

**原因 3：关注点分离**

Gateway 做的是纯无状态的快速判断（查缓存、扫关键词），LangGraph 做的是有状态的推理（BERT → LLM 递进，可能需要升级到 Multimodal）。把无状态预过滤和有状态推理混在同一个图里，会让图变得臃肿且难以调试。

**原因 4：独立扩缩容**

热路径是 CPU 密集型（哈希计算、AC 自动机扫描、向量相似度），冷路径是 GPU/LLM 密集型。分开后，热路径可以在 API 进程中直接完成，冷路径可以异步推送到 Ray/Celery 队列。

### 1.3 当前设计的问题

#### 问题 1：Triage 节点是死代码

`triage.py`（105 行）实现了缓存查询 + 关键词过滤 + 路由逻辑，但 **graph.py 根本没把它加进图**。入口直接是 `image_agent`。

```
graph.py:
  builder.set_entry_point("image_agent")   ← 不是 "triage"

CLAUDE.md 设计:
  builder.set_entry_point("triage")        ← 设计是 Triage 入口
```

| 文件 | 行数 | 是否被引用 |
|------|------|-----------|
| agents/triage.py | 105 | `git grep triage_orchestrator` 仅自己定义，无调用方 |
| gateway.py | 154 | api.py 直接调用 `gateway.check()` |

Gateway 吃掉了 Triage 的所有职责，但 Gateway 不在图中。这导致：

- **设计文档和代码不一致**，新人看 CLAUDE.md 以为 Triage 是核心，实际上 Gateway 才是
- **Triage 里的零容忍短路逻辑（politics/violence 直接 block）与 Gateway 的关键词 block 逻辑有微妙差异**，容易让人困惑哪个才是真正的入口判断

#### 问题 2：关键词匹配跑了两次

对于进入冷路径的请求：

```
Gateway.keyword_filter.match(text)     ← 第一次（gateway.py:62）
    ↓ 匹配到 conf=0.6（嵌入词），不拦截，升级
Text Agent.keyword_filter.match(text)  ← 第二次（text_agent.py:34）
    ↓ 同样 conf=0.6，不拦截，继续 L2
```

AC 自动机扫描是 O(n) 很快（对 200 字文本约 0.3ms），但逻辑上这是冗余。Text Agent 不知道 Gateway 已经扫过了。如果关键词词库扩展到 10 万+，两次扫描的浪费会放大。

#### 问题 3：Gateway 冷路径 trace 丢失

```python
# api.py:82-96
gw_result = gateway.check(text, ...)

if gw_result is not None:
    # 热路径返回，带 Gateway traces ✓
    return gw_result

# 冷路径：gateway 返回了 None
# 注释承认：Gateway 在 miss 时不返回 traces
# 冷路径的 response 里 traces 只来自 LangGraph
state = _make_state(req, text)
result = await graph.ainvoke(state)
```

Gateway 冷路径实际上做了内存缓存查询 + 关键词扫描 + ChromaDB 查询，这些耗时数据在返回 None 时全部丢弃。调试冷路径延迟问题时，看不到 Gateway 层花了多少时间。

#### 问题 4：图片处理一刀切

```python
# gateway.py:35-36
if (image_url and image_url.strip()) or (image_base64 and image_base64.strip()):
    return None  # 直接升级
```

任何带图片的请求直接走冷路径。但图片也有缓存可能性 —— 比如同一张图片的 perceptual hash 完全可以做热路径去重。当前设计中图片请求 100% 进 LangGraph，即使是一张之前审过的完全相同的图片。

#### 问题 5：Gateway 变成了 God Object

Gateway 类承担了 4 个职责：

1. 内存缓存管理（L0）
2. 关键词过滤
3. ChromaDB 缓存查询（L1）
4. 统计数据收集

在 154 行代码里不算多，但这些都是不同的关注点。如果要单独测试关键词逻辑、单独测试缓存逻辑，都绕不开 Gateway 类。

### 1.4 结论

**双路径设计本身是正确的**，理由充分（避免图开销、缓存短路、独立扩缩容）。但当前实现有三个需要修正的工程问题：

| 问题 | 严重程度 | 建议 |
|------|----------|------|
| Triage 死代码 | 中 | 删除 `triage.py`，或让 graph.py 以 `triage` 为入口并让 Triage 内部调 Gateway |
| 关键词重复扫描 | 低 | Text Agent 在 state 中加标记 `keyword_prefiltered: true`，跳过 L1 |
| Gateway 冷 trace 丢失 | 中 | 让 `gateway.check()` 返回 `(decision, traces)` 元组，而非仅 decision |
| 图片一刀切 | 低（POC） | POC 阶段可接受，生产需补充图片哈希去重 |
| Gateway God Object | 低 | 拆分为 `KeywordGate` + `CacheGate` 两个独立类 |

**不影响效率的判断**：双路径对效率是正向的。Gateway 拦截的请求不会产生 LangGraph 开销，这是设计意图。问题不在于"该不该分"，而在于分完之后两边的代码没有清理干净（死代码、重复扫描、trace 丢失）。

### 1.5 一个更好的结构（供参考）

将 Gateway 整合回 LangGraph 作为 Triage 节点，但通过条件边直接 `END` 实现短路：

```python
builder.add_node("triage", triage_with_gateway)
builder.set_entry_point("triage")

builder.add_conditional_edges(
    "triage",
    lambda state: "done" if state.get("cache_hit") or state.get("keyword_block")
                  else "route",
    {
        "done": END,              # 热路径 → 直接结束
        "route": "image_agent",   # 冷路径 → 进入推理
    }
)
```

好处：

- 所有路由逻辑统一在图中，单一入口
- 热路径 trace 也在 LangGraph 的 state 里，不会丢失
- 删除 Gateway 独立类，减少代码路径
- keyword_filter 只在一个地方（Triage）执行

代价：

- 热路径也要走 `ainvoke()`，多 5-10ms 开销
- 但相比 LLM 调用的 500ms-1s，5ms 可忽略

**建议**：Phase 2 重构时考虑这个方案。POC 阶段不改，当前够用。

---

## 讨论记录

**2026-05-05**：热/冷路径分析

- 确认双路径设计动机：避免 LangGraph 开销、缓存短路、独立扩缩容
- 发现 5 个工程问题：Triage 死代码、关键词重复扫描、冷路径 trace 丢失、图片一刀切、Gateway 膨胀
- 结论：设计方向正确，实现需要清理。建议 Phase 2 将 Gateway 整合回 LangGraph Triage 节点

---

---

## 2. 热路径技术栈分析

热路径的目标：在进入 LangGraph 之前，用最低成本拦截最多的请求。技术选型的核心约束是 **延迟 < 5ms 且成本为零（纯 CPU，不调 API）**。

### 2.1 执行流程

```
Gateway.check(text)
  │
  ├── [图片存在?] → SHA256(image_ref) → 查 MemoryCache → 命中则返回
  │                                                      → 未命中则直接升级冷路径
  ├── [空文本?] → 直接 pass
  │
  ├── Step 0: L0 Memory Cache
  │   └── SHA256(text) → Python dict 查找 (TTLCache, 1h TTL)
  │       └── 命中 (< 0.01ms) → 直接返回缓存结果
  │
  ├── Step 1: AC Automaton 关键词匹配
  │   └── pyahocorasick.iter(text)  O(n) 单遍扫描
  │       ├── 白名单命中 → pass
  │       ├── 独立词 (conf=1.0) → block
  │       └── 嵌入词 (conf=0.6) → 升级冷路径
  │
  └── Step 2: ChromaDB 语义缓存
      └── BGE-small-zh → embed(text) → ChromaDB.query(cosine)
          ├── similarity ≥ 0.95 → 返回缓存结果
          └── 未命中 → 升级冷路径
```

### 2.2 技术 1: L0 Memory Cache (SHA256 + TTLCache)

**选型**：Python `cachetools.TTLCache` + `hashlib.sha256`

| 属性 | 值 |
|------|------|
| 数据结构 | Python dict (哈希表) |
| Key | SHA256(text) 64 字符 hex |
| TTL | 3600 秒 (1 小时) |
| 最大容量 | 100,000 条 |
| 内存占用 | ~50 MB (满载) |
| 查找延迟 | < 0.01ms (纯内存) |
| 成本 | 零 |

**为什么选 SHA256 而不是 MD5**：MD5 有碰撞风险，两个不同文本可能产生相同 hash。对于内容审核系统，hash 碰撞意味着一个违规文本可能因为与安全文本同 hash 而被放行。SHA256 碰撞概率 < 10^-60，可忽略。虽然比 MD5 慢，但都 < 0.01ms，差异无意义。

**为什么选 TTLCache 而不是 Redis**：
- POC 阶段单机部署，Redis 需要额外进程
- `cachetools.TTLCache` 纯 Python，`pip install` 即用，零运维
- 生产阶段可迁移到 Redis（需网络调用，延迟升至 0.1-0.5ms）

**为什么 TTL 设为 1 小时**：内容审核中完全相同的文本重复出现，通常发生在短时间内（同一用户重复发、垃圾广告批量刷）。1 小时后大概率不再出现，缓存条目自动过期释放内存。

**问题**：
- 进程重启后缓存丢失（Python 内存），冷启动期间命中率为 0
- 100K 容量上限对小规模足够，但大规模生产可能需要更大容量或 LRU 淘汰策略（TTLCache 已内置 TTL+LRU）

### 2.3 技术 2: AC 自动机关键词匹配 (pyahocorasick)

**选型**：`pyahocorasick` (Aho-Corasick 算法)

| 属性 | 值 |
|------|------|
| 算法 | Aho-Corasick 自动机 |
| 时间复杂度 | O(n + m)，n = 文本长度，m = 匹配数 |
| 空间复杂度 | O(词典总字符数 × 字母表大小) |
| 匹配延迟 | < 0.5ms (200 字文本，50 个关键词) |
| 成本 | 零 |

**为什么不是正则**：正则的 `word1|word2|word3|...` 在词库大时性能崩溃。AC 自动机无论词库多大（百万级），扫描时间始终为 O(n)，因为它在 Trie 上构建了失败转移边。举个例子：

```
正则：傻逼|SB|脑残|弱智|...|(50个词)
  → 每个位置回溯尝试所有 alternation，最坏 O(n × k)

AC 自动机：
  → 每个字符走一次状态转移，O(n)，与词库大小无关
```

**为什么不是 Bloom Filter**：Bloom Filter 只能回答"关键词存在吗"，不能回答"匹配了哪个关键词、在什么位置、属于什么类别"。AC 自动机返回每个匹配的精确位置和类别，这对后续的上下文验证（jieba 分词判断独立词/嵌入词）至关重要。

**为什么需要 jieba 分词做上下文验证**：

```
例子 1："你是个傻逼" → jieba: ["你","是","个","傻逼"]
  → "傻逼" 是独立 token → conf=1.0 → 直接拦截 ✓

例子 2："接口交换技术" → jieba: ["接口","交换","技术"]
  → "口交" 没有出现在分词结果中，它只是子串 → conf=0.0 → 放行 ✓

例子 3："操场上在操练" → 白名单匹配 "操场上" + "操练"
  → 虽然是关键词子串，但在白名单短语内 → 放行 ✓
```

**问题**：
- pyahocorasick 是 C 扩展，编译安装需要 C 编译器。如果未安装，代码回退到 `if word in text` 的 O(n×k) 暴力匹配（keyword_filter.py:105-109）
- 白名单用正则实现，O(白名单长度)。当白名单短语很多时，正则 alternation 性能退化
- jieba 加载耗时 0.5-1 秒（构建前缀词典），进程启动时首次 match 触发

### 2.4 技术 3: ChromaDB 语义缓存

**选型**：ChromaDB PersistentClient + HNSW 索引

| 属性 | 值 |
|------|------|
| 数据库 | ChromaDB (嵌入式) |
| 索引算法 | HNSW (Hierarchical Navigable Small World) |
| 相似度 | Cosine similarity |
| 命中阈值 | ≥ 0.95 |
| 向量维度 | 512 (BGE-small-zh-v1.5) |
| 查询延迟 | < 5ms (包括 embedding + 数据库查询) |
| 成本 | 零（本地 CPU） |

**为什么是 ChromaDB 而不是 Milvus**：
- POC 用 ChromaDB 因为 `pip install chromadb` 一行即用，无需 Docker
- Milvus 需要独立部署 server + etcd + MinIO，POC 阶段运维成本太高
- CLAUDE.md 设计文档也明确：POC 用 ChromaDB，生产切 Milvus

**为什么 similarity 阈值设 0.95**：余弦相似度 0.95 意味着两个向量几乎完全同向。只有换了一两个近义词但仍表达相同意思的文本才会命中。阈值设太高 → 命中率太低（缓存白做），设太低 → 误判增多（"你今天过得怎么样"和"你去死吧"可能因为句式相似而错误匹配）。0.95 是经验值，需通过 benchmark 调参。

**为什么是 BGE-small-zh 而不是 text-embedding-3-small**：
- BGE 本地运行，零 API 成本，零网络延迟
- BGE-small-zh-v1.5 专为中文优化（BAAI 北京智源）
- 512 维向量在 ChromaDB 中存储和查询效率高
- text-embedding-3-small 需调用 OpenAI API，有成本和延迟

**问题**：
- 首次调用需加载 BGE 模型（~95MB 进内存），冷启动 200-500ms
- ChromaDB 持久化目录（`data/chroma/`）会持续增长，需要定期清理过期条目
- HNSW 索引在内存中，大缓存时内存占用上升

### 2.5 热路径技术栈总览

| 层 | 技术 | 数据结构/算法 | 延迟 | 拦截能力 |
|------|------|------|------|------|
| L0 内存缓存 | cachetools.TTLCache | SHA256 + Hash Table | < 0.01ms | 精确重复文本 (10-15%) |
| L1a 关键词 (独立词) | pyahocorasick + jieba | AC Automaton + Trie | < 0.5ms | 明显违规关键词 (15-20%) |
| L1b 关键词 (嵌入词) | pyahocorasick + jieba | AC Automaton + Token Span | < 0.5ms | 疑似违规 → 升级冷路径 |
| L2 语义缓存 | ChromaDB + BGE-small-zh | HNSW + Cosine Similarity | < 5ms | 语义相似文本 (10-15%) |
| 白名单 | re (正则) | Regex Alternation | < 0.1ms | 已知误杀短语 |

**预期热路径总拦截率**：35-50%（L0 10-15% + L1a 15-20% + L2 10-15%）

### 2.6 关键设计决策讨论

**决策 1：AC 自动机的 "独立词 vs 嵌入词" 二分法**

当前设计将关键词匹配置信度分为两级：1.0（独立词，直接拦截）和 0.6（嵌入词，升级冷路径）。这个二分是否合理？

```
"傻逼"                  → jieba: ["傻逼"]        → standalone → conf=1.0 → block ✓
"你傻逼了吧"             → jieba: ["你","傻逼","了"] → standalone → conf=1.0 → block ✓ (他人在骂人)
"操场上"                → 白名单                  → whitelist  → conf=0.0 → pass ✓
"操场上有个傻逼"         → "操场上"白名单 + "傻逼"独立 → conf=1.0 → block (正确，"傻逼"确实在骂人)
```

二分法在当前 50 个关键词的小词库下是可行的。但如果词库扩展到千级以上，可能出现大量 conf=0.6 的升级请求涌入冷路径，导致 BERT 负载上升。届时需要引入三级置信度或统计模型来更精细地判断。

**决策 2：先关键词后缓存的顺序**

Gateway 先做关键词匹配再做 ChromaDB 缓存查询。这个顺序是合理的：

- 关键词匹配 < 0.5ms，缓存查询需 embedding（首次 200ms+，后续 1-3ms）
- 关键词命中率 15-20%，拦截后可跳过更昂贵的 embedding 计算
- 如果先做缓存，那这 15-20% 的请求都要白花 embedding 时间

**决策 3：嵌入词为什么升级而不是拦截**

关键词 conf=0.6（嵌入词）的处理是"升级到冷路径"而不是"拦截"或"放行"。这体现了内容审核的核心权衡：

```
"接口交换" → 包含"口交" → 嵌入词 → 升级 BERT
  → BERT 判断：技术术语 → 安全 ✓
  → 如果直接拦截 → 误杀

"各种口交视频" → 包含"口交" → 也是嵌入词 → 升级 BERT  
  → BERT 判断：色情内容 → 拦截 ✓
  → 如果直接放行 → 漏网
```

仅靠子串匹配无法区分这两种情况，必须引入语义理解（BERT）。所以嵌入词的处理是正确的——既不盲目拦截（误杀），也不盲目放行（漏网），而是交给更强的模型做二次判断。

---

## 3. 热路径多服务高并发场景分析

### 3.1 场景假设

> 实际项目中，多个服务共用审核系统，每个服务每天产生上百万日志数据。

假设 10 个服务，每个服务 500 万条/天：

```
总请求量: 10 × 5,000,000 = 50,000,000 条/天
平均 QPS: 50,000,000 / 86,400 ≈ 578 QPS
峰值 QPS: 578 × 10 = 5,780 QPS (日常峰值)
突发 QPS: 578 × 20 = 11,560 QPS (营销活动/热点事件)
```

### 3.2 单机热路径各层吞吐上限

以一个 Python 进程（单核）测试：

| 层 | 单次延迟 | 单核理论 QPS | 说明 |
|------|----------|------|------|
| L0 内存缓存 | < 0.01ms | **~100,000** | 纯 dict.get()，不涉及 GIL 竞争 |
| AC 自动机关键词 | ~0.3ms | **~3,000** | C 扩展，释放 GIL，纯 CPU |
| BGE Embedding | ~5ms | **~200** | PyTorch 推理，受 GIL 限制 |
| ChromaDB 查询 | ~1ms | **~1,000** | HNSW 图遍历，内存密集 |
| ChromaDB 写入 | ~3ms | **~300** | HNSW 图插入，比查询慢 |

**关键瓶颈：BGE Embedding**。单核 200 QPS，而 578 QPS 的均值已经需要 3 个核来覆盖 embedding。5780 QPS 峰值需要 29 个核。

### 3.3 流量路径分解

以 50M/天的规模，按预期的拦截率分布：

```
                    ┌── 35% L0 内存缓存命中 ──── 202 QPS (直接返回, 不受限)
                    │
50M/天 ──Gateway───┼── 18% 关键词命中/白名单 ── 104 QPS (直接返回, 不受限)
(578 QPS avg)      │
                    ├── 12% ChromaDB 缓存命中 ── 69 QPS  (需要 embedding + 查询)
                    │
                    └── 35% 升级冷路径 ────────── 202 QPS (需要 embedding + BERT + LLM)
```

实际上 ChromaDB 缓存查询和冷路径都需要先做 embedding：

```
真正需要 embedding 的请求: 12% + 35% = 47% → 272 QPS (平均)
                                                 → 2,720 QPS (峰值)
                                                 → 5,433 QPS (突发)
```

**272 QPS 均值已经超过单核 BGE 的 200 QPS 上限**。

### 3.4 瓶颈分析结论

**热路径在高并发下会出问题**，但不是所有层都有问题：

- L0 内存缓存 + AC 自动机：**不受影响**。即使 10,000 QPS 也能单核应付
- BGE Embedding：**是唯一的瓶颈**。单核 200 QPS，多服务场景下必须扩容
- ChromaDB：单核 1000 QPS 查询够用，但写入 (300 QPS) 在峰值时会积压

### 3.5 解决方案（由简到繁）

#### 方案 0：Embedding 缓存（立即可做，改动量小）

在 BGE embedding 之前加一层 LRU 缓存，key 是文本 SHA256。相同的文本不用重新 embedding。

```python
# embedder.py 新增 embedding_cache
from functools import lru_cache

class Embedder:
    def __init__(self, ...):
        ...
        self._emb_cache = TTLCache(maxsize=500_000, ttl=3600)  # 50 万条 × 1h
    
    def embed(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode()).hexdigest()
        cached = self._emb_cache.get(key)
        if cached is not None:
            return cached
        vec = self._model.encode(text[:8191], normalize_embeddings=True).tolist()
        self._emb_cache[key] = vec
        return vec
```

效果：对重复出现的文本（垃圾广告批量刷），embedding 命中率可达 30-40%，等效提升 1.5x 吞吐。

成本：50 万条 × 512 维 × 4 字节 = ~1 GB 内存。

#### 方案 1：FastAPI 多 Worker（立即可做，无代码改动）

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

4 个 worker 进程 → 4 个独立的 BGE 模型实例 → 4 × 200 QPS = 800 QPS embedding 吞吐。

代价：每个 worker 加载一份 BGE 模型（4 × 95 MB = 380 MB 额外内存）。L0 内存缓存进程间不共享（每个 worker 独立的 TTLCache），命中率下降为原来的 1/N。

#### 方案 2：BGE 批量推理（中等改动）

sentence-transformers 的 `encode()` 支持批量输入，n 条文本一起编码比 n 条单独编码快 2-3 倍：

```python
# 单条：5ms/条
vec = model.encode(text)

# 批量 32 条：~50ms → 1.6ms/条，3x 加速
vecs = model.encode(texts, batch_size=32)
```

实现方式：Gateway 内部用一个 `asyncio.Queue` 攒批，达到 32 条或等待 10ms 后批量推理。

代价：增加了 10ms 的人为延迟（对审核系统可接受）。实现复杂度中等。

#### 方案 3：ONNX 导出 BGE 模型（中等改动）

将 BGE-small-zh 导出为 ONNX 格式，用 `onnxruntime` 推理：

```
PyTorch BGE: ~5ms/条
ONNX BGE:   ~1.5ms/条 (3x 加速)
```

参考已有的 `bert_onnx.py` 实现模式。ONNX Runtime 还支持多线程并行（`intra_op_num_threads`），单进程可跑到 600+ QPS。

代价：需要预先导出 ONNX 模型（一次性操作），增加 onnxruntime 依赖。

#### 方案 4：独立 Embedding 服务（较大改动）

将 BGE 模型部署为独立的推理微服务（类似 vLLM 但针对 embedding）：

```
API (Gateway) ──gRPC/HTTP──→ Embedding Service (GPU/多CPU)
                                  └── BGE model × N replicas
```

API 进程只做 L0 缓存 + AC 自动机，embedding 请求异步发到专门的 embedding 服务。

代价：需要部署和运维额外服务，引入网络延迟（内网 ~1ms）。

#### 方案 5：Redis 共享 L0 缓存（中等改动）

用 Redis 替代进程内 TTLCache：

```python
# 所有 API worker 共享同一个 Redis 缓存
redis.get(f"mod:{sha256}")  # ~0.5ms (内网 Redis)
```

代价：延迟从 < 0.01ms 升到 ~0.5ms，但多 worker 之间缓存命中率保持一致。

#### 方案 6：全链路水平扩展（较大改动）

```
         Nginx/Envoy (LB)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 API-1     API-2     API-N    (FastAPI × M workers each)
    │         │         │
    └─────────┼─────────┘
              ▼
    Redis Cluster (共享 L0 缓存)
              │
              ▼
    ChromaDB Server (独立部署)
              │
              ▼
    Embedding Service Pool (BGE × K replicas)
              │
              ▼
    LangGraph Workers (Ray/Celery, 异步)
```

### 3.6 推荐实施路径

```
Phase 1 (当前 POC，578 QPS avg)
  └── 方案 0: Embedding 缓存     ← 改动 10 行代码
  └── 方案 1: uvicorn --workers 4 ← 改动启动命令
  → 目标: 800 QPS embedding 吞吐，覆盖 3x 峰值余量

Phase 2 (灰度上线，5000 QPS peak)
  └── 方案 3: ONNX BGE 加速      ← 3x embedding 推理速度
  └── 方案 5: Redis 共享缓存     ← 多 worker 间缓存一致
  → 目标: 2400 QPS/实例，2 实例覆盖峰值

Phase 3 (全量上线，10000+ QPS peak)
  └── 方案 6: 全链路水平扩展
  └── 方案 4: 独立 Embedding 服务 (GPU)
  └── 方案 2: 批量推理攒批
  → 目标: 100K QPS，水平无限扩容
```

### 3.7 一个重要的认知

**热路径的瓶颈不在代码逻辑，而在模型推理**。L0 内存缓存和 AC 自动机是纯 CPU 操作，可以轻松跑到万级 QPS。BGE Embedding 调用一次 PyTorch 推理的开销是关键词匹配的 100 倍以上。所以优化方向很明确：

```
优化目标：让更少的请求走到 BGE Embedding

手段：
1. L0 内存缓存命中 → 跳过 embedding
2. 关键词直接拦截 → 跳过 embedding  
3. Embedding 结果缓存 → 相同文本不重复计算
4. 批量推理 → 均摊模型加载开销
5. 独立部署 → 解除 API 进程的 CPU 竞争
```

---

## 讨论记录

**2026-05-05**：热路径多服务高并发分析

- 假设 10 服务 × 500 万条/天 = 50M/天，均值 578 QPS，峰值 5,780 QPS
- 唯一瓶颈：BGE Embedding（单核 200 QPS），L0 缓存和 AC 自动机无压力
- 推荐路线：Embedding 缓存 (1.5x) → 多 Worker (4x) → ONNX (3x) → Redis → 水平扩展
- 详细方案见 `poc/research/` 目录

---

## 4. 冷路径整体框架与逐层技术分析

### 4.1 冷路径是什么

热路径（Gateway）解决不了的请求（没有缓存命中、没有关键词匹配、没有已知有害图片哈希），进入冷路径 LangGraph 做真正的 AI 推理。

```
冷路径流量占比: 预期 20-50%（具体取决于缓存热度和关键词命中率）

请求进入条件:
  - 文本: L0 缓存 miss + 关键词 miss + ChromaDB miss
  - 图片: dHash miss + URL 缓存 miss
  - 图文混合: 以上任一情况
```

### 4.2 DAG 拓扑

```python
# graph.py
builder.set_entry_point("image_agent")

builder.add_conditional_edges("image_agent", _route_after_image, {
    "text_agent": "text_agent",    # 有文本 + 图片 OCR 结果
    "decision": "decision",        # 纯图片无文本
})
builder.add_edge("text_agent", "decision")
builder.add_edge("decision", "action")
builder.add_edge("action", END)
```

三条实际路径：

```
路径 A: text_only  → image_agent(空过) → text_agent → decision → action
路径 B: image_only → image_agent(NSFW+OCR) → decision → action
路径 C: mixed      → image_agent(NSFW+OCR+追加OCR文本) → text_agent → decision → action
```

**注意**：入口是 `image_agent`，不是 `text_agent`。纯文本请求进 image_agent 后检测到无图片直接返回 `{"image_result": normal, "text": unchanged}`，然后路由到 text_agent。这个设计是因为图文混合时需要 Image Agent 先把 OCR 文字提取出来再交给 Text Agent。

### 4.3 逐层技术分解

#### Node 1: Image Agent（poc/src/agents/image_agent.py:23）

**职责**：下载图片 → NSFW 分类 → OCR 文字提取。

**技术栈**：

| 步骤 | 技术 | 模型 | 延迟 | 成本 |
|------|------|------|------|------|
| 1. 图片获取 | `requests.get()` / `base64.b64decode()` | - | 100-500ms (网络) | zero |
| 2. NSFW 分类 | HuggingFace pipeline | Falconsai/nsfw_image_detection (ViT) | ~100ms (GPU), ~500ms (CPU) | low |
| 3. OCR 提取 | EasyOCR | ch_sim + en | ~200ms (CPU) | zero |

**POC 状态**：NSFW 模型被 skip_model=True 跳过，仅做图片尺寸/格式校验。OCR 需要首次下载 EasyOCR 模型（~450MB）。

**关键技术决策：混合内容的串行处理**

图文混合先跑 Image Agent（含 OCR），OCR 文字追加到 `state.text`，再路由到 Text Agent。这是 POC 的妥协方案：

```
原因: 如果 Text Agent 和 Image Agent 并行跑，Text Agent 拿不到 OCR 提取的文字
后果: 图片里嵌入的违规文字（如"加微信"）只有 OCR 能抓到
```

生产优化方向：两阶段并行——第一轮 Text+Image 并行跑各自直接特征；第二轮如果 Image 发现 OCR 文字，单独再跑一次 Text Agent 审 OCR 文字。

**当前问题**：

1. `skip_model=True` 意味着 NSFW 检测完全不工作——POC 阶段所有图片都被标记为 normal
2. EasyOCR 首次下载可能很慢（~450MB 模型文件），冷启动需要等待
3. 图片下载用同步 `requests.get()`——在 async 函数中这会阻塞事件循环

#### Node 2: Text Agent（poc/src/agents/text_agent.py:21）

**职责**：文本三层漏斗审核 —— L1 关键词 → L2 BERT → L3 LLM。

**技术栈**：

| 层 | 技术 | 模型 | 延迟 | 成本 | 短路条件 |
|------|------|------|------|------|------|
| L1 | AC 自动机 + jieba | 内置词库 | < 0.5ms | zero | conf > 0.99 → 直接返回 |
| L2 | BERT (ONNX 优先, HF 回退) | unitary/toxic-bert | ~50ms (ONNX), ~150ms (HF) | low | conf ≥ 0.95 → 跳过 LLM |
| L3 | LLM API | DeepSeek Chat / GPT-4o-mini / Claude Haiku | 500-2000ms | **high** | 无 |

**三层漏斗递进逻辑**：

```
L1 关键词:
  - 命中独立词 → 直接返回 (block, conf=1.0, cost=zero)
  - 命中嵌入词 → 标记但继续 L2
  - 未命中   → 继续 L2
  - Gateway 已扫描过 → 跳过 L1 (keyword_prefiltered=True)

L2 BERT:
  - conf ≥ 0.95 → 直接返回 (跳过 LLM, cost=low)
  - 中文 + English-only BERT → 跳过 L2，直接 L3
  - BERT 不可用 → 降级到关键词 → LLM

L3 LLM:
  - 带 BERT 预判结果作为上下文
  - 8 秒超时 → 回退到 BERT 结果
  - 返回 label + confidence + reason (中文解释)
```

**关键设计决策**：

1. **BERT 高置信短路**：BERT 认为 > 95% 确定的不需要 LLM 复审。这是成本控制的核心——LLM 调用一次 ≈ BERT 调用 20 次的价格。

2. **中文检测 → 跳过 English BERT**：`unitary/toxic-bert` 只训练了英文数据。中文文本直接跳过 L2 走 L3，避免 BERT 给出随机结果。这是 POC 的已知局限——需要换多语言 BERT。

3. **LLM 超时回退**：8 秒超时后使用 BERT 结果。保证了可用性——LLM API 挂了不会导致审核中断。

**当前问题**：

1. English-only BERT 对中文无效——中文文本 100% 走到 L3 LLM
2. BERT 高置信阈值 0.95 可能太激进——大量 L3 调用增加成本
3. L2 BERT 和 L3 LLM 的判断可能矛盾（BERT 说 safe，LLM 说 unsafe），没有矛盾处理机制（当前直接信任 LLM）

#### Node 3: Decision Agent（poc/src/agents/decision.py:11）

**职责**：聚合 Text Agent + Image Agent 结果，应用策略规则，灰度区分流。

**技术栈**：纯规则引擎（无模型调用）。

**决策树**（4 条路径）：

```
Path 1: Cache hit
  → 直接复用缓存结果

Path 2: 零容忍关键词 (politics/violence)
  → 硬覆盖为 block，无论 BERT/LLM 怎么说

Path 3: 聚合 text_result + image_result
  → 纯图片无文本: NSFW → block, 正常 → pass
  → 图文混合: 文本 label + 图片 NSFW → 联合判断
  → 图片 NSFW + 文本 safe → label 升级为 unsafe

Path 4: 灰度区判断
  → confidence ∈ [0.3, 0.7] → review (人工复核)
  → confidence < 0.3 → pass (低置信违规可能是误判，宁可放过)
  → confidence > 0.7 → 直接决策 (pass 或 block)
```

**关键设计意图**：

- **零容忍硬覆盖**：politics 和 violence 类别的关键词命中后，即使 LLM 说安全也无效（强制 block）。合规层面必须"宁可错杀不可放过"。
- **灰度区 `[0.3, 0.7]`**：故意不做决策，交给人工。confidence < 0.3 的违规很可能是误判，放过；> 0.7 的违规足够确定，拦截；中间的不确定区间交给人工复核。

**当前问题**：

1. 没有加权聚合——只是简单传递 text_result 作为最终结果，`score_aggregate` 逻辑没有实现
2. 灰度区判断只用 `text_result.confidence`，图片结果没有被纳入
3. 零容忍类别静态硬编码（politics/violence），缺少可配置的策略引擎

#### Node 4: Action Agent（poc/src/agents/action.py:12）

**职责**：执行决策 + 写回三层缓存。

**技术栈**：

| 操作 | 目标 | 延迟 | 说明 |
|------|------|------|------|
| 写回 L0a 本地缓存 | `memory_cache.set()` | < 0.01ms | SHA256 → TTLCache |
| 写回 L0b Redis | `redis_cache.set()` | ~0.5ms | 跨 Worker 共享，降级无影响 |
| 写回 L1 ChromaDB | `vector_cache.store()` | ~15ms | 包括 BGE embedding 计算 |
| 日志记录 | `logger.info()` | < 0.01ms | 结构化日志 |

**POC 不做的**：
- 不执行实际的 block/pass 动作（不接外部执法系统）
- 不发送用户通知
- 不入审计数据库（PostgreSQL）
- 不触发 Feedback Agent

### 4.4 冷路径完整技术总览

```
请求进入
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Image Agent                                             │
│  ├── requests.get / base64.b64decode  (图片获取)         │
│  ├── Falconsai/nsfw_image_detection   (ViT NSFW分类)     │
│  └── EasyOCR (ch_sim+en)             (文字提取)         │
│      └── OCR 文本追加到 state.text                       │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Text Agent                                              │
│  ├── L1: AC 自动机 + jieba          (关键词, <0.5ms)      │
│  │    └── conf>0.99 → 直接返回                           │
│  ├── L2: BERT ONNX / HF pipeline    (毒性分类, ~50ms)     │
│  │    ├── 中文+English BERT → 跳过L2                      │
│  │    └── conf≥0.95 → 跳过L3                            │
│  └── L3: DeepSeek/GPT-4o/Claude     (深度审核, 500ms+)   │
│       └── 8s超时 → 回退BERT                              │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Decision Agent                                          │
│  ├── 缓存命中 → 复用                                     │
│  ├── 零容忍 (politics/violence) → 硬覆盖 block            │
│  ├── text + image 聚合                                  │
│  ├── 灰度区 [0.3, 0.7] → review                         │
│  └── 正常 → pass / block                                │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Action Agent                                            │
│  ├── 写回 L0a 本地 TTLCache                              │
│  ├── 写回 L0b Redis                                     │
│  ├── 写回 L1 ChromaDB                                   │
│  └── 日志输出                                           │
└─────────────────────────────────────────────────────────┘
```

### 4.5 成本分析

| 路径 | 经过的层 | 总延迟 | 成本 |
|------|------|------|------|
| 纯文本 L1 命中 | Text L1 → Decision → Action | < 1ms | $0 |
| 纯文本 L2 命中 | Text L1+L2 → Decision → Action | ~50ms | ~$0.0001 |
| 纯文本 L3 | Text L1+L2+L3 → Decision → Action | 500-2000ms | ~$0.002 |
| 纯图片 (POC) | Image(无NSFW) → Decision → Action | ~200ms (OCR) | $0 |
| 纯图片 (生产) | Image(NSFW+OCR) → Decision → Action | ~400ms | ~$0.0002 |
| 图文混合 L3 | Image(NSFW+OCR) → Text L1+L2+L3 → Decision → Action | 1000-3000ms | ~$0.003 |

### 4.6 设计问题

**问题 1：入口点是 image_agent 而非 text_agent**

`graph.py` 入口设为 `image_agent`，纯文本请求也要先进 Image Agent 绕一圈。这引入了不必要的开销：
- 每次文本请求都要创建 `image_result = {normal, no_image}` 对象
- 纯文本请求在 image_agent 中检查 `not image_url and not image_base64` 后直接返回

建议：将入口改回 triage/gateway 或设为条件入口（纯文本直接从 text_agent 开始）。

**问题 2：NSFW 模型在 POC 阶段完全不工作**

`skip_model=True` 意味着所有图片都被判为 normal。这意味着 POC 阶段没法验证图片审核的准确率。

**问题 3：没有图文矛盾处理**

Text Agent 说 safe、Image Agent 说 NSFW 时，Decision Agent 把 label 升级为 unsafe（line 66-67）。但反向情况（Text 说 unsafe、Image 说 normal）没有特殊处理。表情包配正常图 + 违规文的情况只能依赖 Text Agent 的判断。

**问题 4：BERT → LLM 矛盾没有处理**

L2 BERT 说 safe（conf=0.92）、L3 LLM 说 unsafe（conf=0.85）时，当前代码直接信任 LLM。但 LLM 的 0.85 置信度可能不如 BERT 的 0.92 可靠（BERT 在自己的训练分布内更稳定）。

**问题 5：Decision Agent 缺少加权聚合**

`score_aggregate` 是 CLAUDE.md 设计文档中明确要求的（Text 40% + Image 35% + Multi 25%），但当前代码完全没有实现。Decision 简单地把 text_result 当作最终结果。这意味着图片审核的结果在 Decision 层几乎被忽略。

---

## 6. Action Agent 深入分析

### 6.1 职责

Action Agent 是冷路径最后一个节点。它不是"执行 block/pass"的业务操作（POC 阶段不接外部执法系统），而是**将审核结果写回缓存**，供后续请求复用。

```python
# action.py:12-45
async def action_executor(state: ModerationState) -> dict:
    # 1. 写回 L0a 本地内存缓存
    memory_cache.set(text, decision, confidence, reason)

    # 2. 写回 L0b Redis 共享缓存
    redis_cache.set(text, decision, confidence, reason)

    # 3. 写回 L1 ChromaDB 语义缓存（含 BGE embedding）
    embedding = embedder.embed(text)
    vector_cache.store(embedding, text, decision, confidence, reason)

    # 4. 日志输出
    logger.info("ACTION | id=%s | decision=%s | confidence=%.3f | ...")
```

### 6.2 各缓存写入的延迟与成本

| 操作 | 延迟 | 是否阻塞 | 失败影响 |
|------|------|------|------|
| L0a 本地 TTLCache | < 0.01ms | 同步 | 无（纯内存） |
| L0b Redis | ~0.5ms | 同步 | 无（Redis 不可用时静默跳过） |
| L1 ChromaDB 写入 | **~15ms** | 同步 | 无（try/except 包裹） |
| BGE Embedding (ChromaDB 写入前) | ~0.2ms (缓存命中) / ~5ms (miss) | 同步 | 嵌入缓存已实装，miss 也很少 |

总计：每次冷路径结果写回约 **15ms**，其中绝大部分（14ms+）是 ChromaDB 的 HNSW 图插入。

### 6.3 问题：同步写入阻塞

当前所有缓存写入都是**同步**的。15ms 虽然不大，但在高 QPS 下会累积。而且 ChromaDB 写入只有 36 QPS（之前 benchmark 结果），如果 20% 的流量进入冷路径（均值 115 QPS），**ChromaDB 写入会成为瓶颈**。

```
均值 578 QPS × 30% 冷路径 = 173 次写入/秒
ChromaDB 写入能力 = 36 QPS ← 严重不足
```

**解决方案**：改为异步写入。Action Agent 不需要等缓存写完才返回结果：

```python
# 改进方案：异步写入
async def action_executor(state):
    decision = state["decision"]
    # 立即返回结果
    result = {"action_taken": decision}

    # 后台异步写缓存（不阻塞返回）
    asyncio.create_task(_write_caches_async(state))

    return result
```

### 6.4 POC 缺失的功能

对照 CLAUDE.md 设计文档，Action Agent 在 POC 阶段缺失：

| 设计功能 | POC 状态 | 说明 |
|------|------|------|
| `block_content()` | 未实现 | 硬删除，通知平台执法层 |
| `shadow_remove()` | 未实现 | 软删除（对发布者可见，对他人隐藏） |
| `queue_human_review()` | 未实现 | 推送到人工审核队列 |
| `pass_content()` | 未实现 | 放行 + 递增用户信任分 |
| `notify_user()` | 未实现 | 发送通知（含违规原因和申诉链接） |
| `audit_log` | 部分实现 | 仅有 logger.info，未持久化到 Kafka/DB |
| 缓存写回 | 已实现 | L0a + L0b + L1 |

---

## 7. 冷路径容错与降级策略

### 7.1 各层故障模式

| 故障 | 当前处理 | 影响 |
|------|------|------|
| BERT 模型加载超时 (60s) | 抛异常 → ONNX 尝试 → 仍失败 → `label=unsafe, conf=0.5` → 强行走 L3 | 准确率下降，成本上升 |
| BERT ONNX 不可用 | 回退到 HF pipeline | 延迟从 17ms 升到 49ms（3x 慢） |
| LLM API 超时 (8s) | 回退到 BERT 结果 | 准确率略降，可用性不受影响 |
| LLM API 返回非 JSON | `json.loads()` 失败 → `label=safe, conf=0.5` | 高风险：违规内容可能被放行 |
| NSFW ViT 未下载 | `skip_model=True` → 所有图判 normal | POC 阶段无法验证图片准确率 |
| EasyOCR 未加载 | 返回空字符串 → OCR 文字为空 | 图片中嵌入的违规文字无法检测 |
| ChromaDB 写入失败 | try/except 静默跳过 | 该条结果不进语义缓存，后续相同内容仍需重新审核 |
| Redis 不可用 | 标记 `_available=False` → 静默跳过 | 回退到本地 TTLCache 单机模式 |
| Image download 超时 (10s) | 返回 `label=normal, conf=0.5` | 图片无法审核，降级为"假定正常" |

### 7.2 降级路径图

```
BERT 不可用:
  L2 BERT → L3 LLM（成本上升，准确率不变）

LLM API 不可用:
  L3 LLM → 回退 BERT（成本下降，准确率略降）

BERT + LLM 都不可用:
  L1 关键词作为最终决策（准确率显著下降，但可用）

所有模型不可用:
  Gateway 关键词 + 缓存（退化到纯规则模式）
```

### 7.3 问题：LLM JSON 解析失败的处理

当前代码在 `llm_audit.py:128` 中，如果 `json.loads()` 失败（LLM 返回了非 JSON 格式），整个 `audit()` 抛异常 → 返回 `label=safe, conf=0.5`。这意味着 **LLM 输出的格式错误会导致违规内容被放行**。

应该增加 JSON 修复逻辑：

```python
# 当前：JSON 解析失败 → 直接返回 safe
# 改进：尝试从非 JSON 输出中提取 label
try:
    result = json.loads(content)
except json.JSONDecodeError:
    # Try to extract from malformed JSON
    label = _extract_label_from_text(content)  # regex fallback
    return {"label": label, "confidence": 0.5, "reason": "JSON parse error"}
```

---

## 8. Multimodal Agent 缺口

### 8.1 设计意图

CLAUDE.md 设计的 Multimodal Agent 解决的是一个关键盲区：**图文矛盾**。

```
Text Agent 说 safe ──┐
                      ├── 矛盾！→ 升级 Multimodal Agent
Image Agent 说 NSFW ──┘

Text Agent 说 unsafe ──┐
                        ├── 矛盾！→ 升级 Multimodal Agent  
Image Agent 说 normal ──┘
```

当前 Decision Agent 的处理极其简单：图片 NSFW → label 直接升级为 unsafe。但反过来的情况（文本 unsafe + 图片 normal）没有特殊处理。

### 8.2 什么场景需要 Multimodal

| 场景 | Text | Image | 需要 Multimodal? |
|------|------|------|------|
| 纯违规文本 + 正常图 | unsafe 0.95 | normal 0.90 | **是** — 可能是表情包配文 |
| 正常文本 + 违规图 | safe 0.90 | NSFW 0.95 | 当前已处理（升级为 unsafe） |
| 两个都违规 | unsafe 0.90 | NSFW 0.85 | 不需要（结论一致） |
| 两个都正常 | safe 0.95 | normal 0.90 | 不需要 |
| 两个都不确定 | unsafe 0.50 | NSFW 0.45 | **是** — 需要图文联合理解 |

### 8.3 为什么 POC 没实现

POC 阶段范围限定为文本审核 + 基础图片。Multimodal Agent 需要 VLM（视觉语言模型，如 Qwen2.5-VL 或 InternVL），计算成本远高于纯文本 LLM。POC 的目标是验证三层漏斗（缓存+关键词+BERT+LLM）的可行性，Multimodal 留到 Phase 2。

### 8.4 当前过渡方案

Decision Agent 的加权聚合（刚实现的 P1-2）部分弥补了这个缺口：图片和文本的结果都会被纳入最终评分。但由于没有真正的 VLM 做图文联合推理，跨模态矛盾只能依赖规则处理。

---

## 9. 冷路径与设计文档的差异总结

| 设计组件 | 设计意图 | POC 实际 | 差距 |
|------|------|------|------|
| Triage Orchestrator | 路由中枢，缓存查询 + 关键词 + 模态分类 | 功能被 Gateway 替代，Triage 节点已删除 | 架构简化，功能完整 |
| Text Agent L1-L3 | 三层漏斗递进 | ✅ 完整实现 | 无 |
| Image Agent | NSFW + OCR + face_detect | NSFW 跳过 + OCR 实现 | 缺 face_detect + NSFW |
| Multimodal Agent | 图文矛盾联合审核 | 未实现 | Phase 2 |
| Decision Agent | 加权聚合 + 灰度区 + 可解释性 | 加权聚合已实现 + 灰度区已修复 | 缺 explainability generation |
| Action Agent | block/shadow/queue/notify/audit_log | 仅缓存写回 + 日志 | 缺所有执行动作 |
| Feedback Agent | 标注收集 + 微调触发 + 漂移检测 | 未实现 | Phase 2 |

---

---

## 10. 多语言 BERT 实测与结论

### 10.1 测试结果

用 15 条中文 + 8 条英文标注数据测试 `unitary/multilingual-toxic-xlm-roberta`：

| 语言 | 准确率 | Safe 准确率 | Unsafe 准确率 | L2 短路率 |
|------|------|------|------|------|
| 英文 | **100%** | 100% | 100% | 7/8 (87.5%) |
| 中文 | **33.3%** | 83.3% | **0%** | 0/15 (0%) |

中文 unsafe 全部被误判为 safe，置信度集中在 0.59-0.71（灰色区）。模型对中文没有区分能力。

### 10.2 根因

1. `unitary/multilingual-toxic-xlm-roberta` 基于 Jigsaw Multilingual Toxic Comment 数据集训练
2. 数据集中中文占比极低（< 2%），且主要是翻译文本而非原生中文
3. XLM-RoBERTa 的 SentencePiece tokenizer 对中文分词效果差（tokenizer regex 对 CJK 字符有已知 bug）
4. 模型对中文的策略是"不确定就判 safe"——所有置信度都在灰色区，不会误拦截但也不会有用

### 10.3 结论：回退策略

**POC 最优方案**：保持 `unitary/toxic-bert`（英文专用），中文跳过 BERT 直接走 LLM。

```
英文: toxic-bert → 100% 准确 → L2 短路率高 → 成本低
中文: 跳过 BERT → 直接 LLM → 成本高但准确
```

这个方案在"修复 P0"之前就是正确的。切换到多语言 BERT 是一次**过度修复**——把"中文跳过 BERT"当成 bug，但实际上是正确的设计决策。

多语言模型选择指南（生产阶段）：

| 模型 | 中文准确率 | 大小 | 推荐 |
|------|------|------|------|
| `unitary/multilingual-toxic-xlm-roberta` | ~30% | 280M params | ✗ 不推荐 |
| `KoalaAI/Text-Moderation` | 待测 | 7B (已缓存 2.7G) | 待验证 |
| `govtech/lionguard-2` | 待测 | 1.3B | 推荐尝试 |
| 自定义微调 Qwen2.5-1.5B 做二分类 | 可训练 | 1.5B | 最佳长期方案 |

### 10.4 全部候选模型测试结果

| 模型 | 中文准确率 | 中文 Unsafe | 可用 | 结论 |
|------|------|------|------|------|
| `unitary/toxic-bert` | N/A (跳过) | N/A | ✓ | **英文专用，中文跳过** |
| `unitary/multilingual-toxic-xlm-roberta` | 33.3% | **0%** | ✓ (需修复bug) | 中文无区分能力 |
| `KoalaAI/Text-Moderation` | 30.0% | **0%** | ✓ | 同上，所有中文判 OK=0.99 |
| `govtech/lionguard-2` | 无法加载 | - | ✗ | 自定义架构，不兼容 pipeline |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 未测试 | - | ✗ | 情感分析，非毒性分类 |

**结论**：缓存中的三个多语言模型，没有一个能有效检测中文违规内容。它们的训练数据中中文占比极低（< 2%），模型学到的策略是"不确定就判 OK"。这不是模型加载或配置的问题——是训练数据分布的问题。

### 10.5 正确的 POC 策略

```
英文 → toxic-bert → 100%准确, L2短路率高 → 低成本
中文 → 跳过BERT → 直接LLM → 成本高但准确率有保障
其他语言 → 跳过BERT → 直接LLM
```

这不是妥协，而是基于实测数据的最优解。用不准确的 BERT 做中文预判，不如不做。生产阶段的正确方案是：

1. 收集 5000-10000 条中文标注数据
2. 用 LoRA 微调 Qwen2.5-1.5B 做中文二分类（safe/unsafe）
3. 离线反馈飞轮持续积累标注 → 定期微调 → 准确率持续提升

### 10.4 修复的 Bug

修复了 `bert_classify.py` 中硬编码 `model.bert.encoder.layer` 的 bug——改为 model-agnostic 方式（尝试 bert/roberta/xlm_roberta/distilbert），避免非 BERT 架构模型在 warmup 时崩溃。

---

## 讨论记录

**2026-05-06**：多语言 BERT 实测

- XLM-RoBERTa 中文 unsafe 检测率 0%，确认不适合中文内容审核
- 回退 BERT 配置到 toxic-bert + 中文跳过策略
- 中文内容 100% 走 L3 LLM 是当前最优解（准确率 > 跳过 BERT 后的误判率）
- 生产方案：自定义微调 Qwen2.5-1.5B 做中文二分类，或使用专门的 safety model

**2026-05-06**：冷路径 Action Agent + 容错 + Multimodal 缺口分析

- Action Agent: 同步写入 ChromaDB（15ms）是瓶颈，建议改异步
- 容错：LLM JSON 解析失败直接返回 safe 是安全隐患
- Multimodal Agent: 图文矛盾盲区，当前加权聚合部分弥补
- POC vs 设计文档：Text Agent 最完整，Multimodal/Feedback/Action 执行动作缺失

**2026-05-06**：冷路径框架与技术分析

- 梳理了 4 节点 LangGraph DAG + 每层的技术栈
- Image Agent: requests + ViT NSFW + EasyOCR（POC 阶段 NSFW 被跳过）
- Text Agent: L1(AC自动机) → L2(BERT/ONNX) → L3(LLM) 三层漏斗
- Decision Agent: 4 条路径的规则引擎（缓存/零容忍/聚合/灰度区）
- Action Agent: 写回 L0a/L0b/L1 三层缓存
- 发现 5 个设计问题：入口点诡异、NSFW 不工作、图文矛盾无处理、BERT-LLM 矛盾无处理、加权聚合缺失

---

## 5. 冷路径关键问题深入分析

### 5.1 为什么 100% 的测试请求都走到了 L3 LLM？

全流程 benchmark 中 6/6 条文本全部走了 L3_llm。这是冷路径最大的成本问题。逐条分析原因：

| 文本 | 为什么走 L3 | 根本原因 |
|------|------|------|
| "today is a beautiful day" (EN) | BERT conf=0.999 → **不应该走 L3** | 需排查 `should_skip_llm` 逻辑 |
| "you are a worthless piece of shit" (EN) | BERT conf=0.985 ≥ 0.95 → **也不应该走 L3** | 同上 |
| "今天天气真好" (CN) | 中文检测 → 跳过 BERT → 直接 L3 | English-only BERT |
| "你真是个傻逼" (CN) | 中文检测 → 跳过 BERT → 直接 L3 | English-only BERT |
| "支持台独" (CN) | 中文检测 → 跳过 BERT → 直接 L3 | English-only BERT |
| "你说的也不是完全没道理" | BERT conf=? → 走了 L3 | 可能 conf < 0.95 |

**英文文本的 BERT 短路失效**是 bug。查看 text_agent.py 的 L2 决策逻辑：

```python
# text_agent.py:105
if not bert_skipped and bert_classifier.should_skip_llm(bert_result):
    # 返回 L2 结果，跳过 L3
    return {"text_result": {..., "tier": "L2_bert"}}

# should_skip_llm 的条件:
def should_skip_llm(self, result: dict) -> bool:
    if result.get("label") == "unsafe" and result.get("confidence", 0) >= 0.95:
        return True
    if result.get("label") == "safe" and result.get("confidence", 0) >= 0.95:
        return True
    return False
```

HF pipeline 对 "today is a beautiful day" 返回 `label=safe, confidence=0.999`，理应触发短路。但实际没有。

**问题可能出在**：`text_agent.py` 中调用 `should_skip_llm` 之前先做了 ONNX 尝试，而 ONNX 返回了 `label=unsafe, confidence=0.846`，不满足短路条件。然后代码即使 HF 修正了结果，也没有再检查 HF 的置信度。

看代码路径：
```python
# L2: 先尝试 ONNX
try:
    bert_result = bert_onnx.classify(text)     # ONNX: unsafe 0.846
except Exception:
    bert_result = bert_classifier.classify(text) # HF: safe 0.999

# 然后检查是否跳过 LLM
if not bert_skipped and bert_classifier.should_skip_llm(bert_result):
    # bert_result 是 ONNX 的结果 (unsafe 0.846)，不满足 ≥0.95
    # 所以不会短路，继续到 L3！
```

**这就是根因**：ONNX 成功返回了（没有抛异常），所以 `bert_result` 是 ONNX 的结果。ONNX 的置信度偏低（0.5-0.85），不满足 0.95 阈值。代码没有在 ONNX 成功后对比 HF 的结果。

### 5.2 中文问题：English-only BERT 的连锁反应

当前 BERT 模型 `unitary/toxic-bert` 只在英文数据上训练。对中文文本：
1. BERT 输入中文 token → tokenizer 产生大量 `[UNK]` → 输出无意义
2. 代码检测到 CJK 字符 > 30% → 跳过 BERT → 直接 L3
3. 中文文本 100% 走 LLM → 成本飙升

**量化影响**：如果业务 70% 是中文内容，按当前设计：
```
英文 30%: 其中 50% BERT 高置信拦截 → 15% 走 L3
中文 70%: 100% 走 L3 → 70% 走 L3
总 L3 调用率: 85% ← 远超 20% 的设计目标！
```

**解决方案**：

| 方案 | 效果 | 成本 |
|------|------|------|
| 换多语言 BERT (XLM-RoBERTa) | 中文也能走 L2 | 已有缓存 `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| 用安全专用模型 (LionGuard-2) | 直接替代 BERT+LLM | 已有缓存 `govtech/lionguard-2` (1.3B) |
| 微调 Qwen2.5-1.5B 做分类 | 最高准确率 | 需要标注数据 + GPU 微调 |

最简单的改进：把 BERT 模型换成已缓存的多语言版本，配置改一行：
```python
# config.py
BERT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
```

### 5.3 BERT 置信度阈值：0.95 是否合理？

阈值 0.95 是经验值，没有数据支撑。过高或过低的影响：

| 阈值 | BERT 拦截率 | L3 调用率 | 风险 |
|------|------|------|------|
| 0.99 | 极低 (<5%) | 极高 | 几乎不省钱 |
| 0.95 (当前) | 低 (~10%) | 高 (~90%) | 英文效果好但中文无效 |
| 0.90 | 中 (~25%) | 中 (~75%) | 可能放过一些边缘违规 |
| 0.85 | 较高 (~40%) | 较低 (~60%) | 误拦截风险上升 |
| 0.70 | 高 (~60%) | 低 (~40%) | 准确率显著下降 |

需要对照真实标注数据做 ROC 曲线才能确定最优阈值。POC 阶段缺少足够的标注数据做这个分析。

### 5.4 系统提示词分析

当前 `SYSTEM_PROMPT`（llm_audit.py:12-28）的几个问题：

**问题 1：缺乏示例（few-shot）**

当前是 zero-shot 提示词，只列出了分类名称。加上 1-2 个示例可以大幅提升准确率：

```
# 当前（zero-shot）
Categories to check:
- toxic: insults, hate speech, harassment
- politics: politically sensitive content
...

# 改进（few-shot）
Examples:
"你真是个傻逼，什么都不懂" → {"label": "toxic", "confidence": 0.95, "reason": "直接辱骂，使用脏话攻击他人"}
"今天天气真好，适合出去玩" → {"label": "safe", "confidence": 0.98, "reason": "日常寒暄，无违规内容"}
```

**问题 2：分类体系没有优先级**

当一段文本同时包含 toxic + politics（如"操你妈的台独分子"），模型如何选择？当前提示词没有说。应该加：

```
If multiple categories apply, choose the most severe one.
Severity order: politics > violence > porn > gambling > toxic > spam > safe
```

**问题 3：confidence 的锚定不清晰**

当前只说 `<0.0 to 1.0>`，没有说明什么情况给什么分：

```
Confidence guidelines:
- 0.95-1.0: Absolutely certain (explicit keywords, clear intent)
- 0.80-0.95: High confidence (strong indicators present)
- 0.50-0.80: Moderate (some indicators but could go either way)
- 0.30-0.50: Low confidence (subtle hints, needs human review)
```

### 5.5 加权聚合缺失的影响

CLAUDE.md 设计了 `score_aggregate` 逻辑（Text 40% + Image 35% + Multimodal 25%），但当前 Decision Agent 完全没有实现。

当前行为：
```python
# decision.py — 简单地把 text_result 作为最终结果
label = text_result["label"]       # 完全忽略 image_result
confidence = text_result["confidence"]
```

这意味着：**如果图片被 NSFW 模型判为 0.99 违规，但文本被判为 safe 0.90，最终结果会是 pass。** 图片审核的结果被忽略了（除非是纯图片无文本的情况）。

正确的加权逻辑应该是：
```python
def score_aggregate(text_result, image_result, multimodal_result=None):
    scores = []
    weights = []

    if text_result:
        scores.append(text_result["score"])
        weights.append(0.40)
    if image_result:
        scores.append(image_result["score"])
        weights.append(0.35)
    if multimodal_result:
        scores.append(multimodal_result["score"])
        weights.append(0.25)

    # 归一化权重
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    final_score = sum(s * w for s, w in zip(scores, weights))
    return final_score
```

但当前没有 Multimodal Agent，所以实际权重会是 Text 53% + Image 47%（归一化后）。

### 5.6 冷路径延迟构成

全流程 benchmark 中 P50=1241ms。各步骤延迟拆解：

| 步骤 | 延迟 | 占比 | 可优化 |
|------|------|------|------|
| L1 关键词 (AC自动机) | < 1ms | 0% | 已最优 |
| L2 BERT 推理 | ~50ms | 4% | ONNX 可降到 17ms |
| **L3 LLM API 调用** | **1000-1800ms** | **80-95%** | 本地 LLM 可降到 200ms |
| Decision 规则 | < 1ms | 0% | 已最优 |
| Action 缓存写回 | ~15ms | 1% | 批量异步写入 |

**瓶颈极其明确：LLM API 的网络延迟 + 模型推理时间。** 将 LLM 本地化（方案 C：llama.cpp + Qwen2.5-1.5B）可以将冷路径延迟从 1241ms 降到 ~300ms（4x 加速）。

### 5.7 改进优先级

| 优先级 | 改进项 | 影响 | 工作量 |
|------|------|------|------|
| P0 | 修复 BERT 短路失效（ONNX 优先导致） | 英文 L3 调用率降低 50% | 10 行 |
| P0 | 换多语言 BERT | 中文不再 100% 走 L3 | 1 行配置 |
| P1 | 系统提示词加入 few-shot | LLM 准确率 +10-15% | 20 行 |
| P1 | 实现加权聚合 | 图片审核结果不被忽略 | 30 行 |
| P2 | 本地 LLM 替代 API | 延迟降 4x，成本降为 0 | 50 行 + 模型下载 |
| P2 | BERT 阈值数据驱动调优 | 找到最优平衡点 | 需要标注数据 |

- 确认 4 层技术栈：L0 SHA256 + L1a AC 自动机独立词 + L1b AC 嵌入词 + L2 ChromaDB
- 讨论 AC 自动机 vs 正则 vs Bloom Filter 的取舍
- 讨论独立词/嵌入词二分法的合理性和局限
- 讨论 ChromaDB 选型（POC）和 milestone 迁移路径
- 发现的问题：pyahocorasick 未安装时回退到暴力匹配、白名单正则在大规模下的性能退化、BGE 模型首次加载延迟

---

## 附录：2026-05-05 修复记录

5 个问题中的 4 个已在 POC 阶段修复，1 个（Gateway 整合回 LangGraph）留到 Phase 2。

### 修复 1：删除 Triage 死代码

- 删除了 `poc/src/agents/triage.py`（105 行，无任何调用方）
- graph.py 入口保持为 `image_agent`，路由逻辑完全由 Gateway 承担

### 修复 2：消除关键词重复扫描

- `state.py` 新增 `keyword_prefiltered: bool` 字段
- Gateway 在所有路径（命中/升级）都返回 `keyword_prefiltered` 状态
- `text_agent.py` L1 层检查 `keyword_prefiltered`：为 True 时跳过 AC 自动机扫描，仅记录 trace
- 冷路径请求不再对同一段文本扫描两次关键词

### 修复 3：Gateway 冷路径 trace 不丢失

- `gateway.check()` 返回值从 `dict | None` 改为始终返回 `dict`：
  ```python
  # 旧接口
  gw_result = gateway.check(...)  # dict | None
  if gw_result is not None: ...   # 热路径
  # 冷路径：所有 gateway trace 丢失

  # 新接口
  gw = gateway.check(...)         # 始终返回 dict
  if gw["decision"] is not None: ...  # 热路径
  # 冷路径：gw["traces"] 仍然可用
  ```
- `api.py` 冷路径响应中合并 Gateway traces + LangGraph traces
- 流式接口和批量接口同步修复

### 修复 4：图片 URL 哈希去重

- Gateway 新增 `_hash_image_ref()` 方法，对图片 URL/base64 做 SHA256 哈希
- 在 L0 内存缓存中以 `[IMG]{hash}` 为 key 查询，相同图片引用可走热路径缓存
- 首次出现的图片仍升级到冷路径（POC 阶段不做 perceptual hash 去重）

### 修复 5：Gateway 膨胀

- **POC 阶段不拆分**。Gateway 154 行代码，职责清晰，拆分收益不大
- 将类内方法按职责分为三组（public API / helpers / stats），用 section comment 分隔
- 评估标准：当 Gateway 超过 300 行或需要独立测试缓存/关键词逻辑时，再拆分为 `KeywordGate` + `CacheGate`

### 变更文件清单

| 文件 | 变更 |
|------|------|
| `src/agents/triage.py` | 删除 |
| `src/gateway.py` | 重写：新返回格式 + 图片哈希 + section 注释 |
| `src/state.py` | 新增 `keyword_prefiltered` 字段 |
| `src/api.py` | 适配新 gateway 接口，合并 cold path traces |
| `src/agents/text_agent.py` | L1 检查 `keyword_prefiltered` 跳过重复扫描 |
| `tests/test_e2e.py` | 移除 pytest 硬依赖，补充 `keyword_prefiltered` |
| `eval/benchmark.py` | 补充 `keyword_prefiltered` 字段 |

### 验证

- `check_env.py`：10/10 项通过
- `tests/test_e2e.py`：5/5 项通过
- 手动 6 项 sanity check：全部通过
