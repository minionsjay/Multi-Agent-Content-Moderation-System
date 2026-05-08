# 06 Gateway Full Flow

## 测试什么

将前 5 个测试的每一层串联起来，测试完整的 Gateway 热路径流程：

```
text → L0 Memory Cache → AC Automaton → ChromaDB Cache → 返回或升级
```

测试项覆盖：各层延迟分解、流量分布、整体吞吐量。

## 测试项

### Test 1: 各层延迟分解

每种请求类型，对应走 Gateway 的不同路径：

| 请求类型 | 总延迟 | 内存缓存 | 关键词 | ChromaDB | 走向 |
|------|------|------|------|------|------|
| 空文本 | 0.00ms | 0.00 | 0.00 | - | Hot (pass) |
| 关键词拦截 | 0.22ms | 0.03 | **0.18** | - | Hot (block) |
| 英文关键词 | 0.17ms | 0.01 | **0.16** | - | Hot (block) |
| 缓存命中 | 1.77ms | 0.01 | 0.00 | **1.74** | Hot (pass) |
| 白名单放行 | 1.91ms | 0.02 | 0.01 | **1.86** | Hot (pass) |
| 干净文本（命中） | 1.66ms | 0.02 | 0.01 | **1.61** | Hot (pass) |
| 干净文本（升级） | 1.61ms | 0.03 | 0.01 | **1.55** | Cold (escalate) |

**关键发现**：

- **关键词拦截最快**（0.17-0.22ms）：不需要 embedding，不需要 ChromaDB
- **ChromaDB 是热路径主要延迟来源**（1.5-1.9ms）：即使 miss 也要先做 embedding + 查询
- **内存缓存按理应该最快**（< 0.01ms），但测试中显示为 1.77ms —— 这是因为"缓存命中"的请求实际上走的是 ChromaDB 缓存路径（第一次调用的文本被 ChromaDB 缓存了，第二次走 ChromaDB 命中了）

### Test 2: 流量分布（500 条混合流量）

模拟包含正常评论 + 辱骂 + 重复模板的混合流量：

| 结果 | 数量 | 占比 |
|------|------|------|
| Hot path (block) | 100 | 20.8% |
| Hot path (pass) | 60 | 12.5% |
| Cold escalated | 320 | 66.7% |
| **Hot path total** | **160** | **33.3%** |

**注意**：本测试中 cold escalated 比率很高（66.7%），因为测试数据主要生成了随机唯一文本（带随机数后缀 `[{random}]`），这些文本极少重复，所以 L0 缓存命中率低、ChromaDB 缓存也找不到。

在真实生产流量中（有大量模板化内容、垃圾广告批量复制），hot path 比率预计 50-70%。

### Test 3: 吞吐量

1200 条文本，顺序处理：

| 指标 | 值 |
|------|------|
| 总请求 | 1,200 |
| 总耗时 | 1.24s |
| **QPS** | **968 req/s** |
| Avg latency | 1.03ms/req |

**Gateway Stats（累计）**：

| 指标 | 值 |
|------|------|
| Hot path rate | 69.0% |
| Memory cache hit | 0.0% |
| Keyword hit | 29.7% |
| Whitelist hit | 0.0% |
| ChromaDB hit | 39.3% |
| Escalated | 30.9% |
| LangGraph calls saved | 1,170 |

**结论**：单核 Gateway 吞吐约 **968 QPS**。内存缓存命中率为 0 是因为之前测试中生成的唯一文本没有重复。关键词拦截 29.7% + ChromaDB 缓存 39.3% = 69% 热路径率。30.9% 升级到冷路径（LangGraph）。

## 实际意义

在日均 5000 万条的场景下（均值 578 QPS），一个 968 QPS 的 Gateway 实例刚好够用（1.67x 余量）。峰值 5,780 QPS 时需要 6 个 API Worker（通过方案 01 多 Worker 实现）。
