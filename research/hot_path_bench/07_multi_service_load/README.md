# 07 Multi-Service Load Test

## 测试什么

模拟 5 个不同业务线同时向审核系统发请求，每个服务的流量特征不同：

| Service | 类型 | 正常 | 违规 | 垃圾 | 政治 |
|------|------|------|------|------|------|
| social_comments | 社交评论 | 80% | 15% | 5% | - |
| live_chat | 直播弹幕 | 60% | 30% | 10% | - |
| product_reviews | 商品评价 | 90% | 5% | 5% | - |
| forum_posts | 论坛帖子 | 70% | 10% | 10% | 10% |
| bot_spam | 机器人刷屏 | 30% | 20% | 50% | - |

目的：验证热路径在不同流量模式下的表现，发现瓶颈点。

## 测试配置

```
5 services × 500 requests = 2,500 total
Concurrency: 20 (asyncio.Semaphore)
Hardware: 6.7 GB RAM, no GPU, single process
```

## 结果

### 总体指标

| 指标 | 值 |
|------|------|
| Total time | **11.02s** |
| Total requests | **2,500** |
| Throughput | **227 req/s** |
| Concurrency | 20 |

### 流量分布

| 路径 | 数量 | 占比 |
|------|------|------|
| Hot block | 484 | 19.4% |
| Hot pass | 44 | 1.8% |
| Cold escalated | 1,972 | 78.9% |
| **Hot path total** | **528** | **21.1%** |

### 延迟分布

| 分位 | 延迟 |
|------|------|
| P50 | **1.55ms** |
| P95 | **14.03ms** |
| P99 | **16.34ms** |
| Avg | **4.41ms** |
| Max | **617.00ms** |

P50 的 1.55ms 说明一半的请求在 1.55ms 内完成（热路径）。Max 的 617ms 来自首次 BGE 模型加载的冷启动（测试中约在 batch 中间发生了一次模型重载）。

### 各服务表现

| Service | Requests | Hot Path % | Avg Lat | P99 Lat |
|------|------|------|------|------|
| social_comments | 500 | 23.4% | 5.38ms | 15.85ms |
| live_chat | 500 | 15.6% | 4.43ms | 15.78ms |
| product_reviews | 500 | 0.8% | 5.16ms | 16.72ms |
| forum_posts | 500 | 12.6% | 4.76ms | 19.49ms |
| bot_spam | 500 | **53.2%** | **2.30ms** | 14.63ms |

**关键发现**：

- **bot_spam 的热路径率最高（53.2%）**，延迟最低（2.30ms）：大量重复垃圾广告被 L0 缓存和 ChromaDB 缓存命中
- **product_reviews 最低（0.8%）**：商品评价模板多样且没有关键词，几乎全部升级到冷路径
- **live_chat 的延迟合理（4.43ms）**：短文本占主导，embedding 计算快

### Gateway 统计

| 指标 | 值 |
|------|------|
| Memory cache hit | 0.0% |
| Keyword hit | **19.4%** |
| Whitelist hit | 0.0% |
| ChromaDB hit | **1.8%** |
| Escalated | **78.1%** |

### 瓶颈分析

```
BGE Embedding cache hit rate: 68.7%
单核 BGE QPS limit:          ~200
当前 throughput:              227 req/s
Embedding QPS needed:         183 req/s (227 × 81.2%)
需要 Worker 数 (CPU):         1 (183 / 200 = 0.92)
```

**结论**：当前单核刚好覆盖 5 服务 × 500 条的测试场景。Embedding QPS 需求（183）略低于单核上限（200），但有风险：

1. 如果峰值流量 10x（5,780 QPS），需要 **29 个 CPU 核**
2. 如果 ChromaDB 缓存写入量增大（当前 1.8% 命中率很低，缓存数据少），写入延迟会拖慢整体
3. Max 延迟 617ms 来自冷启动 —— 生产环境需预热模型

## 生产环境推算

| 场景 | QPS | 需要 API Worker | 需要 BGE 核 |
|------|------|------|------|
| 当前测试 | 227 | 1 | 1 |
| 50M/天均值 | 578 | 1 | 3 |
| 50M/天峰值 (10x) | 5,780 | 6 | 29 |
| 50M/天突发 (20x) | 11,560 | 12 | 58 |

BGE Embedding 是**唯一瓶颈**。方案 01（多 Worker）+ 方案 03（ONNX 加速）可以将需要的 CPU 核数从 29 降到 10 核，从 58 降到 20 核。

## 实际意义

多服务负载测试证明了：

1. **热路径确实能拦截大量流量**（bot_spam 53% 热路径率），不同业务线差异大
2. **延迟可控**：P50=1.55ms，P99=16ms，都在可接受范围
3. **单核刚够用**：227 QPS 单核跑满，峰值必须扩容
4. **Embedding 缓存有效**：68.7% 的 embedding 请求被缓存拦截
