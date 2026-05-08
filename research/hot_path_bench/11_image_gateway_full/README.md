# 11 Image Gateway Full Flow

## 测试什么

图片热路径全流程端到端测试：

```
Image → I1: dHash → known DB check
     → I2: pHash → memory cache
     → I3: URL exact match
     → 全部未命中 → escalate 冷路径
```

## 测试结果

### Test 1: 各路径延迟分解

| 请求类型 | 延迟 | 走向 | 链路 |
|------|------|------|------|
| 已知有害图片 | **0.93ms** | Hot | [image_phash_block] |
| 正常新图片 | **0.98ms** | Cold | [image_phash_ok, image_escalate] |
| 纯文本 | 12.76ms | Cold | [L0_miss, Redis_miss, KW_miss, Chroma_miss] |
| 纯 URL (无 base64) | 36.27ms | Cold | [image_phash_ok, image_escalate] |

**关键发现**：
- 已知有害图片在 0.93ms 内被 hot-block 拦截 —— dHash 匹配 + 直接返回
- 正常图片在 0.98ms 内完成全链路检查并升级 —— 开销仅 dHash 计算
- 纯 URL（无 base64 bytes）无法计算 dHash → 直接 escalate，但仍需 36ms（含图片 URL 哈希缓存查询 + ChromaDB embedding）

### Test 2: 流量分布

300 张图片（5% 已知有害，30% 重复，65% 全新）：

| 路径 | 数量 | 占比 |
|------|------|------|
| Hot block (dHash known DB) | 15 | 5.0% |
| Hot pass (pHash cache) | 0 | 0.0% |
| Cold escalate | 285 | 95.0% |

pHash cache 命中率 0% 是因为测试图片全部为随机生成的唯一图案，没有真正"相同视觉内容"的变体。在真实场景中（同一违规图片换 CDN 反复传播），I2 层应该能拦截 10-30%。

### Test 3: 吞吐量

```
200 requests: 0.19s
QPS: 1,080 req/s
Avg: 0.93ms/req
```

图片热路径吞吐 **1,080 QPS**，与纯文本 Gateway（968 QPS）相当。瓶颈都是 BGE embedding，图片热路径不涉及 embedding（dHash 是纯 Pillow 操作）。

### Test 4: 文本 vs 图片对比

| 路径 | 延迟 |
|------|------|
| 文本关键词拦截 (hot) | 6.37ms |
| 图片 dHash escalate (cold) | **0.95ms** |

图片热路径比文本还快，因为：
- 文本热路径可能要走 L0 → Redis → Keyword → ChromaDB（多层 miss 累积延迟）
- 图片热路径只做一次 dHash，然后直接 escalate（如果不在 known DB 中）

## 发现的问题

测试中暴露了一个边界 case：Gateway 对 URL-only（无 base64 bytes）的图片处理效率低 —— 尝试 `base64.b64decode("no_base64_here")` 会解码出垃圾数据，然后 PIL 在尝试打开时失败。这导致不必要的 decode 尝试。

**修复建议**: 在调用 b64decode 前做 base64 格式校验，或使用 `validate=True` 参数。
