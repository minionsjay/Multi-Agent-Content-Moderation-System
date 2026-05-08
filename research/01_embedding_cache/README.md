# 方案 00: Embedding 结果缓存

## 问题

BGE Embedding 是热路径唯一瓶颈（单核 200 QPS）。在多服务场景下，大量文本是重复的：

```
"加微信买六合彩稳赢不赔发财机会难得"  ← 同一个垃圾广告发了 10,000 次
                                          → 每次都要 BGE 推理 → 浪费 10,000 × 5ms = 50 秒
```

**即使文本完全相同，当前代码也会重复调用 BGE 模型。**

## 原理

在 BGE 模型之前加一层 LRU 缓存（TTLCache），key 是文本的 SHA256 哈希：

```
embed(text):
    key = SHA256(text)
    if key in cache:
        return cache[key]   ← < 0.01ms，跳过 5ms BGE 推理
    vec = BGE_model.encode(text)
    cache[key] = vec
    return vec
```

## 为什么 SHA256 而不是直接用 text 做 key

直接用 text 做 key，100 万条文本的 dict 内存占用约 200-500 MB（文本本身很大）。
SHA256 hex 固定 64 字节/key，内存占用可预测且小得多。

## 内存估算

```
缓存容量: 500,000 条
每条目: 64 字节 (SHA256 key) + 512 × 4 字节 (float32 vector) = 2,112 字节
总内存: 500,000 × 2,112 ≈ 1 GB
```

对于日均 5000 万条的系统，500K 条缓存覆盖了约 1% 的独特文本。实际命中率取决于重复率。

## 命中率分析

| 流量特征 | 预期命中率 | 场景 |
|----------|------|------|
| 完全随机文本 | < 1% | 无重复，缓存无效 |
| 正常用户评论 | 10-20% | 常用短语、问候语重复 |
| 垃圾广告刷屏 | 60-80% | 同一广告大量复制 |
| 模板化内容 | 30-50% | "好评返现"、"踩踩" 等 |

## 基准测试

```bash
python bench_embed_cache.py          # 完整测试 (1K/5K/10K)
python bench_embed_cache.py --quick  # 快速冒烟测试
```

预期结果：
- 无缓存：每 1000 条 ~5000ms (5ms × 1000)
- 有缓存（60% 重复率）：每 1000 条 ~2000ms (只算 400 条未命中)
- 加速比：~2.5x

## 代码实现

已在 `poc/src/skills/embedder.py` 中实现。核心改动：

```python
# embedder.py
from cachetools import TTLCache
import hashlib

class Embedder:
    def __init__(self, ...):
        ...
        self._emb_cache = TTLCache(maxsize=500_000, ttl=3600)

    def embed(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode()).hexdigest()
        cached = self._emb_cache.get(key)
        if cached is not None:
            return cached
        vec = self._model.encode(text[:8191], normalize_embeddings=True).tolist()
        self._emb_cache[key] = vec
        return vec
```

## 局限性

1. **进程重启丢失**：TTLCache 在内存中，重启后冷启动
2. **进程间不共享**：多 worker 各自维护缓存，无法受益于其他 worker 的计算
3. **内存上限**：500K 条 ≈ 1GB，超过后 LRU 淘汰旧条目
4. **对真正独特的内容无效**：如果每条文本都是唯一的，缓存命中率为 0

## 下一步

方案 01（多 Worker）+ 方案 04（Redis 共享缓存）可以解决进程间不共享和重启丢失的问题。
