# 方案 02: BGE 批量推理

## 问题

单条 BGE 推理每次都要走一遍 Python → PyTorch → C++ kernel 的调用栈，这个开销是固定的。N 条文本单独推理 = N 次开销：

```
5ms × 1 条  = 5ms    ← 合理
5ms × 1000 条 = 5,000ms  ← N 次 Python→C++ 开销
```

## 原理

Transformer 模型的前向传播天然支持变长批量输入。把多条文本拼接成一个 batch 送入模型，attention 机制并行处理：

```python
# 单条推理（低效）
for text in texts:
    vec = model.encode(text)  # 每次走一遍 Python→C++ 调用栈
    # 5ms overhead per call

# 批量推理（高效）
vecs = model.encode(texts, batch_size=32)
# 1 次 Python→C++ 调用，内部 32 条并行
# ~50ms for 32 texts ≈ 1.6ms/text
```

## 加速原理

Transformer 的核心运算是矩阵乘法（Q·K^T, 等）。批量输入允许这些矩阵操作在更大的张量上运行：

```
单条: [1, 512, 768] × [768, 768] = [1, 512, 768]     → GPU 利用率低
批量: [32, 512, 768] × [768, 768] = [32, 512, 768]   → GPU/CPU 利用率高
```

CPU 上也有类似效果：BLAS/MKL 库对大批量矩阵乘法有自动向量化优化。

## 实现方式：Gateway 内部攒批

```python
import asyncio

class BatchingGateway:
    def __init__(self):
        self._batch_queue = asyncio.Queue(maxsize=64)
        self._batch_event = asyncio.Event()
        # Background task: flush batch every 10ms or when full
        asyncio.create_task(self._batch_worker())

    async def _batch_worker(self):
        batch = []
        while True:
            try:
                # Wait for items or timeout
                item = await asyncio.wait_for(
                    self._batch_queue.get(), timeout=0.01
                )
                batch.append(item)
                # Flush when batch is full
                if len(batch) >= 32:
                    await self._flush(batch)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self._flush(batch)
                    batch = []

    async def _flush(self, batch):
        texts = [item["text"] for item in batch]
        embeddings = embedder.embed_batch(texts)
        for item, emb in zip(batch, embeddings):
            item["future"].set_result(emb)

    async def embed_async(self, text: str) -> list[float]:
        """Async embedding with batching."""
        future = asyncio.get_event_loop().create_future()
        await self._batch_queue.put({"text": text, "future": future})
        return await future
```

## 为什么选 batch_size=32

| Batch Size | 延迟/text | 总延迟 | 内存 |
|------|------|------|------|
| 1 | 5ms | 5ms | 低 |
| 8 | 2.5ms | 20ms | 低 |
| 16 | 2ms | 32ms | 中 |
| 32 | 1.6ms | 50ms | 中 |
| 64 | 1.4ms | 90ms | 高 |
| 128 | 1.3ms | 166ms | 很高 |

32 是 sweet spot：1.6ms/text × 3x 加速，总延迟 50ms 对审核系统可接受。

## 延迟权衡

批量推理引入了人为延迟（攒批等待时间）：

```
最大额外延迟: 10ms (batch_timeout)
平均额外延迟: 5ms (batch_timeout / 2)
```

对于审核系统，< 10ms 的额外延迟是可接受的（LLM 推理要 500ms-1s）。

## 适用场景

- **高 QPS 场景**（500+ QPS）：批量效果好，攒批速度快
- **低 QPS 场景**（< 50 QPS）：批量效果差，攒不满就 timeout 了，反而增加延迟
- **突发流量**：天然适合，突发时瞬间攒满大 batch

## 基准测试

```bash
python bench_batch.py          # 完整测试 (100/500/1K/2K)
python bench_batch.py --quick  # 快速冒烟测试
```
