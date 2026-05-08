# 方案 01: FastAPI 多 Worker

## 问题

单进程 Python 受 GIL（全局解释器锁）限制。BGE 模型在 CPU 上推理时，每次 `encode()` 调用都会持有 GIL（虽然 PyTorch 内部会释放，但 Python 层仍有开销）。一核跑满，其余核闲置：

```
进程 1 (核 1): BGE Embedding → 200 QPS
核 2, 3, 4:   空闲
                ↑ 浪费
```

## 原理

Uvicorn 的 `--workers N` 启动 N 个独立子进程，每个进程有独立的 Python 解释器、独立的 GIL、独立的 BGE 模型实例：

```
              Nginx :80
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 Worker 1     Worker 2     Worker 3     Worker 4
 (核 1)       (核 2)       (核 3)       (核 4)
 BGE ×1       BGE ×1       BGE ×1       BGE ×1
 cache ×1     cache ×1     cache ×1     cache ×1
 L0 ×1        L0 ×1        L0 ×1        L0 ×1
                          
 总计: 4 × 200 = 800 QPS embedding 吞吐
```

**为什么 Uvicorn 多 Worker 有效而 Python 多线程无效？**

Python 的 `threading` 受 GIL 限制，同一时刻只有一个线程执行 Python 字节码。即使开 4 个线程调用 BGE，同一时刻也只有一个在跑。

`--workers` 启动的是独立**进程**（用 `os.fork()` 或 `subprocess`），每个进程有自己的 GIL。4 个进程真正并行跑在 4 个核上。

## 配置

```bash
# 单 worker (当前 POC 默认)
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 4 worker (生产推荐)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

# 自动检测 CPU 核数
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers $(nproc)
```

Kubernetes 部署时，worker 数应与容器的 CPU limit 匹配：

```yaml
# Pod with 4 CPUs
resources:
  limits:
    cpu: "4000m"
    memory: "8Gi"
env:
  - name: UVICORN_WORKERS
    value: "4"
```

## 效果预估

| Worker 数 | BGE QPS | 总内存 | L0 缓存命中率 |
|------|------|------|------|
| 1 | 200 | ~1.5 GB | 100% |
| 2 | 400 | ~2.5 GB | ~50% (轮询导致) |
| 4 | 800 | ~4.5 GB | ~25% (轮询导致) |
| 8 | 1,600 | ~8.5 GB | ~12.5% |

> 注：内存中包含 Python 基础开销 (~200MB/进程) + BGE 模型 (95MB/进程) + 各类缓存。

## 副作用：L0 缓存命中率下降

每个 Worker 的 TTLCache 是进程独立的。Nginx/负载均衡器默认用 round-robin 分发请求：

```
请求 1: "加微信买六合彩" → Worker-1 → 未命中 → Embedding → 写入 Worker-1 缓存
请求 2: "加微信买六合彩" → Worker-3 → 未命中 → 又要 Embedding！
                                    ↑ 没有命中 Worker-1 的缓存
```

**缓解措施**：

1. Nginx 用 `ip_hash` 分发 —— 同一用户/IP 的请求始终到同一 Worker。对重复提交有效，但对批量刷屏（不同 IP 发相同内容）无效。
2. 方案 04（Redis 共享缓存）—— 彻底解决。所有 Worker 共享同一个 Redis，缓存命中率恢复 100%。

## 副作用：启动时间 × N

每个 Worker 独立启动，各自加载 BERT、BGE、ChromaDB。4 个 Worker = 4 次模型加载 = 4 倍启动时间。在 K8s 中应配置足够长的 `initialDelaySeconds`：

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 90  # 给足模型加载时间
  periodSeconds: 10
```

## 验证脚本

```bash
# 启动服务器
python -m src.api &
sleep 5

# 检查运行的进程数
curl http://localhost:8000/health
# 如果只有 1 个进程：ps aux | grep uvicorn
# 用 --workers 4 启动后应该有 1 个主进程 + 4 个子进程

# 并发压测
pip install wrk
wrk -t4 -c100 -d30s --latency \
  -s research/02_multi_worker/wrk_test.lua \
  http://localhost:8000/moderate
```

## 与方案 00（Embedding 缓存）的结合

两个方案是互补的：

- **方案 00** 让每个 Worker 内部的重复文本不吃 BGE 推理
- **方案 01** 让多个 Worker 并行处理，提升总吞吐

```
单 Worker 无缓存:    200 QPS
单 Worker + 缓存:    300 QPS (方案 00 单独效果)
4 Worker 无缓存:     800 QPS (方案 01 单独效果)
4 Worker + 缓存:    1200 QPS (两者叠加)
```
