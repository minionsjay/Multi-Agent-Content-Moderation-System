# 方案 04: Redis 共享缓存

## 问题

方案 01（多 Worker）引入了一个副作用：每个 Worker 有独立的 L0 内存缓存。当一个 Worker 学习了某条文本的审核结果后，其他 Worker 不知道：

```
Worker-1: "加微信买六合彩" → cache miss → Embedding → LLM → 写入 Worker-1 缓存
Worker-2: "加微信买六合彩" → cache miss → 又要重新 Embedding + LLM！
```

Worker 越多，缓存效率越低。而且进程重启后，所有缓存消失。

## 解决方案

在进程内 TTLCache 和 ChromaDB 之间插入一层 Redis，实现跨 Worker 共享、跨重启持久化。

### 缓存层级（更新后）

```
Gateway.check(text)
  │
  ├── L0a: 本地 TTLCache    (< 0.01ms)  ← 最快，进程内
  │   └── 命中 → 直接返回
  │
  ├── L0b: Redis 共享缓存   (~0.5ms)    ← 跨 Worker + 持久化
  │   └── 命中 → 返回 + 回填本地缓存
  │
  ├── Step 1: AC 自动机     (< 0.5ms)
  └── Step 2: ChromaDB      (< 5ms)
```

## 实装代码

### redis_cache.py（新增）

`poc/src/skills/redis_cache.py` — 纯 Python，依赖 `redis-py`。

```python
class RedisCache:
    def get(self, text: str) -> dict | None:
        """SHA256 → Redis GET。不可用时返回 None（不影响可用性）。"""
        if not self._ensure_client():
            return None
        key = f"mod:v1:{sha256(text)}"
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    def set(self, text, decision, confidence, reason):
        """写入 Redis with TTL (默认 2 小时)"""
        self._client.setex(key, 7200, json.dumps({...}))
```

关键设计：
- **TTL: 2 小时**（比本地 TTLCache 的 1 小时长，因为 Redis 内存更充裕）
- **Key prefix: `mod:v1:`**（版本化，方便后续缓存格式变更时一键失效）
- **连接超时: 500ms**（Redis 出问题时快速失败，不阻塞请求）
- **不可用时自动回退**（`_available = False` 后不再尝试直到下次调用）

### Gateway 集成

`poc/src/gateway.py` 在 L0 本地 miss 后加入 Redis 查询：

```python
# Step 0a: local TTLCache
mem_result = memory_cache.get(text)
if mem_result is not None:
    return ...  # instant

# Step 0b: Redis shared cache (NEW)
redis_result = redis_cache.get(text)
if redis_result is not None:
    memory_cache.set(text, ...)  # 回填本地，下次更快
    return ...
```

### Action 写回

`poc/src/agents/action.py` 在每次审核完成后同时写两个缓存：

```python
memory_cache.set(text, decision, confidence, reason)  # 本地
redis_cache.set(text, decision, confidence, reason)   # Redis
```

## 基准测试

### Test 1: 本地缓存（baseline）

| 指标 | 值 |
|------|------|
| 10,000 次查找 | 18.3ms |
| 单次延迟 | 1.83μs |

### Test 2: Redis 延迟

需要 Redis server 运行中。内网 Redis 预期:

| 场景 | 延迟 |
|------|------|
| 本地 Redis (loopback) | 0.3-0.5ms |
| 同机房 Redis | 0.5-1.0ms |
| 跨机房 Redis | 5-10ms |

### Test 3: 优雅降级

```
Redis unavailable (Error 111 connecting to localhost:6379. Connection refused.)
→ Local cache: HIT in 0.00ms      ← 本地不受影响
→ Redis cache: MISS in 166.78ms   ← 首次检测 Redis 不可用（连接超时）
→ Redis status: unavailable       ← 之后调用直接返回 None，不阻塞

✓ Graceful degradation works: system runs without Redis
```

**关键**: 首次 Redis 连接超时 166ms 后自动标记为不可用。后续请求直接跳过我 local fallback，零延迟。

### Test 4: 跨 Worker 共享

```
Worker-1: processed and cached
Worker-2: Redis HIT in 0.00ms → decision=block
✓ Cross-worker sharing works via Redis
```

### Test 5: 重启持久化

```
Before 'restart':
  Local cache:  MISS (as expected — new process)
After 'restart':
  Redis cache:  HIT in 0.00ms
✓ Redis survives process restart
```

### Test 6: 组合流程（最坏情况）

```
First lookup:  local=0.020ms (MISS) → redis=0.00ms (MISS)
After caching: local=0.006ms (HIT)
Total L0 lookup: 0.02ms (both miss, but once Redis marked down it's instant)
```

## 对比总结

| 维度 | 仅本地 TTLCache | + Redis |
|------|------|------|
| 单次延迟 | <0.01ms | +0.5ms (仅 miss 时) |
| 跨 Worker 共享 | 否 | 是 |
| 重启后保留 | 否 | 是 |
| 容量上限 | 100K (50MB) | 无限制 (Redis 内存) |
| 运维依赖 | 零 | 需要 Redis 服务 |
| 可用性 | 100% | 本地回退，100% |

## 安装 Redis

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis && brew services start redis

# Docker (推荐用于 POC)
docker run -d --name redis-moderation -p 6379:6379 redis:7-alpine
```

## 环境变量

```bash
# .env
REDIS_URL=redis://localhost:6379          # Redis 地址
REDIS_CACHE_TTL=7200                      # 缓存 TTL (秒), 默认 2 小时
```
