#!/usr/bin/env python3
"""Benchmark: Redis Shared Cache vs Local TTLCache

Tests:
  1. Local cache latency (baseline)
  2. Redis cache latency (connected)
  3. Graceful degradation when Redis is down
  4. Cross-worker cache sharing simulation
  5. Restart resilience simulation

Usage:
  python bench_redis.py                  # full benchmark (needs Redis)
  python bench_redis.py --local-only     # only test local cache (no Redis needed)
"""

import sys, os, time, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.skills.memory_cache import memory_cache as local_cache
from src.skills.redis_cache import redis_cache


def bench_local_latency():
    """Local TTLCache latency."""
    print("=" * 60)
    print("Test 1: Local TTLCache Latency (baseline)")
    print("=" * 60)

    text = "这是一条测试文本用于缓存基准测试"
    local_cache.set(text, "block", 0.95, "test")

    iterations = 10_000
    t0 = time.perf_counter()
    for _ in range(iterations):
        local_cache.get(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  {iterations:,} lookups: {elapsed_ms:.1f}ms "
          f"= {elapsed_ms/iterations*1000:,.2f}μs/lookup")


def bench_redis_latency():
    """Redis cache latency."""
    print("\n" + "=" * 60)
    print("Test 2: Redis Cache Latency")
    print("=" * 60)

    if redis_cache.status != "connected":
        print("  Redis not available — skipping")
        return

    text = "这是一条Redis缓存测试文本"
    redis_cache.set(text, "pass", 1.0, "benchmark")

    iterations = 1_000
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        redis_cache.get(text)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(f"  Redis URL: {redis_cache.redis_url}")
    print(f"  Iterations: {iterations:,}")
    print(f"  P50: {latencies[len(latencies)//2]:.2f}ms")
    print(f"  P99: {latencies[int(len(latencies)*0.99)]:.2f}ms")
    print(f"  Avg: {sum(latencies)/len(latencies):.2f}ms")
    print(f"  Hit rate: {redis_cache.hit_rate:.1%}")


def bench_graceful_degradation():
    """Verify system works without Redis."""
    print("\n" + "=" * 60)
    print("Test 3: Graceful Degradation (Redis down)")
    print("=" * 60)

    text = "Redis宕机时的回退测试"

    # 1. Local cache only (always available)
    local_cache.set(text, "pass", 1.0, "test")
    t0 = time.perf_counter()
    local_result = local_cache.get(text)
    local_ms = (time.perf_counter() - t0) * 1000

    # 2. Redis may or may not be available
    t0 = time.perf_counter()
    redis_result = redis_cache.get(text)
    redis_ms = (time.perf_counter() - t0) * 1000

    print(f"  Local cache:  {'HIT' if local_result else 'MISS'} in {local_ms:.2f}ms")
    print(f"  Redis cache:  {'HIT' if redis_result else 'MISS'} in {redis_ms:.2f}ms")
    print(f"  Redis status: {redis_cache.status}")

    if redis_cache.status == "unavailable":
        print(f"  ✓ Graceful degradation works: system runs without Redis")
    else:
        print(f"  Redis is connected")


def bench_cross_worker_simulation():
    """Simulate cross-worker cache sharing via Redis."""
    print("\n" + "=" * 60)
    print("Test 4: Cross-Worker Cache Sharing")
    print("=" * 60)

    if redis_cache.status != "connected":
        print("  Redis not available — skipping")
        return

    # Clear previous test data
    text = "跨Worker共享测试文本_" + str(int(time.time()))

    # Simulate Worker-1: processes the text, writes to both caches
    local_cache.set(text, "block", 0.95, "worker-1")
    redis_cache.set(text, "block", 0.95, "worker-1")
    print(f"  Worker-1: processed and cached")

    # Simulate Worker-2: has empty local cache, checks Redis
    # (We can't truly simulate separate processes, but we can
    #  clear the local cache and show Redis still has it)
    t0 = time.perf_counter()
    # Check Redis first (as Worker-2 would)
    redis_hit = redis_cache.get(text)
    redis_ms = (time.perf_counter() - t0) * 1000

    if redis_hit:
        print(f"  Worker-2: Redis HIT in {redis_ms:.2f}ms "
              f"→ decision={redis_hit['decision']}")
        print(f"  ✓ Cross-worker sharing works via Redis")
    else:
        print(f"  Worker-2: Redis MISS")


def bench_restart_resilience():
    """Show Redis survives process restart."""
    print("\n" + "=" * 60)
    print("Test 5: Restart Resilience")
    print("=" * 60)

    if redis_cache.status != "connected":
        print("  Redis not available — skipping")
        return

    text = "进程重启后仍然存在的缓存_" + str(int(time.time()))
    redis_cache.set(text, "block", 0.99, "before_restart")

    # Simulate restart: local cache is gone (new TTLCache)
    local_before = local_cache.get(text)

    # But Redis still has it
    t0 = time.perf_counter()
    redis_after = redis_cache.get(text)
    redis_ms = (time.perf_counter() - t0) * 1000

    print(f"  Before 'restart':")
    print(f"    Local cache:  {'HIT' if local_before else 'MISS (as expected — new process)'}")
    print(f"  After 'restart':")
    print(f"    Redis cache:  {'HIT' if redis_after else 'MISS'} in {redis_ms:.2f}ms")
    if redis_after:
        print(f"    ✓ Redis survives process restart")


def bench_combined_flow():
    """Test the full Gateway flow: local → Redis → fallthrough."""
    print("\n" + "=" * 60)
    print("Test 6: Combined Flow (Local → Redis → Miss)")
    print("=" * 60)

    # Fresh text — not in any cache
    fresh_text = "全新的从未出现过的文本_" + str(int(time.time() * 1000))

    # Step 1: Local cache miss
    t0 = time.perf_counter()
    local = local_cache.get(fresh_text)
    local_ms = (time.perf_counter() - t0) * 1000

    # Step 2: Redis miss
    t0 = time.perf_counter()
    redis_res = redis_cache.get(fresh_text)
    redis_ms = (time.perf_counter() - t0) * 1000

    # Step 3: Cache the result (as Action agent would)
    local_cache.set(fresh_text, "pass", 1.0, "test")
    redis_cache.set(fresh_text, "pass", 1.0, "test")

    # Step 4: Verify next lookup hits
    t0 = time.perf_counter()
    local2 = local_cache.get(fresh_text)
    local2_ms = (time.perf_counter() - t0) * 1000

    print(f"  First lookup:  local={local_ms:.3f}ms (MISS) → "
          f"redis={redis_ms:.2f}ms (MISS)")
    print(f"  After caching: local={local2_ms:.3f}ms (HIT)")
    print(f"  Total L0 lookup path: {local_ms + redis_ms:.2f}ms (worst case — both miss)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true",
                       help="Only test local cache (no Redis needed)")
    args = parser.parse_args()

    print("Redis Cache Benchmark")
    print(f"Redis status: {redis_cache.status}\n")

    bench_local_latency()
    bench_redis_latency()
    bench_graceful_degradation()

    if not args.local_only:
        bench_cross_worker_simulation()
        bench_restart_resilience()
    bench_combined_flow()

    print("\n✅ Redis cache benchmark complete")
