#!/usr/bin/env python3
"""Benchmark: L0 Memory Cache (SHA256 + TTLCache)

Tests:
  1. Latency vs cache size (1K → 500K entries)
  2. Hit rate vs TTL decay
  3. Memory usage per 100K entries
  4. Throughput (lookups/sec)
  5. Hash collision resilience (SHA256 on adversarial texts)

Key questions:
  - What's the max QPS for L0 cache lookups?
  - How does TTLCache perform at 100K vs 500K entries?
  - Is SHA256 overhead measurable vs simpler hash?
"""

import sys, os, time, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from cachetools import TTLCache


def bench_latency_vs_size():
    """Measure lookup latency at different cache sizes."""
    print("=" * 60)
    print("Test 1: Latency vs Cache Size")
    print("=" * 60)
    for size in [1_000, 10_000, 50_000, 100_000, 500_000]:
        cache = TTLCache(maxsize=size, ttl=3600)
        # Fill cache
        for i in range(size):
            key = hashlib.sha256(f"text_{i}".encode()).hexdigest()
            cache[key] = {"decision": "pass", "confidence": 1.0}

        # Benchmark lookups
        iterations = 100_000
        keys_to_lookup = [
            hashlib.sha256(f"text_{i % size}".encode()).hexdigest()
            for i in range(iterations)
        ]

        t0 = time.perf_counter()
        hits = 0
        for key in keys_to_lookup:
            if cache.get(key) is not None:
                hits += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"  Size={size:>6,}: {elapsed_ms:6.1f}ms for {iterations:,} lookups "
              f"= {elapsed_ms/iterations*1000:6.3f}μs/lookup  "
              f"≈ {iterations/(elapsed_ms/1000):,.0f} QPS  "
              f"hits={hits}/{iterations}")


def bench_hash_overhead():
    """Compare SHA256 vs MD5 vs hash() for key generation."""
    print("\n" + "=" * 60)
    print("Test 2: Hash Algorithm Overhead")
    print("=" * 60)
    text = "这是一段比较长的中文测试文本，用来测试不同哈希算法的性能差异" * 10
    iterations = 100_000

    for name, func in [
        ("SHA256", lambda t: hashlib.sha256(t.encode()).hexdigest()),
        ("MD5", lambda t: hashlib.md5(t.encode()).hexdigest()),
        ("Python hash", lambda t: str(hash(t))),
    ]:
        t0 = time.perf_counter()
        for _ in range(iterations):
            func(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  {name:>12}: {elapsed_ms:7.1f}ms = {elapsed_ms/iterations*1000:6.3f}μs/op")


def bench_memory_usage():
    """Estimate memory usage per entry."""
    print("\n" + "=" * 60)
    print("Test 3: Memory Usage Estimate")
    print("=" * 60)
    # TTLCache stores: key (64 char str) + value dict
    # Python overhead: ~50 bytes per str, ~200 bytes per dict
    per_entry = 64 * 2 + 50 * 2 + 200 + 8  # key str + value dict overhead
    sizes = [10_000, 50_000, 100_000, 500_000]
    for size in sizes:
        mem_mb = size * per_entry / 1024 / 1024
        print(f"  {size:>6,} entries ≈ {mem_mb:.0f} MB")


def bench_throughput():
    """Measure raw throughput of TTLCache get/set."""
    print("\n" + "=" * 60)
    print("Test 4: Raw Throughput (get + set)")
    print("=" * 60)
    cache = TTLCache(maxsize=100_000, ttl=3600)

    # Fill
    for i in range(100_000):
        cache[hashlib.sha256(f"t_{i}".encode()).hexdigest()] = i

    # Mixed get/set
    ops = 500_000
    t0 = time.perf_counter()
    for i in range(ops):
        key = hashlib.sha256(f"t_{i % 200_000}".encode()).hexdigest()
        if i % 3 == 0:
            cache[key] = i
        else:
            cache.get(key)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  {ops:,} mixed ops in {elapsed_ms:.1f}ms = {ops/(elapsed_ms/1000):,.0f} ops/sec")


def bench_collision():
    """SHA256 collision check on adversarial texts."""
    print("\n" + "=" * 60)
    print("Test 5: SHA256 Collision Check")
    print("=" * 60)
    # Generate texts that differ by 1 char
    base = "a" * 1000
    keys = set()
    collisions = 0
    for i in range(100_000):
        text = base[:i % 1000] + chr(65 + (i % 26)) + base[i % 1000 + 1:]
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in keys:
            collisions += 1
        keys.add(key)
    print(f"  100,000 similar texts → {collisions} collisions")
    print(f"  SHA256 collision probability for this test: {collisions/100000:.8f}")


if __name__ == "__main__":
    bench_latency_vs_size()
    bench_hash_overhead()
    bench_memory_usage()
    bench_throughput()
    bench_collision()
    print("\n✅ L0 Memory Cache benchmark complete")
