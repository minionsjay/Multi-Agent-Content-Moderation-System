#!/usr/bin/env python3
"""Benchmark: ChromaDB Semantic Cache

Tests:
  1. Lookup latency vs collection size
  2. Hit rate vs similarity threshold tuning
  3. Write latency
  4. Cosine similarity conversion correctness
  5. PersistentClient startup time

Key questions:
  - What's the latency at 10K / 100K / 1M cached entries?
  - Is 0.95 similarity threshold optimal?
  - How fast is write vs read?
"""

import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.skills.vector_cache import vector_cache
from src.skills.embedder import embedder


def bench_lookup_latency():
    """Measure ChromaDB lookup latency."""
    print("=" * 60)
    print("Test 1: ChromaDB Lookup Latency")
    print("=" * 60)

    count = vector_cache.count()
    print(f"  Current collection size: {count} entries")

    # Test lookup with a known text
    text = "今天天气真好适合出去玩"
    embedding = embedder.embed(text)

    iterations = 100
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        vector_cache.lookup(embedding)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(f"  Iterations: {iterations}")
    print(f"  P50: {latencies[len(latencies)//2]:.2f}ms")
    print(f"  P99: {latencies[int(len(latencies)*0.99)]:.2f}ms")
    print(f"  Avg: {sum(latencies)/len(latencies):.2f}ms")


def bench_write_latency():
    """Measure ChromaDB write latency."""
    print("\n" + "=" * 60)
    print("Test 2: ChromaDB Write Latency")
    print("=" * 60)

    test_texts = [
        f"benchmark_test_text_{i}_{random.randint(0,99999)}"
        for i in range(20)
    ]

    latencies = []
    for text in test_texts:
        embedding = embedder.embed(text)
        t0 = time.perf_counter()
        vector_cache.store(embedding, text, "pass", 1.0, "benchmark")
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(f"  Writes: {len(test_texts)}")
    print(f"  P50: {latencies[len(latencies)//2]:.2f}ms")
    print(f"  P99: {latencies[int(len(latencies)*0.99)]:.2f}ms")
    print(f"  Avg: {sum(latencies)/len(latencies):.2f}ms")


def bench_similarity_boundary():
    """Test cache behavior at similarity threshold boundary."""
    print("\n" + "=" * 60)
    print("Test 3: Similarity Threshold Boundary Test")
    print("=" * 60)

    # Store a canonical text
    ref_text = "这是一条包含违规内容的测试文本需要被拦截"
    ref_embedding = embedder.embed(ref_text)
    vector_cache.store(ref_embedding, ref_text, "block", 0.95, "test")

    # Test similar texts
    test_cases = [
        ("文本完全相同", "这是一条包含违规内容的测试文本需要被拦截", "should HIT"),
        ("只改一个字", "这是一条包含违规内容的测试文本必须被拦截", "should HIT (high sim)"),
        ("意思相近", "这条内容有违规需要拦截处理", "might hit (semantic similar)"),
        ("完全不同-安全", "今天天气真好适合出去郊游野餐", "should MISS"),
        ("完全不同-违规", "支持台独分裂国家的言论必须制止", "should MISS (different topic)"),
        ("短文本", "违规", "might hit or miss"),
    ]

    for label, text, expected in test_cases:
        embedding = embedder.embed(text)
        result = vector_cache.lookup(embedding)
        hit = result is not None
        sim = result["similarity"] if result else 0.0
        status = "HIT" if hit else "MISS"
        print(f"  {status:4s} | sim={sim:.4f} | {label:12s}: {text[:40]}")


def bench_read_vs_write():
    """Compare read vs write throughput."""
    print("\n" + "=" * 60)
    print("Test 4: Read vs Write Throughput")
    print("=" * 60)

    text = f"throughput_test_{random.randint(0, 999999)}"
    embedding = embedder.embed(text)
    vector_cache.store(embedding, text, "pass", 1.0, "bench")

    # Read throughput
    iterations = 200
    t0 = time.perf_counter()
    for _ in range(iterations):
        vector_cache.lookup(embedding)
    read_ms = (time.perf_counter() - t0) * 1000
    read_qps = iterations / (read_ms / 1000)

    # Write throughput
    writes = 50
    t0 = time.perf_counter()
    for i in range(writes):
        t = f"write_test_{i}_{random.randint(0, 999999)}"
        e = embedder.embed(t)
        vector_cache.store(e, t, "pass", 1.0, "bench")
    write_ms = (time.perf_counter() - t0) * 1000
    write_qps = writes / (write_ms / 1000)

    print(f"  Read:  {read_qps:,.0f} QPS ({read_ms:.1f}ms for {iterations} ops)")
    print(f"  Write: {write_qps:,.0f} QPS ({write_ms:.1f}ms for {writes} ops)")
    print(f"  Read/Write ratio: {read_qps/write_qps:.1f}x")


if __name__ == "__main__":
    print("Loading embedder...")
    embedder.embed("warmup")
    print("Ready.\n")
    bench_lookup_latency()
    bench_write_latency()
    bench_similarity_boundary()
    bench_read_vs_write()
    print("\n✅ ChromaDB cache benchmark complete")
