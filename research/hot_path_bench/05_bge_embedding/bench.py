#!/usr/bin/env python3
"""Benchmark: BGE Embedding (with and without embedding cache)

Tests:
  1. Single-text latency (cold start vs warm)
  2. Batch latency at different batch sizes
  3. Embedding cache: hit rate and speedup
  4. Throughput: single vs batch vs cached
  5. Vector dimension and memory usage

Key questions:
  - What's the real latency distribution (P50/P99)?
  - How much does the embedding cache help?
  - What batch size gives the best throughput?
"""

import sys, os, time, hashlib, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.skills.embedder import embedder
from cachetools import TTLCache


def bench_single_warmup():
    """Single text latency with warm model (after first call)."""
    print("=" * 60)
    print("Test 1: Single Text Latency (Warm Model)")
    print("=" * 60)

    texts = [
        "今天天气真好",                                 # short
        "这是一个中等长度的中文测试文本包含二十个字符",      # medium
        ("这是一个比较长的中文测试文本" * 20)[:500],      # long
    ]

    for label, text in [("short (7 chars)", texts[0]),
                         ("medium (20 chars)", texts[1]),
                         ("long (500 chars)", texts[2])]:
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            embedder.embed(text)
            latencies.append((time.perf_counter() - t0) * 1000)
        latencies.sort()
        print(f"  {label:20s}: P50={latencies[25]:5.2f}ms  "
              f"P99={latencies[49]:5.2f}ms  avg={sum(latencies)/50:5.2f}ms")


def bench_batch_sizes():
    """Batch embedding at different sizes."""
    print("\n" + "=" * 60)
    print("Test 2: Batch Embedding vs Batch Size")
    print("=" * 60)

    base_texts = ["测试文本内容编号" + str(i) for i in range(128)]

    for bs in [1, 4, 8, 16, 32, 64, 128]:
        batch = base_texts[:bs]
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            embedder.embed_batch(batch)
            latencies.append((time.perf_counter() - t0) * 1000)
        total = sum(latencies) / len(latencies)
        per_text = total / bs
        print(f"  batch_size={bs:>3}: {total:6.1f}ms total  {per_text:5.2f}ms/text  "
              f"≈ {1000/per_text:5.0f} QPS (single core)")


def bench_embedding_cache_effect():
    """Measure embedding cache hit rate and speedup with realistic traffic."""
    print("\n" + "=" * 60)
    print("Test 3: Embedding Cache Speedup")
    print("=" * 60)

    # Generate traffic with 40% unique + 60% duplicates
    random.seed(42)
    unique_templates = [
        f"unique_text_{i}_{random.randint(0, 99999)}" for i in range(400)
    ]
    duplicate_texts = [
        "加微信买六合彩稳赢不赔发大财机会难得快来",
        "真人百家乐在线赌博日赚千元免费注册",
        "同城约炮交友软件下载链接在简介里面",
        "你真是个傻逼这个产品太垃圾了千万别买",
        "今天天气真好适合出去玩推荐给大家一起",
    ]
    traffic = unique_templates + duplicate_texts * 120  # 400 + 600 = 1000
    random.shuffle(traffic)

    # Without embedding cache: count BGE calls
    bge_calls = 0
    t0 = time.perf_counter()
    for text in traffic:
        embedder.embed(text)  # uses internal cache from embedder.py
        bge_calls += 1  # but we can't easily track internal cache, so just measure time
    elapsed = time.perf_counter() - t0

    # The embedder has internal cache now. We can check the hit rate.
    hit_rate = embedder.cache_hit_rate if hasattr(embedder, 'cache_hit_rate') else 0.0
    print(f"  Total texts: {len(traffic)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Embedding cache hit rate: {hit_rate:.1%}")
    print(f"  Effective throughput: {len(traffic)/elapsed:.0f} texts/sec")


def bench_dimension_memory():
    """Vector dimension and memory per embedding."""
    print("\n" + "=" * 60)
    print("Test 4: Vector Properties")
    print("=" * 60)

    dim = embedder.DIM
    vec = embedder.embed("测试")

    import numpy as np
    arr = np.array(vec)
    print(f"  Dimension: {dim}")
    print(f"  Memory per vector: {dim * 4} bytes ({dim * 4 / 1024:.1f} KB)")
    print(f"  Memory for 1M vectors: {dim * 4 * 1_000_000 / 1024 / 1024:.0f} MB")
    print(f"  L2 norm: {np.linalg.norm(arr):.4f} (should be ~1.0 after normalize)")
    print(f"  Value range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  Non-zero elements: {np.count_nonzero(arr)}/{dim}")


if __name__ == "__main__":
    print("Warming up BGE model...")
    embedder.embed("warmup")
    print("Ready.\n")
    bench_single_warmup()
    bench_batch_sizes()
    bench_embedding_cache_effect()
    bench_dimension_memory()
    print("\n✅ BGE embedding benchmark complete")
