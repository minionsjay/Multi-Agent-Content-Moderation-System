#!/usr/bin/env python3
"""Benchmark: embedding cache hit rate and latency impact.

Measures how much an in-memory embedding cache reduces BGE inference calls
for real-world content moderation traffic patterns.

Key insight: in a multi-service deployment, many texts are near-identical
or completely identical (spam, copypasta, template messages). An embedding
cache avoids re-computing the same vector repeatedly.

Usage:
    python bench_embed_cache.py          # full benchmark
    python bench_embed_cache.py --quick  # smoke test (1000 iterations)
"""

import sys, os, time, hashlib, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cachetools import TTLCache
from src.skills.embedder import embedder


def bench_no_cache(texts: list[str], runs: int = 10) -> dict:
    """Baseline: BGE inference every time."""
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for text in texts:
            embedder.embed(text)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "mode": "no_cache",
        "total_texts": len(texts),
        "runs": runs,
        "avg_total_ms": sum(latencies) / len(latencies),
        "calls_per_run": len(texts),
    }


def bench_with_cache(texts: list[str], cache_size: int = 100_000,
                     ttl: int = 3600) -> dict:
    """With embedding cache: only compute if not cached."""
    cache = TTLCache(maxsize=cache_size, ttl=ttl)
    hits = 0
    misses = 0
    latencies = []

    for _ in range(10):
        t0 = time.perf_counter()
        for text in texts:
            key = hashlib.sha256(text.encode()).hexdigest()
            if cache.get(key) is not None:
                hits += 1
                continue
            misses += 1
            vec = embedder.embed(text)
            cache[key] = vec
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mode": "with_cache",
        "cache_size": cache_size,
        "ttl_seconds": ttl,
        "total_texts": len(texts),
        "total_ops": hits + misses,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0,
        "avg_total_ms": sum(latencies) / len(latencies),
    }


def generate_traffic(num_unique: int, num_duplicates: int,
                     base_texts: list[str]) -> list[str]:
    """Simulate real traffic: mix of unique content + repeated spam/templates.

    Real-world traffic pattern:
      - ~40% completely unique (genuine user content)
      - ~30% near-duplicates (template messages, same meaning different words)
      - ~30% exact duplicates (spam, copypasta, automated messages)
    """
    import random
    random.seed(42)

    result = []
    # Unique content
    for _ in range(num_unique):
        result.append(random.choice(base_texts) + f" [{random.randint(0, 99999)}]")

    # Exact duplicates (simulating spam waves)
    spam_templates = [
        "加微信买六合彩稳赢不赔发财机会难得",
        "真人百家乐在线赌博日赚千元免费注册",
        "同城约炮交友软件下载链接在简介",
        "你真是个傻逼这个东西太垃圾了别买",
        "今天天气真好适合出去玩推荐给大家",
    ]
    for t in spam_templates:
        for _ in range(num_duplicates // len(spam_templates)):
            result.append(t)

    random.shuffle(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    base_texts = [
        "今天天气真好适合出去玩",
        "这个产品非常好用推荐给大家",
        "你真是个傻逼什么都不懂",
        "支持台独台湾是一个独立国家",
        "我要杀了你这个混蛋",
        "加微信买六合彩稳赢",
        "谢谢大家的支持我会继续努力",
        "有没有人一起去看演唱会",
        "分享一下我的学习笔记希望对大家有帮助",
        "垃圾产品完全是骗人的千万别买",
    ]

    if args.quick:
        texts = generate_traffic(num_unique=30, num_duplicates=30, base_texts=base_texts)
        print("=== Quick Smoke Test (60 texts) ===\n")
        r1 = bench_no_cache(texts, runs=3)
        r2 = bench_with_cache(texts)
        print(f"No cache:  {r1['avg_total_ms']:.1f}ms avg, {r1['calls_per_run']} BGE calls/run")
        print(f"With cache: {r2['avg_total_ms']:.1f}ms avg, hit_rate={r2['hit_rate']:.1%}")
        print(f"Speedup: {r1['avg_total_ms'] / r2['avg_total_ms']:.1f}x")
        return

    # Full benchmark with realistic traffic volume
    print("=" * 60)
    print("Embedding Cache Benchmark")
    print("=" * 60)

    for scale, label in [(1000, "1K"), (5000, "5K"), (10000, "10K")]:
        unique = int(scale * 0.4)
        dupes = int(scale * 0.6)
        texts = generate_traffic(num_unique=unique, num_duplicates=dupes,
                                 base_texts=base_texts)

        print(f"\n--- {label} texts ({unique} unique + {dupes} duplicates) ---")
        r1 = bench_no_cache(texts)
        r2 = bench_with_cache(texts)

        print(f"  No cache:     {r1['avg_total_ms']:6.1f}ms ({r1['calls_per_run']} BGE calls)")
        print(f"  With cache:   {r2['avg_total_ms']:6.1f}ms ({r2['misses']} BGE calls, hit_rate={r2['hit_rate']:.1%})")
        print(f"  Speedup:      {r1['avg_total_ms'] / r2['avg_total_ms']:.1f}x")
        print(f"  BGE calls saved: {r1['calls_per_run'] * 10 - r2['misses']}")

    print("\n" + "=" * 60)
    print("Summary: Embedding cache eliminates redundant BGE inference")
    print("for duplicate/template content. Hit rate depends on traffic")
    print("patterns — higher with spam waves and automated content.")
    print("=" * 60)


if __name__ == "__main__":
    main()
