#!/usr/bin/env python3
"""Benchmark: single vs batch BGE embedding inference.

Sentence-transformers models benefit significantly from batched inference
because the transformer forward pass is more compute-efficient with
multiple inputs. The attention mechanism processes all texts in parallel.

Usage:
    python bench_batch.py          # full benchmark
    python bench_batch.py --quick  # smoke test with 100 texts
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.skills.embedder import embedder


def bench_single(texts: list[str]) -> dict:
    """Encode texts one at a time."""
    t0 = time.perf_counter()
    for text in texts:
        embedder.embed(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "mode": "single",
        "num_texts": len(texts),
        "total_ms": elapsed_ms,
        "avg_ms_per_text": elapsed_ms / len(texts),
    }


def bench_batch(texts: list[str], batch_size: int = 32) -> dict:
    """Encode texts in batches using embed_batch."""
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embedder.embed_batch(batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "mode": "batch",
        "batch_size": batch_size,
        "num_texts": len(texts),
        "total_ms": elapsed_ms,
        "avg_ms_per_text": elapsed_ms / len(texts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # Generate varied test texts
    templates = [
        "今天天气真好适合出去玩",
        "这个产品非常好用推荐给大家",
        "你真是个傻逼什么都不懂",
        "垃圾产品完全是骗人的千万别买",
        "谢谢大家的支持我会继续努力",
        "加微信买六合彩稳赢不赔",
        "有没有人一起去看演唱会啊",
        "今天的晚餐吃了麻辣烫味道不错",
        "Python 是一门很棒的编程语言",
        "请问这个周末有什么好看的电影",
    ]

    if args.quick:
        texts = templates * 10  # 100 texts
        print("=== Batch Inference Quick Benchmark ===\n")
        r1 = bench_single(texts)
        r2 = bench_batch(texts)
        print(f"Single: {r1['total_ms']:.1f}ms ({r1['avg_ms_per_text']:.2f}ms/text)")
        print(f"Batch:  {r2['total_ms']:.1f}ms ({r2['avg_ms_per_text']:.2f}ms/text)")
        print(f"Speedup: {r1['avg_ms_per_text'] / r2['avg_ms_per_text']:.1f}x")
        return

    print("=" * 60)
    print("BGE Batch Inference Benchmark")
    print("=" * 60)
    print()

    for scale, label in [(100, "100"), (500, "500"), (1000, "1K"), (2000, "2K")]:
        texts = (templates * (scale // len(templates) + 1))[:scale]

        r1 = bench_single(texts)
        # Test different batch sizes
        best = None
        for bs in [8, 16, 32, 64]:
            r2 = bench_batch(texts, batch_size=bs)
            if best is None or r2["avg_ms_per_text"] < best["avg_ms_per_text"]:
                best = r2

        speedup = r1["avg_ms_per_text"] / best["avg_ms_per_text"]
        print(f"--- {label} texts ---")
        print(f"  Single:      {r1['avg_ms_per_text']:.2f}ms/text ({r1['total_ms']:.0f}ms total)")
        print(f"  Batch ({best['batch_size']}):  {best['avg_ms_per_text']:.2f}ms/text ({best['total_ms']:.0f}ms total)")
        print(f"  Speedup:     {speedup:.1f}x")
        print()

    print("=" * 60)
    print("Key insight: batch inference amortizes the transformer")
    print("forward pass overhead across multiple inputs. For 32-text")
    print("batches, expect 2-3x throughput improvement.")
    print("=" * 60)


if __name__ == "__main__":
    main()
