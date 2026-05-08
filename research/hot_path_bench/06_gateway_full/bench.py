#!/usr/bin/env python3
"""Benchmark: Full Gateway Hot Path Flow

Tests the complete Gateway pipeline end-to-end:
  L0 Memory Cache → AC Automaton → ChromaDB Cache

Measures:
  1. Latency breakdown per layer
  2. Hit/miss distribution for each layer
  3. Overall hot path throughput
  4. Escalation rate to cold path

Usage:
  python bench.py            # full benchmark
  python bench.py --reset    # clear ChromaDB first
"""

import sys, os, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.gateway import gateway
from src.skills.embedder import embedder


def bench_latency_breakdown():
    """Measure latency at each Gateway layer."""
    print("=" * 60)
    print("Test 1: Gateway Layer Latency Breakdown")
    print("=" * 60)

    # Warm up
    embedder.embed("warmup")

    test_cases = [
        # (label, text, expected_path)
        ("empty", "", "hot"),
        ("keyword_block", "你真是个傻逼什么都不懂", "hot"),
        ("keyword_block_en", "you are a fucking idiot", "hot"),
        ("memory_cache_hit", "今天天气真好", "hot"),  # second call hits cache
        ("whitelist", "操场上同学们在操练队列", "hot"),
        ("clean_short", "今天天气真好适合出去玩", "cold"),  # first time → cold
        ("clean_long", ("这是一个正常的评论内容用于测试系统的处理能力" * 5)[:300], "cold"),
    ]

    # First call to each to populate cache where applicable
    for label, text, _ in test_cases:
        gateway.check(text, "", "")

    print()
    for label, text, expected in test_cases:
        t0 = time.perf_counter()
        result = gateway.check(text, "", "")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        path = "hot" if result["decision"] is not None else "cold"
        tier = result["decision"].get("tier", "cache") if result["decision"] else "escalated"

        # Trace breakdown
        steps = {}
        for t in result["traces"]:
            steps[t["step"]] = t["latency_ms"]

        print(f"  [{path:4s}] {label:20s}: {elapsed_ms:7.2f}ms total | "
              f"tier={tier:15s} | "
              f"mem={steps.get('L0_memory_hit', steps.get('L0_memory_miss', 0)):.2f}ms "
              f"kw={steps.get('keyword_block', steps.get('keyword_miss', steps.get('whitelist_pass', steps.get('keyword_ambiguous', 0)))):.2f}ms "
              f"chroma={steps.get('L1_chroma_hit', steps.get('L1_chroma_miss', 0)):.2f}ms")


def bench_traffic_distribution():
    """Run a large batch and measure hit/miss distribution."""
    print("\n" + "=" * 60)
    print("Test 2: Traffic Distribution (500 requests)")
    print("=" * 60)

    random.seed(42)

    # Generate varied traffic
    normal_texts = [
        f"正常评论内容编号{random.randint(0, 99999)}" for _ in range(200)
    ]
    toxic_texts = [
        random.choice([
            "你真是个傻逼", "操你妈的", "这垃圾产品", "脑残吧你",
            "fucking idiot", "what a piece of shit",
        ]) for _ in range(100)
    ]
    clean_texts = [
        "今天天气真好适合出去玩",
        "这个产品非常好用推荐给大家",
        "谢谢大家的支持我会继续努力",
    ] * 60  # 180 duplicates
    all_texts = normal_texts + toxic_texts + clean_texts
    random.shuffle(all_texts)

    results = {"hot_block": 0, "hot_pass": 0, "cold_escalated": 0}
    total_ms = 0.0

    for text in all_texts:
        t0 = time.perf_counter()
        result = gateway.check(text, "", "")
        total_ms += (time.perf_counter() - t0) * 1000

        if result["decision"] is not None:
            if result["decision"]["decision"] == "block":
                results["hot_block"] += 1
            else:
                results["hot_pass"] += 1
        else:
            results["cold_escalated"] += 1

    total = len(all_texts)
    print(f"  Total requests:   {total}")
    print(f"  Hot path (block): {results['hot_block']:>4} ({results['hot_block']/total*100:5.1f}%)")
    print(f"  Hot path (pass):  {results['hot_pass']:>4} ({results['hot_pass']/total*100:5.1f}%)")
    print(f"  Cold escalated:   {results['cold_escalated']:>4} ({results['cold_escalated']/total*100:5.1f}%)")
    print(f"  Hot path total:   {results['hot_block']+results['hot_pass']:>4} ({(results['hot_block']+results['hot_pass'])/total*100:5.1f}%)")
    print(f"  Avg latency:      {total_ms/total:.2f}ms")


def bench_throughput():
    """Measure Gateway throughput (requests/sec)."""
    print("\n" + "=" * 60)
    print("Test 3: Gateway Throughput")
    print("=" * 60)

    texts = [
        "今天天气真好适合出去玩",
        "你真是个傻逼",
        "这个产品非常好用",
        "fucking idiot",
        "操场上同学们在操练",
        "正常评论内容测试",
    ] * 200  # 1200 texts

    t0 = time.perf_counter()
    for text in texts:
        gateway.check(text, "", "")
    elapsed = time.perf_counter() - t0

    print(f"  Total: {len(texts)} requests")
    print(f"  Time:  {elapsed:.2f}s")
    print(f"  QPS:   {len(texts)/elapsed:.0f} req/s")
    print(f"  Avg:   {elapsed/len(texts)*1000:.2f}ms/req")

    # Gateway stats
    stats = gateway.get_stats()
    print(f"\n  Gateway Stats:")
    print(f"    Hot path rate:    {stats['hot_path_rate']*100:.1f}%")
    print(f"    Memory cache:     {stats['memory_cache_hit_rate']*100:.1f}%")
    print(f"    Keyword hit:      {stats['keyword_hit_rate']*100:.1f}%")
    print(f"    Whitelist:        {stats['whitelist_hit_rate']*100:.1f}%")
    print(f"    ChromaDB hit:     {stats['chroma_cache_hit_rate']*100:.1f}%")
    print(f"    Escalated:        {stats['escalated_rate']*100:.1f}%")
    print(f"    LangGraph calls saved: {stats['langgraph_calls_saved']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear ChromaDB first")
    args = parser.parse_args()

    print("Warming up...")
    embedder.embed("warmup")
    gateway.check("warmup", "", "")
    print("Ready.\n")

    bench_latency_breakdown()
    bench_traffic_distribution()
    bench_throughput()
    print("\n✅ Gateway full flow benchmark complete")
