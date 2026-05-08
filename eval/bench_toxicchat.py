#!/usr/bin/env python3
"""Benchmark using lmsys/toxic-chat dataset through full gateway + graph pipeline."""

import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gateway import gateway
from src.graph import graph
from src.state import ModerationState
from src.skills.vector_cache import vector_cache
from src.skills.embedder import embedder


def load_toxic_chat(n: int | None = None, seed: int = 42) -> list[dict]:
    """Load lmsys/toxic-chat and map to our expected format.

    Returns list of {"text": str, "expected": "safe"|"unsafe"}
    """
    from datasets import load_dataset

    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    print(f"Loaded {len(ds)} samples from lmsys/toxic-chat (toxicchat0124)")

    cases = []
    for row in ds:
        label = "unsafe" if row["toxicity"] == 1 else "safe"
        cases.append({"text": row["user_input"], "expected": label})

    # Balance: take all unsafe + equal number of safe, shuffled
    unsafe = [c for c in cases if c["expected"] == "unsafe"]
    safe = [c for c in cases if c["expected"] == "safe"]

    import random
    random.seed(seed)
    random.shuffle(safe)
    random.shuffle(unsafe)

    if n:
        n_unsafe = max(5, min(len(unsafe), n // 3))
        n_safe = min(len(safe), n - n_unsafe)
        sampled = unsafe[:n_unsafe] + safe[:n_safe]
    else:
        sampled = unsafe + safe[:min(len(safe), len(unsafe) * 3)]

    random.shuffle(sampled)
    print(f"Using {len(sampled)} cases (unsafe={min(len(unsafe), n_unsafe if n else len(unsafe))}, "
          f"safe={min(len(safe), n_safe if n else len(safe))})")
    return sampled


async def run_one(text: str, content_id: str) -> dict:
    """Run full pipeline: gateway → graph (if escalated)."""
    t0 = time.perf_counter()

    # Step 1: Gateway check (hot path)
    gw = gateway.check(text)
    gw_traces = gw.get("traces", [])

    if gw["decision"] is not None:
        # Hot path hit
        resp = dict(gw["decision"])
        resp["traces"] = gw_traces
        resp["latency_ms"] = (time.perf_counter() - t0) * 1000
        resp["path"] = "hot"
        resp.setdefault("tier", resp.get("tier", "L0_memory"))
        return resp

    # Step 2: Cold path — LangGraph
    state: ModerationState = {
        "content_id": content_id,
        "text": text,
        "image_url": "",
        "image_base64": "",
        "user_id": "benchmark",
        "source": "toxicchat",
        "content_type": "text_only",
        "cache_hit": False,
        "cached_decision": None,
        "keyword_confidence": gw.get("keyword_confidence", 0.0),
        "keyword_label": gw.get("keyword_label"),
        "keyword_prefiltered": gw.get("keyword_prefiltered", False),
        "priority_score": 0.3,
        "text_result": None,
        "decision": "pass",
        "confidence": 0.0,
        "reason": "",
        "traces": [],
    }

    result = await graph.ainvoke(state)
    text_result = result.get("text_result") or {}
    tier = text_result.get("tier", "L3_llm") if text_result else "L3_llm"

    resp = {
        "content_id": content_id,
        "decision": result.get("decision", "pass"),
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason", ""),
        "tier": tier,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "traces": gw_traces + result.get("traces", []),
        "path": "cold",
    }
    return resp


def compute_metrics(cases: list[dict], results: list[dict]) -> dict:
    """Compute accuracy, precision, recall, F1, and tier distribution."""
    tp = tn = fp = fn = 0
    tier_counts: dict[str, int] = {}
    path_counts: dict[str, int] = {}
    latencies = []

    for case, res in zip(cases, results):
        expected = case["expected"]
        predicted = res.get("decision", "pass")

        tier = res.get("tier", "?")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        path = res.get("path", "?")
        path_counts[path] = path_counts.get(path, 0) + 1

        latencies.append(res.get("latency_ms", 0))

        if expected == "unsafe" and predicted in ("block", "review"):
            tp += 1
        elif expected == "safe" and predicted == "pass":
            tn += 1
        elif expected == "unsafe" and predicted == "pass":
            fn += 1
        elif expected == "safe" and predicted in ("block", "review"):
            fp += 1

    total = len(cases)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Latency stats
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0

    return {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tier_distribution": tier_counts,
        "path_distribution": path_counts,
        "llm_call_rate": round(tier_counts.get("L3_llm", 0) / total, 4) if total > 0 else 0,
        "bert_intercept_rate": round(tier_counts.get("L2_bert", 0) / total, 4) if total > 0 else 0,
        "keyword_intercept_rate": round(tier_counts.get("L1_keyword", 0) / total, 4) if total > 0 else 0,
        "cache_hit_rate": round(
            (tier_counts.get("L0_memory", 0) + tier_counts.get("L0_redis", 0) +
             tier_counts.get("L0_phash", 0) + tier_counts.get("L0_empty", 0) +
             tier_counts.get("L0_whitelist", 0) + tier_counts.get("L1_chroma", 0)) / total, 4
        ) if total > 0 else 0,
        "hot_path_rate": round(path_counts.get("hot", 0) / total, 4) if total > 0 else 0,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "avg_latency_ms": round(sum(latencies) / total, 2) if total > 0 else 0,
    }


def print_report(metrics: dict, elapsed_s: float):
    """Pretty-print the benchmark report."""
    print()
    print("=" * 60)
    print("  POC Benchmark — lmsys/toxic-chat (English)")
    print("=" * 60)
    print(f"  Total cases:       {metrics['total']}")
    print(f"  Time elapsed:      {elapsed_s:.1f}s")
    print(f"  Avg latency:       {metrics['avg_latency_ms']:.1f}ms/req")
    print(f"  Throughput:        {metrics['total'] / elapsed_s:.1f} req/s")
    print()
    print("  Confusion Matrix:")
    print(f"    TP={metrics['tp']:4d}  FP={metrics['fp']:4d}")
    print(f"    FN={metrics['fn']:4d}  TN={metrics['tn']:4d}")
    print()
    print(f"  Accuracy:          {metrics['accuracy']:.2%}")
    print(f"  Precision:         {metrics['precision']:.2%}")
    print(f"  Recall:            {metrics['recall']:.2%}")
    print(f"  F1-score:          {metrics['f1']:.2%}")
    print()
    print("  Latency:")
    print(f"    P50: {metrics['latency_p50_ms']:.0f}ms")
    print(f"    P95: {metrics['latency_p95_ms']:.0f}ms")
    print(f"    P99: {metrics['latency_p99_ms']:.0f}ms")
    print()
    print("  Path Distribution:")
    for path, count in sorted(metrics["path_distribution"].items()):
        pct = count / metrics["total"] * 100
        print(f"    {path}: {count:4d} ({pct:.1f}%)")
    print()
    print("  Tier Distribution (L0→L1→L2→L3 funnel):")
    tier_order = ["L0_memory", "L0_redis", "L0_empty", "L0_whitelist",
                  "L1_keyword", "L1_chroma", "L2_bert", "L3_llm"]
    for tier in tier_order:
        count = metrics["tier_distribution"].get(tier, 0)
        if count > 0:
            pct = count / metrics["total"] * 100
            print(f"    {tier:16s}: {count:4d} ({pct:.1f}%)")
    # Any tiers not in order
    for tier, count in sorted(metrics["tier_distribution"].items()):
        if tier not in tier_order:
            pct = count / metrics["total"] * 100
            print(f"    {tier:16s}: {count:4d} ({pct:.1f}%)")
    print()
    print("  Key Metrics:")
    print(f"    Hot path rate:     {metrics['hot_path_rate']:.2%}")
    print(f"    LLM call rate:     {metrics['llm_call_rate']:.2%}")
    print(f"    BERT intercept:    {metrics['bert_intercept_rate']:.2%}")
    print(f"    Keyword intercept: {metrics['keyword_intercept_rate']:.2%}")
    print(f"    Cache hit rate:    {metrics['cache_hit_rate']:.2%}")
    print()
    print("  Gateway Stats:")
    gw_stats = gateway.get_stats()
    for k, v in gw_stats.items():
        print(f"    {k}: {v}")
    print("=" * 60)

    # POC acceptance criteria
    print()
    print("  Acceptance Criteria:")
    checks = [
        ("LLM call rate < 20%", metrics["llm_call_rate"] < 0.20),
        ("BERT intercept > 30%", metrics["bert_intercept_rate"] > 0.30),
        ("F1-score > 90%", metrics["f1"] > 0.90),
        ("Cache hit rate > 15%", metrics["cache_hit_rate"] > 0.15),
        ("Avg latency < 500ms", metrics["avg_latency_ms"] < 500),
    ]
    for criteria, passed in checks:
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"    [{status}] {criteria}")

    return checks


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark with lmsys/toxic-chat")
    parser.add_argument("-n", type=int, default=200, help="Number of samples (default: 200)")
    parser.add_argument("--full", action="store_true", help="Run all 5082 samples (takes >30min)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent LLM calls (default: 4)")
    parser.add_argument("--no-cache", action="store_true", help="Clear caches before run")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    n = None if args.full else args.n

    print("=" * 60)
    print("  POC Benchmark: lmsys/toxic-chat (English)")
    print(f"  Samples: {n or 'ALL (~5000)'} | Concurrency: {args.concurrency}")
    print("=" * 60)
    print()

    # Clear caches if requested
    if args.no_cache:
        from src.skills.memory_cache import memory_cache
        memory_cache._cache.clear()
        # Clear ChromaDB by removing directory
        import shutil
        if os.path.exists(vector_cache.persist_dir):
            shutil.rmtree(vector_cache.persist_dir)
        print("Caches cleared.\n")

    # Load data
    cases = load_toxic_chat(n=n)
    print()

    # Run benchmark
    sem = asyncio.Semaphore(args.concurrency)

    async def process_one(i: int, case: dict) -> dict:
        async with sem:
            res = await run_one(case["text"], f"toxicchat_{i}")
            res["text"] = case["text"]
            res["expected"] = case["expected"]
            return res

    t0 = time.perf_counter()
    tasks = [process_one(i, c) for i, c in enumerate(cases)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(cases, results)
    checks = print_report(metrics, elapsed)

    # Save results
    if args.output:
        output = {
            "dataset": "lmsys/toxic-chat (toxicchat0124)",
            "config": {"n": len(cases), "concurrency": args.concurrency},
            "metrics": metrics,
            "checks": {c: p for c, p in checks},
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

    # Show some misclassified examples
    print()
    print("--- Sample Misclassifications ---")
    shown = 0
    for case, res in zip(cases, results):
        expected = case["expected"]
        predicted = res.get("decision", "pass")
        pred_ok = (expected == "unsafe" and predicted in ("block", "review")) or \
                  (expected == "safe" and predicted == "pass")
        if not pred_ok and shown < 8:
            print(f"\n  Text: {case['text'][:120]}")
            print(f"  Expected: {expected} | Predicted: {predicted} "
                  f"| Conf: {res.get('confidence', 0):.2f} | Tier: {res.get('tier', '?')}")
            print(f"  Reason: {res.get('reason', '')[:150]}")
            shown += 1

    if shown == 0:
        print("  No misclassifications!")


if __name__ == "__main__":
    asyncio.run(main())
