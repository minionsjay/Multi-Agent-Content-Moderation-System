#!/usr/bin/env python3
"""Benchmark using nvidia/Aegis-AI-Content-Safety-Dataset-2.0."""

import asyncio
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gateway import gateway
from src.graph import graph
from src.state import ModerationState
from src.skills.vector_cache import vector_cache


def load_aegis(n: int | None = None, seed: int = 42) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    print(f"Loaded {len(ds)} samples from Aegis")

    cases = []
    for row in ds:
        label = row["prompt_label"]  # safe / unsafe
        cases.append({"text": row["prompt"], "expected": label, "category": row.get("violated_categories", "")})

    unsafe = [c for c in cases if c["expected"] == "unsafe"]
    safe = [c for c in cases if c["expected"] == "safe"]

    import random
    random.seed(seed)
    random.shuffle(safe)
    random.shuffle(unsafe)

    if n:
        n_unsafe = max(5, min(len(unsafe), n // 2))
        n_safe = min(len(safe), n - n_unsafe)
        sampled = unsafe[:n_unsafe] + safe[:n_safe]
    else:
        sampled = unsafe + safe[:len(unsafe)]

    random.shuffle(sampled)
    print(f"Using {len(sampled)} cases (unsafe={n_unsafe}, safe={n_safe})")
    return sampled


async def run_one(text: str, content_id: str) -> dict:
    t0 = time.perf_counter()
    gw = gateway.check(text)
    gw_traces = gw.get("traces", [])

    if gw["decision"] is not None:
        resp = dict(gw["decision"])
        resp["traces"] = gw_traces
        resp["latency_ms"] = (time.perf_counter() - t0) * 1000
        resp["path"] = "hot"
        resp.setdefault("tier", resp.get("tier", "L0_memory"))
        return resp

    state: ModerationState = {
        "content_id": content_id, "text": text, "image_url": "", "image_base64": "",
        "user_id": "benchmark", "source": "aegis", "content_type": "text_only",
        "cache_hit": False, "cached_decision": None,
        "keyword_confidence": gw.get("keyword_confidence", 0.0),
        "keyword_label": gw.get("keyword_label"),
        "keyword_prefiltered": gw.get("keyword_prefiltered", False),
        "priority_score": 0.3, "text_result": None,
        "decision": "pass", "confidence": 0.0, "reason": "", "traces": [],
    }

    result = await graph.ainvoke(state)
    text_result = result.get("text_result") or {}
    tier = text_result.get("tier", "L3_llm") if text_result else "L3_llm"

    return {
        "content_id": content_id, "decision": result.get("decision", "pass"),
        "confidence": result.get("confidence", 0.0), "reason": result.get("reason", ""),
        "tier": tier, "latency_ms": (time.perf_counter() - t0) * 1000,
        "traces": gw_traces + result.get("traces", []), "path": "cold",
    }


def compute_metrics(cases: list[dict], results: list[dict]) -> dict:
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

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0

    # Category breakdown
    cat_errors = Counter()
    for case, res in zip(cases, results):
        expected = case["expected"]
        predicted = res.get("decision", "pass")
        correct = (expected == "unsafe" and predicted in ("block", "review")) or \
                  (expected == "safe" and predicted == "pass")
        if not correct and case.get("category"):
            cat_errors[case["category"]] += 1

    return {
        "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "tier_distribution": dict(tier_counts),
        "path_distribution": dict(path_counts),
        "llm_call_rate": round(tier_counts.get("L3_llm", 0) / total, 4) if total > 0 else 0,
        "bert_intercept_rate": round(tier_counts.get("L2_bert", 0) / total, 4) if total > 0 else 0,
        "keyword_intercept_rate": round(tier_counts.get("L1_keyword", 0) / total, 4) if total > 0 else 0,
        "cache_hit_rate": round(
            sum(tier_counts.get(t, 0) for t in ["L0_memory", "L0_redis", "L0_whitelist",
                 "L0_phash", "L0_empty", "L1_chroma"]) / total, 4) if total > 0 else 0,
        "hot_path_rate": round(path_counts.get("hot", 0) / total, 4) if total > 0 else 0,
        "latency_p50_ms": round(p50, 2), "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2), "avg_latency_ms": round(sum(latencies) / total, 2) if total > 0 else 0,
        "top_error_categories": dict(cat_errors.most_common(5)),
    }


def print_report(metrics: dict, elapsed_s: float):
    print()
    print("=" * 60)
    print("  POC Benchmark — nvidia/Aegis")
    print("=" * 60)
    print(f"  Total: {metrics['total']} | Time: {elapsed_s:.1f}s | "
          f"Throughput: {metrics['total']/elapsed_s:.1f} req/s")
    print()
    print(f"  Confusion: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%} | Precision: {metrics['precision']:.1%} | "
          f"Recall: {metrics['recall']:.1%} | F1: {metrics['f1']:.1%}")
    print(f"  Latency: P50={metrics['latency_p50_ms']:.0f}ms P95={metrics['latency_p95_ms']:.0f}ms "
          f"P99={metrics['latency_p99_ms']:.0f}ms Avg={metrics['avg_latency_ms']:.0f}ms")
    print()
    print("  Funnel:")
    for tier, count in sorted(metrics["tier_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {tier:16s}: {count:4d} ({count/metrics['total']*100:.1f}%)")
    print()
    print(f"  Hot path: {metrics['hot_path_rate']:.1%} | LLM call: {metrics['llm_call_rate']:.1%} | "
          f"BERT intercept: {metrics['bert_intercept_rate']:.1%}")
    if metrics.get("top_error_categories"):
        print(f"  Top error categories: {metrics['top_error_categories']}")
    print()
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
    print("=" * 60)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Aegis Benchmark: n={args.n} concurrency={args.concurrency}")
    cases = load_aegis(n=args.n)

    sem = asyncio.Semaphore(args.concurrency)

    async def process_one(i: int, case: dict) -> dict:
        async with sem:
            res = await run_one(case["text"], f"aegis_{i}")
            res["text"] = case["text"]
            res["expected"] = case["expected"]
            res["category"] = case.get("category", "")
            return res

    t0 = time.perf_counter()
    tasks = [process_one(i, c) for i, c in enumerate(cases)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(cases, results)
    print_report(metrics, elapsed)

    # Show sample errors
    print("\n--- Sample Errors (first 8) ---")
    shown = 0
    for r in results:
        expected = r.get("expected", "?")
        predicted = r.get("decision", "?")
        correct = (expected == "unsafe" and predicted in ("block", "review")) or \
                  (expected == "safe" and predicted == "pass")
        if not correct and shown < 8:
            print(f"\n  Text: {r.get('text', '')[:150]}")
            print(f"  Expected: {expected} | Predicted: {predicted} | "
                  f"Conf: {r.get('confidence', 0):.2f} | Tier: {r.get('tier', '?')}")
            print(f"  Reason: {r.get('reason', '')[:150]}")
            shown += 1

    if args.output:
        output = {
            "dataset": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
            "config": {"n": len(cases), "concurrency": args.concurrency},
            "metrics": metrics,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
