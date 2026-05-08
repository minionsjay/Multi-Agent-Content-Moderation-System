#!/usr/bin/env python3
"""POC Benchmark: run the moderation graph over a test dataset and report metrics."""

import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.dataset import load_dataset
from src.graph import graph
from src.state import ModerationState


async def run_one(text: str, content_id: str = "bench") -> dict:
    """Run moderation on a single text."""
    state: ModerationState = {
        "content_id": content_id,
        "text": text,
        "user_id": "benchmark_user",
        "source": "benchmark",
        "content_type": "text_only",
        "cache_hit": False,
        "cached_decision": None,
        "keyword_confidence": 0.0,
        "keyword_label": None,
        "keyword_prefiltered": False,
        "priority_score": 0.3,
        "text_result": None,
        "decision": "pass",
        "confidence": 0.0,
        "reason": "",
    }
    result = await graph.ainvoke(state)
    return result


def compute_metrics(cases: list[dict], results: list[dict]) -> dict:
    """Compute accuracy, precision, recall, F1, and tier distribution."""
    tp = tn = fp = fn = 0
    tier_counts = {"cache": 0, "L1_keyword": 0, "L2_bert": 0, "L3_llm": 0, "L0_empty": 0}

    for case, res in zip(cases, results):
        expected = case.get("expected", case.get("label", "safe"))
        predicted = res.get("decision", "pass")
        tier = res.get("text_result", {}).get("tier", "cache")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

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

    return {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tier_distribution": tier_counts,
        "llm_call_rate": round(tier_counts.get("L3_llm", 0) / total, 4) if total > 0 else 0,
        "bert_intercept_rate": round(tier_counts.get("L2_bert", 0) / total, 4) if total > 0 else 0,
        "keyword_intercept_rate": round(tier_counts.get("L1_keyword", 0) / total, 4) if total > 0 else 0,
        "cache_hit_rate": round(tier_counts.get("cache", 0) / total, 4) if total > 0 else 0,
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="POC Benchmark")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset")
    parser.add_argument("--slow", action="store_true", help="Run sequentially (easier to debug)")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    print(f"Loaded {len(cases)} test cases\n")

    t0 = time.perf_counter()

    if args.slow:
        results = []
        for i, case in enumerate(cases):
            res = await run_one(case["text"], f"bench_{i}")
            results.append(res)
    else:
        tasks = [run_one(c["text"], f"bench_{i}") for i, c in enumerate(cases)]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(cases, results)

    print("=" * 50)
    print("POC Benchmark Results")
    print("=" * 50)
    print(f"Total cases:       {metrics['total']}")
    print(f"Time elapsed:      {elapsed:.2f}s")
    print(f"Avg latency:       {elapsed / metrics['total'] * 1000:.2f}ms/req")
    print(f"Throughput:        {metrics['total'] / elapsed:.1f} req/s")
    print()
    print("Confusion Matrix:")
    print(f"  TP={metrics['tp']:3d}  FP={metrics['fp']:3d}")
    print(f"  FN={metrics['fn']:3d}  TN={metrics['tn']:3d}")
    print()
    print(f"Accuracy:          {metrics['accuracy']:.2%}")
    print(f"Precision:         {metrics['precision']:.2%}")
    print(f"Recall:            {metrics['recall']:.2%}")
    print(f"F1-score:          {metrics['f1']:.2%}")
    print()
    print("Tier Distribution (L1/L2/L3 funnel):")
    for tier, count in sorted(metrics["tier_distribution"].items()):
        pct = count / metrics["total"] * 100
        print(f"  {tier:12s}: {count:3d} ({pct:5.1f}%)")
    print()
    print("Key Metrics:")
    print(f"  LLM call rate:    {metrics['llm_call_rate']:.2%}")
    print(f"  BERT intercept:   {metrics['bert_intercept_rate']:.2%}")
    print(f"  Keyword intercept:{metrics['keyword_intercept_rate']:.2%}")
    print(f"  Cache hit rate:   {metrics['cache_hit_rate']:.2%}")
    print("=" * 50)

    # POC acceptance criteria check
    print()
    print("Acceptance Criteria:")
    checks = []
    checks.append(("LLM call rate < 20%", metrics["llm_call_rate"] < 0.20))
    checks.append(("BERT intercept > 30%", metrics["bert_intercept_rate"] > 0.30))
    checks.append(("F1-score > 90%", metrics["f1"] > 0.90))
    for criteria, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criteria}")


if __name__ == "__main__":
    asyncio.run(main())
