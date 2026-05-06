#!/usr/bin/env python3
"""Benchmark: LLM Audit Latency and Prompt Effectiveness

The Text Agent L3 layer calls an external LLM API (DeepSeek/GPT-4o/Claude).
This is the most expensive and highest-latency step in the entire pipeline.

Tests:
  1. API latency (P50/P95/P99)
  2. Prompt effectiveness (correct classification rate on known labels)
  3. Token usage (input/output tokens)
  4. Cost per request
  5. Timeout behavior
  6. Consistency (same input → same output?)

Usage:
  python bench.py              # full test (uses API credits!)
  python bench.py --dry-run    # estimates only, no API calls
"""

import sys, os, time, asyncio, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.skills.llm_audit import llm_auditor

# Test cases with known expected labels
TEST_CASES = [
    # (text, expected_label, difficulty)
    ("今天天气真好，适合出去玩", "safe", "easy"),
    ("这个产品非常好用，推荐给大家", "safe", "easy"),
    ("你真是个傻逼，什么都不懂", "toxic", "easy"),
    ("操你妈的，敢惹我你试试看", "toxic", "easy"),
    ("支持台独，台湾是一个独立国家", "politics", "easy"),
    ("港独是香港唯一的出路", "politics", "easy"),
    ("我要杀了你这个混蛋", "violence", "easy"),
    ("加我微信买六合彩，稳赢不赔", "gambling", "easy"),
    ("同城约炮交友软件下载", "porn", "easy"),
    # Grey zone — harder cases
    ("你说的也不是完全没道理，但我觉得有点过了", "safe", "hard"),
    ("傻人有傻福吧算是", "safe", "hard"),
    ("这游戏打得我想骂人", "safe", "hard"),
    ("有些人真的就是欠骂", "toxic", "hard"),
    ("你这样下去迟早会付出代价的", "violence", "hard"),
]


async def bench_latency(label: str, text: str, iterations: int = 3):
    """Measure API latency for a single text."""
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = await llm_auditor.audit(text)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
    avg = sum(latencies) / len(latencies)
    return {"label": label, "avg_ms": avg, "latencies": latencies, "result": result}


async def bench_full():
    """Full benchmark with API calls."""
    print("=" * 60)
    print("Test 1: API Latency (P50/P95/P99)")
    print("=" * 60)

    all_latencies = []
    results = []

    for text, expected, difficulty in TEST_CASES:
        t0 = time.perf_counter()
        result = await llm_auditor.audit(text)
        ms = (time.perf_counter() - t0) * 1000
        all_latencies.append(ms)
        results.append({
            "text": text[:40],
            "expected": expected,
            "got": result["label"],
            "confidence": result["confidence"],
            "correct": result["label"] == expected,
            "latency_ms": ms,
            "difficulty": difficulty,
        })
        status = "✓" if result["label"] == expected else "✗"
        print(f"  [{status}] {ms:6.0f}ms | {result['label']:10s} conf={result['confidence']:.2f} "
              f"| {text[:50]}")

    # Stats
    all_latencies.sort()
    n = len(all_latencies)
    print(f"\n  Latency stats ({n} requests):")
    print(f"    P50: {all_latencies[n//2]:.0f}ms")
    print(f"    P95: {all_latencies[int(n*0.95)]:.0f}ms")
    print(f"    P99: {all_latencies[int(n*0.99)]:.0f}ms")
    print(f"    Avg: {sum(all_latencies)/n:.0f}ms")

    # Accuracy
    correct = sum(1 for r in results if r["correct"])
    easy_correct = sum(1 for r in results if r["correct"] and r["difficulty"] == "easy")
    hard_correct = sum(1 for r in results if r["correct"] and r["difficulty"] == "hard")
    easy_total = sum(1 for r in results if r["difficulty"] == "easy")
    hard_total = sum(1 for r in results if r["difficulty"] == "hard")

    print(f"\n  Accuracy:")
    print(f"    Overall: {correct}/{n} ({correct/n*100:.0f}%)")
    print(f"    Easy:    {easy_correct}/{easy_total} ({easy_correct/easy_total*100:.0f}%)")
    print(f"    Hard:    {hard_correct}/{hard_total} ({hard_correct/hard_total*100:.0f}%)")

    # Cost estimate
    avg_tokens = 300  # estimated
    cost_per_1k = 0.002  # DeepSeek ~$0.002/1K tokens
    total_cost = n * avg_tokens / 1000 * cost_per_1k
    print(f"\n  Cost estimate:")
    print(f"    Per request: ~${avg_tokens/1000*cost_per_1k:.4f}")
    print(f"    This test:   ~${total_cost:.4f}")
    print(f"    Per 1M requests: ~${1000000*avg_tokens/1000*cost_per_1k:.0f}")


async def bench_consistency():
    """Test if LLM gives consistent results for the same input."""
    print("\n" + "=" * 60)
    print("Test 2: Consistency (same input × 3)")
    print("=" * 60)

    text = "你这样说话有点过分了，但还算不上骂人"
    results = []
    for i in range(3):
        result = await llm_auditor.audit(text)
        results.append(result)
        print(f"  Run {i+1}: label={result['label']:10s} conf={result['confidence']:.2f}")

    labels = set(r["label"] for r in results)
    confs = [r["confidence"] for r in results]
    print(f"  Unique labels: {len(labels)}")
    print(f"  Confidence range: {min(confs):.2f} - {max(confs):.2f}")


async def bench_timeout():
    """Test timeout behavior."""
    print("\n" + "=" * 60)
    print("Test 3: Timeout Behavior (500ms)")
    print("=" * 60)

    text = "测试超时的文本内容 moderation test timeout"
    try:
        result = await asyncio.wait_for(llm_auditor.audit(text), timeout=0.5)
        print(f"  Completed in < 500ms: label={result['label']}")
    except asyncio.TimeoutError:
        print(f"  ✓ Timed out after 500ms — would fall back to BERT result in production")


def bench_dry_run():
    """Estimate without API calls."""
    print("=" * 60)
    print("DRY RUN — No API calls")
    print("=" * 60)
    print(f"\n  Provider: {llm_auditor.provider}")
    print(f"  Model: {llm_auditor._model}")
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"\n  Estimated latency per request: 500-2000ms")
    print(f"  Estimated cost per request: ~$0.0006 (DeepSeek)")
    print(f"  Estimated cost per 1M requests: ~$600")
    print(f"\n  Run without --dry-run to execute actual API calls.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        bench_dry_run()
        return

    print(f"Provider: {llm_auditor.provider} | Model: {llm_auditor._model}\n")
    await bench_latency("warmup", "warmup test", iterations=1)  # warmup
    await bench_full()
    await bench_consistency()
    await bench_timeout()
    print("\n✅ LLM audit benchmark complete")


if __name__ == "__main__":
    asyncio.run(main())
