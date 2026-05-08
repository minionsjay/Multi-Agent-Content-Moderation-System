#!/usr/bin/env python3
"""Benchmark: Full Cold Path End-to-End

Runs complete LangGraph pipeline on various inputs and measures:
  1. End-to-end latency for each path
  2. Tier distribution (which layer made the final decision)
  3. Cost per path
  4. Traces completeness

Usage:
  python bench.py              # full cold path test
  python bench.py --count 5    # run 5 iterations per case
"""

import sys, os, time, asyncio, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.graph import graph
from src.state import ModerationState


def make_state(text: str = "", image_url: str = "",
               image_base64: str = "", **overrides) -> ModerationState:
    """Build initial state for LangGraph."""
    return {
        "content_id": f"bench_{int(time.time()*1000)}",
        "text": text,
        "image_url": image_url,
        "image_base64": image_base64,
        "user_id": "bench",
        "source": "bench",
        "content_type": "text_only" if not image_url else "mixed",
        "cache_hit": False,
        "cached_decision": None,
        "keyword_confidence": 0.0,
        "keyword_label": None,
        "keyword_prefiltered": True,  # Gateway already scanned
        "priority_score": 0.3,
        "text_result": None,
        "image_result": None,
        "decision": "pass",
        "confidence": 0.0,
        "reason": "",
        "traces": [],
    }


async def run_case(name: str, state: ModerationState, expected_path: str):
    """Run one case through the full graph."""
    t0 = time.perf_counter()
    result = await graph.ainvoke(state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    decision = result.get("decision", "?")
    text_result = result.get("text_result") or {}
    image_result = result.get("image_result") or {}
    tier = text_result.get("tier", image_result.get("tier", "?"))
    traces = [t["step"] for t in result.get("traces", [])]

    print(f"  {name:30s}: {elapsed_ms:7.0f}ms | decision={decision:6s} "
          f"tier={tier:12s} | traces={traces}")
    return {"name": name, "latency_ms": elapsed_ms, "decision": decision,
            "tier": tier, "traces": traces}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1, help="Iterations per case")
    args = parser.parse_args()

    print("=" * 70)
    print("Cold Path End-to-End Benchmark")
    print("=" * 70)
    print(f"Iterations per case: {args.count}\n")

    # Warmup
    await graph.ainvoke(make_state("warmup test"))

    all_results = []

    # ---- Text-only cases ----
    print("--- Text-Only Path ---")
    cases = [
        ("safe EN (L2 BERT)", "today is a beautiful day for a walk", "pass"),
        ("toxic EN (L2 BERT)", "you are a worthless piece of shit", "block"),
        ("safe CN (L3 LLM)", "今天天气真好适合出去玩", "pass"),
        ("toxic CN (L3 LLM)", "你真是个傻逼什么都不懂垃圾", "block"),
        ("political CN (L3 LLM)", "支持台独台湾是一个独立国家", "block"),
        ("grey zone case", "你说的也不是完全没道理但有点过了", "review"),
    ]

    for name, text, expected in cases:
        for i in range(args.count):
            state = make_state(text=text)
            r = await run_case(f"{name} #{i+1}" if args.count > 1 else name, state, expected)
            all_results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    tiers = {}
    for r in all_results:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

    lats = [r["latency_ms"] for r in all_results]
    lats.sort()
    n = len(lats)

    print(f"  Total requests: {n}")
    print(f"  Latency: P50={lats[n//2]:.0f}ms P95={lats[int(n*0.95)]:.0f}ms "
          f"Avg={sum(lats)/n:.0f}ms")
    print(f"  Tier distribution:")
    for tier, count in sorted(tiers.items()):
        print(f"    {tier:15s}: {count:>3} ({count/n*100:5.1f}%)")

    # Cost estimate
    l3_count = tiers.get("L3_llm", 0)
    l2_count = tiers.get("L2_bert", 0)
    total_cost = l3_count * 0.002 + l2_count * 0.0001
    print(f"\n  Estimated cost:")
    print(f"    L2 BERT calls: {l2_count} × $0.0001 = ${l2_count*0.0001:.4f}")
    print(f"    L3 LLM calls:  {l3_count} × $0.002  = ${l3_count*0.002:.4f}")
    print(f"    Total: ${total_cost:.4f}")

    print(f"\n✅ Cold path end-to-end benchmark complete")


if __name__ == "__main__":
    asyncio.run(main())
