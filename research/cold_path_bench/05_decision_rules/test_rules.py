#!/usr/bin/env python3
"""Test: Decision Agent Rule Engine Validation

The Decision Agent applies policy rules AFTER Text Agent and Image Agent
produce their results. It handles:
  1. Cache hit reuse
  2. Zero-tolerance hard override
  3. Text + Image result aggregation
  4. Grey zone triage ([0.3, 0.7] → human review)
  5. Normal pass/block classification

Tests verify each rule fires correctly and edge cases are handled.

Usage:
  python test_rules.py
"""

import sys, os, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.agents.decision import decision_aggregator
from src.state import ModerationState


def make_state(**overrides) -> ModerationState:
    """Build minimal state for testing Decision Agent."""
    base: ModerationState = {
        "content_id": "test_001",
        "text": "test text",
        "user_id": "test_user",
        "source": "test",
        "content_type": "text_only",
        "cache_hit": False,
        "cached_decision": None,
        "keyword_confidence": 0.0,
        "keyword_label": None,
        "keyword_prefiltered": False,
        "priority_score": 0.3,
        "text_result": None,
        "image_result": None,
        "decision": "pass",
        "confidence": 0.0,
        "reason": "",
        "traces": [],
    }
    base.update(overrides)  # type: ignore
    return base


async def run_test(name: str, state: ModerationState, expected_decision: str):
    result = await decision_aggregator(state)
    actual = result.get("decision", "?")
    passed = actual == expected_decision
    status = "✓" if passed else "✗"
    print(f"  [{status}] {name:35s}: expect={expected_decision:8s} got={actual:8s} "
          f"conf={result.get('confidence', 0):.2f}")
    return passed


async def main():
    print("=" * 70)
    print("Decision Agent Rule Engine Validation")
    print("=" * 70)

    passed = 0
    failed = 0

    # ---- Path 1: Cache hit ----
    print("\n--- Path 1: Cache Hit ---")
    p = await run_test("cache hit → pass",
        make_state(cache_hit=True, cached_decision={"decision": "pass", "confidence": 0.95, "reason": "cached"}),
        "pass")
    passed += p; failed += not p

    p = await run_test("cache hit → block",
        make_state(cache_hit=True, cached_decision={"decision": "block", "confidence": 1.0, "reason": "cached"}),
        "block")
    passed += p; failed += not p

    # ---- Path 2: Zero-tolerance ----
    print("\n--- Path 2: Zero-Tolerance Hard Override ---")
    p = await run_test("zero-tol politics → block",
        make_state(keyword_confidence=1.0, keyword_label="politics", decision="block", confidence=1.0, reason="ZT"),
        "block")
    passed += p; failed += not p

    p = await run_test("zero-tol violence → block",
        make_state(keyword_confidence=1.0, keyword_label="violence", decision="block", confidence=1.0, reason="ZT"),
        "block")
    passed += p; failed += not p

    p = await run_test("toxic (not zero-tol) → pass through",
        make_state(keyword_confidence=1.0, keyword_label="toxic", decision="block", confidence=1.0),
        "block")
    passed += p; failed += not p

    # ---- Path 3: Text + Image aggregation ----
    print("\n--- Path 3: Text + Image Aggregation ---")
    p = await run_test("text safe, no image → pass",
        make_state(text_result={"label": "safe", "confidence": 0.95, "tier": "L2_bert"}),
        "pass")
    passed += p; failed += not p

    p = await run_test("text unsafe, no image → block",
        make_state(text_result={"label": "unsafe", "confidence": 0.95, "tier": "L2_bert"}),
        "block")
    passed += p; failed += not p

    p = await run_test("text safe + image NSFW → unsafe/block",
        make_state(text_result={"label": "safe", "confidence": 0.9, "tier": "L3_llm"},
                   image_result={"label": "nsfw", "confidence": 0.8}),
        "block")
    passed += p; failed += not p

    p = await run_test("text unsafe + image NSFW → block",
        make_state(text_result={"label": "unsafe", "confidence": 0.95, "tier": "L3_llm"},
                   image_result={"label": "nsfw", "confidence": 0.8}),
        "block")
    passed += p; failed += not p

    p = await run_test("image only, NSFW → block",
        make_state(text_result=None, image_result={"label": "nsfw", "confidence": 0.8}),
        "block")
    passed += p; failed += not p

    p = await run_test("image only, normal → pass",
        make_state(text_result=None, image_result={"label": "normal", "confidence": 0.9}),
        "pass")
    passed += p; failed += not p

    p = await run_test("image only, low conf NSFW → pass",
        make_state(text_result=None, image_result={"label": "nsfw", "confidence": 0.4}),
        "pass")
    passed += p; failed += not p

    p = await run_test("no text, no image → fallback pass",
        make_state(text_result=None, image_result={}),
        "pass")
    passed += p; failed += not p

    # ---- Path 4: Grey zone ----
    print("\n--- Path 4: Grey Zone Triage ---")
    p = await run_test("conf=0.50 (in grey zone) → review",
        make_state(text_result={"label": "unsafe", "confidence": 0.50, "tier": "L3_llm"}),
        "review")
    passed += p; failed += not p

    p = await run_test("conf=0.30 (grey zone low) → review",
        make_state(text_result={"label": "unsafe", "confidence": 0.30, "tier": "L3_llm"}),
        "review")
    passed += p; failed += not p

    p = await run_test("conf=0.70 (grey zone high) → review",
        make_state(text_result={"label": "unsafe", "confidence": 0.70, "tier": "L3_llm"}),
        "review")
    passed += p; failed += not p

    p = await run_test("conf=0.29 (below grey) → pass",
        make_state(text_result={"label": "unsafe", "confidence": 0.29, "tier": "L3_llm"}),
        "pass")
    passed += p; failed += not p

    p = await run_test("conf=0.71 (above grey) → block",
        make_state(text_result={"label": "unsafe", "confidence": 0.71, "tier": "L3_llm"}),
        "block")
    passed += p; failed += not p

    p = await run_test("conf=0.20 (deep below) → pass",
        make_state(text_result={"label": "unsafe", "confidence": 0.20, "tier": "L3_llm"}),
        "pass")
    passed += p; failed += not p

    p = await run_test("conf=0.95 (deep above) → block",
        make_state(text_result={"label": "unsafe", "confidence": 0.95, "tier": "L2_bert"}),
        "block")
    passed += p; failed += not p

    # ---- Edge cases ----
    print("\n--- Edge Cases ---")
    p = await run_test("text safe conf=0.50 → pass (not grey)",
        make_state(text_result={"label": "safe", "confidence": 0.50, "tier": "L3_llm"}),
        "pass")
    passed += p; failed += not p

    # Summary
    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    print(f"{'='*70}")

    # Grey zone coverage analysis
    print(f"\nGrey Zone Analysis:")
    print(f"  Zone: [{0.3}, {0.7}]")
    print(f"  Below zone (< 0.3): pass (宁可放过)")
    print(f"  In zone [0.3-0.7]: review (人工复核)")
    print(f"  Above zone (> 0.7): block (直接拦截)")
    print(f"  Note: safe label always passes regardless of confidence")


if __name__ == "__main__":
    asyncio.run(main())
