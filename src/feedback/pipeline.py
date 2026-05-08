#!/usr/bin/env python3
"""
Offline Feedback Pipeline — scheduled data asset generation.

This runs OUTSIDE the online request path, on a schedule (e.g., cron, Airflow).
It reads the event store + review queue, and produces:

  1. Labeled datasets (for model fine-tuning)
  2. Keyword discoveries (new slang/evasion terms)
  3. False positive patterns (candidates for whitelist)
  4. Hard case collections (evaluation sets)
  5. Accuracy & health reports

Usage:
  python -m src.feedback.pipeline                    # full pipeline
  python -m src.feedback.pipeline --dataset-only     # only build datasets
  python -m src.feedback.pipeline --since 7          # last 7 days of events
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.feedback.event_collector import event_collector
from src.feedback.dataset_builder import dataset_builder
from src.feedback.finetune_trigger import finetune_trigger
from src.skills.review_queue import review_queue
from src.skills.keyword_filter import keyword_filter


def load_events(limit: int = 50000) -> list[dict]:
    """Load events from the persistent store."""
    return event_collector.read_all(limit)


def load_human_reviewed() -> list[dict]:
    """Load human-reviewed items and convert to event format."""
    all_items = review_queue.get_pending(1000)  # gets pending only
    # Also get reviewed items by reading the queue file directly
    reviewed = []
    queue_path = review_queue.path
    if os.path.exists(queue_path):
        with open(queue_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("status") == "reviewed":
                    reviewed.append({
                        "content_id": item.get("content_id", ""),
                        "text": item.get("text", ""),
                        "decision": item.get("human_decision", "pass"),
                        "confidence": 1.0,
                        "reason": item.get("human_reason", ""),
                        "source_label": "human",
                        "ai_decision": item.get("ai_decision", ""),
                        "ai_confidence": item.get("ai_confidence", 0),
                        "human_decision": item.get("human_decision", ""),
                        "human_reason": item.get("human_reason", ""),
                        "tier": "human_review",
                    })
    return reviewed


def run_full_pipeline():
    """Run the complete offline feedback pipeline."""
    print("=" * 60)
    print("Offline Feedback Pipeline")
    print("=" * 60)

    # Load data
    t0 = time.perf_counter()
    events = load_events()
    human_events = load_human_reviewed()
    all_events = events + human_events
    print(f"\nLoaded: {len(events)} AI events + {len(human_events)} human-reviewed")
    print(f"Event store: {event_collector.path} ({event_collector.count()} total)")

    if not all_events:
        print("\nNo events to process. Run some moderation requests first.")
        return

    # 1. Build labeled dataset
    print("\n--- 1. Labeled Dataset ---")
    ds = dataset_builder.build_labeled_dataset(all_events, min_confidence=0.90)
    print(f"  Silver (AI high-conf): {ds['silver']}")
    print(f"  Gold (human-reviewed): {ds['gold']}")
    print(f"  Skipped (low conf):    {ds['skipped_low_conf']}")
    print(f"  Total dataset:         {ds['dataset_size']} samples")
    print(f"  Labels: {ds['label_distribution']}")

    # 2. Discover new keywords
    print("\n--- 2. Keyword Discovery ---")
    existing = set()
    for words in keyword_filter.keywords.values():
        existing.update(w.lower() for w in words)
    discovered = dataset_builder.discover_keywords(all_events, existing)
    if discovered:
        for kw in discovered[:10]:
            print(f"  {kw['term']:15s} (×{kw['frequency']:<3}) ← {kw['example_text'][:40]}")
    else:
        print("  No new keywords discovered")

    # 3. False positive patterns
    print("\n--- 3. False Positive Patterns ---")
    fp = dataset_builder.find_false_positives(all_events)
    print(f"  AI-blocked but human-passed: {len(fp)} cases")
    if fp:
        for case in fp[:5]:
            print(f"  {case['text'][:50]}")

    # 4. Hard case collection
    print("\n--- 4. Hard Case Collection ---")
    hc = dataset_builder.build_hard_cases(all_events)
    print(f"  Hard cases: {hc['total']}")

    # 5. Accuracy report
    print("\n--- 5. Accuracy & Health Report ---")
    report = dataset_builder.generate_report(all_events)
    print(f"  Tier distribution:     {report.get('tier_distribution', {})}")
    print(f"  Human review rate:     {report.get('human_review_rate', 0):.1%}")
    print(f"  Human override rate:   {report.get('human_override_rate', 0):.1%}")
    if report.get('bert_llm_agreement') is not None:
        print(f"  BERT/LLM agreement:    {report['bert_llm_agreement']:.1%}")
    print(f"  Confidence dist:       {report.get('confidence_distribution', {})}")

    # 6. Fine-tune readiness
    print("\n--- 6. Fine-Tune Trigger ---")
    # Record new labels from human review
    human_label_count = len(human_events)
    if human_label_count > 0:
        finetune_trigger.add_labels(human_label_count)

    should_ft, ft_reason = finetune_trigger.should_finetune()
    status = finetune_trigger.get_status()
    print(f"  Labels since last FT:  {status['labels_since_last_finetune']}/{status['finetune_threshold']}")
    print(f"  Total labels:          {status['total_labels_collected']}")
    print(f"  Should fine-tune:      {should_ft}")
    print(f"  Reason:                {ft_reason}")

    # 7. Drift detection
    print("\n--- 7. Drift Detection ---")
    # Record accuracy from report (if available)
    if report.get("human_override_rate") is not None:
        accuracy = 1.0 - report["human_override_rate"]
        finetune_trigger.record_accuracy(accuracy, len(all_events))

    has_drift, drift_msg, drift_metrics = finetune_trigger.detect_drift()
    print(f"  Drift detected: {has_drift}")
    print(f"  {drift_msg}")
    if drift_metrics:
        print(f"  Recent 7d: {drift_metrics.get('recent_7d_avg', '?')}")
        print(f"  Baseline:  {drift_metrics.get('baseline_30d_avg', '?')}")

    # 8. Threshold calibration
    print("\n--- 8. Threshold Calibration ---")
    old_thresholds = dict(finetune_trigger.state["thresholds"])
    human_accuracy = report.get("human_override_rate", 0)
    human_accuracy = 1.0 - human_accuracy if human_accuracy else 0.95
    new_thresholds = finetune_trigger.calibrate_thresholds(
        human_accuracy=human_accuracy,
        bert_accuracy=report.get("bert_llm_agreement") or 0.5,
        sample_size=len(all_events),
    )
    for key in old_thresholds:
        old = old_thresholds[key]
        new = new_thresholds.get(key, old)
        if old != new:
            print(f"  {key}: {old} → {new} (updated)")
        else:
            print(f"  {key}: {old} (unchanged)")

    # Asset inventory
    print(f"\n{'='*60}")
    print("Generated Assets:")
    print(f"  {dataset_builder.dir}/datasets/   ← labeled datasets")
    print(f"  {dataset_builder.dir}/keywords/   ← discovered keywords + FP patterns")
    print(f"  {dataset_builder.dir}/reports/    ← accuracy reports")
    print(f"  {finetune_trigger.path}           ← fine-tune state")
    if should_ft:
        print(f"\n  ⚠  Fine-tune recommended! Run:")
        print(f"  python -m src.feedback.train_lora --dry-run")
    print(f"\nElapsed: {time.perf_counter() - t0:.1f}s")
    print("=" * 60)


def run_dataset_only():
    """Build dataset only (faster, for frequent runs)."""
    events = load_events()
    human_events = load_human_reviewed()
    all_events = events + human_events
    if all_events:
        ds = dataset_builder.build_labeled_dataset(all_events)
        print(f"Dataset: {ds['dataset_size']} samples → {ds['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Feedback Pipeline")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--since", type=int, default=30, help="Days of events to process")
    args = parser.parse_args()

    if args.dataset_only:
        run_dataset_only()
    else:
        run_full_pipeline()
