"""
Data Asset Builder — transforms raw moderation events into reusable assets.

From the event stream, this builds:

  1. Labeled Dataset (fine-tuning)
     - High-confidence AI decisions as silver labels
     - Human-reviewed decisions as gold labels
     - Format: JSONL with {text, label, confidence, source}

  2. Keyword Discoveries
     - New slang/evasion terms found by LLM but not in keyword dict
     - Candidate keywords for dictionary expansion

  3. False Positive Patterns
     - Content blocked by AI but human-overridden to pass
     - Candidates for whitelist expansion

  4. Hard Case Collection
     - Grey zone + low confidence cases
     - For model evaluation and stress testing

  5. Model Accuracy Report
     - Agreement between BERT and LLM
     - Human override rate
     - Drift indicators

Usage (offline, scheduled):
  python -m src.feedback.dataset_builder --since 7d
"""

import json
import os
import time
import hashlib
import logging
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

ASSETS_DIR = os.getenv("FEEDBACK_ASSETS_DIR", "./data/assets")


class DatasetBuilder:
    """Build data assets from the moderation event stream."""

    def __init__(self, assets_dir: str | None = None):
        self.dir = assets_dir or ASSETS_DIR
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(f"{self.dir}/datasets", exist_ok=True)
        os.makedirs(f"{self.dir}/keywords", exist_ok=True)
        os.makedirs(f"{self.dir}/reports", exist_ok=True)

    # ---- 1. Labeled Dataset ----

    def build_labeled_dataset(
        self,
        events: list[dict],
        min_confidence: float = 0.90,
        include_human_reviewed: bool = True,
    ) -> dict:
        """Build a labeled dataset from moderation events.

        Silver labels: AI decisions with confidence >= min_confidence.
        Gold labels: human-reviewed decisions (from review queue).

        Returns summary dict with paths to generated files.
        """
        dataset = []
        stats = {"silver": 0, "gold": 0, "skipped_low_conf": 0}

        for event in events:
            text = event.get("text", "")
            if not text or not text.strip():
                continue

            decision = event.get("decision", "")
            confidence = event.get("confidence", 0.0)
            source = event.get("source_label", "ai")

            # Gold: human-reviewed
            if source == "human" and include_human_reviewed:
                dataset.append({
                    "id": hashlib.sha256(text.encode()).hexdigest()[:12],
                    "text": text,
                    "label": decision,
                    "confidence": 1.0,
                    "source": "human_review",
                    "timestamp": event.get("timestamp_iso", ""),
                })
                stats["gold"] += 1

            # Silver: high-confidence AI
            elif source == "ai" and confidence >= min_confidence:
                # Map decision to label
                label = _decision_to_label(decision)
                dataset.append({
                    "id": hashlib.sha256(text.encode()).hexdigest()[:12],
                    "text": text,
                    "label": label,
                    "confidence": confidence,
                    "source": "ai_high_conf",
                    "model_tier": event.get("tier", "unknown"),
                    "ai_reason": event.get("reason", ""),
                    "timestamp": event.get("timestamp_iso", ""),
                })
                stats["silver"] += 1
            else:
                stats["skipped_low_conf"] += 1

        # Write dataset
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = f"{self.dir}/datasets/labeled_{timestamp}.jsonl"
        with open(path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Write summary
        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "path": path,
            "total_events": len(events),
            "dataset_size": len(dataset),
            **stats,
            "label_distribution": dict(Counter(
                d["label"] for d in dataset
            )),
        }
        summary_path = path.replace(".jsonl", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Dataset built: %d samples → %s", len(dataset), path)
        return summary

    # ---- 2. Keyword Discovery ----

    def discover_keywords(
        self,
        events: list[dict],
        existing_keywords: set[str] | None = None,
    ) -> list[dict]:
        """Discover candidate keywords from LLM reasoning.

        When LLM blocks content for a specific reason (e.g., "使用了黑话'菠菜'
        指代赌博"), extract the new term that isn't in the existing dictionary.
        """
        existing = existing_keywords or set()
        candidates = []

        for event in events:
            reason = event.get("reason", "")
            decision = event.get("decision", "")
            text = event.get("text", "")

            if decision != "block":
                continue

            # Look for quoted terms in LLM reasoning
            # Pattern: "使用了'XXX'" or "contains 'XXX'" or "涉及"XXX""
            import re
            quoted = re.findall(r"['\"「『](.+?)['\"」』]", reason)
            for term in quoted:
                if len(term) >= 2 and term not in existing:
                    candidates.append({
                        "term": term,
                        "source_text": text[:100],
                        "ai_reason": reason,
                        "event_id": event.get("event_id", ""),
                    })

        # Deduplicate and rank by frequency
        term_counts = Counter(c["term"] for c in candidates)
        ranked = []
        for term, count in term_counts.most_common(50):
            # Find one example
            example = next(c for c in candidates if c["term"] == term)
            ranked.append({
                "term": term,
                "frequency": count,
                "example_text": example["source_text"],
                "example_reason": example["ai_reason"],
            })

        # Write discoveries
        if ranked:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"{self.dir}/keywords/discovered_{timestamp}.json"
            with open(path, "w") as f:
                json.dump(ranked, f, ensure_ascii=False, indent=2)
            logger.info("Keywords discovered: %d candidates → %s", len(ranked), path)

        return ranked

    # ---- 3. False Positive Patterns ----

    def find_false_positives(self, events: list[dict]) -> list[dict]:
        """Find patterns where AI blocked but human overrode to pass.

        These are candidates for whitelist expansion or BERT threshold tuning.
        """
        fp_cases = []

        for event in events:
            if event.get("ai_decision") == "block" and event.get("human_decision") == "pass":
                fp_cases.append({
                    "text": event.get("text", ""),
                    "ai_decision": event.get("ai_decision"),
                    "ai_confidence": event.get("ai_confidence", 0),
                    "ai_reason": event.get("ai_reason", ""),
                    "human_reason": event.get("human_reason", ""),
                    "event_id": event.get("event_id", ""),
                })

        if fp_cases:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"{self.dir}/keywords/false_positives_{timestamp}.json"
            with open(path, "w") as f:
                json.dump(fp_cases, f, ensure_ascii=False, indent=2)

        return fp_cases

    # ---- 4. Hard Case Collection ----

    def build_hard_cases(self, events: list[dict]) -> dict:
        """Collect hard/edge cases for model evaluation.

        Hard cases = grey zone decisions + low confidence + human overrides.
        These are the most valuable for testing model improvements.
        """
        hard = []

        for event in events:
            confidence = event.get("confidence", 0)
            decision = event.get("decision", "")
            is_grey = 0.3 <= confidence <= 0.7

            # Grey zone cases
            if is_grey:
                hard.append({
                    "text": event.get("text", ""),
                    "label": _decision_to_label(decision),
                    "confidence": confidence,
                    "difficulty": "grey_zone",
                    "ai_reason": event.get("reason", ""),
                })
            # Human overrides (AI was wrong)
            elif event.get("ai_decision") != event.get("human_decision"):
                hard.append({
                    "text": event.get("text", ""),
                    "label": _decision_to_label(event.get("human_decision", "pass")),
                    "confidence": confidence,
                    "difficulty": "ai_wrong",
                    "ai_reason": event.get("ai_reason", ""),
                    "human_reason": event.get("human_reason", ""),
                })

        if hard:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"{self.dir}/datasets/hard_cases_{timestamp}.jsonl"
            with open(path, "w") as f:
                for item in hard:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return {
            "total": len(hard),
            "path": path if hard else None,
        }

    # ---- 5. Accuracy Report ----

    def generate_report(self, events: list[dict]) -> dict:
        """Generate model accuracy and health report."""
        if not events:
            return {"error": "no events"}

        total = len(events)

        # Tier distribution
        tiers = Counter(
            e.get("tier", e.get("text_result", {}).get("tier", "unknown"))
            for e in events
        )

        # Human override rate
        human_reviewed = [e for e in events if e.get("ai_decision") == "review"]
        overrides = [e for e in events
                     if e.get("ai_decision") and e.get("human_decision")
                     and e.get("ai_decision") != e.get("human_decision")]

        # BERT vs LLM agreement (for cold path events with both)
        bert_llm_agree = 0
        bert_llm_total = 0
        for e in events:
            tr = e.get("text_result", {})
            if tr.get("tier") == "L3_llm" and tr.get("bert_preliminary"):
                bert_llm_total += 1
                if tr.get("bert_preliminary") == tr.get("label"):
                    bert_llm_agree += 1

        # Confidence distribution
        confs = [e.get("confidence", 0) for e in events if e.get("confidence")]
        conf_bins = {"0.0-0.3": 0, "0.3-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
        for c in confs:
            if c < 0.3: conf_bins["0.0-0.3"] += 1
            elif c < 0.7: conf_bins["0.3-0.7"] += 1
            elif c < 0.9: conf_bins["0.7-0.9"] += 1
            else: conf_bins["0.9-1.0"] += 1

        # Decision distribution
        decisions = Counter(e.get("decision", "?") for e in events)

        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_events": total,
            "tier_distribution": dict(tiers),
            "human_review_rate": round(len(human_reviewed) / max(total, 1), 4),
            "human_override_rate": round(len(overrides) / max(len(human_reviewed), 1), 4),
            "bert_llm_agreement": round(bert_llm_agree / max(bert_llm_total, 1), 4) if bert_llm_total > 0 else None,
            "confidence_distribution": conf_bins,
            "decision_distribution": dict(decisions),
            "events_per_decision": {
                "pass": decisions.get("pass", 0),
                "block": decisions.get("block", 0),
                "review": decisions.get("review", 0),
            },
        }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = f"{self.dir}/reports/accuracy_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("Report generated: %s", path)
        return report


def _decision_to_label(decision: str) -> str:
    """Normalize decision to dataset label."""
    if decision in ("pass", "safe", "normal"):
        return "safe"
    elif decision in ("block", "unsafe"):
        return "unsafe"
    elif decision == "review":
        return "unsafe"  # review cases are suspicious
    return "safe"


# Singleton
dataset_builder = DatasetBuilder()
