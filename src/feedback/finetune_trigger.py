"""
Fine-tune Trigger & Model Health Monitoring.

Decides WHEN to trigger model updates based on data accumulation
and accuracy metrics. Separate from the training itself.

Three mechanisms:
  1. Label Threshold: N new human labels since last fine-tune → trigger
  2. Drift Detection: 7-day accuracy < 30-day baseline − 5% → alert
  3. Threshold Calibration: Bayesian update of BERT confidence thresholds
     based on human reviewer accuracy rates
"""

import json
import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

STATE_PATH = os.getenv("FINETUNE_STATE_PATH", "./data/finetune_state.json")

# Defaults from CLAUDE.md design
FINETUNE_LABEL_THRESHOLD = int(os.getenv("FINETUNE_LABEL_THRESHOLD", "5000"))
DRIFT_ALERT_DROP = float(os.getenv("DRIFT_ALERT_DROP", "0.05"))  # 5% drop

# BERT confidence thresholds (adjustable via calibration)
from src.config import BERT_HIGH_CONFIDENCE, BERT_LOW_CONFIDENCE, GREY_ZONE_LOW, GREY_ZONE_HIGH


class FinetuneTrigger:
    """Monitors data accumulation and triggers fine-tuning when ready."""

    def __init__(self, state_path: str | None = None):
        self.path = state_path or STATE_PATH
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {
            "labels_since_last_finetune": 0,
            "total_labels_collected": 0,
            "last_finetune_at": None,
            "last_finetune_dataset_size": 0,
            "finetune_count": 0,
            "thresholds": {
                "bert_high": BERT_HIGH_CONFIDENCE,
                "bert_low": BERT_LOW_CONFIDENCE,
                "grey_low": GREY_ZONE_LOW,
                "grey_high": GREY_ZONE_HIGH,
            },
            "accuracy_history": [],  # list of {date, accuracy, sample_size}
            "drift_alerts": [],
        }

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    # ---- Label Collection ----

    def add_labels(self, count: int):
        """Record new labels collected (from human review resolutions)."""
        self.state["labels_since_last_finetune"] += count
        self.state["total_labels_collected"] += count
        self._save()

    # ---- Fine-tune Trigger ----

    def should_finetune(self) -> tuple[bool, str]:
        """Check if enough labels have accumulated to trigger fine-tuning.

        Returns (should_trigger, reason).
        """
        count = self.state["labels_since_last_finetune"]
        threshold = FINETUNE_LABEL_THRESHOLD

        if count >= threshold:
            return True, (
                f"{count} labels accumulated (threshold={threshold}) — "
                f"ready for fine-tuning"
            )
        else:
            progress = count / threshold * 100
            return False, (
                f"{count}/{threshold} labels ({progress:.1f}%) — "
                f"need {threshold - count} more"
            )

    def mark_finetune_started(self, dataset_size: int):
        """Record that a fine-tuning run has started."""
        self.state["labels_since_last_finetune"] = 0
        self.state["last_finetune_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.state["last_finetune_dataset_size"] = dataset_size
        self.state["finetune_count"] += 1
        self._save()
        logger.info("Fine-tune #%d started with %d samples",
                    self.state["finetune_count"], dataset_size)

    # ---- Threshold Calibration ----

    def calibrate_thresholds(
        self,
        human_accuracy: float,
        bert_accuracy: float,
        sample_size: int,
    ) -> dict:
        """Bayesian update of confidence thresholds based on human accuracy.

        If human reviewers consistently overturn AI decisions at a certain
        confidence level, the threshold should be adjusted.

        POC implementation: simple rolling average. Production would use
        proper Bayesian calibration (Beta distribution conjugate prior).

        Returns updated thresholds dict.
        """
        old_thresholds = dict(self.state["thresholds"])

        # If human accuracy > 95%, we can slightly lower BERT high threshold
        # (BERT is doing well, let it handle more cases)
        if human_accuracy > 0.95 and sample_size >= 100:
            new_high = max(0.85, old_thresholds["bert_high"] - 0.02)
            self.state["thresholds"]["bert_high"] = round(new_high, 2)

        # If human overturn rate > 20%, raise thresholds (AI too aggressive)
        ai_error_rate = 1.0 - human_accuracy
        if ai_error_rate > 0.20 and sample_size >= 100:
            new_high = min(0.98, old_thresholds["bert_high"] + 0.03)
            self.state["thresholds"]["bert_high"] = round(new_high, 2)
            new_grey_high = min(0.80, old_thresholds["grey_high"] + 0.05)
            self.state["thresholds"]["grey_high"] = round(new_grey_high, 2)

        self._save()

        logger.info(
            "Thresholds calibrated: BERT high %.2f→%.2f, grey high %.2f→%.2f",
            old_thresholds["bert_high"], self.state["thresholds"]["bert_high"],
            old_thresholds["grey_high"], self.state["thresholds"]["grey_high"],
        )
        return dict(self.state["thresholds"])

    # ---- Drift Detection ----

    def record_accuracy(self, accuracy: float, sample_size: int):
        """Record a periodic accuracy measurement."""
        today = time.strftime("%Y-%m-%d")
        self.state["accuracy_history"].append({
            "date": today,
            "accuracy": round(accuracy, 4),
            "sample_size": sample_size,
        })
        # Keep last 60 days
        if len(self.state["accuracy_history"]) > 60:
            self.state["accuracy_history"] = self.state["accuracy_history"][-60:]
        self._save()

    def detect_drift(self) -> tuple[bool, str, dict]:
        """Check for model drift.

        Compares 7-day rolling accuracy vs 30-day baseline.
        Returns (drift_detected, message, metrics).
        """
        history = self.state["accuracy_history"]
        if len(history) < 7:
            return False, "Not enough data for drift detection (need ≥7 days)", {}

        # 7-day window
        recent = [h["accuracy"] for h in history[-7:]]
        recent_avg = sum(recent) / len(recent)

        # 30-day baseline (exclude last 7 days)
        baseline = [h["accuracy"] for h in history[:-7]] if len(history) > 7 else recent
        baseline_avg = sum(baseline) / len(baseline)

        drop = baseline_avg - recent_avg
        metrics = {
            "recent_7d_avg": round(recent_avg, 4),
            "baseline_30d_avg": round(baseline_avg, 4),
            "drop": round(drop, 4),
            "alert_threshold": DRIFT_ALERT_DROP,
        }

        if drop > DRIFT_ALERT_DROP:
            alert = {
                "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "recent_accuracy": recent_avg,
                "baseline": baseline_avg,
                "drop": drop,
            }
            self.state["drift_alerts"].append(alert)
            self._save()
            return True, (
                f"DRIFT DETECTED: accuracy dropped {drop:.1%} "
                f"({recent_avg:.1%} vs baseline {baseline_avg:.1%})"
            ), metrics

        return False, (
            f"No drift: recent {recent_avg:.1%} vs baseline {baseline_avg:.1%} "
            f"(drop {drop:.1%} < threshold {DRIFT_ALERT_DROP:.1%})"
        ), metrics

    # ---- Report ----

    def get_status(self) -> dict:
        """Get current fine-tuning system status."""
        should_ft, reason = self.should_finetune()
        has_drift, drift_msg, drift_metrics = self.detect_drift()

        return {
            "labels_since_last_finetune": self.state["labels_since_last_finetune"],
            "total_labels_collected": self.state["total_labels_collected"],
            "finetune_threshold": FINETUNE_LABEL_THRESHOLD,
            "finetune_count": self.state["finetune_count"],
            "should_finetune": should_ft,
            "finetune_reason": reason,
            "drift_detected": has_drift,
            "drift_details": drift_msg,
            "drift_metrics": drift_metrics,
            "thresholds": dict(self.state["thresholds"]),
            "last_finetune_at": self.state.get("last_finetune_at"),
        }


# Singleton
finetune_trigger = FinetuneTrigger()
