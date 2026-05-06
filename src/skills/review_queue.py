"""
Human Review Queue — persistence for grey-zone content.

When Decision Agent returns "review", the content is too ambiguous for
automatic decision. It goes into a review queue for human moderators.

POC implementation: JSON file (simple, zero-dependency).
Production: PostgreSQL / Kafka / dedicated review platform.

Flow:
  1. Decision → "review" → Action calls review_queue.enqueue()
  2. Human moderator reviews via review_ui or API
  3. Human decision → review_queue.resolve() → Feedback Agent

The queue is a JSON Lines file (append-only, atomic writes per line).
Each entry contains the full moderation context so the reviewer sees
everything the AI saw.
"""

import json
import os
import time
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

REVIEW_QUEUE_PATH = os.getenv("REVIEW_QUEUE_PATH", "./data/review_queue.jsonl")


class ReviewQueue:
    """Persistent human review queue backed by a JSONL file.

    Production upgrade path:
      - POC: JSONL file (this implementation)
      - Phase 2: PostgreSQL table with priority queue
      - Phase 3: Dedicated review platform API (e.g., Checkstep, Hive)
    """

    def __init__(self, queue_path: str | None = None):
        self.path = queue_path or REVIEW_QUEUE_PATH
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write("")

    def enqueue(self, entry: dict) -> str:
        """Add a content item to the human review queue.

        Args:
            entry: {
                "content_id": str,
                "text": str,
                "image_url": str | None,
                "ai_decision": str,       # original AI decision before review
                "ai_confidence": float,
                "ai_reason": str,
                "signals": {              # all agent outputs for context
                    "text_result": dict | None,
                    "image_result": dict | None,
                },
                "priority": float,        # 0.0 (low) to 1.0 (urgent)
                "traces": list[dict],
            }

        Returns:
            review_id: unique ID for this review item
        """
        review_id = hashlib.sha256(
            f"{entry.get('content_id')}:{time.time()}".encode()
        ).hexdigest()[:16]

        record = {
            "review_id": review_id,
            "status": "pending",          # pending | reviewed | dismissed
            "enqueued_at": time.time(),
            "enqueued_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reviewed_by": None,
            "reviewed_at": None,
            "human_decision": None,       # pass | block (set by reviewer)
            "human_reason": None,
            **entry,
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("REVIEW | id=%s | priority=%.2f | queued for human review",
                    review_id, entry.get("priority", 0.5))
        return review_id

    def resolve(self, review_id: str, human_decision: str,
                reviewer: str, reason: str = "") -> dict | None:
        """Record a human moderator's decision for a queued item.

        Returns the updated record, or None if review_id not found.
        In POC (JSONL file), this rewrites the entire file. For production
        with a database, this would be an UPDATE.
        """
        lines = []
        resolved = None
        found = False

        # Read all lines
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue

                if record.get("review_id") == review_id:
                    record["status"] = "reviewed"
                    record["human_decision"] = human_decision
                    record["human_reason"] = reason
                    record["reviewed_by"] = reviewer
                    record["reviewed_at"] = time.time()
                    record["reviewed_at_iso"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    resolved = record
                    found = True

                lines.append(json.dumps(record, ensure_ascii=False))

        if not found:
            logger.warning("REVIEW | id=%s not found in queue", review_id)
            return None

        # Rewrite file
        with open(self.path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("REVIEW | id=%s | decision=%s | reviewer=%s | resolved",
                    review_id, human_decision, reviewer)
        return resolved

    def get_pending(self, limit: int = 50) -> list[dict]:
        """Get pending review items, ordered by priority (highest first)."""
        items = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "pending":
                    items.append(record)

        items.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
        return items[:limit]

    def get_stats(self) -> dict:
        """Queue statistics."""
        total = 0
        pending = 0
        reviewed = 0
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if record.get("status") == "pending":
                    pending += 1
                elif record.get("status") == "reviewed":
                    reviewed += 1

        return {
            "total": total,
            "pending": pending,
            "reviewed": reviewed,
            "pending_rate": round(pending / max(total, 1), 4),
        }


# Singleton
review_queue = ReviewQueue()
