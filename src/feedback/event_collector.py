"""
Event Collector — captures every moderation decision into a persistent store.

This is the foundation of the feedback loop. Every request that passes through
the system (both hot path and cold path) generates an event record containing:
  - The original content
  - The AI's decision + confidence + reasoning
  - All agent traces
  - The final action taken

These events are the raw material for:
  - Building labeled datasets
  - Analyzing model accuracy
  - Detecting new evasion patterns
  - Auditing decisions

POC storage: JSONL file (append-only, one event per line).
Production: Kafka topic + S3/Data Lake.
"""

import json
import os
import time
import logging

logger = logging.getLogger(__name__)

EVENT_STORE_PATH = os.getenv("EVENT_STORE_PATH", "./data/events.jsonl")


class EventCollector:
    """Append-only event log for all moderation decisions."""

    def __init__(self, store_path: str | None = None):
        self.path = store_path or EVENT_STORE_PATH
        self._ensure()

    def _ensure(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write("")

    def record(self, event: dict) -> str:
        """Record a moderation event.

        Args:
            event: Full moderation result including state, traces, and final decision.

        Returns the event_id.
        """
        event_id = f"evt_{int(time.time() * 1000)}_{hash(event.get('content_id', '')) & 0xFFFF:04x}"

        record = {
            "event_id": event_id,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **event,
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return event_id

    def read_all(self, limit: int = 10000) -> list[dict]:
        """Read recent events for offline analysis."""
        events = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events[-limit:]

    def count(self) -> int:
        """Total events stored."""
        if not os.path.exists(self.path):
            return 0
        count = 0
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


# Singleton
event_collector = EventCollector()
