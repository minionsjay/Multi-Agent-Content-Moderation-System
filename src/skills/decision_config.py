"""
Runtime-adjustable decision parameters for the Decision Agent.

Exposed via API endpoints so the frontend can display and adjust them.
Default values match the CLAUDE.md architecture design.
"""

import copy
import logging

logger = logging.getLogger(__name__)

DEFAULTS = {
    # ---- Agent weights (must sum to 1.0) ----
    "agent_weights": {
        "text": 0.40,
        "image": 0.35,
        "multimodal": 0.25,
    },

    # ---- Label severity scores (0.0 = safe, 1.0 = most severe) ----
    "label_severity": {
        "safe": 0.0,
        "normal": 0.0,
        "spam": 0.3,
        "toxic": 0.5,
        "gambling": 0.6,
        "porn": 0.8,
        "nsfw": 0.85,
        "violence": 0.9,
        "politics": 1.0,
        "unsafe": 0.7,
    },

    # ---- Score-to-label mapping thresholds ----
    "score_thresholds": {
        "politics": 0.85,
        "violence": 0.85,
        "porn": 0.65,
        "gambling": 0.45,
        "toxic": 0.25,
        "spam": 0.10,
    },

    # ---- Grey zone: confidence between low and high → human review ----
    "grey_zone": {
        "low": 0.30,
        "high": 0.70,
    },

    # ---- Zero-tolerance categories (bypass model scoring, always block) ----
    "zero_tolerance": ["politics", "violence"],

    # ---- BERT: confidence >= this skips LLM ----
    "bert_high_confidence": 0.95,

    # ---- BERT: confidence < this means model is unsure, escalate to LLM even if safe ----
    "bert_low_confidence": 0.40,
}

# Runtime state — starts as a copy of defaults
_config = copy.deepcopy(DEFAULTS)


def get_config() -> dict:
    """Return current decision configuration."""
    return copy.deepcopy(_config)


def update_config(updates: dict) -> dict:
    """Deep-merge updates into current config. Returns new config."""
    _deep_merge(_config, updates)
    logger.info("Decision config updated: %s", list(updates.keys()))
    return get_config()


def reset_config() -> dict:
    """Reset to defaults. Returns default config."""
    global _config
    _config = copy.deepcopy(DEFAULTS)
    logger.info("Decision config reset to defaults")
    return get_config()


def _deep_merge(base: dict, updates: dict):
    """Recursively merge updates into base dict."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
