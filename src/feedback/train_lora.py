#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for BERT Content Moderation Model.

Triggered when enough labeled data accumulates (default: 5000 samples).
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning — only trains
a small adapter layer (~2MB) instead of the full BERT model (~400MB).

This is designed to run OFFLINE, not during request processing.
It reads the labeled datasets from data/assets/datasets/ and produces
a fine-tuned LoRA adapter that can be loaded alongside the base BERT.

Usage:
  python -m src.feedback.train_lora                    # train with latest dataset
  python -m src.feedback.train_lora --dry-run          # check if ready to train
  python -m src.feedback.train_lora --dataset path.jsonl  # train with specific dataset

Requirements:
  pip install peft transformers datasets accelerate

POC note:
  - LoRA fine-tuning on CPU works for small datasets (1000-5000 samples)
  - Expected training time on CPU: ~5 min / 1000 samples
  - GPU: ~30 sec / 1000 samples
  - The LoRA adapter is saved to data/models/lora_adapter/
"""

import sys
import os
import json
import time
import glob
import logging
import argparse
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from src.config import BERT_MODEL
from src.feedback.finetune_trigger import finetune_trigger

LORA_OUTPUT_DIR = os.getenv("LORA_OUTPUT_DIR", "./data/models/lora_adapter")


def find_latest_dataset() -> str | None:
    """Find the most recent labeled dataset."""
    pattern = "./data/assets/datasets/labeled_*.jsonl"
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def load_dataset(path: str) -> tuple[list[str], list[int]]:
    """Load labeled dataset, convert to (texts, labels) for training.

    Returns (texts, labels) where labels are 0=safe, 1=unsafe.
    """
    texts = []
    labels = []
    label_map = {"safe": 0, "pass": 0, "normal": 0, "unsafe": 1, "block": 1}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = item.get("text", "")
            label = item.get("label", "safe")

            if text and text.strip():
                texts.append(text)
                labels.append(label_map.get(label, 0))

    logger.info("Loaded %d samples from %s (safe=%d, unsafe=%d)",
                len(texts), path,
                labels.count(0), labels.count(1))
    return texts, labels


def train_lora(
    dataset_path: str,
    base_model: str | None = None,
    output_dir: str | None = None,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    dry_run: bool = False,
) -> dict:
    """Fine-tune BERT with LoRA on labeled moderation data.

    LoRA config:
      - r=8: rank of the low-rank matrices (smaller = fewer params)
      - alpha=16: scaling factor
      - target_modules: query and value projection layers

    Returns training summary dict.
    """
    base_model = base_model or BERT_MODEL
    output_dir = output_dir or LORA_OUTPUT_DIR

    texts, labels = load_dataset(dataset_path)
    if len(texts) < 100:
        return {"error": f"Need ≥100 samples, got {len(texts)}", "status": "skipped"}

    if dry_run:
        return {
            "status": "dry_run",
            "samples": len(texts),
            "safe_count": labels.count(0),
            "unsafe_count": labels.count(1),
            "base_model": base_model,
            "output_dir": output_dir,
            "epochs": epochs,
            "ready": True,
        }

    # ---- Actual training (requires peft + transformers) ----
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import torch
        import numpy as np
    except ImportError as e:
        return {
            "error": f"Missing dependencies: {e}. Install: pip install peft transformers datasets accelerate",
            "status": "dependency_missing",
        }

    logger.info("Starting LoRA fine-tuning: %d samples, %d epochs", len(texts), epochs)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding=False
        )

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],  # BERT attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    logger.info("Trainable params: %d / %d (%.2f%%)",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad)
                / sum(p.numel() for p in model.parameters()) * 100)

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    train_time = time.perf_counter() - t0

    # Evaluate
    eval_result = trainer.evaluate()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Record fine-tune completion
    finetune_trigger.mark_finetune_started(len(texts))

    summary = {
        "status": "completed",
        "samples": len(texts),
        "train_time_seconds": round(train_time, 1),
        "epochs": epochs,
        "train_loss": round(train_result.training_loss or 0, 4),
        "eval_loss": round(eval_result.get("eval_loss", 0), 4),
        "output_dir": output_dir,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }

    logger.info("Fine-tuning complete: eval_loss=%.4f, time=%.0fs",
                summary["eval_loss"], train_time)
    return summary


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for BERT")
    parser.add_argument("--dataset", type=str, help="Path to labeled dataset JSONL")
    parser.add_argument("--dry-run", action="store_true", help="Check readiness only")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    # Check if we should fine-tune
    should, reason = finetune_trigger.should_finetune()
    print(f"Fine-tune trigger: {reason}")

    if not should and not args.dry_run:
        print("Not enough labels yet. Run with --dry-run to see status.")
        return

    # Find dataset
    dataset_path = args.dataset or find_latest_dataset()
    if not dataset_path:
        print("No dataset found. Run the feedback pipeline first:")
        print("  python -m src.feedback.pipeline")
        return

    print(f"Dataset: {dataset_path}")

    # Train
    result = train_lora(
        dataset_path=dataset_path,
        epochs=args.epochs,
        learning_rate=args.lr,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
