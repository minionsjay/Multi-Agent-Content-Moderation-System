#!/usr/bin/env python3
"""
Pre-download all models for offline deployment.

Run this ONCE on a machine with internet access. Models are saved to
the HuggingFace cache (~/.cache/huggingface/hub/) by default, or to
./models/ when --local is used.

Usage:
  python download_models.py              # download text-only models
  python download_models.py --all        # download ALL models (large)
  python download_models.py --local      # save to ./models/ (portable)
  python download_models.py --llm        # also download LLM models

Downloads:
  1. KoalaAI/Text-Moderation          (~400MB)  L2 BERT safety classifier
  2. BAAI/bge-small-zh-v1.5           (~95MB)   text embeddings
  3. Qwen/Qwen3Guard-Gen-0.6B         (~1.2GB) L3 safety classifier (optional)
  4. Qwen/Qwen2.5-1.5B-Instruct       (~3GB)   L3 general LLM (optional)
  5. EasyOCR (ch_sim + en)            (~450MB)  image text extraction (optional)
  6. Falconsai/nsfw_image_detection   (~350MB)  image NSFW classifier (optional)

After download, use HF_LOCAL_FILES_ONLY=true in .env to prevent auto-downloads.
"""

import sys
import os
import time
import argparse


def download_model(model_id: str, label: str, save_dir: str | None = None):
    """Download a HuggingFace model (text classification or embedding)."""
    print(f"\n{'='*50}")
    print(f"[{label}] {model_id}")
    print(f"{'='*50}")

    from transformers import AutoTokenizer, AutoModel

    t0 = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        if save_dir:
            local_path = os.path.join(save_dir, model_id.replace("/", "--"))
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            print(f"  ✓ Saved to {local_path}")
        else:
            print(f"  ✓ Cached in ~/.cache/huggingface/")

        params = sum(p.numel() for p in model.parameters())
        elapsed = time.perf_counter() - t0
        print(f"  ✓ {params/1e6:.0f}M params, {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_causal_lm(model_id: str, label: str, save_dir: str | None = None):
    """Download a CausalLM model (Qwen3Guard, Qwen2.5, etc.)."""
    print(f"\n{'='*50}")
    print(f"[{label}] {model_id}")
    print(f"{'='*50}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    t0 = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        if save_dir:
            local_path = os.path.join(save_dir, model_id.replace("/", "--"))
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            print(f"  ✓ Saved to {local_path}")
        else:
            print(f"  ✓ Cached in ~/.cache/huggingface/")

        params = sum(p.numel() for p in model.parameters())
        elapsed = time.perf_counter() - t0
        print(f"  ✓ {params/1e9:.2f}B params, {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_embedding(model_id: str, label: str, save_dir: str | None = None):
    """Download SentenceTransformer embedding model."""
    print(f"\n{'='*50}")
    print(f"[{label}] {model_id}")
    print(f"{'='*50}")

    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(model_id)
        dim = model.get_sentence_embedding_dimension()

        if save_dir:
            local_path = os.path.join(save_dir, model_id.replace("/", "--"))
            model.save(local_path)
            print(f"  ✓ Saved to {local_path}")
        else:
            print(f"  ✓ Cached in ~/.cache/huggingface/")

        elapsed = time.perf_counter() - t0
        print(f"  ✓ dim={dim}, {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_pipeline(model_id: str, task: str, label: str, save_dir: str | None = None):
    """Download a transformers pipeline model (e.g. NSFW ViT)."""
    print(f"\n{'='*50}")
    print(f"[{label}] {model_id}")
    print(f"{'='*50}")

    from transformers import pipeline

    t0 = time.perf_counter()
    try:
        pipe = pipeline(task, model=model_id)

        if save_dir:
            local_path = os.path.join(save_dir, model_id.replace("/", "--"))
            pipe.save_pretrained(local_path)
            print(f"  ✓ Saved to {local_path}")
        else:
            print(f"  ✓ Cached in ~/.cache/huggingface/")

        elapsed = time.perf_counter() - t0
        print(f"  ✓ {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_easyocr():
    """EasyOCR: Chinese + English text extraction from images."""
    model_id = "EasyOCR (ch_sim + en)"
    print(f"\n{'='*50}")
    print(f"[OCR] {model_id}")
    print(f"{'='*50}")
    try:
        import easyocr
        t0 = time.perf_counter()
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        elapsed = time.perf_counter() - t0
        print(f"  ✓ {elapsed:.0f}s")
        return True
    except ImportError:
        print("  ! easyocr not installed. Install: pip install easyocr")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download all models for offline deployment")
    parser.add_argument("--all", action="store_true", help="Download ALL models including large LLMs")
    parser.add_argument("--llm", action="store_true", help="Also download LLM models (Qwen3Guard + Qwen2.5)")
    parser.add_argument("--text-only", action="store_true", help="Only text models (BERT + Embedding)")
    parser.add_argument("--local", action="store_true", help="Save models to ./models/ instead of HF cache")
    args = parser.parse_args()

    save_dir = os.path.abspath("models") if args.local else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("MODEL DOWNLOADER")
    print("=" * 50)
    print(f"Save mode: {'./models/' if save_dir else 'HF cache (~/.cache/huggingface/)'}")
    print(f"Scope: {'ALL' if args.all else 'LLM included' if args.llm else 'text-only' if args.text_only else 'text essential'}")

    results = {}

    if not args.text_only:
        # Essential: BERT + Embedding
        results["BERT (KoalaAI)"] = download_model(
            "KoalaAI/Text-Moderation", "BERT", save_dir,
        )
        results["Embedding (BGE)"] = download_embedding(
            "BAAI/bge-small-zh-v1.5", "Embedding", save_dir,
        )

    if args.all or args.llm:
        results["Qwen3Guard (0.6B)"] = download_causal_lm(
            "Qwen/Qwen3Guard-Gen-0.6B", "LLM", save_dir,
        )
        results["Qwen2.5 (1.5B)"] = download_causal_lm(
            "Qwen/Qwen2.5-1.5B-Instruct", "LLM", save_dir,
        )

    if args.all:
        results["NSFW ViT"] = download_pipeline(
            "Falconsai/nsfw_image_detection", "image-classification", "NSFW", save_dir,
        )
        results["EasyOCR"] = download_easyocr()

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    if save_dir:
        print(f"\nModels saved to: {save_dir}")
        print(f"Total size: ", end="")
        total = 0
        for root, dirs, files in os.walk(save_dir):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        print(f"{total/1024**3:.1f} GB")
        print(f"\nSet these in .env:")
        print(f"  BERT_MODEL=./models/KoalaAI--Text-Moderation")
        print(f"  EMBED_MODEL=./models/BAAI--bge-small-zh-v1.5")
        if args.all or args.llm:
            print(f"  QWEN_GUARD_MODEL=./models/Qwen--Qwen3Guard-Gen-0.6B")
            print(f"  TRANSFORMERS_LLM_MODEL=./models/Qwen--Qwen2.5-1.5B-Instruct")
        print(f"  HF_LOCAL_FILES_ONLY=true")

    print(f"\nDone. Set HF_LOCAL_FILES_ONLY=true in .env to prevent further downloads.")


if __name__ == "__main__":
    main()
