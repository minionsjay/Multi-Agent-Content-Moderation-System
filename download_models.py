#!/usr/bin/env python3
"""
Pre-download all models for offline deployment.

Run this ONCE on a machine with internet access. It downloads all
HuggingFace models to the local cache (~/.cache/huggingface/hub/).
Then copy the entire cache to the offline machine.

Usage:
  python download_models.py              # download all models
  python download_models.py --text-only  # skip image/OptionalCR models
  python download_models.py --llm-only   # only download LLM (GGUF)

Downloads:
  1. KoalaAI/Text-Moderation          (~400MB)  L2 BERT safety classifier
  2. BAAI/bge-small-zh-v1.5           (~95MB)   text embeddings
  3. EasyOCR (ch_sim + en)            (~450MB)  image text extraction (optional)
  4. Falconsai/nsfw_image_detection   (~350MB)  image NSFW classifier (optional)

After download, pack the cache:
  tar -czf hf_cache.tar.gz -C ~/.cache/huggingface hub/
  tar -czf easyocr.tar.gz -C ~/.EasyOCR model/
"""

import sys
import os
import time
import argparse
import subprocess


def download_bert():
    """L2 BERT: KoalaAI/Text-Moderation (9-label safety classifier)."""
    print("=" * 50)
    print("[1/4] KoalaAI/Text-Moderation (L2 BERT)")
    print("=" * 50)
    from transformers import pipeline
    t0 = time.perf_counter()
    pipe = pipeline("text-classification", model="KoalaAI/Text-Moderation")
    print(f"  ✓ Downloaded in {time.perf_counter() - t0:.0f}s")
    return True


def download_embedding():
    """BGE-small-zh-v1.5: text embeddings, 512-dim."""
    print("\n" + "=" * 50)
    print("[2/4] BAAI/bge-small-zh-v1.5 (Embedding)")
    print("=" * 50)
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    dim = model.get_sentence_embedding_dimension()
    print(f"  ✓ Downloaded in {time.perf_counter() - t0:.0f}s (dim={dim})")
    return True


def download_easyocr():
    """EasyOCR: Chinese + English text extraction from images."""
    print("\n" + "=" * 50)
    print("[3/4] EasyOCR (ch_sim + en)")
    print("=" * 50)
    try:
        import easyocr
        t0 = time.perf_counter()
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        print(f"  ✓ Downloaded in {time.perf_counter() - t0:.0f}s")
        return True
    except ImportError:
        print("  ⚠ easyocr not installed — skipping. Install: pip install easyocr")
        return False


def download_nsfw():
    """NSFW ViT: Falconsai/nsfw_image_detection."""
    print("\n" + "=" * 50)
    print("[4/4] Falconsai/nsfw_image_detection (NSFW ViT)")
    print("=" * 50)
    from transformers import pipeline
    t0 = time.perf_counter()
    try:
        pipe = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
        print(f"  ✓ Downloaded in {time.perf_counter() - t0:.0f}s")
        return True
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        return False


def download_llm():
    """Download Qwen2.5-1.5B GGUF for local LLM (optional)."""
    print("\n" + "=" * 50)
    print("[LLM] Qwen2.5-1.5B-Instruct GGUF Q4_K_M")
    print("=" * 50)

    url = ("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/"
           "resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    dest = "models/qwen2.5-1.5b.gguf"

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✓ Already exists: {size_mb:.0f}MB")
        return True

    os.makedirs("models", exist_ok=True)

    try:
        import urllib.request
        print(f"  Downloading ~1GB...")
        t0 = time.perf_counter()
        urllib.request.urlretrieve(url, dest)
        elapsed = time.perf_counter() - t0
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✓ Downloaded {size_mb:.0f}MB in {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  ⚠ Download failed: {e}")
        print(f"  Manual download: wget {url} -O {dest}")
        return False


def print_summary(results):
    """Print download summary and pack instructions."""
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    print(f"""
=============================================
PACK FOR OFFLINE DEPLOYMENT
=============================================

# 1. Pack HuggingFace models
tar -czf hf_cache.tar.gz -C ~/.cache/huggingface hub/

# 2. Pack EasyOCR (if downloaded)
tar -czf easyocr.tar.gz -C ~/.EasyOCR model/

# 3. Pack LLM (if downloaded)
cp models/qwen2.5-1.5b.gguf . 2>/dev/null

# 4. Pack pip packages
mkdir -p offline_packages
pip download -r requirements.txt -d offline_packages/
# (uncomment next line if using local LLM)
# pip download llama-cpp-python -d offline_packages/

=============================================
ON OFFLINE TARGET MACHINE
=============================================

# 1. Extract models
tar -xzf hf_cache.tar.gz -C ~/.cache/huggingface/
tar -xzf easyocr.tar.gz -C ~/.EasyOCR/
mkdir -p models && mv qwen2.5-1.5b.gguf models/

# 2. Install packages
pip install --no-index --find-links offline_packages/ -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: set LLM_PROVIDER=local (or keep deepseek for API)

# 4. Verify
python check_env.py

# 5. Start
python -m src.api
""")


def main():
    parser = argparse.ArgumentParser(description="Pre-download all models for offline deployment")
    parser.add_argument("--text-only", action="store_true", help="Skip image models (OCR, NSFW)")
    parser.add_argument("--llm-only", action="store_true", help="Only download LLM GGUF")
    parser.add_argument("--llm", action="store_true", help="Also download LLM GGUF")
    args = parser.parse_args()

    results = {}

    if args.llm_only:
        results["LLM GGUF"] = download_llm()
        print_summary(results)
        return

    results["BERT (KoalaAI)"] = download_bert()
    results["Embedding (BGE)"] = download_embedding()

    if not args.text_only:
        results["EasyOCR"] = download_easyocr()
        results["NSFW ViT"] = download_nsfw()

    if args.llm:
        results["LLM GGUF"] = download_llm()

    print_summary(results)


if __name__ == "__main__":
    main()
