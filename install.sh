#!/bin/bash
# ============================================================
# Content Moderation POC v0.5.0 — One-Click Install Script
# ============================================================
# Run this on the OFFLINE target machine after copying:
#   - hf_cache.tar.gz        (HuggingFace models)
#   - easyocr.tar.gz          (optional, for image OCR)
#   - qwen2.5-1.5b.gguf      (optional, for local LLM)
#   - offline_packages/       (pip packages)
#   - The poc/ project directory
#
# Usage:
#   chmod +x install.sh
#   ./install.sh              # full install (text + image + local LLM)
#   ./install.sh --text-only  # text moderation only, no image models
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-full}"  # full | text-only

echo "============================================"
echo " Content Moderation POC v0.5.0 — Install"
echo " Mode: $MODE"
echo "============================================"
echo ""

# ---- 1. Python environment ----
echo "[1/6] Python version check..."
python3 --version || { echo "ERROR: Python 3.11+ required"; exit 1; }

# ---- 2. Install pip packages ----
echo "[2/6] Installing Python packages..."
if [ -d "offline_packages" ] && [ "$(ls -A offline_packages 2>/dev/null)" ]; then
    pip install --no-index --find-links offline_packages/ -r requirements.txt
    echo "  ✓ Installed from offline packages"
else
    echo "  offline_packages/ not found — installing from PyPI (needs internet)"
    pip install -r requirements.txt
fi

# Local LLM (optional)
if [ "$MODE" != "text-only" ]; then
    if [ -f "offline_packages/llama_cpp_python"* ] 2>/dev/null || \
       [ -d "offline_packages" ] && ls offline_packages/llama* 2>/dev/null; then
        pip install --no-index --find-links offline_packages/ llama-cpp-python 2>/dev/null || true
    fi
fi

# ---- 3. Extract HuggingFace models ----
echo "[3/6] Extracting HuggingFace models..."
if [ -f "hf_cache.tar.gz" ]; then
    mkdir -p ~/.cache/huggingface
    tar -xzf hf_cache.tar.gz -C ~/.cache/huggingface/
    echo "  ✓ HF models extracted"
else
    echo "  hf_cache.tar.gz not found — downloading models (needs internet)"
    python download_models.py --text-only
fi

# ---- 4. Extract EasyOCR models (optional) ----
if [ "$MODE" != "text-only" ]; then
    echo "[4/6] Extracting EasyOCR models..."
    if [ -f "easyocr.tar.gz" ]; then
        mkdir -p ~/.EasyOCR
        tar -xzf easyocr.tar.gz -C ~/.EasyOCR/
        echo "  ✓ EasyOCR models extracted"
    else
        echo "  easyocr.tar.gz not found — skip (image OCR won't be available)"
    fi
else
    echo "[4/6] Text-only mode — skipping EasyOCR"
fi

# ---- 5. Setup local LLM (optional) ----
echo "[5/6] Setting up local LLM..."
if [ -f "qwen2.5-1.5b.gguf" ]; then
    mkdir -p models
    cp qwen2.5-1.5b.gguf models/
    echo "  ✓ LLM model placed: models/qwen2.5-1.5b.gguf"
elif [ -f "models/qwen2.5-1.5b.gguf" ]; then
    echo "  ✓ LLM model already in place"
else
    echo "  No local LLM model found — will use API (needs DEEPSEEK_API_KEY)"
fi

# ---- 6. Configure .env ----
echo "[6/6] Configuring environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✓ Created .env from .env.example"
    echo "  ⚠ Edit .env and set DEEPSEEK_API_KEY (or LLM_PROVIDER=local)"
else
    echo "  .env already exists — skipping"
fi

# ---- Done ----
echo ""
echo "============================================"
echo " INSTALL COMPLETE"
echo "============================================"
echo ""
echo " Next steps:"
echo "  1. Edit .env and set your API key or LLM_PROVIDER=local"
echo "  2. Run: python check_env.py"
echo "  3. Start: python -m src.api"
echo "  4. Open: http://localhost:8000"
echo ""
