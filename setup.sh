#!/bin/bash
# ============================================================
# Content Moderation POC — Quick Setup Script
# ============================================================
# Usage:
#   chmod +x setup.sh
#   ./setup.sh                    # Create venv + install deps
#   ./setup.sh --with-local-llm   # Also install llama.cpp
#   ./setup.sh --with-gpu         # GPU-accelerated llama.cpp
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

MODE="${1:-base}"

echo "============================================"
echo " Content Moderation POC — Setup"
echo " Mode: $MODE"
echo "============================================"
echo ""

# ---- Step 1: Check Python ----
log "Checking Python version..."
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
if [ "$(echo "$PYVER >= 3.11" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    log "Python $PYVER OK"
else
    err "Python 3.11+ required, found $PYVER"
fi

# ---- Step 2: Create venv ----
if [ ! -d "venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv venv
else
    log "venv already exists"
fi

source venv/bin/activate
log "Activated venv ($(which python3))"

# ---- Step 3: Upgrade pip ----
log "Upgrading pip..."
pip install --upgrade pip -q

# ---- Step 4: Install dependencies ----
log "Installing dependencies..."
pip install -r requirements.txt -q
log "Base dependencies installed"

# ---- Step 5: Optional: local LLM ----
if [ "$MODE" = "--with-local-llm" ]; then
    log "Installing llama-cpp-python..."
    pip install llama-cpp-python -q
    log "llama-cpp-python installed"

    if [ ! -f "models/qwen2.5-1.5b.gguf" ]; then
        warn "LLM model not found. Download with:"
        echo "  mkdir -p models"
        echo "  wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf -O models/qwen2.5-1.5b.gguf"
    fi
elif [ "$MODE" = "--with-gpu" ]; then
    log "Installing llama-cpp-python with CUDA..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python -q
    log "llama-cpp-python (CUDA) installed"
fi

# ---- Step 6: Download models (if internet available) ----
log "Checking models..."
python3 -c "
import os
cache = os.path.expanduser('~/.cache/huggingface/hub')
bert = any('KoalaAI' in f or 'toxic-bert' in f for f in os.listdir(cache) if os.path.isdir(os.path.join(cache, f))) if os.path.exists(cache) else False
print(f'HuggingFace cache: {\"found\" if bert else \"empty/missing\"}')" 2>/dev/null

if [ -f "download_models.py" ]; then
    warn "Downloading models (may take a while on first run)..."
    python3 download_models.py --text-only 2>&1 | tail -3
fi

# ---- Step 7: Create .env if missing ----
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn "Created .env from .env.example"
    echo "  Edit .env and set your DEEPSEEK_API_KEY"
else
    log ".env already exists"
fi

# ---- Done ----
echo ""
echo "============================================"
echo " SETUP COMPLETE"
echo "============================================"
echo ""
echo " Quick start:"
echo "   source venv/bin/activate"
echo "   python check_env.py       # verify everything works"
echo "   python -m src.api          # start server on :8000"
echo ""
echo " With workers:"
echo "   uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4"
echo ""
