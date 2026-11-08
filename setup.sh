#!/usr/bin/env bash
# Reproducible environment setup for the Support Ticket Router project.
# Tested on macOS (Apple M2, 16 GB) with Apple-GPU (MPS) acceleration.
set -euo pipefail

LLAMA_FACTORY_COMMIT="f6355ab6a426eedb800742ecf61896f7bff95ebb"   # 2026-11-06, v0.9.6.dev0
BASE_MODEL="Qwen/Qwen3-1.7B-Base"

cd "$(dirname "$0")"

# 1. LLaMA-Factory at a pinned commit
if [ ! -d LLaMA-Factory ]; then
  git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
git -C LLaMA-Factory fetch --depth 1 origin "$LLAMA_FACTORY_COMMIT"
git -C LLaMA-Factory checkout -q "$LLAMA_FACTORY_COMMIT"

# 2. Python 3.11 virtual environment (LLaMA-Factory requires >= 3.11)
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
uv venv .venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ./LLaMA-Factory -r requirements.txt
# macOS quirk: on this machine files inside the dot-folder .venv get the UF_HIDDEN flag, and
# Python >= 3.11.13 silently skips *hidden* .pth files, which breaks the editable install
# ("No module named 'llamafactory'"). Clearing the flag fixes it.
if [ "$(uname)" = "Darwin" ]; then chflags -R nohidden .venv; fi

# 3. Base model weights (~3.4 GB into the Hugging Face cache)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$BASE_MODEL"

# 4. Sanity check
python scripts/check_env.py
