#!/bin/bash

# Quick project setup: installs uv (if needed), creates venv, installs all dependencies.
# Usage: bash scripts/setup.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_VERSION="3.11"

# --- 1. Ensure uv is installed ---
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # make uv available in this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "uv is already installed ($(uv --version))."
fi

# --- 2. Create virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python $PYTHON_VERSION..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
    echo "Virtual environment already exists at $VENV_DIR."
fi

# --- 3. Install dependencies ---
echo "Installing dependencies from llm_bench/requirements.txt..."
uv pip install --python "$VENV_DIR/bin/python" -r "$REPO_ROOT/llm_bench/requirements.txt"

echo ""
echo "Setup complete! Activate with:"
echo "  source $VENV_DIR/bin/activate"