#!/usr/bin/env bash
# Rerank load test: sweep num_documents 100,200,300,400,500 with 128 tokens per doc.
# Set FIREWORKS_API_KEY (or API_KEY) before running.
#
# Required: use a local tokenizer (Locust+gevent conflicts with transformers/hub/trio).
# Run once: python download_tokenizer.py cross-encoder/ms-marco-MiniLM-L6-v2
# The script will then use tokenizers/cross-encoder_ms-marco-MiniLM-L6-v2 automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

API_KEY="${FIREWORKS_API_KEY:-${API_KEY:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "Error: Set FIREWORKS_API_KEY or API_KEY before running." >&2
  exit 1
fi

HOST="https://api.fireworks.ai/inference"
MODEL="accounts/thumbtack/models/mismatch-model-v2#accounts/thumbtack/deployments/mismatch-model-v2-auto"
# Must use local tokenizer (run download_tokenizer.py first)
LOCAL_TOKENIZER="${SCRIPT_DIR}/tokenizers/cross-encoder_ms-marco-MiniLM-L6-v2"
if [[ -d "$LOCAL_TOKENIZER" ]]; then
  TOKENIZER="${TOKENIZER:-$LOCAL_TOKENIZER}"
else
  echo "Error: Local tokenizer not found at $LOCAL_TOKENIZER" >&2
  echo "Run once: python download_tokenizer.py cross-encoder/ms-marco-MiniLM-L6-v2" >&2
  exit 1
fi
RESULTS_DIR="${RESULTS_DIR:-Results}"
mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/mismatch-v2-auto-test.csv"

for num_docs in 20 40 60 80 100 200 300 400 500; do
  echo "=== num_documents=$num_docs ==="
  locust -f load_test.py --headless \
    -u 1 -r 1 -t 150s \
    -H "$HOST" \
    --api-key "$API_KEY" \
    -m "$MODEL" \
    --rerank \
    --num-documents "$num_docs" \
    --tokens-per-document 256 \
    --dataset limericks \
    --tokenizer "$TOKENIZER" \
    --summary-file "$SUMMARY_CSV" \
    --csv "$RESULTS_DIR/batch_${num_docs}" \
    --html "$RESULTS_DIR/batch_${num_docs}.html" \
    2>&1 | tee "$RESULTS_DIR/batch_${num_docs}.log"
  echo ""
done

echo "Done. Summary: $SUMMARY_CSV"
