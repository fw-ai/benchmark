#!/usr/bin/env bash
# Run the same load test matrix as the GPT-OSS-120B analysis doc:
# - Default dataset (limericks)
# - 6 input/output configs × 8 concurrency levels (4,8,12,16,24,32,48,64)
# - 120s per run, prompt cache disabled
#
# Usage:
#   export FIREWORKS_API_KEY=...   # or API_KEY (never printed or passed on cmdline)
#   ./run_production_load_tests.sh HOST PROVIDER MODEL TOKENIZER
#   ./run_production_load_tests.sh "https://api.fireworks.ai/inference" fireworks "accounts/pyroworks/deployments/atlassian-gpt-oss-120b-h200-eagle3-fa" openai/gpt-oss-120b
#
# Args: HOST PROVIDER MODEL TOKENIZER (e.g. openai/gpt-oss-120b for GPT-OSS-120B limericks)

set -e
HOST="${1:?Usage: $0 HOST PROVIDER MODEL TOKENIZER}"
PROVIDER="${2:?Usage: $0 HOST PROVIDER MODEL TOKENIZER}"
MODEL="${3:?Usage: $0 HOST PROVIDER MODEL TOKENIZER}"
TOKENIZER="${4:?Usage: $0 HOST PROVIDER MODEL TOKENIZER}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# API key must be set in the environment (never printed or passed on cmdline)
if [ -z "${FIREWORKS_API_KEY:-}" ] && [ -z "${API_KEY:-}" ]; then
  echo "Set FIREWORKS_API_KEY or API_KEY in the environment before running." >&2
  exit 1
fi
[ -n "${FIREWORKS_API_KEY:-}" ] && export API_KEY="$FIREWORKS_API_KEY"

TIMESTAMP="$(date +%Y%m%d_%H%M)"
RESULTS_ROOT="results/production_load_test_${PROVIDER}_${TIMESTAMP}"
mkdir -p "$RESULTS_ROOT"

# 6 configs: "prompt_tokens max_tokens" (from the doc)
CONFIGS=(
  "256 128"
  "256 512"
  "256 1024"
  "1024 256"
  "1024 1024"
  "2048 256"
)
CONCURRENCIES=(4 8 12 16 24 32 48 64)

for config in "${CONFIGS[@]}"; do
  read -r P O <<< "$config"
  CONFIG_DIR="$RESULTS_ROOT/input${P}_output${O}"
  mkdir -p "$CONFIG_DIR"
  for C in "${CONCURRENCIES[@]}"; do
    OUT="$CONFIG_DIR/c${C}"
    mkdir -p "$OUT"
    echo "=== Config input=${P} output=${O} concurrency=${C} ==="
    locust -f load_test.py --headless --only-summary \
      -H "$HOST" \
      --provider "$PROVIDER" \
      --model "$MODEL" \
      --api-key "$API_KEY" \
      --tokenizer "$TOKENIZER" \
      -u "$C" -r "$C" \
      -t 120s \
      -p "$P" -o "$O" \
      --prompt-cache-max-len 0 \
      --stream \
      --csv "$OUT/stats" \
      --html "$OUT/report.html" \
      || true
  done
done

echo "Results under: $RESULTS_ROOT"
