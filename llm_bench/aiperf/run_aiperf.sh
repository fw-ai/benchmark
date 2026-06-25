#!/usr/bin/env bash
#
# Simplified aiperf benchmark runner.
# Sweeps across concurrency levels and collects results into an artifacts directory.
#
# Usage:
#   ./run_aiperf.sh                          # uses defaults
#   ./run_aiperf.sh --model accounts/foo/deployments/bar --concurrencies "1 4 16 64"
#
# Environment:
#   FIREWORKS_API_KEY  - required API key (or pass --api-key)
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL="${MODEL:-}"
URL="${URL:-https://api.fireworks.ai/inference}"
ENDPOINT_TYPE="${ENDPOINT_TYPE:-completions}"
TOKENIZER="${TOKENIZER:-}"
SEQ_DIST="${SEQ_DIST:-3000|99,100|3:100}"
REQUEST_COUNT="${REQUEST_COUNT:-1000}"
CONCURRENCIES="${CONCURRENCIES:-2 4 8 16 32 64 128}"
API_KEY="${FIREWORKS_API_KEY:-}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
EXTRA_INPUTS="${EXTRA_INPUTS:-prompt_cache_max_len:0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2";          shift 2 ;;
    --url)            URL="$2";            shift 2 ;;
    --endpoint-type)  ENDPOINT_TYPE="$2";  shift 2 ;;
    --tokenizer)      TOKENIZER="$2";      shift 2 ;;
    --seq-dist)       SEQ_DIST="$2";       shift 2 ;;
    --request-count)  REQUEST_COUNT="$2";  shift 2 ;;
    --concurrencies)  CONCURRENCIES="$2";  shift 2 ;;
    --api-key)        API_KEY="$2";        shift 2 ;;
    --artifacts-dir)  ARTIFACTS_DIR="$2";  shift 2 ;;
    --extra-inputs)   EXTRA_INPUTS="$2";   shift 2 ;;
    --extra-args)     EXTRA_ARGS="$2";     shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --model          MODEL            Model name (default: $MODEL)"
      echo "  --url            URL              API endpoint URL (default: $URL)"
      echo "  --endpoint-type  TYPE             Endpoint type: completions|chat (default: $ENDPOINT_TYPE)"
      echo "  --tokenizer      TOKENIZER        HF tokenizer name (default: $TOKENIZER)"
      echo "  --seq-dist       DIST             Sequence distribution (default: $SEQ_DIST)"
      echo "  --request-count  N                Requests per concurrency level (default: $REQUEST_COUNT)"
      echo "  --concurrencies  'C1 C2 ...'      Space-separated concurrency levels (default: $CONCURRENCIES)"
      echo "  --api-key        KEY              API key (default: \$FIREWORKS_API_KEY)"
      echo "  --artifacts-dir  DIR              Output directory (default: $ARTIFACTS_DIR)"
      echo "  --extra-inputs   K:V              Extra inputs for aiperf (default: $EXTRA_INPUTS)"
      echo "  --extra-args     ARGS             Additional aiperf arguments"
      echo "  -h, --help                        Show this help"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required (e.g. --model accounts/foo/deployments/bar)"
  exit 1
fi

if [[ -z "$TOKENIZER" ]]; then
  echo "Error: --tokenizer is required (e.g. --tokenizer Qwen/Qwen3-235B-A22B-Thinking-2507-FP8)"
  exit 1
fi

if [[ -z "$API_KEY" ]]; then
  echo "Error: FIREWORKS_API_KEY env var or --api-key is required"
  exit 1
fi

# ── Derived names ────────────────────────────────────────────────────────────
# Sanitize model name for directory: accounts/foo/deployments/bar -> accounts_foo_deployments_bar
MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"

echo "=========================================="
echo " aiperf benchmark sweep"
echo "=========================================="
echo " Model:         $MODEL"
echo " URL:           $URL"
echo " Endpoint:      $ENDPOINT_TYPE"
echo " Tokenizer:     $TOKENIZER"
echo " Seq dist:      $SEQ_DIST"
echo " Requests:      $REQUEST_COUNT per level"
echo " Concurrencies: $CONCURRENCIES"
echo " Artifacts:     $ARTIFACTS_DIR"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/$ARTIFACTS_DIR"

PASS=0
FAIL=0
TOTAL=0

for C in $CONCURRENCIES; do
  TOTAL=$((TOTAL + 1))
  RUN_DIR="$SCRIPT_DIR/$ARTIFACTS_DIR/${MODEL_SLUG}-openai-${ENDPOINT_TYPE}-concurrency${C}"
  mkdir -p "$RUN_DIR/logs"

  echo ""
  echo "──────────────────────────────────────────"
  echo " Running concurrency=$C  →  $RUN_DIR"
  echo "──────────────────────────────────────────"

  CMD=(
    aiperf profile
    --model "$MODEL"
    --url "$URL"
    --endpoint-type "$ENDPOINT_TYPE"
    --concurrency "$C"
    --request-count "$REQUEST_COUNT"
    --tokenizer "$TOKENIZER"
    --sequence-distribution "$SEQ_DIST"
    --api-key "$API_KEY"
    --no-gpu-telemetry
    --no-server-metrics
  )

  # Add extra inputs if specified
  if [[ -n "$EXTRA_INPUTS" ]]; then
    CMD+=(--extra-inputs "$EXTRA_INPUTS")
  fi

  # Add any extra arguments
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    CMD+=($EXTRA_ARGS)
  fi

  LOGFILE="$RUN_DIR/logs/aiperf_$(date +%s).log"

  echo "Command: ${CMD[*]}"
  echo "Log:     $LOGFILE"
  echo ""

  if "${CMD[@]}" 2>&1 | tee "$LOGFILE"; then
    echo "✓ concurrency=$C completed successfully"
    PASS=$((PASS + 1))

    # Move aiperf outputs into the run directory if they exist in cwd
    for f in profile_export_aiperf.json profile_export_aiperf.csv profile_export.jsonl inputs.json; do
      if [[ -f "$SCRIPT_DIR/$f" ]]; then
        mv "$SCRIPT_DIR/$f" "$RUN_DIR/$f"
      fi
    done
  else
    echo "✗ concurrency=$C FAILED (see $LOGFILE)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=========================================="
echo " Done: $PASS/$TOTAL passed, $FAIL failed"
echo " Results in: $ARTIFACTS_DIR/"
echo "=========================================="#!/usr/bin/env bash
#
# Simplified aiperf benchmark runner.
# Sweeps across concurrency levels and collects results into an artifacts directory.
#
# Usage:
#   ./run_aiperf.sh                          # uses defaults
#   ./run_aiperf.sh --model accounts/foo/deployments/bar --concurrencies "1 4 16 64"
#
# Environment:
#   FIREWORKS_API_KEY  - required API key (or pass --api-key)
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL="${MODEL:-}"
URL="${URL:-https://api.fireworks.ai/inference}"
ENDPOINT_TYPE="${ENDPOINT_TYPE:-completions}"
TOKENIZER="${TOKENIZER:-}"
SEQ_DIST="${SEQ_DIST:-3000|99,100|3:100}"
REQUEST_COUNT="${REQUEST_COUNT:-1000}"
CONCURRENCIES="${CONCURRENCIES:-2 4 8 16 32 64 128}"
API_KEY="${FIREWORKS_API_KEY:-}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
EXTRA_INPUTS="${EXTRA_INPUTS:-prompt_cache_max_len:0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2";          shift 2 ;;
    --url)            URL="$2";            shift 2 ;;
    --endpoint-type)  ENDPOINT_TYPE="$2";  shift 2 ;;
    --tokenizer)      TOKENIZER="$2";      shift 2 ;;
    --seq-dist)       SEQ_DIST="$2";       shift 2 ;;
    --request-count)  REQUEST_COUNT="$2";  shift 2 ;;
    --concurrencies)  CONCURRENCIES="$2";  shift 2 ;;
    --api-key)        API_KEY="$2";        shift 2 ;;
    --artifacts-dir)  ARTIFACTS_DIR="$2";  shift 2 ;;
    --extra-inputs)   EXTRA_INPUTS="$2";   shift 2 ;;
    --extra-args)     EXTRA_ARGS="$2";     shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --model          MODEL            Model name (default: $MODEL)"
      echo "  --url            URL              API endpoint URL (default: $URL)"
      echo "  --endpoint-type  TYPE             Endpoint type: completions|chat (default: $ENDPOINT_TYPE)"
      echo "  --tokenizer      TOKENIZER        HF tokenizer name (default: $TOKENIZER)"
      echo "  --seq-dist       DIST             Sequence distribution (default: $SEQ_DIST)"
      echo "  --request-count  N                Requests per concurrency level (default: $REQUEST_COUNT)"
      echo "  --concurrencies  'C1 C2 ...'      Space-separated concurrency levels (default: $CONCURRENCIES)"
      echo "  --api-key        KEY              API key (default: \$FIREWORKS_API_KEY)"
      echo "  --artifacts-dir  DIR              Output directory (default: $ARTIFACTS_DIR)"
      echo "  --extra-inputs   K:V              Extra inputs for aiperf (default: $EXTRA_INPUTS)"
      echo "  --extra-args     ARGS             Additional aiperf arguments"
      echo "  -h, --help                        Show this help"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required (e.g. --model accounts/foo/deployments/bar)"
  exit 1
fi

if [[ -z "$TOKENIZER" ]]; then
  echo "Error: --tokenizer is required (e.g. --tokenizer Qwen/Qwen3-235B-A22B-Thinking-2507-FP8)"
  exit 1
fi

if [[ -z "$API_KEY" ]]; then
  echo "Error: FIREWORKS_API_KEY env var or --api-key is required"
  exit 1
fi

# ── Derived names ────────────────────────────────────────────────────────────
# Sanitize model name for directory: accounts/foo/deployments/bar -> accounts_foo_deployments_bar
MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"

echo "=========================================="
echo " aiperf benchmark sweep"
echo "=========================================="
echo " Model:         $MODEL"
echo " URL:           $URL"
echo " Endpoint:      $ENDPOINT_TYPE"
echo " Tokenizer:     $TOKENIZER"
echo " Seq dist:      $SEQ_DIST"
echo " Requests:      $REQUEST_COUNT per level"
echo " Concurrencies: $CONCURRENCIES"
echo " Artifacts:     $ARTIFACTS_DIR"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/$ARTIFACTS_DIR"

PASS=0
FAIL=0
TOTAL=0

for C in $CONCURRENCIES; do
  TOTAL=$((TOTAL + 1))
  RUN_DIR="$SCRIPT_DIR/$ARTIFACTS_DIR/${MODEL_SLUG}-openai-${ENDPOINT_TYPE}-concurrency${C}"
  mkdir -p "$RUN_DIR/logs"

  echo ""
  echo "──────────────────────────────────────────"
  echo " Running concurrency=$C  →  $RUN_DIR"
  echo "──────────────────────────────────────────"

  CMD=(
    aiperf profile
    --model "$MODEL"
    --url "$URL"
    --endpoint-type "$ENDPOINT_TYPE"
    --concurrency "$C"
    --request-count "$REQUEST_COUNT"
    --tokenizer "$TOKENIZER"
    --sequence-distribution "$SEQ_DIST"
    --api-key "$API_KEY"
    --no-gpu-telemetry
    --no-server-metrics
  )

  # Add extra inputs if specified
  if [[ -n "$EXTRA_INPUTS" ]]; then
    CMD+=(--extra-inputs "$EXTRA_INPUTS")
  fi

  # Add any extra arguments
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    CMD+=($EXTRA_ARGS)
  fi

  LOGFILE="$RUN_DIR/logs/aiperf_$(date +%s).log"

  echo "Command: ${CMD[*]}"
  echo "Log:     $LOGFILE"
  echo ""

  if "${CMD[@]}" 2>&1 | tee "$LOGFILE"; then
    echo "✓ concurrency=$C completed successfully"
    PASS=$((PASS + 1))

    # Move aiperf outputs into the run directory if they exist in cwd
    for f in profile_export_aiperf.json profile_export_aiperf.csv profile_export.jsonl inputs.json; do
      if [[ -f "$SCRIPT_DIR/$f" ]]; then
        mv "$SCRIPT_DIR/$f" "$RUN_DIR/$f"
      fi
    done
  else
    echo "✗ concurrency=$C FAILED (see $LOGFILE)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=========================================="
echo " Done: $PASS/$TOTAL passed, $FAIL failed"
echo " Results in: $ARTIFACTS_DIR/"
echo "=========================================="