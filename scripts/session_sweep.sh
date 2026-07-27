#!/usr/bin/env bash
# Concurrency sweep over --session-mode (progressive-cache / agentic) traffic.
#
# Runs locust at each requested concurrency level with per-session unique
# init + growing history (exact-continuation cache), writing one summary CSV
# + log per level. Use it to map the concurrency-vs-latency/cache curve for a
# deployment that requires exact-continuation prompt caching (e.g. DeepSeek-V4).
#
# Required env:
#   PERF_API_KEY   Fireworks API key with access to --model
# Optional env / defaults:
#   MODEL          accounts/perf/deployments/<name>        (required if not passed)
#   HOST           https://api.fireworks.ai/inference
#   TOKENIZER      deepseek-ai/DeepSeek-V4-Pro
#   LEVELS         "1 2 4 8 16 32 64 128"
#   DUR            120s            (per-level run time)
#   OUTDIR         ./sweep_out
#
# Example:
#   PERF_API_KEY=fw_... \
#   MODEL=accounts/perf/deployments/camv33p6 \
#   LEVELS="1 2 4 8 16 32" DUR=120s \
#   bash scripts/session_sweep.sh
set -u

: "${PERF_API_KEY:?PERF_API_KEY env var is required}"
: "${MODEL:?MODEL env var is required (e.g. accounts/perf/deployments/camv33p6)}"
HOST="${HOST:-https://api.fireworks.ai/inference}"
TOKENIZER="${TOKENIZER:-deepseek-ai/DeepSeek-V4-Pro}"
LEVELS="${LEVELS:-1 2 4 8 16 32 64 128}"
DUR="${DUR:-120s}"
OUTDIR="${OUTDIR:-./sweep_out}"

# Session-mode workload parameters (override via env if desired).
P="${P:-60000}"                 # avg prompt length target
PCML="${PCML:-0}"               # local turn sizing only in session mode (not sent to server)
O="${O:-600}"                   # generation length (mean)
INIT="${INIT:-6000}"            # per-session init size (turn 1)
TURN="${TURN:-6000}"            # per-session increment (turns 2+)
MAXT="${MAXT:-120000}"          # session reset cap
TOL="${TOL:-500}"               # token validation tolerance
# Optional Prometheus scrape (leave empty to skip):
METRICS_URL="${METRICS_URL:-}"
METRICS_INTERVAL="${METRICS_INTERVAL:-5}"
METRICS_NAMES="${METRICS_NAMES:-}"
METRICS_FILE="${METRICS_FILE:-}"

mkdir -p "$OUTDIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOCUST_FILE="$SCRIPT_DIR/llm_bench/load_test.py"

metrics_args=""
[ -n "$METRICS_URL" ] && metrics_args="--metrics-url $METRICS_URL --metrics-interval $METRICS_INTERVAL"
[ -n "$METRICS_NAMES" ] && metrics_args="$metrics_args --metrics-names $METRICS_NAMES"
[ -n "$METRICS_FILE" ] && metrics_args="$metrics_args --metrics-file $METRICS_FILE"

echo "Sweep: model=$MODEL levels=[$LEVELS] dur=$DUR out=$OUTDIR"
for c in $LEVELS; do
  r=$(( c / 8 )); [ "$r" -lt 1 ] && r=1
  echo "=== [$(date +%H:%M:%S)] concurrency $c (spawn $r, dur $DUR) ==="
  locust -f "$LOCUST_FILE" --headless \
    -u "$c" -r "$r" -t "$DUR" \
    --provider fireworks -m "$MODEL" -H "$HOST" --api-key "$PERF_API_KEY" --tokenizer "$TOKENIZER" \
    --session-mode --chat --stream \
    -p "$P" -pcml "$PCML" -o "$O" --max-tokens-distribution uniform \
    --session-init-tokens "$INIT" --session-turn-tokens "$TURN" --session-max-tokens "$MAXT" \
    --token-validation-tolerance "$TOL" \
    $metrics_args \
    --summary-file "$OUTDIR/sweep_${c}u.csv" \
    > "$OUTDIR/sweep_${c}u.log" 2>&1
  echo "   concurrency $c done (locust exit $?)"
done
echo "=== SWEEP COMPLETE -> $OUTDIR ==="
