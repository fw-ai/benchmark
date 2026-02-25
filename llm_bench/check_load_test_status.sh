#!/usr/bin/env bash
# Quick status for production load test (Eagle or other).
# Usage: ./check_load_test_status.sh [eagle|nodraft|vllm|path/to/log]
#   eagle  -> results/run_eagle3_fa.log (default)
#   nodraft -> results/run_nodraft.log
#   vllm   -> results/run_vllm.log
#   path   -> use that log file

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NAME="${1:-eagle}"
if [[ "$NAME" == eagle ]]; then
  LOG="results/run_eagle3_fa.log"
elif [[ "$NAME" == nodraft ]]; then
  LOG="results/run_nodraft.log"
elif [[ "$NAME" == vllm ]]; then
  LOG="results/run_vllm.log"
else
  LOG="$NAME"
fi

TOTAL_RUNS=48   # 6 configs × 8 concurrency levels

echo "=============================================="
echo "Load test status: ${NAME}"
echo "Log: $LOG"
echo "=============================================="

# Is the load test or locust still running?
if pgrep -f "run_production_load_tests.sh" >/dev/null 2>&1; then
  echo "State: RUNNING (run_production_load_tests.sh)"
elif pgrep -f "locust -f load_test.py" >/dev/null 2>&1; then
  echo "State: RUNNING (locust)"
else
  echo "State: not running"
fi

if [[ ! -f "$LOG" ]]; then
  echo "Progress: no log file yet"
  exit 0
fi

# Count completed runs (each run prints "Qps\s*:" in the summary)
DONE=$(grep -c "Qps\s*:" "$LOG" 2>/dev/null || echo 0)
DONE=$(echo "$DONE" | head -1 | tr -d ' \r\n'); DONE=${DONE:-0}
echo "Progress: $DONE / $TOTAL_RUNS runs"

if [[ "$DONE" -gt 0 ]] && [[ "$DONE" -lt "$TOTAL_RUNS" ]]; then
  REMAIN=$((TOTAL_RUNS - DONE))
  # ~2.5 min per run on average (120s + spawn/overhead)
  MINS=$((REMAIN * 3))
  echo "Estimated remaining: ~${MINS} min ($REMAIN runs)"
fi

echo "----------------------------------------------"
echo "Last 20 lines of log:"
echo "----------------------------------------------"
tail -20 "$LOG"
