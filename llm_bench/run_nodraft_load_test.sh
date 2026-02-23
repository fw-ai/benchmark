#!/usr/bin/env bash
# Run production load test for NoDraft deployment.
# Use this once the NoDraft model is updated to READY.
#
# 1. Probes the endpoint (curl). If that fails, exits without running the matrix.
# 2. Runs the full 6×8 matrix (same as run_production_load_tests.sh).
# Output: results/run_nodraft.log and results/production_load_test_fireworks_<timestamp>/
#
# Usage (from llm_bench/):
#   export FIREWORKS_API_KEY=...   # or API_KEY
#   ./run_nodraft_load_test.sh              # foreground
#   nohup ./run_nodraft_load_test.sh &      # background
#
# Check progress: ./check_load_test_status.sh nodraft

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="https://api.fireworks.ai/inference"
PROVIDER="fireworks"
MODEL="accounts/pyroworks/deployments/atlassian-gpt-oss-120b-h200-nodraft"
TOKENIZER="openai/gpt-oss-120b"
LOG="results/run_nodraft.log"

if [ -z "${FIREWORKS_API_KEY:-}" ] && [ -z "${API_KEY:-}" ]; then
  echo "Set FIREWORKS_API_KEY or API_KEY in the environment before running." >&2
  exit 1
fi
[ -n "${FIREWORKS_API_KEY:-}" ] && export API_KEY="$FIREWORKS_API_KEY"

echo "Checking NoDraft endpoint..."
CODE=$(curl -s -o /tmp/nodraft_probe.json -w "%{http_code}" -X POST "${HOST}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"stream\":false}")
if [[ "$CODE" != 200 ]]; then
  echo "NoDraft endpoint returned HTTP $CODE (expected 200). Is the deployment READY?" >&2
  exit 1
fi
if ! python3 -c "import json; d=json.load(open('/tmp/nodraft_probe.json')); exit(0 if d.get('choices') and len(d['choices'])>0 else 1)" 2>/dev/null; then
  echo "NoDraft response missing valid choices. Is the deployment READY?" >&2
  exit 1
fi
echo "NoDraft endpoint OK (HTTP 200, valid response). Starting full load test."
rm -f /tmp/nodraft_probe.json

mkdir -p results
./run_production_load_tests.sh "$HOST" "$PROVIDER" "$MODEL" "$TOKENIZER" 2>&1 | tee "$LOG"
