# Production Load Tests (GPT-OSS-120B style)

Run the same load test matrix as in the [GPT-OSS-120B Load Test Analysis](https://docs.google.com/document/d/1hTf5OxYLTNO1uYCEgU9pYM7xwwdHUj8mB87YzkxPHW8/edit) using this repo with the **default dataset** (limericks).

## Test configuration (matches the doc)

- **Dataset**: default (`limericks`) — no need to pass `--dataset`.
- **Concurrency levels**: 4, 8, 12, 16, 24, 32, 48, 64
- **Duration**: 120s per run (after spawn)
- **Prompt cache**: disabled (`--prompt-cache-max-len 0`)
- **Engines** (set host and model per deployment):
  - vLLM
  - FA+Eagle3
  - FA NoDraft

## Six input/output configs (from the doc)

| Config | `--prompt-tokens` | `--max-tokens` |
|--------|--------------------|----------------|
| 1. Short output   | 256 | 128 |
| 2. Medium output | 256 | 512 |
| 3. Long output   | 256 | 1024 |
| 4. Atlassian-like| 1024 | 256 |
| 5. Long I/O      | 1024 | 1024 |
| 6. Long prompt   | 2048 | 256 |

## Requirements

- **API key**: Set `FIREWORKS_API_KEY` or `API_KEY` in the environment before running. The script never prints it or passes it on the command line.
- **Tokenizer**: Pass as the 4th argument to the script; required for the limericks dataset. For **GPT-OSS-120B** use **`openai/gpt-oss-120b`** (HuggingFace).
- **Host**: Your deployment base URL (e.g. `https://api.fireworks.ai/inference` for Fireworks, or a direct deployment URL).

## Smoke test first (Eagle3-FA deployment)

Test with the Eagle3-FA deployment (pyroworks account) before running the full matrix.

**API key required:** Set `FIREWORKS_API_KEY` or `API_KEY` in the environment before running; otherwise you'll get **401 Unauthorized**.

From `llm_bench/`:

```bash
cd llm_bench
source .venv/bin/activate
export FIREWORKS_API_KEY=...   # or API_KEY
export TOKENIZER=openai/gpt-oss-120b

locust -f load_test.py --headless --only-summary \
  -H "https://api.fireworks.ai/inference" \
  --provider fireworks \
  --model "accounts/pyroworks/deployments/atlassian-gpt-oss-120b-h200-eagle3-fa" \
  --api-key "$FIREWORKS_API_KEY" \
  --tokenizer "$TOKENIZER" \
  -u 4 -r 4 \
  -t 30s \
  -p 256 -o 128 \
  --prompt-cache-max-len 0 \
  --stream \
  --csv results/smoke_stats --html results/smoke_report.html
```

If that succeeds, run the full matrix for that deployment:

```bash
./run_production_load_tests.sh "https://api.fireworks.ai/inference" fireworks "accounts/pyroworks/deployments/atlassian-gpt-oss-120b-h200-eagle3-fa" openai/gpt-oss-120b
```

## Single run example

When you run the load test, pass the base model as the tokenizer path (for limericks prompt token counts):

```bash
cd /path/to/benchmark/llm_bench
# Set FIREWORKS_API_KEY or API_KEY in the environment (never printed)

locust -f load_test.py --headless --only-summary \
  -H "https://YOUR_VLLM_DEPLOYMENT_URL" \
  --provider vllm \
  --model "YOUR_MODEL_ID" \
  --api-key "$FIREWORKS_API_KEY" \
  --tokenizer YOUR_TOKENIZER_PATH \
  -u 4 -r 4 \
  -t 120s \
  -p 256 -o 128 \
  --prompt-cache-max-len 0 \
  --stream \
  --csv results/stats --html results/report.html
```

## Full matrix

Use the provided script. Pass tokenizer as the 4th argument (e.g. `openai/gpt-oss-120b` for GPT-OSS-120B):

```bash
cd llm_bench
# Set FIREWORKS_API_KEY or API_KEY in the environment before running

# Example: Eagle3-FA deployment (pyroworks account)
./run_production_load_tests.sh "https://api.fireworks.ai/inference" fireworks "accounts/pyroworks/deployments/atlassian-gpt-oss-120b-h200-eagle3-fa" openai/gpt-oss-120b
```

Results are written under `results/production_load_test_<provider>_<timestamp>/`.

## NoDraft: when the deployment is READY

Once the NoDraft model is updated to **READY**, run the same full matrix for NoDraft. The script checks the endpoint first (curl); if it returns 200 and valid JSON, it runs the full load test and tees output to `results/run_nodraft.log`.

```bash
cd llm_bench
# Set FIREWORKS_API_KEY or API_KEY

# Foreground (~2.5–3 hours)
./run_nodraft_load_test.sh

# Background
nohup ./run_nodraft_load_test.sh > results/run_nodraft.log 2>&1 &
```

Track progress: `./check_load_test_status.sh nodraft`

## Tracking a running load test

From `llm_bench/`:

```bash
# Eagle (default): progress, state, last 20 log lines
./check_load_test_status.sh eagle

# NoDraft run
./check_load_test_status.sh nodraft

# Custom log path
./check_load_test_status.sh results/run_eagle3_fa.log
```

Or watch the log live: `tail -f results/run_eagle3_fa.log`

## Updating the Google Doc with results

After a load test completes, append the results to the [comparison doc](https://docs.google.com/document/d/1wlkryDkAlAperola_hE-jZj8H1MFFXlxBGCZE-kMNgk/edit):

```bash
# Eagle results (already run)
python update_doc_with_results.py eagle > results/eagle_results_for_doc.md
# Then copy-paste results/eagle_results_for_doc.md at the bottom of the doc

# NoDraft results (when run completes)
python update_doc_with_results.py nodraft > results/nodraft_results_for_doc.md
# Then copy-paste results/nodraft_results_for_doc.md at the bottom of the doc

# Both at once
python update_doc_with_results.py both
```
