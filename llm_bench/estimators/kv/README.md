# KV Cache Estimator

Estimates KV/cache memory in bytes from a model `config.json`. Use exact,
case-sensitive Hugging Face model names or a local config/model directory.

```bash
python -m llm_bench.estimators.kv.kv_estimator \
  --hf-model-name deepseek-ai/DeepSeek-V4-Pro \
  --context-length=50000 \
  --batch-size=500 \
  --gib | jq
```