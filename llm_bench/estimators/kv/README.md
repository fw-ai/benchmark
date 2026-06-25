# KV Cache Estimator

Estimates KV/cache memory in bytes for the given model.

```bash
python -m llm_bench.estimators.kv.kv_estimator \
  --hf-model-name deepseek-ai/DeepSeek-V4-Pro \
  --context-length=50000 \
  --batch-size=100 \
  --gib | jq
```

Sample output:

```json
{
  "attention": {
    "c128a": {
      "swa_attn": 1.2011001110076904,
      "sparse_attn": 2.5066347122192383,
      "total": 3.7077348232269287,
      "pct": 11.79
    },
    "c4a": {
      "swa_attn": 1.1623549461364746,
      "sparse_attn": 23.461790084838867,
      "indexer": 3.1229782104492188,
      "total": 27.74712324142456,
      "pct": 88.21
    },
    "total": 31.45485806465149,
    "pct": 100.0
  },
  "total": 31.45485806465149
}
```
