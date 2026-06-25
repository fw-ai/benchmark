# Prefill FLOPs Estimator

Estimates analytical prefill FLOPs for the given model.

```bash
python -m llm_bench.estimators.flops.flops_estimator \
  --hf-model-name deepseek-ai/DeepSeek-V4-Pro \
  --context-length=50000 \
  --batch-size=100 \
  --peta | jq
```

Sample output:

```json
{
  "attention": {
    "c128a": {
      "proj": 92.96674816,
      "compress_attn": 2.27540992,
      "swa_attn": 5.1943317700608,
      "sparse_attn": 7.916012961792,
      "total": 108.3525028118528,
      "pct": 19.65
    },
    "c4a": {
      "proj": 89.9678208,
      "compress_attn": 4.4040192,
      "compress_indexer": 1.1010048,
      "indexer_proj": 3.9124992,
      "indexer_score": 15.3593856,
      "swa_attn": 5.026772680704,
      "sparse_attn": 38.615245651968,
      "total": 158.386747932672,
      "pct": 28.72
    },
    "total": 266.7392507445248,
    "pct": 48.37
  },
  "moe": {
    "router": 1.67903232,
    "experts": 282.07742976,
    "total": 283.75646208,
    "pct": 51.45
  },
  "hyper_connection": {
    "pre": 0.83951616,
    "post": 0.13991936,
    "total": 0.97943552,
    "pct": 0.18
  },
  "total": 551.4751483445248
}
```
