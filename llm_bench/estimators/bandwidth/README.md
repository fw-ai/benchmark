# Decode Bandwidth Estimator

Estimates analytical decode byte movement by memory fabric for the given model.

```bash
python -m llm_bench.estimators.bandwidth.bandwidth_estimator \
  --hf-model-name deepseek-ai/DeepSeek-V4-Pro \
  --context-length=50000 \
  --batch-size=100 \
  --n-sequences=500 \
  --world-size=8 \
  --attn-sharding=dp \
  --moe-sharding=ep \
  --gib | jq
```

Sample output:

```json
{
  "hbm": {
    "attention_weights": {
      "c4a": {
        "weights": 352.69775390625,
        "activations": 5.559325218200684,
        "total": 358.2570791244507,
        "pct": 4.41
      },
      "c128a": {
        "weights": 350.56640625,
        "activations": 5.0258636474609375,
        "total": 355.59226989746094,
        "pct": 4.37
      },
      "total": 713.8493490219116,
      "pct": 8.78
    },
    "moe": {
      "routed": {
        "weights": 7205.625,
        "activations": 8.028030395507812,
        "total": 7213.653030395508,
        "pct": 88.75
      },
      "shared": {
        "weights": 150.1171875,
        "activations": 1.3380050659179688,
        "total": 151.45519256591797,
        "pct": 1.86
      },
      "total": 7365.108222961426,
      "pct": 90.61
    },
    "kv": {
      "c128a": {
        "swa_attn": 7.597923278808594,
        "sparse_attn": 3.30469012260437,
        "total": 10.902613401412964,
        "pct": 0.13
      },
      "c4a": {
        "swa_attn": 14.705657958984375,
        "sparse_attn": 8.362345397472382,
        "indexer": 15.551727265119553,
        "total": 38.61973062157631,
        "pct": 0.48
      },
      "total": 49.52234402298927,
      "pct": 0.61
    },
    "total": 8128.479916006327,
    "pct": 100
  },
  "nvlink": {
    "moe": {
      "dispatch": 2.137899398803711,
      "combine": 2.137899398803711,
      "total": 4.275798797607422,
      "pct": 100
    },
    "total": 4.275798797607422,
    "pct": 100
  }
}
```
