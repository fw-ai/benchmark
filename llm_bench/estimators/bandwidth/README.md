# Decode Bandwidth Estimator

Estimates analytical decode byte movement by memory fabric for the given model.

```bash
python -m llm_bench.estimators.bandwidth.bandwidth_estimator \
  --hf-model-name deepseek-ai/DeepSeek-V4-Pro \
  --context-length=50000 \
  --batch-size=500 \
  --n-sequences=500 \
  --world-size=8 \
  --attn-sharding=dp \
  --moe-sharding=ep \
  --gib | jq
```

