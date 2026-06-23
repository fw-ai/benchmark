# Prefill FLOPs Estimator

Estimates analytical prefill FLOPs for the given model.

```bash
python -m llm_bench.estimators.flops.flops_estimator \
  --hf-model-name deepseek-ai/DeepSeek-V4-Pro \
  --context-length=50000 \
  --batch-size=500 \
  --peta | jq
```