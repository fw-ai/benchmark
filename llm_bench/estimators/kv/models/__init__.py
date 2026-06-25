"""Registered model-specific KV-cache estimators."""

from __future__ import annotations

from llm_bench.estimators.kv.models.kv_model_base import KvModelBase
from llm_bench.estimators.kv.models.deepseek4 import Deepseek4KvModel

MODEL_ESTIMATORS: tuple[KvModelBase, ...] = (Deepseek4KvModel(),)

__all__ = ["MODEL_ESTIMATORS"]
