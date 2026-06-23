"""Registered model-specific prefill FLOPs estimators."""

from __future__ import annotations

from llm_bench.estimators.flops.models.deepseek4 import Deepseek4FlopsModel
from llm_bench.estimators.flops.models.flops_model_base import FlopsModelBase

MODEL_ESTIMATORS: tuple[FlopsModelBase, ...] = (Deepseek4FlopsModel(),)

__all__ = ["MODEL_ESTIMATORS"]
