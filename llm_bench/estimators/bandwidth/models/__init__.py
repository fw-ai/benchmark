"""Registered model-specific decode bandwidth estimators."""

from __future__ import annotations

from llm_bench.estimators.bandwidth.models.bandwidth_model_base import BandwidthModelBase
from llm_bench.estimators.bandwidth.models.deepseek4 import Deepseek4BandwidthModel

MODEL_ESTIMATORS: tuple[BandwidthModelBase, ...] = (Deepseek4BandwidthModel(),)

__all__ = ["MODEL_ESTIMATORS"]

