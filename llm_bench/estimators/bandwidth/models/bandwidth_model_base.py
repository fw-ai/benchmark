"""Base class for model-specific decode bandwidth estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BandwidthModelBase(ABC):
    name: str

    @abstractmethod
    def matches(self, config: dict[str, Any]) -> bool:
        """Return true when this estimator handles the given text config."""

    @abstractmethod
    def estimate(
        self,
        config: dict[str, Any],
        *,
        context_length: int,
        batch_size: int,
        n_sequences: int,
        world_size: int,
        attn_sharding: str,
        moe_sharding: str,
        convert_to_precision: str | None,
        activation_dtype: str,
    ) -> dict[str, Any]:
        """Return decode byte movement keyed by memory fabric."""
