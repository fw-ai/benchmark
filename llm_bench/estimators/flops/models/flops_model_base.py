"""Base class for model-specific prefill FLOPs estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FlopsModelBase(ABC):
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
    ) -> dict[str, Any]:
        """Return prefill FLOPs keyed by component name."""
