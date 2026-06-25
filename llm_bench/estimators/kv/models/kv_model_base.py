"""Base class for model-specific KV-cache estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrecisionSelection:
    base_precision: str
    overrides: dict[str, str]
    kv_dtype_name: str
    kv_dtype_bytes: int


class KvModelBase(ABC):
    name: str

    @abstractmethod
    def matches(self, config: dict[str, Any]) -> bool:
        """Return true when this estimator handles the given text config."""

    @abstractmethod
    def default_convert_to_precision(self, config: dict[str, Any]) -> str | None:
        """Return the model's Fireworks default convert-to-precision recipe."""

    def adjust_precision(
        self,
        base_precision: str,
        overrides: dict[str, str],
    ) -> tuple[str, dict[str, str]]:
        """Apply model-specific precision-name behavior before KV dtype selection."""
        return base_precision, overrides

    @abstractmethod
    def estimate(
        self,
        config: dict[str, Any],
        precision: PrecisionSelection,
        *,
        context_length: int,
        batch_size: int,
    ) -> dict[str, Any]:
        """Return KV-cache bytes keyed by consumer name."""
