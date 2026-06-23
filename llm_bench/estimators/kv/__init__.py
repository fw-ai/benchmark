"""KV-cache memory estimators."""

from typing import Any

__all__ = ["estimate_kv_cache_memory_use"]


def __getattr__(name: str) -> Any:
    if name == "estimate_kv_cache_memory_use":
        from .kv_estimator import estimate_kv_cache_memory_use

        return estimate_kv_cache_memory_use
    raise AttributeError(name)
