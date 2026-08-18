"""Shared generator-worker targeting helpers for benchmark clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SERVICE_INDEX_HEADER = "x-fireworks-generator-worker-service-index"
LOCAL_INDEX_HEADER = "x-fireworks-generator-worker-local-index"


@dataclass(frozen=True)
class RoutingConfig:
    """Generator-worker routing dimensions exposed by the deployment Envoy."""

    num_servers: int = 1
    num_gens: int = 1

    @property
    def enabled(self) -> bool:
        return self.num_servers > 1 or self.num_gens > 1

    @property
    def num_workers(self) -> int:
        return self.num_servers * self.num_gens

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        return f"server-first {self.num_servers}x{self.num_gens}"


def routing_headers_for_worker(cfg: Optional[RoutingConfig], worker_idx: int) -> dict[str, str]:
    """Return service/local targeting headers for a flat worker index.

    Server-first cycling balances partial worker sets across services. For
    example, the first four workers in a 2-service x 4-local deployment are
    (s0,l0), (s1,l0), (s0,l1), and (s1,l1).
    """
    if cfg is None or not cfg.enabled:
        return {}
    flat = worker_idx % cfg.num_workers
    server = flat % cfg.num_servers
    local = flat // cfg.num_servers
    return {SERVICE_INDEX_HEADER: str(server), LOCAL_INDEX_HEADER: str(local)}


def split_batch_across_workers(batch_size: int, cfg: RoutingConfig) -> list[tuple[int, int]]:
    """Return ``(worker_idx, n)`` assignments whose ``n`` values sum to the batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    worker_count = min(batch_size, cfg.num_workers)
    per_worker, remainder = divmod(batch_size, worker_count)
    return [(worker_idx, per_worker + (worker_idx < remainder)) for worker_idx in range(worker_count)]
