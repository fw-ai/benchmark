"""DeepSeek-V4 KV-cache estimator.

The constants and formulas mirror the serving-side cache layout in
``fireworks.models.deepseek4`` without importing Fireworks at runtime.
"""

from __future__ import annotations

import math
from typing import Any

from llm_bench.estimators.kv.models.kv_model_base import KvModelBase, PrecisionSelection

DEFAULT_CONVERT_TO_PRECISION = "bf16,moe=mxfp8_mxfp4_ll,indexer=mxfp4,mm=mxfp8"
KV_CACHE_BLOCK_SIZE = 256
ROLLING_CACHE_PHYSICAL_SLOTS = 256
FLASH_MLA_SPARSE_K_BYTES = 584
FLASH_MLA_TMA_BLOCK_STRIDE_BYTES = 576
MXFP4_QUANT_BLOCK = 32


class Deepseek4KvModel(KvModelBase):
    name = "deepseek_v4"

    def matches(self, config: dict[str, Any]) -> bool:
        model_type = str(config.get("model_type") or "")
        return model_type == "deepseek_v4" or "compress_ratios" in config and "index_head_dim" in config

    def default_convert_to_precision(self, config: dict[str, Any]) -> str | None:
        return DEFAULT_CONVERT_TO_PRECISION

    def adjust_precision(
        self,
        base_precision: str,
        overrides: dict[str, str],
    ) -> tuple[str, dict[str, str]]:
        if base_precision != "fp4" or "indexer" in overrides:
            return base_precision, overrides

        default_overrides = {
            "moe": "mxfp8_mxfp4_ll",
            "indexer": "mxfp4",
            "mm": "mxfp8",
        }
        return "bf16", {**default_overrides, **overrides}

    def estimate(
        self,
        config: dict[str, Any],
        precision: PrecisionSelection,
        *,
        context_length: int,
        batch_size: int,
    ) -> dict[str, Any]:
        n_layers = int(config["num_hidden_layers"])
        ratios = _compress_ratios(config=config, n_layers=n_layers)
        c4_layers = sum(1 for ratio in ratios if ratio == 4)
        c128_layers = sum(1 for ratio in ratios if ratio == 128)

        num_blocks = batch_size * _ceil_div(context_length, KV_CACHE_BLOCK_SIZE)
        head_dim = int(config["head_dim"])
        index_head_dim = int(config.get("index_head_dim") or config.get("indexer_head_dim") or 128)
        pool_size = batch_size + 1

        c4a_attn = _compressed_paged_bytes(
            num_layers=c4_layers,
            num_blocks=num_blocks,
            compress_ratio=4,
        )
        c4a_attn += _rolling_cache_bytes(
            pool_size=pool_size,
            num_layers=c4_layers,
            per_entry_elements=4 * head_dim,
            dtype_bytes=2,
        )

        c128_attn = _compressed_paged_bytes(
            num_layers=c128_layers,
            num_blocks=num_blocks,
            compress_ratio=128,
        )
        c128_attn += _rolling_cache_bytes(
            pool_size=pool_size,
            num_layers=c128_layers,
            per_entry_elements=2 * head_dim,
            dtype_bytes=2,
        )

        c4a_rolling_swa_attn = _swa_rolling_bytes(
            pool_size=pool_size,
            num_layers=c4_layers,
            head_dim=head_dim,
            precision=precision,
        )
        c128_rolling_swa_attn = _swa_rolling_bytes(
            pool_size=pool_size,
            num_layers=c128_layers,
            head_dim=head_dim,
            precision=precision,
        )

        c4a_indexer = (
            c4_layers
            * num_blocks
            * _compressed_block_size(KV_CACHE_BLOCK_SIZE, 4)
            * _indexer_row_bytes(precision=precision, index_head_dim=index_head_dim)
        )
        c4a_indexer += _rolling_cache_bytes(
            pool_size=pool_size,
            num_layers=c4_layers,
            per_entry_elements=4 * index_head_dim,
            dtype_bytes=2,
        )

        c128a = {
            "swa_attn": c128_rolling_swa_attn,
            "c128_attn": c128_attn,
        }
        c128a["total"] = c128a["swa_attn"] + c128a["c128_attn"]

        c4a = {
            "swa_attn": c4a_rolling_swa_attn,
            "c4a_attn": c4a_attn,
            "c4a_indexer": c4a_indexer,
        }
        c4a["total"] = c4a["swa_attn"] + c4a["c4a_attn"] + c4a["c4a_indexer"]

        attention = {
            "c128a": c128a,
            "c4a": c4a,
        }
        attention["total"] = c128a["total"] + c4a["total"]

        total = attention["total"]
        return {
            "attention": attention,
            "total": total,
        }


def _compress_ratios(*, config: dict[str, Any], n_layers: int) -> list[int]:
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list):
        raise ValueError("DeepSeek-V4 config requires a list-valued compress_ratios")
    if len(ratios) < n_layers:
        raise ValueError(f"compress_ratios has {len(ratios)} entries, fewer than num_hidden_layers={n_layers}")

    ret = [int(ratio) for ratio in ratios[:n_layers]]
    invalid = [ratio for ratio in ret if ratio not in {0, 4, 128}]
    if invalid:
        raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {invalid}")
    return ret


def _compressed_paged_bytes(*, num_layers: int, num_blocks: int, compress_ratio: int) -> int:
    compressed_rows = _compressed_block_size(KV_CACHE_BLOCK_SIZE, compress_ratio)
    return num_layers * num_blocks * _sparse_fp8_padded_block_bytes(compressed_rows)


def _rolling_cache_bytes(
    *,
    pool_size: int,
    num_layers: int,
    per_entry_elements: int,
    dtype_bytes: int,
) -> int:
    return pool_size * ROLLING_CACHE_PHYSICAL_SLOTS * num_layers * per_entry_elements * dtype_bytes


def _swa_rolling_bytes(
    *,
    pool_size: int,
    num_layers: int,
    head_dim: int,
    precision: PrecisionSelection,
) -> int:
    ret = _rolling_cache_bytes(
        pool_size=pool_size,
        num_layers=num_layers,
        per_entry_elements=head_dim,
        dtype_bytes=precision.kv_dtype_bytes,
    )
    if precision.kv_dtype_name == "bf16":
        ret += num_layers * pool_size * _sparse_fp8_padded_block_bytes(ROLLING_CACHE_PHYSICAL_SLOTS)
    return ret


def _sparse_fp8_padded_block_bytes(block_size: int) -> int:
    logical_block_bytes = block_size * FLASH_MLA_SPARSE_K_BYTES
    return _ceil_div(logical_block_bytes, FLASH_MLA_TMA_BLOCK_STRIDE_BYTES) * FLASH_MLA_TMA_BLOCK_STRIDE_BYTES


def _ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator)


def _compressed_block_size(block_size: int, compress_ratio: int) -> int:
    if block_size % compress_ratio != 0:
        return 1
    return max(1, block_size // compress_ratio)


def _indexer_row_bytes(*, precision: PrecisionSelection, index_head_dim: int) -> int:
    indexer_dtype = precision.overrides.get("indexer", "fp8")
    if indexer_dtype == "mxfp4":
        return index_head_dim // 2 + index_head_dim // MXFP4_QUANT_BLOCK
    if indexer_dtype == "fp8":
        return index_head_dim + 4
    raise ValueError(f"Unsupported DeepSeek-V4 indexer dtype {indexer_dtype!r}")
