"""DeepSeek-V4 decode byte-movement estimator.

The estimator models one decode forward for ``n_sequences`` split into rounds
of ``batch_size``. It reports bytes moved across all rounds, separated by HBM
and NVLink/NVSwitch traffic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

DEFAULT_HF_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_CONVERT_TO_PRECISION = "bf16,moe=mxfp8_mxfp4_ll,indexer=mxfp4,mm=mxfp8"

KV_CACHE_BLOCK_SIZE = 256
ROLLING_CACHE_PHYSICAL_SLOTS = 256
FLASH_MLA_SPARSE_K_BYTES = 584
MXFP4_QUANT_BLOCK = 32


def estimate_deepseek4_bandwidth(
    *,
    hf_model_name: str = DEFAULT_HF_MODEL_NAME,
    context_length: int,
    batch_size: int,
    n_sequences: int,
    world_size: int,
    attn_sharding: str,
    moe_sharding: str,
    convert_to_precision: str | None = None,
    activation_dtype: str = "bf16",
) -> dict[str, Any]:
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_sequences <= 0:
        raise ValueError(f"n_sequences must be positive, got {n_sequences}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if attn_sharding != "dp":
        raise ValueError(f"Unsupported attn_sharding {attn_sharding!r}; currently only 'dp' is supported")
    if moe_sharding != "ep":
        raise ValueError(f"Unsupported moe_sharding {moe_sharding!r}; currently only 'ep' is supported")

    config = _text_config(_load_model_config(hf_model_name))
    precision = _parse_precision(convert_to_precision or DEFAULT_CONVERT_TO_PRECISION)
    activation_bytes = _dtype_bytes(activation_dtype)

    total = _empty_estimate()
    remaining = n_sequences
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        total = _add_estimates(
            total,
            _estimate_round(
                config=config,
                precision=precision,
                context_length=context_length,
                batch_size=current_batch,
                world_size=world_size,
                attn_sharding=attn_sharding,
                moe_sharding=moe_sharding,
                activation_bytes=activation_bytes,
            ),
        )
        remaining -= current_batch

    return _with_totals_and_percentages(total)


def _estimate_round(
    *,
    config: dict[str, Any],
    precision: dict[str, str],
    context_length: int,
    batch_size: int,
    world_size: int,
    attn_sharding: str,
    moe_sharding: str,
    activation_bytes: float,
) -> dict[str, Any]:
    if attn_sharding == "dp":
        active_attention_weight_replicas = float(world_size)
    else:
        raise ValueError(f"Unsupported attn_sharding {attn_sharding!r}")

    hbm_attention = _attention_hbm_bytes(
        config=config,
        precision=precision,
        batch_size=batch_size,
        active_replicas=active_attention_weight_replicas,
        activation_bytes=activation_bytes,
    )
    if moe_sharding != "ep":
        raise ValueError(f"Unsupported moe_sharding {moe_sharding!r}")
    hbm_moe = _moe_hbm_bytes(
        config=config,
        precision=precision,
        batch_size=batch_size,
        world_size=world_size,
        activation_bytes=activation_bytes,
    )
    hbm_kv = _kv_moved_bytes(
        config=config,
        precision=precision,
        context_length=context_length,
        batch_size=batch_size,
    )
    nvlink_moe = _moe_nvlink_bytes(
        config=config,
        batch_size=batch_size,
        world_size=world_size,
        activation_bytes=activation_bytes,
    )

    return {
        "hbm": {
            "attention_weights": hbm_attention,
            "moe": hbm_moe,
            "kv": hbm_kv,
        },
        "nvlink": {
            "moe": nvlink_moe,
        },
    }


def _attention_hbm_bytes(
    *,
    config: dict[str, Any],
    precision: dict[str, str],
    batch_size: int,
    active_replicas: float,
    activation_bytes: float,
) -> dict[str, Any]:
    n_layers = int(config["num_hidden_layers"])
    ratios = _compress_ratios(config=config, n_layers=n_layers)
    c4_layers = sum(1 for ratio in ratios if ratio == 4)
    c128_layers = sum(1 for ratio in ratios if ratio == 128)

    hidden_size = int(config["hidden_size"])
    head_dim = int(config["head_dim"])
    num_attention_heads = int(config["num_attention_heads"])
    q_lora_rank = int(config["q_lora_rank"])
    o_lora_rank = int(config["o_lora_rank"])
    o_groups = int(config["o_groups"])
    index_head_dim = int(config.get("index_head_dim") or config.get("indexer_head_dim") or 128)
    index_n_heads = int(config.get("index_n_heads", 64))

    q_width = num_attention_heads * head_dim
    common_projection_elems = (
        hidden_size * (q_lora_rank + head_dim)
        + q_lora_rank * q_width
        + q_width * o_lora_rank
        + o_groups * o_lora_rank * hidden_size
    )
    c128_compressor_elems = hidden_size * head_dim
    c4_attention_compressor_elems = 2 * hidden_size * head_dim
    c4_indexer_compressor_elems = 2 * hidden_size * index_head_dim
    c4_indexer_proj_elems = q_lora_rank * index_n_heads * index_head_dim + hidden_size * index_n_heads

    mm_bytes = _component_dtype_bytes(precision, "mm")
    indexer_bytes = _component_dtype_bytes(precision, "indexer")

    c128a_weights = c128_layers * active_replicas * (common_projection_elems + c128_compressor_elems) * mm_bytes
    c4a_weights = c4_layers * active_replicas * (
        (common_projection_elems + c4_attention_compressor_elems + c4_indexer_compressor_elems) * mm_bytes
        + c4_indexer_proj_elems * indexer_bytes
    )

    # Approximate HBM activation movement for decode attention projection and
    # compressor/indexer paths. KV movement itself is reported under hbm.kv.
    common_projection_activation_elems = batch_size * (
        hidden_size
        + q_lora_rank
        + head_dim
        + q_lora_rank
        + q_width
        + q_width
        + o_lora_rank
        + o_groups * o_lora_rank
        + hidden_size
    )
    c128_compressor_activation_elems = batch_size * (hidden_size + head_dim)
    c4_attention_compressor_activation_elems = batch_size * (hidden_size + 2 * head_dim)
    c4_indexer_compressor_activation_elems = batch_size * (hidden_size + 2 * index_head_dim)
    c4_indexer_projection_activation_elems = batch_size * (
        q_lora_rank + index_n_heads * index_head_dim + hidden_size + index_n_heads
    )

    c128a_activations = (
        c128_layers * (common_projection_activation_elems + c128_compressor_activation_elems) * activation_bytes
    )
    c4a_activations = (
        c4_layers
        * (
            common_projection_activation_elems
            + c4_attention_compressor_activation_elems
            + c4_indexer_compressor_activation_elems
            + c4_indexer_projection_activation_elems
        )
        * activation_bytes
    )
    return {
        "c128a": {
            "weights": c128a_weights,
            "activations": c128a_activations,
        },
        "c4a": {
            "weights": c4a_weights,
            "activations": c4a_activations,
        },
    }


def _moe_hbm_bytes(
    *,
    config: dict[str, Any],
    precision: dict[str, str],
    batch_size: int,
    world_size: int,
    activation_bytes: float,
) -> dict[str, Any]:
    n_layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    n_routed_experts = int(config["n_routed_experts"])
    num_experts_per_tok = int(config["num_experts_per_tok"])
    n_shared_experts = int(config.get("n_shared_experts", 0))

    expert_weight_elems = 3 * hidden_size * intermediate_size
    moe_weight_bytes = _component_dtype_bytes(precision, "moe")

    # Wide EP keeps routed experts sharded across the full world. Count the
    # local expert shards across all GPUs, matching the DP attention accounting
    # that counts every GPU's local HBM movement.
    routed_expert_weights = n_layers * n_routed_experts * expert_weight_elems * moe_weight_bytes
    shared_expert_weights = n_layers * world_size * n_shared_experts * expert_weight_elems * moe_weight_bytes

    # Approximate HBM activation movement for routed expert matmuls:
    # gate/up reads hidden and writes 2I, down reads I and writes hidden.
    routed_activation_elems = batch_size * num_experts_per_tok * (2 * hidden_size + 3 * intermediate_size)
    shared_activation_elems = batch_size * n_shared_experts * (2 * hidden_size + 3 * intermediate_size)

    return {
        "routed": {
            "weights": routed_expert_weights,
            "activations": n_layers * routed_activation_elems * activation_bytes,
        },
        "shared": {
            "weights": shared_expert_weights,
            "activations": n_layers * shared_activation_elems * activation_bytes,
        },
    }


def _kv_moved_bytes(
    *,
    config: dict[str, Any],
    precision: dict[str, str],
    context_length: int,
    batch_size: int,
) -> dict[str, Any]:
    n_layers = int(config["num_hidden_layers"])
    ratios = _compress_ratios(config=config, n_layers=n_layers)
    c4_layers = sum(1 for ratio in ratios if ratio == 4)
    c128_layers = sum(1 for ratio in ratios if ratio == 128)

    head_dim = int(config["head_dim"])
    index_head_dim = int(config.get("index_head_dim") or config.get("indexer_head_dim") or 128)
    index_topk = int(config.get("index_topk", 1024))
    kv_dtype_bytes = _kv_dtype_bytes(precision, config=config)
    indexer_row_bytes = _indexer_row_bytes(precision=precision, index_head_dim=index_head_dim)

    c128a = {
        "swa_attn": _rolling_read_write_bytes(
            batch_size=batch_size,
            num_layers=c128_layers,
            context_length=context_length,
            per_entry_elements=2 * head_dim,
            dtype_bytes=2,
        ),
        "sparse_attn": c128_layers
        * batch_size
        * (_ceil_div(context_length, 128) + 1)
        * FLASH_MLA_SPARSE_K_BYTES,
    }
    c4a = {
        "swa_attn": _rolling_read_write_bytes(
            batch_size=batch_size,
            num_layers=c4_layers,
            context_length=context_length,
            per_entry_elements=4 * head_dim,
            dtype_bytes=kv_dtype_bytes,
        ),
        "sparse_attn": c4_layers
        * batch_size
        * (min(_ceil_div(context_length, 4), index_topk) + 1)
        * FLASH_MLA_SPARSE_K_BYTES,
        "indexer": c4_layers
        * batch_size
        * (
            (_ceil_div(context_length, 4) + 1) * indexer_row_bytes
            + (min(context_length, ROLLING_CACHE_PHYSICAL_SLOTS) + 1) * 4 * index_head_dim * 2
        ),
    }
    return {
        "c128a": c128a,
        "c4a": c4a,
    }


def _moe_nvlink_bytes(
    *,
    config: dict[str, Any],
    batch_size: int,
    world_size: int,
    activation_bytes: float,
) -> dict[str, float]:
    n_layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    num_experts_per_tok = int(config["num_experts_per_tok"])
    remote_fraction = 0.0 if world_size <= 1 else 1.0 - 1.0 / world_size
    one_way = n_layers * batch_size * num_experts_per_tok * hidden_size * activation_bytes * remote_fraction
    return {
        "dispatch": one_way,
        "combine": one_way,
    }


def _empty_estimate() -> dict[str, Any]:
    return {
        "hbm": {
            "attention_weights": {
                "c4a": {
                    "weights": 0.0,
                    "activations": 0.0,
                },
                "c128a": {
                    "weights": 0.0,
                    "activations": 0.0,
                },
            },
            "moe": {
                "routed": {
                    "weights": 0.0,
                    "activations": 0.0,
                },
                "shared": {
                    "weights": 0.0,
                    "activations": 0.0,
                },
            },
            "kv": {
                "c128a": {
                    "swa_attn": 0.0,
                    "sparse_attn": 0.0,
                },
                "c4a": {
                    "swa_attn": 0.0,
                    "sparse_attn": 0.0,
                    "indexer": 0.0,
                },
            },
        },
        "nvlink": {
            "moe": {
                "dispatch": 0.0,
                "combine": 0.0,
            },
        },
    }


def _with_totals_and_percentages(estimate: dict[str, Any]) -> dict[str, Any]:
    hbm = estimate["hbm"]
    hbm["attention_weights"]["c4a"]["total"] = (
        hbm["attention_weights"]["c4a"]["weights"] + hbm["attention_weights"]["c4a"]["activations"]
    )
    hbm["attention_weights"]["c128a"]["total"] = (
        hbm["attention_weights"]["c128a"]["weights"] + hbm["attention_weights"]["c128a"]["activations"]
    )
    hbm["attention_weights"]["total"] = (
        hbm["attention_weights"]["c4a"]["total"] + hbm["attention_weights"]["c128a"]["total"]
    )
    hbm["moe"]["routed"]["total"] = hbm["moe"]["routed"]["weights"] + hbm["moe"]["routed"]["activations"]
    hbm["moe"]["shared"]["total"] = hbm["moe"]["shared"]["weights"] + hbm["moe"]["shared"]["activations"]
    hbm["moe"]["total"] = hbm["moe"]["routed"]["total"] + hbm["moe"]["shared"]["total"]
    hbm["kv"]["c128a"]["total"] = hbm["kv"]["c128a"]["swa_attn"] + hbm["kv"]["c128a"]["sparse_attn"]
    hbm["kv"]["c4a"]["total"] = (
        hbm["kv"]["c4a"]["swa_attn"] + hbm["kv"]["c4a"]["sparse_attn"] + hbm["kv"]["c4a"]["indexer"]
    )
    hbm["kv"]["total"] = hbm["kv"]["c128a"]["total"] + hbm["kv"]["c4a"]["total"]
    hbm["total"] = hbm["attention_weights"]["total"] + hbm["moe"]["total"] + hbm["kv"]["total"]

    nvlink = estimate["nvlink"]
    nvlink["moe"]["total"] = nvlink["moe"]["dispatch"] + nvlink["moe"]["combine"]
    nvlink["total"] = nvlink["moe"]["total"]

    hbm["attention_weights"]["c4a"]["pct"] = _pct(hbm["attention_weights"]["c4a"]["total"], hbm["total"])
    hbm["attention_weights"]["c128a"]["pct"] = _pct(hbm["attention_weights"]["c128a"]["total"], hbm["total"])
    hbm["attention_weights"]["pct"] = _pct(hbm["attention_weights"]["total"], hbm["total"])
    hbm["moe"]["routed"]["pct"] = _pct(hbm["moe"]["routed"]["total"], hbm["total"])
    hbm["moe"]["shared"]["pct"] = _pct(hbm["moe"]["shared"]["total"], hbm["total"])
    hbm["moe"]["pct"] = _pct(hbm["moe"]["total"], hbm["total"])
    hbm["kv"]["c128a"]["pct"] = _pct(hbm["kv"]["c128a"]["total"], hbm["total"])
    hbm["kv"]["c4a"]["pct"] = _pct(hbm["kv"]["c4a"]["total"], hbm["total"])
    hbm["kv"]["pct"] = _pct(hbm["kv"]["total"], hbm["total"])
    nvlink["moe"]["pct"] = _pct(nvlink["moe"]["total"], nvlink["total"])
    hbm["pct"] = _pct(hbm["total"], hbm["total"])
    nvlink["pct"] = _pct(nvlink["total"], nvlink["total"])
    return _normalize_numbers(estimate)


def _add_estimates(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    ret: dict[str, Any] = {}
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, dict):
            ret[key] = _add_estimates(left_value, right_value)
        else:
            ret[key] = float(left_value) + float(right_value)
    return ret


def _load_model_config(hf_model_name: str) -> dict[str, Any]:
    path = Path(hf_model_name).expanduser()
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.is_file():
        with config_path.open() as f:
            return json.load(f)

    downloaded = hf_hub_download(repo_id=hf_model_name, filename="config.json")
    with open(downloaded) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{hf_model_name} config.json did not contain a JSON object")
    return config


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        merged = dict(text_config)
        for key in ("model_type", "quantization_config"):
            if key not in merged and key in config:
                merged[key] = config[key]
        return merged
    return config


def _parse_precision(value: str) -> dict[str, str]:
    value = value.strip()
    if value.startswith("{"):
        data = json.loads(value)
        name = data.get("name")
        if not isinstance(name, str):
            raise ValueError(f"convert_to_precision JSON must contain a string name: {value}")
        return {"base": _normalize_dtype_name(name)}

    precision: dict[str, str] = {}
    base: str | None = None
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" in part:
            key, override = part.split("=", 1)
            precision[key.strip()] = _normalize_dtype_name(override.strip())
        elif base is None:
            base = _normalize_dtype_name(part)
        else:
            raise ValueError(f"More than one base precision in {value!r}")
    precision["base"] = base or "bf16"
    return precision


def _component_dtype_bytes(precision: dict[str, str], component: str) -> float:
    return _dtype_bytes(precision.get(component) or precision["base"])


def _kv_dtype_bytes(precision: dict[str, str], *, config: dict[str, Any]) -> float:
    if "kv" in precision:
        return _dtype_bytes(precision["kv"])
    base = precision["base"]
    is_moe = bool(config.get("n_routed_experts") or config.get("num_experts") or config.get("moe"))
    if base in {"bf16", "fp16", "fp32"}:
        return _dtype_bytes(base)
    if is_moe and base == "fp4":
        return _dtype_bytes("fp8")
    if base in {"fp8", "fp4"}:
        return _dtype_bytes("fp8")
    return _dtype_bytes("bf16")


def _dtype_bytes(value: str) -> float:
    value = _normalize_dtype_name(value)
    if value == "fp32":
        return 4.0
    if value in {"bf16", "fp16"}:
        return 2.0
    if value in {"fp8", "mxfp8", "mxfp8_mxfp4_ll"}:
        return 1.0
    if value in {"fp4", "mxfp4", "nvfp4"}:
        return 0.5
    if "fp4" in value:
        return 0.5
    if "fp8" in value:
        return 1.0
    raise ValueError(f"Unsupported dtype {value!r}")


def _normalize_dtype_name(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8": "fp8",
        "float8_e4m3": "fp8",
        "float8_e4m3fn": "fp8",
        "float4": "fp4",
    }
    return aliases.get(value, value)


def _compress_ratios(*, config: dict[str, Any], n_layers: int) -> list[int]:
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list):
        raise ValueError("DeepSeek-V4 config requires a list-valued compress_ratios")
    if len(ratios) < n_layers:
        raise ValueError(f"compress_ratios has {len(ratios)} entries, fewer than num_hidden_layers={n_layers}")
    return [int(ratio) for ratio in ratios[:n_layers]]


def _rolling_read_write_bytes(
    *,
    batch_size: int,
    num_layers: int,
    context_length: int,
    per_entry_elements: int,
    dtype_bytes: float,
) -> float:
    entries = min(context_length, ROLLING_CACHE_PHYSICAL_SLOTS) + 1
    return batch_size * num_layers * entries * per_entry_elements * dtype_bytes


def _indexer_row_bytes(*, precision: dict[str, str], index_head_dim: int) -> float:
    indexer_dtype = precision.get("indexer", "fp8")
    if indexer_dtype == "mxfp4":
        return index_head_dim // 2 + index_head_dim // MXFP4_QUANT_BLOCK
    if indexer_dtype in {"fp8", "mxfp8"}:
        return index_head_dim + 4
    if "fp4" in indexer_dtype:
        return index_head_dim / 2
    raise ValueError(f"Unsupported DeepSeek-V4 indexer dtype {indexer_dtype!r}")


def _expected_active_bins(draws: float, bins: int) -> float:
    if bins <= 0 or draws <= 0:
        return 0.0
    return bins * (1.0 - (1.0 - 1.0 / bins) ** draws)


def _ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator)


def _pct(value: float, total: float) -> float:
    if not total:
        return 0.0
    pct = value / total * 100.0
    if value != 0 and pct < 0.01:
        return round(pct, 6)
    return round(pct, 2)


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_numbers(child) for key, child in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

