"""Estimate KV-cache memory by attention consumer."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from llm_bench.estimators.kv.models import MODEL_ESTIMATORS
from llm_bench.estimators.kv.models.kv_model_base import KvModelBase, PrecisionSelection


def estimate_kv_cache_memory_use(
    hf_model_name: str,
    convert_to_precision: str | None = None,
    *,
    context_length: int | None = None,
    batch_size: int = 1,
) -> dict[str, int]:
    """Return estimated KV-cache bytes keyed by attention consumer.

    Args:
        hf_model_name: Case-sensitive HuggingFace model id, local model
            directory, or local ``config.json`` path.
        convert_to_precision: Fireworks ``--convert-to-precision`` value. When
            omitted, the selected model plugin may provide a default.
        context_length: Full-token capacity to estimate. Defaults to the model
            config's max context length.
        batch_size: Number of independent sequences, each with ``context_length`` tokens.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    config = _load_model_config(hf_model_name)
    text_config = _text_config(config)
    model_estimator = _select_model_estimator(text_config)
    effective_context = context_length or _context_length(text_config)
    if effective_context <= 0:
        raise ValueError(f"context_length must be positive, got {effective_context}")

    precision = _precision_selection(
        config=text_config,
        convert_to_precision=convert_to_precision,
        model_estimator=model_estimator,
    )
    if model_estimator is not None:
        return model_estimator.estimate(
            text_config,
            precision,
            context_length=effective_context,
            batch_size=batch_size,
        )

    return _estimate_standard_attention(
        text_config,
        precision=precision,
        context_length=effective_context,
        batch_size=batch_size,
    )


def estimate_kv_cache_memory_use_json(
    hf_model_name: str,
    convert_to_precision: str | None = None,
    *,
    context_length: int | None = None,
    batch_size: int = 1,
) -> str:
    """Return :func:`estimate_kv_cache_memory_use` encoded as JSON."""

    return json.dumps(
        estimate_kv_cache_memory_use(
            hf_model_name=hf_model_name,
            convert_to_precision=convert_to_precision,
            context_length=context_length,
            batch_size=batch_size,
        ),
        sort_keys=True,
    )


def _load_model_config(hf_model_name: str) -> dict[str, Any]:
    path = Path(hf_model_name).expanduser()
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.is_file():
        with config_path.open() as f:
            return json.load(f)

    return _download_hf_config_json(hf_model_name)


def _download_hf_config_json(hf_model_name: str) -> dict[str, Any]:
    config_path = hf_hub_download(repo_id=hf_model_name, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{hf_model_name} config.json did not contain a JSON object")
    return config


def _select_model_estimator(config: dict[str, Any]) -> KvModelBase | None:
    for estimator in MODEL_ESTIMATORS:
        if estimator.matches(config):
            return estimator
    return None


def _estimate_standard_attention(
    config: dict[str, Any],
    *,
    precision: PrecisionSelection,
    context_length: int,
    batch_size: int,
) -> dict[str, int]:
    n_layers = int(config["num_hidden_layers"])

    if "kv_lora_rank" in config and "qk_rope_head_dim" in config:
        per_token_elements = int(config["kv_lora_rank"]) + int(config["qk_rope_head_dim"])
        total = batch_size * n_layers * context_length * per_token_elements * precision.kv_dtype_bytes
        return {"mla": total, "total": total}

    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
    head_dim = int(config.get("head_dim", int(config["hidden_size"]) // num_attention_heads))
    total = batch_size * n_layers * context_length * num_key_value_heads * head_dim * 2 * precision.kv_dtype_bytes
    return {"attention": total, "total": total}


def _precision_selection(
    *,
    config: dict[str, Any],
    convert_to_precision: str | None,
    model_estimator: KvModelBase | None,
) -> PrecisionSelection:
    model_default = None
    if model_estimator is not None:
        model_default = model_estimator.default_convert_to_precision(config)
    effective = convert_to_precision or _default_convert_to_precision(config=config, model_default=model_default)
    base_precision, overrides = _parse_convert_to_precision(effective)

    if model_estimator is not None:
        base_precision, overrides = model_estimator.adjust_precision(base_precision, overrides)

    kv_dtype = overrides.get("kv") or _default_kv_dtype(base_precision=base_precision, config=config)
    kv_dtype = _normalize_dtype_name(kv_dtype)
    return PrecisionSelection(
        base_precision=base_precision,
        overrides=overrides,
        kv_dtype_name=kv_dtype,
        kv_dtype_bytes=_dtype_bytes(kv_dtype),
    )


def _default_convert_to_precision(*, config: dict[str, Any], model_default: str | None) -> str:
    if model_default is not None:
        return model_default

    quantization_config = config.get("quantization_config")
    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method")
        fmt = quantization_config.get("fmt")
        if quant_method == "fp8" and fmt == "e4m3":
            return "fp8"

    return "bf16"


def _parse_convert_to_precision(value: str) -> tuple[str, dict[str, str]]:
    value = value.strip()
    if value.startswith("{"):
        data = json.loads(value)
        name = data.get("name")
        if not isinstance(name, str):
            raise ValueError(f"convert_to_precision JSON must contain a string name: {value}")
        return _normalize_base_precision(name), {}

    base_precision: str | None = None
    overrides: dict[str, str] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" in part:
            key, override_value = part.split("=", 1)
            overrides[key.strip()] = _normalize_dtype_name(override_value.strip())
        elif base_precision is None:
            base_precision = _normalize_base_precision(part)
        else:
            raise ValueError(f"More than one base precision in {value!r}")

    if base_precision is None:
        raise ValueError(f"No base precision in {value!r}")
    return base_precision, overrides


def _default_kv_dtype(*, base_precision: str, config: dict[str, Any]) -> str:
    is_moe = bool(config.get("n_routed_experts") or config.get("num_experts") or config.get("moe"))
    if base_precision in {"bf16", "fp16", "fp32"}:
        return base_precision
    if is_moe and base_precision == "fp8":
        return "bf16"
    if is_moe and base_precision == "fp4":
        return "fp8"
    if base_precision in {"fp8", "fp4"}:
        return "fp8"
    raise ValueError(f"Unsupported base precision {base_precision!r}")


def _normalize_base_precision(value: str) -> str:
    value = _normalize_dtype_name(value)
    aliases = {
        "float8_mm": "fp8",
        "float8_mm_v2": "fp8",
        "float8_kv": "fp8",
        "float8_blockscaled_ll_mm_v2": "fp8",
        "bfloat16_mm": "bf16",
    }
    value = aliases.get(value, value)
    if value not in {"bf16", "fp16", "fp32", "fp8", "fp4"}:
        raise ValueError(f"Unsupported base precision {value!r}")
    return value


def _normalize_dtype_name(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("-", "_")
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8": "fp8",
        "float8_e4m3": "fp8",
        "fp8_cast": "fp8",
        "float4": "fp4",
    }
    return aliases.get(value, value)


def _dtype_bytes(value: str) -> int:
    if value == "fp32":
        return 4
    if value in {"bf16", "fp16"}:
        return 2
    if value == "fp8":
        return 1
    raise ValueError(f"Unsupported byte-sized dtype {value!r}")


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        merged = dict(text_config)
        for key in ("model_type", "quantization_config"):
            if key not in merged and key in config:
                merged[key] = config[key]
        return merged
    return config


def _context_length(config: dict[str, Any]) -> int:
    return int(
        config.get("max_position_embeddings")
        or config.get("max_seq_length")
        or config.get("seq_length")
        or config.get("n_positions")
        or 0
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate KV-cache memory use in bytes.")
    parser.add_argument(
        "--hf-model-name",
        required=True,
        help='Case-sensitive HuggingFace model id (for example "deepseek-ai/DeepSeek-V4-Pro") or local path.',
    )
    parser.add_argument("--convert-to-precision", default=None, help="Fireworks convert-to-precision string.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Capacity to estimate. Defaults to the model config max context.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of independent sequences, each with --context-length tokens.",
    )
    parser.add_argument(
        "-g", "--gib", action="store_true", help="Print values in GiB instead of bytes."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    estimate = estimate_kv_cache_memory_use(
        hf_model_name=args.hf_model_name,
        convert_to_precision=args.convert_to_precision,
        context_length=args.context_length,
        batch_size=args.batch_size,
    )
    if args.gib:
        gib = 1024**3
        estimate = {key: value / gib for key, value in estimate.items()}
    print(json.dumps(estimate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
