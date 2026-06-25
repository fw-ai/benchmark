"""Estimate decode byte movement by memory fabric."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from llm_bench.estimators.bandwidth.modes.deepseek4 import (
    DEFAULT_HF_MODEL_NAME,
    estimate_deepseek4_bandwidth,
)


def estimate_bandwidth(
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
    return estimate_deepseek4_bandwidth(
        hf_model_name=hf_model_name,
        context_length=context_length,
        batch_size=batch_size,
        n_sequences=n_sequences,
        world_size=world_size,
        attn_sharding=attn_sharding,
        moe_sharding=moe_sharding,
        convert_to_precision=convert_to_precision,
        activation_dtype=activation_dtype,
    )


def estimate_bandwidth_json(
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
) -> str:
    return json.dumps(
        estimate_bandwidth(
            hf_model_name=hf_model_name,
            context_length=context_length,
            batch_size=batch_size,
            n_sequences=n_sequences,
            world_size=world_size,
            attn_sharding=attn_sharding,
            moe_sharding=moe_sharding,
            convert_to_precision=convert_to_precision,
            activation_dtype=activation_dtype,
        )
    )


def _scale_estimate_preserving_pct(value: Any, factor: float, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _scale_estimate_preserving_pct(child, factor, key=child_key)
            for child_key, child in value.items()
        }
    if isinstance(value, (int, float)):
        if key == "pct":
            return value
        return value / factor
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model-name",
        default=DEFAULT_HF_MODEL_NAME,
        help="Case-sensitive HuggingFace model id or local model/config path.",
    )
    parser.add_argument("--context-length", type=int, required=True, help="Decode context length to model.")
    parser.add_argument("--batch-size", type=int, required=True, help="Sequences processed per decode round.")
    parser.add_argument("--n-sequences", type=int, required=True, help="Total sequences across all rounds.")
    parser.add_argument("--world-size", type=int, required=True, help="Number of GPUs/ranks serving the model.")
    parser.add_argument(
        "--attn-sharding",
        required=True,
        choices=("dp",),
        help="Attention sharding mode. Currently only dp is supported.",
    )
    parser.add_argument(
        "--moe-sharding",
        required=True,
        choices=("ep",),
        help="MoE sharding mode. Currently only ep is supported.",
    )
    parser.add_argument(
        "--convert-to-precision",
        default=None,
        help="Fireworks convert-to-precision string. Defaults to the DeepSeek-V4 serving recipe.",
    )
    parser.add_argument("--activation-dtype", default="bf16", help="Activation dtype for activation/EP traffic.")
    parser.add_argument("-g", "--gib", action="store_true", help="Print byte values in GiB.")
    parser.add_argument("-t", "--tib", action="store_true", help="Print byte values in TiB.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    estimate = estimate_bandwidth(
        hf_model_name=args.hf_model_name,
        context_length=args.context_length,
        batch_size=args.batch_size,
        n_sequences=args.n_sequences,
        world_size=args.world_size,
        attn_sharding=args.attn_sharding,
        moe_sharding=args.moe_sharding,
        convert_to_precision=args.convert_to_precision,
        activation_dtype=args.activation_dtype,
    )
    if args.gib and args.tib:
        raise ValueError("Use at most one of --gib or --tib")
    if args.gib:
        estimate = _scale_estimate_preserving_pct(estimate, 1024**3)
    if args.tib:
        estimate = _scale_estimate_preserving_pct(estimate, 1024**4)
    print(json.dumps(estimate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

