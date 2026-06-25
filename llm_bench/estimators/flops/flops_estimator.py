"""Estimate prefill FLOPs by model component."""

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

from llm_bench.estimators.flops.models import MODEL_ESTIMATORS
from llm_bench.estimators.flops.models.flops_model_base import FlopsModelBase


def estimate_prefill_flops(
    hf_model_name: str,
    *,
    context_length: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Return estimated prefill FLOPs keyed by model component.

    Args:
        hf_model_name: Case-sensitive HuggingFace model id, local model
            directory, or local ``config.json`` path.
        context_length: Full-token prompt length to estimate. Defaults to the
            model config's max context length.
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

    if model_estimator is not None:
        estimate = model_estimator.estimate(
            text_config,
            context_length=effective_context,
            batch_size=batch_size,
        )
        return _with_percentages(estimate)

    estimate = _estimate_standard_prefill(
        text_config,
        context_length=effective_context,
        batch_size=batch_size,
    )
    return _with_percentages(estimate)


def estimate_prefill_flops_json(
    hf_model_name: str,
    *,
    context_length: int | None = None,
    batch_size: int = 1,
) -> str:
    """Return :func:`estimate_prefill_flops` encoded as JSON."""

    return json.dumps(
        estimate_prefill_flops(
            hf_model_name=hf_model_name,
            context_length=context_length,
            batch_size=batch_size,
        ),
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


def _select_model_estimator(config: dict[str, Any]) -> FlopsModelBase | None:
    for estimator in MODEL_ESTIMATORS:
        if estimator.matches(config):
            return estimator
    return None


def _estimate_standard_prefill(
    config: dict[str, Any],
    *,
    context_length: int,
    batch_size: int,
) -> dict[str, Any]:
    n_layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))
    total_tokens = batch_size * context_length

    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    attention_projection = n_layers * (
        _linear_flops(total_tokens, hidden_size, q_width + 2 * kv_width)
        + _linear_flops(total_tokens, q_width, hidden_size)
    )
    attention = n_layers * _attention_flops(
        batch_size=batch_size,
        context_length=context_length,
        num_heads=num_attention_heads,
        head_dim=head_dim,
    )

    mlp = _estimate_standard_mlp_flops(
        config=config,
        total_tokens=total_tokens,
        hidden_size=hidden_size,
        n_layers=n_layers,
    )

    total = attention_projection + attention + mlp
    return {
        "attention": {
            "standard": {
                "proj": attention_projection,
                "attn": attention,
                "total": attention_projection + attention,
            },
            "total": attention_projection + attention,
        },
        "mlp": {
            "total": mlp,
        },
        "total": total,
    }


def _estimate_standard_mlp_flops(
    *,
    config: dict[str, Any],
    total_tokens: int,
    hidden_size: int,
    n_layers: int,
) -> int:
    moe_intermediate_size = config.get("moe_intermediate_size")
    if moe_intermediate_size is not None:
        num_experts = int(config.get("n_routed_experts") or config.get("num_experts"))
        experts_per_token = int(config.get("num_experts_per_tok") or config.get("num_experts_per_token"))
        n_shared_experts = int(config.get("n_shared_experts", 0))
        router = _linear_flops(total_tokens, hidden_size, num_experts)
        experts = (experts_per_token + n_shared_experts) * _swiglu_ffn_flops(
            total_tokens,
            hidden_size,
            int(moe_intermediate_size),
        )
        return n_layers * (router + experts)

    intermediate_size = int(
        config.get("intermediate_size")
        or config.get("ffn_dim")
        or config.get("filter_size")
        or 0
    )
    if intermediate_size <= 0:
        return 0
    return n_layers * _swiglu_ffn_flops(total_tokens, hidden_size, intermediate_size)


def _linear_flops(num_rows: int, in_features: int, out_features: int) -> int:
    return 2 * num_rows * in_features * out_features


def _swiglu_ffn_flops(num_tokens: int, hidden_size: int, intermediate_size: int) -> int:
    return 3 * _linear_flops(num_tokens, hidden_size, intermediate_size)


def _attention_flops(
    *,
    batch_size: int,
    context_length: int,
    num_heads: int,
    head_dim: int,
) -> int:
    causal_pairs = context_length * (context_length + 1) // 2
    return 4 * batch_size * causal_pairs * num_heads * head_dim


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
    parser = argparse.ArgumentParser(description="Estimate prefill FLOPs.")
    parser.add_argument(
        "--hf-model-name",
        required=True,
        help='Case-sensitive HuggingFace model id (for example "deepseek-ai/DeepSeek-V4-Pro") or local path.',
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Prompt length to estimate. Defaults to the model config max context.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of independent sequences, each with --context-length tokens.",
    )
    parser.add_argument(
        "-p",
        "--peta",
        action="store_true",
        help="Print values in petaFLOPs instead of raw FLOPs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    estimate = estimate_prefill_flops(
        hf_model_name=args.hf_model_name,
        context_length=args.context_length,
        batch_size=args.batch_size,
    )
    if args.peta:
        peta = 10**15
        estimate = _scale_estimate(estimate, peta)
    print(json.dumps(estimate))
    return 0


def _scale_estimate(estimate: dict[str, Any], factor: int) -> dict[str, Any]:
    scaled: dict[str, Any] = {}
    for key, value in estimate.items():
        if isinstance(value, dict):
            scaled[key] = _scale_estimate(value, factor)
        elif key == "pct":
            scaled[key] = value
        else:
            scaled[key] = value / factor
    return scaled


def _with_percentages(estimate: dict[str, Any]) -> dict[str, Any]:
    total = int(estimate["total"])
    return _add_percentages(estimate, global_total=total, is_root=True)


def _add_percentages(
    estimate: dict[str, Any],
    *,
    global_total: int,
    is_root: bool,
) -> dict[str, Any]:
    ret: dict[str, Any] = {}
    for key, value in estimate.items():
        if isinstance(value, dict):
            ret[key] = _add_percentages(value, global_total=global_total, is_root=False)
        else:
            ret[key] = value
        if key == "total" and not is_root:
            ret["pct"] = round(value / global_total * 100, 2) if global_total else 0.0
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
