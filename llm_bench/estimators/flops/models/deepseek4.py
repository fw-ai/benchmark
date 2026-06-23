"""DeepSeek-V4 prefill FLOPs estimator.

The formulas mirror the standalone attention, compressor, indexer, mHC, and
MoE layout in ``fireworks.models.deepseek4`` without importing Fireworks at
runtime. Multiply-adds are counted as two FLOPs.
"""

from __future__ import annotations

from typing import Any

from llm_bench.estimators.flops.models.flops_model_base import FlopsModelBase


class Deepseek4FlopsModel(FlopsModelBase):
    name = "deepseek_v4"

    def matches(self, config: dict[str, Any]) -> bool:
        model_type = str(config.get("model_type") or "")
        return model_type == "deepseek_v4" or "compress_ratios" in config and "index_head_dim" in config

    def estimate(
        self,
        config: dict[str, Any],
        *,
        context_length: int,
        batch_size: int,
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
        index_topk = int(config.get("index_topk", 512))
        sliding_window = int(config.get("sliding_window", 128))
        total_tokens = batch_size * context_length

        proj_per_layer = _attention_projection_flops(
            total_tokens=total_tokens,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
        )
        swa_attn_per_layer = _attention_flops_from_pairs(
            batch_size=batch_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_pairs=_sliding_window_pairs(context_length, sliding_window),
        )
        c4_compressed_attn_per_layer = _attention_flops_from_pairs(
            batch_size=batch_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_pairs=_compressed_pairs(context_length, 4, topk=index_topk),
        )
        c128_compressed_attn_per_layer = _attention_flops_from_pairs(
            batch_size=batch_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_pairs=_compressed_pairs(context_length, 128),
        )

        c128_compressor_per_layer = _compressor_flops(
            total_tokens=total_tokens,
            hidden_size=hidden_size,
            output_dim=head_dim,
            compress_ratio=128,
        )
        c4_attention_compressor_per_layer = _compressor_flops(
            total_tokens=total_tokens,
            hidden_size=hidden_size,
            output_dim=head_dim,
            compress_ratio=4,
        )
        c4_indexer_compressor_per_layer = _compressor_flops(
            total_tokens=total_tokens,
            hidden_size=hidden_size,
            output_dim=index_head_dim,
            compress_ratio=4,
        )
        indexer_per_layer = _indexer_components(
            total_tokens=total_tokens,
            batch_size=batch_size,
            context_length=context_length,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
        )
        moe_per_layer = _moe_components(
            config=config,
            total_tokens=total_tokens,
            hidden_size=hidden_size,
        )
        hyper_connection_per_layer = _hyper_connection_components(
            total_tokens=total_tokens,
            hidden_size=hidden_size,
            hc_mult=int(config.get("hc_mult", 1)),
        )

        c4a = {
            "proj": c4_layers * proj_per_layer,
            "compress_attn": c4_layers * c4_attention_compressor_per_layer,
            "compress_indexer": c4_layers * c4_indexer_compressor_per_layer,
            "indexer_proj": c4_layers * indexer_per_layer["proj"],
            "indexer_score": c4_layers * indexer_per_layer["score"],
            "swa_attn": c4_layers * swa_attn_per_layer,
            "c4a_attn": c4_layers * c4_compressed_attn_per_layer,
        }
        c4a["total"] = (
            c4a["proj"]
            + c4a["swa_attn"]
            + c4a["c4a_attn"]
            + c4a["compress_attn"]
            + c4a["compress_indexer"]
            + c4a["indexer_proj"]
            + c4a["indexer_score"]
        )

        c128a = {
            "proj": c128_layers * proj_per_layer,
            "compress_attn": c128_layers * c128_compressor_per_layer,
            "swa_attn": c128_layers * swa_attn_per_layer,
            "c128_attn": c128_layers * c128_compressed_attn_per_layer,
        }
        c128a["total"] = (
            c128a["proj"]
            + c128a["swa_attn"]
            + c128a["c128_attn"]
            + c128a["compress_attn"]
        )

        attention = {
            "c128a": c128a,
            "c4a": c4a,
        }
        attention["total"] = c4a["total"] + c128a["total"]

        moe = {
            "router": n_layers * moe_per_layer["router"],
            "experts": n_layers * moe_per_layer["experts"],
        }
        moe["total"] = moe["router"] + moe["experts"]

        hyper_connection = {
            "pre": n_layers * hyper_connection_per_layer["pre"],
            "post": n_layers * hyper_connection_per_layer["post"],
        }
        hyper_connection["total"] = hyper_connection["pre"] + hyper_connection["post"]

        total = attention["total"] + moe["total"] + hyper_connection["total"]
        return {
            "attention": attention,
            "moe": moe,
            "hyper_connection": hyper_connection,
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


def _attention_projection_flops(
    *,
    total_tokens: int,
    hidden_size: int,
    head_dim: int,
    num_attention_heads: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
) -> int:
    q_width = num_attention_heads * head_dim
    fused_wqa_wkv = _linear_flops(total_tokens, hidden_size, q_lora_rank + head_dim)
    wq_b = _linear_flops(total_tokens, q_lora_rank, q_width)
    wo_a = 2 * total_tokens * q_width * o_lora_rank
    wo_b = _linear_flops(total_tokens, o_groups * o_lora_rank, hidden_size)
    return fused_wqa_wkv + wq_b + wo_a + wo_b


def _attention_flops_from_pairs(
    *,
    batch_size: int,
    num_attention_heads: int,
    head_dim: int,
    attention_pairs: int,
) -> int:
    return 4 * batch_size * attention_pairs * num_attention_heads * head_dim


def _compressor_flops(
    *,
    total_tokens: int,
    hidden_size: int,
    output_dim: int,
    compress_ratio: int,
) -> int:
    coefficient = 2 if compress_ratio == 4 else 1
    return 2 * _linear_flops(total_tokens, hidden_size, coefficient * output_dim)


def _indexer_components(
    *,
    total_tokens: int,
    batch_size: int,
    context_length: int,
    hidden_size: int,
    q_lora_rank: int,
    index_n_heads: int,
    index_head_dim: int,
) -> int:
    q_projection = _linear_flops(total_tokens, q_lora_rank, index_n_heads * index_head_dim)
    weights_projection = _linear_flops(total_tokens, hidden_size, index_n_heads)
    score_pairs = _compressed_pairs(context_length, 4)
    scores = 2 * batch_size * score_pairs * index_n_heads * index_head_dim
    return {
        "proj": q_projection + weights_projection,
        "score": scores,
    }


def _moe_components(
    *,
    config: dict[str, Any],
    total_tokens: int,
    hidden_size: int,
) -> int:
    moe_intermediate_size = int(config["moe_intermediate_size"])
    n_routed_experts = int(config["n_routed_experts"])
    num_experts_per_tok = int(config["num_experts_per_tok"])
    n_shared_experts = int(config.get("n_shared_experts", 0))

    router = _linear_flops(total_tokens, hidden_size, n_routed_experts)
    experts = (num_experts_per_tok + n_shared_experts) * _swiglu_ffn_flops(
        total_tokens,
        hidden_size,
        moe_intermediate_size,
    )
    return {
        "router": router,
        "experts": experts,
    }


def _hyper_connection_components(*, total_tokens: int, hidden_size: int, hc_mult: int) -> dict[str, int]:
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_pre = _linear_flops(total_tokens, hc_mult * hidden_size, hc_mult3)
    hc_post = 2 * total_tokens * hidden_size * hc_mult * hc_mult
    return {
        "pre": 2 * hc_pre,
        "post": 2 * hc_post,
    }


def _linear_flops(num_rows: int, in_features: int, out_features: int) -> int:
    return 2 * num_rows * in_features * out_features


def _swiglu_ffn_flops(num_tokens: int, hidden_size: int, intermediate_size: int) -> int:
    return 3 * _linear_flops(num_tokens, hidden_size, intermediate_size)


def _sliding_window_pairs(context_length: int, window_size: int) -> int:
    if context_length <= window_size:
        return context_length * (context_length + 1) // 2
    return window_size * (window_size + 1) // 2 + (context_length - window_size) * window_size


def _compressed_pairs(context_length: int, compress_ratio: int, topk: int | None = None) -> int:
    if topk is None:
        return _floor_div_sum(context_length, compress_ratio)
    if topk <= 0:
        return 0

    cap_start = topk * compress_ratio
    if context_length < cap_start:
        return _floor_div_sum(context_length, compress_ratio)
    return _floor_div_sum(cap_start - 1, compress_ratio) + topk * (context_length - cap_start + 1)


def _floor_div_sum(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return denominator * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)
