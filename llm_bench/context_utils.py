from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import transformers
from huggingface_hub import hf_hub_download

MAX_SANE_CONTEXT_LENGTH = 1 << 24

# Fireworks `/v1/completions` copies a non-empty list[int] prompt directly into
# TokenizationOutput; it does not add BOS, EOS, or chat-template tokens. The
# benchmark applies the chat template before sending those IDs. RequestAdmitter
# then reserves one token for speculative lookahead:
#   max_gen_tokens = model_max_seq_len - prompt_tokens - 1
# and rejects when max_gen_tokens <= 0, including max_tokens=0 requests.
# Keep this named for that server contract rather than treating -1/-2 as
# unexplained safety margins.
FIREWORKS_SPECULATIVE_LOOKAHEAD_TOKENS = 1

_CONTEXT_LENGTH_FIELDS = (
    "max_position_embeddings",
    "model_max_length",
    "max_sequence_length",
    "seq_length",
    "n_positions",
)
_NESTED_CONFIG_FIELDS = (
    "text_config",
    "language_config",
    "llm_config",
    "model_config",
)


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _config_candidates(config: Any) -> list[Any]:
    candidates: list[Any] = []
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            candidates.append(get_text_config())
        except Exception:
            pass
    candidates.append(config)

    result: list[Any] = []
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        result.append(candidate)
        for field in _NESTED_CONFIG_FIELDS:
            nested = _config_value(candidate, field)
            if nested is not None:
                candidates.append(nested)
    return result


def _context_length_from_config(config: Any) -> tuple[int | None, list[str]]:
    rejected: list[str] = []
    for candidate in _config_candidates(config):
        for field in _CONTEXT_LENGTH_FIELDS:
            value = _config_value(candidate, field)
            if value is None:
                continue
            if type(value) is int and 0 < value <= MAX_SANE_CONTEXT_LENGTH:
                return value, rejected
            rejected.append(f"{field}={value!r}")
    return None, rejected


def tokenizer_asset_path(tokenizer_path: str, filename: str) -> str:
    if os.path.isdir(tokenizer_path):
        path = os.path.join(tokenizer_path, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {filename} in local tokenizer directory {tokenizer_path}")
        return path
    return hf_hub_download(repo_id=tokenizer_path, filename=filename)


def read_config_json(tokenizer_path: str) -> dict[str, Any]:
    with open(tokenizer_asset_path(tokenizer_path, "config.json")) as config_file:
        value = json.load(config_file)
    if not isinstance(value, dict):
        raise ValueError("config.json must contain a JSON object")
    return value


def resolve_max_seq_len(tokenizer_path: str) -> int:
    """Resolve a finite model context length from local or Hugging Face config."""
    errors: list[str] = []
    rejected: list[str] = []

    try:
        config = transformers.AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as error:
        errors.append(f"AutoConfig: {error}")
    else:
        value, invalid = _context_length_from_config(config)
        rejected.extend(invalid)
        if value is not None:
            return value

    try:
        raw_config = read_config_json(tokenizer_path)
    except Exception as error:
        errors.append(f"config.json: {error}")
    else:
        value, invalid = _context_length_from_config(raw_config)
        rejected.extend(invalid)
        if value is not None:
            return value

    details = errors + ([f"rejected values: {', '.join(dict.fromkeys(rejected))}"] if rejected else [])
    suffix = f" ({'; '.join(details)})" if details else ""
    raise ValueError(
        "Could not infer a finite max sequence length from config; " f"pass --max-seq-len explicitly.{suffix}"
    )


def generate_geometric_lengths(min_seq_len: int, max_seq_len: int, factor: int) -> list[int]:
    """Return geometric points plus the exact maximum, without duplicates."""
    if min_seq_len < 1:
        raise ValueError("min_seq_len must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if min_seq_len > max_seq_len:
        raise ValueError("min_seq_len must not exceed max_seq_len")
    if factor <= 1:
        raise ValueError("factor must be greater than one")

    lengths: list[int] = []
    seq_len = min_seq_len
    while seq_len <= max_seq_len:
        lengths.append(seq_len)
        seq_len *= factor
    if max_seq_len not in lengths:
        lengths.append(max_seq_len)
    return lengths


def generation_sequence_limit(model_max_seq_len: int) -> int:
    """Maximum prompt-plus-requested-output length admitted by completions."""
    if model_max_seq_len <= FIREWORKS_SPECULATIVE_LOOKAHEAD_TOKENS:
        raise ValueError("model max sequence length is too small for generation")
    return model_max_seq_len - FIREWORKS_SPECULATIVE_LOOKAHEAD_TOKENS


def prefill_prompt_limit(model_max_seq_len: int, max_tokens: int = 0) -> int:
    """Maximum prompt length admitted for a prefill benchmark request."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    required_generation_capacity = max(1, max_tokens)
    prompt_limit = model_max_seq_len - FIREWORKS_SPECULATIVE_LOOKAHEAD_TOKENS - required_generation_capacity
    if prompt_limit < 1:
        raise ValueError("model max sequence length is too small for prefill")
    return prompt_limit
