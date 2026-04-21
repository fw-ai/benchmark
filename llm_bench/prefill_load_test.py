#!/usr/bin/env python3
"""
Prefill benchmark for Fireworks /v1/completions.

For each (prompt_tokens, cached_tokens) pair, warms the KV cache and sends a batch
of prompts as a single multi-prompt request, reporting server and client durations.

Each prompt is wrapped client-side in the model's chat template (a single user
message with the generation prompt appended) and sent as token ids. The chat
template is split into prefix/suffix token lists so cached_tokens corresponds to
the exact length of the shared head (chat_prefix + shared content) across all
prompts in a batch and the warmup call.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

import requests
import transformers

from tabulate import tabulate

FW_HEADER_PREFIX = "fireworks-"
STAGE_HEADER_KEYS = (
    "tokenizer-queue-duration",
    "tokenizer-duration",
    "prefill-queue-duration",
    "prefill-duration",
    "generation-queue-duration",
)

# NB: don't use power of 2 as we will use multiples of this to generate seq pairs
# and in some cases it will batch max seq len of a model, which is the edge case we don't want to benchmark.
_DEFAULT_MIN_SEQ_LEN = 500


def _load_auto_tokenizer(tokenizer_path: str) -> transformers.PreTrainedTokenizer:
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def resolve_max_seq_len(tokenizer_path: str) -> int:
    config = transformers.AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    # For VLMs (e.g. Kimi K2.5, LLaMA Vision), max_position_embeddings lives
    # under text_config rather than at the top level.
    config = config.get_text_config()
    for name in (
        "max_position_embeddings",
        "model_max_length",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    ):
        v = getattr(config, name, None)
        if isinstance(v, int) and v > 0:
            return v
    raise ValueError("Could not infer max sequence length from config; pass --max-seq-len explicitly.")


def generate_pairs(max_seq_len: int, min_seq_len: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    s = min_seq_len
    while s <= max_seq_len:
        step = s // 8
        for multiplier in [0, 1, 3, 5, 7]:
            c = step * multiplier
            if c <= s:
                pairs.append((s, c))
        s *= 4
    return pairs


def _dataset_path(dataset: str) -> str:
    name = "limericks.txt" if dataset == "limericks" else "code.txt"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def load_chunks(dataset: str) -> list[str]:
    path = _dataset_path(dataset)
    with open(path, "r") as f:
        text = f.read()
    chunks = [p for p in text.split("\n\n") if p.strip()]
    if not chunks:
        raise ValueError(f"No chunks in {path}")
    return chunks


def build_ids_to_length(
    tokenizer: transformers.PreTrainedTokenizer,
    chunks: list[str],
    target_len: int,
) -> list[int]:
    """Token ids from repeated limericks/code chunks, truncated to target_len (prefix kept)."""
    ids: list[int] = []
    i = 0
    while len(ids) < target_len and i < 1_000_000:
        lim = chunks[i % len(chunks)]
        ids.extend(tokenizer.encode(lim + "\n\n"))
        i += 1
    return ids[:target_len]


def build_random_suffix_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    chunks: list[str],
    need_len: int,
    rng: random.Random,
) -> list[int]:
    """Token ids from random limericks/code chunks (same idea as TranslationDataset.__next__)."""
    if need_len <= 0:
        return []
    ids: list[int] = []
    i = 0
    while len(ids) < need_len and i < 1_000_000:
        lim = rng.choice(chunks)
        ids.extend(tokenizer.encode(lim + "\n\n"))
        i += 1
    return ids[:need_len]


def _normalize_ids(obj: Any) -> list[int]:
    """Normalize tokenizer output (list, Encoding, BatchEncoding, tensor) to a plain list[int]."""
    ids = getattr(obj, "ids", None)
    if ids is not None:
        return [int(t) for t in ids]
    input_ids = getattr(obj, "input_ids", None)
    if input_ids is not None:
        seq = (
            input_ids[0]
            if hasattr(input_ids, "__len__") and len(input_ids) and hasattr(input_ids[0], "__iter__")
            else input_ids
        )
        return [int(t) for t in seq]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        flat = tolist()
        if flat and isinstance(flat[0], list):
            flat = flat[0]
        return [int(t) for t in flat]
    return [int(t) for t in obj]


def split_chat_template(
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[list[int], list[int]]:
    """Return (prefix_ids, suffix_ids) for the chat template wrapping a single user message.

    Templates a multi-token sentinel content, locates it in the templated id sequence,
    and returns the surrounding prefix and suffix token lists. Lets us compose
    `prefix + content_ids + suffix` directly without round-tripping through text.
    """
    sentinel = "abcdefghij"
    sentinel_ids = _normalize_ids(tokenizer.encode(sentinel, add_special_tokens=False))
    if not sentinel_ids:
        raise RuntimeError("Sentinel tokenized to empty id list; cannot split chat template.")
    templated = _normalize_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    n = len(sentinel_ids)
    for i in range(len(templated) - n + 1):
        if templated[i : i + n] == sentinel_ids:
            return templated[:i], templated[i + n :]
    raise RuntimeError(f"Could not locate sentinel ids {sentinel_ids} in templated ids {templated}")


def build_pair_ids(
    chat_prefix_ids: list[int],
    chat_suffix_ids: list[int],
    base_ids: list[int],
    tokenizer: transformers.PreTrainedTokenizer,
    chunks: list[str],
    prompt_tokens: int,
    cached_tokens: int,
    rng: random.Random,
) -> list[int]:
    """Build a chat-templated prompt id sequence of exactly `prompt_tokens` tokens.

    Layout: chat_prefix_ids ++ shared_content ++ random_content ++ chat_suffix_ids,
    truncated to `prompt_tokens`. The first `cached_tokens` ids are deterministically
    shared across all prompts in the batch (and across warmup), so the prompt cache
    hits exactly `cached_tokens` tokens.
    """
    chat_overhead = len(chat_prefix_ids) + len(chat_suffix_ids)
    if prompt_tokens < chat_overhead:
        raise ValueError(f"prompt_tokens {prompt_tokens} is smaller than chat overhead {chat_overhead}")
    if cached_tokens > prompt_tokens:
        raise ValueError(f"cached_tokens {cached_tokens} exceeds prompt_tokens {prompt_tokens}")

    content_target = prompt_tokens - chat_overhead
    shared_content_len = max(0, cached_tokens - len(chat_prefix_ids))
    shared_content_len = min(shared_content_len, content_target)

    if shared_content_len > len(base_ids):
        raise ValueError(f"shared_content_len {shared_content_len} exceeds base_ids length {len(base_ids)}")

    shared = base_ids[:shared_content_len]
    need_random = content_target - shared_content_len
    random_ids = build_random_suffix_ids(tokenizer, chunks, need_random, rng)
    content = shared + random_ids
    if len(content) < content_target:
        raise RuntimeError(f"Could not build {content_target} content tokens from dataset (got {len(content)})")

    return chat_prefix_ids + content[:content_target] + chat_suffix_ids


def build_warmup_ids(
    chat_prefix_ids: list[int],
    base_ids: list[int],
    cached_tokens: int,
) -> list[int]:
    """Warmup prompt = first `cached_tokens` of the shared template + content prefix.

    Sending this as a /v1/completions prompt populates the prompt cache up to exactly
    `cached_tokens` tokens, matching the head of every per-pair prompt built above.
    """
    if cached_tokens <= len(chat_prefix_ids):
        return chat_prefix_ids[:cached_tokens]
    extra = cached_tokens - len(chat_prefix_ids)
    if extra > len(base_ids):
        raise ValueError(f"cached_tokens {cached_tokens} exceeds chat_prefix + base_ids capacity")
    return chat_prefix_ids + base_ids[:extra]


def parse_pairs_arg(s: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        pairs.append((int(a.strip()), int(b.strip())))
    if not pairs:
        raise ValueError("No pairs parsed from --seq-pairs")
    return pairs


def get_header(headers: Mapping[str, str], short_key: str) -> Optional[float]:
    full = FW_HEADER_PREFIX + short_key
    v = headers.get(full)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def get_int_header(headers: Mapping[str, str], short_key: str) -> Optional[int]:
    """e.g. short_key 'prompt-tokens' -> fireworks-prompt-tokens."""
    full = FW_HEADER_PREFIX + short_key
    v = headers.get(full)
    if v is None:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def sum_stage_seconds(headers: Mapping[str, str]) -> Optional[float]:
    parts: list[float] = []
    for k in STAGE_HEADER_KEYS:
        v = get_header(headers, k)
        if v is None:
            return None
        parts.append(v)
    return sum(parts)


def prompt_tokens_from_response(data: dict[str, Any], headers: Mapping[str, str]) -> int:
    usage = data.get("usage") or {}
    pt = usage.get("prompt_tokens")
    if isinstance(pt, int):
        return pt
    h = headers.get(FW_HEADER_PREFIX + "prompt-tokens")
    if h is not None:
        return int(h)
    raise ValueError("Could not read prompt token count from usage or fireworks-prompt-tokens header")


def completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/completions"


@dataclass
class PairBenchmarkResult:
    prompt_tokens: int
    cached_tokens: int
    num_prompts: int
    duration: float
    client_duration: float
    mean_fireworks_prompt_tokens: Optional[float]
    mean_fireworks_cached_prompt_tokens: Optional[float]


def post_completion(
    session: requests.Session,
    url: str,
    api_key: str,
    model: Optional[str],
    prompt: str | list[str] | list[int] | list[list[int]],
    max_tokens: int,
    prompt_cache_max_len: int | None = None,
) -> requests.Response:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
        "user": "0",  # NB: in case of DP w/ inline prefill we need to ensure generator stickiness
    }
    if prompt_cache_max_len is not None:
        payload["prompt_cache_max_len"] = prompt_cache_max_len
    if model is not None:
        payload["model"] = model
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return session.post(url, headers=headers, json=payload, timeout=3600)


def run_benchmark(
    tokenizer_path: str,
    model: Optional[str],
    base_url: str,
    api_key: str,
    dataset: str,
    pairs: list[tuple[int, int]],
    min_tokens_to_batch: int,
    max_tokens: int,
    rng_seed: Optional[int],
    retries: int = 3,
    retry_delay: float = 30.0,
) -> list[PairBenchmarkResult]:
    """Per-pair prefill benchmark with multi-prompt batching."""
    tokenizer = _load_auto_tokenizer(tokenizer_path)
    max_seq = max(prompt_tokens for prompt_tokens, _ in pairs)
    chunks = load_chunks(dataset)

    chat_prefix_ids, chat_suffix_ids = split_chat_template(tokenizer)
    chat_overhead = len(chat_prefix_ids) + len(chat_suffix_ids)
    logger.info(
        "Chat template: %d prefix tokens + %d suffix tokens = %d overhead",
        len(chat_prefix_ids),
        len(chat_suffix_ids),
        chat_overhead,
    )

    base_ids = build_ids_to_length(tokenizer, chunks, max_seq)
    if len(base_ids) != max_seq:
        raise RuntimeError(f"Internal error: base_ids length {len(base_ids)} != max_seq {max_seq}")

    url = completions_url(base_url)
    session = requests.Session()

    def do_warmup(cached_tokens: int) -> None:
        warmup_ids = build_warmup_ids(chat_prefix_ids, base_ids, cached_tokens)
        w = post_completion(session, url, api_key, model, warmup_ids, max_tokens)
        if w.status_code != 200:
            raise RuntimeError(f"Warmup failed HTTP {w.status_code}: {w.text[:500]}")

    results: list[PairBenchmarkResult] = []
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    for prompt_tokens, cached_tokens in pairs:
        if not (0 <= cached_tokens <= prompt_tokens):
            raise ValueError(f"Invalid pair ({prompt_tokens}, {cached_tokens}): need 0 <= cached <= prompt")
        if prompt_tokens < chat_overhead:
            raise ValueError(
                f"prompt_tokens {prompt_tokens} smaller than chat template overhead {chat_overhead}; "
                "skip this pair or raise the prompt length."
            )

        for attempt in range(1, retries + 1):
            try:
                if cached_tokens > 0:
                    do_warmup(cached_tokens)

                prompt_token_ids: list[list[int]] = []
                uncached_tokens = prompt_tokens - cached_tokens
                batch_size = max(1, min_tokens_to_batch // uncached_tokens)
                for _ in range(batch_size):
                    pair_ids = build_pair_ids(
                        chat_prefix_ids,
                        chat_suffix_ids,
                        base_ids,
                        tokenizer,
                        chunks,
                        prompt_tokens,
                        cached_tokens,
                        rng,
                    )
                    if len(pair_ids) != prompt_tokens:
                        raise RuntimeError(f"pair_ids length {len(pair_ids)} != prompt_tokens {prompt_tokens}")
                    prompt_token_ids.append(pair_ids)

                logger.info(
                    "Pair (%d, %d): sending %d prompts in one request",
                    prompt_tokens,
                    cached_tokens,
                    len(prompt_token_ids),
                )
                wall_start = time.perf_counter()
                r = post_completion(
                    session, url, api_key, model, prompt_token_ids, max_tokens, prompt_cache_max_len=cached_tokens
                )
                if r.status_code != 200:
                    raise RuntimeError(f"Request failed HTTP {r.status_code}: {r.text[:500]}")
                wall = time.perf_counter() - wall_start

                fwp = get_int_header(r.headers, "prompt-tokens")
                fwc = get_int_header(r.headers, "cached-prompt-tokens")

                server_processing = get_header(r.headers, "server-processing-time")
                st = server_processing if server_processing is not None else sum_stage_seconds(r.headers)
                if st is None:
                    raise RuntimeError(
                        "Missing Fireworks timing headers (need dedicated deployment?). "
                        f"Got keys: {list(r.headers.keys())[:30]}"
                    )

                logger.info(
                    "  -> server prompt-tokens=%s, cached=%s, server-duration=%.6fs, client=%.2fs",
                    fwp,
                    fwc,
                    st,
                    wall,
                )

                results.append(
                    PairBenchmarkResult(
                        prompt_tokens=prompt_tokens,
                        cached_tokens=cached_tokens,
                        num_prompts=len(prompt_token_ids),
                        duration=st,
                        client_duration=wall,
                        mean_fireworks_prompt_tokens=fwp,
                        mean_fireworks_cached_prompt_tokens=fwc,
                    )
                )
                break
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        "Pair (%d, %d) failed (attempt %d/%d): %s. Retrying in %.0fs ...",
                        prompt_tokens,
                        cached_tokens,
                        attempt,
                        retries,
                        e,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    raise

    return results


def format_table(rows: list[PairBenchmarkResult]) -> str:
    data: list[list[Any]] = []
    for row in rows:
        data.append(
            [
                row.prompt_tokens,
                row.cached_tokens,
                row.num_prompts,
                row.duration,
                row.client_duration,
                row.mean_fireworks_prompt_tokens if row.mean_fireworks_prompt_tokens is not None else "",
                row.mean_fireworks_cached_prompt_tokens if row.mean_fireworks_cached_prompt_tokens is not None else "",
            ]
        )
    return tabulate(
        data,
        headers=[
            "prompt tokens",
            "cached tokens",
            "num prompts",
            "server duration",
            "client duration",
            "server prompt tokens",
            "server cached tokens",
        ],
        tablefmt="pipe",
        floatfmt=".6f",
        colalign=("right",) * 7,
    )


def format_csv(rows: list[PairBenchmarkResult]) -> str:
    headers = [
        "prompt_tokens",
        "cached_tokens",
        "num_prompts",
        "server_duration",
        "client_duration",
        "server_prompt_tokens",
        "server_cached_tokens",
    ]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(
            ",".join(
                str(v)
                for v in [
                    row.prompt_tokens,
                    row.cached_tokens,
                    row.num_prompts,
                    f"{row.duration:.6f}",
                    f"{row.client_duration:.6f}",
                    row.mean_fireworks_prompt_tokens if row.mean_fireworks_prompt_tokens is not None else "",
                    (
                        row.mean_fireworks_cached_prompt_tokens
                        if row.mean_fireworks_cached_prompt_tokens is not None
                        else ""
                    ),
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="Prefill cache + mTPM benchmark (Fireworks completions).")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace model directory (tokenizer + config), same as load_test --tokenizer.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model / deployment id (optional; omit for servers with a single default model).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("FIREWORKS_BASE_URL", "http://localhost"),
        help="API base URL (default: http://localhost, or FIREWORKS_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY") or os.environ.get("FIREWORKS_API_KEY"),
        help="Bearer token (default: API_KEY or FIREWORKS_API_KEY).",
    )
    parser.add_argument("--dataset", choices=("limericks", "code"), default="limericks")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max sequence length (default: read from HF config).",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=_DEFAULT_MIN_SEQ_LEN,
        help=f"Min sequence length for auto-generated pairs (default: {_DEFAULT_MIN_SEQ_LEN})",
    )
    parser.add_argument(
        "-s",
        "--seq-pairs",
        default=None,
        help="Comma-separated prompt_tokens:cached_tokens, e.g. 4096:2048,8192:4096. "
        "If omitted, auto-generated from --max-seq-len.",
    )
    parser.add_argument(
        "--min-tokens-to-batch",
        type=int,
        default=131072,
        help="Minimum total prompt tokens to accumulate per batch (default 64K).",
    )
    parser.add_argument("--max-tokens", type=int, default=0, help="max_tokens for completions (default 0).")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for random limericks/code draws in non-cached suffixes (default: 0).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format: pipe markdown table (default) or CSV.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts per (prompt_tokens, cached_tokens) pair (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=30.0,
        help="Seconds to sleep between retries (default: 30).",
    )

    args = parser.parse_args()
    if not args.api_key:
        parser.error("Pass --api-key or set API_KEY / FIREWORKS_API_KEY")

    max_seq_len = args.max_seq_len
    try:
        hf_max_seq_len = resolve_max_seq_len(args.tokenizer)
        logger.info("Resolved max_seq_len=%s from HF config", hf_max_seq_len)
    except ValueError:
        hf_max_seq_len = None
        if max_seq_len is None:
            parser.error("Could not infer max sequence length from config; pass --max-seq-len explicitly.")
    if max_seq_len is None:
        max_seq_len = hf_max_seq_len
    elif hf_max_seq_len is not None:
        max_seq_len = min(max_seq_len, hf_max_seq_len)

    if args.seq_pairs is not None:
        pairs = parse_pairs_arg(args.seq_pairs)
    else:
        pairs = generate_pairs(max_seq_len, min_seq_len=args.min_seq_len)
        logger.info("Auto-generated %d pairs: %s", len(pairs), pairs)

    rows = run_benchmark(
        tokenizer_path=args.tokenizer,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        dataset=args.dataset,
        pairs=pairs,
        min_tokens_to_batch=args.min_tokens_to_batch,
        max_tokens=args.max_tokens,
        rng_seed=args.seed,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    if args.format == "csv":
        print(format_csv(rows))
    else:
        print(format_table(rows))


if __name__ == "__main__":
    main()
