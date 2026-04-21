#!/usr/bin/env python3
"""
Generation (decode) latency benchmark for Fireworks /v1/completions.

For each (seq_len, batch_size) pair, builds a single user message wrapped in
the model's chat template (applied client-side via the HF tokenizer), sends
the resulting token-id prompt with n=batch_size and max_tokens output tokens,
then reports per-forward-pass generation latency derived from
fireworks-generation-duration and the number of target-model forward passes
(speculation-aware).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

import requests
import transformers
from tabulate import tabulate

FW_HEADER_PREFIX = "fireworks-"

_FAST_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]

# NB: don't use power of 2 as we will use multiples of this to generate seq pairs
# and in some cases it will batch max seq len of a model, which is the edge case we don't want to benchmark.
_DEFAULT_MIN_SEQ_LEN = 1000


def get_profile_batch_sizes(max_batch_size: int) -> list[int]:
    r = [b for b in _FAST_BATCH_SIZES if b <= max_batch_size]
    if not r:
        return [max_batch_size]

    step = 4
    b = r[-1] + step
    while b <= max_batch_size:
        r.append(b)
        if (b & (b - 1)) == 0:
            if 32 <= b < 128:
                step = 16
            else:
                step = b // 2
        b += step

    if r[-1] != max_batch_size:
        r.append(max_batch_size)

    return r


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


def generate_seq_lens(min_seq_len: int, max_seq_len: int) -> list[int]:
    lens: list[int] = []
    s = min_seq_len
    while s <= max_seq_len:
        lens.append(s)
        s *= 2
    return lens


def _load_auto_tokenizer(tokenizer_path: str) -> transformers.PreTrainedTokenizer:
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


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


_DATASET_SUFFIXES = {
    "limericks": "\n\nTranslate the limericks above to Spanish.",
    "code": "\n\nTranslate the code above to C++.",
}


def build_chunk_texts_to_length(
    tokenizer: transformers.PreTrainedTokenizer,
    chunks: list[str],
    target_len: int,
) -> list[str]:
    """Cycle chunks (each followed by `\\n\\n`) until total tokens reach `target_len`.

    Returns one chunk text per element so callers can slice at chunk granularity
    (never cutting a chunk mid-text).
    """
    out: list[str] = []
    total = 0
    i = 0
    while total < target_len and i < 1_000_000:
        text = chunks[i % len(chunks)] + "\n\n"
        out.append(text)
        total += len(tokenizer.encode(text))
        i += 1
    return out


def apply_chat_template_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    content_text: str,
) -> list[int]:
    """Apply the model's chat template to a single user message and return token ids."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": content_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return _normalize_ids(out)


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


def build_chat_prompt_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    suffix_text: str,
    chunk_texts: list[str],
    target_len: int,
) -> list[int]:
    """Build chat-templated prompt ids of at most `target_len` tokens, chunk-aligned.

    Bisects on chunk count to find the largest k such that the chat-templated
    tokenization of `chunks[:k] + suffix` stays within `target_len`. This is
    exact (no token-count estimation drift from chat template wrapping) at the
    cost of O(log n) chat template tokenizations.
    """
    if not chunk_texts:
        raise ValueError("no chunks provided")

    def length_for(k: int) -> int:
        return len(apply_chat_template_ids(tokenizer, "".join(chunk_texts[:k]) + suffix_text))

    if length_for(1) > target_len:
        raise ValueError(f"target_len={target_len} too small to fit even one chunk")

    n = len(chunk_texts)
    if length_for(n) <= target_len:
        return apply_chat_template_ids(tokenizer, "".join(chunk_texts) + suffix_text)

    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if length_for(mid) <= target_len:
            lo = mid
        else:
            hi = mid - 1
    return apply_chat_template_ids(tokenizer, "".join(chunk_texts[:lo]) + suffix_text)


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
    full = FW_HEADER_PREFIX + short_key
    v = headers.get(full)
    if v is None:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/completions"


def parse_pairs_arg(s: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        pairs.append((int(a.strip()), int(b.strip())))
    if not pairs:
        raise ValueError("No pairs parsed from --seq-batch-pairs")
    return pairs


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def post_completion(
    session: requests.Session,
    url: str,
    api_key: str,
    model: Optional[str],
    prompt: list[int],
    max_tokens: int,
    n: int,
    temperature: Optional[float] = None,
    user: Optional[str] = None,
) -> requests.Response:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if model is not None:
        payload["model"] = model
    if user is not None:
        payload["user"] = user
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return session.post(url, headers=headers, json=payload, timeout=3600)


def parse_num_forward_passes(headers: Mapping[str, str], batch_size: int, completion_tokens: int) -> int:
    """Derive the number of target-model forward passes during generation.

    With speculation the acceptance header gives the total per-sequence
    verification steps; dividing by batch_size yields forward passes.
    Without speculation every forward pass produces one token per sequence,
    so forward passes = completion_tokens / batch_size.
    """
    spec_acceptance = headers.get(FW_HEADER_PREFIX + "speculation-acceptance")
    if spec_acceptance:
        verification_steps = int(spec_acceptance.split(",")[0].split("/")[1])
        return verification_steps // batch_size
    return completion_tokens // batch_size


@dataclass
class GenBenchmarkResult:
    seq_len: int
    batch_size: int
    max_tokens: int
    generation_duration: float
    latency_per_forward: float
    client_duration: float


def _run_pair_n_mode(
    session: requests.Session,
    url: str,
    api_key: str,
    model: Optional[str],
    prompt_ids: list[int],
    max_tokens: int,
    seq_len: int,
    batch_size: int,
    temperature: Optional[float],
) -> GenBenchmarkResult:
    """Single request with n=batch_size."""
    logger.info(
        "Pair (seq_len=%d, batch_size=%d): n=%d, max_tokens=%d",
        seq_len,
        batch_size,
        batch_size,
        max_tokens,
    )

    wall_start = time.perf_counter()
    r = post_completion(
        session,
        url,
        api_key,
        model,
        prompt_ids,
        max_tokens=max_tokens,
        n=batch_size,
        temperature=temperature,
    )
    wall = time.perf_counter() - wall_start

    if r.status_code != 200:
        raise RuntimeError(f"Request failed HTTP {r.status_code}: {r.text[:500]}")

    gen_dur = get_header(r.headers, "generation-duration")
    if gen_dur is None:
        raise RuntimeError(
            "Missing fireworks-generation-duration header (need dedicated deployment?). "
            f"Got keys: {[k for k in r.headers.keys() if 'fireworks' in k.lower()]}"
        )

    resp_json = r.json()
    completion_tokens = resp_json.get("usage", {}).get("completion_tokens", max_tokens * batch_size)
    num_fwd = parse_num_forward_passes(r.headers, batch_size, completion_tokens)

    fwp = get_int_header(r.headers, "prompt-tokens")
    logger.info(
        "  -> server prompt-tokens=%s, generation-duration=%.6fs, forward_passes=%s, client=%.2fs",
        fwp,
        gen_dur,
        num_fwd,
        wall,
    )

    return GenBenchmarkResult(
        seq_len=seq_len,
        batch_size=batch_size,
        max_tokens=max_tokens,
        generation_duration=gen_dur,
        latency_per_forward=gen_dur / num_fwd,
        client_duration=wall,
    )


def _generate_users(seed: int, count: int) -> list[str]:
    """Generate `count` deterministic user ids (random ints as strings) from `seed`."""
    rng = random.Random(seed)
    return [str(rng.randint(0, 2**63 - 1)) for _ in range(count)]


def _run_pair_separate_mode(
    url: str,
    api_key: str,
    model: Optional[str],
    prompt_ids: list[int],
    max_tokens: int,
    seq_len: int,
    batch_size: int,
    temperature: Optional[float],
    users: list[str],
) -> GenBenchmarkResult:
    """Send batch_size concurrent requests each with n=1."""
    logger.info(
        "Pair (seq_len=%d, batch_size=%d): %d separate requests, max_tokens=%d",
        seq_len,
        batch_size,
        batch_size,
        max_tokens,
    )
    assert len(users) == batch_size, f"expected {batch_size} users, got {len(users)}"

    def _single_request(user: str) -> requests.Response:
        s = requests.Session()
        return post_completion(
            s,
            url,
            api_key,
            model,
            prompt_ids,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            user=user,
        )

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(_single_request, u) for u in users]
        responses = []
        for fut in as_completed(futures):
            responses.append(fut.result())
    wall = time.perf_counter() - wall_start

    gen_durs: list[float] = []
    fwd_counts: list[int] = []
    for i, r in enumerate(responses):
        if r.status_code != 200:
            raise RuntimeError(f"Request {i} failed HTTP {r.status_code}: {r.text[:500]}")
        gd = get_header(r.headers, "generation-duration")
        if gd is None:
            raise RuntimeError(
                "Missing fireworks-generation-duration header (need dedicated deployment?). "
                f"Got keys: {[k for k in r.headers.keys() if 'fireworks' in k.lower()]}"
            )
        gen_durs.append(gd)
        resp_json = r.json()
        ct = resp_json.get("usage", {}).get("completion_tokens", max_tokens)
        fwd_counts.append(parse_num_forward_passes(r.headers, batch_size=1, completion_tokens=ct))

    avg_gen_dur = sum(gen_durs) / len(gen_durs)
    avg_fwd = sum(fwd_counts) / len(fwd_counts)

    logger.info(
        "  -> gen_dur avg=%.6fs  min=%.6fs  max=%.6fs, fwd_passes avg=%.0f, client=%.2fs",
        avg_gen_dur,
        min(gen_durs),
        max(gen_durs),
        avg_fwd,
        wall,
    )

    return GenBenchmarkResult(
        seq_len=seq_len,
        batch_size=batch_size,
        max_tokens=max_tokens,
        generation_duration=avg_gen_dur,
        latency_per_forward=avg_gen_dur / avg_fwd,
        client_duration=wall,
    )


def _warmup_seq_len(
    *,
    url: str,
    api_key: str,
    model: Optional[str],
    prompt_ids: list[int],
    seq_len: int,
    concurrency: int,
    temperature: Optional[float],
    retries: int,
    retry_delay: float,
    users: Optional[list[str]] = None,
) -> None:
    """Issue `concurrency` warmup completions in parallel for the same prompt."""
    if users is not None:
        assert len(users) == concurrency, f"expected {concurrency} users, got {len(users)}"

    def _single(user: Optional[str]) -> requests.Response:
        return post_completion(
            requests.Session(),
            url,
            api_key,
            model,
            prompt_ids,
            max_tokens=0,
            n=1,
            temperature=temperature,
            user=user,
        )

    for attempt in range(1, retries + 1):
        logger.info(
            "Warmup (seq_len=%d, concurrency=%d, attempt %d/%d) ...",
            seq_len,
            concurrency,
            attempt,
            retries,
        )
        request_users: list[Optional[str]] = list(users) if users is not None else [None] * concurrency
        if concurrency == 1:
            responses = [_single(request_users[0])]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_single, u) for u in request_users]
                responses = [f.result() for f in as_completed(futures)]

        bad = next((r for r in responses if r.status_code != 200), None)
        if bad is None:
            return
        if attempt < retries:
            logger.warning(
                "Warmup seq_len=%d failed HTTP %d: %s. Retrying in %.0fs ...",
                seq_len,
                bad.status_code,
                bad.text[:500],
                retry_delay,
            )
            time.sleep(retry_delay)
        else:
            raise RuntimeError(f"Warmup seq_len={seq_len} failed HTTP {bad.status_code}: {bad.text[:500]}")


def run_benchmark(
    tokenizer_path: str,
    model: Optional[str],
    base_url: str,
    api_key: str,
    dataset: str,
    pairs: list[tuple[int, int]],
    max_tokens: int,
    temperature: Optional[float] = None,
    separate_requests: bool = False,
    retries: int = 3,
    retry_delay: float = 30.0,
    seed: int = 0,
) -> list[GenBenchmarkResult]:
    tokenizer = _load_auto_tokenizer(tokenizer_path)
    max_seq = max(seq_len for seq_len, _ in pairs)
    chunks = load_chunks(dataset)
    suffix = _DATASET_SUFFIXES.get(dataset, "")

    chunk_texts = build_chunk_texts_to_length(tokenizer, chunks, max_seq)

    url = completions_url(base_url)
    session = requests.Session()

    if separate_requests:
        logger.info("Mode: separate concurrent requests (no n>1)")
    else:
        logger.info("Mode: single request with n=batch_size")

    results: list[GenBenchmarkResult] = []
    prev_seq_len: Optional[int] = None

    # Sort by seq_len descending (longer prompts first for full prompt-cache
    # hit rate) and batch_size descending so the largest batch for a given
    # seq_len comes first.
    pairs = sorted(pairs, key=lambda p: (-p[0], -p[1]))

    prompt_ids: list[int] = []
    seq_users: list[str] = []
    for seq_len, batch_size in pairs:
        if seq_len != prev_seq_len:
            prompt_ids = build_chat_prompt_ids(
                tokenizer,
                suffix,
                chunk_texts,
                target_len=seq_len - max_tokens,
            )
            logger.info("Built prompt for seq_len=%d: %d tokens", seq_len, len(prompt_ids))
            if separate_requests:
                if prev_seq_len is None:
                    # HACK: backend OOMs without this pre-warmup on the largest
                    # seq_len; sending the same prompt 64 times sequentially
                    # with a fresh random `user` each time primes the backend
                    # across generators before the user-pinned warmup at full
                    # batch_size. TODO: fix the backend OOM and remove this.
                    logger.info(
                        "Pre-warmup (seq_len=%d): 64 sequential requests with random users (HACK)",
                        seq_len,
                    )
                    rng = random.Random(seed)
                    for _ in range(64):
                        _warmup_seq_len(
                            url=url,
                            api_key=api_key,
                            model=model,
                            prompt_ids=prompt_ids,
                            seq_len=seq_len,
                            concurrency=1,
                            temperature=temperature,
                            retries=retries,
                            retry_delay=retry_delay,
                            users=[str(rng.randint(0, 2**63 - 1))],
                        )
                seq_users = _generate_users(seq_len + seed, batch_size)
                _warmup_seq_len(
                    url=url,
                    api_key=api_key,
                    model=model,
                    prompt_ids=prompt_ids,
                    seq_len=seq_len,
                    concurrency=len(seq_users),
                    temperature=temperature,
                    retries=retries,
                    retry_delay=retry_delay,
                    users=seq_users,
                )
            else:
                _warmup_seq_len(
                    url=url,
                    api_key=api_key,
                    model=model,
                    prompt_ids=prompt_ids,
                    seq_len=seq_len,
                    concurrency=1,
                    temperature=temperature,
                    retries=retries,
                    retry_delay=retry_delay,
                )
            prev_seq_len = seq_len

        users: Optional[list[str]] = None
        if separate_requests:
            users = seq_users[:batch_size]

        for attempt in range(1, retries + 1):
            try:
                if separate_requests:
                    assert users is not None
                    result = _run_pair_separate_mode(
                        url=url,
                        api_key=api_key,
                        model=model,
                        prompt_ids=prompt_ids,
                        max_tokens=max_tokens,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        temperature=temperature,
                        users=users,
                    )
                else:
                    result = _run_pair_n_mode(
                        session=session,
                        url=url,
                        api_key=api_key,
                        model=model,
                        prompt_ids=prompt_ids,
                        max_tokens=max_tokens,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        temperature=temperature,
                    )
                results.append(result)
                break
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        "Pair seq_len=%d batch_size=%d failed (attempt %d/%d): %s. Retrying in %.0fs ...",
                        seq_len,
                        batch_size,
                        attempt,
                        retries,
                        e,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    raise

    return results


def format_table(rows: list[GenBenchmarkResult]) -> str:
    data: list[list[Any]] = []
    for row in rows:
        data.append([row.seq_len, row.batch_size, row.latency_per_forward])
    return tabulate(
        data,
        headers=["seq_len", "batch_size", "latency_per_forward"],
        tablefmt="pipe",
        floatfmt=".6f",
        colalign=("right", "right", "right"),
    )


def format_csv(rows: list[GenBenchmarkResult]) -> str:
    lines = ["seq_len,batch_size,latency_per_forward"]
    for row in rows:
        lines.append(f"{row.seq_len},{row.batch_size},{row.latency_per_forward:.6f}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="Generation (decode) latency benchmark (Fireworks completions).")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace model directory (tokenizer + config).",
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
        "-p",
        "--seq-batch-pairs",
        default=None,
        help="Comma-separated seq_len:batch_size pairs, e.g. 1024:1,1024:4,4096:1,4096:4. "
        "If omitted, auto-generated from --seq-lens x --max-batch-size.",
    )
    parser.add_argument(
        "-s",
        "--seq-lens",
        default=None,
        help="Comma-separated sequence lengths. "
        "If omitted, auto-generated as min-seq-len, min-seq-len*2, ..., max-seq-len. "
        "Ignored when --seq-batch-pairs is given.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max sequence length (default: read from HF config). " "Used for auto-generating --seq-lens.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=_DEFAULT_MIN_SEQ_LEN,
        help=f"Min sequence length for auto-generated seq lens (default: {_DEFAULT_MIN_SEQ_LEN}).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Max batch size for auto-generated pairs (default: 128). " "Ignored when --seq-batch-pairs is given.",
    )
    parser.add_argument(
        "--max-kv-cache-entries",
        type=int,
        default=None,
        help="Cap batch size so (seq_len + max_tokens) * batch_size <= this value. "
        "Applied to both auto-generated and explicit pairs.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate per completion (default: 100).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format: pipe markdown table (default) or CSV.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional; omit to use server default).",
    )
    parser.add_argument(
        "--separate-requests",
        action="store_true",
        default=False,
        help="Send batch_size separate concurrent requests (each with n=1) "
        "instead of a single request with n=batch_size. Useful for testing LLMs in data parallel mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic per-request `user` ids in --separate-requests mode. Default: 0.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts per (seq_len, batch_size) pair (default: 3).",
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

    if args.seq_batch_pairs is not None:
        pairs = parse_pairs_arg(args.seq_batch_pairs)
    else:
        if args.seq_lens is not None:
            seq_lens = parse_int_list(args.seq_lens)
        else:
            max_seq_len = args.max_seq_len
            try:
                hf_max_seq_len = resolve_max_seq_len(args.tokenizer)
                logger.info("Resolved max_seq_len=%d from HF config", hf_max_seq_len)
            except ValueError:
                hf_max_seq_len = None
                if max_seq_len is None:
                    parser.error("Could not infer max sequence length from config; pass --max-seq-len explicitly.")
            if max_seq_len is None:
                max_seq_len = hf_max_seq_len
            elif hf_max_seq_len is not None:
                max_seq_len = min(max_seq_len, hf_max_seq_len)
            seq_lens = generate_seq_lens(args.min_seq_len, max_seq_len)
        batch_sizes = get_profile_batch_sizes(args.max_batch_size)
        pairs = [(s, b) for s in seq_lens for b in batch_sizes]

    if args.max_kv_cache_entries is not None:
        pairs = [(s, b) for s, b in pairs if (s + args.max_tokens) * b <= args.max_kv_cache_entries]

    if len(pairs) == 0:
        raise RuntimeError("No seq:batch pairs")

    logger.info("Using %d seq-batch pairs: %s", len(pairs), pairs)

    rows = run_benchmark(
        tokenizer_path=args.tokenizer,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        dataset=args.dataset,
        pairs=pairs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        separate_requests=args.separate_requests,
        retries=args.retries,
        retry_delay=args.retry_delay,
        seed=args.seed,
    )
    if args.format == "csv":
        print(format_csv(rows))
    else:
        print(format_table(rows))


if __name__ == "__main__":
    main()
