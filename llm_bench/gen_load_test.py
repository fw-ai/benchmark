#!/usr/bin/env python3
"""
Generation (decode) latency benchmark for Fireworks /v1/chat/completions.

For each (seq_len, batch_size) pair, sends a single prompt of length seq_len
with n=batch_size and max_tokens output tokens, then reports per-forward-pass
generation latency derived from fireworks-generation-duration and the number
of target-model forward passes (speculation-aware).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import requests
import transformers
from tabulate import tabulate

FW_HEADER_PREFIX = "fireworks-"

_FAST_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]
_DEFAULT_MIN_SEQ_LEN = 1024


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


def build_ids_to_length(
    tokenizer: transformers.PreTrainedTokenizer,
    chunks: list[str],
    target_len: int,
) -> list[int]:
    ids: list[int] = []
    i = 0
    while len(ids) < target_len and i < 1_000_000:
        lim = chunks[i % len(chunks)]
        ids.extend(tokenizer.encode(lim + "\n\n", add_special_tokens=False))
        i += 1
    return ids[:target_len]


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


def chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/chat/completions"


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


def post_chat_completion(
    session: requests.Session,
    url: str,
    api_key: str,
    model: Optional[str],
    prompt: str,
    max_tokens: int,
    n: int,
    temperature: Optional[float] = None,
) -> requests.Response:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "n": n,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if model is not None:
        payload["model"] = model
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


def run_benchmark(
    tokenizer_path: str,
    model: Optional[str],
    base_url: str,
    api_key: str,
    dataset: str,
    pairs: list[tuple[int, int]],
    max_tokens: int,
    temperature: Optional[float] = None,
) -> list[GenBenchmarkResult]:
    tokenizer = _load_auto_tokenizer(tokenizer_path)
    max_seq = max(seq_len for seq_len, _ in pairs)
    chunks = load_chunks(dataset)
    suffix = _DATASET_SUFFIXES.get(dataset, "")

    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    prefix_ids = build_ids_to_length(tokenizer, chunks, max_seq)
    if len(prefix_ids) + len(suffix_ids) < max_seq:
        raise RuntimeError(f"Could only build {len(prefix_ids) + len(suffix_ids)} tokens from dataset, need {max_seq}")

    url = chat_completions_url(base_url)
    session = requests.Session()

    # Warmup with the full-length prompt
    warmup_ids = prefix_ids[: max_seq - len(suffix_ids)] + suffix_ids
    warmup_prompt = tokenizer.decode(warmup_ids, skip_special_tokens=True)
    print(f"Warmup: prompt tokens ~ {len(warmup_ids)}", file=sys.stderr)
    t0 = time.perf_counter()
    w = post_chat_completion(
        session,
        url,
        api_key,
        model,
        warmup_prompt,
        max_tokens=1,
        n=1,
        temperature=temperature,
    )
    elapsed_w = time.perf_counter() - t0
    if w.status_code != 200:
        raise RuntimeError(f"Warmup failed HTTP {w.status_code}: {w.text[:500]}")
    print(f"Warmup done in {elapsed_w:.2f}s", file=sys.stderr)

    results: list[GenBenchmarkResult] = []

    for seq_len, batch_size in pairs:
        prompt_ids = prefix_ids[: seq_len - len(suffix_ids)] + suffix_ids
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        print(
            f"Pair (seq_len={seq_len}, batch_size={batch_size}): " f"n={batch_size}, max_tokens={max_tokens}",
            file=sys.stderr,
        )

        wall_start = time.perf_counter()
        r = post_chat_completion(
            session,
            url,
            api_key,
            model,
            prompt_text,
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
        print(
            f"  -> server prompt-tokens={fwp}, "
            f"generation-duration={gen_dur:.6f}s, "
            f"forward_passes={num_fwd}, client={wall:.2f}s",
            file=sys.stderr,
        )

        lpf = gen_dur / num_fwd
        results.append(
            GenBenchmarkResult(
                seq_len=seq_len,
                batch_size=batch_size,
                max_tokens=max_tokens,
                generation_duration=gen_dur,
                latency_per_forward=lpf,
                client_duration=wall,
            )
        )

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
    parser = argparse.ArgumentParser(description="Generation (decode) latency benchmark (Fireworks chat completions).")
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
            if max_seq_len is None:
                max_seq_len = resolve_max_seq_len(args.tokenizer)
                print(f"Resolved max_seq_len={max_seq_len} from HF config", file=sys.stderr)
            seq_lens = generate_seq_lens(args.min_seq_len, max_seq_len)
        batch_sizes = get_profile_batch_sizes(args.max_batch_size)
        pairs = [(s, b) for s in seq_lens for b in batch_sizes]

    if args.max_kv_cache_entries is not None:
        pairs = [(s, b) for s, b in pairs if (s + args.max_tokens) * b <= args.max_kv_cache_entries]

    print(f"Using {len(pairs)} seq-batch pairs: {pairs}", file=sys.stderr)

    rows = run_benchmark(
        tokenizer_path=args.tokenizer,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        dataset=args.dataset,
        pairs=pairs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.format == "csv":
        print(format_csv(rows))
    else:
        print(format_table(rows))


if __name__ == "__main__":
    main()
