#!/usr/bin/env python3
"""
Generation (decode) latency benchmark for Fireworks /v1/completions.

For each (seq_len, batch_size) pair, builds a single user message wrapped in
the model's chat template (applied client-side via the HF tokenizer), sends
the resulting token-id prompt with ``n > 1``, verifies that the full shareable
prompt prefix was served from cache, then reports per-forward-pass generation
latency using the fireworks-generation-duration header and the number of
target-model forward passes (speculation-aware). In routed mode the requested
batch is split across generator workers, with one ``n > 1`` request per worker.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import random
import signal
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from threading import Event
from typing import Any, Optional

logger = logging.getLogger(__name__)

import requests
import transformers
from tabulate import tabulate

from context_utils import (
    generate_geometric_lengths,
    generation_sequence_limit,
    read_config_json,
    resolve_max_seq_len,
    tokenizer_asset_path,
)

FW_HEADER_PREFIX = "fireworks-"

_SERVICE_INDEX_HEADER = "x-fireworks-generator-worker-service-index"
_LOCAL_INDEX_HEADER = "x-fireworks-generator-worker-local-index"


@dataclass(frozen=True)
class RoutingConfig:
    """Generator-worker routing knobs for the benchmark client.

    `num_servers` and `num_gens` define the round-robin space across
    (service-index, local-index) workers. Routing headers are only emitted
    when at least one dimension is greater than 1; otherwise existing
    benchmark runs are unchanged byte-for-byte.

    Requires `enableGeneratorWorkerTargeting=true` on the deployment for the
    headers to take effect; otherwise they are ignored by Envoy.
    """

    num_servers: int = 1
    num_gens: int = 1

    @property
    def enabled(self) -> bool:
        return self.num_servers > 1 or self.num_gens > 1

    @property
    def num_workers(self) -> int:
        return self.num_servers * self.num_gens

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        return f"server-first {self.num_servers}x{self.num_gens}"


def routing_headers_for_worker(cfg: Optional[RoutingConfig], worker_idx: int) -> dict[str, str]:
    """Return the routing headers to attach to the request for the given worker.

    Server-first cycling keeps the per-server distribution balanced even when
    `batch_size < num_servers * num_gens`: e.g. with 2 servers x 4 locals and
    batch=4, workers visited are (s0,l0),(s1,l0),(s0,l1),(s1,l1) so each server
    gets two requests rather than one server eating the whole batch.
    """
    if cfg is None or not cfg.enabled:
        return {}
    total = cfg.num_servers * cfg.num_gens
    flat = worker_idx % total
    server = flat % cfg.num_servers
    local = flat // cfg.num_servers
    return {_SERVICE_INDEX_HEADER: str(server), _LOCAL_INDEX_HEADER: str(local)}


def split_batch_across_workers(batch_size: int, cfg: RoutingConfig) -> list[tuple[int, int]]:
    """Return ``(worker_idx, n)`` assignments whose ``n`` values sum to the batch.

    At most one request is sent to each configured worker. The remainder is
    assigned server-first so the per-worker batch sizes differ by at most one.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    worker_count = min(batch_size, cfg.num_workers)
    per_worker, remainder = divmod(batch_size, worker_count)
    return [(worker_idx, per_worker + (worker_idx < remainder)) for worker_idx in range(worker_count)]


_FAST_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]

_DEFAULT_MIN_SEQ_LEN = 1000


def get_profile_batch_sizes(max_batch_size: int, min_batch_size: int = 1) -> list[int]:
    if min_batch_size > max_batch_size:
        return []

    r = [b for b in _FAST_BATCH_SIZES if min_batch_size <= b <= max_batch_size]
    if not r:
        r = [min_batch_size]

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


def resolve_model_type(tokenizer_path: str) -> str:
    try:
        config = transformers.AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
        text_config = config.get_text_config()
        return getattr(text_config, "model_type", None) or getattr(config, "model_type", "")
    except Exception:
        config_json = read_config_json(tokenizer_path)
        text_config = config_json.get("text_config") or {}
        return text_config.get("model_type") or config_json.get("model_type", "")


@lru_cache(maxsize=None)
def load_dsv4_encode_messages(tokenizer_path: str) -> Any:
    path = tokenizer_asset_path(tokenizer_path, "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("hf_dsv4_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DSV4 encoding module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def generate_seq_lens(min_seq_len: int, max_seq_len: int) -> list[int]:
    return generate_geometric_lengths(min_seq_len, max_seq_len, factor=2)


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
    tokenizer_path: str,
    content_text: str,
    model_type: str,
) -> list[int]:
    """Apply the model's chat template to a single user message and return token ids."""
    if model_type == "deepseek_v4":
        encode_messages = load_dsv4_encode_messages(tokenizer_path)
        prompt = encode_messages([{"role": "user", "content": content_text}], thinking_mode="chat")
        return _normalize_ids(tokenizer.encode(prompt, add_special_tokens=False))
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


def split_chat_template_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_path: str,
    model_type: str,
) -> tuple[list[int], list[int]]:
    """Return the template token IDs before and after one user message."""
    sentinel = "abcdefghij"
    sentinel_ids = _normalize_ids(tokenizer.encode(sentinel, add_special_tokens=False))
    if not sentinel_ids:
        raise RuntimeError("Sentinel tokenized to an empty sequence")
    templated = apply_chat_template_ids(tokenizer, tokenizer_path, sentinel, model_type)
    width = len(sentinel_ids)
    for index in range(len(templated) - width + 1):
        if templated[index : index + width] == sentinel_ids:
            return templated[:index], templated[index + width :]
    raise RuntimeError(f"Could not locate sentinel IDs {sentinel_ids} in templated prompt")


def build_chat_prompt_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_path: str,
    model_type: str,
    suffix_text: str,
    chunk_texts: list[str],
    target_len: int,
) -> list[int]:
    """Build an exact-length, client-templated prompt token sequence."""
    if not chunk_texts:
        raise ValueError("no chunks provided")

    prefix_ids, suffix_ids = split_chat_template_ids(tokenizer, tokenizer_path, model_type)
    instruction_ids = _normalize_ids(tokenizer.encode(suffix_text, add_special_tokens=False))
    content_len = target_len - len(prefix_ids) - len(suffix_ids)
    body_len = content_len - len(instruction_ids)
    if body_len < 0:
        raise ValueError(f"target_len={target_len} is too small for the chat template and instruction")

    body_ids: list[int] = []
    chunk_index = 0
    while len(body_ids) < body_len and chunk_index < 1_000_000:
        chunk = chunk_texts[chunk_index % len(chunk_texts)]
        body_ids.extend(_normalize_ids(tokenizer.encode(chunk, add_special_tokens=False)))
        chunk_index += 1
    if len(body_ids) < body_len:
        raise RuntimeError(f"Could not build {body_len} dataset tokens")

    # The completions endpoint accepts these IDs directly; it validates them but
    # does not re-tokenize or add BOS/EOS tokens.
    prompt_ids = prefix_ids + body_ids[:body_len] + instruction_ids + suffix_ids
    if len(prompt_ids) != target_len:
        raise RuntimeError(f"Built {len(prompt_ids)} prompt tokens, expected {target_len}")
    return prompt_ids


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
    api_key: Optional[str],
    model: Optional[str],
    prompt: list[int],
    max_tokens: int,
    n: int,
    temperature: Optional[float] = None,
    user: Optional[str] = None,
    extra_headers: Optional[dict[str, str]] = None,
) -> requests.Response:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "stream": False,
        "ignore_eos": True,
        "context_length_exceeded_behavior": "error",
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if model is not None:
        payload["model"] = model
    if user is not None:
        payload["user"] = user
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    return session.post(url, headers=headers, json=payload, timeout=3600)


def validate_completion_usage(
    response_data: dict[str, Any],
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> None:
    """Verify the server used the exact pre-tokenized request boundary."""
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("Completion response is missing usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(f"Server reported {prompt_tokens} prompt tokens; sent {expected_prompt_tokens} token IDs")
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(f"Server generated {completion_tokens} tokens; requested {expected_completion_tokens}")


class PromptCacheVerificationError(RuntimeError):
    """The measured request did not prove a full prompt-cache hit."""


def validate_full_prompt_cache_hit(
    response_data: dict[str, Any],
    headers: Mapping[str, str],
    expected_prompt_tokens: int,
    request_label: str,
) -> int:
    """Return cached prompt tokens after verifying a full shareable-prefix hit.

    Fireworks keeps the final prompt token out of the shareable prefix because
    that token must be evaluated to produce the first generated token. The
    non-streaming completions response reports the cached count in both the
    ``fireworks-cached-prompt-tokens`` header and, on current servers,
    ``usage.prompt_tokens_details.cached_tokens``.
    """
    header_cached_tokens = get_int_header(headers, "cached-prompt-tokens")

    body_cached_tokens: Optional[int] = None
    usage = response_data.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            try:
                body_cached_tokens = int(details["cached_tokens"])
            except (TypeError, ValueError) as error:
                raise PromptCacheVerificationError(
                    f"{request_label} returned invalid usage.prompt_tokens_details.cached_tokens="
                    f"{details['cached_tokens']!r}"
                ) from error

    if header_cached_tokens is None and body_cached_tokens is None:
        raise PromptCacheVerificationError(
            f"{request_label} did not report cached prompt tokens in either "
            "fireworks-cached-prompt-tokens or usage.prompt_tokens_details.cached_tokens"
        )
    if (
        header_cached_tokens is not None
        and body_cached_tokens is not None
        and header_cached_tokens != body_cached_tokens
    ):
        raise PromptCacheVerificationError(
            f"{request_label} reported inconsistent cached prompt tokens: "
            f"header={header_cached_tokens}, body={body_cached_tokens}"
        )

    cached_tokens = header_cached_tokens if header_cached_tokens is not None else body_cached_tokens
    assert cached_tokens is not None
    expected_cached_tokens = max(expected_prompt_tokens - 1, 0)
    if cached_tokens != expected_cached_tokens:
        raise PromptCacheVerificationError(
            f"{request_label} did not fully hit the prompt cache: "
            f"cached {cached_tokens}/{expected_cached_tokens} shareable prompt tokens "
            f"(prompt_tokens={expected_prompt_tokens})"
        )
    return cached_tokens


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
    api_key: Optional[str],
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
    completion_tokens = max_tokens * batch_size
    validate_completion_usage(
        resp_json,
        expected_prompt_tokens=len(prompt_ids),
        expected_completion_tokens=completion_tokens,
    )
    cached_tokens = validate_full_prompt_cache_hit(
        resp_json,
        r.headers,
        expected_prompt_tokens=len(prompt_ids),
        request_label="n-mode request",
    )
    num_fwd = parse_num_forward_passes(r.headers, batch_size, completion_tokens)

    fwp = get_int_header(r.headers, "prompt-tokens")
    logger.info(
        "  -> server prompt-tokens=%s, cached-prompt-tokens=%d, generation-duration=%.6fs, "
        "forward_passes=%s, client=%.2fs",
        fwp,
        cached_tokens,
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


def _run_pair_routed_n_mode(
    url: str,
    api_key: Optional[str],
    model: Optional[str],
    prompt_ids: list[int],
    max_tokens: int,
    seq_len: int,
    batch_size: int,
    temperature: Optional[float],
    users: list[str],
    routing: Optional[RoutingConfig] = None,
    worker_offset: int = 0,
) -> GenBenchmarkResult:
    """Split the batch across workers using one concurrent ``n > 1`` request each."""
    routing = routing or RoutingConfig()
    assignments = split_batch_across_workers(batch_size, routing)
    logger.info(
        "Pair (seq_len=%d, batch_size=%d): %d routed n-requests %s, max_tokens=%d, routing=%s",
        seq_len,
        batch_size,
        len(assignments),
        [n for _, n in assignments],
        max_tokens,
        routing.describe(),
    )
    assert len(users) == len(assignments), f"expected {len(assignments)} users, got {len(users)}"

    def _single_request(worker_idx: int, n: int, user: str) -> requests.Response:
        s = requests.Session()
        headers = routing_headers_for_worker(routing, worker_offset + worker_idx)
        if headers and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "  worker=%d server=%s local=%s",
                worker_idx,
                headers.get(_SERVICE_INDEX_HEADER),
                headers.get(_LOCAL_INDEX_HEADER),
            )
        return post_completion(
            s,
            url,
            api_key,
            model,
            prompt_ids,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            user=user,
            extra_headers=headers or None,
        )

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        futures = {
            pool.submit(_single_request, worker_idx, n, users[request_idx]): (worker_idx, n)
            for request_idx, (worker_idx, n) in enumerate(assignments)
        }
        responses: list[tuple[int, int, requests.Response]] = []
        for fut in as_completed(futures):
            worker_idx, n = futures[fut]
            responses.append((worker_idx, n, fut.result()))
    wall = time.perf_counter() - wall_start

    gen_durs: list[float] = []
    fwd_counts: list[int] = []
    cached_counts: list[int] = []
    for worker_idx, n, r in responses:
        target_headers = routing_headers_for_worker(routing, worker_offset + worker_idx)
        request_label = (
            f"worker {worker_idx} "
            f"(server={target_headers.get(_SERVICE_INDEX_HEADER)}, "
            f"local={target_headers.get(_LOCAL_INDEX_HEADER)})"
        )
        if r.status_code != 200:
            raise RuntimeError(f"{request_label} failed HTTP {r.status_code}: {r.text[:500]}")
        gd = get_header(r.headers, "generation-duration")
        if gd is None:
            raise RuntimeError(
                f"{request_label} is missing fireworks-generation-duration header "
                "(need dedicated deployment?). "
                f"Got keys: {[k for k in r.headers.keys() if 'fireworks' in k.lower()]}"
            )
        gen_durs.append(gd)
        resp_json = r.json()
        validate_completion_usage(
            resp_json,
            expected_prompt_tokens=len(prompt_ids),
            expected_completion_tokens=max_tokens * n,
        )
        cached_counts.append(
            validate_full_prompt_cache_hit(
                resp_json,
                r.headers,
                expected_prompt_tokens=len(prompt_ids),
                request_label=request_label,
            )
        )
        fwd_counts.append(parse_num_forward_passes(r.headers, batch_size=n, completion_tokens=max_tokens * n))

    avg_gen_dur = sum(gen_durs) / len(gen_durs)
    avg_fwd = sum(fwd_counts) / len(fwd_counts)

    logger.info(
        "  -> full cache hit on %d routed requests (%d tokens each), "
        "gen_dur avg=%.6fs  min=%.6fs  max=%.6fs, fwd_passes avg=%.0f, client=%.2fs",
        len(cached_counts),
        cached_counts[0],
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
    api_key: Optional[str],
    model: Optional[str],
    prompt_ids: list[int],
    seq_len: int,
    concurrency: int,
    temperature: Optional[float],
    retries: int,
    retry_delay: float,
    users: Optional[list[str]] = None,
    routing: Optional[RoutingConfig] = None,
    worker_offset: int = 0,
) -> None:
    """Issue `concurrency` warmup completions in parallel for the same prompt."""
    if users is not None:
        assert len(users) == concurrency, f"expected {concurrency} users, got {len(users)}"

    def _single(worker_idx: int, user: Optional[str]) -> requests.Response:
        headers = routing_headers_for_worker(routing, worker_offset + worker_idx)
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
            extra_headers=headers or None,
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
        try:
            if concurrency == 1:
                responses = [_single(0, request_users[0])]
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(_single, i, u) for i, u in enumerate(request_users)]
                    responses = [f.result() for f in as_completed(futures)]
        except requests.exceptions.RequestException as error:
            if attempt < retries:
                logger.warning(
                    "Warmup seq_len=%d failed (attempt %d/%d): %s. Retrying in %.0fs ...",
                    seq_len,
                    attempt,
                    retries,
                    error,
                    retry_delay,
                )
                time.sleep(retry_delay)
                continue
            raise

        bad = next((r for r in responses if r.status_code != 200), None)
        if bad is None:
            return
        if 400 <= bad.status_code < 500 and bad.status_code not in (408, 429):
            raise RuntimeError(f"Warmup seq_len={seq_len} failed HTTP {bad.status_code}: {bad.text[:500]}")
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
    api_key: Optional[str],
    dataset: str,
    pairs: list[tuple[int, int]],
    max_tokens: int,
    temperature: Optional[float] = None,
    retries: int = 3,
    retry_delay: float = 30.0,
    seed: int = 0,
    routing: Optional[RoutingConfig] = None,
    repeat: int = 1,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> list[GenBenchmarkResult]:
    if repeat < 0:
        raise ValueError("repeat must be >= 0")
    should_stop = stop_requested or (lambda: False)

    tokenizer = _load_auto_tokenizer(tokenizer_path)
    model_type = resolve_model_type(tokenizer_path)
    if model_type == "deepseek_v4":
        logger.info("Using DeepSeek-V4 benchmark prompt encoder")
    max_seq = max(seq_len for seq_len, _ in pairs)
    chunks = load_chunks(dataset)
    suffix = _DATASET_SUFFIXES.get(dataset, "")

    chunk_texts = build_chunk_texts_to_length(tokenizer, chunks, max_seq)

    url = completions_url(base_url)
    session = requests.Session()

    # Routing enabled: split the requested batch across generator workers and
    # send one n>1 request to each selected worker. Routing disabled: send one
    # n=batch_size request through the load balancer.
    routing = routing or RoutingConfig()
    routed_requests = routing.enabled
    if routed_requests:
        logger.info("Mode: routed n>1 requests (batch split across workers)")
    else:
        logger.info("Mode: single request with n=batch_size")
    logger.info("Routing: %s", routing.describe())

    results: list[GenBenchmarkResult] = []

    # Sort by seq_len descending (longer prompts first for full prompt-cache
    # hit rate) and batch_size descending so the largest batch for a given
    # seq_len comes first.
    pairs = sorted(pairs, key=lambda p: (-p[0], -p[1]))

    prepared: dict[int, tuple[list[int], list[str]]] = {}
    round_index = 0
    while (repeat == 0 or round_index < repeat) and not should_stop():
        round_index += 1
        if repeat == 0:
            logger.info("Starting measured round %d (repeating until Ctrl-C)", round_index)
        elif repeat > 1:
            logger.info("Starting measured round %d/%d", round_index, repeat)

        for seq_len, batch_size in pairs:
            if should_stop():
                break

            if seq_len not in prepared:
                prompt_len = seq_len - max_tokens
                if prompt_len < 1:
                    raise ValueError(f"seq_len={seq_len} must exceed max_tokens={max_tokens}")
                prompt_ids = build_chat_prompt_ids(
                    tokenizer,
                    tokenizer_path,
                    model_type,
                    suffix,
                    chunk_texts,
                    target_len=prompt_len,
                )
                logger.info(
                    "Built exact boundary for seq_len=%d: prompt=%d + output=%d",
                    seq_len,
                    len(prompt_ids),
                    max_tokens,
                )
                seq_users: list[str] = []
                if routed_requests:
                    # Pairs are sorted by descending batch size, so the first
                    # pair for a sequence length warms every worker needed by
                    # any later pair or repeat round.
                    worker_count = min(batch_size, routing.num_workers)
                    seq_users = _generate_users(seq_len + seed, worker_count)
                    _warmup_seq_len(
                        url=url,
                        api_key=api_key,
                        model=model,
                        prompt_ids=prompt_ids,
                        seq_len=seq_len,
                        concurrency=worker_count,
                        temperature=temperature,
                        retries=retries,
                        retry_delay=retry_delay,
                        users=seq_users,
                        routing=routing,
                    )
                else:
                    # n-mode measurement is unrouted; warmup must match or cache primes the wrong worker.
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
                prepared[seq_len] = (prompt_ids, seq_users)

            if should_stop():
                break

            prompt_ids, seq_users = prepared[seq_len]
            users: Optional[list[str]] = None
            if routed_requests:
                users = seq_users[: min(batch_size, routing.num_workers)]

            for attempt in range(1, retries + 1):
                try:
                    if routed_requests:
                        assert users is not None
                        result = _run_pair_routed_n_mode(
                            url=url,
                            api_key=api_key,
                            model=model,
                            prompt_ids=prompt_ids,
                            max_tokens=max_tokens,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            temperature=temperature,
                            users=users,
                            routing=routing,
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
                except PromptCacheVerificationError:
                    # A cache miss invalidates the measurement. Retrying would
                    # silently turn that failed measurement into another warmup.
                    raise
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
        help="Bearer token (default: API_KEY or FIREWORKS_API_KEY). "
        "Optional; omit for servers that don't require auth.",
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
        help="Sequence-length cap used to generate geometric points plus the exact endpoint. "
        "The default is the model config limit minus Fireworks generation headroom.",
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
        "--min-batch-size",
        type=int,
        default=1,
        help="Min batch size for auto-generated pairs (default: 1). " "Ignored when --seq-batch-pairs is given.",
    )
    parser.add_argument(
        "--max-kv-cache-entries",
        type=int,
        default=None,
        help="Cap batch size so seq_len * batch_size <= this value. "
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
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic per-request `user` ids when routing is enabled "
        "(--num-servers/--num-gens > 1). Default: 0.",
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
    parser.add_argument(
        "--repeat",
        nargs="?",
        type=int,
        const=0,
        default=1,
        metavar="N",
        help="Repeat the measured pair set N times after warming each sequence length once. "
        "Pass without N (or pass 0) to repeat until Ctrl-C. Default: 1.",
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=1,
        help="Number of generator servers to fan requests out across via "
        "x-fireworks-generator-worker-service-index. Requires the deployment to "
        "be rendered with enableGeneratorWorkerTargeting=true; otherwise the "
        "header is ignored. Default: 1 (no service-index header sent).",
    )
    parser.add_argument(
        "--num-generators-per-server",
        "--num-gens",
        dest="num_generators_per_server",
        type=int,
        default=1,
        help="Number of data-parallel generator groups per server, used as the "
        "round-robin range for x-fireworks-generator-worker-local-index. "
        "Alias matches fw-infer's --num-gens. Default: 1.",
    )
    args = parser.parse_args()

    if args.num_servers < 1:
        parser.error("--num-servers must be >= 1")
    if args.num_generators_per_server < 1:
        parser.error("--num-generators-per-server must be >= 1")
    if args.min_batch_size < 1:
        parser.error("--min-batch-size must be >= 1")
    if args.min_batch_size > args.max_batch_size:
        parser.error("--min-batch-size must be <= --max-batch-size")
    if args.repeat < 0:
        parser.error("--repeat must be >= 0")
    routing = RoutingConfig(
        num_servers=args.num_servers,
        num_gens=args.num_generators_per_server,
    )

    if args.seq_batch_pairs is not None:
        pairs = parse_pairs_arg(args.seq_batch_pairs)
    else:
        if args.seq_lens is not None:
            seq_lens = parse_int_list(args.seq_lens)
        else:
            max_seq_len = args.max_seq_len
            try:
                model_max_seq_len = resolve_max_seq_len(args.tokenizer)
                hf_sequence_limit = generation_sequence_limit(model_max_seq_len)
                logger.info(
                    "Resolved model max_seq_len=%d; completions generation limit=%d",
                    model_max_seq_len,
                    hf_sequence_limit,
                )
            except ValueError as error:
                hf_sequence_limit = None
                if max_seq_len is None:
                    parser.error(str(error))
            if max_seq_len is None:
                max_seq_len = hf_sequence_limit
            elif hf_sequence_limit is not None:
                max_seq_len = min(max_seq_len, hf_sequence_limit)
            seq_lens = generate_seq_lens(args.min_seq_len, max_seq_len)
            logger.info(
                "Auto-generated sequence lengths including exact endpoint=%d: %s",
                max_seq_len,
                seq_lens,
            )
        batch_sizes = get_profile_batch_sizes(args.max_batch_size, args.min_batch_size)
        pairs = [(s, b) for s in seq_lens for b in batch_sizes]

    if args.max_kv_cache_entries is not None:
        pairs = [(s, b) for s, b in pairs if s * b <= args.max_kv_cache_entries]

    if len(pairs) == 0:
        raise RuntimeError("No seq:batch pairs")

    logger.info("Using %d seq-batch pairs: %s", len(pairs), pairs)

    stop_event = Event()
    previous_sigint_handler: Any = None
    if args.repeat == 0:
        previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def request_stop(_signum: int, _frame: Any) -> None:
            if not stop_event.is_set():
                logger.info("Ctrl-C received; stopping after the current in-flight requests finish")
                stop_event.set()

        signal.signal(signal.SIGINT, request_stop)

    try:
        rows = run_benchmark(
            tokenizer_path=args.tokenizer,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            dataset=args.dataset,
            pairs=pairs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            retries=args.retries,
            retry_delay=args.retry_delay,
            seed=args.seed,
            routing=routing,
            repeat=args.repeat,
            stop_requested=stop_event.is_set,
        )
    finally:
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
    if args.format == "csv":
        print(format_csv(rows))
    else:
        print(format_table(rows))


if __name__ == "__main__":
    main()
