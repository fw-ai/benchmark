#!/usr/bin/env python3
"""DSV4-aware cached-prefix benchmark (warmup then load).

Uses prefill_load_test token-id layout and per-generator-worker warmup so
prompt-cache + bundle state are primed before concurrent load.  See
prefill_load_test.build_warmup_ids / build_pair_ids.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import orjson
import requests

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

# Reuse llm_bench helpers (same directory).
from prefill_load_test import (
    RoutingConfig,
    build_ids_to_length,
    build_pair_ids,
    build_warmup_ids,
    completions_url,
    load_chunks,
    post_completion,
    resolve_model_type,
    routing_headers_for_worker,
    split_chat_template,
)
from prefill_load_test import _load_auto_tokenizer  # noqa: PLC2701

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FW_HEADER_PREFIX = "fireworks-"


@dataclass
class RequestResult:
    ok: bool
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    gen_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    decode_tps: float = 0.0
    spec_pos0: Optional[float] = None
    spec_pos1: Optional[float] = None
    error: str = ""


def _parse_spec_acceptance(perf_metrics: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    raw = perf_metrics.get("speculation-acceptance") or perf_metrics.get("speculation_acceptance")
    if not raw:
        return None, None
    pos0 = pos1 = None
    for part in str(raw).split(","):
        if ":" not in part:
            continue
        pos_s, val_s = part.split(":", 1)
        if "/" in val_s:
            num_s, den_s = val_s.split("/", 1)
            pct = 100.0 * float(num_s) / float(den_s) if float(den_s) else 0.0
        else:
            pct = float(val_s) * 100.0
        if pos_s == "0":
            pos0 = pct
        elif pos_s == "1":
            pos1 = pct
    return pos0, pos1


def stream_completion(
    *,
    session: requests.Session,
    url: str,
    api_key: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    cached_tokens: int,
    extra_headers: Optional[dict[str, str]] = None,
    user: str = "0",
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 1.0,
        "stream_options": {"include_usage": True},
        "prompt_cache_max_len": cached_tokens,
        "perf_metrics_in_response": True,
        "user": user,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    t_start = time.perf_counter()
    ttft_ms = 0.0
    gen_tokens = 0
    prompt_tokens = 0
    cached_tokens_out = 0
    perf_metrics: dict[str, Any] = {}
    done = False

    try:
        with session.post(url, headers=headers, json=payload, stream=True, timeout=3600) as resp:
            if resp.status_code != 200:
                return RequestResult(
                    ok=False,
                    total_ms=(time.perf_counter() - t_start) * 1000,
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                )
            for raw_line in resp.iter_lines(delimiter=b"\n\n"):
                if not raw_line:
                    continue
                if not raw_line.startswith(b"data:"):
                    continue
                chunk = raw_line[len(b"data:") :].strip()
                if chunk == b"[DONE]":
                    done = True
                    continue
                data = orjson.loads(chunk)
                if data.get("perf_metrics"):
                    perf_metrics = data["perf_metrics"]
                usage = data.get("usage") or {}
                if usage.get("completion_tokens"):
                    gen_tokens = int(usage["completion_tokens"])
                if usage.get("prompt_tokens"):
                    prompt_tokens = int(usage["prompt_tokens"])
                details = usage.get("prompt_tokens_details") or {}
                if details.get("cached_tokens") is not None:
                    cached_tokens_out = int(details["cached_tokens"])
                choices = data.get("choices") or []
                if choices and ttft_ms == 0.0:
                    delta = choices[0].get("text") or ""
                    if delta or gen_tokens:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
    except Exception as exc:
        return RequestResult(
            ok=False,
            total_ms=(time.perf_counter() - t_start) * 1000,
            error=repr(exc),
        )

    total_ms = (time.perf_counter() - t_start) * 1000
    decode_ms = max(total_ms - ttft_ms, 1.0)
    decode_tps = (gen_tokens / decode_ms) * 1000.0 if gen_tokens else 0.0
    spec0, spec1 = _parse_spec_acceptance(perf_metrics)
    return RequestResult(
        ok=True,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        gen_tokens=gen_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens_out,
        decode_tps=decode_tps,
        spec_pos0=spec0,
        spec_pos1=spec1,
    )


def warmup_workers(
    *,
    url: str,
    api_key: str,
    model: str,
    chat_prefix_ids: list[int],
    base_ids: list[int],
    cached_tokens: int,
    routing: RoutingConfig,
    retries: int = 3,
) -> None:
    """Prime paged KV + DSV4 bundles on each generator worker."""
    for worker_idx in range(routing.total_workers):
        warmup_ids = build_warmup_ids(chat_prefix_ids, base_ids, cached_tokens)
        headers = routing_headers_for_worker(routing, worker_idx)
        for attempt in range(1, retries + 1):
            session = requests.Session()
            try:
                resp = post_completion(
                    session,
                    url,
                    api_key,
                    model,
                    warmup_ids,
                    max_tokens=1,
                    prompt_cache_max_len=cached_tokens,
                    extra_headers=headers or None,
                )
            finally:
                session.close()
            if resp.status_code == 200:
                fwc = resp.headers.get(FW_HEADER_PREFIX + "cached-prompt-tokens")
                logger.info(
                    "Warmup worker %d/%d ok (%d warmup ids, cached_header=%s)",
                    worker_idx,
                    routing.total_workers,
                    len(warmup_ids),
                    fwc,
                )
                break
            if attempt == retries:
                raise RuntimeError(
                    f"Warmup worker {worker_idx} failed HTTP {resp.status_code}: {resp.text[:400]}"
                )
            time.sleep(5)


def probe_cached_tokens(
    *,
    url: str,
    api_key: str,
    model: str,
    chat_prefix_ids: list[int],
    chat_suffix_ids: list[int],
    base_ids: list[int],
    tokenizer,
    chunks: list[str],
    prompt_tokens: int,
    cached_tokens: int,
    max_tokens: int,
    routing: RoutingConfig,
) -> int:
    """Return usage.cached_tokens for a single probed request (non-streaming)."""
    rng = random.Random(0)
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
    session = requests.Session()
    try:
        resp = post_completion(
            session,
            url,
            api_key,
            model,
            pair_ids,
            max_tokens,
            cached_tokens,
            routing_headers_for_worker(routing, 0) or None,
        )
    finally:
        session.close()
    if resp.status_code != 200:
        raise RuntimeError(f"Cache probe failed HTTP {resp.status_code}: {resp.text[:300]}")
    usage = resp.json().get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        hdr = resp.headers.get(FW_HEADER_PREFIX + "cached-prompt-tokens")
        cached = int(hdr) if hdr is not None else 0
    return int(cached)


def calibrate_cached_tokens_param(
    *,
    url: str,
    api_key: str,
    model: str,
    chat_prefix_ids: list[int],
    chat_suffix_ids: list[int],
    base_ids: list[int],
    tokenizer,
    chunks: list[str],
    prompt_tokens: int,
    target_cache_pct: float,
    max_tokens: int,
    routing: RoutingConfig,
) -> int:
    """Find prompt_cache_max_len that achieves target_cache_pct on a probe after warmup.

    DSV4 can exhibit a sharp cache cliff (e.g. param=17500 -> ~48%% reported,
    param=17600 -> ~70%%).  We search the smallest param with reported/prompt >= target.
    """
    target_cached = int(prompt_tokens * target_cache_pct / 100.0)
    # DSV4 cliff near ~17.5k on 25k prompts: param must be >= ~17600 for full credit.
    lo = max(target_cached - 500, 1)
    hi = min(len(base_ids) + len(chat_prefix_ids) - 1, target_cached + 2500)
    best = hi
    logger.info(
        "Calibrating cached_tokens param for ~%.0f%% (%d cached / %d prompt), search [%d, %d]",
        target_cache_pct,
        target_cached,
        prompt_tokens,
        lo,
        hi,
    )
    while lo <= hi:
        mid = (lo + hi) // 2
        warmup_workers(
            url=url,
            api_key=api_key,
            model=model,
            chat_prefix_ids=chat_prefix_ids,
            base_ids=base_ids,
            cached_tokens=mid,
            routing=routing,
        )
        reported = probe_cached_tokens(
            url=url,
            api_key=api_key,
            model=model,
            chat_prefix_ids=chat_prefix_ids,
            chat_suffix_ids=chat_suffix_ids,
            base_ids=base_ids,
            tokenizer=tokenizer,
            chunks=chunks,
            prompt_tokens=prompt_tokens,
            cached_tokens=mid,
            max_tokens=max_tokens,
            routing=routing,
        )
        pct = 100.0 * reported / prompt_tokens
        logger.info("  probe param=%d reported=%d (%.1f%%)", mid, reported, pct)
        if reported >= target_cached:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    logger.info("Calibrated cached_tokens param=%d", best)
    return best


def validate_cache_hit(
    *,
    url: str,
    api_key: str,
    model: str,
    chat_prefix_ids: list[int],
    chat_suffix_ids: list[int],
    base_ids: list[int],
    tokenizer,
    chunks: list[str],
    prompt_tokens: int,
    cached_tokens: int,
    max_tokens: int,
    routing: RoutingConfig,
) -> float:
    """Single probe after warmup; return cached/prompt ratio."""
    reported = probe_cached_tokens(
        url=url,
        api_key=api_key,
        model=model,
        chat_prefix_ids=chat_prefix_ids,
        chat_suffix_ids=chat_suffix_ids,
        base_ids=base_ids,
        tokenizer=tokenizer,
        chunks=chunks,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        max_tokens=max_tokens,
        routing=routing,
    )
    ratio = reported / prompt_tokens if prompt_tokens else 0.0
    logger.info(
        "Cache probe: prompt=%d cached=%d (%.1f%%)",
        prompt_tokens,
        reported,
        ratio * 100,
    )
    return ratio


def run_load_window(
    *,
    url: str,
    api_key: str,
    model: str,
    chat_prefix_ids: list[int],
    chat_suffix_ids: list[int],
    base_ids: list[int],
    tokenizer,
    chunks: list[str],
    prompt_tokens: int,
    cached_tokens: int,
    max_tokens: int,
    concurrency: int,
    duration_s: float,
    routing: RoutingConfig,
    seed: int,
) -> list[RequestResult]:
    """Sustained load after per-worker warmup.

    DSV4 bundle cache is per generator worker: only one in-flight request per
  routed worker so each request reuses the warmed prefix+bundle on that cell.
    """
    stop_at = time.perf_counter() + duration_s
    results: list[RequestResult] = []
    results_lock = threading.Lock()
    req_idx = 0
    idx_lock = threading.Lock()
    worker_slots = routing.total_workers if routing.enabled else 1
    per_worker_sem = [threading.Semaphore(1) for _ in range(worker_slots)]

    def worker_loop(thread_slot: int) -> None:
        nonlocal req_idx
        route_idx = thread_slot % worker_slots
        session = requests.Session()
        rng = random.Random(seed + thread_slot * 100_003)
        try:
            while time.perf_counter() < stop_at:
                with per_worker_sem[route_idx]:
                    with idx_lock:
                        my_idx = req_idx
                        req_idx += 1
                    headers = routing_headers_for_worker(routing, route_idx) or None
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
                    res = stream_completion(
                        session=session,
                        url=url,
                        api_key=api_key,
                        model=model,
                        prompt_ids=pair_ids,
                        max_tokens=max_tokens,
                        cached_tokens=cached_tokens,
                        extra_headers=headers,
                        user=f"bench-w{route_idx}",
                    )
                with results_lock:
                    results.append(res)
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker_loop, i) for i in range(concurrency)]
        for fut in as_completed(futures):
            fut.result()
    return results


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def summarize(
    concurrency: int,
    results: list[RequestResult],
    *,
    num_gpus: int,
    duration_s: float,
) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    fail = len(results) - len(ok)
    qps = len(ok) / duration_s if duration_s > 0 else 0.0
    qps_gpu = qps / num_gpus
    ttfts = [r.ttft_ms for r in ok]
    tps = [r.decode_tps for r in ok if r.decode_tps > 0]
    cache_pcts = [
        100.0 * r.cached_tokens / r.prompt_tokens
        for r in ok
        if r.prompt_tokens > 0
    ]
    spec0 = [r.spec_pos0 for r in ok if r.spec_pos0 is not None]
    spec1 = [r.spec_pos1 for r in ok if r.spec_pos1 is not None]
    return {
        "concurrency": concurrency,
        "requests_ok": len(ok),
        "requests_fail": fail,
        "qps": qps,
        "qps_per_gpu": qps_gpu,
        "p50_ttft_ms": percentile(ttfts, 50),
        "p95_ttft_ms": percentile(ttfts, 95),
        "p50_tps": percentile(tps, 50),
        "p95_tps": percentile(tps, 95),
        "mean_cache_pct": statistics.mean(cache_pcts) if cache_pcts else 0.0,
        "p50_cache_pct": percentile(cache_pcts, 50),
        "mean_spec_pos0": statistics.mean(spec0) if spec0 else None,
        "mean_spec_pos1": statistics.mean(spec1) if spec1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DSV4 cached-prefix benchmark")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--api-key", default=os.environ.get("FIREWORKS_API_KEY", ""))
    parser.add_argument("--base-url", default="https://api.fireworks.ai/inference")
    parser.add_argument("--prompt-tokens", type=int, default=25000)
    parser.add_argument(
        "--cached-tokens",
        type=int,
        default=0,
        help="prompt_cache_max_len; 0 = auto-calibrate from --target-cache-pct",
    )
    parser.add_argument(
        "--target-cache-pct",
        type=float,
        default=70.0,
        help="Target %% of prompt_tokens reported as cached (used when --cached-tokens=0)",
    )
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--dataset", default="limericks", choices=["limericks", "code"])
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--num-gens", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concurrency-sweep",
        default="",
        help="Comma-separated concurrency levels (e.g. 12,24,48,96)",
    )
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--min-cache-pct", type=float, default=60.0)
    args = parser.parse_args()

    if not args.api_key:
        print("FIREWORKS_API_KEY required", file=sys.stderr)
        sys.exit(1)

    routing = RoutingConfig(num_servers=args.num_servers, num_gens=args.num_gens)
    url = completions_url(args.base_url)
    tokenizer = _load_auto_tokenizer(args.tokenizer)
    model_type = resolve_model_type(args.tokenizer)
    if model_type != "deepseek_v4":
        logger.warning("Tokenizer model_type=%s (expected deepseek_v4)", model_type)

    chunks = load_chunks(args.dataset)
    chat_prefix_ids, chat_suffix_ids = split_chat_template(tokenizer, args.tokenizer, model_type)
    base_ids = build_ids_to_length(tokenizer, chunks, args.prompt_tokens)

    levels = [args.concurrency]
    if args.concurrency_sweep:
        levels = [int(x.strip()) for x in args.concurrency_sweep.split(",") if x.strip()]

    cached_tokens = args.cached_tokens
    if cached_tokens <= 0:
        cached_tokens = calibrate_cached_tokens_param(
            url=url,
            api_key=args.api_key,
            model=args.deployment,
            chat_prefix_ids=chat_prefix_ids,
            chat_suffix_ids=chat_suffix_ids,
            base_ids=base_ids,
            tokenizer=tokenizer,
            chunks=chunks,
            prompt_tokens=args.prompt_tokens,
            target_cache_pct=args.target_cache_pct,
            max_tokens=args.max_tokens,
            routing=routing,
        )
    logger.info(
        "Using cached_tokens=%d for %.0f%% target on %d prompt tokens",
        cached_tokens,
        args.target_cache_pct,
        args.prompt_tokens,
    )

    all_rows: list[dict[str, Any]] = []

    for conc in levels:
        logger.info("=== concurrency=%d: warming %d workers ===", conc, routing.total_workers)
        warmup_workers(
            url=url,
            api_key=args.api_key,
            model=args.deployment,
            chat_prefix_ids=chat_prefix_ids,
            base_ids=base_ids,
            cached_tokens=cached_tokens,
            routing=routing,
        )
        cache_pct = (
            validate_cache_hit(
                url=url,
                api_key=args.api_key,
                model=args.deployment,
                chat_prefix_ids=chat_prefix_ids,
                chat_suffix_ids=chat_suffix_ids,
                base_ids=base_ids,
                tokenizer=tokenizer,
                chunks=chunks,
                prompt_tokens=args.prompt_tokens,
                cached_tokens=cached_tokens,
                max_tokens=args.max_tokens,
                routing=routing,
            )
            * 100.0
        )
        if cache_pct < args.min_cache_pct:
            logger.warning(
                "Cache probe %.1f%% < min %.1f%% — continuing but results may be invalid",
                cache_pct,
                args.min_cache_pct,
            )

        logger.info("=== concurrency=%d: load %.0fs ===", conc, args.duration)
        t0 = time.perf_counter()
        results = run_load_window(
            url=url,
            api_key=args.api_key,
            model=args.deployment,
            chat_prefix_ids=chat_prefix_ids,
            chat_suffix_ids=chat_suffix_ids,
            base_ids=base_ids,
            tokenizer=tokenizer,
            chunks=chunks,
            prompt_tokens=args.prompt_tokens,
            cached_tokens=cached_tokens,
            max_tokens=args.max_tokens,
            concurrency=conc,
            duration_s=args.duration,
            routing=routing,
            seed=args.seed + conc,
        )
        elapsed = time.perf_counter() - t0
        row = summarize(conc, results, num_gpus=args.num_gpus, duration_s=elapsed)
        all_rows.append(row)
        logger.info(
            "c=%d qps/gpu=%.3f p50_ttft=%.0f p95_ttft=%.0f p50_tps=%.1f p95_tps=%.1f "
            "cache=%.1f%% spec0=%s spec1=%s ok=%d fail=%d",
            conc,
            row["qps_per_gpu"],
            row["p50_ttft_ms"],
            row["p95_ttft_ms"],
            row["p50_tps"],
            row["p95_tps"],
            row["mean_cache_pct"],
            f"{row['mean_spec_pos0']:.0f}" if row["mean_spec_pos0"] is not None else "—",
            f"{row['mean_spec_pos1']:.0f}" if row["mean_spec_pos1"] is not None else "—",
            row["requests_ok"],
            row["requests_fail"],
        )
        time.sleep(15)

    print("\n| C | QPS/GPU | P50 TPS | P95 TPS | P50 TTFT | P95 TTFT | Cache% | Spec0 | Spec1 | OK | Fail |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in all_rows:
        spec0 = f"{row['mean_spec_pos0']:.0f}" if row["mean_spec_pos0"] is not None else "—"
        spec1 = f"{row['mean_spec_pos1']:.0f}" if row["mean_spec_pos1"] is not None else "—"
        print(
            f"| {row['concurrency']} | {row['qps_per_gpu']:.3f} | {row['p50_tps']:.1f} | "
            f"{row['p95_tps']:.1f} | {row['p50_ttft_ms']:.0f} | {row['p95_ttft_ms']:.0f} | "
            f"{row['mean_cache_pct']:.1f} | {spec0} | {spec1} | "
            f"{row['requests_ok']} | {row['requests_fail']} |"
        )

    if args.output_csv:
        import csv

        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
