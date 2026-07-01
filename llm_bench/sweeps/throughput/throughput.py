#!/usr/bin/env python3
"""Run throughput sweeps across configured API implementations."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
import urllib.request
import yaml
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_bench import gen_load_test, prefill_load_test
from llm_bench.apis.base_api import BaseApi
from llm_bench.apis.fireworks_api import FireworksApi
from llm_bench.apis.sglang_api import SGLangApi
from llm_bench.estimators.bandwidth.bandwidth_estimator import estimate_bandwidth
from llm_bench.estimators.flops.flops_estimator import estimate_prefill_flops
from llm_bench.estimators.gpu_specs import resolve_gpu_config
from llm_bench.utils import log_utils

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        log_utils.create_handler(
            color=log_utils.BRIGHT_PURPLE,
            fmt="%(asctime)s %(levelname)s %(filename)s:%(lineno)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    ],
    force=True,
)

SUMMARY_COLUMNS = [
    "api",
    "batch size",
    "n_sequences",
    "success %",
    "gpu_type",
    "gpu_pflops_s",
    "gpu_hbm_TBps",
    "gpu_nvlink_GBps",
    "prefill_s",
    "prefill_toks",
    "prefill_tps",
    "prefill_pflops",
    "prefill_mfu",
    "decode_s",
    "decode_toks",
    "decode_tps",
    "decode_hbm_bw_tb",
    "decode_hbm_mbu",
    "decode_nvlink_bw_mb",
    "decode_nvlink_mbu",
    "req_decode_tps",
    "req_decode_tps_p90",
    "req_decode_tps_p99",
    "server_config",
]

PETA = 10**15

API_CLASSES: dict[str, type[BaseApi]] = {
    "fireworks": FireworksApi,
    "sglang": SGLangApi,
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = math.ceil((pct / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def round_float(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return value
    if value != 0 and abs(value) < 10 ** (-digits):
        return round(value, max(digits + 3, 6))
    return round(value, digits)


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "avg": round_float(statistics.fmean(values), 3) if values else float("nan"),
        "p90": round_float(percentile(values, 90), 3),
        "p95": round_float(percentile(values, 95), 3),
        "p99": round_float(percentile(values, 99), 3),
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return data


def timestamped_run_dir(output_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    return output_dir / f"throughput_{ts}"


def benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    # The initial stub used top-level `args`; prefer `benchmark` but keep this
    # fallback so older local configs still run.
    return dict(config.get("benchmark") or config.get("args") or {})


def get_batch_sizes(bench_cfg: dict[str, Any]) -> list[int]:
    values = bench_cfg.get("batch_sizes")
    if not values:
        raise ValueError("Config must set benchmark.batch_sizes")
    return [int(v) for v in values]


def parse_batch_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_selected_apis(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "api": row["api"],
        "batch size": row["batch_size"],
        "n_sequences": row["n_sequences"],
        "success %": row["success_pct"],
        "gpu_type": row["gpu_type"],
        "gpu_pflops_s": row["gpu_pflops_s"],
        "gpu_hbm_TBps": row["gpu_hbm_TBps"],
        "gpu_nvlink_GBps": row["gpu_nvlink_GBps"],
        "prefill_s": row["prefill_s"],
        "prefill_toks": row["prefill_toks"],
        "prefill_tps": row["prefill_tps"],
        "prefill_pflops": row["prefill_pflops"],
        "prefill_mfu": row["prefill_mfu"],
        "decode_s": row["decode_s"],
        "decode_toks": row["decode_toks"],
        "decode_tps": row["decode_tps"],
        "decode_hbm_bw_tb": row["decode_hbm_bw_tb"],
        "decode_hbm_mbu": row["decode_hbm_mbu"],
        "decode_nvlink_bw_mb": row["decode_nvlink_bw_mb"],
        "decode_nvlink_mbu": row["decode_nvlink_mbu"],
        "req_decode_tps": row["req_decode_tps"],
        "req_decode_tps_p90": row["req_decode_tps_p90"],
        "req_decode_tps_p99": row["req_decode_tps_p99"],
        "server_config": row["server_config"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_summary_row(row))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def write_summary_png(path: Path, rows: list[dict[str, Any]], *, show_prefill: bool, show_generation: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts: list[tuple[str, str, str]] = []
    if show_prefill:
        charts.append(("prefill_tps", "Prefill", "prefill_tps"))
    if show_generation:
        charts.append(("decode_tps", "Generation", "decode_tps"))
    if not charts:
        return

    fig, axes = plt.subplots(1, len(charts), figsize=(7 * len(charts), 4), squeeze=False)
    api_names = sorted({row["api"] for row in rows})
    for ax, (metric, title, ylabel) in zip(axes[0], charts):
        for api_name in api_names:
            api_rows = sorted((row for row in rows if row["api"] == api_name), key=lambda r: r["batch_size"])
            xs = [row["batch_size"] for row in api_rows if _is_number(row.get(metric))]
            ys = [row[metric] for row in api_rows if _is_number(row.get(metric))]
            if xs:
                ax.plot(xs, ys, marker="o", label=api_name)
        ax.set_title(title)
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def resolve_load_test_dataset(path_value: str | None) -> str:
    if path_value is None:
        return "limericks"
    dataset = Path(path_value).stem
    if dataset not in {"limericks", "code"}:
        raise ValueError(f"Load tests only support limericks/code datasets, got {path_value!r}")
    return dataset


def load_test_routing(bench_cfg: dict[str, Any], module: Any) -> Any:
    return module.RoutingConfig(
        num_servers=int(bench_cfg.get("num_servers", 1)),
        num_gens=int(bench_cfg.get("num_generators_per_server", bench_cfg.get("num_gens", 1))),
    )


def write_load_test_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def resolve_api_model_id(api: BaseApi) -> str:
    with urllib.request.urlopen(f"{api.get_hostname().rstrip('/')}/v1/models", timeout=30) as response:
        payload = json.loads(response.read().decode())
    data = payload.get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError(f"No model id returned by {api.get_hostname()}/v1/models: {payload}")
    return str(data[0]["id"])


def estimate_prefill_flops_total(
    model_path: str,
    *,
    prompt_tokens: int,
    n_sequences: int,
) -> int:
    per_sequence = int(
        estimate_prefill_flops(
            model_path,
            context_length=prompt_tokens,
            batch_size=1,
        )["total"]
    )
    return per_sequence * n_sequences


def estimate_decode_bandwidth_bytes(
    model_path: str,
    *,
    prompt_tokens: int,
    max_tokens: int,
    batch_size: int,
    n_sequences: int,
    world_size: int,
    attn_sharding: str,
    moe_sharding: str,
    convert_to_precision: str | None = None,
) -> dict[str, float]:
    hbm_bytes = 0.0
    nvlink_bytes = 0.0
    for step in range(max_tokens):
        context_length = prompt_tokens + step
        if context_length <= 0:
            continue
        estimate = estimate_bandwidth(
            hf_model_name=model_path,
            context_length=context_length,
            batch_size=batch_size,
            n_sequences=n_sequences,
            world_size=world_size,
            attn_sharding=attn_sharding,
            moe_sharding=moe_sharding,
            convert_to_precision=convert_to_precision,
        )
        hbm_bytes += float(estimate["hbm"]["total"])
        nvlink_bytes += float(estimate["nvlink"]["total"])
    return {
        "hbm": hbm_bytes,
        "nvlink": nvlink_bytes,
    }


def compute_utilization_metrics(
    *,
    model_path: str,
    gpu_cfg: dict[str, Any],
    prompt_tokens: int,
    max_tokens: int,
    batch_size: int,
    n_sequences: int,
    prefill_s: float,
    decode_s: float,
    world_size: int,
    attn_sharding: str,
    moe_sharding: str,
    convert_to_precision: str | None = None,
) -> dict[str, float]:
    num_gpus = int(gpu_cfg["num_gpus"])
    peak_flops_per_s = float(gpu_cfg["gpu_pflops_s"]) * PETA * num_gpus
    peak_hbm_tb_per_s = float(gpu_cfg["gpu_hbm_TBps"]) * world_size
    peak_nvlink_mb_per_s = float(gpu_cfg["gpu_nvlink_GBps"]) * 1000.0 * world_size

    prefill_flops = estimate_prefill_flops_total(
        model_path,
        prompt_tokens=prompt_tokens,
        n_sequences=n_sequences,
    )
    prefill_mfu = (
        100.0 * prefill_flops / prefill_s / peak_flops_per_s if prefill_s > 0 and peak_flops_per_s > 0 else float("nan")
    )

    decode_bytes = estimate_decode_bandwidth_bytes(
        model_path,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        batch_size=batch_size,
        n_sequences=n_sequences,
        world_size=world_size,
        attn_sharding=attn_sharding,
        moe_sharding=moe_sharding,
        convert_to_precision=convert_to_precision,
    )
    decode_hbm_bw_tb = decode_bytes["hbm"] / decode_s / 1e12 if decode_s > 0 else float("nan")
    decode_hbm_mbu = (
        100.0 * decode_hbm_bw_tb / peak_hbm_tb_per_s if decode_s > 0 and peak_hbm_tb_per_s > 0 else float("nan")
    )
    decode_nvlink_bw_mb = decode_bytes["nvlink"] / decode_s / 1e6 if decode_s > 0 else float("nan")
    decode_nvlink_mbu = (
        100.0 * decode_nvlink_bw_mb / peak_nvlink_mb_per_s
        if decode_s > 0 and peak_nvlink_mb_per_s > 0
        else float("nan")
    )

    return {
        "prefill_pflops": prefill_flops / PETA,
        "prefill_mfu": prefill_mfu,
        "decode_hbm_bw_tb": decode_hbm_bw_tb,
        "decode_hbm_mbu": decode_hbm_mbu,
        "decode_nvlink_bw_mb": decode_nvlink_bw_mb,
        "decode_nvlink_mbu": decode_nvlink_mbu,
    }


def summary_row_from_load_tests(
    *,
    api_name: str,
    batch_size: int,
    model_path: str,
    gpu_cfg: dict[str, Any],
    prompt_tokens: int,
    max_tokens: int,
    world_size: int,
    attn_sharding: str,
    moe_sharding: str,
    convert_to_precision: str | None,
    prefill_result: prefill_load_test.PairBenchmarkResult,
    gen_result: gen_load_test.GenBenchmarkResult,
    server_config: str,
) -> dict[str, Any]:
    n_sequences = prefill_result.num_prompts
    prefill_s = prefill_result.duration
    decode_s = gen_result.generation_duration
    prefill_toks = prompt_tokens * n_sequences
    decode_toks = max_tokens * gen_result.num_sequences
    util = compute_utilization_metrics(
        model_path=model_path,
        gpu_cfg=gpu_cfg,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        batch_size=batch_size,
        n_sequences=n_sequences,
        prefill_s=prefill_s,
        decode_s=decode_s,
        world_size=world_size,
        attn_sharding=attn_sharding,
        moe_sharding=moe_sharding,
        convert_to_precision=convert_to_precision,
    )
    row: dict[str, Any] = {
        "api": api_name,
        "batch_size": batch_size,
        "n_sequences": n_sequences,
        "success_pct": 100.0,
        "gpu_type": gpu_cfg["gpu_type"],
        "gpu_pflops_s": round_float(gpu_cfg["gpu_pflops_s"], 3),
        "gpu_hbm_TBps": round_float(gpu_cfg["gpu_hbm_TBps"], 3),
        "gpu_nvlink_GBps": round_float(gpu_cfg["gpu_nvlink_GBps"], 3),
        "prefill_s": round_float(prefill_s, 3),
        "prefill_toks": prefill_toks,
        "prefill_tps": round_float(prefill_toks / prefill_s if prefill_s > 0 else float("nan"), 3),
        "prefill_pflops": round_float(util["prefill_pflops"], 3),
        "prefill_mfu": round_float(util["prefill_mfu"], 3),
        "decode_s": round_float(decode_s, 3),
        "decode_toks": decode_toks,
        "decode_tps": round_float(decode_toks / decode_s if decode_s > 0 else float("nan"), 3),
        "decode_hbm_bw_tb": round_float(util["decode_hbm_bw_tb"], 3),
        "decode_hbm_mbu": round_float(util["decode_hbm_mbu"], 3),
        "decode_nvlink_bw_mb": round_float(util["decode_nvlink_bw_mb"], 3),
        "decode_nvlink_mbu": round_float(util["decode_nvlink_mbu"], 3),
        "req_decode_tps": round_float(gen_result.req_decode_tps, 3),
        "req_decode_tps_p90": round_float(gen_result.req_decode_tps_p90, 3),
        "req_decode_tps_p99": round_float(gen_result.req_decode_tps_p99, 3),
        "server_config": server_config,
    }
    return row


def run_load_test_benchmark(
    *,
    api_name: str,
    api: BaseApi,
    model_path: str,
    bench_cfg: dict[str, Any],
    gpu_cfg: dict[str, Any],
    batch_sizes: list[int],
    out_dir: Path,
    server_config: str,
) -> list[dict[str, Any]]:
    dataset = resolve_load_test_dataset(bench_cfg.get("dataset"))
    prompt_tokens = int(bench_cfg.get("prompt_tokens", 50_000))
    prefill_cached_tokens = int(bench_cfg.get("prefill_cache_tokens", 0))
    max_tokens = int(bench_cfg.get("max_tokens", 600))
    n_sequences = int(bench_cfg.get("n_sequences", 0))
    retries = int(bench_cfg.get("retries", 3))
    retry_delay = float(bench_cfg.get("retry_delay", 30.0))
    temperature = float(bench_cfg.get("temperature", 0.0))
    prefill_max_tokens = int(bench_cfg.get("prefill_max_tokens", 1 if api_name == "sglang" else 0))
    decode_warmup_max_tokens = int(bench_cfg.get("decode_warmup_max_tokens", 1 if api_name == "sglang" else 0))
    convert_to_precision = bench_cfg.get("convert_to_precision")
    try:
        world_size = int(bench_cfg["world_size"])
        attn_sharding = str(bench_cfg["attn_sharding"])
        moe_sharding = str(bench_cfg["moe_sharding"])
    except KeyError as exc:
        raise ValueError(
            "benchmark.world_size, benchmark.attn_sharding, and benchmark.moe_sharding must be set"
        ) from exc
    logging.info(f"resolving model id {api_name=} base_url={api.get_hostname()}")
    resolve_start = time.perf_counter()
    model_id = resolve_api_model_id(api)
    elapsed = time.perf_counter() - resolve_start
    logging.info(f"resolved {api_name=} {model_id=} {elapsed=:.1f}s")
    if not (0 <= prefill_cached_tokens < prompt_tokens):
        raise ValueError("benchmark.prefill_cache_tokens must satisfy 0 <= value < prompt_tokens")
    if n_sequences <= 0:
        raise ValueError("benchmark.n_sequences must be set to a positive integer")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "server_config.txt").write_text(server_config + "\n")
    logging.info(
        f"benchmark config {api_name=} {dataset=} {prompt_tokens=} "
        f"{prefill_cached_tokens=} {max_tokens=} {n_sequences=} {batch_sizes=}"
    )

    prefill_by_batch: dict[int, prefill_load_test.PairBenchmarkResult] = {}
    uncached_tokens = prompt_tokens - prefill_cached_tokens
    for batch_size in batch_sizes:
        logging.info(f"prefill start {api_name=} {batch_size=}")
        prefill_start = time.perf_counter()
        rows = prefill_load_test.run_benchmark(
            tokenizer_path=model_path,
            model=model_id,
            base_url=api.get_hostname(),
            api_key=None,
            dataset=dataset,
            pairs=[(prompt_tokens, prefill_cached_tokens)],
            min_tokens_to_batch=batch_size * uncached_tokens,
            max_tokens=prefill_max_tokens,
            rng_seed=int(bench_cfg.get("seed", 0)),
            n_sequences=n_sequences,
            retries=retries,
            retry_delay=retry_delay,
            routing=load_test_routing(bench_cfg, prefill_load_test),
        )
        prefill_by_batch[batch_size] = rows[0]
        prefill_path = out_dir / f"prefill_b{batch_size}.csv"
        write_load_test_csv(prefill_path, prefill_load_test.format_csv(rows))
        elapsed = time.perf_counter() - prefill_start
        logging.info(f"prefill done {api_name=} {batch_size=} output={prefill_path} {elapsed=:.1f}s")

    seq_len = prompt_tokens + max_tokens
    logging.info(f"decode start {api_name=} {seq_len=} {batch_sizes=}")
    decode_start = time.perf_counter()
    gen_rows = gen_load_test.run_benchmark(
        tokenizer_path=model_path,
        model=model_id,
        base_url=api.get_hostname(),
        api_key=None,
        dataset=dataset,
        pairs=[(seq_len, batch_size) for batch_size in batch_sizes],
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
        seed=int(bench_cfg.get("seed", 0)),
        routing=load_test_routing(bench_cfg, gen_load_test),
        prompt_cache_max_len=prompt_tokens if api_name == "fireworks" else None,
        warmup_max_tokens=decode_warmup_max_tokens,
        n_sequences=n_sequences,
    )
    decode_path = out_dir / "decode.csv"
    write_load_test_csv(decode_path, gen_load_test.format_csv(gen_rows))
    elapsed = time.perf_counter() - decode_start
    logging.info(f"decode done {api_name=} output={decode_path} {elapsed=:.1f}s")
    gen_by_batch = {row.batch_size: row for row in gen_rows}
    return [
        summary_row_from_load_tests(
            api_name=api_name,
            batch_size=batch_size,
            model_path=model_path,
            gpu_cfg=gpu_cfg,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            world_size=world_size,
            attn_sharding=attn_sharding,
            moe_sharding=moe_sharding,
            convert_to_precision=str(convert_to_precision) if convert_to_precision is not None else None,
            prefill_result=prefill_by_batch[batch_size],
            gen_result=gen_by_batch[batch_size],
            server_config=server_config,
        )
        for batch_size in batch_sizes
    ]


async def run_one_api(
    *,
    config: dict[str, Any],
    api_config: dict[str, Any],
    bench_cfg: dict[str, Any],
    run_dir: Path,
    batch_sizes: list[int],
    measure_prefill: bool,
    measure_generation: bool,
) -> list[dict[str, Any]]:
    api_type = str(api_config["type"])
    api_name = api_type
    try:
        api_cls = API_CLASSES[api_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported api type {api_type!r}") from exc
    api_dir = run_dir / "apis" / api_name
    model_path = str(config["model_path"])
    gpu_cfg = resolve_gpu_config(bench_cfg, api_config)

    logging.info(f"api startup start {api_name=} {api_dir=}")
    startup_start = time.perf_counter()
    with api_cls(
        model_path=model_path,
        endpoint=dict(api_config.get("endpoint") or {}),
        args=dict(api_config.get("args") or {}),
        environment=dict(api_config.get("environment") or {}),
        log_dir=api_dir,
    ) as api:
        elapsed = time.perf_counter() - startup_start
        logging.info(f"api startup done {api_name=} base_url={api.get_hostname()} {elapsed=:.1f}s")
        rows = run_load_test_benchmark(
            api_name=api_name,
            api=api,
            model_path=model_path,
            bench_cfg=bench_cfg,
            gpu_cfg=gpu_cfg,
            batch_sizes=batch_sizes,
            out_dir=api_dir,
            server_config=api.server_config,
        )
        logging.info(f"api benchmark done {api_name=} rows={len(rows)}")
        return rows
    raise ValueError("At least one of prefill or generation benchmark must be enabled")


async def async_main(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    logging.info(f"loading config {config_path=}")
    config = load_config(config_path)
    bench_cfg = benchmark_config(config)
    if args.batch_sizes:
        batch_sizes = parse_batch_sizes(args.batch_sizes)
        bench_cfg["batch_sizes"] = batch_sizes
        copied_benchmark = dict(config.get("benchmark") or config.get("args") or {})
        copied_benchmark["batch_sizes"] = batch_sizes
        if "benchmark" in config:
            config["benchmark"] = copied_benchmark
        else:
            config["args"] = copied_benchmark
    else:
        batch_sizes = get_batch_sizes(bench_cfg)
    measure_prefill = True
    measure_generation = True

    selected_apis = parse_selected_apis(args.select_apis)
    apis = [
        dict(api)
        for api in config.get("apis", [])
        if api.get("enabled", True) and (not selected_apis or str(api.get("type")) in selected_apis)
    ]
    if not apis:
        raise ValueError("Config must contain at least one selected enabled API in `apis`")
    logging.info(f"selected apis={ [api.get('type') for api in apis] } {batch_sizes=}")

    run_dir = timestamped_run_dir(args.output_dir.resolve())
    run_dir.mkdir(parents=True, exist_ok=False)
    copied_config = dict(config)
    copied_config["apis"] = apis
    (run_dir / "config.yaml").write_text(yaml.safe_dump(copied_config, sort_keys=False))
    logging.info(f"run dir created {run_dir=}")

    all_rows: list[dict[str, Any]] = []
    for api_config in apis:
        logging.info(f"api run start api_type={api_config.get('type')}")
        rows = await run_one_api(
            config=config,
            api_config=api_config,
            bench_cfg=bench_cfg,
            run_dir=run_dir,
            batch_sizes=batch_sizes,
            measure_prefill=measure_prefill,
            measure_generation=measure_generation,
        )
        all_rows.extend(rows)
        write_summary_csv(run_dir / "summary.csv", all_rows)
        logging.info(f"wrote summary csv rows={len(all_rows)}")
        write_summary_png(
            run_dir / "summary.png",
            all_rows,
            show_prefill=measure_prefill,
            show_generation=measure_generation,
        )
        logging.info("wrote summary png")

    print(f"Wrote {run_dir / 'summary.csv'}")
    print(f"Wrote {run_dir / 'summary.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to throughput YAML config.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where timestamped run output is written."
    )
    parser.add_argument(
        "--select-apis",
        default="",
        help="Comma-separated API types to run, e.g. fireworks,sglang. Empty means all enabled APIs.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="",
        help="Comma-separated batch sizes overriding the config, e.g. 100,200.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
