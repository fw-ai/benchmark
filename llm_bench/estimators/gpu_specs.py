"""Peak GPU specs for MFU/MBU normalization."""

from __future__ import annotations

import re
import subprocess
from typing import Any

# FP8 tensor-core peaks (petaFLOP/s per GPU), HBM bandwidth (TB/s per GPU),
# and NVLink/NVSwitch bandwidth (GB/s per GPU).
GPU_SPECS: dict[str, dict[str, float]] = {
    "b300": {"pflops_s": 4.5, "hbm_TBps": 8.0, "nvlink_GBps": 1800.0},
    "gb300": {"pflops_s": 4.5, "hbm_TBps": 8.0, "nvlink_GBps": 1800.0},
    "b200": {"pflops_s": 4.5, "hbm_TBps": 8.0, "nvlink_GBps": 1800.0},
    "gb200": {"pflops_s": 4.5, "hbm_TBps": 8.0, "nvlink_GBps": 1800.0},
    "h200": {"pflops_s": 2.0, "hbm_TBps": 4.8, "nvlink_GBps": 900.0},
    "h100": {"pflops_s": 2.0, "hbm_TBps": 3.35, "nvlink_GBps": 900.0},
    "a100": {"pflops_s": 1.0, "hbm_TBps": 2.0, "nvlink_GBps": 600.0},
}


def normalize_gpu_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def infer_gpu_type_from_nvidia_smi() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        name = line.strip().lower()
        if not name:
            continue
        for token in ("b300", "gb300", "b200", "gb200", "h200", "h100", "a100"):
            if token in name.replace("-", "").replace(" ", ""):
                return token
    return None


def detect_gpu_count() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    counts = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    if not counts:
        return None
    return max(counts)


def infer_num_gpus(api_config: dict[str, Any], bench_cfg: dict[str, Any]) -> int:
    if bench_cfg.get("num_gpus") is not None:
        return int(bench_cfg["num_gpus"])
    args = dict(api_config.get("args") or {})
    api_type = str(api_config.get("type") or "")
    inferred: int | None = None
    if api_type == "sglang":
        inferred = int(args.get("tp_size", 1))
    elif api_type == "fireworks":
        inferred = int(args.get("num_ranks_override", 1))
    detected = detect_gpu_count()
    if inferred is not None and detected is not None:
        return min(inferred, detected)
    if inferred is not None:
        return inferred
    if detected is not None:
        return detected
    return 1


def resolve_gpu_config(
    bench_cfg: dict[str, Any],
    api_config: dict[str, Any],
) -> dict[str, Any]:
    gpu_type = bench_cfg.get("gpu_type")
    if gpu_type is None:
        gpu_type = infer_gpu_type_from_nvidia_smi()
    if gpu_type is None:
        raise ValueError(
            "benchmark.gpu_type must be set when GPU type cannot be inferred from nvidia-smi"
        )
    gpu_type = normalize_gpu_type(str(gpu_type))
    specs = GPU_SPECS.get(gpu_type)
    if specs is None:
        raise ValueError(f"Unsupported gpu_type {gpu_type!r}; known types: {sorted(GPU_SPECS)}")

    gpu_pflops_s = bench_cfg.get("gpu_pflops_s")
    if gpu_pflops_s is None:
        gpu_pflops_s = specs["pflops_s"]
    gpu_hbm_TBps = bench_cfg.get("gpu_hbm_TBps", bench_cfg.get("gpu_bw_tbs"))
    if gpu_hbm_TBps is None:
        gpu_hbm_TBps = specs["hbm_TBps"]
    gpu_nvlink_GBps = bench_cfg.get("gpu_nvlink_GBps")
    if gpu_nvlink_GBps is None:
        gpu_nvlink_GBps = specs["nvlink_GBps"]

    return {
        "gpu_type": gpu_type,
        "gpu_pflops_s": float(gpu_pflops_s),
        "gpu_hbm_TBps": float(gpu_hbm_TBps),
        "gpu_nvlink_GBps": float(gpu_nvlink_GBps),
        "num_gpus": infer_num_gpus(api_config, bench_cfg),
    }
