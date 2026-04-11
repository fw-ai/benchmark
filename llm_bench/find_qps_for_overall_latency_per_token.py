#!/usr/bin/env python3
"""
Binary-search Locust user count (-u) to maximize achieved QPS while keeping
``Overall Latency Per Token`` (ms) at or below a target.

Each probe runs ``locust`` headless for a fixed duration (default 30s). Spawn rate
(-r) is set high enough to reach the target user count quickly (override with
``--spawn-rate``).

Example (pass your usual flags after ``--``; omit ``-u``, ``-r``, ``-t`` — the
script injects them)::

    ./find_qps_for_overall_latency_per_token.py \\
        --target-ms 16.67 \\
        --duration 30s \\
        --max-users 256 \\
        -- \\
        -f /home/ying/benchmark/llm_bench/load_test.py \\
        --host http://127.0.0.1:80 \\
        --provider fireworks \\
        -m accounts/kimi-k2p5/models/hf \\
        -p 90000 --prompt-cache-max-len 90000 -o 576 \\
        --tokenizer /shared/data2/kimi-k2p5/hf/ \\
        --acceptance-probs-override '[0.78, 0.64, 0.50]'
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

SUMMARY_LINE = re.compile(r"^\s*([^\s].*?)\s*:\s*(.*)\s*$")


def strip_u_r_t(argv: list[str]) -> list[str]:
    """Remove -u/--users, -r/--spawn-rate, -t/--run-time and glued forms (-u10)."""
    out: list[str] = []
    i = 0
    two_arg = frozenset({"-u", "--users", "-r", "--spawn-rate", "-t", "--run-time"})
    eq_keys = frozenset({"-u", "--users", "-r", "--spawn-rate", "-t", "--run-time"})
    while i < len(argv):
        s = argv[i]
        if "=" in s:
            k = s.split("=", 1)[0]
            if k in eq_keys:
                i += 1
                continue
        if s in two_arg:
            i += 2 if i + 1 < len(argv) and not argv[i + 1].startswith("-") else 1
            continue
        if s.startswith("-u") and len(s) > 2 and s[2] != "-":
            i += 1
            continue
        if s.startswith("-r") and len(s) > 2 and s[2] != "-":
            i += 1
            continue
        if s.startswith("-t") and len(s) > 2 and s[2] != "-":
            i += 1
            continue
        out.append(s)
        i += 1
    return out


@dataclass
class ProbeResult:
    users: int
    overall_latency_per_token_ms: Optional[float]
    qps: Optional[float]
    num_requests: Optional[int]
    exit_code: int
    summary_block: str
    full_output: str


def parse_summary(text: str) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """Return (overall_latency_per_token_ms, qps, num_requests) from Summary section."""
    overall: Optional[float] = None
    qps: Optional[float] = None
    num_req: Optional[int] = None
    in_summary = False
    for line in text.splitlines():
        if " Summary " in line and "=" in line:
            in_summary = True
            continue
        if in_summary and line.strip().startswith("=" * 10):
            break
        if not in_summary:
            continue
        m = SUMMARY_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key == "Overall Latency Per Token":
            try:
                overall = float(val)
            except ValueError:
                pass
        elif key == "Qps":
            try:
                qps = float(val)
            except ValueError:
                pass
        elif key == "Num Requests":
            try:
                num_req = int(float(val))
            except ValueError:
                pass
    return overall, qps, num_req


def run_probe(
    locust_bin: str,
    base_args: list[str],
    users: int,
    spawn_rate: int,
    duration: str,
) -> ProbeResult:
    cmd = [
        locust_bin,
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        duration,
        *base_args,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout + "\n" + proc.stderr
    overall, qps, num_req = parse_summary(out)
    block = ""
    if " Summary " in out:
        start = out.index(" Summary ")
        end = out.find("=" * 80, start)
        if end != -1:
            block = out[start : end + 80]
    return ProbeResult(
        users=users,
        overall_latency_per_token_ms=overall,
        qps=qps,
        num_requests=num_req,
        exit_code=proc.returncode,
        summary_block=block,
        full_output=out,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Binary-search Locust -u to max QPS subject to Overall Latency Per Token <= target (ms)."
    )
    p.add_argument("--target-ms", type=float, default=16.67, help="Max allowed overall latency per token (ms).")
    p.add_argument(
        "--duration",
        type=str,
        default="30s",
        help="Locust -t for each probe (e.g. 30s, 2m). Deferred until after spawn in load_test.",
    )
    p.add_argument("--min-users", type=int, default=1)
    p.add_argument("--max-users", type=int, default=512)
    p.add_argument(
        "--spawn-rate",
        type=int,
        default=None,
        help="Locust -r per probe. Default: min(users, spawn-cap).",
    )
    p.add_argument(
        "--spawn-cap",
        type=int,
        default=500,
        help="When --spawn-rate is omitted, use -r min(users, this).",
    )
    p.add_argument(
        "--locust",
        type=str,
        default=None,
        help="Path to locust executable (default: PATH).",
    )
    p.add_argument(
        "--show-probe-output",
        action="store_true",
        help="Print full locust stdout/stderr for every probe (verbose).",
    )
    p.add_argument(
        "--final-duration",
        type=str,
        default=None,
        help="If set, re-run best user count with this -t and print Summary (e.g. 2m).",
    )
    p.add_argument("locust_args", nargs=argparse.REMAINDER, help="Locust args after -- (do not pass -u/-r/-t).")
    args = p.parse_args()
    if args.locust_args and args.locust_args[0] == "--":
        args.locust_args = args.locust_args[1:]

    locust_bin = args.locust or shutil.which("locust")
    if not locust_bin:
        print("locust not found; install locust or pass --locust", file=sys.stderr)
        return 1

    base = strip_u_r_t(args.locust_args)
    if not base:
        print("No locust arguments after -- (need at least -f ...)", file=sys.stderr)
        return 1

    lo = args.min_users
    hi = args.max_users
    target = args.target_ms

    best: Optional[ProbeResult] = None

    def spawn_for(u: int) -> int:
        if args.spawn_rate is not None:
            return max(1, args.spawn_rate)
        return max(1, min(u, args.spawn_cap))

    print(
        f"Binary search: target Overall Latency Per Token <= {target} ms, "
        f"users in [{lo}, {hi}], duration {args.duration} per probe.\n"
    )

    while lo <= hi:
        mid = (lo + hi) // 2
        sr = spawn_for(mid)
        print(f"--- Probe: -u {mid} -r {sr} -t {args.duration} ---", flush=True)
        pr = run_probe(locust_bin, base, mid, sr, args.duration)
        if args.show_probe_output:
            print(pr.full_output)

        if pr.exit_code != 0:
            print(
                f"Probe failed (exit {pr.exit_code}); treating as too much load. "
                f"If this is unexpected, use --show-probe-output.\n",
                flush=True,
            )
            hi = mid - 1
            continue

        if pr.overall_latency_per_token_ms is None:
            print("Could not parse Overall Latency Per Token from output; stderr/stdout tail:", flush=True)
            print(pr.full_output[-4000:], flush=True)
            return 1

        lat = pr.overall_latency_per_token_ms
        qps_s = f"{pr.qps:.6f}" if pr.qps is not None else "?"
        print(
            f"  overall_latency_per_token={lat:.4f} ms  qps={qps_s}  num_requests={pr.num_requests}\n",
            flush=True,
        )

        if lat <= target:
            best = pr
            lo = mid + 1
        else:
            hi = mid - 1

    print("\n" + "=" * 80)
    if best is None:
        print(
            f"No configuration in [{args.min_users}, {args.max_users}] users met "
            f"Overall Latency Per Token <= {target} ms (or all probes failed)."
        )
        print("Try lowering --target-ms, raising --max-users, or checking server capacity.")
        return 2

    print(
        f"Best user count meeting latency target: -u {best.users} "
        f"(Overall Latency Per Token = {best.overall_latency_per_token_ms:.6f} ms, "
        f"QPS ≈ {best.qps})\n"
    )

    final_d = args.final_duration or args.duration
    if args.final_duration:
        print(f"Final run: -u {best.users} -r {spawn_for(best.users)} -t {final_d}\n", flush=True)
        final = run_probe(locust_bin, base, best.users, spawn_for(best.users), final_d)
        if final.summary_block:
            print(final.summary_block)
        else:
            print(final.full_output[-8000:])
        return 0 if final.exit_code == 0 else 1

    if best.summary_block:
        print(best.summary_block)
    else:
        print(best.full_output[-8000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
