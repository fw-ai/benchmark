"""D-117's pre-flight runner: run a tier plan's llm_bench steps against mock_server.py on CPU, and write the
step summary that task 15's per-class D-63 check reads.

    python preflight_run.py PLAN.json --out DIR [--step ramp-3] [--duration 180] [--mock-tokenizer ID]

What it runs, from the plan as task 07 writes it (record + traffic_generator_create_args):
- the mock engine, emulating the plan's measured reuse: record.reuse_probe's overhang as the recomputed window
  and its boundary as the restore block (D-111/D-113; --window/--boundary override), tokenizing with the served
  model's tokenizer (record.served_model_config, else the step's --served-model-config or --tokenizer);
- the plan's fill step (any step whose extra args carry --prefix-fill), then the scored step: --step, or the
  step at the plan's target rate (the last one whose --qps equals record.target_qps), with the plan's exact extra
  args, its concurrency and rate, and the duration capped at --duration;
- the scored step's summary goes to DIR/<step>.summary.csv (the file task 15's check reads), its log beside it.

It prints the applied reuse semantics and a per-class designed vs delivered table for a quick look; the verdict
is task 15's check (D-117). Paths in the extra args that don't exist locally can be remapped with --path SRC=DST.
Exit code: llm_bench's for the scored step (non-zero also when the mock or a step fails to start).
"""

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _flag(args: list, name: str):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if isinstance(a, str) and a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def plan_steps(plan: dict) -> list:
    """[(name, concurrency, rate, duration seconds, extra args list)] in the plan's order."""
    steps = []
    for name, create in plan["traffic_generator_create_args"].items():
        extra = shlex.split(_flag(create, "--extra-args") or "")
        duration = _flag(create, "--duration") or "0s"
        seconds = float(duration[:-1]) if duration.endswith("s") else float(duration)
        rate = _flag(create, "--qps")
        steps.append((name, int(_flag(create, "--concurrency") or 1), float(rate) if rate else None, seconds, extra))
    return steps


def scored_step(plan: dict, steps: list, wanted=None):
    if wanted:
        for step in steps:
            if step[0] == wanted:
                return step
        raise SystemExit(f"preflight: no step {wanted!r} in the plan ({', '.join(s[0] for s in steps)})")
    target = plan.get("record", {}).get("target_qps")
    paced = [s for s in steps if s[2] is not None]
    at_target = [s for s in paced if target is not None and abs(s[2] - float(target)) < 1e-6]
    if at_target:
        return at_target[-1]
    replay = [s for s in steps if "--replay-schedule" in s[4] and "--prefix-fill" not in s[4]]
    if replay:
        return replay[-1]
    if paced:
        return paced[-1]
    raise SystemExit("preflight: the plan has no paced or replay step to score")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _remap(extra: list, paths: dict) -> list:
    out = []
    for a in extra:
        for src, dst in paths.items():
            if isinstance(a, str) and src in a:
                a = a.replace(src, dst)
        out.append(a)
    return out


def _locust(port: int, step, duration_cap: float, summary: str, log: str) -> int:
    name, concurrency, rate, seconds, extra = step
    run_for = min(seconds, duration_cap) if seconds else duration_cap
    cmd = [sys.executable, "-m", "locust", "-f", os.path.join(HERE, "load_test.py"), "--headless"]
    cmd += ["-H", f"http://127.0.0.1:{port}", "--provider", "vllm", "-m", "mock", "--stream"]
    cmd += ["-u", str(concurrency), "-r", str(concurrency), "-t", f"{int(run_for)}s"]
    if rate is not None:
        cmd += ["--qps", str(rate)]
    if summary:
        cmd += ["--summary-file", summary]
    cmd += extra
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE).returncode


def quick_look(summary: str, step) -> None:
    """Designed vs delivered uncached per class (information only; task 15's check decides)."""
    try:
        with open(summary, newline="") as f:
            row = list(csv.DictReader(f))[-1]
    except (OSError, IndexError):
        print("preflight: no summary row to show")
        return
    print(
        f"applied: overhang {row.get('Reuse Overhang Tokens', '?')}, boundary {row.get('Reuse Restore Boundary', '?')}, "
        f"served model config {row.get('Served Model Config', '?')}"
    )
    mix = _flag(step[4], "--shape-mix")
    if not mix:
        return
    if mix.startswith("@"):
        with open(mix[1:]) as f:
            mix = f.read()
    print(f"{'class':22} {'designed uncached':>18} {'delivered':>10} {'ratio':>7}")
    for cls in json.loads(mix).get("classes", []):
        stem = "Class " + " ".join(w.capitalize() for w in cls["name"].split("_"))
        prompt, cached = row.get(f"{stem} Prompt Tokens", ""), row.get(f"{stem} Cached Tokens", "")
        if prompt in ("", None) or cached in ("", None):
            continue
        designed = float(cls["prompt_tokens"]) - float(cls.get("cached_prompt_tokens") or 0)
        delivered = float(prompt) - float(cached)
        print(f"{cls['name']:22} {designed:18.1f} {delivered:10.1f} {delivered / designed if designed else 0:7.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan")
    parser.add_argument("--out", required=True, help="Directory for the step summary and logs.")
    parser.add_argument("--step", default=None, help="The step to score (default: the one at the target rate).")
    parser.add_argument("--duration", type=float, default=180.0, help="Cap on the scored step's seconds.")
    parser.add_argument("--fill-duration", type=float, default=3600.0, help="Cap on the fill step's seconds.")
    parser.add_argument("--window", type=int, default=None, help="Engine recompute (default: the plan's probe).")
    parser.add_argument("--boundary", type=int, default=None, help="Restore block (default: the plan's probe).")
    parser.add_argument("--mock-tokenizer", default=None, help="The served model's tokenizer for the engine.")
    parser.add_argument("--path", action="append", default=[], help="SRC=DST remap for paths in the extra args.")
    args = parser.parse_args(argv)

    with open(args.plan) as f:
        plan = json.load(f)
    record = plan.get("record", {})
    paths = dict(p.split("=", 1) for p in args.path)
    steps = [(n, c, r, s, _remap(e, paths)) for n, c, r, s, e in plan_steps(plan)]
    step = scored_step(plan, steps, args.step)
    fills = [s for s in steps if "--prefix-fill" in s[4]]
    probe = record.get("reuse_probe") or {}
    window = args.window if args.window is not None else int(probe.get("overhang") or 0)
    boundary = args.boundary if args.boundary is not None else int(probe.get("boundary") or 0) or 1
    tokenizer = (
        args.mock_tokenizer
        or record.get("served_model_config")
        or _flag(step[4], "--served-model-config")
        or _flag(step[4], "--tokenizer")
        or "NousResearch/Meta-Llama-3.1-8B-Instruct"
    )
    if not probe and args.window is None:
        print("preflight: !!! the plan records no reuse_probe; the mock engine reuses token-exact with no window")
    os.makedirs(args.out, exist_ok=True)
    port = _free_port()
    server_log = os.path.join(args.out, "mock_server.log")
    with open(server_log, "w") as f:
        server = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_server.py"), "--port", str(port), "--window", str(window)]
            + ["--boundary", str(boundary), "--tokenizer", tokenizer, "--log", os.path.join(args.out, "mock.requests")],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    try:
        for _ in range(120):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if server.poll() is not None:
                    print(f"preflight: the mock server exited (see {server_log})")
                    return 2
                time.sleep(1)
        print(f"preflight: mock engine window {window}, boundary {boundary}, tokenizer {tokenizer}, port {port}")
        for fill in fills:
            print(f"preflight: fill step {fill[0]}")
            rc = _locust(port, fill, args.fill_duration, "", os.path.join(args.out, f"{fill[0]}.log"))
            if rc:
                print(f"preflight: fill step {fill[0]} exited {rc} (see its log)")
                return rc
        summary = os.path.join(args.out, f"{step[0]}.summary.csv")
        print(f"preflight: scored step {step[0]} (rate {step[2]}, concurrency {step[1]}, cap {args.duration:.0f} s)")
        rc = _locust(port, step, args.duration, summary, os.path.join(args.out, f"{step[0]}.log"))
        print(f"preflight: step summary {summary} (llm_bench exit {rc})")
        quick_look(summary, step)
        return rc
    finally:
        server.terminate()


if __name__ == "__main__":
    sys.exit(main())
