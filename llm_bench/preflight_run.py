"""D-117's pre-flight runner: run a tier plan's llm_bench steps against mock_server.py on CPU, and write the
step summary that task 15's per-class D-63 check reads.

    python preflight_run.py PLAN.json --out DIR [--step ramp-3] [--duration S] [--mock-tokenizer ID] [--check]

What it runs, from the plan as task 07 writes it (record + traffic_generator_create_args):
- the mock engine, emulating the plan's measured reuse: record.reuse_probe's overhang as the recomputed window
  and its boundary as the restore block (D-111/D-113; --window/--boundary override), tokenizing with the served
  model's tokenizer (record.served_model_config, else the step's --served-model-config or --tokenizer);
- the plan's fill step (any step whose extra args carry --prefix-fill), then the scored step: --step, or the
  step at the plan's target rate (the last one whose --qps equals record.target_qps), with the plan's exact extra
  args, its concurrency and rate, and the duration capped at --duration;
- the scored step's summary goes to DIR/<step>.summary.csv (the file task 15's check reads), its log beside it.

It prints the applied reuse semantics and a per-class designed vs delivered table for a quick look, and the step
summary's path as its last line. The verdict is task 15's check (D-117): --check runs it after the run, as
`traffic_profile_logs preflight-check PLAN SUMMARY --step STEP`, and exits with its code (0 pass, 1 fail,
2 insufficient rows). Paths in the extra args that don't exist locally can be remapped with --path SRC=DST.
The layout must match the mock: every step's overhang (an explicit --prefix-reuse-overhang, else what llm_bench
resolves from the served model's config) must equal the mock's window, less its chat head (--mock-chat-head) in chat
mode, or the run is refused (exit 2) before it starts, and the verdict is refused if the step's summary reports
otherwise. A mismatch would make the verdict measure the mock, not the plan; --allow-mismatch runs it anyway for
deliberate experiments, labelled in the output and in DIR/MISMATCH_EXPERIMENT, and exits 3 whatever the verdict
(an experiment is never a plan result). With --mock-chat-head and the plan's probe as the window, a chat probe's
overhang (already net of the engine's head) gets the head added back for the mock's raw window.
Every step it runs must name its wire mode (--chat or --no-chat; llm_bench requires one), and every request the
mock logs for a step must hit that mode's path (/v1/chat/completions or /v1/completions), else the run fails.
Exit code: the check's with --check, else llm_bench's for the scored step (non-zero also when the mock or a step
fails to start, 2 when a step names no mode or the layout doesn't match the mock, 1 on a path mismatch, 3 for a
--allow-mismatch experiment).
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
# llm_bench's default counting tokenizer (without --tokenizer)
DEFAULT_TOKENIZER = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def _flag(args: list, name: str):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if isinstance(a, str) and a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


MODE_PATHS = {"--chat": "/v1/chat/completions", "--no-chat": "/v1/completions"}


def step_path(step):
    """The path a step's requests must hit, from its explicit --chat/--no-chat (None: no mode named; the last
    one named wins, as in argparse)."""
    modes = [a for a in step[4] if a in MODE_PATHS]
    return MODE_PATHS[modes[-1]] if modes else None


def path_mismatches(request_log: str, start_line: int, expected: str) -> tuple:
    """(requests logged since start_line, the paths among them that aren't `expected`, with counts)."""
    seen, wrong = 0, {}
    try:
        with open(request_log) as f:
            lines = f.read().splitlines()[start_line:]
    except OSError:
        return 0, {}
    for line in lines:
        fields = line.split()
        path = fields[4] if len(fields) > 4 else "?"
        seen += 1
        if path.split("?", 1)[0] != expected:
            wrong[path] = wrong.get(path, 0) + 1
    return seen, wrong


def expected_overhang(step, window: int, mock_chat_head: int) -> int:
    """The layout overhang that makes a step's prefixes reuse exactly their designed share on this mock: the window
    it recomputes, less (in chat) the head it renders ahead of every prompt, which is shared and so pre-pays part
    of the window."""
    return window - (mock_chat_head if step_path(step) == MODE_PATHS["--chat"] else 0)


def applied_reuse(step) -> tuple:
    """(overhang, restore boundary) llm_bench will apply for a step, resolved by llm_bench itself (the explicit
    --prefix-reuse-overhang / --prefix-restore-boundary, else the served model's config), in a subprocess so this
    process stays unpatched."""
    overhang, boundary = _flag(step[4], "--prefix-reuse-overhang"), _flag(step[4], "--prefix-restore-boundary")
    if overhang is not None and boundary is not None:
        return int(overhang), int(boundary)
    ns = (
        f"argparse.Namespace(prefix_reuse_overhang=None, prefix_restore_boundary=None, "
        f"served_model_config={_flag(step[4], '--served-model-config')!r}, tokenizer={_flag(step[4], '--tokenizer')!r})"
    )
    code = (
        "import argparse, sys; sys.argv = ['x']; import load_test as lt; "
        f"ns = {ns}; print(lt.resolve_prefix_reuse_overhang(ns), lt.resolve_prefix_restore_boundary(ns))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    if out.returncode or not lines:
        raise RuntimeError(f"couldn't resolve step {step[0]}'s reuse semantics: {out.stderr.strip()[-300:]}")
    resolved_overhang, resolved_boundary = (int(x) for x in lines[-1].split())
    return (
        int(overhang) if overhang is not None else resolved_overhang,
        int(boundary) if boundary is not None else resolved_boundary,
    )


def applied_overhang(step) -> int:
    """The overhang llm_bench will apply for a step (see applied_reuse)."""
    return applied_reuse(step)[0]


def tokenizer_fingerprint(name: str):
    """sha256 of a tokenizer's tokenizer.json (a local directory, else the cached or downloaded HF file), so two names
    for the same tokenizer compare equal and two tokenizers sharing a name don't; None when it can't be read."""
    import hashlib

    try:
        path = os.path.join(name, "tokenizer.json")
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=name, filename="tokenizer.json")
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def tokenizer_mismatch(counting: str, mock: str):
    """A warning when the steps count tokens with a different tokenizer than the mock tokenizes with, else None.
    Compared by content (tokenizer.json); by name, said so, when either can't be read."""
    if counting == mock:
        return None
    a, b = tokenizer_fingerprint(counting), tokenizer_fingerprint(mock)
    if a is not None and b is not None:
        if a == b:
            return None
        how = "different tokenizer.json"
    else:
        how = "names differ; tokenizer.json unreadable, so compared by name only"
    return (
        f"the mock tokenizes with {mock} but the step counts with {counting} ({how}): token and cached counts in "
        f"the summary come from the mock's tokenizer (pass --tokenizer {mock} in the steps, or --mock-tokenizer "
        f"{counting})"
    )


def text_step(step) -> bool:
    """A step that sends text requests (not --embeddings, --rerank or --systemone): the ones with reuse columns."""
    return not any(flag in step[4] for flag in ("--embeddings", "--rerank", "--systemone"))


def _token_exact(boundary) -> int:
    """llm_bench's boundary 0 and the mock's 1 both mean token-exact reuse."""
    return 0 if int(boundary) <= 1 else int(boundary)


def boundary_message(step, applied: int, boundary: int) -> str:
    return (
        f"step {step[0]} lays prefixes out for restore boundary {applied}, but the mock restores in blocks of "
        f"{boundary}: the verdict would score the mock's rounding, not the plan. Run the mock with --boundary "
        f"{applied or 1} (or, only if the step's boundary is itself wrong, pass --prefix-restore-boundary {boundary} "
        f"in the step); --allow-mismatch runs it anyway, labelled"
    )


def probe_mode(probe: dict):
    """The wire mode a recorded reuse probe ran in ("chat" or "completions"), from its mode or source label."""
    mode = probe.get("mode")
    if mode in ("chat", "completions"):
        return mode
    source = str(probe.get("source") or "")
    return "chat" if "chat" in source else "completions" if "completions" in source else None


def mock_window(probe: dict, window_arg, mock_chat_head: int):
    """The mock's raw recompute window: --window, else the plan's probe overhang. A chat probe's overhang is already
    net of the engine's head, so with --mock-chat-head the head is added back; None when the probe's mode is
    unknown and a head is given (ambiguous)."""
    if window_arg is not None:
        return window_arg
    window = int(probe.get("overhang") or 0)
    if probe and mock_chat_head:
        mode = probe_mode(probe)
        if mode is None:
            return None
        if mode == "chat":
            window += mock_chat_head
    return window


def mismatch_message(step, applied: int, expected: int, window: int, mock_chat_head: int) -> str:
    chat = expected != window
    head = f" less its chat head {mock_chat_head}" if chat else ""
    # The step's overhang is the engine-measured value (the probe's): fit the mock to it, not the plan to the mock
    fix_window = applied + (mock_chat_head if chat else 0)
    return (
        f"step {step[0]} lays prefixes out with overhang {applied}, but the mock recomputes {window}{head}, so it "
        f"matches only {expected}: the delivered cached share would be off by {applied - expected} tokens per hit, a "
        f"mock artifact, not a plan result. Run the mock to match the layout: --window {fix_window}"
        + (f" --mock-chat-head {mock_chat_head}" if chat else "")
        + f" (or, only if the step's overhang is itself wrong, pass --prefix-reuse-overhang {expected} in the step); "
        f"--allow-mismatch runs it anyway, labelled"
    )


def set_aside(path: str):
    """Move an existing file to <path>.prev (or .prev2, .prev3, ... if taken), never overwriting; the new name, or
    None when there was nothing to move."""
    if not os.path.exists(path):
        return None
    target, n = f"{path}.prev", 1
    while os.path.exists(target):
        n += 1
        target = f"{path}.prev{n}"
    os.rename(path, target)
    return target


def _mark_experiment(out: str, messages: list) -> None:
    with open(os.path.join(out, "MISMATCH_EXPERIMENT"), "w") as f:
        f.write("\n".join(messages) + "\n")


def _log_lines(request_log: str) -> int:
    try:
        with open(request_log) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


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


def _locust(port: int, step, duration_cap, summary: str, log: str) -> int:
    name, concurrency, rate, seconds, extra = step
    if duration_cap is None:
        run_for = seconds or 180.0
    else:
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


def summary_mismatch(summary: str, step, window: int, mock_chat_head: int, mock_tokenizer: str, boundary: int = 1):
    """The applied overhang and restore boundary (and, when the client counts with the mock's tokenizer, the measured
    chat head) from the step's summary, against the mock; a message when they disagree or can't be read (a summary
    that can't vouch for its layout can't be scored), else None. A missing summary is left to the caller."""
    if not os.path.exists(summary):
        return None
    try:
        with open(summary, newline="") as f:
            row = list(csv.DictReader(f))[-1]
    except (OSError, IndexError, csv.Error) as e:
        return (
            f"step {step[0]}'s summary {summary} can't be read ({e!r}), so its layout can't be checked against the mock"
        )
    if not text_step(step):  # embeddings, rerank and systemone summaries carry no reuse columns
        return None
    expected = expected_overhang(step, window, mock_chat_head)
    applied = row.get("Reuse Overhang Tokens")
    if applied in (None, ""):
        return f"step {step[0]}'s summary has no Reuse Overhang Tokens, so its layout can't be checked against the mock"
    if int(float(applied)) != expected:
        return mismatch_message(step, int(float(applied)), expected, window, mock_chat_head)
    applied_boundary = row.get("Reuse Restore Boundary")
    if applied_boundary in (None, ""):
        return (
            f"step {step[0]}'s summary has no Reuse Restore Boundary, so its layout can't be checked against the mock"
        )
    if applied_boundary not in (None, "") and _token_exact(float(applied_boundary)) != _token_exact(boundary):
        return boundary_message(step, int(float(applied_boundary)), boundary)
    head = row.get("Chat Head Tokens")
    counting = _flag(step[4], "--tokenizer")
    if head not in (None, "") and counting == mock_tokenizer and int(float(head)) != mock_chat_head:
        return f"step {step[0]} measured a chat head of {head} tokens on a mock that renders {mock_chat_head}"
    return None


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
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Cap on the scored step's seconds (default: the plan's own duration, so gated classes reach their rows).",
    )
    parser.add_argument("--fill-duration", type=float, default=3600.0, help="Cap on the fill step's seconds.")
    parser.add_argument("--window", type=int, default=None, help="Engine recompute (default: the plan's probe).")
    parser.add_argument("--boundary", type=int, default=None, help="Restore block (default: the plan's probe).")
    parser.add_argument("--mock-tokenizer", default=None, help="The served model's tokenizer for the engine.")
    parser.add_argument(
        "--mock-chat-head",
        type=int,
        default=0,
        help="Tokens the mock engine renders ahead of every chat prompt (DeepSeek V4.1: 26); 0 renders as the client.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Run even when a step's layout overhang doesn't match the mock (a deliberate mismatch experiment; the "
        "output and DIR/MISMATCH_EXPERIMENT say so).",
    )
    parser.add_argument("--path", action="append", default=[], help="SRC=DST remap for paths in the extra args.")
    parser.add_argument(
        "--replay-acceptance-per-row",
        action="store_true",
        default=False,
        help="Forwarded to the steps: send each row's recorded acceptance (matched_per_pos/drafted_per_pos).",
    )
    parser.add_argument(
        "--check",
        nargs="?",
        const="python -m fireworks.perfagent.estimate.traffic_profile_logs preflight-check",
        default=None,
        help="After the run, invoke task 15's per-class check (D-117) as `<command> PLAN SUMMARY --step STEP` and "
        "exit with its code (0 pass, 1 fail, 2 insufficient). Bare --check uses traffic_profile_logs "
        "preflight-check; pass a different command to override.",
    )
    args = parser.parse_args(argv)

    with open(args.plan) as f:
        plan = json.load(f)
    record = plan.get("record", {})
    paths = dict(p.split("=", 1) for p in args.path)
    steps = [(n, c, r, s, _remap(e, paths)) for n, c, r, s, e in plan_steps(plan)]
    if args.replay_acceptance_per_row:
        for st in steps:
            if "--replay-schedule" in st[4] and "--replay-acceptance-per-row" not in st[4]:
                st[4].append("--replay-acceptance-per-row")
    step = scored_step(plan, steps, args.step)
    fills = [s for s in steps if "--prefix-fill" in s[4]]
    probe = record.get("reuse_probe") or {}
    window = mock_window(probe, args.window, args.mock_chat_head)
    if window is None:
        print(
            "preflight: !!! --mock-chat-head with a reuse_probe of unknown mode: pass --window (the engine's raw "
            "window; a chat probe's overhang is net of the head)"
        )
        return 2
    boundary = args.boundary if args.boundary is not None else int(probe.get("boundary") or 0) or 1
    tokenizer = (
        args.mock_tokenizer
        or record.get("served_model_config")
        or _flag(step[4], "--served-model-config")
        or _flag(step[4], "--tokenizer")
        or DEFAULT_TOKENIZER
    )
    unnamed = [s[0] for s in fills + [step] if step_path(s) is None]
    if unnamed:
        print(
            f"preflight: !!! step(s) {', '.join(unnamed)} name no wire mode: pass --chat or --no-chat in their extra args"
        )
        return 2
    if not probe and args.window is None:
        print("preflight: !!! the plan records no reuse_probe; the mock engine reuses token-exact with no window")
    os.makedirs(args.out, exist_ok=True)
    # The layout and the mock must agree, or the verdict measures the mismatch instead of the plan
    mismatched = []
    for s in fills + [step]:
        (applied, applied_boundary), expected = applied_reuse(s), expected_overhang(s, window, args.mock_chat_head)
        if applied != expected:
            mismatched.append(mismatch_message(s, applied, expected, window, args.mock_chat_head))
        if _token_exact(applied_boundary) != _token_exact(boundary):
            mismatched.append(boundary_message(s, applied_boundary, boundary))
    # The mock should count tokens as the client does, or the summary's token counts and the designed ones differ
    # A schedule's acceptance lists must match the replica's draft count, or every row fails on the server.
    # The mock is started with the same count (below), and the schedule's list width is checked against it here
    draft_width = (
        plan.get("speculative_draft_tokens")
        or plan.get("num_draft_tokens")
        or record.get("speculative_draft_tokens")
        or record.get("num_draft_tokens")
    )
    if draft_width is not None:
        draft_width = int(draft_width)
    if draft_width is not None:
        print(f"preflight: the replica's num_draft_tokens is {draft_width}; the mock enforces the override width")
        schedule_path = _flag(step[4], "--replay-schedule")
        # the step may carry the flag itself (07's runner injects it), not only the pre-flight's copy
        if schedule_path and (
            getattr(args, "replay_acceptance_per_row", False) or "--replay-acceptance-per-row" in step[4]
        ):
            widths = set()
            with open(schedule_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    drafted = row.get("drafted_per_pos")
                    if drafted is not None:
                        widths.add(len(drafted))
            if widths and widths != {draft_width}:
                print(
                    f"preflight: !!! refusing: the schedule's acceptance lists are {sorted(widths)} wide but the "
                    f"replica's num_draft_tokens is {draft_width}: every row would fail on the server"
                )
                return 2
    warned = set()
    for s in fills + [step]:
        counting = _flag(s[4], "--tokenizer") or DEFAULT_TOKENIZER
        message = tokenizer_mismatch(counting, tokenizer) if counting not in warned else None
        warned.add(counting)
        if message:
            print(f"preflight: !!! step {s[0]}: {message}")
    if mismatched and not args.allow_mismatch:
        for m in mismatched:
            print(f"preflight: !!! refusing: {m}")
        return 2
    for m in mismatched:
        print(f"preflight: !!! MISMATCH EXPERIMENT (--allow-mismatch): {m}")
    if mismatched:
        _mark_experiment(args.out, mismatched)
    port = _free_port()
    server_log = os.path.join(args.out, "mock_server.log")
    request_log = os.path.join(args.out, "mock.requests")
    with open(server_log, "w") as f:
        # The replica's draft count, when the plan records it: the mock refuses acceptance overrides of any
        # other length, as the engine does, and the schedule's list width is checked against it below
        cmd = (
            [sys.executable, os.path.join(HERE, "mock_server.py"), "--port", str(port), "--window", str(window)]
            + ["--boundary", str(boundary), "--tokenizer", tokenizer, "--log", request_log]
            + ["--chat-head", str(args.mock_chat_head)]
        )
        if draft_width is not None:
            cmd += ["--num-draft-tokens", str(draft_width)]
        server = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
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
        else:
            print(f"preflight: the mock server didn't start listening on port {port} in 120 s (see {server_log})")
            return 2
        print(
            f"preflight: mock engine window {window}, boundary {boundary}, chat head {args.mock_chat_head}, "
            f"tokenizer {tokenizer}, port {port}"
        )

        def paths_ok(s, start_line) -> bool:
            seen, wrong = path_mismatches(request_log, start_line, step_path(s))
            if not seen:
                print(f"preflight: !!! step {s[0]} logged no requests on the mock (expected {step_path(s)})")
                return False
            if wrong:
                print(
                    f"preflight: !!! step {s[0]} expects {step_path(s)} but {sum(wrong.values())} of {seen} requests hit {wrong}"
                )
                return False
            print(f"preflight: step {s[0]}: {seen} requests, all on {step_path(s)}")
            return True

        for fill in fills:
            print(f"preflight: fill step {fill[0]}")
            start_line = _log_lines(request_log)
            rc = _locust(port, fill, args.fill_duration, "", os.path.join(args.out, f"{fill[0]}.log"))
            if rc:
                print(f"preflight: fill step {fill[0]} exited {rc} (see its log)")
                return rc
            if not paths_ok(fill, start_line):
                return 1
        summary = os.path.join(args.out, f"{step[0]}.summary.csv")
        # llm_bench appends a row per run: a reused --out dir would hold two runs' rows, so a reader taking the wrong
        # one scores the old run. An earlier summary is moved aside (never deleted)
        moved = set_aside(summary)
        if moved:
            print(f"preflight: an earlier {os.path.basename(summary)} is kept as {os.path.basename(moved)}")
        run_for = step[3] if args.duration is None else min(step[3] or args.duration, args.duration)
        print(f"preflight: scored step {step[0]} (rate {step[2]}, concurrency {step[1]}, {run_for:.0f} s)")
        start_line = _log_lines(request_log)
        rc = _locust(port, step, args.duration, summary, os.path.join(args.out, f"{step[0]}.log"))
        print(f"preflight: step summary {summary} (llm_bench exit {rc})")
        if not paths_ok(step, start_line):
            print(summary)
            return 1
        quick_look(summary, step)
        # Backstop: what the step's summary says it applied (the authority), against the mock
        verdict = summary_mismatch(summary, step, window, args.mock_chat_head, tokenizer, boundary)
        if verdict and not mismatched:
            print(
                f"preflight: !!! {'MISMATCH EXPERIMENT' if args.allow_mismatch else 'refusing the verdict'}: {verdict}"
            )
            if not args.allow_mismatch:
                print(summary)
                return 2
            mismatched.append(verdict)
            _mark_experiment(args.out, mismatched)
        if args.check:
            if not os.path.exists(summary):
                print("preflight: no summary to check")
                print(summary)
                return rc or 1
            cmd = shlex.split(args.check) + [args.plan, summary, "--step", step[0]]
            print(f"preflight: check: {' '.join(cmd)}", flush=True)
            locust_rc, rc = rc, subprocess.run(cmd).returncode
            if locust_rc:
                # a step that failed can leave a partial summary the check passes: the step's failure wins
                print(f"preflight: !!! llm_bench exited {locust_rc}: that wins over the check's {rc}")
                rc = locust_rc
        if mismatched:
            # Never a plan verdict: a driver reading only the exit code mustn't take an experiment for a pass
            print(f"preflight: !!! MISMATCH EXPERIMENT: verdict exit {rc} is not a plan result; exiting 3")
            rc = 3
        print(summary)  # last line: the step summary's path (the check's input)
        return rc
    finally:
        server.terminate()


if __name__ == "__main__":
    sys.exit(main())
