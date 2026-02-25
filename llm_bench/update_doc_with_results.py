#!/usr/bin/env python3
"""
Parse load test logs and output markdown for the Google Doc.
Usage:
  python update_doc_with_results.py eagle   # Eagle results
  python update_doc_with_results.py nodraft # NoDraft results (when ready)
  python update_doc_with_results.py vllm    # vLLM v6 results
  python update_doc_with_results.py both    # Eagle + NoDraft
  python update_doc_with_results.py all     # Eagle + NoDraft + vLLM
Output is printed to stdout. Copy-paste at the bottom of:
  https://docs.google.com/document/d/1wlkryDkAlAperola_hE-jZj8H1MFFXlxBGCZE-kMNgk
"""
import re
import sys

CONFIG_NAMES = {
    "256/128": "Short Output",
    "256/512": "Medium Output",
    "256/1024": "Long Output",
    "1024/256": "Atlassian-like",
    "1024/1024": "Long I/O",
    "2048/256": "Long Prompt",
}
CONFIGS_ORDER = ["256/128", "256/512", "256/1024", "1024/256", "1024/1024", "2048/256"]


def parse_log(path):
    with open(path) as f:
        text = f.read()
    configs = re.findall(r"=== Config input=(\d+) output=(\d+) concurrency=(\d+) ===", text)
    qps = re.findall(r"Qps\s+:\s+([\d.]+)", text)
    ttft = re.findall(r"P99 Time To First Token\s+:\s+([\d.]+)", text)
    # P99 TPOT from the locust percentile table for latency_per_token
    # Columns: p50 p66 p75 p80 p90 p95 p98 p99 p99.9 p99.99 p100
    tpot_pattern = r'METRIC\s+latency_per_token\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+'
    tpot_matches = re.findall(tpot_pattern, text)
    tpot_p99 = [m[7] for m in tpot_matches]  # index 7 = p99
    n = min(len(configs), len(qps), len(ttft), len(tpot_p99))
    return [(configs[i], qps[i], ttft[i], tpot_p99[i]) for i in range(n)]


def build_markdown(name, log_path, run_id, note=""):
    rows = parse_log(log_path)
    data = {}
    for (p, o, c), q, t, tp in rows:
        cfg = f"{p}/{o}"
        if cfg not in data:
            data[cfg] = []
        data[cfg].append((int(c), float(q), float(t), float(tp)))

    lines = [
        "---",
        f"## {name} Load Test Results (Feb 23, 2026)",
        "",
        f"**Run:** `{run_id}` | {len(rows)}/48 runs completed" + (f" {note}" if note else ""),
        "",
        "### Summary at Concurrency = 16",
        "",
        "| Config | QPS | P99 TTFT (ms) | P99 TPOT (ms) |",
        "|--------|-----|---------------|---------------|",
    ]
    for cfg in CONFIGS_ORDER:
        rows_cfg = sorted(data.get(cfg, []), key=lambda x: x[0])
        c16 = next((r for r in rows_cfg if r[0] == 16), None)
        if c16:
            c, q, t, tp = c16
            lines.append(f"| {cfg} ({CONFIG_NAMES[cfg]}) | {q:.2f} | {t:.0f} | {tp:.0f} |")
        else:
            lines.append(f"| {cfg} ({CONFIG_NAMES[cfg]}) | — | — | — |")

    lines.extend(["", "### Full Data by Config", ""])
    for cfg in CONFIGS_ORDER:
        rows_cfg = sorted(data.get(cfg, []), key=lambda x: x[0])
        lines.append(f"**{cfg} ({CONFIG_NAMES[cfg]})**")
        lines.append("")
        lines.append("| Conc | QPS | P99 TTFT (ms) | P99 TPOT (ms) |")
        lines.append("|------|-----|---------------|---------------|")
        for c, q, t, tp in rows_cfg:
            lines.append(f"| {c} | {q:.2f} | {t:.0f} | {tp:.0f} |")
        lines.append("")

    return "\n".join(lines)


def find_run_id(log_path):
    """Extract the results directory path from the log (e.g. results/production_load_test_fireworks_YYYYMMDD_HHMM)."""
    with open(log_path) as f:
        text = f.read()
    m = re.search(r"(results/production_load_test_fireworks_\d{8}_\d{4})", text)
    return m.group(1) if m else "unknown"


VARIANTS = {
    "eagle": ("Eagle3-FA", "results/run_eagle3_fa.log"),
    "nodraft": ("NoDraft", "results/run_nodraft.log"),
    "vllm": ("vLLM v6", "results/run_vllm.log"),
}

MODE_GROUPS = {
    "both": ["eagle", "nodraft"],
    "all": ["eagle", "nodraft", "vllm"],
}


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "eagle").lower()
    keys = MODE_GROUPS.get(mode, [mode])
    out = []

    for key in keys:
        if key not in VARIANTS:
            print(f"Unknown variant: {key}. Choose from: {', '.join(VARIANTS)}", file=sys.stderr)
            sys.exit(1)
        name, log_path = VARIANTS[key]
        try:
            run_id = find_run_id(log_path)
            out.append(build_markdown(name, log_path, run_id))
        except FileNotFoundError:
            print(f"{name} log not found at {log_path}. Run load test first.", file=sys.stderr)
            sys.exit(1)

    print("\n\n".join(out))


if __name__ == "__main__":
    main()
