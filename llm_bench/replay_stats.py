#!/usr/bin/env python3
"""Sequence-length stats for a captured request-replay tree.

Reads the small per-conversation ``manifest.json`` files (which carry
``numPromptTokens`` / ``numCompletionTokens`` per turn) -- no body
decompression needed -- and reports seqlen distributions:

  * prompt tokens per turn      (input context length at each turn)
  * completion tokens per turn  (output length)
  * total tokens per turn       (prompt + completion)
  * per-conversation peak prompt tokens   (deepest context reached)
  * per-conversation final total tokens   (full conversation length at last turn)
  * turns per conversation

Only turns whose ``model`` matches --target-substr are counted (drops the
per-request classifier turns that hit a different model), so numbers line up
with the replay set produced by convert_replay.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def scan_conversation(conv_dir: str, target_substr: str):
    """Return per-conversation aggregates from its manifest.json.

    Result dict keys:
      prompts, completions, totals: per-turn lists (target turns only)
      peak_prompt: max prompt tokens across turns
      final_total: prompt+completion of the last target turn
      turns: number of target turns
      statuses: {status: count}
    """
    mpath = os.path.join(conv_dir, "manifest.json")
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath) as f:
            man = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: bad manifest {mpath}: {e!r}", file=sys.stderr)
        return None

    prompts: list[int] = []
    completions: list[int] = []
    totals: list[int] = []
    statuses: dict[str, int] = {}
    for entry in man.get("files", []):
        if target_substr not in str(entry.get("model", "")):
            continue
        p = entry.get("numPromptTokens")
        c = entry.get("numCompletionTokens")
        if p is None:
            continue
        p = int(p)
        c = int(c or 0)
        prompts.append(p)
        completions.append(c)
        totals.append(p + c)
        st = str(entry.get("status", "unknown"))
        statuses[st] = statuses.get(st, 0) + 1

    if not prompts:
        return None
    return {
        "prompts": prompts,
        "completions": completions,
        "totals": totals,
        "peak_prompt": max(prompts),
        "final_total": totals[-1],
        "turns": len(prompts),
        "statuses": statuses,
    }


def _summary(name: str, values: list[int]):
    if not values:
        print(f"{name:<34} (no data)")
        return
    s = sorted(values)
    n = len(s)

    def pct(p):
        return s[min(n - 1, int(round(p * (n - 1))))]

    avg = sum(s) / n
    print(
        f"{name:<34} n={n:<8} min={s[0]:<8} p50={pct(0.5):<8} "
        f"avg={avg:<10.1f} p90={pct(0.9):<8} p99={pct(0.99):<8} max={s[-1]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/shared/request-replay-data/matterhorn-fp4-5")
    ap.add_argument("--target-substr", default="matterhorn-fp4-5")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--csv", default=None, help="Optional path to write per-turn rows (conversation_id,turn,prompt,completion,total).")
    ap.add_argument("--by-turn-index", type=int, default=0, metavar="N", help="Also print prompt/response length stats grouped by turn position, for the first N turn indices.")
    args = ap.parse_args()

    conv_dirs = sorted(
        os.path.join(args.src, d) for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))
    )
    if args.limit is not None:
        conv_dirs = conv_dirs[: args.limit]
    print(f"Scanning {len(conv_dirs)} conversation manifests under {args.src}", file=sys.stderr)

    all_prompts: list[int] = []
    all_completions: list[int] = []
    all_totals: list[int] = []
    peak_prompts: list[int] = []
    final_totals: list[int] = []
    turns_per_conv: list[int] = []
    conv_token_sums: list[int] = []  # sum of all turn totals in a conversation (bytes actually moved)
    statuses: dict[str, int] = {}
    n_conv = 0

    # Per-turn-index accumulators: index -> {"prompt": [...], "completion": [...]}
    by_index: dict[int, dict[str, list[int]]] = {}

    csv_f = open(args.csv, "w") if args.csv else None
    if csv_f:
        csv_f.write("conversation_id,turn,prompt_tokens,completion_tokens,total_tokens\n")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_conversation, d, args.target_substr): d for d in conv_dirs}
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            n_conv += 1
            all_prompts.extend(res["prompts"])
            all_completions.extend(res["completions"])
            all_totals.extend(res["totals"])
            peak_prompts.append(res["peak_prompt"])
            final_totals.append(res["final_total"])
            turns_per_conv.append(res["turns"])
            conv_token_sums.append(sum(res["totals"]))
            for k, v in res["statuses"].items():
                statuses[k] = statuses.get(k, 0) + v
            if args.by_turn_index:
                for i, (p, c) in enumerate(zip(res["prompts"], res["completions"])):
                    if i >= args.by_turn_index:
                        break
                    slot = by_index.setdefault(i, {"prompt": [], "completion": []})
                    slot["prompt"].append(p)
                    slot["completion"].append(c)
            if csv_f:
                cid = os.path.basename(futs[fut].rstrip("/"))
                for i, (p, c, t) in enumerate(zip(res["prompts"], res["completions"], res["totals"])):
                    csv_f.write(f"{cid},{i},{p},{c},{t}\n")

    if csv_f:
        csv_f.close()

    print("\n===== SEQUENCE-LENGTH STATS (target turns only) =====")
    print(f"conversations: {n_conv}    turns: {len(all_prompts)}    statuses: {statuses}\n")
    print("Per-turn:")
    _summary("  prompt tokens (input seqlen)", all_prompts)
    _summary("  completion tokens (output)", all_completions)
    _summary("  total tokens (prompt+compl)", all_totals)
    print("\nPer-conversation:")
    _summary("  turns / conversation", turns_per_conv)
    _summary("  peak prompt tokens", peak_prompts)
    _summary("  final-turn total tokens", final_totals)
    _summary("  sum of turn totals", conv_token_sums)
    if args.by_turn_index and by_index:

        def _stats(vals):
            s = sorted(vals)
            n = len(s)
            return n, s[0], s[n // 2], sum(s) / n, s[min(n - 1, int(0.9 * (n - 1)))], s[-1]

        print(f"\nPer-turn-index (first {args.by_turn_index} positions):")
        print(f"  {'turn':>4}  {'convs':>7}  | {'prompt: min':>11} {'p50':>7} {'avg':>8} {'p90':>7} {'max':>7} "
              f"| {'resp: min':>9} {'p50':>5} {'avg':>7} {'p90':>6} {'max':>6}")
        for i in sorted(by_index):
            pn, pmin, pmed, pavg, p90, pmax = _stats(by_index[i]["prompt"])
            _, cmin, cmed, cavg, c90, cmax = _stats(by_index[i]["completion"])
            print(f"  {i:>4}  {pn:>7}  | {pmin:>11} {pmed:>7} {pavg:>8.0f} {p90:>7} {pmax:>7} "
                  f"| {cmin:>9} {cmed:>5} {cavg:>7.0f} {c90:>6} {cmax:>6}")

    if all_totals:
        print(f"\ngrand total tokens across all turns: {sum(all_totals):,}")
    if args.csv:
        print(f"per-turn CSV written to: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
