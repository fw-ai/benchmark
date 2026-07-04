#!/usr/bin/env python3
"""Estimate the prefix-cache hit rate for a captured session-replay dataset.

Reads the per-conversation ``manifest.json`` files (numPromptTokens /
numCompletionTokens per turn -- no body decompression) and computes the
token-weighted prefix-cache hit rate under a PERFECT (from-cold) prefix cache:

  * turn 0 of every conversation is cold (0% cached),
  * every later turn reuses the maximal already-computed prefix, which -- because
    each turn re-sends the full history append-only -- is the previous turn's
    prompt PLUS its generated response (both resident in the KV cache), capped at
    the current prompt length (handles context trimming where the prompt shrank).

Overall hit rate = sum(cached) / sum(prompt) over all turns. This is the
theoretical ceiling for a cold replay: turn-0 prefills and single-turn
conversations can never be cached, so the ceiling is < 100%.

A live run's reported cache hit ABOVE this ceiling implies cross-conversation /
cross-step cache residue (not a true cold replay); at/below means turn-0
prefills are landing cold as expected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def scan(conv_dir: str, target: str):
    """Return (prompts, completions) for target turns of one conversation, ordered."""
    mp = os.path.join(conv_dir, "manifest.json")
    if not os.path.exists(mp):
        return None
    try:
        man = json.load(open(mp))
    except Exception:
        return None
    fs = [
        e
        for e in man.get("files", [])
        if target in str(e.get("model", "")) and e.get("numPromptTokens") is not None
    ]
    if not fs:
        return None
    P = [int(e["numPromptTokens"]) for e in fs]
    C = [int(e.get("numCompletionTokens") or 0) for e in fs]
    return P, C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/shared/request-replay-data/matterhorn-fp4-5")
    ap.add_argument("--target-substr", default="matterhorn-fp4-5")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--model",
        choices=["perfect", "prompt-only"],
        default="perfect",
        help="'perfect' (default): cached = prev prompt + prev response. "
        "'prompt-only': cached = prev prompt only (ignore the generated response).",
    )
    ap.add_argument("--by-turn-index", type=int, default=8, metavar="N",
                    help="Also print per-turn-index hit rate for the first N indices (0 disables).")
    args = ap.parse_args()

    conv_dirs = sorted(
        os.path.join(args.src, d) for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))
    )
    if args.limit is not None:
        conv_dirs = conv_dirs[: args.limit]

    tot_prompt = tot_cached = cold_turn0 = 0
    n_turns = n_conv = n_single = 0
    idx_prompt: dict[int, int] = defaultdict(int)
    idx_cached: dict[int, int] = defaultdict(int)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(scan, d, args.target_substr) for d in conv_dirs]):
            res = fut.result()
            if not res:
                continue
            P, C = res
            n_conv += 1
            if len(P) == 1:
                n_single += 1
            for i, p in enumerate(P):
                tot_prompt += p
                n_turns += 1
                idx_prompt[i] += p
                if i == 0:
                    cold_turn0 += p
                    cached = 0
                elif args.model == "perfect":
                    cached = min(P[i - 1] + C[i - 1], p)
                else:  # prompt-only
                    cached = min(P[i - 1], p)
                tot_cached += cached
                idx_cached[i] += cached

    if tot_prompt == 0:
        print("no data")
        return 1

    print(f"===== PREFIX-CACHE HIT ESTIMATE ({args.model}, from cold) =====")
    print(f"conversations:              {n_conv:,}  (single-turn: {n_single:,} = {100*n_single/n_conv:.1f}%)")
    print(f"turns:                      {n_turns:,}")
    print(f"total prompt tokens:        {tot_prompt:,}")
    print(f"cold turn-0 tokens:         {cold_turn0:,}  ({100*cold_turn0/tot_prompt:.1f}% of all prompt)")
    print(f"cached tokens:              {tot_cached:,}")
    print(f"uncached tokens:            {tot_prompt-tot_cached:,}  (irreducible prefill compute)")
    print(f"TOKEN-WEIGHTED HIT RATE:    {100*tot_cached/tot_prompt:.2f}%")
    if args.by_turn_index:
        print("\nper-turn-index hit rate:")
        for i in sorted(idx_prompt)[: args.by_turn_index]:
            hr = 100 * idx_cached[i] / idx_prompt[i] if idx_prompt[i] else 0.0
            print(f"  turn {i:>2}: {hr:5.1f}%   (prompt tokens {idx_prompt[i]:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
