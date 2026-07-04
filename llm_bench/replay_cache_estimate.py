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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class ConversationTurns:
    """Per-turn token counts for one conversation, in turn order."""

    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)


@dataclass
class CacheTotals:
    """Running aggregates across the whole dataset."""

    conversations: int = 0
    single_turn_conversations: int = 0
    turns: int = 0
    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0
    cold_turn0_tokens: int = 0
    prompt_tokens_by_turn: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    cached_tokens_by_turn: dict[int, int] = field(default_factory=lambda: defaultdict(int))


def read_conversation_turns(conversation_dir: str, target_substr: str) -> ConversationTurns | None:
    """Read one conversation's per-turn prompt/completion token counts from its manifest.

    Keeps only turns whose model matches ``target_substr`` (drops the per-request
    classifier turns), preserving turn order. Returns None if there are no
    matching turns.
    """
    manifest_path = os.path.join(conversation_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        manifest = json.load(open(manifest_path))
    except Exception:
        return None

    turns = ConversationTurns()
    for entry in manifest.get("files", []):
        if target_substr not in str(entry.get("model", "")):
            continue
        if entry.get("numPromptTokens") is None:
            continue
        turns.prompt_tokens.append(int(entry["numPromptTokens"]))
        turns.completion_tokens.append(int(entry.get("numCompletionTokens") or 0))

    return turns if turns.prompt_tokens else None


def cached_tokens_for_turn(turns: ConversationTurns, turn_index: int, model: str) -> int:
    """Tokens served from cache for one turn under a perfect from-cold prefix cache.

    Turn 0 is always cold (0). For later turns the reusable prefix is what was
    already computed on the previous turn:
      * ``perfect``     -> previous prompt + previous response (both in KV cache)
      * ``prompt-only`` -> previous prompt only (ignore the generated response)
    capped at this turn's prompt length (context may have been trimmed).
    """
    if turn_index == 0:
        return 0
    prompt = turns.prompt_tokens[turn_index]
    prev_prompt = turns.prompt_tokens[turn_index - 1]
    if model == "perfect":
        prev_response = turns.completion_tokens[turn_index - 1]
        reusable_prefix = prev_prompt + prev_response
    else:  # prompt-only
        reusable_prefix = prev_prompt
    return min(reusable_prefix, prompt)


def accumulate(totals: CacheTotals, turns: ConversationTurns, model: str) -> None:
    """Fold one conversation's turns into the running totals."""
    totals.conversations += 1
    if len(turns.prompt_tokens) == 1:
        totals.single_turn_conversations += 1

    for turn_index, prompt in enumerate(turns.prompt_tokens):
        totals.turns += 1
        totals.total_prompt_tokens += prompt
        totals.prompt_tokens_by_turn[turn_index] += prompt

        cached = cached_tokens_for_turn(turns, turn_index, model)
        totals.total_cached_tokens += cached
        totals.cached_tokens_by_turn[turn_index] += cached
        if turn_index == 0:
            totals.cold_turn0_tokens += prompt


def print_report(totals: CacheTotals, model: str, by_turn_index: int) -> None:
    prompt = totals.total_prompt_tokens
    cached = totals.total_cached_tokens
    print(f"===== PREFIX-CACHE HIT ESTIMATE ({model}, from cold) =====")
    print(
        f"conversations:              {totals.conversations:,}  "
        f"(single-turn: {totals.single_turn_conversations:,} = "
        f"{100 * totals.single_turn_conversations / totals.conversations:.1f}%)"
    )
    print(f"turns:                      {totals.turns:,}")
    print(f"total prompt tokens:        {prompt:,}")
    print(f"cold turn-0 tokens:         {totals.cold_turn0_tokens:,}  ({100 * totals.cold_turn0_tokens / prompt:.1f}% of all prompt)")
    print(f"cached tokens:              {cached:,}")
    print(f"uncached tokens:            {prompt - cached:,}  (irreducible prefill compute)")
    print(f"TOKEN-WEIGHTED HIT RATE:    {100 * cached / prompt:.2f}%")

    if by_turn_index:
        print("\nper-turn-index hit rate:")
        for turn_index in sorted(totals.prompt_tokens_by_turn)[:by_turn_index]:
            turn_prompt = totals.prompt_tokens_by_turn[turn_index]
            turn_cached = totals.cached_tokens_by_turn[turn_index]
            hit_rate = 100 * turn_cached / turn_prompt if turn_prompt else 0.0
            print(f"  turn {turn_index:>2}: {hit_rate:5.1f}%   (prompt tokens {turn_prompt:,})")


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
    ap.add_argument(
        "--by-turn-index",
        type=int,
        default=8,
        metavar="N",
        help="Also print per-turn-index hit rate for the first N indices (0 disables).",
    )
    args = ap.parse_args()

    conversation_dirs = sorted(
        os.path.join(args.src, name)
        for name in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, name))
    )
    if args.limit is not None:
        conversation_dirs = conversation_dirs[: args.limit]

    totals = CacheTotals()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(read_conversation_turns, d, args.target_substr) for d in conversation_dirs]
        for future in as_completed(futures):
            turns = future.result()
            if turns is not None:
                accumulate(totals, turns, args.model)

    if totals.total_prompt_tokens == 0:
        print("no data")
        return 1

    print_report(totals, args.model, args.by_turn_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
