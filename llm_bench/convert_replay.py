#!/usr/bin/env python3
"""Offline preprocessor for session-based request replay (llm_bench Mode B).

Turns a captured request-replay tree (as downloaded from
``s3://anysphere-fireworks-shared/request-replay-data/<deployment>/``) into a
form the load test can replay with *zero* client-side CPU on the hot path:

  * Each captured turn (``<conversation>/NNN_*.json``) is a zstd-compressed
    JSON object ``{"body": {...}, "headers": {...}, "url": ...}``.
  * We decode it once here, filter to the target deployment (dropping the
    per-request classifier turns that hit a different model), bake the target
    ``model`` and ``stream`` settings into the body, and write the *final
    ready-to-POST JSON bytes* to ``<out>/bodies/<conversation>/NNN.json``.
  * We emit ``<out>/index.jsonl`` with one line per conversation listing the
    ordered body files plus the stable session-routing headers
    (``x-session-affinity`` / ``x-multi-turn-session-id``), with
    ``authorization`` stripped.

At run time the load test only reads a body file and POSTs the bytes verbatim
plus the session headers -- no decompression, no JSON (de)serialization, no
tokenization.

Without ``--model`` the script runs in stats-only mode: it decodes and filters
but writes nothing, so you can inspect the dataset before committing to a bake.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import zstandard

# Headers we replay so the server routes a conversation's turns to the same
# replica and reuses the prefix cache, exactly as in production.
DEFAULT_SESSION_HEADERS = ("x-session-affinity", "x-multi-turn-session-id")


def _turn_sort_key(filename: str) -> tuple[int, str]:
    # Files are named like ``000_<uuid>-<n>-<hash>.json``; sort by the NNN prefix.
    stem = filename.split("_", 1)[0]
    try:
        return (int(stem), filename)
    except ValueError:
        return (1 << 30, filename)


def _decode_turn(path: str) -> Optional[dict[str, Any]]:
    """zstd-decompress + json-parse one captured turn file."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as f:
        raw = dctx.stream_reader(f).read()
    return json.loads(raw)


@dataclass
class ConversationResult:
    conversation_id: str
    session_headers: dict[str, str] = field(default_factory=dict)
    turns: list[str] = field(default_factory=list)  # relative body paths (bake mode)
    turn_prompt_msgs: list[int] = field(default_factory=list)
    turn_bytes: list[int] = field(default_factory=list)
    kept: int = 0
    skipped: int = 0
    had_tools: int = 0


def process_conversation(
    conv_dir: str,
    out_bodies_dir: Optional[str],
    model: Optional[str],
    stream: bool,
    target_substr: str,
    keep_headers: tuple[str, ...],
    max_tokens_mode: str = "recorded",
    force_length: bool = True,
) -> Optional[ConversationResult]:
    """Decode + filter + (optionally) bake one conversation directory.

    Runs in a worker process. When ``out_bodies_dir`` is None this is
    stats-only and no files are written.

    ``max_tokens_mode``:
      * ``recorded`` -- set each turn's max_tokens to the recorded
        numCompletionTokens (from manifest.json), so decode length matches prod.
        With ``force_length`` also set min_tokens + ignore_eos to force exactly
        that many tokens (the served checkpoint may differ from the capture and
        never emit EOS, otherwise running to the captured max_tokens).
      * ``keep`` -- leave the captured max_tokens untouched.
    """
    conversation_id = os.path.basename(conv_dir.rstrip("/"))
    files = sorted(
        (f for f in os.listdir(conv_dir) if f.endswith(".json") and not f.startswith(".")),
        key=_turn_sort_key,
    )
    if not files:
        return None

    # Map original turn filename -> recorded numCompletionTokens (for max_tokens).
    completion_by_file: dict[str, int] = {}
    if max_tokens_mode == "recorded":
        try:
            with open(os.path.join(conv_dir, "manifest.json")) as mf:
                for e in json.load(mf).get("files", []):
                    c = e.get("numCompletionTokens")
                    if e.get("file") and c is not None:
                        completion_by_file[e["file"]] = int(c)
        except Exception:  # noqa: BLE001
            pass

    res = ConversationResult(conversation_id=conversation_id)
    conv_out_dir = None
    if out_bodies_dir is not None:
        conv_out_dir = os.path.join(out_bodies_dir, conversation_id)

    for fname in files:
        src = os.path.join(conv_dir, fname)
        if fname == "manifest.json":
            continue
        try:
            obj = _decode_turn(src)
        except Exception as e:  # noqa: BLE001 - want to skip and keep going
            print(f"WARN: failed to decode {src}: {e!r}", file=sys.stderr)
            continue

        body = obj.get("body")
        if not isinstance(body, dict):
            continue

        model_name = str(body.get("model", ""))
        if target_substr not in model_name:
            # e.g. the gemini classifier turn 0 -> not our deployment.
            res.skipped += 1
            continue

        # Capture the stable session headers from the first qualifying turn.
        if not res.session_headers:
            hdrs = obj.get("headers") or {}
            picked = {}
            for h in keep_headers:
                # header lookups are case-insensitive in HTTP; captured keys are lowercase.
                v = hdrs.get(h) or hdrs.get(h.lower()) or hdrs.get(h.title())
                if v:
                    picked[h] = v
            res.session_headers = picked

        res.turn_prompt_msgs.append(len(body.get("messages", []) or []))
        if body.get("tools"):
            res.had_tools += 1

        if out_bodies_dir is not None and model is not None:
            baked = dict(body)
            baked["model"] = model
            baked["stream"] = stream
            if stream:
                so = dict(baked.get("stream_options") or {})
                so["include_usage"] = True
                baked["stream_options"] = so
            else:
                baked.pop("stream_options", None)
            if max_tokens_mode == "recorded":
                comp = completion_by_file.get(fname)
                if comp is not None:
                    mt = max(1, comp)
                    baked["max_tokens"] = mt
                    if force_length:
                        # Force the exact recorded decode length: the served
                        # checkpoint may differ from the capture and not emit
                        # EOS, so bound + pad to reproduce prod token counts.
                        baked["min_tokens"] = mt
                        baked["ignore_eos"] = True
            data = json.dumps(baked, ensure_ascii=False).encode("utf-8")
            os.makedirs(conv_out_dir, exist_ok=True)
            idx = len(res.turns)
            rel = os.path.join("bodies", conversation_id, f"{idx:03d}.json")
            with open(os.path.join(out_bodies_dir, conversation_id, f"{idx:03d}.json"), "wb") as out:
                out.write(data)
            res.turns.append(rel)
            res.turn_bytes.append(len(data))

        res.kept += 1

    if res.kept == 0:
        return None
    return res


def _percentiles(values: list[int], ps=(0.5, 0.9, 0.99, 1.0)) -> dict[str, int]:
    if not values:
        return {}
    s = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        out[f"p{int(p*100)}"] = s[idx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--src",
        default="/shared/request-replay-data/matterhorn-fp4-5",
        help="Source replay tree (one subdir per conversation).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir for baked bodies + index.jsonl. Required unless stats-only.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Target model string to bake into each body's 'model' field. "
        "If omitted, runs stats-only (no bake, no writes).",
    )
    stream_grp = ap.add_mutually_exclusive_group()
    stream_grp.add_argument("--stream", dest="stream", action="store_true", default=True)
    stream_grp.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument(
        "--target-substr",
        default="matterhorn-fp4-5",
        help="Only keep captured turns whose body.model contains this substring.",
    )
    ap.add_argument(
        "--session-headers",
        default=",".join(DEFAULT_SESSION_HEADERS),
        help="Comma-separated header names to replay (authorization is always dropped).",
    )
    ap.add_argument(
        "--max-tokens-mode",
        choices=["recorded", "keep"],
        default="recorded",
        help="'recorded' (default): set each turn's max_tokens to the recorded numCompletionTokens so "
        "decode length matches prod. 'keep': leave the captured max_tokens (often 20000) untouched.",
    )
    force_grp = ap.add_mutually_exclusive_group()
    force_grp.add_argument("--force-length", dest="force_length", action="store_true", default=True,
                           help="(default) With --max-tokens-mode recorded, also set min_tokens + ignore_eos "
                           "to force exactly the recorded token count (served checkpoint may not emit EOS).")
    force_grp.add_argument("--no-force-length", dest="force_length", action="store_false")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    ap.add_argument("--limit", type=int, default=None, help="Process at most N conversations (debug).")
    args = ap.parse_args()

    keep_headers = tuple(h.strip() for h in args.session_headers.split(",") if h.strip())
    stats_only = args.model is None
    if stats_only:
        print("No --model given: running in STATS-ONLY mode (no bake, no writes).", file=sys.stderr)
        out_bodies_dir = None
        index_path = None
    else:
        if not args.out:
            ap.error("--out is required when --model is given (bake mode).")
        out_bodies_dir = os.path.join(args.out, "bodies")
        os.makedirs(out_bodies_dir, exist_ok=True)
        index_path = os.path.join(args.out, "index.jsonl")

    conv_dirs = sorted(
        os.path.join(args.src, d) for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))
    )
    if args.limit is not None:
        conv_dirs = conv_dirs[: args.limit]
    print(f"Found {len(conv_dirs)} conversation dirs under {args.src}", file=sys.stderr)

    total_conv = 0
    total_turns = 0
    total_skipped = 0
    total_tools = 0
    total_bytes = 0
    turns_per_conv: list[int] = []
    msgs_per_turn: list[int] = []
    body_bytes: list[int] = []
    missing_session_hdr = 0

    index_f = open(index_path, "w") if index_path else None
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    process_conversation,
                    d,
                    out_bodies_dir,
                    args.model,
                    args.stream,
                    args.target_substr,
                    keep_headers,
                    args.max_tokens_mode,
                    args.force_length,
                )
                for d in conv_dirs
            ]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                if res is None:
                    continue
                total_conv += 1
                total_turns += res.kept
                total_skipped += res.skipped
                total_tools += res.had_tools
                turns_per_conv.append(res.kept)
                msgs_per_turn.extend(res.turn_prompt_msgs)
                if res.turn_bytes:
                    body_bytes.extend(res.turn_bytes)
                    total_bytes += sum(res.turn_bytes)
                if not res.session_headers:
                    missing_session_hdr += 1
                if index_f is not None:
                    index_f.write(
                        json.dumps(
                            {
                                "conversation_id": res.conversation_id,
                                "session_headers": res.session_headers,
                                "turns": res.turns,
                            }
                        )
                        + "\n"
                    )
                if i % 1000 == 0:
                    print(f"  processed {i}/{len(conv_dirs)} dirs...", file=sys.stderr)
    finally:
        if index_f is not None:
            index_f.close()

    print("\n===== REPLAY DATASET STATS =====")
    print(f"conversations kept:      {total_conv}")
    print(f"turns kept (target):     {total_turns}")
    print(f"turns skipped (non-tgt): {total_skipped}")
    print(f"turns with tools:        {total_tools} ({100*total_tools/max(1,total_turns):.1f}%)")
    print(f"conversations missing session headers: {missing_session_hdr}")
    if turns_per_conv:
        print(f"turns/conversation:      {_percentiles(turns_per_conv)}  max={max(turns_per_conv)}")
    if msgs_per_turn:
        print(f"messages/turn:           {_percentiles(msgs_per_turn)}  max={max(msgs_per_turn)}")
    if body_bytes:
        print(f"baked body bytes:        {_percentiles(body_bytes)}  max={max(body_bytes)}")
        print(f"total baked size:        {total_bytes/1e9:.2f} GB across {len(body_bytes)} files")
    if index_path:
        print(f"\nindex written to:        {index_path}")
        print(f"bodies written under:    {out_bodies_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
