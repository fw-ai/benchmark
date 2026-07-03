#!/usr/bin/env python3
"""Per-turn token breakdown for a captured request-replay tree.

For every turn splits tokens into four buckets:

  * prompt     -- input context length (exact, from manifest numPromptTokens)
  * thinking   -- generated reasoning_content
  * tool_call  -- generated tool-call arguments (+ function names)
  * response   -- generated visible text content

The response side is decomposed by decompressing bodies: a turn's generated
assistant message reappears as the last assistant message of the *next* turn's
request body. We measure that message's reasoning / tool-call / text character
counts and split the EXACT reported numCompletionTokens proportionally, so the
three output buckets always sum to the real completion count.

Because the split needs the next turn, the final turn of each conversation has
no output breakdown and is excluded from the output-bucket stats (its prompt is
still counted). Prompt counts are exact; output-bucket splits are proportional
estimates of exact totals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import zstandard


def _load(path, dctx):
    try:
        with open(path, "rb") as f:
            return json.loads(dctx.stream_reader(f).read())
    except Exception:
        return None


def _text_chars(m) -> int:
    c = m.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(p["text"]) for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
    return 0


def _reasoning_chars(m) -> int:
    r = m.get("reasoning_content")
    return len(r) if isinstance(r, str) else 0


def _toolcall_chars(m) -> int:
    s = 0
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        a = fn.get("arguments")
        if isinstance(a, str):
            s += len(a)
        elif a is not None:
            s += len(json.dumps(a))
        s += len(fn.get("name", "") or "")
    return s


def scan_conversation(conv_dir: str, target_substr: str):
    """Return list of per-turn rows: (turn_index, prompt, thinking, tool, response)."""
    dctx = zstandard.ZstdDecompressor()
    mpath = os.path.join(conv_dir, "manifest.json")
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath) as f:
            man = json.load(f)
    except Exception:
        return None

    entries = [e for e in man.get("files", []) if target_substr in str(e.get("model", ""))]
    if len(entries) < 2:
        return None

    bodies = [_load(os.path.join(conv_dir, e["file"]), dctx) for e in entries]

    rows = []
    # Turn N's generated assistant message == last assistant msg of turn N+1's body.
    for n in range(len(entries) - 1):
        nxt = bodies[n + 1]
        if nxt is None:
            continue
        asst = [m for m in nxt["body"].get("messages", []) if m.get("role") == "assistant"]
        if not asst:
            continue
        gen = asst[-1]
        r_ch = _reasoning_chars(gen)
        t_ch = _toolcall_chars(gen)
        x_ch = _text_chars(gen)
        tot_ch = r_ch + t_ch + x_ch

        prompt = entries[n].get("numPromptTokens")
        comp = entries[n].get("numCompletionTokens")
        if prompt is None or comp is None:
            continue
        prompt = int(prompt)
        comp = int(comp)
        if tot_ch > 0:
            think = comp * r_ch / tot_ch
            tool = comp * t_ch / tot_ch
            resp = comp * x_ch / tot_ch
        else:
            think = tool = resp = 0.0
        rows.append((n, prompt, think, tool, resp))
    return rows or None


def _summary(name, vals):
    if not vals:
        print(f"{name:<32} (no data)")
        return
    s = sorted(vals)
    n = len(s)
    pct = lambda p: s[min(n - 1, int(round(p * (n - 1))))]
    print(
        f"{name:<32} n={n:<7} min={s[0]:<8.0f} p50={pct(0.5):<8.0f} avg={sum(s)/n:<9.0f} "
        f"p90={pct(0.9):<8.0f} p99={pct(0.99):<8.0f} max={s[-1]:.0f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/shared/request-replay-data/matterhorn-fp4-5")
    ap.add_argument("--target-substr", default="matterhorn-fp4-5")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--by-turn-index", type=int, default=0, metavar="N",
                    help="Also print breakdown grouped by turn position, for the first N indices.")
    ap.add_argument("--csv", default=None,
                    help="Write per-turn rows: conversation_id,turn,prompt,thinking,tool_call,response")
    args = ap.parse_args()

    conv_dirs = sorted(
        os.path.join(args.src, d) for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))
    )
    if args.limit is not None:
        conv_dirs = conv_dirs[: args.limit]
    print(f"Scanning {len(conv_dirs)} conversations under {args.src}", file=sys.stderr)

    P, TH, TL, RS = [], [], [], []
    by_idx: dict[int, dict[str, list[float]]] = {}
    csv_f = open(args.csv, "w") if args.csv else None
    if csv_f:
        csv_f.write("conversation_id,turn,prompt,thinking,tool_call,response\n")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_conversation, d, args.target_substr): d for d in conv_dirs}
        for fut in as_completed(futs):
            rows = fut.result()
            if not rows:
                continue
            cid = os.path.basename(futs[fut].rstrip("/"))
            for (n, p, th, tl, rs) in rows:
                P.append(p); TH.append(th); TL.append(tl); RS.append(rs)
                if args.by_turn_index and n < args.by_turn_index:
                    slot = by_idx.setdefault(n, {"p": [], "th": [], "tl": [], "rs": []})
                    slot["p"].append(p); slot["th"].append(th); slot["tl"].append(tl); slot["rs"].append(rs)
                if csv_f:
                    csv_f.write(f"{cid},{n},{p},{th:.1f},{tl:.1f},{rs:.1f}\n")
    if csv_f:
        csv_f.close()

    print("\n===== PER-TURN TOKEN BREAKDOWN (turns with a following turn) =====")
    print(f"turns counted: {len(P)}\n")
    _summary("prompt (input)", P)
    _summary("thinking (reasoning out)", TH)
    _summary("tool_call (args out)", TL)
    _summary("response (visible text out)", RS)
    tot_out = [th + tl + rs for th, tl, rs in zip(TH, TL, RS)]
    _summary("total output (=completion)", tot_out)
    if tot_out:
        sTH, sTL, sRS = sum(TH), sum(TL), sum(RS)
        so = sTH + sTL + sRS
        print(f"\noutput composition (token share): thinking={sTH/so:.1%}  tool_call={sTL/so:.1%}  text={sRS/so:.1%}")

    if args.by_turn_index and by_idx:
        med = lambda v: sorted(v)[len(v) // 2] if v else 0
        print(f"\nPer-turn-index medians (first {args.by_turn_index}):")
        print(f"  {'turn':>4} {'turns':>7} | {'prompt':>8} {'thinking':>9} {'tool_call':>10} {'response':>9}")
        for i in sorted(by_idx):
            s = by_idx[i]
            print(f"  {i:>4} {len(s['p']):>7} | {med(s['p']):>8.0f} {med(s['th']):>9.0f} "
                  f"{med(s['tl']):>10.0f} {med(s['rs']):>9.0f}")

    if args.csv:
        print(f"\nper-turn CSV written to: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
