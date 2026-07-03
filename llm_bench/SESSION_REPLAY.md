# Session-based request replay (Mode B)

Replay captured production requests against a deployment, preserving multi-turn
**conversation structure** and **client session semantics** so the server's
prefix cache and session-affinity routing behave as they do in production.

Captured requests (e.g. from
`s3://anysphere-fireworks-shared/request-replay-data/<deployment>/`) are stored
as one directory per conversation, with each turn a zstd-compressed JSON object
(`{"body": ..., "headers": ..., "url": ...}`). A single turn's `body` already
contains the **entire accumulated conversation** up to that point (system +
user + assistant + tool messages), so consecutive turns of one conversation
share a long common prefix.

Mode B replays each conversation **in order on a single client connection**,
with the captured session-routing headers (`x-session-affinity`,
`x-multi-turn-session-id`). One Locust user = one concurrent conversation, so
`-u N` means **N concurrent conversations**.

## How it works

1. **`convert_replay.py`** (offline, one-time) pre-bakes each captured turn into
   the *final ready-to-POST JSON bytes* (target `model` + `stream` baked in) and
   writes a lightweight `index.jsonl` listing, per conversation, the ordered
   body files plus the session headers (`authorization` stripped). This keeps
   the load-test hot path free of decompression, JSON (de)serialization, and
   tokenization — so replay QPS is bounded by the server, not the client.
2. **`load_test.py --replay-sessions`** streams whole conversations from the
   index to Locust users, POSTing the pre-baked bytes verbatim with the
   per-request session headers, and reuses the existing streaming-SSE parsing,
   TTFT / token metrics, and summary CSV.

## Usage (2 commands)

### 1. Bake the dataset (once)

```bash
python3 convert_replay.py \
  --src /shared/request-replay-data/matterhorn-fp4-5 \
  --out /shared/request-replay-data/matterhorn-fp4-5-baked \
  --model "" --stream
```

- `--model ""` bakes an empty model string, which the Fireworks server resolves
  to its **base (loaded) model** — so the same baked dataset works against any
  server/snapshot without re-baking. Pass an explicit model FQN instead if you
  need to target a specific addon/deployment.
- `--stream` must match the `--stream` flag you use at run time (the SSE parser
  depends on it). Use `--no-stream` for both if benchmarking non-streaming.
- Omit `--model` entirely to run in **stats-only** mode (prints dataset shape,
  writes nothing).
- By default (`--max-tokens-mode recorded`) each turn's `max_tokens` is set to
  its recorded `numCompletionTokens`, and `min_tokens` + `ignore_eos` are baked
  in (`--force-length`, on by default) so the decode length exactly matches
  prod. This matters when the served checkpoint differs from the captured model
  (e.g. a Kermit snapshot vs. the source deployment): such a model produces
  incoherent output and never emits EOS, so without this it would run to the
  captured `max_tokens` (often 20000) on every turn. Content is meaningless in
  that case, but prompt sizes, decode lengths, session structure, and
  prefix-cache behavior are all faithful — which is what a perf/load replay
  needs. Use `--max-tokens-mode keep` (or `--no-force-length`) to preserve the
  captured limits instead.

Output: `index.jsonl` (one line per conversation) + `bodies/<conversation>/NNN.json`.

### 2. Run the replay

```bash
locust --headless -u 64 -r 8 -t 10min --chat --stream --provider fireworks \
  --replay-sessions \
  --replay-index /shared/request-replay-data/matterhorn-fp4-5-baked/index.jsonl \
  -H http://localhost:8080 \
  --summary-file replay.csv
```

- `-u` = number of **concurrent conversations**; `-r` = ramp rate.
- `--chat --stream` are required (bodies are chat-completions; `--stream` must
  match the bake).
- `-m` can be omitted — the model is baked into each body; the harness
  auto-detects a display model from `/v1/models` for bookkeeping only.
- `--gpus N` (optional): when set, the summary also reports **TPM/GPU total**
  and **TPM/GPU uncached** (real compute after prefix-cache reuse).
- Results: Locust's percentile tables + summary block, plus `--summary-file`
  CSV (consumed by jarvis/perfagent). Server-side metrics are on the server's
  `--request-metrics-port`.

### Derived summary metrics

Beyond the standard fields, the end-of-run summary (and the `--summary-file`
CSV) also reports, for the chat/completions path:

| Field | Meaning |
|---|---|
| `Ttft Max` | max time-to-first-token (ms) |
| `Decode Tokens Per S` | `1000 / mean latency_per_token` (per-stream inter-token rate) |
| `Cache Hit Pct` | token-weighted `cached_tokens / prompt_tokens` |
| `Tpm Total` / `Tpm Uncached` | tokens/min = `qps × tokens × 60` (uncached = after prefix-cache reuse) |
| `Tpm Per Gpu Total` / `Tpm Per Gpu Uncached` | the above ÷ `--gpus` (only when `--gpus` is set) |

### Replay-specific flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--replay-sessions` | off | Enable Mode B session replay. |
| `--replay-index PATH` | — | Path to `index.jsonl` from `convert_replay.py`. |
| `--session-think-time SECS` | `0` | Client-side sleep between consecutive turns of a conversation. |
| `--replay-once` | off | Play each conversation exactly once, then stop. Default recycles conversations to sustain load for `-t`. |
| `--replay-timeout SECS` | `600` | Per-request HTTP timeout (replayed prompts are large with high `max_tokens`). |

## Dataset stats helper

`replay_stats.py` reads the per-conversation `manifest.json` files (no
decompression) and reports sequence-length distributions:

```bash
python3 replay_stats.py --src /shared/request-replay-data/matterhorn-fp4-5
# add --csv turns.csv to dump per-turn (prompt/completion/total) rows
```

Reports per-turn prompt / completion / total tokens and per-conversation
turns, peak prompt tokens, and final-turn length (min / p50 / avg / p90 / p99 /
max). Add `--by-turn-index N` to break prompt/response length down by turn
position.

`replay_turn_breakdown.py` further splits each turn's output into
thinking / tool_call / response buckets (decompresses bodies; the exact
completion count is split proportionally to the generated message composition):

```bash
python3 replay_turn_breakdown.py --src /shared/request-replay-data/matterhorn-fp4-5 --by-turn-index 12
```

## Notes / limitations

- **Single-process only.** The session controller is per-process; running
  Locust with `--processes` / distributed workers would duplicate conversations
  across processes. For a single host this is not a concern.
- Baked size is roughly the decompressed dataset (each turn re-embeds its full
  prior context), e.g. ~30 GB for `matterhorn-fp4-5`. It is passive disk, never
  loaded wholesale into memory — turns are read on demand.
