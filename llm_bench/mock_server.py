"""A local OpenAI-compatible server that emulates an engine's prefix cache, for checking llm_bench on CPU.

Every request's prompt is tokenized with --tokenizer (chat requests through its chat template) and compared with
the prompts this server has seen: the engine reuses the longest common token prefix, rounded down to whole
--boundary blocks, minus the --window tokens it recomputes at the divergence. That is the model llm_bench lays
prompts out for (C-242, D-111), so a run against this server shows per class whether the designed cached tokens
are delivered:

    python mock_server.py --port 8000 --window 130                       # DeepSeek V4.1 as measured on P8
    python mock_server.py --port 8000 --window 128 --boundary 2048       # 2,048-token restore blocks
    python mock_server.py --port 8000 --window 130 --caches 8            # 8 caches (DP ranks), random routing
    python mock_server.py --port 8000 --window 130 --caches 8 --affinity # ... routed by x-session-affinity

Responses carry usage.prompt_tokens (the real token count) and usage.prompt_tokens_details.cached_tokens, as
Fireworks does; streaming and non-streaming both work. It holds no request content beyond token ids in memory.
"""

import argparse
import hashlib
from array import array
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_BLOCK = 64  # tokens per hashed block of the prefix index


class _PrefixIndex:
    """The longest common token prefix of a prompt with any stored prompt, without comparing it to all of them:
    every stored prompt is indexed by a running hash of its first k whole 64-token blocks (and, for prompts
    sharing no whole block, by its first token). A lookup walks whole blocks by hash, then compares tokens only
    inside the next block, against the prompts that share the matched blocks. Exact, and cheap for 300K-token
    prompts: the per-request work is the prompt's blocks plus at most 64 tokens per candidate."""

    def __init__(self):
        self.blocks = {}  # running hash of the first k blocks -> prompts with that prefix
        self.firsts = {}  # first token -> prompts starting with it

    @staticmethod
    def _hashes(ids):
        h = hashlib.blake2b(digest_size=16)
        raw = array("I", ids).tobytes()
        for k in range(len(ids) // _BLOCK):
            h.update(raw[4 * _BLOCK * k : 4 * _BLOCK * (k + 1)])
            yield h.copy().digest()

    def longest(self, ids) -> int:
        if not ids:
            return 0
        matched, candidates = 0, self.firsts.get(ids[0], [])
        for k, digest in enumerate(self._hashes(ids)):
            found = self.blocks.get(digest)
            if found is None:
                break
            matched, candidates = (k + 1) * _BLOCK, found
        best = matched
        for prev in candidates:
            n = min(len(ids), len(prev), matched + _BLOCK)
            i = matched
            while i < n and ids[i] == prev[i]:
                i += 1
            best = max(best, i)
        return best

    def add(self, ids) -> None:
        if not ids:
            return
        ids = array("I", ids)  # 4 bytes a token: a 100M-token fill fits in memory
        self.firsts.setdefault(ids[0], []).append(ids)
        for digest in self._hashes(ids):
            self.blocks.setdefault(digest, []).append(ids)


def _dsv4_encoder(tokenizer, tokenizer_path):
    """The DeepSeek V4 family's encode_messages (encoding_dsv4.py, or encoding.py on V4.1 checkpoints) when
    the tokenizer has no chat template, else None."""
    if getattr(tokenizer, "chat_template", None) or not tokenizer_path:
        return None
    import importlib.util
    import os

    for name in ("encoding/encoding_dsv4.py", "encoding/encoding.py"):  # V4.1 ships it as encoding.py
        path = os.path.join(tokenizer_path, name)
        if os.path.exists(path):
            break
    else:
        try:
            from huggingface_hub import hf_hub_download

            for name in ("encoding/encoding_dsv4.py", "encoding/encoding.py"):
                try:
                    path = hf_hub_download(repo_id=tokenizer_path, filename=name)
                    break
                except Exception:
                    continue
            else:
                return None
        except Exception:
            return None
    spec = importlib.util.spec_from_file_location("mock_dsv4_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


class PrefixCacheEngine:
    """Token-id prompts per cache, and the reuse rule: floor(common prefix / boundary) * boundary - window."""

    def __init__(
        self, tokenizer, window=0, boundary=1, caches=1, affinity=False, seed=0, tokenizer_path=None, chat_head=0
    ):
        self.chat_head = chat_head
        self.tokenizer = tokenizer
        self.encode_messages = _dsv4_encoder(tokenizer, tokenizer_path)
        self.window = window
        self.boundary = max(1, boundary)
        self.caches = max(1, caches)
        self.affinity = affinity
        self.rng = random.Random(seed)
        self.store = {}
        self.lock = threading.Lock()

    def token_ids(self, body: dict) -> list:
        ids = self._content_ids(body)
        # The engine's own chat head (DeepSeek V4.1 renders +26 the client doesn't count), shared by every chat prompt
        return (
            [_HEAD_TOKEN_BASE + i for i in range(self.chat_head)] + ids
            if self.chat_head and "messages" in body
            else ids
        )

    def _content_ids(self, body: dict) -> list:
        if "messages" in body and self.encode_messages is not None:
            # DeepSeek V4 family: no chat_template; the checkpoint ships its own encoding module (as llm_bench uses)
            prompt = self.encode_messages(body["messages"], thinking_mode="chat")
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        if "messages" in body:
            ids = self.tokenizer.apply_chat_template(body["messages"], tokenize=True, add_generation_prompt=True)
            # newer transformers return a BatchEncoding (a dict) here, older ones the ids
            return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
        return self.tokenizer.encode(body["prompt"], add_special_tokens=False)

    def serve(self, ids: list, affinity_key=None) -> int:
        """Cached tokens for a prompt, then remember it (in the cache it was routed to)."""
        with self.lock:
            if self.caches == 1:
                cache = 0
            elif self.affinity and affinity_key:
                cache = int(hashlib.sha1(affinity_key.encode()).hexdigest(), 16) % self.caches
            else:
                cache = self.rng.randrange(self.caches)
            index = self.store.setdefault(cache, _PrefixIndex())
            common = index.longest(ids)
            index.add(ids)
            self.last_cache = cache
        return max(0, (common // self.boundary) * self.boundary - self.window)


# Token ids for the engine's chat head: outside any real vocabulary, so they never match prompt text
_HEAD_TOKEN_BASE = 9_990_000


def make_handler(
    engine: PrefixCacheEngine,
    ttft: float,
    tpot: float,
    log=None,
    slow_fraction=0.0,
    slow_ttft=0.0,
    num_draft_tokens: int = 0,
):
    log_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            arrived = time.time()
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            # The engine refuses an acceptance override whose length differs from its draft count (a UserError),
            # so a schedule padded wider than the replica's num_draft_tokens fails every row: caught here
            override = body.get("acceptance_probs_override")
            if num_draft_tokens and isinstance(override, list) and len(override) != num_draft_tokens:
                raw = json.dumps(
                    {"error": {"message": f"acceptance_probs_override must have {num_draft_tokens} entries"}}
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            ids = engine.token_ids(body)
            cached = engine.serve(ids, self.headers.get("x-session-affinity"))
            if log:
                with log_lock:
                    # The path last, so readers of the first four columns are unchanged
                    log.write(f"{arrived:.6f} {len(ids)} {cached} {engine.last_cache} {self.path}\n")
                    log.flush()
            chat = "messages" in body
            n = max(1, int(body.get("max_tokens") or 16))
            usage = {
                "prompt_tokens": len(ids),
                "completion_tokens": n,
                "prompt_tokens_details": {"cached_tokens": cached},
            }
            if not body.get("stream", False):
                choice = {"index": 0, "message": {"content": " x" * n}} if chat else {"index": 0, "text": " x" * n}
                raw = json.dumps({"choices": [choice], "usage": usage}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(obj):
                data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            # A share of requests is held long (long generations on a healthy server)
            time.sleep(slow_ttft if slow_fraction and random.random() < slow_fraction else ttft)
            for i in range(n):
                if i:
                    time.sleep(tpot)
                choice = {"index": 0, "delta": {"content": " x"}} if chat else {"index": 0, "text": " x"}
                send({"choices": [choice]})
            send({"choices": [], "usage": usage})
            send("[DONE]")
            self.wfile.write(b"0\r\n\r\n")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--tokenizer",
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        help="Tokenizer the engine counts and matches prefixes with: use the one llm_bench counts with (--tokenizer).",
    )
    parser.add_argument("--window", type=int, default=0, help="Tokens recomputed at the divergence (the overhang).")
    parser.add_argument("--boundary", type=int, default=1, help="Restore block in tokens (1: token-exact reuse).")
    parser.add_argument("--caches", type=int, default=1, help="Independent prefix caches (e.g. data-parallel ranks).")
    parser.add_argument("--affinity", action="store_true", help="Route by x-session-affinity, else at random.")
    parser.add_argument("--ttft", type=float, default=0.05, help="Seconds before the first streamed token.")
    parser.add_argument("--tpot", type=float, default=0.002, help="Seconds between streamed tokens.")
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=0,
        help="The replica's speculative draft count: an acceptance_probs_override of any other length is refused "
        "(400), as the engine refuses it (a UserError). 0 doesn't check.",
    )
    parser.add_argument(
        "--chat-head",
        type=int,
        default=0,
        help="Tokens the engine renders ahead of every chat prompt beyond the client's count (DeepSeek V4.1: 26).",
    )
    parser.add_argument("--slow-fraction", type=float, default=0.0, help="Share of requests held --slow-ttft s.")
    parser.add_argument("--slow-ttft", type=float, default=0.0, help="Seconds before the first token when held.")
    parser.add_argument("--log", default=None, help="Append 'arrival prompt_tokens cached_tokens' per request.")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    engine = PrefixCacheEngine(
        AutoTokenizer.from_pretrained(args.tokenizer),
        args.window,
        args.boundary,
        args.caches,
        args.affinity,
        tokenizer_path=args.tokenizer,
        chat_head=args.chat_head,
    )
    log = open(args.log, "a") if args.log else None
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(engine, args.ttft, args.tpot, log, args.slow_fraction, args.slow_ttft, args.num_draft_tokens),
    )
    print(
        f"mock_server: http://{args.host}:{args.port}  window {args.window}  boundary {args.boundary}  "
        f"caches {args.caches}{' (affinity)' if args.affinity else ''}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
