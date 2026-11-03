import random

from mock_server import PrefixCacheEngine, _PrefixIndex


def _brute(ids, seen):
    best = 0
    for prev in seen:
        n = min(len(ids), len(prev))
        k = 0
        while k < n and ids[k] == prev[k]:
            k += 1
        best = max(best, k)
    return best


def test_prefix_index_matches_brute_force() -> None:
    """The block-hash index finds the same longest common prefix as scanning every stored prompt, on prompts that
    share pool prefixes of many lengths and diverge anywhere (mid-block, at a block edge, at token 0)."""
    rng = random.Random(3)
    pools = [[rng.randrange(1000) for _ in range(rng.randrange(1, 900))] for _ in range(12)]
    index, seen = _PrefixIndex(), []
    for _ in range(400):
        pool = rng.choice(pools)
        cut = rng.randrange(len(pool) + 1)
        ids = pool[:cut] + [rng.randrange(1000, 2000) for _ in range(rng.randrange(1, 200))]
        assert index.longest(ids) == _brute(ids, seen)
        index.add(ids)
        seen.append(ids)


class _Tok:
    """A tokenizer stand-in: one id per word; the chat template wraps the last message in two marker ids, and
    returns a dict as newer transformers do."""

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 50_000 for w in text.split()]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return {"input_ids": [7, 8] + self.encode(messages[-1]["content"]) + [9], "attention_mask": []}


def test_engine_reuse_rule_window_boundary_and_routing() -> None:
    words = " ".join(f"w{i}" for i in range(3000))
    tail = lambda k: f" tail{k} x y z"
    # token-exact with a 130-token window
    engine = PrefixCacheEngine(_Tok(), window=130)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}))
    assert engine.serve(engine.token_ids({"prompt": words + tail(1)})) == 3000 - 130
    # 2,048-token restore blocks
    engine = PrefixCacheEngine(_Tok(), window=128, boundary=2048)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}))
    assert engine.serve(engine.token_ids({"prompt": words + tail(1)})) == 2048 - 128
    # chat: the template's dict return is unpacked (it used to yield its 2 keys as the prompt)
    engine = PrefixCacheEngine(_Tok(), window=0)
    chat = {"messages": [{"role": "user", "content": words}]}
    assert len(engine.token_ids(chat)) == 3003
    # several caches: affinity keeps a key on one cache; without it a request can miss the warm prompt
    engine = PrefixCacheEngine(_Tok(), window=0, caches=8, affinity=True)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}), "k1")
    assert all(engine.serve(engine.token_ids({"prompt": words + tail(i)}), "k1") == 3000 for i in range(1, 6))
    engine = PrefixCacheEngine(_Tok(), window=0, caches=8, affinity=False, seed=1)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}))
    hits = [engine.serve(engine.token_ids({"prompt": words + tail(i)})) for i in range(1, 9)]
    assert 0 in hits  # landed on a cache that hadn't seen it


def test_preflight_runner_reads_a_tier_plan_and_picks_the_scored_step() -> None:
    """D-117: the runner takes task 07's plan as written (traffic_generator_create_args) and scores the step at
    the plan's target rate, after the fill; the measured probe comes from record.reuse_probe."""
    from preflight_run import plan_steps, scored_step

    mix = '{"seed":1,"classes":[{"name":"c","weight":1,"prompt_tokens":900,"cached_prompt_tokens":500}]}'

    def create(gid, conc, dur, extra, qps=None):
        args = ["traffic-generator", "create", "--target", "dev-pod/x", "--traffic-generator-id", gid]
        args += ["--concurrency", str(conc), "--spawn-rate", str(conc), "--duration", dur]
        if qps is not None:
            args += ["--qps", str(qps)]
        return args + ["--image-tag", "in-pod", "--extra-args", extra]

    plan = {
        "record": {"target_qps": 3.63164, "reuse_probe": {"overhang": 130, "boundary": 0}},
        "traffic_generator_create_args": {
            "fill-fill": create("f", 1, "3600s", f"--shape-mix '{mix}' --prefix-fill --prefix-reuse-overhang 130"),
            "warm-warm": create("w", 599, "600s", f"--shape-mix '{mix}' --qps-distribution gamma", 3.63164),
            "probe-p1": create("p1", 300, "180s", f"--shape-mix '{mix}'", 1.81582),
            "ramp-3": create("r3", 599, "900s", f"--shape-mix '{mix}' --qps-cv2 2.75", 3.63164),
        },
    }
    steps = plan_steps(plan)
    assert [s[0] for s in steps] == ["fill-fill", "warm-warm", "probe-p1", "ramp-3"]
    name, conc, rate, seconds, extra = scored_step(plan, steps)
    assert (name, conc, rate, seconds) == ("ramp-3", 599, 3.63164, 900.0)  # the LAST step at the target rate
    assert extra[:2] == ["--shape-mix", mix] and "--qps-cv2" in extra  # the plan's exact args, unquoted
    assert "--prefix-fill" in steps[0][4]
    assert scored_step(plan, steps, "probe-p1")[0] == "probe-p1"


def test_preflight_requires_a_mode_and_checks_the_logged_paths(tmp_path) -> None:
    """Every step must name --chat or --no-chat, and the requests the mock logged for it must hit that path."""
    from preflight_run import path_mismatches, step_path

    assert step_path(("s", 1, None, None, ["--shape-mix", "x", "--no-chat"])) == "/v1/completions"
    assert step_path(("s", 1, None, None, ["--no-chat", "--chat"])) == "/v1/chat/completions"  # the last wins
    assert step_path(("s", 1, None, None, ["--shape-mix", "x"])) is None
    assert path_mismatches(str(tmp_path / "missing.requests"), 0, "/v1/completions") == (
        0,
        {},
    )  # 0 seen: main() fails it
    log = tmp_path / "mock.requests"
    log.write_text("1.0 900 0 0 /v1/completions\n2.0 900 400 0 /v1/completions\n3.0 900 400 0 /v1/chat/completions\n")
    assert path_mismatches(str(log), 0, "/v1/completions") == (3, {"/v1/chat/completions": 1})
    assert path_mismatches(str(log), 0, "/v1/chat/completions") == (3, {"/v1/completions": 2})
    assert path_mismatches(str(log), 2, "/v1/chat/completions") == (1, {})  # only this step's lines
    # A log from before the path column counts as a mismatch, not a silent pass
    log.write_text("1.0 900 0 0\n")
    assert path_mismatches(str(log), 0, "/v1/completions") == (1, {"?": 1})


def test_preflight_refuses_a_layout_that_doesnt_match_the_mock(tmp_path, capsys) -> None:
    """The layout overhang must equal what the mock withholds (its window, less its chat head in chat), or the
    verdict measures the mismatch: P9's 2-log pre-flight laid prefixes out +128 on a window-0 mock and "failed"
    its short classes by -67%. The pre-flight refuses before starting; --allow-mismatch is for experiments."""
    import json

    from preflight_run import applied_overhang, expected_overhang, main, summary_mismatch

    chat = ("s", 1, None, None, ["--shape-mix", "x", "--chat", "--prefix-reuse-overhang", "102"])
    completions = ("s", 1, None, None, ["--shape-mix", "x", "--no-chat", "--prefix-reuse-overhang", "128"])
    assert expected_overhang(chat, 128, 26) == 102 and expected_overhang(completions, 128, 26) == 128
    assert applied_overhang(chat) == 102
    # resolved by llm_bench from the served model's config when the step passes no overhang
    config = tmp_path / "dsv4"
    config.mkdir()
    (config / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "sliding_window": 128}))
    assert applied_overhang(("s", 1, None, None, ["--no-chat", "--served-model-config", str(config)])) == 128

    mix = '{"seed":1,"classes":[{"name":"c","weight":1,"prompt_tokens":900,"cached_prompt_tokens":500}]}'
    plan = {
        "record": {"target_qps": 1.0},
        "traffic_generator_create_args": {
            "ramp-1": [
                "traffic-generator",
                "create",
                "--concurrency",
                "2",
                "--duration",
                "10s",
                "--qps",
                "1.0",
                "--extra-args",
                f"--shape-mix '{mix}' --no-chat --prefix-reuse-overhang 128",
            ],
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert main([str(path), "--out", str(tmp_path / "out"), "--window", "0"]) == 2  # before any mock starts
    assert "refusing" in capsys.readouterr().out
    assert not (tmp_path / "out" / "mock.requests").exists()

    summary = tmp_path / "s.csv"
    summary.write_text("Reuse Overhang Tokens,Chat Head Tokens\n128,20\n")
    assert "overhang 128" in summary_mismatch(str(summary), completions, 0, 0, "tok")
    assert summary_mismatch(str(summary), completions, 128, 0, "tok") is None
    headed = ("s", 1, None, None, ["--chat", "--tokenizer", "tok", "--prefix-reuse-overhang", "102"])
    summary.write_text("Reuse Overhang Tokens,Chat Head Tokens\n102,20\n")
    assert "chat head of 20" in summary_mismatch(str(summary), headed, 128, 26, "tok")
    assert summary_mismatch(str(summary), headed, 128, 26, "another-tok") is None  # a counting-tokenizer head


def test_preflight_mock_window_nets_a_chat_probe_and_leads_with_the_mock_fix(tmp_path) -> None:
    """N10-146-5: a chat probe's overhang (102) is net of the engine's head, so with --mock-chat-head 26 the mock's raw
    window is 128 and the probe's own 102 is accepted, not refused for 76. N10-146-6: the advice fits the mock to
    the layout first. N10-146-7: an --allow-mismatch run is marked and never exits as a plan verdict."""
    from preflight_run import expected_overhang, mismatch_message, mock_window, probe_mode

    chat = ("s", 1, None, None, ["--chat", "--prefix-reuse-overhang", "102"])
    probe = {"overhang": 102, "boundary": 0, "source": "probe-chat-fresh-102"}
    assert probe_mode(probe) == "chat" and probe_mode({"source": "probe-completions-fresh-129"}) == "completions"
    window = mock_window(probe, None, 26)
    assert window == 128 and expected_overhang(chat, window, 26) == 102  # the correct 102 passes
    assert mock_window({"overhang": 129, "source": "probe-completions-x"}, None, 26) == 129
    assert mock_window({"overhang": 102}, None, 26) is None  # mode unknown with a head: ambiguous, refused
    assert mock_window({"overhang": 102}, None, 0) == 102 and mock_window(probe, 120, 26) == 120
    message = mismatch_message(chat, 102, 76, 102, 26)
    assert message.index("--window 128 --mock-chat-head 26") < message.index("--prefix-reuse-overhang")
