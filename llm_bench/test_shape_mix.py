from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import pytest
from locust.stats import RequestStats

import load_test
from load_test import (
    FireworksProvider,
    arrival_summary_entries,
    count_dispersion,
    gap_cv2,
    modulation_rate_variance,
    resolve_prefix_reuse_overhang,
    resolve_qps_dispersion,
    FixedQPSPacer,
    OpenAIProvider,
    ShapeMixDataset,
    VllmProvider,
    build_shape_mix_request,
    parse_shape_mix,
    resolve_qps_cv2,
    resolve_request_seed,
    sample_interarrival,
    shape_mix_summary_entries,
    validate_shape_mix_options,
)

LIMERICKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limericks.txt")
MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"

MIX = {
    "seed": 7,
    "classes": [
        {
            "name": "chat",
            "weight": 0.6,
            "prompt_tokens": 600,
            "cached_prompt_tokens": 400,
            "generated_tokens": 50,
            "prefix_pool_size": 3,
        },
        {
            "name": "agent",
            "weight": 0.3,
            "prompt_tokens": 1200,
            "cached_prompt_tokens": 1000,
            "generated_tokens": 120,
            "prefix_pool_size": 2,
        },
        {"name": "cold", "weight": 0.1, "prompt_tokens": 300, "cached_prompt_tokens": 0, "generated_tokens": 20},
    ],
}


class WordTokenizer:
    """Offline stand-in for a HF tokenizer: one token per whitespace-separated word, stable vocab."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._words)
                self._words.append(word)
            ids.append(self._ids[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._words[i] for i in ids)


def _dataset(mix=None, tokenizer=None, **kwargs) -> ShapeMixDataset:
    kwargs.setdefault("corpus_path", LIMERICKS)
    return ShapeMixDataset(
        mix=parse_shape_mix(json.dumps(mix or MIX)),
        tokenizer=tokenizer or WordTokenizer(),
        **kwargs,
    )


def _take(ds: ShapeMixDataset, n: int):
    return [next(ds) for _ in range(n)]


def _pool(ds: ShapeMixDataset, class_index: int) -> list[str]:
    return [ds.prefix(class_index, j) for j in range(ds.pool_sizes[class_index])]


def _options(chat: bool, **overrides) -> argparse.Namespace:
    values = dict(
        systemone=False,
        rerank=False,
        embeddings=False,
        chat=chat,
        stream=True,
        temperature=1.0,
        n=1,
        top_k=None,
        logprobs=None,
        reasoning_effort=None,
        clear_assistant=False,
        force_min_tokens=False,
        prompt_cache_max_len=0,
        acceptance_probs_override=None,
        forced_generation_file=None,
        forced_generation_from_dataset=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parse_inline_file_and_defaults(tmp_path) -> None:
    path = tmp_path / "mix.json"
    path.write_text(json.dumps(MIX))

    inline = parse_shape_mix(json.dumps(MIX))
    from_file = parse_shape_mix(f"@{path}")

    assert inline == from_file
    assert inline.seed == 7
    assert [c.name for c in inline.classes] == ["chat", "agent", "cold"]
    assert inline.classes[2].prefix_pool_size == 1
    unnamed = parse_shape_mix(json.dumps({"classes": [{"weight": 1, "prompt_tokens": 64, "generated_tokens": 8}]}))
    assert unnamed.seed == 0
    assert unnamed.classes[0].name == "class0"
    assert unnamed.classes[0].cached_prompt_tokens == 0


@pytest.mark.parametrize(
    "spec",
    [
        "not json",
        {"classes": []},
        [{"weight": 1, "prompt_tokens": 10, "generated_tokens": 1}],
        {"classes": [{"weight": 1, "prompt_tokens": 10}]},
        {"classes": [{"weight": 1, "prompt_tokens": 10, "cached_prompt_tokens": 11, "generated_tokens": 1}]},
        {"classes": [{"weight": -1, "prompt_tokens": 10, "generated_tokens": 1}]},
        {"classes": [{"weight": 0, "prompt_tokens": 10, "generated_tokens": 1}]},
        {"classes": [{"weight": 1, "prompt_tokens": "10", "generated_tokens": 1}]},
        {"classes": [{"weight": 1, "prompt_tokens": 10, "generated_tokens": 0.4}]},
        {"classes": [{"weight": 1, "prompt_tokens": 10.2, "cached_prompt_tokens": 10.4, "generated_tokens": 1}]},
        {"arrivals": {"cv2": 0}, "classes": [{"weight": 1, "prompt_tokens": 10, "generated_tokens": 1}]},
        {"arrivals": {"cv2": "4"}, "classes": [{"weight": 1, "prompt_tokens": 10, "generated_tokens": 1}]},
        {"classes": [{"weight": 1, "prompt_tokens": 10, "generated_tokens": 1, "prefix_pool_size": 0}]},
        {"classes": [{"name": "a b", "weight": 1, "prompt_tokens": 10, "generated_tokens": 1}]},
        {
            "classes": [
                {"name": "x", "weight": 1, "prompt_tokens": 10, "generated_tokens": 1},
                {"name": "x", "weight": 1, "prompt_tokens": 10, "generated_tokens": 1},
            ]
        },
        {"seed": "1", "classes": [{"weight": 1, "prompt_tokens": 10, "generated_tokens": 1}]},
    ],
)
def test_parse_rejects_invalid(spec) -> None:
    with pytest.raises(ValueError):
        parse_shape_mix(spec if isinstance(spec, str) else json.dumps(spec))


PROFILE = {
    "profile_version": 1,
    "model": "accounts/fireworks/models/x",
    "window": {"start": "2026-10-01T00:00:00Z", "hours": 24},
    "arrivals": {"qps": 3.2, "cv2": 4.5},
    "seed": 3,
    "classes": [
        {
            "cache_class": "warm",
            "prompt_band": "4k-16k",
            "weight": 0.5,
            "prompt_tokens": 6123.6,
            "cached_prompt_tokens": 5000.5,
            "generated_tokens": 211.49,
        },
        {
            "cache_class": "warm",
            "prompt_band": "4k-16k",
            "weight": 0.2,
            "prompt_tokens": 9000,
            "cached_prompt_tokens": 100,
            "generated_tokens": 64,
            "prefix_pool_size": 4,
        },
        {"cache_class": "cold", "prompt_band": "<1k", "weight": 0.2, "prompt_tokens": 700.2, "generated_tokens": 30.7},
        {"cache_class": None, "prompt_band": None, "weight": 0.1, "prompt_tokens": 1500, "generated_tokens": 99},
    ],
}


def test_profile_json_passes_as_is(caplog) -> None:
    load_test._parse_shape_mix.cache_clear()
    with caplog.at_level("WARNING", logger="load_test"):
        mix = parse_shape_mix(json.dumps(PROFILE))

    assert [c.name for c in mix.classes] == ["warm_4k-16k", "warm_4k-16k_2", "cold_1k", "marginal"]
    first = mix.classes[0]
    assert (first.prompt_tokens, first.cached_prompt_tokens, first.generated_tokens) == (6124, 5001, 211)
    assert (mix.classes[2].prompt_tokens, mix.classes[2].generated_tokens) == (700, 31)
    assert [c.prefix_pool_size for c in mix.classes] == [1, 4, 1, 1]
    assert mix.seed == 3
    assert mix.arrivals_cv2 == 4.5
    # only the warm class without prefix_pool_size warns; top-level extras are ignored silently
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "warm_4k-16k" in warnings[0] and "prefix_pool_size" in warnings[0]


def test_derived_names_avoid_explicit_ones() -> None:
    mix = parse_shape_mix(
        json.dumps(
            {
                "classes": [
                    {"cache_class": None, "prompt_band": None, "weight": 1, "prompt_tokens": 64, "generated_tokens": 1},
                    {"name": "marginal", "weight": 1, "prompt_tokens": 64, "generated_tokens": 1},
                    {"cache_class": "hot", "weight": 1, "prompt_tokens": 64, "generated_tokens": 1},
                    {"weight": 1, "prompt_tokens": 64, "generated_tokens": 1},
                ]
            }
        )
    )

    assert [c.name for c in mix.classes] == ["marginal_2", "marginal", "hot", "class3"]


def test_cached_rounding_never_exceeds_prompt() -> None:
    mix = parse_shape_mix(
        json.dumps(
            {"classes": [{"weight": 1, "prompt_tokens": 100.4, "cached_prompt_tokens": 100.4, "generated_tokens": 1}]}
        )
    )

    assert (mix.classes[0].prompt_tokens, mix.classes[0].cached_prompt_tokens) == (100, 100)


def test_qps_cv2_defaults_to_profile_arrivals() -> None:
    profile = json.dumps(PROFILE)
    no_arrivals = json.dumps({k: v for k, v in PROFILE.items() if k != "arrivals"})

    assert resolve_qps_cv2(argparse.Namespace(qps_cv2=None, shape_mix=profile)) == 4.5
    assert resolve_qps_cv2(argparse.Namespace(qps_cv2=2.0, shape_mix=profile)) == 2.0
    assert resolve_qps_cv2(argparse.Namespace(qps_cv2=None, shape_mix=no_arrivals)) == 1.0
    assert resolve_qps_cv2(argparse.Namespace(qps_cv2=None, shape_mix=None)) == 1.0


def test_class_sampling_follows_weights() -> None:
    mix = json.loads(json.dumps(MIX))
    mix["classes"].append(
        {"name": "never", "weight": 0, "prompt_tokens": 100, "cached_prompt_tokens": 0, "generated_tokens": 1}
    )
    n = 6000
    counts = Counter(r.class_name for r in _take(_dataset(mix), n))

    assert counts["never"] == 0
    for cls in MIX["classes"]:
        assert counts[cls["name"]] / n == pytest.approx(cls["weight"], abs=0.025)


def test_prefix_pool_reuse_and_unique_tails() -> None:
    ds = _dataset()
    requests = _take(ds, 600)

    for ci, cls in enumerate(ds.mix.classes):
        reqs = [r for r in requests if r.class_index == ci]
        assert reqs
        if cls.cached_prompt_tokens == 0:
            assert ds.pool_sizes[ci] == 0
            assert all(r.prefix_index is None and r.affinity_key is None for r in reqs)
            continue
        pool = _pool(ds, ci)
        assert len(pool) == cls.prefix_pool_size == len(set(pool))
        assert {r.prefix_index for r in reqs} == set(range(cls.prefix_pool_size))
        for r in reqs:
            assert r.prompt.startswith(pool[r.prefix_index])
            assert r.affinity_key == f"shape-7-{cls.name}-{r.prefix_index}"

    all_prefixes = [p for ci in range(len(ds.mix.classes)) for p in _pool(ds, ci)]
    assert len(set(all_prefixes)) == len(all_prefixes)
    assert len({r.prompt for r in requests}) == len(requests)


def test_tails_diverge_right_after_the_shared_prefix() -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer)
    requests = _take(ds, 300)

    def common(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    by_slot: dict[tuple, list[list[int]]] = {}
    for r in requests:
        by_slot.setdefault((r.class_index, r.prefix_index), []).append(tokenizer.encode(r.prompt))
    for (ci, _), prompts in by_slot.items():
        # the shared prefix, then the fixed tail-start token, then the per-request nonce
        expected = ds.prefix_tokens[ci] + 1
        shared = [common(a, b) for a, b in zip(prompts, prompts[1:])]
        # the first nonce word is one of 256, so a pair rarely shares it too
        assert all(expected <= n <= expected + 1 for n in shared)
        assert shared.count(expected) >= 0.9 * len(shared)


def test_exact_prompt_token_lengths() -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer)

    for r in _take(ds, 300):
        cls = ds.mix.classes[r.class_index]
        ids = tokenizer.encode(r.prompt)
        assert len(ids) == r.prompt_tokens == cls.prompt_tokens
        if r.prefix_index is not None:
            assert ds.prefix_tokens[r.class_index] == cls.cached_prompt_tokens
            assert ids[: cls.cached_prompt_tokens] == tokenizer.encode(ds.prefix(r.class_index, r.prefix_index))


def test_chat_template_tokens_count_toward_prompt_and_cache() -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer, chat_overhead_tokens=12, chat_head_tokens=5)

    assert ds.prefix_tokens == [400 - 5, 1000 - 5, 0]
    for r in _take(ds, 100):
        assert r.prompt_tokens == ds.mix.classes[r.class_index].prompt_tokens
        assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens - 12


def test_fully_cached_class_keeps_a_unique_tail() -> None:
    mix = {
        "seed": 1,
        "classes": [{"weight": 1, "prompt_tokens": 200, "cached_prompt_tokens": 200, "generated_tokens": 4}],
    }
    tokenizer = WordTokenizer()
    ds = _dataset(mix, tokenizer=tokenizer)
    requests = _take(ds, 50)

    assert ds.prefix_tokens == [168]
    assert len({r.prompt for r in requests}) == 50
    assert all(len(tokenizer.encode(r.prompt)) == 200 for r in requests)


def test_prompt_too_short_for_a_unique_tail() -> None:
    mix = {"classes": [{"weight": 1, "prompt_tokens": 16, "generated_tokens": 4}]}
    with pytest.raises(ValueError):
        _dataset(mix)


def test_seeded_prompts_are_reproducible() -> None:
    first = [r.prompt for r in _take(_dataset(), 50)]
    again = [r.prompt for r in _take(_dataset(), 50)]
    other_seed = [r.prompt for r in _take(_dataset({**MIX, "seed": 8}), 50)]

    assert first == again
    assert first != other_seed


def test_workers_share_prefixes_but_not_prompts() -> None:
    w0 = _dataset(worker_index=0)
    w1 = _dataset(worker_index=1)

    assert all(_pool(w0, ci) == _pool(w1, ci) for ci in range(len(MIX["classes"])))
    prompts = {r.prompt for r in _take(w0, 200)} | {r.prompt for r in _take(w1, 200)}
    assert len(prompts) == 400


def test_runs_with_different_request_seeds_share_prefixes_but_no_prompt() -> None:
    tokenizer = WordTokenizer()
    run_a = _dataset(tokenizer=tokenizer)
    run_b = _dataset(tokenizer=tokenizer, request_seed=1234)
    prompts_a = [r.prompt for r in _take(run_a, 300)]
    prompts_b = [r.prompt for r in _take(run_b, 300)]

    assert run_a.request_seed == MIX["seed"]
    assert run_b.request_seed == 1234
    assert all(_pool(run_a, ci) == _pool(run_b, ci) for ci in range(len(MIX["classes"])))
    assert not set(prompts_a) & set(prompts_b)
    shared = [p for p in prompts_b if any(p.startswith(prefix) for prefix in _pool(run_a, 0))]
    assert shared


def test_request_seed_from_mix_and_cli_override() -> None:
    in_mix = json.dumps({**MIX, "request_seed": 99})

    assert parse_shape_mix(json.dumps(MIX)).request_seed == MIX["seed"]
    assert parse_shape_mix(in_mix).request_seed == 99
    assert resolve_request_seed(argparse.Namespace(request_seed=None, shape_mix=in_mix)) == 99
    assert resolve_request_seed(argparse.Namespace(request_seed=5, shape_mix=in_mix)) == 5
    assert resolve_request_seed(argparse.Namespace(request_seed=None, shape_mix=None)) is None
    from_mix = [r.prompt for r in _take(_dataset({**MIX, "request_seed": 99}), 20)]
    from_cli = [r.prompt for r in _take(_dataset(request_seed=99), 20)]
    assert from_mix == from_cli


def test_prefixes_are_built_lazily_and_cached() -> None:
    huge = {
        "seed": 1,
        "classes": [
            {
                "name": "huge",
                "weight": 1,
                "prompt_tokens": 600,
                "cached_prompt_tokens": 500,
                "generated_tokens": 1,
                "prefix_pool_size": 1_000_000,
            }
        ],
    }
    ds = _dataset(huge)

    assert ds.built_prefixes == 0
    requests = _take(ds, 20)
    assert ds.built_prefixes == len({r.prefix_index for r in requests}) <= 20
    # a prefix depends only on (seed, class, index), not on when it was first built
    fresh = _dataset(huge)
    assert fresh.prefix(0, requests[-1].prefix_index) == ds.prefix(0, requests[-1].prefix_index)
    assert ds.built_prefixes == len({r.prefix_index for r in requests})


def test_build_prefixes_upfront_matches_lazy_and_stays_compact() -> None:
    mix = json.loads(json.dumps(MIX))
    mix["classes"][0]["prefix_pool_size"] = 400
    eager = _dataset(mix)
    lazy = _dataset(mix)

    seconds = eager.build_prefixes()
    assert seconds >= 0
    assert eager.built_prefixes == sum(eager.pool_sizes) == 400 + 2
    assert eager.pool_tokens == 400 * 400 + 2 * 1000
    for ci, size in enumerate(eager.pool_sizes):
        for j in range(size):
            assert eager.prefix(ci, j) == lazy.prefix(ci, j)
    assert eager.built_prefixes == 402
    # (offset, width) per prefix plus the corpus, not the rendered text
    descriptors = 8 * 402 + 8 * eager.tail_variants * len(mix["classes"])
    assert eager.resident_bytes() == eager._corpus.resident_bytes() + descriptors
    assert sum(len(p.encode()) for ci in range(3) for p in _pool(eager, ci)) > 10 * eager.resident_bytes()


def test_no_affinity_for_cold_or_single_prefix_classes() -> None:
    mix = {
        "seed": 2,
        "classes": [
            {
                "name": "pool",
                "weight": 1,
                "prompt_tokens": 300,
                "cached_prompt_tokens": 200,
                "generated_tokens": 1,
                "prefix_pool_size": 3,
            },
            {"name": "single", "weight": 1, "prompt_tokens": 300, "cached_prompt_tokens": 200, "generated_tokens": 1},
            {"name": "cold", "weight": 1, "prompt_tokens": 300, "generated_tokens": 1},
        ],
    }
    provider = FireworksProvider(MODEL, _options(chat=True))
    seen = {}
    for r in _take(_dataset(mix), 200):
        _, headers = build_shape_mix_request(provider, r)
        seen.setdefault(r.class_name, set()).add(None if headers is None else headers["x-session-affinity"])

    assert seen["pool"] == {"shape-2-pool-0", "shape-2-pool-1", "shape-2-pool-2"}
    assert seen["single"] == {None}
    assert seen["cold"] == {None}


def test_prompt_tokens_checked_against_chat_overhead_at_parse_time() -> None:
    mix = json.dumps(
        {
            "classes": [
                {"name": "ok", "weight": 1, "prompt_tokens": 100, "generated_tokens": 1},
                {"name": "tiny", "weight": 1, "prompt_tokens": 40, "generated_tokens": 1},
            ]
        }
    )

    parse_shape_mix(mix)
    with pytest.raises(ValueError, match=r"at least 52 .*chat template overhead 20.*tiny \(40\)"):
        parse_shape_mix(mix, chat_overhead_tokens=20)
    with pytest.raises(ValueError, match="tiny"):
        validate_shape_mix_options(argparse.Namespace(chat=False, tokenizer=None, shape_mix=mix.replace("40", "31")))


@pytest.mark.parametrize("chat", [True, False])
@pytest.mark.parametrize("provider_cls", [FireworksProvider, VllmProvider, OpenAIProvider])
def test_forced_output_length_and_affinity(provider_cls, chat: bool) -> None:
    provider = provider_cls(MODEL, _options(chat))
    by_class = {}
    for r in _take(_dataset(), 200):
        by_class.setdefault(r.class_name, r)

    for name, r in by_class.items():
        data, headers = build_shape_mix_request(provider, r)
        assert data["max_tokens"] == data["min_tokens"] == r.generated_tokens
        assert data["ignore_eos"] is True
        if chat:
            assert data["messages"] == [{"role": "user", "content": r.prompt}]
            assert "prompt" not in data
        else:
            assert data["prompt"] == r.prompt
            assert "messages" not in data
        if name == "cold":
            assert headers is None
        else:
            assert headers == {"x-session-affinity": r.affinity_key}
        json.dumps(data)


@pytest.mark.parametrize("cv2", [0.25, 1.0, 4.0, 10.0])
def test_gamma_interarrival_mean_and_cv2(cv2: float) -> None:
    rng = random.Random(0)
    mean_wait = 1 / 20
    samples = [sample_interarrival("gamma", mean_wait, cv2, rng) for _ in range(200_000)]
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)

    assert mean == pytest.approx(mean_wait, rel=0.03)
    assert var / mean**2 == pytest.approx(cv2, rel=0.08)


def test_exponential_is_the_cv2_one_case() -> None:
    rng = random.Random(1)
    samples = [sample_interarrival("exponential", 0.1, rng=rng) for _ in range(100_000)]
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)

    assert var / mean**2 == pytest.approx(1.0, rel=0.05)


def test_pacer_rejects_non_positive_cv2() -> None:
    with pytest.raises(ValueError):
        FixedQPSPacer(10, "gamma", 0)


def _gaps(pacer: FixedQPSPacer, n: int) -> list[float]:
    times = [pacer.next_offset() for _ in range(n + 1)]
    return [b - a for a, b in zip(times, times[1:])]


def test_seeded_pacer_repeats_the_arrival_sequence() -> None:
    incumbent = _gaps(FixedQPSPacer(5, "gamma", 4.0, seed=11), 50)
    candidate = _gaps(FixedQPSPacer(5, "gamma", 4.0, seed=11), 50)
    other = _gaps(FixedQPSPacer(5, "gamma", 4.0, seed=12), 50)

    assert candidate == pytest.approx(incumbent, abs=1e-9)
    assert other != pytest.approx(incumbent, abs=1e-9)


def test_per_class_summary_columns() -> None:
    stats = RequestStats()
    for ttft, total, cached in [(100, 1000, 380), (120, 1100, 400), (140, 1200, 400)]:
        stats.log_request("METRIC", "class_chat_time_to_first_token", ttft, 0)
        stats.log_request("METRIC", "class_chat_total_latency", total, 0)
        stats.log_request("METRIC", "class_chat_cached_tokens", cached, 0)
    stats.log_request("METRIC", "class_cold_total_latency", 500, 0)
    stats.log_request("METRIC", "class_cold_cached_tokens", 0, 0)

    entries = shape_mix_summary_entries(stats, ["chat", "cold", "unused"], stream=True)

    assert entries["class_chat_requests"] == 3
    assert entries["class_chat_time_to_first_token"] == 120
    assert entries["class_chat_total_latency"] == 1100
    assert entries["class_chat_cached_tokens"] == pytest.approx(393.33, abs=0.01)
    assert entries["class_chat_P50_time_to_first_token"] == 120
    assert entries["class_chat_P99_total_latency"] == 1200
    assert entries["class_cold_requests"] == 1
    assert entries["class_cold_cached_tokens"] == 0
    assert entries["class_cold_time_to_first_token"] == ""
    assert entries["class_unused_requests"] == 0
    assert entries["class_unused_total_latency"] == ""

    non_stream = shape_mix_summary_entries(stats, ["chat"], stream=False)
    assert non_stream["class_chat_time_to_first_token"] == ""


@pytest.mark.parametrize("chat", [False, True])
def test_real_tokenizer_exact_lengths(chat: bool) -> None:
    """Exercises BPE merges and chat-template trimming; skipped when the tokenizer can't be loaded."""
    from load_test import (
        DEFAULT_TOKENIZER,
        _load_auto_tokenizer,
        chat_template_head_tokens,
        chat_template_token_ids,
    )

    path = os.environ.get("SHAPE_MIX_TEST_TOKENIZER", DEFAULT_TOKENIZER)
    try:
        tokenizer = _load_auto_tokenizer(path)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    overhead = len(chat_template_token_ids(tokenizer, path)) if chat else 0
    head = chat_template_head_tokens(tokenizer, path) if chat else 0
    ds = _dataset(tokenizer=tokenizer, chat_overhead_tokens=overhead, chat_head_tokens=head)
    ds.build_prefixes()

    for r in _take(ds, 100):
        if chat:
            ids = chat_template_token_ids(tokenizer, path, r.prompt)
        else:
            ids = tokenizer.encode(r.prompt, add_special_tokens=False)
        assert len(ids) == r.prompt_tokens
        if r.prefix_index is not None:
            prefix = ds.prefix(r.class_index, r.prefix_index)
            cached = head + ds.prefix_tokens[r.class_index]
            assert cached == ds.mix.classes[r.class_index].cached_prompt_tokens
            expected = (chat_template_token_ids(tokenizer, path, prefix)[:head] if chat else []) + tokenizer.encode(
                prefix, add_special_tokens=False
            )
            assert ids[:cached] == expected


def _schedule(pacer: FixedQPSPacer, seconds: float) -> list[float]:
    times = []
    while True:
        t = pacer.next_offset()
        if t > seconds:
            return times
        times.append(t)


@pytest.mark.parametrize(
    "distribution,cv2,window,dispersion",
    [
        ("exponential", 1.0, 10.0, 3.0),
        ("gamma", 1.4, 10.0, 3.0),
        ("exponential", 1.0, 5.0, 6.0),
        ("gamma", 4.0, 20.0, 10.0),
    ],
)
def test_modulated_rate_reaches_target_10s_dispersion(distribution, cv2, window, dispersion) -> None:
    qps, seconds = 5.0, 40_000
    times = _schedule(
        FixedQPSPacer(qps, distribution, cv2, seed=7, modulation_window=window, dispersion=dispersion), seconds
    )

    assert len(times) / seconds == pytest.approx(qps, rel=0.05)
    assert count_dispersion(times, 10.0) == pytest.approx(dispersion, rel=0.12)


def test_unmodulated_dispersion_and_cv2_match_the_renewal_process() -> None:
    times = _schedule(FixedQPSPacer(5.0, "gamma", 1.4, seed=3), 20_000)

    assert gap_cv2(times) == pytest.approx(1.4, rel=0.06)
    assert count_dispersion(times, 10.0) == pytest.approx(1.4, rel=0.12)
    assert gap_cv2([0.1 * i for i in range(100)]) == pytest.approx(0, abs=1e-9)
    assert count_dispersion([0.1 * i + 0.05 for i in range(1000)], 1.0) == pytest.approx(0, abs=0.01)


def test_modulation_needs_a_dispersion_and_skips_reachable_targets(caplog) -> None:
    with pytest.raises(ValueError, match="needs --qps-dispersion"):
        FixedQPSPacer(5.0, "exponential", modulation_window=10.0)
    with caplog.at_level("WARNING", logger="load_test"):
        pacer = FixedQPSPacer(5.0, "gamma", 4.0, seed=1, modulation_window=10.0, dispersion=3.0)
    assert pacer.rate_variance == 0
    assert "not modulated" in caplog.text
    # a 10 s window over a 10 s rate grid: g = 100 - 1000 / 30, so var = (D - c) * qps * 10 / g
    assert modulation_rate_variance(5.0, 3.0, 1.0, 10.0) == pytest.approx(2 * 5 * 10 / (100 - 1000 / 30))
    assert modulation_rate_variance(5.0, 3.0, 1.0, 2.0) == pytest.approx(2 * 5 * 10 / (10 * 2 - 4 / 3))


def test_modulated_schedule_is_seeded() -> None:
    a = [FixedQPSPacer(5, "gamma", 1.4, seed=9, modulation_window=10, dispersion=3).next_offset() for _ in range(1)]
    pacers = [FixedQPSPacer(5, "gamma", 1.4, seed=s, modulation_window=10, dispersion=3) for s in (9, 9, 10)]
    runs = [[p.next_offset() for _ in range(200)] for p in pacers]

    assert a[0] == runs[0][0]
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]


def test_qps_dispersion_defaults_to_the_profile() -> None:
    top = json.dumps({**MIX, "arrivals": {"cv2": 2.0, "dispersion_10s": 4.2}})
    router = json.dumps({**MIX, "arrivals": {"router_level": {"dispersion_10s": 7.5}}})
    ns = lambda **kw: argparse.Namespace(**{"qps_dispersion": None, "shape_mix": None, **kw})

    assert resolve_qps_dispersion(ns(shape_mix=top)) == 4.2
    assert resolve_qps_dispersion(ns(shape_mix=router)) == 7.5
    assert resolve_qps_dispersion(ns(shape_mix=router, qps_dispersion=2.0)) == 2.0
    assert resolve_qps_dispersion(ns(shape_mix=json.dumps(MIX))) is None


def test_arrival_summary_reports_intended_delivered_and_lateness() -> None:
    rng = random.Random(0)
    intended, t = [], 0.0
    for _ in range(3000):
        t += rng.expovariate(10)
        intended.append(t)
    actual = [i + 0.002 for i in intended]
    actual[5] += 0.5

    entries = arrival_summary_entries(intended, actual)

    assert entries["intended_gap_cv2"] == pytest.approx(1.0, rel=0.1)
    assert entries["delivered_gap_cv2"] == pytest.approx(entries["intended_gap_cv2"], rel=0.05)
    assert entries["delivered_dispersion_1s"] == pytest.approx(entries["intended_dispersion_1s"], rel=0.05)
    assert entries["intended_dispersion_10s"] == pytest.approx(1.0, rel=0.35)
    assert entries["send_lateness_p50_ms"] == pytest.approx(2.0, abs=0.01)
    assert entries["send_lateness_max_ms"] == pytest.approx(502.0, abs=0.01)
    assert arrival_summary_entries([], []) == {}


def test_reuse_overhang_lengthens_prefixes_not_prompts(caplog) -> None:
    tokenizer = WordTokenizer()
    mix = json.loads(json.dumps(MIX))
    mix["classes"].append(
        {
            "name": "tight",
            "weight": 0.1,
            "prompt_tokens": 500,
            "cached_prompt_tokens": 450,
            "generated_tokens": 1,
            "prefix_pool_size": 2,
        }
    )
    with caplog.at_level("WARNING", logger="load_test"):
        ds = _dataset(mix, tokenizer=tokenizer, chat_head_tokens=5, chat_overhead_tokens=12, reuse_overhang_tokens=128)

    assert ds.prefix_tokens[:3] == [400 - 5 + 128, 1000 - 5 + 128, 0]
    assert ds.reusable_tokens[:3] == [400, 1000, 0]
    assert ds.prefix_tokens[3] == 500 - 12 - 32 and ds.reusable_tokens[3] == 5 + 456 - 128
    assert "only 333 of 450 cached tokens are reusable" in caplog.text
    for r in _take(ds, 200):
        assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens - 12
        assert r.reusable_tokens == ds.reusable_tokens[r.class_index]


def test_prefix_reuse_overhang_resolution(monkeypatch) -> None:
    ns = lambda **kw: argparse.Namespace(**{"prefix_reuse_overhang": None, "tokenizer": "some/model", **kw})
    monkeypatch.setattr(load_test, "resolve_model_type", lambda path: "deepseek_v4")
    monkeypatch.setattr(load_test, "_read_config_json", lambda path: {"sliding_window": 128})
    assert resolve_prefix_reuse_overhang(ns()) == 128
    assert resolve_prefix_reuse_overhang(ns(prefix_reuse_overhang=0)) == 0
    monkeypatch.setattr(load_test, "resolve_model_type", lambda path: "llama")
    assert resolve_prefix_reuse_overhang(ns()) == 0
    assert resolve_prefix_reuse_overhang(ns(prefix_reuse_overhang=64)) == 64


def test_requests_need_no_tokenizer_calls() -> None:
    ds = _dataset()
    ds.build_prefixes()
    assert ds.fixed_nonce_tokens is not None and ds.tail_variants > 0

    def no_tokenizer(*_args, **_kwargs):
        raise AssertionError("tokenizer called while building a request")

    ds._encode = ds._encode_lengths = ds._decode = no_tokenizer
    requests = _take(ds, 300)
    assert len({r.prompt for r in requests}) == 300


class CharTokenizer:
    """One token per character: word lengths vary, so the word nonce has no fixed token length."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_tails_fitted_per_request_when_the_nonce_length_varies() -> None:
    mix = {
        "seed": 2,
        "classes": [
            {
                "name": "warm",
                "weight": 1,
                "prompt_tokens": 900,
                "cached_prompt_tokens": 600,
                "generated_tokens": 1,
                "prefix_pool_size": 2,
            },
            {"name": "cold", "weight": 1, "prompt_tokens": 400, "generated_tokens": 1},
        ],
    }
    tokenizer = CharTokenizer()
    ds = _dataset(mix, tokenizer=tokenizer)

    assert ds.fixed_nonce_tokens is None and ds.tail_variants == 0
    requests = _take(ds, 40)
    assert all(len(tokenizer.encode(r.prompt)) == r.prompt_tokens for r in requests)
    assert len({r.prompt for r in requests}) == 40


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


@pytest.mark.parametrize(
    "tokenizer_path,overhang",
    [
        ("NousResearch/Meta-Llama-3.1-8B-Instruct", 0),
        ("NousResearch/Meta-Llama-3.1-8B-Instruct", 128),
        ("deepseek-ai/DeepSeek-V4-Flash", None),
    ],
)
@pytest.mark.parametrize("chat", [True, False])
def test_real_tokenizer_shared_tokens_match_the_class_prefix(tokenizer_path: str, overhang, chat: bool) -> None:
    """Render requests with a real tokenizer (and chat template) and count the tokens two requests on the
    same prefix share: that's the shared prefix the server sees, cached_prompt_tokens plus the overhang."""
    from load_test import _load_auto_tokenizer, chat_template_head_tokens, chat_template_token_ids

    try:
        tokenizer = _load_auto_tokenizer(tokenizer_path)
        if overhang is None:
            overhang = resolve_prefix_reuse_overhang(
                argparse.Namespace(prefix_reuse_overhang=None, tokenizer=tokenizer_path)
            )
        head = chat_template_head_tokens(tokenizer, tokenizer_path) if chat else 0
        overhead = len(chat_template_token_ids(tokenizer, tokenizer_path)) if chat else 0
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    ds = _dataset(
        tokenizer=tokenizer, chat_overhead_tokens=overhead, chat_head_tokens=head, reuse_overhang_tokens=overhang
    )
    ds.build_prefixes()
    render = (
        (lambda p: chat_template_token_ids(tokenizer, tokenizer_path, p))
        if chat
        else (lambda p: tokenizer.encode(p, add_special_tokens=False))
    )

    by_slot: dict = {}
    for r in _take(ds, 120):
        ids = render(r.prompt)
        assert len(ids) == r.prompt_tokens
        by_slot.setdefault((r.class_index, r.prefix_index), []).append(ids)
    for (ci, prefix_index), prompts in by_slot.items():
        cached = ds.mix.classes[ci].cached_prompt_tokens
        for a, b in zip(prompts, prompts[1:]):
            shared = _common_prefix(a, b)
            if prefix_index is None:
                assert shared <= head + 2
                continue
            # the prefix, then the fixed tail start ("\n\nRequest"), then rarely one shared nonce word
            assert cached + overhang <= shared <= cached + overhang + ds._tail_start_tokens + 1
            assert shared - overhang - ds._tail_start_tokens <= ds.reusable_tokens[ci] == cached


def test_a_stall_moves_the_schedule_once_without_a_burst(monkeypatch) -> None:
    import time as _time
    from types import SimpleNamespace

    from load_test import InitTracker, LLMUser, SendTimeline

    InitTracker.spawned.set()
    SendTimeline.reset()
    monkeypatch.setattr(InitTracker, "last_late_warning", 0.0)
    pacer = FixedQPSPacer(10, "constant", seed=1)
    pacer.start = _time.monotonic() - 3.0  # the generator stalled for 3 s: the first ~30 sends are overdue
    sent = []
    user = SimpleNamespace(
        qps_pacer=pacer,
        environment=SimpleNamespace(parsed_options=argparse.Namespace(qps_max_lateness=0.5)),
        _prepare_request=lambda: "prepared",
        _in_flight=SimpleNamespace(spawn=lambda fn, prepared, intended: sent.append(intended)),
        _send_paced=None,
    )
    user._pace_and_send = lambda *args: LLMUser._pace_and_send(user, *args)

    LLMUser._paced_request(user)  # 2.9 s late: the schedule moves ~2.9 s later and this send moves with it
    assert SendTimeline.skipped == 0 and pacer.epoch == 1
    assert pacer.shifted_seconds == pytest.approx(2.9, abs=0.05)
    assert pacer.resync(0, 5.0) is False  # a second stale send from the same stall doesn't shift again
    assert len(sent) == 1 and abs(sent[0] - _time.monotonic()) < 0.1
    LLMUser._paced_request(user)  # the next slot follows 0.1 s later: no burst, no hole
    assert len(sent) == 2 and sent[1] - sent[0] == pytest.approx(0.1, abs=0.02)


def test_users_asleep_during_a_stall_keep_the_schedule_spacing(monkeypatch) -> None:
    import time as _time
    from types import SimpleNamespace

    import gevent

    import load_test
    from load_test import InitTracker, LLMUser, SendTimeline

    InitTracker.spawned.set()
    SendTimeline.reset()
    monkeypatch.setattr(InitTracker, "last_late_warning", 0.0)
    pacer = FixedQPSPacer(10, "constant", seed=1)
    pacer.start = _time.monotonic()
    real_sleep = gevent.sleep
    stall_end = _time.monotonic() + 1.5
    # Every user asleep before the stall ends wakes when it ends (a 1.5 s stall of the whole process)
    monkeypatch.setattr(
        load_test.gevent, "sleep", lambda seconds=0: real_sleep(max(seconds, stall_end - _time.monotonic()))
    )
    sent = []
    user = SimpleNamespace(
        qps_pacer=pacer,
        environment=SimpleNamespace(parsed_options=argparse.Namespace(qps_max_lateness=0.5)),
        _prepare_request=lambda: "prepared",
        _in_flight=SimpleNamespace(spawn=lambda fn, prepared, intended: sent.append((intended, _time.monotonic()))),
        _send_paced=None,
    )
    user._pace_and_send = lambda *args: LLMUser._pace_and_send(user, *args)
    gevent.joinall([gevent.spawn(LLMUser._paced_request, user) for _ in range(20)])  # slots over the next 2 s

    assert all(at - intended <= 0.5 + 0.05 for intended, at in sent)  # nothing goes out more than 0.5 s late
    assert pacer.epoch == 1  # the first late waker shifts the schedule, once
    # N6-140-1: every claimed send goes out, at the schedule's own spacing, from the end of the stall: no burst,
    # no hole (the old skip left a gap of min(stall, users / QPS))
    assert len(sent) == 20 and SendTimeline.skipped == 0
    times = sorted(at for _, at in sent)
    assert times[0] - stall_end < 0.1
    gaps = [round(b - a, 3) for a, b in zip(times, times[1:])]
    assert all(g == pytest.approx(0.1, abs=0.03) for g in gaps), (gaps, times[0] - stall_end, pacer.shifts)


def test_prefix_pool_cache_is_reused_across_processes(tmp_path) -> None:
    tokenizer = WordTokenizer()
    cache = str(tmp_path / "pool")
    first = _dataset(tokenizer=tokenizer, pool_cache=cache)
    first.build_prefixes()
    assert first.cache_loaded_prefixes == 0
    files = sorted(os.listdir(cache))
    assert len([f for f in files if f.startswith("prefix-")]) == 2  # cold class has no prefixes
    assert len([f for f in files if f.startswith("filler-")]) == 3

    second = _dataset(tokenizer=tokenizer, pool_cache=cache)
    searched = []
    second._run_search_batch = lambda searches: searched.extend(searches)
    second.build_prefixes()
    assert second.cache_loaded_prefixes == sum(second.pool_sizes) == second.built_prefixes
    assert searched == []

    uncached = _dataset(tokenizer=tokenizer)
    assert [r.prompt for r in _take(second, 100)] == [r.prompt for r in _take(uncached, 100)]


def test_prefix_pool_cache_grows_with_the_pool_and_misses_on_other_keys(tmp_path) -> None:
    tokenizer = WordTokenizer()
    cache = str(tmp_path / "pool")
    _dataset(tokenizer=tokenizer, pool_cache=cache).build_prefixes()

    bigger = json.loads(json.dumps(MIX))
    bigger["classes"][0]["prefix_pool_size"] = 10
    grown = _dataset(bigger, tokenizer=tokenizer, pool_cache=cache)
    grown.build_prefixes()
    assert grown.cache_loaded_prefixes == 3 + 2
    again = _dataset(bigger, tokenizer=tokenizer, pool_cache=cache)
    again.build_prefixes()
    assert again.cache_loaded_prefixes == 10 + 2
    assert _pool(again, 0) == _pool(_dataset(bigger, tokenizer=tokenizer), 0)

    other_seed = _dataset({**MIX, "seed": 99}, tokenizer=tokenizer, pool_cache=cache)
    other_seed.build_prefixes()
    assert other_seed.cache_loaded_prefixes == 0
    longer = _dataset(tokenizer=tokenizer, pool_cache=cache, reuse_overhang_tokens=16)
    longer.build_prefixes()
    assert longer.cache_loaded_prefixes == 0


def test_corrupt_prefix_pool_cache_is_rebuilt(tmp_path) -> None:
    tokenizer = WordTokenizer()
    cache = tmp_path / "pool"
    _dataset(tokenizer=tokenizer, pool_cache=str(cache)).build_prefixes()
    for f in cache.iterdir():
        f.write_bytes(f.read_bytes()[:-3])

    ds = _dataset(tokenizer=tokenizer, pool_cache=str(cache))
    ds.build_prefixes()
    assert ds.cache_loaded_prefixes == 0
    assert [r.prompt for r in _take(ds, 50)] == [r.prompt for r in _take(_dataset(tokenizer=tokenizer), 50)]


def test_request_outcomes_classify_timeouts_and_299s() -> None:
    import requests
    import urllib3

    from load_test import RequestOutcomes, _is_timeout, request_outcome_entries, request_timeout

    read_timeout = requests.exceptions.ConnectionError(
        urllib3.exceptions.ReadTimeoutError(None, "/v1/chat/completions", "Read timed out.")
    )
    assert _is_timeout(read_timeout)
    assert _is_timeout(requests.exceptions.ReadTimeout("slow"))
    assert not _is_timeout(requests.exceptions.ConnectionError("refused"))
    assert RequestOutcomes.classify(None, 299) == "http_299"
    assert RequestOutcomes.classify(read_timeout, 0) == "timeout"
    assert RequestOutcomes.classify(None, 500) == "other_error"
    assert request_timeout(argparse.Namespace(request_timeout=0)) is None
    assert request_timeout(argparse.Namespace(request_timeout=120.0)) == 120.0

    RequestOutcomes.reset()
    for kind, cls in [("timeout", "chat"), ("timeout", "chat"), ("http_299", "rag"), ("cancel_rate", "chat")]:
        RequestOutcomes.record(kind, cls)
    stats = RequestStats()
    for _ in range(10):
        stats.log_request("POST", "/v1/chat/completions", 100, 0)
        stats.log_request("METRIC", "total_latency", 100, 0)
    entries = request_outcome_entries(stats, ["chat", "rag"])
    RequestOutcomes.reset()

    assert (
        entries["requests_timeout"] == 2 and entries["requests_http_299"] == 1 and entries["requests_cancel_rate"] == 1
    )
    assert entries["client_cancelled_fraction"] == pytest.approx(3 / 10)
    assert entries["class_chat_timeouts"] == 2 and entries["class_rag_http_299"] == 1


def test_prefix_fill_sends_every_prefix_once_shaped_like_traffic() -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer)
    ds.build_prefixes()
    plan = ds.fill_plan()

    assert sorted(plan) == [(ci, j) for ci, n in enumerate(ds.pool_sizes) for j in range(n)]
    assert plan[:4] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    requests = [ds.fill_request(ci, j) for ci, j in plan]
    assert all(r.generated_tokens == 1 for r in requests)
    assert len({r.prompt for r in requests}) == len(plan)
    for (ci, j), r in zip(plan, requests):
        assert r.prompt.startswith(ds.prefix(ci, j))
        assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens == ds.mix.classes[ci].prompt_tokens
    data, _ = build_shape_mix_request(FireworksProvider(MODEL, _options(chat=True)), requests[0])
    assert data["max_tokens"] == data["min_tokens"] == 1


def test_prefix_fill_summary(monkeypatch) -> None:
    from load_test import PrefixFill, prefix_fill_summary_entries

    monkeypatch.setattr(PrefixFill, "requests", 3)
    monkeypatch.setattr(PrefixFill, "prompt_tokens_sent", 9000)
    monkeypatch.setattr(PrefixFill, "prefix_tokens_sent", 7500)
    monkeypatch.setattr(PrefixFill, "seconds", 2.0)
    stats = RequestStats()
    for prompt, cached in [(3000, 0), (3000, 2500), (3000, 0)]:
        stats.log_request("METRIC", "class_chat_prompt_tokens", prompt, 0)
        stats.log_request("METRIC", "class_chat_cached_tokens", cached, 0)

    entries = prefix_fill_summary_entries(stats, ["chat", "cold"])

    assert entries["prefix_fill_requests"] == 3
    assert entries["prefix_fill_server_prompt_tokens"] == 9000
    assert entries["prefix_fill_server_cached_tokens"] == 2500
    assert entries["prefix_fill_prompt_tokens_per_s"] == 4500


def test_delivered_time_is_stamped_once_the_request_is_on_the_wire() -> None:
    import threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import requests

    from load_test import _begin_wire_timing, _timed_adapter, _wire_send_time

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            _time.sleep(0.3)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session = requests.Session()
    session.mount("http://", _timed_adapter(pool_maxsize=4))
    try:
        dispatched = _begin_wire_timing()
        session.post(f"http://127.0.0.1:{server.server_port}/v1/completions", data=b"x" * 200_000)
        answered = _time.monotonic()
        wire = _wire_send_time(dispatched)
    finally:
        server.shutdown()

    assert dispatched < wire < answered - 0.25  # after the body went out, before the 0.3 s server delay
    assert _wire_send_time(123.0) == 123.0  # consumed: a request that never went out falls back


@pytest.mark.parametrize(
    "distribution,cv2,dispersion", [("exponential", 1.0, 10.0), ("gamma", 1.4, 10.0), ("gamma", 1.4, 42.0)]
)
def test_modulation_keeps_the_gap_cv2_and_reaches_the_dispersion(distribution, cv2, dispersion) -> None:
    qps, seconds = 20.0, 30_000
    pacer = FixedQPSPacer(qps, distribution, cv2, seed=5, modulation_window=10.0, dispersion=dispersion)
    times = _schedule(pacer, seconds)

    assert pacer.base_cv2 < cv2
    assert gap_cv2(times) == pytest.approx(cv2, rel=0.1)
    assert count_dispersion(times, 10.0) == pytest.approx(dispersion, rel=0.15)


def test_modulated_rate_blocks_average_exactly_the_nominal_qps() -> None:
    # A 5-minute tier with W=10 is one block of 30 windows: its mean rate is exactly --qps, whatever the seed
    for seed in range(5):
        pacer = FixedQPSPacer(
            20.0, "exponential", seed=seed, modulation_window=10.0, dispersion=42.0, rate_block_windows=30
        )
        rates = [pacer._window_rate] + [pacer._draw_rate() for _ in range(29)]
        assert sum(rates) / 30 == pytest.approx(20.0, rel=1e-9)
        assert len(
            _schedule(
                FixedQPSPacer(
                    20.0, "exponential", seed=seed, modulation_window=10.0, dispersion=42.0, rate_block_windows=30
                ),
                300,
            )
        ) == pytest.approx(6000, rel=0.03)


def test_unreachable_modulated_gap_cv2_warns(caplog) -> None:
    with caplog.at_level("WARNING", logger="load_test"):
        pacer = FixedQPSPacer(1.0, "exponential", seed=1, modulation_window=10.0, dispersion=42.0)
    assert pacer.base_cv2 == pytest.approx(0.05)
    assert "can't stay at 1" in caplog.text


def test_fail_ratio_counts_only_requests() -> None:
    from load_test import request_counts

    stats = RequestStats()
    for i in range(10):
        stats.log_request("POST", "/v1/chat/completions", 100, 0)
        for metric in ("total_latency", "prompt_tokens", "cached_tokens", "time_to_first_token"):
            stats.log_request("METRIC", metric, 100, 0)
    stats.log_error("POST", "/v1/chat/completions", "timeout")
    stats.log_error("POST", "/v1/chat/completions", "timeout")

    assert request_counts(stats) == (10, 2)
    assert stats.total.num_requests == 50  # what the ratio used to divide by


def test_acceptance_probs_override_reaches_shape_mix_payloads() -> None:
    provider = FireworksProvider(MODEL, _options(chat=True, acceptance_probs_override="[0.9, 0.7, 0.5]"))
    for r in _take(_dataset(), 20):
        data, _ = build_shape_mix_request(provider, r)
        assert data["acceptance_probs_override"] == [0.9, 0.7, 0.5]
        assert data["max_tokens"] == data["min_tokens"] == r.generated_tokens and data["ignore_eos"] is True
        assert "prompt_cache_max_len" not in data


@pytest.mark.parametrize("chat", [True, False])
@pytest.mark.parametrize("provider_cls", [FireworksProvider, VllmProvider, OpenAIProvider])
def test_max_only_output_mode_caps_without_forcing(provider_cls, chat: bool) -> None:
    provider = provider_cls(MODEL, _options(chat))
    for r in _take(_dataset(), 30):
        data, headers = build_shape_mix_request(provider, r, output_mode="max")
        assert data["max_tokens"] == r.generated_tokens
        assert "min_tokens" not in data and "ignore_eos" not in data
        exact, _ = build_shape_mix_request(provider, r)
        assert exact["min_tokens"] == r.generated_tokens and exact["ignore_eos"] is True
        assert headers == (None if r.affinity_key is None else {"x-session-affinity": r.affinity_key})


@pytest.mark.parametrize("chat", [True, False])
def test_unrelated_prompts_diverge_within_the_first_block(chat: bool) -> None:
    """Only requests on the same prefix slot may share more than the template head, the tail start and
    rarely one nonce word; anything longer would hand the server a common leading text to cache."""
    from load_test import DEFAULT_TOKENIZER, _load_auto_tokenizer, chat_template_head_tokens, chat_template_token_ids

    try:
        tokenizer = _load_auto_tokenizer(DEFAULT_TOKENIZER)
        head = chat_template_head_tokens(tokenizer, DEFAULT_TOKENIZER) if chat else 0
        overhead = len(chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER)) if chat else 0
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    render = (
        (lambda p: chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER, p))
        if chat
        else (lambda p: tokenizer.encode(p, add_special_tokens=False))
    )
    runs = [
        _dataset(tokenizer=tokenizer, chat_overhead_tokens=overhead, chat_head_tokens=head, request_seed=seed)
        for seed in (1, 2)
    ]
    requests = [(r, render(r.prompt)) for ds in runs for r in _take(ds, 80)]

    longest = 0
    for i, (ra, ia) in enumerate(requests):
        for rb, ib in requests[i + 1 :]:
            if ra.prefix_index is not None and (ra.class_index, ra.prefix_index) == (rb.class_index, rb.prefix_index):
                continue
            longest = max(longest, _common_prefix(ia, ib))
    assert longest <= head + 8 < 64


INSTRUCTION = "Summarize the text above in two sentences."


def test_instruction_suffix_ends_every_prompt_at_exact_length(tmp_path) -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer, instruction=f"  {INSTRUCTION}\n")
    ds.build_prefixes()

    requests = _take(ds, 200) + [ds.fill_request(0, 0)]
    for r in requests:
        assert r.prompt.endswith("\n\n" + INSTRUCTION)
        assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens == ds.mix.classes[r.class_index].prompt_tokens
    assert len({r.prompt for r in requests}) == len(requests)
    # the instruction changes the fillers, so it's part of their cache key
    cache = str(tmp_path / "pool")
    _dataset(tokenizer=tokenizer, pool_cache=cache).build_prefixes()
    other = _dataset(tokenizer=tokenizer, pool_cache=cache, instruction=INSTRUCTION)
    assert [other._load_cached(*other._filler_cache_slot(ci)) for ci in range(3)] == [16, 16, 16]
    assert [r.prompt for r in _take(other, 20)] == [
        r.prompt for r in _take(_dataset(tokenizer=tokenizer, instruction=INSTRUCTION), 20)
    ]


def test_instruction_too_long_for_a_class_is_rejected() -> None:
    mix = {"classes": [{"name": "tiny", "weight": 1, "prompt_tokens": 40, "generated_tokens": 1}]}
    with pytest.raises(ValueError, match="tiny"):
        _dataset(mix, instruction=" ".join(["word"] * 30))


def test_custom_corpus_file(tmp_path) -> None:
    from load_test import resolve_shape_mix_corpus

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n\n".join(f"Paragraph {i} about turbines, rotors and blades spinning in the wind." for i in range(50))
    )
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer, corpus_path=resolve_shape_mix_corpus(str(corpus)))

    r = next(r for r in _take(ds, 20) if r.prefix_index is not None)
    assert "turbines" in r.prompt and "limerick" not in r.prompt.lower()
    assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens
    assert resolve_shape_mix_corpus("code").endswith("code_corpus.txt.gz")
    import gzip as _gzip

    with _gzip.open(resolve_shape_mix_corpus("code"), "rt", encoding="utf-8") as f:
        assert len(f.read()) > 1_500_000  # not a few KB reused cyclically (N5-140-2)
    assert resolve_shape_mix_corpus(None).endswith("prose_corpus.txt.gz")
    assert resolve_shape_mix_corpus("limericks").endswith("limericks.txt")
    with pytest.raises(ValueError):
        resolve_shape_mix_corpus(str(tmp_path / "missing.txt"))


@pytest.mark.parametrize("chat", [True, False])
def test_real_tokenizer_code_corpus_and_instruction(chat: bool) -> None:
    from load_test import (
        DEFAULT_TOKENIZER,
        _load_auto_tokenizer,
        chat_template_head_tokens,
        chat_template_token_ids,
        resolve_shape_mix_corpus,
    )

    try:
        tokenizer = _load_auto_tokenizer(DEFAULT_TOKENIZER)
        head = chat_template_head_tokens(tokenizer, DEFAULT_TOKENIZER) if chat else 0
        overhead = len(chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER)) if chat else 0
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    ds = ShapeMixDataset(
        mix=parse_shape_mix(json.dumps(MIX)),
        tokenizer=tokenizer,
        corpus_path=resolve_shape_mix_corpus("code"),
        chat_overhead_tokens=overhead,
        chat_head_tokens=head,
        instruction=INSTRUCTION,
    )
    ds.build_prefixes()
    assert ds.fixed_nonce_tokens is not None

    for r in _take(ds, 60):
        ids = (
            chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER, r.prompt)
            if chat
            else tokenizer.encode(r.prompt, add_special_tokens=False)
        )
        assert len(ids) == r.prompt_tokens
        assert r.prompt.endswith(INSTRUCTION)


SAMPLED_MIX = {
    "seed": 4,
    "classes": [
        {
            "name": "chat",
            "weight": 0.7,
            "prompt_tokens": 1500,
            "cached_prompt_tokens": 1000,
            "generated_tokens": 100,
            "prefix_pool_size": 3,
            "uncached_tokens_quantiles": {"p0": 100, "p50": 400, "p90": 900, "p100": 1500},
            "generated_tokens_quantiles": [[0.0, 10], [0.5, 80], [1.0, 400]],
        },
        {
            "name": "cold",
            "weight": 0.3,
            "prompt_tokens": 800,
            "generated_tokens": 50,
            "uncached_tokens_quantiles": {"0.1": 300, "0.9": 1300},
        },
    ],
}


def test_quantile_parsing_and_interpolation() -> None:
    from load_test import quantile_value

    mix = parse_shape_mix(json.dumps(SAMPLED_MIX))
    chat, cold = mix.classes
    assert chat.uncached_tokens_quantiles == ((0.0, 100.0), (0.5, 400.0), (0.9, 900.0), (1.0, 1500.0))
    assert chat.generated_tokens_quantiles == ((0.0, 10.0), (0.5, 80.0), (1.0, 400.0))
    assert cold.generated_tokens_quantiles is None
    assert quantile_value(chat.uncached_tokens_quantiles, 0.25) == pytest.approx(250)
    assert quantile_value(cold.uncached_tokens_quantiles, 0.05) == 300  # flat below the lowest point
    assert quantile_value(cold.uncached_tokens_quantiles, 0.95) == 1300
    for bad in [{"p50": 10, "p90": 5}, {"p150": 3}, {"median": 3}, {"p50": -1}, "lots"]:
        spec = json.loads(json.dumps(MIX))
        spec["classes"][0]["uncached_tokens_quantiles"] = bad
        with pytest.raises(ValueError):
            parse_shape_mix(json.dumps(spec))


def test_length_sampling_is_off_by_default() -> None:
    plain = json.loads(json.dumps(SAMPLED_MIX))
    for c in plain["classes"]:
        c.pop("uncached_tokens_quantiles", None)
        c.pop("generated_tokens_quantiles", None)
    with_quantiles = [(r.prompt, r.generated_tokens) for r in _take(_dataset(SAMPLED_MIX), 100)]
    without = [(r.prompt, r.generated_tokens) for r in _take(_dataset(plain), 100)]
    assert with_quantiles == without


REALISTIC_SAMPLED_MIX = {
    "seed": 4,
    "classes": [
        {
            "name": "chat",
            "weight": 1,
            "prompt_tokens": 3000,
            "cached_prompt_tokens": 1000,
            "generated_tokens": 420,
            "prefix_pool_size": 2,
            "uncached_tokens_quantiles": {"p10": 300, "p50": 1400, "p90": 4200, "p99": 9000},
            "generated_tokens_quantiles": {"p10": 40, "p50": 220, "p90": 900, "p99": 2600},
        }
    ],
}


def test_sampled_lengths_keep_the_quantiles_and_the_class_mean(caplog) -> None:
    import statistics as st

    tokenizer = WordTokenizer()
    with caplog.at_level("WARNING", logger="load_test"):
        ds = _dataset(REALISTIC_SAMPLED_MIX, tokenizer=tokenizer, sample_lengths=True)
    assert "scaled" not in caplog.text  # reachable without moving any given quantile
    ds.build_prefixes()
    requests = _take(ds, 40000)
    q = lambda xs, p: xs[int(p * (len(xs) - 1))]
    out = sorted(r.generated_tokens for r in requests)
    # N6-140-2: given quantiles stay where the profile puts them (they used to move with a global scale)
    for level, value in ((0.1, 40), (0.5, 220), (0.9, 900), (0.99, 2600)):
        assert q(out, level) == pytest.approx(value, rel=0.1), level
    assert st.mean(out) == pytest.approx(420, rel=0.04)
    assert out[-1] > 2600  # draws go past the top quantile (the tail is extrapolated)
    unc = sorted(r.prompt_tokens - ds.reusable_tokens[0] for r in requests)
    for level, value in ((0.5, 1400), (0.9, 4200), (0.99, 9000)):
        assert q(unc, level) == pytest.approx(value, rel=0.12), level  # strata are discrete
    assert st.mean(unc) == pytest.approx(2000, rel=0.02)
    assert unc[-1] > 9000
    for r in requests[:200]:
        assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens


def test_sampled_lengths_bend_the_body_before_scaling(caplog) -> None:
    """N8-140-3: a mean below what a flat tail allows bends each segment toward its lower point, keeping every
    given quantile (it used to scale them all)."""
    import statistics as st

    spec = json.loads(json.dumps(REALISTIC_SAMPLED_MIX))
    spec["classes"][0]["generated_tokens"] = 250
    with caplog.at_level("WARNING", logger="load_test"):
        ds = _dataset(spec, sample_lengths=True)
    assert "scaled" not in caplog.text
    assert ds._generated_dist[0].straight < 1
    ds.build_prefixes()
    out = sorted(r.generated_tokens for r in _take(ds, 40000))
    assert st.mean(out) == pytest.approx(250, rel=0.04)
    q = lambda xs, p: xs[int(p * (len(xs) - 1))]
    for level, value in ((0.5, 220), (0.9, 900), (0.99, 2600)):
        assert q(out, level) == pytest.approx(value, rel=0.1), level


def test_sampled_lengths_fall_back_to_scaling_when_the_mean_is_out_of_reach(caplog) -> None:
    import statistics as st

    spec = json.loads(json.dumps(REALISTIC_SAMPLED_MIX))
    spec["classes"][0]["generated_tokens"] = 150  # below even a step through the given quantiles
    with caplog.at_level("WARNING", logger="load_test"):
        ds = _dataset(spec, sample_lengths=True)
    assert "all output quantiles scaled" in caplog.text
    ds.build_prefixes()
    assert st.mean(r.generated_tokens for r in _take(ds, 20000)) == pytest.approx(150, rel=0.05)
    # the old single-point profile (one interior quantile) also scales, keeping its mean
    ds2 = _dataset(SAMPLED_MIX, sample_lengths=True)
    ds2.build_prefixes()
    chat = [r.generated_tokens for r in _take(ds2, 8000) if r.class_index == 0]
    assert st.mean(chat) == pytest.approx(100, rel=0.04)


def test_sampled_prompts_stay_in_their_band_and_the_context(caplog) -> None:
    """N8-140-3: uncached draws are capped so the prompt stays in the class's prompt_band (widened by bands
    merged into it) and, with the largest output, inside the model's context."""
    import statistics as st

    from load_test import LengthQuantiles

    spec = {
        "seed": 4,
        "classes": [
            {
                "cache_class": "partial",
                "prompt_band": "lt_2k",
                "weight": 1,
                "prompt_tokens": 800,
                "cached_prompt_tokens": 300,
                "generated_tokens": 200,
                "prefix_pool_size": 2,
                "uncached_tokens_quantiles": {"p10": 80, "p50": 400, "p90": 1100, "p99": 1600},
            },
            {
                "cache_class": "cold",
                "prompt_band": "8k_32k",
                "weight": 1,
                "prompt_tokens": 14000,
                "generated_tokens": 300,
                "uncached_tokens_quantiles": {"p10": 9000, "p50": 13000, "p90": 22000, "p99": 30000},
                "generated_tokens_quantiles": {"p10": 50, "p50": 200, "p90": 700, "p99": 1500},
            },
        ],
    }
    mix = parse_shape_mix(json.dumps(spec))
    assert [c.prompt_tokens_max for c in mix.classes] == [2047, 32767]
    merged = dict(
        spec,
        validity={
            "merged_classes": [
                {
                    "from": {"cache_class": "partial", "prompt_band": "2k_8k"},
                    "into": {"cache_class": "partial", "prompt_band": "lt_2k"},
                }
            ]
        },
    )
    assert parse_shape_mix(json.dumps(merged)).classes[0].prompt_tokens_max == 8191
    no_band = json.loads(json.dumps(spec))
    del no_band["classes"][0]["prompt_band"]
    assert parse_shape_mix(json.dumps(no_band)).classes[0].prompt_tokens_max is None

    with caplog.at_level("WARNING", logger="load_test"):
        ds = _dataset(spec, sample_lengths=True, context_tokens=32768)
    assert "scaled" not in caplog.text
    ds.build_prefixes()
    requests = _take(ds, 60000)
    for ci, ceiling in ((0, 2047), (1, 32767)):
        prompts = [r.prompt_tokens for r in requests if r.class_index == ci]
        assert max(prompts) <= ceiling
        # the class mean still holds under the cap
        assert st.mean(prompts) == pytest.approx(spec["classes"][ci]["prompt_tokens"], rel=0.03)
    assert max(r.prompt_tokens + r.generated_tokens for r in requests) <= 32768
    assert 0 < ds.context_cut_requests < len(requests) / 1000  # a long output drawn for a long prompt is cut
    # a cap never moves a given quantile below its value
    dist = LengthQuantiles(((0.5, 400.0), (0.99, 1600.0)), cap=1000)
    assert dist.high == 1600 and dist.value(0.999999) <= 1600


def test_sampled_tails_never_drop_below_the_nonce() -> None:
    spec = json.loads(json.dumps(SAMPLED_MIX))
    spec["classes"][0]["uncached_tokens_quantiles"] = {"p0": 0, "p99": 50, "p100": 20000}
    ds = _dataset(spec, sample_lengths=True)
    assert min(ds._filler_tokens[0]) == ds._min_tail_tokens(0)
    assert all(t >= ds._min_tail_tokens(0) for t in ds._filler_tokens[0])


def test_real_tokenizer_sampled_lengths_are_exact() -> None:
    from load_test import DEFAULT_TOKENIZER, _load_auto_tokenizer

    try:
        tokenizer = _load_auto_tokenizer(DEFAULT_TOKENIZER)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    ds = _dataset(SAMPLED_MIX, tokenizer=tokenizer, sample_lengths=True)
    ds.build_prefixes()
    assert ds.fixed_nonce_tokens is not None
    for r in _take(ds, 80):
        assert len(tokenizer.encode(r.prompt, add_special_tokens=False)) == r.prompt_tokens


def _write_schedule(path, rows) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_replay_schedule_parsing(tmp_path) -> None:
    from load_test import load_replay_schedule

    names = ("chat", "agent", "cold")
    path = _write_schedule(
        tmp_path / "s.jsonl",
        [
            {"timestamp": 5000, "class": "agent", "prefix_index": 1},
            {"timestamp": 2000, "class": "chat", "output_length": 7},
            {"time_s": 2.5, "class": "cold"},
        ],
    )
    entries = load_replay_schedule(path, names)

    assert [(e.offset, e.class_index, e.prefix_index, e.output_length) for e in entries] == [
        (0.0, 0, None, 7),
        (0.5, 2, None, None),
        (3.0, 1, 1, None),
    ]
    for bad in [
        [{"class": "chat"}],
        [{"timestamp": 1, "class": "nope"}],
        [{"timestamp": 1, "class": "chat", "output_length": -1}],
    ]:
        with pytest.raises(ValueError):
            load_replay_schedule(_write_schedule(tmp_path / f"bad{len(str(bad))}.jsonl", bad), names)


def test_replay_pacer_hands_out_recorded_offsets(tmp_path) -> None:
    from load_test import ReplayPacer, load_replay_schedule

    path = _write_schedule(tmp_path / "s.jsonl", [{"timestamp": t, "class": "chat"} for t in (0, 0, 250, 1000)])
    pacer = ReplayPacer(load_replay_schedule(path, ("chat",)), path)
    claims = [pacer.claim_entry() for _ in range(5)]

    assert claims[-1] is None and pacer.exhausted
    start = claims[0][0]
    assert [round(c[0] - start, 6) for c in claims[:4]] == [0, 0, 0.25, 1.0]
    assert pacer.resync(0, 0.5) and not pacer.resync(0, 0.5)
    assert "4 requests over 1s" in pacer.describe()


def test_scheduled_requests_use_the_class_pool(tmp_path) -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer)
    ds.build_prefixes()

    fixed = ds.scheduled_request(1, prefix_index=1, generated_tokens=9)
    assert fixed.class_name == "agent" and fixed.prefix_index == 1 and fixed.generated_tokens == 9
    assert fixed.prompt.startswith(ds.prefix(1, 1))
    drawn = [ds.scheduled_request(0) for _ in range(60)]
    assert {r.prefix_index for r in drawn} == {0, 1, 2}
    assert all(r.generated_tokens == 50 and len(tokenizer.encode(r.prompt)) == r.prompt_tokens for r in drawn)
    cold = ds.scheduled_request(2)
    assert cold.prefix_index is None and cold.affinity_key is None
    for ci, prefix_index in ((0, 3), (2, 0)):  # out of range, and a class with no shared prefix
        with pytest.raises(ValueError):
            ds.scheduled_request(ci, prefix_index=prefix_index)


def test_replay_schedule_is_checked_at_init(tmp_path) -> None:
    path = _write_schedule(tmp_path / "s.jsonl", [{"timestamp": 0, "class": "chat", "prefix_index": 3}])
    options = argparse.Namespace(
        shape_mix=json.dumps(MIX), replay_schedule=path, chat=False, tokenizer=None, shape_mix_corpus=None
    )
    with pytest.raises(ValueError, match="outside class chat's pool of 3"):
        validate_shape_mix_options(options)


def _popular_mix(popularity=None, pool=5):
    spec = json.loads(json.dumps(MIX))
    spec["classes"][0]["prefix_pool_size"] = pool
    if popularity is not None:
        spec["classes"][0]["prefix_popularity"] = popularity
    return spec


def test_prefix_popularity_parsing() -> None:
    assert parse_shape_mix(json.dumps(_popular_mix({"zipf": 1.2}))).classes[0].prefix_popularity == ("zipf", 1.2)
    assert parse_shape_mix(json.dumps(_popular_mix([5, 0, 3, 1, 1]))).classes[0].prefix_popularity == (
        "weights",
        (5.0, 0.0, 3.0, 1.0, 1.0),
    )
    for bad in [{"zipf": -1}, [1, 2], [0, 0, 0, 0, 0], [1, -1, 1, 1, 1], {"pareto": 2}, "hot"]:
        with pytest.raises(ValueError):
            parse_shape_mix(json.dumps(_popular_mix(bad)))


def test_zipf_and_explicit_popularity_draws() -> None:
    n = 20000
    zipf = _dataset(_popular_mix({"zipf": 1.0}))
    counts = Counter(r.prefix_index for r in _take(zipf, n) if r.class_index == 0)
    total = sum(counts.values())
    harmonic = sum(1 / (j + 1) for j in range(5))
    for j in range(5):
        assert counts[j] / total == pytest.approx(1 / (j + 1) / harmonic, abs=0.015)
    assert zipf.prefix_popularity_share(0, 1) == pytest.approx(1 / harmonic)

    explicit = _dataset(_popular_mix([5, 0, 3, 1, 1]))
    counts = Counter(r.prefix_index for r in _take(explicit, n) if r.class_index == 0)
    assert counts[1] == 0
    assert counts[0] / sum(counts.values()) == pytest.approx(0.5, abs=0.02)


def test_uniform_popularity_is_unchanged_and_the_cli_default_yields_to_the_class() -> None:
    assert [r.prompt for r in _take(_dataset(), 100)] == [
        r.prompt for r in _take(_dataset(default_popularity_zipf=0.0), 100)
    ]
    ds = _dataset(_popular_mix([0, 0, 0, 0, 1]), default_popularity_zipf=2.0)
    reqs = _take(ds, 3000)
    assert {r.prefix_index for r in reqs if r.class_index == 0} == {4}  # class weights beat the default
    agent = Counter(r.prefix_index for r in reqs if r.class_index == 1)
    assert agent[0] / sum(agent.values()) == pytest.approx(4 / 5, abs=0.03)  # zipf 2 over a pool of 2


def test_large_zipf_pool_draws_fast() -> None:
    import time as _time

    ds = _dataset(_popular_mix({"zipf": 1.1}, pool=1_000_000))
    started = _time.perf_counter()
    top = Counter(ds._draw_prefix_index(0) for _ in range(20000))
    assert _time.perf_counter() - started < 2
    assert top[0] > top[1] > top[2] > 0
    assert max(top) < 1_000_000


def test_replay_draws_prefixes_by_popularity() -> None:
    ds = _dataset(_popular_mix([0, 0, 1, 0, 0]))
    ds.build_prefixes()
    assert {ds.scheduled_request(0).prefix_index for _ in range(50)} == {2}
    assert ds.scheduled_request(0, prefix_index=0).prefix_index == 0


def _llama():
    from load_test import DEFAULT_TOKENIZER, _load_auto_tokenizer, chat_template_head_tokens, chat_template_token_ids

    try:
        tokenizer = _load_auto_tokenizer(DEFAULT_TOKENIZER)
        return (
            tokenizer,
            chat_template_head_tokens(tokenizer, DEFAULT_TOKENIZER),
            len(chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER)),
        )
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.mark.parametrize("subtract", [True, False])
def test_replay_input_lengths_with_template_subtraction(tmp_path, subtract: bool) -> None:
    from load_test import DEFAULT_TOKENIZER, chat_messages_overhead, chat_template_token_ids, load_replay_schedule

    tokenizer, head, overhead = _llama()
    path = _write_schedule(
        tmp_path / "s.jsonl",
        [
            {"timestamp": 0, "class": "chat", "input_length": 900},
            {"timestamp": 10, "class": "chat", "input_length": 1234, "output_length": 5},
            {"timestamp": 20, "class": "cold", "input_length": 333},
            {"timestamp": 30, "class": "agent"},
        ],
    )
    ds = _dataset(tokenizer=tokenizer, chat_overhead_tokens=overhead, chat_head_tokens=head)
    ds.build_prefixes()
    entries = load_replay_schedule(path, ("chat", "agent", "cold"))
    ds.prepare_replay(
        entries,
        message_overhead=lambda n: chat_messages_overhead(tokenizer, DEFAULT_TOKENIZER, n),
        subtract_template=subtract,
    )

    for entry in entries:
        r = ds.replay_request(entry)
        rendered = len(chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER, r.prompt))
        content = len(tokenizer.encode(r.prompt, add_special_tokens=False))
        want = entry.input_length or ds.mix.classes[entry.class_index].prompt_tokens
        if entry.input_length is None or subtract:
            assert rendered == r.prompt_tokens == want
        else:
            assert content == want and r.prompt_tokens == want + overhead
        if entry.class_index == 0:
            assert r.prompt.startswith(ds.prefix(0, r.prefix_index))
    assert ds.replay_request(entries[1]).generated_tokens == 5
    # building a replay request needs no tokenizer call
    ds._encode = ds._encode_lengths = ds._decode = lambda *a, **k: (_ for _ in ()).throw(AssertionError("tokenizer"))
    assert [ds.replay_request(e).prompt for e in entries]


def test_replay_pods_partition_the_schedule(tmp_path) -> None:
    from load_test import load_replay_schedule

    rows = [
        {"timestamp": 100 * i, "class": "chat", **({"session_id": f"s{i % 7}"} if i % 3 == 0 else {})}
        for i in range(300)
    ]
    path = _write_schedule(tmp_path / "s.jsonl", rows)
    whole = load_replay_schedule(path, ("chat",))
    pods = [load_replay_schedule(path, ("chat",), 4, i) for i in range(4)]

    assert sorted(e.index for pod in pods for e in pod) == [e.index for e in whole]
    for pod in pods:
        assert 40 < len(pod) < 110
        assert all(e.offset == whole[e.index].offset for e in pod)  # global offsets, not per pod
    for session in {e.session for e in whole if e.session}:
        assert sum(any(e.session == session for e in pod) for pod in pods) == 1


def test_replay_pacer_counts_claims_until_released(tmp_path) -> None:
    from load_test import ReplayPacer, load_replay_schedule

    path = _write_schedule(tmp_path / "s.jsonl", [{"timestamp": t, "class": "chat"} for t in (0, 10)])
    pacer = ReplayPacer(load_replay_schedule(path, ("chat",)), path)
    pacer.claim_entry(), pacer.claim_entry()
    assert pacer.exhausted and pacer.pending == 2
    pacer.release()
    pacer.release()
    assert pacer.pending == 0


def _replay_dataset(tmp_path, rows, tokenizer=None, chat=True, **kwargs):
    from load_test import DEFAULT_TOKENIZER, chat_messages_overhead, load_replay_schedule

    head = overhead = 0
    if tokenizer is None:
        tokenizer, head, overhead = _llama()
    elif not chat:
        head = overhead = 0
    path = _write_schedule(tmp_path / "s.jsonl", rows)
    kwargs.setdefault("split_restore_classes", False)  # as DatasetHolder builds it for --replay-schedule
    ds = _dataset(
        tokenizer=tokenizer,
        chat_overhead_tokens=overhead if chat else 0,
        chat_head_tokens=head if chat else 0,
        **kwargs,
    )
    ds.build_prefixes()
    entries = load_replay_schedule(path, ("chat", "agent", "cold"))
    message_overhead = (lambda n: chat_messages_overhead(tokenizer, DEFAULT_TOKENIZER, n)) if chat else None
    ds.prepare_replay(entries, message_overhead=message_overhead)
    return ds, entries, tokenizer


def test_replay_rows_send_their_recorded_lengths(tmp_path) -> None:
    from load_test import DEFAULT_TOKENIZER, chat_template_token_ids

    rows = [
        {
            "timestamp": 0,
            "class": "chat",
            "input_length": 900,
            "cached_length": 300,
            "output_length": 7,
            "prefix_index": 1,
        },
        {
            "timestamp": 10,
            "class": "chat",
            "input_length": 900,
            "cached_length": 400,
            "output_length": 9,
            "prefix_index": 1,
        },
        {"timestamp": 20, "class": "agent", "prompt_tokens": 1500, "cached_tokens": 0, "generated_tokens": 3},
        {"timestamp": 30, "class": "cold", "input_length": 333, "cached_length": 0},
    ]
    ds, entries, tokenizer = _replay_dataset(tmp_path, rows)
    full = ds.prefix(0, 1)
    for entry in entries:
        r = ds.replay_request(entry)
        assert (
            len(chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER, r.prompt))
            == r.prompt_tokens
            == entry.input_length
        )
        assert r.recorded_cached_tokens == entry.cached_length
        assert r.generated_tokens == (entry.output_length or ds.mix.classes[entry.class_index].generated_tokens)
        if entry.cached_length:
            assert r.reusable_tokens == entry.cached_length
            reused_chars = len(full) if entry.cached_length >= 400 else None
            if reused_chars is None:
                # a shorter cached_length reuses a byte prefix of the same pool prefix
                cut = r.prompt.index("\n\nRequest")
                assert full.startswith(r.prompt[:cut])
                assert len(tokenizer.encode(r.prompt[:cut], add_special_tokens=False)) == entry.cached_length - ds._head
            else:
                assert r.prompt.startswith(full)
        else:
            assert r.reusable_tokens == 0 and r.prompt.startswith("Request ")
    assert (ds.replay_shortfall["rows"], ds.replay_shortfall["tokens"]) == (0, 0)


def test_session_rows_reuse_the_previous_prompt_and_reply(tmp_path) -> None:
    rows = [
        {
            "timestamp": 0,
            "class": "chat",
            "session_id": "a",
            "input_length": 900,
            "cached_length": 400,
            "output_length": 40,
        },
        {
            "timestamp": 5000,
            "class": "chat",
            "session_id": "a",
            "input_length": 1500,
            "cached_length": 930,
            "output_length": 60,
        },
        # branching: two rows at once both continue from the row before them in time order
        {
            "timestamp": 9000,
            "class": "chat",
            "session_id": "a",
            "input_length": 1300,
            "cached_length": 1200,
            "output_length": 20,
        },
        {
            "timestamp": 9000,
            "class": "chat",
            "session_id": "a",
            "input_length": 700,
            "cached_length": 300,
            "output_length": 20,
        },
        {"timestamp": 9500, "class": "chat", "session_id": "b", "input_length": 1000, "cached_length": 200},
    ]
    tokenizer = WordTokenizer()
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=tokenizer, chat=False)
    requests = [ds.replay_request(e) for e in entries]
    for entry, r in zip(entries, requests):
        assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens == entry.input_length
        assert r.affinity_key == f"shape-{ds.replay_salt}-session-{entry.session}"
    a0, a1, a2, a3 = requests[:4]
    # each row's reusable text is a byte prefix of (previous prompt + "\n\n" + its synthetic reply)
    assert a1.prompt.startswith(a0.prompt + "\n\n")
    assert a1.reusable_tokens == 930
    assert a2.reusable_tokens == 1200 and a2.prompt[:2000] == a1.prompt[:2000]
    assert a3.reusable_tokens == 300
    # 300 tokens fit inside the earlier text, so row 3 reuses the start of the same conversation
    assert a3.prompt.split("\n\nRequest")[0] == a2.prompt[: len(a3.prompt.split("\n\nRequest")[0])]
    assert requests[4].reusable_tokens == 200 and ds.replay_shortfall["rows"] == 0


def test_session_shorter_than_cached_length_falls_back_to_the_pool(tmp_path) -> None:
    rows = [
        {
            "timestamp": 0,
            "class": "agent",
            "session_id": "s",
            "input_length": 300,
            "cached_length": 100,
            "output_length": 5,
        },
        # the conversation so far is ~300 + 5 tokens; 900 can only come from the class pool (1000 tokens)
        {
            "timestamp": 10,
            "class": "agent",
            "session_id": "s",
            "input_length": 1100,
            "cached_length": 900,
            "output_length": 5,
        },
        # 1500 is more than any source: reuse what there is, count the shortfall
        {
            "timestamp": 20,
            "class": "agent",
            "session_id": "s",
            "input_length": 1600,
            "cached_length": 1500,
            "output_length": 5,
        },
    ]
    tokenizer = WordTokenizer()
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=tokenizer, chat=False)
    r0, r1, r2 = (ds.replay_request(e) for e in entries)
    assert r1.reusable_tokens == 900 and ds.prefix(1, r1.prefix_index).startswith(r1.prompt.split("\n\nRequest")[0])
    assert r2.reusable_tokens < 1500
    assert ds.replay_shortfall["rows"] == 1 and ds.replay_shortfall["tokens"] == 1500 - r2.reusable_tokens
    assert all(len(tokenizer.encode(r.prompt)) == r.prompt_tokens for r in (r0, r1, r2))


def test_replay_row_fields_validation_and_overrides(tmp_path) -> None:
    from load_test import load_replay_schedule

    names = ("chat", "agent", "cold")
    some = _write_schedule(
        tmp_path / "some.jsonl",
        [
            {"timestamp": 0, "class": "chat", "cached_length": 10, "input_length": 100},
            {"timestamp": 1, "class": "chat", "input_length": 100},
        ],
    )
    with pytest.raises(ValueError, match="every row needs it; missing on line\\(s\\) 2"):
        load_replay_schedule(some, names)
    for bad in [
        {"cached_length": 200, "input_length": 100},
        {"cancelled": "yes"},
        {"input_length": 10, "prompt_tokens": 10},
        {"pod": [1]},
    ]:
        with pytest.raises(ValueError):
            load_replay_schedule(
                _write_schedule(tmp_path / f"b{abs(hash(str(bad)))}.jsonl", [{"timestamp": 0, "class": "chat", **bad}]),
                names,
            )
    # Invalid optional settings drop the field for that row (counted) instead of rejecting the schedule
    lenient = _write_schedule(
        tmp_path / "lenient.jsonl",
        [
            {
                "timestamp": 0,
                "class": "chat",
                "temperature": -1,
                "top_p": 1.5,
                "reasoning_effort": "max",
                "output_length": 0,
            },
            {"timestamp": 1, "class": "chat", "temperature": 0.7},
        ],
    )
    first, second = load_replay_schedule(lenient, names)
    assert (first.temperature, first.top_p, first.reasoning_effort, first.output_length) == (None, None, None, 1)
    assert set(first.defaulted) == {"temperature", "top_p", "reasoning_effort", "output_length"}
    assert second.temperature == 0.7 and second.defaulted == ()

    rows = [
        {
            "timestamp": 0,
            "class": "chat",
            "input_length": 900,
            "cached_length": 405,
            "temperature": 0.2,
            "top_p": 0.9,
            "reasoning_effort": "low",
            "cancelled": True,
        }
    ]
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=WordTokenizer(), chat=False)
    r = ds.replay_request(entries[0])
    data, _ = build_shape_mix_request(FireworksProvider(MODEL, _options(chat=True)), r)
    assert (data["temperature"], data["top_p"], data["reasoning_effort"]) == (0.2, 0.9, "low")
    assert r.cancelled and entries[0].cancelled


@pytest.mark.parametrize("mode", ["send", "skip"])
def test_late_replay_rows_are_counted(tmp_path, mode, monkeypatch) -> None:
    import time as _time
    from types import SimpleNamespace

    from load_test import InitTracker, LLMUser, ReplayPacer, SendTimeline, load_replay_schedule, replay_summary_entries

    SendTimeline.reset()
    monkeypatch.setattr(InitTracker, "last_late_warning", 0.0)
    path = _write_schedule(
        tmp_path / "s.jsonl", [{"timestamp": t, "class": "chat", "cancelled": t == 0} for t in (0, 10, 20)]
    )
    pacer = ReplayPacer(load_replay_schedule(path, ("chat",)), path)
    sent = []
    user = SimpleNamespace(
        qps_pacer=pacer,
        environment=SimpleNamespace(parsed_options=argparse.Namespace(qps_max_lateness=0.5, replay_late_rows=mode)),
        _in_flight=SimpleNamespace(spawn=lambda fn, prepared, intended: sent.append(prepared)),
        _send_paced=None,
    )
    now = _time.monotonic()
    LLMUser._pace_and_send(user, "late", now - 2.0, 0, replay_entry=pacer.entries[0])
    LLMUser._pace_and_send(user, "on time", now, 0, replay_entry=pacer.entries[1])

    summary = replay_summary_entries(pacer, None)
    assert summary["replay_late_rows"] == 1 and summary["replay_late_share"] == pytest.approx(1 / 3)
    assert summary["replay_max_lateness_ms"] == pytest.approx(2000, abs=50)
    if mode == "send":
        assert (
            sent == ["late", "on time"]
            and summary["replay_rows_skipped"] == 0
            and summary["replay_cancelled_rows"] == 1
        )
    else:
        assert sent == ["on time"] and summary["replay_rows_skipped"] == 1 and SendTimeline.skipped == 1


def test_replay_layout_before_the_pool_build_builds_what_it_needs(tmp_path) -> None:
    from load_test import load_replay_schedule

    tokenizer = WordTokenizer()
    path = _write_schedule(
        tmp_path / "s.jsonl", [{"timestamp": 0, "class": "chat", "input_length": 900, "cached_length": 300}]
    )
    ds = _dataset(tokenizer=tokenizer)
    ds.prepare_replay(load_replay_schedule(path, ("chat", "agent", "cold")))  # no build_prefixes() first
    r = ds.replay_request(load_replay_schedule(path, ("chat", "agent", "cold"))[0])
    assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens == 900 and r.reusable_tokens == 300
    assert len(r.prompt) < 20_000


def test_replay_input_too_short_is_counted(tmp_path) -> None:
    rows = [
        {"timestamp": 0, "class": "cold", "input_length": 3, "cached_length": 0},
        {"timestamp": 1, "class": "cold", "input_length": 500, "cached_length": 0},
    ]
    ds, entries, tokenizer = _replay_dataset(tmp_path, rows, tokenizer=WordTokenizer(), chat=False)
    short, ok = (ds.replay_request(e) for e in entries)
    assert short.prompt_tokens > 3 and ok.prompt_tokens == 500
    assert ds.replay_shortfall["input_rows"] == 1 and ds.replay_shortfall["input_tokens"] == short.prompt_tokens - 3


def test_truncated_session_tails_fit_with_the_real_nonce(tmp_path) -> None:
    from load_test import DEFAULT_TOKENIZER, chat_template_token_ids

    rows, t = [], 0
    for s in range(6):
        for k, (inp, cached) in enumerate([(1500, 400), (2200, 1400), (1300, 1150), (2600, 1250)]):
            rows.append(
                {
                    "timestamp": t,
                    "class": "chat",
                    "session_id": f"s{s}",
                    "input_length": inp + 7 * s,
                    "cached_length": cached + (13 * s if k else 0),
                    "output_length": 20,
                }
            )
            t += 10
    ds, entries, tokenizer = _replay_dataset(tmp_path, rows)
    missing = 0
    for entry in entries:
        r = ds.replay_request(entry)
        assert (
            len(chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER, r.prompt))
            == r.prompt_tokens
            == entry.input_length
        )
        # a multi-token word straddling the cut can leave it a token or two short; that is counted
        assert entry.cached_length - 2 <= r.reusable_tokens <= entry.cached_length
        missing += entry.cached_length - r.reusable_tokens
    assert ds.replay_shortfall["tokens"] == missing


def _heavy_user_rows(n: int = 2400, seed: int = 3) -> list:
    """A bursty recorded stream where one session key holds ~40% of the rows (an API caller, not a chat), the
    rest being short conversations and session-less rows."""
    rng = __import__("random").Random(seed)
    rows, t, turns = [], 0.0, {}
    for i in range(n):
        t += rng.choice((20_000, 5, 5, 5, 300, 900))  # clumped
        r = rng.random()
        if r < 0.4:
            sid = "heavy"
        elif r < 0.8:
            sid = f"c{rng.randrange(n // 6)}"
        else:
            sid = None
        k = turns[sid] = turns.get(sid, -1) + 1
        row = {"timestamp": t, "class": "chat" if sid else "cold", "input_length": 600 + k, "cached_length": 0}
        if sid is not None:
            row["session_id"] = sid
        if i % 7 == 0:
            row["temperature"] = 0.5
        rows.append(row)
    return rows


def test_replay_retime_controls_every_row_with_a_heavy_user(tmp_path) -> None:
    import statistics as st

    from load_test import load_replay_schedule, parse_replay_retime, retime_summary_entries, ReplayPacer

    names = ("chat", "agent", "cold")
    rows = _heavy_user_rows()
    assert sum(r.get("session_id") == "heavy" for r in rows) / len(rows) == pytest.approx(0.4, abs=0.03)
    path = _write_schedule(tmp_path / "s.jsonl", rows)
    original = load_replay_schedule(path, names)
    by_index = {e.index: e for e in original}

    def gap_cv2(entries):
        gaps = [b.offset - a.offset for a, b in zip(entries, entries[1:])]
        return st.pvariance(gaps) / st.mean(gaps) ** 2

    assert gap_cv2(original) > 4
    for spec, target in (("poisson", 1.0), ("gamma:2", 2.0), ("gamma:4", 4.0), ("gamma:8", 8.0)):
        retimed = load_replay_schedule(path, names, retime=spec, seed=5)
        # every row of the stream is on the requested process, at the stream's own rate
        assert gap_cv2(retimed) == pytest.approx(target, rel=0.03)
        assert len(retimed) == len(original)
        assert retimed[0].offset == original[0].offset
        assert retimed[-1].offset == pytest.approx(original[-1].offset)
        # recorded order kept: no row overtakes an earlier one, so no follow-up precedes its parent
        assert [e.index for e in retimed] == [e.index for e in original]
        assert all(a.offset <= b.offset for a, b in zip(retimed, retimed[1:]))
        strip = lambda e: (e.class_index, e.input_length, e.cached_length, e.session, e.temperature)
        assert all(strip(e) == strip(by_index[e.index]) for e in retimed)
        assert all(e.recorded_offset == by_index[e.index].offset for e in retimed)
        assert load_replay_schedule(path, names, retime=spec, seed=5) == retimed  # seeded
        assert load_replay_schedule(path, names, retime=spec, seed=6) != retimed
        # schedules without a pod field: each generator's share is retimed on its own
        pods = [load_replay_schedule(path, names, 3, k, retime=spec, seed=5) for k in range(3)]
        assert sorted(e.index for p in pods for e in p) == sorted(by_index)
        for p in pods:
            assert gap_cv2(p) == pytest.approx(target, rel=0.03)
        pacer = ReplayPacer(retimed, path)
        pacer.retime = spec
        summary = retime_summary_entries(pacer)
        assert summary["replay_retime_requested_gap_cv2"] == target
        assert summary["replay_retime_slot_gap_cv2"] == pytest.approx(target, rel=0.03)
        span = original[-1].offset - original[0].offset
        assert summary["replay_retime_requested_qps"] == pytest.approx((len(original) - 1) / span)
        assert 0 < summary["replay_retime_followups_within_0_5s"] <= summary["replay_retime_followups_within_5s"]
        assert summary["replay_retime_followups_within_5s"] <= summary["replay_retime_followups"]
        last, recorded_close = {}, 0
        for e in original:
            if e.session is None:
                continue
            if e.session in last:
                recorded_close += e.offset - last[e.session] < 0.5
            last[e.session] = e.offset
        assert summary["replay_retime_followups_within_0_5s_recorded"] == recorded_close
    # a replay that isn't retimed still reports its follow-up gaps (C-180), as sent equal to as recorded
    plain = retime_summary_entries(ReplayPacer(original, path))
    assert set(plain) == {"replay_retime_followups"} | {
        f"replay_retime_followups_within_{t}{suffix}" for t in ("0_5s", "5s") for suffix in ("", "_recorded")
    }
    assert plain["replay_retime_followups_within_5s"] == plain["replay_retime_followups_within_5s_recorded"]
    assert plain["replay_retime_followups_within_0_5s"] <= plain["replay_retime_followups_within_5s"]
    for bad in ("gamma", "gamma:0", "gamma:x", "uniform"):
        with pytest.raises(ValueError):
            parse_replay_retime(bad)


def test_remainder_lead_ins_and_cross_session_cover(tmp_path) -> None:
    """C-164 / D-50: shared prefixes are cross-session. A shared-prefix warm-up lead-in (cached_length ==
    input_length) is skipped when any session's earlier row sent that pool cut; a remainder lead-in (the shared
    part plus the session's own context) is always sent, and its session's first scored row then reuses all of
    its recorded cached length: the shared part from the pool prefix and the remainder from the lead-in."""
    rows = [
        # another session's lead-in sends chat prefix 0 (a remainder lead-in: 400 shared + private context)
        {
            "time_s": -9,
            "class": "chat",
            "session_id": "b",
            "input_length": 900,
            "cached_length": 400,
            "output_length": 5,
            "prefix_index": 0,
            "lead_in": True,
        },
        # a pure shared-prefix warm-up of the same prefix: covered by that earlier send
        {
            "time_s": -8,
            "class": "chat",
            "session_id": "c",
            "input_length": 400,
            "cached_length": 400,
            "output_length": 5,
            "prefix_index": 0,
            "lead_in": True,
        },
        # session a: remainder lead-in, 400 shared + 600 own context
        {
            "time_s": -5,
            "class": "chat",
            "session_id": "a",
            "input_length": 1000,
            "cached_length": 400,
            "output_length": 5,
            "prefix_index": 0,
            "lead_in": True,
        },
        # a's first scored row: 1000 cached (400 shared + 600 own), 300 fresh
        {
            "time_s": 0,
            "class": "chat",
            "session_id": "a",
            "input_length": 1300,
            "cached_length": 1000,
            "output_length": 5,
            "prefix_index": 0,
        },
        # a session with no lead-in: its first row reuses the shared pool prefix only
        {
            "time_s": 1,
            "class": "chat",
            "session_id": "d",
            "input_length": 700,
            "cached_length": 400,
            "output_length": 5,
            "prefix_index": 0,
        },
    ]
    tokenizer = WordTokenizer()
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=tokenizer, chat=False)
    sent = ds.lead_in_requests(entries)
    assert [s for s, _ in sent] == ["b", "a"] and ds.lead_in_skipped == 1
    ids = {s: tokenizer.encode(r.prompt) for s, r in sent}
    first_a = ds.replay_request(entries[3])
    first_d = ds.replay_request(entries[4])
    a = tokenizer.encode(first_a.prompt)
    assert first_a.reusable_tokens == 1000 and _common_prefix(a, ids["a"]) >= 1000
    # the shared 400 tokens are the same text in every session; the remainder is not shared across sessions
    assert _common_prefix(ids["a"], ids["b"]) >= 400
    assert _common_prefix(ids["a"], ids["b"]) < 450
    d = tokenizer.encode(first_d.prompt)
    assert first_d.reusable_tokens == 400 and _common_prefix(d, ids["b"]) >= 400
    assert ds.replay_shortfall["rows"] == 0
    assert sum(1 for e in entries if e.lead_in and e.cached_length < e.input_length) == 2


def _pool_meta(tmp_path, per_slot) -> str:
    """A schedule meta file with pool_prefix[slot][class]["*"] = {shared_max, shared_p99}."""
    pool = {
        slot: {name: {"*": {"shared_max": mx, "shared_p99": p99}} for name, (mx, p99) in classes.items()}
        for slot, classes in per_slot.items()
    }
    path = tmp_path / "s.jsonl.meta.json"
    path.write_text(json.dumps({"version": 1, "pool_prefix": pool}))
    return str(path)


def test_replay_pool_prefix_sizes_come_from_the_schedule_meta(tmp_path, caplog) -> None:
    """C-178 / D-61: the pool prefix is the longest recorded shared length per class on the replayed replica."""
    from load_test import load_replay_schedule, replay_pool_prefix_tokens, replay_sized_mix

    rows = [
        {"time_s": 0, "pod": "3", "class": "chat", "input_length": 1500, "cached_length": 900},
        {"time_s": 1, "pod": "3", "class": "agent", "input_length": 2000, "cached_length": 0},
        {"time_s": 2, "pod": "4", "class": "chat", "input_length": 1500, "cached_length": 100},
    ]
    schedule = _write_schedule(tmp_path / "s.jsonl", rows)
    names = ("chat", "agent", "cold")
    _pool_meta(tmp_path, {"3": {"chat": (950, 700), "agent": (1800, 1500)}, "4": {"chat": (200, 150)}})
    options = argparse.Namespace(replay_schedule=schedule, replay_schedule_meta=None, replay_pool_prefix="max")

    entries = load_replay_schedule(schedule, names, pod_select="3")
    sizes = replay_pool_prefix_tokens(options, entries, names)
    assert sizes == {"chat": 950, "agent": 1800}  # the replayed slot only; cold has no rows, so no warning
    options.replay_pool_prefix = "p99"
    assert replay_pool_prefix_tokens(options, entries, names) == {"chat": 700, "agent": 1500}
    merged = load_replay_schedule(schedule, names, pod_select="all")
    options.replay_pool_prefix = "max"
    assert replay_pool_prefix_tokens(options, merged, names)["chat"] == 950  # merged pods: the longest of them

    mix = replay_sized_mix(parse_shape_mix(json.dumps(MIX)), {"chat": 950})
    chat, agent, cold = mix.classes
    assert chat.cached_prompt_tokens == 950 and chat.prompt_tokens >= 950 + 32  # tail kept
    assert (agent.cached_prompt_tokens, agent.prompt_tokens) == (1000, 1200) and cold.cached_prompt_tokens == 0

    # a class the meta doesn't list (no session-first row with a shared part) keeps its mean quietly; a meta with
    # nothing for the replica, and no meta at all, warn
    _pool_meta(tmp_path, {"3": {"chat": (950, 700)}})
    caplog.clear()
    with caplog.at_level("INFO", logger="load_test"):
        assert replay_pool_prefix_tokens(options, entries, names) == {"chat": 950}
    assert "WARNING" not in caplog.text and "no shared length for agent" in caplog.text
    _pool_meta(tmp_path, {"9": {"chat": (950, 700)}})
    caplog.clear()
    with caplog.at_level("WARNING", logger="load_test"):
        assert replay_pool_prefix_tokens(options, entries, names) == {}
    assert "has no usable 'pool_prefix' for the replayed replica" in caplog.text
    os.remove(tmp_path / "s.jsonl.meta.json")
    caplog.clear()
    with caplog.at_level("WARNING", logger="load_test"):
        assert replay_pool_prefix_tokens(options, entries, names) == {}
    assert "no schedule meta found" in caplog.text
    (tmp_path / "bad.json").write_text("{")
    options.replay_schedule_meta = str(tmp_path / "bad.json")
    with pytest.raises(ValueError):
        replay_pool_prefix_tokens(options, entries, names)


def test_sized_pool_prefix_closes_the_first_row_shortfall(tmp_path) -> None:
    from load_test import first_row_shortfall_entries, load_replay_schedule, replay_sized_mix

    rows = [
        # two sessions whose first rows share more than the class mean (400): 900 and 700 recorded, cross-session
        {
            "time_s": 0,
            "class": "chat",
            "session_id": "a",
            "input_length": 1500,
            "cached_length": 900,
            "prefix_index": 0,
        },
        {
            "time_s": 1,
            "class": "chat",
            "session_id": "b",
            "input_length": 1300,
            "cached_length": 700,
            "prefix_index": 0,
        },
        {
            "time_s": 2,
            "class": "chat",
            "session_id": "a",
            "input_length": 1700,
            "cached_length": 1400,
            "prefix_index": 0,
        },
        {"time_s": 3, "class": "cold", "input_length": 500, "cached_length": 0},
    ]
    schedule = _write_schedule(tmp_path / "s.jsonl", rows)
    names = ("chat", "agent", "cold")
    entries = load_replay_schedule(schedule, names)
    tokenizer = WordTokenizer()

    def layout(sizes):
        mix = replay_sized_mix(parse_shape_mix(json.dumps(MIX)), sizes)
        ds = ShapeMixDataset(mix, tokenizer, LIMERICKS, request_seed=3, tail_variants=0)
        ds.build_prefixes()
        ds.prepare_replay(entries)
        return ds

    mean = layout({})  # today: the class mean (400) is the pool prefix
    summary = first_row_shortfall_entries(mean)
    assert summary["class_chat_first_rows"] == 2 and summary["class_chat_first_row_shortfall_rows"] == 2
    assert summary["class_chat_first_row_shortfall_tokens"] == (900 - 400) + (700 - 400)
    assert "class_chat_pool_prefix_tokens" not in summary  # nothing sized it

    sized = layout({"chat": 900})
    summary = first_row_shortfall_entries(sized)
    assert summary["replay_first_rows"] == 2  # cold's row records no cached tokens, so it isn't a first row to check
    assert summary["replay_first_row_shortfall_rows"] == 0 and summary["replay_first_row_shortfall_tokens"] == 0
    sized.replay_pool_sizes = {"chat": 900}
    assert first_row_shortfall_entries(sized)["class_chat_pool_prefix_tokens"] == 900
    reqs = {e.index: sized.replay_request(e) for e in entries}
    assert [reqs[i].reusable_tokens for i in (0, 1)] == [900, 700]  # each row cuts the shared prefix at its own length
    a, b = (tokenizer.encode(reqs[i].prompt) for i in (0, 1))
    assert _common_prefix(a, b) >= 700  # one shared copy of the prefix text across sessions
    # the second turn of session a reuses its own first prompt (1400 cached) even though the pool stops at 900
    assert reqs[2].reusable_tokens == 1400


def test_replay_class_metrics_split_first_rows_and_followups(monkeypatch) -> None:
    """C-180: per class, the server's and the recorded cached tokens over first rows and over follow-up rows (the
    same rows, those whose server cached tokens are known), with the names task 07 parses."""
    calls = []
    monkeypatch.setattr(load_test, "add_custom_metric", lambda name, value: calls.append((name, value)))
    record = load_test.record_shape_class_metrics
    record("chat", 10, 100, 900, 400, 5, recorded_cached_tokens=500, laid_out_cached_tokens=500, replay_role="first")
    record("chat", 10, 100, 900, 800, 5, recorded_cached_tokens=850, laid_out_cached_tokens=850, replay_role="followup")
    record("chat", 10, 100, 900, 600, 5, recorded_cached_tokens=700, laid_out_cached_tokens=700, replay_role="followup")
    record(
        "chat", 10, 100, 900, None, 5, recorded_cached_tokens=900, laid_out_cached_tokens=900, replay_role="followup"
    )
    record("chat", 10, 100, 900, 100, 5)  # not a replay row: no role metrics
    names = [n for n, _ in calls]
    assert names.count("class_chat_followup_cached_tokens") == 2  # the row without a server count isn't in either mean
    assert names.count("class_chat_followup_recorded_cached_tokens") == 2
    assert names.count("class_chat_first_row_cached_tokens") == 1

    stats = RequestStats()
    for name, value in calls:
        stats.log_request("METRIC", name, value, 0)
    entries = shape_mix_summary_entries(stats, ["chat", "cold"], True)
    assert entries["class_chat_followup_requests"] == 2
    assert entries["class_chat_followup_cached_tokens"] == pytest.approx(700)
    assert entries["class_chat_followup_recorded_cached_tokens"] == pytest.approx(775)
    assert entries["class_chat_first_row_cached_tokens"] == 400
    assert entries["class_chat_first_row_recorded_cached_tokens"] == 500
    # a class with no replay rows, and a run that isn't a replay, print no role columns
    assert "class_cold_followup_requests" not in entries
    plain = shape_mix_summary_entries(RequestStats(), ["chat"], True)
    assert not [k for k in plain if "followup" in k or "first_row" in k]


def test_replay_schedule_warns_on_unknown_fields(tmp_path, caplog) -> None:
    """N8-140-1: fields llm_bench doesn't read are named in a warning, not dropped silently."""
    from load_test import load_replay_schedule

    rows = [
        {"time_s": 0, "class": "chat", "input_length": 900, "cached_length": 0, "lead_in_length": 300},
        {"time_s": 1, "class": "chat", "input_length": 900, "cached_length": 0, "lead_in_length": 0, "extra": 1},
        {"time_s": 2, "class": "cold", "prompt_tokens": 400, "cached_tokens": 0, "temperature": 0.3},
    ]
    path = _write_schedule(tmp_path / "s.jsonl", rows)
    with caplog.at_level("WARNING", logger="load_test"):
        assert len(load_replay_schedule(path, ("chat", "agent", "cold"))) == 3
    assert "unknown fields extra (1 rows), lead_in_length (2 rows)" in caplog.text
    assert '"lead_in": true' in caplog.text
    caplog.clear()
    path = _write_schedule(tmp_path / "ok.jsonl", rows[2:])
    with caplog.at_level("WARNING", logger="load_test"):
        load_replay_schedule(path, ("chat", "agent", "cold"))
    assert "unknown" not in caplog.text


def test_replay_start_overrun_warning_counts_the_rows_due(tmp_path, caplog, monkeypatch) -> None:
    """N8-140-6: the warning loads the rows itself when no user has built the pacer yet (it said 0 due)."""
    import argparse
    import time as _time

    import load_test
    from load_test import InitTracker, ReplayPacer

    rows = [{"time_s": float(i), "class": "cold", "input_length": 400, "cached_length": 0} for i in range(20)]
    path = _write_schedule(tmp_path / "s.jsonl", rows)
    options = argparse.Namespace(
        replay_schedule=path,
        replay_start_at=_time.time() - 10.5,
        replay_late_rows="send",
        qps_max_lateness=1.0,
        shape_mix=json.dumps(MIX),
        replay_pods=1,
        replay_pod_index=0,
        replay_retime=None,
        replay_pod=None,
        request_seed=None,
    )
    monkeypatch.setattr(ReplayPacer, "_instance", None)
    monkeypatch.setattr(InitTracker, "environment", argparse.Namespace(parsed_options=options))
    monkeypatch.setattr(load_test, "resolve_request_seed", lambda o: 1)
    with caplog.at_level("WARNING", logger="load_test"):
        assert InitTracker._replay_start_wait() == 0.0
    assert "11 rows already due" in caplog.text  # rows at 0..10 s are due 10.5 s after the start
    assert ReplayPacer._instance is not None


def _pod_rows():
    rows = []
    for i in range(60):
        pod = ["p1", "p2", "p10"][i % 3]
        rows.append(
            {
                "timestamp": 1000 + i * 100,
                "class": "chat",
                "pod": pod,
                "session_id": f"s{i % 2}",
                "input_length": 900,
                "cached_length": 0,
                "output_length": 5,
            }
        )
    return rows


def test_replay_pod_selects_one_recorded_pod_at_its_exact_times(tmp_path) -> None:
    from argparse import Namespace

    from load_test import ReplayPacer, load_replay_schedule, replay_summary_entries

    names = ("chat", "agent", "cold")
    path = _write_schedule(tmp_path / "pods.jsonl", _pod_rows())
    everything = [r for r in _pod_rows()]
    with pytest.raises(ValueError, match="records 3 pods; one test replica replays one"):
        load_replay_schedule(path, names)
    p2 = load_replay_schedule(path, names, pod_select="p2")
    assert {e.pod for e in p2} == {"p2"} and len(p2) == 20
    # exact recorded times, relative to the start of the whole schedule (so replicas line up)
    assert [round(e.offset, 3) for e in p2] == [
        round((r["timestamp"] - 1000) / 1000, 3) for r in everything if r["pod"] == "p2"
    ]
    # sessions are scoped to their pod
    assert {e.session for e in p2} == {"p2/s0", "p2/s1"}
    with pytest.raises(ValueError, match="not in the schedule's pods"):
        load_replay_schedule(path, names, pod_select="p9")
    # --replay-pods N: replica I gets the I-th recorded pod (numeric order: p1, p2, p10)
    replicas = [load_replay_schedule(path, names, 3, i) for i in range(3)]
    assert [{e.pod for e in r} for r in replicas] == [{"p1"}, {"p2"}, {"p10"}]
    with pytest.raises(ValueError):
        load_replay_schedule(path, names, 4, 0)
    # merging is explicit and labelled
    merged = load_replay_schedule(path, names, pod_select="all")
    pacer = ReplayPacer(merged, path)
    assert "MERGED 3 recorded pods" in pacer.describe()
    assert replay_summary_entries(pacer, None)["replay_pods_merged"] == 3
    single = ReplayPacer(p2, path)
    assert "pod p2" in single.describe() and replay_summary_entries(single, None)["replay_pods_merged"] == 0
    # retiming a pod keeps its own rate and span
    retimed = load_replay_schedule(path, names, pod_select="p2", retime="poisson", seed=1)
    assert retimed[0].offset == p2[0].offset and retimed[-1].offset == pytest.approx(p2[-1].offset)
    assert [e.index for e in retimed] == [e.index for e in p2]
    # the pacer starts at a shared wall-clock time
    import time as _time

    pacer = ReplayPacer.instance.__func__  # noqa: F841  (instance() reads options; exercise start_at directly)
    shared = ReplayPacer(p2, path)
    shared.start_at = _time.time() + 5.0
    intended, _, entry = shared.claim_entry()
    assert intended - _time.monotonic() == pytest.approx(5.0 + entry.offset, abs=0.05)


def test_session_salt_is_new_per_run(tmp_path) -> None:
    rows = [{"timestamp": 0, "class": "chat", "session_id": "a", "input_length": 900, "cached_length": 400}]
    ds1, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=WordTokenizer(), chat=False)
    ds2, _, _ = _replay_dataset(tmp_path, rows, tokenizer=WordTokenizer(), chat=False)
    assert ds1.replay_salt != ds2.replay_salt
    assert ds1.replay_request(entries[0]).affinity_key == f"shape-{ds1.replay_salt}-session-a"
    ds2.prepare_replay(entries, session_salt="fixed")
    assert ds2.replay_request(entries[0]).affinity_key == "shape-fixed-session-a"


def test_session_rows_reuse_the_prompt_that_covers_their_cached_length(tmp_path) -> None:
    rows = [
        {
            "timestamp": 0,
            "class": "chat",
            "session_id": "a",
            "input_length": 2000,
            "cached_length": 400,
            "output_length": 30,
        },
        {
            "timestamp": 10,
            "class": "chat",
            "session_id": "a",
            "input_length": 700,
            "cached_length": 400,
            "output_length": 30,
        },
        # 1500 is more than row 1's prompt (+ reply) holds, but row 0's prompt covers it
        {
            "timestamp": 20,
            "class": "chat",
            "session_id": "a",
            "input_length": 2500,
            "cached_length": 1500,
            "output_length": 30,
        },
    ]
    tokenizer = WordTokenizer()
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=tokenizer, chat=False)
    r0, r1, r2 = (ds.replay_request(e) for e in entries)
    assert r2.reusable_tokens == 1500
    assert r2.prompt[:2000] == r0.prompt[:2000] and r2.prompt[:2000] != r1.prompt[:2000]
    assert ds.replay_shortfall["rows"] == 0
    assert all(len(tokenizer.encode(r.prompt)) == r.prompt_tokens for r in (r0, r1, r2))


def test_rows_with_little_uncached_room_get_a_bare_tail(tmp_path) -> None:
    from load_test import DEFAULT_TOKENIZER, chat_template_token_ids

    from load_test import chat_template_head_tokens, empty_chat_template_token_ids

    tokenizer, _, _ = _llama()
    # the template's own tokens after the content count as uncached too
    trailing = len(empty_chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER)) - chat_template_head_tokens(
        tokenizer, DEFAULT_TOKENIZER
    )
    rows = [
        {
            "timestamp": 0,
            "class": "chat",
            "session_id": "a",
            "input_length": 1500,
            "cached_length": 400,
            "output_length": 5,
        },
        # only 4 uncached content tokens: no room for the nonce, so the tail is bare and cached_length is kept
        {
            "timestamp": 10,
            "class": "chat",
            "session_id": "a",
            "input_length": 1400,
            "cached_length": 1396 - trailing,
            "output_length": 5,
        },
        {"timestamp": 20, "class": "chat", "input_length": 900, "cached_length": 400, "output_length": 5},
        # outside a session: the whole 400-token pool prefix and 4 uncached content tokens
        {"timestamp": 30, "class": "chat", "input_length": 404 + trailing, "cached_length": 400, "output_length": 5},
    ]
    ds, entries, tokenizer = _replay_dataset(tmp_path, rows)
    for entry in entries:
        r = ds.replay_request(entry)
        assert (
            len(chat_template_token_ids(tokenizer, DEFAULT_TOKENIZER, r.prompt))
            == r.prompt_tokens
            == entry.input_length
        )
        assert r.reusable_tokens == entry.cached_length
    assert ds._replay[entries[1].index]["segments"][-1][0] == "bare"
    assert ds.replay_shortfall["rows"] == 0


def test_replay_fillers_vary_by_row(tmp_path) -> None:
    rows = [
        {"timestamp": i, "class": "cold", "input_length": 600, "cached_length": 0, "output_length": 5} for i in range(8)
    ]
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=WordTokenizer(), chat=False)
    fillers = {" ".join(ds.replay_request(e).prompt.split()[6:40]) for e in entries}
    assert len(fillers) == 8
    assert ds.replay_layout_seconds > 0


def test_replay_rows_asleep_during_a_stall_count_as_late(tmp_path, monkeypatch) -> None:
    import time as _time
    from types import SimpleNamespace

    import gevent

    import load_test
    from load_test import InitTracker, LLMUser, ReplayPacer, SendTimeline, load_replay_schedule, replay_summary_entries

    InitTracker.spawned.set()
    SendTimeline.reset()
    monkeypatch.setattr(InitTracker, "last_late_warning", 0.0)
    path = _write_schedule(tmp_path / "s.jsonl", [{"timestamp": 100 * i, "class": "chat"} for i in range(20)])
    pacer = ReplayPacer(load_replay_schedule(path, ("chat",)), path)
    real_sleep = gevent.sleep
    stall_end = _time.monotonic() + 1.5
    monkeypatch.setattr(
        load_test.gevent, "sleep", lambda seconds=0: real_sleep(max(seconds, stall_end - _time.monotonic()))
    )
    for mode in ("send", "skip"):
        sent = []
        user = SimpleNamespace(
            qps_pacer=pacer,
            environment=SimpleNamespace(parsed_options=argparse.Namespace(qps_max_lateness=0.5, replay_late_rows=mode)),
            _in_flight=SimpleNamespace(spawn=lambda fn, prepared, intended: sent.append(intended)),
            _send_paced=None,
        )
        if mode == "send":
            now = _time.monotonic()
            gevent.joinall(
                [gevent.spawn(LLMUser._pace_and_send, user, "x", now + 0.1 * i, 0, pacer.entries[i]) for i in range(10)]
            )
            summary = replay_summary_entries(pacer, None)
            # rows due before ~1.0 s woke more than 0.5 s late: all counted, all still sent
            assert summary["replay_late_rows"] >= 9 and len(sent) == 10
        else:
            stall_end = _time.monotonic() + 1.5
            before = pacer.late_rows
            now = _time.monotonic()
            gevent.joinall(
                [
                    gevent.spawn(LLMUser._pace_and_send, user, "x", now + 0.1 * i, 0, pacer.entries[10 + i])
                    for i in range(10)
                ]
            )
            assert pacer.late_rows - before >= 9 and pacer.skipped_rows >= 9 and len(sent) <= 1


def test_mixed_pod_fields_are_rejected(tmp_path) -> None:
    from load_test import load_replay_schedule

    path = _write_schedule(
        tmp_path / "mixed.jsonl",
        [
            {"timestamp": 0, "class": "chat", "pod": "p1"},
            {"timestamp": 1, "class": "chat"},
            {"timestamp": 2, "class": "chat"},
        ],
    )
    with pytest.raises(ValueError, match="'pod' is set on some rows.*line\\(s\\) 2, 3"):
        load_replay_schedule(path, ("chat",), pod_select="p1")


def _common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_restore_boundary_comes_from_the_glm_config(monkeypatch) -> None:
    import load_test
    from load_test import resolve_prefix_restore_boundary

    monkeypatch.setattr(load_test, "resolve_model_type", lambda path: "glm_moe_dsa" if "GLM" in path else "llama")
    monkeypatch.setattr(load_test, "_read_config_json", lambda path: {"index_topk": 2048})
    assert (
        resolve_prefix_restore_boundary(argparse.Namespace(tokenizer="zai-org/GLM-5.2", prefix_restore_boundary=None))
        == 2048
    )
    assert (
        resolve_prefix_restore_boundary(argparse.Namespace(tokenizer="meta-llama/x", prefix_restore_boundary=None)) == 0
    )
    assert (
        resolve_prefix_restore_boundary(argparse.Namespace(tokenizer="zai-org/GLM-5.2", prefix_restore_boundary=0)) == 0
    )


def test_shape_mix_prefixes_align_to_the_restore_boundary() -> None:
    import statistics as st

    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer, restore_boundary=256)  # chat 400 cached, agent 1000 cached
    # N8-140-4: each class lays out floor(cached / 256) * 256 or one block more, in the share that keeps its mean
    assert ds.reusable_tokens == [256, 768, 0, 512, 1024]  # chat, agent, cold, then chat's and agent's upper blocks
    assert ds.prefix_tokens[:2] == [256 + 16, 768 + 16]  # laid out a margin past the boundary
    assert ds.class_reusable_tokens() == pytest.approx([400, 1000, 0])
    # the pools split in the same ratio: chat 3 -> 1 + 2 (share 144/256), agent 2 -> 1 + 1 (share 232/256)
    assert ds.pool_sizes == [1, 1, 0, 2, 1]
    seen = {}
    requests = _take(ds, 4000)
    for r in requests[:300]:
        ids = tokenizer.encode(r.prompt)
        assert len(ids) == r.prompt_tokens == ds.mix.classes[r.class_index].prompt_tokens
        key = (r.class_index, r.reusable_tokens, r.prefix_index)
        if r.prefix_index is not None and key in seen:
            shared = _common_prefix(ids, seen[key])
            assert shared // 256 * 256 == r.reusable_tokens
        seen.setdefault(key, ids)
    assert {r.class_name for r in requests} == {"chat", "agent", "cold"}  # companions report as their class
    for ci, cached in ((0, 400), (1, 1000)):
        assert st.mean(r.reusable_tokens for r in requests if r.class_index == ci) == pytest.approx(cached, rel=0.04)
    # without a boundary nothing splits
    assert len(_dataset(tokenizer=tokenizer).classes) == 3


def test_replay_rows_align_to_the_restore_boundary(tmp_path) -> None:
    rows = [
        {
            "timestamp": 0,
            "class": "chat",
            "session_id": "a",
            "input_length": 1500,
            "cached_length": 0,
            "output_length": 5,
        },
        {
            "timestamp": 10,
            "class": "chat",
            "session_id": "a",
            "input_length": 2400,
            "cached_length": 1300,
            "output_length": 5,
        },
        {
            "timestamp": 20,
            "class": "chat",
            "session_id": "a",
            "input_length": 2600,
            "cached_length": 2290,
            "output_length": 5,
        },
        {
            "timestamp": 30,
            "class": "chat",
            "session_id": "a",
            "input_length": 900,
            "cached_length": 200,
            "output_length": 5,
        },
    ]
    tokenizer = WordTokenizer()
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=tokenizer, chat=False, restore_boundary=512)
    sent = []
    for entry in entries:
        r = ds.replay_request(entry)
        ids = tokenizer.encode(r.prompt)
        assert len(ids) == r.prompt_tokens == entry.input_length
        assert r.reusable_tokens == entry.cached_length // 512 * 512
        if sent and r.reusable_tokens:
            shared = max(_common_prefix(ids, earlier) for earlier in sent)
            assert shared // 512 * 512 == r.reusable_tokens
        sent.append(ids)
    assert ds.replay_shortfall["rows"] == 0


def test_glm_tokenizer_prefixes_restore_whole_blocks() -> None:
    from load_test import (
        _load_auto_tokenizer,
        chat_template_head_tokens,
        chat_template_token_ids,
        empty_chat_template_token_ids,
    )

    path = "zai-org/GLM-5.2"
    try:
        tokenizer = _load_auto_tokenizer(path)
    except Exception as e:
        pytest.skip(f"GLM tokenizer unavailable: {e}")
    overhead = len(empty_chat_template_token_ids(tokenizer, path))
    head = chat_template_head_tokens(tokenizer, path)
    spec = {
        "seed": 5,
        "classes": [
            {
                "name": "chat",
                "weight": 1,
                "prompt_tokens": 9000,
                "cached_prompt_tokens": 5000,
                "generated_tokens": 10,
                "prefix_pool_size": 2,
            },
            {"name": "cold", "weight": 0.3, "prompt_tokens": 3000, "cached_prompt_tokens": 0, "generated_tokens": 10},
        ],
    }
    ds = ShapeMixDataset(
        parse_shape_mix(json.dumps(spec), chat_overhead_tokens=overhead),
        tokenizer,
        "limericks.txt",
        overhead,
        head,
        request_seed=3,
        restore_boundary=2048,
    )
    ds.build_prefixes()
    # 5,000 cached: 4,096 on 56% of requests and 6,144 on 44%, so the class keeps its mean (N8-140-4)
    assert ds.reusable_tokens == [4096, 0, 6144]
    assert ds.class_reusable_tokens()[0] == pytest.approx(5000)
    seen = {}
    for r in _take(ds, 40):
        ids = chat_template_token_ids(tokenizer, path, r.prompt)
        assert len(ids) == r.prompt_tokens
        key = (r.class_index, r.reusable_tokens, r.prefix_index)
        if r.prefix_index is not None and key in seen:
            shared = _common_prefix(ids, seen[key])
            assert r.reusable_tokens <= shared < r.reusable_tokens + 64
            assert shared // 2048 * 2048 == r.reusable_tokens
        seen.setdefault(key, ids)


def test_a_session_prompt_ending_at_the_boundary_still_covers_the_row(tmp_path) -> None:
    # turn 1's prompt is exactly one 512-token block; turn 2 records it all as cached, with a short reply
    rows = [
        {
            "timestamp": 0,
            "class": "agent",
            "session_id": "a",
            "input_length": 512,
            "cached_length": 0,
            "output_length": 2,
        },
        {
            "timestamp": 10,
            "class": "agent",
            "session_id": "a",
            "input_length": 1200,
            "cached_length": 512,
            "output_length": 2,
        },
    ]
    tokenizer = WordTokenizer()
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=tokenizer, chat=False, restore_boundary=512)
    first, second = (ds.replay_request(e) for e in entries)
    # reuses the session's own prompt, not the (longer) class pool prefix
    assert _common_prefix(tokenizer.encode(second.prompt), tokenizer.encode(first.prompt)) >= 512
    assert second.reusable_tokens == 512 and ds.replay_shortfall["rows"] == 0


def test_lead_in_rows_warm_sessions_before_time_zero(tmp_path) -> None:
    from load_test import ReplayPacer, load_replay_schedule

    rows = [
        # lead-ins recorded before the window: the session's earlier context, and a shorter one it already covers
        {
            "timestamp": 1000,
            "class": "chat",
            "session_id": "a",
            "input_length": 1500,
            "cached_length": 0,
            "output_length": 5,
            "lead_in": True,
        },
        {
            "timestamp": 1500,
            "class": "chat",
            "session_id": "a",
            "input_length": 900,
            "cached_length": 0,
            "output_length": 5,
            "lead_in": True,
        },
        # the first scored row already records cached tokens: they come from the lead-in
        {
            "timestamp": 5000,
            "class": "chat",
            "session_id": "a",
            "input_length": 2000,
            "cached_length": 1400,
            "output_length": 5,
        },
        {"timestamp": 6000, "class": "cold", "input_length": 700, "cached_length": 0, "output_length": 5},
    ]
    tokenizer = WordTokenizer()
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=tokenizer, chat=False)
    assert [e.lead_in for e in entries] == [True, True, False, False]
    assert [e.offset for e in entries] == [-4.0, -3.5, 0.0, 1.0]  # time 0 is the first scored row
    pacer = ReplayPacer(entries, "s.jsonl")
    assert [e.index for e in pacer.entries] == [2, 3] and len(pacer.lead_ins) == 2
    sends = ds.lead_in_requests(entries)
    assert len(sends) == 1 and ds.lead_in_skipped == 1  # the shorter lead-in is covered by the first
    lead = sends[0][1]
    scored = ds.replay_request(entries[2])
    assert scored.reusable_tokens == 1400
    assert _common_prefix(tokenizer.encode(scored.prompt), tokenizer.encode(lead.prompt)) >= 1400
    # retiming leaves lead-ins where they are
    path = _write_schedule(tmp_path / "r.jsonl", rows)
    retimed = load_replay_schedule(path, ("chat", "agent", "cold"), retime="poisson", seed=1)
    assert [e.offset for e in retimed if e.lead_in] == [-4.0, -3.5]


def test_default_prose_corpus_doesnt_repeat_in_long_prompts() -> None:
    from load_test import resolve_shape_mix_corpus

    tokenizer = WordTokenizer()
    spec = {
        "seed": 2,
        "classes": [
            {
                "name": "long",
                "weight": 1,
                "prompt_tokens": 60000,
                "cached_prompt_tokens": 50000,
                "generated_tokens": 5,
                "prefix_pool_size": 1,
            }
        ],
    }
    ds = _dataset(spec, tokenizer=tokenizer, corpus_path=resolve_shape_mix_corpus(None))
    ds.build_prefixes()
    words = next(ds).prompt.split()
    # no 20-word run appears twice in a 60,000-word prompt (the limericks repeat every ~4,000 words)
    runs = [" ".join(words[i : i + 20]) for i in range(0, len(words) - 20, 20)]
    assert len(set(runs)) == len(runs)


def test_synthetic_replies_vary_by_row(tmp_path) -> None:
    rows = [
        {
            "timestamp": i,
            "class": "chat",
            "session_id": f"s{i}",
            "input_length": 900,
            "cached_length": 0,
            "output_length": 40,
        }
        for i in range(6)
    ]
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=WordTokenizer(), chat=False)
    replies = {ds._response_segment(40, e.index)[1:3] for e in entries}
    assert len(replies) == 6


def test_prompt_token_estimate_is_once_per_process_without_tokenizing(monkeypatch) -> None:
    import load_test
    from load_test import DatasetHolder

    monkeypatch.setattr(DatasetHolder, "_prompt_tokens_estimate", None)
    calls = []
    monkeypatch.setattr(load_test.InitTracker, "load_tokenizer", lambda path: calls.append(path) or WordTokenizer())
    get_input = lambda: ("a long prompt " * 1000, 16000, None)
    opts = argparse.Namespace(tokenizer=None)
    # C-151: every user used to tokenize a whole prompt in on_start, inside the run window
    assert [DatasetHolder.prompt_token_estimate(opts, get_input) for _ in range(100)] == [16000] * 100
    assert calls == []
    monkeypatch.setattr(DatasetHolder, "_prompt_tokens_estimate", None)
    assert DatasetHolder.prompt_token_estimate(opts, lambda: ("three short words", 0, None)) == 3  # no count: once
    assert len(calls) == 1


def test_client_cpu_summary_reports_mean_and_peak() -> None:
    import time as _time

    from load_test import ClientCpu

    ClientCpu._samples = []
    ClientCpu.reset()
    end = _time.monotonic() + 0.3
    while _time.monotonic() < end:
        sum(range(1000))
    entries = ClientCpu.summary_entries()
    assert 50 <= entries["client_cpu_percent"] <= 105
    assert entries["client_cpu_peak_5s_percent"] >= entries["client_cpu_percent"] - 1e-9


def test_a_session_lead_in_isnt_covered_by_another_keys_pool_send(tmp_path) -> None:
    rows = [
        # another key sends the chat pool prefix first
        {
            "timestamp": 500,
            "class": "chat",
            "input_length": 1500,
            "cached_length": 400,
            "output_length": 5,
            "prefix_index": 0,
            "lead_in": True,
        },
        # this session's lead-in starts from the same pool prefix: still needed, its replica is different
        {
            "timestamp": 1000,
            "class": "chat",
            "session_id": "a",
            "input_length": 1200,
            "cached_length": 400,
            "output_length": 5,
            "prefix_index": 0,
            "lead_in": True,
        },
        {
            "timestamp": 5000,
            "class": "chat",
            "session_id": "a",
            "input_length": 1600,
            "cached_length": 1100,
            "output_length": 5,
        },
    ]
    ds, entries, _ = _replay_dataset(tmp_path, rows, tokenizer=WordTokenizer(), chat=False)
    assert len(ds.lead_in_requests(entries)) == 2 and ds.lead_in_skipped == 0
    assert ds.replay_request(entries[2]).reusable_tokens == 1100
