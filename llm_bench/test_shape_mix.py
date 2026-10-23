from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import pytest
from locust.stats import RequestStats

from load_test import (
    FireworksProvider,
    FixedQPSPacer,
    OpenAIProvider,
    ShapeMixDataset,
    VllmProvider,
    build_shape_mix_request,
    parse_shape_mix,
    sample_interarrival,
    shape_mix_summary_entries,
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
    return ShapeMixDataset(
        mix=parse_shape_mix(json.dumps(mix or MIX)),
        tokenizer=tokenizer or WordTokenizer(),
        corpus_path=LIMERICKS,
        **kwargs,
    )


def _take(ds: ShapeMixDataset, n: int):
    return [next(ds) for _ in range(n)]


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
        {"classes": [{"weight": 1, "prompt_tokens": 10.5, "generated_tokens": 1}]},
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
            assert ds.prefixes[ci] == []
            assert all(r.prefix_index is None and r.affinity_key is None for r in reqs)
            continue
        pool = ds.prefixes[ci]
        assert len(pool) == cls.prefix_pool_size == len(set(pool))
        assert {r.prefix_index for r in reqs} == set(range(cls.prefix_pool_size))
        for r in reqs:
            assert r.prompt.startswith(pool[r.prefix_index])
            assert r.affinity_key == f"shape-7-{cls.name}-{r.prefix_index}"

    all_prefixes = [p for pool in ds.prefixes for p in pool]
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
        for a, b in zip(prompts, prompts[1:]):
            assert common(a, b) == expected


def test_exact_prompt_token_lengths() -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer)

    for r in _take(ds, 300):
        cls = ds.mix.classes[r.class_index]
        ids = tokenizer.encode(r.prompt)
        assert len(ids) == r.prompt_tokens == cls.prompt_tokens
        if r.prefix_index is not None:
            assert ds.prefix_tokens[r.class_index] == cls.cached_prompt_tokens
            assert ids[: cls.cached_prompt_tokens] == tokenizer.encode(ds.prefixes[r.class_index][r.prefix_index])


def test_chat_template_tokens_count_toward_prompt_and_cache() -> None:
    tokenizer = WordTokenizer()
    ds = _dataset(tokenizer=tokenizer, chat_overhead_tokens=12, chat_head_tokens=5)

    assert ds.prefix_tokens == [400 - 5, 1000 - 5, 0]
    for r in _take(ds, 100):
        assert r.prompt_tokens == ds.mix.classes[r.class_index].prompt_tokens
        assert len(tokenizer.encode(r.prompt)) == r.prompt_tokens - 12


def test_fully_cached_class_keeps_a_unique_tail() -> None:
    mix = {"seed": 1, "classes": [{"weight": 1, "prompt_tokens": 200, "cached_prompt_tokens": 200, "generated_tokens": 4}]}
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

    assert w0.prefixes == w1.prefixes
    prompts = {r.prompt for r in _take(w0, 200)} | {r.prompt for r in _take(w1, 200)}
    assert len(prompts) == 400


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

    for r in _take(ds, 100):
        if chat:
            ids = chat_template_token_ids(tokenizer, path, r.prompt)
        else:
            ids = tokenizer.encode(r.prompt, add_special_tokens=False)
        assert len(ids) == r.prompt_tokens
        if r.prefix_index is not None:
            prefix = ds.prefixes[r.class_index][r.prefix_index]
            cached = head + ds.prefix_tokens[r.class_index]
            assert cached == ds.mix.classes[r.class_index].cached_prompt_tokens
            expected = (chat_template_token_ids(tokenizer, path, prefix)[:head] if chat else []) + tokenizer.encode(
                prefix, add_special_tokens=False
            )
            assert ids[:cached] == expected
