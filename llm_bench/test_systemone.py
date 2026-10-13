from __future__ import annotations

import argparse
import json
import os

import pytest
from locust.stats import RequestStats

from load_test import (
    FireworksProvider,
    SystemOneDataset,
    SystemOneMixDataset,
    systemone_option_key,
    systemone_summary_entries,
    validate_systemone_response,
)

LIMERICKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limericks.txt")
MODEL = "accounts/pyroworks/deployments/pkrpl2xp"


class RoundTripWordTokenizer:
    """Offline stand-in for a HF tokenizer: one token per whitespace-separated word."""

    def __init__(self) -> None:
        self._vocab: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            self._vocab.append(word)
            ids.append(len(self._vocab) - 1)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._vocab[i] for i in ids)


def _dataset(tokenizer=None, **overrides) -> SystemOneDataset:
    kwargs = dict(
        path=LIMERICKS,
        tokenizer=tokenizer,
        num_questions=3,
        num_options=5,
        state_tokens=768,
        question_tokens=48,
        unique_state=True,
        seed=0,
    )
    kwargs.update(overrides)
    return SystemOneDataset(**kwargs)


def _options(**overrides) -> argparse.Namespace:
    values = dict(
        systemone=True,
        rerank=False,
        embeddings=False,
        chat=True,
        stream=True,
        acceptance_probs_override=None,
        forced_generation_file=None,
        forced_generation_from_dataset=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _response(question_ids: list[str], output_tokens: int = 0) -> dict:
    return {
        "model": MODEL,
        "answers": {
            qid: {"type": "choice", "choice": "option_a", "probabilities": {"option_a": 0.9, "option_b": 0.1}}
            for qid in question_ids
        },
        "usage": {"input_tokens": 812, "cached_input_tokens": 0, "output_tokens": output_tokens},
    }


def test_option_keys() -> None:
    assert [systemone_option_key(i) for i in (0, 1, 25, 26, 27)] == [
        "option_a",
        "option_b",
        "option_z",
        "option_aa",
        "option_ab",
    ]


@pytest.mark.parametrize("num_questions,num_options", [(1, 4), (3, 5), (16, 30)])
def test_payload_shape(num_questions: int, num_options: int) -> None:
    ds = _dataset(num_questions=num_questions, num_options=num_options)
    provider = FireworksProvider(MODEL, _options())
    prompt, _ = next(ds)
    payload = provider.format_payload(prompt, 0, None)

    assert provider.get_url() == "/v1/systemone"
    assert set(payload) == {"model", "state", "questions"}
    assert payload["model"] == MODEL
    assert isinstance(payload["state"], dict)
    assert list(payload["questions"]) == [f"q{i}" for i in range(num_questions)]
    for question in payload["questions"].values():
        assert question["type"] == "choice"
        assert isinstance(question["instructions"], str) and question["instructions"]
        assert list(question["criteria"]) == [systemone_option_key(j) for j in range(num_options)]
        assert all(isinstance(v, str) and v for v in question["criteria"].values())
    json.dumps(payload)


def test_heuristic_sizing_without_tokenizer() -> None:
    ds = _dataset(state_tokens=2000, question_tokens=64)
    prompt, reported_tokens = next(ds)
    state_chars = len(json.dumps(prompt["state"]))

    assert reported_tokens == ds.state_tokens
    assert 2000 <= ds.state_tokens <= 2000 * 1.15
    assert abs(state_chars / 4 - ds.state_tokens) <= 1
    for question in prompt["questions"].values():
        assert len(question["instructions"]) == 64 * 4


def test_tokenizer_sizing() -> None:
    tokenizer = RoundTripWordTokenizer()
    ds = _dataset(tokenizer=tokenizer, state_tokens=500, question_tokens=20)
    prompt, _ = next(ds)

    assert ds.state_tokens == len(tokenizer.encode(json.dumps(prompt["state"])))
    assert 500 <= ds.state_tokens <= 500 * 1.15
    for question in prompt["questions"].values():
        assert len(tokenizer.encode(question["instructions"])) == 20
    assert ds.question_tokens == [20, 20, 20]


def test_unique_state_nonce_differs_and_leads() -> None:
    ds = _dataset(unique_state=True)
    first, _ = next(ds)
    second, _ = next(ds)

    assert list(first["state"])[0] == "nonce"
    assert json.dumps(first["state"]).startswith('{"nonce": ')
    assert first["state"]["nonce"] != second["state"]["nonce"]
    first_rest = {k: v for k, v in first["state"].items() if k != "nonce"}
    second_rest = {k: v for k, v in second["state"].items() if k != "nonce"}
    assert first_rest == second_rest
    assert first["questions"] == second["questions"]


def test_fixed_state_reused_and_seeded() -> None:
    ds = _dataset(unique_state=False)
    first, _ = next(ds)
    second, _ = next(ds)

    assert json.dumps(first) == json.dumps(second)
    same_seed, _ = next(_dataset(unique_state=False))
    other_seed, _ = next(_dataset(unique_state=False, seed=1))
    assert json.dumps(same_seed) == json.dumps(first)
    assert json.dumps(other_seed["state"]) != json.dumps(first["state"])


def test_unique_and_fixed_states_same_size() -> None:
    assert _dataset(unique_state=True).state_tokens == _dataset(unique_state=False).state_tokens


def test_validate_and_parse_pass() -> None:
    ids = ["q0", "q1", "q2"]
    data = _response(ids)
    provider = FireworksProvider(MODEL, _options())

    assert validate_systemone_response(data, ids) is None
    out = provider.parse_output_json(data)
    assert out.prompt_tokens == 812
    assert out.completion_tokens == 0
    assert out.cached_tokens == 0
    assert out.text == "q0:option_a, q1:option_a, q2:option_a"


@pytest.mark.parametrize(
    "data",
    [
        _response(["q0", "q1"]),
        _response(["q0", "q1", "q2", "q3"]),
        _response(["q0", "q1", "x"]),
        {"error": {"message": "boom"}},
        {"answers": []},
        [],
    ],
)
def test_validate_fail(data) -> None:
    assert validate_systemone_response(data, ["q0", "q1", "q2"]) is not None


def test_fanout_output_tokens_parsed() -> None:
    out = FireworksProvider(MODEL, _options()).parse_output_json(_response(["q0"], output_tokens=3))

    assert out.completion_tokens == 3


def test_summary_fields_present() -> None:
    stats = RequestStats()
    for latency, packed in [(100, 1), (200, 1), (300, 0)]:
        stats.log_request("POST", "/v1/systemone", latency, 10)
        stats.log_request("METRIC", "total_latency", latency, 0)
        stats.log_request("METRIC", "input_tokens", 800, 0)
        stats.log_request("METRIC", "output_tokens", 0 if packed else 4, 0)
        stats.log_request("METRIC", "packed_fraction", packed, 0)
    stats.log_request("POST", "/v1/systemone", 50, 0)
    stats.log_error("POST", "/v1/systemone", "HTTP 500")

    entries = systemone_summary_entries(stats, num_questions=4)

    for key in ["requests", "failures", "questions_per_s", "total_latency", "input_tokens", "packed_fraction"]:
        assert key in entries
    assert entries["requests"] == 4
    assert entries["failures"] == 1
    assert entries["input_tokens"] == 800
    assert entries["packed_fraction"] == pytest.approx(2 / 3)
    assert entries["questions_per_s"] == pytest.approx(stats.entries[("total_latency", "METRIC")].total_rps * 4)


def test_summary_counts_requests_under_host_base_path() -> None:
    stats = RequestStats()
    stats.log_request("POST", "/inference/v1/systemone", 120, 10)
    stats.log_request("POST", "/inference/v1/systemone", 140, 10)
    stats.log_error("POST", "/inference/v1/systemone", "HTTP 503")
    stats.log_request("METRIC", "total_latency", 120, 0)

    entries = systemone_summary_entries(stats, num_questions=4)

    assert entries["requests"] == 2
    assert entries["failures"] == 1


def test_record_systemone_server_timings(monkeypatch) -> None:
    import load_test

    recorded = []
    monkeypatch.setattr(load_test, "add_custom_metric", lambda name, value, length_value=0: recorded.append((name, value)))
    load_test.record_systemone_server_timings(
        {"fireworks-prefill-duration": "0.022", "fireworks-prefill-queue-duration": "bad", "fireworks-other": "1"}
    )
    load_test.record_systemone_server_timings({"fireworks-prefill-queue-duration": ""})

    assert recorded == [("server_side_prefill_latency", pytest.approx(22.0))]


def test_record_systemone_server_timings_per_bucket(monkeypatch) -> None:
    import load_test

    recorded = []
    monkeypatch.setattr(load_test, "add_custom_metric", lambda name, value, length_value=0: recorded.append((name, value)))
    load_test.record_systemone_server_timings(
        {"fireworks-prefill-duration": "0.010", "fireworks-server-processing-time": "0.031"}, "1k-2k"
    )

    assert recorded == [
        ("server_side_prefill_latency", pytest.approx(10.0)),
        ("server_side_prefill_latency@1k-2k", pytest.approx(10.0)),
        ("server_side_total_latency@1k-2k", pytest.approx(31.0)),
    ]


def _mix_spec() -> dict:
    return {
        "num_options": 4,
        "question_tokens": 16,
        "prefix_pool": 2,
        "buckets": [
            {"name": "small", "weight": 3, "state_tokens": 200, "num_questions": 2, "cache": [{"p": 1, "fraction": 0}]},
            {
                "name": "big",
                "weight": 1,
                "state_tokens": 1000,
                "num_questions": 5,
                "cache": [{"p": 1, "fraction": 0.5}],
            },
        ],
    }


def test_mix_buckets_weighted_and_shaped() -> None:
    ds = SystemOneMixDataset(path=LIMERICKS, tokenizer=None, spec=_mix_spec(), seed=0)
    counts = {"small": 0, "big": 0}
    for _ in range(2000):
        prompt, _ = next(ds)
        name = ds.last_bucket["name"]
        counts[name] += 1
        assert len(prompt["questions"]) == (2 if name == "small" else 5)
    assert 0.70 < counts["small"] / 2000 < 0.80


def test_mix_shared_prefix_precedes_nonce_and_is_shared_across_workers() -> None:
    a = SystemOneMixDataset(path=LIMERICKS, tokenizer=None, spec=_mix_spec(), seed=0)
    b = SystemOneMixDataset(path=LIMERICKS, tokenizer=None, spec=_mix_spec(), seed=1)

    def shared_prefixes(ds):
        prefixes = set()
        for _ in range(400):
            prompt, _ = next(ds)
            state = prompt["state"]
            keys = list(state)
            nonce_at = keys.index("nonce")
            if ds.last_bucket["name"] == "small":
                assert nonce_at == 0
            else:
                assert nonce_at > 0 and all(k.startswith("a_shared") for k in keys[:nonce_at])
                assert all(k.startswith("z_private") for k in keys[nonce_at + 1 :])
                prefixes.add(json.dumps({k: state[k] for k in keys[:nonce_at]}))
        return prefixes

    pa, pb = shared_prefixes(a), shared_prefixes(b)
    assert len(pa) == 2 and pa == pb


def test_mix_shared_fraction_sizing() -> None:
    ds = SystemOneMixDataset(path=LIMERICKS, tokenizer=None, spec=_mix_spec(), seed=0)
    while True:
        prompt, _ = next(ds)
        if ds.last_bucket["name"] == "big":
            break
    state = prompt["state"]
    keys = list(state)
    nonce_at = keys.index("nonce")
    shared = ds.count_tokens(json.dumps({k: state[k] for k in keys[:nonce_at]}))
    total = ds.count_tokens(json.dumps(state))
    assert 0.4 < shared / total < 0.6


def test_mix_rejects_empty_spec() -> None:
    with pytest.raises(ValueError):
        SystemOneMixDataset(path=LIMERICKS, tokenizer=None, spec={"buckets": []}, seed=0)
