from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests
from requests.structures import CaseInsensitiveDict

import gen_load_test


def _response(status_code: int = 200, text: str = "") -> Mock:
    return Mock(status_code=status_code, text=text)


def _warmup(*, concurrency: int = 1, retries: int = 3) -> None:
    gen_load_test._warmup_seq_len(
        url="https://example.test/v1/completions",
        api_key=None,
        model="model",
        prompt_ids=[1, 2, 3],
        seq_len=3,
        concurrency=concurrency,
        temperature=None,
        retries=retries,
        retry_delay=0,
    )


def test_warmup_retries_transport_exception_from_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        side_effect=[
            requests.exceptions.SSLError("connection closed"),
            _response(),
            _response(),
            _response(),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(gen_load_test, "post_completion", post)
    monkeypatch.setattr(gen_load_test.time, "sleep", sleep)

    _warmup(concurrency=2)

    assert post.call_count == 4
    sleep.assert_called_once_with(0)


def test_warmup_transport_retry_exhaustion_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(side_effect=requests.exceptions.SSLError("connection closed"))
    sleep = Mock()
    monkeypatch.setattr(gen_load_test, "post_completion", post)
    monkeypatch.setattr(gen_load_test.time, "sleep", sleep)

    with pytest.raises(requests.exceptions.SSLError, match="connection closed"):
        _warmup(concurrency=2)

    assert post.call_count == 6
    assert sleep.call_count == 2


def test_successful_warmup_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(return_value=_response())
    sleep = Mock()
    monkeypatch.setattr(gen_load_test, "post_completion", post)
    monkeypatch.setattr(gen_load_test.time, "sleep", sleep)

    _warmup()

    post.assert_called_once()
    sleep.assert_not_called()


def test_warmup_does_not_retry_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(return_value=_response(status_code=400, text="invalid prompt"))
    sleep = Mock()
    monkeypatch.setattr(gen_load_test, "post_completion", post)
    monkeypatch.setattr(gen_load_test.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="failed HTTP 400: invalid prompt"):
        _warmup()

    post.assert_called_once()
    sleep.assert_not_called()


class _RoutedResponse:
    status_code = 200
    text = ""

    def __init__(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        self.headers = CaseInsensitiveDict(
            {
                "fireworks-generation-duration": "1.5",
                "fireworks-cached-prompt-tokens": str(prompt_tokens - 1),
            }
        )
        self._body = {
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens_details": {"cached_tokens": prompt_tokens - 1},
            }
        }

    def json(self) -> dict:
        return self._body


def test_split_batch_across_workers() -> None:
    routing = gen_load_test.RoutingConfig(num_servers=4, num_gens=1)

    assert gen_load_test.split_batch_across_workers(1, routing) == [(0, 1)]
    assert gen_load_test.split_batch_across_workers(3, routing) == [(0, 1), (1, 1), (2, 1)]
    assert gen_load_test.split_batch_across_workers(10, routing) == [
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 2),
    ]


def test_routed_mode_sends_one_n_request_per_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, dict[str, str]]] = []

    def fake_post_completion(*_args: object, **kwargs: object) -> _RoutedResponse:
        n = int(kwargs["n"])
        headers = kwargs["extra_headers"]
        assert isinstance(headers, dict)
        seen.append((n, headers))
        return _RoutedResponse(prompt_tokens=2, completion_tokens=5 * n)

    monkeypatch.setattr(gen_load_test, "post_completion", fake_post_completion)
    routing = gen_load_test.RoutingConfig(num_servers=4, num_gens=1)
    result = gen_load_test._run_pair_routed_n_mode(
        url="http://unused",
        api_key=None,
        model=None,
        prompt_ids=[1, 2],
        max_tokens=5,
        seq_len=7,
        batch_size=10,
        temperature=None,
        users=["u0", "u1", "u2", "u3"],
        routing=routing,
    )

    assert sorted(n for n, _ in seen) == [2, 2, 3, 3]
    assert sorted(headers[gen_load_test._SERVICE_INDEX_HEADER] for _, headers in seen) == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert result.batch_size == 10
    assert result.latency_per_forward == pytest.approx(0.3)


def _stub_benchmark_inputs(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(gen_load_test, "_load_auto_tokenizer", lambda _path: object())
    monkeypatch.setattr(gen_load_test, "resolve_model_type", lambda _path: "test")
    monkeypatch.setattr(gen_load_test, "load_chunks", lambda _dataset: ["chunk"])
    monkeypatch.setattr(
        gen_load_test,
        "build_chunk_texts_to_length",
        lambda _tokenizer, chunks, _target_len: chunks,
    )
    monkeypatch.setattr(
        gen_load_test,
        "build_chat_prompt_ids",
        lambda *_args, target_len, **_kwargs: list(range(target_len)),
    )
    warmup = Mock()
    monkeypatch.setattr(gen_load_test, "_warmup_seq_len", warmup)
    return warmup


def _result() -> gen_load_test.GenBenchmarkResult:
    return gen_load_test.GenBenchmarkResult(
        seq_len=10,
        batch_size=1,
        max_tokens=2,
        generation_duration=1.0,
        latency_per_forward=0.5,
        client_duration=1.1,
    )


def test_repeat_count_warms_once_and_runs_requested_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    warmup = _stub_benchmark_inputs(monkeypatch)
    measurement = Mock(return_value=_result())
    monkeypatch.setattr(gen_load_test, "_run_pair_n_mode", measurement)

    rows = gen_load_test.run_benchmark(
        tokenizer_path="unused",
        model=None,
        base_url="https://example.test",
        api_key=None,
        dataset="code",
        pairs=[(10, 1)],
        max_tokens=2,
        repeat=3,
    )

    assert len(rows) == 3
    warmup.assert_called_once()
    assert measurement.call_count == 3


def test_unbounded_repeat_stops_between_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    warmup = _stub_benchmark_inputs(monkeypatch)
    measurement = Mock(return_value=_result())
    monkeypatch.setattr(gen_load_test, "_run_pair_n_mode", measurement)

    rows = gen_load_test.run_benchmark(
        tokenizer_path="unused",
        model=None,
        base_url="https://example.test",
        api_key=None,
        dataset="code",
        pairs=[(10, 1)],
        max_tokens=2,
        repeat=0,
        stop_requested=lambda: measurement.call_count >= 3,
    )

    assert len(rows) == 3
    warmup.assert_called_once()
    assert measurement.call_count == 3
