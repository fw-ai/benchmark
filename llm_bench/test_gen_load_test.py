from unittest.mock import Mock

import pytest
import requests

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
