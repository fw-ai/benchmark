from __future__ import annotations

from load_test import FirstNTokenTimer


def _word_counter(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def _timer(n: int) -> FirstNTokenTimer:
    return FirstNTokenTimer(n, _word_counter)


def test_stamps_on_the_chunk_that_crosses_the_threshold() -> None:
    timer = _timer(3)

    timer.observe("a", 1.0)
    assert timer.timestamp is None

    timer.observe("a b", 2.0)
    assert timer.timestamp is None

    timer.observe("a b c", 3.0)
    assert timer.timestamp == 3.0


def test_stamps_when_a_chunk_jumps_past_the_threshold() -> None:
    timer = _timer(3)

    timer.observe("a b c d e", 1.5)

    assert timer.timestamp == 1.5
    assert timer.tokens_seen == 5


def test_does_not_restamp_on_later_chunks() -> None:
    timer = _timer(2)

    timer.observe("a b", 1.0)
    timer.observe("a b c", 2.0)
    timer.observe("a b c d", 3.0)

    assert timer.timestamp == 1.0


def test_stays_unstamped_below_the_threshold() -> None:
    timer = _timer(15)

    for i, now in enumerate([1.0, 2.0, 3.0]):
        timer.observe(" ".join(["tok"] * (i + 1)), now)

    assert timer.timestamp is None
    assert timer.tokens_seen == 3


def test_counts_the_whole_prefix_so_split_tokens_are_not_double_counted() -> None:
    # "he" then "hello" is what a tokenizer sees when a token straddles two chunks;
    # summing per-chunk encodes would report 2 tokens instead of 1.
    timer = _timer(2)

    timer.observe("he", 1.0)
    timer.observe("hello", 2.0)

    assert timer.timestamp is None
    assert timer.tokens_seen == 1

    timer.observe("hello world", 3.0)
    assert timer.timestamp == 3.0


def test_disabled_when_n_is_zero_or_negative() -> None:
    for n in (0, -1):
        timer = FirstNTokenTimer(n, _word_counter)
        timer.observe("a b c d e", 1.0)
        assert timer.timestamp is None
        assert timer.tokens_seen == 0


def test_empty_text_does_not_invoke_the_tokenizer() -> None:
    def explode(text: str) -> int:
        raise AssertionError("tokenizer should not be called for empty text")

    timer = FirstNTokenTimer(3, explode)
    timer.observe("", 1.0)

    assert timer.timestamp is None


def test_metric_name_is_derived_from_n() -> None:
    assert _timer(15).metric_name == "time_to_first_15_tokens"
    assert _timer(5).metric_name == "time_to_first_5_tokens"
