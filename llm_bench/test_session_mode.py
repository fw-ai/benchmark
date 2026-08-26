from __future__ import annotations

import pytest

from load_test import ChatSession, session_limits


def test_session_limits_follow_prompt_and_cache_lengths() -> None:
    turn_tokens, max_prompt_tokens = session_limits(prompt_tokens=60_000, prompt_cache_max_len=54_000)

    assert turn_tokens == 6_000
    assert max_prompt_tokens == 120_000


@pytest.mark.parametrize("prompt_cache_max_len", [0, -1, 60_000, 60_001])
def test_session_limits_reject_unusable_cache_lengths(prompt_cache_max_len: int) -> None:
    with pytest.raises(ValueError):
        session_limits(prompt_tokens=60_000, prompt_cache_max_len=prompt_cache_max_len)


def run_session(session: ChatSession, turns: int, turn_tokens: int, completion_tokens: int) -> list[int]:
    """Drive `turns` complete turns and return the prompt length each one was sent with."""
    prompt_lengths = []
    for turn in range(turns):
        payload, prompt_tokens = session.start_turn(f"new text {turn}", turn_tokens)
        prompt_lengths.append(prompt_tokens)
        session.complete_turn(
            f"reply {turn}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    return prompt_lengths


def test_turns_are_exact_continuations_of_the_previous_request() -> None:
    session = ChatSession(max_prompt_tokens=120_000, session_id="s")

    first, _ = session.start_turn("first user text", 6_000)
    session.complete_turn("first reply", prompt_tokens=6_000, completion_tokens=600)
    second, _ = session.start_turn("second user text", 6_000)

    assert first["messages"] == [{"role": "user", "content": "first user text"}]
    assert second["messages"] == [
        {"role": "user", "content": "first user text"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second user text"},
    ]


def test_every_turn_reuses_the_same_session_id() -> None:
    session = ChatSession(max_prompt_tokens=12_000, session_id="session-42")

    first, _ = session.start_turn("a", 6_000)
    session.complete_turn("reply", prompt_tokens=6_000, completion_tokens=600)
    second, _ = session.start_turn("b", 6_000)

    assert first["user"] == "session-42"
    assert second["user"] == "session-42"


def test_prompt_grows_by_a_turn_plus_the_previous_response() -> None:
    session = ChatSession(max_prompt_tokens=120_000, session_id="s")

    prompt_lengths = run_session(session, turns=4, turn_tokens=6_000, completion_tokens=600)

    assert prompt_lengths == [6_000, 12_600, 19_200, 25_800]


def test_average_prompt_length_matches_the_requested_length() -> None:
    session = ChatSession(max_prompt_tokens=120_000, session_id="s")

    prompt_lengths = run_session(session, turns=100, turn_tokens=6_000, completion_tokens=600)

    average = sum(prompt_lengths) / len(prompt_lengths)
    assert 54_000 <= average <= 66_000


def test_session_restarts_once_it_passes_the_maximum_prompt_length() -> None:
    session = ChatSession(max_prompt_tokens=12_000, session_id="s")

    _, first = session.start_turn("a", 6_000)
    assert session.complete_turn("reply", prompt_tokens=first, completion_tokens=600) is False

    _, second = session.start_turn("b", 6_000)
    assert second == 12_600
    assert session.complete_turn("reply", prompt_tokens=second, completion_tokens=600) is True

    payload, third = session.start_turn("c", 6_000)
    assert third == 6_000
    assert payload["messages"] == [{"role": "user", "content": "c"}]


def test_failed_turn_leaves_the_conversation_untouched() -> None:
    session = ChatSession(max_prompt_tokens=120_000, session_id="s")
    session.start_turn("first user text", 6_000)
    session.complete_turn("first reply", prompt_tokens=6_000, completion_tokens=600)

    # A request that never completes never commits, so the next turn replaces it.
    session.start_turn("text that failed", 6_000)
    payload, prompt_tokens = session.start_turn("text after the failure", 6_000)

    assert session.turns == 1
    assert prompt_tokens == 12_600
    assert payload["messages"] == [
        {"role": "user", "content": "first user text"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "text after the failure"},
    ]


def test_token_estimate_falls_back_to_client_side_counts() -> None:
    session = ChatSession(max_prompt_tokens=120_000, session_id="s")

    _, first = session.start_turn("a", 6_000)
    session.complete_turn("reply", prompt_tokens=None, completion_tokens=None)
    _, second = session.start_turn("b", 6_000)

    assert first == 6_000
    assert second == 12_000
