from __future__ import annotations

import argparse
import itertools
import json
from types import SimpleNamespace

import pytest
from locust.argument_parser import get_empty_argument_parser, setup_parser_arguments

import load_test
from load_test import LLMUser, parse_lora_adapters, select_round_robin


def test_parse_lora_adapters_splits_strips_and_rejects_empty() -> None:
    assert parse_lora_adapters("a, b ,c") == ("a", "b", "c")
    assert parse_lora_adapters("accounts/x/models/m#accounts/x/deployments/d") == (
        "accounts/x/models/m#accounts/x/deployments/d",
    )
    assert parse_lora_adapters(" , ,a,, ") == ("a",)
    for empty in ("", " , "):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_lora_adapters(empty)


def test_round_robin_is_deterministic() -> None:
    adapters = ("a", "b", "c")
    assert [select_round_robin(adapters, i) for i in range(7)] == ["a", "b", "c", "a", "b", "c", "a"]
    assert select_round_robin((), 3) is None


def test_next_lora_adapter_cycles_across_calls(monkeypatch) -> None:
    monkeypatch.setattr(LLMUser, "_lora_adapter_counter", itertools.count())
    adapters = ("a", "b")
    assert [LLMUser._next_lora_adapter(adapters) for _ in range(5)] == ["a", "b", "a", "b", "a"]
    assert LLMUser._next_lora_adapter(()) is None  # no adapters: the counter isn't consumed
    assert LLMUser._next_lora_adapter(adapters) == "b"


def _user(adapters):
    options = SimpleNamespace(embeddings=False, embeddings_batch_size=1)
    formatter = SimpleNamespace(
        parsed_options=options, format_payload=lambda prompt, max_tokens, images: {"model": "base", "prompt": prompt}
    )
    return SimpleNamespace(
        provider_formatter=formatter,
        environment=SimpleNamespace(parsed_options=options),
        max_tokens_sampler=SimpleNamespace(sample=lambda: 8),
        _get_input=lambda: ("hello", 5, None),
        lora_adapters=adapters,
        _next_lora_adapter=LLMUser._next_lora_adapter,
    )


def test_the_adapter_is_in_the_serialized_body(monkeypatch) -> None:
    monkeypatch.setattr(LLMUser, "_lora_adapter_counter", itertools.count())
    user = _user(("a", "b"))
    bodies = [json.loads(LLMUser._prepare_text_request(user)[3]) for _ in range(3)]
    assert [b["model"] for b in bodies] == ["a", "b", "a"]  # the body that is sent, not only the dict
    plain = _user(())
    data, max_tokens, prompt_tokens, body = LLMUser._prepare_text_request(plain)
    assert data["model"] == "base" and json.loads(body)["model"] == "base" and (max_tokens, prompt_tokens) == (8, 5)


def test_lora_flag_and_default() -> None:
    parser = get_empty_argument_parser()
    setup_parser_arguments(parser)
    load_test.init_parser(parser)
    default = parser.parse_args([])
    assert default.lora_adapters == ()
    given = parser.parse_args(["--lora-adapters", "x,y"])
    assert given.lora_adapters == ("x", "y")
