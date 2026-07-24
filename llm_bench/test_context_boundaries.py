from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import context_utils
import gen_load_test
import prefill_load_test
from context_utils import (
    generate_geometric_lengths,
    generation_sequence_limit,
    prefill_prompt_limit,
    resolve_max_seq_len,
)


@pytest.mark.parametrize("maximum", [131072, 420000, 1048576])
@pytest.mark.parametrize("factor", [2, 4])
def test_geometric_lengths_append_exact_maximum(maximum: int, factor: int) -> None:
    lengths = generate_geometric_lengths(1000, maximum, factor=factor)

    assert lengths[-1] == maximum
    assert lengths.count(maximum) == 1
    assert lengths[:-1] == [value for value in lengths[:-1] if value < maximum]


@pytest.mark.parametrize("maximum", [131072, 420000, 1048576])
def test_generation_lengths_include_exact_endpoint(maximum: int) -> None:
    lengths = gen_load_test.generate_seq_lens(1000, maximum)

    assert lengths[-1] == maximum
    assert lengths.count(maximum) == 1


def test_geometric_lengths_do_not_duplicate_geometric_maximum() -> None:
    assert generate_geometric_lengths(1000, 128000, factor=2).count(128000) == 1
    assert generate_geometric_lengths(500, 128000, factor=4).count(128000) == 1


@pytest.mark.parametrize("maximum", [131072, 420000, 1048576])
def test_prefill_pairs_include_exact_prompt_endpoint(maximum: int) -> None:
    prompt_lengths = {prompt_tokens for prompt_tokens, _ in prefill_load_test.generate_pairs(maximum, 500)}

    assert maximum in prompt_lengths


@pytest.mark.parametrize("model_max", [131072, 420000, 1048576])
def test_endpoint_limits_match_fireworks_admission_contract(model_max: int) -> None:
    lookahead = context_utils.FIREWORKS_SPECULATIVE_LOOKAHEAD_TOKENS

    generation_limit = generation_sequence_limit(model_max)
    assert generation_limit + lookahead == model_max

    zero_token_prefill_limit = prefill_prompt_limit(model_max)
    assert zero_token_prefill_limit + lookahead + 1 == model_max

    generation_prefill_limit = prefill_prompt_limit(model_max, max_tokens=100)
    assert generation_prefill_limit + lookahead + 100 == model_max


def test_resolver_falls_back_after_custom_config_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        context_utils.transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("custom config failed")),
    )
    (tmp_path / "config.json").write_text(json.dumps({"language_config": {"max_position_embeddings": 420000}}))

    assert resolve_max_seq_len(str(tmp_path)) == 420000


def test_resolver_uses_sane_alternate_after_hf_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        context_utils.transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unsupported")),
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "model_max_length": 1000000000000000019884624838656,
                    "n_positions": 1048576,
                }
            }
        )
    )

    assert resolve_max_seq_len(str(tmp_path)) == 1048576


@pytest.mark.parametrize(
    "config_text",
    [
        '{"model_max_length": 1000000000000000019884624838656}',
        '{"max_position_embeddings": true}',
        '{"max_position_embeddings": 0}',
        '{"max_position_embeddings": 16777217}',
        "{malformed",
    ],
)
def test_resolver_normalizes_invalid_configs(
    config_text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        context_utils.transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("load failed")),
    )
    (tmp_path / "config.json").write_text(config_text)

    with pytest.raises(ValueError, match="Could not infer a finite max sequence length"):
        resolve_max_seq_len(str(tmp_path))


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        return [1] + self.encode(messages[0]["content"]) + [2]


def test_generation_prompt_builder_hits_exact_token_target() -> None:
    prompt = gen_load_test.build_chat_prompt_ids(
        _FakeTokenizer(),
        tokenizer_path="unused",
        model_type="test",
        suffix_text="XY",
        chunk_texts=["abc"],
        target_len=25,
    )

    assert len(prompt) == 25
    assert prompt[:1] == [1]
    assert prompt[-3:] == [ord("X"), ord("Y"), 2]


class _RecordingSession:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def post(self, _url: str, **kwargs: Any) -> object:
        self.payload = kwargs["json"]
        return object()


def test_generation_completion_sends_raw_ids_with_strict_context_behavior() -> None:
    session = _RecordingSession()
    prompt_ids = [1, 2, 3]

    gen_load_test.post_completion(
        session,
        "https://example.test/v1/completions",
        api_key=None,
        model="model",
        prompt=prompt_ids,
        max_tokens=7,
        n=1,
    )

    assert session.payload is not None
    assert session.payload["prompt"] is prompt_ids
    assert session.payload["ignore_eos"] is True
    assert session.payload["context_length_exceeded_behavior"] == "error"


def test_prefill_completion_sends_raw_ids_with_strict_context_behavior() -> None:
    session = _RecordingSession()
    prompt_ids = [[1, 2], [3, 4]]

    prefill_load_test.post_completion(
        session,
        "https://example.test/v1/completions",
        api_key=None,
        model="model",
        prompt=prompt_ids,
        max_tokens=0,
    )

    assert session.payload is not None
    assert session.payload["prompt"] is prompt_ids
    assert session.payload["context_length_exceeded_behavior"] == "error"


def test_usage_validation_rejects_server_added_prompt_tokens() -> None:
    with pytest.raises(RuntimeError, match="sent 3 token IDs"):
        gen_load_test.validate_completion_usage(
            {"usage": {"prompt_tokens": 4, "completion_tokens": 7}},
            expected_prompt_tokens=3,
            expected_completion_tokens=7,
        )
