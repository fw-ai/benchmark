from __future__ import annotations

import orjson
import pytest

from load_test import extract_prompt_tokens


def _embeddings_body(batch_size: int, dims: int = 4096, prompt_tokens: int = 21132) -> bytes:
    return orjson.dumps(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.0123456789] * dims} for i in range(batch_size)
            ],
            "model": "accounts/fireworks/models/some-embed-model",
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
    )


@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_extract_prompt_tokens_matches_full_parse(batch_size: int) -> None:
    body = _embeddings_body(batch_size)

    assert extract_prompt_tokens(body) == orjson.loads(body)["usage"]["prompt_tokens"]


def test_extract_prompt_tokens_falls_back_when_field_absent() -> None:
    body = orjson.dumps({"object": "list", "data": [], "usage": {"total_tokens": 7}})

    assert extract_prompt_tokens(body) is None


def test_extract_prompt_tokens_survives_malformed_body() -> None:
    assert extract_prompt_tokens(b"not json at all") is None


def test_extract_prompt_tokens_tolerates_whitespace() -> None:
    assert extract_prompt_tokens(b'{"usage": {"prompt_tokens" :  4096 }}') == 4096
