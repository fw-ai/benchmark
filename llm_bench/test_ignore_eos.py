from __future__ import annotations

import pytest
from locust.argument_parser import get_parser

from load_test import FireworksProvider

MODEL = "accounts/pyroworks/deployments/test"


def _payload(*flags: str) -> dict:
    options = get_parser().parse_args(
        ["-H", "https://api.fireworks.ai/inference", "--provider", "fireworks", "--model", MODEL, "--no-chat", *flags]
    )
    return FireworksProvider(MODEL, options).format_payload("Hello", 200, None)


@pytest.mark.parametrize(("flags", "expected"), [((), None), (("--ignore-eos",), True)])
def test_ignore_eos_payload(flags: tuple[str, ...], expected: bool | None) -> None:
    payload = _payload(*flags)
    assert payload.get("ignore_eos") is expected
    assert "min_tokens" not in payload
    assert payload["max_tokens"] == 200
