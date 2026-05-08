"""Minimal DeepSeek-V4 prompt encoder for benchmark single-user prompts.

DeepSeek-V4 does not ship a Hugging Face Jinja chat template. The model card
publishes an ``encoding/encoding_dsv4.py`` helper instead; this file keeps the
small subset needed by the load tests so they can still produce token-id
prompts for /v1/completions.
"""

from __future__ import annotations

from typing import Any

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str = "chat",
    add_default_bos_token: bool = True,
) -> str:
    """Encode OpenAI-style messages into the DSV4 prompt format.

    The benchmarks only construct a single user message. Supporting exactly
    that shape keeps this helper small and makes unexpected usage fail loudly.
    """
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError(f"Invalid thinking_mode: {thinking_mode}")
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("DSV4 benchmark encoding only supports a single user message")

    content = messages[0].get("content") or ""
    prompt = BOS_TOKEN if add_default_bos_token else ""
    prompt += USER_TOKEN + content + ASSISTANT_TOKEN
    prompt += THINKING_START_TOKEN if thinking_mode == "thinking" else THINKING_END_TOKEN
    return prompt
