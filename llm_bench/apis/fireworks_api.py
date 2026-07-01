#!/usr/bin/env python3
"""Fireworks OpenAI-compatible API server launcher."""

from __future__ import annotations
from typing import Any
from llm_bench.apis.base_api import BaseApi, to_cli_args


class FireworksApi(BaseApi):
    api_type = "fireworks"
    venv_name = ".venv-fireworks"
    init_script_name = "fireworks_init_venv.sh"

    def _build_command(
        self, model_path: str, endpoint: dict[str, Any], args: dict[str, Any]
    ) -> list[str]:
        return [
            str(self.venv_bin / "python"),
            "-m",
            "fireworks.serving",
            "--server-type=text",
            f"--host={endpoint['host']}",
            f"--port={endpoint['port']}",
            model_path,
            *to_cli_args(args),
        ]

    def stream_payload(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        cache_tokens: int,
        force_min_tokens: bool,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "prompt_cache_max_len": cache_tokens,
        }
        if force_min_tokens:
            payload["min_tokens"] = max_tokens
        return "/v1/completions", payload

    def has_stream_content(self, response_chunk: dict[str, Any]) -> bool:
        for choice in response_chunk.get("choices", []):
            if choice.get("text"):
                return True
        return False
