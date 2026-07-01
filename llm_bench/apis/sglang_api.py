#!/usr/bin/env python3
"""SGLang OpenAI-compatible API server launcher."""

from __future__ import annotations
from typing import Any
from llm_bench.apis.base_api import BaseApi, to_cli_args


class SGLangApi(BaseApi):
    api_type = "sglang"
    venv_name = ".venv-sglang"
    init_script_name = "sglang_init_venv.sh"

    def _build_command(
        self, model_path: str, endpoint: dict[str, Any], args: dict[str, Any]
    ) -> list[str]:
        return [
            str(self.venv_bin / "sglang"),
            "serve",
            f"--model-path={model_path}",
            f"--host={endpoint['host']}",
            f"--port={endpoint['port']}",
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
        return "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }

    def has_stream_content(self, response_chunk: dict[str, Any]) -> bool:
        for choice in response_chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                return True
        return False
