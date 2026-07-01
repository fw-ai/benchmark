#!/usr/bin/env python3
"""API server process abstractions for benchmark sweeps."""

from __future__ import annotations

import abc
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def to_cli_args(args: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in args.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        out.append(flag)
        if isinstance(value, (list, tuple)):
            out.extend(str(v) for v in value)
        else:
            out.append(str(value))
    return out


class BaseApi(abc.ABC):
    """Base class that starts one OpenAI-compatible API server."""

    api_type: str
    venv_name: str
    init_script_name: str

    def __init__(
        self,
        model_path: str,
        endpoint: dict[str, Any],
        args: dict[str, Any],
        environment: dict[str, Any],
        *,
        log_dir: str | Path,
    ) -> None:
        self.model_path = model_path
        self.endpoint = dict(endpoint)
        self.args = dict(args)
        self.environment = dict(environment)
        self.name = self.api_type
        self.host = str(self.endpoint["host"])
        self.port = int(self.endpoint["port"])
        self.startup_timeout_s = float(self.endpoint["startup_timeout_s"])
        self.repo_root = Path(__file__).resolve().parents[2]
        self.venv_path = self.repo_root / self.venv_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[bytes] | None = None
        self.server_config = ""

        self.ensure_venv()
        self.start()

    def get_hostname(self) -> str:
        return f"http://{self.host}:{self.port}"

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def has_stream_content(self, response_chunk: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _build_command(
        self, model_path: str, endpoint: dict[str, Any], args: dict[str, Any]
    ) -> list[str]:
        raise NotImplementedError

    @property
    def venv_bin(self) -> Path:
        return self.venv_path / "bin"

    def ensure_venv(self) -> None:
        script = Path(__file__).resolve().parent / self.init_script_name
        subprocess.run(["bash", str(script)], cwd=self.repo_root, check=True)

    def start(self) -> None:
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.environment.items()})
        command = self._build_command(self.model_path, self.endpoint, self.args)
        env_prefix = [f"{k}={shlex.quote(str(v))}" for k, v in self.environment.items()]
        self.server_config = " ".join(
            env_prefix + [shlex.quote(part) for part in command]
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=env,
            start_new_session=True,
        )
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.startup_timeout_s
        url = f"{self.get_hostname().rstrip('/')}/v1/models"
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited during startup with code {self.process.returncode}. "
                    "See parent shell output."
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
                    last_error = f"HTTP {response.status}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = repr(exc)
            time.sleep(5)
        raise TimeoutError(
            f"{self.name} did not become ready at {url}: {last_error}. See parent shell output."
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)

    def __enter__(self) -> BaseApi:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
