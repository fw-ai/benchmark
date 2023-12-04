import abc
import argparse
import csv
from dataclasses import dataclass
from functools import partial
import os
import random
import sys
import traceback
from typing import Optional
from locust import HttpUser, task, events, constant_pacing
import copy
import json
import time
import orjson
import threading

try:
    import locust_plugins
except ImportError:
    print("locust-plugins is not installed, Grafana won't work")


def add_custom_metric(name, value, length_value=0):
    events.request.fire(
        request_type="METRIC",
        name=name,
        response_time=value,
        response_length=length_value,
        exception=None,
        context=None,
    )


prompt_prefix = "Pad "  # exactly one token
# "Lengthy" prompt borrowed from nat.dev
prompt = """Generate a Django application with Authentication, JWT, Tests, DB support. Show docker-compose for python and postgres. Show the complete code for every file!"""
prompt_tokens = 35  # from Llama tokenizer tool (so we don't import it here)


class FixedQPSPacer:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, qps, distribution):
        self.qps = qps
        self.distribution = distribution

        # It's kind of thread safe thanks to GIL as the only state is `t` - good enough for a loadtest
        def gen():
            t = time.time()
            mean_wait = 1 / self.qps
            while True:
                if self.distribution == "exponential":
                    wait = random.expovariate(1 / mean_wait)
                elif self.distribution == "uniform":
                    wait = random.uniform(0, 2 * mean_wait)
                elif self.distribution == "constant":
                    wait = mean_wait
                else:
                    print("Unknown distribution {self.distribution}")
                    os._exit(1)
                t += wait
                yield t

        self.iterator = gen()

    @classmethod
    def instance(cls, qps, distribution):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(qps, distribution)
            else:
                assert cls._instance.qps == qps
                assert cls._instance.distribution == distribution
            return cls._instance

    def wait_time_till_next(self):
        with self._lock:
            t = next(self.iterator)
        now = time.time()
        if now > t:
            print(
                f"WARNING: not enough locust users to keep up with the desired QPS. Either the number of locust users is too low or the server is overloaded. Delay: {now-t:.3f}s"
            )
            return 0
        return t - now


class LengthSampler:
    def __init__(self, distribution: str, mean: int, cap: Optional[int], alpha: float):
        self.distribution = distribution
        self.mean = mean
        self.cap = cap
        self.alpha = alpha

        if self.distribution == "exponential":
            self.sample_func = lambda: int(random.expovariate(1 / self.mean))
        elif self.distribution == "uniform":
            mx = self.mean + int(self.alpha * self.mean)
            if self.cap is not None:
                mx = min(mx, self.cap)
            self.sample_func = lambda: random.randint(
                max(1, self.mean - int(self.alpha * self.mean)), mx
            )
        elif self.distribution == "constant":
            self.sample_func = lambda: self.mean
        elif self.distribution == "normal":
            self.sample_func = lambda: int(
                random.gauss(self.mean, self.mean * self.alpha)
            )
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

    def sample(self) -> int:
        for _ in range(1000):
            sample = self.sample_func()
            if sample <= 0:
                continue
            if self.cap is not None and sample > self.cap:
                continue
            return sample
        else:
            raise ValueError(
                "Can't sample a value after 1000 attempts, check distribution parameters"
            )

    def __str__(self):
        r = int(self.mean * self.alpha)
        if self.distribution == "constant":
            s = str(self.mean)
        elif self.distribution == "uniform":
            s = f"uniform({self.mean} +/- {r})"
        elif self.distribution == "normal":
            s = f"normal({self.mean}, {r})"
        elif self.distribution == "exponential":
            s = f"exponential({self.mean})"
        else:
            assert False
        if self.cap is not None:
            s += f" capped at {self.cap}"
        return s


class InitTracker:
    lock = threading.Lock()
    users = None
    first_request_done = 0
    logging_params = None
    environment = None
    tokenizer = None

    @classmethod
    def notify_init(cls, environment, logging_params):
        with cls.lock:
            if cls.environment is None:
                cls.environment = environment
            if cls.logging_params is None:
                cls.logging_params = logging_params
            else:
                assert (
                    cls.logging_params == logging_params
                ), f"Inconsistent settings between workers: {cls.logging_params} != {logging_params}"

    @classmethod
    def notify_first_request(cls):
        with cls.lock:
            if (
                cls.environment.parsed_options.qps is not None
                and cls.first_request_done == 0
            ):
                # if in QPS mode, reset after first successful request comes back
                cls.reset_stats()
            cls.first_request_done += 1
            if (
                cls.environment.parsed_options.qps is not None
                and cls.first_request_done == 0
                and cls.users == cls.first_request_done
            ):
                # if in fixed load mode, reset after all users issued one request (we're in a steady state)
                cls.reset_stats()

    @classmethod
    def notify_spawning_complete(cls, user_count):
        with cls.lock:
            cls.users = user_count
            if cls.users == cls.first_request_done:
                cls.reset_stats()

    @classmethod
    def reset_stats(cls):
        assert cls.environment.runner, "only local mode is supported"
        print("Resetting stats after traffic reach a steady state")
        cls.environment.events.reset_stats.fire()
        cls.environment.runner.stats.reset_all()

    @classmethod
    def load_tokenizer(cls, dir):
        if not dir:
            return None
        with cls.lock:
            if cls.tokenizer:
                return cls.tokenizer
            import transformers

            cls.tokenizer = transformers.AutoTokenizer.from_pretrained(dir)
            cls.tokenizer.add_bos_token = False
            cls.tokenizer.add_eos_token = False
            return cls.tokenizer


events.spawning_complete.add_listener(InitTracker.notify_spawning_complete)


@dataclass
class ChunkMetadata:
    text: str
    logprob_tokens: Optional[int]
    usage_tokens: Optional[int]
    prompt_usage_tokens: Optional[int]


class BaseProvider(abc.ABC):
    DEFAULT_MODEL_NAME = None

    def __init__(self, model, parsed_options):
        self.model = model
        self.parsed_options = parsed_options

    @abc.abstractmethod
    def get_url(self):
        ...

    @abc.abstractmethod
    def format_payload(self, prompt, max_tokens):
        ...

    @abc.abstractmethod
    def parse_output_json(self, json, prompt):
        ...


class OpenAIProvider(BaseProvider):
    def get_url(self):
        if self.parsed_options.chat:
            return "/v1/chat/completions"
        else:
            return "/v1/completions"

    def format_payload(self, prompt, max_tokens):
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
        }
        if self.parsed_options.chat:
            data["messages"] = [{"role": "user", "content": prompt}]
        else:
            data["prompt"] = prompt
        if self.parsed_options.logprobs is not None:
            data["logprobs"] = self.parsed_options.logprobs
        return data

    def parse_output_json(self, data, prompt):
        usage = data.get("usage", None)

        assert len(data["choices"]) == 1, f"Too many choices {len(data['choices'])}"
        choice = data["choices"][0]
        if self.parsed_options.chat:
            if self.parsed_options.stream:
                text = choice["delta"].get("content", "")
            else:
                text = choice["message"]["content"]
        else:
            text = choice["text"]

        logprobs = choice.get("logprobs", None)
        return ChunkMetadata(
            text=text,
            logprob_tokens=len(logprobs["tokens"]) if logprobs else None,
            usage_tokens=usage["completion_tokens"] if usage else None,
            prompt_usage_tokens=usage.get("prompt_tokens", None) if usage else None,
        )


class FireworksProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens):
        data = super().format_payload(prompt, max_tokens)
        data["min_tokens"] = max_tokens
        return data


class VllmProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens):
        data = super().format_payload(prompt, max_tokens)
        data["ignore_eos"] = True
        return data


class TogetherProvider(OpenAIProvider):
    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        return "/"

    def format_payload(self, prompt, max_tokens):
        data = super().format_payload(prompt, max_tokens)
        data["ignore_eos"] = True
        data["stream_tokens"] = data.pop("stream")
        return data

    def parse_output_json(self, data, prompt):
        if not self.parsed_options.stream:
            data = data["output"]
        return super().parse_output_json(data, prompt)

class UpstageProvider(OpenAIProvider):
    def get_url(self):
        if self.parsed_options.chat:
            return "/v1/solar/chat/completions"
        else:
            assert self.parsed_options.chat, "--chat option is mandatory for Upstage"
            return "/"

    def format_payload(self, prompt, max_tokens):
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
        }
        if self.parsed_options.chat:
            data["messages"] = [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}]
        else:
            assert self.parsed_options.chat, "--chat option is mandatory for Upstage"
        return data

class TritonInferProvider(BaseProvider):
    DEFAULT_MODEL_NAME = "ensemble"

    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        assert not self.parsed_options.stream, "Stream is not supported"
        return f"/v2/models/{self.model}/infer"

    def format_payload(self, prompt, max_tokens):
        # matching latest TRT-LLM example, your model configuration might be different
        data = {
            "inputs": [
                {
                    "name": "text_input",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [[prompt]],
                },
                {
                    "name": "max_tokens",
                    "datatype": "UINT32",
                    "shape": [1, 1],
                    "data": [[max_tokens]],
                },
                {
                    "name": "bad_words",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [[""]],
                },
                {
                    "name": "stop_words",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [[""]],
                },
                {
                    "name": "temperature",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [[self.parsed_options.temperature]],
                },
            ]
        }
        assert self.parsed_options.logprobs is None, "logprobs are not supported"
        return data

    def parse_output_json(self, data, prompt):
        for output in data["outputs"]:
            if output["name"] == "text_output":
                assert output["datatype"] == "BYTES"
                assert output["shape"] == [1]
                text = output["data"][0]
                # Triton returns the original prompt in the output, cut it off
                text = text.removeprefix("<s> ")
                if text.startswith(prompt):
                    # HF tokenizers get confused by the leading space
                    text = text[len(prompt) :].removeprefix(" ")
                else:
                    print("WARNING: prompt not found in the output")
                return ChunkMetadata(
                    text=text,
                    logprob_tokens=None,
                    usage_tokens=None,
                    prompt_usage_tokens=None,
                )
        raise ValueError("text_output not found in the response")


class TritonGenerateProvider(BaseProvider):
    DEFAULT_MODEL_NAME = "ensemble"

    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        stream_suffix = "_stream" if self.parsed_options.stream else ""
        return f"/v2/models/{self.model}/generate{stream_suffix}"

    def format_payload(self, prompt, max_tokens):
        data = {
            "text_input": prompt,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
            # for whatever reason these has to be provided
            "bad_words": "",
            "stop_words": "",
        }
        assert self.parsed_options.logprobs is None, "logprobs are not supported"
        return data

    def parse_output_json(self, data, prompt):
        text = data["text_output"]
        if not self.parsed_options.stream:
            # Triton returns the original prompt in the output, cut it off
            text = text.removeprefix("<s> ")
            if text.startswith(prompt):
                # HF tokenizers get confused by the leading space
                text = text[len(prompt) :].removeprefix(" ")
            else:
                print("WARNING: prompt not found in the output")
        return ChunkMetadata(
            text=text,
            logprob_tokens=None,
            usage_tokens=None,
            prompt_usage_tokens=None,
        )


class TgiProvider(BaseProvider):
    DEFAULT_MODEL_NAME = "<unused>"

    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        stream_suffix = "_stream" if self.parsed_options.stream else ""
        return f"/generate{stream_suffix}"

    def format_payload(self, prompt, max_tokens):
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": self.parsed_options.temperature,
                "top_n_tokens": self.parsed_options.logprobs,
                "details": self.parsed_options.logprobs is not None,
            },
        }
        return data

    def parse_output_json(self, data, prompt):
        if "token" in data:
            # streaming chunk
            return ChunkMetadata(
                text=data["token"]["text"],
                logprob_tokens=1,
                usage_tokens=None,
                prompt_usage_tokens=None,
            )
        else:
            # non-streaming response
            return ChunkMetadata(
                text=data["generated_text"],
                logprob_tokens=len(data["details"]["tokens"])
                if "details" in data
                else None,
                usage_tokens=data["details"]["generated_tokens"]
                if "details" in data
                else None,
                prompt_usage_tokens=None,
            )


PROVIDER_CLASS_MAP = {
    "fireworks": FireworksProvider,
    "vllm": VllmProvider,
    "openai": OpenAIProvider,
    "anyscale": OpenAIProvider,
    "together": TogetherProvider,
    "triton-infer": TritonInferProvider,
    "triton-generate": TritonGenerateProvider,
    "tgi": TgiProvider,
    "upstage": UpstageProvider,
}


def _load_curl_like_data(text):
    """
    Either use the passed string or load from a file if the string is `@filename`
    """
    if text.startswith("@"):
        try:
            with open(text[1:], "r") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {text[1:]}") from e
    else:
        return text


class LLMUser(HttpUser):
    # no wait time, so every user creates a continuous load, sending requests as quickly as possible

    def on_start(self):
        try:
            self._on_start()
        except Exception as e:
            print(f"Failed to initialize: {repr(e)}")
            print(traceback.format_exc())
            sys.exit(1)

    def _guess_provider(self):
        self.model = self.environment.parsed_options.model
        self.provider = self.environment.parsed_options.provider
        # guess based on URL
        if self.provider is None:
            if "fireworks.ai" in self.host:
                self.provider = "fireworks"
            elif "together" in self.host:
                self.provider = "together"
            elif "openai" in self.host:
                self.provider = "openai"
            elif "anyscale" in self.host:
                self.provider = "anyscale"
            elif "apistage" in self.host:
                self.provider = "upstage"

        if (
            self.model is None
            and self.provider is not None
            and PROVIDER_CLASS_MAP[self.provider].DEFAULT_MODEL_NAME is not None
        ):
            self.model = PROVIDER_CLASS_MAP[self.provider].DEFAULT_MODEL_NAME

        if self.model and self.provider:
            return

        # vllm doesn't support /model/<name> endpoint, so iterate over all models
        try:
            resp = self.client.get("/v1/models")
            resp.raise_for_status()
            resp = resp.json()
        except Exception as e:
            raise ValueError(
                "Argument --model or --provider was not specified and /v1/models failed"
            ) from e

        models = resp["data"]
        assert len(models) > 0, "No models found in /v1/models"
        owned_by = None
        # pick the first model
        for m in models:
            if self.model is None or m["id"] == self.model:
                self.model = m["id"]
                owned_by = m["owned_by"]
                break
        if self.provider is None:
            if not owned_by:
                raise ValueError(
                    f"Model {self.model} not found in /v1/models. Specify --provider explicitly"
                )
            if owned_by in ["vllm", "fireworks"]:
                self.provider = owned_by
            else:
                raise ValueError(
                    f"Can't detect provider, specify it explicitly with --provider, owned_by={owned_by}"
                )

    def _on_start(self):
        self.client.headers["Content-Type"] = "application/json"
        if self.environment.parsed_options.api_key:
            self.client.headers["Authorization"] = (
                "Bearer " + self.environment.parsed_options.api_key
            )
        self._guess_provider()
        print(f" Provider {self.provider} using model {self.model} ".center(80, "*"))
        self.provider_formatter = PROVIDER_CLASS_MAP[self.provider](
            self.model, self.environment.parsed_options
        )

        self.stream = self.environment.parsed_options.stream
        prompt_chars = self.environment.parsed_options.prompt_chars
        if self.environment.parsed_options.prompt_text:
            self.prompt = _load_curl_like_data(
                self.environment.parsed_options.prompt_text
            )
        elif prompt_chars:
            self.prompt = (
                prompt_prefix * (prompt_chars // len(prompt_prefix) + 1) + prompt
            )[:prompt_chars]
        else:
            assert (
                self.environment.parsed_options.prompt_tokens >= prompt_tokens
            ), f"Minimal prompt length is {prompt_tokens}"
            self.prompt = (
                prompt_prefix
                * (self.environment.parsed_options.prompt_tokens - prompt_tokens)
                + prompt
            )
        self.max_tokens_sampler = LengthSampler(
            distribution=self.environment.parsed_options.max_tokens_distribution,
            mean=self.environment.parsed_options.max_tokens,
            cap=self.environment.parsed_options.max_tokens_cap,
            alpha=self.environment.parsed_options.max_tokens_range,
        )
        self.temperature = self.environment.parsed_options.temperature

        logging_params = {
            # TODO: add some server info with git version
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.environment.parsed_options.prompt_tokens,  # might be overwritten based on metric
            "generation_tokens": str(self.max_tokens_sampler),
            "stream": self.stream,
            "temperature": self.temperature,
            "logprobs": self.environment.parsed_options.logprobs,
        }
        InitTracker.notify_init(self.environment, logging_params)

        self.tokenizer = InitTracker.load_tokenizer(
            self.environment.parsed_options.tokenizer
        )
        if self.tokenizer:
            self.prompt_tokenizer_tokens = len(self.tokenizer.encode(self.prompt))
        else:
            self.prompt_tokenizer_tokens = None

        if self.environment.parsed_options.qps is not None:
            if self.environment.parsed_options.burst:
                raise ValueError("Burst and QPS modes are mutually exclusive")
            pacer = FixedQPSPacer.instance(
                self.environment.parsed_options.qps,
                self.environment.parsed_options.qps_distribution,
            )
            # it will be called by Locust after each task
            self.wait_time = pacer.wait_time_till_next
            self.wait()
        elif self.environment.parsed_options.burst:
            self.wait_time = partial(
                constant_pacing(self.environment.parsed_options.burst), self
            )
        else:
            # introduce initial delay to avoid all users hitting the service at the same time
            time.sleep(random.random())

        self.first_done = False

    @task
    def generate_text(self):
        max_tokens = self.max_tokens_sampler.sample()
        data = self.provider_formatter.format_payload(self.prompt, max_tokens)
        t_start = time.perf_counter()

        with self.client.post(
            self.provider_formatter.get_url(),
            data=json.dumps(data),
            stream=True,
            catch_response=True,
        ) as response:
            dur_chunks = []
            combined_text = ""
            done = False
            prompt_usage_tokens = self.prompt_tokenizer_tokens
            total_usage_tokens = None
            total_logprob_tokens = None
            try:
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Error in response: {response.text}") from e
            t_first_token = None
            for chunk in response.iter_lines(delimiter=b"\n\n"):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    t_prev = time.perf_counter()

                if len(chunk) == 0:
                    continue  # come providers send empty lines between data chunks
                if done:
                    if chunk != b"data: [DONE]":
                        print(f"WARNING: Received more chunks after [DONE]: {chunk}")
                try:
                    now = time.perf_counter()
                    dur_chunks.append(now - t_prev)
                    t_prev = now
                    if self.stream:
                        assert chunk.startswith(
                            b"data:"
                        ), f"Unexpected chunk not starting with 'data': {chunk}"
                        chunk = chunk[len(b"data:") :]
                        if chunk.strip() == b"[DONE]":
                            done = True
                            continue
                    data = orjson.loads(chunk)
                    out = self.provider_formatter.parse_output_json(data, self.prompt)
                    if out.usage_tokens:
                        total_usage_tokens = (
                            total_usage_tokens or 0
                        ) + out.usage_tokens
                    if out.prompt_usage_tokens:
                        prompt_usage_tokens = out.prompt_usage_tokens
                    combined_text += out.text

                    if out.logprob_tokens:
                        total_logprob_tokens = (
                            total_logprob_tokens or 0
                        ) + out.logprob_tokens
                except Exception as e:
                    print(f"Failed to parse response: {chunk} with error {repr(e)}")
                    response.failure(e)
                    return
            assert t_first_token is not None, "empty response received"
            if (
                (total_logprob_tokens is not None)
                and (total_usage_tokens is not None)
                and total_logprob_tokens != total_usage_tokens
            ):
                print(
                    f"WARNING: usage_tokens {total_usage_tokens} != logprob_tokens {total_logprob_tokens}"
                )
            if total_logprob_tokens is not None:
                num_tokens = total_logprob_tokens
            else:
                num_tokens = total_usage_tokens
            if self.tokenizer:
                num_tokenizer_tokens = len(self.tokenizer.encode(combined_text))
                if num_tokens is None:
                    num_tokens = num_tokenizer_tokens
                elif num_tokens != num_tokenizer_tokens:
                    print(
                        f"WARNING: tokenizer token count {num_tokenizer_tokens} != {num_tokens} received from server"
                    )
            num_tokens = num_tokens or 0
            num_chars = len(combined_text)
            now = time.perf_counter()
            dur_total = now - t_start
            dur_generation = now - t_first_token
            dur_first_token = t_first_token - t_start
            print(
                f"Response received: total {dur_total*1000:.2f} ms, first token {dur_first_token*1000:.2f} ms, {num_chars} chars, {num_tokens} tokens"
            )
            if self.environment.parsed_options.show_response:
                print("---")
                print(combined_text)
                print("---")
            if num_chars:
                add_custom_metric(
                    "latency_per_char", dur_generation / num_chars * 1000, num_chars
                )
            add_custom_metric("time_to_first_token", dur_first_token * 1000)
            add_custom_metric("total_latency", dur_total * 1000)
            if num_tokens:
                if num_tokens != max_tokens:
                    print(
                        f"WARNING: wrong number of tokens: {num_tokens}, expected {max_tokens}"
                    )
                add_custom_metric("num_tokens", num_tokens)
                add_custom_metric(
                    "latency_per_token", dur_generation / num_tokens * 1000, num_tokens
                )
                add_custom_metric(
                    "overall_latency_per_token",
                    dur_total / num_tokens * 1000,
                    num_tokens,
                )
            if (
                prompt_usage_tokens is not None
                and self.prompt_tokenizer_tokens is not None
                and prompt_usage_tokens != self.prompt_tokenizer_tokens
            ):
                print(
                    f"WARNING: prompt usage tokens {prompt_usage_tokens} != {self.prompt_tokenizer_tokens} derived from local tokenizer"
                )
            prompt_tokens = prompt_usage_tokens or self.prompt_tokenizer_tokens
            if prompt_tokens:
                add_custom_metric("prompt_tokens", prompt_tokens)

            if not self.first_done:
                self.first_done = True
                InitTracker.notify_first_request()


@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CLASS_MAP.keys()),
        type=str,
        help="Which flavor of API to use. If not specified, we'll try to guess based on the URL and /v1/models output",
    )
    parser.add_argument(
        "-m",
        "--model",
        env_var="MODEL",
        type=str,
        help="The model to use for generating text. If not specified we will pick the first model from the service as returned by /v1/models",
    )
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /v1/chat/completions API",
    )
    parser.add_argument(
        "-p",
        "--prompt-tokens",
        env_var="PROMPT_TOKENS",
        type=int,
        default=512,
        help="Length of the prompt in tokens. Default 512",
    )
    parser.add_argument(
        "--prompt-chars",
        env_var="PROMPT_CHARS",
        type=int,
        help="Length of the prompt in characters.",
    )
    parser.add_argument(
        "--prompt-text",
        env_var="PROMPT_TEXT",
        type=str,
        help="Prompt text to use instead of generating one. It can be a file reference starting with an ampersand, e.g. `@prompt.txt`",
    )
    parser.add_argument(
        "-o",
        "--max-tokens",
        env_var="MAX_TOKENS",
        type=int,
        default=64,
        help="Max number of tokens to generate. If --max-tokens-distribution is non-constant this is going to be the mean. Defaults to 64",
    )
    parser.add_argument(
        "--max-tokens-cap",
        env_var="MAX_TOKENS_CAP",
        type=int,
        help="If --max-tokens-distribution is non-constant, this truncates the distribition at the specified limit",
    )
    parser.add_argument(
        "--max-tokens-distribution",
        env_var="MAX_TOKENS_DISTRIBUTION",
        type=str,
        choices=["constant", "uniform", "exponential", "normal"],
        default="constant",
        help="How to sample `max-tokens` on each request",
    )
    parser.add_argument(
        "--max-tokens-range",
        env_var="MAX_TOKENS_RANGE",
        type=float,
        default=0.3,
        help="Specifies the width of the distribution. Specified value `alpha` is relative to `max-tokens`. For uniform distribution we'd sample from [max_tokens - max_tokens * alpha, max_tokens + max_tokens * alpha]. For normal distribution we'd sample from `N(max_tokens, max_tokens * alpha)`. Defaults to 0.3",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the streaming API",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        env_var="API_KEY",
        help="Auth for the API",
    )
    parser.add_argument(
        "--temperature",
        env_var="TEMPERATURE",
        type=float,
        default=1.0,
        help="Temperature parameter for the API",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Whether to ask for logprobs, it makes things slower for some providers but is necessary for token count in streaming (unless it's Fireworks API that returns usage in streaming mode)",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        help="Append the line with the summary to the specified CSV file. Useful for generating a spreadsheet with perf sweep results. If the file doesn't exist, writes out the header first",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="Enabled 'fixed QPS' mode where requests are issues at the specified rate regardless of how long the processing takes. In this case --users and --spawn-rate need to be set to a sufficiently high value (e.g. 100)",
    )
    parser.add_argument(
        "--qps-distribution",
        type=str,
        choices=["constant", "uniform", "exponential"],
        default="constant",
        help="Must be used with --qps. Specifies how to space out requests: equally ('constant') or by sampling wait times from a distribution ('uniform' or 'exponential'). Expected QPS is going to match --qps",
    )
    parser.add_argument(
        "--burst",
        type=float,
        default=None,
        help="Makes requests to arrive in bursts every specified number of seconds. Note that burst duration has to be longer than maximum time of the response. Size of the burst is controlled by --users. The spawn rate -r is best set to a high value",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Specify HF tokenizer to use for validating the output of the model. It's optional, we're going to rely on 'usage' or 'logprobs' field to get token count information",
    )
    parser.add_argument(
        "--show-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the result of each generation",
    )


@events.quitting.add_listener
def _(environment, **kw):
    total_latency = environment.stats.entries[("total_latency", "METRIC")]
    if environment.stats.total.num_failures > 0 or total_latency.num_requests == 0:
        print("Test failed due to failed requests")
        environment.process_exit_code = 1
        return

    entries = copy.copy(InitTracker.logging_params)
    if environment.parsed_options.qps is not None:
        entries[
            "concurrency"
        ] = f"QPS {environment.parsed_options.qps} {environment.parsed_options.qps_distribution}"
    else:
        entries["concurrency"] = InitTracker.users
    for metric_name in [
        "time_to_first_token",
        "latency_per_token",
        "num_tokens",
        "total_latency",
        "prompt_tokens",  # might overwrite the static value based on server side tokenization
    ]:
        entries[metric_name] = environment.stats.entries[
            (metric_name, "METRIC")
        ].avg_response_time
    if not environment.parsed_options.stream:
        # if there's no streaming these metrics are meaningless
        entries["time_to_first_token"] = ""
        entries["latency_per_token"] = ""
    entries["num_requests"] = total_latency.num_requests
    entries["qps"] = total_latency.total_rps

    pretty_name = lambda s: " ".join([w.capitalize() for w in s.split("_")])
    entries = {pretty_name(k): v for k, v in entries.items()}

    # print in the final event handler to make sure our output is the last one
    @events.quit.add_listener
    def exit_printer(**kw):
        max_width = max(len(k) for k in entries.keys())
        print(" Summary ".center(80, "="))
        for k, v in entries.items():
            print(f"{k:<{max_width}}: {v}")
        print("=" * 80)

    if environment.parsed_options.summary_file:
        with open(environment.parsed_options.summary_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=entries.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(entries)
