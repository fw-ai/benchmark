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
import base64
import io
import itertools
from PIL import Image
import transformers
import re
import gevent
from locust.util.timespan import parse_timespan as _locust_parse_timespan

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


PROMPT_CHAT_IMAGE_PLACEHOLDER = "<image>"


class TranslationDataset:
    def __init__(
        self,
        path: str,
        prompt: str,
        tokenizer_path: str,
        chat: bool,
        num_tokens: int,
        common_tokens: int,
    ):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self._num_tokens = num_tokens

        self._all_limericks = []
        with open(path, "r") as f:
            text = f.read()
            lims = text.split("\n\n")
            for i, lim in enumerate(lims):
                num_tokens = len(self._tokenizer.encode(lim, add_special_tokens=False))
                self._all_limericks.append((lim, num_tokens))

        self._prefix = ""
        self._suffix = prompt
        self._prefix_suffix_tokens = len(self._tokenizer.encode(prompt, add_special_tokens=False))
        # Use deterministic selection (sequential iteration) to ensure all workers
        # get the same prefix for the same common_tokens value
        idx = 0
        while self._prefix_suffix_tokens < common_tokens:
            lim, num_tokens = self._all_limericks[idx % len(self._all_limericks)]
            self._prefix += lim + "\n\n"
            self._prefix_suffix_tokens += num_tokens
            idx += 1

        if chat:
            empty_tempalate_tokens = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=True,
            )
            self._prefix_suffix_tokens += len(empty_tempalate_tokens)

    def __next__(self):
        prompt_tokens = self._prefix_suffix_tokens
        prompt = self._prefix
        while prompt_tokens < self._num_tokens:
            lim, num_tokens = self._all_limericks[random.randint(0, len(self._all_limericks) - 1)]

            prompt += lim + "\n\n"
            prompt_tokens += num_tokens
        prompt += self._suffix

        return prompt, prompt_tokens

    def __iter__(self):
        return self


class JsonlDataset:
    def __init__(
        self,
        path: str,
        shuffle_seed: Optional[int] = None,
        dataset_limit: Optional[int] = None,
    ):
        self.path = path
        self.shuffle_seed = shuffle_seed
        self.dataset_limit = dataset_limit

    def __iter__(self):
        return itertools.cycle(self._read_data())

    def _read_data(self):
        # Read all data into a list first (needed for shuffling and limiting)
        data = []
        with open(self.path, "r") as f:
            for line in f:
                data.append((json.loads(line), 0))

        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(data)
            print(f"Shuffled dataset with seed {self.shuffle_seed}")

        # Limit dataset size if specified
        if self.dataset_limit is not None:
            data = data[: self.dataset_limit]
            print(f"Limited dataset to first {len(data)} items")

        # Yield all items
        for item in data:
            yield item


class DatasetHolder:
    _instance = None

    @classmethod
    def _create_dataset(cls, options: argparse.Namespace):
        if options.dataset.startswith("@"):
            return JsonlDataset(
                options.dataset[1:],
                shuffle_seed=getattr(options, "dataset_shuffle_seed", None),
                dataset_limit=getattr(options, "dataset_limit", None),
            )
        elif options.dataset in ("limericks", "code"):
            assert options.tokenizer is not None, "--tokenizer is required for limericks or code dataset"
            if options.dataset == "limericks":
                if options.prompt is None:
                    prompt = "Translate the limericks above to Spanish."
                else:
                    prompt = options.prompt
                dataset_file = "limericks.txt"
            elif options.dataset == "code":
                if options.prompt is None:
                    prompt = "Translate the code above to C++."
                else:
                    prompt = options.prompt
                dataset_file = "code.txt"

            return TranslationDataset(
                path=os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_file),
                prompt="\n\n" + prompt,
                tokenizer_path=options.tokenizer,
                chat=options.chat,
                num_tokens=options.prompt_tokens,
                common_tokens=options.prompt_cache_max_len,
            )
        else:
            raise ValueError(f"Unknown dataset: {options.dataset}")

    @classmethod
    def get_instance(cls, options: argparse.Namespace):
        if cls._instance is None:
            cls._instance = cls._create_dataset(options)
        return cls._instance


class FixedQPSPacer:
    _instance = None

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
        if cls._instance is None:
            cls._instance = cls(qps, distribution)
        else:
            assert cls._instance.qps == qps
            assert cls._instance.distribution == distribution
        return cls._instance

    def wait_time_till_next(self):
        t = next(self.iterator)
        now = time.time()
        if now > t:
            print(
                f"WARNING: not enough locust users to keep up with the desired QPS. Either the number of locust users is too low or the server is overloaded. Delay: {now-t:.3f}s"
            )
            return 0
        return t - now


class RampingPacer:
    """
    A pacer that gradually increases concurrent requests from min_users to max_users.

    Tracks the number of in-flight requests and controls concurrency by making
    users wait when above the target. Starts at min_users concurrent requests,
    runs for warmup_time seconds, then increases by pct percent every second
    until max_users is reached.
    """

    _instance = None

    def __init__(self, min_users, max_users, pct, warmup_time=30, max_spawn_rate=1):
        self.min_users = min_users
        self.max_users = max_users
        self.pct = pct
        self.warmup_time = warmup_time
        self.max_spawn_rate = max_spawn_rate
        self.start_time = time.time()
        self._logged_target = min_users
        self._warmup_complete = False
        self._active_requests = 0  # Track number of concurrent requests

    def _get_target_users(self):
        now = time.time()
        elapsed = now - self.start_time

        # During initial warmup period, stay at min_users
        if elapsed < self.warmup_time:
            return self.min_users

        # After warmup, increase users by pct every second
        if not self._warmup_complete:
            self._warmup_complete = True
            print(f"RampingPacer: warmup complete, starting to increase concurrency from {self.min_users}")

        # Calculate target users based on time since warmup ended
        seconds_since_warmup = now - (self.start_time + self.warmup_time)
        # Compound growth: users = min_users * (1 + pct/100)^seconds
        multiplier = (1 + self.pct / 100) ** seconds_since_warmup
        target = self.min_users * multiplier

        # Cap ramp-up rate at max_spawn_rate users per second
        max_linear_target = self.min_users + seconds_since_warmup * self.max_spawn_rate
        target = min(target, max_linear_target)

        # Cap at max_users
        if target >= self.max_users:
            target = self.max_users

        # Log target changes
        if abs(target - self._logged_target) >= 0.5:
            print(f"RampingPacer: target concurrency now at {target:.1f}, active: {self._active_requests}")
            self._logged_target = target

        return target

    def request_start(self):
        """Called when a request is about to start. Increments active count."""
        self._active_requests += 1

    def request_end(self):
        """Called when a request completes. Decrements active count."""
        self._active_requests -= 1

    @classmethod
    def instance(cls, min_users, max_users, pct, warmup_time=30, max_spawn_rate=1):
        if cls._instance is None:
            cls._instance = cls(min_users, max_users, pct, warmup_time, max_spawn_rate)
        else:
            assert cls._instance.min_users == min_users
            assert cls._instance.max_users == max_users
            assert cls._instance.pct == pct
            assert cls._instance.warmup_time == warmup_time
            assert cls._instance.max_spawn_rate == max_spawn_rate
        return cls._instance

    def can_start(self):
        """Returns True if we're below target concurrency and can start a new request."""
        target = self._get_target_users()
        return self._active_requests < target


class LengthSampler:
    def __init__(self, distribution: str, mean: int, cap: Optional[int], alpha: float):
        self.distribution = distribution
        self.mean = mean
        self.cap = cap
        self.alpha = alpha

        if self.mean == 0:
            self.sample_func = lambda: 0
        elif self.distribution == "exponential":
            self.sample_func = lambda: int(random.expovariate(1 / self.mean))
        elif self.distribution == "uniform":
            mx = self.mean + int(self.alpha * self.mean)
            if self.cap is not None:
                mx = min(mx, self.cap)
            self.sample_func = lambda: random.randint(max(1, self.mean - int(self.alpha * self.mean)), mx)
        elif self.distribution == "constant":
            self.sample_func = None
        elif self.distribution == "normal":
            self.sample_func = lambda: int(random.gauss(self.mean, self.mean * self.alpha))
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

    def sample(self) -> int:
        if self.mean == 0:
            return 0
        if self.distribution == "constant":
            return self.mean

        for _ in range(1000):
            sample = self.sample_func()
            if sample <= 0:
                continue
            if self.cap is not None and sample > self.cap:
                continue
            return sample
        else:
            raise ValueError("Can't sample a value after 1000 attempts, check distribution parameters")

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
    users = None
    first_request_done = 0
    logging_params = None
    environment = None
    tokenizer = None
    deferred_run_time_seconds = None
    deferred_max_requests = None
    request_count = 0
    stop_scheduled = False
    max_requests_stop_scheduled = False
    stats_reset_done = False

    @classmethod
    def notify_init(cls, environment, logging_params):
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
        cls.first_request_done += 1

    @classmethod
    def notify_request_complete(cls):
        """Notify that a request has completed and check if we should stop."""
        if cls.deferred_max_requests is None:
            return
        cls.request_count += 1
        print(f"Request {cls.request_count}/{cls.deferred_max_requests} completed")
        if cls.request_count >= cls.deferred_max_requests:
            print(f"Reached max requests limit ({cls.deferred_max_requests}), stopping test")
            cls.stop_scheduled = True
            # Use a small delay to ensure the current request completes
            gevent.spawn_later(0.1, cls._do_quit)

    @classmethod
    def _do_quit(cls):
        """Actually stop the runner."""
        if not cls.environment:
            print("WARNING: environment is None, cannot stop")
            return
        if not cls.environment.runner:
            print("WARNING: runner is None, cannot stop")
            return

        runner = cls.environment.runner
        try:
            if hasattr(runner, "stop"):
                runner.stop()
            runner.quit()
        except Exception as e:
            print(f"Failed to stop runner: {e}")

    @classmethod
    def notify_spawning_complete(cls, user_count):
        cls.users = user_count
        # Start steady-state measurement exactly when all users have spawned
        if not cls.stats_reset_done:
            cls.reset_stats()
            cls.stats_reset_done = True
        # If -t/--run-time was provided, schedule test stop relative to spawn complete
        if (
            cls.deferred_run_time_seconds is not None
            and not cls.stop_scheduled
            and cls.environment is not None
            and cls.environment.runner is not None
        ):
            delay = float(cls.deferred_run_time_seconds)
            print(f"Scheduling stop {delay}s after spawning complete (deferred -t)")
            gevent.spawn_later(delay, cls.environment.runner.quit)
            cls.stop_scheduled = True

    @classmethod
    def reset_stats(cls):
        if cls.environment is None or cls.environment.runner is None:
            return
        print("Resetting stats after traffic reached a steady state")
        cls.environment.events.reset_stats.fire()
        cls.environment.runner.stats.reset_all()

    @classmethod
    def load_tokenizer(cls, dir):
        if not dir:
            return None
        if cls.tokenizer:
            return cls.tokenizer
        import transformers

        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(dir, trust_remote_code=True)
        cls.tokenizer.add_bos_token = False
        cls.tokenizer.add_eos_token = False
        return cls.tokenizer


events.spawning_complete.add_listener(InitTracker.notify_spawning_complete)


def _parse_run_time_to_seconds(run_time_value):
    """Parse Locust -t/--run-time value into seconds (float). Supports both
    already-parsed numeric values and human strings like '30s', '5m', '1h30m'.
    """
    if not run_time_value:
        return None
    # If Locust already parsed it to a number (seconds), just use it
    if isinstance(run_time_value, (int, float)):
        return float(run_time_value)
    # Try Locust's own parser first
    if _locust_parse_timespan is not None:
        try:
            return float(_locust_parse_timespan(run_time_value))
        except Exception:
            pass
    # Fallback simple parser for strings like '1h30m15s'
    s = str(run_time_value).strip().lower()
    total = 0.0
    for value, unit in re.findall(r"(\d+)\s*([smhd])", s):
        n = float(value)
        if unit == "s":
            total += n
        elif unit == "m":
            total += n * 60
        elif unit == "h":
            total += n * 3600
        elif unit == "d":
            total += n * 86400
    if total == 0.0:
        raise ValueError(f"Unable to parse run time value: {run_time_value}")
    return total


@events.init.add_listener
def _defer_run_time_to_after_spawn(environment, **_kwargs):
    """Capture -t/--run-time and --max-requests and defer them appropriately.

    For run-time: we store the desired duration, null out the original option to prevent
    Locust from scheduling an early stop, and then schedule our own stop in
    InitTracker.notify_spawning_complete.

    For max-requests: we store the limit and check it after each request completes.
    """
    try:
        run_time_value = getattr(environment.parsed_options, "run_time", None)
    except Exception:
        run_time_value = None
    seconds = _parse_run_time_to_seconds(run_time_value) if run_time_value else None
    if seconds:
        # Disable Locust's default run_time handling by clearing it
        try:
            environment.parsed_options.run_time = None
        except Exception:
            pass
        InitTracker.deferred_run_time_seconds = seconds
        InitTracker.environment = environment
        print(f"Deferring -t/--run-time to start after spawning complete: {seconds}s")

    # Capture max_requests if specified
    try:
        max_requests = getattr(environment.parsed_options, "max_requests", None)
    except Exception:
        max_requests = None
    if max_requests is not None:
        InitTracker.deferred_max_requests = max_requests
        InitTracker.environment = environment
        print(f"Will stop after {max_requests} requests complete")


@dataclass
class ChunkMetadata:
    text: str
    logprob_tokens: Optional[int]
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    cached_tokens: Optional[int]


class BaseProvider(abc.ABC):
    DEFAULT_MODEL_NAME = None

    def __init__(self, model, parsed_options):
        self.model = model
        self.parsed_options = parsed_options

    @abc.abstractmethod
    def get_url(self): ...

    @abc.abstractmethod
    def format_payload(self, prompt, max_tokens, images): ...

    @abc.abstractmethod
    def parse_output_json(self, json): ...

    def post_response_hook(self, headers, num_tokens, perf_metrics=None):
        """Hook for provider-specific post-response processing.

        Override this method to extract and emit custom metrics from response headers
        or perf_metrics (for streaming responses).

        Args:
            headers: Response headers dict
            num_tokens: Number of tokens generated in this response
            perf_metrics: Optional dict of performance metrics from response body (streaming)
        """
        pass


class OpenAIProvider(BaseProvider):
    def get_url(self):
        if self.parsed_options.embeddings:
            return "/v1/embeddings"
        elif self.parsed_options.chat:
            return "/v1/chat/completions"
        else:
            return "/v1/completions"

    def format_payload(self, prompt, max_tokens, images):
        if self.parsed_options.embeddings:
            data = {
                "model": self.model,
                "input": prompt,
            }
            # Add embeddings-specific parameters
            if self.parsed_options.return_logits is not None:
                data["return_logits"] = self.parsed_options.return_logits
            if self.parsed_options.normalize is not None:
                data["normalize"] = self.parsed_options.normalize
            return data

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
            "n": self.parsed_options.n,
        }
        if self.parsed_options.stream:
            data["stream_options"] = {"include_usage": True}
        if self.parsed_options.top_k is not None:
            data["top_k"] = self.parsed_options.top_k
        if self.parsed_options.logprobs is not None:
            data["logprobs"] = self.parsed_options.logprobs
        if self.parsed_options.reasoning_effort is not None:
            data["reasoning_effort"] = self.parsed_options.reasoning_effort
        if isinstance(prompt, str):
            if self.parsed_options.chat:
                if images is None:
                    data["messages"] = [{"role": "user", "content": prompt}]
                else:
                    image_urls = []
                    for image in images:
                        image_urls.append({"type": "image_url", "image_url": {"url": image}})
                    data["messages"] = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}, *image_urls],
                        }
                    ]
            else:
                data["prompt"] = prompt
                if images is not None:
                    data["images"] = images
        else:
            assert isinstance(prompt, dict), "prompt must be a dict"
            for k, v in prompt.items():
                data[k] = v

        # Clear last assistant message if requested
        if (
            self.parsed_options.clear_assistant
            and "messages" in data
            and isinstance(data["messages"], list)
            and len(data["messages"]) > 0
        ):
            # Remove the last message if it's from assistant
            if data["messages"][-1].get("role") == "assistant":
                data["messages"].pop()

        return data

    def parse_output_json(self, data):
        if self.parsed_options.embeddings:
            return ChunkMetadata(
                text=data["data"][0]["embedding"],
                logprob_tokens=None,
                completion_tokens=None,
                prompt_tokens=None,
                cached_tokens=None,
            )
        usage = data.get("usage", None)

        assert len(data["choices"]) == 1, f"Too many choices {len(data['choices'])}"
        choice = data["choices"][0]
        if self.parsed_options.chat:
            if self.parsed_options.stream:
                block = choice["delta"]
            else:
                block = choice["message"]
            text = (
                (block.get("reasoning", "") or "")
                + (block.get("reasoning_content", "") or "")
                + (block.get("content", "") or "")
            )
        else:
            text = choice["text"]

        logprobs = choice.get("logprobs", None)
        if logprobs and "tokens" in logprobs:
            logprob_tokens = len(logprobs["tokens"])
        else:
            logprob_tokens = None

        cached_tokens = None
        if usage:
            prompt_tokens_details = usage.get("prompt_tokens_details") or {}
            cached_tokens = prompt_tokens_details.get("cached_tokens", None)

        return ChunkMetadata(
            text=text,
            logprob_tokens=logprob_tokens,
            completion_tokens=usage["completion_tokens"] if usage else None,
            prompt_tokens=usage.get("prompt_tokens", None) if usage else None,
            cached_tokens=cached_tokens,
        )


class FireworksProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens, images):
        data = super().format_payload(prompt, max_tokens, images)
        # Enable perf_metrics_in_response to get speculation stats in streaming responses
        data["perf_metrics_in_response"] = True
        # Add prompt_cache_max_pct if specified (Fireworks-specific parameter)
        if self.parsed_options.prompt_cache_max_pct is not None:
            data["prompt_cache_max_pct"] = int(self.parsed_options.prompt_cache_max_pct)
        return data

    def post_response_hook(self, headers, num_tokens, perf_metrics=None):
        """Process Fireworks-specific response for speculation hit rate tracking.

        For streaming responses, speculation stats come from perf_metrics in the response body.
        For non-streaming responses, they come from headers.

        Only records speculation hit rates for generations with >= 30 tokens to ensure
        meaningful statistics.
        """
        # Only track speculation hit rates for sufficiently long generations
        if num_tokens is None or num_tokens < 30:
            return

        # Try to get speculation acceptance from perf_metrics first (streaming),
        # then fall back to headers (non-streaming)
        speculation_hit_rates = None
        if perf_metrics:
            speculation_hit_rates = perf_metrics.get("speculation-acceptance")
            if self.parsed_options.show_response:
                print(f"DEBUG: perf_metrics: {perf_metrics}")
        if not speculation_hit_rates:
            speculation_hit_rates = headers.get("fireworks-speculation-acceptance")
            if self.parsed_options.show_response:
                print(f"DEBUG: Response headers: {dict(headers)}")

        if not speculation_hit_rates:
            if self.parsed_options.show_response:
                print("DEBUG: speculation-acceptance not found in perf_metrics or headers")
            return

        try:
            positions = speculation_hit_rates.split(",")
            for position in positions:
                position, hit_rate_frac = position.split(":")
                hits, total = map(int, hit_rate_frac.split("/"))
                if total > 0:
                    hit_rate = hits / total
                    add_custom_metric(
                        f"speculation_hit_rate_position_{position}",
                        hit_rate * 100,
                    )
        except Exception as e:
            print(f"WARNING: Failed to parse speculation hit rates '{speculation_hit_rates}': {e}")


class VllmProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens, images):
        data = super().format_payload(prompt, max_tokens, images)
        data["ignore_eos"] = True
        if data.get("stream"):
            data["stream_options"] = {"include_usage": True}
        return data


class TogetherProvider(OpenAIProvider):
    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        return "/"

    def format_payload(self, prompt, max_tokens, images):
        data = super().format_payload(prompt, max_tokens, images)
        data["ignore_eos"] = True
        data["stream_tokens"] = data.pop("stream")
        return data

    def parse_output_json(self, data):
        if not self.parsed_options.stream:
            data = data["output"]
        return super().parse_output_json(data)


class TgiProvider(BaseProvider):
    DEFAULT_MODEL_NAME = "<unused>"

    def get_url(self):
        assert self.parsed_options.n == 1, "n > 1 is not supported"
        assert not self.parsed_options.chat, "Chat is not supported"
        stream_suffix = "_stream" if self.parsed_options.stream else ""
        return f"/generate{stream_suffix}"

    def format_payload(self, prompt, max_tokens, images):
        assert isinstance(prompt, str), "prompt must be a string"
        assert images is None, "images are not supported"
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

    def parse_output_json(self, data):
        if "token" in data:
            # streaming chunk
            return ChunkMetadata(
                text=data["token"]["text"],
                logprob_tokens=1,
                completion_tokens=None,
                prompt_tokens=None,
                cached_tokens=None,
            )
        else:
            # non-streaming response
            return ChunkMetadata(
                text=data["generated_text"],
                logprob_tokens=(len(data["details"]["tokens"]) if "details" in data else None),
                completion_tokens=(data["details"]["generated_tokens"] if "details" in data else None),
                prompt_tokens=None,
                cached_tokens=None,
            )


PROVIDER_CLASS_MAP = {
    "fireworks": FireworksProvider,
    "vllm": VllmProvider,
    "sglang": VllmProvider,
    "openai": OpenAIProvider,
    "together": TogetherProvider,
    "tgi": TgiProvider,
}


def _load_curl_like_data(text):
    """
    Either use the passed string or load from a file if the string is `@filename`
    """
    if text.startswith("@"):
        try:
            if text.endswith(".jsonl"):
                with open(text[1:], "r") as f:
                    return [json.loads(line) for line in f]
            else:
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
            raise ValueError("Argument --model or --provider was not specified and /v1/models failed") from e

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
                raise ValueError(f"Model {self.model} not found in /v1/models. Specify --provider explicitly")
            if owned_by in PROVIDER_CLASS_MAP:
                self.provider = owned_by
            else:
                raise ValueError(f"Can't detect provider, specify it explicitly with --provider, owned_by={owned_by}")

    def _on_start(self):
        self.client.headers["Content-Type"] = "application/json"
        if self.environment.parsed_options.api_key:
            self.client.headers["Authorization"] = "Bearer " + self.environment.parsed_options.api_key
        if self.environment.parsed_options.header:
            for header in self.environment.parsed_options.header:
                key, val = header.split(":", 1)
                self.client.headers[key] = val
        self._guess_provider()
        print(f" Provider {self.provider} using model {self.model} ".center(80, "*"))
        self.provider_formatter = PROVIDER_CLASS_MAP[self.provider](self.model, self.environment.parsed_options)

        self.stream = self.environment.parsed_options.stream
        self.ramping_pacer = None  # Will be set if --ramping-time is used

        image_resolutions = self.environment.parsed_options.prompt_images_with_resolutions
        self.prompt_images = None
        if image_resolutions:
            if not self.environment.parsed_options.chat:
                # Using regular /completions endpoint, each model has it's own image placeholder
                # e.g., <|image|> for Phi, <|image_pad|> for Qwen, <image> for Llava
                # So using /completions endpoint requires a bit more work to support this
                raise AssertionError("--prompt-images-with-resolutions is only supported with --chat mode.")
            self.prompt_images = [self._create_base64_image(width, height) for width, height in image_resolutions]

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
            "completion_tokens": str(self.max_tokens_sampler),
            "stream": self.stream,
            "temperature": self.temperature,
            "logprobs": self.environment.parsed_options.logprobs,
        }

        if self.environment.parsed_options.top_k is not None:
            logging_params["top_k"] = self.environment.parsed_options.top_k

        InitTracker.notify_init(self.environment, logging_params)

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
        elif self.environment.parsed_options.ramping_time is not None:
            if self.environment.parsed_options.burst:
                raise ValueError("Burst and ramping modes are mutually exclusive")
            max_users = self.environment.parsed_options.num_users
            if max_users is None:
                raise ValueError("--ramping-time requires -u/--users to be specified")
            min_users = 1.0
            ramping_time = self.environment.parsed_options.ramping_time
            # Compute pct such that: max_users = min_users * (1 + pct/100)^ramping_time
            # pct = 100 * ((max_users/min_users)^(1/ramping_time) - 1)
            ramping_pct = 100 * ((max_users / min_users) ** (1 / ramping_time) - 1)
            self.ramping_pacer = RampingPacer.instance(
                min_users,
                max_users,
                ramping_pct,
                self.environment.parsed_options.ramping_warmup_time,
            )
            # Use custom wait that blocks until below target concurrency
            self.wait_time = lambda: 0
            self._wait_for_ramping_capacity()
        elif self.environment.parsed_options.burst:
            self.wait_time = partial(constant_pacing(self.environment.parsed_options.burst), self)
        else:
            # introduce initial delay to avoid all users hitting the service at the same time
            time.sleep(random.random())

        self.first_done = False

        dataset = DatasetHolder.get_instance(self.environment.parsed_options)
        self.dataset = iter(dataset)

    def _create_base64_image(self, width, height):
        """Create a random RGB image with the given dimensions and return as base64 data URI."""
        img = Image.new("RGB", (width, height))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _get_input(self):
        prompt, prompt_tokens = next(self.dataset)

        if self.prompt_images:
            images = self.prompt_images
            prompt_images_positioning = self.environment.parsed_options.prompt_images_positioning
            prompt = self.insert_image_placeholders(prompt, len(images), prompt_images_positioning)
        else:
            images = None

        return prompt, prompt_tokens, images

    def _wait_for_ramping_capacity(self):
        """Block until ramping pacer allows a new request (below target concurrency)."""
        while not self.ramping_pacer.can_start():
            gevent.sleep(0.1)

    def insert_image_placeholders(self, prompt, num_images, prompt_images_positioning):
        if num_images <= 0:
            return prompt

        prompt_length = len(prompt)
        if prompt_length == 0:
            return PROMPT_CHAT_IMAGE_PLACEHOLDER * num_images

        if prompt_images_positioning == "space-evenly":
            """
            Insert <image> placeholders evenly throughout the prompt.
            E.g., for 3 images, a prompt "abcdefgh" is changed to "ab<image>cd<image>ef<image>gh"

            Images are spaced out evenly based on on character length.
            This may result in a few extra tokens if the image tags are placed in the middle of tokens.
            But shouldn't affect results meaningfully.
            """
            # we need num_images + 1 segments to place between <image> tags
            segment_length = prompt_length / (num_images + 1)
            result = ""
            for i in range(num_images):
                # Move a sliding window of segment_length across the prompt
                # Truncating to ensure all segments are non-overlapping
                # If segment_end is truncated, that character will be included in the next segment
                segment_start = int(i * segment_length)
                segment_end = int((i + 1) * segment_length)
                result += prompt[segment_start:segment_end] + PROMPT_CHAT_IMAGE_PLACEHOLDER

            # Final segment
            result += prompt[int(num_images * segment_length) :]

            return result
        elif prompt_images_positioning == "end":
            return prompt + PROMPT_CHAT_IMAGE_PLACEHOLDER * num_images
        else:
            raise ValueError(f"Invalid prompt images positioning: {prompt_images_positioning}")

    @task
    def generate_text(self):
        # Wait for capacity and track concurrent requests for ramping pacer
        if self.ramping_pacer:
            self._wait_for_ramping_capacity()
            self.ramping_pacer.request_start()
        try:
            self._do_generate_text()
        finally:
            if self.ramping_pacer:
                self.ramping_pacer.request_end()

    def _do_generate_text(self):
        max_tokens = self.max_tokens_sampler.sample()
        prompt, prompt_tokens, images = self._get_input()
        data = self.provider_formatter.format_payload(prompt, max_tokens, images)
        if self.environment.parsed_options.show_request:
            print("--- Request payload ---")
            print(json.dumps(data, indent=2))
            print("---")
        t_start = time.perf_counter()

        with self.client.post(
            self.provider_formatter.get_url(),
            data=json.dumps(data),
            stream=True,
            catch_response=True,
        ) as response:
            combined_text = ""
            done_empty_chunk = False
            done = False
            completion_tokens = None
            total_logprob_tokens = None
            cached_tokens = None
            perf_metrics = None  # Capture perf_metrics from response body (streaming)
            try:
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Error in response: {response.text}") from e
            t_first_token = None
            for chunk in response.iter_lines(delimiter=b"\n\n"):
                if len(chunk) == 0:
                    continue  # come providers send empty lines between data chunks
                if done:
                    if chunk != b"data: [DONE]":
                        print(f"WARNING: Received more chunks after [DONE]: {chunk}")
                try:
                    now = time.perf_counter()
                    if self.provider_formatter.parsed_options.embeddings:
                        t_first_token = now
                        if self.environment.parsed_options.show_response:
                            out = self.provider_formatter.parse_output_json(orjson.loads(chunk))
                            combined_text = out.text
                        break
                    if self.stream:
                        assert chunk.startswith(b"data:"), f"Unexpected chunk not starting with 'data': {chunk}"
                        chunk = chunk[len(b"data:") :]
                        if chunk.strip() == b"[DONE]":
                            done = True
                            continue
                    if done_empty_chunk:
                        print(f"WARNING: Received more chunks after the trailing last chunk: {chunk}")
                    data = orjson.loads(chunk)
                    # Capture perf_metrics if present (usually in final usage chunk)
                    if data.get("perf_metrics"):
                        perf_metrics = data["perf_metrics"]
                    if not data.get("choices"):
                        usage = data.get("usage")
                        if usage and usage.get("completion_tokens"):
                            completion_tokens = usage["completion_tokens"]
                        if usage and usage.get("prompt_tokens"):
                            prompt_tokens = usage["prompt_tokens"]
                        if usage:
                            prompt_tokens_details = usage.get("prompt_tokens_details") or {}
                            if prompt_tokens_details.get("cached_tokens") is not None:
                                cached_tokens = prompt_tokens_details["cached_tokens"]
                        done_empty_chunk = True
                        continue
                    out = self.provider_formatter.parse_output_json(data)
                    if out.completion_tokens:
                        completion_tokens = out.completion_tokens
                    if out.prompt_tokens:
                        prompt_tokens = out.prompt_tokens
                    if out.cached_tokens is not None:
                        cached_tokens = out.cached_tokens
                    combined_text += out.text

                    # some providers (SGLang) send an empty chunk first skewing the TTFT
                    if combined_text and t_first_token is None:
                        t_first_token = now

                    if out.logprob_tokens:
                        total_logprob_tokens = (total_logprob_tokens or 0) + out.logprob_tokens
                except Exception as e:
                    print(f"Failed to parse response: {chunk} with error {repr(e)}")
                    response.failure(e)
                    return
            if t_first_token is None:
                if max_tokens == 0:
                    t_first_token = time.perf_counter()
                else:
                    response.failure(Exception("empty response received"))
                    return

            if (
                (total_logprob_tokens is not None)
                and (completion_tokens is not None)
                and total_logprob_tokens != completion_tokens
            ):
                print(f"WARNING: completion_tokens {completion_tokens} != logprob_tokens {total_logprob_tokens}")
            if total_logprob_tokens is not None:
                num_tokens = total_logprob_tokens
            else:
                num_tokens = completion_tokens

            num_tokens = num_tokens or 0
            num_chars = len(combined_text)
            now = time.perf_counter()
            dur_total = now - t_start
            dur_generation = now - t_first_token
            dur_first_token = t_first_token - t_start

            if not self.provider_formatter.parsed_options.embeddings:
                prompt_tokens = prompt_tokens or self.prompt_tokenizer_tokens

            token_parts = []
            if prompt_tokens:
                token_parts.append(f"{prompt_tokens} prompt")
            if cached_tokens is not None:
                token_parts.append(f"{cached_tokens} cached")
            token_parts.append(f"{num_tokens} completion")
            token_str = ", ".join(token_parts)

            print(
                f"Response received: total {dur_total*1000:.2f} ms, first token {dur_first_token*1000:.2f} ms, {num_chars} chars, {token_str}"
            )
            if self.environment.parsed_options.show_response:
                print("---")
                print(combined_text)
                print("---")
            if num_chars:
                add_custom_metric("latency_per_char", dur_generation / num_chars * 1000, num_chars)
            if self.stream:
                add_custom_metric("time_to_first_token", dur_first_token * 1000)
            add_custom_metric("total_latency", dur_total * 1000)
            if num_tokens:
                if num_tokens != max_tokens:
                    print(f"WARNING: wrong number of tokens: {num_tokens}, expected {max_tokens}")
                add_custom_metric("completion_tokens", num_tokens)
                add_custom_metric("latency_per_token", dur_generation / num_tokens * 1000, num_tokens)
                add_custom_metric(
                    "overall_latency_per_token",
                    dur_total / num_tokens * 1000,
                    num_tokens,
                )

            if not self.provider_formatter.parsed_options.embeddings:
                if prompt_tokens:
                    add_custom_metric("prompt_tokens", prompt_tokens)
                if cached_tokens is not None:
                    add_custom_metric("cached_tokens", cached_tokens)

            # Allow provider to process response (e.g., for custom metrics)
            self.provider_formatter.post_response_hook(response.headers, num_tokens, perf_metrics)

            # Mark response as success (required when using catch_response=True)
            response.success()

            if not self.first_done:
                self.first_done = True
                InitTracker.notify_first_request()

            # Notify request completion and check if we should stop
            InitTracker.notify_request_complete()


def parse_resolution(res_str):
    """Parse a resolution string like '3084x1080' into a tuple of integers (width, height)."""
    try:
        width, height = map(int, res_str.split("x"))
        return (width, height)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Invalid resolution format: {res_str}. Expected format: WIDTHxHEIGHT (e.g. 1024x1024)"
        )


@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CLASS_MAP.keys()),
        type=str,
        help="Which flavor of API to use. If not specified, we'll try to guess based on the URL and /v1/models output",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        env_var="DATASET",
        type=str,
        help="Either 'limericks', 'code' or a path to a JSONL file",
        default="limericks",
    )
    parser.add_argument(
        "--dataset-shuffle-seed",
        type=int,
        default=None,
        help="Random seed for shuffling the dataset. If provided, the dataset will be shuffled with this seed before use. Useful for reproducible sampling.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit the dataset to the first N items after shuffling (if shuffle seed is provided). Useful for sampling a subset of a large dataset.",
    )
    parser.add_argument(
        "-m",
        "--model",
        env_var="MODEL",
        type=str,
        help="The model to use for generating text. If not specified we will pick the first model from the service as returned by /v1/models",
    )
    parser.add_argument(
        "--tokenizer",
        env_var="TOKENIZER",
        type=str,
        help="Specify HF tokenizer to use for validating the output of the model. It's optional, we're going to rely on 'usage' or 'logprobs' field to get token count information",
    )
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use /v1/chat/completions API",
    )
    parser.add_argument(
        "--embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /v1/embeddings API",
    )
    parser.add_argument(
        "--return-logits",
        type=int,
        nargs="*",
        default=None,
        help="For embeddings: return per-token or per-class logits. Provide specific token/class indices, or empty list for all. Only works with certain models.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For embeddings: apply L2 normalization to activations when return_logits is None, or softmax to selected logits when return_logits is provided.",
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
        "--prompt-images-with-resolutions",
        type=parse_resolution,
        nargs="+",
        default=[],
        help="Images to add to the prompt for vision models, defined by their resolutions in format WIDTHxHEIGHT. "
        'For example, "--prompt-images-with-resolutions 3084x1080 1024x1024" will insert 2 images '
        "(3084 width x 1080 height and 1024 width x 1024 height) into the prompt. "
        "Images will be spaced out evenly across the prompt."
        "Only supported with --chat mode.",
    )
    parser.add_argument(
        "--prompt-images-positioning",
        type=str,
        choices=["space-evenly", "end"],
        default="space-evenly",
        help="How to position the images in the prompt. "
        "space-evenly: images are spaced out evenly across the prompt. E.g., 3 images in 'abcdefgh' is 'ab<image>cd<image>ef<image>gh'"
        "end: images are added to the end of the prompt. E.g., 3 images in 'abcdefgh' is 'abcdefgh<image><image><image>'"
        "Only relevant with --prompt-images-with-resolutions.",
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
        "--top-k",
        env_var="TOP_K",
        type=int,
        default=None,
        help="Specifies the top-k sampling parameter.",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        "--ramping-time",
        type=float,
        default=None,
        help="Enables ramping mode. Time in seconds to ramp from 1 user to -u/--users. "
        "Example: -u 50 --ramping-time 60 ramps from 1 to 50 users over 60 seconds.",
    )
    parser.add_argument(
        "--ramping-warmup-time",
        type=float,
        default=30.0,
        help="Time in seconds to hold at 1 user before starting to ramp up. Default 30 seconds.",
    )
    parser.add_argument(
        "--show-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the result of each generation",
    )
    parser.add_argument(
        "--show-request",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the request payload for each request",
    )
    parser.add_argument(
        "-pcml",
        "--prompt-cache-max-len",
        env_var="PROMPT_CACHE_MAX_LEN",
        type=int,
        default=0,
        help="Maximum length of the prompt cache to use. Defaults to 0 (no caching).",
    )
    parser.add_argument(
        "--prompt-cache-max-pct",
        env_var="PROMPT_CACHE_MAX_PCT",
        type=float,
        default=None,
        help="(Fireworks only) Maximum percentage of prompt tokens to use for prompt cache (0-100). "
        "Passed as prompt_cache_max_pct in the API request. Ignored for other providers.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Arbitrary headers to add to the inference request. Can be used multiple times. For example, --header header1:value1 --header header2:value2",
    )
    parser.add_argument(
        "-n",
        "--n",
        default=1,
        type=int,
        help="How many sequences to generate (makes sense to use with non-zero temperature).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop the test after the specified number of successful requests complete. "
        "Useful for running a fixed number of requests regardless of time or dataset size.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to use for the dataset. If not specified, a default prompt will be used.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Set the reasoning_effort parameter for the API request (e.g., 'none', 'low', 'medium', 'high'). "
        "If not specified and using a JSONL dataset, will use the value from the dataset if present.",
    )
    parser.add_argument(
        "--clear-assistant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If the last message in the messages array is from the assistant role, remove it. "
        "This allows the model to generate new content instead of continuing from existing assistant content.",
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
        entries["concurrency"] = f"QPS {environment.parsed_options.qps} {environment.parsed_options.qps_distribution}"
    else:
        entries["concurrency"] = InitTracker.users
    for metric_name in [
        "time_to_first_token",
        "latency_per_token",
        "overall_latency_per_token",
        "total_latency",
        "completion_tokens",
        "prompt_tokens",
    ]:
        entries[metric_name] = environment.stats.entries[(metric_name, "METRIC")].avg_response_time
    if ("cached_tokens", "METRIC") in environment.stats.entries:
        entries["cached_tokens"] = environment.stats.entries[("cached_tokens", "METRIC")].avg_response_time
    if not environment.parsed_options.stream:
        # if there's no streaming these metrics are meaningless
        entries["time_to_first_token"] = ""
        entries["latency_per_token"] = ""
    entries["num_requests"] = total_latency.num_requests
    entries["qps"] = total_latency.total_rps
    percentile_to_report = [50, 90, 95, 99, 99.9]
    percentile_metrics = ["time_to_first_token", "total_latency"]
    for percentile_metric in percentile_metrics:
        metrics = environment.stats.entries[percentile_metric, "METRIC"]
        for percentile in percentile_to_report:
            name = f"P{percentile}_{percentile_metric}"
            entries[name] = metrics.get_response_time_percentile(percentile / 100)

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
