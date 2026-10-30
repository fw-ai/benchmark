import abc
import argparse
from array import array
import csv
from dataclasses import dataclass
from functools import lru_cache, partial
import importlib.util
import dataclasses
import gzip
import json
import statistics
import logging
import math
import os
import random
import secrets
import sys
import threading
import traceback
from typing import Any, Optional
from locust import HttpUser, task, events, constant_pacing
import copy
import json
import time
import uuid
import orjson
import requests
import urllib3
import urllib3.connection
import urllib3.connectionpool
import base64
import bisect
import hashlib
import io
import itertools
from PIL import Image
import transformers
import re
import gevent
import gevent.event
import gevent.pool
from locust.clients import LocustHttpAdapter
from locust.runners import MasterRunner
from locust.util.timespan import parse_timespan as _locust_parse_timespan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import locust_plugins
except ImportError:
    logger.warning("locust-plugins is not installed, Grafana won't work")


def _install_transformers_tokenizer_compat_shim():
    """Patch moved helpers required by older remote tokenizer code."""
    try:
        import transformers.models.gpt2.tokenization_gpt2 as gpt2_module
        from transformers.convert_slow_tokenizer import bytes_to_unicode
    except ImportError:
        return

    if not hasattr(gpt2_module, "bytes_to_unicode"):
        gpt2_module.bytes_to_unicode = bytes_to_unicode


DEFAULT_TOKENIZER = "NousResearch/Meta-Llama-3.1-8B-Instruct"
QPS_CONNECTION_POOL_SIZE = 1024
LATE_SEND_WARNING_SECONDS = 0.05
DEFAULT_MAX_LATENESS_SECONDS = 0.5
_MIN_MODULATED_BASE_CV2 = 0.05
_DEFAULT_RATE_BLOCK_WINDOWS = 30


def _load_auto_tokenizer(tokenizer_path: str):
    _install_transformers_tokenizer_compat_shim()
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _tokenizer_asset_path(tokenizer_path: str, filename: str) -> str:
    if os.path.isdir(tokenizer_path):
        path = os.path.join(tokenizer_path, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {filename} in local tokenizer directory {tokenizer_path}")
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=tokenizer_path, filename=filename)


def _read_config_json(tokenizer_path: str) -> dict[str, Any]:
    config_path = _tokenizer_asset_path(tokenizer_path, "config.json")
    with open(config_path) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def resolve_model_type(tokenizer_path: str) -> str:
    try:
        config = transformers.AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
        text_config = config.get_text_config()
        return getattr(text_config, "model_type", None) or getattr(config, "model_type", "")
    except ValueError:
        config_json = _read_config_json(tokenizer_path)
        text_config = config_json.get("text_config") or {}
        return text_config.get("model_type") or config_json.get("model_type", "")


@lru_cache(maxsize=None)
def load_dsv4_encode_messages(tokenizer_path: str) -> Any:
    path = _tokenizer_asset_path(tokenizer_path, "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("hf_dsv4_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DSV4 encoding module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def _normalize_token_ids(obj: Any) -> list[int]:
    ids = getattr(obj, "ids", None)
    if ids is not None:
        return [int(t) for t in ids]
    input_ids = getattr(obj, "input_ids", None)
    if input_ids is not None:
        flat = input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
        return [int(t) for t in flat]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        flat = tolist()
        if flat and isinstance(flat[0], list):
            flat = flat[0]
        return [int(t) for t in flat]
    return [int(t) for t in obj]


def chat_template_token_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_path: str,
    content: str = "",
) -> list[int]:
    """Token ids for a single user message rendered with the chat template + generation prompt."""
    model_type = resolve_model_type(tokenizer_path)
    if model_type == "deepseek_v4":
        encode_messages = load_dsv4_encode_messages(tokenizer_path)
        prompt = encode_messages([{"role": "user", "content": content}], thinking_mode="chat")
        return _normalize_token_ids(tokenizer.encode(prompt, add_special_tokens=False))
    return _normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def chat_messages_token_ids(tokenizer, tokenizer_path: str, messages: list[dict]) -> list[int]:
    """Token ids for a whole conversation rendered with the chat template + generation prompt."""
    if resolve_model_type(tokenizer_path) == "deepseek_v4":
        prompt = load_dsv4_encode_messages(tokenizer_path)(messages, thinking_mode="chat")
        return _normalize_token_ids(tokenizer.encode(prompt, add_special_tokens=False))
    return _normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def chat_messages_overhead(tokenizer, tokenizer_path: str, num_messages: int) -> int:
    """Template tokens of a conversation of `num_messages` empty messages (user, assistant, ..., user)."""
    roles = ["user" if i % 2 == 0 else "assistant" for i in range(num_messages)]
    return len(chat_messages_token_ids(tokenizer, tokenizer_path, [{"role": r, "content": ""} for r in roles]))


def empty_chat_template_token_ids(
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_path: str,
) -> list[int]:
    """Token ids for chat template overhead (empty user message + generation prompt)."""
    return chat_template_token_ids(tokenizer, tokenizer_path)


def chat_template_head_tokens(tokenizer, tokenizer_path: str) -> int:
    """Number of template tokens rendered before the user content (shared by every chat request)."""
    empty = chat_template_token_ids(tokenizer, tokenizer_path)
    filled = chat_template_token_ids(tokenizer, tokenizer_path, "Shared context follows.")
    head = 0
    while head < min(len(empty), len(filled)) and empty[head] == filled[head]:
        head += 1
    return head


_PROMPT_TOKENS_RE = re.compile(rb'"prompt_tokens"\s*:\s*(\d+)')


def extract_prompt_tokens(body: bytes) -> Optional[int]:
    """Read usage.prompt_tokens out of a response body without fully decoding it.

    A batched embeddings response is almost entirely the float arrays under `data`.
    Decoding those into Python objects just to read `usage` costs orders of magnitude
    more than scanning for the field, and that cost lands on the load generator.
    Falls back to a real parse if the scan misses.
    """
    match = _PROMPT_TOKENS_RE.search(body)
    if match:
        return int(match.group(1))
    try:
        return (orjson.loads(body).get("usage") or {}).get("prompt_tokens")
    except orjson.JSONDecodeError:
        return None


def add_custom_metric(name, value, length_value=0):
    events.request.fire(
        request_type="METRIC",
        name=name,
        response_time=value,
        response_length=length_value,
        exception=None,
        context=None,
    )


def track_performance_metrics(
    response_time: float, token_count: int, char_count: int, prompt_tokens: int = None, ttft: float = None
):
    """Track comprehensive performance metrics for parity with Customers load_test.py."""
    metrics = {
        "response_time": response_time,
        "token_count": token_count,
        "char_count": char_count,
        "tokens_per_second": token_count / response_time if response_time > 0 else 0,
        "chars_per_second": char_count / response_time if response_time > 0 else 0,
    }

    if prompt_tokens:
        metrics["total_tokens"] = prompt_tokens + token_count
        metrics["efficiency"] = token_count / (prompt_tokens + token_count) if (prompt_tokens + token_count) > 0 else 0

    if ttft:
        metrics["ttft"] = ttft
        metrics["generation_time"] = response_time - ttft
        metrics["generation_tokens_per_second"] = (
            token_count / (response_time - ttft) if (response_time - ttft) > 0 else 0
        )

    for metric_name, metric_value in metrics.items():
        add_custom_metric(metric_name, metric_value, token_count)


PROMPT_PREFIX_TOKEN = "Pad "  # exactly one token
PROMPT_CHAT_IMAGE_PLACEHOLDER = "<image>"


class TranslationDataset:
    def __init__(
        self,
        path: str,
        prompt: str,
        tokenizer,
        tokenizer_path: str,
        chat: bool,
        num_tokens: int,
        common_tokens: int,
    ):
        self._tokenizer = tokenizer
        self._tokenizer_path = tokenizer_path
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
            empty_template_tokens = empty_chat_template_token_ids(self._tokenizer, self._tokenizer_path)
            self._prefix_suffix_tokens += len(empty_template_tokens)

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
            logger.info(f"Shuffled dataset with seed {self.shuffle_seed}")

        # Limit dataset size if specified
        if self.dataset_limit is not None:
            data = data[: self.dataset_limit]
            logger.info(f"Limited dataset to first {len(data)} items")

        # Yield all items
        for item in data:
            yield item


SYSTEMONE_NONCE_CHARS = 32  # uuid4 hex length, so unique and fixed states size identically


def systemone_option_key(index: int) -> str:
    """Spreadsheet-style criteria key: option_a .. option_z, option_aa, ..."""
    suffix = ""
    index += 1
    while index > 0:
        index, rem = divmod(index - 1, 26)
        suffix = chr(ord("a") + rem) + suffix
    return f"option_{suffix}"


def validate_systemone_response(data: Any, question_ids: list[str]) -> Optional[str]:
    """Return an error message if a /v1/systemone response doesn't answer exactly the asked questions."""
    if not isinstance(data, dict):
        return f"response is not a JSON object: {type(data).__name__}"
    answers = data.get("answers")
    if not isinstance(answers, dict):
        return "response has no 'answers' object"
    if set(answers.keys()) != set(question_ids):
        missing = sorted(set(question_ids) - set(answers.keys()))
        extra = sorted(set(answers.keys()) - set(question_ids))
        return f"answers keys mismatch: missing {missing}, unexpected {extra}"
    return None


class SystemOneDataset:
    """Synthetic /v1/systemone requests: a ~state_tokens JSON state plus N choice questions, seeded per worker."""

    def __init__(
        self,
        path: str,
        tokenizer,
        num_questions: int,
        num_options: int,
        state_tokens: int,
        question_tokens: int,
        unique_state: bool,
        seed: int,
    ):
        if num_questions < 1:
            raise ValueError("--systemone-num-questions must be >= 1")
        if num_options < 1:
            raise ValueError("--systemone-num-options must be >= 1")
        self._tokenizer = tokenizer
        self._unique_state = unique_state
        self._rng = random.Random(seed)

        with open(path, "r") as f:
            self._texts = [t.strip() for t in f.read().split("\n\n") if t.strip()]
        self._rng.shuffle(self._texts)
        self._text_tokens = [self.count_tokens(json.dumps(t)) for t in self._texts]

        self._fixed_nonce = "%032x" % self._rng.getrandbits(128)
        self._sections = self._build_sections(state_tokens)
        self.question_ids = [f"q{i}" for i in range(num_questions)]
        self._questions = {
            qid: {
                "type": "choice",
                "instructions": self._build_instructions(i, question_tokens),
                "criteria": {
                    systemone_option_key(j): f"The state best matches profile {j + 1} of {num_options}."
                    for j in range(num_options)
                },
            }
            for i, qid in enumerate(self.question_ids)
        }
        self.state_tokens = self.count_tokens(self._serialize_state(self._fixed_nonce))
        self.question_tokens = [self.count_tokens(q["instructions"]) for q in self._questions.values()]

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        # ~4 chars/token heuristic when no tokenizer is given
        return (len(text) + 3) // 4

    def _build_sections(self, state_tokens: int) -> dict[str, str]:
        # Nonce key + braces overhead is counted up front so the total lands near the target
        used = self.count_tokens(json.dumps({"nonce": "0" * SYSTEMONE_NONCE_CHARS}))
        sections = {}
        idx = 0
        while True:
            if used >= state_tokens:
                # Per-piece estimates drift from the joined count, so confirm against the real serialization
                used = self.count_tokens(json.dumps({"nonce": "0" * SYSTEMONE_NONCE_CHARS, **sections}))
                if used >= state_tokens:
                    return sections
            text_idx = idx % len(self._texts)
            key = f"section_{idx}"
            sections[key] = self._texts[text_idx]
            used += self._text_tokens[text_idx] + self.count_tokens(f', "{key}": ')
            idx += 1

    def _build_instructions(self, index: int, question_tokens: int) -> str:
        text = f"Which option best describes the state for check {index}?"
        filler = " ".join(self._rng.sample(self._texts, min(len(self._texts), 8))).replace("\n", " ")
        while self.count_tokens(text) < question_tokens:
            text += " " + filler
        if self._tokenizer is not None:
            ids = self._tokenizer.encode(text, add_special_tokens=False)[:question_tokens]
            return self._tokenizer.decode(ids)
        return text[: question_tokens * 4]

    def _serialize_state(self, nonce: str) -> str:
        return json.dumps(self.make_state(nonce))

    def make_state(self, nonce: str) -> dict[str, str]:
        return {"nonce": nonce, **self._sections}

    def __next__(self):
        nonce = uuid.uuid4().hex if self._unique_state else self._fixed_nonce
        return {"state": self.make_state(nonce), "questions": self._questions}, self.state_tokens

    def __iter__(self):
        return self


SHAPE_CLASS_FIELDS = {
    "name",
    "weight",
    "prompt_tokens",
    "cached_prompt_tokens",
    "generated_tokens",
    "prefix_pool_size",
    "cache_class",
    "prompt_band",
    "uncached_tokens_quantiles",
    "generated_tokens_quantiles",
    "prefix_popularity",
}
_SHAPE_CLASS_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SHAPE_NAME_INVALID_RE = re.compile(r"[^A-Za-z0-9_-]+")
# Every unique tail starts with this, so prefixes are checked against it once to keep token counts additive
SHAPE_MIX_TAIL_START = "\n\nRequest"
# Room left after the shared prefix for the per-request nonce that keeps every prompt unique
SHAPE_MIX_TAIL_RESERVE_TOKENS = 32
_UNBUILT = 0xFFFFFFFF
_FIT_ATTEMPTS = 4
SHAPE_MIX_TAIL_VARIANTS = 16
# Filler variants per tail length in --replay-schedule layouts (chosen by row)
REPLAY_FILLER_VARIANTS = 8
_PREFIX_OFFSET_ATTEMPTS = 32
_PREFIX_BUILD_BATCH_TOKENS = 1_000_000
# Bump when window search or corpus layout changes, so caches from older code aren't reused
_POOL_CACHE_FORMAT = "shape-mix-pool-v2"


@dataclass(frozen=True)
class ShapeClass:
    name: str
    weight: float
    prompt_tokens: int
    cached_prompt_tokens: int
    generated_tokens: int
    prefix_pool_size: int
    # (quantile, tokens) points, sorted; sampled from with --shape-mix-sample-lengths
    uncached_tokens_quantiles: Optional[tuple] = None
    generated_tokens_quantiles: Optional[tuple] = None
    # How often each prefix of the pool is used: ("zipf", s) or ("weights", (w0, w1, ...)); None is uniform
    prefix_popularity: Optional[tuple] = None
    # Largest prompt (tokens) the class's prompt_band allows, widened by bands merged into it; None: no ceiling
    prompt_tokens_max: Optional[int] = None


@dataclass(frozen=True)
class ShapeMix:
    classes: list[ShapeClass]
    seed: int
    arrivals_cv2: Optional[float] = None
    request_seed: Optional[int] = None
    arrivals_dispersion_10s: Optional[float] = None

    def __post_init__(self):
        if self.request_seed is None:
            object.__setattr__(self, "request_seed", self.seed)

    def describe(self) -> str:
        return "; ".join(
            f"{c.name} w={c.weight:g} p={c.prompt_tokens} c={c.cached_prompt_tokens} "
            f"g={c.generated_tokens} n={c.prefix_pool_size}"
            for c in self.classes
        )

    def weighted_mean(self, field: str) -> float:
        total = sum(c.weight for c in self.classes)
        return sum(c.weight * getattr(c, field) for c in self.classes) / total


def _shape_number(raw: dict, key: str, where: str, default: Optional[float] = None) -> float:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"--shape-mix {where}: missing '{key}'")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"--shape-mix {where}: '{key}' must be a number, got {value!r}")
    return value


def _shape_int(raw: dict, key: str, where: str, minimum: int, default: Optional[int] = None) -> int:
    # Profiles carry per-class means, so token counts may be fractional; round half up
    value = math.floor(_shape_number(raw, key, where, default) + 0.5)
    if value < minimum:
        raise ValueError(f"--shape-mix {where}: '{key}' must be >= {minimum} after rounding, got {raw.get(key)!r}")
    return value


def _quantile_level(key, where: str, field: str) -> float:
    text = str(key).strip().lower()
    try:
        level = float(text[1:]) / 100 if text.startswith("p") else float(text)
    except ValueError:
        raise ValueError(f"--shape-mix {where}: '{field}' keys must be like 'p50' or 0.5, got {key!r}")
    if not 0 <= level <= 1:
        raise ValueError(f"--shape-mix {where}: '{field}' quantile {key!r} is outside [0, 1]")
    return level


def _parse_quantiles(raw: dict, field: str, where: str, minimum: int) -> Optional[tuple]:
    """Quantiles as {"p10": v, "p50": v, ...} (or 0.1-style keys) or [[q, v], ...]; values may be fractional."""
    value = raw.get(field)
    if value is None:
        return None
    items = value.items() if isinstance(value, dict) else value
    try:
        points = sorted((_quantile_level(q, where, field), float(v)) for q, v in items)
    except (TypeError, ValueError) as e:
        if isinstance(e, ValueError) and str(e).startswith("--shape-mix"):
            raise
        raise ValueError(f"--shape-mix {where}: '{field}' must map quantiles to token counts, got {value!r}")
    if not points:
        return None
    values = [v for _, v in points]
    if any(not math.isfinite(v) or v < minimum for v in values) or values != sorted(values):
        raise ValueError(
            f"--shape-mix {where}: '{field}' must be finite, >= {minimum} and non-decreasing, got {value!r}"
        )
    return tuple(points)


def _parse_popularity(raw: dict, where: str, pool_size: int) -> Optional[tuple]:
    """prefix_popularity: {"zipf": s} (prefix j weighted 1 / (j + 1)^s) or a list of prefix_pool_size weights."""
    value = raw.get("prefix_popularity")
    if value is None:
        return None
    if isinstance(value, dict) and set(value) == {"zipf"}:
        exponent = _shape_number(value, "zipf", f"{where} prefix_popularity")
        if exponent < 0:
            raise ValueError(f"--shape-mix {where}: prefix_popularity zipf must be >= 0, got {exponent}")
        return ("zipf", float(exponent))
    if isinstance(value, list):
        if len(value) != pool_size:
            raise ValueError(
                f"--shape-mix {where}: prefix_popularity has {len(value)} weights for a pool of {pool_size}"
            )
        weights = tuple(_shape_number({"w": w}, "w", f"{where} prefix_popularity") for w in value)
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"--shape-mix {where}: prefix_popularity weights must be >= 0 with a positive sum")
        return ("weights", weights)
    raise ValueError(
        f'--shape-mix {where}: prefix_popularity must be {{"zipf": s}} or a list of weights, got {value!r}'
    )


def quantile_value(points: tuple, level: float) -> float:
    """Piecewise-linear inverse CDF through (quantile, value) points, flat beyond the outermost points."""
    if level <= points[0][0]:
        return points[0][1]
    for (q0, v0), (q1, v1) in zip(points, points[1:]):
        if level <= q1:
            return v0 if q1 == q0 else v0 + (v1 - v0) * (level - q0) / (q1 - q0)
    return points[-1][1]


def quantile_average(points: tuple, low: float, high: float) -> float:
    """Exact mean of quantile_value over levels [low, high] (it is linear between knots, flat outside them)."""
    if high <= low:
        return quantile_value(points, low)
    cuts = [low] + [q for q, _ in points if low < q < high] + [high]
    area = sum((b - a) * (quantile_value(points, a) + quantile_value(points, b)) / 2 for a, b in zip(cuts, cuts[1:]))
    return area / (high - low)


class LengthQuantiles:
    """A length distribution through a profile's (quantile, value) points, with every given quantile kept.

    Between points, the value follows log(value) linear in the normal z of the quantile (blended toward linear in
    value by `alpha`); past the outermost points it is extrapolated (the lower tail with the first segment's slope,
    the upper with `tail_slope`), capped at p100 when given, else at 4x the top point, and at `cap` (the class's
    prompt band or the model's context) when lower. `fit_mean` moves only what the quantiles don't pin (the upper
    tail's slope, then the blend, then a bend of each segment toward its lower point) to hit a mean; scaling every
    value is the last resort, with a warning.
    """

    _GRID = 4001
    _Z = 8.5
    _TAIL_CAP = 4.0

    def __init__(self, points: tuple, cap: Optional[float] = None):
        normal = statistics.NormalDist()
        self.low = max(1.0, next((v for q, v in points if q <= 0.0), 1.0))
        interior = [(q, max(1.0, v)) for q, v in points if 0.0 < q < 1.0]
        top = interior[-1][1] if interior else max(v for _, v in points)
        self.high = next((v for q, v in points if q >= 1.0), self._TAIL_CAP * top)
        if cap is not None:
            # Never below the given quantiles: those are kept as recorded
            self.high = min(self.high, max(cap, top, self.low))
        self.ceiling = self.high if cap is not None else math.inf  # also holds when every value is scaled
        self.z = [normal.inv_cdf(q) for q, _ in interior]
        self.y = [math.log(v) for _, v in interior]
        self.v = [v for _, v in interior]
        self.ok = len(interior) >= 2 and all(b > a for a, b in zip(self.z, self.z[1:]))
        self.low_slope = self.base_slope = self.tail_slope = 0.0
        self.alpha = 0.0
        # Segment interpolation runs at f ** (1 / straight): 1 is log-linear; toward 0 each segment stays near its
        # lower point until its upper one, which lowers the mean with every given quantile kept
        self.straight = 1.0
        if self.ok:
            self.low_slope = (self.y[1] - self.y[0]) / (self.z[1] - self.z[0])
            self.base_slope = (self.y[-1] - self.y[-2]) / (self.z[-1] - self.z[-2])
            self.tail_slope = self.base_slope
        self.scale = 1.0
        self._points = points
        step = 2 * self._Z / (self._GRID - 1)
        self._grid = [-self._Z + i * step for i in range(self._GRID)]
        self._cdf = [normal.cdf(z) for z in self._grid]
        self._w = [math.exp(-z * z / 2) / math.sqrt(2 * math.pi) * step for z in self._grid]
        self._normal = normal
        self._parts: dict = {}

    def _clamp(self, value: float) -> float:
        return min(self.high, max(self.low, value))

    def _body(self, z: float) -> tuple[float, float]:
        """(log-linear value, linear value) at z inside or below the points."""
        if z <= self.z[0]:
            value = math.exp(self.y[0] + self.low_slope * (z - self.z[0]))
            return value, value
        i = min(bisect.bisect_right(self.z, z) - 1, len(self.z) - 2)
        f = (z - self.z[i]) / (self.z[i + 1] - self.z[i])
        if self.straight < 1.0:
            f = f ** (1 / self.straight)
        return math.exp(self.y[i] + (self.y[i + 1] - self.y[i]) * f), self.v[i] + (self.v[i + 1] - self.v[i]) * f

    def _tail(self, dz: float) -> float:
        return self._clamp(math.exp(min(700.0, self.y[-1] + self.tail_slope * dz)))

    def value(self, level: float) -> float:
        if not self.ok:
            return min(self.ceiling, self.scale * quantile_value(self._points, level))
        level = min(max(level, 1e-12), 1 - 1e-12)
        z = self._normal.inv_cdf(level)
        if z >= self.z[-1]:
            return min(self.ceiling, self.scale * self._tail(z - self.z[-1]))
        loglin, lin = self._body(z)
        return min(self.ceiling, self.scale * self._clamp((1 - self.alpha) * loglin + self.alpha * lin))

    def average(self, low: float, high: float) -> float:
        """Mean value over quantile levels [low, high]."""
        if not self.ok:
            return self.scale * quantile_average(self._points, low, high)
        key = (low, high, self.straight)
        part = self._parts.get(key)
        if part is None:
            loglin_sum = lin_sum = weight = 0.0
            tail = []
            for z, c, w in zip(self._grid, self._cdf, self._w):
                if low <= c <= high:
                    weight += w
                    if z >= self.z[-1]:
                        tail.append((z - self.z[-1], w))
                    else:
                        loglin, lin = self._body(z)
                        loglin_sum += w * self._clamp(loglin)
                        lin_sum += w * self._clamp(lin)
            part = self._parts[key] = (loglin_sum, lin_sum, weight, tail)
        loglin_sum, lin_sum, weight, tail = part
        if not weight:
            return self.value((low + high) / 2)
        body = (1 - self.alpha) * loglin_sum + self.alpha * lin_sum
        return self.scale * (body + sum(w * self._tail(dz) for dz, w in tail)) / weight

    def _solve(self, attr: str, lo: float, hi: float, excess) -> bool:
        """Set `attr` in [lo, hi] so excess(self) (increasing in it) is 0; False if out of reach."""
        setattr(self, attr, lo)
        if excess(self) > 0:
            return False
        setattr(self, attr, hi)
        if excess(self) < 0:
            return False
        for _ in range(50):
            mid = (lo + hi) / 2
            setattr(self, attr, mid)
            lo, hi = (mid, hi) if excess(self) < 0 else (lo, mid)
        setattr(self, attr, (lo + hi) / 2)
        return True

    def fit_mean(self, excess) -> str:
        """Hit excess(self) == 0 (the mean minus its target): upper-tail slope first, then the body blend; else
        scale everything (returns "scale")."""
        if self.ok:
            if self._solve("tail_slope", 0.0, 3 * max(self.base_slope, 0.1), excess):
                return "tail"
            self.tail_slope = 3 * max(self.base_slope, 0.1) if excess(self) < 0 else 0.0
            if self._solve("alpha", 0.0, 1.0, excess):
                return "body"
            if excess(self) > 0:
                # Even a flat tail and log-linear segments overshoot: bend the segments toward their lower points
                self.alpha = 0.0
                if self._solve("straight", 1 / 64, 1.0, excess):
                    self._parts = {k: v for k, v in self._parts.items() if k[2] == self.straight}
                    return "bend"
                self.straight = 1.0
            self.alpha, self.tail_slope = 0.0, self.base_slope
        lo, hi = 0.0, 4.0
        self.scale = hi
        while excess(self) < 0 and hi < 1e6:
            hi *= 2
            self.scale = hi
        self._solve("scale", lo, hi, excess)
        return "scale"


def sampling_strata(n: int) -> list[float]:
    """Level boundaries for uncached-length strata: n equal strata with the top one split into four narrower
    ones (down to the top 0.3% for n = 16), so the top stratum lands at the distribution's top quantile."""
    top = 1 - 1 / n
    return [k / n for k in range(n)] + [top + (1 - top) * f for f in (0.5, 0.8, 0.95)] + [1.0]


# Task 01's prompt bands: whole-prompt token range [low, high)
PROMPT_BAND_BOUNDS = {
    "lt_2k": (0, 2_048),
    "2k_8k": (2_048, 8_192),
    "8k_32k": (8_192, 32_768),
    "32k_128k": (32_768, 131_072),
    "128k_plus": (131_072, None),
}


def _prompt_tokens_max(raw: dict, spec: dict) -> Optional[int]:
    """The class's prompt ceiling: its band's upper edge less one, or the highest band merged into it (task 01
    folds thin classes into a neighbouring band of the same cache class, listed in validity.merged_classes)."""
    band = raw.get("prompt_band")
    if band not in PROMPT_BAND_BOUNDS:
        return None
    highs = [PROMPT_BAND_BOUNDS[band][1]]
    validity = spec.get("validity") if isinstance(spec.get("validity"), dict) else {}
    for merge in validity.get("merged_classes") or ():
        into, source = (merge or {}).get("into") or {}, (merge or {}).get("from") or {}
        if into.get("prompt_band") == band and into.get("cache_class") == raw.get("cache_class"):
            highs.append(PROMPT_BAND_BOUNDS.get(source.get("prompt_band"), (0, None))[1])
    return None if None in highs else max(highs) - 1


def _derived_shape_class_name(raw: dict, index: int) -> str:
    parts = [str(raw[key]) for key in ("cache_class", "prompt_band") if raw.get(key) is not None]
    name = "_".join(_SHAPE_NAME_INVALID_RE.sub("_", p).strip("_") for p in parts).strip("_")
    return name or ("marginal" if "cache_class" in raw or "prompt_band" in raw else f"class{index}")


def parse_shape_mix(text: str, chat_overhead_tokens: int = 0) -> ShapeMix:
    """Parse --shape-mix: inline JSON or @file with a profile `classes` list, `seed`, optional `request_seed`
    (defaults to `seed`) and optional `arrivals.cv2`.

    Other top-level profile fields are ignored, so a traffic profile can be passed as is. Every class needs
    room for the chat template (`chat_overhead_tokens`) plus a unique tail of SHAPE_MIX_TAIL_RESERVE_TOKENS.
    """
    mix = _parse_shape_mix(text)
    minimum = chat_overhead_tokens + SHAPE_MIX_TAIL_RESERVE_TOKENS
    too_short = [f"{c.name} ({c.prompt_tokens})" for c in mix.classes if c.prompt_tokens < minimum]
    if too_short:
        raise ValueError(
            f"--shape-mix: prompt_tokens must be at least {minimum} (chat template overhead {chat_overhead_tokens} "
            f"+ {SHAPE_MIX_TAIL_RESERVE_TOKENS}-token unique tail); too short: {', '.join(too_short)}"
        )
    return mix


@lru_cache(maxsize=None)
def _parse_shape_mix(text: str) -> ShapeMix:
    if text.startswith("@"):
        with open(text[1:], "r") as f:
            text = f.read()
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"--shape-mix must be valid JSON: {e}")
    if not isinstance(spec, dict) or not isinstance(spec.get("classes"), list) or not spec["classes"]:
        raise ValueError("--shape-mix must be a JSON object with a non-empty 'classes' list")
    seed = spec.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"--shape-mix 'seed' must be an integer, got {seed!r}")
    request_seed = spec.get("request_seed")
    if request_seed is not None and (isinstance(request_seed, bool) or not isinstance(request_seed, int)):
        raise ValueError(f"--shape-mix 'request_seed' must be an integer, got {request_seed!r}")
    arrivals_cv2 = None
    arrivals = spec.get("arrivals")
    if isinstance(arrivals, dict) and arrivals.get("cv2") is not None:
        arrivals_cv2 = _shape_number(arrivals, "cv2", "arrivals")
        if arrivals_cv2 <= 0:
            raise ValueError(f"--shape-mix arrivals: 'cv2' must be > 0, got {arrivals_cv2}")
    arrivals_dispersion = None
    if isinstance(arrivals, dict):
        router_level = arrivals.get("router_level")
        for where, source in (("arrivals", arrivals), ("arrivals.router_level", router_level)):
            if isinstance(source, dict) and source.get("dispersion_10s") is not None:
                arrivals_dispersion = _shape_number(source, "dispersion_10s", where)
                if arrivals_dispersion <= 0:
                    raise ValueError(f"--shape-mix {where}: 'dispersion_10s' must be > 0, got {arrivals_dispersion}")
                break

    explicit_names = [raw.get("name") for raw in spec["classes"] if isinstance(raw, dict) and "name" in raw]
    if len(set(map(str, explicit_names))) != len(explicit_names):
        raise ValueError(f"--shape-mix class names must be unique, got {explicit_names}")
    used_names = set(map(str, explicit_names))

    classes = []
    for i, raw in enumerate(spec["classes"]):
        where = f"class {i}"
        if not isinstance(raw, dict):
            raise ValueError(f"--shape-mix {where}: must be a JSON object")
        unknown = sorted(set(raw) - SHAPE_CLASS_FIELDS)
        if unknown:
            logger.warning(f"--shape-mix {where}: ignoring unknown fields {unknown}")
        if "name" in raw:
            name = raw["name"]
            if not isinstance(name, str) or not _SHAPE_CLASS_NAME_RE.match(name):
                raise ValueError(f"--shape-mix {where}: 'name' must match [A-Za-z0-9_-]+, got {name!r}")
        else:
            base = name = _derived_shape_class_name(raw, i)
            suffix = 2
            while name in used_names:
                name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(name)
        weight = raw.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"--shape-mix {where}: 'weight' must be a non-negative number, got {weight!r}")
        if _shape_number(raw, "cached_prompt_tokens", where, 0) > _shape_number(raw, "prompt_tokens", where):
            raise ValueError(
                f"--shape-mix {where}: cached_prompt_tokens ({raw['cached_prompt_tokens']}) > "
                f"prompt_tokens ({raw['prompt_tokens']})"
            )
        prompt_tokens = _shape_int(raw, "prompt_tokens", where, 1)
        cached_prompt_tokens = min(_shape_int(raw, "cached_prompt_tokens", where, 0, default=0), prompt_tokens)
        if cached_prompt_tokens > 0 and "prefix_pool_size" not in raw:
            logger.warning(
                f"--shape-mix class {name}: no prefix_pool_size, so all its requests share 1 prefix "
                f"({cached_prompt_tokens} cached tokens); set it to size the prefix working set"
            )
        classes.append(
            ShapeClass(
                name=name,
                weight=float(weight),
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                generated_tokens=_shape_int(raw, "generated_tokens", where, 1),
                prefix_pool_size=_shape_int(raw, "prefix_pool_size", where, 1, default=1),
                uncached_tokens_quantiles=_parse_quantiles(raw, "uncached_tokens_quantiles", where, 0),
                generated_tokens_quantiles=_parse_quantiles(raw, "generated_tokens_quantiles", where, 1),
                prefix_popularity=_parse_popularity(
                    raw, where, _shape_int(raw, "prefix_pool_size", where, 1, default=1)
                ),
                prompt_tokens_max=_prompt_tokens_max(raw, spec),
            )
        )
    if sum(c.weight for c in classes) <= 0:
        raise ValueError("--shape-mix class weights must sum to a positive number")
    return ShapeMix(
        classes=classes,
        seed=seed,
        arrivals_cv2=arrivals_cv2,
        request_seed=request_seed,
        arrivals_dispersion_10s=arrivals_dispersion,
    )


def rate_block_windows(options: argparse.Namespace) -> Optional[int]:
    """Modulation windows per rate block: the whole deferred -t run when known, so the tier averages --qps."""
    window = getattr(options, "qps_modulation_window", None)
    if not window or not InitTracker.deferred_run_time_seconds:
        return None
    return math.ceil(InitTracker.deferred_run_time_seconds / window)


def resolve_qps_dispersion(options: argparse.Namespace) -> Optional[float]:
    """--qps-dispersion if given, else the profile's arrivals.dispersion_10s (or arrivals.router_level.dispersion_10s)."""
    if getattr(options, "qps_dispersion", None) is not None:
        return options.qps_dispersion
    if getattr(options, "shape_mix", None):
        return parse_shape_mix(options.shape_mix).arrivals_dispersion_10s
    return None


# Prefix tokens a server can't reuse where a prompt diverges from a cached one, by model type: DeepSeek V4
# keeps sliding-window KV only for a sequence's last tokens, so it recomputes the window before the divergence.
_REUSE_OVERHANG_FROM_SLIDING_WINDOW = {"deepseek_v4"}
# Model types whose servers restore cached KV only in whole blocks of the config's index_topk tokens
# (GLM 5.x DSA: a shared prefix of N tokens reuses floor(N / 2048) * 2048)
_RESTORE_BOUNDARY_FROM_INDEX_TOPK = {"glm_moe_dsa"}
# Shared tokens laid out past a restore boundary, so a cut a few tokens short (a word straddling it) keeps the block
RESTORE_BOUNDARY_MARGIN_TOKENS = 16


def resolve_prefix_reuse_overhang(options: argparse.Namespace) -> int:
    """--prefix-reuse-overhang if given, else the model's sliding window for model types that need it, else 0."""
    if getattr(options, "prefix_reuse_overhang", None) is not None:
        return options.prefix_reuse_overhang
    tokenizer_path = getattr(options, "tokenizer", None) or DEFAULT_TOKENIZER
    try:
        if resolve_model_type(tokenizer_path) not in _REUSE_OVERHANG_FROM_SLIDING_WINDOW:
            return 0
        return int(_read_config_json(tokenizer_path).get("sliding_window") or 0)
    except Exception as e:
        logger.warning(f"--shape-mix: couldn't read the model config for the prefix reuse overhang ({e!r}); using 0")
        return 0


def resolve_context_tokens(options: argparse.Namespace) -> Optional[int]:
    """The model's context for --shape-mix-sample-lengths caps: the config's max_position_embeddings (its text
    config's for multimodal models), or None when it can't be read (no context cap)."""
    if not getattr(options, "shape_mix_sample_lengths", False):
        return None
    tokenizer_path = getattr(options, "tokenizer", None) or DEFAULT_TOKENIZER
    try:
        config = _read_config_json(tokenizer_path)
        value = (config.get("text_config") or {}).get("max_position_embeddings") or config.get(
            "max_position_embeddings"
        )
        return int(value) if value else None
    except Exception as e:
        logger.warning(f"--shape-mix: couldn't read the model context from its config ({e!r}); prompts aren't capped")
        return None


def resolve_prefix_restore_boundary(options: argparse.Namespace) -> int:
    """--prefix-restore-boundary if given, else the model's KV restore block for model types that restore in
    blocks (GLM 5.x: the config's index_topk, 2048), else 0 (token-exact reuse)."""
    if getattr(options, "prefix_restore_boundary", None) is not None:
        return options.prefix_restore_boundary
    tokenizer_path = getattr(options, "tokenizer", None) or DEFAULT_TOKENIZER
    try:
        if resolve_model_type(tokenizer_path) not in _RESTORE_BOUNDARY_FROM_INDEX_TOPK:
            return 0
        return int(_read_config_json(tokenizer_path).get("index_topk") or 0)
    except Exception as e:
        logger.warning(f"--shape-mix: couldn't read the model config for the prefix restore boundary ({e!r}); using 0")
        return 0


def resolve_request_seed(options: argparse.Namespace) -> Optional[int]:
    """--request-seed if given, else the shape mix's request_seed (which defaults to its seed), else None."""
    if getattr(options, "request_seed", None) is not None:
        return options.request_seed
    if getattr(options, "shape_mix", None):
        return parse_shape_mix(options.shape_mix).request_seed
    return None


def resolve_qps_cv2(options: argparse.Namespace) -> float:
    """--qps-cv2 if given, else the profile's arrivals.cv2 from --shape-mix, else 1 (Poisson)."""
    if getattr(options, "qps_cv2", None) is not None:
        return options.qps_cv2
    if getattr(options, "shape_mix", None):
        cv2 = parse_shape_mix(options.shape_mix).arrivals_cv2
        if cv2 is not None:
            return cv2
    return 1.0


@dataclass(frozen=True)
class ShapeRequest:
    prompt: str
    prompt_tokens: int
    class_index: int
    class_name: str
    generated_tokens: int
    prefix_index: Optional[int]
    affinity_key: Optional[str]
    reusable_tokens: int = 0
    # A multi-turn conversation; `prompt` is then the last user message
    messages: Optional[list] = None
    # --replay-schedule rows: the recorded cached length, payload overrides, and whether it was cancelled
    recorded_cached_tokens: Optional[int] = None
    overrides: Optional[dict] = None
    cancelled: bool = False


class _CyclicText:
    """Corpus text addressed by word index, repeating forever.

    A window of `width` words starting at word `offset` is a plain slice of the text, so rendering a
    prompt is a memory copy: it starts at a word start, ends at a word end (never in whitespace), and
    needs no tokenizer call.
    """

    def __init__(self, text: str):
        self.text = text + "\n\n"
        starts, ends = array("I"), array("I")
        for m in re.finditer(r"\S+", self.text):
            starts.append(m.start())
            ends.append(m.end())
        self.starts, self.ends = starts, ends
        self.words = len(starts)
        # Cumulative token count at the end of each word (set_token_counts), to predict window widths
        self.token_ends = array("I", range(1, self.words + 1))
        self.cycle_tokens = self.words

    def set_token_counts(self, token_starts: list[int]) -> None:
        """From the start offsets (in characters) of the tokens of `text`, in order."""
        self.token_ends = array("I", (bisect.bisect_left(token_starts, end) for end in self.ends))
        self.cycle_tokens = self.token_ends[-1]

    def _tokens_through(self, word: int) -> int:
        cycle, i = divmod(word, self.words)
        return cycle * self.cycle_tokens + self.token_ends[i]

    def tokens_between(self, offset: int, width: int) -> int:
        """Approximate tokens in the window, as tokenized in the running corpus."""
        return self._tokens_through(offset + width - 1) - self._tokens_through(offset - 1)

    def width_for(self, offset: int, goal: int) -> int:
        """Fewest words from `offset` whose approximate token count reaches `goal`."""
        if goal <= 0:
            return 0
        lo, hi = 1, max(2, (goal // max(1, self.cycle_tokens) + 2) * self.words)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.tokens_between(offset, mid) >= goal:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def slice(self, offset: int, width: int) -> str:
        if width <= 0:
            return ""
        first_cycle, first = divmod(offset, self.words)
        last_cycle, last = divmod(offset + width - 1, self.words)
        if first_cycle == last_cycle:
            return self.text[self.starts[first] : self.ends[last]]
        return "".join(
            [self.text[self.starts[first] :]]
            + [self.text] * (last_cycle - first_cycle - 1)
            + [self.text[: self.ends[last]]]
        )

    def resident_bytes(self) -> int:
        return len(self.text) + sum(a.buffer_info()[1] * a.itemsize for a in (self.starts, self.ends))


# Later user messages of a replayed session start like a cold tail
_SESSION_TAIL_START = SHAPE_MIX_TAIL_START.lstrip()

# 'prose' (the default) is 1.7 MB of public-domain novels, so long prompts don't repeat the text the way the
# 4.9k-token limericks do; 'code' is 1.76 MB of CPython 3.12 standard-library source (PSF license, see
# code_corpus.LICENSE.txt); code.txt stays --dataset code's file
_BUILTIN_CORPORA = {
    name: os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
    for name, file in (("prose", "prose_corpus.txt.gz"), ("limericks", "limericks.txt"), ("code", "code_corpus.txt.gz"))
}


def resolve_shape_mix_corpus(value: Optional[str]) -> str:
    """--shape-mix-corpus: a built-in name ('prose', the default, 'limericks' or 'code') or a text file path."""
    if not value:
        return _BUILTIN_CORPORA["prose"]
    path = _BUILTIN_CORPORA.get(value, value[1:] if value.startswith("@") else value)
    if not os.path.isfile(path):
        raise ValueError(f"--shape-mix-corpus: no such file {path!r} (or use one of {sorted(_BUILTIN_CORPORA)})")
    return path


def resolve_shape_mix_instruction(value: Optional[str]) -> Optional[str]:
    """--shape-mix-instruction: inline text or @file."""
    if not value:
        return None
    if value.startswith("@"):
        with open(value[1:], "r") as f:
            return f.read()
    return value


# Headers are never truncated for prefixes longer than this, so rendering them needs no tokenizer call
_MAX_PREFIX_HEADER_TOKENS = 64


def _prefix_header(class_index: int, prefix_index: int) -> str:
    return f"Shared context {class_index}.{prefix_index}:\n\n"


class ShapeMixDataset:
    """Class-mix prompts: a seeded shared prefix per class (one of N) plus a unique tail.

    Prefixes depend only on `seed`, so every worker and every run sends the same working set. Each prefix
    is a header plus a window of the cyclic word corpus, so the pool stores only (offset, width) per prefix
    and slices the text per request; `build_prefixes()` finds every window before traffic starts.

    A tail is a nonce of fixed token length (single-token words encoding the request seed's random bits,
    the worker index and the request counter) followed by one of a few precomputed fillers of the class.
    Every prompt is therefore unique and diverges right after its prefix, and building a request needs no
    tokenizer call. If the tokenizer can't give the nonce a fixed length, tails are fitted per request.

    Class choice, prefix choice and the nonce come from `request_seed`: runs that must not replay each
    other's tails (e.g. load tiers against one deployment) need distinct request seeds. Token counts include
    the chat template; `chat_head_tokens` (template tokens before the user content) count toward the cached
    tokens. `reuse_overhang_tokens` is how many shared tokens the server can't reuse where a prompt diverges
    from a cached one (DeepSeek V4 recomputes its 128-token sliding window there); prefixes are that much
    longer, so the server reuses exactly `cached_prompt_tokens`. `restore_boundary` (GLM 5.x: 2048) is the
    block the server restores cached KV in: a shared prefix of N tokens reuses floor(N / boundary) * boundary,
    so cached lengths are aligned down to a boundary and prefixes laid out a few tokens past it.
    """

    def __init__(
        self,
        mix: ShapeMix,
        tokenizer,
        corpus_path: str,
        chat_overhead_tokens: int = 0,
        chat_head_tokens: int = 0,
        worker_index: int = 0,
        request_seed: Optional[int] = None,
        reuse_overhang_tokens: int = 0,
        restore_boundary: int = 0,
        tail_variants: int = SHAPE_MIX_TAIL_VARIANTS,
        pool_cache: Optional[str] = None,
        instruction: Optional[str] = None,
        sample_lengths: bool = False,
        default_popularity_zipf: Optional[float] = None,
        context_tokens: Optional[int] = None,
        split_restore_classes: bool = True,
    ):
        self.mix = mix
        # The model's context: sampled prompts plus outputs stay inside it
        self.context_tokens = context_tokens
        self.context_cut_requests = 0
        self.sample_lengths = sample_lengths
        # Ends every prompt, after the unique tail; counted in prompt_tokens
        self.trailer = "\n\n" + instruction.strip() if instruction and instruction.strip() else ""
        self.pool_cache = pool_cache
        self.cache_loaded_prefixes = 0
        self.request_seed = mix.request_seed if request_seed is None else request_seed
        self.reuse_overhang_tokens = reuse_overhang_tokens
        self.restore_boundary = restore_boundary
        self._tokenizer = tokenizer
        backend = getattr(tokenizer, "_tokenizer", None)
        self._backend = backend if hasattr(backend, "encode_batch") else None
        self._overhead = chat_overhead_tokens
        self._head = chat_head_tokens
        self._worker_index = worker_index
        self._lock = threading.Lock()
        self._prefix_lock = threading.Lock()
        self._counter = 0
        self._rng = random.Random(f"shape-mix:{self.request_seed}:requests:{worker_index}")
        self._fill_rng = random.Random(f"shape-mix:{self.request_seed}:fill:{worker_index}")
        # The mix's classes, then companions one restore block up (indexes past the mix's report as their class)
        self._upper: dict[int, tuple[int, float]] = {}
        self._class_pools: dict[int, int] = {}
        self.classes = self._with_restore_companions(mix, chat_overhead_tokens) if split_restore_classes else []
        self.classes = self.classes or list(mix.classes)
        self._report = list(range(len(mix.classes))) + sorted(self._upper)
        self._weights = [c.weight for c in self.classes]

        with (
            gzip.open(corpus_path, "rt", encoding="utf-8") if corpus_path.endswith(".gz") else open(corpus_path, "r")
        ) as f:
            texts = [t.strip() for t in f.read().split("\n\n") if t.strip()]
        random.Random(f"shape-mix:{mix.seed}:corpus").shuffle(texts)
        self._corpus = _CyclicText("\n\n".join(texts))
        corpus_ids = self._encode(self._corpus.text)
        self._corpus.set_token_counts(self._token_starts(self._corpus.text))
        self._fingerprint = self._pool_fingerprint(corpus_ids)
        self._tail_start_tokens = len(self._encode(SHAPE_MIX_TAIL_START))
        self._nonce_words = self._single_token_words()
        self._nonce_digits = self._nonce_layout()
        self.fixed_nonce_tokens = self._fixed_nonce_length()
        self._trailer_tokens = len(self._encode(self.trailer)) if self.trailer else 0

        self.prefix_tokens: list[int] = []
        self.reusable_tokens: list[int] = []
        self.pool_sizes: list[int] = []
        self._prefix_offsets: list[array] = []
        self._prefix_widths: list[array] = []
        self.build_seconds = 0.0
        for ci, cls in enumerate(self.classes):
            content_tokens = cls.prompt_tokens - chat_overhead_tokens
            if content_tokens < SHAPE_MIX_TAIL_RESERVE_TOKENS:
                raise ValueError(
                    f"--shape-mix class {cls.name}: prompt_tokens ({cls.prompt_tokens}) must be at least "
                    f"{SHAPE_MIX_TAIL_RESERVE_TOKENS + chat_overhead_tokens} (chat template + unique tail)"
                )
            prefix_tokens = self._shared_target(cls.cached_prompt_tokens)
            aligned = self._aligned_cached(cls.cached_prompt_tokens)
            if aligned != cls.cached_prompt_tokens and ci not in self._upper:
                logger.warning(
                    f"--shape-mix class {cls.name}: cached_prompt_tokens {cls.cached_prompt_tokens} aligned to {aligned} "
                    f"(the server restores cached KV in {restore_boundary}-token blocks)"
                )
            if prefix_tokens > content_tokens - SHAPE_MIX_TAIL_RESERVE_TOKENS:
                prefix_tokens = content_tokens - SHAPE_MIX_TAIL_RESERVE_TOKENS
                logger.warning(
                    f"--shape-mix class {cls.name}: shared prefix capped at {prefix_tokens} content tokens "
                    f"to leave room for a unique tail"
                )
            pool_size = self._class_pools.get(ci, cls.prefix_pool_size) if prefix_tokens > 0 else 0
            self.prefix_tokens.append(prefix_tokens)
            self.reusable_tokens.append(min(aligned, self._reusable_of(prefix_tokens)))
            if aligned > 0 and self.reusable_tokens[-1] < aligned:
                logger.warning(
                    f"--shape-mix class {cls.name}: only {self.reusable_tokens[-1]} of {aligned} "
                    f"cached tokens are reusable (prompt too short for the prefix plus reuse overhang)"
                )
            self.pool_sizes.append(pool_size)
            self._prefix_offsets.append(array("I", bytes(4 * pool_size)))
            self._prefix_widths.append(array("I", [_UNBUILT]) * pool_size)

        nonce_tokens = (self.fixed_nonce_tokens or {}).get(SHAPE_MIX_TAIL_START, SHAPE_MIX_TAIL_RESERVE_TOKENS // 2)
        too_short = [
            f"{c.name} ({self._tail_tokens(ci)}-token tail)"
            for ci, c in enumerate(self.classes)
            if self._tail_tokens(ci) < nonce_tokens + self._trailer_tokens + 2
        ]
        if too_short:
            raise ValueError(
                f"--shape-mix-instruction ({self._trailer_tokens} tokens) doesn't fit after the unique nonce in: "
                f"{', '.join(too_short)}; shorten it or raise prompt_tokens"
            )
        self.tail_variants = tail_variants if self.fixed_nonce_tokens is not None else 0
        # With --shape-mix-sample-lengths: weighted strata of uncached lengths and the factors that keep each
        # class's mean (the profile quantiles' own mean generally differs from the class mean)
        self._variant_cumulative: list[Optional[list[float]]] = [None] * len(self.classes)
        self._uncached_dist: list[Optional[LengthQuantiles]] = [None] * len(self.classes)
        self._generated_dist = [self._generated_dist_for(c) for c in self.classes]
        self._filler_tokens = [self._tail_lengths(ci) for ci in range(len(self.classes))]
        self._prefix_cumulative = [
            self._popularity_cumulative(ci, default_popularity_zipf) for ci in range(len(self.classes))
        ]
        self._filler_offsets = [array("I", bytes(4 * len(t))) for t in self._filler_tokens]
        self._filler_widths = [array("I", [_UNBUILT]) * len(t) for t in self._filler_tokens]
        if self.tail_variants:
            missing = [ci for ci in range(len(self.classes)) if not self._load_cached(*self._filler_cache_slot(ci))]
            self._build_fillers(missing)
            for ci in missing:
                self._save_cached(*self._filler_cache_slot(ci))

    def _with_restore_companions(self, mix: ShapeMix, chat_overhead_tokens: int) -> list[ShapeClass]:
        """With a restore boundary b, a class's cached mean c lays out as floor(c / b) * b, which can lose most of
        it (613 -> 0). Instead a share p = (c mod b) / b of its requests use a companion class one block up, so the
        class's mean cached tokens is kept; its pool is split between the two in the same ratio, so the prefix
        working set is too. Companions have weight 0 (drawn through their class) and report as their class.
        Returns the mix's classes followed by the companions, or [] when no class splits."""
        b = self.restore_boundary
        classes, companions = list(mix.classes), []
        for ci, cls in enumerate(mix.classes):
            c = cls.cached_prompt_tokens
            if not b or not c % b:
                continue
            upper = c // b * b + b
            if self._shared_target(upper) > cls.prompt_tokens - chat_overhead_tokens - SHAPE_MIX_TAIL_RESERVE_TOKENS:
                continue  # the next block doesn't fit in the prompt: floor only (warned below)
            share = (c % b) / b
            n = cls.prefix_pool_size
            # The working set stays n prefixes (one each when n is 1)
            upper_n = max(1, min(n - 1, round(n * share)))
            lower_n = max(1, n - upper_n)
            upper_popularity = cls.prefix_popularity
            if upper_popularity is not None and upper_popularity[0] == "weights" and n > 1:
                weights = upper_popularity[1]
                upper_popularity = ("weights", weights[:upper_n])
                classes[ci] = dataclasses.replace(cls, prefix_popularity=("weights", weights[n - lower_n :]))
            self._class_pools[ci] = lower_n
            self._upper[ci] = (len(mix.classes) + len(companions), share)
            companions.append(
                dataclasses.replace(
                    cls,
                    name=f"{cls.name}_upper",
                    weight=0.0,
                    cached_prompt_tokens=upper,
                    prefix_pool_size=upper_n,
                    prefix_popularity=upper_popularity,
                )
            )
            logger.info(
                f"--shape-mix class {cls.name}: cached_prompt_tokens {c} laid out as {upper - b} on {1 - share:.0%} of "
                f"its requests ({lower_n} prefixes) and {upper} on {share:.0%} ({upper_n}), keeping the mean with "
                f"{b}-token restore blocks"
            )
        return classes + companions if companions else []

    def class_reusable_tokens(self) -> list[float]:
        """Mean reusable tokens per class of the mix (a split class's two block lengths, weighted)."""
        out = []
        for ci in range(len(self.mix.classes)):
            upper = self._upper.get(ci)
            if upper is None:
                out.append(self.reusable_tokens[ci])
            else:
                k, share = upper
                out.append((1 - share) * self.reusable_tokens[ci] + share * self.reusable_tokens[k])
        return out

    def _aligned_cached(self, cached: int) -> int:
        """The cached length the server can deliver: `cached`, or with a restore boundary what it restores from a
        `cached`-token shared prefix, floor(cached / boundary) * boundary."""
        boundary = self.restore_boundary
        return cached // boundary * boundary if boundary else cached

    def _shared_target(self, cached: int) -> int:
        """Content tokens to share with earlier prompts so the server reuses `cached` (aligned) tokens."""
        aligned = self._aligned_cached(cached)
        if aligned <= self._head:
            return 0
        margin = RESTORE_BOUNDARY_MARGIN_TOKENS if self.restore_boundary else 0
        return aligned - self._head + self.reuse_overhang_tokens + margin

    def _reusable_of(self, shared_content: int) -> int:
        """Tokens the server reuses from `shared_content` shared content tokens (after the template head)."""
        if shared_content <= 0:
            return 0
        effective = max(0, self._head + shared_content - self.reuse_overhang_tokens)
        boundary = self.restore_boundary
        return effective // boundary * boundary if boundary else effective

    @property
    def built_prefixes(self) -> int:
        return sum(len(w) - w.count(_UNBUILT) for w in self._prefix_widths)

    @property
    def pool_tokens(self) -> int:
        return sum(n * size for n, size in zip(self.prefix_tokens, self.pool_sizes))

    def resident_bytes(self) -> int:
        """Memory held by the prefix pool, tail fillers and the corpus they index into."""
        arrays = [*self._prefix_offsets, *self._prefix_widths, *self._filler_offsets, *self._filler_widths]
        return self._corpus.resident_bytes() + sum(a.buffer_info()[1] * a.itemsize for a in arrays)

    def build_prefixes(self) -> float:
        """Find every prefix window now, so none is built inside the measured window; returns seconds taken.

        With `pool_cache`, windows found by earlier processes are loaded first and only the rest are built."""
        started = time.perf_counter()
        for ci in range(len(self.classes)):
            if self.pool_sizes[ci]:
                self.cache_loaded_prefixes += self._load_cached(*self._prefix_cache_slot(ci))
        missing = [
            (ci, j)
            for ci, size in enumerate(self.pool_sizes)
            for j in range(size)
            if self._prefix_widths[ci][j] == _UNBUILT
        ]
        self._build_prefix_keys(missing)
        for ci in {ci for ci, _ in missing}:
            self._save_cached(*self._prefix_cache_slot(ci))
        return time.perf_counter() - started

    def _pool_fingerprint(self, corpus_ids: list[int]) -> str:
        """What every window depends on besides seed and lengths: format, corpus text and tokenizer."""
        h = hashlib.sha256(f"{_POOL_CACHE_FORMAT}\n".encode())
        h.update(self._corpus.text.encode())
        h.update(array("I", corpus_ids).tobytes())
        to_str = getattr(self._backend, "to_str", None)
        h.update((to_str() if callable(to_str) else type(self._tokenizer).__qualname__).encode())
        return h.hexdigest()

    def _prefix_cache_slot(self, ci: int):
        key = f"prefix:{self.mix.seed}:{ci}:{self.prefix_tokens[ci]}:{self._fingerprint}"
        return "prefix", key, self._prefix_offsets[ci], self._prefix_widths[ci]

    def _filler_cache_slot(self, ci: int):
        lead = self._nonce_text(self._tail_start(ci), 0, 0)
        trailer = hashlib.sha256(self.trailer.encode()).hexdigest()[:16]
        lengths = hashlib.sha256(repr(self._filler_tokens[ci]).encode()).hexdigest()[:16]
        key = f"filler:{self.mix.seed}:{ci}:{lengths}:{lead}:{trailer}:{self._fingerprint}"
        return "filler", key, self._filler_offsets[ci], self._filler_widths[ci]

    def _cache_path(self, kind: str, key: str) -> str:
        return os.path.join(self.pool_cache, f"{kind}-{hashlib.sha256(key.encode()).hexdigest()[:40]}.bin")

    def _load_cached(self, kind: str, key: str, offsets: array, widths: array) -> int:
        """Fill offsets/widths from the cache file for `key`; returns how many windows were loaded."""
        if not self.pool_cache or not len(widths):
            return 0
        try:
            with open(self._cache_path(kind, key), "rb") as f:
                header = json.loads(f.readline())
                if header.get("key") != key:
                    return 0
                count = int(header["count"])
                stored_offsets, stored_widths = array("I"), array("I")
                stored_offsets.frombytes(f.read(4 * count))
                stored_widths.frombytes(f.read(4 * count))
        except (OSError, ValueError, KeyError, EOFError):
            return 0
        if len(stored_offsets) != count or len(stored_widths) != count:
            return 0
        n = min(count, len(widths))
        offsets[:n] = stored_offsets[:n]
        widths[:n] = stored_widths[:n]
        return n

    def _save_cached(self, kind: str, key: str, offsets: array, widths: array) -> None:
        """Write this process's windows for `key`, unless the file already holds at least as many."""
        if not self.pool_cache or not len(widths) or _UNBUILT in widths:
            return
        path = self._cache_path(kind, key)
        try:
            with open(path, "rb") as f:
                if json.loads(f.readline()).get("count", 0) >= len(widths):
                    return
        except (OSError, ValueError):
            pass
        try:
            os.makedirs(self.pool_cache, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            with open(tmp, "wb") as f:
                f.write((json.dumps({"key": key, "count": len(widths)}) + "\n").encode())
                f.write(offsets.tobytes())
                f.write(widths.tobytes())
            os.replace(tmp, path)
        except OSError as e:
            logger.warning(f"--prefix-pool-cache: couldn't write {path}: {e!r}")

    def prefix(self, class_index: int, prefix_index: int) -> str:
        """Render shared prefix `prefix_index` of a class (its window is found first if not built yet)."""
        if self._prefix_widths[class_index][prefix_index] == _UNBUILT:
            with self._prefix_lock:
                if self._prefix_widths[class_index][prefix_index] == _UNBUILT:
                    self._build_prefix_keys([(class_index, prefix_index)])
        if self.prefix_tokens[class_index] > _MAX_PREFIX_HEADER_TOKENS:
            lead = _prefix_header(class_index, prefix_index)
        else:
            lead = self._prefix_lead(class_index, prefix_index)[0]
        return lead + self._corpus.slice(
            self._prefix_offsets[class_index][prefix_index], self._prefix_widths[class_index][prefix_index]
        )

    def _encode(self, text: str) -> list[int]:
        if self._backend is not None:
            return self._backend.encode(text, add_special_tokens=False).ids
        return _normalize_token_ids(self._tokenizer.encode(text, add_special_tokens=False))

    def _token_starts(self, text: str) -> list[int]:
        if self._backend is not None:
            return [start for start, _ in self._backend.encode(text, add_special_tokens=False).offsets]
        # Without offsets, credit each word with its standalone token count, at its start
        starts = []
        for start, end in zip(self._corpus.starts, self._corpus.ends):
            starts.extend([start] * len(self._encode(" " + text[start:end])))
        return starts

    def _encode_lengths(self, texts: list[str]) -> list[int]:
        if self._backend is not None:
            return [len(e.ids) for e in self._backend.encode_batch(texts, add_special_tokens=False)]
        return [len(self._encode(t)) for t in texts]

    def _decode(self, ids: list[int]) -> str:
        if self._backend is not None:
            return self._backend.decode(ids, skip_special_tokens=False)
        return self._tokenizer.decode(ids)

    def _single_token_words(self) -> list[str]:
        # The built-in limericks always contribute, so a corpus with few plain words (e.g. code) keeps the fast path
        with open(_BUILTIN_CORPORA["limericks"], "r") as f:
            words = self._corpus.text.split() + f.read().split()
        candidates = sorted({w for w in words if w.isalpha() and w.islower() and 2 <= len(w) <= 10})
        lengths = self._encode_lengths([" " + w for w in candidates])
        words = [w for w, n in zip(candidates, lengths) if n == 1]
        random.Random("shape-mix:nonce-words").shuffle(words)
        return words[:256]

    def _nonce_layout(self) -> Optional[tuple[int, int, int]]:
        """Digits (in base len(nonce words)) for the random bits, worker index and request counter."""
        base = len(self._nonce_words)
        if base < 64:
            return None
        digits = lambda bits: math.ceil(bits / math.log2(base))
        return digits(64), digits(16), digits(32)

    def _nonce_text(self, tail_start: str, nonce: int, counter: int) -> str:
        if self._nonce_digits is None:
            return f"{tail_start} {nonce:016x}-{self._worker_index}-{counter}:"
        base = len(self._nonce_words)
        words = []
        for value, count in zip((nonce, self._worker_index, counter), self._nonce_digits):
            for _ in range(count):
                value, digit = divmod(value, base)
                words.append(self._nonce_words[digit])
        return f"{tail_start} {' '.join(words)}."

    def _fixed_nonce_length(self) -> Optional[dict[str, int]]:
        """Token length of the nonce for each tail start, or None if the tokenizer doesn't keep it fixed."""
        if self._nonce_digits is None:
            return None
        rng = random.Random("shape-mix:nonce-check")
        word = " " + self._corpus.slice(0, 1)
        word_tokens = len(self._encode(word))
        lengths = {}
        for tail_start in (SHAPE_MIX_TAIL_START, SHAPE_MIX_TAIL_START.lstrip()):
            texts = [self._nonce_text(tail_start, rng.getrandbits(64), rng.getrandbits(32)) + word for _ in range(128)]
            seen = set(self._encode_lengths(texts))
            if len(seen) != 1:
                logger.warning(
                    "--shape-mix: the tokenizer doesn't keep the word nonce at a fixed length; fitting tails per request"
                )
                return None
            lengths[tail_start] = seen.pop() - word_tokens
        return lengths

    def _window_text(self, lead: str, offset: int, width: int, trailer: str = "") -> str:
        return lead + self._corpus.slice(offset, width) + trailer

    def _prefix_lead(self, class_index: int, prefix_index: int) -> tuple[str, int]:
        num_tokens = self.prefix_tokens[class_index]
        header = _prefix_header(class_index, prefix_index)
        header_ids = self._encode(header)
        if len(header_ids) >= num_tokens:
            header = self._decode(header_ids[:num_tokens])
            return header, len(self._encode(header))
        return header, len(header_ids)

    def _tail_start(self, class_index: int) -> str:
        return SHAPE_MIX_TAIL_START if self.pool_sizes[class_index] else SHAPE_MIX_TAIL_START.lstrip()

    def _tail_tokens(self, class_index: int) -> int:
        return self.classes[class_index].prompt_tokens - self._overhead - self.prefix_tokens[class_index]

    def _min_tail_tokens(self, class_index: int) -> int:
        nonce = (self.fixed_nonce_tokens or {}).get(self._tail_start(class_index), SHAPE_MIX_TAIL_RESERVE_TOKENS // 2)
        return nonce + self._trailer_tokens + 2

    def _tail_for_uncached(self, class_index: int, uncached_tokens: float) -> int:
        """Tail length that leaves `uncached_tokens` for the server to compute (the rest of the prompt is the
        reusable prefix plus fixed template tokens), floored at what the nonce and instruction need."""
        cls = self.classes[class_index]
        offset = (cls.prompt_tokens - self.reusable_tokens[class_index]) - self._tail_tokens(class_index)
        return max(self._min_tail_tokens(class_index), round(uncached_tokens) - offset)

    def _generated_dist_for(self, cls) -> Optional[LengthQuantiles]:
        points = cls.generated_tokens_quantiles if self.sample_lengths else None
        if not points:
            return None
        dist = LengthQuantiles(points)
        how = dist.fit_mean(lambda d: d.average(0.0, 1.0) - cls.generated_tokens)
        if how == "scale":
            logger.warning(
                f"--shape-mix class {cls.name}: no upper tail reaches the output mean {cls.generated_tokens} with the "
                f"given quantiles kept; all output quantiles scaled by {dist.scale:.3f}"
            )
        return dist

    def _uncached_cap(self, class_index: int) -> Optional[float]:
        """Most uncached tokens a sampled prompt may have: inside the class's prompt band, and inside the model's
        context with the class's mean output (a longer output drawn for a long prompt is cut in _render)."""
        cls = self.classes[class_index]
        reusable = self.reusable_tokens[class_index]
        caps = []
        if cls.prompt_tokens_max is not None:
            caps.append(cls.prompt_tokens_max - reusable)
        if self.context_tokens:
            caps.append(self.context_tokens - reusable - cls.generated_tokens)
        return max(1.0, min(caps)) if caps else None

    def _tail_lengths(self, class_index: int) -> list[int]:
        """Tail length of each filler: the class mean, or with sampling one per stratum of the uncached quantiles
        (sampling_strata: each stratum's exact quantile average, drawn by its width). The values are scaled so the
        class's mean uncached tokens (prompt_tokens minus the reusable share) is kept, floors included."""
        cls = self.classes[class_index]
        points = cls.uncached_tokens_quantiles if self.sample_lengths else None
        if not points:
            return [self._tail_tokens(class_index)] * self.tail_variants
        # Strata end at the profile's own quantile levels too, so none straddles a given quantile
        bounds = sorted(
            set(sampling_strata(self.tail_variants or SHAPE_MIX_TAIL_VARIANTS)) | {q for q, _ in points if 0 < q < 1}
        )
        weights = [b - a for a, b in zip(bounds, bounds[1:])]
        target = self._tail_tokens(class_index)
        dist = LengthQuantiles(points, cap=self._uncached_cap(class_index))

        def mean_tail(d: LengthQuantiles) -> float:
            values = [d.average(a, b) for a, b in zip(bounds, bounds[1:])]
            return sum(w * self._tail_for_uncached(class_index, v) for w, v in zip(weights, values)) - target

        how = dist.fit_mean(mean_tail)
        self._uncached_dist[class_index] = dist
        if how == "scale":
            logger.warning(
                f"--shape-mix class {cls.name}: no upper tail reaches the uncached mean "
                f"{cls.prompt_tokens - self.reusable_tokens[class_index]} with the given quantiles kept; all uncached "
                f"quantiles scaled by {dist.scale:.3f}"
            )
        if not self.tail_variants:
            return []
        values = [dist.average(a, b) for a, b in zip(bounds, bounds[1:])]
        lengths = [self._tail_for_uncached(class_index, v) for v in values]
        self._variant_cumulative[class_index] = list(itertools.accumulate(weights))
        if self._min_tail_tokens(class_index) in lengths:
            logger.warning(
                f"--shape-mix class {cls.name}: low uncached-token quantiles are raised to "
                f"{self._min_tail_tokens(class_index)}-token tails (unique nonce and instruction)"
            )
        return lengths

    def _build_prefix_keys(self, keys: list[tuple[int, int]]) -> None:
        jobs = []
        for ci, j in keys:
            lead, lead_tokens = self._prefix_lead(ci, j)
            jobs.append(
                _WindowSearch(
                    offsets=self._prefix_offsets[ci],
                    widths=self._prefix_widths[ci],
                    index=j,
                    lead=lead,
                    lead_tokens=lead_tokens,
                    num_tokens=self.prefix_tokens[ci],
                    rng=random.Random(f"shape-mix:{self.mix.seed}:prefix:{ci}:{j}"),
                    check_tail_start=True,
                )
            )
        self._run_searches(jobs)

    def _build_fillers(self, class_indexes: Optional[list[int]] = None) -> None:
        """Fillers follow a sample nonce: every nonce has the same token length and ends in '.', so the
        filler's token count holds for every request."""
        jobs = []
        for ci in range(len(self.classes)) if class_indexes is None else class_indexes:
            lead = self._nonce_text(self._tail_start(ci), 0, 0) + " "
            for k in range(len(self._filler_tokens[ci])):
                jobs.append(
                    _WindowSearch(
                        offsets=self._filler_offsets[ci],
                        widths=self._filler_widths[ci],
                        index=k,
                        lead=lead,
                        lead_tokens=len(self._encode(lead)),
                        num_tokens=self._filler_tokens[ci][k],
                        rng=random.Random(f"shape-mix:{self.mix.seed}:filler:{ci}:{k}"),
                        check_tail_start=False,
                        trailer=self.trailer,
                        trailer_tokens=self._trailer_tokens,
                    )
                )
        self._run_searches(jobs)

    def _run_searches(self, jobs: list["_WindowSearch"]) -> None:
        """Find each window whose text (lead + words) re-encodes to exactly the target tokens and, for prefixes,
        doesn't merge with the tail start. Batched (bounded by tokens to cap memory) for multi-threaded
        tokenizer calls; each search follows its own seeded offsets, so results don't depend on batching."""
        batch: list[_WindowSearch] = []
        batch_tokens = 0
        for job in jobs:
            batch.append(job)
            batch_tokens += job.num_tokens
            if batch_tokens >= _PREFIX_BUILD_BATCH_TOKENS:
                self._run_search_batch(batch)
                batch, batch_tokens = [], 0
        if batch:
            self._run_search_batch(batch)

    def _run_search_batch(self, searches: list["_WindowSearch"]) -> None:
        for s in searches:
            s.corpus = self._corpus
            s.next_offset(self._corpus.words)
        while searches:
            texts = [self._window_text(s.lead, s.offset, s.width, s.trailer) for s in searches]
            fitted, retry = [], []
            for s, text, got in zip(searches, texts, self._encode_lengths(texts)):
                if got == s.num_tokens:
                    fitted.append((s, text))
                    continue
                s.adjust(got)
                if s.attempts == _FIT_ATTEMPTS:
                    s.next_offset(self._corpus.words)
                retry.append(s)
            boundary = iter(
                self._encode_lengths([text + SHAPE_MIX_TAIL_START for s, text in fitted if s.check_tail_start])
            )
            for s, _ in fitted:
                if s.check_tail_start and next(boundary) != s.num_tokens + self._tail_start_tokens:
                    s.next_offset(self._corpus.words)
                    retry.append(s)
                    continue
                s.offsets[s.index] = s.offset
                s.widths[s.index] = s.width
            searches = retry

    def _fit_tail(self, nonce_text: str, num_tokens: int, nonce: int) -> str:
        offsets = random.Random(nonce)
        search = _WindowSearch(
            offsets=array("I", [0]),
            widths=array("I", [0]),
            index=0,
            lead=nonce_text + " ",
            lead_tokens=len(self._encode(nonce_text + " ")),
            num_tokens=num_tokens,
            rng=offsets,
            check_tail_start=False,
            trailer=self.trailer,
            trailer_tokens=self._trailer_tokens,
        )
        self._run_search_batch([search])
        return self._window_text(search.lead, search.offsets[0], search.widths[0], self.trailer)

    def __next__(self) -> ShapeRequest:
        return self._draw()

    def scheduled_request(
        self, class_index: int, prefix_index: Optional[int] = None, generated_tokens: Optional[int] = None
    ) -> ShapeRequest:
        """A request of a given class (from --replay-schedule); the prefix and lengths are drawn as usual
        unless the schedule fixes them."""
        if prefix_index is not None and not 0 <= prefix_index < self.pool_sizes[class_index]:
            raise ValueError(
                f"--replay-schedule: prefix_index {prefix_index} is outside class "
                f"{self.classes[class_index].name}'s pool of {self.pool_sizes[class_index]}"
            )
        return self._draw(class_index, prefix_index, generated_tokens)

    def _draw(self, ci=None, prefix_index=None, generated=None) -> ShapeRequest:
        with self._lock:
            counter = self._counter
            self._counter += 1
            if ci is None:
                ci = self._rng.choices(range(len(self._weights)), weights=self._weights)[0]
                upper = self._upper.get(ci)
                if upper is not None and self._rng.random() < upper[1]:
                    ci = upper[0]
            pool_size = self.pool_sizes[ci]
            if not pool_size:
                prefix_index = None
            elif prefix_index is None:
                prefix_index = self._draw_prefix_index(ci)
            nonce = self._rng.getrandbits(64)
            variant = self._draw_variant(ci, self._rng)
            cls = self.classes[ci]
            tail_tokens = None
            if generated is None:
                generated = cls.generated_tokens
                if self.sample_lengths and cls.generated_tokens_quantiles:
                    drawn = self._generated_dist[ci].value(self._rng.random())
                    generated = max(1, round(drawn))
            if self.sample_lengths and cls.uncached_tokens_quantiles and variant is None:
                drawn = self._uncached_dist[ci].value(self._rng.random())
                tail_tokens = self._tail_for_uncached(ci, drawn)
        return self._render(ci, prefix_index, nonce, counter, variant, generated, tail_tokens)

    def _draw_variant(self, ci: int, rng: random.Random) -> Optional[int]:
        count = len(self._filler_tokens[ci])
        if not count:
            return None
        cumulative = self._variant_cumulative[ci]
        if cumulative is None:
            return rng.randrange(count)
        return min(bisect.bisect_right(cumulative, rng.random() * cumulative[-1]), count - 1)

    def _draw_prefix_index(self, ci: int) -> int:
        cumulative = self._prefix_cumulative[ci]
        if cumulative is None:
            return self._rng.randrange(self.pool_sizes[ci])
        return min(bisect.bisect_right(cumulative, self._rng.random() * cumulative[-1]), len(cumulative) - 1)

    def _popularity_cumulative(self, ci: int, default_zipf: Optional[float]) -> Optional[array]:
        """Cumulative prefix weights for weighted draws, or None for uniform (the class has no prefix_popularity
        and no --prefix-popularity-zipf default, or the pool has one prefix)."""
        n = self.pool_sizes[ci]
        popularity = self.classes[ci].prefix_popularity
        if popularity is None and default_zipf:
            popularity = ("zipf", default_zipf)
        if popularity is None or n <= 1 or popularity == ("zipf", 0.0):
            return None
        kind, value = popularity
        weights = (1 / (j + 1) ** value for j in range(n)) if kind == "zipf" else value
        return array("d", itertools.accumulate(weights))

    def prefix_popularity_share(self, ci: int, top: int) -> float:
        """Fraction of the class's requests expected on its `top` most popular prefixes."""
        n = self.pool_sizes[ci]
        if not n:
            return 0.0
        cumulative = self._prefix_cumulative[ci]
        if cumulative is None:
            return min(top, n) / n
        return cumulative[min(top, n) - 1] / cumulative[-1]

    def prepare_replay(
        self, entries: tuple, message_overhead=None, subtract_template: bool = True, session_salt: Optional[str] = None
    ) -> None:
        """Lay out every --replay-schedule row before traffic starts, so replay requests need no tokenizer call.

        Recorded lengths are authoritative: a row sends input_length prompt tokens as the server counts them
        (with `subtract_template`, the chat template's tokens come out of the content), of which cached_length
        are reusable, and output_length is forced. The reusable text is the first cached_length tokens of a
        source: the class pool prefix for rows outside a session and a session's first row; for later rows of a
        session, the most recent earlier prompt of the session that covers cached_length (the server holds
        prompts, not our synthetic replies), else the previous prompt plus a synthetic reply of its
        output_length, else the class pool prefix. Rows are taken in time order, so branching and concurrent
        rows work; nothing waits for responses. A fresh tail (unique nonce + filler; a bare filler when the
        uncached room is too small for the nonce) then fills the prompt up to input_length. Rows without cached_length reuse their class's cached share
        (a session row: its whole source). Shortfalls (a recorded length the layout can't meet) keep
        input_length exact, are counted, and show in the summary.
        """
        overhead = message_overhead(1) if message_overhead else self._overhead
        self._replay_sizing_overhead = overhead if subtract_template else 0
        self._replay_template = overhead
        self._replay: dict[int, dict] = {}
        self._replay_windows: dict[tuple, tuple[int, int]] = {}
        self._sep_tokens = len(self._encode("\n\n"))
        self.replay_shortfall = {"rows": 0, "tokens": 0, "input_rows": 0, "input_tokens": 0, "length_rows": 0}
        # Per class: [first scored rows (a session's first, or a row without a session), of which short, tokens short]
        self.replay_first_rows: dict[str, list] = {}
        self._scored_sessions: set = set()
        self.replay_salt = session_salt or secrets.token_hex(4)
        self._edge_memo: dict[tuple, tuple[str, str]] = {}
        self._junction_memo: dict[tuple, int] = {}
        started = time.perf_counter()
        self._prebuild_replay_windows(entries)
        prompts: dict[str, list] = {}  # session -> [(prompt segments, prompt content tokens)], in time order
        sources: dict[str, list] = {}  # session -> previous prompt + "\n\n" + synthetic reply
        self.lead_in_skipped = 0
        pool_sent: dict[tuple, int] = {}  # (class, prefix) -> longest pool cut sent so far
        for entry in entries:
            session = entry.session
            plan = self._plan_row(entry, prompts.get(session), sources.get(session))
            if entry.lead_in and self._lead_in_covered(entry, plan, prompts.get(session), pool_sent):
                # Something earlier already holds this text: don't send it, and don't count it as sent
                plan["skip"] = True
                self.lead_in_skipped += 1
                self._replay[entry.index] = plan
                continue
            self._replay[entry.index] = plan
            first = plan["segments"][0] if plan["segments"] else None
            if first is not None and first[0] == "pool":
                key = (first[1], first[2])
                pool_sent[key] = max(pool_sent.get(key, 0), first[-1])
            if session is not None:
                prompts.setdefault(session, []).append(
                    (plan["segments"], plan["prompt_tokens"] - self._replay_template)
                )
                sources[session] = plan["segments"] + [("sep",), self._response_segment(plan["output"], entry.index)]
        self.replay_layout_seconds = time.perf_counter() - started
        if self.replay_shortfall["rows"]:
            logger.warning(
                f"--replay-schedule: {self.replay_shortfall['rows']} rows reuse {self.replay_shortfall['tokens']} "
                f"fewer tokens in total than their cached_length (source or input too short); input lengths are exact"
            )
        if self.replay_shortfall["input_rows"]:
            logger.warning(
                f"--replay-schedule: {self.replay_shortfall['input_rows']} rows have an input_length below the chat "
                f"template + nonce + instruction; they are sent {self.replay_shortfall['input_tokens']} tokens longer in "
                f"total than recorded (Replay Input Overflow Rows / Tokens)"
            )

    def _plan_row(self, entry: "ScheduleEntry", prompts: Optional[list], previous: Optional[list]) -> dict:
        ci = entry.class_index
        cls = self.classes[ci]
        prefix_index = entry.prefix_index
        if prefix_index is None and self.pool_sizes[ci]:
            with self._lock:
                prefix_index = self._draw_prefix_index(ci)
        pool = [self._pool_segment(ci, prefix_index)] if self.pool_sizes[ci] and prefix_index is not None else []
        pool_tokens = self.prefix_tokens[ci] if pool else 0
        recorded = entry.cached_length
        want = self._replay_want(entry)  # None: the class's cached share, or a session's whole source
        # The restore margin is headroom, not a requirement: a source covering the aligned length itself will do
        need = want - RESTORE_BOUNDARY_MARGIN_TOKENS if want and self.restore_boundary else want
        source = self._replay_source(need, prompts, previous, pool, pool_tokens)
        if want is None:
            want = sum(self._segment_tokens(seg) for seg in source)
        output = entry.output_length or cls.generated_tokens
        prompt = entry.input_length
        content = None if prompt is None else prompt - self._replay_sizing_overhead
        if content is not None:
            want = min(want, max(0, content - self._bare_min()))
        segments, reused = self._cut(source, want)
        # Segment token counts add up except where text merges across a boundary (".\n\n" is one Llama-3 token);
        # measure each boundary so the reused count is exact and the fresh tail makes up the rest
        reused += sum(self._segment_junction(x, y) for x, y in zip(segments, segments[1:]))
        tail_start = SHAPE_MIX_TAIL_START if reused else SHAPE_MIX_TAIL_START.lstrip()
        min_tail = self._min_tail_tokens_for(tail_start)
        variant = entry.index % REPLAY_FILLER_VARIANTS
        if content is not None and reused and content - reused < min_tail + 2:
            # Too little uncached room for the nonce: a bare filler keeps the recorded cached length. Rare, so the
            # whole prompt is encoded to confirm the length (and adjusted once); what's still off is counted
            if content - reused < self._bare_min():
                self.replay_shortfall["input_rows"] += 1
                self.replay_shortfall["input_tokens"] += self._bare_min() - (content - reused)
            # The row's own length, short of what's already counted as input overflow (N8-140-5: checking against
            # the planned length missed tails whose junction didn't settle)
            target = reused + max(content - reused, self._bare_min())
            tail_segment = self._bare_tail(content - reused, segments[-1], variant)
            tail = tail_segment[-1]
            for attempt in range(2):
                actual = len(self._encode("".join(self._render_segment(sg) for sg in segments + [tail_segment])))
                joint = actual - reused - tail  # report what is actually sent
                off = actual - target
                if not off:
                    break
                if attempt:
                    self.replay_shortfall["length_rows"] += 1
                    break
                tail_segment = self._bare_tail(tail - off, segments[-1], variant)
                tail = tail_segment[-1]
        else:
            joint = (
                self._junction_delta(self._edges(segments[-1])[1], self._row_nonce_text(tail_start, entry.index))
                if segments
                else 0
            )
            tail = (content - reused - joint) if content is not None else self._tail_tokens(ci) + (pool_tokens - reused)
            if tail < min_tail:
                # input_length below template + nonce + instruction: sent longer than recorded, and counted
                if content is not None:
                    self.replay_shortfall["input_rows"] += 1
                    self.replay_shortfall["input_tokens"] += min_tail - tail
                tail = min_tail
            tail_segment = self._tail_segment(tail_start, tail, entry.index, variant)
        reusable = self._reusable_of(reused)
        target = None if recorded is None else self._aligned_cached(recorded)
        first_scored = not entry.lead_in and (entry.session is None or entry.session not in self._scored_sessions)
        if not entry.lead_in and entry.session is not None:
            self._scored_sessions.add(entry.session)
        if target is not None and first_scored and target > 0:
            first = self.replay_first_rows.setdefault(cls.name, [0, 0, 0])
            first[0] += 1
            if reusable < target:
                first[1] += 1
                first[2] += target - reusable
        if target is not None and reusable < target and not entry.lead_in:  # lead-ins are unscored
            self.replay_shortfall["rows"] += 1
            self.replay_shortfall["tokens"] += target - reusable
        segments = segments + [tail_segment]
        return {
            "class": ci,
            "prefix_index": prefix_index,
            "segments": segments,
            "prompt_tokens": self._replay_template + reused + joint + tail,
            "reusable": max(0, reusable),
            "recorded_cached": recorded,
            "output": output,
        }

    def _lead_in_covered(self, entry: "ScheduleEntry", plan: dict, prompts: Optional[list], pool_sent: dict) -> bool:
        """A lead-in is covered, and not sent, when
        - an earlier sent prompt of its own session is at least as long (the session's private text is held), or
        - it is a shared-prefix warm-up (cached_length == input_length: nothing private) and an earlier row, of any
          session, already sent that class pool cut at least as long (D-50: shared prefixes are cross-session).
        A remainder lead-in (cached_length < input_length: the shared part plus the session's own context) always
        carries private text, so only the first rule can cover it."""
        content = plan["prompt_tokens"] - self._replay_template
        if entry.session is not None and any(tokens >= content for _, tokens in prompts or ()):
            return True
        shared_only = entry.session is None or (
            entry.cached_length is not None and entry.cached_length >= entry.input_length
        )
        first = plan["segments"][0] if plan["segments"] else None
        if shared_only and first is not None and first[0] == "pool":
            return pool_sent.get((first[1], first[2]), 0) >= first[-1]
        return False

    def lead_in_requests(self, entries: tuple) -> list[tuple[Optional[str], "ShapeRequest"]]:
        """(session, request) for each lead-in row to send, in time order."""
        out = []
        for entry in entries:
            plan = self._replay.get(entry.index)
            if entry.lead_in and plan is not None and not plan.get("skip"):
                out.append((entry.session, self.replay_request(entry)))
        return out

    def _replay_want(self, entry: "ScheduleEntry") -> Optional[int]:
        """Content tokens a row should share with what was sent before (None: the class or session default)."""
        if entry.cached_length is None:
            return None
        return self._shared_target(entry.cached_length)

    def _prebuild_replay_windows(self, entries: tuple) -> None:
        """Build, in one multi-threaded batch, the tail windows rows will most likely need (their tail length if
        the reused part comes out exactly as wanted); the row-by-row layout then mostly finds them built."""
        keys = set()
        bare_min = self._bare_min()
        for entry in entries:
            want = self._replay_want(entry)
            if entry.input_length is None or want is None:
                continue
            content = entry.input_length - self._replay_sizing_overhead
            want = min(want, max(0, content - bare_min))
            start = SHAPE_MIX_TAIL_START if want else SHAPE_MIX_TAIL_START.lstrip()
            min_tail = self._min_tail_tokens_for(start)
            variant = entry.index % REPLAY_FILLER_VARIANTS
            if want and content - want < min_tail + 2:
                if content - want >= self._trailer_tokens + 2:
                    keys.add(("bare", "\n", content - want, variant))
            else:
                keys.add(("tail", start, max(content - want, min_tail), variant))
        missing = sorted(k for k in keys if k not in self._replay_windows)
        if missing:
            self._build_replay_windows(missing)

    def _replay_source(
        self, want: Optional[int], prompts: Optional[list], previous: Optional[list], pool: list, pool_tokens: int
    ) -> list:
        """What a row reuses from: the session's most recent earlier prompt covering `want`, else the previous
        prompt plus its synthetic reply, else the class pool prefix, else the longest of those."""
        if not prompts:
            return pool
        if want is None:
            return previous
        for segments, tokens in reversed(prompts):
            if tokens >= want:
                return segments
        previous_tokens = sum(self._segment_tokens(seg) for seg in previous)
        if previous_tokens >= want or previous_tokens >= pool_tokens:
            return previous
        return pool

    def _bare_min(self) -> int:
        # A newline, filler and the instruction; without an instruction a single one-token word will do
        return self._trailer_tokens + 2 if self.trailer else 1

    def _bare_tail(self, n: int, left: tuple, variant: int) -> tuple:
        """A tail of exactly n tokens after `left` with no nonce: a newline, filler and the instruction."""
        if n <= 1 and not self.trailer and self._nonce_words:
            return ("word", self._nonce_words[variant % len(self._nonce_words)], 1)
        n = max(n, self._trailer_tokens + 2)
        segment = None
        for _ in range(3):
            offset, width = self._replay_window("bare", "\n", n, variant)
            segment = ("bare", offset, width, True, n)
            fixed = n - self._segment_junction(left, segment)
            if fixed == n or fixed < self._trailer_tokens + 2:
                break
            n = fixed
        return segment

    def _edges(self, seg: tuple) -> tuple[str, str]:
        edges = self._edge_memo.get(seg)
        if edges is None:
            text = self._render_segment(seg)
            edges = self._edge_memo[seg] = (text[:120], text[-120:])
        return edges

    def _segment_junction(self, left: tuple, right: tuple) -> int:
        key = (left, right)
        delta = self._junction_memo.get(key)
        if delta is None:
            delta = self._junction_memo[key] = self._junction_delta(self._edges(left)[1], self._edges(right)[0])
        return delta

    def _row_nonce_text(self, tail_start: str, origin: int) -> str:
        # Deterministic per (request seed, pod, row): a reused tail renders exactly as it was sent
        nonce = random.Random(f"shape-mix:{self.request_seed}:replay:{self._worker_index}:{origin}").getrandbits(64)
        return self._nonce_text(tail_start, nonce, origin)

    def _junction_delta(self, left: str, right: str) -> int:
        """Tokens gained (<= 0) by encoding `left + right` rather than each on its own, measured on the text
        around the boundary (cut at whitespace, where pretokens start)."""
        a = left[-120:]
        space = next((i for i, ch in enumerate(a) if ch.isspace()), None)
        a = a[space:] if space is not None and len(left) > 120 else a
        b = right[:120]
        if len(right) > 120:
            end = max(b.rfind(" "), b.rfind("\n"))
            b = b[:end] if end > 0 else b
        if not a or not b:
            return 0
        return len(self._encode(a + b)) - len(self._encode(a)) - len(self._encode(b))

    def _pool_segment(self, ci: int, j: int, width: Optional[int] = None, tokens: Optional[int] = None) -> tuple:
        if self._prefix_widths[ci][j] == _UNBUILT:
            self.prefix(ci, j)  # builds its window
        if width is None:
            width, tokens = self._prefix_widths[ci][j], self.prefix_tokens[ci]
        return ("pool", ci, j, width, tokens)

    def _tail_segment(self, tail_start: str, n: int, origin: int, variant: int = 0) -> tuple:
        offset, width = self._replay_window("tail", tail_start, n, variant)
        return ("tail", tail_start, offset, width, origin, True, n)

    def _response_segment(self, n: int, origin: int = 0) -> tuple:
        offset, width = self._replay_window("assistant", "", n, origin % REPLAY_FILLER_VARIANTS)
        return ("resp", offset, width, n)

    def _segment_tokens(self, seg: tuple) -> int:
        return self._sep_tokens if seg[0] == "sep" else seg[-1]

    def _cut(self, source: list, want: int) -> tuple[list, int]:
        """The first `want` tokens of `source`, as whole segments plus one truncated segment (the same window
        with fewer words, so the text is a byte prefix of what was sent). May come up a few tokens short when a
        multi-token word straddles the cut, or when the source is shorter."""
        out, used = [], 0
        for seg in source:
            n = self._segment_tokens(seg)
            if used + n <= want:
                out.append(seg)
                used += n
                continue
            part = self._truncate(seg, want - used)
            if part is not None:
                out.append(part)
                used += self._segment_tokens(part)
            break
        if out and out[-1][0] == "sep":
            # A cut ending on the separator would join the tail to a bare "\n\n", whose tokens merge with the text
            # before it in ways a pairwise boundary measurement misses
            used -= self._segment_tokens(out.pop())
        return out, used

    def _truncate(self, seg: tuple, tokens: int) -> Optional[tuple]:
        kind = seg[0]
        if kind == "pool":
            _, ci, j, _, _ = seg
            lead = (
                _prefix_header(ci, j)
                if self.prefix_tokens[ci] > _MAX_PREFIX_HEADER_TOKENS
                else self._prefix_lead(ci, j)[0]
            )
            fit = self._fit_width(lead, self._prefix_offsets[ci][j], tokens)
            return None if fit is None else ("pool", ci, j, fit[0], fit[1])
        if kind == "tail":
            _, start, offset, _, origin, _, _ = seg
            fit = self._fit_width(self._row_nonce_text(start, origin) + " ", offset, tokens)
            return None if fit is None else ("tail", start, offset, fit[0], origin, False, fit[1])
        if kind == "resp":
            fit = self._fit_width("", seg[1], tokens)
            return None if fit is None else ("resp", seg[1], fit[0], fit[1])
        if kind == "bare":
            fit = self._fit_width("\n", seg[1], tokens)
            return None if fit is None else ("bare", seg[1], fit[0], False, fit[1])
        return None

    def _fit_width(self, lead: str, offset: int, tokens: int) -> Optional[tuple[int, int]]:
        """Widest window from `offset` whose text (after `lead`) has at most `tokens` tokens: (width, tokens)."""
        lead_tokens = len(self._encode(lead)) if lead else 0
        if tokens <= lead_tokens:
            return None
        width = self._corpus.width_for(offset, tokens - lead_tokens)
        best = None
        for _ in range(6):
            if width <= 0:
                break
            got = len(self._encode(lead + self._corpus.slice(offset, width)))
            if got <= tokens and (best is None or got > best[1]):
                best = (width, got)
            if got == tokens:
                break
            step = tokens - got
            width = width + (
                step if abs(step) <= 4 else round(step / max(0.5, self._corpus.cycle_tokens / self._corpus.words))
            )
        return best

    def _replay_window(self, kind: str, tail_start: str, n: int, variant: int = 0) -> tuple[int, int]:
        key = (kind, tail_start, n, variant)
        if key not in self._replay_windows:
            self._build_replay_windows([key])
        return self._replay_windows[key]

    def _min_tail_tokens_for(self, tail_start: str) -> int:
        nonce = (self.fixed_nonce_tokens or {}).get(tail_start, SHAPE_MIX_TAIL_RESERVE_TOKENS // 2)
        return nonce + self._trailer_tokens + 2

    def _build_replay_windows(self, needed: list[tuple]) -> None:
        jobs = []
        for kind, tail_start, n, variant in needed:
            if kind == "tail":
                lead = self._nonce_text(tail_start, 0, 0) + " "
                trailer, trailer_tokens = self.trailer, self._trailer_tokens
            elif kind == "bare":
                lead, trailer, trailer_tokens = "\n", self.trailer, self._trailer_tokens
            else:
                lead, trailer, trailer_tokens = "", "", 0
            jobs.append(
                _WindowSearch(
                    offsets=array("I", [0]),
                    widths=array("I", [_UNBUILT]),
                    index=0,
                    lead=lead,
                    lead_tokens=len(self._encode(lead)) if lead else 0,
                    num_tokens=n,
                    rng=random.Random(
                        f"shape-mix:{self.mix.seed}:replay:{kind}:{tail_start}:{n}" + (f":{variant}" if variant else "")
                    ),
                    check_tail_start=False,
                    trailer=trailer,
                    trailer_tokens=trailer_tokens,
                )
            )
        self._run_searches(jobs)
        for key, job in zip(needed, jobs):
            self._replay_windows[key] = (job.offsets[0], job.widths[0])

    def _render_segment(self, seg: tuple) -> str:
        kind = seg[0]
        if kind == "pool":
            _, ci, j, width, _ = seg
            lead = (
                _prefix_header(ci, j)
                if self.prefix_tokens[ci] > _MAX_PREFIX_HEADER_TOKENS
                else self._prefix_lead(ci, j)[0]
            )
            return lead + self._corpus.slice(self._prefix_offsets[ci][j], width)
        if kind == "tail":
            _, start, offset, width, origin, with_trailer, _ = seg
            return (
                self._row_nonce_text(start, origin)
                + " "
                + self._corpus.slice(offset, width)
                + (self.trailer if with_trailer else "")
            )
        if kind == "sep":
            return "\n\n"
        if kind == "bare":
            _, offset, width, with_trailer, _ = seg
            return "\n" + self._corpus.slice(offset, width) + (self.trailer if with_trailer else "")
        if kind == "word":
            return " " + seg[1]
        return self._corpus.slice(seg[1], seg[2])

    def replay_request(self, entry: "ScheduleEntry") -> ShapeRequest:
        """The request for a --replay-schedule row (see prepare_replay)."""
        plan = getattr(self, "_replay", {}).get(entry.index)
        if plan is None:
            return self.scheduled_request(entry.class_index, entry.prefix_index, entry.output_length)
        ci = plan["class"]
        cls = self.classes[ci]
        prefix_index = plan["prefix_index"]
        if entry.session is not None:
            affinity = f"shape-{self.replay_salt}-session-{entry.session}"
        elif plan["reusable"] and prefix_index is not None and self.pool_sizes[ci] > 1:
            affinity = f"shape-{self.mix.seed}-{cls.name}-{prefix_index}"
        else:
            affinity = None
        overrides = {
            k: v
            for k, v in (
                ("temperature", entry.temperature),
                ("top_p", entry.top_p),
                ("reasoning_effort", entry.reasoning_effort),
            )
            if v is not None
        }
        return ShapeRequest(
            prompt="".join(self._render_segment(seg) for seg in plan["segments"]),
            prompt_tokens=plan["prompt_tokens"],
            class_index=ci,
            class_name=cls.name,
            generated_tokens=plan["output"],
            prefix_index=prefix_index,
            affinity_key=affinity,
            reusable_tokens=plan["reusable"],
            recorded_cached_tokens=plan["recorded_cached"],
            overrides=overrides or None,
            cancelled=entry.cancelled,
        )

    def fill_plan(self) -> list[tuple[int, int]]:
        """Every (class, prefix) once, classes interleaved so each warms at the same pace."""
        return [
            (ci, j) for j in range(max(self.pool_sizes, default=0)) for ci, n in enumerate(self.pool_sizes) if j < n
        ]

    def fill_request(self, class_index: int, prefix_index: int) -> ShapeRequest:
        """A request on one prefix shaped like live traffic (prefix + unique tail), generating one token.

        Live traffic diverges right after the prefix, so filling with the same shape leaves the cache as
        live traffic would (a fill ending at the prefix would keep e.g. DeepSeek V4's sliding window there)."""
        with self._lock:
            counter = self._counter
            self._counter += 1
            nonce = self._fill_rng.getrandbits(64)
            variant = self._draw_variant(class_index, self._fill_rng)
        return self._render(class_index, prefix_index, nonce, counter, variant, 1)

    def _render(self, ci, prefix_index, nonce, counter, variant, generated_tokens, tail_tokens=None) -> ShapeRequest:
        cls = self.classes[ci]
        pool_size = self.pool_sizes[ci]
        prefix = self.prefix(ci, prefix_index) if pool_size else ""
        if tail_tokens is None:
            tail_tokens = self._filler_tokens[ci][variant] if variant is not None else self._tail_tokens(ci)
        nonce_text = self._nonce_text(self._tail_start(ci), nonce, counter)
        if variant is not None:
            tail = (
                nonce_text
                + " "
                + self._corpus.slice(self._filler_offsets[ci][variant], self._filler_widths[ci][variant])
                + self.trailer
            )
        else:
            if len(self._encode(nonce_text)) + self._trailer_tokens >= tail_tokens:
                raise RuntimeError(
                    f"--shape-mix class {cls.name}: no room for the unique nonce in a {tail_tokens}-token tail"
                )
            tail = self._fit_tail(nonce_text, tail_tokens, nonce)
        prompt_tokens = self._overhead + self.prefix_tokens[ci] + tail_tokens
        if self.context_tokens and prompt_tokens + generated_tokens > self.context_tokens:
            # A sampled long prompt with a sampled long output: cut the output to fit the context (counted)
            generated_tokens = max(1, self.context_tokens - prompt_tokens)
            self.context_cut_requests += 1
        return ShapeRequest(
            prompt=prefix + tail,
            prompt_tokens=prompt_tokens,
            class_index=self._report[ci],
            class_name=self.classes[self._report[ci]].name,
            generated_tokens=generated_tokens,
            prefix_index=prefix_index,
            # A single key would pin the class's whole share to one replica, so only pools of 2+ get one
            affinity_key=f"shape-{self.mix.seed}-{cls.name}-{prefix_index}" if pool_size > 1 else None,
            reusable_tokens=self.reusable_tokens[ci],
        )

    def __iter__(self):
        return self


@dataclass
class _WindowSearch:
    """State of one exact-length window search (see ShapeMixDataset._run_search_batch)."""

    offsets: array
    widths: array
    index: int
    lead: str
    lead_tokens: int
    num_tokens: int
    rng: random.Random
    check_tail_start: bool
    offset: int = 0
    width: int = 0
    attempts: int = 0
    offsets_tried: int = 0
    corpus: Optional[_CyclicText] = None
    trailer: str = ""
    trailer_tokens: int = 0

    def next_offset(self, corpus_words: int) -> None:
        if self.offsets_tried == _PREFIX_OFFSET_ATTEMPTS:
            raise ValueError(f"--shape-mix: could not build a {self.num_tokens}-token window with the given tokenizer")
        self.offsets_tried += 1
        self.offset = self.rng.randrange(corpus_words)
        self.width = self.corpus.width_for(self.offset, self.num_tokens - self.lead_tokens - self.trailer_tokens)
        self.attempts = 0

    def adjust(self, got: int) -> None:
        """Re-predict the width, correcting the corpus estimate by how far the last text was off."""
        fixed = self.lead_tokens + self.trailer_tokens
        error = got - fixed - self.corpus.tokens_between(self.offset, self.width)
        width = self.corpus.width_for(self.offset, self.num_tokens - fixed - error)
        # No new width to try (a multi-token word straddles the target): spend the attempts on another offset
        self.attempts = _FIT_ATTEMPTS if width == self.width else self.attempts + 1
        self.width = width


def _process_rss_bytes(field: str = "VmRSS") -> Optional[int]:
    """Current (VmRSS) or peak (VmHWM) resident memory of this process, if /proc is available."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _release_free_memory() -> None:
    # Building prefixes churns through tokenizer buffers that glibc keeps mapped unless asked to trim
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def build_shape_mix_request(provider, request: ShapeRequest, output_mode: str = "exact") -> tuple[dict, Optional[dict]]:
    """Payload for the class's output length, plus the affinity header that routes by shared prefix.

    "exact" forces generated_tokens (max_tokens = min_tokens, ignore_eos); "max" only caps the output at
    generated_tokens and lets the model stop, dropping the ignore_eos some providers add on their own.
    """
    data = provider.format_payload(request.prompt, request.generated_tokens, None)
    if request.messages is not None:
        data["messages"] = request.messages
    if request.overrides:
        data.update(request.overrides)
    if output_mode == "max":
        data.pop("min_tokens", None)
        data.pop("ignore_eos", None)
    else:
        data["min_tokens"] = request.generated_tokens
        data["ignore_eos"] = True
    headers = {"x-session-affinity": request.affinity_key} if request.affinity_key is not None else None
    return data, headers


class DatasetHolder:
    _instance = None
    _forced_generation_instance = None
    _shape_mix_instance = None
    _shape_mix_lock = threading.Lock()

    @classmethod
    def get_shape_mix_instance(cls, options: argparse.Namespace, worker_index: int) -> ShapeMixDataset:
        with cls._shape_mix_lock:
            if cls._shape_mix_instance is None:
                tokenizer = InitTracker.load_tokenizer(options.tokenizer)
                tokenizer_path = options.tokenizer or DEFAULT_TOKENIZER
                overhead = head = 0
                if options.chat:
                    overhead = len(empty_chat_template_token_ids(tokenizer, tokenizer_path))
                    head = chat_template_head_tokens(tokenizer, tokenizer_path)
                mix = parse_shape_mix(options.shape_mix, chat_overhead_tokens=overhead)
                pool_sizes = None
                if getattr(options, "replay_schedule", None):
                    names = tuple(c.name for c in mix.classes)
                    pool_sizes = replay_pool_prefix_tokens(options, replay_entries(options, names), names)
                    mix = replay_sized_mix(mix, pool_sizes)
                cls._shape_mix_instance = ShapeMixDataset(
                    mix=mix,
                    tokenizer=tokenizer,
                    corpus_path=resolve_shape_mix_corpus(getattr(options, "shape_mix_corpus", None)),
                    chat_overhead_tokens=overhead,
                    chat_head_tokens=head,
                    worker_index=worker_index,
                    request_seed=resolve_request_seed(options),
                    reuse_overhang_tokens=resolve_prefix_reuse_overhang(options),
                    restore_boundary=resolve_prefix_restore_boundary(options),
                    pool_cache=getattr(options, "prefix_pool_cache", None),
                    instruction=resolve_shape_mix_instruction(getattr(options, "shape_mix_instruction", None)),
                    sample_lengths=getattr(options, "shape_mix_sample_lengths", False),
                    default_popularity_zipf=getattr(options, "prefix_popularity_zipf", None),
                    tail_variants=getattr(options, "shape_mix_tail_variants", SHAPE_MIX_TAIL_VARIANTS),
                    context_tokens=resolve_context_tokens(options),
                    # Replay rows carry their own cached lengths: its class pools stay at the floor
                    split_restore_classes=not getattr(options, "replay_schedule", None),
                )
                ds = cls._shape_mix_instance
                ds.replay_pool_sizes = pool_sizes
                if pool_sizes:
                    logger.info(
                        f"--replay-schedule: pool prefixes sized to the longest recorded shared length "
                        f"({getattr(options, 'replay_pool_prefix', 'max')}) per class: "
                        + ", ".join(f"{n} {t}" for n, t in sorted(pool_sizes.items())[:12])
                        + (" ..." if len(pool_sizes) > 12 else "")
                    )
                if ds.reuse_overhang_tokens:
                    logger.info(
                        f"--shape-mix: prefixes are {ds.reuse_overhang_tokens} tokens longer than cached_prompt_tokens "
                        f"(prefix reuse overhang), so the server reuses exactly the cached share"
                    )
                if ds.restore_boundary:
                    logger.info(
                        f"--shape-mix: the server restores cached KV in {ds.restore_boundary}-token blocks; cached "
                        f"lengths are aligned down to a block and shared prefixes run {RESTORE_BOUNDARY_MARGIN_TOKENS} "
                        f"tokens past it"
                    )
                logger.info(
                    f"--shape-mix: building {sum(ds.pool_sizes)} prefixes "
                    f"({ds.pool_tokens / 1e6:.1f}M tokens) before traffic starts"
                )
                seconds = ds.build_seconds = ds.build_prefixes()
                if ds.pool_cache:
                    logger.info(
                        f"--prefix-pool-cache {ds.pool_cache}: loaded {ds.cache_loaded_prefixes} of "
                        f"{sum(ds.pool_sizes)} prefixes, built {sum(ds.pool_sizes) - ds.cache_loaded_prefixes}"
                    )
                _release_free_memory()
                rss, peak = _process_rss_bytes(), _process_rss_bytes("VmHWM")
                logger.info(
                    f"--shape-mix: built {ds.built_prefixes} prefixes ({ds.pool_tokens / 1e6:.1f}M tokens) in "
                    f"{seconds:.1f}s; pool resident size {ds.resident_bytes() / 2**20:.2f} MiB"
                    + (f", process RSS {rss / 2**20:.0f} MiB (peak {peak / 2**20:.0f} MiB)" if rss and peak else "")
                )
                if getattr(options, "replay_schedule", None):
                    # After the pool: replay rows reuse (truncated) pool prefixes
                    cls._prepare_replay(ds, options, tokenizer, tokenizer_path)
            return cls._shape_mix_instance

    @staticmethod
    def _prepare_replay(ds: "ShapeMixDataset", options: argparse.Namespace, tokenizer, tokenizer_path: str) -> None:
        names = tuple(c.name for c in ds.mix.classes)
        entries = replay_entries(options, names)
        overheads: dict[int, int] = {}

        def message_overhead(n: int) -> int:
            if n not in overheads:
                overheads[n] = chat_messages_overhead(tokenizer, tokenizer_path, n)
            return overheads[n]

        ds.prepare_replay(
            entries,
            message_overhead=message_overhead if options.chat else None,
            subtract_template=options.replay_template_subtraction,
            session_salt=getattr(options, "replay_session_salt", None),
        )
        sessions = len({e.session for e in entries if e.session is not None})
        pods = sorted({e.pod for e in entries if e.pod is not None}, key=_pod_sort_key)
        where = (
            (f"recorded pod {pods[0]}" if len(pods) == 1 else f"MERGED recorded pods {', '.join(pods[:8])}")
            if pods
            else (f"generator {options.replay_pod_index} of {options.replay_pods}")
        )
        defaulted = sum(1 for e in entries if e.defaulted)
        lead_ins = sum(1 for e in entries if e.lead_in)
        logger.info(
            f"--replay-schedule: {where} replays {len(entries) - lead_ins} requests ({sessions} sessions"
            + (f", plus {lead_ins} lead-in rows before time 0" if lead_ins else "")
            + "); laid out in "
            f"{ds.replay_layout_seconds:.1f}s ({1000 * ds.replay_layout_seconds / max(1, len(entries)):.2f} ms/row)"
            + (f"; {defaulted} rows had invalid optional fields dropped" if defaulted else "")
        )

    @classmethod
    def _create_dataset(cls, options: argparse.Namespace):
        if options.dataset.startswith("@"):
            return JsonlDataset(
                options.dataset[1:],
                shuffle_seed=getattr(options, "dataset_shuffle_seed", None),
                dataset_limit=getattr(options, "dataset_limit", None),
            )
        elif options.dataset in ("limericks", "code"):
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

            common_tokens = options.prompt_cache_max_len
            if common_tokens > options.prompt_tokens:
                common_tokens = options.prompt_tokens
            tokenizer_path = options.tokenizer or DEFAULT_TOKENIZER
            return TranslationDataset(
                path=os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_file),
                prompt="\n\n" + prompt,
                tokenizer=InitTracker.load_tokenizer(options.tokenizer),
                tokenizer_path=tokenizer_path,
                chat=options.chat and not getattr(options, "rerank", False),
                num_tokens=options.prompt_tokens,
                common_tokens=common_tokens,
            )
        else:
            raise ValueError(f"Unknown dataset: {options.dataset}")

    @classmethod
    def get_instance(cls, options: argparse.Namespace):
        if cls._instance is None:
            cls._instance = cls._create_dataset(options)
        return cls._instance

    _prompt_tokens_estimate = None

    @classmethod
    def prompt_token_estimate(cls, options: argparse.Namespace, get_input) -> int:
        """Prompt tokens to report when the server doesn't: the dataset's own count when it has one, else one
        prompt tokenized, once per process."""
        if cls._prompt_tokens_estimate is None:
            prompt, prompt_tokens, _ = get_input()
            if not prompt_tokens:
                tokenizer = InitTracker.load_tokenizer(options.tokenizer)
                prompt_tokens = len(tokenizer.encode(prompt if isinstance(prompt, str) else str(prompt)))
            cls._prompt_tokens_estimate = prompt_tokens
        return cls._prompt_tokens_estimate

    @classmethod
    def get_forced_generation_instance(cls, dataset, tokenizer, max_tokens):
        if cls._forced_generation_instance is None:
            if dataset.startswith("@"):
                raise ValueError(
                    "--forced-generation-from-dataset only works with 'limericks' or 'code' datasets, "
                    "not JSONL files. Use --forced-generation-file instead."
                )
            dataset_files = {"limericks": "limericks.txt", "code": "code.txt"}
            if dataset not in dataset_files:
                raise ValueError(
                    f"--forced-generation-from-dataset requires --dataset to be one of "
                    f"{list(dataset_files.keys())}, got '{dataset}'"
                )
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_files[dataset])
            cls._forced_generation_instance = TranslationDataset(
                path=path,
                prompt="",
                tokenizer=InitTracker.load_tokenizer(tokenizer),
                tokenizer_path=tokenizer or DEFAULT_TOKENIZER,
                chat=False,
                num_tokens=max_tokens,
                common_tokens=0,
            )
        return cls._forced_generation_instance


def sample_interarrival(distribution: str, mean_wait: float, cv2: float = 1.0, rng=random) -> float:
    """One inter-arrival time with the given mean; for 'gamma', cv2 is its squared coefficient of variation."""
    if distribution == "exponential":
        return rng.expovariate(1 / mean_wait)
    if distribution == "gamma":
        # shape k = 1/cv2, scale = mean * cv2: mean stays mean_wait, variance / mean^2 = cv2 (k=1 is exponential)
        return rng.gammavariate(1 / cv2, mean_wait * cv2)
    if distribution == "uniform":
        return rng.uniform(0, 2 * mean_wait)
    if distribution == "constant":
        return mean_wait
    raise ValueError(f"Unknown distribution {distribution}")


def base_process_cv2(distribution: str, cv2: float = 1.0) -> float:
    """Squared coefficient of variation of the inter-arrival times of each --qps-distribution."""
    return {"constant": 0.0, "uniform": 1 / 3, "exponential": 1.0, "gamma": cv2}[distribution]


def modulation_rate_variance(
    qps: float, dispersion: float, base_cv2: float, window: float, horizon: float = 10.0
) -> float:
    """Variance of the per-window rate that gives `horizon`-second counts an index of dispersion `dispersion`.

    With the rate redrawn every `window` seconds and arrivals of squared CV `base_cv2` within a window, a
    randomly aligned count window of length T has Var/mean = base_cv2 + var * g(T, W) / (qps * T), where
    g = T^2 - T^3 / (3W) for T <= W and T*W - W^2 / 3 for T >= W (the integral of the rate's triangular
    autocovariance). Returns 0 if the arrivals alone already reach the target.
    """
    t, w = horizon, window
    g = t * t - t**3 / (3 * w) if t <= w else t * w - w * w / 3
    return max(0.0, (dispersion - base_cv2) * qps * t / g)


def modulated_base_cv2(qps: float, dispersion: float, target_cv2: float, window: float) -> tuple[float, float]:
    """Within-window arrival CV^2 and rate variance for a modulated schedule whose overall gap CV^2 is
    `target_cv2` and whose 10 s counts have index of dispersion `dispersion`.

    With rates gamma(k) of mean q, gaps of a window at rate r have mean 1/r and CV^2 c, and a window holds
    gaps in proportion to r, so the overall gap CV^2 is (1 + c) * q * E[1/r] - 1 = (1 + c) * k / (k - 1) - 1.
    Solved jointly with modulation_rate_variance (which depends on c); c is floored at
    _MIN_MODULATED_BASE_CV2 when the target is out of reach.
    """
    c = target_cv2
    variance = 0.0
    for _ in range(50):
        variance = modulation_rate_variance(qps, dispersion, c, window)
        if variance == 0:
            return target_cv2, 0.0
        k = qps * qps / variance
        wanted = (1 + target_cv2) * (k - 1) / k - 1 if k > 1 else _MIN_MODULATED_BASE_CV2
        wanted = max(_MIN_MODULATED_BASE_CV2, wanted)
        if abs(wanted - c) < 1e-9:
            break
        c = wanted
    return c, variance


def modulated_gap_cv2(qps: float, variance: float, base_cv2: float) -> float:
    """Overall gap CV^2 of a modulated schedule (see modulated_base_cv2); infinite for gamma shape <= 1."""
    if not variance:
        return base_cv2
    k = qps * qps / variance
    return (1 + base_cv2) * k / (k - 1) - 1 if k > 1 else math.inf


class FixedQPSPacer:
    """Open-loop send schedule: `claim()` hands out the next intended send time.

    Inter-arrival times follow `distribution` with mean 1/qps (`cv2` for gamma). With `modulation_window`,
    the rate is redrawn every that many seconds from a gamma with mean qps, sized so 10-second counts reach
    index of dispersion `dispersion`; arrivals are time-rescaled through the rate. For exponential and gamma
    arrivals the within-window CV^2 is lowered so the overall gap CV^2 still equals `cv2` (1 for
    exponential), since rate variation alone adds to it. Rates are drawn in blocks of `rate_block_windows`
    windows (the whole run, when its length is known) and scaled so each block averages exactly qps, so a
    load tier sends its nominal rate. All draws are seeded from `seed`.

    `resync()` moves the rest of the schedule later after a stall, so the backlog isn't sent as a burst.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        qps,
        distribution,
        cv2=1.0,
        seed=None,
        modulation_window=None,
        dispersion=None,
        rate_block_windows=None,
    ):
        if distribution == "gamma" and not cv2 > 0:
            raise ValueError(f"--qps-cv2 must be > 0, got {cv2}")
        if modulation_window is not None and not modulation_window > 0:
            raise ValueError(f"--qps-modulation-window must be > 0, got {modulation_window}")
        if modulation_window is not None and dispersion is None:
            raise ValueError(
                "--qps-modulation-window needs --qps-dispersion (or arrivals.dispersion_10s in --shape-mix)"
            )
        sample_interarrival(distribution, 1.0, cv2)
        self.qps = qps
        self.distribution = distribution
        self.cv2 = cv2
        self.seed = seed
        self.modulation_window = modulation_window
        self.dispersion = dispersion
        # Seeded so runs compared against each other (e.g. incumbent vs candidate) see the same arrivals
        self.rng = random.Random(f"arrivals:{seed}") if seed is not None else random.Random()
        self._rate_rng = random.Random(f"arrivals:{seed}:rate") if seed is not None else random.Random()
        self.rate_variance = 0.0
        self.base_distribution, self.base_cv2 = distribution, cv2
        self.rate_block_windows = max(2, int(rate_block_windows or _DEFAULT_RATE_BLOCK_WINDOWS))
        if modulation_window is not None:
            if distribution in ("exponential", "gamma"):
                target = base_process_cv2(distribution, cv2)
                self.base_cv2, self.rate_variance = modulated_base_cv2(qps, dispersion, target, modulation_window)
                self.base_distribution = "gamma"
                expected = modulated_gap_cv2(qps, self.rate_variance, self.base_cv2)
                if expected > target * 1.05:
                    logger.warning(
                        f"--qps-dispersion {dispersion} at {qps} QPS varies the rate so much that the gap CV^2 "
                        f"can't stay at {target:g}; expect about {expected:.2f}"
                    )
            else:
                self.rate_variance = modulation_rate_variance(
                    qps, dispersion, base_process_cv2(distribution, cv2), modulation_window
                )
            if self.rate_variance == 0:
                logger.warning(
                    f"--qps-dispersion {dispersion} is already reached by {distribution} arrivals "
                    f"(cv2 {base_process_cv2(distribution, cv2):g}); the rate is not modulated"
                )
        self.start = None
        self.epoch = 0
        self.shifted_seconds = 0.0
        self.shifts: list[float] = []
        self._pending: dict[int, tuple[float, int]] = {}
        self._pending_seq = 0
        self._rate_block: list[float] = []
        self._unit_time = 0.0
        self._window_start = 0.0
        self._window_unit_start = 0.0
        self._window_rate = self._draw_rate()

    def _draw_rate(self) -> float:
        if not self.rate_variance:
            return self.qps
        if not self._rate_block:
            k = self.rate_block_windows
            # Scaling a block to its mean removes about 1/k of the variance; draw wider to keep the target
            variance = self.rate_variance * k / (k - 1)
            shape = self.qps * self.qps / variance
            draws = [self._rate_rng.gammavariate(shape, self.qps / shape) for _ in range(k)]
            mean = sum(draws) / k
            self._rate_block = [max(d * self.qps / mean, 1e-9) for d in reversed(draws)]
        return self._rate_block.pop()

    def next_offset(self) -> float:
        """Seconds from the schedule start to the next intended send."""
        self._unit_time += sample_interarrival(self.base_distribution, 1.0, self.base_cv2, self.rng)
        if self.modulation_window is None:
            return self._unit_time / self.qps
        while self._unit_time > self._window_unit_start + self._window_rate * self.modulation_window:
            self._window_unit_start += self._window_rate * self.modulation_window
            self._window_start += self.modulation_window
            self._window_rate = self._draw_rate()
        return self._window_start + (self._unit_time - self._window_unit_start) / self._window_rate

    def claim_with_epoch(self) -> tuple[float, int]:
        """Intended send time (time.monotonic()) of the next request, and the schedule epoch it belongs to.
        The schedule starts at the first claim; monotonic, so a wall-clock step or VM pause isn't a backlog."""
        with self._lock:
            if self.start is None:
                self.start = time.monotonic()
            return self.start + self.next_offset(), self.epoch

    def claim(self) -> float:
        return self.claim_with_epoch()[0]

    def track(self, intended: float, epoch: int) -> int:
        """Register a claimed send until it goes out (untrack), so a resync can find the oldest one."""
        with self._lock:
            self._pending_seq += 1
            self._pending[self._pending_seq] = (intended, epoch)
            return self._pending_seq

    def oldest_lateness(self) -> float:
        """How late the oldest pending send is now (0 with none), in the current epoch's schedule."""
        with self._lock:
            now = time.monotonic()
            return max((now - (t + sum(self.shifts[e:])) for t, e in self._pending.values()), default=0.0)

    def untrack(self, key: int) -> None:
        with self._lock:
            self._pending.pop(key, None)

    def resync(self, epoch: int, late_seconds: float) -> bool:
        """Shift the rest of the schedule later, once per stall, by enough that the oldest send still pending is
        due now (at least `late_seconds`); sends claimed before the shift move by the same amount (shift_since),
        so the stall leaves neither a burst nor a hole."""
        with self._lock:
            if epoch != self.epoch:
                return False
            now = time.monotonic()
            oldest = min((t + sum(self.shifts[e:]) for t, e in self._pending.values()), default=now - late_seconds)
            shift = max(late_seconds, now - oldest)
            self.start += shift
            self.shifted_seconds += shift
            self.shifts.append(shift)
            self.epoch += 1
            return True

    def shift_since(self, epoch: int) -> tuple[float, int]:
        """(Seconds the schedule moved since `epoch`, the current epoch)."""
        with self._lock:
            return sum(self.shifts[epoch:]), self.epoch

    def describe(self) -> str:
        label = f"QPS {self.qps} {self.distribution}"
        if self.distribution == "gamma":
            label += f" cv2={self.cv2:g}"
        if self.modulation_window is not None:
            label += f" modulated W={self.modulation_window:g}s D10={self.dispersion:g}"
            if self.base_cv2 != self.cv2 or self.base_distribution != self.distribution:
                label += f" (within-window gamma cv2={self.base_cv2:.2f})"
        return label

    @classmethod
    def instance(
        cls, qps, distribution, cv2=1.0, seed=None, modulation_window=None, dispersion=None, rate_block_windows=None
    ):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(qps, distribution, cv2, seed, modulation_window, dispersion, rate_block_windows)
            else:
                assert cls._instance.qps == qps
                assert cls._instance.distribution == distribution
                assert cls._instance.cv2 == cv2
                assert cls._instance.seed == seed
                assert cls._instance.modulation_window == modulation_window
                assert cls._instance.dispersion == dispersion
            return cls._instance


_WIRE_SEND_TIMES: dict = {}


class _TimedHTTPConnection(urllib3.connection.HTTPConnection):
    """Stamps when the request (headers and body) has been written to the socket, per greenlet."""

    def request(self, *args, **kwargs):
        super().request(*args, **kwargs)
        _WIRE_SEND_TIMES[id(gevent.getcurrent())] = time.monotonic()


class _TimedHTTPSConnection(urllib3.connection.HTTPSConnection):
    def request(self, *args, **kwargs):
        super().request(*args, **kwargs)
        _WIRE_SEND_TIMES[id(gevent.getcurrent())] = time.monotonic()


class _TimedHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):
    ConnectionCls = _TimedHTTPConnection


class _TimedHTTPSConnectionPool(urllib3.connectionpool.HTTPSConnectionPool):
    ConnectionCls = _TimedHTTPSConnection


def _timed_adapter(pool_maxsize: int) -> LocustHttpAdapter:
    adapter = LocustHttpAdapter(pool_manager=None, pool_maxsize=pool_maxsize)
    adapter.poolmanager.pool_classes_by_scheme = {"http": _TimedHTTPConnectionPool, "https": _TimedHTTPSConnectionPool}
    return adapter


def _begin_wire_timing() -> float:
    _WIRE_SEND_TIMES.pop(id(gevent.getcurrent()), None)
    return time.monotonic()


def _wire_send_time(dispatched: float) -> float:
    """When this greenlet's request finished going out on the wire, or `dispatched` if it never did."""
    return _WIRE_SEND_TIMES.pop(id(gevent.getcurrent()), dispatched)


class PrefixFill:
    """Progress of a --prefix-fill run, reported in the summary."""

    started = False
    requests = 0
    prompt_tokens_sent = 0
    prefix_tokens_sent = 0
    seconds = 0.0


def prefix_fill_summary_entries(stats, class_names: list[str]) -> dict[str, Any]:
    """Tokens sent and time taken by --prefix-fill, with server-reported prompt and cached tokens."""

    def total(metric):
        return sum(
            e.total_response_time
            for name in class_names
            if (e := stats.entries.get((f"class_{name}_{metric}", "METRIC"))) is not None
        )

    seconds = PrefixFill.seconds
    return {
        "mode": "prefix_fill",
        "prefix_fill_requests": PrefixFill.requests,
        "prefix_fill_prompt_tokens_sent": PrefixFill.prompt_tokens_sent,
        "prefix_fill_shared_prefix_tokens_sent": PrefixFill.prefix_tokens_sent,
        "prefix_fill_server_prompt_tokens": total("prompt_tokens"),
        "prefix_fill_server_cached_tokens": total("cached_tokens"),
        "prefix_fill_seconds": seconds,
        "prefix_fill_prompt_tokens_per_s": PrefixFill.prompt_tokens_sent / seconds if seconds else "",
    }


@dataclass(frozen=True)
class ScheduleEntry:
    offset: float  # seconds after the first entry of the whole schedule (all pods)
    class_index: int
    prefix_index: Optional[int] = None
    output_length: Optional[int] = None
    input_length: Optional[int] = None  # prompt tokens as the server counts them (chat template included)
    session: Optional[str] = None
    index: int = 0  # position in the whole time-sorted schedule
    cached_length: Optional[int] = None  # recorded reusable (cached) prompt tokens, as the server counts them
    cancelled: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    pod: Optional[str] = None  # the recorded pod (the row's "pod" field)
    lead_in: bool = False  # a warm-up row: sent unscored before time 0, only if nothing earlier covers it
    recorded_offset: Optional[float] = None  # the recorded offset of a retimed row
    defaulted: tuple = ()  # optional fields that were invalid and dropped (or output_length 0 sent as 1)


# Row fields and the names task 15 / the B3 classes may use for them
_REPLAY_ALIASES = {
    "input_length": ("input_length", "prompt_tokens"),
    "cached_length": ("cached_length", "cached_tokens", "cached_prompt_tokens"),
    "output_length": ("output_length", "generated_tokens"),
}


# Every field a schedule row may carry; others are reported (not silently dropped)
_REPLAY_FIELDS = frozenset(
    {"timestamp", "time_s", "class", "session_id", "pod", "prefix_index", "cancelled", "lead_in"}
    | {"temperature", "top_p", "reasoning_effort"}
    | {name for names in _REPLAY_ALIASES.values() for name in names}
)


def _warn_unknown_replay_fields(path: str, unknown: dict) -> None:
    if not unknown:
        return
    fields = ", ".join(f"{k} ({n} rows)" for k, n in sorted(unknown.items()))
    hint = ' (lead-ins are separate rows with "lead_in": true)' if "lead_in_length" in unknown else ""
    logger.warning(f"--replay-schedule {path}: ignoring unknown fields {fields}{hint}")


def _replay_int(row: dict, keys: tuple, where: str, minimum: int) -> Optional[int]:
    present = [k for k in keys if row.get(k) is not None]
    if len(present) > 1:
        raise ValueError(f"{where}: {' and '.join(present)} both set; use one")
    return _shape_int(row, present[0], where, minimum) if present else None


def _replay_float(row: dict, key: str, where: str, minimum: float, maximum: Optional[float]) -> Optional[float]:
    if row.get(key) is None:
        return None
    value = _shape_number(row, key, where)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(
            f"{where}: '{key}' must be in [{minimum}, {maximum if maximum is not None else 'inf'}], got {value}"
        )
    return float(value)


def _replay_bool(row: dict, key: str, where: str) -> bool:
    value = row.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"{where}: '{key}' must be true or false, got {value!r}")
    return value


def _replay_effort(row: dict, where: str) -> Optional[str]:
    value = row.get("reasoning_effort")
    if value is not None and value not in ("none", "low", "medium", "high"):
        raise ValueError(f"{where}: 'reasoning_effort' must be none, low, medium or high, got {value!r}")
    return value


def _replay_pod(entry: ScheduleEntry, pods: int) -> int:
    """Pod an entry belongs to: by session, so a conversation stays on one generator, else by position."""
    if entry.session is None:
        return entry.index % pods
    return int(hashlib.sha256(entry.session.encode()).hexdigest()[:8], 16) % pods


@lru_cache(maxsize=None)
def parse_replay_retime(value: Optional[str]) -> Optional[tuple[str, float]]:
    """--replay-retime: 'poisson' or 'gamma:<cv2>' -> (kind, gap CV²), or None."""
    if not value:
        return None
    if value == "poisson":
        return ("poisson", 1.0)
    if value.startswith("gamma:"):
        try:
            cv2 = float(value.split(":", 1)[1])
        except ValueError:
            cv2 = -1.0
        if cv2 > 0:
            return ("gamma", cv2)
    raise ValueError(f"--replay-retime must be 'poisson' or 'gamma:<cv2>' with cv2 > 0, got {value!r}")


RETIME_DRAWS = 16  # seeded draws tried per retime; the one whose gap CV² is closest to the request is used


def _draw_gaps(rng: random.Random, n: int, kind: str, cv2: float) -> list[float]:
    return [rng.expovariate(1.0) if kind == "poisson" else rng.gammavariate(1 / cv2, cv2) for _ in range(n)]


def retimed_offsets(count: int, span: float, retime: tuple[str, float], seed) -> list[float]:
    """`count` send times over [0, span]: seeded Poisson (or gamma at the given gap CV²) gaps, scaled to the span,
    so the mean rate is the schedule's own. A finite draw's gap CV² scatters around the request (about ±10% for
    2,000 gamma:4 gaps), so of RETIME_DRAWS seeded draws the one closest to it (in log ratio) is used."""
    if count < 2 or span <= 0:
        return [0.0] * count
    kind, cv2 = retime
    rng = random.Random(f"replay-retime:{seed}:{kind}:{cv2}")
    best, best_error = None, None
    for _ in range(RETIME_DRAWS if count > 2 else 1):
        gaps = _draw_gaps(rng, count - 1, kind, cv2)
        mean = sum(gaps) / len(gaps)
        drawn = sum((g - mean) ** 2 for g in gaps) / len(gaps) / (mean * mean)
        error = abs(math.log(drawn / cv2)) if drawn > 0 else math.inf
        if best_error is None or error < best_error:
            best, best_error = gaps, error
    scale = span / sum(best)
    offsets, t = [0.0], 0.0
    for gap in best:
        t += gap * scale
        offsets.append(t)
    offsets[-1] = span
    return offsets


@lru_cache(maxsize=None)
def retime_rows(entries: tuple, retime: tuple[str, float], seed) -> tuple:
    """--replay-retime: redraw this stream's send times as one seeded Poisson / gamma process (at the requested
    gap CV²) with as many arrivals as it has rows, over its own span, so its rate is unchanged. Rows fill the new
    slots in their recorded order: a follow-up still comes after its parent (and every earlier row), and the
    layout (what each row reuses, which lead-ins are covered) is the same as the recorded replay's. Only the
    recorded gaps between turns are not kept. Lead-in rows keep their times (before time 0)."""
    lead_ins = tuple(e for e in entries if e.lead_in)
    scored = [e for e in entries if not e.lead_in]
    if len(scored) < 3:
        return entries
    first, last = scored[0].offset, scored[-1].offset
    offsets = retimed_offsets(len(scored), last - first, retime, seed)
    retimed = tuple(dataclasses.replace(e, offset=first + o, recorded_offset=e.offset) for e, o in zip(scored, offsets))
    return lead_ins + retimed


def load_replay_schedule(
    path: str,
    class_names: tuple,
    pods: int = 1,
    pod_index: int = 0,
    retime: Optional[str] = None,
    seed=None,
    pod_select: Optional[str] = None,
) -> tuple:
    """--replay-schedule JSONL: one request per line, {"timestamp": <ms>, "class": "<name>"} (AIPerf's
    millisecond timestamps; "time_s" in seconds also works), optionally "prefix_index" (a prefix of the class
    pool; drawn otherwise), "input_length" (prompt tokens as the server counts them), "output_length" and
    "session_id" (turns of one multi-turn conversation). Sorted by time, relative to the first entry of the
    whole schedule; with `pods` > 1 only this pod's share is returned."""
    index = {name: i for i, name in enumerate(class_names)}
    raw = []
    unknown: dict[str, int] = {}
    with open(path, "r") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            where = f"--replay-schedule {path}:{n}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON ({e})")
            if not isinstance(row, dict):
                raise ValueError(f"{where}: each line must be a JSON object")
            for key in row.keys() - _REPLAY_FIELDS:
                unknown[key] = unknown.get(key, 0) + 1
            if "timestamp" in row:
                t = _shape_number(row, "timestamp", where) / 1000
            elif "time_s" in row:
                t = _shape_number(row, "time_s", where)
            else:
                raise ValueError(f"{where}: needs 'timestamp' (ms) or 'time_s'")
            name = row.get("class")
            if name not in index:
                raise ValueError(f"{where}: class {name!r} is not in --shape-mix ({', '.join(class_names)})")
            session = row.get("session_id")
            if session is not None and not isinstance(session, (str, int)):
                raise ValueError(f"{where}: 'session_id' must be a string or integer, got {session!r}")
            pod = row.get("pod")
            if pod is not None and not isinstance(pod, (str, int)):
                raise ValueError(f"{where}: 'pod' must be a string or integer, got {pod!r}")
            defaulted = []

            def lenient(field, parse):
                # Optional per-row settings: an invalid value drops the field for this row (counted), rather
                # than rejecting the whole schedule
                try:
                    return parse()
                except ValueError:
                    defaulted.append(field)
                    return None

            output_length = _replay_int(row, _REPLAY_ALIASES["output_length"], where, 0)
            if output_length == 0:
                output_length = 1  # a recorded zero-output row: prefill only, sent with one output token
                defaulted.append("output_length")
            fields = {
                "prefix_index": _replay_int(row, ("prefix_index",), where, 0),
                "output_length": output_length,
                "input_length": _replay_int(row, _REPLAY_ALIASES["input_length"], where, 1),
                "cached_length": _replay_int(row, _REPLAY_ALIASES["cached_length"], where, 0),
                "cancelled": _replay_bool(row, "cancelled", where),
                "temperature": lenient("temperature", lambda: _replay_float(row, "temperature", where, 0.0, None)),
                "top_p": lenient("top_p", lambda: _replay_float(row, "top_p", where, 0.0, 1.0)),
                "reasoning_effort": lenient("reasoning_effort", lambda: _replay_effort(row, where)),
                "pod": None if pod is None else str(pod),
                "lead_in": _replay_bool(row, "lead_in", where),
            }
            fields["defaulted"] = tuple(defaulted)
            if (
                fields["cached_length"] is not None
                and fields["input_length"] is not None
                and fields["cached_length"] > fields["input_length"]
            ):
                raise ValueError(
                    f"{where}: cached_length {fields['cached_length']} > input_length {fields['input_length']}"
                )
            if session is not None:
                # Session keys are scoped to their recorded pod
                session = f"{fields['pod']}/{session}" if fields["pod"] is not None else str(session)
            raw.append((t, n, index[name], session, fields))
    if not raw:
        raise ValueError(f"--replay-schedule {path}: no entries")
    _warn_unknown_replay_fields(path, unknown)
    with_pod = [r[1] for r in raw if r[4]["pod"] is not None]
    if with_pod and len(with_pod) != len(raw):
        missing = sorted(set(r[1] for r in raw) - set(with_pod))
        raise ValueError(
            f"--replay-schedule {path}: 'pod' is set on some rows, so every row needs it (rows without one would "
            f"be dropped from a per-pod replay); missing on line(s) {', '.join(map(str, missing[:10]))}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
    with_cached = [r[1] for r in raw if r[4]["cached_length"] is not None]
    if with_cached and len(with_cached) != len(raw):
        missing = sorted(set(r[1] for r in raw) - set(with_cached))
        raise ValueError(
            f"--replay-schedule {path}: cached_length is set on some rows, so every row needs it; missing on "
            f"line(s) {', '.join(map(str, missing[:10]))}{' ...' if len(missing) > 10 else ''}"
        )
    raw.sort(key=lambda r: (r[0], r[1]))
    # Time 0 is the first scored row; lead-in rows before it get negative offsets
    start = next((r[0] for r in raw if not r[4]["lead_in"]), raw[0][0])
    entries = tuple(
        ScheduleEntry(t - start, ci, session=session, index=i, **fields)
        for i, (t, _, ci, session, fields) in enumerate(raw)
    )
    retime_spec = parse_replay_retime(retime)
    recorded_pods = sorted({e.pod for e in entries if e.pod is not None}, key=_pod_sort_key)
    if recorded_pods:
        chosen = select_replay_pods(recorded_pods, pod_select, pods, pod_index)
        entries = tuple(e for e in entries if e.pod in chosen)
        if not entries:
            raise ValueError(f"--replay-schedule {path}: no rows for pod(s) {', '.join(chosen)}")
        if retime_spec is not None:
            entries = retime_rows(entries, retime_spec, f"{seed}:{','.join(chosen)}")
        return entries
    if pod_select:
        raise ValueError(f"--replay-pod: {path} has no 'pod' field on its rows")
    if pods > 1:
        # No recorded pods: split by session (else position) across generators
        entries = tuple(e for e in entries if _replay_pod(e, pods) == pod_index)
    if retime_spec is not None:
        # After the split: each generator's own stream gets the requested arrival process
        entries = retime_rows(entries, retime_spec, seed if pods == 1 else f"{seed}:{pod_index}/{pods}")
    return entries


def replay_entries(options: argparse.Namespace, class_names: tuple) -> tuple:
    """This generator's replay rows from the options (cached: init, layout and the pacer share one parse)."""
    return load_replay_schedule(
        options.replay_schedule,
        class_names,
        getattr(options, "replay_pods", 1),
        getattr(options, "replay_pod_index", 0),
        getattr(options, "replay_retime", None),
        resolve_request_seed(options),
        getattr(options, "replay_pod", None),
    )


# Task 15's schedule meta: pool_prefix[<slot>][<class>]["*"] = {shared_max, shared_p99, ...}, the recorded shared
# length (what an earlier same-class row on the replica had cached) of that class's session-first rows, over all
# of its prefix indexes. Field names live here so a rename is one edit.
REPLAY_META_POOL_KEY = "pool_prefix"
REPLAY_META_CLASS_ALL = "*"
REPLAY_META_POOL_FIELDS = {"max": "shared_max", "p99": "shared_p99"}


def replay_meta_path(options: argparse.Namespace) -> Optional[str]:
    """--replay-schedule-meta, else `<schedule>.meta.json` next to the schedule when it exists."""
    explicit = getattr(options, "replay_schedule_meta", None)
    if explicit:
        return explicit
    default = f"{options.replay_schedule}.meta.json"
    return default if os.path.exists(default) else None


def replay_pool_prefix_tokens(options: argparse.Namespace, entries: tuple, class_names: tuple) -> dict[str, int]:
    """Pool prefix length per class for a replay: the longest recorded shared length (or its p99 with
    --replay-pool-prefix p99) on the replayed replica(s), from the schedule meta. A class the meta doesn't cover
    keeps its mean cached length, with a warning (rows recorded with a longer shared part then fall short)."""
    which = getattr(options, "replay_pool_prefix", "max") or "max"
    field = REPLAY_META_POOL_FIELDS[which]
    path = replay_meta_path(options)
    pool = None
    if path is not None:
        try:
            with open(path) as f:
                pool = json.load(f).get(REPLAY_META_POOL_KEY)
        except (OSError, ValueError) as e:
            raise ValueError(f"--replay-schedule-meta {path}: can't read it ({e})")
    slots = sorted({e.pod for e in entries if e.pod is not None})
    sizes: dict[str, int] = {}
    if isinstance(pool, dict) and slots:
        for name in class_names:
            values = []
            for slot in slots:
                entry = ((pool.get(slot) or {}).get(name) or {}).get(REPLAY_META_CLASS_ALL) or {}
                value = entry.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    values.append(int(math.ceil(value)))
            if values:
                sizes[name] = max(values)
    used = {e.class_index for e in entries}
    missing = [name for i, name in enumerate(class_names) if name not in sizes and i in used]
    if missing and not sizes:
        why = (
            "no schedule meta found (--replay-schedule-meta, or <schedule>.meta.json)"
            if path is None
            else f"{path} has no usable '{REPLAY_META_POOL_KEY}' for the replayed replica(s)"
        )
        logger.warning(
            f"--replay-schedule: {why}: pool prefixes stay at the class mean cached length; session-first rows "
            f"recorded with a longer shared part will get less cache than recorded (see First Row Shortfall in "
            f"the summary)"
        )
    elif missing:
        # The meta covers the replica, so these classes have no session-first row with a shared part to size for
        logger.info(
            f"--replay-schedule: the schedule meta has no shared length for {', '.join(missing[:12])}"
            f"{' ...' if len(missing) > 12 else ''}; their pool prefixes stay at the class mean cached length"
        )
    return sizes


def replay_sized_mix(mix: ShapeMix, sizes: dict[str, int]) -> ShapeMix:
    """The mix with each class's cached_prompt_tokens (its pool prefix length) set to `sizes`, and its prompt
    long enough to keep its tail: replay rows carry their own lengths, the class values only size the pool."""
    classes = []
    for cls in mix.classes:
        length = sizes.get(cls.name)
        if length is None:
            classes.append(cls)
            continue
        tail = max(cls.prompt_tokens - cls.cached_prompt_tokens, 2 * SHAPE_MIX_TAIL_RESERVE_TOKENS)
        classes.append(dataclasses.replace(cls, cached_prompt_tokens=length, prompt_tokens=length + tail))
    return dataclasses.replace(mix, classes=classes)


def _pod_sort_key(pod: str):
    # Natural order: p2 before p10
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"(\d+)", pod) if part]


def select_replay_pods(recorded: list, pod_select: Optional[str], pods: int, pod_index: int) -> tuple:
    """Recorded pods this generator replays. One test replica stands for one recorded pod:
    --replay-pod ID replays that pod's rows at their exact times; --replay-pods N (with --replay-pod-index I)
    gives replica I the I-th of the first N recorded pods (or of the --replay-pod list). Several pods in one
    stream (--replay-pod a,b or all) is a merge, only meant for tiers above 1x, and is labelled in the summary."""
    listed = None
    if pod_select and pod_select != "all":
        listed = [p.strip() for p in pod_select.split(",") if p.strip()]
        unknown = [p for p in listed if p not in recorded]
        if unknown:
            raise ValueError(
                f"--replay-pod: {', '.join(unknown)} not in the schedule's pods ({', '.join(recorded[:20])}"
                f"{' ...' if len(recorded) > 20 else ''})"
            )
    if pods > 1:
        if pod_select == "all":
            raise ValueError("--replay-pods N gives each replica one recorded pod; drop --replay-pod all")
        candidates = listed or recorded
        if len(candidates) < pods:
            raise ValueError(f"--replay-pods {pods}: only {len(candidates)} recorded pods to replay")
        return (candidates[pod_index],)
    if pod_select == "all":
        return tuple(recorded)
    if listed:
        return tuple(listed)
    if len(recorded) == 1:
        return tuple(recorded)
    raise ValueError(
        f"--replay-schedule records {len(recorded)} pods; one test replica replays one of them: pick it with "
        f"--replay-pod ID (e.g. {recorded[0]}), or --replay-pods N --replay-pod-index I. Merging pods "
        f"(--replay-pod a,b or all) is only for tiers above 1x"
    )


class ReplayPacer:
    """Schedule-backed pacer for --replay-schedule: hands out the recorded send times at recorded speed
    (no speed-up), with the same claim / resync interface as FixedQPSPacer."""

    _instance = None
    _lock = threading.Lock()
    finishing = False

    def __init__(self, entries: tuple, path: str = ""):
        # Lead-in rows are sent before time 0 (send_lead_ins), not paced or scored
        self.lead_ins = tuple(e for e in entries if e.lead_in)
        entries = tuple(e for e in entries if not e.lead_in)
        self.entries = entries
        self.path = path
        self.start = None
        self.epoch = 0
        self.shifted_seconds = 0.0
        self._next = 0
        # Claimed entries not yet sent or skipped; the run may only end once this is back to 0
        self.pending = 0
        self.sent_rows = self.late_rows = self.skipped_rows = self.cancelled_rows = 0
        self.max_lateness_seen = 0.0
        self.duration = entries[-1].offset if entries else 0.0
        self.pods = sorted({e.pod for e in entries if e.pod is not None}, key=_pod_sort_key)

    def claim_entry(self) -> Optional[tuple[float, int, ScheduleEntry]]:
        """The next entry with its intended send time; call release() once it is sent or skipped."""
        with self._lock:
            if self._next >= len(self.entries):
                return None
            if self.start is None:
                self.start = time.monotonic()
                start_at = getattr(self, "start_at", None)
                if start_at is not None:
                    # A start time shared by every generator (--replay-start-at): offsets count from it
                    self.start += start_at - time.time()
            entry = self.entries[self._next]
            self._next += 1
            self.pending += 1
            return self.start + entry.offset, self.epoch, entry

    def release(self) -> None:
        with self._lock:
            self.pending -= 1

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self.entries)

    def resync(self, epoch: int, late_seconds: float) -> bool:
        with self._lock:
            if epoch != self.epoch:
                return False
            self.start += late_seconds
            self.shifted_seconds += late_seconds
            self.epoch += 1
            return True

    def describe(self) -> str:
        span = self.entries[-1].offset - self.entries[0].offset if self.entries else 0.0
        rate = len(self.entries) / span if span else 0.0
        retimed = f", retimed {self.retime}" if getattr(self, "retime", None) else ""
        if len(self.pods) > 1:
            pods = f", MERGED {len(self.pods)} recorded pods"
        elif self.pods:
            pods = f", pod {self.pods[0]}"
        else:
            pods = ""
        return (
            f"replay {os.path.basename(self.path)}: {len(self.entries)} requests over {span:.0f}s "
            f"({rate:.2f} QPS{pods}{retimed})"
        )

    @classmethod
    def instance(cls, options: argparse.Namespace, class_names: tuple):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(replay_entries(options, class_names), options.replay_schedule)
                cls._instance.retime = getattr(options, "replay_retime", None)
                cls._instance.start_at = getattr(options, "replay_start_at", None)
            return cls._instance


class SendTimeline:
    """Intended and actual send times of paced requests, for delivered-arrival stats in the summary."""

    intended = array("d")
    actual = array("d")
    skipped = 0

    @classmethod
    def record(cls, intended: float, actual: float) -> None:
        cls.intended.append(intended)
        cls.actual.append(actual)

    @classmethod
    def reset(cls) -> None:
        cls.intended = array("d")
        cls.actual = array("d")
        cls.skipped = 0


def gap_cv2(times) -> Optional[float]:
    if len(times) < 3:
        return None
    gaps = [b - a for a, b in zip(times, times[1:])]
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return None
    return sum((g - mean) ** 2 for g in gaps) / len(gaps) / (mean * mean)


def count_dispersion(times, window: float, offsets: int = 10) -> Optional[float]:
    """Index of dispersion (variance / mean) of counts in `window`-second bins, averaged over `offsets` bin
    alignments so the result doesn't depend on where the bins happen to start."""
    if len(times) < 2:
        return None
    t0, t1 = times[0], times[-1]
    values = []
    for k in range(offsets):
        start = t0 + window * k / offsets
        bins = int((t1 - start) // window)
        if bins < 2:
            continue
        counts = [0] * bins
        for t in times:
            i = int((t - start) // window)
            if 0 <= i < bins:
                counts[i] += 1
        mean = sum(counts) / bins
        if mean > 0:
            values.append(sum((c - mean) ** 2 for c in counts) / bins / mean)
    return sum(values) / len(values) if values else None


def arrival_summary_entries(intended, actual) -> dict[str, Any]:
    """Intended vs delivered gap CV^2 and 1 s / 10 s count dispersion, plus how late sends left."""
    if len(actual) < 3:
        return {}
    lateness = sorted((a - i) * 1000 for i, a in zip(intended, actual))
    intended, actual = sorted(intended), sorted(actual)
    pct = lambda q: lateness[min(len(lateness) - 1, int(q * len(lateness)))]
    entries = {}
    for name, times in (("intended", intended), ("delivered", actual)):
        entries[f"{name}_gap_cv2"] = gap_cv2(times)
        entries[f"{name}_dispersion_1s"] = count_dispersion(times, 1.0)
        entries[f"{name}_dispersion_10s"] = count_dispersion(times, 10.0)
    entries["send_lateness_p50_ms"] = pct(0.5)
    entries["send_lateness_p99_ms"] = pct(0.99)
    entries["send_lateness_max_ms"] = lateness[-1]
    return {k: ("" if v is None else v) for k, v in entries.items()}


class RampingPacer:
    """
    A pacer that gradually increases concurrent requests from min_users to max_users.

    Tracks the number of in-flight requests and controls concurrency by making
    users wait when above the target. Starts at min_users concurrent requests,
    runs for warmup_time seconds, then increases by pct percent every second
    until max_users is reached.
    """

    _instance = None

    def __init__(self, min_users, max_users, pct, warmup_time=30, max_spawn_rate=10):
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
            logger.info(f"RampingPacer: warmup complete, starting to increase concurrency from {self.min_users}")

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
            logger.info(f"RampingPacer: target concurrency now at {target:.1f}, active: {self._active_requests}")
            self._logged_target = target

        return target

    def request_start(self):
        """Called when a request is about to start. Increments active count."""
        self._active_requests += 1

    def request_end(self):
        """Called when a request completes. Decrements active count."""
        self._active_requests -= 1

    @classmethod
    def instance(cls, min_users, max_users, pct, warmup_time=30, max_spawn_rate=10):
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
    _tokenizer_lock = threading.Lock()
    deferred_run_time_seconds = None
    deferred_max_requests = None
    request_count = 0
    stop_scheduled = False
    max_requests_stop_scheduled = False
    stats_reset_done = False
    spawned = gevent.event.Event()
    late_sends = 0
    last_late_warning = 0.0

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
    def warn_late_send(cls, late_seconds: float, max_lateness: float = 0.5) -> None:
        cls.late_sends += 1
        now = time.time()
        if now - cls.last_late_warning >= 10:
            logger.warning(
                f"{cls.late_sends} sends left more than {LATE_SEND_WARNING_SECONDS * 1000:.0f} ms after their "
                f"scheduled time (latest {late_seconds * 1000:.0f} ms; sends over {max_lateness:g} s late are "
                f"skipped and the schedule moves later): the load generator's CPU can't keep up, or -u is too small to prepare requests ahead of "
                f"a burst"
            )
            cls.last_late_warning = now

    @classmethod
    def notify_first_request(cls):
        cls.first_request_done += 1

    @classmethod
    def notify_request_complete(cls):
        """Notify that a request has completed and check if we should stop."""
        if cls.deferred_max_requests is None:
            return
        cls.request_count += 1
        logger.info(f"Request {cls.request_count}/{cls.deferred_max_requests} completed")
        if cls.request_count >= cls.deferred_max_requests:
            logger.info(f"Reached max requests limit ({cls.deferred_max_requests}), stopping test")
            cls.stop_scheduled = True
            # Use a small delay to ensure the current request completes
            gevent.spawn_later(0.1, cls._do_quit)

    @classmethod
    def _do_quit(cls):
        """Actually stop the runner."""
        if not cls.environment:
            logger.warning("environment is None, cannot stop")
            return
        if not cls.environment.runner:
            logger.warning("runner is None, cannot stop")
            return

        runner = cls.environment.runner
        try:
            if hasattr(runner, "stop"):
                runner.stop()
            runner.quit()
        except Exception as e:
            logger.error(f"Failed to stop runner: {e}")

    @classmethod
    def notify_spawning_complete(cls, user_count):
        cls.users = user_count
        # Start steady-state measurement exactly when all users have spawned
        if not cls.stats_reset_done:
            cls.reset_stats()
            cls.stats_reset_done = True
        ClientCpu.reset()
        SendTimeline.reset()
        RequestOutcomes.reset()
        # Paced users start claiming send times only now, so the schedule doesn't begin with half the users
        cls.spawned.set()
        start_wait = cls._replay_start_wait()
        if start_wait > 0:
            # The window (stats, QPS, -t) starts at the shared --replay-start-at, not while waiting for it
            def _start_window():
                cls.reset_stats()
                SendTimeline.reset()
                RequestOutcomes.reset()
                ClientCpu.reset()

            # Just before the start: no scored row goes out earlier, and the first sends' stamps must survive it
            gevent.spawn_later(max(0.0, start_wait - 0.1), _start_window)
        # If -t/--run-time was provided, schedule test stop relative to spawn complete
        if (
            cls.deferred_run_time_seconds is not None
            and not cls.stop_scheduled
            and cls.environment is not None
            and cls.environment.runner is not None
        ):
            delay = float(cls.deferred_run_time_seconds) + start_wait
            logger.info(f"Scheduling stop {delay}s after spawning complete (deferred -t)")
            gevent.spawn_later(delay, cls.environment.runner.quit)
            cls.stop_scheduled = True

    @classmethod
    def _replay_start_wait(cls) -> float:
        """Seconds until --replay-start-at (0 if unset or past; a passed start is reported with its due rows)."""
        options = getattr(cls.environment, "parsed_options", None) if cls.environment is not None else None
        start_at = getattr(options, "replay_start_at", None) if options is not None else None
        if start_at is None or not getattr(options, "replay_schedule", None):
            return 0.0
        wait = start_at - time.time()
        if wait < -getattr(options, "qps_max_lateness", DEFAULT_MAX_LATENESS_SECONDS):
            # The pacer may not exist yet (it is built on a user's first claim): load the same rows it will hold
            names = tuple(c.name for c in parse_shape_mix(options.shape_mix).classes)
            due = sum(1 for e in ReplayPacer.instance(options, names).entries if e.offset < -wait)
            logger.warning(
                f"--replay-start-at passed {-wait:.1f}s before this generator was ready: {due} rows already due are "
                f"late and go out {'at once' if options.replay_late_rows == 'send' else 'skipped'} "
                f"(--replay-late-rows {options.replay_late_rows}); give generators more lead time"
            )
        return max(0.0, wait)

    @classmethod
    def reset_stats(cls):
        if cls.environment is None or cls.environment.runner is None:
            return
        logger.info("Resetting stats after traffic reached a steady state")
        cls.environment.events.reset_stats.fire()
        cls.environment.runner.stats.reset_all()

    @classmethod
    def load_tokenizer(cls, explicit_path: str):
        if cls.tokenizer:
            return cls.tokenizer
        with cls._tokenizer_lock:
            if cls.tokenizer:
                return cls.tokenizer
            source = explicit_path or DEFAULT_TOKENIZER
            if not explicit_path:
                logger.warning(f"No --tokenizer specified, falling back to default: {DEFAULT_TOKENIZER}")
            cls.tokenizer = _load_auto_tokenizer(source)
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
        logger.info(f"Deferring -t/--run-time to start after spawning complete: {seconds}s")

    # Capture max_requests if specified
    try:
        max_requests = getattr(environment.parsed_options, "max_requests", None)
    except Exception:
        max_requests = None
    if max_requests is not None:
        InitTracker.deferred_max_requests = max_requests
        InitTracker.environment = environment
        logger.info(f"Will stop after {max_requests} requests complete")


def validate_shape_mix_options(options: argparse.Namespace) -> None:
    """Parse --shape-mix against the chat template overhead; raises ValueError naming any class too short."""
    resolve_shape_mix_corpus(getattr(options, "shape_mix_corpus", None))
    if getattr(options, "replay_schedule", None):
        pods, pod_index = getattr(options, "replay_pods", 1), getattr(options, "replay_pod_index", 0)
        if pods < 1 or not 0 <= pod_index < pods:
            raise ValueError(f"--replay-pod-index must be in [0, --replay-pods={pods}), got {pod_index}")
        classes = parse_shape_mix(options.shape_mix).classes
        schedule = replay_entries(options, tuple(c.name for c in classes))
        for entry in schedule:
            cls = classes[entry.class_index]
            pool = cls.prefix_pool_size if cls.cached_prompt_tokens else 0
            if entry.prefix_index is not None and entry.prefix_index >= pool:
                raise ValueError(
                    f"--replay-schedule: prefix_index {entry.prefix_index} is outside class {cls.name}'s pool of "
                    f"{pool}" + ("" if pool else " (the class has no shared prefix)")
                )
    overhead = 0
    if options.chat:
        tokenizer = InitTracker.load_tokenizer(options.tokenizer)
        overhead = len(empty_chat_template_token_ids(tokenizer, options.tokenizer or DEFAULT_TOKENIZER))
    parse_shape_mix(options.shape_mix, chat_overhead_tokens=overhead)


@events.init.add_listener
def _validate_shape_mix_on_init(environment, **_kwargs):
    options = getattr(environment, "parsed_options", None)
    if options is None or not getattr(options, "shape_mix", None):
        return
    try:
        validate_shape_mix_options(options)
    except (OSError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)


@events.test_start.add_listener
def _build_shape_mix_before_spawning(environment, **_kwargs):
    """Build the --shape-mix dataset (tokenizer, prefix pool) before any user spawns.

    Locust spawns every user and fires spawning_complete, which starts the run window (stats reset,
    deferred -t, QPS sends), before the first user's on_start runs; built there, the pool would eat
    into the window.
    """
    options = getattr(environment, "parsed_options", None)
    if options is None or isinstance(environment.runner, MasterRunner):
        return
    ClientCpu.start()
    if not getattr(options, "shape_mix", None):
        if not (options.systemone or options.embeddings or options.rerank):
            try:
                # The text dataset (it tokenizes its corpus) is built before spawning, outside the run window
                DatasetHolder.get_instance(options)
            except Exception as e:
                logger.warning(f"Dataset setup before spawning failed ({e!r}); users will build it")
        return
    try:
        worker_index = getattr(environment.runner, "worker_index", 0) or 0
        if getattr(options, "replay_schedule", None) and options.replay_pods > 1:
            # Pods are separate processes with worker index 0: tell their nonces apart by pod
            worker_index = options.replay_pod_index
        ds = DatasetHolder.get_shape_mix_instance(options, worker_index)
        if getattr(options, "replay_schedule", None):
            send_replay_lead_ins(environment, options, ds)
    except Exception as e:
        logger.error(f"--shape-mix setup failed: {e!r}\n{traceback.format_exc()}")
        os._exit(1)


class ClientCpu:
    """This process's CPU use over the run window (mean, and the busiest 5 s), sampled each second: a client near
    100% sends late and evenly, which voids send-timing results (C-151)."""

    _samples: list = []
    _window_start = None
    _started = False

    @classmethod
    def start(cls) -> None:
        if cls._started:
            return
        cls._started = True

        def sample():
            while True:
                cls._samples.append((time.monotonic(), time.process_time()))
                if len(cls._samples) > 7200:
                    del cls._samples[:3600]
                gevent.sleep(1.0)

        gevent.spawn(sample)

    @classmethod
    def reset(cls) -> None:
        cls._window_start = (time.monotonic(), time.process_time())

    @classmethod
    def summary_entries(cls) -> dict:
        if cls._window_start is None:
            return {}
        wall0, cpu0 = cls._window_start
        wall, cpu = time.monotonic(), time.process_time()
        if wall - wall0 <= 0:
            return {}
        mean = 100 * (cpu - cpu0) / (wall - wall0)
        points = [p for p in cls._samples if p[0] >= wall0] + [(wall, cpu)]
        peak = 0.0
        j = 0
        for i in range(len(points)):
            while points[i][0] - points[j][0] > 5.0:
                j += 1
            if points[i][0] - points[j][0] >= 4.0:
                peak = max(peak, 100 * (points[i][1] - points[j][1]) / (points[i][0] - points[j][0]))
        peak = max(peak, mean)
        if peak > 90:
            logger.warning(
                f"Client CPU peaked at {peak:.0f}% (mean {mean:.0f}%): this process was saturated, so its send timing "
                f"is unreliable; use more processes or a lower rate"
            )
        return {"client_cpu_percent": mean, "client_cpu_peak_5s_percent": peak}


class LeadIns:
    """Lead-in rows of a --replay-schedule: warm-up sent before time 0, outside the stats and -t."""

    rows = sent = failed = skipped = 0
    remainder_rows = 0  # lead-ins with cached_length < input_length: a shared part plus the session's own context
    seconds = 0.0


def send_replay_lead_ins(environment, options: argparse.Namespace, ds: "ShapeMixDataset") -> None:
    """Send the schedule's lead-in rows (unscored, before any user spawns): each session's in order, sessions
    concurrently (--prefix-fill-concurrency), with max_tokens 1 since only the prompt cache matters. Rows an
    earlier prompt already covers were dropped at layout."""
    entries = replay_entries(options, tuple(c.name for c in ds.mix.classes))
    LeadIns.rows = sum(1 for e in entries if e.lead_in)
    LeadIns.remainder_rows = sum(1 for e in entries if e.lead_in and (e.cached_length or 0) < (e.input_length or 0))
    if not LeadIns.rows:
        return
    LeadIns.skipped = getattr(ds, "lead_in_skipped", 0)
    to_send = ds.lead_in_requests(entries)
    if not to_send:
        return
    if not options.provider:
        logger.warning(
            f"--replay-schedule: {len(to_send)} lead-in rows need --provider to be sent before time 0; not sent"
        )
        LeadIns.failed = len(to_send)
        return
    formatter = PROVIDER_CLASS_MAP[options.provider](options.model, options)
    url = environment.host.rstrip("/") + formatter.get_url()
    base = {"Content-Type": "application/json"}
    if options.api_key:
        base["Authorization"] = "Bearer " + options.api_key
    for header in options.header or []:
        key, val = header.split(":", 1)
        base[key] = val
    groups: dict = {}
    for n, (session, request) in enumerate(to_send):
        groups.setdefault(session if session is not None else f"#row{n}", []).append(request)
    http = requests.Session()
    errors: list[str] = []

    def send(requests_in_order):
        for request in requests_in_order:
            data, headers = build_shape_mix_request(formatter, request, options.shape_mix_output_mode)
            data["max_tokens"] = 1
            for key in ("min_tokens", "ignore_eos", "stream_options"):
                data.pop(key, None)
            data["stream"] = False
            try:
                response = http.post(
                    url,
                    data=json.dumps(data),
                    headers={**base, **(headers or {})},
                    timeout=request_timeout(options) or 600,
                )
                response.raise_for_status()
                LeadIns.sent += 1
            except Exception as e:
                LeadIns.failed += 1
                if len(errors) < 3:
                    errors.append(repr(e)[:200])

    started = time.monotonic()
    logger.info(
        f"--replay-schedule: sending {len(to_send)} lead-in rows before time 0 ({LeadIns.skipped} already covered by "
        f"earlier prompts, not sent)"
    )
    pool = gevent.pool.Pool(max(1, options.prefix_fill_concurrency))
    for group in groups.values():
        pool.spawn(send, group)
    pool.join()
    LeadIns.seconds = time.monotonic() - started
    logger.info(
        f"--replay-schedule: lead-in rows sent {LeadIns.sent}, failed {LeadIns.failed} in {LeadIns.seconds:.1f}s"
        + (f"; first errors: {errors}" if errors else "")
    )


# Status a proxy may answer with when it gave up on a request for the client (nginx uses 499)
CLIENT_CANCELLED_STATUS = 299


def request_timeout(options: argparse.Namespace) -> Optional[float]:
    """--request-timeout in seconds (connect and per-read for streams), or None for no timeout."""
    value = getattr(options, "request_timeout", 60.0)
    return value if value and value > 0 else None


def _is_timeout(error: Optional[BaseException]) -> bool:
    seen = set()
    stack = [error]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen or not isinstance(e, BaseException):
            continue
        seen.add(id(e))
        if isinstance(e, (requests.exceptions.Timeout, urllib3.exceptions.TimeoutError, TimeoutError)):
            return True
        # requests wraps urllib3's ReadTimeoutError in a ConnectionError while streaming
        stack.extend([e.__cause__, e.__context__, *[a for a in e.args if isinstance(a, BaseException)]])
        reason = getattr(e, "reason", None)
        if isinstance(reason, BaseException):
            stack.append(reason)
    return False


def _guard_stream(chunks, errors: list):
    try:
        yield from chunks
    except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError, OSError) as e:
        errors.append(e)


class RequestOutcomes:
    """Client-side cancellations and failures by kind, overall and per --shape-mix class."""

    KINDS = ("timeout", "http_299", "cancel_rate", "other_error")
    counts: dict = {}
    by_class: dict = {}

    @classmethod
    def classify(cls, error: Optional[BaseException] = None, status_code: Optional[int] = None) -> str:
        if status_code == CLIENT_CANCELLED_STATUS:
            return "http_299"
        if _is_timeout(error):
            return "timeout"
        return "other_error"

    @classmethod
    def record(cls, kind: str, class_name: Optional[str] = None) -> None:
        cls.counts[kind] = cls.counts.get(kind, 0) + 1
        if class_name is not None:
            per_class = cls.by_class.setdefault(class_name, {})
            per_class[kind] = per_class.get(kind, 0) + 1

    @classmethod
    def reset(cls) -> None:
        cls.counts = {}
        cls.by_class = {}


def request_counts(stats) -> tuple[int, int]:
    """Requests sent and failed, excluding the METRIC rows the custom metrics are logged as."""
    entries = [e for (name, method), e in stats.entries.items() if method != "METRIC"]
    return sum(e.num_requests for e in entries), sum(e.num_failures for e in entries)


def request_outcome_entries(stats, class_names: Optional[list[str]] = None) -> dict[str, Any]:
    """Timeouts, HTTP 299s, --cancel-rate cancellations and other failures, plus the client-cancelled fraction
    (timeouts and 299s) of all requests sent."""
    sent, _ = request_counts(stats)
    entries = {f"requests_{kind}": RequestOutcomes.counts.get(kind, 0) for kind in RequestOutcomes.KINDS}
    cancelled = entries["requests_timeout"] + entries["requests_http_299"]
    entries["client_cancelled_fraction"] = cancelled / sent if sent else 0.0
    for name in class_names or []:
        per_class = RequestOutcomes.by_class.get(name, {})
        entries[f"class_{name}_timeouts"] = per_class.get("timeout", 0)
        entries[f"class_{name}_http_299"] = per_class.get("http_299", 0)
    return entries


@dataclass
class ChunkMetadata:
    text: str
    logprob_tokens: Optional[int]
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    cached_tokens: Optional[int]
    has_reasoning_only: bool = False  # chunk has reasoning/thinking content but no visible content


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
        if getattr(self.parsed_options, "systemone", False):
            return "/v1/systemone"
        if self.parsed_options.rerank:
            return "/v1/rerank"
        elif self.parsed_options.embeddings:
            return "/v1/embeddings"
        elif self.parsed_options.chat:
            return "/v1/chat/completions"
        else:
            return "/v1/completions"

    def format_payload(self, prompt, max_tokens, images):
        if getattr(self.parsed_options, "systemone", False):
            # Model is sent raw: the gateway keys its packed-path profile on the exact string
            return {"model": self.model, "state": prompt["state"], "questions": prompt["questions"]}

        if self.parsed_options.rerank:
            if isinstance(prompt, list):
                documents = prompt
            else:
                documents = [doc.strip() for doc in prompt.split("\n\n") if doc.strip()]
            query = self.parsed_options.rerank_query or "Find the most relevant document."
            data = {
                "model": self.model,
                "query": query,
                "documents": documents,
            }
            if self.parsed_options.rerank_top_n is not None:
                data["top_n"] = self.parsed_options.rerank_top_n
            if not self.parsed_options.rerank_return_documents:
                data["return_documents"] = False
            return data

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
            if getattr(self.parsed_options, "embeddings_dimensions", None) is not None:
                data["dimensions"] = self.parsed_options.embeddings_dimensions
            if getattr(self.parsed_options, "embeddings_prompt_template", None) is not None:
                data["prompt_template"] = self.parsed_options.embeddings_prompt_template
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
        if getattr(self.parsed_options, "systemone", False):
            usage = data.get("usage") or {}
            answers = data.get("answers") or {}
            choices = [f"{qid}:{(answers[qid] or {}).get('choice')}" for qid in sorted(answers)]
            return ChunkMetadata(
                text=", ".join(choices),
                logprob_tokens=None,
                completion_tokens=usage.get("output_tokens"),
                prompt_tokens=usage.get("input_tokens"),
                cached_tokens=usage.get("cached_input_tokens"),
            )

        if self.parsed_options.rerank:
            usage = data.get("usage", {})
            scores = [
                f"{item['index']}:{item['relevance_score']:.4f}" for item in data.get("results", data.get("data", []))
            ]
            return ChunkMetadata(
                text=", ".join(scores),
                logprob_tokens=None,
                completion_tokens=None,
                prompt_tokens=usage.get("prompt_tokens"),
                cached_tokens=None,
            )

        if self.parsed_options.embeddings:
            return ChunkMetadata(
                text=data["data"][0]["embedding"],
                logprob_tokens=None,
                completion_tokens=None,
                prompt_tokens=None,
                cached_tokens=None,
            )
        usage = data.get("usage", None)

        # Trailing usage-only SSE chunk (choices absent or empty) — return metadata
        # without trying to extract text from a missing choice.
        if len(data.get("choices", [])) == 0:
            prompt_tokens_details = (usage or {}).get("prompt_tokens_details") or {}
            return ChunkMetadata(
                text="",
                logprob_tokens=None,
                completion_tokens=usage.get("completion_tokens") if usage else None,
                prompt_tokens=usage.get("prompt_tokens") if usage else None,
                cached_tokens=prompt_tokens_details.get("cached_tokens"),
            )

        assert len(data["choices"]) == 1, f"Too many choices {len(data['choices'])}"
        choice = data["choices"][0]
        if self.parsed_options.chat:
            if self.parsed_options.stream:
                block = choice["delta"]
            else:
                block = choice["message"]
            has_content = bool(block.get("content"))
            has_reasoning = bool(block.get("reasoning_content") or block.get("reasoning"))
            has_reasoning_only = has_reasoning and not has_content
            text = (
                (block.get("reasoning", "") or "")
                + (block.get("reasoning_content", "") or "")
                + (block.get("content", "") or "")
            )
        else:
            # Completions API: no reasoning_content field exists in the schema;
            # thinking tokens (if any) appear raw in text and cannot be separated.
            has_reasoning_only = False
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
            has_reasoning_only=has_reasoning_only,
            logprob_tokens=logprob_tokens,
            completion_tokens=usage["completion_tokens"] if usage else None,
            prompt_tokens=usage.get("prompt_tokens", None) if usage else None,
            cached_tokens=cached_tokens,
        )


class FireworksProvider(OpenAIProvider):
    def __init__(self, model, parsed_options):
        super().__init__(model, parsed_options)
        raw = parsed_options.acceptance_probs_override
        if raw is not None:
            try:
                val = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"--acceptance-probs-override must be valid JSON: {e}")
            if not isinstance(val, list) or not all(isinstance(x, (int, float)) for x in val):
                raise ValueError("--acceptance-probs-override must be a JSON array of numbers")
            self._acceptance_probs_override = val
        else:
            self._acceptance_probs_override = None

        forced_gen_path = parsed_options.forced_generation_file
        forced_gen_from_dataset = parsed_options.forced_generation_from_dataset
        if forced_gen_path and forced_gen_from_dataset:
            raise ValueError("--forced-generation-file and --forced-generation-from-dataset are mutually exclusive")

        if forced_gen_path is not None:
            self._forced_generation_pool = itertools.cycle(self._load_forced_generation_texts(forced_gen_path))
        elif forced_gen_from_dataset:
            assert parsed_options.tokenizer is not None, "--tokenizer is required for --forced-generation-from-dataset"
            ds = DatasetHolder.get_forced_generation_instance(
                dataset=parsed_options.dataset,
                tokenizer=parsed_options.tokenizer,
                max_tokens=parsed_options.max_tokens,
            )
            self._forced_generation_pool = (text for text, _tokens in ds)
        else:
            self._forced_generation_pool = None

    @staticmethod
    def _load_forced_generation_texts(path):
        texts = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "text" in obj:
                        texts.append(obj["text"])
                    else:
                        texts.append(line)
                except json.JSONDecodeError:
                    texts.append(line)
        if not texts:
            raise ValueError(f"--forced-generation-file '{path}' contains no text entries")
        logger.info(f"Loaded {len(texts)} forced generation texts from {path}")
        return texts

    def format_payload(self, prompt, max_tokens, images):
        data = super().format_payload(prompt, max_tokens, images)
        if self.parsed_options.rerank or getattr(self.parsed_options, "systemone", False):
            return data
        if self.parsed_options.force_min_tokens:
            data["min_tokens"] = max_tokens
        if getattr(self.parsed_options, "ignore_eos", False):
            data["ignore_eos"] = True
        # Enable perf_metrics_in_response to get speculation stats in streaming responses
        data["perf_metrics_in_response"] = True
        # Only send prompt_cache_max_len when the user explicitly opted in (>0). The
        # default (0 = no caching) is a no-op for the server, but unconditionally
        # adding the key breaks deployments whose OpenAI-compat schema is configured
        # with extra="forbid" (e.g. some TRT-LLM and vLLM-style serving images).
        if self.parsed_options.prompt_cache_max_len > 0:
            data["prompt_cache_max_len"] = self.parsed_options.prompt_cache_max_len
        if self._acceptance_probs_override is not None:
            data["acceptance_probs_override"] = self._acceptance_probs_override
        if self._forced_generation_pool is not None:
            data["forced_generation"] = next(self._forced_generation_pool)
        return data

    def post_response_hook(self, headers, num_tokens, perf_metrics=None):
        """Process Fireworks-specific response for speculation hit rate tracking.

        For streaming responses, speculation stats come from perf_metrics in the response body.
        For non-streaming responses, they come from headers.

        Only records speculation hit rates for generations with >= 30 tokens to ensure
        meaningful statistics.
        """
        def get_perf_metric(name):
            value = (perf_metrics or {}).get(name)
            if value is None:
                value = headers.get(f"fireworks-{name}")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        # FireAttention and fw-proxy expose these in seconds. Record them as
        # Locust metrics in milliseconds so they use the same units as existing
        # client-side latency rows. This is best-effort: it runs before
        # response.success() inside a catch_response block, so any failure here
        # must never mark the request as failed or perturb the existing
        # client-side tables.
        try:
            server_processing_time = get_perf_metric("server-processing-time")
            if server_processing_time is not None:
                add_custom_metric("server_side_total_latency", server_processing_time * 1000)

            server_ttft = get_perf_metric("server-time-to-first-token")
            if server_ttft is not None:
                add_custom_metric("server_side_time_to_first_token", server_ttft * 1000)
        except Exception as e:
            logger.debug(f"Skipping server-side perf metrics: {e}")

        # Only track speculation hit rates for sufficiently long generations
        if num_tokens is None or num_tokens < 30:
            return

        # Try to get speculation acceptance from perf_metrics first (streaming),
        # then fall back to headers (non-streaming)
        speculation_hit_rates = None
        if perf_metrics:
            speculation_hit_rates = perf_metrics.get("speculation-acceptance")
            if self.parsed_options.show_response:
                logger.debug(f"perf_metrics: {perf_metrics}")
        if not speculation_hit_rates:
            speculation_hit_rates = headers.get("fireworks-speculation-acceptance")
            if self.parsed_options.show_response:
                logger.debug(f"Response headers: {dict(headers)}")

        if not speculation_hit_rates:
            if self.parsed_options.show_response:
                logger.debug("speculation-acceptance not found in perf_metrics or headers")
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
            logger.warning(f"Failed to parse speculation hit rates '{speculation_hit_rates}': {e}")


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


class TritonInferProvider(BaseProvider):
    """Triton Inference Server — /v2/models/{model}/infer endpoint (non-streaming only)."""

    DEFAULT_MODEL_NAME = "ensemble"

    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        assert not self.parsed_options.stream, "Streaming is not supported"
        assert self.parsed_options.n == 1, "n > 1 is not supported"
        return f"/v2/models/{self.model}/infer"

    def format_payload(self, prompt, max_tokens, images):
        assert isinstance(prompt, str), "prompt must be a string"
        assert images is None, "images are not supported"
        assert self.parsed_options.logprobs is None, "logprobs are not supported"
        # Store prompt so parse_output_json can strip the echoed prefix from the response.
        self._last_prompt = prompt
        return {
            "inputs": [
                {"name": "text_input", "datatype": "BYTES", "shape": [1, 1], "data": [[prompt]]},
                {"name": "max_tokens", "datatype": "UINT32", "shape": [1, 1], "data": [[max_tokens]]},
                {"name": "bad_words", "datatype": "BYTES", "shape": [1, 1], "data": [[""]]},
                {"name": "stop_words", "datatype": "BYTES", "shape": [1, 1], "data": [[""]]},
                {
                    "name": "temperature",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [[self.parsed_options.temperature]],
                },
            ]
        }

    def parse_output_json(self, data):
        prompt = getattr(self, "_last_prompt", "")
        for output in data["outputs"]:
            if output["name"] == "text_output":
                assert output["datatype"] == "BYTES"
                assert output["shape"] == [1]
                text = output["data"][0]
                # Triton echoes the prompt in the response; strip it.
                text = text.removeprefix("<s> ")
                if text.startswith(prompt):
                    text = text[len(prompt) :].removeprefix(" ")
                else:
                    logger.warning("prompt not found in the output")
                return ChunkMetadata(
                    text=text,
                    logprob_tokens=None,
                    completion_tokens=None,
                    prompt_tokens=None,
                    cached_tokens=None,
                )
        raise ValueError("text_output not found in the response")


class TritonGenerateProvider(BaseProvider):
    """Triton Inference Server — /v2/models/{model}/generate[_stream] endpoint."""

    DEFAULT_MODEL_NAME = "ensemble"

    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        assert self.parsed_options.n == 1, "n > 1 is not supported"
        stream_suffix = "_stream" if self.parsed_options.stream else ""
        return f"/v2/models/{self.model}/generate{stream_suffix}"

    def format_payload(self, prompt, max_tokens, images):
        assert isinstance(prompt, str), "prompt must be a string"
        assert images is None, "images are not supported"
        assert self.parsed_options.logprobs is None, "logprobs are not supported"
        # Store prompt so parse_output_json can strip the echoed prefix in non-streaming mode.
        self._last_prompt = prompt
        return {
            "text_input": prompt,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
            "bad_words": "",
            "stop_words": "",
        }

    def parse_output_json(self, data):
        text = data["text_output"]
        if not self.parsed_options.stream:
            # Non-streaming: Triton echoes the prompt in the response; strip it.
            prompt = getattr(self, "_last_prompt", "")
            text = text.removeprefix("<s> ")
            if text.startswith(prompt):
                text = text[len(prompt) :].removeprefix(" ")
            else:
                logger.warning("prompt not found in the output")
        return ChunkMetadata(
            text=text,
            logprob_tokens=None,
            completion_tokens=None,
            prompt_tokens=None,
            cached_tokens=None,
        )


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
    "anyscale": OpenAIProvider,
    "together": TogetherProvider,
    "tgi": TgiProvider,
    "triton-infer": TritonInferProvider,
    "triton-generate": TritonGenerateProvider,
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


_SYSTEMONE_WORKER_SEEDS = itertools.count()


class LLMUser(HttpUser):
    # no wait time, so every user creates a continuous load, sending requests as quickly as possible

    # In-flight paced sends (--qps, --replay-schedule) of all users, so the end of a replay can wait for them
    _paced_sends = gevent.pool.Group()

    def on_start(self):
        try:
            self._on_start()
        except Exception as e:
            logger.error(f"Failed to initialize: {repr(e)}")
            logger.error(traceback.format_exc())
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
            elif self.environment.parsed_options.systemone:
                # /v1/systemone only exists on the Fireworks gateway
                self.provider = "fireworks"

        if self.environment.parsed_options.systemone and self.model is None:
            raise ValueError("--systemone requires --model (sent verbatim, e.g. accounts/<acct>/deployments/<id>)")

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
        logger.info(f" Provider {self.provider} using model {self.model} ".center(80, "*"))
        self.provider_formatter = PROVIDER_CLASS_MAP[self.provider](self.model, self.environment.parsed_options)

        self.stream = self.environment.parsed_options.stream
        # Rerank, embeddings and systemone endpoints are non-streaming single-response APIs
        if (
            self.environment.parsed_options.rerank
            or self.environment.parsed_options.embeddings
            or self.environment.parsed_options.systemone
        ):
            self.stream = False
        if self.environment.parsed_options.systemone and (
            self.environment.parsed_options.rerank or self.environment.parsed_options.embeddings
        ):
            raise ValueError("--systemone is mutually exclusive with --rerank and --embeddings")
        self.shape_mix_dataset = None
        if self.environment.parsed_options.shape_mix:
            self._validate_shape_mix_options()
            # Built before the pacer starts its clock so the setup cost doesn't delay the first arrivals
            self.shape_mix_dataset = DatasetHolder.get_shape_mix_instance(
                self.environment.parsed_options, getattr(self.environment.runner, "worker_index", 0) or 0
            )
            self.prompt_tokenizer_tokens = None
        self.ramping_pacer = None  # Will be set if --ramping-time is used
        self.qps_pacer = None  # Will be set if --qps is used

        image_resolutions = self.environment.parsed_options.prompt_images_with_resolutions
        self.prompt_images = None
        if self.environment.parsed_options.first_visible_token:
            if not self.environment.parsed_options.chat:
                raise AssertionError(
                    "--first-visible-token requires --chat. The completions API exposes thinking tokens "
                    "inline in `text` without a separate `reasoning_content` field, so thinking and "
                    "visible tokens cannot be distinguished."
                )
            if not self.environment.parsed_options.stream:
                raise AssertionError(
                    "--first-visible-token requires --stream. Non-streaming responses arrive all at once, "
                    "so there is no per-chunk timestamp to distinguish thinking from visible tokens."
                )

        if image_resolutions:
            if self.environment.parsed_options.systemone:
                raise AssertionError("--prompt-images-with-resolutions is not supported with --systemone.")
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
            # "generation_tokens" is a backward-compat alias for "completion_tokens";
            # jarvis/perfagent/gcs.go reads the CSV column "Generation Tokens".
            "generation_tokens": str(self.max_tokens_sampler),
            "completion_tokens": str(self.max_tokens_sampler),
            "stream": self.stream,
            "temperature": self.temperature,
            "logprobs": self.environment.parsed_options.logprobs,
        }

        if self.environment.parsed_options.top_k is not None:
            logging_params["top_k"] = self.environment.parsed_options.top_k

        if self.environment.parsed_options.systemone:
            # Generation params don't apply to systemone; log the sweep params instead
            logging_params = {
                "provider": self.provider,
                "model": self.model,
                "mode": "systemone",
                "num_questions": self.environment.parsed_options.systemone_num_questions,
                "num_options": self.environment.parsed_options.systemone_num_options,
                "state_tokens": self.environment.parsed_options.systemone_state_tokens,
                "question_tokens": self.environment.parsed_options.systemone_question_tokens,
                "unique_state": self.environment.parsed_options.systemone_unique_state,
            }

        if self.environment.parsed_options.shape_mix:
            mix = parse_shape_mix(self.environment.parsed_options.shape_mix)
            logging_params["prompt_tokens"] = round(mix.weighted_mean("prompt_tokens"))
            logging_params["generation_tokens"] = f"shape-mix mean {mix.weighted_mean('generated_tokens'):.0f}"
            logging_params["completion_tokens"] = logging_params["generation_tokens"]
            logging_params["shape_mix"] = mix.describe()
            logging_params["shape_mix_seed"] = mix.seed
            logging_params["shape_mix_request_seed"] = resolve_request_seed(self.environment.parsed_options)
            logging_params["shape_mix_output_mode"] = self.environment.parsed_options.shape_mix_output_mode
            logging_params["shape_mix_sample_lengths"] = self.environment.parsed_options.shape_mix_sample_lengths
            logging_params["prefix_popularity_zipf"] = self.environment.parsed_options.prefix_popularity_zipf or ""
            logging_params["prefix_restore_boundary"] = resolve_prefix_restore_boundary(self.environment.parsed_options)

        InitTracker.notify_init(self.environment, logging_params)

        if getattr(self.environment.parsed_options, "replay_schedule", None):
            options = self.environment.parsed_options
            if not options.shape_mix:
                raise ValueError("--replay-schedule needs --shape-mix (its classes and prefix pools)")
            if options.qps is not None or options.burst or options.ramping_time is not None or options.prefix_fill:
                raise ValueError(
                    "--replay-schedule sets the send times; drop --qps, --burst, --ramping-time and --prefix-fill"
                )
            names = tuple(c.name for c in parse_shape_mix(options.shape_mix).classes)
            self.qps_pacer = ReplayPacer.instance(options, names)
            self.wait_time = lambda: 0
            self._in_flight = LLMUser._paced_sends
            adapter = _timed_adapter(pool_maxsize=QPS_CONNECTION_POOL_SIZE)
            self.client.mount("http://", adapter)
            self.client.mount("https://", adapter)
        elif getattr(self.environment.parsed_options, "prefix_fill", False):
            options = self.environment.parsed_options
            if not options.shape_mix:
                raise ValueError("--prefix-fill needs --shape-mix")
            if options.qps is not None or options.burst or options.ramping_time is not None:
                raise ValueError(
                    "--prefix-fill sends at --prefix-fill-concurrency; drop --qps, --burst and --ramping-time"
                )
            if options.prefix_fill_concurrency < 1:
                raise ValueError("--prefix-fill-concurrency must be >= 1")
            self.wait_time = lambda: 0
            adapter = _timed_adapter(pool_maxsize=max(QPS_CONNECTION_POOL_SIZE, options.prefix_fill_concurrency))
            self.client.mount("http://", adapter)
            self.client.mount("https://", adapter)
        elif self.environment.parsed_options.qps is not None:
            if self.environment.parsed_options.burst:
                raise ValueError("Burst and QPS modes are mutually exclusive")
            self.qps_pacer = FixedQPSPacer.instance(
                self.environment.parsed_options.qps,
                self.environment.parsed_options.qps_distribution,
                resolve_qps_cv2(self.environment.parsed_options),
                resolve_request_seed(self.environment.parsed_options),
                self.environment.parsed_options.qps_modulation_window,
                resolve_qps_dispersion(self.environment.parsed_options),
                rate_block_windows(self.environment.parsed_options),
            )
            # Open loop: the task claims a send time, prepares the request, sleeps until that time and sends in a
            # child greenlet, so slow responses never delay later sends and users only bound preparation lookahead
            self.wait_time = lambda: 0
            self._in_flight = LLMUser._paced_sends
            adapter = _timed_adapter(pool_maxsize=QPS_CONNECTION_POOL_SIZE)
            self.client.mount("http://", adapter)
            self.client.mount("https://", adapter)
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

        if self.environment.parsed_options.systemone:
            self._init_systemone_dataset()
            return

        if self.shape_mix_dataset is not None:
            return

        dataset = DatasetHolder.get_instance(self.environment.parsed_options)
        self.dataset = iter(dataset)
        # Once per process, not per user: users start inside the run window, and tokenizing a long prompt for
        # each of them blocked the event loop there (C-151)
        self.prompt_tokenizer_tokens = DatasetHolder.prompt_token_estimate(
            self.environment.parsed_options, self._get_input
        )

        # Override dataset with synthetic rerank documents if num_documents or tokens_per_document is set
        if self.environment.parsed_options.rerank and (
            getattr(self.environment.parsed_options, "num_documents", None) is not None
            or getattr(self.environment.parsed_options, "tokens_per_document", None) is not None
        ):
            num_docs = getattr(self.environment.parsed_options, "num_documents", None) or 10
            tokens_per_doc = getattr(self.environment.parsed_options, "tokens_per_document", None) or (
                self.environment.parsed_options.prompt_tokens // num_docs
            )
            synthetic_prompt = "\n\n".join(PROMPT_PREFIX_TOKEN * tokens_per_doc for _ in range(num_docs))
            self.dataset = iter(itertools.cycle([(synthetic_prompt, num_docs * tokens_per_doc)]))

    def _validate_shape_mix_options(self):
        options = self.environment.parsed_options
        if options.systemone or options.embeddings or options.rerank:
            raise ValueError("--shape-mix only applies to text generation (not --systemone, --embeddings, --rerank)")
        if options.dataset != "limericks":
            raise ValueError("--shape-mix builds its own prompts and can't be combined with --dataset")
        if options.prompt_cache_max_len > 0:
            raise ValueError("--shape-mix sets cached tokens per class; drop -pcml/--prompt-cache-max-len")
        if options.prompt_images_with_resolutions:
            raise ValueError("--shape-mix doesn't support --prompt-images-with-resolutions")
        if not isinstance(self.provider_formatter, OpenAIProvider):
            raise ValueError(f"--shape-mix needs an OpenAI-compatible provider, got {self.provider}")
        if options.shape_mix_output_mode == "max" and options.force_min_tokens:
            raise ValueError("--shape-mix-output-mode max lets the model stop early; drop --force-min-tokens")

    def _init_systemone_dataset(self):
        options = self.environment.parsed_options
        # Only size with a real tokenizer when one is given; otherwise use the chars/token heuristic
        tokenizer = InitTracker.load_tokenizer(options.tokenizer) if options.tokenizer else None
        self.systemone_dataset = SystemOneDataset(
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "limericks.txt"),
            tokenizer=tokenizer,
            num_questions=options.systemone_num_questions,
            num_options=options.systemone_num_options,
            state_tokens=options.systemone_state_tokens,
            question_tokens=options.systemone_question_tokens,
            unique_state=options.systemone_unique_state,
            seed=next(_SYSTEMONE_WORKER_SEEDS),
        )
        self.dataset = self.systemone_dataset
        self.prompt_tokenizer_tokens = self.systemone_dataset.state_tokens
        ds = self.systemone_dataset
        logger.info(
            f"systemone: state ~{ds.state_tokens} tokens, "
            f"{len(ds.question_ids)} questions of ~{max(ds.question_tokens)} tokens"
        )

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
        if getattr(self.environment.parsed_options, "prefix_fill", False):
            self._run_prefix_fill()
            return
        if self.qps_pacer is not None:
            self._paced_request()
            return
        # Wait for capacity and track concurrent requests for ramping pacer
        if self.ramping_pacer:
            self._wait_for_ramping_capacity()
            self.ramping_pacer.request_start()
        try:
            self._do_generate_text()
        finally:
            if self.ramping_pacer:
                self.ramping_pacer.request_end()

    def _run_prefix_fill(self):
        """Send every prefix once (see ShapeMixDataset.fill_request) at --prefix-fill-concurrency, then quit."""
        if PrefixFill.started:
            gevent.sleep(1)
            return
        PrefixFill.started = True
        InitTracker.spawned.wait()
        ds = self.shape_mix_dataset
        plan = ds.fill_plan()
        concurrency = self.environment.parsed_options.prefix_fill_concurrency
        logger.info(
            f"--prefix-fill: sending {len(plan)} prefixes ({ds.pool_tokens / 1e6:.1f}M prefix tokens) at concurrency {concurrency}"
        )
        pool = gevent.pool.Pool(concurrency)
        started = time.monotonic()
        for ci, j in plan:
            request = ds.fill_request(ci, j)
            data, headers = build_shape_mix_request(self.provider_formatter, request)
            PrefixFill.prompt_tokens_sent += request.prompt_tokens
            PrefixFill.prefix_tokens_sent += ds._head + ds.prefix_tokens[ci]
            PrefixFill.requests += 1
            pool.spawn(self._send_paced, (data, headers, 1, request.prompt_tokens, request, json.dumps(data)), None)
        pool.join()
        PrefixFill.seconds = time.monotonic() - started
        logger.info(
            f"--prefix-fill: sent {PrefixFill.requests} requests, {PrefixFill.prompt_tokens_sent} prompt tokens in "
            f"{PrefixFill.seconds:.1f}s; quitting"
        )
        # From its own greenlet: stopping the runner kills the user greenlets, including this one
        gevent.spawn(InitTracker._do_quit)
        gevent.sleep(1)

    def _paced_request(self):
        InitTracker.spawned.wait()
        if isinstance(self.qps_pacer, ReplayPacer):
            claimed = self.qps_pacer.claim_entry()
            if claimed is None:
                self._finish_replay()
                return
            try:
                intended, epoch, entry = claimed
                prepared = self._prepare_text_request(self.shape_mix_dataset.replay_request(entry))
                self._pace_and_send(prepared, intended, epoch, replay_entry=entry)
            finally:
                self.qps_pacer.release()
            return
        intended, epoch = self.qps_pacer.claim_with_epoch()
        self._pace_and_send(self._prepare_request(), intended, epoch)

    def _pace_and_send(self, prepared, intended: float, epoch: int, replay_entry=None) -> None:
        delay = intended - time.monotonic()
        max_lateness = getattr(self.environment.parsed_options, "qps_max_lateness", DEFAULT_MAX_LATENESS_SECONDS)
        if replay_entry is not None:
            pacer = self.qps_pacer
            if delay > 0:
                gevent.sleep(delay)
                # Measured after waking: a stall while asleep makes the send late too
                delay = intended - time.monotonic()
            late = max(0.0, -delay)
            pacer.max_lateness_seen = max(pacer.max_lateness_seen, late)
            if late > max_lateness:
                # Rows are recorded traffic: count every late one; send it unless asked to skip
                pacer.late_rows += 1
                InitTracker.warn_late_send(late, max_lateness)
                if self.environment.parsed_options.replay_late_rows == "skip":
                    pacer.skipped_rows += 1
                    SendTimeline.skipped += 1
                    return
            elif late > LATE_SEND_WARNING_SECONDS:
                InitTracker.warn_late_send(late)
            pacer.sent_rows += 1
            pacer.cancelled_rows += bool(replay_entry.cancelled)
            self._in_flight.spawn(self._send_paced, prepared, intended)
            return
        tracked = self.qps_pacer.track(intended, epoch)
        try:
            while True:
                if delay > 0:
                    gevent.sleep(delay)
                # Adopt any shift made meanwhile (users asleep during a stall wake together past their times)
                shift, epoch = self.qps_pacer.shift_since(epoch)
                intended += shift
                delay = intended - time.monotonic()
                if shift and delay > 0:
                    continue
                if delay >= -max_lateness and self.qps_pacer.oldest_lateness() <= max_lateness:
                    break
                # A stall: the schedule moves later so the oldest pending send is due now, and every send claimed
                # before the shift moves along, so the stall's backlog is neither sent as a burst nor dropped
                if self.qps_pacer.resync(epoch, max(0.0, -delay)):
                    InitTracker.warn_late_send(self.qps_pacer.shifts[-1], max_lateness)
        finally:
            self.qps_pacer.untrack(tracked)
        if delay < -LATE_SEND_WARNING_SECONDS:
            InitTracker.warn_late_send(-delay)
        self._in_flight.spawn(self._send_paced, prepared, intended)

    def _finish_replay(self):
        """The schedule is exhausted: once (from the first user to notice), wait for in-flight requests, then quit."""
        if ReplayPacer.finishing:
            gevent.sleep(1)
            return
        ReplayPacer.finishing = True
        # Every entry is claimed, but users may still be sleeping until their claimed send times
        while self.qps_pacer.pending:
            gevent.sleep(0.05)
        logger.info(f"--replay-schedule: all {len(self.qps_pacer.entries)} requests sent; waiting for responses")
        timeout = request_timeout(self.environment.parsed_options)
        LLMUser._paced_sends.join(timeout=None if timeout is None else timeout + 5)
        # From its own greenlet: stopping the runner kills the user greenlets, including this one
        gevent.spawn(InitTracker._do_quit)
        gevent.sleep(1)

    def _send_paced(self, prepared, intended: Optional[float]):
        try:
            if self.environment.parsed_options.systemone:
                self._do_systemone_request(prepared, intended)
            else:
                self._do_generate_text(prepared, intended)
        except Exception as e:
            logger.error(f"Paced request failed: {e!r}\n{traceback.format_exc()}")

    def _prepare_request(self):
        if self.environment.parsed_options.systemone:
            return next(self.dataset)[0]
        return self._prepare_text_request()

    def _do_systemone_request(self, prompt=None, intended=None):
        if prompt is None:
            prompt, _ = next(self.dataset)
        question_ids = self.systemone_dataset.question_ids
        data = self.provider_formatter.format_payload(prompt, 0, None)
        if self.environment.parsed_options.show_request:
            print("--- Request payload ---")
            print(json.dumps(data, indent=2))
            print("---")
        dispatched = _begin_wire_timing()
        t_start = time.perf_counter()

        with self.client.post(
            self.provider_formatter.get_url(),
            data=json.dumps(data),
            catch_response=True,
            timeout=request_timeout(self.environment.parsed_options),
        ) as response:
            if intended is not None:
                SendTimeline.record(intended, _wire_send_time(dispatched))
            body = response.content
            dur_total = time.perf_counter() - t_start
            if response.status_code != 200:
                outcome = RequestOutcomes.classify(getattr(response, "error", None), response.status_code)
                RequestOutcomes.record(outcome)
                response.failure(f"Error in response ({outcome}): HTTP {response.status_code}: {response.text}")
                return
            try:
                resp_data = orjson.loads(body)
            except Exception as e:
                logger.error(f"Failed to parse response body with error {repr(e)}")
                response.failure(e)
                return
            error = validate_systemone_response(resp_data, question_ids)
            if error is not None:
                response.failure(error)
                return
            out = self.provider_formatter.parse_output_json(resp_data)
            input_tokens = out.prompt_tokens or 0
            output_tokens = out.completion_tokens or 0
            # The packed single-pass path reads logits without decoding, so it reports zero output tokens
            packed = output_tokens == 0

            logger.info(
                f"Response received: total {dur_total*1000:.2f} ms, {len(question_ids)} questions, "
                f"{input_tokens} input, {out.cached_tokens} cached, {output_tokens} output, packed={packed}"
            )
            if self.environment.parsed_options.show_response:
                print("---")
                print(out.text)
                print("---")
            add_custom_metric("total_latency", dur_total * 1000)
            add_custom_metric("num_questions", len(question_ids))
            add_custom_metric("input_tokens", input_tokens)
            add_custom_metric("output_tokens", output_tokens)
            if out.cached_tokens is not None:
                add_custom_metric("cached_input_tokens", out.cached_tokens)
            add_custom_metric("packed_fraction", 1 if packed else 0)
            add_custom_metric("response_bytes", len(body))
            self.provider_formatter.post_response_hook(response.headers, 0)
            record_systemone_server_timings(response.headers)

            response.success()

            if not self.first_done:
                self.first_done = True
                InitTracker.notify_first_request()

            InitTracker.notify_request_complete()

    def _prepare_text_request(self, shape_request: Optional[ShapeRequest] = None):
        request_headers = None
        is_embeddings = self.provider_formatter.parsed_options.embeddings
        batch_size = getattr(self.environment.parsed_options, "embeddings_batch_size", 1) or 1
        if self.shape_mix_dataset is not None:
            if shape_request is None:
                shape_request = next(self.shape_mix_dataset)
            max_tokens = shape_request.generated_tokens
            prompt_tokens = shape_request.prompt_tokens
            data, request_headers = build_shape_mix_request(
                self.provider_formatter, shape_request, self.environment.parsed_options.shape_mix_output_mode
            )
        elif is_embeddings and batch_size > 1:
            max_tokens = self.max_tokens_sampler.sample()
            prompts = []
            for _ in range(batch_size):
                p, _, _ = self._get_input()
                prompts.append(p)
            prompt, prompt_tokens, images = prompts, 0, None
        else:
            max_tokens = self.max_tokens_sampler.sample()
            prompt, prompt_tokens, images = self._get_input()
        if shape_request is None:
            data = self.provider_formatter.format_payload(prompt, max_tokens, images)
        return data, request_headers, max_tokens, prompt_tokens, shape_request, json.dumps(data)

    def _do_generate_text(self, prepared=None, intended=None):
        if self.environment.parsed_options.systemone:
            self._do_systemone_request()
            return
        if prepared is None:
            prepared = self._prepare_text_request()
        data, request_headers, max_tokens, prompt_tokens, shape_request, body = prepared
        batch_size = getattr(self.environment.parsed_options, "embeddings_batch_size", 1) or 1
        if self.environment.parsed_options.show_request:
            print("--- Request payload ---")
            print(json.dumps(data, indent=2))
            print("---")
        dispatched = _begin_wire_timing()
        t_start = time.perf_counter()

        with self.client.post(
            self.provider_formatter.get_url(),
            data=body,
            headers=request_headers,
            stream=True,
            catch_response=True,
            timeout=request_timeout(self.environment.parsed_options),
        ) as response:
            if intended is not None:
                SendTimeline.record(intended, _wire_send_time(dispatched))
            class_name = shape_request.class_name if shape_request is not None else None
            combined_text = ""
            done = False
            completion_tokens = None
            total_logprob_tokens = None
            cached_tokens = None
            perf_metrics = None  # Capture perf_metrics from response body (streaming)
            try:
                response.raise_for_status()
            except Exception as e:
                # Use locust's failure-recording API instead of raising so the
                # failure is counted in stats.total.num_failures (visible in
                # the summary, stats CSV, HTML report, and stats_failures.csv)
                # rather than being routed to runner.exceptions.
                outcome = RequestOutcomes.classify(getattr(response, "error", None) or e, response.status_code)
                RequestOutcomes.record(outcome, class_name)
                response.failure(f"Error in response ({outcome}): {response.text or repr(e)}")
                return
            if response.status_code == CLIENT_CANCELLED_STATUS:
                RequestOutcomes.record("http_299", class_name)
                response.failure(f"Request cancelled (HTTP {CLIENT_CANCELLED_STATUS})")
                return
            t_first_token = None
            t_first_visible_token = None

            # Rerank and embeddings are single-response JSON APIs. Draining them through
            # iter_lines(delimiter=b"\n\n") makes the interpreter scan the whole body for a
            # separator a JSON document never contains, so a multi-megabyte batched
            # embeddings response gets buffered and rescanned in Python. Under gevent that
            # scan is not cooperative, so it serializes across greenlets and caps client
            # throughput well below what the endpoint can serve. Read the body once here
            # and leave the chunk loop to the streaming paths.
            if self.provider_formatter.parsed_options.rerank or self.provider_formatter.parsed_options.embeddings:
                try:
                    body = response.content
                except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError) as e:
                    outcome = RequestOutcomes.classify(e)
                    RequestOutcomes.record(outcome, class_name)
                    response.failure(f"Error reading response ({outcome}): {e!r}")
                    return
                # Recorded so a client-side ceiling is visible in the results rather than
                # being mistaken for server latency.
                add_custom_metric("response_bytes", len(body))
                show_response = self.environment.parsed_options.show_response

                # An empty or unparseable body is a failed request, not a fast success. Record
                # it via locust's failure API so fail_ratio/--max-fail-ratio catch it instead
                # of the parse error landing in runner.exceptions.
                if not body:
                    response.failure(Exception("empty response received"))
                    return
                try:
                    if self.provider_formatter.parsed_options.rerank:
                        out = self.provider_formatter.parse_output_json(orjson.loads(body))
                        if out.prompt_tokens:
                            prompt_tokens = out.prompt_tokens
                        if show_response:
                            combined_text = out.text
                    elif show_response:
                        resp_data = orjson.loads(body)
                        server_prompt_tokens = (resp_data.get("usage") or {}).get("prompt_tokens")
                        out = self.provider_formatter.parse_output_json(resp_data)
                        combined_text = str(out.text)
                    else:
                        server_prompt_tokens = extract_prompt_tokens(body)
                except Exception as e:
                    logger.error(f"Failed to parse response body with error {repr(e)}")
                    response.failure(e)
                    return

                # Only count a first token once the body has parsed successfully, so an empty
                # or malformed response is reported as a failure rather than a zero-latency success.
                now = time.perf_counter()
                t_first_token = now
                if self.provider_formatter.parsed_options.embeddings:
                    if server_prompt_tokens:
                        prompt_tokens = server_prompt_tokens
                    add_custom_metric("latency_per_embedding", (now - t_start) / batch_size * 1000)

                # Body is already consumed; fall through to the shared accounting below.
                chunks = ()
            else:
                chunks = response.iter_lines(delimiter=b"\n\n")
            # A transport error mid-stream (e.g. the read timeout) would otherwise escape the response
            # context manager and never be counted as a failed request
            stream_errors: list[Exception] = []
            chunks = _guard_stream(chunks, stream_errors)

            for chunk in chunks:
                if len(chunk) == 0:
                    continue  # come providers send empty lines between data chunks
                if done:
                    if chunk != b"data: [DONE]":
                        logger.warning(f"Received more chunks after [DONE]: {chunk}")
                try:
                    now = time.perf_counter()
                    if self.stream:
                        assert chunk.startswith(b"data:"), f"Unexpected chunk not starting with 'data': {chunk}"
                        chunk = chunk[len(b"data:") :]
                        if chunk.strip() == b"[DONE]":
                            done = True
                            continue
                    data = orjson.loads(chunk)
                    # Capture perf_metrics if present (usually in final usage chunk)
                    if data.get("perf_metrics"):
                        perf_metrics = data["perf_metrics"]
                    out = self.provider_formatter.parse_output_json(data)
                    if out.completion_tokens:
                        completion_tokens = out.completion_tokens
                    if out.prompt_tokens:
                        prompt_tokens = out.prompt_tokens
                    if out.cached_tokens is not None:
                        cached_tokens = out.cached_tokens
                    combined_text += out.text

                    # some providers (SGLang) send an empty chunk first skewing the TTFT;
                    # completion_tokens handles invisible-token deployments where combined_text
                    # stays empty but usage.completion_tokens is non-zero (e.g. Maverick-style)
                    if (combined_text or completion_tokens) and t_first_token is None:
                        t_first_token = now
                        # Randomly cancel after first token if cancel_rate is set
                        cancel_rate = getattr(self.environment.parsed_options, "cancel_rate", 0.0)
                        if cancel_rate > 0.0 and random.random() < cancel_rate:
                            RequestOutcomes.record("cancel_rate", class_name)
                            response.success()
                            return

                    # Track first visible (non-reasoning) token separately.
                    # has_reasoning_only is False for transition chunks (both reasoning + content),
                    # so t_first_visible_token fires correctly at the end of the thinking phase.
                    if combined_text and not out.has_reasoning_only and t_first_visible_token is None:
                        t_first_visible_token = now

                    if out.logprob_tokens:
                        total_logprob_tokens = (total_logprob_tokens or 0) + out.logprob_tokens
                except Exception as e:
                    logger.error(f"Failed to parse response: {chunk} with error {repr(e)}")
                    response.failure(e)
                    return
            if stream_errors:
                outcome = RequestOutcomes.classify(stream_errors[0])
                RequestOutcomes.record(outcome, class_name)
                response.failure(f"Error reading stream ({outcome}): {stream_errors[0]!r}")
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
                logger.warning(f"completion_tokens {completion_tokens} != logprob_tokens {total_logprob_tokens}")
            if total_logprob_tokens is not None:
                num_tokens = total_logprob_tokens
            else:
                num_tokens = completion_tokens

            num_tokens = num_tokens or 0
            num_chars = len(combined_text)
            now = time.perf_counter()
            dur_total = now - t_start

            use_first_visible = getattr(self.environment.parsed_options, "first_visible_token", False)
            # Fall back to t_first_token if no visible content ever arrived (e.g. all-thinking response)
            t_effective_first = (t_first_visible_token or t_first_token) if use_first_visible else t_first_token

            dur_generation = now - t_effective_first
            dur_first_token = t_effective_first - t_start

            if not (
                self.provider_formatter.parsed_options.embeddings or self.provider_formatter.parsed_options.rerank
            ):
                prompt_tokens = prompt_tokens or self.prompt_tokenizer_tokens

            token_parts = []
            if prompt_tokens:
                token_parts.append(f"{prompt_tokens} prompt")
            if cached_tokens is not None:
                token_parts.append(f"{cached_tokens} cached")
            token_parts.append(f"{num_tokens} completion")
            token_str = ", ".join(token_parts)

            logger.info(
                f"Response received: total {dur_total*1000:.2f} ms, first token {dur_first_token*1000:.2f} ms, {num_chars} chars, {token_str}"
            )
            if self.environment.parsed_options.show_response:
                print("---")
                print(combined_text)
                print("---")
            track_performance_metrics(
                response_time=dur_total,
                token_count=num_tokens,
                char_count=num_chars,
                prompt_tokens=prompt_tokens,
                ttft=dur_first_token,
            )
            if num_chars:
                add_custom_metric("latency_per_char", dur_generation / num_chars * 1000, num_chars)
            if self.stream:
                add_custom_metric("time_to_first_token", dur_first_token * 1000)
            add_custom_metric("total_latency", dur_total * 1000)
            if num_tokens:
                exact_length = shape_request is None or self.environment.parsed_options.shape_mix_output_mode == "exact"
                if num_tokens != max_tokens and exact_length:
                    logger.warning(f"wrong number of tokens: {num_tokens}, expected {max_tokens}")
                add_custom_metric("generation_tokens", num_tokens)  # backward-compat alias; see logging_params
                add_custom_metric("completion_tokens", num_tokens)
                add_custom_metric("num_tokens", num_tokens)
                add_custom_metric("latency_per_token", dur_generation / num_tokens * 1000, num_tokens)
                add_custom_metric(
                    "overall_latency_per_token",
                    dur_total / num_tokens * 1000,
                    num_tokens,
                )

            if self.provider_formatter.parsed_options.embeddings or self.provider_formatter.parsed_options.rerank:
                if prompt_tokens:
                    add_custom_metric("prompt_tokens", prompt_tokens)
            else:
                if prompt_tokens:
                    add_custom_metric("prompt_tokens", prompt_tokens)
                if cached_tokens is not None:
                    add_custom_metric("cached_tokens", cached_tokens)

            if shape_request is not None:
                record_shape_class_metrics(
                    shape_request.class_name,
                    ttft_ms=dur_first_token * 1000 if self.stream else None,
                    total_latency_ms=dur_total * 1000,
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached_tokens,
                    completion_tokens=num_tokens,
                    recorded_cached_tokens=shape_request.recorded_cached_tokens,
                    laid_out_cached_tokens=shape_request.reusable_tokens,
                )

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
        help=(
            "Tokenizer for counting prompt tokens. Accepts a local directory path or a HuggingFace model ID "
            f"(e.g. 'Qwen/Qwen3.5-9B' or './tokenizers/Llama-3.1-8B-Instruct'). "
            f"If omitted, defaults to '{DEFAULT_TOKENIZER}' (downloaded from HF on first use and cached locally). "
            "Use download_tokenizer.py to pre-download a tokenizer for offline or CI use."
        ),
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
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /v1/rerank API. The generated prompt text is split into documents (by paragraph), "
        "and --rerank-query is used as the query.",
    )
    parser.add_argument(
        "--rerank-query",
        type=str,
        default=None,
        help="For rerank: the search query string. Defaults to a generic query if not specified.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="For rerank: number of top results to return.",
    )
    parser.add_argument(
        "--rerank-return-documents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For rerank: whether to return document text in response.",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=None,
        help="For rerank: number of synthetic documents to generate per request. "
        "Used with --tokens-per-document to control document payload size.",
    )
    parser.add_argument(
        "--tokens-per-document",
        type=int,
        default=None,
        help="For rerank: approximate token count per synthetic document. "
        "If set without --num-documents, defaults to 10 documents.",
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
        "--embeddings-batch-size",
        type=int,
        default=1,
        help="For embeddings: number of texts per request (sent as an array). Default 1.",
    )
    parser.add_argument(
        "--embeddings-dimensions",
        type=int,
        default=None,
        help="For embeddings: requested number of output dimensions.",
    )
    parser.add_argument(
        "--embeddings-prompt-template",
        type=str,
        default=None,
        help="For embeddings: Jinja2 template for structured input. Example: 'Embed this passage: {text}'",
    )
    parser.add_argument(
        "--systemone",
        action="store_true",
        default=False,
        help="Use the Fireworks /v1/systemone API (non-streaming) with synthetic choice questions over a JSON state. "
        "--model is sent verbatim.",
    )
    parser.add_argument(
        "--systemone-num-questions",
        type=int,
        default=1,
        help="For systemone: number of questions per request.",
    )
    parser.add_argument(
        "--systemone-num-options",
        type=int,
        default=4,
        help="For systemone: number of criteria (options) per choice question.",
    )
    parser.add_argument(
        "--systemone-state-tokens",
        type=int,
        default=768,
        help="For systemone: approximate state size in tokens (uses --tokenizer if given, else ~4 chars/token).",
    )
    parser.add_argument(
        "--systemone-question-tokens",
        type=int,
        default=48,
        help="For systemone: approximate instructions size per question in tokens.",
    )
    parser.add_argument(
        "--systemone-unique-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For systemone: start each state with a random nonce so every request misses the prefix cache. "
        "With --no-systemone-unique-state each worker reuses one fixed state (warm-cache path).",
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
        help="Enables 'fixed QPS' mode: requests are sent open loop at the specified rate regardless of how long the "
        "processing takes (each send runs in its own greenlet, so slow responses don't delay later sends). Users "
        "prepare upcoming requests ahead of their send times, so -u bounds how many sends can be prepared ahead of a "
        "burst (e.g. -u 100). Sending starts once all users have spawned. The summary reports intended against "
        "delivered gap CV^2, 1 s / 10 s count dispersion and send lateness.",
    )
    parser.add_argument(
        "--qps-distribution",
        type=str,
        choices=["constant", "uniform", "exponential", "gamma"],
        default="constant",
        help="Must be used with --qps. Specifies how to space out requests: equally ('constant') or by sampling wait times from a distribution ('uniform', 'exponential' or 'gamma'). Expected QPS is going to match --qps",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for a connection and, for streaming responses, between reads (default 60). "
        "0 waits forever. Timed-out requests count as failures and are reported as Requests Timeout in the summary.",
    )
    parser.add_argument(
        "--qps-max-lateness",
        type=float,
        default=DEFAULT_MAX_LATENESS_SECONDS,
        help="With --qps: a send more than this many seconds behind schedule (the generator stalled) is skipped and "
        "the rest of the schedule moves later by the stall, so the backlog isn't sent as one burst. Skipped sends "
        "and the total shift are in the summary. Default 0.5.",
    )
    parser.add_argument(
        "--qps-modulation-window",
        type=float,
        default=None,
        help="With --qps: redraw the rate every this many seconds from a gamma with mean --qps, so counts are "
        "overdispersed like production traffic (a renewal process can't be). Arrivals keep --qps-distribution "
        "within each window. Needs --qps-dispersion (or arrivals.dispersion_10s in --shape-mix). E.g. 10.",
    )
    parser.add_argument(
        "--qps-dispersion",
        type=float,
        default=None,
        help="With --qps-modulation-window: target index of dispersion (variance / mean) of request counts in "
        "10-second windows. Defaults to arrivals.dispersion_10s (or arrivals.router_level.dispersion_10s) from "
        "--shape-mix. Values at or below what the arrivals give on their own (1 for Poisson, cv2 for gamma) "
        "leave the rate unmodulated.",
    )
    parser.add_argument(
        "--shape-mix-corpus",
        type=str,
        default=None,
        help="For --shape-mix: text that prefixes and tails are cut from. 'prose' (default: 1.7 MB of public-domain "
        "novels, so long prompts don't repeat), 'limericks' (4.9k tokens, repeats past that), 'code' (1.76 MB of CPython "
        "source), or a path to a text file, optionally gzipped (paragraphs separated by blank lines are shuffled by "
        "seed). The corpus is part of the --prefix-pool-cache key.",
    )
    parser.add_argument(
        "--shape-mix-instruction",
        type=str,
        default=None,
        help="For --shape-mix: instruction appended to every prompt after a blank line (inline text or @file), e.g. "
        "'Summarize the text above.'. It counts toward prompt_tokens; the unique tail is shortened to fit.",
    )
    parser.add_argument(
        "--shape-mix-sample-lengths",
        action="store_true",
        default=False,
        help="For --shape-mix: draw each request's uncached prompt tokens and output tokens from the class's "
        "uncached_tokens_quantiles and generated_tokens_quantiles ({'p10': n, 'p50': n, ...} or [[q, n], ...]), "
        "by piecewise-linear inverse CDF, instead of using the class means. The cached share stays fixed. Uncached "
        "lengths are sampled in 19 weighted strata reaching the top quantile (one precomputed tail each), output lengths "
        "continuously; both are scaled to keep the class mean. "
        "Classes without quantiles keep their means. Off by default.",
    )
    parser.add_argument(
        "--prefix-popularity-zipf",
        type=float,
        default=None,
        help="For --shape-mix: default prefix popularity for classes without prefix_popularity: prefix j of a pool "
        "is drawn with weight 1 / (j + 1)^S (0 or unset is uniform). A class's prefix_popularity ({'zipf': s} or "
        "a list of prefix_pool_size weights) takes precedence. --prefix-fill still sends each prefix once.",
    )
    parser.add_argument(
        "--shape-mix-tail-variants",
        type=int,
        default=SHAPE_MIX_TAIL_VARIANTS,
        help=f"For --shape-mix: precomputed fillers per class that follow each request's unique nonce (default "
        f"{SHAPE_MIX_TAIL_VARIANTS}). 0 fits a fresh filler for every request instead (tokenizer work per request), "
        "e.g. to check that repeated filler text doesn't change speculative-decoding acceptance.",
    )
    parser.add_argument(
        "--shape-mix-output-mode",
        choices=["exact", "max"],
        default="exact",
        help="For --shape-mix: 'exact' (default) forces each class's generated_tokens (max_tokens = min_tokens, "
        "ignore_eos). 'max' only sets max_tokens = generated_tokens, with no min_tokens or ignore_eos, so the model "
        "stops where it would (for output-length models where generated_tokens is a cap); the summary's per-class "
        "Completion Tokens shows what was generated. --prefix-fill always sends max_tokens 1.",
    )
    parser.add_argument(
        "--replay-schedule",
        type=str,
        default=None,
        help="For --shape-mix: send at recorded times instead of --qps. JSONL, one request per line: "
        '{"timestamp": <ms>, "class": "<class name>"} (or "time_s": <seconds>), optionally "prefix_index" (a prefix '
        'of the class pool; drawn otherwise) and "output_length" (overrides the class). Times are replayed at '
        "recorded speed relative to the first entry, open loop like --qps (same lateness handling and delivered-"
        'arrival stats). Prompts come from the class prefix pools. Entries may also set "input_length" (prompt '
        'tokens as the server counts them; see --replay-template-subtraction) and "session_id" (turns of one '
        "multi-turn chat, re-sending the conversation so far; needs --chat). The run ends when the schedule is "
        "done and responses are in (or at -t).",
    )
    parser.add_argument(
        "--replay-pods",
        type=int,
        default=1,
        help="With --replay-schedule: N test replicas, one recorded pod's stream each (rows' \"pod\" field): replica "
        "--replay-pod-index I replays the I-th of the first N recorded pods (or of the --replay-pod list) at its "
        "exact times; use --replay-start-at for a shared start. Schedules without pods are split by session.",
    )
    parser.add_argument(
        "--replay-pod",
        default=None,
        metavar="ID[,ID...]|all",
        help='With --replay-schedule: replay only this recorded pod\'s rows (the "pod" field) at their exact times; '
        "one test replica stands for one recorded pod. Several IDs or 'all' merge pods into one stream, which is only "
        "for tiers above 1x and is labelled in the summary. Required when the schedule records more than one pod "
        "(unless --replay-pods N).",
    )
    parser.add_argument(
        "--replay-start-at",
        type=float,
        default=float(os.environ["REPLAY_START_AT"]) if os.environ.get("REPLAY_START_AT") else None,
        metavar="UNIX_SECONDS",
        help="With --replay-schedule: wall-clock start (Unix time) shared by all generators, so replicas started "
        "at different moments replay in step (default $REPLAY_START_AT, else when this one is ready). Rows already "
        "due when it starts count as late.",
    )
    parser.add_argument(
        "--replay-session-salt",
        default=None,
        help="With --replay-schedule: salt in session affinity keys (default: a new random value per run, so "
        "runs don't share session routing).",
    )
    parser.add_argument(
        "--replay-pod-index",
        type=int,
        default=int(os.environ.get("REPLAY_POD_INDEX", 0)),
        help="With --replay-pods: this pod's index, 0 to N-1 (default $REPLAY_POD_INDEX or 0).",
    )
    parser.add_argument(
        "--replay-schedule-meta",
        type=str,
        default=None,
        help="With --replay-schedule: the schedule's meta JSON (default <schedule>.meta.json when it exists). Its "
        "pool_prefix gives each class's longest recorded shared length on the replayed replica, which sizes the "
        "class's pool prefix; without it pool prefixes stay at the class mean cached length (with a warning).",
    )
    parser.add_argument(
        "--replay-pool-prefix",
        choices=["max", "p99"],
        default="max",
        help="With --replay-schedule: size each class's pool prefix to the longest ('max', default) or the p99 "
        "recorded shared length, for when the longer prefixes push warm-up cache occupancy above production's.",
    )
    parser.add_argument(
        "--replay-retime",
        default=None,
        metavar="poisson|gamma:CV2",
        help="With --replay-schedule: keep every row (lengths, class, session, sampling fields, order) and redraw "
        "only the send times, as a seeded Poisson process (or gamma gaps at the given CV²) at the schedule's own "
        "mean rate over the same span. Seeded by --request-seed / the mix's request_seed. Compare a replay against "
        "the same replay retimed to see whether arrival shaping matters.",
    )
    parser.add_argument(
        "--replay-late-rows",
        choices=["send", "skip"],
        default="send",
        help="With --replay-schedule: a row more than --qps-max-lateness behind its time is counted as late and "
        "still sent ('send', default; rows are recorded traffic), or counted and skipped ('skip'). The summary "
        "reports Replay Late Rows and their share either way.",
    )
    parser.add_argument(
        "--replay-template-subtraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --replay-schedule: input_length counts the prompt as the server does, chat template included, "
        "so the template's tokens are subtracted from the content (default). --no-replay-template-subtraction "
        "treats input_length as content tokens only.",
    )
    parser.add_argument(
        "--prefix-fill",
        action="store_true",
        default=False,
        help="For --shape-mix: prefill-only cache fill. Sends every prefix of every class once (classes "
        "interleaved), each as a request shaped like live traffic (prefix + unique tail) with max_tokens 1, at "
        "--prefix-fill-concurrency, then exits. The summary reports requests, prompt tokens and shared-prefix tokens sent, "
        "server-reported prompt and cached tokens, and the time taken. Run it in one process, without -t.",
    )
    parser.add_argument(
        "--prefix-fill-concurrency",
        type=int,
        default=16,
        help="With --prefix-fill: requests in flight at once (default 16).",
    )
    parser.add_argument(
        "--prefix-pool-cache",
        type=str,
        default=None,
        help="For --shape-mix: directory to keep built prefix windows in, so llm_bench processes after the first "
        "load the pool instead of rebuilding it. Files are keyed by seed, class, prefix length, tokenizer and "
        "corpus (not request seed or pool size: a larger pool builds only the missing prefixes). Safe to share "
        "between concurrent processes.",
    )
    parser.add_argument(
        "--prefix-reuse-overhang",
        type=int,
        default=None,
        help="For --shape-mix: tokens before a divergence point the server can't reuse from its prefix cache. "
        "Shared prefixes are made this much longer (the unique tail shorter), so the server reuses exactly "
        "cached_prompt_tokens. Defaults to the model's sliding window for DeepSeek V4 (128, which recomputes it "
        "at the divergence), else 0.",
    )
    parser.add_argument(
        "--prefix-restore-boundary",
        type=int,
        default=None,
        help="For --shape-mix and --replay-schedule: the block (tokens) the server restores cached KV in; a shared "
        "prefix of N tokens reuses floor(N / block) * block. Cached lengths are aligned down to a block and "
        f"prefixes laid out {RESTORE_BOUNDARY_MARGIN_TOKENS} tokens past it. Defaults to the config's index_topk for "
        "GLM 5.x (glm_moe_dsa: 2048), else 0 (token-exact reuse).",
    )
    parser.add_argument(
        "--qps-cv2",
        type=float,
        default=None,
        help="For --qps-distribution gamma: squared coefficient of variation of inter-arrival times. "
        "1 is Poisson (same as exponential), > 1 clumps arrivals into bursts, < 1 spaces them more evenly. "
        "Defaults to arrivals.cv2 from --shape-mix if present, else 1.",
    )
    parser.add_argument(
        "--request-seed",
        type=int,
        default=None,
        help="Seed for per-request randomness: --shape-mix class, prefix and tail choice, and --qps arrival times. "
        "Defaults to the shape mix's request_seed (itself defaulting to its seed); without --shape-mix, arrivals "
        "are unseeded unless this is set. Give each run against the same deployment its own value so unique "
        "tails aren't replayed into the cache; reuse it to replay identical traffic (e.g. incumbent vs candidate).",
    )
    parser.add_argument(
        "--shape-mix",
        type=str,
        default=None,
        help="Class-mix workload as inline JSON or @file: "
        '{"seed": 0, "classes": [{"name": "chat", "weight": 0.7, "prompt_tokens": 8000, '
        '"cached_prompt_tokens": 6000, "generated_tokens": 200, "prefix_pool_size": 16}, ...]}. '
        "Each request picks a class by weight, reuses one of the class's prefix_pool_size seeded shared prefixes "
        "(cached_prompt_tokens long, sent with x-session-affinity) plus a unique tail up to prompt_tokens, and "
        "forces generated_tokens output (max_tokens = min_tokens, ignore_eos). Replaces -p/-o/--dataset/-pcml.",
    )
    parser.add_argument(
        "--cancel-rate",
        type=float,
        default=0.0,
        help="Probability (0.0-1.0) of cancelling a request after receiving the first token. E.g. 0.25 cancels 25%% of requests.",
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
        "--acceptance-probs-override",
        type=str,
        default=None,
        help="JSON array of acceptance probabilities for speculative decoding override (e.g. '[0.9, 0.7, 0.5]'). "
        "Each element is the probability of accepting the draft token at that position (0.0-1.0). "
        "Injected as acceptance_probs_override in every request body. Fireworks-specific.",
    )
    parser.add_argument(
        "--forced-generation-file",
        type=str,
        default=None,
        help="Path to a file containing natural text completions for forced generation. "
        "Supports plain text (one completion per line) or JSONL (one JSON object per line with a 'text' field). "
        "Each request cycles through the pool so different requests get different forced texts. Fireworks-specific.",
    )
    parser.add_argument(
        "--forced-generation-from-dataset",
        action="store_true",
        default=False,
        help="Use the current --dataset (limericks or code) as forced generation text. "
        "Parses the dataset file into paragraph-separated entries and cycles through them. "
        "Each request gets a different entry for diverse expert routing. "
        "Use with --acceptance-probs-override for representative MoE benchmarking. "
        "Mutually exclusive with --forced-generation-file. Fireworks-specific.",
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
        choices=["none", "low", "medium", "high"],
        help="Set the reasoning_effort parameter for the API request. "
        "Use 'none' to disable thinking/reasoning where supported. "
        "If not specified and using a JSONL dataset, will use the value from the dataset if present.",
    )
    parser.add_argument(
        "--clear-assistant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If the last message in the messages array is from the assistant role, remove it. "
        "This allows the model to generate new content instead of continuing from existing assistant content.",
    )
    parser.add_argument(
        "--force-min-tokens",
        action="store_true",
        default=False,
        help="Set min_tokens = max_tokens in Fireworks requests to force exact output length. "
        "This causes ignore_eos=True on the serving side, which can cause some models to "
        "generate far more tokens than requested.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=False,
        help="Send ignore_eos=true in Fireworks requests so every request generates max_tokens. "
        "Replaces the deprecated min_tokens alias used by --force-min-tokens.",
    )
    parser.add_argument(
        "--first-visible-token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When set, TTFT and latency_per_token measure from the first visible (non-reasoning) token. "
        "By default, TTFT fires on the first thinking token for reasoning models, which is consistent "
        "with how Customers measures TTFT. Use this flag to compare apples-to-apples TTFT across "
        "reasoning and non-reasoning providers. Requires --chat and --stream.",
    )
    parser.add_argument(
        "--max-fail-ratio",
        type=float,
        default=0.01,
        help="Maximum fraction of failed requests tolerated before the run is considered failed. "
        "Defaults to 0.01 (1%%). Set to 0 to fail on any failed request.",
    )


SYSTEMONE_SERVER_TIMING_HEADERS = {
    "fireworks-prefill-duration": "server_side_prefill_latency",
    "fireworks-prefill-queue-duration": "server_side_prefill_queue_latency",
}


def record_systemone_server_timings(headers) -> None:
    # The gateway forwards the serving pod's prefill timings (seconds); record them in ms like the other latency rows.
    for header, metric_name in SYSTEMONE_SERVER_TIMING_HEADERS.items():
        try:
            value = headers.get(header)
            if value not in (None, ""):
                add_custom_metric(metric_name, float(value) * 1000)
        except (TypeError, ValueError):
            logger.debug(f"Skipping unparsable {header}: {headers.get(header)!r}")


SERVER_SIDE_TEXT_METRICS = ["server_side_time_to_first_token", "server_side_total_latency"]


def text_summary_entries(stats, stream: bool) -> tuple[dict[str, Any], list[str]]:
    """Summary entries and percentile metrics for text completion runs.

    Server-side metrics are only reported when the provider recorded them
    (Fireworks perf_metrics or response headers), so other providers keep the
    same CSV columns. Reporting them next to the client-side values separates
    network/gateway time from time spent on the deployment.
    """

    def _avg(metric_name):
        entry = stats.entries.get((metric_name, "METRIC"))
        return entry.avg_response_time if entry else ""

    entries: dict[str, Any] = {}
    for metric_name in [
        "time_to_first_token",
        "latency_per_token",
        "overall_latency_per_token",
        "total_latency",
        "generation_tokens",
        "completion_tokens",
        "num_tokens",
        "prompt_tokens",
    ]:
        entries[metric_name] = _avg(metric_name)
    if ("cached_tokens", "METRIC") in stats.entries:
        entries["cached_tokens"] = stats.entries[("cached_tokens", "METRIC")].avg_response_time
    if not stream:
        # if there's no streaming these metrics are meaningless
        entries["time_to_first_token"] = ""
        entries["latency_per_token"] = ""
    percentile_metrics = ["time_to_first_token", "total_latency"]
    # Speculative decoding acceptance per draft position, when the server reports it (Fireworks perf_metrics)
    for (name, method), entry in sorted(stats.entries.items()):
        if method == "METRIC" and name.startswith("speculation_hit_rate_position_"):
            entries[name] = entry.avg_response_time
    for metric_name in SERVER_SIDE_TEXT_METRICS:
        if (metric_name, "METRIC") in stats.entries:
            entries[metric_name] = _avg(metric_name)
            percentile_metrics.append(metric_name)
    return entries, percentile_metrics


SHAPE_CLASS_PERCENTILES = [50, 90, 99]


def record_shape_class_metrics(
    class_name: str,
    ttft_ms: Optional[float],
    total_latency_ms: float,
    prompt_tokens: Optional[int],
    cached_tokens: Optional[int],
    completion_tokens: Optional[int],
    recorded_cached_tokens: Optional[int] = None,
    laid_out_cached_tokens: Optional[int] = None,
) -> None:
    prefix = f"class_{class_name}_"
    add_custom_metric(prefix + "total_latency", total_latency_ms)
    if recorded_cached_tokens is not None:
        # Per row: recorded (replay schedule), laid out by the generator, and server-reported (cached_tokens)
        add_custom_metric(prefix + "recorded_cached_tokens", recorded_cached_tokens)
        add_custom_metric(prefix + "laid_out_cached_tokens", laid_out_cached_tokens or 0)
    if ttft_ms is not None:
        add_custom_metric(prefix + "time_to_first_token", ttft_ms)
    if prompt_tokens:
        add_custom_metric(prefix + "prompt_tokens", prompt_tokens)
    if cached_tokens is not None:
        add_custom_metric(prefix + "cached_tokens", cached_tokens)
    if completion_tokens:
        add_custom_metric(prefix + "completion_tokens", completion_tokens)


RETIME_CLOSE_FOLLOWUP_S = 0.5


def retime_summary_entries(pacer) -> dict[str, Any]:
    """For a retimed replay: the requested gap CV² and rate against the stream's (drawn slot times) and what was
    delivered (send times on the wire), plus follow-ups that now land within 0.5 s of their session's previous
    row (recorded gaps between turns aren't kept, so such a row can arrive while its parent is in prefill)."""
    spec = parse_replay_retime(getattr(pacer, "retime", None))
    if spec is None:
        return {}
    offsets = [e.offset for e in pacer.entries]
    span = offsets[-1] - offsets[0] if len(offsets) > 1 else 0.0
    actual = sorted(SendTimeline.actual)
    delivered_span = actual[-1] - actual[0] if len(actual) > 1 else 0.0
    previous: dict[str, ScheduleEntry] = {}
    close = recorded_close = followups = 0
    for e in pacer.entries:
        if e.session is not None:
            parent = previous.get(e.session)
            if parent is not None:
                followups += 1
                close += e.offset - parent.offset < RETIME_CLOSE_FOLLOWUP_S
                if e.recorded_offset is not None and parent.recorded_offset is not None:
                    recorded_close += e.recorded_offset - parent.recorded_offset < RETIME_CLOSE_FOLLOWUP_S
            previous[e.session] = e
    slot_cv2 = gap_cv2(offsets)
    return {
        "replay_retime_requested_gap_cv2": spec[1],
        "replay_retime_slot_gap_cv2": "" if slot_cv2 is None else slot_cv2,
        "replay_retime_requested_qps": (len(offsets) - 1) / span if span else "",
        "replay_retime_delivered_qps": (len(actual) - 1) / delivered_span if delivered_span else "",
        "replay_retime_followups": followups,
        "replay_retime_followups_within_0_5s": close,
        "replay_retime_followups_within_0_5s_recorded": recorded_close,
    }


def first_row_shortfall_entries(dataset) -> dict[str, Any]:
    """Session-first rows whose recorded cached length the layout couldn't meet, in total and per class: the
    shortfall left once pool prefixes are sized (rows with cached_length > 0 only)."""
    first = getattr(dataset, "replay_first_rows", None)
    if not first:
        return {}
    entries: dict[str, Any] = {
        "replay_first_rows": sum(v[0] for v in first.values()),
        "replay_first_row_shortfall_rows": sum(v[1] for v in first.values()),
        "replay_first_row_shortfall_tokens": sum(v[2] for v in first.values()),
    }
    sizes = getattr(dataset, "replay_pool_sizes", None) or {}
    for name, (rows, short_rows, short_tokens) in sorted(first.items()):
        entries[f"class_{name}_first_rows"] = rows
        entries[f"class_{name}_first_row_shortfall_rows"] = short_rows
        entries[f"class_{name}_first_row_shortfall_tokens"] = short_tokens
        if name in sizes:
            entries[f"class_{name}_pool_prefix_tokens"] = sizes[name]
    return entries


def replay_summary_entries(pacer, dataset) -> dict[str, Any]:
    """Rows of this pod, how many went out late (and their share; none are dropped silently), cancelled rows,
    and rows whose recorded cached length the layout couldn't meet."""
    rows = len(pacer.entries)
    shortfall = {
        "rows": 0,
        "tokens": 0,
        "input_rows": 0,
        "input_tokens": 0,
        "length_rows": 0,
        **(getattr(dataset, "replay_shortfall", None) or {}),
    }
    return {
        "replay_rows": rows,
        "replay_pods": ",".join(pacer.pods[:8]) + (f",... ({len(pacer.pods)})" if len(pacer.pods) > 8 else ""),
        "replay_pods_merged": len(pacer.pods) if len(pacer.pods) > 1 else 0,
        "replay_rows_defaulted": sum(1 for e in pacer.entries if e.defaulted),
        "replay_layout_s": getattr(dataset, "replay_layout_seconds", 0.0),
        "replay_layout_ms_per_row": 1000 * getattr(dataset, "replay_layout_seconds", 0.0) / rows if rows else 0.0,
        "replay_session_salt": getattr(dataset, "replay_salt", ""),
        "replay_retime": getattr(pacer, "retime", None) or "",
        **retime_summary_entries(pacer),
        "replay_rows_sent": pacer.sent_rows,
        "replay_late_rows": pacer.late_rows,
        "replay_late_share": pacer.late_rows / rows if rows else 0.0,
        "replay_rows_skipped": pacer.skipped_rows,
        "replay_max_lateness_ms": pacer.max_lateness_seen * 1000,
        "replay_cancelled_rows": pacer.cancelled_rows,
        "replay_cached_shortfall_rows": shortfall["rows"],
        "replay_cached_shortfall_tokens": shortfall["tokens"],
        "replay_input_overflow_rows": shortfall["input_rows"],
        "replay_input_overflow_tokens": shortfall["input_tokens"],
        "replay_length_mismatch_rows": shortfall["length_rows"],
        **first_row_shortfall_entries(dataset),
        "replay_lead_in_rows": LeadIns.rows,
        "replay_lead_in_remainder_rows": LeadIns.remainder_rows,
        "replay_lead_in_sent": LeadIns.sent,
        "replay_lead_in_covered": LeadIns.skipped,
        "replay_lead_in_failed": LeadIns.failed,
        "replay_lead_in_s": LeadIns.seconds,
    }


def shape_mix_summary_entries(
    stats, class_names: list[str], stream: bool, reusable_tokens: Optional[list[int]] = None
) -> dict[str, Any]:
    """Per-class summary columns for --shape-mix; requests count the class's successful responses.

    `reusable_tokens` is what the generator lays out for the server to reuse per class (cached_prompt_tokens,
    unless the prompt is too short); compare it with the server-reported cached tokens.
    """
    entries: dict[str, Any] = {}
    for ci, name in enumerate(class_names):
        prefix = f"class_{name}_"
        latency = stats.entries.get((prefix + "total_latency", "METRIC"))
        entries[prefix + "requests"] = latency.num_requests if latency else 0
        if reusable_tokens is not None:
            entries[prefix + "reusable_tokens"] = reusable_tokens[ci]
        for metric in ["time_to_first_token", "total_latency", "prompt_tokens", "cached_tokens", "completion_tokens"]:
            entry = stats.entries.get((prefix + metric, "METRIC"))
            entries[prefix + metric] = entry.avg_response_time if entry else ""
        recorded = stats.entries.get((prefix + "recorded_cached_tokens", "METRIC"))
        if recorded is not None:
            laid_out = stats.entries.get((prefix + "laid_out_cached_tokens", "METRIC"))
            entries[prefix + "recorded_cached_tokens"] = recorded.avg_response_time
            entries[prefix + "laid_out_cached_tokens"] = laid_out.avg_response_time if laid_out else ""
            delivered = entries[prefix + "cached_tokens"]
            entries[prefix + "delivered_over_recorded_cached"] = (
                delivered / recorded.avg_response_time if delivered != "" and recorded.avg_response_time else ""
            )
        for metric in ["time_to_first_token", "total_latency"]:
            entry = stats.entries.get((prefix + metric, "METRIC"))
            for percentile in SHAPE_CLASS_PERCENTILES:
                entries[f"{prefix}P{percentile}_{metric}"] = (
                    entry.get_response_time_percentile(percentile / 100) if entry else ""
                )
        if not stream:
            entries[prefix + "time_to_first_token"] = ""
    return entries


def systemone_summary_entries(stats, num_questions: int) -> dict[str, Any]:
    """Systemone summary columns; latency percentiles are appended by the shared summary code."""
    # Locust names the entry after the full request path, which includes any host base path (e.g. /inference).
    request_entry = next(
        (e for (name, method), e in stats.entries.items() if method == "POST" and name.endswith("/v1/systemone")),
        None,
    )
    latency_entry = stats.entries.get(("total_latency", "METRIC"))
    qps = latency_entry.total_rps if latency_entry else 0

    def _avg(metric_name):
        entry = stats.entries.get((metric_name, "METRIC"))
        return entry.avg_response_time if entry else ""

    entries = {
        "requests": request_entry.num_requests if request_entry else 0,
        "failures": request_entry.num_failures if request_entry else 0,
        "questions_per_s": qps * num_questions,
    }
    for metric_name in [
        "total_latency",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "packed_fraction",
        "response_bytes",
        "server_side_total_latency",
    ]:
        entries[metric_name] = _avg(metric_name)
    return entries


@events.quitting.add_listener
def _(environment, **kw):
    total_latency = environment.stats.entries.get(("total_latency", "METRIC"))
    num_requests, num_failures = request_counts(environment.stats)
    fail_ratio = (num_failures / num_requests) if num_requests > 0 else 1.0
    outcomes = request_outcome_entries(environment.stats)
    if fail_ratio > environment.parsed_options.max_fail_ratio or total_latency is None:
        logger.error(
            f"Test failed: {num_failures}/{num_requests} requests failed ({fail_ratio:.2%}); "
            f"timeouts {outcomes['requests_timeout']}, HTTP {CLIENT_CANCELLED_STATUS} {outcomes['requests_http_299']}, "
            f"other errors {outcomes['requests_other_error']} (--request-timeout "
            f"{request_timeout(environment.parsed_options)} s)"
        )
        environment.process_exit_code = 1
        return
    # Explicitly set exit code 0 so that locust's default policy does not force
    # a non-zero exit when runner.exceptions is non-empty (e.g. unhandled
    # exceptions from non-request paths like dataset loading).
    environment.process_exit_code = 0

    entries = copy.copy(InitTracker.logging_params)
    replaying = bool(getattr(environment.parsed_options, "replay_schedule", None))
    if replaying and ReplayPacer._instance is not None:
        entries["concurrency"] = ReplayPacer._instance.describe()
    elif environment.parsed_options.qps is not None:
        pacer = FixedQPSPacer._instance
        if pacer is not None:
            entries["concurrency"] = pacer.describe()
        else:
            entries["concurrency"] = (
                f"QPS {environment.parsed_options.qps} {environment.parsed_options.qps_distribution}"
            )
    else:
        entries["concurrency"] = InitTracker.users

    def _avg(metric_name):
        entry = environment.stats.entries.get((metric_name, "METRIC"))
        return entry.avg_response_time if entry else ""

    if getattr(environment.parsed_options, "embeddings", False):
        for metric_name in [
            "total_latency",
            "latency_per_embedding",
            "prompt_tokens",
            "response_bytes",
            "server_side_total_latency",
        ]:
            entries[metric_name] = _avg(metric_name)
        percentile_metrics = ["total_latency", "latency_per_embedding", "server_side_total_latency"]
    elif getattr(environment.parsed_options, "systemone", False):
        entries.update(systemone_summary_entries(environment.stats, environment.parsed_options.systemone_num_questions))
        percentile_metrics = [
            "total_latency",
            "server_side_total_latency",
            "server_side_prefill_latency",
            "server_side_prefill_queue_latency",
        ]
    elif getattr(environment.parsed_options, "rerank", False):
        for metric_name in ["total_latency", "prompt_tokens", "response_bytes", "server_side_total_latency"]:
            entries[metric_name] = _avg(metric_name)
        # Include rerank sweep params when set
        num_docs = getattr(environment.parsed_options, "num_documents", None)
        tokens_per_doc = getattr(environment.parsed_options, "tokens_per_document", None)
        if num_docs is not None:
            entries["num_documents"] = num_docs
        if tokens_per_doc is not None:
            entries["tokens_per_document"] = tokens_per_doc
        percentile_metrics = ["total_latency", "server_side_total_latency"]
    else:
        text_entries, percentile_metrics = text_summary_entries(environment.stats, environment.parsed_options.stream)
        entries.update(text_entries)

    entries["num_requests"] = total_latency.num_requests
    entries["qps"] = total_latency.total_rps
    percentile_to_report = [50, 90, 95, 99, 99.9]
    for percentile_metric in percentile_metrics:
        metric_entry = environment.stats.entries.get((percentile_metric, "METRIC"))
        if metric_entry is None:
            continue
        for percentile in percentile_to_report:
            name = f"P{percentile}_{percentile_metric}"
            entries[name] = metric_entry.get_response_time_percentile(percentile / 100)

    if environment.parsed_options.qps is not None or replaying:
        entries.update(arrival_summary_entries(SendTimeline.intended, SendTimeline.actual))
        entries["late_sends_skipped"] = SendTimeline.skipped
        pacer = ReplayPacer._instance if replaying else FixedQPSPacer._instance
        if replaying and pacer is not None:
            entries.update(replay_summary_entries(pacer, DatasetHolder._shape_mix_instance))
        entries["schedule_shift_s"] = pacer.shifted_seconds if pacer is not None else 0.0
    entries.update(ClientCpu.summary_entries())

    class_names = None
    if getattr(environment.parsed_options, "shape_mix", None):
        class_names = [c.name for c in parse_shape_mix(environment.parsed_options.shape_mix).classes]
    if getattr(environment.parsed_options, "prefix_fill", False):
        entries["concurrency"] = f"prefix fill at {environment.parsed_options.prefix_fill_concurrency}"
        entries.update(prefix_fill_summary_entries(environment.stats, class_names))
    entries["request_timeout_s"] = request_timeout(environment.parsed_options) or ""
    entries.update(request_outcome_entries(environment.stats, class_names))

    if getattr(environment.parsed_options, "shape_mix", None):
        mix = parse_shape_mix(environment.parsed_options.shape_mix)
        dataset = DatasetHolder._shape_mix_instance
        if dataset is not None:
            entries["prefix_pool_prefixes"] = sum(dataset.pool_sizes)
            entries["prefix_pool_tokens"] = dataset.pool_tokens
            entries["prefix_pool_loaded_from_cache"] = dataset.cache_loaded_prefixes
            entries["prefix_pool_build_s"] = dataset.build_seconds
            if dataset.context_tokens:
                entries["shape_mix_context_tokens"] = dataset.context_tokens
                entries["shape_mix_context_cut_requests"] = dataset.context_cut_requests
        entries.update(
            shape_mix_summary_entries(
                environment.stats,
                [c.name for c in mix.classes],
                environment.parsed_options.stream,
                reusable_tokens=dataset.class_reusable_tokens() if dataset is not None else None,
            )
        )

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
