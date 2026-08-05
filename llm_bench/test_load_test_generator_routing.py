from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import load_test
from generator_routing import RoutingConfig, routing_headers_for_worker


def setup_function() -> None:
    load_test.GeneratorLoadCoordinator.reset()


def test_reused_prompt_dataset_materializes_one_item() -> None:
    dataset = load_test.ReusedPromptDataset(iter([("first", 10), ("second", 20)]))

    first_user = iter(dataset)
    second_user = iter(dataset)

    assert next(first_user) == ("first", 10)
    assert next(first_user) == ("first", 10)
    assert next(second_user) == ("first", 10)


def test_locust_users_are_balanced_across_generator_workers() -> None:
    routing = RoutingConfig(num_servers=8, num_gens=1)

    assignments = [load_test.GeneratorLoadCoordinator.allocate_user(routing) for _ in range(16)]

    assert [worker for _, worker in assignments] == list(range(8)) * 2
    assert [
        routing_headers_for_worker(routing, worker)["x-fireworks-generator-worker-service-index"]
        for _, worker in assignments[:8]
    ] == [str(i) for i in range(8)]


def test_cache_warmup_runs_once_per_generator_worker() -> None:
    warmup = Mock(return_value="ready")

    assert load_test.GeneratorLoadCoordinator.warm_worker_once(3, warmup) == "ready"
    assert load_test.GeneratorLoadCoordinator.warm_worker_once(3, warmup) == "ready"
    assert load_test.GeneratorLoadCoordinator.warm_worker_once(4, warmup) == "ready"

    assert warmup.call_count == 2


def test_generator_users_on_same_worker_share_session_id() -> None:
    user = object.__new__(load_test.LLMUser)
    user.generator_session_user = "llm-bench-generator-worker-2"
    payload = {}

    user._apply_generator_session(payload)

    assert payload["user"] == "llm-bench-generator-worker-2"


def test_openai_parser_accepts_multiple_n_choices() -> None:
    provider = load_test.OpenAIProvider(
        "model",
        SimpleNamespace(rerank=False, embeddings=False, chat=False, stream=False),
    )

    result = provider.parse_output_json(
        {
            "choices": [
                {"text": "first", "logprobs": None},
                {"text": "second", "logprobs": None},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    )

    assert result.text == "firstsecond"
    assert result.completion_tokens == 20
