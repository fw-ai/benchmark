#!/usr/bin/env python3
"""Unit tests for strict token-id prompt construction in load_test.py."""

import random
import unittest
from unittest import mock

import transformers

from load_test import TranslationDataset, _build_pair_ids, _load_chunks


class StrictPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        cls.tokenizer.add_bos_token = False
        cls.tokenizer.add_eos_token = False
        cls.path = __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.abspath(__file__)), "limericks.txt"
        )
        cls.chunks = _load_chunks(cls.path)

    def test_exact_prompt_length_no_chat(self):
        ds = TranslationDataset(
            path=self.path,
            prompt="",
            tokenizer=self.tokenizer,
            tokenizer_path="gpt2",
            chat=False,
            num_tokens=512,
            common_tokens=0,
            seed=1,
        )
        ids, reported = next(iter(ds))
        self.assertIsInstance(ids, list)
        self.assertEqual(reported, 512)
        self.assertEqual(len(ids), 512)

    def test_exact_prompt_length_with_chat(self):
        fake_prefix, fake_suffix = [1, 2, 3], [4, 5]
        with mock.patch("load_test.resolve_model_type", return_value="gpt2"), mock.patch(
            "load_test._split_chat_template", return_value=(fake_prefix, fake_suffix)
        ):
            ds = TranslationDataset(
                path=self.path,
                prompt="",
                tokenizer=self.tokenizer,
                tokenizer_path="gpt2",
                chat=True,
                num_tokens=512,
                common_tokens=0,
                seed=2,
            )
            ids, reported = next(iter(ds))
        self.assertEqual(len(ids), 512)
        self.assertEqual(reported, 512)
        self.assertEqual(ids[:3], fake_prefix)
        self.assertEqual(ids[-2:], fake_suffix)

    def test_shared_prefix_matches_cached_tokens(self):
        prompt_tokens = 1000
        cached_tokens = 600
        ds = TranslationDataset(
            path=self.path,
            prompt="",
            tokenizer=self.tokenizer,
            tokenizer_path="gpt2",
            chat=False,
            num_tokens=prompt_tokens,
            common_tokens=cached_tokens,
            seed=3,
        )
        first, _ = next(iter(ds))
        second, _ = next(iter(ds))
        self.assertEqual(first[:cached_tokens], second[:cached_tokens])
        self.assertNotEqual(first, second)

    def test_shared_prefix_with_chat_template(self):
        prompt_tokens = 800
        cached_tokens = 400
        prefix, suffix = [11, 12, 13], [21, 22]
        rng = random.Random(4)
        base_ids = [i % 1000 for i in range(prompt_tokens)]
        a = _build_pair_ids(prefix, suffix, base_ids, self.tokenizer, self.chunks, prompt_tokens, cached_tokens, rng)
        b = _build_pair_ids(prefix, suffix, base_ids, self.tokenizer, self.chunks, prompt_tokens, cached_tokens, rng)
        self.assertEqual(len(a), prompt_tokens)
        self.assertEqual(len(b), prompt_tokens)
        self.assertEqual(a[:cached_tokens], b[:cached_tokens])

    def test_openai_provider_token_ids_use_completions(self):
        from load_test import OpenAIProvider

        opts = mock.Mock(chat=True, rerank=False, embeddings=False)
        provider = OpenAIProvider("test-model", opts)
        self.assertEqual(provider.get_url([1, 2, 3]), "/v1/completions")
        self.assertEqual(provider.get_url("hello"), "/v1/chat/completions")

    def test_per75_shape_shared_prefix(self):
        """Scaled-down check: first cached_tokens ids are identical across requests."""
        prompt_tokens = 5000
        cached_tokens = 3750  # 75% of 5000, mirrors deployment experiment ratio
        ds = TranslationDataset(
            path=self.path,
            prompt="",
            tokenizer=self.tokenizer,
            tokenizer_path="gpt2",
            chat=False,
            num_tokens=prompt_tokens,
            common_tokens=cached_tokens,
            seed=99,
        )
        prompts = [next(iter(ds))[0] for _ in range(5)]
        for p in prompts:
            self.assertEqual(len(p), prompt_tokens)
        prefix = prompts[0][:cached_tokens]
        for p in prompts[1:]:
            self.assertEqual(p[:cached_tokens], prefix)
            self.assertNotEqual(p, prompts[0])

    def test_format_payload_accepts_token_ids(self):
        from load_test import OpenAIProvider

        opts = mock.Mock(
            chat=True,
            rerank=False,
            embeddings=False,
            stream=False,
            temperature=1.0,
            n=1,
            top_k=None,
            logprobs=None,
            reasoning_effort=None,
            clear_assistant=False,
        )
        provider = OpenAIProvider("test-model", opts)
        payload = provider.format_payload([10, 20, 30], max_tokens=16, images=None)
        self.assertEqual(payload["prompt"], [10, 20, 30])
        self.assertNotIn("messages", payload)


if __name__ == "__main__":
    unittest.main()
