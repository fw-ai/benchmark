#!/usr/bin/env python3
"""Unit tests for strict token-id prompt construction in load_test.py."""

import random
import unittest
from unittest import mock

import transformers

from load_test import TranslationDataset
from prefill_load_test import build_pair_ids, load_chunks


class StrictPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        cls.tokenizer.add_bos_token = False
        cls.tokenizer.add_eos_token = False
        cls.chunks = load_chunks("limericks")

    def test_exact_prompt_length_no_chat(self):
        ds = TranslationDataset(
            dataset="limericks",
            tokenizer=self.tokenizer,
            tokenizer_path="gpt2",
            chat=False,
            num_tokens=512,
            common_tokens=0,
            seed=1,
        )
        ids, reported = next(iter(ds))
        self.assertEqual(reported, 512)
        self.assertEqual(len(ids), 512)

    def test_exact_prompt_length_with_chat(self):
        fake_prefix, fake_suffix = [1, 2, 3], [4, 5]
        with mock.patch("load_test.resolve_model_type", return_value="gpt2"), mock.patch(
            "load_test.split_chat_template", return_value=(fake_prefix, fake_suffix)
        ):
            ds = TranslationDataset(
                dataset="limericks",
                tokenizer=self.tokenizer,
                tokenizer_path="gpt2",
                chat=True,
                num_tokens=512,
                common_tokens=0,
                seed=2,
            )
            ids, reported = next(iter(ds))
        self.assertEqual(len(ids), 512)
        self.assertEqual(ids[:3], fake_prefix)
        self.assertEqual(ids[-2:], fake_suffix)

    def test_shared_prefix_matches_cached_tokens(self):
        ds = TranslationDataset(
            dataset="limericks",
            tokenizer=self.tokenizer,
            tokenizer_path="gpt2",
            chat=False,
            num_tokens=1000,
            common_tokens=600,
            seed=3,
        )
        first, _ = next(iter(ds))
        second, _ = next(iter(ds))
        self.assertEqual(first[:600], second[:600])
        self.assertNotEqual(first, second)

    def test_per75_shape_shared_prefix(self):
        prompt_tokens = 5000
        cached_tokens = 3750
        ds = TranslationDataset(
            dataset="limericks",
            tokenizer=self.tokenizer,
            tokenizer_path="gpt2",
            chat=False,
            num_tokens=prompt_tokens,
            common_tokens=cached_tokens,
            seed=99,
        )
        prompts = [next(iter(ds))[0] for _ in range(5)]
        prefix = prompts[0][:cached_tokens]
        for p in prompts[1:]:
            self.assertEqual(len(p), prompt_tokens)
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
