#!/usr/bin/env python3
"""Unit tests for strict token-id prompt construction in load_test.py."""

import os
import random
import unittest
from unittest import mock

import transformers

from load_test import OpenAIProvider, TranslationDataset
from prefill_load_test import load_chunks


class StrictPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        cls.tokenizer.add_bos_token = False
        cls.tokenizer.add_eos_token = False
        cls.dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limericks.txt")

    def _make_dataset(self, **kwargs):
        defaults = dict(
            path=self.dataset_path,
            prompt="\n\nTranslate the limericks above to Spanish.",
            tokenizer=self.tokenizer,
            tokenizer_path="gpt2",
            chat=False,
            num_tokens=512,
            common_tokens=0,
            strict=True,
            seed=1,
        )
        defaults.update(kwargs)
        return TranslationDataset(**defaults)

    def test_exact_prompt_length_no_chat(self):
        ds = self._make_dataset()
        ids, reported = next(iter(ds))
        self.assertEqual(reported, 512)
        self.assertEqual(len(ids), 512)

    def test_exact_prompt_length_with_chat(self):
        fake_prefix, fake_suffix = [1, 2, 3], [4, 5]
        with mock.patch("load_test.resolve_model_type", return_value="gpt2"), mock.patch(
            "load_test.split_chat_template", return_value=(fake_prefix, fake_suffix)
        ):
            ds = self._make_dataset(chat=True, seed=2)
            ids, reported = next(iter(ds))
        self.assertEqual(len(ids), 512)
        self.assertEqual(ids[:3], fake_prefix)
        self.assertEqual(ids[-2:], fake_suffix)

    def test_shared_prefix_matches_cached_tokens(self):
        ds = self._make_dataset(num_tokens=1000, common_tokens=600, seed=3)
        first, _ = next(iter(ds))
        second, _ = next(iter(ds))
        self.assertEqual(first[:600], second[:600])
        self.assertNotEqual(first, second)

    def test_custom_prompt_suffix_is_included(self):
        custom = "\n\nMy custom instruction."
        ds = self._make_dataset(prompt=custom, num_tokens=256, common_tokens=0, seed=4)
        suffix_ids = self.tokenizer.encode(custom, add_special_tokens=False)
        ids, _ = next(iter(ds))
        self.assertEqual(ids[-len(suffix_ids) :], suffix_ids)

    def test_rerank_mode_returns_text_not_token_ids(self):
        ds = self._make_dataset(strict=False, num_tokens=200, common_tokens=0)
        prompt, _ = next(iter(ds))
        self.assertIsInstance(prompt, str)

    def test_format_payload_accepts_token_ids(self):
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

    def test_rerank_format_payload_splits_text_not_token_ids(self):
        opts = mock.Mock(
            rerank=True,
            rerank_query=None,
            rerank_top_n=None,
            rerank_return_documents=True,
        )
        provider = OpenAIProvider("test-model", opts)
        payload = provider.format_payload("doc one\n\ndoc two", max_tokens=16, images=None)
        self.assertEqual(payload["documents"], ["doc one", "doc two"])


if __name__ == "__main__":
    unittest.main()
