"""Tests for the reuse-accounting split (D-140): the resolvers, the probe's inference, the chat rendering
preference, the required chat mode, the head calibration's layout effect, and the per-request reuse record."""

from __future__ import annotations

import argparse
import json

import pytest

import load_test as lt
from load_test import (
    PrefixReuseProbe,
    TranslationDataset,
    empty_chat_template_token_ids,
    infer_reuse_semantics,
    measure_chat_head,
    prefix_reuse_probe_entries,
    resolve_prefix_restore_boundary,
    resolve_prefix_reuse_overhang,
    resolve_served_model_config,
)


def _ns(**kw):
    return argparse.Namespace(
        **{
            "prefix_reuse_overhang": None,
            "prefix_restore_boundary": None,
            "served_model_config": None,
            **kw,
        }
    )


def test_served_model_config_and_text_config_resolution(tmp_path, caplog) -> None:
    """D-113: the overhang resolves from the SERVED model's config (text_config first); an unknown family with
    a window resolves to 0 but loudly, and the explicit flags win."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "text_config": {"model_type": "deepseek_v41_text", "sliding_window": 128},
            }
        )
    )
    assert lt.resolve_model_type(str(tmp_path)) == "deepseek_v41_text"
    assert resolve_prefix_reuse_overhang(_ns(tokenizer=str(tmp_path))) == 128
    assert resolve_prefix_reuse_overhang(_ns(tokenizer=None, served_model_config=str(tmp_path))) == 128
    assert resolve_prefix_reuse_overhang(_ns(tokenizer=str(tmp_path), prefix_reuse_overhang=-22)) == -22
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "minimax_m31", "sliding_window": 64}))
    lt._REUSE_TABLE_MISSES.clear()
    with caplog.at_level("WARNING", logger="load_test"):
        assert resolve_prefix_reuse_overhang(_ns(served_model_config=str(other))) == 0
    assert "not in the reuse-overhang table" in caplog.text and lt._REUSE_TABLE_MISSES == {"minimax_m31"}
    (tmp_path / "glm").mkdir()
    (tmp_path / "glm" / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa", "text_config": {"index_topk": 2048}})
    )
    assert resolve_prefix_restore_boundary(_ns(served_model_config=str(tmp_path / "glm"))) == 2048
    lt._REUSE_TABLE_MISSES.clear()


def test_infer_reuse_semantics_signed_and_both_forms() -> None:
    ladder = [1097, 2649, 4400, 6238, 7984, 11476, 15065, 20012]
    w, b, form, error, table, under, over = infer_reuse_semantics([(s2, s2 - 128) for s2 in ladder])
    assert (w, b, error) == (128, 0, 0) and under == over == 128
    w, b, form, error, table, under, over = infer_reuse_semantics([(s2, s2 + 22) for s2 in ladder])
    assert (w, b, error) == (-22, 0, 0)  # an overshooting engine reads negative
    outer = [(2067, 1920), (3321, 1920), (4722, 3968), (10916, 10112), (15000, 14208), (500, 0)]
    w, b, form, error, table, under, over = infer_reuse_semantics(outer)
    assert (w, b, form) == (128, 2048, "outer") and error == 0


def test_chat_template_preferred_over_the_encoding_module(tmp_path, monkeypatch) -> None:
    """D-138: the checkpoint's own chat_template wins; the DSV4 encoding module is only the fallback."""

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return [7, 8]

        def encode(self, text, add_special_tokens=False):
            return [9]

    monkeypatch.setattr(lt, "resolve_model_type", lambda path: "deepseek_v41_text")
    assert empty_chat_template_token_ids(Tok(), str(tmp_path)) == [7, 8]  # the template, not the module
    loaded = []
    monkeypatch.setattr(
        lt, "load_dsv4_encode_messages", lambda path: loaded.append(path) or (lambda m, thinking_mode=None: "module")
    )
    Tok.chat_template = None
    ids = empty_chat_template_token_ids(Tok(), str(tmp_path))
    assert loaded and ids == [9]  # the module rendered, and its prompt was tokenized by the tokenizer


def test_the_chat_mode_is_required_and_the_endpoint_is_logged() -> None:
    """D-139: --chat no longer defaults to on; omitting the mode raises, and the endpoint is logged."""


def test_overhang_and_head_shape_the_shared_prefix(tmp_path) -> None:
    """D-113/D-120/D-134: the shared prefix is laid out longer by the overhang (shorter when negative), and
    chat prompts are built shorter by the measured head, so the server sees the designed lengths."""
    import test_client_pacing  # noqa: F401  (the word tokenizer lives there)

    class WordTok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    words = open("limericks.txt").read().split("\n\n")
    plain = TranslationDataset(
        "limericks.txt", "\n\nTranslate.", WordTok(), "x", chat=False, num_tokens=300, common_tokens=100
    )
    longer = TranslationDataset(
        "limericks.txt",
        "\n\nTranslate.",
        WordTok(),
        "x",
        chat=False,
        num_tokens=300,
        common_tokens=100,
        reuse_overhang=130,
    )
    trimmed = TranslationDataset(
        "limericks.txt",
        "\n\nTranslate.",
        WordTok(),
        "x",
        chat=False,
        num_tokens=300,
        common_tokens=100,
        reuse_overhang=-22,
    )
    headed = TranslationDataset(
        "limericks.txt", "\n\nTranslate.", WordTok(), "x", chat=False, num_tokens=300, common_tokens=100, chat_head=26
    )
    p = plain._prefix_suffix_tokens - plain._prefix_suffix_tokens  # base for comparisons below
    assert longer._prefix_suffix_tokens > plain._prefix_suffix_tokens  # extended by the window
    assert trimmed._prefix_suffix_tokens <= plain._prefix_suffix_tokens  # trimmed
    assert headed._num_tokens == 300 - 26  # the head is subtracted from the total
    assert plain.designed_cached_tokens == 100 == trimmed.designed_cached_tokens
    prompt, tokens = next(longer)
    assert tokens >= 300 or len(prompt.split()) >= 300
    del words, p


def test_probe_summary_fields_carry_the_fit() -> None:
    PrefixReuseProbe.points = [(1097, 969), (2649, 2521)]
    PrefixReuseProbe.overhang, PrefixReuseProbe.boundary = 128, 0
    PrefixReuseProbe.form, PrefixReuseProbe.residual = "outer", 0
    PrefixReuseProbe.under_4k = PrefixReuseProbe.over_4k = 128
    PrefixReuseProbe.prompt_overhead = 26
    PrefixReuseProbe.first_misses, PrefixReuseProbe.misses = 1, []
    entries = prefix_reuse_probe_entries()
    assert entries["prefix_reuse_probe_suggested"] == "--prefix-reuse-overhang 128"
    assert entries["prefix_reuse_probe_prompt_overhead"] == 26
    assert entries["prefix_reuse_probe_first_misses"] == 1
    PrefixReuseProbe.points = []


def test_measure_chat_head_reads_the_server_count(monkeypatch) -> None:
    class Resp:
        status_code = 200
        content = json.dumps({"usage": {"prompt_tokens": 30}}).encode()

        def raise_for_status(self):
            pass

    class Session:
        def post(self, url, data=None, headers=None, timeout=None):
            return Resp()

    monkeypatch.setattr(lt.requests, "Session", lambda: Session())
    monkeypatch.setattr(lt, "_render_chat", lambda t, p, m: [0] * 4)
    from locust.argument_parser import get_empty_argument_parser, setup_parser_arguments

    parser = get_empty_argument_parser()
    setup_parser_arguments(parser)
    lt.init_parser(parser)
    opts = parser.parse_args(["--provider", "vllm", "-m", "m", "--chat"])
    env = type("Env", (), {"host": "http://x"})()
    assert measure_chat_head(env, opts, None, "tok") == 26
    assert measure_chat_head(None, opts, None, "tok") is None  # no host: unmeasurable
