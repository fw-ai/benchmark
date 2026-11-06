"""Tests for the reuse-accounting split (D-140): the resolvers, the probe's inference, the chat rendering
preference, the required chat mode, the head calibration's layout effect, and the per-request reuse record."""

from __future__ import annotations

import argparse
import json
import os

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

_LIMERICKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limericks.txt")


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
    # Values from DSV4.1/MiniMax measurements (D-134/D-120); any value exercises the path
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
    # Values from DSV4.1/MiniMax measurements (D-134/D-120); any value exercises the path
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
    """D-139: --chat no longer defaults to on; omitting the mode raises before any request."""
    from locust.argument_parser import get_empty_argument_parser, setup_parser_arguments

    parser = get_empty_argument_parser()
    setup_parser_arguments(parser)
    lt.init_parser(parser)
    opts = parser.parse_args(["--provider", "vllm", "-m", "m"])  # neither --chat nor --no-chat
    assert opts.chat is None  # no implicit default
    with pytest.raises(ValueError, match="the mode is required"):
        lt.LLMUser._require_explicit_chat_mode(opts)
    assert (
        lt.LLMUser._require_explicit_chat_mode(parser.parse_args(["--provider", "vllm", "-m", "m", "--chat"])) is None
    )
    assert (
        lt.LLMUser._require_explicit_chat_mode(parser.parse_args(["--provider", "vllm", "-m", "m", "--no-chat"]))
        is None
    )


def test_overhang_and_head_shape_the_shared_prefix() -> None:
    """D-113/D-120/D-134: the shared prefix grows by about the overhang (shrinks when negative), the head is
    subtracted from the total, and designed stays the requested share."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limericks.txt")

    class WordTok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def dataset(**kw):
        kw.setdefault("path", path)
        kw.setdefault("prompt", "\n\nTranslate.")
        kw.setdefault("tokenizer", WordTok())
        kw.setdefault("tokenizer_path", "x")
        kw.setdefault("chat", False)
        kw.setdefault("num_tokens", 400)
        kw.setdefault("common_tokens", 100)
        return TranslationDataset(**kw)

    plain = dataset()  # common_tokens 100: the -pcml share
    longer = dataset(reuse_overhang=130)
    # Values from DSV4.1/MiniMax measurements (D-134/D-120); any value exercises the path
    trimmed = dataset(reuse_overhang=-22)
    headed = dataset(chat_head=26)
    assert plain._prefix  # the shared prefix exists (a -pcml run)
    assert longer._prefix.startswith(plain._prefix)  # the same limericks, then more
    assert (
        abs((longer._prefix_suffix_tokens - plain._prefix_suffix_tokens) - 130) <= 45
    )  # ~the overhang (whole-limerick granularity on both sides)
    assert trimmed._prefix_suffix_tokens < plain._prefix_suffix_tokens  # actually trims
    assert headed._num_tokens == 400 - 26  # the head is subtracted from the total
    assert plain.designed_cached_tokens == 100 == trimmed.designed_cached_tokens


def test_probe_summary_fields_carry_the_fit() -> None:
    PrefixReuseProbe.points = [(1097, 969), (2649, 2521)]
    PrefixReuseProbe.overhang, PrefixReuseProbe.boundary = 128, 0
    PrefixReuseProbe.form, PrefixReuseProbe.residual = "outer", 0
    PrefixReuseProbe.under_4k = PrefixReuseProbe.over_4k = 128
    # Values from DSV4.1/MiniMax measurements (D-134/D-120); any value exercises the path
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
        # Values from DSV4.1/MiniMax measurements (D-134/D-120); any value exercises the path
        content = json.dumps({"usage": {"prompt_tokens": 30}}).encode()

        def raise_for_status(self):
            pass

    class Session:
        def post(self, url, data=None, headers=None, timeout=None):
            return Resp()

    monkeypatch.setattr(lt.requests, "Session", lambda: Session())
    monkeypatch.setattr(lt, "_render_chat", lambda t, p, m: [0] * 4)
    lt._CHAT_HEAD_CALIBRATION.clear()  # another test measured this key
    from locust.argument_parser import get_empty_argument_parser, setup_parser_arguments

    parser = get_empty_argument_parser()
    setup_parser_arguments(parser)
    lt.init_parser(parser)
    opts = parser.parse_args(["--provider", "vllm", "-m", "m", "--chat"])
    env = type("Env", (), {"host": "http://x"})()
    assert measure_chat_head(env, opts, None, "tok") == 26
    assert measure_chat_head(None, opts, None, "tok") is None  # no host: unmeasurable


def test_chat_head_zero_is_distinguishable_from_unmeasured(caplog) -> None:
    """N10-145-3: an unmeasured head is None (a loud warning, no calibration, a blank summary field); a
    measured 0 is 0 (the server adds nothing, calibrated, 0 in the summary)."""
    ds = TranslationDataset(
        _LIMERICKS, "\n\nTranslate.", _WordTok(), "x", chat=False, num_tokens=50, common_tokens=10, chat_head=None
    )
    assert ds.chat_head is None and ds._num_tokens == 50
    ds = TranslationDataset(
        _LIMERICKS, "\n\nTranslate.", _WordTok(), "x", chat=False, num_tokens=50, common_tokens=10, chat_head=0
    )
    assert ds.chat_head == 0 and ds._num_tokens == 50  # measured 0: nothing subtracted, but calibrated
    with caplog.at_level("WARNING", logger="load_test"):
        assert measure_chat_head(None, argparse.Namespace(provider="vllm"), None, "t") is None
    assert "chat head can't be measured" in caplog.text and "no registered host" in caplog.text


class _WordTok:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2]


def test_an_unmeasured_head_flows_through_datasetholder_as_none(caplog, monkeypatch) -> None:
    """N10-145-5: without a measurable head (no --provider), DatasetHolder must keep None - not record 0 - so
    the layout warns, the info line stays silent and the summary field is blank. Tested through the holder,
    not by setting ds.chat_head directly."""
    from locust.argument_parser import get_empty_argument_parser, setup_parser_arguments

    parser = get_empty_argument_parser()
    setup_parser_arguments(parser)
    lt.init_parser(parser)
    opts = parser.parse_args(["--chat", "-p", "120", "-pcml", "40", "--dataset", "limericks"])
    monkeypatch.setattr(lt.DatasetHolder, "_instance", None)
    monkeypatch.setattr(
        lt, "InitTracker", type("IT", (), {"environment": None, "load_tokenizer": staticmethod(lambda p: _WordTok())})
    )
    with caplog.at_level("WARNING", logger="load_test"):
        ds = lt.DatasetHolder.get_instance(opts)
    assert ds.chat_head is None  # not 0
    assert "chat head can't be measured" in caplog.text
    assert "chat head is unmeasured" in caplog.text
    assert "chat head 0 tokens" not in caplog.text and "chat head None" not in caplog.text
    lt.DatasetHolder._instance = None


def test_no_pcml_run_gets_no_silent_shared_prefix() -> None:
    """Review fix 2: without -pcml the overhang must not lay a shared prefix out (the server would serve
    prefix-cache hits to a no-cache benchmark, with no reuse record to show it)."""
    import random as _random

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    plain = TranslationDataset(
        _LIMERICKS, "\n\nT.", _Tok(), "x", chat=False, num_tokens=80, common_tokens=0, reuse_overhang=128
    )
    assert plain._prefix == "" and plain._prefix_suffix_tokens == len("\n\nT.".split())  # nothing shared
    assert plain.designed_cached_tokens == 0


def test_a_reply_without_usage_is_an_unmeasured_head(monkeypatch, caplog) -> None:
    """Review fix 4: a 200 reply with no usable prompt-token count is unmeasured (None, warned), not a
    negative head applied to the layout."""

    class Resp:
        status_code = 200
        content = json.dumps({"choices": []}).encode()  # no usage

        def raise_for_status(self):
            pass

    class Session:
        def post(self, *a, **kw):
            return Resp()

    monkeypatch.setattr(lt.requests, "Session", lambda: Session())
    monkeypatch.setattr(lt, "_render_chat", lambda t, p, m: [0] * 4)
    lt._CHAT_HEAD_CALIBRATION.clear()  # an earlier test measured this (host, model, tokenizer) key
    from locust.argument_parser import get_empty_argument_parser, setup_parser_arguments

    parser = get_empty_argument_parser()
    setup_parser_arguments(parser)
    lt.init_parser(parser)
    opts = parser.parse_args(["--provider", "vllm", "-m", "m", "--chat"])
    env = type("Env", (), {"host": "http://x"})()
    with caplog.at_level("WARNING", logger="load_test"):
        assert measure_chat_head(env, opts, None, "tok") is None
    assert "no prompt token count" in caplog.text
    assert not lt._CHAT_HEAD_CALIBRATION  # never cached as a measurement


def test_provider_from_host_writes_back_for_setup() -> None:
    """Review fix 3: the host-based guess is available without --provider and -t, so the calibration runs for
    the common invocation."""
    from load_test import provider_from_host

    assert provider_from_host("https://api.fireworks.ai/inference") == "fireworks"
    assert provider_from_host("https://api.together.xyz") == "together"
    assert provider_from_host("http://localhost:8000") is None
    assert provider_from_host("http://localhost:8000", systemone=True) == "fireworks"


def test_probe_point_prefixes_share_no_leading_text() -> None:
    """Review fix 5: each probe point's prefix starts with its own header, so no point's text is a prefix of
    another's."""

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids):
            return " ".join(ids)

    tok = Tok()
    headers = [tok.encode(f"Probe ab12 {i}.\n") for i in range(3)]
    base = ["corpus"] * 50
    points = [tuple(headers[i] + base) for i in range(3)]
    from load_test import _common_prefix_count

    for a in points:
        for b in points:
            if a is not b:
                # they diverge inside the header (at the point index), long before either's corpus slice,
                # so no point's prefix is a prefix of another's
                assert _common_prefix_count(a, b) < min(len(a), len(b)) and _common_prefix_count(a, b) <= len(
                    headers[0]
                )


def test_pcml_prompt_field_is_the_designed_server_length() -> None:
    """N10-15-23 (D-183): in chat, -pcml builds the client's count net of the calibrated head (num_tokens - head), so
    the reuse record's prompt= adds that one constant head back: the designed SERVER length, as on shape-mix rows. At
    500 tokens and head 26 the client count alone read the harness-delivery check +5.5% off a correct layout."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limericks.txt")

    class WordTok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def first_count(**kw):
        import random as _random

        _random.seed(5)  # -pcml draws limericks from the global RNG
        ds = TranslationDataset(
            path, "\n\nTranslate.", WordTok(), "x", chat=False, num_tokens=500, common_tokens=250, **kw
        )
        return next(iter(ds))[1]

    # Values from DSV4.1/MiniMax measurements (D-134/D-120); any value exercises the path
    head = 26
    client = first_count(chat_head=head)  # what the client builds and counts
    limerick = 60  # whole limericks: the count lands at or a little past its target
    assert 500 - head <= client < 500 - head + limerick
    # the record the response path logs for this row (LLMUser calls pcml_reuse_record with the dataset's chat_head)
    record = lt.pcml_reuse_record(250, 252, client, head)
    designed = int(record.split("prompt=")[1].split()[0])
    assert designed == client + head and 500 <= designed < 500 + limerick
    assert lt.pcml_reuse_record(250, None, client, 0).endswith(f" delivered=none prompt={client}")  # completions
    assert abs(client / designed - 1) > 0.04  # the client count alone would read the server's length ~5% long
    assert first_count(chat_head=None) == first_count()  # completions / unmeasured: no head, prompt= = client count
