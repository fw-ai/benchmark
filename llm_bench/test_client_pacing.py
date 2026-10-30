from __future__ import annotations

import argparse
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import gevent
import pytest
import requests

import load_test
from load_test import (
    ClientCpu,
    DatasetHolder,
    FixedQPSPacer,
    InitTracker,
    LLMUser,
    SendTimeline,
    arrival_summary_entries,
    gap_cv2,
    sample_interarrival,
)


@pytest.mark.parametrize("cv2", [0.25, 1.0, 4.0, 10.0])
def test_gamma_interarrival_mean_and_cv2(cv2: float) -> None:
    rng = random.Random(0)
    mean_wait = 1 / 20
    samples = [sample_interarrival("gamma", mean_wait, cv2, rng) for _ in range(200_000)]
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)

    assert mean == pytest.approx(mean_wait, rel=0.03)
    assert var / mean**2 == pytest.approx(cv2, rel=0.08)


def test_exponential_is_the_cv2_one_case() -> None:
    rng = random.Random(1)
    samples = [sample_interarrival("exponential", 0.1, rng=rng) for _ in range(100_000)]
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)

    assert var / mean**2 == pytest.approx(1.0, rel=0.05)


def test_pacer_rejects_non_positive_cv2_and_unknown_distributions() -> None:
    with pytest.raises(ValueError):
        FixedQPSPacer(10, "gamma", 0)
    with pytest.raises(ValueError):
        FixedQPSPacer(10, "weibull")


@pytest.mark.parametrize(
    "distribution,cv2,target", [("constant", 1.0, 0.0), ("exponential", 1.0, 1.0), ("gamma", 4.0, 4.0)]
)
def test_pacer_schedule_has_the_requested_rate_and_cv2(distribution, cv2, target) -> None:
    pacer = FixedQPSPacer(20, distribution, cv2)
    times = [pacer.next_offset() for _ in range(20_000)]
    assert (len(times) - 1) / (times[-1] - times[0]) == pytest.approx(20, rel=0.05)
    gaps = [b - a for a, b in zip(times, times[1:])]
    mean = sum(gaps) / len(gaps)
    assert sum((g - mean) ** 2 for g in gaps) / len(gaps) / mean**2 == pytest.approx(target, abs=0.1 * max(target, 0.1))


def test_arrival_summary_reports_intended_delivered_and_lateness() -> None:
    rng = random.Random(0)
    intended, t = [], 0.0
    for _ in range(3000):
        t += rng.expovariate(10)
        intended.append(t)
    actual = [i + 0.002 for i in intended]
    actual[5] += 0.5

    entries = arrival_summary_entries(intended, actual)

    assert entries["intended_gap_cv2"] == pytest.approx(1.0, rel=0.1)
    assert entries["delivered_gap_cv2"] == pytest.approx(entries["intended_gap_cv2"], rel=0.05)
    assert entries["delivered_dispersion_1s"] == pytest.approx(entries["intended_dispersion_1s"], rel=0.05)
    assert entries["intended_dispersion_10s"] == pytest.approx(1.0, rel=0.35)
    assert entries["send_lateness_p50_ms"] == pytest.approx(2.0, abs=0.01)
    assert entries["send_lateness_max_ms"] == pytest.approx(502.0, abs=0.01)
    assert entries["dispersion_10s_bins"] == int((actual[-1] - actual[0]) // 10)  # how few bins the 10 s figure has
    assert entries["timed_sends"] == 3000
    assert arrival_summary_entries([], []) == {}
    assert gap_cv2([0.0, 1.0]) is None  # too few arrivals


def _stalled_user(pacer, sent, sleep_stall=None):
    user = SimpleNamespace(
        qps_pacer=pacer,
        environment=SimpleNamespace(parsed_options=argparse.Namespace(qps_max_lateness=0.5)),
        _prepare_request=lambda: "prepared",
        _in_flight=SimpleNamespace(spawn=lambda fn, prepared, intended: sent.append((intended, time.monotonic()))),
        _send_paced=None,
    )
    user._pace_and_send = lambda *args: LLMUser._pace_and_send(user, *args)
    return user


def test_a_stall_moves_the_schedule_once_without_a_burst(monkeypatch) -> None:
    InitTracker.spawned.set()
    SendTimeline.reset()
    monkeypatch.setattr(InitTracker, "last_late_warning", 0.0)
    pacer = FixedQPSPacer(10, "constant")
    pacer.start = time.monotonic() - 3.0  # the generator stalled for 3 s: the first ~30 sends are overdue
    sent = []
    user = _stalled_user(pacer, sent)

    LLMUser._paced_request(user)  # 2.9 s late: the schedule moves ~2.9 s later and this send moves with it
    assert pacer.epoch == 1
    assert pacer.shifted_seconds == pytest.approx(2.9, abs=0.05)
    assert pacer.resync(0, 5.0) is False  # a second stale send from the same stall doesn't shift again
    assert len(sent) == 1 and abs(sent[0][0] - time.monotonic()) < 0.1
    LLMUser._paced_request(user)  # the next slot follows 0.1 s later: no burst, no hole
    assert len(sent) == 2 and sent[1][0] - sent[0][0] == pytest.approx(0.1, abs=0.02)


def test_users_asleep_during_a_stall_keep_the_schedule_spacing(monkeypatch) -> None:
    InitTracker.spawned.set()
    SendTimeline.reset()
    monkeypatch.setattr(InitTracker, "last_late_warning", 0.0)
    pacer = FixedQPSPacer(10, "constant")
    pacer.start = time.monotonic()
    real_sleep = gevent.sleep
    stall_end = time.monotonic() + 1.5
    # Every user asleep before the stall ends wakes when it ends (a 1.5 s stall of the whole process)
    monkeypatch.setattr(
        load_test.gevent, "sleep", lambda seconds=0: real_sleep(max(seconds, stall_end - time.monotonic()))
    )
    sent = []
    user = _stalled_user(pacer, sent)
    gevent.joinall([gevent.spawn(LLMUser._paced_request, user) for _ in range(20)])  # slots over the next 2 s

    assert all(at - intended <= 0.5 + 0.05 for intended, at in sent)  # nothing goes out more than 0.5 s late
    assert pacer.epoch == 1  # the first late waker shifts the schedule, once
    # every claimed send goes out, at the schedule's own spacing, from the end of the stall: no burst, no hole
    assert len(sent) == 20
    times = sorted(at for _, at in sent)
    assert times[0] - stall_end < 0.1
    gaps = [round(b - a, 3) for a, b in zip(times, times[1:])]
    assert all(g == pytest.approx(0.1, abs=0.03) for g in gaps), (gaps, times[0] - stall_end, pacer.shifts)


class _SlowHandler(BaseHTTPRequestHandler):
    delay = 0.4

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(self.delay)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


def test_a_send_is_on_the_timeline_once_it_is_on_the_wire_not_when_answered() -> None:
    """N9-143-1: sends still waiting for their response at the end of a run must be in the figures."""
    from load_test import _begin_wire_timing, _finish_wire_timing, _timed_adapter

    server = HTTPServer(("127.0.0.1", 0), _SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session = requests.Session()
    session.mount("http://", _timed_adapter(pool_maxsize=4))
    SendTimeline.reset()
    answered = []

    def send():
        _begin_wire_timing(intended=time.monotonic() - 0.05)
        try:
            session.post(f"http://127.0.0.1:{server.server_port}/v1/completions", data=b"x" * 200_000)
            answered.append(time.monotonic())
        finally:
            _finish_wire_timing()

    try:
        sent_at = time.monotonic()
        greenlet = gevent.spawn(send)
        gevent.sleep(0.2)  # the server is still holding the response
        assert not answered and len(SendTimeline.actual) == 1  # already recorded, before the headers arrive
        assert SendTimeline.actual[0] - sent_at < 0.15 and SendTimeline.actual[0] - SendTimeline.intended[0] >= 0.04
        greenlet.join()
    finally:
        server.shutdown()
    assert len(SendTimeline.actual) == 1  # not recorded a second time when the response came


def test_a_send_that_never_reaches_the_wire_is_still_counted() -> None:
    from load_test import _begin_wire_timing, _finish_wire_timing, _timed_adapter

    session = requests.Session()
    session.mount("http://", _timed_adapter(pool_maxsize=4))
    SendTimeline.reset()
    dispatched = _begin_wire_timing(intended=time.monotonic())
    with pytest.raises(requests.ConnectionError):
        session.post("http://127.0.0.1:9/v1/completions", data=b"x")  # nothing listens on the discard port
    _finish_wire_timing()
    assert len(SendTimeline.actual) == 1 and SendTimeline.actual[0] == pytest.approx(dispatched, abs=0.5)
    _finish_wire_timing()  # idempotent
    assert len(SendTimeline.actual) == 1


def test_oldest_pending_matches_a_scan_across_stalls() -> None:
    """N9-143-4: the heap and cumulative shift give the same oldest pending send as scanning every pending one."""
    rng = random.Random(4)
    pacer = FixedQPSPacer(100, "constant")
    pacer.start = time.monotonic()
    live = {}
    for step in range(400):
        action = rng.random()
        if action < 0.5:
            intended, epoch = pacer.claim_with_epoch()
            live[pacer.track(intended, epoch)] = (intended, epoch)
        elif action < 0.8 and live:
            key = rng.choice(list(live))
            pacer.untrack(key)
            del live[key]
        elif action < 0.9:
            pacer.resync(pacer.epoch, rng.uniform(0.0, 0.5))
        if live:
            expected = max(0.0, time.monotonic() - min(t + sum(pacer.shifts[e:]) for t, e in live.values()))
            assert pacer.oldest_lateness() == pytest.approx(expected, abs=1e-3)
        else:
            assert pacer.oldest_lateness() == 0.0
        e = rng.randrange(pacer.epoch + 1)
        assert pacer.shift_since(e)[0] == pytest.approx(sum(pacer.shifts[e:]))


def test_summary_file_append_never_drops_the_row(tmp_path, caplog) -> None:
    """N9-143-2: an older client's header, or a --qps row after a plain one, extends the header; no row is lost."""
    import csv

    from load_test import append_summary_row

    path = str(tmp_path / "sweep.csv")
    append_summary_row(path, {"a": 1, "b": 2})  # new file: header and row
    append_summary_row(path, {"a": 3, "b": 4})  # same columns
    append_summary_row(path, {"b": 5})  # fewer columns: blank
    with caplog.at_level("WARNING", logger="load_test"):
        append_summary_row(path, {"a": 6, "b": 7, "client_cpu_percent": 9.5, "qps": 20})  # more columns
    assert "header lacked 2 column(s)" in caplog.text and "client_cpu_percent" in caplog.text
    append_summary_row(path, {"qps": 21, "a": 8, "b": 9, "client_cpu_percent": 1.5})  # a subset of the widened header
    rows = list(csv.DictReader(open(path)))
    assert list(rows[0]) == ["a", "b", "client_cpu_percent", "qps"]
    assert [(r["a"], r["b"], r["client_cpu_percent"], r["qps"]) for r in rows] == [
        ("1", "2", "", ""),
        ("3", "4", "", ""),
        ("", "5", "", ""),
        ("6", "7", "9.5", "20"),
        ("8", "9", "1.5", "21"),
    ]
    # an empty existing file gets its header
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    append_summary_row(str(empty), {"x": 1})
    assert empty.read_text().splitlines() == ["x", "1"]
    # a path that can't be written still saves the row beside it
    caplog.clear()
    blocked = tmp_path / "dir.csv"
    blocked.mkdir()
    with caplog.at_level("ERROR", logger="load_test"):
        append_summary_row(str(blocked), {"x": 2})
    assert "couldn't append the row" in caplog.text
    saved = [f for f in tmp_path.iterdir() if f.name.startswith("dir.csv.") and f.name.endswith(".csv")]
    assert len(saved) == 1 and saved[0].read_text().splitlines() == ["x", "2"]


def test_prompt_token_estimate_is_once_per_process_without_tokenizing(monkeypatch) -> None:
    monkeypatch.setattr(DatasetHolder, "_prompt_tokens_estimate", None)
    calls = []
    monkeypatch.setattr(
        InitTracker,
        "load_tokenizer",
        lambda path: calls.append(path) or SimpleNamespace(encode=lambda text: text.split()),
    )
    get_input = lambda: ("a long prompt " * 1000, 16000, None)
    opts = argparse.Namespace(tokenizer=None)
    # every user used to tokenize a whole prompt in on_start, inside the run window
    assert [DatasetHolder.prompt_token_estimate(opts, get_input) for _ in range(100)] == [16000] * 100
    assert calls == []
    monkeypatch.setattr(DatasetHolder, "_prompt_tokens_estimate", None)
    assert DatasetHolder.prompt_token_estimate(opts, lambda: ("three short words", 0, None)) == 3  # no count: once
    assert len(calls) == 1


def test_prompt_token_estimate_averages_a_variable_length_dataset(monkeypatch) -> None:
    """N9-143-7: with dataset-provided counts the one per-process estimate is a mean over a few prompts."""
    monkeypatch.setattr(DatasetHolder, "_prompt_tokens_estimate", None)
    counts = iter([100, 200] * 8 + [999])
    estimate = DatasetHolder.prompt_token_estimate(
        argparse.Namespace(tokenizer=None), lambda: ("p", next(counts), None)
    )
    assert estimate == 150 and next(counts) == 999  # 16 prompts drawn, once


def test_client_cpu_summary_reports_mean_and_peak() -> None:
    ClientCpu._samples = []
    ClientCpu.reset()
    end = time.monotonic() + 0.3
    while time.monotonic() < end:
        sum(range(1000))
    entries = ClientCpu.summary_entries()
    assert 50 <= entries["client_cpu_percent"] <= 105
    assert entries["client_cpu_peak_5s_percent"] >= entries["client_cpu_percent"] - 1e-9
