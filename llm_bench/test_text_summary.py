from __future__ import annotations

from locust.stats import RequestStats

from load_test import text_summary_entries


def _client_metrics(stats: RequestStats) -> None:
    for ttft, total in [(340, 2100), (360, 2200), (380, 2300)]:
        stats.log_request("METRIC", "time_to_first_token", ttft, 0)
        stats.log_request("METRIC", "total_latency", total, 0)


def test_server_side_metrics_reported_when_recorded() -> None:
    stats = RequestStats()
    _client_metrics(stats)
    for ttft, total in [(70, 1800), (80, 1900), (90, 2000)]:
        stats.log_request("METRIC", "server_side_time_to_first_token", ttft, 0)
        stats.log_request("METRIC", "server_side_total_latency", total, 0)

    entries, percentile_metrics = text_summary_entries(stats, stream=True)

    assert entries["time_to_first_token"] == 360
    assert entries["server_side_time_to_first_token"] == 80
    assert entries["server_side_total_latency"] == 1900
    assert percentile_metrics == [
        "time_to_first_token",
        "total_latency",
        "server_side_time_to_first_token",
        "server_side_total_latency",
    ]


def test_server_side_metrics_omitted_when_not_recorded() -> None:
    stats = RequestStats()
    _client_metrics(stats)

    entries, percentile_metrics = text_summary_entries(stats, stream=True)

    assert "server_side_time_to_first_token" not in entries
    assert "server_side_total_latency" not in entries
    assert percentile_metrics == ["time_to_first_token", "total_latency"]


def test_non_streaming_blanks_client_ttft() -> None:
    stats = RequestStats()
    _client_metrics(stats)

    entries, _ = text_summary_entries(stats, stream=False)

    assert entries["time_to_first_token"] == ""
    assert entries["latency_per_token"] == ""
