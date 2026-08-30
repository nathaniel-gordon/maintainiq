"""Run reporting: the markdown report and the four-panel chart."""
from __future__ import annotations

import pytest

from conftest import make_daemon, make_source
from pme.core.events import ScoredRecord
from pme.reporting.report import _markdown_table, run_report, stream_chart

#: PNG magic number, so "the chart was written" means an actual image.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def report_and_state(tmp_path_factory, models):
    """One short run whose report artifacts the tests in this module inspect."""
    out = tmp_path_factory.mktemp("reporting")
    daemon = make_daemon(models, make_source(jitter=6, late_fraction=0.02), out,
                         max_out_of_orderness_h=2)
    report = daemon.run()
    return report, daemon


def test_the_markdown_table_renders_a_header_and_a_rule():
    text = _markdown_table(["a", "b"], [[1, 2], [3, 4]])
    assert text.splitlines()[0] == "| a | b |"
    assert text.splitlines()[1] == "| --- | --- |"
    assert text.splitlines()[-1] == "| 3 | 4 |"


def test_an_empty_table_says_none_rather_than_rendering_nothing():
    assert _markdown_table(["a"], []) == "_(none)_"


def test_the_report_carries_the_stream_numbers_it_was_given(report_and_state, tmp_path):
    report, daemon = report_and_state
    text = run_report(report, {"risk_auc": 0.91, "rul_mae_h": 12.3, "train_machines": 4,
                               "test_machines": 2, "feature_rows": 100, "min_window_samples": 20},
                      daemon.schedule_sink.latest, tmp_path / "fleet_report.md")
    assert "# Predictive Maintenance — Stream Run Report" in text
    assert str(report.events_consumed) in text
    assert str(report.engine_stats["events_dropped_late"]) in text
    assert "0.91" in text
    assert (tmp_path / "fleet_report.md").read_text(encoding="utf-8") == text


def test_a_holdout_that_could_not_be_scored_reports_none_not_a_number(report_and_state, tmp_path):
    report, daemon = report_and_state
    text = run_report(report, {"risk_auc": None, "rul_mae_h": None}, daemon.schedule_sink.latest,
                      tmp_path / "r.md")
    assert "| risk AUC | None |" in text
    assert "| RUL MAE (h) | None |" in text
    quality = text.split("## Stream health")[0].lower()
    assert "nan" not in quality.replace("maintenance", "")


def test_the_report_creates_missing_parent_directories(report_and_state, tmp_path):
    report, daemon = report_and_state
    target = tmp_path / "deep" / "nested" / "r.md"
    run_report(report, {}, daemon.schedule_sink.latest, target)
    assert target.is_file()


def test_the_chart_is_a_real_png(report_and_state, tmp_path):
    report, daemon = report_and_state
    path = tmp_path / "fleet_health.png"
    stream_chart(report, daemon.schedule_sink.latest, daemon.metrics_sink.heartbeats, path)
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_the_chart_survives_a_run_with_no_late_events_and_no_heartbeats(report_and_state, tmp_path):
    """The lateness panel has an explicit empty state; exercise it."""
    report, _ = report_and_state
    path = tmp_path / "empty.png"
    latest = {"M1": ScoredRecord("M1", 10, 0.6, 30.0, 0)}
    stripped = report.engine_stats | {"lateness_histogram": []}
    report.engine_stats, original = stripped, report.engine_stats
    try:
        stream_chart(report, latest, [], path)
    finally:
        report.engine_stats = original
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_the_run_report_dict_matches_the_report_object(report_and_state):
    report, _ = report_and_state
    payload = report.to_dict()
    assert payload["events_consumed"] == report.events_consumed
    assert payload["alerts"] == len(report.alerts)
    assert payload["engine"]["drop_rate"] == report.drop_rate
    assert set(payload["sinks"]) >= {"alerts", "schedule", "metrics", "late"}
