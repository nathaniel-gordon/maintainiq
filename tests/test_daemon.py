"""The daemon: sink fan-out, drop-rate budget, checkpoint restore and its refusals."""
from __future__ import annotations

from typing import Any

import pytest

from conftest import LIVE_CONFIG, SEED, make_daemon, make_source
from pme.core.events import Alert, LateEvent
from pme.errors import CheckpointError, DropRateExceededError, StreamMismatchError, TrainServeSkewError
from pme.runtime.config import OutputPaths, StreamConfig
from pme.runtime.daemon import MaintenanceDaemon
from pme.sinks.base import Sink


class CollectingSink(Sink):
    """A third-party sink that only wants alerts."""

    name = "collector"
    accepts = (Alert,)

    def __init__(self) -> None:
        self.records: list[Any] = []
        self.opened = False
        self.flushes = 0

    def open(self) -> None:
        self.opened = True

    def write(self, record: Any) -> None:
        self.records.append(record)

    def flush(self) -> None:
        self.flushes += 1


def test_a_clean_run_produces_alerts_and_a_schedule(models, ordered_source, tmp_path):
    report = make_daemon(models, ordered_source, tmp_path).run()
    assert report.events_consumed == len(ordered_source)
    assert report.engine_stats["events_dropped_late"] == 0
    assert report.drop_rate == 0.0
    assert (tmp_path / "alerts.jsonl").exists()
    assert (tmp_path / "schedule.csv").exists()
    assert (tmp_path / "late.jsonl").read_text(encoding="utf-8") == ""


def test_dropped_events_reach_the_late_side_output(models, tmp_path):
    source = make_source(jitter=6, late_fraction=0.05)
    daemon = make_daemon(models, source, tmp_path, max_out_of_orderness_h=2)
    report = daemon.run()
    dropped = report.engine_stats["events_dropped_late"]
    assert dropped > 0
    lines = (tmp_path / "late.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == dropped
    assert report.sink_summaries["late"]["late_events"] == dropped
    assert sum(report.engine_stats["dropped_by_machine"].values()) == dropped


def test_a_run_over_the_drop_budget_fails(models, tmp_path):
    source = make_source(jitter=6, late_fraction=0.05)
    daemon = make_daemon(models, source, tmp_path, max_out_of_orderness_h=2, max_drop_rate=0.001)
    with pytest.raises(DropRateExceededError) as excinfo:
        daemon.run()
    assert excinfo.value.exit_code != 0
    # Output is still written: the run happened, it just is not acceptable.
    assert (tmp_path / "alerts.jsonl").exists()
    assert excinfo.value.details["drop_rate"] > 0.001


def test_extra_sinks_are_routed_by_what_they_accept(models, ordered_source, tmp_path):
    collector = CollectingSink()
    daemon = MaintenanceDaemon(ordered_source, models, LIVE_CONFIG, OutputPaths.under(tmp_path),
                               extra_sinks=[collector])
    report = daemon.run()
    assert collector.opened
    assert collector.flushes >= 1
    assert [a.key for a in collector.records] == [a.key for a in report.alerts]
    assert all(isinstance(r, Alert) for r in collector.records)


def test_a_sink_declaring_late_events_receives_them(models, tmp_path):
    class LateCollector(CollectingSink):
        name = "late_collector"
        accepts = (LateEvent,)

    collector = LateCollector()
    source = make_source(jitter=6, late_fraction=0.05)
    daemon = MaintenanceDaemon(source, models, StreamConfig(max_out_of_orderness_h=2, max_drop_rate=1.0,
                                                            scoring_batch_size=128),
                               OutputPaths.under(tmp_path), extra_sinks=[collector])
    report = daemon.run()
    assert len(collector.records) == report.engine_stats["events_dropped_late"] > 0


def test_results_land_in_the_database(models, ordered_source, tmp_path, store):
    daemon = make_daemon(models, ordered_source, tmp_path, store=store)
    report = daemon.run()
    run = store.get_run(report.run_id)
    assert run["status"] == "completed"
    assert run["events_consumed"] == report.events_consumed
    assert run["source_fingerprint"] == ordered_source.fingerprint()
    assert sum(store.count_alerts_by_severity(report.run_id).values()) == len(report.alerts)
    assert store.list_machine_state(report.run_id).total > 0


def test_a_failed_run_is_recorded_as_failed(models, tmp_path, store):
    source = make_source(jitter=6, late_fraction=0.05)
    daemon = make_daemon(models, source, tmp_path, store=store, max_out_of_orderness_h=2, max_drop_rate=0.0)
    with pytest.raises(DropRateExceededError):
        daemon.run()
    assert store.get_run(daemon.run_id)["status"] == "failed"


@pytest.mark.parametrize("fraction", [0.05, 0.33, 0.5, 0.9])
def test_restore_reproduces_the_uninterrupted_feeds(models, tmp_path, fraction):
    source = make_source(jitter=6)
    whole = make_daemon(models, source, tmp_path, prefix="whole_")
    whole.run()
    alerts = (tmp_path / "whole_alerts.jsonl").read_bytes()
    schedule = (tmp_path / "whole_schedule.csv").read_bytes()

    cut = max(1, int(len(source) * fraction))
    first = make_daemon(models, make_source(jitter=6), tmp_path, prefix="part_",
                        checkpoint_every=max(1, cut // 3), stop_after_events=cut)
    first.run()
    second = make_daemon(models, make_source(jitter=6), tmp_path, prefix="part_",
                         checkpoint_every=max(1, cut // 3))
    assert second.restore_latest()
    second.run()

    assert (tmp_path / "part_alerts.jsonl").read_bytes() == alerts
    assert (tmp_path / "part_schedule.csv").read_bytes() == schedule


def test_restore_refuses_a_differently_ordered_stream(models, tmp_path):
    first = make_daemon(models, make_source(jitter=6), tmp_path, checkpoint_every=500,
                        stop_after_events=1000)
    first.run()
    # Same source, same length, same knobs — only the broker's delivery order
    # differs. That is a different stream and must not be resumed into.
    second = make_daemon(models, make_source(jitter=6, delivery_seed=SEED + 99), tmp_path)
    with pytest.raises(StreamMismatchError, match="fingerprint"):
        second.restore_latest()


def test_restore_refuses_a_different_source(models, tmp_path):
    first = make_daemon(models, make_source(jitter=6), tmp_path, checkpoint_every=500, stop_after_events=1000)
    first.run()
    second = make_daemon(models, make_source(jitter=0), tmp_path)
    with pytest.raises(StreamMismatchError, match="different source"):
        second.restore_latest()


def test_restore_refuses_a_different_lateness_policy(models, tmp_path):
    first = make_daemon(models, make_source(jitter=6), tmp_path, checkpoint_every=500, stop_after_events=1000)
    first.run()
    second = make_daemon(models, make_source(jitter=6), tmp_path, allowed_lateness_h=5)
    with pytest.raises(CheckpointError, match="semantics"):
        second.restore_latest()


def test_restore_refuses_changed_alert_thresholds(models, tmp_path):
    first = make_daemon(models, make_source(jitter=6), tmp_path, checkpoint_every=500, stop_after_events=1000)
    first.run()
    second = make_daemon(models, make_source(jitter=6), tmp_path, critical_risk=0.9)
    with pytest.raises(CheckpointError, match="semantics"):
        second.restore_latest()


def test_a_bundle_trained_under_a_different_gate_is_refused(models, ordered_source, tmp_path):
    with pytest.raises(TrainServeSkewError):
        make_daemon(models, ordered_source, tmp_path, min_window_samples=models.min_window_samples - 1)


def test_a_stop_request_checkpoints_and_stops(models, tmp_path):
    source = make_source()
    daemon = make_daemon(models, source, tmp_path, checkpoint_every=250)
    daemon.request_stop("test")
    report = daemon.run()
    assert report.stopped_early
    assert report.events_consumed == 1
    assert daemon.checkpoints.exists()


def test_metrics_file_keeps_one_schema_for_the_whole_run(models, ordered_source, tmp_path):
    import json

    make_daemon(models, ordered_source, tmp_path).run()
    payload = json.loads((tmp_path / "stream_metrics.json").read_text(encoding="utf-8"))
    assert isinstance(payload["runtime"]["watermark"], int)
    assert payload["runtime"]["stream_closed"] is True
    edges = [b["upper_h"] for b in payload["runtime"]["lateness_histogram"]]
    assert edges == [*sorted(e for e in edges if e is not None), None]
    for beat in payload["heartbeats"]:
        assert beat["watermark"] is None or isinstance(beat["watermark"], int)
