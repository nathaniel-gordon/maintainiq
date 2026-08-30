"""End-to-end passes through the command line.

These drive ``main()`` exactly as an operator would and then look at what landed
on disk and in the database. They are the slowest tests in the suite because
every one of them fits a real gradient-boosting bundle; the fleets are kept just
large enough to produce a failure event, because a training set without one is
refused on purpose.
"""
from __future__ import annotations

import json

import pytest

from pme.__main__ import main
from pme.errors import EXIT_OK
from pme.learning.bundle import manifest_path
from pme.sources.synthetic import _simulate
from pme.store.db import open_database
from pme.store.repository import MetadataStore

#: Small enough to fit in a test run, large enough that the wear simulator
#: produces at least one failure so the risk classifier can be fitted.
FLEET = {"machines": 10, "days": 45}


@pytest.fixture(scope="module")
def historian(tmp_path_factory):
    """A settled historian export, event-time ordered, with failures in it."""
    path = tmp_path_factory.mktemp("e2e") / "historian.csv"
    frame = _simulate(FLEET["machines"], 24 * FLEET["days"], seed=7)
    assert frame["failure"].sum() > 0, "the fixture fleet must fail at least once"
    frame.to_csv(path, index=False)
    return path


def test_the_demo_runs_the_whole_pipeline_and_writes_every_artifact(tmp_path, capsys):
    """``pme demo`` is the one command that exercises train, run, crash and resume."""
    code = main(["demo", "--machines", str(FLEET["machines"]), "--days", str(FLEET["days"]),
                 "--output", str(tmp_path), "--log-level", "ERROR"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "alert feed byte-identical to the uninterrupted run: True" in out

    for name in ("fleet_report.md", "fleet_health.png", "alerts.jsonl", "schedule.csv",
                 "stream_metrics.json", "late.jsonl", "models.joblib", "maintainiq.db"):
        assert (tmp_path / name).is_file(), name
    assert (tmp_path / "fleet_health.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    metrics = json.loads((tmp_path / "stream_metrics.json").read_text(encoding="utf-8"))
    assert metrics["runtime"]["events_consumed"] == 24 * FLEET["days"] * FLEET["machines"]

    with open_database(tmp_path / "maintainiq.db") as db:
        store = MetadataStore(db)
        runs = store.list_runs()
        assert runs.total == 1
        assert runs.items[0]["status"] == "completed"
        assert store.latest_checkpoint(runs.items[0]["run_id"]) is not None


def test_train_writes_a_bundle_beside_a_manifest(historian, tmp_path, capsys):
    bundle = tmp_path / "models.joblib"
    code = main(["train", "--data", str(historian), "--out", str(bundle),
                 "--output", str(tmp_path), "--log-level", "ERROR"])
    assert code == EXIT_OK
    assert bundle.is_file()
    manifest = json.loads(manifest_path(bundle).read_text(encoding="utf-8"))
    assert manifest["sha256"]
    assert manifest["columns"]
    printed = json.loads(capsys.readouterr().out.split("saved bundle")[0])
    assert printed["feature_rows"] > 0


@pytest.fixture(scope="module")
def trained_bundle(historian, tmp_path_factory):
    path = tmp_path_factory.mktemp("e2e-bundle") / "models.joblib"
    assert main(["train", "--data", str(historian), "--out", str(path),
                 "--output", str(path.parent), "--log-level", "ERROR"]) == EXIT_OK
    return path


def test_a_crashed_run_is_finished_by_resume(historian, trained_bundle, tmp_path, capsys):
    """Stop halfway, then pick the same stream up from the checkpoint."""
    common = ["--data", str(historian), "--models", str(trained_bundle), "--output", str(tmp_path),
              "--log-level", "ERROR", "--run-id", "run-e2e", "--checkpoint-every", "2000"]
    assert main(["run", *common, "--stop-after", "5000"]) == EXIT_OK
    first = json.loads(capsys.readouterr().out)
    assert first["stopped_early"] is True
    assert (tmp_path / "checkpoint.json").is_file()

    assert main(["resume", *common]) == EXIT_OK
    second = json.loads(capsys.readouterr().out)
    assert second["stopped_early"] is False
    assert second["events_consumed"] == 24 * FLEET["days"] * FLEET["machines"]
    assert second["run_id"] == "run-e2e"

    with open_database(tmp_path / "maintainiq.db") as db:
        run = MetadataStore(db).get_run("run-e2e")
    assert run["status"] == "completed"
    assert run["events_consumed"] == second["events_consumed"]


def test_the_metrics_subcommand_prints_what_the_endpoint_would_serve(historian, trained_bundle,
                                                                    tmp_path, capsys):
    assert main(["run", "--data", str(historian), "--models", str(trained_bundle),
                 "--output", str(tmp_path), "--log-level", "ERROR"]) == EXIT_OK
    capsys.readouterr()

    assert main(["metrics", "--output", str(tmp_path), "--log-level", "ERROR"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "# TYPE pme_rows_total gauge" in text
    assert 'pme_rows_total{table="runs"} 1' in text
    assert "pme_last_run_events_consumed" in text
    # Every HELP line is unique: a repeated one makes the exposition unparseable.
    helps = [line for line in text.splitlines() if line.startswith("# HELP")]
    assert len(helps) == len(set(helps))


def test_a_run_with_no_db_writes_no_database(historian, trained_bundle, tmp_path, capsys):
    assert main(["run", "--data", str(historian), "--models", str(trained_bundle),
                 "--output", str(tmp_path), "--log-level", "ERROR", "--no-db"]) == EXIT_OK
    capsys.readouterr()
    assert not (tmp_path / "maintainiq.db").exists()
    assert (tmp_path / "alerts.jsonl").is_file()
