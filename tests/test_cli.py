"""The command line: argument placement, exit codes and the error envelope."""
from __future__ import annotations

import json

import pytest

from conftest import STREAM_FLEET
from pme.__main__ import build_parser, main
from pme.errors import EXIT_DATA, EXIT_INTEGRITY, EXIT_OK, EXIT_USAGE
from pme.sources.synthetic import _simulate


@pytest.fixture(scope="module")
def telemetry(tmp_path_factory):
    """A machine-major export: every hour of M000, then every hour of M001…"""
    path = tmp_path_factory.mktemp("cli") / "telemetry.csv"
    frame = _simulate(STREAM_FLEET["n_machines"], 24 * 12, seed=42)
    frame.sort_values(["machine_id", "hour"], kind="stable").to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, models):
    path = tmp_path_factory.mktemp("cli-models") / "models.joblib"
    models.save(path)
    return path


def test_version_is_available():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0


def test_common_options_come_after_the_subcommand():
    args = build_parser().parse_args(["run", "--data", "x.csv", "--models", "m.joblib", "--output", "out"])
    assert args.output == "out"
    assert args.command == "run"


def test_run_consumes_a_machine_major_file_without_dropping_it(telemetry, bundle, tmp_path, capsys):
    code = main(["run", "--data", str(telemetry), "--models", str(bundle), "--output", str(tmp_path)])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"]["events_dropped_late"] == 0
    assert payload["engine"]["drop_rate"] == 0.0
    assert payload["events_consumed"] == payload["engine"]["events_released"]


def test_file_order_on_a_machine_major_file_blows_the_drop_budget(telemetry, bundle, tmp_path, capsys):
    code = main(["run", "--data", str(telemetry), "--models", str(bundle), "--output", str(tmp_path),
                 "--order", "file"])
    assert code == EXIT_DATA
    error = json.loads(capsys.readouterr().err.strip().splitlines()[-1])["error"]
    assert error["code"] == "drop_rate_exceeded"
    assert error["details"]["drop_rate"] > 0.5


def test_a_missing_telemetry_file_is_a_data_error(bundle, tmp_path, capsys):
    code = main(["run", "--data", str(tmp_path / "nope.csv"), "--models", str(bundle), "--output", str(tmp_path)])
    assert code == EXIT_DATA
    assert json.loads(capsys.readouterr().err.strip().splitlines()[-1])["error"]["code"] == "path_rejected"


def test_an_unverifiable_bundle_is_an_integrity_error(telemetry, bundle, tmp_path, capsys):
    lonely = tmp_path / "unmanifested.joblib"
    lonely.write_bytes(bundle.read_bytes())
    code = main(["run", "--data", str(telemetry), "--models", str(lonely), "--output", str(tmp_path)])
    assert code == EXIT_INTEGRITY
    assert json.loads(capsys.readouterr().err.strip().splitlines()[-1])["error"]["code"] == "model_integrity"


def test_resume_without_a_checkpoint_says_so(telemetry, bundle, tmp_path, capsys):
    code = main(["resume", "--data", str(telemetry), "--models", str(bundle), "--output", str(tmp_path)])
    assert code == EXIT_USAGE
    assert "no checkpoint" in capsys.readouterr().err
