"""Model bundle integrity: nothing is unpickled without a matching manifest."""
from __future__ import annotations

import json

import pytest

from pme.errors import ModelIntegrityError
from pme.learning.bundle import MaintenanceModels, manifest_path


@pytest.fixture(scope="module")
def saved(tmp_path_factory, models):
    path = tmp_path_factory.mktemp("bundle") / "models.joblib"
    models.save(path)
    return path


def test_save_writes_a_manifest_beside_the_bundle(saved):
    side = manifest_path(saved)
    assert side.is_file()
    manifest = json.loads(side.read_text(encoding="utf-8"))
    assert manifest["bundle"] == saved.name
    assert len(manifest["sha256"]) == 64
    assert manifest["min_window_samples"] > 0


def test_a_verified_bundle_loads(saved):
    loaded = MaintenanceModels.load(saved)
    assert loaded.columns
    assert loaded.min_window_samples > 0


def test_a_bundle_without_a_manifest_is_refused(saved, tmp_path):
    lonely = tmp_path / "copy.joblib"
    lonely.write_bytes(saved.read_bytes())
    with pytest.raises(ModelIntegrityError, match="manifest"):
        MaintenanceModels.load(lonely)


def test_a_modified_bundle_is_refused(saved, tmp_path):
    target = tmp_path / "models.joblib"
    target.write_bytes(saved.read_bytes() + b"\x00")
    manifest_path(target).write_bytes(manifest_path(saved).read_bytes())
    with pytest.raises(ModelIntegrityError, match="digest"):
        MaintenanceModels.load(target)


def test_a_missing_bundle_is_refused(tmp_path):
    with pytest.raises(ModelIntegrityError, match="does not exist"):
        MaintenanceModels.load(tmp_path / "nope.joblib")


def test_verification_can_be_waived_explicitly(saved, tmp_path):
    lonely = tmp_path / "copy.joblib"
    lonely.write_bytes(saved.read_bytes())
    assert MaintenanceModels.load(lonely, verify=False).columns
