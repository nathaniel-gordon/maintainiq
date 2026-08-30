"""The model bundle: three estimators plus the contract they were trained under.

* ``risk``    GradientBoostingClassifier — P(failure within RISK_HORIZON_H hours)
* ``rul``     GradientBoostingRegressor  — remaining useful life in hours
* ``anomaly`` IsolationForest            — novel sensor signatures the classifier never saw

Whole machines are held out, never rows, so no machine contributes to both train
and test. Scoring is a pure array-in / arrays-out call so the streaming operator
can micro-batch it.

**On loading a bundle.** ``joblib`` is pickle, and unpickling runs code. A bundle
is therefore only ever loaded alongside the manifest written next to it at save
time, and the digest in that manifest must match the file. That catches
truncation and after-the-fact edits; it does *not* make an attacker-supplied
(bundle, manifest) pair safe, because nothing signs the manifest. Treat a .joblib
from anywhere you would not run a script from as executable code, because it is.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ..core.schema import FEATURE_NAMES, MIN_WINDOW_SAMPLES, RUL_CAP_H
from ..errors import DataQualityError, ModelIntegrityError
from ..logging_setup import get_logger
from ..paths import normalize

log = get_logger(__name__)

#: Suffix of the manifest written beside every saved bundle.
MANIFEST_SUFFIX = ".manifest.json"

#: Version of the manifest layout.
MANIFEST_VERSION = 1

#: Rows above this RUL are "no failure in sight" and are the only ones the anomaly
#: detector is fitted on, so novelty means "unlike a healthy machine".
HEALTHY_RUL_H = 200.0


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: str | Path) -> Path:
    """Where the manifest for ``path`` lives."""
    resolved = Path(path)
    return resolved.with_name(resolved.name + MANIFEST_SUFFIX)


class MaintenanceModels:
    """Trained bundle plus the held-out metrics that justify it."""

    def __init__(self, seed: int = 42, min_window_samples: int = MIN_WINDOW_SAMPLES) -> None:
        self.seed = seed
        #: Emission gate the training features were built under. The daemon
        #: refuses to serve a bundle whose gate differs from its own, which is
        #: the one train/serve skew this design cannot rule out structurally.
        self.min_window_samples = min_window_samples
        self.risk = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3)
        self.rul = GradientBoostingRegressor(random_state=seed, n_estimators=150, max_depth=3)
        self.anomaly = IsolationForest(random_state=seed, contamination=0.03, n_estimators=200)
        self.columns: tuple[str, ...] = FEATURE_NAMES
        self.metrics: dict[str, Any] = {}

    def fit(
        self,
        x: np.ndarray,
        machine_ids: np.ndarray,
        rul_h: np.ndarray,
        fail_within: np.ndarray,
    ) -> dict[str, Any]:
        """Fit all three estimators on a machine-level holdout split.

        Whole machines are held out (never rows), so no machine contributes to both
        train and test — the leakage discipline the baseline established.
        """
        machines = np.array(sorted(set(machine_ids.tolist())))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(machines)
        cut = max(1, int(len(machines) * 0.75))
        train_m, test_m = set(machines[:cut].tolist()), set(machines[cut:].tolist())
        is_train = np.array([m in train_m for m in machine_ids])
        is_test = ~is_train

        if len(np.unique(fail_within[is_train])) < 2:
            raise DataQualityError(
                "the training set contains no failure events, so the risk classifier cannot be fitted",
                train_rows=int(is_train.sum()), train_machines=len(train_m),
            )
        self.risk.fit(x[is_train], fail_within[is_train])
        # A holdout that contains only one class cannot produce an AUC. Reporting
        # a bare NaN put a token strict JSON parsers reject into the metrics file
        # and read like a result; None plus a note says what actually happened.
        notes: list[str] = []
        auc: float | None = None
        if len(np.unique(fail_within[is_test])) > 1:
            auc = round(float(roc_auc_score(fail_within[is_test], self.risk.predict_proba(x[is_test])[:, 1])), 4)
        else:
            notes.append("risk AUC not computed: the held-out machines contain a single class")

        tr_r = is_train & (rul_h < RUL_CAP_H)
        te_r = is_test & (rul_h < RUL_CAP_H)
        self.rul.fit(x[tr_r], rul_h[tr_r])
        mae: float | None = None
        if te_r.sum():
            mae = round(float(mean_absolute_error(rul_h[te_r], self.rul.predict(x[te_r]))), 1)
        else:
            notes.append("RUL MAE not computed: no held-out row is below the RUL cap")

        healthy = is_train & (rul_h > HEALTHY_RUL_H)
        self.anomaly.fit(x[healthy])

        self.metrics = {
            "risk_auc": auc,
            "rul_mae_h": mae,
            "holdout_notes": notes,
            "train_machines": len(train_m),
            "test_machines": len(test_m),
            "train_rows": int(is_train.sum()),
            "test_rows": int(is_test.sum()),
            "anomaly_fit_rows": int(healthy.sum()),
        }
        return self.metrics

    def score_batch(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score a micro-batch. Row-independent, so batching never changes a result."""
        risk = self.risk.predict_proba(x)[:, 1]
        rul = np.clip(self.rul.predict(x), 0.0, RUL_CAP_H)
        anomaly = (self.anomaly.predict(x) == -1).astype(int)
        return risk, rul, anomaly

    def save(self, path: str | Path) -> Path:
        """Write the bundle and its manifest. Returns the manifest path."""
        target = normalize(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "bundle": target.name,
            "sha256": sha256_of(target),
            "bytes": target.stat().st_size,
            "seed": self.seed,
            "min_window_samples": self.min_window_samples,
            "columns": list(self.columns),
            "metrics": self.metrics,
        }
        side = manifest_path(target)
        side.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return side

    @staticmethod
    def load(path: str | Path, verify: bool = True) -> MaintenanceModels:
        """Load a bundle, refusing one whose manifest is missing or disagrees.

        ``verify=False`` exists for the operator who knowingly loads a bundle they
        produced before manifests existed. It is logged at warning level every
        time, because it means unpickling an unchecked file.
        """
        target = normalize(path)
        if not target.is_file():
            raise ModelIntegrityError("model bundle does not exist", path=str(target))
        if verify:
            side = manifest_path(target)
            if not side.is_file():
                raise ModelIntegrityError(
                    "model bundle has no manifest; loading it would unpickle an unchecked file",
                    path=str(target), expected_manifest=str(side),
                )
            try:
                manifest = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelIntegrityError("model manifest is unreadable", path=str(side),
                                          detail=str(exc)) from exc
            actual = sha256_of(target)
            if manifest.get("sha256") != actual:
                raise ModelIntegrityError("model bundle does not match its manifest digest",
                                          path=str(target), expected=str(manifest.get("sha256"))[:16],
                                          actual=actual[:16])
        else:
            log.warning("loading a model bundle without verifying it", path=str(target))
        bundle: MaintenanceModels = joblib.load(target)
        if not isinstance(bundle, MaintenanceModels):
            raise ModelIntegrityError("file did not contain a model bundle", path=str(target),
                                      got=type(bundle).__name__)
        if not hasattr(bundle, "min_window_samples"):
            bundle.min_window_samples = MIN_WINDOW_SAMPLES
        return bundle
