"""The model bundle — three estimators carried over unchanged from the validated engine.

* ``risk``    GradientBoostingClassifier — P(failure within RISK_HORIZON_H hours)
* ``rul``     GradientBoostingRegressor  — remaining useful life in hours
* ``anomaly`` IsolationForest            — novel sensor signatures the classifier never saw

Hyper-parameters, the machine-level holdout and the healthy-subset fit for the
anomaly detector are all preserved verbatim; only the *source* of the design
matrix changed (stream replay instead of a pandas re-scan). Scoring is exposed as
a pure array-in / arrays-out call so the streaming operator can micro-batch it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ..core.schema import FEATURE_NAMES, RUL_CAP_H

#: RUL rows at or above the cap mean "no failure within the observable horizon"
#: and are excluded from the regressor, exactly as in the baseline.
HEALTHY_RUL_H = 200.0


class MaintenanceModels:
    """Trained bundle plus the held-out metrics that justify it."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
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

        self.risk.fit(x[is_train], fail_within[is_train])
        if len(np.unique(fail_within[is_test])) > 1:
            auc = float(roc_auc_score(fail_within[is_test], self.risk.predict_proba(x[is_test])[:, 1]))
        else:
            auc = float("nan")

        tr_r = is_train & (rul_h < RUL_CAP_H)
        te_r = is_test & (rul_h < RUL_CAP_H)
        self.rul.fit(x[tr_r], rul_h[tr_r])
        mae = float(mean_absolute_error(rul_h[te_r], self.rul.predict(x[te_r]))) if te_r.sum() else float("nan")

        healthy = is_train & (rul_h > HEALTHY_RUL_H)
        self.anomaly.fit(x[healthy])

        self.metrics = {
            "risk_auc": round(auc, 4),
            "rul_mae_h": round(mae, 1),
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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> MaintenanceModels:
        return joblib.load(Path(path))
