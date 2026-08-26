# MaintainIQ — Predictive Maintenance for Industrial Fleets

> Predict failures before they happen. MaintainIQ delivers end-to-end predictive maintenance: failure-risk scoring, remaining-useful-life (RUL) estimation, anomaly early-warning, and a cost-optimized maintenance scheduler — one pipeline from raw telemetry to a crew-ready service plan.

## What MaintainIQ Does

- **Failure risk scoring** — gradient-boosting classifier on sensor telemetry with calibrated probabilities
- **RUL estimation** — regression model estimating remaining useful life per asset
- **Anomaly early-warning** — IsolationForest flags assets drifting before failure threshold
- **Cost-optimized scheduling** — balances maintenance cost vs downtime risk per asset
- **Fleet dashboard** — ranked asset list with risk scores, RUL estimates, and recommended actions

## Architecture

```
Sensor Telemetry (asset x timestamp x readings)
    └─> FeatureBuilder      (rolling stats, degradation trends)
    └─> RiskClassifier      (failure probability, calibrated)
    └─> RULEstimator        (remaining-useful-life regression)
    └─> AnomalyDetector     (IsolationForest early-warning)
    └─> MaintenanceScheduler(cost-optimized service plan)
    └─> FleetReport         (ranked asset dashboard)
```

## Quickstart

```bash
python -m pme demo           # full pipeline on synthetic fleet telemetry
python -m pme score --data telemetry.csv
```

## Test

```bash
python tests/test_smoke.py
```

---

## 👤 Author & Contact

- **Author**: Nathaniel Gordon
- **Role**: Senior AI & Machine Learning Engineer
- **GitHub**: [github.com/nathaniel-gordon](https://github.com/nathaniel-gordon)
- **Portfolio / Upwork**: [upwork.com/freelancers/~015fe5a704f8943797](https://www.upwork.com/freelancers/~015fe5a704f8943797)
- **Email**: nathanielgordon346@gmail.com
- **Location**: Tallahassee, FL, USA
