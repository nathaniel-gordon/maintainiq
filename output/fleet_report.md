# Predictive Maintenance — Stream Run Report

## Model quality (machine-level holdout)

| metric | value | baseline (batch engine) |
| --- | --- | --- |
| risk AUC | 0.9993 | 0.9993 |
| RUL MAE (h) | 24.3 | 24.3 |
| train / test machines | 30 / 10 |  |
| feature rows | 85480 |  |

## Stream health

| metric | value |
| --- | --- |
| events consumed | 86400 |
| events released by watermark | 86058 |
| events dropped as too-late | 342 |
| late-drop rate | 0.003958 |
| max observed lateness (h) | 87 |
| mean watermark lag (h) | 6.0 |
| max watermark lag (h) | 6.0 |
| feature windows emitted | 85298 |
| records scored | 85298 |
| checkpoints written | 4 |
| wall seconds | 5.156 |

## Alerts

Total **604** CRITICAL **155** · WARNING **55** · ANOMALY **394**

| machine_id | event_time | severity | risk_prob | rul_est_h | reason |
| --- | --- | --- | --- | --- | --- |
| M022 | 781 | CRITICAL | 1.0 | 110.4 | P(fail<=72h)=1.00, RUL~110h |
| M015 | 1494 | CRITICAL | 0.9999 | 4.8 | P(fail<=72h)=1.00, RUL~5h |
| M033 | 1358 | CRITICAL | 0.9997 | 12.0 | P(fail<=72h)=1.00, RUL~12h |
| M039 | 1888 | CRITICAL | 0.9997 | 7.2 | P(fail<=72h)=1.00, RUL~7h |
| M021 | 1685 | CRITICAL | 0.9996 | 19.4 | P(fail<=72h)=1.00, RUL~19h |
| M001 | 896 | CRITICAL | 0.9996 | 18.2 | P(fail<=72h)=1.00, RUL~18h |
| M000 | 1362 | CRITICAL | 0.9996 | 15.6 | P(fail<=72h)=1.00, RUL~16h |
| M016 | 1764 | CRITICAL | 0.9995 | 7.6 | P(fail<=72h)=1.00, RUL~8h |
| M016 | 720 | CRITICAL | 0.9995 | 14.0 | P(fail<=72h)=1.00, RUL~14h |
| M008 | 1807 | CRITICAL | 0.9993 | 10.8 | P(fail<=72h)=1.00, RUL~11h |

## Fleet state at end of stream

Machines scored: **40** · High risk (>50%): **0** · Watchlist (20–50%): **1** · Anomalies: **5**

| machine_id | event_time | risk_prob | rul_est_h | anomaly |
| --- | --- | --- | --- | --- |
| M019 | 2159 | 0.2617 | 50.7 | 1 |
| M032 | 2159 | 0.0207 | 93.5 | 1 |
| M020 | 2159 | 0.0024 | 104.5 | 1 |
| M018 | 2159 | 0.0002 | 212.5 | 1 |
| M025 | 2159 | 0.0001 | 255.2 | 1 |
| M022 | 2159 | 0.0001 | 481.4 | 0 |
| M023 | 2159 | 0.0001 | 491.0 | 0 |
| M031 | 2159 | 0.0001 | 482.5 | 0 |
| M033 | 2159 | 0.0001 | 465.6 | 0 |
| M038 | 2159 | 0.0001 | 486.2 | 0 |

## Maintenance schedule (next 7 days)

| machine_id | day | risk_prob | rul_est_h | expected_saving | reason |
| --- | --- | --- | --- | --- | --- |
| M019 | 1 | 0.2617 | 50.7 | 1272.95 | risk 26% makes service cost-positive; anomalous sensor signature |
| M032 | 1 | 0.0207 | 93.5 | 0.0 | anomalous sensor signature |
| M020 | 1 | 0.0024 | 104.5 | 0.0 | anomalous sensor signature |
| M018 | 2 | 0.0002 | 212.5 | 0.0 | anomalous sensor signature |
| M025 | 2 | 0.0001 | 255.2 | 0.0 | anomalous sensor signature |
