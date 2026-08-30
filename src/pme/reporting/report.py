"""Run reporting: a markdown fleet/stream report and a four-panel chart.

Two of the four panels — watermark lag and the lateness histogram — describe the
*stream*, not the fleet: they are the diagnostics an operator needs to decide
whether the configured out-of-orderness and lateness budgets match reality.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: the daemon may run on a server with no display
import matplotlib.pyplot as plt

from ..core.events import ScoredRecord
from ..runtime.daemon import RunReport

_SEVERITY_ORDER = ("CRITICAL", "WARNING", "ANOMALY")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(none)_"
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def stream_chart(report: RunReport, latest: dict[str, ScoredRecord], heartbeats: list[dict[str, Any]],
                 path: Path) -> None:
    """Fleet risk + stream-health panels."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    risks = sorted((r.risk_prob for r in latest.values()), reverse=True)

    axes[0][0].hist(risks, bins=20, color="#4477aa", edgecolor="white")
    axes[0][0].set_title("Fleet failure-risk distribution (latest window)")
    axes[0][0].set_xlabel("P(failure <= 72h)")
    axes[0][0].set_ylabel("machines")

    top = sorted(latest.values(), key=lambda r: (-r.risk_prob, r.key))[:12][::-1]
    colors = ["#cc3311" if r.risk_prob > 0.5 else "#ee7733" if r.risk_prob > 0.2 else "#4477aa" for r in top]
    axes[0][1].barh([r.key for r in top], [r.risk_prob for r in top], color=colors)
    axes[0][1].set_title("Highest-risk machines")
    axes[0][1].set_xlabel("P(failure <= 72h)")

    beats = [h for h in heartbeats if h.get("watermark_lag") is not None]
    if beats:
        xs = [h["events_consumed"] for h in beats]
        axes[1][0].plot(xs, [h["watermark_lag"] for h in beats], color="#228833", label="watermark lag (h)")
        axes[1][0].plot(xs, [h["reorder_buffer"] for h in beats], color="#aa3377", label="reorder buffer")
        axes[1][0].legend(loc="upper right", fontsize=8)
    axes[1][0].set_title("Watermark lag & reorder-buffer depth")
    axes[1][0].set_xlabel("events consumed")

    buckets = report.engine_stats.get("lateness_histogram", [])
    if buckets and sum(b["count"] for b in buckets) > 0:
        labels = [f"<={b['upper_h']}h" if b["upper_h"] is not None else "more" for b in buckets]
        axes[1][1].bar(labels, [b["count"] for b in buckets], color="#ccbb44", edgecolor="white")
        axes[1][1].tick_params(axis="x", rotation=45, labelsize=8)
    else:
        axes[1][1].text(0.5, 0.5, "no events dropped as late", ha="center", va="center")
    axes[1][1].set_title("Dropped-late events by lateness")
    axes[1][1].set_ylabel("events")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_report(
    report: RunReport,
    metrics: dict[str, Any],
    latest: dict[str, ScoredRecord],
    path: Path,
) -> str:
    """Write the markdown run report and return its text."""
    stats = report.engine_stats
    severity_counts = {sev: sum(1 for a in report.alerts if a.severity == sev) for sev in _SEVERITY_ORDER}
    high = [r for r in latest.values() if r.risk_prob > 0.5]
    watch = [r for r in latest.values() if 0.2 < r.risk_prob <= 0.5]
    anomalies = [r for r in latest.values() if r.anomaly == 1]
    top = sorted(latest.values(), key=lambda r: (-r.risk_prob, r.key))[:10]

    lines: list[str] = [
        "# Predictive Maintenance — Stream Run Report",
        "",
        "## Model quality (machine-level holdout)",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["risk AUC", metrics.get("risk_auc")],
                ["RUL MAE (h)", metrics.get("rul_mae_h")],
                ["train / test machines", f"{metrics.get('train_machines')} / {metrics.get('test_machines')}"],
                ["feature rows", metrics.get("feature_rows")],
                ["window gate (min samples)", metrics.get("min_window_samples")],
            ],
        ),
        "",
        "Fitted on synthetic telemetry from this repository's wear simulator, not on plant data.",
        "",
        "## Stream health",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["events consumed", report.events_consumed],
                ["events released by watermark", stats["events_released"]],
                ["events dropped as too-late", stats["events_dropped_late"]],
                ["late-drop rate", stats["drop_rate"]],
                ["late events in side output", report.sink_summaries.get("late", {}).get("late_events", 0)],
                ["max observed lateness (h)", stats["lateness_max_h"]],
                ["mean watermark lag (h)", stats["watermark_lag_mean"]],
                ["max watermark lag (h)", stats["watermark_lag_max"]],
                ["feature windows emitted", report.features_emitted],
                ["records scored", report.records_scored],
                ["checkpoints written", report.checkpoints_written],
                ["wall seconds", report.wall_seconds],
            ],
        ),
        "",
        "## Alerts",
        "",
        f"Total **{len(report.alerts)}** "
        + " · ".join(f"{sev} **{severity_counts[sev]}**" for sev in _SEVERITY_ORDER),
        "",
        _markdown_table(
            ["machine_id", "event_time", "severity", "risk_prob", "rul_est_h", "reason"],
            [
                [a.key, a.event_time, a.severity, round(a.risk_prob, 4), round(a.rul_est_h, 1), a.reason]
                for a in sorted(report.alerts, key=lambda a: (-a.risk_prob, a.key))[:10]
            ],
        ),
        "",
        "## Fleet state at end of stream",
        "",
        (f"Machines scored: **{len(latest)}** · High risk (>50%): **{len(high)}** · "
         f"Watchlist (20-50%): **{len(watch)}** · Anomalies: **{len(anomalies)}**"),
        "",
        _markdown_table(
            ["machine_id", "event_time", "risk_prob", "rul_est_h", "anomaly"],
            [[r.key, r.event_time, round(r.risk_prob, 4), round(r.rul_est_h, 1), r.anomaly] for r in top],
        ),
        "",
        "## Maintenance schedule (next 7 days)",
        "",
        _markdown_table(
            ["machine_id", "day", "risk_prob", "rul_est_h", "expected_saving", "reason"],
            [[o.machine_id, o.day, o.risk_prob, o.rul_est_h, o.expected_saving, o.reason] for o in report.orders],
        ),
        "",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
