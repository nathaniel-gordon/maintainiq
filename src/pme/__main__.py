"""Daemon entry point.

Subcommands
-----------
``demo``    seeded end-to-end run: historian backfill -> training -> live stream ->
            mid-stream checkpoint/restore proof -> report.
``train``   fit the model bundle from a historian CSV by replaying it as a stream.
``run``     start the daemon against a telemetry CSV and consume it until drained.
``resume``  restore the daemon from its checkpoint and finish the same stream.

The daemon always terminates on its own: sources are bounded, the drain phase
closes every open window, and ``--max-runtime`` is a hard guard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .learning.bundle import MaintenanceModels
from .learning.trainer import train_from_stream
from .reporting.report import run_report, stream_chart
from .runtime.config import OutputPaths, StreamConfig
from .runtime.daemon import MaintenanceDaemon
from .sources.csvfile import CsvTelemetrySource
from .sources.synthetic import SyntheticFleetSource

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "output"

#: Metrics recorded from the pre-refactor batch engine on the same seeded fleet
#: (40 machines x 90 days, seed 42). The demo recomputes today's numbers at
#: runtime and prints both so the refactor's effect on quality is visible.
BASELINE_METRICS: dict[str, Any] = {"risk_auc": 0.9993, "rul_mae_h": 24.3, "engine": "batch pandas.rolling"}

#: Historian backfill: complete and perfectly ordered, so training sees no gaps.
BACKFILL_CONFIG = StreamConfig(max_out_of_orderness_h=0, allowed_lateness_h=0)


def _live_config(args: argparse.Namespace, checkpoint_every: int, stop_after: int | None = None) -> StreamConfig:
    return StreamConfig(
        max_out_of_orderness_h=args.max_out_of_orderness,
        allowed_lateness_h=args.allowed_lateness,
        scoring_batch_size=args.batch_size,
        alert_cooldown_h=args.cooldown,
        checkpoint_every=checkpoint_every,
        heartbeat_every=args.heartbeat_every,
        stop_after_events=stop_after,
        max_runtime_s=args.max_runtime,
    )


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    machines, hours = args.machines, 24 * args.days

    print(f"[1/6] Historian backfill: {machines} machines x {args.days} days, in event-time order")
    backfill = SyntheticFleetSource(machines, hours, seed=args.seed)
    print(f"      {len(backfill):,} events, {int(backfill.history()['failure'].sum())} failure events")

    print("[2/6] Training by replaying the backfill through the streaming operators")
    trained = train_from_stream(backfill, seed=args.seed, config=BACKFILL_CONFIG)
    metrics = trained.metrics
    print(f"      risk AUC   {metrics['risk_auc']}   (batch baseline {BASELINE_METRICS['risk_auc']})")
    print(f"      RUL MAE    {metrics['rul_mae_h']} h (batch baseline {BASELINE_METRICS['rul_mae_h']} h)")
    print(f"      {metrics['feature_rows']:,} windowed feature rows x {metrics['feature_columns']} columns")
    bundle_path = out / "models.joblib"
    trained.models.save(bundle_path)

    print(f"[3/6] Live stream: same telemetry, jittered up to {args.max_out_of_orderness}h, "
          f"{args.late_fraction:.1%} stragglers")
    live_paths = OutputPaths.under(out)
    live = SyntheticFleetSource(
        machines, hours, seed=args.seed,
        out_of_order_delay=args.max_out_of_orderness,
        late_fraction=args.late_fraction,
        delivery_seed=args.seed + 1,
    )
    checkpoint_every = max(1, len(live) // 4)
    daemon = MaintenanceDaemon(live, trained.models, _live_config(args, checkpoint_every), live_paths)
    report = daemon.run()
    stats = report.engine_stats
    print(f"      consumed {report.events_consumed:,} | released {stats['events_released']:,} | "
          f"dropped-late {stats['events_dropped_late']:,} ({stats['drop_rate']:.4%})")
    print(f"      watermark lag mean {stats['watermark_lag_mean']}h max {stats['watermark_lag_max']}h | "
          f"max lateness {stats['lateness_max_h']}h")
    print(f"      alerts {len(report.alerts)} ({daemon.alert_sink.by_severity}) | "
          f"service orders {len(report.orders)} | {report.wall_seconds}s")

    print("[4/6] Checkpoint/restore proof: crash the same stream mid-flight and resume")
    replay_paths = OutputPaths.under(out, prefix="replay_")
    cut = max(1, int(len(live) * 0.55))
    crashed = SyntheticFleetSource(
        machines, hours, seed=args.seed, out_of_order_delay=args.max_out_of_orderness,
        late_fraction=args.late_fraction, delivery_seed=args.seed + 1,
    )
    part1 = MaintenanceDaemon(crashed, trained.models, _live_config(args, checkpoint_every, cut), replay_paths)
    first = part1.run()
    print(f"      run stopped after {first.events_consumed:,} events; checkpoint written")

    resumed_source = SyntheticFleetSource(
        machines, hours, seed=args.seed, out_of_order_delay=args.max_out_of_orderness,
        late_fraction=args.late_fraction, delivery_seed=args.seed + 1,
    )
    part2 = MaintenanceDaemon(resumed_source, trained.models, _live_config(args, checkpoint_every), replay_paths)
    if not part2.restore_latest():
        raise RuntimeError("expected a checkpoint to resume from")
    second = part2.run()
    identical = second.alert_bytes == report.alert_bytes
    print(f"      resumed at event {first.events_consumed:,}, finished {second.events_consumed:,}")
    print(f"      alert feed byte-identical to the uninterrupted run: {identical} "
          f"({len(report.alert_bytes):,} bytes, {len(report.alerts)} alerts)")
    if not identical:
        raise RuntimeError("checkpoint restore did not reproduce the uninterrupted alert feed")

    print("[5/6] Fleet-health tumbling windows + stream metrics")
    ms = daemon.metrics_sink.summary()
    print(f"      {ms['tumbling_windows']} weekly windows, {ms['windows_with_failures']} contained a failure")

    print("[6/6] Writing report + chart")
    chart_path = out / "fleet_health.png"
    report_path = out / "fleet_report.md"
    stream_chart(report, daemon.schedule_sink.latest, daemon.metrics_sink.heartbeats, chart_path)
    run_report(report, metrics, daemon.schedule_sink.latest, report_path, baseline=BASELINE_METRICS)

    print()
    print("Top service orders:")
    for order in report.orders[:5]:
        print(f"  day {order.day}  {order.machine_id}  risk={order.risk_prob:.3f}  "
              f"RUL~{order.rul_est_h:.0f}h  saving=${order.expected_saving:,.0f}  ({order.reason})")
    print(f"\nArtifacts -> {report_path}, {chart_path}, {live_paths.alerts}, "
          f"{live_paths.schedule}, {live_paths.metrics}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    source = CsvTelemetrySource(args.data)
    trained = train_from_stream(source, seed=args.seed, config=BACKFILL_CONFIG)
    print(json.dumps(trained.metrics, indent=2, sort_keys=True))
    trained.models.save(args.out)
    print(f"saved bundle -> {args.out}")
    return 0


def _daemon_from_args(args: argparse.Namespace, checkpoint_every: int) -> MaintenanceDaemon:
    source = CsvTelemetrySource(
        args.data,
        out_of_order_delay=args.max_out_of_orderness if args.jitter else 0,
        late_fraction=args.late_fraction if args.jitter else 0.0,
        delivery_seed=args.seed,
    )
    models = MaintenanceModels.load(args.models)
    paths = OutputPaths.under(args.output)
    return MaintenanceDaemon(source, models, _live_config(args, checkpoint_every, args.stop_after), paths)


def _print_run(report: Any) -> None:
    print(json.dumps(
        {
            "events_consumed": report.events_consumed,
            "features_emitted": report.features_emitted,
            "records_scored": report.records_scored,
            "alerts": len(report.alerts),
            "service_orders": len(report.orders),
            "stopped_early": report.stopped_early,
            "checkpoints_written": report.checkpoints_written,
            "wall_seconds": report.wall_seconds,
            "engine": report.engine_stats,
        },
        indent=2,
        sort_keys=True,
    ))


def cmd_run(args: argparse.Namespace) -> int:
    daemon = _daemon_from_args(args, args.checkpoint_every)
    _print_run(daemon.run())
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    daemon = _daemon_from_args(args, args.checkpoint_every)
    if not daemon.restore_latest():
        print(f"no checkpoint at {daemon.checkpoints.path}", file=sys.stderr)
        return 2
    print(f"restored at event {daemon.events_consumed:,}; continuing", file=sys.stderr)
    _print_run(daemon.run())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pme", description="Predictive maintenance streaming daemon")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-out-of-orderness", type=int, default=6,
                        help="watermark budget in event-time hours")
    parser.add_argument("--allowed-lateness", type=int, default=2,
                        help="extra hours a window stays open past its watermark")
    parser.add_argument("--batch-size", type=int, default=512, help="scoring micro-batch size")
    parser.add_argument("--cooldown", type=int, default=24, help="alert cooldown in event-time hours")
    parser.add_argument("--heartbeat-every", type=int, default=5000)
    parser.add_argument("--max-runtime", type=float, default=900.0)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="seeded end-to-end run")
    demo.add_argument("--machines", type=int, default=40)
    demo.add_argument("--days", type=int, default=90)
    demo.add_argument("--late-fraction", type=float, default=0.004)
    demo.set_defaults(func=cmd_demo)

    train = sub.add_parser("train", help="fit the bundle from a historian CSV")
    train.add_argument("--data", required=True)
    train.add_argument("--out", default=str(DEFAULT_OUTPUT / "models.joblib"))
    train.set_defaults(func=cmd_train)

    for name, func, help_text in (("run", cmd_run, "consume a telemetry CSV until drained"),
                                  ("resume", cmd_resume, "restore from checkpoint and finish the stream")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--models", required=True)
        cmd.add_argument("--checkpoint-every", type=int, default=0)
        cmd.add_argument("--stop-after", type=int, default=None,
                         help="stop after N events (checkpointing first) to simulate a crash")
        cmd.add_argument("--jitter", action="store_true", help="re-deliver rows out of order to exercise watermarks")
        cmd.add_argument("--late-fraction", type=float, default=0.0)
        cmd.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
