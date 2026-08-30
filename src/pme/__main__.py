"""Daemon entry point.

Subcommands
-----------
``demo``    seeded end-to-end run: historian backfill -> training -> live stream ->
            mid-stream checkpoint/restore proof -> report.
``train``   fit the model bundle from a historian CSV by replaying it as a stream.
``run``     start the daemon against a telemetry CSV and consume it until drained.
``resume``  restore the daemon from its checkpoint and finish the same stream.
``serve``   expose health, metrics and the run history over HTTP.
``metrics`` print the Prometheus exposition once and exit.

Common options are attached to every subcommand rather than to the top-level
parser, so ``pme run --data x --output y`` works instead of failing with
"unrecognized arguments".

Every deliberate failure is a :class:`~pme.errors.PmeError`, printed as one JSON
object on stderr and mapped onto a process exit status an operator can branch on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core.schema import MIN_WINDOW_SAMPLES
from .errors import EXIT_INTERNAL, EXIT_OK, PmeError
from .learning.bundle import MaintenanceModels
from .learning.trainer import train_from_stream
from .logging_setup import configure_logging, correlation_scope, get_logger, log_level_from_env
from .paths import ensure_directory
from .reporting.report import run_report, stream_chart
from .runtime.config import OutputPaths, StreamConfig
from .runtime.daemon import MaintenanceDaemon
from .sources.csvfile import CsvTelemetrySource
from .sources.synthetic import SyntheticFleetSource
from .store.db import open_database
from .store.repository import MetadataStore

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "output"

#: Historian backfill: complete and perfectly ordered, so training sees no gaps.
BACKFILL_CONFIG = StreamConfig(max_out_of_orderness_h=0, allowed_lateness_h=0)


def _live_config(args: argparse.Namespace, checkpoint_every: int, stop_after: int | None = None) -> StreamConfig:
    return StreamConfig(
        max_out_of_orderness_h=args.max_out_of_orderness,
        allowed_lateness_h=args.allowed_lateness,
        min_window_samples=args.min_window_samples,
        scoring_batch_size=args.batch_size,
        alert_cooldown_h=args.cooldown,
        max_drop_rate=args.max_drop_rate,
        checkpoint_every=checkpoint_every,
        heartbeat_every=args.heartbeat_every,
        stop_after_events=stop_after,
        max_runtime_s=args.max_runtime,
    )


def _store(args: argparse.Namespace, paths: OutputPaths) -> MetadataStore | None:
    if getattr(args, "no_db", False):
        return None
    return MetadataStore(open_database(paths.database))


def cmd_demo(args: argparse.Namespace) -> int:
    out = ensure_directory(args.output)
    machines, hours = args.machines, 24 * args.days

    print(f"[1/6] Historian backfill: {machines} machines x {args.days} days, in event-time order")
    backfill = SyntheticFleetSource(machines, hours, seed=args.seed)
    print(f"      {len(backfill):,} events, {int(backfill.history()['failure'].sum())} failure events")

    print("[2/6] Training by replaying the backfill through the streaming operators")
    trained = train_from_stream(backfill, seed=args.seed, config=BACKFILL_CONFIG)
    metrics = trained.metrics
    print(f"      risk AUC {metrics['risk_auc']} | RUL MAE {metrics['rul_mae_h']} h "
          f"(synthetic telemetry, machine-level holdout)")
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
    store = _store(args, live_paths)
    daemon = MaintenanceDaemon(live, trained.models, _live_config(args, checkpoint_every), live_paths, store=store)
    report = daemon.run()
    alert_bytes = live_paths.alerts.read_bytes()
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
    part1 = MaintenanceDaemon(crashed, trained.models, _live_config(args, checkpoint_every, cut), replay_paths,
                              store=_store(args, replay_paths))
    first = part1.run()
    print(f"      run stopped after {first.events_consumed:,} events; checkpoint written")

    resumed_source = SyntheticFleetSource(
        machines, hours, seed=args.seed, out_of_order_delay=args.max_out_of_orderness,
        late_fraction=args.late_fraction, delivery_seed=args.seed + 1,
    )
    part2 = MaintenanceDaemon(resumed_source, trained.models, _live_config(args, checkpoint_every), replay_paths,
                              store=_store(args, replay_paths))
    if not part2.restore_latest():
        raise RuntimeError("expected a checkpoint to resume from")
    second = part2.run()
    identical = replay_paths.alerts.read_bytes() == alert_bytes
    print(f"      resumed at event {first.events_consumed:,}, finished {second.events_consumed:,}")
    print(f"      alert feed byte-identical to the uninterrupted run: {identical} "
          f"({len(alert_bytes):,} bytes, {len(report.alerts)} alerts)")
    if not identical:
        raise RuntimeError("checkpoint restore did not reproduce the uninterrupted alert feed")

    print("[5/6] Fleet-health tumbling windows + stream metrics")
    ms = daemon.metrics_sink.summary()
    print(f"      {ms['tumbling_windows']} weekly windows, {ms['windows_with_failures']} contained a failure")

    print("[6/6] Writing report + chart")
    chart_path = out / "fleet_health.png"
    report_path = out / "fleet_report.md"
    stream_chart(report, daemon.schedule_sink.latest, daemon.metrics_sink.heartbeats, chart_path)
    run_report(report, metrics, daemon.schedule_sink.latest, report_path)

    print()
    print("Top service orders:")
    for order in report.orders[:5]:
        print(f"  day {order.day}  {order.machine_id}  risk={order.risk_prob:.3f}  "
              f"RUL~{order.rul_est_h:.0f}h  saving=${order.expected_saving:,.0f}  ({order.reason})")
    print(f"\nArtifacts -> {report_path}, {chart_path}, {live_paths.alerts}, "
          f"{live_paths.schedule}, {live_paths.metrics}, {live_paths.late}")
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    source = CsvTelemetrySource(args.data, order=args.order, duplicate_policy=args.on_duplicate)
    _report_validation(source)
    trained = train_from_stream(source, seed=args.seed, config=BACKFILL_CONFIG)
    print(json.dumps(trained.metrics, indent=2, sort_keys=True, allow_nan=False))
    trained.models.save(args.out)
    print(f"saved bundle -> {args.out}")
    return EXIT_OK


def _report_validation(source: CsvTelemetrySource) -> None:
    summary = source.validation_summary()
    log.info("telemetry validated", source=source.name, **summary)
    if summary["rejected"]:
        print(f"warning: {summary['rejected']} of {summary['rows_in']} rows quarantined "
              f"{summary['reject_codes']}", file=sys.stderr)


def _daemon_from_args(args: argparse.Namespace, checkpoint_every: int) -> MaintenanceDaemon:
    source = CsvTelemetrySource(
        args.data,
        order=args.order,
        duplicate_policy=args.on_duplicate,
        out_of_order_delay=args.jitter,
        late_fraction=args.late_fraction,
        delivery_seed=args.seed,
    )
    _report_validation(source)
    models = MaintenanceModels.load(args.models)
    paths = OutputPaths.under(ensure_directory(args.output))
    store = _store(args, paths)
    daemon = MaintenanceDaemon(
        source, models, _live_config(args, checkpoint_every, args.stop_after), paths,
        store=store, run_id=args.run_id,
    )
    if store is not None and source.rejected_rows:
        store.quarantine_rows(source.name, [r.to_dict() for r in source.rejected_rows])
    daemon.install_signal_handlers()
    return daemon


def cmd_run(args: argparse.Namespace) -> int:
    daemon = _daemon_from_args(args, args.checkpoint_every)
    print(json.dumps(daemon.run().to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return EXIT_OK


def cmd_resume(args: argparse.Namespace) -> int:
    daemon = _daemon_from_args(args, args.checkpoint_every)
    if not daemon.restore_latest():
        print(f"no checkpoint at {daemon.checkpoints.path}", file=sys.stderr)
        return 2
    print(f"restored at event {daemon.events_consumed:,}; continuing", file=sys.stderr)
    print(json.dumps(daemon.run().to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    from .service.app import serve  # imported lazily: the CLI does not need a socket

    paths = OutputPaths.under(ensure_directory(args.output))
    return serve(paths, host=args.host, port=args.port)


def cmd_metrics(args: argparse.Namespace) -> int:
    """Print what ``/metrics`` would serve, without binding a socket.

    Useful from a cron job or a container shell. Only the database-derived gauges
    have values here: the request counter and the latency histogram belong to the
    serving process, and this one has answered no requests, so they render their
    headers and no samples.
    """
    from .service.app import Api  # lazy for the same reason as serve

    paths = OutputPaths.under(ensure_directory(args.output))
    store = MetadataStore(open_database(paths.database))
    try:
        body = Api(store).dispatch("GET", "/metrics", {}, None, "cli")[1]
        print(body if isinstance(body, str) else json.dumps(body), end="")
    finally:
        store.db.close()
    return EXIT_OK


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log-level", default=log_level_from_env(),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-format", default="plain", choices=["json", "plain"])
    parser.add_argument("--no-db", action="store_true", help="skip the sqlite metadata store")


def _add_stream_knobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-out-of-orderness", type=int, default=6,
                        help="watermark budget in event-time hours")
    parser.add_argument("--allowed-lateness", type=int, default=2,
                        help="extra hours a window stays open past its watermark")
    parser.add_argument("--min-window-samples", type=int, default=MIN_WINDOW_SAMPLES,
                        help="samples a window needs before it emits features")
    parser.add_argument("--batch-size", type=int, default=512, help="scoring micro-batch size")
    parser.add_argument("--cooldown", type=int, default=24, help="alert cooldown in event-time hours")
    parser.add_argument("--max-drop-rate", type=float, default=0.01,
                        help="fail the run above this share of events dropped as too-late")
    parser.add_argument("--heartbeat-every", type=int, default=5000)
    parser.add_argument("--max-runtime", type=float, default=900.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pme", description="Predictive maintenance streaming daemon")
    parser.add_argument("--version", action="version", version=f"pme {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="seeded end-to-end run")
    _add_common(demo)
    _add_stream_knobs(demo)
    demo.add_argument("--machines", type=int, default=40)
    demo.add_argument("--days", type=int, default=90)
    demo.add_argument("--late-fraction", type=float, default=0.004)
    demo.set_defaults(func=cmd_demo)

    train = sub.add_parser("train", help="fit the bundle from a historian CSV")
    _add_common(train)
    train.add_argument("--data", required=True)
    train.add_argument("--out", default=str(DEFAULT_OUTPUT / "models.joblib"))
    train.add_argument("--order", default="event_time", choices=["event_time", "file"])
    train.add_argument("--on-duplicate", default="last", choices=["last", "first", "reject"])
    train.set_defaults(func=cmd_train)

    for name, func, help_text in (("run", cmd_run, "consume a telemetry CSV until drained"),
                                  ("resume", cmd_resume, "restore from checkpoint and finish the stream")):
        cmd = sub.add_parser(name, help=help_text)
        _add_common(cmd)
        _add_stream_knobs(cmd)
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--models", required=True)
        cmd.add_argument("--run-id", default=None, help="reuse a run id so results group in the database")
        cmd.add_argument("--order", default="event_time", choices=["event_time", "file"],
                         help="delivery order: sorted by event time, or the file's own row order")
        cmd.add_argument("--on-duplicate", default="last", choices=["last", "first", "reject"])
        cmd.add_argument("--checkpoint-every", type=int, default=0)
        cmd.add_argument("--stop-after", type=int, default=None,
                         help="stop after N events (checkpointing first) to simulate a crash")
        cmd.add_argument("--jitter", type=int, default=0, metavar="HOURS",
                         help="re-deliver rows up to N hours out of order to exercise watermarks")
        cmd.add_argument("--late-fraction", type=float, default=0.0)
        cmd.set_defaults(func=func)

    serve = sub.add_parser("serve", help="serve health, metrics and run history over HTTP")
    _add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=cmd_serve)

    metrics = sub.add_parser("metrics", help="print the Prometheus exposition without serving")
    _add_common(metrics)
    metrics.set_defaults(func=cmd_metrics)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level, fmt=args.log_format)
    with correlation_scope() as cid:
        try:
            return int(args.func(args))
        except PmeError as exc:
            log.error("command failed", code=exc.code, reason=exc.message, **exc.details)
            print(json.dumps({"error": {**exc.to_dict(), "correlation_id": cid}}, sort_keys=True, default=str),
                  file=sys.stderr)
            return exc.exit_code
        except KeyboardInterrupt:  # pragma: no cover - interactive
            print("interrupted", file=sys.stderr)
            return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
