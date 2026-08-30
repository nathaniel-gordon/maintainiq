"""Properties the streaming core must hold for *any* delivery order.

The per-module tests pin behaviour on hand-written examples. These check the laws
the design rests on, over a few hundred generated delivery orders each. There is
no hypothesis dependency: a seeded ``random.Random`` walks the space, and the
seeds are fixed so a failure is reproducible from the test name alone.

Three laws matter here:

* **Conservation.** Every event pushed is released, dropped as too-late, or still
  sitting in the reorder buffer. Nothing evaporates.
* **Order independence.** Any delivery order that keeps every event inside the
  out-of-orderness budget produces exactly the output of a perfectly ordered
  stream. This is the reason the watermark exists.
* **Restore equivalence.** Cutting a run at an arbitrary point and resuming from
  its checkpoint produces the byte-identical feeds of the uninterrupted run.
"""
from __future__ import annotations

import random

import pytest

from conftest import make_daemon, make_source
from pme.core.assigners import SlidingWindows, TumblingWindows
from pme.core.engine import LATENESS_BUCKETS, WindowingEngine
from pme.core.events import LateEvent, SlidingTrigger, TelemetryEvent, TumblingResult
from pme.core.watermark import BoundedOutOfOrdernessWatermark

SENSOR_COUNT = 5
HORIZON = 60


def build_engine(ooo: int, lateness: int) -> WindowingEngine:
    return WindowingEngine(
        sliding=SlidingWindows(4, 1),
        tumbling=TumblingWindows(8),
        watermark=BoundedOutOfOrdernessWatermark(ooo),
        allowed_lateness=lateness,
    )


def event(machine: str, t: int) -> TelemetryEvent:
    return TelemetryEvent(machine, t, tuple(float(t + i) for i in range(SENSOR_COUNT)), 1.0, 0)


def ordered_stream(machines: int = 3, horizon: int = HORIZON) -> list[TelemetryEvent]:
    return [event(f"M{m}", t) for t in range(horizon) for m in range(machines)]


def jittered(events: list[TelemetryEvent], budget: int, rng: random.Random) -> list[TelemetryEvent]:
    """Re-deliver ``events`` out of order, but never past the budget.

    The delivery key is ``event_time + jitter`` with ``0 <= jitter <= budget``,
    and a stable sort on it. That is the same construction the synthetic broker
    uses, and it guarantees no event arrives more than ``budget`` hours behind
    the newest one delivered so far.
    """
    keyed = [(e.event_time + rng.randint(0, budget), i, e) for i, e in enumerate(events)]
    keyed.sort(key=lambda row: (row[0], row[1]))
    return [e for _, _, e in keyed]


def drain(engine: WindowingEngine, events: list[TelemetryEvent]) -> list[object]:
    out: list[object] = []
    for e in events:
        out.extend(engine.push(e))
    out.extend(engine.close())
    return out


# ------------------------------------------------------------------ conservation


@pytest.mark.parametrize("seed", range(12))
def test_every_pushed_event_is_released_dropped_or_still_buffered(seed):
    rng = random.Random(seed)
    engine = build_engine(ooo=rng.randint(0, 6), lateness=rng.randint(0, 4))
    events = jittered(ordered_stream(), budget=rng.randint(0, 12), rng=rng)
    for i, e in enumerate(events, start=1):
        engine.push(e)
        assert engine.released + engine.late_dropped + engine.reorder_buffer_size == i
    engine.close()
    assert engine.released + engine.late_dropped == len(events)
    assert engine.reorder_buffer_size == 0


@pytest.mark.parametrize("seed", range(12))
def test_a_sliding_trigger_is_emitted_for_exactly_the_released_events(seed):
    rng = random.Random(100 + seed)
    engine = build_engine(ooo=rng.randint(0, 6), lateness=rng.randint(0, 4))
    events = jittered(ordered_stream(), budget=rng.randint(0, 12), rng=rng)
    out = drain(engine, events)
    triggers = [o for o in out if isinstance(o, SlidingTrigger)]
    lates = [o for o in out if isinstance(o, LateEvent)]
    assert len(triggers) == engine.released
    assert len(lates) == engine.late_dropped
    assert len(triggers) + len(lates) == len(events)


@pytest.mark.parametrize("seed", range(8))
def test_the_lateness_histogram_and_per_machine_counts_add_up(seed):
    rng = random.Random(200 + seed)
    engine = build_engine(ooo=rng.randint(0, 3), lateness=0)
    events = jittered(ordered_stream(), budget=rng.randint(4, 20), rng=rng)
    drain(engine, events)
    stats = engine.stats()
    dropped = stats["events_dropped_late"]
    assert sum(b["count"] for b in stats["lateness_histogram"]) == dropped
    assert sum(stats["dropped_by_machine"].values()) == dropped
    assert len(stats["lateness_histogram"]) == len(LATENESS_BUCKETS) + 1
    assert 0.0 <= stats["drop_rate"] <= 1.0


@pytest.mark.parametrize("seed", range(8))
def test_a_tumbling_window_never_reports_more_samples_than_were_released(seed):
    rng = random.Random(300 + seed)
    engine = build_engine(ooo=rng.randint(0, 4), lateness=rng.randint(0, 2))
    events = jittered(ordered_stream(), budget=rng.randint(0, 8), rng=rng)
    out = drain(engine, events)
    rollups = [o for o in out if isinstance(o, TumblingResult)]
    assert sum(r.samples for r in rollups) == engine.released
    assert all(r.samples > 0 for r in rollups)
    assert all(r.window.start < r.window.end for r in rollups)


# ------------------------------------------------------------------- watermark


@pytest.mark.parametrize("seed", range(10))
def test_the_watermark_never_moves_backwards(seed):
    rng = random.Random(400 + seed)
    clock = BoundedOutOfOrdernessWatermark(rng.randint(0, 8))
    previous = clock.current
    for _ in range(400):
        clock.observe(rng.randint(-50, 50))
        assert clock.current >= previous
        previous = clock.current
    assert clock.close() >= previous


@pytest.mark.parametrize("seed", range(10))
def test_nothing_is_released_before_the_frontier_reaches_it(seed):
    """A released event's time is always at or below the frontier at release."""
    rng = random.Random(500 + seed)
    engine = build_engine(ooo=rng.randint(0, 6), lateness=rng.randint(0, 3))
    for e in jittered(ordered_stream(), budget=rng.randint(0, 10), rng=rng):
        for out in engine.push(e):
            if isinstance(out, SlidingTrigger):
                assert out.event.event_time <= engine.release_frontier


@pytest.mark.parametrize("seed", range(10))
def test_the_reported_watermark_never_exceeds_the_newest_event_seen(seed):
    rng = random.Random(600 + seed)
    engine = build_engine(ooo=rng.randint(0, 6), lateness=0)
    assert engine.stats()["watermark"] is None
    for e in jittered(ordered_stream(), budget=rng.randint(0, 10), rng=rng):
        engine.push(e)
        stats = engine.stats()
        assert stats["watermark"] <= engine.watermark.max_seen
        assert stats["stream_closed"] is False
    engine.close()
    closed = engine.stats()
    assert isinstance(closed["watermark"], int)
    assert closed["stream_closed"] is True


# ------------------------------------------------------------- order independence


@pytest.mark.parametrize("seed", range(10))
def test_any_delivery_order_inside_the_budget_matches_a_perfectly_ordered_stream(seed):
    """The equivalence the whole design is built on.

    Both engines see the same events; one gets them in event-time order, the
    other jittered by up to the out-of-orderness budget. Held open for the
    lateness budget on top, each machine's windows must fire on the same events
    in the same order, and each event time must release the same set of machines.

    What is *not* claimed: the interleaving between machines. The reorder buffer
    breaks ties on arrival sequence, so two machines reporting the same hour come
    out in whichever order the broker delivered them. Every downstream operator
    keys on the machine, so that interleaving changes no result.
    """
    rng = random.Random(700 + seed)
    budget = rng.randint(1, 6)
    events = ordered_stream()

    ordered_engine = build_engine(ooo=budget, lateness=budget)
    ordered_out = [o.event for o in drain(ordered_engine, events) if isinstance(o, SlidingTrigger)]

    shuffled_engine = build_engine(ooo=budget, lateness=budget)
    shuffled_out = [o.event for o in drain(shuffled_engine, jittered(events, budget, rng))
                    if isinstance(o, SlidingTrigger)]

    assert shuffled_engine.late_dropped == 0
    for machine in {e.machine_id for e in events}:
        assert [e.event_time for e in shuffled_out if e.machine_id == machine] == \
               [e.event_time for e in ordered_out if e.machine_id == machine]
    assert _by_time(shuffled_out) == _by_time(ordered_out)


def _by_time(events: list[TelemetryEvent]) -> list[tuple[int, list[str]]]:
    """Release order collapsed to ``(event_time, sorted machines)`` groups."""
    groups: dict[int, list[str]] = {}
    for e in events:
        groups.setdefault(e.event_time, []).append(e.machine_id)
    return [(t, sorted(groups[t])) for t in sorted(groups)]


@pytest.mark.parametrize("seed", range(6))
def test_jitter_past_the_budget_is_reported_rather_than_silently_absorbed(seed):
    """The other half of the contract: exceeding the budget must be visible."""
    rng = random.Random(800 + seed)
    engine = build_engine(ooo=1, lateness=0)
    out = drain(engine, jittered(ordered_stream(), budget=24, rng=rng))
    lates = [o for o in out if isinstance(o, LateEvent)]
    assert lates, "a 24h jitter against a 1h budget must produce late events"
    assert engine.drop_rate > 0
    for late in lates:
        assert late.lateness_h > 0
        assert late.event_time <= late.watermark


# ------------------------------------------------------------------ persistence


@pytest.mark.parametrize("seed", range(8))
def test_engine_state_is_a_fixed_point_at_any_point_mid_stream(seed):
    rng = random.Random(900 + seed)
    events = jittered(ordered_stream(), budget=rng.randint(0, 10), rng=rng)
    cut = rng.randrange(1, len(events))
    engine = build_engine(ooo=rng.randint(0, 6), lateness=rng.randint(0, 3))
    for e in events[:cut]:
        engine.push(e)
    snapshot = engine.state()

    restored = build_engine(ooo=engine.watermark.max_out_of_orderness,
                            lateness=engine.allowed_lateness)
    restored.restore(snapshot)
    assert restored.state() == snapshot

    tail = events[cut:]
    original = [type(o).__name__ for o in drain(engine, tail)]
    replayed = [type(o).__name__ for o in drain(restored, tail)]
    assert replayed == original
    assert restored.stats() == engine.stats()


# ------------------------------------------------------------- crash recovery


@pytest.mark.slow
@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.4, 0.6, 0.75, 0.95])
def test_a_crash_at_any_point_resumes_to_the_same_feeds(models, tmp_path, fraction):
    """Restore equivalence at six cut points, not just the two convenient ones."""
    reference = make_daemon(models, make_source(jitter=6, late_fraction=0.01), tmp_path, prefix="whole_")
    whole = reference.run()
    alerts = (tmp_path / "whole_alerts.jsonl").read_bytes()
    late = (tmp_path / "whole_late.jsonl").read_bytes()

    length = len(make_source(jitter=6, late_fraction=0.01))
    cut = max(1, int(length * fraction))
    first = make_daemon(models, make_source(jitter=6, late_fraction=0.01), tmp_path, prefix="part_",
                        checkpoint_every=max(1, cut // 2), stop_after_events=cut)
    first_report = first.run()
    assert first_report.stopped_early
    assert first.checkpoints.exists()

    second = make_daemon(models, make_source(jitter=6, late_fraction=0.01), tmp_path, prefix="part_",
                         checkpoint_every=max(1, cut // 2))
    assert second.restore_latest()
    resumed = second.run()

    assert resumed.events_consumed == whole.events_consumed
    assert resumed.engine_stats["events_dropped_late"] == whole.engine_stats["events_dropped_late"]
    assert resumed.engine_stats["lateness_histogram"] == whole.engine_stats["lateness_histogram"]
    assert (tmp_path / "part_alerts.jsonl").read_bytes() == alerts
    assert (tmp_path / "part_late.jsonl").read_bytes() == late


@pytest.mark.slow
def test_two_runs_of_the_same_stream_agree_on_everything_that_is_not_a_clock(models, tmp_path):
    """Determinism, minus the fields that are wall-clock by definition."""
    # run_id is a uuid, wall_seconds is a clock, and every sink summary carries
    # the path it wrote to, which differs by prefix.
    volatile = {"wall_seconds", "run_id", "sinks"}
    first = make_daemon(models, make_source(jitter=4), tmp_path, prefix="a_").run().to_dict()
    second = make_daemon(models, make_source(jitter=4), tmp_path, prefix="b_").run().to_dict()
    assert {k: v for k, v in first.items() if k not in volatile} == \
           {k: v for k, v in second.items() if k not in volatile}
    assert _without_paths(first["sinks"]) == _without_paths(second["sinks"])
    assert (tmp_path / "a_alerts.jsonl").read_bytes() == (tmp_path / "b_alerts.jsonl").read_bytes()


def _without_paths(summaries: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {name: {k: v for k, v in body.items() if k != "path"} for name, body in summaries.items()}
