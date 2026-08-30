"""Windowing engine: release frontier, late side output and reported statistics."""
from __future__ import annotations

from pme.core.assigners import SlidingWindows, TumblingWindows
from pme.core.engine import WindowingEngine
from pme.core.events import LateEvent, SlidingTrigger, TelemetryEvent, TumblingResult
from pme.core.watermark import BoundedOutOfOrdernessWatermark

SENSOR_COUNT = 5


def _engine(ooo: int = 2, lateness: int = 0, tumbling: int = 8) -> WindowingEngine:
    return WindowingEngine(
        sliding=SlidingWindows(4, 1),
        tumbling=TumblingWindows(tumbling),
        watermark=BoundedOutOfOrdernessWatermark(ooo),
        allowed_lateness=lateness,
    )


def _event(machine: str, t: int) -> TelemetryEvent:
    return TelemetryEvent(machine, t, tuple(float(t + i) for i in range(SENSOR_COUNT)), 1.0, 0)


def _push_all(engine: WindowingEngine, times: list[int], machine: str = "M1") -> list:
    out: list = []
    for t in times:
        out.extend(engine.push(_event(machine, t)))
    return out


def test_events_are_released_only_once_the_frontier_passes_them():
    engine = _engine()
    assert engine.push(_event("M1", 0)) == []
    released = [o for o in _push_all(engine, [1, 2, 3, 4]) if isinstance(o, SlidingTrigger)]
    assert [o.event.event_time for o in released] == [0, 1]


def test_a_late_event_comes_back_as_a_side_output():
    engine = _engine()
    _push_all(engine, list(range(20)))
    out = engine.push(_event("M1", 3))
    late = [o for o in out if isinstance(o, LateEvent)]
    assert len(late) == 1
    assert late[0].event_time == 3
    assert late[0].lateness_h == engine.release_frontier - 3 + 1
    assert engine.late_by_machine == {"M1": 1}
    assert engine.drop_rate > 0


def test_out_of_order_inside_the_budget_is_not_late():
    engine = _engine(ooo=6)
    out = _push_all(engine, [0, 5, 3, 8, 6, 12, 9])
    assert not [o for o in out if isinstance(o, LateEvent)]
    assert engine.late_dropped == 0


def test_close_drains_every_window():
    engine = _engine()
    _push_all(engine, list(range(10)))
    drained = engine.close()
    assert [o.event.event_time for o in drained if isinstance(o, SlidingTrigger)] == [7, 8, 9]
    assert any(isinstance(o, TumblingResult) for o in drained)
    assert engine.reorder_buffer_size == 0


def test_watermark_is_reported_as_an_integer_after_drain():
    engine = _engine()
    _push_all(engine, list(range(10)))
    engine.close()
    stats = engine.stats()
    assert isinstance(stats["watermark"], int)
    assert stats["stream_closed"] is True
    assert stats["watermark"] == 9


def test_watermark_is_null_before_the_first_event():
    stats = _engine().stats()
    assert stats["watermark"] is None
    assert stats["stream_closed"] is False


def test_lateness_histogram_keeps_its_bucket_order():
    engine = _engine()
    _push_all(engine, list(range(200)))
    for t in (198, 150, 20):
        engine.push(_event("M1", t))
    buckets = engine.stats()["lateness_histogram"]
    edges = [b["upper_h"] for b in buckets]
    assert edges == [1, 2, 4, 8, 16, 32, 64, 128, None]
    assert sum(b["count"] for b in buckets) == engine.late_dropped


def test_state_round_trip_restores_counters_and_buffer():
    engine = _engine()
    _push_all(engine, list(range(30)))
    engine.push(_event("M1", 1))
    state = engine.state()

    restored = _engine()
    restored.restore(state)
    assert restored.state() == state
    assert restored.late_by_machine == engine.late_by_machine
    assert restored.stats() == engine.stats()
