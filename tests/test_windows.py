"""Window geometry and the incremental per-machine aggregate."""
from __future__ import annotations

import numpy as np
import pytest

from pme.core import rolling
from pme.core.assigners import SlidingWindows, TumblingWindows
from pme.core.rolling import MachineWindowState
from pme.core.schema import WINDOW_SIZE_H

SENSOR_COUNT = 5


def test_tumbling_windows_partition_the_timeline():
    tumbling = TumblingWindows(8)
    seen = [tumbling.window_of(t) for t in range(40)]
    for t, window in enumerate(seen):
        assert window.start <= t < window.end
        assert window.start % 8 == 0
        assert window.end - window.start == 8
    starts = sorted({w.start for w in seen})
    assert starts == list(range(0, 40, 8))


def test_a_window_boundary_belongs_to_the_later_window():
    tumbling = TumblingWindows(8)
    assert tumbling.window_of(7).start == 0
    assert tumbling.window_of(8).start == 8


def test_the_closing_window_is_one_of_the_assigned_windows():
    sliding = SlidingWindows(WINDOW_SIZE_H, 1)
    for t in (0, 1, 23, 24, 100, 1_000):
        closing = sliding.closing_window(t)
        assigned = sliding.assign(t)
        assert closing in assigned
        assert closing.max_timestamp == t
        assert all(w.start <= t < w.end for w in assigned)


def _admit(state: MachineWindowState, times, values) -> None:
    for t, row in zip(times, values, strict=True):
        state.admit(int(t), tuple(float(v) for v in row), 1.0)


def test_incremental_statistics_match_a_direct_recomputation():
    rng = np.random.default_rng(7)
    times = np.arange(40)
    values = rng.normal(50.0, 5.0, size=(40, SENSOR_COUNT))
    state = MachineWindowState()
    _admit(state, times, values)
    state.evict_before(16)

    window = values[16:]
    window_times = times[16:].astype(float)
    features = state.feature_vector()
    for k in range(SENSOR_COUNT):
        latest, mean, std, slope = features[4 * k: 4 * k + 4]
        assert latest == pytest.approx(window[-1, k])
        assert mean == pytest.approx(window[:, k].mean())
        assert std == pytest.approx(window[:, k].std(ddof=1))
        expected = np.polyfit(window_times, window[:, k], 1)[0]
        assert slope == pytest.approx(expected, abs=1e-9)


def test_a_constant_channel_has_zero_variance_not_a_negative_one():
    state = MachineWindowState()
    _admit(state, range(30), [[4.0] * SENSOR_COUNT] * 30)
    features = state.feature_vector()
    for k in range(SENSOR_COUNT):
        _, mean, std, slope = features[4 * k: 4 * k + 4]
        assert mean == pytest.approx(4.0)
        assert std == 0.0
        assert slope == pytest.approx(0.0)


def test_a_single_sample_has_no_spread_and_no_slope():
    state = MachineWindowState()
    _admit(state, [5], [[1.0, 2.0, 3.0, 4.0, 5.0]])
    features = state.feature_vector()
    assert features[2] == 0.0
    assert features[3] == 0.0


def test_an_empty_window_cannot_produce_features():
    with pytest.raises(ValueError, match="empty window"):
        MachineWindowState().feature_vector()


def test_rebasing_the_time_origin_keeps_the_statistics(monkeypatch):
    """The rebase path is unreachable at the shipped limit, so lower it."""
    monkeypatch.setattr(rolling, "REBASE_LIMIT", 16)
    rng = np.random.default_rng(3)
    times = np.arange(0, 60)
    values = rng.normal(10.0, 2.0, size=(60, SENSOR_COUNT))

    rebased = MachineWindowState()
    _admit(rebased, times, values)
    rebased.evict_before(40)
    assert rebased._origin != 0, "the rebase path did not run"

    monkeypatch.setattr(rolling, "REBASE_LIMIT", 1 << 20)
    plain = MachineWindowState()
    _admit(plain, times, values)
    plain.evict_before(40)

    for a, b in zip(rebased.feature_vector(), plain.feature_vector(), strict=True):
        assert a == pytest.approx(b, abs=1e-9)


def test_state_round_trip_is_a_fixed_point():
    rng = np.random.default_rng(11)
    state = MachineWindowState()
    _admit(state, range(24), rng.normal(0.0, 1.0, size=(24, SENSOR_COUNT)))
    snapshot = state.state()
    restored = MachineWindowState.from_state(snapshot)
    assert restored.state() == snapshot
    assert restored.feature_vector() == state.feature_vector()
