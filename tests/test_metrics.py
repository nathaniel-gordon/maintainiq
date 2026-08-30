"""The in-process metric registry: counting, bucketing and rendering."""
from __future__ import annotations

import json
import threading

import pytest

from pme.metrics import DURATION_BUCKETS, Counter, Histogram, Registry, escape_label


@pytest.fixture
def registry() -> Registry:
    return Registry()


def _samples(text: str, prefix: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefix) and not line.startswith("#")]


def test_a_counter_starts_at_zero_and_adds_up(registry):
    hits = registry.counter("pme_hits_total", "Hits.", ("route",))
    assert hits.value(route="/a") == 0.0
    hits.inc(route="/a")
    hits.inc(2.0, route="/a")
    hits.inc(route="/b")
    assert hits.value(route="/a") == 3.0
    assert hits.value(route="/b") == 1.0


def test_a_counter_cannot_go_backwards(registry):
    hits = registry.counter("pme_hits_total", "Hits.")
    with pytest.raises(ValueError, match="cannot decrease"):
        hits.inc(-1.0)


def test_label_order_does_not_split_a_series(registry):
    hits = registry.counter("pme_hits_total", "Hits.", ("route", "status"))
    hits.inc(route="/a", status="200")
    hits.inc(status="200", route="/a")
    assert hits.value(route="/a", status="200") == 2.0


def test_a_wrong_label_set_is_a_programming_error(registry):
    hits = registry.counter("pme_hits_total", "Hits.", ("route",))
    with pytest.raises(ValueError, match="expects labels"):
        hits.inc(path="/a")
    with pytest.raises(ValueError, match="expects labels"):
        hits.inc()


def test_an_unobserved_metric_renders_a_header_and_no_samples(registry):
    registry.counter("pme_hits_total", "Hits.", ("route",))
    text = registry.render()
    assert "# TYPE pme_hits_total counter" in text
    assert _samples(text, "pme_hits_total") == []


def test_counter_renders_one_line_per_series(registry):
    hits = registry.counter("pme_hits_total", "Hits.", ("route",))
    hits.inc(2, route="/a")
    hits.inc(route="/b")
    text = registry.render()
    assert text.count("# HELP pme_hits_total") == 1
    assert _samples(text, "pme_hits_total") == ['pme_hits_total{route="/a"} 2', 'pme_hits_total{route="/b"} 1']


def test_histogram_buckets_are_cumulative(registry):
    latency = registry.histogram("pme_seconds", "Latency.", (), buckets=(0.1, 1.0))
    for value in (0.05, 0.5, 0.5, 42.0):
        latency.observe(value)
    lines = _samples(registry.render(), "pme_seconds")
    assert lines == [
        'pme_seconds_bucket{le="0.1"} 1',
        'pme_seconds_bucket{le="1.0"} 3',
        'pme_seconds_bucket{le="+Inf"} 4',
        "pme_seconds_sum 43.05",
        "pme_seconds_count 4",
    ]


def test_an_observation_on_a_bucket_edge_falls_inside_it(registry):
    latency = registry.histogram("pme_seconds", "Latency.", (), buckets=(0.1,))
    latency.observe(0.1)
    assert 'pme_seconds_bucket{le="0.1"} 1' in registry.render()


def test_histogram_count_and_sum_track_observations(registry):
    latency = registry.histogram("pme_seconds", "Latency.", ("route",))
    latency.observe(0.2, route="/a")
    latency.observe(0.3, route="/a")
    assert latency.count(route="/a") == 2
    assert latency.total(route="/a") == pytest.approx(0.5)
    assert latency.count(route="/b") == 0


def test_the_timer_records_even_when_the_block_raises(registry):
    latency = registry.histogram("pme_seconds", "Latency.", ("route",))
    with pytest.raises(RuntimeError), latency.time(route="/a"):
        raise RuntimeError("boom")
    assert latency.count(route="/a") == 1


def test_buckets_are_sorted_however_they_are_declared():
    histogram = Histogram("pme_seconds", "Latency.", (), buckets=(1.0, 0.1, 0.5))
    assert histogram.buckets == (0.1, 0.5, 1.0)


def test_a_histogram_needs_at_least_one_edge():
    with pytest.raises(ValueError, match="bucket edge"):
        Histogram("pme_seconds", "Latency.", (), buckets=())


def test_registering_the_same_name_twice_returns_the_same_metric(registry):
    first = registry.counter("pme_hits_total", "Hits.", ("route",))
    second = registry.counter("pme_hits_total", "Hits.", ("route",))
    assert first is second


def test_redeclaring_a_metric_with_a_different_shape_is_refused(registry):
    registry.counter("pme_hits_total", "Hits.", ("route",))
    with pytest.raises(ValueError, match="different shape"):
        registry.counter("pme_hits_total", "Hits.", ("route", "status"))
    with pytest.raises(ValueError, match="different shape"):
        registry.histogram("pme_hits_total", "Hits.", ("route",))


def test_metric_names_render_in_a_stable_order(registry):
    registry.counter("pme_zzz_total", "Z.").inc()
    registry.counter("pme_aaa_total", "A.").inc()
    names = [line.split()[2] for line in registry.render().splitlines() if line.startswith("# TYPE")]
    assert names == sorted(names)


def test_clear_empties_the_registry(registry):
    registry.counter("pme_hits_total", "Hits.").inc()
    registry.clear()
    assert registry.render() == ""


def test_snapshot_is_json_shaped(registry):
    registry.counter("pme_hits_total", "Hits.", ("route",)).inc(3, route="/a")
    registry.histogram("pme_seconds", "Latency.", (), buckets=(1.0,)).observe(0.5)
    snapshot = registry.snapshot()
    assert snapshot["pme_hits_total"]["samples"]["route=/a"] == 3.0
    assert snapshot["pme_seconds"]["type"] == "histogram"
    assert snapshot["pme_seconds"]["samples"][""]["count"] == 1
    assert json.dumps(snapshot)


def test_integral_values_render_without_a_decimal_point(registry):
    registry.counter("pme_hits_total", "Hits.").inc(2.0)
    assert "pme_hits_total 2" in registry.render()


def test_concurrent_increments_are_not_lost(registry):
    hits = registry.counter("pme_hits_total", "Hits.", ("route",))
    latency = registry.histogram("pme_seconds", "Latency.", ("route",))

    def work() -> None:
        for _ in range(500):
            hits.inc(route="/a")
            latency.observe(0.001, route="/a")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert hits.value(route="/a") == 2000.0
    assert latency.count(route="/a") == 2000


def test_default_duration_buckets_are_ascending():
    assert list(DURATION_BUCKETS) == sorted(DURATION_BUCKETS)
    assert DURATION_BUCKETS[0] > 0


def test_a_bare_counter_has_no_label_braces():
    counter = Counter("pme_hits_total", "Hits.")
    counter.inc()
    assert "pme_hits_total 1" in "\n".join(counter.render())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('a"b', 'a\\"b'),
        ("a\\b", "a\\\\b"),
        ("a\nb", "a\\nb"),
        ("plain", "plain"),
    ],
)
def test_label_values_are_escaped(raw, expected):
    assert escape_label(raw) == expected
