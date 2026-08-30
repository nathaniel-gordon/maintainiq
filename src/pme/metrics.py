"""In-process counters and latency histograms, rendered as Prometheus text.

The ``/metrics`` endpoint already exposed gauges read back out of the database —
how many runs exist, what the last one consumed. Those describe the *stream*.
They say nothing about the service itself: how many requests it answered, how
many were rejected, how long they took. That is what this module adds.

Everything lives in one process-wide :class:`Registry`. There is no push, no
scrape agent and no external dependency: the HTTP handler increments a counter
and observes a duration, and ``/metrics`` renders whatever has been recorded.
A metric with no observations renders its ``# HELP``/``# TYPE`` header and no
samples, so the exposition never claims a measurement that was not taken.

Buckets are fixed at construction. Prometheus histograms are cumulative — each
``_bucket{le="x"}`` counts every observation at or below ``x`` — and the renderer
does that accumulation, so callers only ever record a raw value.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

#: Bucket edges (seconds) for request latency. Spans a sub-millisecond in-memory
#: read to a ten-second timeout, which is the range this service actually covers.
DURATION_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

#: A metric sample's label values, sorted by label name so two calls that pass
#: the same labels in a different keyword order land on the same series.
LabelKey = tuple[tuple[str, str], ...]


def escape_label(value: str) -> str:
    """Escape a label value for the text exposition format.

    Label values can carry caller-supplied text (a run id from the CLI, say). An
    unescaped quote or newline there would let that text forge extra metric
    lines, so nothing reaches the exposition without passing through here.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(key: LabelKey, extra: str = "") -> str:
    parts = [f'{name}="{escape_label(value)}"' for name, value in key]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


def _format(value: float) -> str:
    """Integral values render without a decimal point; the rest keep full precision."""
    if value == int(value) and abs(value) < 1 << 53:
        return str(int(value))
    return repr(value)


class Metric:
    """Shared naming, labelling and locking for every metric type."""

    kind = "untyped"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.help_text = help_text
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str]) -> LabelKey:
        if set(labels) != set(self.labelnames):
            raise ValueError(
                f"{self.name} expects labels {sorted(self.labelnames)}, got {sorted(labels)}"
            )
        return tuple(sorted((k, str(v)) for k, v in labels.items()))

    def _header(self) -> list[str]:
        return [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} {self.kind}"]

    def render(self) -> list[str]:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


class Counter(Metric):
    """Monotonically increasing count, one series per label combination."""

    kind = "counter"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("a counter cannot decrease")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        return self._header() + [f"{self.name}{_render_labels(k)} {_format(v)}" for k, v in items]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"type": self.kind, "samples": {"|".join(f"{a}={b}" for a, b in k): v
                                                   for k, v in sorted(self._values.items())}}


class Histogram(Metric):
    """Cumulative-bucket distribution with a running sum and count."""

    kind = "histogram"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = (),
                 buckets: Sequence[float] = DURATION_BUCKETS) -> None:
        super().__init__(name, help_text, labelnames)
        edges = tuple(sorted(float(b) for b in buckets))
        if not edges:
            raise ValueError("a histogram needs at least one bucket edge")
        self.buckets = edges
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._counts.get(key)
            if counts is None:
                counts = [0] * (len(self.buckets) + 1)
                self._counts[key] = counts
                self._sums[key] = 0.0
            self._sums[key] += value
            for i, edge in enumerate(self.buckets):
                if value <= edge:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1

    @contextmanager
    def time(self, **labels: str) -> Iterator[None]:
        """Observe the wall-clock duration of the block, even if it raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - started, **labels)

    def count(self, **labels: str) -> int:
        with self._lock:
            return sum(self._counts.get(self._key(labels), []))

    def total(self, **labels: str) -> float:
        with self._lock:
            return self._sums.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = [(k, list(v), self._sums[k]) for k, v in sorted(self._counts.items())]
        lines = self._header()
        for key, counts, total in items:
            cumulative = 0
            for edge, count in zip(self.buckets, counts, strict=False):
                cumulative += count
                bucket = _render_labels(key, 'le="' + repr(edge) + '"')
                lines.append(f"{self.name}_bucket{bucket} {cumulative}")
            cumulative += counts[-1]
            infinite = _render_labels(key, 'le="+Inf"')
            lines.append(f"{self.name}_bucket{infinite} {cumulative}")
            lines.append(f"{self.name}_sum{_render_labels(key)} {_format(total)}")
            lines.append(f"{self.name}_count{_render_labels(key)} {cumulative}")
        return lines

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": self.kind,
                "buckets": list(self.buckets),
                "samples": {
                    "|".join(f"{a}={b}" for a, b in k): {"counts": list(v), "sum": self._sums[k],
                                                         "count": sum(v)}
                    for k, v in sorted(self._counts.items())
                },
            }


MetricT = TypeVar("MetricT", bound=Metric)


class Registry:
    """Named collection of metrics. Get-or-create, so import order does not matter."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._get_or_create(Counter(name, help_text, labelnames))

    def histogram(self, name: str, help_text: str, labelnames: Sequence[str] = (),
                  buckets: Sequence[float] = DURATION_BUCKETS) -> Histogram:
        return self._get_or_create(Histogram(name, help_text, labelnames, buckets))

    def _get_or_create(self, metric: MetricT) -> MetricT:
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is None:
                self._metrics[metric.name] = metric
                return metric
        if type(existing) is not type(metric) or existing.labelnames != metric.labelnames:
            raise ValueError(f"{metric.name} is already registered with a different shape")
        return existing

    def render(self) -> str:
        """Prometheus text exposition of every registered metric."""
        with self._lock:
            metrics = [self._metrics[name] for name in sorted(self._metrics)]
        lines: list[str] = []
        for metric in metrics:
            lines.extend(metric.render())
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable dump, for the CLI and for assertions in tests."""
        with self._lock:
            metrics = dict(self._metrics)
        return {name: metrics[name].snapshot() for name in sorted(metrics)}

    def clear(self) -> None:
        """Drop every metric. Only used to isolate tests from one another."""
        with self._lock:
            self._metrics.clear()


#: The process-wide registry. One process, one exposition.
REGISTRY = Registry()


def http_requests_total() -> Counter:
    return REGISTRY.counter(
        "pme_http_requests_total",
        "HTTP requests answered, by method, route template and status.",
        ("method", "route", "status"),
    )


def http_request_duration_seconds() -> Histogram:
    return REGISTRY.histogram(
        "pme_http_request_duration_seconds",
        "Wall-clock time spent handling a request, by route template.",
        ("route",),
    )


__all__ = [
    "DURATION_BUCKETS",
    "REGISTRY",
    "Counter",
    "Histogram",
    "Metric",
    "Registry",
    "escape_label",
    "http_request_duration_seconds",
    "http_requests_total",
]
