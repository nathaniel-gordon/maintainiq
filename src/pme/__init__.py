"""Predictive Maintenance Engine — an event-time stream-processing pipeline.

Topology::

    TelemetrySource ─► WindowingEngine ─► FeatureOperator ─► ScoringOperator ─► AlertOperator
      (out of order)     watermarks,        incremental        micro-batched      event-time
                         lateness,          rolling stats      inference          cooldown
                         tumbling roll-ups        │                  │                 │
                                                  └──────────────────┴────────► sink fan-out
                                                       AlertSink · ScheduleSink · MetricsSink

Run it with ``python . demo`` from the project root, or drive
:class:`pme.runtime.MaintenanceDaemon` directly.
"""

__version__ = "2.0.0"
