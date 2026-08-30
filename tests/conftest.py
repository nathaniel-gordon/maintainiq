"""Shared fixtures.

Training the gradient-boosting bundle is the only expensive thing in the suite,
so it happens once per session and every daemon test reuses it against short
streams.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest

from pme.learning.trainer import TrainingResult, train_from_stream
from pme.runtime.config import OutputPaths, StreamConfig
from pme.runtime.daemon import MaintenanceDaemon
from pme.sources.synthetic import SyntheticFleetSource
from pme.store.db import MEMORY, Database
from pme.store.repository import MetadataStore

SEED = 42

#: Fleet the bundle is fitted on: large enough to hold out whole machines and
#: still see failures on both sides of the split.
QUALITY_FLEET = {"n_machines": 10, "hours": 24 * 50}

#: Short fleet for the streaming tests, where model accuracy is irrelevant but
#: alert volume must be non-trivial.
STREAM_FLEET = {"n_machines": 6, "hours": 24 * 25}

#: Watermark/lateness budget used by the streaming tests.
LIVE_CONFIG = StreamConfig(
    max_out_of_orderness_h=6,
    allowed_lateness_h=2,
    scoring_batch_size=128,
    heartbeat_every=500,
    max_drop_rate=1.0,
)

BACKFILL_CONFIG = StreamConfig(max_out_of_orderness_h=0, allowed_lateness_h=0)


@pytest.fixture(scope="session")
def trained() -> TrainingResult:
    """Bundle fitted once, by replaying a historian backfill through the operators."""
    source = SyntheticFleetSource(seed=SEED, **QUALITY_FLEET)
    return train_from_stream(source, seed=SEED, config=BACKFILL_CONFIG)


@pytest.fixture(scope="session")
def models(trained: TrainingResult):
    return trained.models


@pytest.fixture
def ordered_source() -> SyntheticFleetSource:
    """Short stream delivered in perfect event-time order."""
    return SyntheticFleetSource(seed=SEED, **STREAM_FLEET)


@pytest.fixture
def store() -> Iterator[MetadataStore]:
    """An in-memory metadata store, migrated and ready, closed on teardown."""
    db = Database(MEMORY)
    db.migrate()
    yield MetadataStore(db)
    db.close()


def make_source(jitter: int = 0, late_fraction: float = 0.0, delivery_seed: int = SEED + 1,
                **overrides) -> SyntheticFleetSource:
    """Build the short test fleet under a chosen delivery regime."""
    params = {**STREAM_FLEET, **overrides}
    return SyntheticFleetSource(
        seed=SEED,
        out_of_order_delay=jitter,
        late_fraction=late_fraction,
        delivery_seed=delivery_seed,
        **params,
    )


def make_daemon(models, source, tmp_path, prefix: str = "", store: MetadataStore | None = None,
                **config_overrides) -> MaintenanceDaemon:
    """Wire a daemon writing into ``tmp_path`` under an optional filename prefix."""
    config = replace(LIVE_CONFIG, **config_overrides)
    return MaintenanceDaemon(source, models, config, OutputPaths.under(tmp_path, prefix=prefix), store=store)
