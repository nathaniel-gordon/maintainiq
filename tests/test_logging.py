"""Structured logging: JSON shape, correlation ids, redaction and key collisions."""
from __future__ import annotations

import io
import json
import logging

import pytest

from pme.logging_setup import (
    JsonFormatter,
    PlainFormatter,
    configure_logging,
    correlation_id,
    correlation_scope,
    get_logger,
    log_level_from_env,
    new_correlation_id,
    redact,
    set_correlation_id,
)


@pytest.fixture
def sink() -> io.StringIO:
    """A stream carrying JSON log lines, with logging reconfigured to write to it."""
    stream = io.StringIO()
    configure_logging(level="DEBUG", fmt="json", stream=stream)
    yield stream
    configure_logging(level="INFO", fmt="plain", stream=io.StringIO())


def lines(sink: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]


def test_get_logger_returns_the_structured_wrapper():
    log = get_logger("pme.test")
    assert hasattr(log, "info")
    assert isinstance(log.stdlib, logging.Logger)


def test_a_bare_name_is_placed_under_the_pme_root():
    assert get_logger("widgets").stdlib.name == "pme.widgets"
    assert get_logger("pme.widgets").stdlib.name == "pme.widgets"


def test_every_record_is_one_json_object(sink):
    get_logger("pme.test").info("run started", events=3)
    records = lines(sink)
    assert len(records) == 1
    assert records[0]["event"] == "run started"
    assert records[0]["level"] == "INFO"
    assert records[0]["events"] == 3
    assert records[0]["logger"] == "pme.test"
    assert "ts" in records[0]


def test_keyword_fields_land_beside_the_message(sink):
    get_logger("pme.test").warning("late drop", machine_id="M001", lateness_h=7)
    record = lines(sink)[0]
    assert record["machine_id"] == "M001"
    assert record["lateness_h"] == 7


def test_a_correlation_id_is_attached_inside_a_scope(sink):
    log = get_logger("pme.test")
    with correlation_scope("abc123"):
        log.info("inside")
    log.info("outside")
    inside, outside = lines(sink)
    assert inside["correlation_id"] == "abc123"
    assert "correlation_id" not in outside


def test_a_scope_restores_the_previous_id():
    set_correlation_id("outer")
    with correlation_scope("inner"):
        assert correlation_id() == "inner"
    assert correlation_id() == "outer"
    set_correlation_id("")


def test_a_scope_without_an_id_invents_one():
    with correlation_scope() as cid:
        assert len(cid) == 16
        assert correlation_id() == cid


def test_generated_ids_differ():
    assert new_correlation_id() != new_correlation_id()


@pytest.mark.parametrize(
    "field",
    ["token", "api_key", "apikey", "password", "passwd", "secret", "authorization", "bearer", "PME_API_TOKEN"],
)
def test_credential_shaped_fields_are_masked(field):
    assert redact({field: "hunter2"})[field] == "***"


def test_redaction_reaches_into_nested_mappings():
    out = redact({"config": {"api_key": "abc", "window_size_h": 24}, "run": "r1"})
    assert out["config"]["api_key"] == "***"
    assert out["config"]["window_size_h"] == 24
    assert out["run"] == "r1"


def test_ordinary_fields_survive_redaction():
    assert redact({"machine_id": "M001", "risk_prob": 0.4}) == {"machine_id": "M001", "risk_prob": 0.4}


def test_a_logged_token_never_reaches_the_stream(sink):
    get_logger("pme.test").info("configured", api_token="s3cret-value")
    assert "s3cret-value" not in sink.getvalue()
    assert lines(sink)[0]["api_token"] == "***"


@pytest.mark.parametrize("field", ["message", "name", "module", "levelname", "args", "asctime"])
def test_a_field_named_like_a_log_record_attribute_is_renamed_not_fatal(sink, field):
    """These used to raise KeyError inside the stdlib's own ``makeRecord``."""
    get_logger("pme.test").info("collision", **{field: "value"})
    record = lines(sink)[0]
    assert record[f"field_{field}"] == "value"
    assert record["event"] == "collision"


def test_an_exception_is_rendered_into_the_record(sink):
    log = get_logger("pme.test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("it failed", stage="scoring")
    record = lines(sink)[0]
    assert "ValueError: boom" in record["exception"]
    assert record["stage"] == "scoring"


def test_levels_below_the_threshold_are_dropped():
    stream = io.StringIO()
    configure_logging(level="WARNING", fmt="json", stream=stream)
    log = get_logger("pme.test")
    log.debug("quiet")
    log.info("also quiet")
    log.error("loud")
    assert [json.loads(line)["event"] for line in stream.getvalue().splitlines()] == ["loud"]


def test_configure_logging_replaces_rather_than_stacks_handlers():
    first, second = io.StringIO(), io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=first)
    configure_logging(level="INFO", fmt="json", stream=second)
    get_logger("pme.test").info("once")
    assert first.getvalue() == ""
    assert len(second.getvalue().splitlines()) == 1


def test_the_plain_formatter_stays_human_readable():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="plain", stream=stream)
    with correlation_scope("cid00001"):
        get_logger("pme.test").info("run finished", alerts=4, api_key="nope")
    text = stream.getvalue()
    assert "run finished" in text
    assert "alerts=4" in text
    assert "[cid00001]" in text
    assert "nope" not in text


def test_the_formatters_can_be_used_directly():
    record = logging.LogRecord("pme.test", logging.INFO, __file__, 1, "hello", None, None)
    assert json.loads(JsonFormatter().format(record))["event"] == "hello"
    assert "hello" in PlainFormatter().format(record)


def test_the_log_level_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("PME_LOG_LEVEL", raising=False)
    assert log_level_from_env() == "INFO"
    monkeypatch.setenv("PME_LOG_LEVEL", "DEBUG")
    assert log_level_from_env() == "DEBUG"
