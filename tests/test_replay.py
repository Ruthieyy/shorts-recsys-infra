"""Kafka producer startup retries against kafka-python 3.x."""

from kafka.errors import KafkaTimeoutError

from shorts_recsys.replay import make_producer


def test_make_producer_retries_kafka_timeout(monkeypatch):
    calls = {"n": 0}

    class FakeProducer:
        def __init__(self, **kwargs):
            calls["n"] += 1
            calls["servers"] = kwargs["bootstrap_servers"]
            if calls["n"] < 3:
                raise KafkaTimeoutError("broker down")

    monkeypatch.setattr("kafka.KafkaProducer", FakeProducer)
    monkeypatch.setattr("shorts_recsys.replay.time.sleep", lambda _seconds: None)
    producer = make_producer("kafka:19092", attempts=5, delay=0)
    assert calls["n"] == 3
    assert calls["servers"] == ["kafka:19092"]
    assert isinstance(producer, FakeProducer)


def test_make_producer_raises_after_attempts(monkeypatch):
    class FakeProducer:
        def __init__(self, **kwargs):
            raise OSError("connection refused")

    monkeypatch.setattr("kafka.KafkaProducer", FakeProducer)
    monkeypatch.setattr("shorts_recsys.replay.time.sleep", lambda _seconds: None)
    try:
        make_producer("localhost:9092", attempts=2, delay=0)
    except RuntimeError as exc:
        assert "localhost:9092" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
