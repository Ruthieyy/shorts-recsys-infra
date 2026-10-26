"""Kafka consumer that folds view events into Redis user features."""

from __future__ import annotations

import logging
import os
import signal
import time

from shorts_recsys.config import KAFKA_TOPIC, setup_logging
from shorts_recsys.features import UserFeatureState
from shorts_recsys.replay import kafka_servers, make_producer

logger = logging.getLogger(__name__)

_STOP = False


def _handle_stop(signum, _frame) -> None:
    del signum
    global _STOP
    _STOP = True


def redis_client():
    import redis

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(
        url, decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0
    )


def wait_for_redis(attempts: int = 30, delay: float = 2.0):
    last_error: Exception | None = None
    for attempt in range(attempts):
        client = redis_client()
        try:
            client.ping()
            return client
        except Exception as exc:
            last_error = exc
            logger.warning("redis not ready (%s/%s): %s", attempt + 1, attempts, exc)
            time.sleep(delay)
    raise RuntimeError("could not connect to redis") from last_error


def make_consumer(servers: str, topic: str, group: str):
    from kafka import KafkaConsumer

    # make_producer already retries broker metadata; reuse that wait, then close it.
    probe = make_producer(servers, attempts=30, delay=2.0)
    probe.close()
    return KafkaConsumer(
        topic,
        bootstrap_servers=servers.split(","),
        group_id=group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda raw: __import__("json").loads(raw.decode("utf-8")),
    )


def flush(client, states: dict[int, UserFeatureState], dirty: set[int]) -> None:
    if not dirty:
        return
    pipe = client.pipeline()
    for user_id in dirty:
        pipe.hset(f"user:{user_id}", mapping=states[user_id].to_redis())
    pipe.execute()
    dirty.clear()


def consume_forever() -> None:
    global _STOP
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    servers = kafka_servers()
    group = os.environ.get("KAFKA_GROUP", "shorts-feature-worker")
    client = wait_for_redis()
    consumer = make_consumer(servers, KAFKA_TOPIC, group)
    states: dict[int, UserFeatureState] = {}
    dirty: set[int] = set()
    last_flush = time.monotonic()
    logger.info("consuming %s from %s", KAFKA_TOPIC, servers)
    while not _STOP:
        polled = consumer.poll(timeout_ms=500)
        for records in polled.values():
            for record in records:
                event = record.value
                user_id = int(event["user_id"])
                state = states.setdefault(user_id, UserFeatureState())
                state.update(event)
                dirty.add(user_id)
        if dirty and (len(dirty) >= 100 or time.monotonic() - last_flush > 0.25):
            flush(client, states, dirty)
            last_flush = time.monotonic()
    flush(client, states, dirty)
    consumer.close()


def main() -> None:
    setup_logging()
    consume_forever()


if __name__ == "__main__":
    main()
