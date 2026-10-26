"""Replay MovieLens ratings into Kafka as short-video view events."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from shorts_recsys.config import (
    KAFKA_TOPIC,
    TrainConfig,
    env_path,
    setup_logging,
)
from shorts_recsys.data import build_events, ensure_dataset, load_tables

logger = logging.getLogger(__name__)


def kafka_servers() -> str:
    import os

    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def make_producer(servers: str, attempts: int = 30, delay: float = 2.0):
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    # kafka-python 3.x dropped NoBrokersAvailable. The constructor raises
    # KafkaTimeoutError (a KafkaError) when the broker list is unreachable.
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return KafkaProducer(
                bootstrap_servers=servers.split(","),
                acks=1,
                linger_ms=20,
                retries=3,
                request_timeout_ms=10000,
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                key_serializer=lambda key: str(key).encode("utf-8"),
            )
        except (KafkaError, OSError) as exc:
            last_error = exc
            logger.warning("kafka not ready (%s/%s): %s", attempt + 1, attempts, exc)
            time.sleep(delay)
    raise RuntimeError(f"could not connect to kafka at {servers}") from last_error


def load_event_log(data_dir: Path, config: TrainConfig, max_events: int | None) -> list[dict]:
    dataset_dir = ensure_dataset(data_dir)
    ratings, movies = load_tables(dataset_dir)
    events = build_events(
        ratings,
        movies,
        positive_threshold=config.positive_threshold,
        short_length_sec=config.short_length_sec,
    )
    if max_events is not None:
        events = events[: max(0, max_events)]
    return events


def publish(events: list[dict], servers: str, topic: str, rate: float) -> int:
    producer = make_producer(servers)
    batch = 200
    sent = 0
    started = time.perf_counter()
    for event in events:
        producer.send(topic, key=event["user_id"], value=event)
        sent += 1
        if rate > 0 and sent % batch == 0:
            expected = sent / rate
            elapsed = time.perf_counter() - started
            if expected > elapsed:
                time.sleep(expected - elapsed)
    producer.flush()
    producer.close()
    logger.info("published %s events to %s", sent, topic)
    return sent


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Replay interaction events into Kafka")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--rate", type=float, default=2000.0, help="events per second; 0 is unlimited"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    events = load_event_log(
        args.data_dir or env_path("DATA_DIR", "data"),
        TrainConfig(),
        args.max_events,
    )
    if args.dry_run:
        print(f"events={len(events)}")
        if events:
            print(
                json.dumps(
                    {k: events[0][k] for k in ("user_id", "item_id", "liked", "watch_time_sec")}
                )
            )
        return
    publish(events, kafka_servers(), KAFKA_TOPIC, args.rate)


if __name__ == "__main__":
    main()
