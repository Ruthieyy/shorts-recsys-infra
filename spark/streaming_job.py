"""Spark Structured Streaming job: Kafka user_events -> feature snapshot.

Counters match ``UserFeatureState`` (view count, like count, watch-time sum).
The job uses complete-mode aggregation, which is correct at demo volume and
recomputes state every micro-batch. A production cluster would keep incremental
state with flatMapGroupsWithState. The Redis feature worker, not this job,
keeps the last-N item list the ranker reads.

Run inside the Spark image:

    spark-submit --master local[1] \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9 \\
      spark/streaming_job.py
"""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

EVENT_SCHEMA = StructType(
    [
        StructField("user_id", LongType()),
        StructField("item_id", LongType()),
        StructField("liked", BooleanType()),
        StructField("watch_time_sec", DoubleType()),
        StructField("ts", LongType()),
        StructField("rating", DoubleType()),
        StructField("event_type", StringType()),
        StructField("genres", ArrayType(StringType())),
        StructField("title", StringType()),
    ]
)


def aggregate_events(events_df):
    """Cumulative counters. ``liked`` true contributes one like."""
    like_flag = F.when(F.col("liked") == F.lit(True), F.lit(1)).otherwise(F.lit(0))
    return events_df.groupBy("user_id").agg(
        F.count(F.lit(1)).alias("view_count"),
        F.sum(like_flag).alias("like_count"),
        F.sum("watch_time_sec").alias("watch_time_sum"),
        F.avg("watch_time_sec").alias("avg_watch_time_sec"),
        F.max("ts").alias("last_ts"),
    )


def write_snapshot(df, epoch_id: int, path: Path) -> None:
    rows = df.toJSON().collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    temporary.write_text(payload)
    temporary.replace(path)
    print(f"wrote batch {epoch_id} users={len(rows)} path={path}", flush=True)


def main() -> None:
    kafka = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.environ.get("KAFKA_TOPIC", "user_events")
    output = Path(os.environ.get("FEATURE_SNAPSHOT_PATH", "/data/stream_features.jsonl"))
    checkpoint = os.environ.get("CHECKPOINT_PATH", "/data/spark_ckpt")
    spark = SparkSession.builder.appName("shorts-feature-stream").getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )
    events = (
        raw.select(F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("event"))
        .select("event.*")
        .where(F.col("user_id").isNotNull())
    )
    aggregated = aggregate_events(events)

    def _write(batch_df, epoch_id: int) -> None:
        write_snapshot(batch_df, int(epoch_id), output)

    query = (
        aggregated.writeStream.outputMode("complete")
        .foreachBatch(_write)
        .option("checkpointLocation", checkpoint)
        .trigger(processingTime="5 seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
