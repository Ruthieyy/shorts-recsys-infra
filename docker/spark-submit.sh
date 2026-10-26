#!/bin/bash
# Submit the streaming job with the vendored Kafka connector jars so startup
# does not download from Maven.
set -euo pipefail
# The Spark image defaults to uid 185, which cannot create the checkpoint on a
# root-owned named volume. Compose runs this service as root; mkdir here so
# Hadoop does not fail the query before the first micro-batch.
mkdir -p /data/spark_ckpt
jars=$(find /opt/jars -name '*.jar' | paste -sd, -)
if [[ -z "${jars}" ]]; then
  echo "no jars in /opt/jars" >&2
  exit 1
fi
exec /opt/spark/bin/spark-submit \
  --master local[1] \
  --driver-memory 1g \
  --jars "${jars}" \
  /opt/job/streaming_job.py
