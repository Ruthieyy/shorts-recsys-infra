"""Measure live /recommend latency and write results/latency.json."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import urllib.request
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no samples")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _post(url: str, user_id: int, k: int) -> tuple[int, float]:
    payload = json.dumps({"user_id": user_id, "k": k}).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/recommend",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
        status = response.status
    return status, (time.perf_counter() - started) * 1000.0


def measure(base_url: str, user_ids: list[int], requests: int, k: int) -> dict:
    samples_ms: list[float] = []
    for index in range(requests):
        user_id = user_ids[index % len(user_ids)]
        status, elapsed = _post(base_url, user_id, k)
        if status != 200:
            raise RuntimeError(f"unexpected status {status}")
        samples_ms.append(elapsed)
    return {
        "base_url": base_url,
        "k": k,
        "n": len(samples_ms),
        "user_ids": user_ids,
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe recommendation latency")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--user-ids", default="1,2,3,4,5")
    parser.add_argument("--output", type=Path, default=Path("results/latency.json"))
    args = parser.parse_args()
    user_ids = [int(part) for part in args.user_ids.split(",") if part]
    # One warmup request so import/cache effects are not in the sample.
    _post(args.url, user_ids[0], args.k)
    report = measure(args.url, user_ids, args.requests, args.k)
    report["cpu_count"] = os.cpu_count()
    report["python"] = platform.python_version()
    report["platform"] = platform.platform()
    report["note"] = (
        "Client-side wall time for POST /recommend, excluding one warmup request. "
        "Single uvicorn process, Redis unset, model bundle loaded from local artifacts."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"n={report['n']} p50_ms={report['p50_ms']:.3f} p95_ms={report['p95_ms']:.3f} "
        f"mean_ms={report['mean_ms']:.3f}"
    )


if __name__ == "__main__":
    main()
