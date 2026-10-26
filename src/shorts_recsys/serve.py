"""FastAPI recommendation service with Prometheus latency metrics."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from shorts_recsys.artifacts import load_bundle
from shorts_recsys.config import setup_logging
from shorts_recsys.features import parse_redis_hash
from shorts_recsys.recommend import recommend

LATENCY = Histogram(
    "shorts_recommend_latency_seconds",
    "Recommendation handler latency in seconds.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
REQUESTS = Counter(
    "shorts_recommend_requests_total",
    "Recommendation requests by outcome.",
    ["status"],
)


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="MovieLens user id")
    k: int = Field(10, ge=1, le=50)


class OnlineFeatures:
    def __init__(self, url: str | None) -> None:
        self.url = url or None
        self._client = None
        self._down_until = 0.0

    def lookup(self, user_id: int) -> tuple[list[int] | None, dict | None]:
        if not self.url or time.monotonic() < self._down_until:
            return None, None
        try:
            client = self._connect()
            raw = client.hgetall(f"user:{user_id}")
        except Exception:
            self._client = None
            self._down_until = time.monotonic() + 5.0
            return None, None
        if not raw:
            return None, None
        parsed = parse_redis_hash(raw)
        return parsed["recent_item_ids"], parsed

    def _connect(self):
        if self._client is not None:
            return self._client
        import redis

        self._client = redis.Redis.from_url(
            self.url,
            decode_responses=True,
            socket_timeout=0.05,
            socket_connect_timeout=0.05,
        )
        return self._client


def create_app(artifact_dir: Path | None = None, redis_url: str | None = None) -> FastAPI:
    setup_logging()
    root = artifact_dir or Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL") or None
    bundle = load_bundle(root / "current")
    online = OnlineFeatures(redis_url)
    app = FastAPI(title="Shorts recommendation service", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        if bundle is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        metrics = bundle.manifest.get("metrics", {})
        two_stage = metrics.get("retrieval_plus_ranker", {})
        return {
            "status": "ok",
            "model_loaded": True,
            "n_users": int(bundle.manifest.get("n_users", 0)),
            "n_items": int(bundle.manifest.get("n_items", 0)),
            "champion_recall_at_10": two_stage.get("recall@10"),
            "champion_ndcg_at_10": two_stage.get("ndcg@10"),
        }

    @app.get("/model")
    def model() -> dict:
        if bundle is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return bundle.manifest

    @app.get("/features/{user_id}")
    def features(user_id: int) -> dict:
        recent, parsed = online.lookup(user_id)
        if parsed is None:
            raise HTTPException(status_code=404, detail="no online features for user")
        del recent
        return parsed

    @app.post("/recommend")
    def recommend_endpoint(body: RecommendRequest) -> dict:
        status = "error"
        started = time.perf_counter()
        try:
            if bundle is None:
                status = "unavailable"
                raise HTTPException(status_code=503, detail="model is not loaded")
            recent, parsed = online.lookup(body.user_id)
            result = recommend(bundle, body.user_id, body.k, online_recent=recent)
            result["online_features"] = parsed
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            status = "cold_start" if result["cold_start"] else "ok"
            return result
        finally:
            LATENCY.observe(time.perf_counter() - started)
            REQUESTS.labels(status=status).inc()

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.state.bundle = bundle
    return app


app = create_app()


def main() -> None:
    import uvicorn

    # Pass the app object. `python -m shorts_recsys.serve` loads this file as
    # __main__, and a string import would load it a second time and register
    # the Prometheus collectors twice.
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
