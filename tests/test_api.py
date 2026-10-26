import numpy as np
from fastapi.testclient import TestClient

from shorts_recsys.ranker import make_ranker
from shorts_recsys.serve import create_app

from .test_recommend import _save_toy


def test_api_recommend_health_and_metrics(tmp_path):
    model = make_ranker(0)
    features = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=float)
    model.fit(features, np.array([0, 1]))
    _save_toy(tmp_path / "current", model)
    # _save_toy writes the bundle directly into the given directory, while the
    # service loads ARTIFACT_DIR/current. Move the files down one level.
    bundle_dir = tmp_path / "root" / "current"
    bundle_dir.parent.mkdir()
    (tmp_path / "current").rename(bundle_dir)
    app = create_app(artifact_dir=bundle_dir.parent, redis_url="")
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert health.json()["n_users"] == 1
    cold = client.post("/recommend", json={"user_id": 999, "k": 2})
    assert cold.status_code == 200
    assert cold.json()["cold_start"] is True
    assert "latency_ms" in cold.json()
    known = client.post("/recommend", json={"user_id": 7, "k": 2})
    assert known.status_code == 200
    assert known.json()["cold_start"] is False
    assert len(known.json()["recommendations"]) == 2
    invalid = client.post("/recommend", json={"user_id": 7, "k": 0})
    assert invalid.status_code == 422
    missing = client.get("/features/7")
    assert missing.status_code == 404
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert "shorts_recommend_latency_seconds" in body
    assert "shorts_recommend_requests_total" in body


def test_api_503_without_model(tmp_path):
    app = create_app(artifact_dir=tmp_path / "empty", redis_url="")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    recommend = client.post("/recommend", json={"user_id": 1, "k": 5})
    assert recommend.status_code == 503
