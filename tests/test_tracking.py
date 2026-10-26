import os

import numpy as np

from shorts_recsys.ranker import make_ranker
from shorts_recsys.tracking import is_better, log_run, promote_staging, write_json


def _metrics(recall: float, ndcg: float, mrr: float) -> dict:
    section = {"mrr": mrr, "n_users": 10, "ndcg@10": ndcg, "recall@10": recall, "recall@50": recall}
    return {
        "best_epoch": 1,
        "epochs_ran": 1,
        "n_eval_users": 10,
        "n_items": 4,
        "n_train_pairs": 8,
        "n_users": 2,
        "popularity": section,
        "retrieval": section,
        "retrieval_plus_ranker": section,
        "val_retrieval_recall_at_10": recall,
    }


def test_gate_is_lexicographic():
    champion = _metrics(0.20, 0.10, 0.10)
    assert is_better(_metrics(0.20, 0.10, 0.10), None)
    assert not is_better(_metrics(0.19, 0.90, 0.90), champion)
    assert is_better(_metrics(0.21, 0.01, 0.01), champion)
    assert not is_better(_metrics(0.20, 0.10, 0.10), champion)
    assert is_better(_metrics(0.20, 0.11, 0.01), champion)
    assert is_better(_metrics(0.20, 0.10, 0.12), champion)


def test_promote_replaces_current(tmp_path):
    staging = tmp_path / "staging"
    current = tmp_path / "current"
    staging.mkdir()
    (staging / "marker.txt").write_text("new")
    current.mkdir()
    (current / "marker.txt").write_text("old")
    promote_staging(staging, current)
    assert (current / "marker.txt").read_text() == "new"
    assert (tmp_path / "current.prev" / "marker.txt").read_text() == "old"
    assert not staging.exists()


def test_mlflow_registry_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    model = make_ranker(0)
    model.fit(np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=float), np.array([0, 1]))
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "note.txt").write_text("bundle")
    database = tmp_path / "mlflow.db"
    info = log_run(
        tracking_uri=f"sqlite:///{database}",
        experiment="shorts-test",
        params={"seed": 0, "dataset": "toy"},
        metrics=_metrics(0.3, 0.2, 0.2),
        artifact_dir=artifact_dir,
        ranker=model,
        register=True,
        model_name="shorts-ranker-test",
    )
    from mlflow.tracking import MlflowClient

    alias = MlflowClient(tracking_uri=f"sqlite:///{database}").get_model_version_by_alias(
        "shorts-ranker-test", "champion"
    )
    assert info["registered"] is True
    assert str(alias.version) == str(info["model_version"])


def test_write_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "metrics.json"
    write_json(path, {"recall@10": 0.1})
    assert path.read_text().strip().startswith("{")
