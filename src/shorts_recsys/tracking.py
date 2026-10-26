"""Champion gate and MLflow run/registry logging."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

COMPARE_KEYS = (
    ("retrieval_plus_ranker", "recall@10"),
    ("retrieval_plus_ranker", "ndcg@10"),
    ("retrieval_plus_ranker", "mrr"),
)


def is_better(candidate: dict, champion: dict | None) -> bool:
    """Lexicographic improvement on the two-stage test metrics."""
    if champion is None:
        return True
    for section, key in COMPARE_KEYS:
        new_value = float(candidate[section][key])
        old_value = float(champion[section][key])
        if new_value > old_value + 1e-6:
            return True
        if new_value < old_value - 1e-6:
            return False
    return False


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def promote_staging(staging: Path, current: Path) -> None:
    current.parent.mkdir(parents=True, exist_ok=True)
    backup = current.parent / "current.prev"
    if backup.exists():
        shutil.rmtree(backup)
    if current.exists():
        current.rename(backup)
    staging.rename(current)


def _flatten(metrics: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for section in ("popularity", "retrieval", "retrieval_plus_ranker"):
        for key, value in metrics[section].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            name = f"{section}_{key}".replace("@", "_at_")
            flat[name] = float(value)
    for key in (
        "val_retrieval_recall_at_10",
        "epochs_ran",
        "best_epoch",
        "n_train_pairs",
        "n_eval_users",
        "n_users",
        "n_items",
    ):
        if key in metrics and isinstance(metrics[key], (int, float)):
            flat[key] = float(metrics[key])
    return flat


def _configure_experiment(tracking_uri: str, experiment: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        mlflow.set_experiment(experiment)
        return
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        if tracking_uri.startswith("sqlite:///"):
            database = tracking_uri[len("sqlite:///") :]
            root = Path(database).expanduser().resolve().parent / "artifacts"
        else:
            root = Path("mlartifacts")
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment, artifact_location=root.as_uri())
    mlflow.set_experiment(experiment)


def log_run(
    *,
    tracking_uri: str,
    experiment: str,
    params: dict,
    metrics: dict,
    artifact_dir: Path,
    ranker,
    register: bool,
    model_name: str = "shorts-ranker",
) -> dict:
    """Log params, metrics, artifacts, and optionally set the champion alias."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow.tracking import MlflowClient

    attempts = 5 if tracking_uri.startswith("http") else 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            _configure_experiment(tracking_uri, experiment)
            with mlflow.start_run(run_name="shorts-two-tower") as run:
                mlflow.log_params({key: value for key, value in params.items()})
                mlflow.log_metrics(_flatten(metrics))
                mlflow.log_artifacts(str(artifact_dir), artifact_path="recommender")
                mlflow.sklearn.log_model(ranker, name="ranker")
                run_id = run.info.run_id
            version = None
            if register:
                registered = mlflow.register_model(f"runs:/{run_id}/ranker", model_name)
                MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(
                    model_name, "champion", registered.version
                )
                version = str(registered.version)
            return {
                "run_id": run_id,
                "registered": register,
                "model_version": version,
                "model_name": model_name,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("MLflow log attempt %s failed: %s", attempt + 1, exc)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("MLflow logging failed") from last_error
