"""Train the two-tower and ranker, then promote the bundle if it beats the champion."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from shorts_recsys.artifacts import save_bundle
from shorts_recsys.config import TrainConfig, env_path, experiment_name, setup_logging, tracking_uri
from shorts_recsys.data import DATASET_NAME, ensure_dataset, load_tables, prepare, sha256_file
from shorts_recsys.evaluate import evaluate_embeddings, evaluate_popularity, evaluate_two_stage
from shorts_recsys.model import train_two_tower
from shorts_recsys.ranker import fit_ranker
from shorts_recsys.tracking import is_better, load_metrics, log_run, promote_staging, write_json

logger = logging.getLogger(__name__)


def _section(summary: dict) -> dict:
    return {
        "mrr": summary["mrr"],
        "n_users": int(summary["n_users"]),
        "ndcg@10": summary["ndcg@10"],
        "recall@10": summary["recall@10"],
        "recall@50": summary["recall@50"],
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "unknown"


def build_metrics(
    prepared, popularity, retrieval, reranked, train_info, val_recall, ratings_sha, seed: int
) -> dict:
    return {
        "best_epoch": int(train_info["best_epoch"]),
        "dataset": DATASET_NAME,
        "epochs_ran": int(train_info["epochs_ran"]),
        "final_loss": float(train_info["final_loss"]),
        "n_eval_users": int(retrieval["n_users"]),
        "n_items": int(len(prepared.item_ids)),
        "n_ratings": int(prepared.n_ratings),
        "n_skipped_test_target_not_in_catalog": int(prepared.n_skipped_test),
        "n_train_pairs": int(len(prepared.train_user_idx)),
        "n_users": int(len(prepared.user_ids)),
        "n_users_qualified": int(prepared.n_users_qualified),
        "package_versions": {
            "faiss": _package_version("faiss-cpu"),
            "mlflow": _package_version("mlflow"),
            "numpy": _package_version("numpy"),
            "scikit-learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
        },
        "popularity": _section(popularity),
        "protocol": (
            "MovieLens ml-latest-small implicit leave-one-out. "
            "A positive is rating >= 4. Users need at least 5 positives. "
            "The last positive is the test item and the previous positive is validation. "
            "The two-tower trains on earlier positives. At test time every other rated "
            "title is removed from the candidate list. The ranker reorders the top 100 "
            "retrieved titles. Users whose test title never appears as a training positive "
            "for anyone are excluded from the reported averages."
        ),
        "ratings_sha256": ratings_sha,
        "retrieval": _section(retrieval),
        "retrieval_plus_ranker": _section(reranked),
        "seed": seed,
        "trained_at": datetime.now(UTC).isoformat(),
        "val_retrieval_recall_at_10": float(val_recall),
        "watch_time_proxy": (
            "MovieLens has no dwell time. Replay events set "
            "watch_time_sec = (rating / 5) * 30 and liked = rating >= 4. "
            "Retrieval and the ranker do not train on watch_time_sec."
        ),
    }


def run(
    config: TrainConfig, data_dir: Path, artifact_dir: Path, results_path: Path, force: bool
) -> dict:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    dataset_dir = ensure_dataset(data_dir)
    ratings_sha = sha256_file(dataset_dir / "ratings.csv")
    ratings, movies = load_tables(dataset_dir)
    prepared = prepare(ratings, movies, config)
    logger.info(
        "prepared users=%s items=%s train_pairs=%s val=%s test=%s skipped_test=%s",
        len(prepared.user_ids),
        len(prepared.item_ids),
        len(prepared.train_user_idx),
        len(prepared.val_pairs),
        len(prepared.test_pairs),
        prepared.n_skipped_test,
    )

    def val_recall(user_vecs: np.ndarray, item_vecs: np.ndarray) -> float:
        summary = evaluate_embeddings(user_vecs, item_vecs, prepared, "val")
        logger.info("val recall@10=%.4f ndcg@10=%.4f", summary["recall@10"], summary["ndcg@10"])
        return float(summary["recall@10"])

    user_vecs, item_vecs, train_info = train_two_tower(
        n_users=len(prepared.user_ids),
        n_items=len(prepared.item_ids),
        genre_multi=prepared.genre_multi,
        train_users=prepared.train_user_idx,
        train_items=prepared.train_item_idx,
        config=config,
        score_fn=val_recall,
    )
    logger.info(
        "selected epoch %s/%s val_recall@10=%.4f",
        train_info["best_epoch"],
        train_info["epochs_ran"],
        train_info["best_score"],
    )
    ranker = fit_ranker(prepared, user_vecs, item_vecs, config)
    popularity = evaluate_popularity(prepared, "test")
    retrieval = evaluate_embeddings(user_vecs, item_vecs, prepared, "test")
    reranked = evaluate_two_stage(
        user_vecs, item_vecs, ranker, prepared, "test", config.candidate_pool
    )
    metrics = build_metrics(
        prepared,
        popularity,
        retrieval,
        reranked,
        train_info,
        train_info["best_score"],
        ratings_sha,
        config.seed,
    )
    logger.info(
        "test popularity recall@10=%.4f retrieval recall@10=%.4f ndcg@10=%.4f "
        "reranked recall@10=%.4f ndcg@10=%.4f",
        popularity["recall@10"],
        retrieval["recall@10"],
        retrieval["ndcg@10"],
        reranked["recall@10"],
        reranked["ndcg@10"],
    )

    champion = load_metrics(results_path)
    has_bundle = (artifact_dir / "current" / "manifest.json").exists()
    better = is_better(metrics, champion)
    # The published metrics file is replaced only when the new test numbers win.
    # A checkout that has those numbers but no local bundle still gets a serving
    # directory so the API can start; that write does not move the MLflow alias.
    write_metrics = force or champion is None or better
    write_bundle = force or (not has_bundle) or better
    register_alias = write_metrics
    staging = artifact_dir / "staging"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    manifest = {
        "candidate_pool": config.candidate_pool,
        "dataset": DATASET_NAME,
        "metrics": metrics,
        "positive_threshold": config.positive_threshold,
        "recent_window": config.recent_window,
        "seed": config.seed,
    }
    save_bundle(
        staging,
        user_ids=prepared.user_ids,
        item_ids=prepared.item_ids,
        user_embeddings=user_vecs,
        item_embeddings=item_vecs,
        genre_names=prepared.genre_names,
        genre_multi=prepared.genre_multi,
        log_popularity=prepared.log_popularity,
        mean_rating=prepared.mean_rating,
        popularity=prepared.popularity,
        titles=prepared.titles,
        genres=prepared.genres,
        history_raw=prepared.history_raw,
        profile_raw=prepared.profile_raw,
        recent_for_serve=prepared.recent_for_serve,
        ranker=ranker,
        manifest=manifest,
    )
    params = {
        "batch_size": config.batch_size,
        "candidate_pool": config.candidate_pool,
        "dataset": DATASET_NAME,
        "embedding_dim": config.embedding_dim,
        "epochs": config.epochs,
        "lr": config.lr,
        "positive_threshold": config.positive_threshold,
        "seed": config.seed,
        "temperature": config.temperature,
    }
    mlflow_error: Exception | None = None
    try:
        mlflow_info = log_run(
            tracking_uri=tracking_uri(),
            experiment=experiment_name(),
            params=params,
            metrics=metrics,
            artifact_dir=staging,
            ranker=ranker,
            register=register_alias,
        )
    except Exception as exc:
        logger.exception("MLflow logging failed")
        mlflow_info = {"run_id": None, "registered": False, "error": str(exc)}
        mlflow_error = exc
    if write_bundle:
        promote_staging(staging, artifact_dir / "current")
    else:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
    if write_metrics:
        write_json(results_path, metrics)
        if champion is None:
            reason = "no champion yet"
        elif force and not better:
            reason = "forced replacement of the champion"
        else:
            reason = "beat champion on two-stage test metrics"
    else:
        reason = "did not beat champion on two-stage test metrics"
        if write_bundle:
            reason = "wrote a local bundle; metrics did not beat the published champion"
    decision = {
        "bundle_written": write_bundle,
        "champion_updated": write_metrics,
        "forced": force,
        "mlflow": mlflow_info,
        "reason": reason,
        "retrieval_plus_ranker": metrics["retrieval_plus_ranker"],
        "retrieval": metrics["retrieval"],
        "popularity": metrics["popularity"],
    }
    write_json(artifact_dir / "last_run.json", decision)
    logger.info("champion_updated=%s reason=%s", better, reason)
    if mlflow_error is not None and os.environ.get("MLFLOW_REQUIRED", "1") == "1":
        raise mlflow_error
    return decision


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Train and maybe promote the short-video recommender"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace the champion even if metrics regress"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    decision = run(
        TrainConfig(),
        args.data_dir or env_path("DATA_DIR", "data"),
        args.artifact_dir or env_path("ARTIFACT_DIR", "artifacts"),
        args.results or env_path("RESULTS_PATH", "results/offline_metrics.json"),
        args.force,
    )
    print(decision["reason"])
    section = decision["retrieval_plus_ranker"]
    print(
        f"two-stage recall@10={section['recall@10']:.4f} "
        f"ndcg@10={section['ndcg@10']:.4f} mrr={section['mrr']:.4f}"
    )


if __name__ == "__main__":
    main()
