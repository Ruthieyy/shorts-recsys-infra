"""Paths, hyperparameters, and logging."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


def setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")


def experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "shorts-retrieval")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    embedding_dim: int = 64
    epochs: int = 60
    batch_size: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-6
    temperature: float = 0.07
    patience: int = 12
    min_epochs: int = 25
    min_user_positives: int = 5
    positive_threshold: float = 4.0
    short_length_sec: float = 30.0
    candidate_pool: int = 100
    ranker_negatives: int = 30
    recent_window: int = 5


# MovieLens ml-latest-small zip retrieved 2026-10-26 from files.grouplens.org.
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_ZIP_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"

KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "user_events")
FEATURE_NAMES = (
    "retrieval_score",
    "log_popularity",
    "genre_overlap",
    "recent_genre_affinity",
    "item_mean_rating",
)
