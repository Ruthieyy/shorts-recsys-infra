"""Lightweight pointwise ranker over retrieval candidates."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shorts_recsys.config import FEATURE_NAMES, TrainConfig


def distribution(item_rows: np.ndarray | list[int], genre_multi: np.ndarray) -> np.ndarray:
    """Normalized genre mass of a sequence of catalog row indices."""
    n_genres = int(genre_multi.shape[1]) if genre_multi.ndim == 2 else 0
    if n_genres == 0 or item_rows is None or len(item_rows) == 0:
        return np.zeros(n_genres, dtype=np.float64)
    rows = np.asarray(item_rows, dtype=np.int64)
    counts = genre_multi[rows].sum(axis=0).astype(np.float64, copy=False)
    total = float(counts.sum())
    if total <= 0.0:
        return np.zeros(n_genres, dtype=np.float64)
    return counts / total


def features_for_items(
    item_rows: np.ndarray | list[int],
    retrieval_scores: np.ndarray,
    log_popularity: np.ndarray,
    mean_rating: np.ndarray,
    genre_multi: np.ndarray,
    user_dist: np.ndarray,
    recent_dist: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(item_rows, dtype=np.int64)
    scores = np.asarray(retrieval_scores, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    multi = genre_multi[rows]
    counts = multi.sum(axis=1)
    overlap_mass = multi @ user_dist
    recent_mass = multi @ recent_dist
    overlap = np.divide(overlap_mass, counts, out=np.zeros(rows.size), where=counts > 0)
    recent = np.divide(recent_mass, counts, out=np.zeros(rows.size), where=counts > 0)
    return np.column_stack(
        [
            scores,
            log_popularity[rows],
            overlap,
            recent,
            mean_rating[rows],
        ]
    ).astype(np.float64, copy=False)


def make_ranker(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_ranker(
    prepared, user_vecs: np.ndarray, item_vecs: np.ndarray, config: TrainConfig
) -> Pipeline:
    """Fit on validation users. Positives are val items; negatives are retrieved candidates."""
    from shorts_recsys.evaluate import topk_indices

    rng = np.random.default_rng(config.seed)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for user_row, target in prepared.val_pairs:
        scores = np.ascontiguousarray(user_vecs[user_row] @ item_vecs.T)
        seen = prepared.seen_for_val[user_row]
        if len(seen):
            scores[seen] = -np.inf
        top = topk_indices(scores, config.candidate_pool)
        if target not in top:
            top = [*top[: max(config.candidate_pool - 1, 0)], int(target)]
        negatives = [item for item in top if item != int(target)]
        if len(negatives) > config.ranker_negatives:
            pick = rng.choice(len(negatives), size=config.ranker_negatives, replace=False)
            negatives = [negatives[int(i)] for i in pick]
        if not negatives:
            continue
        rows = [int(target), *negatives]
        retrieval = user_vecs[user_row] @ item_vecs[rows].T
        user_dist = distribution(prepared.profile_items[user_row], prepared.genre_multi)
        recent_dist = distribution(prepared.recent_for_val[user_row], prepared.genre_multi)
        block = features_for_items(
            rows,
            retrieval,
            prepared.log_popularity,
            prepared.mean_rating,
            prepared.genre_multi,
            user_dist,
            recent_dist,
        )
        blocks.append(block)
        labels.append(np.array([1, *([0] * len(negatives))], dtype=np.int64))
    if not blocks:
        raise RuntimeError("ranker received no training rows")
    x_train = np.vstack(blocks)
    y_train = np.concatenate(labels)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("ranker training labels have a single class")
    model = make_ranker(config.seed)
    model.fit(x_train, y_train)
    return model
