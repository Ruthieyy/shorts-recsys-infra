"""Offline evaluation for popularity, retrieval, and the two-stage ranker."""

from __future__ import annotations

import numpy as np

from shorts_recsys.metrics import RANK_MISS, ranks_from_scores, summarize_ranks
from shorts_recsys.ranker import distribution, features_for_items


def topk_indices(scores: np.ndarray, k: int) -> list[int]:
    finite = np.isfinite(scores)
    if not finite.any() or k <= 0:
        return []
    idx = np.flatnonzero(finite)
    values = scores[idx]
    keep = min(k, len(idx))
    if keep == len(idx):
        order = np.argsort(-values, kind="mergesort")
    else:
        part = np.argpartition(-values, keep - 1)[:keep]
        order = part[np.argsort(-values[part], kind="mergesort")]
    return [int(idx[i]) for i in order[:keep]]


def _pairs_and_seen(prepared, split: str):
    if split == "test":
        return prepared.test_pairs, prepared.seen_for_test, prepared.recent_for_test
    if split == "val":
        return prepared.val_pairs, prepared.seen_for_val, prepared.recent_for_val
    raise ValueError(f"unknown split {split}")


def evaluate_embeddings(user_vecs: np.ndarray, item_vecs: np.ndarray, prepared, split: str) -> dict:
    pairs, seen_map, _ = _pairs_and_seen(prepared, split)
    if not pairs:
        return summarize_ranks(np.array([], dtype=np.int64))
    user_rows = np.asarray([user for user, _ in pairs], dtype=np.int64)
    targets = np.asarray([item for _, item in pairs], dtype=np.int64)
    scores = user_vecs[user_rows] @ item_vecs.T
    seen = [seen_map[int(user)] for user, _ in pairs]
    ranks = ranks_from_scores(scores, targets, seen)
    return summarize_ranks(ranks)


def evaluate_popularity(prepared, split: str) -> dict:
    pairs, seen_map, _ = _pairs_and_seen(prepared, split)
    if not pairs:
        return summarize_ranks(np.array([], dtype=np.int64))
    n_items = len(prepared.item_ids)
    base = prepared.popularity.astype(np.float64) - np.arange(n_items, dtype=np.float64) * 1e-12
    scores = np.tile(base, (len(pairs), 1))
    targets = np.asarray([item for _, item in pairs], dtype=np.int64)
    seen = [seen_map[int(user)] for user, _ in pairs]
    ranks = ranks_from_scores(scores, targets, seen)
    return summarize_ranks(ranks)


def evaluate_two_stage(
    user_vecs: np.ndarray,
    item_vecs: np.ndarray,
    ranker,
    prepared,
    split: str,
    candidate_pool: int,
) -> dict:
    pairs, seen_map, recent_map = _pairs_and_seen(prepared, split)
    if not pairs:
        return summarize_ranks(np.array([], dtype=np.int64))
    ranks = np.empty(len(pairs), dtype=np.int64)
    for row, (user_row, target) in enumerate(pairs):
        scores = np.ascontiguousarray(user_vecs[user_row] @ item_vecs.T)
        seen = seen_map[user_row]
        if len(seen):
            scores[seen] = -np.inf
        top = topk_indices(scores, candidate_pool)
        top_set = set(top)
        if target not in top_set:
            ranks[row] = RANK_MISS
            continue
        retrieval = user_vecs[user_row] @ item_vecs[top].T
        user_dist = distribution(prepared.profile_items[user_row], prepared.genre_multi)
        recent_dist = distribution(recent_map[user_row], prepared.genre_multi)
        features = features_for_items(
            top,
            retrieval,
            prepared.log_popularity,
            prepared.mean_rating,
            prepared.genre_multi,
            user_dist,
            recent_dist,
        )
        pred = ranker.predict_proba(features)[:, 1]
        order = np.argsort(-pred, kind="mergesort")
        ranked = [top[int(i)] for i in order]
        ranks[row] = ranked.index(int(target))
    return summarize_ranks(ranks)
