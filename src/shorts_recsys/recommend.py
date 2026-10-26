"""Pure recommendation function used by FastAPI and tests."""

from __future__ import annotations

import numpy as np

from shorts_recsys.artifacts import Bundle
from shorts_recsys.config import TrainConfig
from shorts_recsys.ranker import distribution, features_for_items


def _item_payload(
    bundle: Bundle, item_row: int, score: float | None, retrieval_score: float | None
) -> dict:
    raw_id = int(bundle.item_ids[item_row])
    return {
        "item_id": raw_id,
        "title": bundle.titles.get(raw_id, str(raw_id)),
        "genres": bundle.genres.get(raw_id, []),
        "score": None if score is None else round(float(score), 6),
        "retrieval_score": None if retrieval_score is None else round(float(retrieval_score), 6),
    }


def _rows_for(raw_ids: list[int], item_index: dict[int, int]) -> np.ndarray:
    return np.asarray([item_index[item] for item in raw_ids if item in item_index], dtype=np.int64)


def recommend(
    bundle: Bundle,
    user_id: int,
    k: int,
    online_recent: list[int] | None = None,
) -> dict:
    pool = int(bundle.manifest.get("candidate_pool", TrainConfig.candidate_pool))
    window = int(bundle.manifest.get("recent_window", TrainConfig.recent_window))
    if user_id not in bundle.user_index:
        recommendations = []
        watched = bundle.history.get(user_id, set())
        for raw_id in bundle.popular_item_ids:
            if raw_id in watched:
                continue
            row = bundle.item_index.get(raw_id)
            if row is None:
                continue
            recommendations.append(_item_payload(bundle, row, None, None))
            if len(recommendations) >= k:
                break
        return {
            "user_id": user_id,
            "cold_start": True,
            "used_online_recent": False,
            "recommendations": recommendations,
        }

    user_row = bundle.user_index[user_id]
    watched = bundle.history.get(user_id, set())
    search_k = min(len(bundle.item_ids), max(pool, k + len(watched) + 50))
    query = np.ascontiguousarray(bundle.user_embeddings[user_row : user_row + 1], dtype=np.float32)
    scores, indices = bundle.index.search(query, search_k)
    candidates: list[int] = []
    retrieval_scores: list[float] = []
    for index, score in zip(indices[0], scores[0], strict=True):
        if int(index) < 0:
            continue
        raw_id = int(bundle.item_ids[int(index)])
        if raw_id in watched:
            continue
        candidates.append(int(index))
        retrieval_scores.append(float(score))
        if len(candidates) >= pool:
            break
    recent_raw = online_recent if online_recent else bundle.recent_items.get(user_id, [])
    recent_raw = list(recent_raw)[-window:]
    user_dist = distribution(
        _rows_for(bundle.profile_items.get(user_id, []), bundle.item_index), bundle.genre_multi
    )
    recent_dist = distribution(_rows_for(recent_raw, bundle.item_index), bundle.genre_multi)
    features = features_for_items(
        candidates,
        np.asarray(retrieval_scores, dtype=np.float64),
        bundle.log_popularity,
        bundle.mean_rating,
        bundle.genre_multi,
        user_dist,
        recent_dist,
    )
    if len(candidates) == 0:
        ordered: list[int] = []
        rank_scores = np.array([], dtype=np.float64)
    else:
        rank_scores = bundle.ranker.predict_proba(features)[:, 1]
        ordered = [candidates[int(i)] for i in np.argsort(-rank_scores, kind="mergesort")]
    recommendations = []
    retrieval_by_row = {row: score for row, score in zip(candidates, retrieval_scores, strict=True)}
    rank_by_row = {row: float(score) for row, score in zip(candidates, rank_scores, strict=True)}
    for row in ordered[:k]:
        recommendations.append(_item_payload(bundle, row, rank_by_row[row], retrieval_by_row[row]))
    return {
        "user_id": user_id,
        "cold_start": False,
        "used_online_recent": online_recent is not None,
        "recommendations": recommendations,
    }
