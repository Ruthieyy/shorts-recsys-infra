"""Ranking metrics. A single relevant item makes recall@K an indicator of a hit."""

from __future__ import annotations

import numpy as np

RANK_MISS = 10**9


def summarize_ranks(ranks: np.ndarray, ks: tuple[int, ...] = (10, 50)) -> dict[str, float]:
    """Macro-average recall@K, NDCG@10, and MRR from 0-based ranks.

    A rank of ``RANK_MISS`` contributes nothing. NDCG uses the binary gain
    ``1 / log2(rank + 2)`` because each evaluated user has one relevant item,
    so IDCG@10 is 1.
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    n = int(ranks.size)
    out: dict[str, float] = {"n_users": float(n)}
    if n == 0:
        for k in ks:
            out[f"recall@{k}"] = 0.0
        out["ndcg@10"] = 0.0
        out["mrr"] = 0.0
        return out
    finite = ranks < RANK_MISS
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks < k))
    ndcg = np.zeros(n, dtype=np.float64)
    hit10 = ranks < 10
    ndcg[hit10] = 1.0 / np.log2(ranks[hit10].astype(np.float64) + 2.0)
    out["ndcg@10"] = float(np.mean(ndcg))
    mrr = np.zeros(n, dtype=np.float64)
    mrr[finite] = 1.0 / (ranks[finite].astype(np.float64) + 1.0)
    out["mrr"] = float(np.mean(mrr))
    return out


def ranks_from_scores(
    scores: np.ndarray,
    targets: np.ndarray,
    seen: list[np.ndarray],
) -> np.ndarray:
    """0-based rank of each target after masking seen item columns."""
    scores = np.array(scores, dtype=np.float64, copy=True)
    n_rows, n_cols = scores.shape
    for row, cols in enumerate(seen):
        if len(cols):
            scores[row, cols] = -np.inf
    targets = np.asarray(targets, dtype=np.int64)
    target_scores = scores[np.arange(n_rows), targets]
    order = np.argsort(-scores, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(n_rows)[:, None], order] = np.arange(n_cols)[None, :]
    ranks = positions[np.arange(n_rows), targets]
    ranks = ranks.astype(np.int64, copy=False)
    ranks[~np.isfinite(target_scores)] = RANK_MISS
    return ranks
