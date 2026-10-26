import math

import numpy as np

from shorts_recsys.metrics import RANK_MISS, ranks_from_scores, summarize_ranks


def test_summarize_ranks_matches_hand_calculation():
    ranks = np.array([0, 2, 10], dtype=np.int64)
    summary = summarize_ranks(ranks)
    assert summary["recall@10"] == 2 / 3
    assert summary["recall@50"] == 1.0
    expected_ndcg = (1.0 / math.log2(2) + 1.0 / math.log2(4) + 0.0) / 3
    expected_mrr = (1.0 + 1.0 / 3 + 1.0 / 11) / 3
    assert summary["ndcg@10"] == expected_ndcg
    assert summary["mrr"] == expected_mrr
    assert summary["n_users"] == 3.0


def test_rank_miss_contributes_zero():
    summary = summarize_ranks(np.array([RANK_MISS], dtype=np.int64))
    assert summary["recall@10"] == 0.0
    assert summary["ndcg@10"] == 0.0
    assert summary["mrr"] == 0.0


def test_ranks_mask_seen_items_and_keep_target():
    scores = np.array(
        [
            [0.1, 0.9, 0.2, 0.0],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    targets = np.array([2, 0])
    seen = [np.array([1]), np.array([], dtype=np.int64)]
    ranks = ranks_from_scores(scores, targets, seen)
    # Row 0: item 1 is masked, so the order is item 2 (0.2), item 0 (0.1), item 3.
    assert int(ranks[0]) == 0
    # Row 1: item 0 is the highest score.
    assert int(ranks[1]) == 0
