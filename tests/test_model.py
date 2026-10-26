import numpy as np
import torch

from shorts_recsys.config import TrainConfig
from shorts_recsys.evaluate import evaluate_embeddings
from shorts_recsys.model import TwoTower, train_two_tower


class _Prepared:
    def __init__(self, pairs, seen):
        self.val_pairs = pairs
        self.seen_for_val = seen
        self.recent_for_val = {user: np.array([], dtype=np.int64) for user, _ in pairs}


def test_two_tower_shapes_and_unit_norm():
    model = TwoTower(n_users=3, n_items=4, n_genres=2, dim=8)
    users = torch.arange(3)
    items = torch.arange(4)
    genres = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    user_vec, item_vec = model(users, items, genres[items])
    assert user_vec.shape == (3, 8)
    assert item_vec.shape == (4, 8)
    assert torch.allclose(user_vec.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_two_tower_overfits_identity_pairs():
    n = 6
    genre_multi = np.zeros((n, 1), dtype=np.float32)
    config = TrainConfig(
        seed=7,
        embedding_dim=16,
        epochs=40,
        batch_size=n,
        lr=5e-2,
        weight_decay=0.0,
        temperature=0.05,
        patience=100,
        min_epochs=40,
    )
    user_vecs, item_vecs, info = train_two_tower(
        n_users=n,
        n_items=n,
        genre_multi=genre_multi,
        train_users=np.arange(n, dtype=np.int64),
        train_items=np.arange(n, dtype=np.int64),
        config=config,
    )
    assert info["epochs_ran"] == 40
    prepared = _Prepared(
        pairs=[(i, i) for i in range(n)],
        seen={i: np.array([], dtype=np.int64) for i in range(n)},
    )
    summary = evaluate_embeddings(user_vecs, item_vecs, prepared, "val")
    assert summary["recall@10"] == 1.0
    assert np.allclose(np.linalg.norm(item_vecs, axis=1), 1.0, atol=1e-5)
