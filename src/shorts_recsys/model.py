"""Two-tower retrieval model trained with in-batch softmax."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shorts_recsys.config import TrainConfig

logger = logging.getLogger(__name__)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_genres: int, dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.genre_emb = nn.Embedding(max(n_genres, 1), dim)
        self.n_genres = n_genres
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.genre_emb.weight, std=0.02)

    def user_vectors(self, users: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.user_emb(users), dim=-1)

    def item_vectors(self, items: torch.Tensor, genre_multi: torch.Tensor) -> torch.Tensor:
        if self.n_genres == 0:
            return nn.functional.normalize(self.item_emb(items), dim=-1)
        genre_sum = genre_multi @ self.genre_emb.weight
        counts = genre_multi.sum(dim=1, keepdim=True)
        genre_mean = genre_sum / counts.clamp(min=1.0)
        has_genre = (counts > 0).to(dtype=genre_mean.dtype)
        return nn.functional.normalize(self.item_emb(items) + has_genre * genre_mean, dim=-1)

    def forward(
        self, users: torch.Tensor, items: torch.Tensor, genre_multi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_vectors(users), self.item_vectors(items, genre_multi)


def inbatch_softmax(
    user_vec: torch.Tensor, item_vec: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = user_vec @ item_vec.T / temperature
    labels = torch.arange(user_vec.size(0), device=user_vec.device)
    return nn.functional.cross_entropy(logits, labels)


@torch.no_grad()
def export_embeddings(model: TwoTower, genre_multi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    users = torch.arange(model.user_emb.num_embeddings, device=device)
    user_vecs = model.user_vectors(users).cpu().numpy().astype(np.float32)
    items = torch.arange(model.item_emb.num_embeddings, device=device)
    genres = torch.from_numpy(np.ascontiguousarray(genre_multi, dtype=np.float32)).to(device)
    item_vecs = model.item_vectors(items, genres).cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(user_vecs), np.ascontiguousarray(item_vecs)


def train_two_tower(
    *,
    n_users: int,
    n_items: int,
    genre_multi: np.ndarray,
    train_users: np.ndarray,
    train_items: np.ndarray,
    config: TrainConfig,
    score_fn=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train and return the embedding checkpoint with the best ``score_fn``.

    ``score_fn(user_vecs, item_vecs) -> float`` is maximized. When it is omitted,
    the negative training loss is the score, so a lower loss wins.
    """
    if len(train_users) == 0:
        raise RuntimeError("empty training pairs")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cpu")
    torch.set_num_threads(min(4, torch.get_num_threads() or 1))
    model = TwoTower(n_users, n_items, genre_multi.shape[1], config.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    users_t = torch.from_numpy(train_users.astype(np.int64, copy=False))
    items_t = torch.from_numpy(train_items.astype(np.int64, copy=False))
    genre_t = torch.from_numpy(np.ascontiguousarray(genre_multi, dtype=np.float32))
    dataset = TensorDataset(users_t, items_t)
    batch_size = max(1, min(config.batch_size, len(dataset)))
    drop_last = len(dataset) >= batch_size * 2
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )
    best_score = -np.inf
    best_users: np.ndarray | None = None
    best_items: np.ndarray | None = None
    best_epoch = 0
    stale = 0
    last_loss = 0.0
    epochs_ran = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch_users, batch_items in loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            optimizer.zero_grad(set_to_none=True)
            user_vec, item_vec = model(batch_users, batch_items, genre_t[batch_items])
            loss = inbatch_softmax(user_vec, item_vec, config.temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        last_loss = total / max(steps, 1)
        epochs_ran = epoch
        logger.info("epoch %s train_loss=%.4f", epoch, last_loss)
        user_vecs, item_vecs = export_embeddings(model, genre_multi)
        score = float(score_fn(user_vecs, item_vecs)) if score_fn is not None else -last_loss
        if score > best_score + 1e-6:
            best_score = score
            best_users = user_vecs
            best_items = item_vecs
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= config.min_epochs and stale >= config.patience:
            break
    assert best_users is not None and best_items is not None
    info = {
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "epochs_ran": epochs_ran,
        "final_loss": float(last_loss),
    }
    return best_users, best_items, info
