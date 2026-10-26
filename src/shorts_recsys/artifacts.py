"""Save and load the serving bundle written by training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np

from shorts_recsys.config import FEATURE_NAMES


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path):
    return json.loads(path.read_text())


@dataclass
class Bundle:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict[int, int]
    item_index: dict[int, int]
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    index: Any
    genre_names: list[str]
    genre_multi: np.ndarray
    log_popularity: np.ndarray
    mean_rating: np.ndarray
    popularity: np.ndarray
    titles: dict[int, str]
    genres: dict[int, list[str]]
    history: dict[int, set[int]]
    profile_items: dict[int, list[int]]
    recent_items: dict[int, list[int]]
    ranker: Any
    popular_item_ids: list[int]
    manifest: dict


def save_bundle(
    directory: Path,
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    genre_names: list[str],
    genre_multi: np.ndarray,
    log_popularity: np.ndarray,
    mean_rating: np.ndarray,
    popularity: np.ndarray,
    titles: dict[int, str],
    genres: dict[int, list[str]],
    history_raw: dict[int, list[int]],
    profile_raw: dict[int, list[int]],
    recent_for_serve: dict[int, list[int]],
    ranker,
    manifest: dict,
) -> None:
    if directory.exists():
        for child in directory.iterdir():
            if child.is_file():
                child.unlink()
    else:
        directory.mkdir(parents=True, exist_ok=True)
    users = _normalize(user_embeddings)
    items = _normalize(item_embeddings)
    np.save(directory / "user_embeddings.npy", users)
    np.save(directory / "item_embeddings.npy", items)
    np.save(directory / "genre_multi.npy", np.ascontiguousarray(genre_multi, dtype=np.float32))
    np.save(
        directory / "log_popularity.npy", np.ascontiguousarray(log_popularity, dtype=np.float32)
    )
    np.save(directory / "mean_rating.npy", np.ascontiguousarray(mean_rating, dtype=np.float32))
    np.save(directory / "popularity.npy", np.ascontiguousarray(popularity, dtype=np.float32))
    _write_json(directory / "user_ids.json", [int(i) for i in user_ids])
    _write_json(directory / "item_ids.json", [int(i) for i in item_ids])
    _write_json(directory / "genre_names.json", list(genre_names))
    _write_json(
        directory / "item_meta.json",
        {
            str(int(item_id)): {"title": titles[int(item_id)], "genres": genres[int(item_id)]}
            for item_id in item_ids
        },
    )
    profiles = {}
    for user_id in user_ids:
        uid = int(user_id)
        profiles[str(uid)] = {
            "history_item_ids": [int(i) for i in history_raw.get(uid, [])],
            "profile_item_ids": [int(i) for i in profile_raw.get(uid, [])],
            "recent_item_ids": [int(i) for i in recent_for_serve.get(uid, [])],
        }
    _write_json(directory / "user_profiles.json", profiles)
    pop = popularity.astype(np.float64)
    ids = item_ids.astype(np.int64)
    order = np.lexsort((ids, -pop))
    _write_json(directory / "popular_items.json", [int(ids[i]) for i in order])
    joblib.dump(ranker, directory / "ranker.joblib")
    index = faiss.IndexFlatIP(items.shape[1])
    index.add(items)
    faiss.write_index(index, str(directory / "faiss.index"))
    manifest = dict(manifest)
    manifest["feature_names"] = list(FEATURE_NAMES)
    manifest["embedding_dim"] = int(items.shape[1])
    manifest["n_users"] = int(users.shape[0])
    manifest["n_items"] = int(items.shape[0])
    _write_json(directory / "manifest.json", manifest)


def load_bundle(directory: Path) -> Bundle | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    user_ids = np.asarray(_read_json(directory / "user_ids.json"), dtype=np.int64)
    item_ids = np.asarray(_read_json(directory / "item_ids.json"), dtype=np.int64)
    meta = _read_json(directory / "item_meta.json")
    profiles = _read_json(directory / "user_profiles.json")
    titles = {int(item_id): meta[str(int(item_id))]["title"] for item_id in item_ids}
    genres = {int(item_id): list(meta[str(int(item_id))]["genres"]) for item_id in item_ids}
    history: dict[int, set[int]] = {}
    profile_items: dict[int, list[int]] = {}
    recent_items: dict[int, list[int]] = {}
    for user_id in user_ids:
        raw = profiles[str(int(user_id))]
        history[int(user_id)] = {int(i) for i in raw["history_item_ids"]}
        profile_items[int(user_id)] = [int(i) for i in raw["profile_item_ids"]]
        recent_items[int(user_id)] = [int(i) for i in raw["recent_item_ids"]]
    return Bundle(
        user_ids=user_ids,
        item_ids=item_ids,
        user_index={int(user_id): i for i, user_id in enumerate(user_ids)},
        item_index={int(item_id): i for i, item_id in enumerate(item_ids)},
        user_embeddings=np.load(directory / "user_embeddings.npy"),
        item_embeddings=np.load(directory / "item_embeddings.npy"),
        index=faiss.read_index(str(directory / "faiss.index")),
        genre_names=list(_read_json(directory / "genre_names.json")),
        genre_multi=np.load(directory / "genre_multi.npy"),
        log_popularity=np.load(directory / "log_popularity.npy"),
        mean_rating=np.load(directory / "mean_rating.npy"),
        popularity=np.load(directory / "popularity.npy"),
        titles=titles,
        genres=genres,
        history=history,
        profile_items=profile_items,
        recent_items=recent_items,
        ranker=joblib.load(directory / "ranker.joblib"),
        popular_item_ids=[int(i) for i in _read_json(directory / "popular_items.json")],
        manifest=_read_json(manifest_path),
    )
