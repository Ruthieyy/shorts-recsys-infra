"""MovieLens download, event adaptation, and leave-one-out splits."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import shutil
import ssl
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from shorts_recsys.config import MOVIELENS_URL, MOVIELENS_ZIP_SHA256, TrainConfig

logger = logging.getLogger(__name__)

DATASET_NAME = "movielens-latest-small"


def parse_genres(value: object) -> list[str]:
    if not isinstance(value, str) or not value or value == "(no genres listed)":
        return []
    return [part for part in value.split("|") if part]


def watch_time_seconds(rating: float, short_length_sec: float) -> float:
    """Map a 0.5–5 star rating onto a short-video watch time.

    MovieLens has no watch time. This is a linear proxy so the event schema
    can carry the field a short-video pipeline expects. Retrieval labels do
    not use it.
    """
    return float(rating) / 5.0 * float(short_length_sec)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    cert_path = Path(__file__).resolve().parents[2] / "certs" / "emsign_ecc_root_g3.pem"
    if cert_path.exists():
        ctx.load_verify_locations(cafile=str(cert_path))
    try:
        import certifi

        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        logger.debug("certifi bundle not added", exc_info=True)
    return ctx


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "shorts-recsys/0.1"})
    context = _ssl_context()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, context=context, timeout=120) as response:
                dest.write_bytes(response.read())
            return
        except Exception as exc:
            last_error = exc
            logger.warning("download attempt %s failed: %s", attempt + 1, exc)
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def ensure_dataset(data_dir: Path) -> Path:
    """Download and extract MovieLens ml-latest-small. Returns the dataset directory."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock_path = raw / ".download.lock"
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return _ensure_dataset_locked(raw)


def _ensure_dataset_locked(raw: Path) -> Path:
    dataset_dir = raw / "ml-latest-small"
    stamp = raw / "ml-latest-small.sha256"
    ratings = dataset_dir / "ratings.csv"
    if stamp.exists() and stamp.read_text().strip() == MOVIELENS_ZIP_SHA256 and ratings.exists():
        return dataset_dir
    zip_path = raw / "ml-latest-small.zip"
    if not zip_path.exists() or sha256_file(zip_path) != MOVIELENS_ZIP_SHA256:
        logger.info("downloading %s", MOVIELENS_URL)
        partial = zip_path.with_suffix(".zip.partial")
        _download(MOVIELENS_URL, partial)
        digest = sha256_file(partial)
        if digest != MOVIELENS_ZIP_SHA256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"MovieLens zip sha256 {digest} does not match pinned {MOVIELENS_ZIP_SHA256}"
            )
        partial.replace(zip_path)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw)
    if not ratings.exists():
        raise RuntimeError(f"ratings.csv missing after extract under {dataset_dir}")
    stamp.write_text(MOVIELENS_ZIP_SHA256 + "\n")
    return dataset_dir


def load_tables(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(dataset_dir / "ratings.csv")
    movies = pd.read_csv(dataset_dir / "movies.csv")
    required = {"userId", "movieId", "rating", "timestamp"}
    if not required <= set(ratings.columns):
        raise ValueError(f"ratings.csv missing columns: {required - set(ratings.columns)}")
    if not {"movieId", "title", "genres"} <= set(movies.columns):
        raise ValueError("movies.csv missing movieId, title, or genres")
    return ratings, movies


def build_events(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    positive_threshold: float,
    short_length_sec: float,
) -> list[dict]:
    """One view event per rating, ordered by time.

    ``liked`` is the implicit positive used by the feature counters. Watch time
    is the documented rating proxy, not a measured dwell time.
    """
    genre_map = {
        int(row.movieId): parse_genres(row.genres) for row in movies.itertuples(index=False)
    }
    title_map = {int(row.movieId): str(row.title) for row in movies.itertuples(index=False)}
    frame = ratings.sort_values(["timestamp", "userId", "movieId"], kind="mergesort")
    events: list[dict] = []
    for row in frame.itertuples(index=False):
        rating = float(row.rating)
        item_id = int(row.movieId)
        events.append(
            {
                "user_id": int(row.userId),
                "item_id": item_id,
                "rating": rating,
                "ts": int(row.timestamp),
                "genres": genre_map.get(item_id, []),
                "title": title_map.get(item_id, str(item_id)),
                "liked": rating >= positive_threshold,
                "watch_time_sec": watch_time_seconds(rating, short_length_sec),
                "event_type": "view",
            }
        )
    return events


@dataclass
class PreparedData:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict[int, int]
    item_index: dict[int, int]
    genre_names: list[str]
    genre_multi: np.ndarray
    train_user_idx: np.ndarray
    train_item_idx: np.ndarray
    val_pairs: list[tuple[int, int]]
    test_pairs: list[tuple[int, int]]
    seen_for_val: dict[int, np.ndarray]
    seen_for_test: dict[int, np.ndarray]
    profile_items: dict[int, np.ndarray]
    recent_for_val: dict[int, np.ndarray]
    recent_for_test: dict[int, np.ndarray]
    recent_for_serve: dict[int, list[int]]
    history_raw: dict[int, list[int]]
    profile_raw: dict[int, list[int]]
    log_popularity: np.ndarray
    mean_rating: np.ndarray
    popularity: np.ndarray
    titles: dict[int, str]
    genres: dict[int, list[str]]
    n_skipped_test: int
    n_skipped_val: int
    n_users_qualified: int
    n_ratings: int
    extra: dict = field(default_factory=dict)


def _map_rows(raw_ids: list[int] | np.ndarray, item_index: dict[int, int]) -> np.ndarray:
    return np.asarray([item_index[int(i)] for i in raw_ids if int(i) in item_index], dtype=np.int64)


def prepare(ratings: pd.DataFrame, movies: pd.DataFrame, config: TrainConfig) -> PreparedData:
    """Leave-one-out split on implicit positives (rating >= threshold).

    The last positive is test, the previous positive is validation, and earlier
    positives train the two-tower. At evaluation time every other title the
    user rated is removed from the candidate list.
    """
    users = ratings["userId"].to_numpy(dtype=np.int64)
    items = ratings["movieId"].to_numpy(dtype=np.int64)
    values = ratings["rating"].to_numpy(dtype=np.float64)
    stamps = ratings["timestamp"].to_numpy(dtype=np.int64)
    order = np.lexsort((items, stamps, users))
    users, items, values, stamps = users[order], items[order], values[order], stamps[order]

    boundaries = np.flatnonzero(np.diff(users)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(users)]

    train_pairs_raw: list[tuple[int, int]] = []
    val_raw: dict[int, int] = {}
    test_raw: dict[int, int] = {}
    seen_val_raw: dict[int, list[int]] = {}
    seen_test_raw: dict[int, list[int]] = {}
    profile_raw: dict[int, list[int]] = {}
    recent_val_raw: dict[int, list[int]] = {}
    recent_test_raw: dict[int, list[int]] = {}
    recent_serve_raw: dict[int, list[int]] = {}
    history_raw: dict[int, list[int]] = {}

    for start, end in zip(starts, ends, strict=True):
        user_id = int(users[start])
        user_items = items[start:end]
        user_values = values[start:end]
        history_raw[user_id] = [int(i) for i in user_items]
        positive_at = np.flatnonzero(user_values >= config.positive_threshold)
        if positive_at.size < config.min_user_positives:
            continue
        test_at = int(positive_at[-1])
        val_at = int(positive_at[-2])
        train_at = positive_at[:-2]
        test_raw[user_id] = int(user_items[test_at])
        val_raw[user_id] = int(user_items[val_at])
        train_ids = [int(user_items[i]) for i in train_at]
        profile_raw[user_id] = train_ids
        for item_id in train_ids:
            train_pairs_raw.append((user_id, item_id))
        seen_test_raw[user_id] = [int(i) for j, i in enumerate(user_items) if j != test_at]
        seen_val_raw[user_id] = [
            int(i) for j, i in enumerate(user_items) if j != test_at and j != val_at
        ]
        recent_val_raw[user_id] = [int(i) for i in user_items[:val_at][-config.recent_window :]]
        recent_test_raw[user_id] = [int(i) for i in user_items[:test_at][-config.recent_window :]]
        recent_serve_raw[user_id] = [int(i) for i in user_items[-config.recent_window :]]

    if not train_pairs_raw:
        raise RuntimeError("no training pairs; check the positive threshold and minimum history")

    item_ids = np.asarray(sorted({item for _, item in train_pairs_raw}), dtype=np.int64)
    user_ids = np.asarray(sorted(profile_raw), dtype=np.int64)
    item_index = {int(item): i for i, item in enumerate(item_ids)}
    user_index = {int(user): i for i, user in enumerate(user_ids)}

    genre_by_item = {
        int(row.movieId): parse_genres(row.genres) for row in movies.itertuples(index=False)
    }
    title_by_item = {int(row.movieId): str(row.title) for row in movies.itertuples(index=False)}
    genre_names = sorted({genre for item in item_ids for genre in genre_by_item.get(int(item), [])})
    genre_index = {name: i for i, name in enumerate(genre_names)}
    genre_multi = np.zeros((len(item_ids), len(genre_names)), dtype=np.float32)
    titles: dict[int, str] = {}
    genres: dict[int, list[str]] = {}
    for row, item_id in enumerate(item_ids):
        item_genres = genre_by_item.get(int(item_id), [])
        genres[int(item_id)] = item_genres
        titles[int(item_id)] = title_by_item.get(int(item_id), str(int(item_id)))
        for genre in item_genres:
            genre_multi[row, genre_index[genre]] = 1.0

    popularity = np.zeros(len(item_ids), dtype=np.float64)
    for _, item_id in train_pairs_raw:
        popularity[item_index[item_id]] += 1.0
    log_popularity = np.log1p(popularity)

    excluded = {(user, val_raw[user]) for user in val_raw} | {
        (user, test_raw[user]) for user in test_raw
    }
    rating_sum: dict[int, float] = {}
    rating_count: dict[int, int] = {}
    included_sum = 0.0
    included_count = 0
    for user_id, item_id, rating in zip(users, items, values, strict=True):
        pair = (int(user_id), int(item_id))
        if pair in excluded:
            continue
        rating_sum[pair[1]] = rating_sum.get(pair[1], 0.0) + float(rating)
        rating_count[pair[1]] = rating_count.get(pair[1], 0) + 1
        included_sum += float(rating)
        included_count += 1
    global_mean = included_sum / included_count if included_count else 3.5
    mean_rating = np.full(len(item_ids), global_mean, dtype=np.float64)
    for item_id, row in item_index.items():
        count = rating_count.get(item_id, 0)
        if count:
            mean_rating[row] = rating_sum[item_id] / count

    def pairs_for(mapping: dict[int, int]) -> tuple[list[tuple[int, int]], int]:
        kept: list[tuple[int, int]] = []
        skipped = 0
        for user_id, item_id in mapping.items():
            if item_id not in item_index:
                skipped += 1
                continue
            kept.append((user_index[user_id], item_index[item_id]))
        return kept, skipped

    val_pairs, n_skipped_val = pairs_for(val_raw)
    test_pairs, n_skipped_test = pairs_for(test_raw)

    def seen_index(raw_map: dict[int, list[int]]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for user_id, raw_ids in raw_map.items():
            out[user_index[user_id]] = _map_rows(raw_ids, item_index)
        return out

    profile_items = {
        user_index[user_id]: _map_rows(raw_ids, item_index)
        for user_id, raw_ids in profile_raw.items()
    }
    train_user_idx = np.asarray([user_index[user] for user, _ in train_pairs_raw], dtype=np.int64)
    train_item_idx = np.asarray([item_index[item] for _, item in train_pairs_raw], dtype=np.int64)

    qualified_history = {user_id: history_raw[user_id] for user_id in profile_raw}
    return PreparedData(
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
        genre_names=genre_names,
        genre_multi=genre_multi,
        train_user_idx=train_user_idx,
        train_item_idx=train_item_idx,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        seen_for_val=seen_index(seen_val_raw),
        seen_for_test=seen_index(seen_test_raw),
        profile_items=profile_items,
        recent_for_val={
            user_index[user_id]: _map_rows(raw_ids, item_index)
            for user_id, raw_ids in recent_val_raw.items()
        },
        recent_for_test={
            user_index[user_id]: _map_rows(raw_ids, item_index)
            for user_id, raw_ids in recent_test_raw.items()
        },
        recent_for_serve=recent_serve_raw,
        history_raw=qualified_history,
        profile_raw=profile_raw,
        log_popularity=log_popularity.astype(np.float32),
        mean_rating=mean_rating.astype(np.float32),
        popularity=popularity.astype(np.float32),
        titles=titles,
        genres=genres,
        n_skipped_test=n_skipped_test,
        n_skipped_val=n_skipped_val,
        n_users_qualified=len(profile_raw),
        n_ratings=int(len(ratings)),
    )


def main() -> None:
    import os

    from shorts_recsys.config import setup_logging

    setup_logging()
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    path = ensure_dataset(data_dir)
    print(path)


if __name__ == "__main__":
    main()
