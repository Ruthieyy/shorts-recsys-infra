import pandas as pd

from shorts_recsys.config import TrainConfig
from shorts_recsys.data import build_events, prepare, watch_time_seconds


def _tables():
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
            {"userId": 1, "movieId": 2, "rating": 5.0, "timestamp": 2},
            {"userId": 1, "movieId": 3, "rating": 4.0, "timestamp": 3},
            {"userId": 1, "movieId": 4, "rating": 4.5, "timestamp": 4},
            {"userId": 1, "movieId": 5, "rating": 5.0, "timestamp": 5},
            {"userId": 2, "movieId": 5, "rating": 5.0, "timestamp": 1},
            {"userId": 2, "movieId": 6, "rating": 5.0, "timestamp": 2},
            {"userId": 2, "movieId": 7, "rating": 4.0, "timestamp": 3},
            {"userId": 2, "movieId": 8, "rating": 4.0, "timestamp": 4},
            {"userId": 2, "movieId": 9, "rating": 5.0, "timestamp": 5},
            {"userId": 3, "movieId": 1, "rating": 5.0, "timestamp": 1},
            {"userId": 3, "movieId": 2, "rating": 3.0, "timestamp": 2},
        ]
    )
    movies = pd.DataFrame(
        [
            {
                "movieId": i,
                "title": f"Title {i}",
                "genres": "Comedy|Drama" if i % 2 == 0 else "Action",
            }
            for i in range(1, 10)
        ]
    )
    return ratings, movies


def test_watch_time_proxy_and_like_flag():
    assert watch_time_seconds(5.0, 30.0) == 30.0
    assert watch_time_seconds(2.5, 30.0) == 15.0
    ratings, movies = _tables()
    events = build_events(ratings, movies, positive_threshold=4.0, short_length_sec=30.0)
    by_key = {(event["user_id"], event["item_id"]): event for event in events}
    assert by_key[(3, 2)]["liked"] is False
    assert by_key[(1, 1)]["liked"] is True
    assert by_key[(1, 1)]["watch_time_sec"] == 30.0
    assert by_key[(1, 1)]["event_type"] == "view"
    assert events == sorted(
        events, key=lambda event: (event["ts"], event["user_id"], event["item_id"])
    )


def test_leave_one_out_split_and_catalog_skip():
    ratings, movies = _tables()
    prepared = prepare(ratings, movies, TrainConfig())
    assert prepared.n_users_qualified == 2
    assert prepared.n_skipped_test == 1
    assert len(prepared.test_pairs) == 1
    user_row, item_row = prepared.test_pairs[0]
    assert int(prepared.user_ids[user_row]) == 1
    assert int(prepared.item_ids[item_row]) == 5
    train = {
        (int(prepared.user_ids[u]), int(prepared.item_ids[i]))
        for u, i in zip(prepared.train_user_idx, prepared.train_item_idx, strict=True)
    }
    assert (1, 4) not in train
    assert (1, 5) not in train
    assert train == {(1, 1), (1, 2), (1, 3), (2, 5), (2, 6), (2, 7)}
    assert prepared.profile_raw[1] == [1, 2, 3]
    assert 4 not in set(prepared.item_ids)
    assert prepared.genres[2] == ["Comedy", "Drama"]
