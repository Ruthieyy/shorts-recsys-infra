from shorts_recsys.features import UserFeatureState, parse_redis_hash


def test_feature_state_counts_likes_watch_time_and_recent_window():
    state = UserFeatureState(recent_limit=2)
    state.update({"user_id": 1, "item_id": 10, "liked": True, "watch_time_sec": 30, "ts": 1})
    state.update({"user_id": 1, "item_id": 11, "liked": False, "watch_time_sec": 12, "ts": 2})
    state.update({"user_id": 1, "item_id": 12, "liked": True, "watch_time_sec": 18, "ts": 3})
    assert state.view_count == 3
    assert state.like_count == 2
    assert state.like_rate == 2 / 3
    assert state.watch_time_sum == 60
    assert state.avg_watch_time == 20
    assert state.recent_item_ids == [11, 12]
    assert state.last_ts == 3
    parsed = parse_redis_hash(state.to_redis())
    assert parsed["view_count"] == 3
    assert parsed["like_count"] == 2
    assert parsed["recent_item_ids"] == [11, 12]
    assert abs(parsed["avg_watch_time_sec"] - 20) < 1e-6
