"""Online user-feature state shared by the Kafka worker and unit tests.

Spark Structured Streaming computes the same counters (view count, like count,
watch-time sum). The worker additionally keeps a short recent-item list, which
the ranker reads as recent genre affinity. See the README for that split.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserFeatureState:
    view_count: int = 0
    like_count: int = 0
    watch_time_sum: float = 0.0
    recent_item_ids: list[int] = field(default_factory=list)
    last_ts: int = 0
    recent_limit: int = 20

    def update(self, event: dict) -> None:
        self.view_count += 1
        if event.get("liked"):
            self.like_count += 1
        self.watch_time_sum += float(event.get("watch_time_sec") or 0.0)
        self.recent_item_ids.append(int(event["item_id"]))
        if len(self.recent_item_ids) > self.recent_limit:
            self.recent_item_ids = self.recent_item_ids[-self.recent_limit :]
        ts = int(event.get("ts") or 0)
        if ts > self.last_ts:
            self.last_ts = ts

    @property
    def avg_watch_time(self) -> float:
        if self.view_count == 0:
            return 0.0
        return self.watch_time_sum / self.view_count

    @property
    def like_rate(self) -> float:
        if self.view_count == 0:
            return 0.0
        return self.like_count / self.view_count

    def to_redis(self) -> dict[str, str]:
        return {
            "view_count": str(self.view_count),
            "like_count": str(self.like_count),
            "watch_time_sum": f"{self.watch_time_sum:.6f}",
            "avg_watch_time_sec": f"{self.avg_watch_time:.6f}",
            "like_rate": f"{self.like_rate:.6f}",
            "recent_item_ids": ",".join(str(item_id) for item_id in self.recent_item_ids),
            "last_ts": str(self.last_ts),
        }


def parse_redis_hash(raw: dict) -> dict:
    """Decode a Redis hash written by ``UserFeatureState.to_redis``."""
    recent_raw = raw.get("recent_item_ids") or ""
    recent = [int(part) for part in str(recent_raw).split(",") if part]
    views = int(raw.get("view_count") or 0)
    likes = int(raw.get("like_count") or 0)
    return {
        "view_count": views,
        "like_count": likes,
        "avg_watch_time_sec": float(raw.get("avg_watch_time_sec") or 0.0),
        "like_rate": float(raw.get("like_rate") or 0.0),
        "recent_item_ids": recent,
        "last_ts": int(raw.get("last_ts") or 0),
    }
