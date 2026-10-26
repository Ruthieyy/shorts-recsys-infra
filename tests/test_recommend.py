import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shorts_recsys.artifacts import load_bundle, save_bundle
from shorts_recsys.ranker import features_for_items, make_ranker
from shorts_recsys.recommend import recommend


def _ranker_sensitive_to_recent_affinity() -> Pipeline:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(900, 5))
    labels = (features[:, 3] > 0).astype(np.int64)
    model = make_ranker(0)
    model.fit(features, labels)
    coefficient = model.named_steps["clf"].coef_[0, 3]
    assert coefficient > 0
    return model


def _save_toy(directory, ranker):
    user_ids = np.array([7], dtype=np.int64)
    item_ids = np.array([100, 101, 102], dtype=np.int64)
    vector = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    genre_multi = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    ones = np.ones(3, dtype=np.float32)
    save_bundle(
        directory,
        user_ids=user_ids,
        item_ids=item_ids,
        user_embeddings=vector,
        item_embeddings=np.repeat(vector, 3, axis=0),
        genre_names=["A", "B"],
        genre_multi=genre_multi,
        log_popularity=ones,
        mean_rating=np.full(3, 4.0, dtype=np.float32),
        popularity=ones,
        titles={100: "A1", 101: "B1", 102: "A2"},
        genres={100: ["A"], 101: ["B"], 102: ["A"]},
        history_raw={7: []},
        profile_raw={7: []},
        recent_for_serve={7: []},
        ranker=ranker,
        manifest={"candidate_pool": 10, "recent_window": 5, "metrics": {}},
    )


def test_online_recent_items_change_ranking(tmp_path):
    ranker = _ranker_sensitive_to_recent_affinity()
    _save_toy(tmp_path, ranker)
    bundle = load_bundle(tmp_path)
    assert bundle is not None
    assert bundle.manifest["n_users"] == 1
    assert bundle.manifest["n_items"] == 3
    assert bundle.manifest["embedding_dim"] == 4
    toward_b = recommend(bundle, 7, k=3, online_recent=[101])
    toward_a = recommend(bundle, 7, k=3, online_recent=[100])
    assert toward_b["used_online_recent"] is True
    assert toward_b["recommendations"][0]["item_id"] == 101
    assert toward_a["recommendations"][0]["item_id"] in {100, 102}
    assert toward_a["recommendations"][0]["genres"] == ["A"]


def test_cold_start_uses_popularity(tmp_path):
    ranker = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=0)),
        ]
    )
    ranker.fit(np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=float), np.array([0, 1]))
    _save_toy(tmp_path / "bundle", ranker)
    bundle = load_bundle(tmp_path / "bundle")
    result = recommend(bundle, 999, k=2)
    assert result["cold_start"] is True
    assert len(result["recommendations"]) == 2
    assert result["recommendations"][0]["score"] is None


def test_feature_matrix_shape():
    multi = np.array([[1, 0], [0, 1]], dtype=np.float32)
    features = features_for_items(
        [0, 1],
        np.array([0.2, 0.4]),
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        multi,
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )
    assert features.shape == (2, 5)
    assert features[0, 2] == 1.0
    assert features[1, 3] == 1.0
