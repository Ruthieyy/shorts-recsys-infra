import json
from pathlib import Path


def test_readme_quotes_measured_metrics():
    readme = Path("README.md").read_text()
    metrics = json.loads(Path("results/offline_metrics.json").read_text())
    latency = json.loads(Path("results/latency.json").read_text())
    for section in ("popularity", "retrieval", "retrieval_plus_ranker"):
        for key in ("recall@10", "recall@50", "ndcg@10", "mrr"):
            rendered = f"{metrics[section][key]:.4f}"
            assert rendered in readme, f"{section} {key} {rendered} missing from README"
    assert f"{metrics['val_retrieval_recall_at_10']:.4f}" in readme
    assert f"{latency['p50_ms']:.3f}" in readme
    assert f"{latency['p95_ms']:.3f}" in readme
    assert f"{latency['mean_ms']:.3f}" in readme
    assert str(latency["n"]) in readme
    assert str(latency["cpu_count"]) in readme
