from shorts_recsys.latency_probe import percentile


def test_percentile_even_and_odd():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10, 20, 30], 0.0) == 10
    assert percentile([10, 20, 30], 1.0) == 30
    assert percentile([5], 0.95) == 5
