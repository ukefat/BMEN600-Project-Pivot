import numpy as np
import pytest

from temporal_ssl.baselines import causal_baselines, error_summary, fill_masked
from temporal_ssl.probes import fit_logistic, fit_ridge, sigmoid
from temporal_ssl.probes import observation_representations
from temporal_ssl.data import Trajectories


def test_interpolation_and_boundary_filling():
    x = np.arange(12, dtype=float).reshape(6, 2, 1)
    mask = np.array([False, True, True, False, False, True])
    expected = x.copy()
    expected[-1] = 9
    np.testing.assert_allclose(fill_masked(x, mask), expected)
    corrupted = x.copy()
    corrupted[mask] = 10000
    for method in ("linear", "nearest"):
        np.testing.assert_array_equal(fill_masked(x, mask, method), fill_masked(corrupted, mask, method))


def test_causal_baseline_horizon_and_no_leakage():
    x = np.arange(20, dtype=float).reshape(10, 2, 1)
    pred = causal_baselines(x, horizon=3, context=2)
    np.testing.assert_allclose(pred["linear_trend"], x[3:])
    changed = x.copy()
    changed[7:] += 1000
    altered = causal_baselines(changed, horizon=3, context=2)
    for method in pred:
        np.testing.assert_array_equal(pred[method], altered[method])


def test_metrics_empty_regions_and_probe_sanity():
    summary = error_summary(np.arange(4), np.ones(4, dtype=bool), np.zeros(4, dtype=bool))
    assert summary["near_transition"] is None
    assert summary["all"] == 1.5
    rng = np.random.default_rng(8)
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    y = x @ np.array([[1], [2], [-3]])
    np.testing.assert_allclose(x @ fit_ridge(x, y, 1e-8), y, atol=1e-6)
    labels = (x[:, 1:] @ np.array([[1], [-1]]) > 0).astype(float)
    pred = sigmoid(x @ fit_logistic(x, labels, 0.001))
    assert ((pred > 0.5) == labels).mean() > 0.98


def test_observation_control_excludes_future_and_distant_context():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    original = Trajectories([{"x": x}])
    for statistics in [False, True]:
        before = observation_representations(original, 2, 3, statistics)
        changed = x.copy()
        changed[16:] += 100
        after = observation_representations(Trajectories([{"x": changed}]), 2, 3, statistics)
        np.testing.assert_array_equal(before["local"][:, :8], after["local"][:, :8])
        changed = x.copy()
        changed[:10] += 100
        after = observation_representations(Trajectories([{"x": changed}]), 2, 3, statistics)
        np.testing.assert_array_equal(before["local"][:, 7:], after["local"][:, 7:])
