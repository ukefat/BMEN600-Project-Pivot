import copy

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from temporal_ssl.cli import load_config
from temporal_ssl.eeg_course.features import feature_names
from temporal_ssl.eeg_data import aggregate_participants
from temporal_ssl.eeg_spectral_recoverability import (
    check_splits, fit_probe, load_encoder, observation_window, spectral_metrics, spectral_targets)


def test_matched_window_spectral_dimensions_and_encoder_access(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 4000, 19)).astype(np.float32)
    changed = x.copy()
    changed[:, 2000:] += 1000
    np.testing.assert_array_equal(observation_window(x), observation_window(changed))
    targets = spectral_targets(x, 500.)
    assert targets.shape == (2, 152) and len(feature_names()) == 152
    np.testing.assert_array_equal(targets, spectral_targets(changed, 500.))
    # Every channel has four relative powers partitioning 1–30 Hz.
    np.testing.assert_allclose(targets.reshape(2, 19, 8)[:, :, 4:].sum(-1), 1)
    cfg = load_config("configs/eeg_pilot.json")
    cfg["model"].update(d_model=8, nhead=2, layers=1)
    for name in ("untrained_causal", "untrained_masked"):
        model = load_encoder(tmp_path, cfg, name, (4000, 19), "cpu")
        with torch.no_grad():
            a = model.representations(torch.from_numpy(observation_window(x)))["sequence"]
            b = model.representations(torch.from_numpy(observation_window(changed)))["sequence"]
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError):
        observation_window(np.zeros((2, 7, 19)))


def test_participant_splits_and_mean_aggregation():
    splits = {"train": [{"participant_id": "a"}], "validation": [{"participant_id": "b"}],
              "test": [{"participant_id": "c"}]}
    check_splits(splits)
    bad = copy.deepcopy(splits)
    bad["test"][0]["participant_id"] = "a"
    with pytest.raises(ValueError, match="overlap"):
        check_splits(bad)
    ids, features, _ = aggregate_participants(np.array([[1., 2.], [5., 8.], [7., 10.]]),
        [{"participant_id": "a", "group": "AD"}, {"participant_id": "a", "group": "AD"},
         {"participant_id": "b", "group": "HC"}])
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(features, [[3., 5.], [7., 10.]])


def test_probe_train_only_scaling_selection_and_affine_units():
    rng = np.random.default_rng(2)
    x = {k:rng.normal(size=(n, 3)) for k, n in (("train", 40), ("validation", 10), ("test", 12))}
    y = {k:np.column_stack([v[:, 0]*50+100, v[:, 1]*.01-.02]) for k,v in x.items()}
    probe, predictions, scores = fit_probe(x, y, alphas=(.001, 10000.))
    assert probe["ridge"].alpha == max(scores, key=lambda r:r["validation_r2"])["alpha"] == .001
    np.testing.assert_allclose(probe["x_scaler"].mean_, x["train"].mean(0))
    np.testing.assert_allclose(probe["y_scaler"].mean_, y["train"].mean(0))
    altered_x, altered_y = copy.deepcopy(x), copy.deepcopy(y)
    altered_x["test"] += 10000
    altered_y["test"] -= 9000
    again, _, again_scores = fit_probe(altered_x, altered_y, alphas=(.001, 10000.))
    np.testing.assert_array_equal(probe["ridge"].coef_, again["ridge"].coef_)
    assert scores == again_scores
    metrics, r2 = spectral_metrics(y["test"], predictions["test"], y["train"])
    assert metrics["r2"] > .999
    converted = spectral_metrics(y["test"]*[100, .1]+[9, 5],
                predictions["test"]*[100, .1]+[9, 5], y["train"]*[100, .1]+[9, 5])[1]
    np.testing.assert_allclose(r2, converted)


def test_r2_uniform_mean_negative_and_constant_targets():
    training = np.array([[0., 0., 4., 8.], [1., 100., 4., 9.], [2., 200., 4., 10.]])
    truth = np.array([[0., 0., 4., 11.], [1., 100., 4., 11.], [2., 200., 4., 11.]])
    prediction = truth.copy()
    prediction[:, 1] = 100
    metrics, scores = spectral_metrics(truth, prediction, training)
    np.testing.assert_allclose(scores[:2], [1, 0])
    assert np.isnan(scores[2:]).all()
    assert metrics["r2"] == .5 and metrics["features_excluded"] == 2
    _, negative = spectral_metrics(truth, np.zeros_like(truth), training)
    assert negative[1] < 0
    near = training.copy()
    near[:, 2] = [4, 4+1e-12, 4-1e-12]
    assert spectral_metrics(truth, prediction, near)[0]["features_excluded"] == 2
    with pytest.raises(ValueError, match="No nonconstant"):
        spectral_metrics(np.ones((3, 2)), np.ones((3, 2)), np.ones((5, 2)))


def test_encoder_seed_and_checkpoint_loading(tmp_path):
    cfg = load_config("configs/eeg_pilot.json")
    cfg["model"].update(patch_size=5, d_model=8, nhead=2, layers=1, context=4)
    shape = (40, 3)
    for control, trained in (("untrained_causal", "causal"), ("untrained_masked", "masked_short")):
        initial = load_encoder(tmp_path, cfg, control, shape, "cpu")
        repeated = load_encoder(tmp_path, cfg, control, shape, "cpu")
        assert not initial.training and all(not p.requires_grad for p in initial.parameters())
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[key], rtol=0, atol=0)
        with pytest.raises(FileNotFoundError, match="will not be retrained"):
            load_encoder(tmp_path, cfg, trained, shape, "cpu")
        path = tmp_path/trained
        path.mkdir()
        state = initial.state_dict()
        state["embedding.bias"] = state["embedding.bias"]+3
        torch.save(state, path/"best.pt")
        restored = load_encoder(tmp_path, cfg, trained, shape, "cpu")
        torch.testing.assert_close(restored.embedding.bias, initial.embedding.bias+3)
