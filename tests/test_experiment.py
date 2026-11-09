import copy
import json

import numpy as np
import pytest
import torch

from temporal_ssl.cli import load_config
from temporal_ssl.data import make_splits
from temporal_ssl.experiment import make_model, run_experiment, seed_everything, train
from temporal_ssl.plots import analyze


@pytest.mark.parametrize("objective", ["causal", "masked"])
def test_end_to_end_and_resume(tmp_path, objective):
    cfg = load_config("configs/smoke.json")
    cfg["model"]["objective"] = objective
    cfg["training"]["epochs"] = 1
    cfg["probes"]["controls"] = True
    path = tmp_path/objective
    metrics = run_experiment(cfg, path)
    assert metrics["epochs"] == 1 and "local/u/linear" in metrics["probes"]
    assert metrics["reconstruction"]["neural"]["all"]["n_trajectories"] == 4
    assert set(metrics["probe_controls"]) == {"untrained_encoder", "raw_patch", "raw_statistics"}
    assert metrics["probe_controls"]["raw_patch"]["local/u/linear"]["test_pooled"]["r2"] > 0.95
    for file in ["config.json", "environment.json", "metrics.json", "history.json", "last.pt", "best.pt",
                 "normalization.npz", "split_manifest.json", "reconstruction_trajectories.csv",
                 "trajectory.png", "reconstruction.png", "training.png"]:
        assert (path/file).stat().st_size > 0
    with np.load(path/"reconstruction.npz") as example:
        assert example["prediction"].shape[-1] == 3
    assert json.loads((path/"metrics.json").read_text()) == metrics
    assert run_experiment(cfg, path, resume=True) == metrics
    with pytest.raises(FileExistsError):
        run_experiment(cfg, path)
    changed = copy.deepcopy(cfg)
    changed["data"]["seed"] = 100
    with pytest.raises(ValueError):
        run_experiment(changed, path, resume=True)
    summary = analyze(tmp_path)
    assert summary.stat().st_size > 0


def test_exact_training_resume_with_dropout(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["model"].update(objective="masked", dropout=0.1)
    splits, _ = make_splits(cfg["data"])
    full, partial = tmp_path/"full", tmp_path/"partial"
    full.mkdir()
    partial.mkdir()
    seed_everything(11)
    net = make_model(cfg)
    train(net, splits, cfg, full, torch.device("cpu"))
    expected = torch.load(full/"last.pt", weights_only=False)
    seed_everything(11)
    net = make_model(cfg)
    first = copy.deepcopy(cfg)
    first["training"]["epochs"] = 1
    train(net, splits, first, partial, torch.device("cpu"))
    train(net, splits, cfg, partial, torch.device("cpu"), resume=True)
    actual = torch.load(partial/"last.pt", weights_only=False)
    assert expected["history"] == actual["history"]
    for key in expected["model"]:
        torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)


def test_train_only_and_explicit_budget_extension(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["training"]["epochs"] = 1
    root = tmp_path/"run"
    run_experiment(cfg, root, evaluate=False)
    assert not (root/"metrics.json").exists()
    cfg["training"]["epochs"] = 2
    with pytest.raises(ValueError):
        run_experiment(cfg, root, resume=True, evaluate=False)
    history = run_experiment(cfg, root, resume=True, evaluate=False, allow_epoch_extension=True)
    assert len(history) == 2
    assert len(json.loads((root/"budget_history.json").read_text())) == 1
    cfg["training"]["lr"] = 0.01
    with pytest.raises(ValueError):
        run_experiment(cfg, root, resume=True, evaluate=False, allow_epoch_extension=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_and_resume(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["model"]["objective"] = "masked"
    cfg["training"].update(device="cuda", epochs=1)
    root = tmp_path/"cuda"
    run_experiment(cfg, root, evaluate=False)
    cfg["training"]["epochs"] = 2
    result = run_experiment(cfg, root, resume=True, allow_epoch_extension=True)
    assert result["epochs"] == 2
    assert np.isfinite(result["reconstruction"]["neural"]["all"]["mean"])
