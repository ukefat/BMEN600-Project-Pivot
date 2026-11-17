import copy

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from temporal_ssl.eeg_long_training import evaluate_fixed, fixed_baselines
from temporal_ssl.experiment import seed_everything, train, validation_loss
from temporal_ssl.models import TemporalSSL


class SmallEEG(Dataset):
    def __init__(self):
        self.x = np.random.default_rng(7).normal(size=(6, 40, 3)).astype(np.float32)
        self.metadata = [{"participant_id": f"s-{i//2}", "group": ("AD", "FTD", "HC")[i//2]}
                         for i in range(6)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": torch.from_numpy(self.x[index]), "latents": None, "parameters": None,
                "metadata": self.metadata[index]}


def model():
    return TemporalSSL(3, 40, objective="masked", patch_size=5, d_model=8,
                       nhead=2, layers=1, dropout=.1, mask_ratio=.25, mask_span=1)


def test_fixed_scores_align_with_original_validation_protocol(tmp_path):
    seed_everything(600)
    data, encoder = SmallEEG(), model()
    baselines, masks = fixed_baselines(data, encoder, 2, 603, tmp_path)
    scores = evaluate_fixed(encoder, data, masks, 2, tmp_path/"scores")
    expected = validation_loss(encoder, data, 2, torch.device("cpu"), 603)
    assert scores["mean"] == pytest.approx(expected, rel=1e-6)
    patches = data.x.reshape(6, 8, 5, 3)
    expected_mean = (patches[masks]**2).mean()
    assert baselines["training_mean"]["mean"] == pytest.approx(expected_mean, rel=1e-6)
    assert scores["target_rms"]**2 == pytest.approx(expected_mean, rel=1e-6)
    repeated_baselines, repeated_masks = fixed_baselines(data, encoder, 2, 603, tmp_path)
    np.testing.assert_array_equal(masks, repeated_masks)
    assert baselines == repeated_baselines


def test_chunked_resume_with_diagnostics_preserves_training(tmp_path):
    data = SmallEEG()
    cfg = {"training": {"epochs": 2, "batch_size": 2, "seed": 600, "lr": .001,
                         "weight_decay": .01, "grad_clip": 1.}}
    full, chunk = tmp_path/"full", tmp_path/"chunk"
    full.mkdir()
    chunk.mkdir()
    seed_everything(600)
    uninterrupted = model()
    train(uninterrupted, {"train":data, "validation":data}, cfg, full, torch.device("cpu"))
    seed_everything(600)
    continued = model()
    first_cfg = copy.deepcopy(cfg)
    first_cfg["training"]["epochs"] = 1
    train(continued, {"train":data, "validation":data}, first_cfg, chunk, torch.device("cpu"))
    _, masks = fixed_baselines(data, continued, 2, 603, tmp_path)
    evaluate_fixed(continued, data, masks, 2, tmp_path/"diagnostic")
    torch.rand(100)  # Even explicit diagnostic RNG draws are restored on resume.
    train(continued, {"train":data, "validation":data}, cfg, chunk, torch.device("cpu"), resume=True)
    a = torch.load(full/"last.pt", map_location="cpu", weights_only=False)
    b = torch.load(chunk/"last.pt", map_location="cpu", weights_only=False)
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
