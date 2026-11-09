import numpy as np
import pytest

from temporal_ssl.data import Normalizer, collate_trajectories, make_splits, simulate


def config(system="predictable"):
    return {"system": system, "length": 64, "channels": 3, "seed": 8,
            "splits": {"train": 6, "validation": 3, "test": 3}}


@pytest.mark.parametrize("system", ["ar2", "predictable", "random"])
def test_deterministic_shapes_and_splits(system):
    cfg = config(system)
    splits, norm = make_splits(cfg)
    again, _ = make_splits(cfg)
    ids = []
    for name, dataset in splits.items():
        for i, record in enumerate(dataset):
            assert record["x"].shape == (64, 3)
            assert all(v.shape[0] == 64 for v in record["latents"].values())
            assert np.isfinite(record["x"].numpy()).all()
            np.testing.assert_array_equal(record["x"], again[name][i]["x"])
            ids.append(tuple(record["metadata"]["spawn_key"]))
    assert len(set(ids)) == len(ids)
    assert not np.array_equal(splits["train"][0]["x"], splits["test"][0]["x"])
    train = np.concatenate([r["x"].numpy() for r in splits["train"]])
    np.testing.assert_allclose(train.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(train.std(0), 1, atol=1e-5)
    expected = Normalizer.fit(splits["train"].records)
    np.testing.assert_array_equal(norm.mean, expected.mean)
    cfg["splits"]["test"] = 5
    changed, changed_norm = make_splits(cfg)
    np.testing.assert_array_equal(norm.mean, changed_norm.mean)
    np.testing.assert_array_equal(splits["train"][0]["x"], changed["train"][0]["x"])


@pytest.mark.parametrize("mode", ["wrap", "reset"])
def test_predictable_switch_dynamics(mode):
    cfg = {**config(), "delta": 0.2, "threshold": 1.0, "noise": 0, "reset": mode}
    record = simulate(cfg, np.random.SeedSequence(4))
    u, q, phi = [record["latents"][k][:, 0] for k in ("u", "q", "phi")]
    for i in range(len(u)-1):
        switch = u[i]+0.2 >= 1-1e-6
        assert (q[i] != q[i+1]) == switch
        expected = (u[i]+0.2-1 if mode == "wrap" else 0) if switch else u[i]+0.2
        assert u[i+1] == pytest.approx(expected, abs=1e-6)
        omega = record["parameters"]["omega0" if q[i] == 0 else "omega1"]
        assert np.cos(phi[i+1]) == pytest.approx(np.cos(phi[i]+omega), abs=1e-6)
    np.testing.assert_allclose(record["x"][:, 2], 0.6*u, atol=1e-6)


def test_random_control_and_frequency_matching():
    for hazard in [0, 1]:
        r = simulate({**config("random"), "hazard": hazard}, np.random.SeedSequence(1))
        assert np.all(r["latents"]["transition"][1:] == hazard)
    rates = []
    for system in ["predictable", "random"]:
        r = simulate({**config(system), "length": 20000, "delta": 0.04, "threshold": 1.0},
                     np.random.SeedSequence(12))
        rates.append(r["latents"]["transition"].mean())
    assert abs(rates[0]-rates[1]) < 0.005


def test_optional_labels_and_configurable_mapping():
    import torch
    batch = collate_trajectories([{"x": torch.zeros(12, 5)}, {"x": torch.ones(12, 5)}])
    assert batch["x"].shape == (2, 12, 5)
    assert batch["latents"] == [None, None]
    cfg = {**config(), "channels": 2, "observation_matrix": [[1, 0, 0], [0, 1, 0]]}
    assert simulate(cfg, np.random.SeedSequence(4))["x"].shape == (64, 2)
    with pytest.raises(ValueError):
        simulate({**cfg, "channels": 5}, np.random.SeedSequence(4))


def test_ar2_equation():
    r = simulate({**config("ar2"), "innovation_noise": 0, "noise": 0, "burn_in": 0},
                 np.random.SeedSequence(9))
    x = r["latents"]["state"][:, 0]
    radius, omega = r["parameters"]["r"], r["parameters"]["omega"]
    np.testing.assert_allclose(x[2:], 2*radius*np.cos(omega)*x[1:-1]-radius**2*x[:-2], atol=2e-6)
