"""Independent trajectory simulation and the shared (future EEG-compatible) batch contract."""
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectorySource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, Any]: ...


def sample_range(rng, cfg, key, default):
    value = cfg.get(key, default)
    return float(rng.uniform(*value)) if isinstance(value, (list, tuple)) else float(value)


def simulate(cfg: dict, seed: np.random.SeedSequence) -> dict:
    rng = np.random.default_rng(seed)
    system, length = cfg["system"], int(cfg["length"])
    channels = int(cfg.get("channels", 3))
    noise = float(cfg.get("noise", 0.03))
    if length < 4 or channels < 1 or noise < 0:
        raise ValueError("Require length >= 4, channels >= 1, noise >= 0")
    if system == "ar2":
        r = sample_range(rng, cfg, "r", [0.92, 0.99])
        omega = sample_range(rng, cfg, "omega", [0.15, 0.5])
        sigma = float(cfg.get("innovation_noise", 0.08))
        if not 0 < r < 1 or not 0 < omega < np.pi or sigma < 0:
            raise ValueError("AR(2) requires 0 < r < 1, 0 < omega < pi, sigma >= 0")
        burn = int(cfg.get("burn_in", 256))
        if burn < 0:
            raise ValueError("burn_in must be nonnegative")
        state = np.zeros(length + burn + 2)
        state[:2] = rng.normal(size=2)
        for t in range(2, len(state)):
            state[t] = 2*r*np.cos(omega)*state[t-1] - r*r*state[t-2] + sigma*rng.normal()
        current, previous = state[burn+2:], state[burn+1:-1]
        basis = np.stack([current, previous], -1)
        default_map = np.tile([1.0, 0.0], (channels, 1))
        latents = {"state": basis}
        parameters = {"r": r, "omega": omega, "innovation_noise": sigma}
    elif system in {"predictable", "random"}:
        delta = sample_range(rng, cfg, "delta", [0.025, 0.05])
        tau = sample_range(rng, cfg, "threshold", [0.9, 1.1])
        w0 = sample_range(rng, cfg, "omega0", [0.12, 0.2])
        w1 = sample_range(rng, cfg, "omega1", [0.35, 0.5])
        visibility = float(cfg.get("visibility", 0.6))
        hazard = float(cfg.get("hazard", delta/tau))
        mode = cfg.get("reset", "wrap")
        if not 0 < delta < tau or not 0 <= hazard <= 1 or mode not in {"wrap", "reset"}:
            raise ValueError("Require 0 < delta < threshold, hazard in [0,1], reset=wrap|reset")
        u, phase, q = np.empty(length), np.empty(length), np.empty(length, dtype=np.int64)
        u[0], phase[0], q[0] = rng.uniform(0, tau), rng.uniform(0, 2*np.pi), rng.integers(2)
        for t in range(length-1):
            phase[t+1] = (phase[t] + (w0 if q[t] == 0 else w1)) % (2*np.pi)
            next_u = u[t] + delta
            switch = next_u >= tau if system == "predictable" else rng.random() < hazard
            q[t+1] = 1-q[t] if switch else q[t]
            u[t+1] = (next_u-tau if mode == "wrap" and system == "predictable" else 0) if switch else next_u
        phase_xy = np.stack([np.cos(phase), np.sin(phase)], -1)
        basis = np.column_stack([phase_xy, visibility*u])
        default_map = np.eye(3)[np.arange(channels) % 3]
        latents = {"u": u[:, None], "phase": phase_xy, "q": q[:, None],
                   "phi": phase[:, None]}
        parameters = {"delta": delta, "threshold": tau, "omega0": w0, "omega1": w1,
                      "visibility": visibility, "hazard": hazard}
    else:
        raise ValueError(f"Unknown system: {system}")
    mapping = np.asarray(cfg.get("observation_matrix", default_map), dtype=float)
    if mapping.shape != (channels, basis.shape[1]):
        raise ValueError(f"observation_matrix must have shape {(channels, basis.shape[1])}")
    x = basis @ mapping.T + noise*rng.normal(size=(length, channels))
    transition = np.zeros(length, dtype=bool)
    if "q" in latents:
        transition[1:] = np.diff(latents["q"][:, 0]) != 0
    latents["transition"] = transition[:, None]
    return {"x": x.astype(np.float32),
            "latents": {k: v.astype(np.float32) for k, v in latents.items()},
            "parameters": parameters,
            "metadata": {"system": system, "seed_entropy": seed.entropy,
                         "spawn_key": list(seed.spawn_key), "simulation": cfg}}


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, records):
        x = np.concatenate([record["x"] for record in records], axis=0)
        return cls(x.mean(0, dtype=np.float64).astype(np.float32),
                   np.maximum(x.std(0, dtype=np.float64), 1e-6).astype(np.float32))

    def apply(self, x):
        return (x-self.mean)/self.std


class Trajectories(Dataset):
    def __init__(self, records, normalizer=None):
        self.records, self.normalizer = records, normalizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        x = record["x"] if self.normalizer is None else self.normalizer.apply(record["x"])
        return {**record, "x": torch.as_tensor(x, dtype=torch.float32)}


def collate_trajectories(records):
    """Fixed T,C per batch; optional labels are kept per trajectory, never model inputs."""
    if len({tuple(r["x"].shape) for r in records}) != 1:
        raise ValueError("Batch trajectories must have equal T,C; window future EEG before batching")
    return {"x": torch.stack([torch.as_tensor(r["x"]) for r in records]),
            "latents": [r.get("latents") for r in records],
            "parameters": [r.get("parameters") for r in records],
            "metadata": [r.get("metadata", {}) for r in records]}


def make_splits(cfg):
    counts = cfg.get("splits", {"train": 48, "validation": 16, "test": 16})
    if set(counts) != {"train", "validation", "test"} or min(counts.values()) < 2:
        raise ValueError("Specify train, validation, test counts, each >= 2")
    seeds = np.random.SeedSequence(int(cfg.get("seed", 0))).spawn(3)
    raw = {name: [simulate(cfg, child) for child in seeds[i].spawn(counts[name])]
           for i, name in enumerate(("train", "validation", "test"))}
    normalizer = Normalizer.fit(raw["train"]) if cfg.get("normalize", True) else None
    return {k: Trajectories(v, normalizer) for k, v in raw.items()}, normalizer


def save_dataset(path, dataset):
    """Numeric arrays only; metadata/config is saved alongside by the CLI."""
    arrays = {"x": np.stack([r["x"].numpy() for r in dataset])}
    for key in dataset.records[0].get("latents", {}):
        arrays[f"latent_{key}"] = np.stack([r["latents"][key] for r in dataset.records])
    for key in dataset.records[0].get("parameters", {}):
        arrays[f"parameter_{key}"] = np.array([r["parameters"][key] for r in dataset.records])
    np.savez_compressed(path, **arrays)
