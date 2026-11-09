"""Small deterministic training loop, resumable checkpoints, and held-out evaluation."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import causal_baselines, error_summary, fill_masked, near_transitions
from .data import collate_trajectories, make_splits
from .models import TemporalSSL, patchify
from .probes import evaluate_probes, evaluate_representations, observation_representations


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def make_model(cfg):
    return TemporalSSL(channels=cfg["data"].get("channels", 3), length=cfg["data"]["length"],
                       **cfg["model"])


def loader(dataset, batch_size, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=collate_trajectories, num_workers=0)


@torch.no_grad()
def validation_loss(model, dataset, batch_size, device, seed):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total, count = 0, 0
    for batch in loader(dataset, batch_size):
        output = model(batch["x"].to(device), generator=generator)
        total += float(output["loss"])*len(batch["x"])
        count += len(batch["x"])
    return total/count


def train(model, splits, cfg, output_dir, device, resume=False):
    options = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.get("lr", 0.001),
                                 weight_decay=options.get("weight_decay", 0.01))
    seed = options.get("seed", 0)
    shuffle = torch.Generator().manual_seed(seed+1)
    masking = torch.Generator().manual_seed(seed+2)
    checkpoint_path = output_dir/"last.pt"
    history, start, best = [], 0, float("inf")
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Optimizer state follows parameters onto the selected device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        history, start, best = checkpoint["history"], checkpoint["epoch"]+1, checkpoint["best"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        shuffle.set_state(checkpoint["shuffle_rng"])
        masking.set_state(checkpoint["mask_rng"])
        # best.pt is derived from last.pt, so interruption between writes is recoverable.
        torch.save(checkpoint["best_model"], output_dir/"best.pt")
        best_model = checkpoint["best_model"]
    else:
        best_model = None
    for epoch in range(start, options["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        total, count = 0, 0
        for batch in loader(splits["train"], options["batch_size"], True, shuffle):
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["x"].to(device), generator=masking)
            if not torch.isfinite(result["loss"]):
                raise RuntimeError("Nonfinite training loss")
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.get("grad_clip", 1.0))
            optimizer.step()
            total += float(result["loss"].detach())*len(batch["x"])
            count += len(batch["x"])
        val = validation_loss(model, splits["validation"], options["batch_size"], device, seed+3)
        if not np.isfinite(val):
            raise RuntimeError("Nonfinite validation loss")
        history.append({"epoch": epoch+1, "train_loss": total/count, "validation_loss": val})
        if val < best:
            best = val
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "best": best, "best_model": best_model, "history": history,
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                      "shuffle_rng": shuffle.get_state(), "mask_rng": masking.get_state()}
        torch.save(checkpoint, output_dir/"last.pt.tmp")
        os.replace(output_dir/"last.pt.tmp", checkpoint_path)
        torch.save(best_model, output_dir/"best.pt")
        save_json(output_dir/"history.json", history)
        print(f"{output_dir.name} epoch {epoch+1}: train={total/count:.5f} val={val:.5f} ({time.perf_counter()-epoch_start:.2f}s)", flush=True)
    save_json(output_dir/"history.json", history)
    model.load_state_dict(torch.load(output_dir/"best.pt", map_location=device, weights_only=True))
    return history


@torch.no_grad()
def evaluate_reconstruction(model, dataset, cfg, device, output_dir):
    model.eval()
    rows = []
    generator = torch.Generator().manual_seed(cfg.get("evaluation", {}).get("seed", 2000))
    radius = cfg.get("evaluation", {}).get("transition_radius", 4)
    if radius < 0:
        raise ValueError("transition_radius must be nonnegative")
    p = model.patch_size
    for i in range(len(dataset)):
        record = dataset[i]
        x = record["x"].numpy()
        result = model(record["x"][None].to(device), generator=generator)
        indices = result["indices"].cpu().numpy()
        selected = result["selected"][0].cpu().numpy()
        patches = patchify(record["x"][None], p)[0].numpy().reshape(-1, p, x.shape[-1])
        prediction = result["prediction"][0].cpu().numpy().reshape(-1, p, x.shape[-1])
        if model.objective == "masked":
            baselines = {method: fill_masked(patches, selected, method) for method in ("linear", "nearest")}
        else:
            baselines = causal_baselines(patches, model.horizon, model.context)
        target = patches[indices]
        near = near_transitions(record.get("latents"), len(x), radius).reshape(-1, p)[indices].reshape(-1)
        sample_mask = np.repeat(selected, p)
        for name, pred in {"neural": prediction, **baselines}.items():
            error = ((pred-target)**2).mean(-1).reshape(-1)
            rows.append({"trajectory": i, "method": name, **error_summary(error, sample_mask, near)})
        if i == 0:
            np.savez_compressed(output_dir/"reconstruction.npz", x=x, indices=indices, selected=selected,
                                prediction=prediction, patch_size=p, near=near,
                                **{f"baseline_{k}": v for k, v in baselines.items()})
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        summary[method] = {}
        for region in ("all", "near_transition", "stable"):
            values = [r[region] for r in rows if r["method"] == method and r[region] is not None]
            summary[method][region] = {
                "mean": float(np.mean(values)) if values else None,
                "sem": float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None,
                "n_trajectories": len(values)}
    with (output_dir/"reconstruction_trajectories.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def run_experiment(cfg, output_dir, resume=False, evaluate=True, allow_epoch_extension=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir/"config.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"{output_dir} already contains a run; use --resume or a new output")
        previous = json.loads(config_path.read_text())
        if previous != cfg:
            comparable = json.loads(json.dumps(cfg))
            comparable["training"]["epochs"] = previous["training"]["epochs"]
            if not (allow_epoch_extension and comparable == previous
                    and cfg["training"]["epochs"] > previous["training"]["epochs"]):
                raise ValueError("Resume config must match the saved run exactly (explicit epoch extension excepted)")
            archive_path = output_dir/"budget_history.json"
            archive = json.loads(archive_path.read_text()) if archive_path.exists() else []
            archive.append(previous)
            save_json(archive_path, archive)
    epochs = cfg["training"]["epochs"]
    if epochs < 1 or cfg["training"]["batch_size"] < 1:
        raise ValueError("Require epochs >= 1 and batch_size >= 1")
    save_json(config_path, cfg)
    seed_everything(cfg["training"].get("seed", 0), cfg["training"].get("threads", 1))
    device = torch.device(cfg["training"].get("device", "cpu"))
    splits, normalizer = make_splits(cfg["data"])
    if normalizer is not None:
        np.savez(output_dir/"normalization.npz", mean=normalizer.mean, std=normalizer.std)
    save_json(output_dir/"split_manifest.json", {
        k: [r["metadata"] for r in dataset.records] for k, dataset in splits.items()})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    save_json(output_dir/"environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                                            "numpy": np.__version__, "git_revision": revision,
                                            "device": str(device), "platform": platform.platform(),
                                            "packages": {d.metadata["Name"]: d.version
                                                         for d in importlib.metadata.distributions()},
                                            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                              for p in Path(__file__).parent.glob("*.py")}})
    model = make_model(cfg).to(device)
    history = train(model, splits, cfg, output_dir, device, resume)
    if not evaluate:
        return history
    model.requires_grad_(False)
    reconstruction = evaluate_reconstruction(model, splits["test"], cfg, device, output_dir)
    probes = evaluate_probes(model, splits, cfg.get("probes", {}), device)
    controls = {}
    if cfg.get("probes", {}).get("controls", False):
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg["training"].get("seed", 0))
            untrained = make_model(cfg).to(device).eval().requires_grad_(False)
        controls["untrained_encoder"] = evaluate_probes(untrained, splits, cfg.get("probes", {}), device)
        for name, statistics in (("raw_patch", False), ("raw_statistics", True)):
            features = {k: observation_representations(v, model.patch_size, model.context, statistics)
                        for k, v in splits.items()}
            controls[name] = evaluate_representations(features, splits, cfg.get("probes", {}), model.patch_size)
    metrics = {"system": cfg["data"]["system"], "objective": model.objective,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "epochs": len(history), "best_validation_loss": min(h["validation_loss"] for h in history),
               "reconstruction_units": "MSE in train-normalized observation units" if normalizer else "raw observation MSE",
               "local_probe_access": "past and current patch, bounded context" if model.objective == "causal"
               else "full unmasked sequence, including future observations",
               "reconstruction": reconstruction, "probes": probes}
    if controls:
        metrics["probe_controls"] = controls
    from .plots import plot_run
    plot_run(output_dir, splits["test"].records[0])
    save_json(output_dir/"metrics.json", metrics)
    return metrics
