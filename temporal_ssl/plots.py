"""Headless artifact plots; plotting never supplies data to training or probes."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trajectory(record, path):
    latents = record.get("latents") or {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(record["x"])
    axes[0].set_ylabel("Observation")
    axes[0].set_title(record["metadata"]["system"]+" — raw observations and simulator state")
    keys = [key for key in ("u", "phase", "state") if key in latents]
    for key in keys:
        axes[1].plot(latents[key], label=[f"{key}_{i}" for i in range(latents[key].shape[-1])])
    if keys:
        axes[1].legend(ncol=4)
    if "q" in latents:
        axes[2].step(np.arange(len(record["x"])), latents["q"][:, 0], where="post")
    axes[2].set_ylabel("Regime")
    axes[2].set_xlabel("Sample")
    if "transition" in latents:
        for index in np.flatnonzero(latents["transition"][:, 0]):
            for ax in axes:
                ax.axvline(index, color="black", alpha=0.2, linestyle=":")
    finish(fig, path)


def plot_run(output_dir, record):
    output_dir = Path(output_dir)
    plot_trajectory(record, output_dir/"trajectory.png")
    example = np.load(output_dir/"reconstruction.npz")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x, p = example["x"], int(example["patch_size"])
    ax.plot(x[:, 0], color="black", label="observed target")
    t = (example["indices"][:, None]*p+np.arange(p)).reshape(-1)
    selected = np.repeat(example["selected"], p)
    for key in ["prediction"]+[key for key in example.files if key.startswith("baseline_")]:
        y = example[key][..., 0].reshape(-1).copy()
        y[~selected] = np.nan
        ax.plot(t, y, label="neural" if key == "prediction" else key.removeprefix("baseline_"), alpha=0.8)
    ax.set(xlabel="Sample", ylabel="Channel 0", title="Test reconstruction (scored targets only)")
    ax.legend()
    finish(fig, output_dir/"reconstruction.png")
    history = json.loads((output_dir/"history.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 3))
    for key in ("train_loss", "validation_loss"):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
    ax.set(xlabel="Epoch", ylabel="Objective MSE", title="Optimization history")
    ax.legend()
    finish(fig, output_dir/"training.png")


def analyze(root, output=None):
    root = Path(root)
    output = Path(output) if output else root/"analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs, rows = [], []
    for path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(path.read_text())
        cfg = json.loads(path.with_name("config.json").read_text())
        runs.append((path.parent.name, cfg, metrics))
        identity = {"run": str(path.parent), "system": metrics["system"], "objective": metrics["objective"],
                    "data_seed": cfg["data"]["seed"], "model_seed": cfg["training"]["seed"],
                    "horizon": cfg["model"].get("horizon", 1), "context": cfg["model"].get("context", 8),
                    "mask_ratio": cfg["model"].get("mask_ratio", 0.5), "mask_span": cfg["model"].get("mask_span", 2),
                    "noise": cfg["data"].get("noise", 0.03), "visibility": cfg["data"].get("visibility", 0.6)}
        for method, regions in metrics["reconstruction"].items():
            for region, values in regions.items():
                rows.append({**identity, "kind": "reconstruction", "factor": method, "region": region,
                             "metric": "mse", "value": values["mean"]})
        for factor, values in metrics["probes"].items():
            for metric, value in values["test_pooled"].items():
                rows.append({**identity, "kind": "probe", "factor": factor, "region": "pooled_test",
                             "metric": metric, "value": value})
            for metric, value in values["test_trajectory_aggregate"].items():
                if value is not None:
                    rows.append({**identity, "kind": "probe", "factor": factor, "region": "trajectory_mean",
                                 "metric": metric, "value": value["mean"]})
    if not runs:
        raise ValueError(f"No completed metrics.json files found under {root}")
    with (output/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save = lambda fig, name: finish(fig, output/name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for objective, ax in zip(("causal", "masked"), axes):
        for name, cfg, metrics in runs:
            if metrics["objective"] != objective:
                continue
            key = "horizon" if objective == "causal" else "mask_span"
            for method, values in metrics["reconstruction"].items():
                ax.scatter(cfg["model"][key], values["all"]["mean"],
                           label=f"{name}: {method}", marker="o" if method == "neural" else "x")
        ax.set(xlabel="Horizon (patches)" if objective == "causal" else "Requested mask span (patches)",
               ylabel="Test trajectory mean MSE", title=objective)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=5)
    save(fig, "error_vs_task.png")
    factors = sorted({key for _, _, m in runs for key in m["probes"] if key.endswith("/linear")})
    fig, ax = plt.subplots(figsize=(max(9, len(factors)*1.2), 5))
    for name, _, metrics in runs:
        x, y = [], []
        for i, key in enumerate(factors):
            if key not in metrics["probes"]:
                continue
            values = metrics["probes"][key]
            score = values["test_pooled"].get("r2", values["test_pooled"].get("balanced_accuracy"))
            if score is not None:
                x.append(i)
                y.append(score)
        ax.scatter(x, y, label=name)
    ax.set_xticks(range(len(factors)), factors, rotation=35, ha="right", fontsize=7)
    ax.set(ylabel="Test R² / regime balanced accuracy", title="Frozen encoder linear recoverability (pooled descriptive metrics)")
    ax.legend(fontsize=5)
    save(fig, "recoverability.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = []
    labeled_regions = set()
    for i, (name, _, metrics) in enumerate(runs):
        labels.append(name)
        for offset, region in ((-0.12, "near_transition"), (0.12, "stable")):
            value = metrics["reconstruction"]["neural"][region]
            if value["mean"] is not None:
                ax.errorbar(i+offset, value["mean"], yerr=value["sem"], fmt="o",
                            color="tab:orange" if offset < 0 else "tab:blue",
                            label=region if region not in labeled_regions else None)
                labeled_regions.add(region)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=6)
    ax.set(ylabel="Neural test MSE ± trajectory SEM", title="Transitions and stable intervals")
    ax.legend()
    save(fig, "transition_errors.png")
    systems = sorted({m["system"] for _, _, m in runs})
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, _, m in runs:
        offset = -0.12 if m["objective"] == "causal" else 0.12
        ax.scatter(systems.index(m["system"])+offset, m["reconstruction"]["neural"]["all"]["mean"], label=name)
    ax.set_xticks(range(len(systems)), systems)
    ax.set(ylabel="Neural test MSE", title="Complete pretraining regimes by synthetic system")
    ax.legend(fontsize=5)
    save(fig, "system_comparison.png")
    return output/"summary.csv"
