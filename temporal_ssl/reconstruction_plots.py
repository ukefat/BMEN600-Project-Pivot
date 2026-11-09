"""Reconstruction comparisons from completed paired-seed study artifacts; no retraining."""
import argparse
import csv
import json
from pathlib import Path

from . import plots as _headless_backend  # Select the same headless backend as other plots.

import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "causal": [("neural", "Trained causal model"), ("constant", "Last-value persistence"),
               ("linear_trend", "Past-window linear trend")],
    "masked": [("neural", "Trained masked model"), ("linear", "Linear interpolation"),
               ("nearest", "Nearest visible sample")],
}
COLORS = ["#2878B5", "#E69F00", "#009E73"]
SYSTEM_NAMES = {"ar2": "Oscillatory AR(2)", "predictable": "Predictable switching",
                "random": "Random switching"}


def save_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def load_runs(root):
    plan = json.loads((root/"study_plan.json").read_text())
    runs = []
    for entry in plan["matrix"]:
        for seed in plan["config"]["study"]["seeds"]:
            path = root/f"{entry['name']}-d{seed['data']}-m{seed['model']}"
            metrics = json.loads((path/"metrics.json").read_text())
            cfg = json.loads((path/"config.json").read_text())
            runs.append({"path": path, "condition": entry["name"], "cfg": cfg,
                         "system": metrics["system"], "objective": metrics["objective"],
                         "seed": (seed["data"], seed["model"]), "values": metrics["reconstruction"]})
    if len({run["cfg"]["data"].get("normalize", True) for run in runs}) != 1:
        raise ValueError("Plot studies with a consistent normalization setting")
    return runs


def values_for(runs, condition, method, region):
    return np.array([run["values"][method][region]["mean"] for run in runs
                     if run["condition"] == condition
                     and method in run["values"]
                     and run["values"][method][region]["mean"] is not None], dtype=float)


def bars(ax, runs, conditions, labels, objective, region="all", annotate=True):
    width = 0.23
    peak = 0
    for j, (method, label) in enumerate(METHODS[objective]):
        for i, condition in enumerate(conditions):
            values = values_for(runs, condition, method, region)
            if not len(values):
                continue
            center = i+(j-1)*width
            mean, sd = float(values.mean()), float(values.std(ddof=1)) if len(values)>1 else 0
            peak = max(peak, mean+sd)
            ax.bar(center, mean, width=width*0.9, yerr=sd, capsize=3,
                   color=COLORS[j], alpha=0.85, label=label if i == 0 else None,
                   error_kw={"linewidth": 1.1})
            ax.scatter(center+np.linspace(-width*0.18, width*0.18, len(values)), values,
                       color="black", s=11, zorder=4, alpha=0.6)
            if annotate:
                ax.annotate(f"{mean:.3f}", (center, mean+sd), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, peak*1.38 if peak else 1)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left")


def plot_overall(runs, output, ylabel):
    systems = [s for s in SYSTEM_NAMES if any(r["system"] == s for r in runs)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, objective in zip(axes, ("causal", "masked")):
        conditions = [f"{s}-{objective}" for s in systems]
        if not any(run["condition"] in conditions for run in runs):
            ax.set_visible(False)
            continue
        bars(ax, runs, conditions, [SYSTEM_NAMES[s] for s in systems], objective)
        ax.set_ylabel(ylabel)
        ax.set_title("Causal future-patch prediction" if objective == "causal" else "Bidirectional masked reconstruction")
    fig.suptitle("Held-out reconstruction by method — lower MSE is better", fontsize=15)
    fig.text(0.5, 0.025, "Bars: mean across seed pairs; error bars: sample SD; black dots: individual seed-pair means.\n"
             "Each seed score first averages within trajectories, then across test trajectories. Causal and masked targets/information differ.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save_figure(fig, output, "reconstruction_methods")


def plot_regions(runs, output, ylabel):
    systems = [s for s in ("predictable", "random") if any(r["system"] == s for r in runs)]
    if not systems:
        return
    fig, axes = plt.subplots(len(systems), 2, figsize=(13, 4.1*len(systems)), squeeze=False)
    for row, system in enumerate(systems):
        for col, objective in enumerate(("causal", "masked")):
            ax = axes[row, col]
            condition = f"{system}-{objective}"
            if not any(run["condition"] == condition for run in runs):
                ax.set_visible(False)
                continue
            width = 0.23
            peak = 0
            for j, (method, label) in enumerate(METHODS[objective]):
                for i, region in enumerate(("stable", "near_transition")):
                    values = values_for(runs, condition, method, region)
                    if not len(values):
                        continue
                    center = i+(j-1)*width
                    mean, sd = values.mean(), values.std(ddof=1) if len(values)>1 else 0
                    peak = max(peak, mean+sd)
                    ax.bar(center, mean, width=width*0.9, yerr=sd, capsize=3,
                           color=COLORS[j], alpha=0.85, label=label if i == 0 else None)
                    ax.scatter(center+np.linspace(-0.04, 0.04, len(values)), values,
                               s=11, color="black", alpha=0.6, zorder=4)
                    ax.annotate(f"{mean:.3f}", (center, mean+sd), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8)
            ax.set_xticks([0, 1], ["Stable intervals", "Near transitions"])
            ax.set(title=f"{SYSTEM_NAMES[system]} · {objective}", ylabel=ylabel, ylim=(0, peak*1.38 if peak else 1))
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
    radius = runs[0]["cfg"].get("evaluation", {}).get("transition_radius", 4)
    fig.suptitle("Reconstruction during stable intervals and near regime switches", fontsize=15)
    fig.text(0.5, 0.02, f"Near transition: target samples within ±{radius} samples of an actual switch. "
             "Mean ± SD across seed pairs; lower is better.\nAR(2) has no switching events. Regions and eligible target samples differ across objectives.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    save_figure(fig, output, "reconstruction_transitions")


def plot_difficulty(runs, output, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    specifications = [("causal", ["predictable-causal", "predictable-long-horizon"], "horizon", "Prediction horizon"),
                      ("masked", ["predictable-masked", "predictable-long-span"], "mask_span", "Requested mask span")]
    for ax, (objective, conditions, key, label) in zip(axes, specifications):
        if not all(any(run["condition"] == condition for run in runs) for condition in conditions):
            ax.set_visible(False)
            continue
        labels = []
        for condition in conditions:
            cfg = next(run["cfg"] for run in runs if run["condition"] == condition)
            labels.append(f"{label}: {cfg['model'][key]} patches")
        bars(ax, runs, conditions, labels, objective)
        ax.set_yscale("log")
        observed = np.concatenate([values_for(runs, condition, method, "all")
                                   for condition in conditions for method, _ in METHODS[objective]])
        positive = observed[observed > 0]
        ax.set_ylim(positive.min()*0.6, positive.max()*2.8)
        ax.set_ylabel(ylabel+" (log scale)")
        ax.set_title(f"Predictable switching · {objective}")
    fig.suptitle("Reconstruction as the prediction horizon or mask span increases", fontsize=15)
    fig.text(0.5, 0.025, "Logarithmic y-axis keeps neural and baseline errors visible together. Mean ± SD across seed pairs.\n"
             "Longer horizons/spans change the task and target samples; these are separate configurations.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save_figure(fig, output, "reconstruction_difficulty")


def plot_examples(runs, output, ylabel):
    systems = [s for s in SYSTEM_NAMES if any(r["system"] == s for r in runs)]
    fig, axes = plt.subplots(len(systems), 2, figsize=(14, 3.0*len(systems)), squeeze=False)
    first_seed = runs[0]["seed"]
    for row, system in enumerate(systems):
        for col, objective in enumerate(("causal", "masked")):
            ax = axes[row, col]
            run = next((r for r in runs if r["condition"] == f"{system}-{objective}" and r["seed"] == first_seed), None)
            if run is None:
                ax.set_visible(False)
                continue
            with np.load(run["path"]/"reconstruction.npz") as example:
                x, p = example["x"], int(example["patch_size"])
                times = (example["indices"][:, None]*p+np.arange(p)).reshape(-1)
                selected = np.repeat(example["selected"], p)
                ax.plot(x[:, 0], color="black", linewidth=1.2, label="Ground truth")
                for j, (method, label) in enumerate(METHODS[objective]):
                    key = "prediction" if method == "neural" else "baseline_"+method
                    prediction = example[key][..., 0].reshape(-1).copy()
                    prediction[~selected] = np.nan
                    ax.plot(times, prediction, color=COLORS[j], label=label, linewidth=1.1,
                            alpha=1 if method == "neural" else 0.75)
                for index in example["indices"][example["selected"]]:
                    ax.axvspan(index*p-0.5, (index+1)*p-0.5, color="gray", alpha=0.055, linewidth=0)
            ax.set(title=f"{SYSTEM_NAMES[system]} · {objective}", xlabel="Time sample", ylabel="Channel 0")
            if row == 0:
                ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
                ax.set_title(f"{SYSTEM_NAMES[system]} · {objective}", pad=40)
    units = "train-normalized observations" if "normalized" in ylabel else "observations"
    fig.suptitle(f"Example test reconstructions — {units}", fontsize=15)
    fig.text(0.5, 0.01, f"First test trajectory, data/model seed pair {first_seed}; channel 0 only. "
             "Predictions are shown only at scored targets (gray shading).\n"
             "Summary MSE plots average all observation channels and all test trajectories; examples are illustrative.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    save_figure(fig, output, "reconstruction_examples")


def plot_reconstruction_comparisons(root, output=None):
    root = Path(root)
    output = Path(output) if output else root/"analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    ylabel = "Test MSE (train-normalized units)" if runs[0]["cfg"]["data"].get("normalize", True) else "Test MSE (raw units)"
    table = []
    for condition in dict.fromkeys(run["condition"] for run in runs):
        representative = next(run for run in runs if run["condition"] == condition)
        for method, _ in METHODS[representative["objective"]]:
            for region in ("all", "stable", "near_transition"):
                values = values_for(runs, condition, method, region)
                table.append({"condition": condition, "system": representative["system"],
                              "objective": representative["objective"], "method": method, "region": region,
                              "mean_mse": float(values.mean()) if len(values) else None,
                              "sd_across_seed_pairs": float(values.std(ddof=1)) if len(values)>1 else None,
                              "seed_pairs": len(values)})
    with (output/"reconstruction_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    plot_overall(runs, output, ylabel)
    plot_regions(runs, output, ylabel)
    plot_difficulty(runs, output, ylabel)
    plot_examples(runs, output, ylabel)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(plot_reconstruction_comparisons(args.root, args.output))


if __name__ == "__main__":
    main()
