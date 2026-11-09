"""Paired-seed study: validation-only budget selection, controls, and seed-level reports."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cli import load_config, override
from .experiment import run_experiment, save_json
from .plots import finish


def validation_trend(history, window):
    if len(history) < 2*window:
        raise ValueError("Need at least two validation windows")
    previous = min(row["validation_loss"] for row in history[-2*window:-window])
    recent = min(row["validation_loss"] for row in history[-window:])
    return {"previous_window_best": previous, "recent_window_best": recent,
            "relative_improvement": (previous-recent)/max(previous, 1e-12),
            "best_epoch": min(history, key=lambda r: r["validation_loss"])["epoch"]}


def plan_runs(cfg, matrix):
    runs = []
    for seed in cfg["study"]["seeds"]:
        for entry in matrix:
            changes = {**entry["overrides"], "data.seed": seed["data"], "training.seed": seed["model"]}
            runs.append((f"{entry['name']}-d{seed['data']}-m{seed['model']}", entry["name"], override(cfg, changes)))
    if len({name for name, _, _ in runs}) != len(runs):
        raise ValueError("Study run names must be unique")
    return runs


def run_study(cfg, matrix, root, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = {"config": cfg, "matrix": matrix}
    plan_path = root/"study_plan.json"
    if plan_path.exists():
        if not resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing study requires --resume with an identical plan")
    save_json(plan_path, plan)
    runs = plan_runs(cfg, matrix)
    options = cfg["study"]
    pilot, maximum = options["pilot_epochs"], options["maximum_epochs"]
    if not 2*options["window"] <= pilot <= maximum:
        raise ValueError("Require 2*window <= pilot_epochs <= maximum_epochs")
    decision_path = root/"budget_decision.json"
    if not decision_path.exists():
        trends = {}
        for i, (name, _, config) in enumerate(runs):
            config = override(config, {"training.epochs": pilot})
            save_json(root/"study_status.json", {"stage": "pilot_training", "run": name, "index": i+1, "runs": len(runs)})
            history = run_experiment(config, root/name, resume=(root/name/"config.json").exists(), evaluate=False)
            trends[name] = validation_trend(history, options["window"])
        extend = any(t["relative_improvement"] >= options["relative_improvement"] for t in trends.values())
        save_json(decision_path, {"selection_data": "validation only; no test metrics inspected",
                                 "rule": "Extend every run to the common maximum budget if any pilot run exceeds the relative improvement threshold",
                                 "threshold": options["relative_improvement"], "window": options["window"],
                                 "epochs": maximum if extend else pilot, "pilot_trends": trends})
    decision = json.loads(decision_path.read_text())
    # Complete the common training budget before opening test metrics for any condition.
    for i, (name, _, config) in enumerate(runs):
        config = override(config, {"training.epochs": decision["epochs"]})
        save_json(root/"study_status.json", {"stage": "final_training", "run": name, "index": i+1, "runs": len(runs), "epochs": decision["epochs"]})
        run_experiment(config, root/name, resume=True, evaluate=False, allow_epoch_extension=True)
    for i, (name, _, config) in enumerate(runs):
        save_json(root/"study_status.json", {"stage": "test_and_controls", "run": name, "index": i+1, "runs": len(runs)})
        if (root/name/"metrics.json").exists():
            continue
        config = override(config, {"training.epochs": decision["epochs"]})
        run_experiment(config, root/name, resume=True)
    report_study(root)
    save_json(root/"study_status.json", {"stage": "complete", "runs": len(runs), "epochs": decision["epochs"]})


def report_study(root):
    root = Path(root)
    analysis = root/"analysis"
    analysis.mkdir(exist_ok=True)
    plan = json.loads((root/"study_plan.json").read_text())
    conditions = {name: condition for name, condition, _ in plan_runs(plan["config"], plan["matrix"])}
    rows, curves, trends = [], [], {}
    for name, condition in conditions.items():
        run = root/name
        metrics = json.loads((run/"metrics.json").read_text())
        cfg = json.loads((run/"config.json").read_text())
        identity = {"run": name, "condition": condition, "system": metrics["system"],
                    "objective": metrics["objective"], "data_seed": cfg["data"]["seed"],
                    "model_seed": cfg["training"]["seed"]}
        for method, regions in metrics["reconstruction"].items():
            for region, values in regions.items():
                rows.append({**identity, "source": method, "factor": f"reconstruction/{region}",
                             "metric": "mse", "value": values["mean"]})
        for source, probes in {"trained_encoder": metrics["probes"], **metrics.get("probe_controls", {})}.items():
            for factor, values in probes.items():
                metric = "balanced_accuracy" if factor.startswith("local/q/") else "r2"
                rows.append({**identity, "source": source, "factor": factor,
                             "metric": metric, "value": values["test_pooled"][metric]})
                if source == "trained_encoder":
                    original = metrics["probe_controls"]["untrained_encoder"][factor]["test_pooled"][metric]
                    rows.append({**identity, "source": "trained_minus_untrained", "factor": factor,
                                 "metric": metric, "value": values["test_pooled"][metric]-original
                                 if values["test_pooled"][metric] is not None and original is not None else None})
        history = json.loads((run/"history.json").read_text())
        trends[name] = validation_trend(history, plan["config"]["study"]["window"])
        curves.extend({**identity, **r} for r in history)
    for filename, table in (("seed_metrics.csv", rows), ("learning_curves.csv", curves)):
        with (analysis/filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    aggregates = []
    keys = sorted({(r["condition"], r["source"], r["factor"], r["metric"]) for r in rows})
    for condition, source, factor, metric in keys:
        values = [r["value"] for r in rows if (r["condition"], r["source"], r["factor"], r["metric"]) == (condition, source, factor, metric) and r["value"] is not None]
        aggregates.append({"condition": condition, "source": source, "factor": factor, "metric": metric,
                           "mean": float(np.mean(values)) if values else None,
                           "sd_across_seed_pairs": float(np.std(values, ddof=1)) if len(values)>1 else None,
                           "seed_pairs": len(values)})
    save_json(analysis/"aggregate_metrics.json", aggregates)
    save_json(analysis/"final_validation_trends.json", trends)
    with (analysis/"aggregate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    plot_study(analysis, rows, curves)
    from .reconstruction_plots import plot_reconstruction_comparisons
    plot_reconstruction_comparisons(root, analysis)
    write_report(root, aggregates, trends)


def plot_study(output, rows, curves):
    import matplotlib.pyplot as plt
    sources = ["trained_encoder", "untrained_encoder", "raw_patch", "raw_statistics"]
    for system in sorted({r["system"] for r in rows}):
        base = [r for r in rows if r["system"] == system and r["condition"] in {system+"-causal", system+"-masked"}]
        factors = sorted({r["factor"] for r in base if r["source"] == "trained_encoder"})
        fig, axes = plt.subplots(2, (len(factors)+1)//2, figsize=(max(11, len(factors)*1.5), 7), squeeze=False)
        for ax, factor in zip(axes.flat, factors):
            for i, source in enumerate(sources):
                for offset, objective in ((-0.12, "causal"), (0.12, "masked")):
                    values = [r["value"] for r in base if r["source"] == source and r["factor"] == factor and r["objective"] == objective and r["value"] is not None]
                    if values:
                        ax.errorbar(i+offset, np.mean(values), yerr=np.std(values, ddof=1), fmt="o",
                                    color="tab:blue" if objective == "causal" else "tab:orange", label=objective if i == 0 else None)
            ax.axhline(0.5 if factor.startswith("local/q/") else 0, color="gray", linestyle=":")
            ax.set_title(factor, fontsize=9)
            ax.set_ylabel("Balanced accuracy" if factor.startswith("local/q/") else "Test R²")
            ax.set_xticks(range(len(sources)), ["trained", "untrained", "raw patch", "raw stats"], rotation=35, fontsize=7)
            ax.legend(fontsize=6)
        for ax in list(axes.flat)[len(factors):]:
            ax.set_visible(False)
        seed_count = len({(r["data_seed"], r["model_seed"]) for r in base})
        fig.suptitle(f"{system}: mean ± SD across {seed_count} seed pairs; independent axes per latent")
        finish(fig, output/f"{system}_recoverability_controls.png")
    conditions = sorted({r["condition"] for r in curves})
    fig, axes = plt.subplots(2, (len(conditions)+1)//2, figsize=(16, 7), squeeze=False)
    for ax, condition in zip(axes.flat, conditions):
        for seed in sorted({r["data_seed"] for r in curves}):
            selected = [r for r in curves if r["condition"] == condition and r["data_seed"] == seed]
            ax.plot([r["epoch"] for r in selected], [r["validation_loss"] for r in selected], label=f"data seed {seed}")
        ax.set(title=condition, xlabel="Epoch", ylabel="Validation objective MSE")
        ax.legend(fontsize=6)
    finish(fig, output/"validation_curves.png")


def write_report(root, aggregates, trends):
    decision = json.loads((root/"budget_decision.json").read_text())
    plan = json.loads((root/"study_plan.json").read_text())
    cfg = plan["config"]
    counts = cfg["data"]["splits"]
    seed_pairs = [(s["data"], s["model"]) for s in cfg["study"]["seeds"]]
    def cell(condition, source, factor):
        r = next((r for r in aggregates if r["condition"] == condition and r["source"] == source and r["factor"] == factor), None)
        return f"{r['mean']:.3f} ± {r['sd_across_seed_pairs']:.3f}" if r and r["mean"] is not None else "—"
    conditions = sorted({r["condition"] for r in aggregates})
    lines = ["# Larger paired-seed study", "", f"{counts['train']} training / {counts['validation']} validation / {counts['test']} test independent trajectories per seed; {cfg['data']['length']} samples, {cfg['data']['channels']} channels. Paired simulation/initialization seeds: {seed_pairs}. Same architecture, AdamW, batch size {cfg['training']['batch_size']} and update budget for both objectives.", "",
             f"All {len(conditions)*len(seed_pairs)} runs completed {decision['epochs']} epochs. The {cfg['study']['pilot_epochs']}-epoch pilot budget decision used validation only; test evaluation followed completion of the final common training budget. Values below are mean ± sample SD across {len(seed_pairs)} seed pairs, not confidence intervals.", "",
             "## Reconstruction", "", "| Condition | Neural MSE | Persistence / interpolation MSE |", "| --- | ---: | ---: |"]
    for c in conditions:
        lines.append(f"| {c} | {cell(c, 'neural', 'reconstruction/all')} | {cell(c, 'linear' if 'masked' in c or 'span' in c else 'constant', 'reconstruction/all')} |")
    lines += ["", "## Local state recovery", "", "| Condition | Accumulator R² | Phase R² | Regime balanced accuracy | AR state R² |", "| --- | ---: | ---: | ---: | ---: |"]
    for c in conditions:
        lines.append("| "+c+" | "+" | ".join(cell(c,"trained_encoder",f"local/{f}/linear") for f in ("u","phase","q","state"))+" |")
    lines += ["", "## Change from the same encoder before training", "", "| Condition | Δ accumulator R² | Δ phase R² | Δ regime balanced accuracy | Δ AR state R² |", "| --- | ---: | ---: | ---: | ---: |"]
    for c in conditions:
        lines.append("| "+c+" | "+" | ".join(cell(c,"trained_minus_untrained",f"local/{f}/linear") for f in ("u","phase","q","state"))+" |")
    lines += ["", "## Trajectory parameters", "", "| Condition | r R² | ω R² | δ R² | τ R² | ω₀ R² | ω₁ R² |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in conditions:
        lines.append("| "+c+" | "+" | ".join(cell(c,"trained_encoder",f"sequence/{f}/linear") for f in ("r","omega","delta","threshold","omega0","omega1"))+" |")
    threshold = decision["threshold"]
    improving = [name for name,t in trends.items() if t["relative_improvement"] >= threshold]
    lines += ["", "## Validation and interpretation", "",
              f"{len(improving)} of {len(trends)} runs still improved their best validation loss by at least {threshold:.0%} comparing the final two 10-epoch windows. This descriptive stopping diagnostic does not prove convergence. Full trends and best epochs are in `analysis/final_validation_trends.json`.", "",
              "Untrained controls use the identical initial architecture/weights and the same input access as their trained counterpart. Raw-patch features use the current flattened patch locally and patch means globally. Raw-statistics features add past-window mean/std and centered per-channel lag correlations (lags 1,2,4,8) locally, using at most the causal context; sequence features use full-sequence mean/std/lag correlations. They use observations only, no latent labels. Local raw controls have no future access; masked encoder locals do. All feature scaling and probe fitting use training data, and regularization uses validation data.", "",
              "Masked locals see the full unmasked test sequence, including future observations. Causal locals see bounded past/current patches. Sequence embeddings mean-pool local states. Reconstruction targets and information sets differ across objectives. Pooled R² scores are descriptive; per-trajectory results remain in each run's metrics. Simulation and initialization seeds vary jointly, so this study does not isolate their individual sources of variability. No test score selected the training budget. The study is a stronger pilot, not proof of shortcut learning or general superiority.", "",
              "See `analysis/seed_metrics.csv`, `aggregate_metrics.csv`, `learning_curves.csv` and per-system `*_recoverability_controls.png` figures. Each latent has its own plot axis to keep positive local-state scores visible alongside negative parameter scores."]
    (root/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.json")
    parser.add_argument("--matrix", default="configs/matrix.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report_study(args.output)
    else:
        run_study(load_config(args.config), load_config(args.matrix), args.output, args.resume)


if __name__ == "__main__":
    main()
