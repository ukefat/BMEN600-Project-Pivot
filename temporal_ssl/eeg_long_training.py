"""Extend the short-mask EEG pilot; fixed-mask train/validation diagnostics only."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from .baselines import fill_masked
from .data import Normalizer
from .eeg_course.download import sha256
from .eeg_data import EEGDataset
from .eeg_pilot import make_eeg_model, summarize_participant_mse
from .experiment import loader, save_json, seed_everything, train
from .models import patchify, span_mask
from .plots import plt, finish


def fixed_baselines(dataset, model, batch_size, seed, path):
    """Identical masks for every checkpoint; only observed samples enter interpolation."""
    generator = torch.Generator().manual_seed(seed)
    rows, masks = [], []
    offset = 0
    for batch in loader(dataset, batch_size):
        x = batch["x"].numpy()
        patches = x.reshape(len(x), -1, model.patch_size, x.shape[-1])
        selected = span_mask(len(x), patches.shape[1], model.mask_ratio, model.mask_span, generator).numpy()
        masks.extend(selected)
        for j, (target, mask) in enumerate(zip(patches, selected)):
            predictions = {"training_mean": np.zeros_like(target),
                           "linear": fill_masked(target, mask, "linear"),
                           "nearest": fill_masked(target, mask, "nearest")}
            for method, prediction in predictions.items():
                rows.append({"epoch": offset+j, "participant_id": dataset.metadata[offset+j]["participant_id"],
                             "method": method, "mse": float(np.square(prediction[mask]-target[mask]).mean())})
        offset += len(x)
    pd.DataFrame(rows).to_csv(path/"baseline_epochs.csv", index=False)
    summary, participants = summarize_participant_mse(rows)
    participants.to_csv(path/"baseline_participants.csv", index=False)
    masks = np.stack(masks)
    np.savez_compressed(path/"fixed_masks.npz", mask=masks)
    return summary, masks


@torch.no_grad()
def evaluate_fixed(model, dataset, masks, batch_size, path):
    model.eval()
    rows, examples = [], []
    squared_prediction, squared_target, coordinates = 0., 0., 0
    offset = 0
    device = next(model.parameters()).device
    for batch in loader(dataset, batch_size):
        x = batch["x"].to(device)
        mask = torch.from_numpy(masks[offset:offset+len(x)]).to(device)
        result = model(x, mask=mask)
        target = patchify(x, model.patch_size)
        prediction = result["prediction"]
        error = (prediction-target).square().mean(-1)
        for j in range(len(x)):
            metadata = dataset.metadata[offset+j]
            rows.append({"epoch": offset+j, "participant_id": metadata["participant_id"],
                         "method": "neural", "mse": float(error[j][mask[j]].mean())})
            if metadata["group"] not in {example["group"] for example in examples}:
                examples.append({"x": x[j].cpu().numpy(), "mask": mask[j].cpu().numpy(),
                                 "prediction": prediction[j].cpu().numpy().reshape(-1, model.patch_size, x.shape[-1]),
                                 "participant_id": metadata["participant_id"], "group": metadata["group"]})
        squared_prediction += float(prediction[mask].double().square().sum())
        squared_target += float(target[mask].double().square().sum())
        coordinates += prediction[mask].numel()
        offset += len(x)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    summary, participant = summarize_participant_mse(rows)
    participant.to_csv(path.parent/f"{path.name}_participants.csv", index=False)
    np.savez_compressed(path.parent/f"{path.name}_examples.npz",
                        **{f"{i}_{k}":v for i,e in enumerate(examples) for k,v in e.items()})
    return {**summary["neural"], "prediction_rms": float(np.sqrt(squared_prediction/coordinates)),
            "target_rms": float(np.sqrt(squared_target/coordinates))}


def write_report(output):
    protocol = json.loads((output/"protocol.json").read_text())
    history = pd.DataFrame(json.loads((output/"masked_short/history.json").read_text()))
    metrics = pd.read_csv(output/"fixed_evaluations.csv")
    baselines = json.loads((output/"baselines.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, split in zip(axes, ("train", "validation")):
        ax.plot(history.epoch, history["train_loss" if split == "train" else "validation_loss"],
                color="C0", alpha=.6, label="Online train loss (changing masks)" if split == "train" else "Validation (fixed masks)")
        frame = metrics[(metrics.split == split) & (metrics.checkpoint == "last")]
        ax.plot(frame.epoch, frame["mean"], "o-", color="C1", label="Epoch-end model (fixed masks)")
        for method, label, color in (("training_mean", "Training-channel mean", "gray"),
                                      ("linear", "Linear interpolation", "C2")):
            ax.axhline(baselines[split][method]["mean"], color=color, linestyle="--", label=label)
        ax.axvline(protocol["original_epochs"], color="black", lw=.8, linestyle=":")
        ax.set(title=f"{split.capitalize()} participants", xlabel="Total training epochs", ylabel="Normalized masked MSE")
        ax.legend(fontsize=8)
    fig.suptitle("Same short-mask EEG model and data; longer training")
    finish(fig, output/"learning_curves.png")

    start, end = protocol["original_epochs"], int(history.epoch.max())
    with np.load(output/f"epoch_{start:03d}_last_validation_examples.npz") as before, \
         np.load(output/f"epoch_{end:03d}_last_validation_examples.npz") as after:
        fig, axes = plt.subplots(3, 2, figsize=(13, 7), sharex=True)
        p, fs = protocol["config"]["model"]["patch_size"], protocol["config"]["sampling_hz"]
        for i in range(3):
            for ax, example, epoch in zip(axes[i], (before, after), (start, end)):
                x, mask, pred = [example[f"{i}_{k}"] for k in ("x", "mask", "prediction")]
                hidden = np.repeat(mask, p)
                neural = pred[..., 0].ravel().copy()
                linear = fill_masked(x.reshape(-1,p,x.shape[-1]), mask)[...,0].ravel()
                neural[~hidden], linear[~hidden] = np.nan, np.nan
                t = np.arange(len(x))/fs
                ax.plot(t, x[:,0], color="black", lw=.8, label="Observed EEG")
                ax.plot(t, neural, color="C1", lw=1, label="Model at hidden samples")
                ax.plot(t, linear, color="C2", lw=.8, alpha=.8, label="Interpolation")
                ax.set(xlim=(0,2), ylabel="Normalized Fp1", title=f"{example[f'{i}_participant_id']} ({example[f'{i}_group']}), epoch {epoch}")
        axes[0,0].legend(fontsize=8)
        axes[-1,0].set_xlabel("Seconds (first 2 s of an 8 s window)")
        axes[-1,1].set_xlabel("Seconds (first 2 s of an 8 s window)")
        finish(fig, output/"validation_examples.png")

    table = ["| Checkpoint | Train MSE | Validation MSE | Train prediction RMS | Validation prediction RMS |",
             "|---|---:|---:|---:|---:|"]
    for (epoch, checkpoint), frame in metrics.groupby(["epoch", "checkpoint"], sort=False):
        values = frame.set_index("split")
        table.append(f"| {epoch}: {checkpoint} | {values.loc['train','mean']:.6f} | {values.loc['validation','mean']:.6f} | "
                     f"{values.loc['train','prediction_rms']:.4f} | {values.loc['validation','prediction_rms']:.4f} |")
    baseline_table = ["| Baseline | Train MSE | Validation MSE |", "|---|---:|---:|"]
    for method in ("training_mean", "linear", "nearest"):
        baseline_table.append(f"| {method} | {baselines['train'][method]['mean']:.6f} | {baselines['validation'][method]['mean']:.6f} |")
    best = min(history.to_dict("records"), key=lambda row:row["validation_loss"])
    text = f"""# Longer short-mask EEG training diagnostic

Question: does the near-mean pilot reconstruction improve with more training on the same data, or does fitting separate from validation generalization?

Continued the original epoch-{start} `masked_short/last.pt` to **{end} total epochs**, preserving model, AdamW, shuffle/mask RNG, Torch/CUDA RNG, and validation-selected best state. The original pilot artifacts are unchanged. Same 56,411-parameter model, 25-sample (50 ms) patches, 25% masked reconstruction, batch size 16, learning rate 0.001, weight decay 0.01, clip 1.0, seed 600. No architecture/optimizer/data sweep. Each training epoch uses new masks.

The original 52 training participants (1,248 windows) and 18 validation participants (432 windows) are loaded. All 8 s × 19-channel windows and training-only channel normalization are unchanged. **No test participant EEG is loaded or evaluated.** This is a diagnostic using the existing validation split, not a fresh held-out confirmation. Epochs {start}–{end} correspond to {end*78:,} total optimizer updates and {end} presentations per training window.

Fixed evaluation masks use seed 604 for training and 603 for validation, matching the original validation-loss protocol. The same missing samples are scored for models, channel-mean zero predictions, and visible-sample interpolation. Metrics first average epochs within participant, then participants; all participants have 24 windows. Online training loss averages losses while weights and masks change; epoch-end fixed-mask training loss evaluates a single checkpoint. Their curves have different sampling conventions.

## Fixed-mask reconstruction

{chr(10).join(baseline_table)}

{chr(10).join(table)}

`last` is the model after that total epoch count. `best` is the lowest validation-loss state seen by that milestone; it may come from an earlier epoch. Final selected epoch: **{best['epoch']}**, validation MSE **{best['validation_loss']:.6f}**. Prediction RMS measures the size of the output around the training-channel means (zero in normalized units); target RMS is recorded in the numerical results for context. Low output RMS together with near-mean error supports the visual impression of small-amplitude predictions.

![Training and validation curves](eeg_long_training_assets/learning_curves.png)
![Same validation examples before and after](eeg_long_training_assets/validation_examples.png)

## Reproducibility

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_long_training \\
  --pilot results/eeg_pilot --output results/eeg_long_training --epochs 100
# To continue an interrupted run, add --resume with the identical configuration.
```

Use a new output directory for a fresh repeat. The runner copies the original last checkpoint into its own directory; it never writes to the pilot. It saves the exact evaluation masks, per-window/per-participant metrics, examples, original checkpoint hash, input/configuration/code hashes, original and extended training history, and continuation checkpoints. Runtime: {(json.loads((output/'timing.json').read_text()))['seconds']:.1f} seconds including cache loading, baselines, training, and evaluation (excluding final report rendering).

Numerical results: [fixed evaluations](eeg_long_training_assets/fixed_evaluations.csv), [epoch history](eeg_long_training_assets/history.csv), [baselines](eeg_long_training_assets/baselines.json).
"""
    # Interpretation is added from measured endpoints, not a preselected outcome.
    first = metrics[(metrics.epoch == start) & (metrics.checkpoint == "last")].set_index("split")
    last = metrics[(metrics.epoch == end) & (metrics.checkpoint == "last")].set_index("split")
    gains = {s:1-last.loc[s,"mean"]/baselines[s]["training_mean"]["mean"] for s in ("train", "validation")}
    interpretation = (f"## Interpretation\n\nAt the final epoch, reconstruction reduces MSE versus the fitted channel-mean baseline by "
        f"**{gains['train']*100:.2f}% on training** and **{gains['validation']*100:.2f}% on validation**. "
        f"From epoch {start} to {end}, fixed-mask train MSE changes from {first.loc['train','mean']:.6f} to {last.loc['train','mean']:.6f}; "
        f"validation changes from {first.loc['validation','mean']:.6f} to {last.loc['validation','mean']:.6f}. ")
    if min(gains.values()) < .1 and max(gains.values()) < .1:
        interpretation += ("Both errors remain close to the channel-mean baseline. Longer training at these settings did not produce useful "
            "reconstruction of the readily interpolable gaps, and there is no large training/validation separation showing successful fitting "
            "followed by failed transfer. This does not establish insufficient training data as the main cause. Optimization/representation/decoder "
            "design remain candidates; these results do not distinguish them.\n\n")
    elif min(gains.values()) >= .1:
        interpolation_ratio = last.loc['validation','mean']/baselines['validation']['linear']['mean']
        interpretation += ("Both training and validation reconstruction improve substantially on unchanged data. The original 20-epoch pilot "
            "was too short for this configuration to leave its near-mean behavior. This supports training duration as an important contributor "
            "to that earlier failure, rather than establishing that more training data was required. "
            f"Final validation error is still {interpolation_ratio:.2f} times the linear interpolation error; the remaining reconstruction gap "
            "should not be overlooked.\n\n")
    else:
        interpretation += ("Assess the training/validation curves and interpolation gap together: improvement alone does not show that more data is "
            "needed, and any remaining train/validation difference can have multiple causes.\n\n")
    interpretation += ("This is one seed and one unchanged learning rate/budget extension. It cannot rule out improvements with other optimization "
        "settings or more data. It evaluates masked waveform reconstruction, not causal predictability or diagnosis accuracy; no classifier was retrained. "
        "Earlier classification and spectral-probe results still refer to the original pilot checkpoints. They cannot be assumed to describe these "
        "longer-trained weights, or used to establish an intrinsic limitation of the SSL objectives.\n\n")
    text = text.replace("## Reproducibility", interpretation+"## Reproducibility")
    assets = Path("reports/eeg_long_training_assets")
    assets.mkdir(parents=True, exist_ok=True)
    history.to_csv(output/"history.csv", index=False)
    for filename in ("learning_curves.png", "validation_examples.png", "fixed_evaluations.csv", "history.csv", "baselines.json"):
        shutil.copyfile(output/filename, assets/filename)
    Path("reports/eeg_long_training.md").write_text(text)
    (output/"report.md").write_text(text.replace("eeg_long_training_assets/", ""))
    save_json(output/"report_provenance.json", {"report_source_sha256":sha256(Path(__file__)),
              "fixed_evaluations_sha256":sha256(output/"fixed_evaluations.csv"),
              "note":"Training source snapshot is retained separately in environment.json."})


def run(pilot, output, epochs=100, device="cuda", resume=False):
    pilot, output = Path(pilot), Path(output)
    original_checkpoint = pilot/"masked_short/last.pt"
    checkpoint_hash = sha256(original_checkpoint)
    original = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
    start = original["epoch"]+1
    if epochs <= start:
        raise ValueError(f"Total epochs must exceed original {start}")
    cfg = json.loads((pilot/"config.json").read_text())
    cfg["training"].update(epochs=epochs, device=device)
    partitions = json.loads((pilot/"participants.json").read_text())
    ids = {k:{r['participant_id'] for r in rows} for k,rows in partitions.items()}
    if any(ids[a]&ids[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Participant partitions overlap")
    protocol = {"config":cfg, "pilot":str(pilot.resolve()), "original_epochs":start,
                "original_last_sha256":checkpoint_hash, "evaluated_splits":["train", "validation"],
                "fixed_mask_seeds":{"train":604, "validation":603},
                "normalization_sha256":sha256(pilot/"normalization.npz"),
                "partitions_sha256":sha256(pilot/"participants.json")}
    if (output/"protocol.json").exists():
        if not resume or json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("Use a new output directory, or --resume with identical configuration")
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/"protocol.json",protocol)
    model_path = output/"masked_short"
    model_path.mkdir(exist_ok=True)
    if not (model_path/"last.pt").exists():
        shutil.copyfile(original_checkpoint, model_path/"last.pt")
    save_json(output/"environment.json", {"torch":torch.__version__,"numpy":np.__version__,
        "gpu":torch.cuda.get_device_name(0) if device=="cuda" else "cpu",
        "source_sha256":{str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')}})
    begin = time.perf_counter()
    with np.load(pilot/"normalization.npz") as stored:
        normalizer = Normalizer(stored["mean"],stored["std"])
    # Only these two partitions are materialized; test cache paths are never opened.
    datasets = {k:EEGDataset(partitions[k], normalizer) for k in ("train", "validation")}
    seed_everything(cfg["training"]["seed"],cfg["training"]["threads"])
    model = make_eeg_model(cfg,"masked_short", datasets["train"].x.shape[1:]).to(device)
    masks, baselines = {}, {}
    for split, dataset in datasets.items():
        path = output/split
        path.mkdir(exist_ok=True)
        baselines[split],masks[split] = fixed_baselines(dataset,model,cfg["training"]["batch_size"],
                                    protocol["fixed_mask_seeds"][split],path)
        print(f"{split} fixed baselines: mean={baselines[split]['training_mean']['mean']:.6f}, "
              f"interpolation={baselines[split]['linear']['mean']:.6f}",flush=True)
    save_json(output/"baselines.json",baselines)
    rows = pd.read_csv(output/"fixed_evaluations.csv").to_dict("records") if (output/"fixed_evaluations.csv").exists() else []
    def evaluate(epoch, checkpoint, state):
        model.load_state_dict(state)
        for split,dataset in datasets.items():
            if any(r["epoch"]==epoch and r["checkpoint"]==checkpoint and r["split"]==split for r in rows):
                continue
            scores = evaluate_fixed(model,dataset,masks[split],cfg["training"]["batch_size"],
                                    output/f"epoch_{epoch:03d}_{checkpoint}_{split}")
            rows.append({"epoch":epoch,"checkpoint":checkpoint,"split":split,**scores})
            pd.DataFrame(rows).to_csv(output/"fixed_evaluations.csv",index=False)
            print(f"Epoch {epoch} {checkpoint} {split}: MSE={scores['mean']:.6f}, output RMS={scores['prediction_rms']:.4f}",flush=True)
    evaluate(start,"last",original["model"])
    evaluate(start,"best",original["best_model"])
    milestones = sorted(set([e for e in range(start+20,epochs+1,20)]+[epochs]))
    for milestone in milestones:
        chunk_cfg = copy.deepcopy(cfg)
        chunk_cfg["training"]["epochs"] = milestone
        # train restores optimizer and all RNG state from last.pt after diagnostic evaluation.
        train(model,datasets,chunk_cfg,model_path,torch.device(device),resume=True)
        checkpoint = torch.load(model_path/"last.pt",map_location="cpu",weights_only=False)
        current = checkpoint["epoch"]+1
        if current != milestone:
            # A resumed later checkpoint can already exceed this milestone.
            continue
        evaluate(current,"last",checkpoint["model"])
        evaluate(current,"best",checkpoint["best_model"])
    save_json(output/"timing.json",{"seconds":time.perf_counter()-begin})
    if sha256(original_checkpoint) != checkpoint_hash:
        raise RuntimeError("Original pilot checkpoint unexpectedly changed")
    write_report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",default="results/eeg_pilot")
    parser.add_argument("--output",default="results/eeg_long_training")
    parser.add_argument("--epochs",type=int,default=100,help="Total epochs including original pilot")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    run(args.pilot,args.output,args.epochs,args.device,args.resume)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    main()
