"""Small inductive ds004504 SSL pilot. Run prepare, smoke, run, or report."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import fill_masked
from .cli import load_config
from .eeg_data import EEGDataset, aggregate_participants, fit_normalizer, prepare_cache, subject_splits
from .eeg_models import FirstHalfPrediction
from .experiment import loader, save_json, seed_everything, train
from .models import TemporalSSL, patchify
from .plots import plt, finish

CLASSES = ["AD", "FTD", "HC"]
ORDER = ["causal", "masked_short", "masked_long", "masked_medium"]


def make_eeg_model(cfg, name, shape):
    options = dict(cfg["model"])
    if name == "causal":
        return FirstHalfPrediction(shape[1], shape[0], **options)
    options["mask_span"] = cfg["spans"][name.split("_")[1]]
    return TemporalSSL(shape[1], shape[0], objective="masked", **options)


def half_baselines(x):
    half = len(x)//2
    history = x[:half]
    time_axis = np.arange(half, dtype=float)-(half-1)
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(half), time_axis]), history, rcond=None)[0]
    forecast = np.column_stack([np.ones(half), np.arange(1, half+1)]) @ coefficients
    return {"constant": np.repeat(history[-1:], half, axis=0), "linear_trend": forecast}


def mask_runs(mask):
    padded = np.r_[False, mask, False].astype(int)
    edges = np.diff(padded)
    return np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)


def summarize_participant_mse(rows):
    table = pd.DataFrame(rows)
    participant = table.groupby(["participant_id", "method"], as_index=False).mse.mean()
    output = {}
    for method, frame in participant.groupby("method"):
        values = frame.mse.to_numpy()
        output[method] = {"mean": float(values.mean()), "sd_across_participants": float(values.std(ddof=1)),
                          "participants": len(values)}
    return output, participant


@torch.no_grad()
def reconstruct(model, dataset, cfg, path):
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(cfg["evaluation_seed"])
    rows, spans, counts, masks, examples = [], [], [], [], []
    p = model.patch_size
    offset = 0
    for batch in loader(dataset, cfg["training"]["batch_size"]):
        result = model(batch["x"].to(device), generator=generator)
        predicted = result["prediction"].cpu().numpy().reshape(len(batch["x"]), -1, p, dataset.x.shape[-1])
        selected = result["selected"].cpu().numpy()
        indices = result["indices"].cpu().numpy()
        for j, tensor in enumerate(batch["x"]):
            x = tensor.numpy()
            patches = x.reshape(-1, p, x.shape[-1])
            if model.objective == "masked":
                baselines = {method: fill_masked(patches, selected[j], method) for method in ("linear", "nearest")}
                spans.extend(mask_runs(selected[j]).tolist())
                counts.append(int(selected[j].sum()))
                masks.append(selected[j])
            else:
                baselines = {k: v.reshape(-1, p, x.shape[-1]) for k, v in half_baselines(x).items()}
            baselines["training_mean"] = np.zeros_like(predicted[j])
            for method, prediction in {"neural": predicted[j], **baselines}.items():
                error = ((prediction-patches[indices])**2).mean(-1)[selected[j]].mean()
                rows.append({"epoch": offset+j, "participant_id": dataset.metadata[offset+j]["participant_id"],
                             "method": method, "mse": float(error)})
            metadata = dataset.metadata[offset+j]
            if len(examples) < 3 and metadata["group"] not in {ex["group"] for ex in examples}:
                examples.append({"x": x, "prediction": predicted[j], "selected": selected[j],
                                 "indices": indices, "group": metadata["group"], "participant_id": metadata["participant_id"],
                                 **{f"baseline_{k}": v for k,v in baselines.items()}})
        offset += len(batch["x"])
    summary, participant = summarize_participant_mse(rows)
    pd.DataFrame(rows).to_csv(path/"reconstruction_epochs.csv", index=False)
    participant.to_csv(path/"reconstruction_participants.csv", index=False)
    if spans:
        np.savez_compressed(path/"test_masks.npz", mask=np.stack(masks))
        values, frequencies = np.unique(spans, return_counts=True)
        duration = 1000*p/cfg["sampling_hz"]
        pd.DataFrame({"patches": values, "milliseconds": values*duration, "count": frequencies}).to_csv(path/"realized_mask_spans.csv", index=False)
        save_json(path/"mask_summary.json", {"ratio": 0.25, "requested_patches": model.mask_span,
                  "requested_ms": model.mask_span*duration, "hidden_patches_per_epoch": sorted(set(counts)),
                  "realized_mean_ms": float(np.mean(spans)*duration), "realized_median_ms": float(np.median(spans)*duration),
                  "realized_min_ms": float(min(spans)*duration), "realized_max_ms": float(max(spans)*duration),
                  "epochs": len(dataset), "distribution_scope": "one fixed mask per held-out epoch"})
    for i, example in enumerate(examples):
        np.savez_compressed(path/f"example_{i}.npz", **example)
    plot_examples(examples, cfg, model, path)
    return summary


def plot_examples(examples, cfg, model, path):
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.7*len(examples)), squeeze=False)
    p, fs = model.patch_size, cfg["sampling_hz"]
    for ax, ex in zip(axes.flat, examples):
        x = ex["x"]
        t = (ex["indices"][:, None]*p+np.arange(p)).ravel()/fs
        keep = np.repeat(ex["selected"], p)
        ax.plot(np.arange(len(x))/fs, x[:, 0], color="black", lw=0.8, label="Observed target signal")
        for key in ["prediction"]+[k for k in ex if k.startswith("baseline_")]:
            y = ex[key][..., 0].ravel().copy()
            y[~keep] = np.nan
            ax.plot(t, y, lw=0.8, alpha=0.8, label="neural" if key == "prediction" else key.removeprefix("baseline_"))
        for index in ex["indices"][ex["selected"]]:
            ax.axvspan(index*p/fs, (index+1)*p/fs, color="gray", alpha=0.06, lw=0)
        ax.set(xlabel="Seconds", ylabel="Normalized Fp1", title=f"{ex['participant_id']} ({ex['group']})")
    axes[0,0].legend(ncol=4, fontsize=8)
    fig.suptitle("Held-out EEG examples; gray regions are scored targets")
    finish(fig, path/"reconstruction_examples.png")


def classification_score(y, prediction):
    return {"balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=CLASSES, average="macro", zero_division=0)),
            **dict(zip([f"recall_{c}" for c in CLASSES],
                       recall_score(y, prediction, labels=CLASSES, average=None, zero_division=0).tolist()))}


@torch.no_grad()
def participant_features(model, dataset, batch_size):
    model.eval()
    device = next(model.parameters()).device
    representations = []
    for batch in loader(dataset, batch_size):
        # Identical 0–4 s observation window for ALL neural and spectral classifiers.
        x = batch["x"][:, :dataset.x.shape[1]//2].to(device)
        representations.append(model.representations(x)["sequence"].cpu().numpy())
    return aggregate_participants(np.concatenate(representations), dataset.metadata)


def classify(features, cfg, path):
    path.mkdir(parents=True, exist_ok=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["classifier"]["C"],
              class_weight=cfg["classifier"]["class_weight"], solver="lbfgs", max_iter=5000,
              tol=1e-6, random_state=cfg["seed"]))
    model.fit(features["train"][1], features["train"][2])
    np.testing.assert_allclose(model[0].mean_, features["train"][1].mean(0, dtype=np.float64), atol=1e-10)
    scores, rows = {}, []
    for split in ("validation", "test"):
        ids, x, y = features[split]
        prediction = model.predict(x)
        scores[split] = classification_score(y, prediction)
        for pid, truth, pred in zip(ids, y, prediction):
            rows.append({"split": split, "participant_id": pid, "truth": truth, "prediction": pred})
        pd.DataFrame(confusion_matrix(y, prediction, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(path/f"{split}_confusion.csv")
    pd.DataFrame(rows).to_csv(path/"predictions.csv", index=False)
    joblib.dump(model, path/"classifier.joblib")
    np.savez_compressed(path/"participant_features.npz", **{
        f"{split}_{key}": value for split, values in features.items()
        for key,value in zip(("ids","x","y"), values)})
    save_json(path/"classification.json", scores)
    return scores


def build_datasets(cfg, output, smoke=False):
    manifest = json.loads((Path(cfg["cache_dir"])/"manifest.json").read_text())
    if manifest["specification"]["max_epochs"] != cfg["max_epochs_per_participant"]:
        raise ValueError("Cache epoch cap differs; rerun prepare with this config")
    people = [r for r in manifest["participants"] if not r["excluded"]]
    splits = subject_splits(people, cfg["seed"])
    if smoke:
        splits = {k: sum(([r for r in rows if r["group"] == group][:2] for group in CLASSES), []) for k,rows in splits.items()}
    sets = {k: {r["participant_id"] for r in rows} for k,rows in splits.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    datasets = {k: EEGDataset(rows, epoch_limit=2 if smoke else None) for k,rows in splits.items()}
    normalizer = fit_normalizer(datasets["train"])
    for dataset in datasets.values():
        dataset.normalizer = normalizer
    fs = {r["sampling_hz"] for rows in splits.values() for r in rows}
    if len(fs) != 1:
        raise ValueError("Pilot requires a common sampling rate; no implicit resampling")
    cfg["sampling_hz"] = fs.pop()
    np.savez(output/"normalization.npz", mean=normalizer.mean, std=normalizer.std)
    save_json(output/"participants.json", splits)
    save_json(output/"dataset.json", {"identity": manifest["identity"], "channels": manifest["channels"],
              "participants": {k: len(v) for k,v in splits.items()}, "epochs": {k: len(v) for k,v in datasets.items()},
              "groups": {k: pd.Series([r["group"] for r in rows]).value_counts().to_dict() for k,rows in splits.items()},
              "sampling_hz": cfg["sampling_hz"], "shape": list(datasets["train"].x.shape[1:]), "smoke": smoke})
    return datasets


def run_pilot(cfg, output, smoke=False, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = copy.deepcopy(cfg)
    if (output/"input_config.json").exists():
        if not resume or json.loads((output/"input_config.json").read_text()) != original:
            raise ValueError("Existing pilot requires --resume with identical config")
    save_json(output/"input_config.json", original)
    if smoke:
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 4
    datasets = build_datasets(cfg, output, smoke)
    shape = datasets["train"].x.shape[1:]
    if shape[0] % (2*cfg["model"]["patch_size"]):
        raise ValueError("Each half epoch must contain an integer number of patches")
    save_json(output/"config.json", cfg)
    save_json(output/"environment.json", {"packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "cuda_device": torch.cuda.get_device_name(0) if cfg["training"]["device"] == "cuda" else "cpu",
              "source_sha256": {str(p.relative_to(Path(__file__).parent)): __import__('hashlib').sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.rglob("*.py")}})
    summary = {"models": {}, "classification": {}}
    for name in ORDER:
        print(f"Starting EEG {name}", flush=True)
        path = output/name
        path.mkdir(exist_ok=True)
        seed_everything(cfg["training"]["seed"], cfg["training"]["threads"])
        model = make_eeg_model(cfg, name, shape).to(cfg["training"]["device"])
        if not name.startswith("masked_") or name == "masked_short":
            key = "untrained_causal" if name == "causal" else "untrained_masked"
            features = {k: participant_features(model, dataset, cfg["training"]["batch_size"]) for k,dataset in datasets.items()}
            summary["classification"][key] = classify(features, cfg, output/key)
        # Feature extraction above is deterministic but DataLoader iteration advances Torch RNG;
        # reset before training so all three masked models have identical training RNG streams.
        seed_everything(cfg["training"]["seed"], cfg["training"]["threads"])
        start = time.perf_counter()
        history = train(model, datasets, cfg, path, torch.device(cfg["training"]["device"]), resume)
        duration = time.perf_counter()-start
        model.requires_grad_(False)
        reconstruction = reconstruct(model, datasets["test"], cfg, path)
        features = {k: participant_features(model, dataset, cfg["training"]["batch_size"]) for k,dataset in datasets.items()}
        classification = classify(features, cfg, path)
        summary["classification"][name] = classification
        summary["models"][name] = {"reconstruction": reconstruction, "epochs": len(history),
                 "training_seconds": duration, "optimizer_updates": len(history)*int(np.ceil(len(datasets['train'])/cfg['training']['batch_size'])),
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "best_epoch": min(history, key=lambda r:r["validation_loss"])["epoch"],
                 "best_validation_mse": min(r["validation_loss"] for r in history)}
        save_json(path/"metrics.json", summary["models"][name])
        save_json(output/"summary.json", summary)
        del model
    spectral = {k: aggregate_participants(dataset.spectral, dataset.metadata) for k,dataset in datasets.items()}
    summary["classification"]["spectral"] = classify(spectral, cfg, output/"spectral")
    save_json(output/"summary.json", summary)
    from .eeg_report import write_report
    write_report(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "run", "report"])
    parser.add_argument("--config", default="configs/eeg_pilot.json")
    parser.add_argument("--output", default="results/eeg_pilot")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        prepare_cache(cfg["data_dir"], cfg["cache_dir"], cfg["max_epochs_per_participant"])
    elif args.command == "report":
        from .eeg_report import write_report
        write_report(Path(args.output))
    else:
        run_pilot(cfg, args.output, args.command == "smoke", args.resume)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
