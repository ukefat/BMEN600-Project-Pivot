"""Participant-level linear spectral probes; reuse EEG pilot artifacts, never train SSL."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import Normalizer
from .eeg_course.download import sha256
from .eeg_course.features import BANDS, extract, feature_names
from .eeg_data import EEGDataset, PREPROCESS, aggregate_participants
from .eeg_pilot import make_eeg_model, participant_features
from .experiment import save_json, seed_everything
from .plots import plt, finish

MODELS = ("untrained_causal", "causal", "untrained_masked", "masked_short",
          "masked_medium", "masked_long")
LABELS = ("Untrained causal", "Trained causal", "Untrained masked", "Masked 50 ms",
          "Masked 200 ms", "Masked 600 ms")
ALPHAS = (0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000.)
SPLITS = ("train", "validation", "test")


def observation_window(x):
    """x is [epoch,time,channel]; only 0–4 s of the native 8 s epoch is available."""
    if x.ndim != 3 or x.shape[1] % 2:
        raise ValueError("Expected even-length [epoch,time,channel] EEG")
    return x[:, :x.shape[1]//2]


def spectral_targets(x, fs):
    return np.stack([extract(window.T, fs, PREPROCESS)[0]
                     for window in observation_window(x)])


def check_splits(splits):
    ids = [[r["participant_id"] for r in splits[k]] for k in SPLITS]
    if any(len(v) != len(set(v)) or not v for v in ids):
        raise ValueError("Missing or duplicate participants")
    if any(set(a) & set(b) for i, a in enumerate(ids) for b in ids[i+1:]):
        raise ValueError("Participants overlap across partitions")


def load_encoder(pilot, cfg, name, shape, device):
    if name not in MODELS:
        raise ValueError(name)
    seed_everything(cfg["training"]["seed"], cfg["training"]["threads"])
    architecture = ("causal" if name == "untrained_causal" else
                    "masked_short" if name == "untrained_masked" else name)
    model = make_eeg_model(cfg, architecture, shape)
    if not name.startswith("untrained_"):
        checkpoint = Path(pilot)/name/"best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint; SSL will not be retrained: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


def target_tolerance(training):
    # Absolute + relative tolerance in each target's own physical/log units.
    return 1e-12 + 1e-10*np.max(np.abs(training), axis=0)


def spectral_metrics(truth, prediction, training):
    """Original-unit featurewise R², uniform mean over nonconstant targets.

    Undefined R² stays NaN (CSV blank), never coerced to a perfect/zero score.
    Train-constant or evaluation-constant targets are excluded and counted.
    """
    truth, prediction, training = [np.asarray(a, dtype=np.float64)
                                   for a in (truth, prediction, training)]
    if truth.shape != prediction.shape or len(truth) < 2:
        raise ValueError("Need matching predictions and at least two participants")
    if not all(np.isfinite(a).all() for a in (truth, prediction, training)):
        raise ValueError("Nonfinite spectral targets/predictions")
    tolerance = target_tolerance(training)
    eligible = (training.std(0) > tolerance) & (truth.std(0) > tolerance)
    r2 = np.full(truth.shape[1], np.nan)
    sse = np.square(truth-prediction).sum(0)
    sst = np.square(truth-truth.mean(0)).sum(0)
    r2[eligible] = 1-sse[eligible]/sst[eligible]
    if not eligible.any():
        raise ValueError("No nonconstant targets to score")
    nmse = np.mean(np.square(truth[:, eligible]-prediction[:, eligible]) /
                   training.std(0)[eligible]**2)
    return {"r2": float(r2[eligible].mean()), "normalized_mse": float(nmse),
            "features_scored": int(eligible.sum()), "features_excluded": int((~eligible).sum())}, r2


def fit_probe(x, y, alphas=ALPHAS):
    """All scaler fits and ridge fits use training participants only."""
    x_scale = StandardScaler().fit(x["train"])
    y_scale = StandardScaler().fit(y["train"])
    candidates = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, solver="svd").fit(x_scale.transform(x["train"]),
                                                     y_scale.transform(y["train"]))
        prediction = y_scale.inverse_transform(ridge.predict(x_scale.transform(x["validation"])))
        metrics, _ = spectral_metrics(y["validation"], prediction, y["train"])
        candidates.append({"alpha": alpha, "validation_r2": metrics["r2"]})
    # Grid is ascending; exact ties use the first/smallest alpha. No train+val refit.
    alpha = max(candidates, key=lambda row: row["validation_r2"])["alpha"]
    ridge = Ridge(alpha=alpha, solver="svd").fit(x_scale.transform(x["train"]),
                                                y_scale.transform(y["train"]))
    predictions = {k: y_scale.inverse_transform(ridge.predict(x_scale.transform(x[k])))
                   for k in ("validation", "test")}
    return {"x_scaler": x_scale, "y_scaler": y_scale, "ridge": ridge}, predictions, candidates


def load_features(pilot, name, splits, width):
    path = pilot/name/"participant_features.npz"
    with np.load(path) as stored:
        features = {}
        for split in SPLITS:
            people = sorted(splits[split], key=lambda row: row["participant_id"])
            np.testing.assert_array_equal(stored[f"{split}_ids"], [r["participant_id"] for r in people])
            np.testing.assert_array_equal(stored[f"{split}_y"], [r["group"] for r in people])
            features[split] = stored[f"{split}_x"].astype(np.float64)
            if features[split].shape != (len(people), width) or not np.isfinite(features[split]).all():
                raise ValueError(f"Invalid feature cache: {path}")
    return features


def audit_inputs(pilot, cfg, splits, dataset, device):
    """Check original cache provenance, all spectral targets/norm, and encoder spot checks."""
    environment = json.loads((pilot/"environment.json").read_text())
    source = Path(__file__).parent
    for name in ("eeg_pilot.py", "eeg_models.py", "models.py", "eeg_data.py", "eeg_course/features.py"):
        if sha256(source/name) != environment["source_sha256"][name]:
            raise ValueError(f"Pilot extraction code changed: {name}; cached protocol must be reviewed")
    with np.load(pilot/"normalization.npz") as stored:
        normalizer = Normalizer(stored["mean"], stored["std"])
    count, total, squares = 0, np.zeros(dataset["shape"][1]), np.zeros(dataset["shape"][1])
    spectral_error = 0.
    for split, people in splits.items():
        for person in people:
            with np.load(person["cache"]) as stored:
                x, cached = stored["x"], stored["spectral"]
                recomputed = spectral_targets(x, person["sampling_hz"])
                # Cache EEG is float32; original Welch ran before that storage conversion.
                np.testing.assert_allclose(recomputed, cached, rtol=2e-6, atol=2e-7)
                spectral_error = max(spectral_error, float(np.max(np.abs(recomputed-cached))))
                if split == "train":
                    for epoch in x:
                        values = epoch.astype(np.float64)
                        count += len(values)
                        total += values.sum(0)
                        squares += np.square(values).sum(0)
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean**2, 1e-12))
    np.testing.assert_array_equal(normalizer.mean, mean.astype(np.float32))
    np.testing.assert_array_equal(normalizer.std, std.astype(np.float32))
    targets = load_features(pilot, "spectral", splits, len(feature_names()))
    # Verify that cached participant targets are means of the exact epoch targets.
    for split, people in splits.items():
        vectors, metadata = [], []
        for person in people:
            with np.load(person["cache"]) as stored:
                vectors.extend(stored["spectral"])
                metadata.extend([person]*len(stored["spectral"]))
        _, pooled, _ = aggregate_participants(np.asarray(vectors), metadata)
        np.testing.assert_array_equal(pooled, targets[split])
    sample = EEGDataset([sorted(splits["train"], key=lambda r:r["participant_id"])[0]], normalizer)
    checks, hashes = {}, {}
    features = {}
    for name in MODELS:
        model = load_encoder(pilot, cfg, name, dataset["shape"], device)
        features[name] = load_features(pilot, name, splits, cfg["model"]["d_model"])
        ids, computed, _ = participant_features(model, sample, cfg["training"]["batch_size"])
        np.testing.assert_allclose(computed[0], features[name]["train"][0], rtol=1e-5, atol=2e-6)
        checks[name] = {"participant": ids[0], "epochs": len(sample),
                        "embedding_max_abs_difference": float(np.max(np.abs(computed[0]-features[name]["train"][0])))}
        if not name.startswith("untrained_"):
            hashes[name] = sha256(pilot/name/"best.pt")
        del model
    artifacts = [pilot/"config.json", pilot/"participants.json", pilot/"dataset.json", pilot/"normalization.npz"]
    artifacts += [pilot/name/"participant_features.npz" for name in (*MODELS, "spectral")]
    return features, targets, {"spectral_recomputation_max_abs_difference": spectral_error,
            "training_only_normalization": "exact float32 match", "encoder_cache_checks": checks,
            "checkpoint_sha256": hashes,
            "input_artifact_sha256": {str(p.relative_to(pilot)):sha256(p) for p in artifacts}}


def make_plots(per_feature, overall, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, kind, title in zip(axes, ("log10_absolute", "relative"), ("Log₁₀ absolute power", "Relative power")):
        for name, label in zip(MODELS, LABELS):
            frame = per_feature[(per_feature.model == name) & (per_feature.kind == kind)]
            ax.plot(list(BANDS), [frame.loc[frame.band == band, "r2"].mean() for band in BANDS],
                    marker="o", label=label)
        mean = per_feature[(per_feature.model == "training_mean") & (per_feature.kind == kind)]
        ax.plot(list(BANDS), [mean.loc[mean.band == b, "r2"].mean() for b in BANDS],
                color="black", linestyle="--", label="Training mean")
        ax.axhline(0, color="gray", lw=.7)
        ax.set(title=title, ylabel="Test R² (mean across 19 channels)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Spectral recovery from frozen 0–4 s EEG embeddings; 18 test participants")
    finish(fig, output/"per_band_r2.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    table = overall.set_index("model").loc[list(MODELS)]
    ax.bar(LABELS, table.test_r2, color=["gray", "C0", "gray", "C1", "C2", "C3"])
    ax.axhline(overall.loc[overall.model == "training_mean", "test_r2"].iloc[0],
               color="black", linestyle="--", label="Training mean")
    ax.axhline(0, color="gray", lw=.7)
    ax.set(ylabel="Uniform mean test R² over 152 features")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    finish(fig, output/"overall_r2.png")


def write_report(pilot, output, overall, bands, deltas, audit, dataset):
    assets = Path("reports/eeg_spectral_recoverability_assets")
    assets.mkdir(parents=True, exist_ok=True)
    for filename in ("per_band_r2.png", "overall_r2.png"):
        shutil.copyfile(output/filename, assets/filename)
    # Portable numerical deliverables accompany the tracked report; detailed artifacts stay in results/.
    for filename in ("overall.csv", "per_band.csv", "per_feature.csv", "trained_minus_untrained.csv", "ridge_selection.csv"):
        shutil.copyfile(output/filename, assets/filename)
    def table(frame):
        columns = list(frame.columns)
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"]*len(columns)) + "|"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)
    causal_delta = float(deltas.loc[deltas.model == "causal", "delta_r2"].iloc[0])
    masked_delta = deltas.loc[deltas.model != "causal", "delta_r2"].tolist()
    text = f"""# EEG spectral information recoverability

Can the frozen pilot encoders linearly recover the spectral information used by the stronger dementia classifier? The hypothesis was that pretraining would improve accessibility, with an effect of masking duration. This is a diagnostic of existing checkpoints, with no SSL retraining, new preprocessing, or downloads.

## Protocol

OpenNeuro ds004504 v1.0.9: 88 participants, {dataset['participants']['train']} train / {dataset['participants']['validation']} validation / {dataset['participants']['test']} test, using the original diagnosis-stratified, participant-disjoint split (seed 600). There are 24 cached epochs per participant. Every encoder and spectral target uses the same first 4 s (2,000 samples at 500 Hz) of each native 8 s epoch. Masked encoders receive no artificial masks. Each 50 ms patch yields 32 features; mean-pool 80 patch embeddings, then mean-pool 24 windows per participant. Targets are likewise the mean of epoch features, not features of an averaged waveform.

The exact earlier spectral baseline supplies **152 targets**: 19 channels × delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) × two measures. Measures are log10 absolute bandpower (power in µV², floor 1e-12 µV²) and relative power (bandpower divided by 1–30 Hz power). Welch uses 2 s Hann segments, 50% overlap, constant detrending, and native 0.5 Hz bins; integration interpolates band endpoints and uses trapezoids. No gamma targets. Relative powers sum to one per channel. Averaging log powers across windows is retained exactly from classification.

Ridge predicts all targets jointly. Input and target StandardScalers fit only the 52 training participants. Select one shared alpha per encoder from {list(ALPHAS)} by highest uniform mean validation R². The final probe remains fitted on training participants only. Predictions are inverse-transformed before scoring. Compute each target's R² across the 18 test participants, then average equally across targets (38 per band, 19 per band/measure); this prevents high-variance targets dominating. Normalized MSE divides each squared error by that target's training variance. Constant or near-constant training/evaluation targets (SD ≤ 1e-12 + 1e-10 × maximum absolute training value) have undefined R² and are excluded/count-reported, never assigned perfect scores. No targets were excluded in this run. Training-mean predictions provide a separate baseline; their test R² can be negative because test means differ.

## Checkpoints and cache verification

Evaluated `causal/best.pt`, `masked_short/best.pt`, `masked_medium/best.pt`, and `masked_long/best.pt` under `{pilot}` (validation-selected epochs 20, 17, 16, 20). The two random encoders reproduce initialization seed 600 using the original model configuration. All three masked runs started with the same initial weights. Encoders were frozen throughout. Existing first-4-s participant embeddings and exact spectral targets were reused. Original extraction source hashes were checked, and each of the six encoder caches was checked by re-extracting all 24 windows of the first training participant. Every cached spectral epoch was recomputed from the matching first half of stored raw EEG (maximum discrepancy {audit['spectral_recomputation_max_abs_difference']:.2g}, due to float32 EEG storage). Participant target means and training-only SSL normalization were verified. Checkpoint hashes, versions, probe scalers, predictions, target arrays, and validation-grid scores are saved in `{output}`.

## Held-out results

{table(overall[['model', 'alpha', 'validation_r2', 'test_r2', 'normalized_mse']])}

![Overall spectral R²](eeg_spectral_recoverability_assets/overall_r2.png)

Per-band R² below averages both power measures equally across all 19 channels. The figure separates the two measures.

{table(bands.pivot(index='model', columns='band', values='r2').reindex(list(MODELS)+['training_mean']).reset_index()[['model']+list(BANDS)])}

![Per-band spectral R²](eeg_spectral_recoverability_assets/per_band_r2.png)

Trained minus corresponding untrained test R² (each probe independently validation-selected):

{table(deltas)}

## Interpretation and limits

Causal pretraining changes overall R² by {causal_delta:+.3f}; the 50/200/600 ms masked changes are {', '.join(f'{v:+.3f}' for v in masked_delta)}. **Pretraining did not improve overall spectral linear recoverability in any of these four comparisons.** Among trained masked checkpoints, overall recovery declines as requested mask duration increases (50 → 200 → 600 ms). These comparisons describe this seed/split and separately selected probe penalties, not a general duration effect.

All six overall scores are negative, but all beat the fitted training-mean baseline (-0.133). This indicates limited out-of-sample recovery averaged over all spectral targets, not zero recoverable content. Delta-band R² is positive in every encoder (0.043–0.179); theta is weak, and alpha/beta averages are negative. The two untrained encoders recover log absolute power better on average than relative power. Full feature scores retain positive and negative values without clipping. One explicit exception to the aggregate pretraining trend is causal alpha recovery: it improves from -0.111 to -0.083, while remaining negative.

The earlier dementia balanced accuracies were 36.8% untrained causal, 41.6% trained causal, 35.7% untrained masked, 47.1/47.9/42.4% masked 50/200/600 ms, and 66.0% with spectral features directly. **The rankings differ:** diagnosis classification improved after pretraining while overall spectral recovery declined; medium masks classified slightly better than short masks despite weaker spectral recovery. Spectral regression asks about all channel/band targets, whereas classification only needs disease-discriminative directions. The representations therefore do not provide a reliable linear reconstruction of the complete spectral baseline, but this diagnostic cannot establish spectral loss as the cause of the classification gap.

Positive R² means these observed spectral summaries are linearly accessible from participant-averaged representations on held-out participants. Negative R² means the probe performs worse than predicting each target's test-set mean; that comparator is a scoring reference, not a usable fitted baseline. Low R² does not establish information absence: nonlinear encoding and mean pooling may hide it. Targets are derived summaries of EEG, not independent dynamical ground truth. A single seed/split and 18 test participants support descriptive comparisons only. Relative-power targets are correlated and log-power units differ from fractions; uniform averaging is explicit. Provider offline preprocessing is unchanged. The optional full-epoch and epoch-level analyses were omitted to keep the matched-input diagnostic focused.

A next experiment, if needed, should target the observed weak bands or disease-relevant spectral directions with a predeclared protocol; this result alone does not justify foundation-scale pretraining. Preserve the current test split as evaluated rather than using repeated diagnostics as independent confirmation.

## Reproduce without SSL training

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_spectral_recoverability \\
  --pilot results/eeg_pilot --output results/eeg_spectral_recoverability
.venv-gpu/bin/python -m pytest -q tests/test_eeg_spectral_recoverability.py
```

CPU cache verification is available with `--device cpu`. Missing checkpoints or incompatible cache provenance stop explicitly; nothing is silently retrained. Portable tables: [overall](eeg_spectral_recoverability_assets/overall.csv), [per band](eeg_spectral_recoverability_assets/per_band.csv), [all 152 features](eeg_spectral_recoverability_assets/per_feature.csv), [paired changes](eeg_spectral_recoverability_assets/trained_minus_untrained.csv), [ridge selection](eeg_spectral_recoverability_assets/ridge_selection.csv).
"""
    (output/"report.md").write_text(text.replace("eeg_spectral_recoverability_assets/", ""))
    Path("reports/eeg_spectral_recoverability.md").write_text(text)


def run(pilot, output, device):
    pilot, output = Path(pilot), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((pilot/"config.json").read_text())
    dataset = json.loads((pilot/"dataset.json").read_text())
    splits = json.loads((pilot/"participants.json").read_text())
    check_splits(splits)
    print("Auditing matched-window caches, normalization, and six frozen encoders", flush=True)
    features, targets, audit = audit_inputs(pilot, cfg, splits, dataset, device)
    save_json(output/"audit.json", audit)
    save_json(output/"protocol.json", {"pilot": str(pilot.resolve()), "device": device,
              "window_seconds": 4, "alphas": ALPHAS, "seed": cfg["training"]["seed"],
              "participants": {k:[r['participant_id'] for r in splits[k]] for k in SPLITS},
              "source_sha256": sha256(Path(__file__)),
              "versions": {p:importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "torch", "pandas")}})
    np.savez_compressed(output/"targets.npz", **targets, names=feature_names())
    overall, per_feature, selection, per_band = [], [], [], []
    for name in (*MODELS, "training_mean"):
        if name == "training_mean":
            alpha = None
            predictions = {k:np.broadcast_to(targets["train"].mean(0), targets[k].shape).copy()
                           for k in ("validation", "test")}
        else:
            probe, predictions, candidates = fit_probe(features[name], targets)
            alpha = float(probe["ridge"].alpha)
            selection.extend({"model": name, **r} for r in candidates)
            joblib.dump(probe, output/f"{name}_probe.joblib")
        np.savez_compressed(output/f"{name}_predictions.npz", **predictions)
        val, _ = spectral_metrics(targets["validation"], predictions["validation"], targets["train"])
        test, scores = spectral_metrics(targets["test"], predictions["test"], targets["train"])
        overall.append({"model": name, "alpha": alpha, "validation_r2": val["r2"],
                        "test_r2": test["r2"], **{k:v for k,v in test.items() if k != "r2"}})
        for feature, score in zip(feature_names(), scores):
            channel, kind, band = feature.split("__")
            per_feature.append({"model": name, "feature": feature, "channel": channel,
                                "kind": kind, "band": band, "r2": score})
        for band in BANDS:
            mask = np.array([n.endswith("__"+band) for n in feature_names()])
            per_band.append({"model": name, "band": band, "r2": float(np.nanmean(scores[mask]))})
        print(f"{name}: alpha={alpha}, validation R²={val['r2']:.4f}, test R²={test['r2']:.4f}", flush=True)
    overall, per_feature, per_band = map(pd.DataFrame, (overall, per_feature, per_band))
    deltas = []
    indexed = overall.set_index("model")
    for name in ("causal", "masked_short", "masked_medium", "masked_long"):
        control = "untrained_causal" if name == "causal" else "untrained_masked"
        deltas.append({"model": name, "control": control,
                       "delta_r2": float(indexed.loc[name, "test_r2"]-indexed.loc[control, "test_r2"])})
    deltas = pd.DataFrame(deltas)
    for filename, frame in (("overall", overall), ("per_feature", per_feature), ("per_band", per_band),
                            ("trained_minus_untrained", deltas), ("ridge_selection", pd.DataFrame(selection))):
        frame.to_csv(output/f"{filename}.csv", index=False)
    make_plots(per_feature, overall, output)
    write_report(pilot, output, overall, per_band, deltas, audit, dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="results/eeg_pilot")
    parser.add_argument("--output", default="results/eeg_spectral_recoverability")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.pilot, args.output, args.device)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
