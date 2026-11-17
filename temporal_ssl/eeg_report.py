"""Small shareable report using only completed EEG pilot artifacts."""
import json
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .plots import plt, finish


def write_report(root):
    root = Path(root).resolve()
    cfg = json.loads((root/"config.json").read_text())
    data = json.loads((root/"dataset.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    if set(summary["models"]) != {"causal", "masked_short", "masked_medium", "masked_long"} or "spectral" not in summary["classification"]:
        raise ValueError("Pilot is incomplete; report requires four runs and controls")
    plot_summary(root, summary)
    fs, p = data["sampling_hz"], cfg["model"]["patch_size"]
    patch_ms = 1000*p/fs
    lines = ["# EEG SSL pilot — ds004504", "", "## Research question", "",
             "How does the length of contiguous masked regions affect bidirectional self-supervised EEG representation learning, compared with a fixed causal future-prediction baseline?", "",
             "## Experimental setup", "",
             "This is an exploratory, inductive **single participant-disjoint split**, not cross-validation. Pretraining and classifier fitting use training participants only. Diagnostic labels are used for stratification and downstream classification, never SSL loss. One data/initialization seed is used; there is no hyperparameter search.", "",
             f"Dataset: OpenNeuro ds004504, version 1.0.9. All 88 participants have usable recordings (36 AD, 23 FTD, 29 HC). We reuse the course-project amplitude-preserving EEGLAB loader, canonical channel handling, metadata diagnosis mapping, boundary-aware epoch extraction, and Welch spectral features. No original course-project files were modified. {'**This report is a smoke test, not the full pilot.**' if data['smoke'] else ''}", "",
             "The supplied derivatives were already cleaned by their authors: histories include 0.5–45 Hz `pop_eegfiltnew`, ASR, ICA and ICLabel component removal. We preserve the supplied A1/A2 reference, filtering and artifact handling, with no additional filtering, resampling, rereferencing or per-epoch normalization. Source documentation calls the filter Butterworth, while histories use EEGLAB FIR; raw documentation/sidecars also disagree on reference/filter details. The verified derivative samples and headers are authoritative for this pilot. Offline provider filtering/cleaning may depend on future samples within a recording; model-level causality applies to the supplied derivative sequence and is not a strict online/raw-EEG causality claim.", "",
             "The complete 19-channel montage is Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz. Native microvolt amplitudes are retained in the cache. Non-overlapping 8-second epochs restart at every discontinuity identified from `clean_sample_mask`, checked against EEGLAB boundary events. Nonfinite/flat-channel epochs are rejected (flat peak-to-peak ≤1e-6 µV); >1,000 µV is a review flag only. No epoch crosses a provider-removed gap. Supplied offline preprocessing is independent per recording; we fit no preprocessing across held-out participants.", "",
             f"To keep the pilot small and participant contributions balanced, at most {cfg['max_epochs_per_participant']} usable epochs are selected per participant using evenly spaced indices across the recording. This is a documented subsample of the original course pipeline, not all available epochs. Exact selected starts and source checksums are cached. Channel mean/std are fit across selected **training** observations only and then fixed for every model and split. Spectral features are computed on physical microvolts before this SSL normalization.", "",
             "| Partition | Participants | AD / FTD / HC | Selected epochs |", "| --- | ---: | --- | ---: |"]
    for split in ("train", "validation", "test"):
        groups = data["groups"][split]
        lines.append(f"| {split} | {data['participants'][split]} | {groups.get('AD',0)} / {groups.get('FTD',0)} / {groups.get('HC',0)} | {data['epochs'][split]} |")
    lines += ["", f"Sampling: **{fs:g} Hz**, {data['shape'][1]} channels, {data['shape'][0]} samples per epoch. Shared patch: **{p} samples / {patch_ms:g} ms**; {data['shape'][0]//p} patches per 8 seconds.", "",
              "**Causal task:** only 0–4 seconds are observation tokens; predict every sample/channel in 4–8 seconds simultaneously. The same Transformer processes causal source tokens plus position-only future queries. Queries never become keys/values at any layer; future observation values are never embedded. The same linear patch head maps each query state to an observation patch. There is no autoregressive rollout or teacher forcing. This is an explicit extension of the synthetic individual-future-patch loss; the original synthetic objectives are unchanged.", "",
              "**Masked tasks:** 25% of whole patches are hidden, keeping the total count fixed while varying requested contiguous span. Masked values are replaced before embedding; only visible keys/values are accessible. Blocks can merge or truncate. Actual held-out run-length histograms, exact masks and summary counts are saved for each run.", "",
              "| Mask condition | Requested patches | Requested duration | Realized median / mean / max |", "| --- | ---: | ---: | --- |"]
    masks = {}
    for name in ("masked_short", "masked_medium", "masked_long"):
        masks[name] = json.loads((root/name/"mask_summary.json").read_text())
        m = masks[name]
        lines.append(f"| {name} | {m['requested_patches']} | {m['requested_ms']:g} ms | {m['realized_median_ms']:.0f} / {m['realized_mean_ms']:.0f} / {m['realized_max_ms']:.0f} ms |")
    parameters = next(iter(summary["models"].values()))["parameters"]
    updates = next(iter(summary["models"].values()))["optimizer_updates"]
    lines += ["", f"All four models: width {cfg['model']['d_model']}, {cfg['model']['layers']} Transformer layers, {cfg['model']['nhead']} heads, {parameters:,} allocated parameters each; AdamW lr={cfg['training']['lr']}, batch={cfg['training']['batch_size']}, **{cfg['training']['epochs']} epochs / {updates} updates**, seed={cfg['seed']}, device={cfg['training']['device']}. Same initialization and optimizer budget; only mask span changes among masked models. Each run selects its minimum validation reconstruction loss checkpoint. Full histories/checkpoints are retained.", "",
              "**Classification access is matched:** every neural and spectral method uses **0–4 seconds only** from each selected 8-second epoch. Causal attention uses past/current source patches; masked embeddings use bidirectional attention within those same four seconds, with no second-half observations at classification. This gives the same observed window but different within-window contextualization; shortening masked extraction from 8-second pretraining to 4-second evaluation is a distribution change. Mean-pool local embeddings per epoch, then mean-pool epochs per participant. The classifier is a training-only StandardScaler plus multinomial logistic regression (C=1, class_weight=None), following the course baseline's estimator/metrics. It is fitted only on training participants; validation scores are descriptive and select no classifier hyperparameters. Synthetic latent labels are absent and unused.", "",
              "Controls use identical initialized encoder weights/architecture/access/aggregation and the same classifier. The three masked initialization controls are identical because mask span is inactive during unmasked representation extraction; report one shared `untrained_masked` control. The copied spectral control uses the same first 4 seconds and selected epoch subset, with 2-second Hann Welch windows, 50% overlap, constant segment detrending, absolute log10 and relative delta/theta/alpha/beta power per channel (152 features). Compared with the original course spectral analysis, observation duration is 4 rather than 8 seconds to match neural classification access. Normalized reconstruction and physical-unit spectral features intentionally use distinct fixed preprocessing appropriate to their tasks.", "",
              "## Results", "", "### Held-out reconstruction", "",
              "Normalized MSE is averaged within an epoch, then within participant, then across held-out participants. Simple methods use exactly the neural target mask/region and available observations. Causal persistence/trend see only the first half; trend is fitted over the complete first half. Masked interpolation/nearest filling use visible samples only, with one-sided constant filling at edges. **Do not rank causal MSE against masked MSE as the same task.**", "",
              "| Configuration | Neural MSE | Persistence / interpolation | Trend / nearest | Training-mean MSE | Best validation epoch |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in summary["models"].items():
        r = result["reconstruction"]
        baseline, other = ("constant", "linear_trend") if name == "causal" else ("linear", "nearest")
        lines.append(f"| {name} | {r['neural']['mean']:.4f} | {r[baseline]['mean']:.4f} | {r[other]['mean']:.4f} | {r['training_mean']['mean']:.4f} | {result['best_epoch']} |")
    lines += ["", f"![Reconstruction comparison]({root/'reconstruction_comparison.png'})", "", "### Frozen participant classification", "",
              "Scores are from one held-out participant split, not cross-validated means. Chance balanced accuracy is 1/3. No epoch is counted as an independent participant.", "",
              "| Features | Validation balanced accuracy | Test balanced accuracy | Test macro-F1 | Recall AD | Recall FTD | Recall HC |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = []
    for name, scores in summary["classification"].items():
        v,t = scores["validation"],scores["test"]
        lines.append(f"| {name} | {v['balanced_accuracy']:.3f} | {t['balanced_accuracy']:.3f} | {t['macro_f1']:.3f} | {t['recall_AD']:.3f} | {t['recall_FTD']:.3f} | {t['recall_HC']:.3f} |")
        rows.append({"features": name, "validation_balanced_accuracy": v["balanced_accuracy"], **t})
    pd.DataFrame(rows).to_csv(root/"classification_results.csv", index=False)
    lines += ["", f"![Participant classification]({root/'classification_comparison.png'})", "",
              "Illustrative traces (one held-out example per diagnostic group; Fp1 only):", ""]
    for name in ("causal", "masked_short", "masked_medium", "masked_long"):
        lines.append(f"![{name}]({root/name/'reconstruction_examples.png'})")
    lines += ["", "## Interpretation and limitations", ""]
    short = summary["models"]["masked_short"]["reconstruction"]["linear"]["mean"]
    long = summary["models"]["masked_long"]["reconstruction"]["linear"]["mean"]
    short_ba = summary["classification"]["masked_short"]["test"]["balanced_accuracy"]
    long_ba = summary["classification"]["masked_long"]["test"]["balanced_accuracy"]
    lines += [f"Interpolation MSE changes from **{short:.4f}** for short masks to **{long:.4f}** for long masks. Over the same two conditions, frozen-classifier balanced accuracy changes from **{short_ba:.3f}** to **{long_ba:.3f}**. These are observed associations across this finite pilot; a low interpolation error establishes that interpolation is an available solution, not that a network actually uses it. A harder objective need not make better features, and lower reconstruction error need not improve dementia classification.", "",
              "One seed, a small participant cohort, one split, residual recording artifacts, possible demographic/acquisition confounding, capped epochs, a modest budget and validation checkpoint selection limit conclusions. Participant labels describe the dataset, not a clinically validated diagnostic tool. The classifier and mean pooling can miss information present in nonlinear or time-local representations. Masked and causal contextualization differ, despite the matched classification window. Four-second exact waveform prediction can be intrinsically difficult for resting EEG; a near-mean forecast may be competitive. The training-mean reference predicts zero in normalized units (each channel's fixed training mean), using the same scored targets; compare it before attributing prediction skill to a low neural MSE. No training loss or interpolation score alone proves shortcut learning or causal superiority. Inspect the untrained and spectral controls before attributing a classifier score to SSL.", "",
              "## Suggested next experiments", "",
              "1. Repeat this fixed protocol over several subject splits/seeds, retraining SSL within each split; report participant-level paired uncertainty rather than treating epochs as replicates.",
              "2. Diagnose masked reconstruction against the strong short-span interpolation reference before scaling training: inspect learning curves and test a declared patch/embedding compression change with the same budget, rather than assuming the current architecture learned interpolation.",
              "3. In the existing AR(2)/switching synthetic systems, match EEG-like channel counts, patch sizes and noise, then vary accumulator visibility and mask span. Check whether the 475-observation-value → 32-dimensional patch bottleneck explains failure to exploit interpolation before attributing it to EEG dynamics.",
              "4. Repeat frozen classification with a predeclared balanced-class-weight readout and participant splits that retain adequate FTD cases; compare matched 4-second versus 8-second observation windows for every method in a subsequent fixed ablation.", "",
              "## Reproduce and hand off", "",
              "Install the optional EEG dependencies into the existing CUDA environment:", "", "```bash", ".venv-gpu/bin/python -m pip install -e '.[eeg,test]'",
              "# Reuse local verified ds004504 data; edit data_dir/cache_dir in configs/eeg_pilot.json if moved.",
              ".venv-gpu/bin/python -m temporal_ssl.eeg_pilot prepare",
              "CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m temporal_ssl.eeg_pilot smoke --output results/eeg_smoke",
              "CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m temporal_ssl.eeg_pilot run --output results/eeg_pilot",
              "# Interrupted runs: repeat the same run command with --resume.",
              ".venv-gpu/bin/python -m temporal_ssl.eeg_pilot report --output results/eeg_pilot",
              "CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m pytest -q",
              "# Only if the source data is absent, download to a NEW directory, then update data_dir:",
              ".venv-gpu/bin/python -m temporal_ssl.eeg_course.download --data-dir data/ds004504 --no-raw-examples", "```", "",
              "Cached arrays, preprocessing identities and exact epoch starts are under `data/eeg_pilot`; dataset snapshot source hashes are verified during preparation. Participant assignments, train-only normalization, effective config, installed versions/source fingerprints, epoch/participant errors, masks/span histograms, participant features/predictions/confusion matrices, classifiers, training histories and last/best checkpoints are saved in the output directory. The training loop supports epoch-boundary resume; all four runs must finish before this report is generated. Cached data and outputs are gitignored. The original course code is copied under `temporal_ssl/eeg_course` with its MIT license; only its spectral import was adapted to a package-relative import.", ""]
    model_scores = {name: summary["classification"][name]["test"]["balanced_accuracy"] for name in summary["models"]}
    best_name = max(model_scores, key=model_scores.get)
    spectral_score = summary["classification"]["spectral"]["test"]["balanced_accuracy"]
    observations = ["### What this completed run shows", "",
                    f"The spectral control scores **{spectral_score:.1%}** balanced accuracy; the highest trained SSL score is **{model_scores[best_name]:.1%}** ({best_name}). This is a descriptive single-split result, not a selected best method for deployment.", ""]
    for name in summary["models"]:
        baseline = "untrained_causal" if name == "causal" else "untrained_masked"
        gain = model_scores[name]-summary["classification"][baseline]["test"]["balanced_accuracy"]
        reconstruction = summary["models"][name]["reconstruction"]
        relative = 1-reconstruction["neural"]["mean"]/reconstruction["training_mean"]["mean"]
        observations.append(f"- {name}: trained-minus-untrained balanced accuracy **{100*gain:+.1f} percentage points**; neural MSE reduction relative to the training-mean prediction **{100*relative:.2f}%** on its own scored targets.")
    observations += ["", "Thus comparisons against persistence alone can overstate useful exact-waveform prediction. On short masked spans, compare the neural value directly with interpolation before claiming the network exploits local continuity. Longer masks can make interpolation worse while still failing to improve classification.", "",
                     f"The test set has {data['groups']['test'].get('AD',0)} AD, {data['groups']['test'].get('FTD',0)} FTD and {data['groups']['test'].get('HC',0)} HC participants. One additional correct classification changes balanced accuracy by 1/(3 × class count), so the small differences among SSL methods should not be treated as reliable rankings. Per-class recalls in the table expose FTD performance rather than hiding it in overall accuracy.", ""]
    index = lines.index("## Suggested next experiments")
    lines[index:index] = observations
    report = "\n".join(lines)
    (root/"report.md").write_text(report.replace(str(root)+"/", ""))
    if not data["smoke"]:
        destination = Path(__file__).resolve().parents[1]/"reports/eeg_pilot.md"
        assets = destination.parent/"eeg_pilot_assets"
        for filename in ["reconstruction_comparison.png", "classification_comparison.png"] + [
                f"{name}/reconstruction_examples.png" for name in summary["models"]]:
            target = assets/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/filename, target)
        destination.write_text(report.replace(str(root)+"/", "eeg_pilot_assets/"))
    (root/"report_provenance.json").write_text(json.dumps({
        "report_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((root/"summary.json").read_bytes()).hexdigest(),
        "note": "Training source snapshot is retained separately in environment.json; report generation does not retrain models."
    }, indent=2)+"\n")
    return report


def plot_summary(root, summary):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax, names in zip(axes, (["causal"], ["masked_short","masked_medium","masked_long"])):
        methods = ["neural","constant","linear_trend","training_mean"] if names[0] == "causal" else ["neural","linear","nearest","training_mean"]
        peak = 0
        for j,method in enumerate(methods):
            values = [summary["models"][n]["reconstruction"][method]["mean"] for n in names]
            x = np.arange(len(names))+(j-1.5)*0.18
            ax.bar(x, values, width=0.17,label=method)
            peak = max(peak, max(values))
            for position,value in zip(x,values):
                ax.annotate(f"{value:.3f}",(position,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7)
        ax.set_xticks(range(len(names)),names)
        ax.set(ylabel="Participant-averaged normalized test MSE", title="Causal: 0–4 → 4–8 s" if names[0] == "causal" else "Masked: 25% hidden, varying span")
        ax.set_ylim(0, peak*1.35)
        ax.legend(ncol=2,fontsize=8)
    finish(fig, root/"reconstruction_comparison.png")
    names = list(summary["classification"])
    fig,ax = plt.subplots(figsize=(11,4))
    ax.bar(range(len(names)),[summary["classification"][n]["test"]["balanced_accuracy"] for n in names])
    for i,name in enumerate(names):
        value=summary["classification"][name]["test"]["balanced_accuracy"]
        ax.text(i,value+0.02,f"{100*value:.1f}%",ha="center",fontsize=9)
    ax.axhline(1/3,color="gray",ls=":",label="Chance balanced accuracy")
    ax.set_xticks(range(len(names)),names,rotation=25,ha="right")
    ax.set(ylabel="Held-out participant balanced accuracy",ylim=(0,1), title="Frozen features; matched first-four-second access; one subject split")
    ax.legend()
    finish(fig,root/"classification_comparison.png")
