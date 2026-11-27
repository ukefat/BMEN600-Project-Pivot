# Reproducing the final study

Run commands from the repository root. Use the quick pathway for grading the reported results. Full reproduction is documented for a fresh working copy; it is not required to regenerate figures and was not rerun during submission preparation.

## Environment

Recorded platform: Linux x86_64, Python **3.12.3**, RTX 3090 (24 GB), PyTorch **2.11.0+cu128**. [Observed package versions](environment_observed.json) agree with saved study environments. EEG tools require Python 3.11+ because the downloader uses `hashlib.file_digest`; Python 3.12 is the tested choice even though the base synthetic package metadata accepts 3.10+.

For CPU tables and figures only:

```bash
python3.12 -m venv .venv-report
source .venv-report/bin/activate
python -m pip install -r requirements-report.txt
```

The report requirements pin the observed NumPy/pandas/Matplotlib versions without introducing a new analysis dependency. Inkscape is an optional external executable for the overview SVG's PNG export, not a Python package. The already committed SVG/PNG remain viewable without it.

For the existing full analysis and tests, create a separate environment:

```bash
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-analysis.txt
python -m pip install -e '.[eeg,test]'
python -m pytest -q
```

The CUDA wheel resolves to the recorded `2.11.0+cu128` build. The non-Torch dependencies are pinned in [requirements-analysis.txt](../requirements-analysis.txt), with package constraints also declared in [pyproject.toml](../pyproject.toml). Both requirements files reflect the observed environment; they are not a complete lock of transitive packages. No historical dependency was upgraded during submission preparation. CPU tests can use an appropriate CPU PyTorch wheel instead; historical training used CUDA, so do not claim byte-identical GPU results from another wheel/platform. Joblib/scikit-learn fitted objects should be loaded with their recorded library versions.

## Quick CPU pathway without private/local artifacts

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reproduce_submission.py
# Optional actual-data overview export, requiring Inkscape:
OPENBLAS_NUM_THREADS=1 python scripts/reproduce_submission.py \
  --overview --output results/submission_preview_with_overview
```

Each command needs a previously nonexistent output directory. The script only reads committed inputs and writes new outputs; it never invokes training, fits a classifier, changes a split or overwrites the original figures. It independently rechecks numerical summaries, the 5,000-replicate paired bootstrap, FTD counts and QC totals. The original synthetic figure builder rechecks all 342 aggregate rows against individual seeds. Figure styles may differ from earlier report figures; numerical values must agree. `verification.json` records checks, input hashes and elapsed time.

This is **numerical/figure reproduction from saved evidence**, not independent model evaluation from waveforms. The predictions and scores remain those of the saved fitted models. A fresh clone contains the necessary CSVs, source scripts, participant assignments and small overview signal extract. No `data/`, checkpoints, GPU or PyTorch is needed. The timed workstation run, including optional overview export, took about 2.6 seconds; runtime varies.

## Map of paper outputs

No final manuscript file or definitive figure numbering is present, so outputs are mapped by content rather than invented paper numbers. All figures below are programmatic; manuscript placement, cropping and final numbering remain the authors' editorial work. Paths in the source/configuration columns are relative to `temporal_ssl/` and `configs/` unless specified.

| Output | Original source and configuration | Committed numerical evidence / final artifact | Quick reproduction output under `results/submission_preview/` |
|---|---|---|---|
| Study overview | `reports/study_overview_assets/build_figure.py`; actual archived signal extract, no model config | `reports/study_overview.svg`, PNG, caption and signal provenance | `study_overview.svg` and PNG with `--overview` |
| EEG cohort/dataset table | `eeg_data.py`, `eeg_pilot.build_datasets`; `eeg_pilot.json` | `reports/submission_assets/participant_split.csv`, supplement `qc_by_participant.csv` | `cohort.csv` |
| Synthetic latent recovery and tradeoff | `study.py`, `probes.py`; `research.json` + `matrix.json`; figure source `reports/synthetic_summary_assets/build_figures.py` | `reports/synthetic_summary_assets/{aggregate_metrics,seed_metrics}.csv` | `horizon_tradeoff.{png,pdf}`, `predictable_controls.{png,pdf}` |
| EEG waveform prediction | `eeg_reconstruction_extension.py`, causal/factorial/span helpers; four study configs | `reports/eeg_reconstruction_extended_assets/reconstruction{,_summary}.csv` and `reconstruction.png` | `waveform_summary.csv`, `waveform_prediction.{png,pdf}` |
| Participant-level diagnosis | Same central pipeline and fixed C=1 classifier in `eeg_pilot.py` | Central `classification{,_summary}.csv`, `classification.png` | `diagnostic_summary.csv`, `diagnostic_performance.{png,pdf}` |
| EEG spectral accessibility | `eeg_window_spectral.py`; saved seed-600 model config, fixed alpha grid selected on validation as originally specified | `reports/eeg_window_spectral_assets/{scores,alpha_selection,paired_bootstrap}.csv` | `spectral_scores.csv`, `spectral_accessibility.{png,pdf}` |
| Paired diagnostic intervals | `eeg_supplement.py`; fixed seed 5600, 5,000 participant resamples | `reports/eeg_supplement_assets/{participant_predictions,paired_bootstrap}.csv` | `paired_diagnostic_intervals.csv`, checked against saved bounds |
| FTD errors and matrices | `eeg_supplement.py`; original classification predictions | Supplement `ftd_counts_per_seed.csv`, `confusion_matrices_per_seed.csv`, `confusion_matrices.png` | `ftd_counts.csv`, `confusion_matrices.csv`; original matrix figure is committed |
| Window QC | `eeg_course/shared.py`, `eeg_data.py`; original QC settings | Supplement `qc_by_participant.csv`, `qc_totals.csv` | `qc_counts.csv`, checked against participant totals |

The quick command for **every row** is `python scripts/reproduce_submission.py`; add `--overview` only for its first row. The final [central report](../reports/eeg_reconstruction_extended.md), [spectral report](../reports/eeg_window_spectral.md) and [supplement](../reports/eeg_supplement.md) contain definitions and interpretation. Historical reports/assets stay unchanged.

## EEG access and preparation for full reproduction

The [versioned dataset](https://openneuro.org/datasets/ds004504/versions/1.0.9) is public and declares CC0. Respect its published acknowledgment instructions, including the [data descriptor](https://doi.org/10.3390/data8060095). The downloader pins version 1.0.9 to its resolved dataset Git commit (`63b2ec152bc3cd4487d131124f10f767ebe543bc` in the saved download manifest), validates Git-annex SHA256 identities and writes `download_manifest.json`.

```bash
python -m temporal_ssl.eeg_course.download \
  --data-dir data/ds004504 --version 1.0.9 --no-raw-examples
```

This retrieves all derivative `.set` recordings plus participant/channel/sidecar metadata; the final study does not require the optional raw examples. Expected root includes `download_manifest.json`, `participants.tsv`, `participants.json`, `dataset_description.json`, `derivatives/sub-*/eeg/*_eeg.set` and per-participant EEG sidecars/channel TSVs. A BIDS download without the generated checksum manifest is not sufficient for the existing preparation code: use the supplied downloader rather than fabricate the manifest.

Historical `configs/eeg_pilot*.json` and old provenance contain workstation-specific source locations. They are preserved to avoid invalidating run records. The new [portable template](../configs/submission/eeg_pilot.json) differs only in `data_dir`, which is `data/ds004504`; scientific parameters and relative cache paths are unchanged. Use it for a fresh reproduction, or copy it to a local file and change **only** the data location. Do not edit archived participants JSON or run protocol files to repair paths in place. On another computer, regenerate the cache/metadata using the same functions and validate the portable participant-ID split.

```bash
python scripts/prepare_eeg_metadata.py --config configs/submission/eeg_pilot.json
```

This invokes the original capped-cache preparation, participant split and training-only normalization, validates all participant assignments against the committed split, and fits only the original handcrafted spectral reference. It does **not** pretrain pilot models. It creates the prerequisite `results/eeg_pilot/{config,participants,dataset}.json`, normalization and spectral classifier/features/predictions. It refuses an existing pilot output directory. The first final training command automatically creates the full cache with the original `max_epochs=1000000` setting, preserving all 8,464 usable windows across 88 participants. No windows were rejected for nonfinite or flat-channel checks; provider-mask removals and boundary tails are separate.

## Full training and evaluation sequence

**Fresh working copy only.** These are the original expensive run entry points, not commands executed during submission preparation. They save into their established `results/` directories and some report functions regenerate `reports/` assets. Use a separate clone to protect the submitted evidence. Identical interrupted runs resume where supported; do not change settings to force a resume. Set the environment once:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
```

1. **Synthetic training and probes** (configured CUDA; CPU is supported by the underlying synthetic pipeline when explicitly configured, but is not the recorded study environment):

   ```bash
   python -m temporal_ssl.study --config configs/research.json \
     --matrix configs/matrix.json --output results/research
   ```

   Generates all 24 runs and `results/research/analysis/`. The original validation-only budget rule gives a common 100 epochs / 1,600 updates in the saved study. `--resume` resumes an identical study; `--report-only` uses existing run metrics/history without training.

2. **Base EEG masking runs** (CUDA required by these modules):

   ```bash
   python -m temporal_ssl.eeg_factorial --config configs/eeg_factorial.json
   ```

   Twelve runs include the historical capped/full × 25/50% crossing. Only the full-data models enter the final primary comparison; the capped runs remain prerequisites of this original orchestration. No need to train the original 20-epoch pilot.

3. **100/200 ms masking at 25/50%** (12 additional CUDA runs; reuses 50 ms):

   ```bash
   python -m temporal_ssl.eeg_span_study --config configs/eeg_span_study.json
   ```

4. **Causal 4→4 and 5.35→2.65 s** (six CUDA runs; collects prior full-data masked models):

   ```bash
   python -m temporal_ssl.eeg_reconstruction_study --config configs/eeg_reconstruction_study.json
   ```

5. **Causal 6→2 s and 75% masking** (12 additional CUDA runs):

   ```bash
   python -m temporal_ssl.eeg_reconstruction_extension --config configs/eeg_reconstruction_extension.json
   ```

   Consolidates 36 primary pretrained models, their initialization controls, spectral reference and matched update-budget audits under `results/eeg_reconstruction_extension/` and `reports/eeg_reconstruction_extended*`.

6. **Frozen evaluations** (no SSL training):

   ```bash
   python -m temporal_ssl.eeg_pooling
   python -m temporal_ssl.eeg_window_spectral --device cuda
   python -m temporal_ssl.eeg_supplement
   ```

   Pooling currently explicitly uses CUDA and creates `results/eeg_pooling/summary.csv`, which the supplementary audit checks. Window spectral probes also accept `--device cpu`, but require saved checkpoints and original waveform panels. Supplementary inference/QC is CPU-capable and needs the original derivative metadata, capped/full caches, saved classifiers/features/predictions, all initialization/best checkpoints and the pooling summary. It verifies checkpoints rather than training them. Saved sklearn objects must come from trusted study artifacts.

## Artifact availability, compute and limits

Committed: numerical tables, participant predictions used by the supplement, audits, exact participant-ID split, final report images/SVG, figure scripts and a small actual-signal extract. This supports the quick pathway from a clone.

Not committed: the full EEG download, capped/full waveform caches, fitted classifier joblib files, normalization NPZ, synthetic per-run metrics/history and full model/checkpoint trees. They exist in the study workspace, but no remote checkpoint archive is configured. Thus model-level re-evaluation on a fresh clone requires downloading/preparing data and rerunning the expensive full sequence, or obtaining an intact trusted artifact tree from the authors. Numerical reproduction alone cannot independently verify waveform preprocessing or re-extract embeddings. Portable QC tables allow checking accounting, while independently rechecking provider masks requires original derivatives.

The source download currently occupies about 2.9 GB locally, the two EEG caches about 3.8 GB combined, with additional space for environments and run artifacts. The full pipeline historically used an RTX 3090 with 24 GB GPU memory; host memory was not profiled, and complete end-to-end wall time was not recorded. Do not infer total runtime from quick figure generation. Each primary EEG model uses 10,000 updates × batch 64; the original full orchestration trains 42 models including six historical capped models, of which 36 enter the primary table. Synthetic runs use 24 × 1,600 updates in the completed study. These counts are clearer computational requirements than an invented timing estimate.

Statistical interpretation is unchanged: seed SD is not a participant confidence interval; fixed-cohort bootstrap intervals condition on fitted models; repeated test-set inspection is exploratory; different targets/access prevent isolating attention direction. The paper-output map is by content because the final manuscript and figure numbering still require author confirmation.
