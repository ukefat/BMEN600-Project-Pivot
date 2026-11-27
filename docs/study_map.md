# Final study and historical experiments

Files stay in their established locations to preserve imports, checkpoint provenance and reproduction commands. “Historical” denotes a role in development, not a failed experiment. The separate [original project](https://github.com/Pouya-Ta/BMEN-600-Course-Project) supplies historical context; this repository contains the final pivoted study.

| Role | Reports | Main code/configurations |
|---|---|---|
| Final synthetic evidence | [Synthesis](../reports/synthetic_summary.md), [results](../reports/research_results.md), [protocol](../reports/research_protocol.md) | `data.py`, `models.py`, `probes.py`, `study.py`; `research.json`, `matrix.json` |
| Final primary EEG comparison | [Extended three-class comparison](../reports/eeg_reconstruction_extended.md) | `eeg_factorial.py` → `eeg_span_study.py` → `eeg_reconstruction_study.py` → `eeg_reconstruction_extension.py`; corresponding configs |
| Supporting final representation analysis | [Pooling](../reports/eeg_pooling.md), [five-band window probes](../reports/eeg_window_spectral.md) | `eeg_pooling.py`, `eeg_window_spectral.py` |
| Final supplementary inference/QC | [Supplement](../reports/eeg_supplement.md) | `eeg_supplement.py`; committed predictions and QC counts |
| Final overview | [SVG](../reports/study_overview.svg), [caption](../reports/study_overview_assets/figure_caption.txt) | `reports/study_overview_assets/build_figure.py` |
| Historical prerequisites reused by final study | [Data coverage](../reports/eeg_factorial.md), [span study](../reports/eeg_span_study.md), [earlier central comparison](../reports/eeg_reconstruction_study.md) | Their full-data checkpoints are reused; capped-data contrasts are supporting development analyses |
| Earlier pilot and optimization diagnostics | `eeg_pilot`, `eeg_learning_checks`, `eeg_long_training`, `eeg_sweep_100`, `eeg_masked_300` reports | Corresponding modules/configs; pilot preprocessing/split/scaler definitions are still used |
| Alternative readouts/targets outside primary study | `eeg_binary`, broad `eeg_spectral_recoverability`, `eeg_latent*` reports | Binary, 152-target participant spectral and latent-prediction modules/configs |
| Execution checks | [Smoke validation](../reports/smoke_validation.md), `smoke.json`, sensitivity configs | Small budgets verify execution; not performance estimates |

Full source paths are under `temporal_ssl/`, configurations under `configs/`, reports under `reports/`; detailed original commands remain in [development history](development_history.md). No exploratory result has been silently substituted for a final primary result.

The final comparison controls architecture dimensions, EEG observations for diagnosis, participant assignments, training-update budgets and downstream readouts. Different forecasting/reconstruction targets, temporal geometry and information access remain. The primary findings concern these task formulations, not attention direction in isolation.
