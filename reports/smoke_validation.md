# Smoke validation — 2026-11-08

Executed locally on CPU. This checks pipeline execution and saved artifacts; the two-epoch budget is insufficient for scientific conclusions.

- `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q`: **19 passed**, including both end-to-end objectives and exact interrupted/uninterrupted training equality with dropout.
- `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m temporal_ssl sweep --config configs/smoke.json --output results/smoke`: all **eight** conditions completed. Resuming the completed sweep also succeeded.
- Standalone generation: `python -m temporal_ssl generate --config configs/smoke.json --output data/smoke` produced all three splits, labels, normalization, metadata and figures.
- Optional diagnostic: a masked run with `--set probes.nonlinear=true` completed in `results/nonlinear-smoke`.
- Audited all run configs, manifests, source fingerprints, checkpoints, histories, CSV row counts, finite NPZ predictions and plots. The combined tidy table contains every run. Both architectures allocate the same parameter count.

| Condition | Neural test MSE | Simple baseline MSE | Accumulator pooled test R² |
| --- | ---: | ---: | ---: |
| ar2-causal | 1.0382 | constant: 0.4564 | — |
| ar2-masked | 0.9972 | linear: 0.9174 | — |
| predictable-causal | 1.3234 | constant: 1.1611 | 0.9085 |
| predictable-long-horizon | 1.2034 | constant: 1.9749 | 0.9020 |
| predictable-long-span | 1.1829 | linear: 1.6776 | 0.9166 |
| predictable-masked | 1.1803 | linear: 1.1726 | 0.9159 |
| random-causal | 1.0660 | constant: 0.7609 | 0.7967 |
| random-masked | 1.0567 | linear: 0.8341 | 0.8098 |

MSE uses training-normalized units, averaged over held-out trajectories. Baselines and targets differ across objectives: causal persistence predicts future patches, while masked linear interpolation accesses both visible sides where available. The accumulator column uses full-sequence future context for masked representations and bounded past context for causal representations. It is descriptive, not a controlled comparison of equal information sets. AR(2) has no accumulator.

Detailed machine-readable results and figures are in `results/smoke/<condition>/`; comparison figures and `summary.csv` are in `results/smoke/analysis/`. Generated datasets and result directories are intentionally gitignored. Every run's `environment.json` captures installed versions and source SHA-256 fingerprints. See the README for equations, observability assumptions, reproduction commands, and limitations.
