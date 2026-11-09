# Larger paired-seed study

## Execution and findings — 2026-11-08

All **24 runs completed on the RTX 3090**, with 512/128/128 independent train/validation/test trajectories per seed and **100 epochs / 1,600 optimizer updates per model**. Models each allocate 26,316 parameters. The common budget was extended from 50 epochs because 17 of the 24 validation pilots met the recorded 2% improvement criterion. Test evaluation followed completion of that common training budget. **23 tests passed**, including CUDA resume, control access, and validation-only study scheduling.

Artifact audit passed for every run: 100-epoch checkpoints/history, configs/budget archives, source hashes, 768 unique split identities, 128 test trajectories, 384 reconstruction CSV rows, finite predictions, controls and plots. Observation-only control scores match exactly between paired objectives, as expected from their identical datasets and access rules.

The larger study changes the interpretation of the smoke results:

- **Reconstruction now beats the corresponding simple baseline in all eight conditions.** Neural causal AR(2) MSE is 0.099 versus persistence 0.496; masked AR(2) is 0.419 versus interpolation 0.932. Each objective uses its own eligible targets and visible information, so their MSEs should not be directly ranked as equal tasks.
- **Several trajectory parameters are now recoverable.** AR frequency R² is about 0.83 for both models. Predictable-system delta R² is 0.666 causal and 0.623 masked; threshold R² is 0.901 and 0.841. The smoke runs' broadly negative parameter R² did not establish a model limitation.
- **Strong local-state scores were partly present before training.** In the predictable system, the default causal model increases accumulator/phase R² by only 0.023/0.017 relative to its own initialization; masked training decreases these scores by 0.077/0.052. Regime balanced accuracy increases by 0.028 causal and 0.102 masked. This is a tradeoff among linear readouts, not evidence that all latent information was gained or lost.
- **Longer causal prediction changes what becomes linearly accessible.** Predictable delta R² rises from 0.666 at horizon 1 to 0.804 at horizon 4, and regime accuracy rises from 0.623 to 0.677. Local phase R² falls from 0.990 to 0.576. Longer horizons do not uniformly improve recovery of every factor.
- **Simple observation statistics remain strong controls.** On the predictable system, raw-statistics delta R² is 0.948, omega0 R² 0.760 and omega1 R² 0.937, exceeding the default trained encoder readouts. AR frequency R² from raw statistics is 0.956. Encoders are not extracting all linearly available information in those engineered statistics. Their sequence pooling and training objectives may contribute; this experiment does not isolate those effects.
- **The comparison is not fully converged.** Ten runs still improve their best validation loss by at least 2% between epochs 81–90 and 91–100; all ten are masked conditions. Default causal validation often plateaus or overfits earlier. Further masked improvements are plausible. The study evaluates a common finite training budget and cannot establish asymptotic superiority or shortcut learning.

All numerical findings are means across three paired simulation/initialization seeds. The tables below report sample SD. These are not confidence intervals. Masked local probes retain future access; causal local probes use bounded past context. Smoke-versus-study comparisons also change sequence length, accumulator range and PyTorch/device, so within-study trained/untrained controls are the clearer evidence for learning.

Artifacts and reproduction:

- Full generated report: `results/research/report.md`.
- Numerical tables: `results/research/analysis/aggregate_metrics.csv` and `seed_metrics.csv`.
- Curves: `results/research/analysis/validation_curves.png` and `learning_curves.csv`.
- Controls by factor: `results/research/analysis/{ar2,predictable,random}_recoverability_controls.png`.
- Run plan and stopping decision: `results/research/study_plan.json` and `budget_decision.json`.
- Protocol: `reports/research_protocol.md`; configuration: `configs/research.json`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \
  .venv-gpu/bin/python -m temporal_ssl.study --output results/research --resume
```

The original smoke artifacts and CPU environment are preserved. Results directories and environments remain gitignored; this report and the protocol are repository files.

512 training / 128 validation / 128 test independent trajectories per seed; 128 samples, 3 channels. Paired simulation/initialization seeds: [(7, 11), (19, 23), (31, 37)]. Same architecture, AdamW, batch size 32 and update budget for both objectives.

All 24 runs completed 100 epochs. The 50-epoch pilot budget decision used validation only; test evaluation followed completion of the final common training budget. Values below are mean ± sample SD across 3 seed pairs, not confidence intervals.

## Reconstruction

| Condition | Neural MSE | Persistence / interpolation MSE |
| --- | ---: | ---: |
| ar2-causal | 0.099 ± 0.006 | 0.496 ± 0.005 |
| ar2-masked | 0.419 ± 0.062 | 0.932 ± 0.052 |
| predictable-causal | 0.127 ± 0.000 | 0.785 ± 0.016 |
| predictable-long-horizon | 0.275 ± 0.008 | 2.390 ± 0.027 |
| predictable-long-span | 0.337 ± 0.017 | 1.524 ± 0.006 |
| predictable-masked | 0.244 ± 0.008 | 1.133 ± 0.002 |
| random-causal | 0.089 ± 0.010 | 0.511 ± 0.004 |
| random-masked | 0.251 ± 0.017 | 0.818 ± 0.017 |

## Local state recovery

| Condition | Accumulator R² | Phase R² | Regime balanced accuracy | AR state R² |
| --- | ---: | ---: | ---: | ---: |
| ar2-causal | — | — | — | 0.993 ± 0.001 |
| ar2-masked | — | — | — | 0.962 ± 0.013 |
| predictable-causal | 0.963 ± 0.003 | 0.990 ± 0.000 | 0.623 ± 0.031 | — |
| predictable-long-horizon | 0.650 ± 0.024 | 0.576 ± 0.013 | 0.677 ± 0.026 | — |
| predictable-long-span | 0.869 ± 0.030 | 0.890 ± 0.035 | 0.713 ± 0.013 | — |
| predictable-masked | 0.875 ± 0.063 | 0.930 ± 0.022 | 0.689 ± 0.034 | — |
| random-causal | 0.991 ± 0.001 | 0.996 ± 0.000 | 0.722 ± 0.030 | — |
| random-masked | 0.891 ± 0.027 | 0.927 ± 0.011 | 0.729 ± 0.014 | — |

## Change from the same encoder before training

| Condition | Δ accumulator R² | Δ phase R² | Δ regime balanced accuracy | Δ AR state R² |
| --- | ---: | ---: | ---: | ---: |
| ar2-causal | — | — | — | 0.020 ± 0.006 |
| ar2-masked | — | — | — | -0.016 ± 0.016 |
| predictable-causal | 0.023 ± 0.001 | 0.017 ± 0.002 | 0.028 ± 0.015 | — |
| predictable-long-horizon | -0.291 ± 0.021 | -0.398 ± 0.011 | 0.082 ± 0.016 | — |
| predictable-long-span | -0.083 ± 0.031 | -0.092 ± 0.034 | 0.126 ± 0.009 | — |
| predictable-masked | -0.077 ± 0.064 | -0.052 ± 0.021 | 0.102 ± 0.019 | — |
| random-causal | 0.040 ± 0.010 | 0.022 ± 0.004 | 0.135 ± 0.023 | — |
| random-masked | -0.071 ± 0.026 | -0.054 ± 0.010 | 0.131 ± 0.015 | — |

## Trajectory parameters

| Condition | r R² | ω R² | δ R² | τ R² | ω₀ R² | ω₁ R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar2-causal | 0.423 ± 0.028 | 0.825 ± 0.032 | — | — | — | — |
| ar2-masked | 0.280 ± 0.036 | 0.829 ± 0.034 | — | — | — | — |
| predictable-causal | — | — | 0.666 ± 0.036 | 0.901 ± 0.014 | 0.214 ± 0.067 | 0.317 ± 0.040 |
| predictable-long-horizon | — | — | 0.804 ± 0.016 | 0.759 ± 0.040 | 0.308 ± 0.015 | 0.474 ± 0.088 |
| predictable-long-span | — | — | 0.586 ± 0.135 | 0.884 ± 0.031 | 0.286 ± 0.091 | 0.473 ± 0.151 |
| predictable-masked | — | — | 0.623 ± 0.085 | 0.841 ± 0.049 | 0.218 ± 0.146 | 0.443 ± 0.018 |
| random-causal | — | — | 0.141 ± 0.129 | 0.015 ± 0.005 | 0.145 ± 0.139 | 0.369 ± 0.133 |
| random-masked | — | — | 0.014 ± 0.013 | 0.001 ± 0.023 | 0.123 ± 0.042 | 0.326 ± 0.142 |

## Validation and interpretation

10 of 24 runs still improved their best validation loss by at least 2% comparing the final two 10-epoch windows. This descriptive stopping diagnostic does not prove convergence. Full trends and best epochs are in `analysis/final_validation_trends.json`.

Untrained controls use the identical initial architecture/weights and the same input access as their trained counterpart. Raw-patch features use the current flattened patch locally and patch means globally. Raw-statistics features add past-window mean/std and centered per-channel lag correlations (lags 1,2,4,8) locally, using at most the causal context; sequence features use full-sequence mean/std/lag correlations. They use observations only, no latent labels. Local raw controls have no future access; masked encoder locals do. All feature scaling and probe fitting use training data, and regularization uses validation data.

Masked locals see the full unmasked test sequence, including future observations. Causal locals see bounded past/current patches. Sequence embeddings mean-pool local states. Reconstruction targets and information sets differ across objectives. Pooled R² scores are descriptive; per-trajectory results remain in each run's metrics. Simulation and initialization seeds vary jointly, so this study does not isolate their individual sources of variability. No test score selected the training budget. The study is a stronger pilot, not proof of shortcut learning or general superiority.

See `analysis/seed_metrics.csv`, `aggregate_metrics.csv`, `learning_curves.csv` and per-system `*_recoverability_controls.png` figures. Each latent has its own plot axis to keep positive local-state scores visible alongside negative parameter scores.
