# Larger experiment protocol

This experiment follows the smoke pipeline check with enough data and updates to assess learning curves and compare frozen-representation probes against initialization and observation controls.

- Three data/model seed pairs: `(7,11)`, `(19,23)`, `(31,37)`. The objectives share trajectories and initialization within each pair. Seeds vary jointly; this does not isolate simulation versus initialization variance.
- Eight conditions per pair: AR(2), predictable switching and random switching, each causal and masked, plus predictable causal horizon 4 and masked span 4. Total: **24 runs**.
- **512 training, 128 validation, 128 test trajectories**, each 128 samples and three channels. Normalization is fit separately on each seed/system's training trajectories only. Generators and parameter ranges are in `configs/research.json`.
- Shared 32-dimensional, two-layer Transformer configuration, AdamW at 0.001, batch size 32. This gives **16 optimizer updates per epoch**, versus three per epoch in the smoke runs.
- Pilot: **50 epochs / 800 updates** for every run. Compare the minimum validation loss in epochs 31–40 against epochs 41–50. If any run improves by at least 2%, extend **every run** to **100 epochs / 1,600 updates**, preserving optimizer/RNG state. Choose the best checkpoint separately by validation objective within that common budget. This rule is a budget heuristic, not a convergence test or a matched compute budget.
- Test metrics are computed only after the final common training budget is complete. Test results never determine the extension decision. Full trajectories are generated when splits are built, but their test values/labels are not used for training or budget selection.
- Controls: identical randomly initialized encoder; raw current-patch features and mean-pooled sequence patches; observation-only mean/std/lag-correlation features. Local statistical controls use bounded past windows and never future observations. Each control gets the same train/validation/test probe fitting procedure.
- Probe regularization grid: `1e-5, 0.001, 0.1, 10`. Regression features/targets are scaled using training data; validation selects regularization. Report local per-trajectory results and pooled descriptive scores; report sequence parameter scores across independent test trajectories. Paired trained-minus-untrained changes measure gains relative to initialization.
- Report mean and sample SD across **three seed pairs**, not confidence intervals or significance tests. All individual trajectory and seed metrics remain available.
- Hardware: RTX 3090, deterministic float32 training, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Isolated `.venv-gpu` uses PyTorch `2.11.0+cu128`; the original CPU environment remains intact. The measured causal GPU step was about 0.027 seconds versus 0.057 seconds on CPU; masked steps were about 0.014 versus 0.016 seconds. These are short representative benchmarks, not full-study timings.

Run and resume:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \
  .venv-gpu/bin/python -u -m temporal_ssl.study --output results/research

CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \
  .venv-gpu/bin/python -u -m temporal_ssl.study --output results/research --resume
```

Outputs: `results/research/report.md`, `analysis/aggregate_metrics.csv`, `analysis/seed_metrics.csv`, `analysis/learning_curves.csv`, validation plots, per-system recoverability/control plots and all individual run checkpoints/configs/metrics. Every factor has its own plot axis so a very negative parameter R² cannot conceal positive local-state scores.

The smoke config uses shorter sequences and a faster accumulator range than this study. Hardware/library versions also differ. Consequently, smoke-versus-study score changes cannot be attributed solely to training duration or dataset size. The paired trained/untrained comparisons within this study provide a cleaner check on the contribution of pretraining. Causal and masked locals still have different information sets; direct observation, observability, linear recoverability, reconstruction performance and learned dynamics remain distinct.
