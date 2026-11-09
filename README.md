# Temporal SSL on synthetic dynamical systems

This small PyTorch framework asks: **when can local temporal interpolation solve bidirectional masked reconstruction, and how does this relate to recovery of dynamical state and parameters compared with causal prediction?** It separates optimization loss, held-out reconstruction, and frozen-encoder recoverability. It makes no assumption that causal pretraining should win.

## Install and quick start

Python 3.10+ is required. A CPU is sufficient; the implementation was validated with Python 3.12. Runtime dependencies are NumPy, PyTorch, and Matplotlib. Probes use NumPy, so scikit-learn and a training framework are unnecessary.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
pytest -q

# Eight small runs, covering both models on every system and horizon/span changes.
python -m temporal_ssl sweep --config configs/smoke.json --output results/smoke
```

For CUDA, install an appropriate PyTorch build instead of the CPU wheel and set `training.device` to `"cuda"`. CPU and RTX 3090 CUDA training/resume are tested. Deterministic algorithms are enabled; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for CUDA. Use `OPENBLAS_NUM_THREADS=1` if NumPy's BLAS creates excessive threads on your machine.

## Repository layout

```text
temporal_ssl/
  data.py          generators, split construction, training normalization, batch contract
  models.py        shared patch embedding/Transformer, causal and masked objectives
  experiment.py    training, checkpoints, reconstruction and probe orchestration
  probes.py        ridge/logistic probes, optional nonlinear random-feature probes
  baselines.py     interpolation, nearest filling, persistence and linear trend
  plots.py         trajectory, reconstruction, optimization and comparison plots
  cli.py           generate, run, sweep and analyze commands
configs/           default, smoke, explicit system matrix and sensitivity matrix
tests/             dynamics, leakage, objective, baseline, probe and pipeline tests
```

## Systems and their ground truth

All generators sample independent trajectories, with independently sampled trajectory parameters. A scalar config value fixes a parameter; a two-element array samples it uniformly. `data.length`, `channels`, `splits`, `seed`, `noise`, and `normalize` apply to every system. Sampling is at unit time steps; frequencies are radians per sample.

**Oscillatory AR(2)** (`data.system="ar2"`):

\[
x_t=2r\cos(\omega)x_{t-1}-r^2x_{t-2}+\sigma\epsilon_t,
\quad 0<r<1,\quad \epsilon_t\sim\mathcal N(0,1).
\]

`r` and `omega` are sampled per trajectory; `innovation_noise` is σ. Two Gaussian initial states precede a configurable `burn_in` (default 256). The observation basis is `[x_t, x_(t-1)]`. By default, every channel measures `x_t`, with independent additive observation noise (`data.noise`). Set `observation_matrix` to a `[channels,2]` matrix to change the mapping. Ground truth includes the two-dimensional `state` and `r`, `omega`, and innovation noise. This stochastic oscillator has no separately simulated phase, so phase/regime/accumulator probes are not fabricated for AR(2). Finite burn-in approximates stationarity; increase it for radii very close to one.

**Predictable switching oscillator** (`"predictable"`):

\[
u_{t+1}=u_t+\delta,\qquad
\phi_{t+1}=(\phi_t+\omega_{q_t})\bmod 2\pi.
\]

If the updated accumulator crosses τ, `q` toggles and `u` wraps by subtracting τ (`reset="wrap"`, the default) or resets to zero (`"reset"`). The phase update uses the **old** regime; the changed regime controls the next increment. Require `0 < delta < threshold`. Initial `u` is uniform on `[0,τ)`, phase on `[0,2π)`, and regime on `{0,1}`. `delta`, `threshold`, `omega0`, and `omega1` can vary per trajectory.

The observation basis is `[cos(phi), sin(phi), visibility*u]`. Default channels cycle through these three coordinates; the primary configuration has three channels and `visibility=0.6`. A `[channels,3]` `observation_matrix` mixes the basis, followed by independent Gaussian observation noise. Latents are `u`, `phase=[cos(phi),sin(phi)]`, `phi`, `q`, and `transition` (true at the first sample of a new regime). All generative parameters and simulation settings are retained.

**Random switching control** (`"random"`): the same phase evolution and observation mapping apply. Each step independently toggles `q` with probability `hazard`. On each switch `u` resets to zero; otherwise it increases by δ. Here `u` measures elapsed time since the last switch and does **not** cause switches. With a constant geometric hazard, age gives no warning of the next event. If omitted, hazard is set per trajectory to `delta/threshold`, matching the long-run predictable wrap transition rate. Reset-to-zero predictable dynamics instead have an approximate rate `1/ceil(threshold/delta)`; specify `hazard` explicitly when matching that variant. Switching times match in average rate, not in distribution or exact realization. Random age is unbounded, unlike the predictable accumulator, so its marginal observation distribution differs. This is a scientific difference to control or discuss when interpreting results.

### Observability and interpretation

With the default three-coordinate mapping, nonzero visibility, and low noise, phase coordinates and accumulator directly influence observations. Distinct regime frequencies allow regime inference from phase increments over time; regime is not an instantaneous phase coordinate. Accumulator slope and switches provide evidence about δ and τ. Noise, short context, patch compression, frequency overlap, and non-injective mappings can impair estimation.

At `visibility=0`, accumulator is absent from the instantaneous observation. Histories of past switches can still constrain timing in the predictable system; this does not make every hidden initial condition recoverable. If frequencies coincide, regime labels have no oscillator signature. Channel counts below three can omit the accumulator even with nonzero visibility. A rank-deficient observation matrix can hide or confound phase/accumulator. AR(2) state is exposed through current observations and history under the default mapping, but estimating its parameters requires a trajectory; short or noisy trajectories need not identify them well. No formal identifiability claim is made. These properties concern the data-generating process; they do not guarantee a learned encoder will preserve information or make it linearly recoverable.

## Data generation and split guarantees

```bash
python -m temporal_ssl generate --output data/predictable
python -m temporal_ssl generate --output data/ar2 --set data.system='"ar2"'
python -m temporal_ssl generate --output data/random --set data.system='"random"'
```

Each saves train/validation/test NPZ arrays, raw trajectory/latent plots, a config, a seed manifest, and normalization statistics. NPZ observations use the configured normalization. Latents retain simulator units. Metadata identifies raw simulation parameters and NumPy SeedSequence spawn keys.

The sample contract is `{x: [T,C], latents: optional dict, parameters: optional dict, metadata: dict}`. `collate_trajectories` produces `x: [B,T,C]` and lists of optional labels/metadata. No latent, parameter, or metadata field enters model forward passes. Dataset splits use distinct SeedSequence branches and complete independently generated trajectories. Validation/test sizes do not change training draws. Per-channel mean and standard deviation are fit on training observations only, then fixed for every split and objective. No per-window or per-trajectory standardization is performed; no target-dependent normalization occurs at inference.

## Objectives and representation access

Both models instantiate the same patch projection, sinusoidal position encoding, Transformer width/depth, LayerNorm, and linear reconstruction head. They have exactly equal allocated parameter counts (the causal model leaves the mask token unused). Defaults: 32-dimensional representations, two layers, four heads, four-sample patches, AdamW, identical epochs and batches. Patch size must divide sequence length. Position encodings contain only indices. Encoder normalization operates within each token, not across observed/hidden patches. Boolean attention masks follow the [PyTorch Transformer convention](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html): true disallows attention.

**Causal reconstruction:** for target patch `j`, predict it from a context window ending at patch `j-horizon`. Horizon and context are measured in patches. Horizon 1 predicts the immediately following patch. With patch size P and horizon h, there are `(h-1)*P` unobserved samples between the input patch end and target patch start. All real tokens use causal attention; each endpoint is encoded in its own bounded window so multiple layers cannot indirectly read beyond `context`. Early windows have only the available history. Future target patches are never inputs to their own prediction. The encoder can ingest later patches in other windows for later prediction endpoints; these windows are independent. Loss covers all eligible future targets.

**Masked reconstruction:** sample a fixed hidden-patch count per trajectory, approximately `mask_ratio*N`, with at least one visible and one hidden patch. Blocks start at randomly selected unmasked patches and extend up to `mask_span`; adjacent blocks can merge, boundaries truncate, and the last block can be shortened to meet the count. The requested span is therefore not an exact distribution of realized contiguous lengths. Whole patches are hidden. Hidden observations are replaced **before** patch projection; a learned mask token supplies hidden queries. At every layer, only visible positions can be keys/values, so hidden queries read the bidirectional visible context. A shared linear head reconstructs observations, and only hidden targets contribute loss. This is a small MAE-style model with a shared encoder operating on visible tokens plus masked query positions, rather than a separate large decoder.

**Representation API:** call `model.eval(); model.representations(x)`, which returns `local: [B,N,D]` and `sequence: [B,D]`. Sequence representation is the arithmetic mean of local embeddings. A local target is always aligned with the **last** sample of its patch, because every sample in the current patch has already been observed. Causal local embeddings see at most the configured number of patches ending there. Masked model evaluation defaults to **unmasked full-sequence** local embeddings, including future observations. Supply an explicit `mask` to its representation API for corrupted-input diagnostics. Default probes deliberately measure these different information sets, and `metrics.json` records this access difference. They are not a matched-information causal state estimator comparison. Both sequence representations eventually have access to the full sequence, with different contextualization and pooling effects.

The number and placement of reconstruction targets, visible information, task difficulty, masking, context, and computation differ between objectives, even with identical architectures and optimizer-step budgets. Independent bounded causal windows are more expensive than a single unrestricted pass. Validation selects each objective's best epoch within the shared budget. These are comparisons of complete pretraining regimes, not isolated causal-attention interventions. Neither reconstruction MSE nor probe scores should be used to rank attention mechanisms without these qualifications.

## Train, resume, and evaluate

```bash
python -m temporal_ssl run --output results/causal
python -m temporal_ssl run --output results/masked --set model.objective='"masked"'

# Resume an interrupted run with the identical config and overrides.
python -m temporal_ssl run --output results/masked --set model.objective='"masked"' --resume

# Useful variations; values after '=' must be valid JSON.
python -m temporal_ssl run --output results/long-context \
  --set model.context=16 --set model.horizon=4 --set training.seed=23
python -m temporal_ssl run --output results/nonlinear \
  --set model.objective='"masked"' --set probes.nonlinear=true
```

Each run automatically evaluates its validation-selected checkpoint, freezes the encoder, fits probes, evaluates held-out test trajectories, and saves plots. Existing runs require `--resume`; changed configs are rejected to prevent mixing experiments. `last.pt` stores model, optimizer, epoch, best weights and Torch/shuffle/mask RNG states for epoch-boundary resume. Incomplete epochs are rerun. Checkpoints and JSON writes use temporary-file replacement where needed. A completed resumed run deterministically repeats evaluation. Load only trusted PyTorch checkpoints.

Linear state probes use ridge regression for accumulator, phase coordinates, or AR state, and binary logistic regression for regime. Sequence probes use ridge regression on nonconstant trajectory parameters such as r, ω, and δ. Features and regression targets are scaled using training data only. Regularization is selected on validation data (standardized regression MSE or regime balanced accuracy). Test data never select hyperparameters. The optional nonlinear diagnostic adds 64 fixed seeded tanh random features before the same readout and validation procedure; it is a nonlinear feature probe, not an optimized MLP.

For local probes, per-trajectory MSE/R² or classification metrics and their mean/SEM are reported alongside pooled descriptive scores. R² is computed per output dimension and averaged over dimensions with nonzero within-trajectory variance; it is null when undefined. Regime balanced accuracy averages recalls over the classes present in that trajectory. Sequence-level R² is computed across independent test trajectories; a single trajectory cannot have a sequence-level R². Probe test predictions and targets are saved in trajectory-major order. Time steps are correlated observations, not independent replicates. Trajectory SEM measures sampling variation among trajectories for a fixed fitted model, not uncertainty across simulation/model seeds. Use independently repeated seeds for scientific comparisons; the default smoke run is too small for conclusions.

## Baselines and transitions

Masked reconstruction uses linear interpolation and nearest visible sample filling in **sample time**, across precisely the same hidden patches as the model. Leading/trailing hidden spans use the nearest available boundary value. Baselines access visible observations only. Causal reconstruction uses last-sample persistence and a per-channel least-squares time trend fit only within the same context window, extrapolated over the same horizon. No bidirectional interpolation is scored as a causal predictor. A fitted short-context AR baseline is left as an optional extension.

Channel-averaged MSE is reported in training-normalized units (raw units if normalization is disabled). Test errors are first averaged within each trajectory, then across trajectories, so long time series do not create extra experimental replicates. `evaluation.transition_radius` defines near-transition samples within ±radius **samples** of an actual regime switch; other samples are stable. Scores use the target sample's region, not the source endpoint's region. A patch can contain both regions. Empty regions are null with count zero, including all AR transition scores. Transition labels are used only for post-hoc scoring/plots; models and baselines never receive them. Region definitions use known actual transitions, so they are descriptive, not online predictors of impending switches.

An interpolation baseline with low reconstruction error establishes that the task admits a local solution. Evidence for shortcut learning additionally requires a pattern across masking/horizon/noise/visibility interventions and representation evaluations. Direct observation of u or phase can itself make a probe easy; good state probes are not proof that dynamics were learned. Probe failure is also not proof that information is absent; nonlinear diagnostics and observability controls help interpret it.

## Sweeps and analysis

```bash
# Eight explicit conditions: three systems × two objectives, plus horizon/span changes.
python -m temporal_ssl sweep --output results/main

# Separate small one-factor sensitivity list: context, ratio, noise, visibility, seeds.
python -m temporal_ssl sweep --matrix configs/sensitivity.json --output results/sensitivity

# Rebuild combined plots/tidy tables from completed runs.
python -m temporal_ssl analyze --root results/main
```

Matrices are lists of `{name, overrides}` records with dotted config keys; no Cartesian grid is generated implicitly. Conditions in the default system matrix explicitly set the dataset system; the sensitivity matrix inherits the base system. To repeat the full matrix under new seeds, use a separate output directory and `--set data.seed=... --set training.seed=...`. Keep simulation and initialization seeds distinct; paired objectives share training trajectories and initial weights for the same seed. Add more repeats before drawing statistical conclusions.

Run artifacts include:

- `config.json`, `split_manifest.json`, `environment.json`, `normalization.npz`: settings, independent trajectory identities, installed package versions, source SHA-256 fingerprints/revision and fitted training statistics.
- `last.pt`, `best.pt`, `history.json`: resumable state, selected weights and optimization loss.
- `metrics.json`: reconstruction aggregates, probe selection settings, pooled/per-trajectory scores, test targets/predictions and encoder access notes.
- `reconstruction_trajectories.csv`: errors and target-sample counts per method, trajectory and transition region.
- `reconstruction.npz`: first test example's observations, scored mask, target indices, neural and baseline predictions.
- `trajectory.png`, `reconstruction.png`, `training.png`: separate data, reconstruction, and optimization views.
- `analysis/summary.csv`, `error_vs_task.png`, `recoverability.png`, `transition_errors.png`, `system_comparison.png`: tidy results and cross-run comparisons. Each point represents a named run; graphs do not silently average different conditions or connect confounded runs. Recoverability plots show pooled descriptive R²/accuracy; trajectory means are separately available in the CSV and JSON.

Test masks use a separate fixed evaluation seed and are identical for runs with equal shape/masking settings, independent of model training. One sampled mask per held-out trajectory is used; repeat evaluation seeds to assess mask variability. AR normalization rescales amplitude but does not remove all r/ω-dependent stationary statistics. Noise and visibility interventions can change effective signal-to-noise after normalization, so inspect saved statistics. Random-control latent age has a different distribution, and full-sequence bidirectional access can make state recovery easier independently of pretraining quality.

## Larger paired-seed experiment

`configs/research.json` defines 512/128/128 trajectories, 128-sample sequences, the same small 32-dimensional two-layer models, batch size 32, and three paired data/model seeds. The eight-condition system matrix produces 24 runs. Its validation-only pilot trains every condition for 50 epochs. If any run improves its best validation loss by at least 2% between epochs 31–40 and 41–50, **every** run extends to the same 100-epoch budget. Test evaluation occurs only after the common training budget is complete. This rule is a practical budget decision, not a convergence guarantee.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \
  .venv-gpu/bin/python -m temporal_ssl.study --output results/research

# Resume the complete study, including its recorded budget decision.
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \
  .venv-gpu/bin/python -m temporal_ssl.study --output results/research --resume

python -m temporal_ssl.study --output results/research --report-only
```

The CUDA environment used here is isolated in `.venv-gpu`, with PyTorch `2.11.0+cu128`. It can be recreated with `python -m venv .venv-gpu`, `.venv-gpu/bin/python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`, then `.venv-gpu/bin/python -m pip install -e '.[test]'`. The original CPU environment remains available.

Set `probes.controls=true` to evaluate the identical encoder before training and two observation-only controls with the same probe fitting and validation selection. The raw-patch control flattens the current patch locally and mean-pools patches globally. Raw-statistics features add bounded-past mean/std and centered per-channel lag correlations at 1,2,4,8 samples; sequence features use full-sequence statistics. Local raw controls use no future observations. Untrained encoders preserve each architecture's original access rules. These controls distinguish easily preserved observation content from improvements due to pretraining.

The study saves `report.md`, per-seed/aggregate metric CSVs, paired trained-minus-untrained changes, validation curves and per-system recovery/control plots with independent axes for each factor. Error bars show sample SD across three jointly varied data/model seed pairs; these are not confidence intervals. `budget_history.json` records explicitly authorized epoch extensions while other resume config changes remain rejected. All individual run artifacts and per-trajectory scores are retained.

Reconstruction comparisons are also generated by the study report, or independently from the saved results without retraining:

```bash
python -m temporal_ssl.reconstruction_plots --root results/research
```

Under `results/research/analysis/`, `reconstruction_methods` compares trained models against their applicable baselines across systems; `reconstruction_transitions` separates stable and near-switch errors; `reconstruction_difficulty` compares the default and longer horizon/span conditions; and `reconstruction_examples` shows channel-0 traces from the first test trajectory. Each figure is saved as PNG and vector PDF. `reconstruction_comparison.csv` contains the plotted aggregate values. Bars show trajectory-averaged test MSE, averaged across seed pairs; error bars show sample SD across those pairs, with individual pair scores as black dots. Lower is better. Causal and masked tasks use different targets/information and are shown in separate panels. Raw-patch and raw-statistics controls are probe feature sources and have no standalone reconstruction decoder; they are not reconstruction methods. The untrained encoder control likewise evaluates latent probes rather than a trained reconstruction readout.

## Future EEG integration

Implement a dataset satisfying `TrajectorySource`: return a float tensor `[T,C]`, metadata, and absent/`None` latent/parameter fields. Segment to a fixed length divisible by patch size before batching. Reuse `collate_trajectories`, the models and training loop; reconstruction evaluation accepts missing transition labels, and probe evaluation skips missing synthetic labels. The current `make_splits` factory is intentionally synthetic: replace its call with externally constructed dataset splits when adding an adapter. Separate participants/sessions before extracting windows; fit normalization on the training partition only. Subject/session splitting and real EEG loading are not implemented. Variable-length/padded EEG batches, JEPA objectives and identifiability proofs are outside this implementation.

## Validation

```bash
pytest -q
```

Tests cover seeded generation, shapes, exact switch/phase dynamics, random transition-rate matching, train-only normalization, independent split seeds, optional label batches, multilevel causal future/context exclusion, masked-value exclusion, loss selection, equal parameter allocation, representation extraction, baselines, probe sanity, both complete pipelines and exact epoch resume with dropout. The eight-run smoke matrix exercises AR(2), predictable and random switching end to end. Its tiny training budget verifies execution and artifacts; it does not establish a research finding.
