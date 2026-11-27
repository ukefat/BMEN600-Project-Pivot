STUDY OVERVIEW FIGURE

Deliverables
  ../study_overview.svg    Self-contained vector artwork with editable Arial text.
  ../study_overview.png    White-background preview exported at 600 dpi.
  build_figure.py          Source; four named Inkscape layers, one per study stage.
  figure_caption.txt       Suggested caption, including schematic/data distinctions.

Layout: 170 x 139.4 mm, suitable for approximately half a page in a report with
one full-width text column. Keep near this width to preserve small-label legibility.
Colors: causal blue #246DA4, masked orange #C66A20, shared components neutral grey.
The SVG contains no external assets, embedded rasters, scripts or font outlines.
Text remains editable; Arial with Helvetica and sans-serif fallbacks.

Reproduce from repository root
  .venv-gpu/bin/python reports/study_overview_assets/build_figure.py

Requires NumPy and the Inkscape command-line executable. The included numerical
extract permits rebuilding without loading the full dataset/checkpoints. If the
extract is removed, the script uses the repository study artifacts and requires
the original PyTorch environment to recreate and verify the exact synthetic run.
No model is trained and no new model predictions are generated.

Signal sources
  signal_extract.npz       Exact saved numerical examples for portable rebuilding.
  signal_provenance.json   Source identities, hashes, sample indices and display rules.

Synthetic: predictable-causal-d7-m11, test trajectory 0, independently reproduced
from its archived seed and verified by exact equality against reconstruction.npz
under the archived channel normalization. The display uses the actual unnormalized
observation x1 and true u/q. Frequency/phase changes arise from the archived dynamics.
EEG: ds004504, training participant sub-001, selected cached epoch 0, samples
0:1000 at 500 Hz. Cached provider-cleaned derivatives, not untouched acquisition
voltages; no new filtering or channel-specific rescaling. Display offsets separate
channels, with common 0.12 SVG viewBox units per microvolt. The EEG panel's time
and voltage scale bars are quantitative. The signal snippets in visible patch
slots use the first 200 samples of that F3 trace, split into 25-sample patches;
these have a separate common drawing scale, without quantitative axes.
Embedding glyphs are abstract vector symbols, not a quantitative embedding heatmap.

Evaluation scope checked against
  reports/research_results.md
  reports/eeg_reconstruction_extended.md
  reports/eeg_window_spectral.md
  reports/eeg_pooling.md
  temporal_ssl/models.py
  temporal_ssl/eeg_models.py
  temporal_ssl/eeg_reconstruction_study.py

The two encoders are independently trained methods, not a shared-weight dual
branch model. The prediction head is retained for waveform evaluation; encoders
are frozen for representation probes/classification. Synthetic regime classification
uses BA, continuous synthetic and EEG spectral targets use R², and EEG diagnosis
uses participant-level BA and macro-F1. EEG spectral probing is window-level.
Masked encoder probing uses unmasked windows; pretraining mask settings are not
test-time classification masks. The figure does not claim convergence, superiority,
clinical validity or a direct equivalence of synthetic and EEG task timescales.
