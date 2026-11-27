# Submission readiness and verification

Prepared for BMEN 600 on 2026-11-26. The experiments are complete; this preparation changes documentation, safe reproduction tooling, a portable data-path template and ignore rules. It does not retrain models, change splits/protocols, replace numerical findings, delete/move historical experiments, commit or push.

## Version to be graded

Preparation started from commit **`21b407b2d7fc6db4cc14e065018cc02e9d3c454e`**. This identifies the pre-preparation evidence, **not the final submitted revision**. A commit containing these reviewed changes does not yet exist. After author review and commit/push, obtain the submission SHA and URL:

```bash
git status --short
git rev-parse HEAD
# URL format: https://github.com/ukefat/BMEN600-Project-Pivot/commit/<FULL_SHA>
```

Paste the actual SHA and commit URL into the course submission and final paper/repository citation as appropriate. A document cannot reliably contain the hash of the very commit that changes that document; use the course submission to identify the reviewed revision. The authoritative repository is [BMEN600-Project-Pivot](https://github.com/ukefat/BMEN600-Project-Pivot), not the historical foundation-model project.

## Executed checks

| Check | Actual outcome |
|---|---|
| Original inventory and Git status | Source, configs, tests, report assets, local result trees and pipeline dependencies inspected; working tree initially clean |
| Existing tests | `OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m pytest -q`: **51 passed in 7.56 s** |
| Imports | **37 modules** imported successfully; `temporal_ssl.__main__` excluded because importing it runs the CLI |
| Entry points | Ten argparse `--help` checks passed: synthetic CLI/study, pilot, factorial, spans, causal study/extension, window spectral, supplement and downloader; both new script help commands passed |
| Numerical reproduction | Quick CPU command, including overview, completed in **2.57 s** on the first run and **2.55 s** on a repeat; output directories were new |
| Numerical consistency | Diagnostic/waveform seed summaries match committed tables; all **13** paired interval point/bound calculations match; FTD confusion counts and QC accounting match; original synthetic builder verifies **342** aggregate rows |
| Clean-export quick pathway | Passed from a temporary export of repository files without local data/results inputs, with imports of PyTorch, SciPy, scikit-learn, joblib and `temporal_ssl` explicitly blocked; NumPy/pandas/Matplotlib and Inkscape remained available |
| Relative links and config paths | New/archived Markdown links checked; scientific configs and original paths left in place; portable template differs from the original only in `data_dir` |
| Existing research outputs | Before/after SHA256 comparison verifies original report assets and numerical evidence were not modified |
| Ignore rules | Data/results, virtual environments, checkpoints and `.env` paths are ignored; intentional small signal extract remains available under report assets |
| Credentials and large tracked files | No matches for tested common private-key/GitHub/AWS/API-token patterns; no tracked file above 10 MB. This is a targeted scan, not a comprehensive security audit |
| GitHub/data links | Public final GitHub repository and versioned OpenNeuro metadata reachable; OpenNeuro landing page responds but is JavaScript rendered. Dataset license/citation checked against local/versioned metadata |

Quick reproduction timing is **not** a full-training estimate. All original scientific outputs remain in their original locations; regenerated outputs are under ignored `results/submission_preview/` and `results/submission_verification_final/`. The clean-export check used the existing workstation interpreter and installed report dependencies, not a new pip installation.

## Requirements assessment

| Assignment requirement | Status and evidence |
|---|---|
| README with question, data, map and execution steps | Met: [root README](../README.md), concise final-study entry point with Project Evolution |
| Dataset access, version, citation and licensing | Met: version 1.0.9 download/preparation steps and CC0 source in [reproduction guide](reproduction.md); no dataset/checkpoint download silently assumed |
| Analysis code | Met: existing `temporal_ssl/` modules plus safe CPU figure/table script and metadata-preparation wrapper; no manuscript-only undisclosed analysis claimed |
| Software/platform/setup | Met: `pyproject.toml`, pinned CPU report requirements, observed environment JSON, tested Linux/Python/CUDA details |
| Organized files | Met with minimal changes: code/configs/reports/tests retained; new `scripts/` and `docs/`; [final/history map](study_map.md) avoids destructive restructuring |
| Expected outputs and figure/table provenance | Met: nine-row [paper-output map](reproduction.md#map-of-paper-outputs), command, configuration/source, committed evidence and output paths |
| Practical reproduction | Met for committed numerical tables/figures: CPU-only clean-export execution verified. Model-level reproduction additionally needs data and untracked artifacts or documented full training |
| Working GitHub link | Existing public repository reachable; **new changes still need author review, commit and push** |
| Submitted-version identifier | **Pending author action**: base SHA supplied above; final reviewed commit SHA must be included in the actual submission |

No definitive final manuscript or paper figure numbering was found. The content-based figure/table map avoids inventing them. Existing report PDFs are scientific summaries, not an assertion that the final paper has been deposited here.

## Reproducibility issues and checks not performed

- Full downloads, preprocessing, checkpoint-based evaluation and training were **not** repeated. The fresh metadata wrapper uses the original functions and validates original participant assignments, but was checked by import/help rather than rerunning it into a data/cache directory.
- Full waveform caches, normalization, fitted classifier objects and checkpoint trees are not committed or remotely hosted by this repository. Numerical evidence is sufficient for quick reproduction; independent re-extraction of representations requires these artifacts or the full fresh pipeline.
- New-machine environment installation, another OS/CUDA stack and end-to-end training wall time were not tested. Observed package versions are recorded; performance may not be bit-identical on other environments.
- Historical `configs/eeg_pilot.json` and `eeg_pilot_100.json` retain machine-specific paths for provenance. They are not the portable setup entry point. Use `configs/submission/eeg_pilot.json`; changing historical configs can invalidate existing resume checks.
- The original scientific limits remain: small reused cohort, only five FTD test participants, test-cohort intervals conditional on fitted models, possible acquisition/demographic effects and different target/access geometry across SSL tasks.

## AI assistance and verification

Codex assisted with code, scientific figures, analysis support and submission documentation during the project. Verification records identify checks that actually ran: deterministic fixtures, information-access tests, train-only scaling, exact participant/control pairing, saved-output audits and independent recomputation from committed numerical evidence. This preparation preserved source and result provenance; it did not create new model-performance results. The authors still need to review scientific claims, final manuscript alignment and the final submission revision.

## Final author checklist

- [ ] Review the root README, Project Evolution, interpretation/limitations and final-versus-historical map.
- [ ] Confirm manuscript tables/figures against the content-based output map; add actual paper figure numbers or the final paper link when available.
- [ ] Run the CPU quick pathway in a new output directory if desired; inspect `verification.json` and the regenerated central figures.
- [ ] Confirm course access to the GitHub URL and decide whether additional checkpoint artifacts should be shared through a separate channel; raw data are downloadable from OpenNeuro.
- [ ] Review `git diff`, new files and `git status`; do not stage local `data/`, `results/`, environments, checkpoints or credentials.
- [ ] Commit and push the reviewed files yourself. No automatic commit/push was performed.
- [ ] Run `git rev-parse HEAD`, open the corresponding GitHub commit URL, and put that exact SHA/URL into the grading submission.
