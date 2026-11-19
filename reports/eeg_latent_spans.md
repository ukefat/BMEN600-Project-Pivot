# Latent prediction: longer masked spans

Same shared-encoder/predictor/SIGReg model, seed 600, 25% masking, all 4,894 training windows, 10,000 updates and 640,000 draws. Only requested mask span changes; patches remain 50 ms. Each run selects its minimum combined validation objective. Single-seed results on the same 18 reused test participants are exploratory.

| Span | Latent balanced accuracy | Macro-F1 | Same-span waveform accuracy | Pooled effective rank |
|---|---:|---:|---:|---:|
| 50 ms | 30.2% | 0.252 | 34.9% | 3.39 |
| 100 ms | 30.5% | 0.250 | 37.6% | 3.52 |
| 200 ms | 36.0% | 0.333 | 41.3% | 3.71 |

The 200 ms selected model improves the point estimate relative to the 50 ms latent run, but does not beat the matched waveform model or establish a reliable improvement. All spans retain low effective rank and very small within-window variation. The untrained reference is 35.7% balanced accuracy. Larger gaps alone have not removed the temporal-homogenization failure mode.

The final 100 ms checkpoint scores 42.7% (macro-F1 0.416), while its validation-selected checkpoint scores 30.5%. Both are retained in the numerical results; test performance is not used to pick a checkpoint. The final 200 ms checkpoint scores 30.5%. This illustrates instability and a possible mismatch between the latent validation objective and diagnostic utility, not a validated optimum.

Details: [100 ms](eeg_latent_span100.md), [200 ms](eeg_latent_span200.md), [all selected/final scores and diagnostics](eeg_latent_span_comparison.csv), [selected models with matched waveform references](eeg_latent_span_selected.csv).

Reproduce both: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m temporal_ssl.eeg_latent_spans`. Existing completed runs resume without additional training; original 50 ms artifacts are preserved.
