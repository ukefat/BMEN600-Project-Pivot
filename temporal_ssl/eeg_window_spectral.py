"""Five-band window-level probes of existing frozen EEG waveform encoders."""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch

from .data import Normalizer
from .eeg_course.download import sha256
from .eeg_course.features import integrate_psd
from .eeg_data import EEGDataset
from .eeg_pilot import make_eeg_model
from .eeg_pooling import pool
from .eeg_reconstruction_study import CausalRatioPrediction
from .eeg_spectral_recoverability import check_splits, fit_probe, spectral_metrics, SPLITS, ALPHAS
from .experiment import save_json, seed_everything
from .plots import plt, finish

BANDS = {'delta': (1., 4.), 'theta': (4., 8.), 'alpha': (8., 13.),
         'beta': (13., 30.), 'gamma': (30., 45.)}
CHECKPOINTS = {
    'causal': 'results/eeg_reconstruction_study/context80/seed_600/best.pt',
    'untrained_causal': 'results/eeg_reconstruction_study/context80/seed_600/untrained.pt',
    'masked': 'results/eeg_factorial/full_mask25/seed_600/best.pt',
    'untrained_masked': 'results/eeg_factorial/full_mask25/seed_600/untrained.pt',
}


def band_targets(windows, fs=500.):
    """[window,time,channel] physical EEG -> five channel-averaged fractions."""
    if windows.ndim != 3 or windows.shape[1] != round(4*fs):
        raise ValueError('Expected exactly four seconds per window')
    result = []
    for window in windows:
        f, psd = welch(window.T.astype(np.float64), fs=fs, window='hann',
                       nperseg=round(2*fs), noverlap=round(fs), detrend='constant',
                       scaling='density', average='mean', axis=-1)
        total = integrate_psd(f, psd, 1., 45.)
        if np.any(total <= 1e-12):
            raise ValueError('Undefined relative power for a channel')
        powers = np.stack([integrate_psd(f, psd, *bounds) for bounds in BANDS.values()], -1)
        result.append((powers / total[:, None]).mean(0))
    values = np.stack(result)
    np.testing.assert_allclose(values.sum(1), 1., atol=1e-8)
    return values


@torch.no_grad()
def extract_windows(model, dataset, device):
    values = {'mean': [], 'mean_std': []}
    for start in range(0, len(dataset), 64):
        # Slice before normalization/encoding; future half is never available.
        x = dataset.normalizer.apply(dataset.x[start:start+64, :2000])
        local = model.representations(torch.from_numpy(x).to(device))['local']
        for key, value in pool(local).items():
            if key in values:
                values[key].append(value.cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in values.items()}


def bootstrap_differences(truth, trained, control, ids, draws=2000):
    """Paired subject bootstrap: retain all windows whenever a participant is drawn."""
    groups = [np.flatnonzero(ids == pid) for pid in np.unique(ids)]
    rng = np.random.default_rng(600)
    values = []
    for _ in range(draws):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        sst = np.square(truth[ix]-truth[ix].mean(0)).sum(0)
        difference = (np.square(truth[ix]-control[ix]).sum(0) -
                      np.square(truth[ix]-trained[ix]).sum(0))/sst
        values.append(np.r_[difference, difference.mean()])
    return np.quantile(values, [.025, .975], axis=0)


def run(root, device):
    seed_everything(600, 1)
    root.mkdir(parents=True, exist_ok=True)
    pilot = Path('results/eeg_pilot')
    cfg = json.loads((pilot/'config.json').read_text())
    splits = json.loads((pilot/'participants.json').read_text())
    check_splits(splits)
    with np.load(pilot/'normalization.npz') as stored:
        normalizer = Normalizer(stored['mean'], stored['std'])
    datasets = {k: EEGDataset(splits[k], normalizer) for k in SPLITS}
    targets, features, metadata, provenance = {}, {}, [], []
    for part, data in datasets.items():
        if any(person['sampling_hz'] != 500. for person in splits[part]):
            raise ValueError('Sampling rate mismatch')
        targets[part] = band_targets(data.x[:, :2000])
        metadata.extend([dict(split=part, row=i, **row) for i, row in enumerate(data.metadata)])
        np.save(root/f'{part}_targets.npy', targets[part])
    pd.DataFrame(metadata).to_csv(root/'windows.csv', index=False)
    scores, candidates, predictions = [], [], {}

    def record(name, pooling, part, prediction, alpha=None):
        metrics, per_band = spectral_metrics(targets[part], prediction, targets['train'])
        scores.append(dict(model=name, pooling=pooling, split=part, ridge_alpha=alpha,
                           **metrics, **dict(zip(BANDS, per_band))))
        pd.DataFrame({**{f'truth_{b}': targets[part][:, i] for i, b in enumerate(BANDS)},
                      **{f'prediction_{b}': prediction[:, i] for i, b in enumerate(BANDS)},
                      'participant_id': [r['participant_id'] for r in datasets[part].metadata],
                      'epoch_index': [r['epoch_index'] for r in datasets[part].metadata]}
                     ).to_csv(root/f'{name}_{pooling}_{part}_predictions.csv', index=False)

    for part in ('validation', 'test'):
        record('training_mean', 'none', part,
               np.broadcast_to(targets['train'].mean(0), targets[part].shape))
    for name, checkpoint in CHECKPOINTS.items():
        seed_everything(600, 1)
        model = (CausalRatioPrediction(19, 4000, context_patches=80, **cfg['model'])
                 if 'causal' in name else make_eeg_model(cfg, 'masked_short', (4000, 19)))
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        model = model.to(device).eval().requires_grad_(False)
        features[name] = {part: extract_windows(model, data, device) for part, data in datasets.items()}
        np.savez(root/f'{name}_features.npz',
                 **{f'{part}_{key}': value for part, pools in features[name].items()
                    for key, value in pools.items()})
        provenance.append(dict(model=name, checkpoint=checkpoint, sha256=sha256(checkpoint)))
        for pooling in ('mean', 'mean_std'):
            x = {part: features[name][part][pooling] for part in SPLITS}
            probe, pred, grid = fit_probe(x, targets)
            joblib.dump(probe, root/f'{name}_{pooling}_probe.joblib')
            candidates.extend([dict(model=name, pooling=pooling, **row) for row in grid])
            for part, prediction in pred.items():
                record(name, pooling, part, prediction, probe['ridge'].alpha)
            predictions[name, pooling] = pred['test']
        print(name, 'completed', flush=True)
    pd.DataFrame(scores).to_csv(root/'scores.csv', index=False)
    pd.DataFrame(candidates).to_csv(root/'alpha_selection.csv', index=False)
    comparisons = []
    ids = np.array([r['participant_id'] for r in datasets['test'].metadata])
    for name in ('causal', 'masked'):
        for pooling in ('mean', 'mean_std'):
            trained, control = predictions[name, pooling], predictions['untrained_'+name, pooling]
            ci = bootstrap_differences(targets['test'], trained, control, ids)
            a = spectral_metrics(targets['test'], trained, targets['train'])[1]
            b = spectral_metrics(targets['test'], control, targets['train'])[1]
            for i, band in enumerate([*BANDS, 'average']):
                comparisons.append(dict(model=name, pooling=pooling, band=band,
                    difference=float(np.r_[a-b, (a-b).mean()][i]), lower_95=ci[0, i], upper_95=ci[1, i]))
    pd.DataFrame(comparisons).to_csv(root/'paired_bootstrap.csv', index=False)
    save_json(root/'protocol.json', dict(seed=600, device=device, bands=BANDS,
        checkpoints=provenance, split_sha256=sha256(pilot/'participants.json'),
        normalization_sha256=sha256(pilot/'normalization.npz'),
        window_counts={k:len(v) for k,v in datasets.items()}, participant_counts={k:len(v) for k,v in splits.items()},
        input_seconds=4, input_samples=2000, channel_aggregation='mean of channel-wise relative powers',
        welch_seconds=2, overlap=.5, denominator_hz=[1,45], alpha_grid=ALPHAS,
        fit='train only; validation selects alpha by uniform mean R2; no refit',
        bootstrap='2000 paired participant draws; conditional on fitted probes; unadjusted 95% intervals',
        source_sha256=sha256(__file__)))
    plot(pd.DataFrame(scores), root)


def plot(scores, root):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    labels = {'causal':'Causal 4→4 s', 'untrained_causal':'Untrained causal',
              'masked':'Masked 50 ms / 25%', 'untrained_masked':'Untrained masked'}
    for ax, pooling in zip(axes, ('mean', 'mean_std')):
        table = scores[(scores.split=='test') & (scores.pooling==pooling)]
        for row in table.to_dict('records'):
            ax.plot(list(BANDS), [row[b] for b in BANDS], marker='o', label=labels[row['model']])
        baseline = scores[(scores.model=='training_mean') & (scores.split=='test')].iloc[0]
        ax.plot(list(BANDS), [baseline[b] for b in BANDS], 'k--', label='Training mean target')
        ax.axhline(0, color='gray', linewidth=.7)
        ax.set_title(pooling.replace('_', ' + ')+' patch pooling')
        ax.set_ylabel('Held-out window R²'); ax.legend(fontsize=8)
    finish(fig, root/'per_band_r2.png')


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/eeg_window_spectral'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run(args.output, args.device)
