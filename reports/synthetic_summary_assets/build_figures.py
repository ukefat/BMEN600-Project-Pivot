"""Rebuild report figures from the archived numerical tables; no training required."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(ROOT/'aggregate_metrics.csv')
SEEDS = pd.read_csv(ROOT/'seed_metrics.csv')


def row(condition, source, factor):
    selected = DATA[(DATA.condition == condition) & (DATA.source == source) & (DATA.factor == factor)]
    assert len(selected) == 1, (condition, source, factor)
    return selected.iloc[0]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(ROOT/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def run():
    # Recompute every archived aggregate, rather than trusting rounded report text.
    for r in DATA.itertuples():
        values = SEEDS[(SEEDS.condition == r.condition) & (SEEDS.source == r.source) &
                       (SEEDS.factor == r.factor) & (SEEDS.metric == r.metric)].value.dropna()
        assert len(values) == r.seed_pairs
        if len(values):
            np.testing.assert_allclose(values.mean(), r.mean, atol=1e-12)
            np.testing.assert_allclose(values.std(ddof=1), r.sd_across_seed_pairs, atol=1e-12)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    sources = ['untrained_encoder', 'trained_encoder', 'raw_patch', 'raw_statistics']
    labels = ['Untrained', 'Trained', 'Raw patch', 'Raw stats']
    colors = ['#9aa5b1', '#2879b9', '#e4b35c', '#398268']
    factors = [('local/u/linear', 'Accumulator'), ('local/phase/linear', 'Phase'),
               ('local/q/linear', 'Regime'), ('sequence/delta/linear', 'Accumulator rate δ'),
               ('sequence/omega0/linear', 'Frequency ω₀'), ('sequence/omega1/linear', 'Frequency ω₁')]
    for ax, (factor, title) in zip(axes.flat, factors):
        for j, source in enumerate(sources):
            rows = [row('predictable-'+obj, source, factor) for obj in ['causal', 'masked']]
            x = np.arange(2)+(j-1.5)*.19
            ax.bar(x, [r['mean'] for r in rows], width=.18,
                   yerr=[r.sd_across_seed_pairs for r in rows], capsize=2, color=colors[j], label=labels[j])
        classification = '/q/' in factor
        ax.axhline(.5 if classification else 0, linestyle=':', color='gray')
        ax.set_xticks([0, 1], ['Causal', 'Masked'])
        ax.set_title(title)
        ax.set_ylabel('Balanced accuracy' if classification else 'Test R²')
        if factor in ['local/u/linear', 'local/phase/linear']:
            ax.set_ylim(0, 1.06)
        elif classification:
            ax.set_ylim(.4, .85)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.03))
    save(fig, 'predictable_controls')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    conditions = ['predictable-causal', 'predictable-long-horizon']
    for ax, factor, title in zip(axes, ['sequence/delta/linear','local/phase/linear','local/q/linear'],
                                 ['Trajectory rate δ', 'Local phase', 'Local regime']):
        r = [row(c, 'trained_encoder', factor) for c in conditions]
        ax.bar([0, 1], [v['mean'] for v in r], yerr=[v.sd_across_seed_pairs for v in r],
               capsize=4, color=['#2879b9','#398268'])
        ax.set_xticks([0, 1], ['1-patch lead','4-patch lead'])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Balanced accuracy' if '/q/' in factor else 'Test R²')
    save(fig, 'horizon_tradeoff')
    print(f'Validated {len(DATA)} aggregate rows against the seed table; saved four figure files.')


if __name__ == '__main__':
    run()
