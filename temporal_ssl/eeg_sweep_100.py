"""Continue the four EEG pilot objectives to 100 epochs and rerun frozen probes."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import eeg_report, eeg_spectral_recoverability
from .eeg_course.download import sha256
from .eeg_pilot import ORDER, run_pilot
from .experiment import save_json
from .plots import plt, finish


def report(original, output, spectral):
    old = json.loads((original/'summary.json').read_text())
    new = json.loads((output/'summary.json').read_text())
    rows = []
    for name, value in new['classification'].items():
        rows.append({'model':name, 'ba_20':old['classification'][name]['test']['balanced_accuracy'],
                     'ba_100':value['test']['balanced_accuracy'],
                     'f1_20':old['classification'][name]['test']['macro_f1'],
                     'f1_100':value['test']['macro_f1']})
    table = pd.DataFrame(rows)
    table.to_csv(output/'classification_comparison.csv',index=False)
    previous = pd.read_csv('results/eeg_spectral_recoverability/overall.csv')
    current = pd.read_csv(spectral/'overall.csv')
    comparison = previous[['model','test_r2']].merge(current[['model','test_r2','alpha']],on='model',suffixes=('_20','_100'))
    comparison.to_csv(output/'spectral_comparison.csv',index=False)
    assets = Path('reports/eeg_sweep_100_assets')
    assets.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,4.5))
    x = np.arange(len(table))
    ax.bar(x-.18,table.ba_20,.36,label='20 epochs')
    ax.bar(x+.18,table.ba_100,.36,label='100 epochs')
    ax.axhline(1/3,color='gray',ls=':',label='Chance balanced accuracy')
    ax.set(xticks=x,xticklabels=table.model,ylabel='Test balanced accuracy',ylim=(0,1))
    ax.tick_params(axis='x',labelrotation=25)
    ax.legend()
    finish(fig,output/'classification_comparison.png')
    fig, axes = plt.subplots(2,2,figsize=(11,7))
    for ax,name in zip(axes.flat,ORDER):
        history=pd.DataFrame(json.loads((output/name/'history.json').read_text()))
        ax.plot(history.epoch,history.train_loss,label='Training')
        ax.plot(history.epoch,history.validation_loss,label='Validation')
        ax.axvline(20,color='gray',ls=':')
        best=new['models'][name]['best_epoch']
        ax.scatter([best],[new['models'][name]['best_validation_mse']],color='black',label=f'Selected epoch {best}')
        ax.set(title=name,xlabel='Epoch',ylabel='Reconstruction MSE')
        ax.legend(fontsize=8)
    finish(fig,output/'training_curves.png')
    for p in [output/'classification_comparison.png',output/'training_curves.png',output/'classification_comparison.csv',
              output/'spectral_comparison.csv',spectral/'per_band_r2.png',spectral/'overall.csv',
              spectral/'per_feature.csv',spectral/'trained_minus_untrained.csv']:
        shutil.copyfile(p,assets/p.name)
    def markdown(frame):
        lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
        for row in frame.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    reconstruction = pd.DataFrame([{'model':name,'selected_epoch':v['best_epoch'],
          'validation_mse':v['best_validation_mse'],'test_neural_mse':v['reconstruction']['neural']['mean'],
          'test_mean_mse':v['reconstruction']['training_mean']['mean']} for name,v in new['models'].items()])
    reconstruction.to_csv(output/'reconstruction_comparison.csv',index=False)
    short_class=table.set_index('model').loc['masked_short']
    short_spectral=comparison.set_index('model').loc['masked_short']
    findings=(f"Short-mask reconstruction test MSE changes from {old['models']['masked_short']['reconstruction']['neural']['mean']:.3f} "
              f"to {new['models']['masked_short']['reconstruction']['neural']['mean']:.3f}, while its dementia balanced accuracy "
              f"changes from {short_class.ba_20:.1%} to {short_class.ba_100:.1%}. Its overall spectral R² changes from "
              f"{short_spectral.test_r2_20:.3f} to {short_spectral.test_r2_100:.3f}. These measured changes should be interpreted separately; "
              "waveform reconstruction is not a surrogate for diagnosis accuracy or complete spectral preservation.")
    text=f'''# EEG objective comparison at 100 epochs

All four original objectives were continued to **100 total epochs (7,800 optimizer updates each)**. The completed 100-epoch short-mask checkpoint was reused. All model, AdamW, shuffle/mask/Torch/CUDA RNG states were restored from the saved continuation checkpoints. Architecture, learning rate, batch size, preprocessing, normalization, seed 600, and participant splits remain unchanged. Checkpoints are selected by lowest validation reconstruction loss over epochs 1–100. There is no test-based tuning or classifier hyperparameter sweep.

The original split contains 52 train / 18 validation / 18 test participants, with 24 eight-second windows per participant. SSL uses full eight-second windows under each objective's access rules. Every frozen classification and spectral probe uses only the first four seconds, mean-pooling patch embeddings and then windows per participant. Classification uses the original training-only standardization and fixed logistic regression. Spectral recovery uses the same 152 Welch features, training-only input/target scaling, and validation-selected ridge grid. Original 20-epoch artifacts remain in `results/eeg_pilot`; new artifacts are in `{output}`. Spectral probe artifacts are in `{spectral}`.

## Dementia classification

{markdown(table)}

![Classification comparison](eeg_sweep_100_assets/classification_comparison.png)

## Reconstruction at validation-selected checkpoints

{markdown(reconstruction)}

Causal and masked errors score different tasks and should not be ranked directly. Masked interpolation baselines and exact test masks are saved in each model directory. Reconstruction improvement does not guarantee better frozen classification.

![Training and validation reconstruction](eeg_sweep_100_assets/training_curves.png)

## Linear spectral recovery

Uniform mean test R² across 152 features in original target units; negative scores are retained. Each training duration has its own validation-selected ridge penalty.

{markdown(comparison)}

![Spectral recovery by band and power measure](eeg_sweep_100_assets/per_band_r2.png)

## Interpretation limits

{findings}

These are exploratory one-seed comparisons on the existing test set, which has already been examined. They are not independent confirmation or evidence of a general objective ranking. Differences on 18 test participants are descriptive. Longer training changes learned representations; reconstruction, diagnosis classification, and linear spectral recovery measure different properties. The original untrained and direct spectral controls use unchanged observations and fitting protocols. No new architecture, EEG preprocessing, data download, or additional tuning sweep was introduced.

## Reproduce

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_sweep_100
# Add --resume only to continue an interrupted run in the same output directory.
```

CSV tables and plots are alongside this report under `eeg_sweep_100_assets/`; detailed checkpoint hashes, probe scalers, predictions and validation scores remain in the results directories.
'''
    Path('reports/eeg_sweep_100.md').write_text(text)
    (output/'report.md').write_text(text.replace('eeg_sweep_100_assets/', '../../reports/eeg_sweep_100_assets/'))
    save_json(output/'report_provenance.json',{'source_sha256':sha256(Path(__file__)),
              'summary_sha256':sha256(output/'summary.json'),
              'spectral_overall_sha256':sha256(spectral/'overall.csv')})


def run(original, output, short, spectral, resume=False):
    original, output, short, spectral = map(Path,(original,output,short,spectral))
    if output.exists() and not resume:
        raise FileExistsError('Use a new output directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    cfg = copy.deepcopy(json.loads((original/'input_config.json').read_text()))
    cfg['training']['epochs']=100
    save_json(Path('configs/eeg_pilot_100.json'),cfg)
    sources={name:(short/'masked_short' if name=='masked_short' else original/name) for name in ORDER}
    hashes={name:sha256(path/'last.pt') for name,path in sources.items()}
    if not (output/'input_config.json').exists():
        save_json(output/'input_config.json',cfg)
        for name,path in sources.items():
            (output/name).mkdir(exist_ok=True)
            for filename in ['last.pt','best.pt']:
                shutil.copyfile(path/filename,output/name/filename)
        save_json(output/'continuation_sources.json',{'sources':{k:str(v.resolve()) for k,v in sources.items()},'last_sha256':hashes})
    # The default report destinations belong to the original experiments. Generate a separate report below.
    original_writer=eeg_report.write_report
    eeg_report.write_report=lambda *_: None
    try:
        run_pilot(cfg,output,resume=True)
    finally:
        eeg_report.write_report=original_writer
    assert json.loads((original/'participants.json').read_text())==json.loads((output/'participants.json').read_text())
    for name,path in sources.items():
        assert sha256(path/'last.pt')==hashes[name]
    original_spectral_writer=eeg_spectral_recoverability.write_report
    eeg_spectral_recoverability.write_report=lambda *_: None
    try:
        eeg_spectral_recoverability.run(output,spectral,cfg['training']['device'])
    finally:
        eeg_spectral_recoverability.write_report=original_spectral_writer
    report(original,output,spectral)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',default='results/eeg_pilot')
    parser.add_argument('--output',default='results/eeg_pilot_100')
    parser.add_argument('--short',default='results/eeg_long_training')
    parser.add_argument('--spectral',default='results/eeg_spectral_recoverability_100')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.original,args.output,args.short,args.spectral,args.resume)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
