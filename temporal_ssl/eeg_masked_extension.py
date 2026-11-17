"""Continue medium/long EEG masking to 300 total epochs; train/validation only."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from .data import Normalizer
from .eeg_course.download import sha256
from .eeg_data import EEGDataset
from .eeg_long_training import evaluate_fixed, fixed_baselines
from .eeg_pilot import make_eeg_model
from .experiment import save_json, seed_everything, train
from .plots import plt, finish

NAMES = ('masked_medium', 'masked_long')


def report(output):
    protocol = json.loads((output/'protocol.json').read_text())
    rows = pd.read_csv(output/'fixed_evaluations.csv')
    baselines = json.loads((output/'baselines.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    results = []
    for axes_row,name in zip(axes,NAMES):
        history = pd.DataFrame(json.loads((output/name/'history.json').read_text()))
        best = history.loc[history.validation_loss.idxmin()]
        for ax,split in zip(axes_row,('train','validation')):
            ax.plot(history.epoch,history.train_loss if split=='train' else history.validation_loss,
                    label='Online training' if split=='train' else 'Fixed-mask validation',color='C0')
            fixed = rows[(rows.model==name)&(rows.split==split)&(rows.checkpoint=='last')]
            ax.plot(fixed.epoch,fixed['mean'],'o-',color='C1',label='Epoch-end fixed masks')
            for method,label,color in [('training_mean','Training-channel mean','gray'),('linear','Interpolation','C2')]:
                ax.axhline(baselines[name][split][method]['mean'],ls='--',color=color,label=label)
            ax.axvline(100,color='black',ls=':',lw=.8)
            ax.set(title=f'{name}: {split}',xlabel='Total epochs',ylabel='Normalized masked MSE')
            ax.legend(fontsize=8)
        for epoch,checkpoint in [(100,'last'),(protocol['epochs'],'last'),(protocol['epochs'],'best')]:
            subset=rows[(rows.model==name)&(rows.epoch==epoch)&(rows.checkpoint==checkpoint)].set_index('split')
            results.append({'model':name,'checkpoint':f'{epoch}: {checkpoint}',
                'weights_epoch':int(best.epoch) if checkpoint=='best' else epoch,
                'train_mse':subset.loc['train','mean'],'validation_mse':subset.loc['validation','mean'],
                'validation_prediction_rms':subset.loc['validation','prediction_rms']})
    finish(fig,output/'training_curves.png')
    results=pd.DataFrame(results)
    results.to_csv(output/'summary.csv',index=False)
    baseline_table=pd.DataFrame([{'model':name,'split':split,'mean_mse':v['training_mean']['mean'],
                                  'interpolation_mse':v['linear']['mean']}
                                 for name,parts in baselines.items() for split,v in parts.items()])
    baseline_table.to_csv(output/'baseline_summary.csv',index=False)
    def table(frame):
        lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
        for values in frame.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in values)+' |')
        return '\n'.join(lines)
    findings=[]
    for name in NAMES:
        frame=results[results.model==name]
        first,last,best=frame.iloc[0],frame.iloc[1],frame.iloc[2]
        baseline=baselines[name]['validation']['training_mean']['mean']
        findings.append(f"- {name}: epoch-100 → epoch-{protocol['epochs']} validation MSE "
            f"{first.validation_mse:.6f} → {last.validation_mse:.6f}; selected epoch {best.weights_epoch} "
            f"has validation MSE {best.validation_mse:.6f} ({100*(1-best.validation_mse/baseline):.2f}% below the mean baseline).")
    selected=results[results.checkpoint==f"{protocol['epochs']}: best"]
    if (selected.weights_epoch<=100).all():
        conclusion=("Neither continuation produced a better validation checkpoint than the original 100-epoch sweep. "
                    "The decline in training error without a corresponding validation decline supports overfitting rather than a delayed "
                    "training transition at these settings. Extending the same settings further is not supported by these curves. "
                    "This does not rule out improvements with a changed optimization or reconstruction design, or additional data.")
    else:
        conclusion=("At least one continuation improved the validation minimum after epoch 100. The tables and curves above distinguish "
                    "that improvement from the final checkpoint's performance; late overfitting can coexist with an earlier gain.")
    text=f'''# Medium/long masked EEG: longer training

Continued the two original **100-epoch** runs to **{protocol['epochs']} total epochs** on the RTX 3090. Same 52 training participants (1,248 windows), 18 validation participants (432 windows), 19 channels, eight-second windows, 50 ms patches, 25% masking, requested 200/600 ms spans, initialization seed 600, AdamW learning rate 0.001, batch size 16, weight decay 0.01 and gradient clip 1.0. The model, optimizer, shuffle/mask RNG and Torch/CUDA RNG states were resumed exactly. The fixed epoch budget was set before observing this continuation. No architecture or optimizer sweep.

**No test EEG was loaded, and no dementia classifier or spectral probe was fitted.** Causal and short-mask runs were not extended. Original 100-epoch results and reports are preserved; outputs are in `{output}`.

## Reconstruction

{table(results)}

`last` evaluates the model at that total epoch; `best` evaluates the minimum-validation-loss checkpoint among all epochs 1–{protocol['epochs']}. The label `{protocol['epochs']}: best` describes the selection budget, not necessarily the epoch of its weights. Both final and selected states are reported to expose overfitting or regression. Prediction RMS measures output amplitude around the training-channel means; target RMS is retained in the numerical data.

{table(baseline_table)}

Models and baselines use identical fixed hidden samples: mask seed 604 for training, 603 for validation. Validation masks match the training loop's checkpoint-selection protocol. Model evaluation averages window MSE within participants and then across participants. Online training loss uses changing masks and changing weights, so it need not match epoch-end fixed-mask loss. Interpolation only uses visible samples. Mask spans can merge/truncate, just as in the pilot.

![Training and validation reconstruction](eeg_masked_300_assets/training_curves.png)

## Measured changes

{chr(10).join(findings)}

{conclusion}

The measured trajectories distinguish a delayed reduction in both errors from continued near-mean behavior or training/validation separation. They do not establish whether more EEG data, another learning rate, or a different decoder would help. These are one-seed diagnostics on an already inspected validation split, not independent test confirmation. Reconstruction performance alone does not establish dementia classification or spectral-information quality.

## Reproduce

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_masked_extension \\
  --pilot results/eeg_pilot_100 --output results/eeg_masked_300 --epochs 300
# Add --resume to continue an interrupted run with identical settings.
```

Use a new output directory for a separate repeat. Saved artifacts include source checkpoint/configuration/normalization hashes, source-code snapshot hashes, unchanged original history prefixes, continuation checkpoints with optimizer/RNG state, exact evaluation masks, per-window and per-participant errors, fixed-mask reconstruction examples at each milestone, baseline errors, and [summary CSV](eeg_masked_300_assets/summary.csv) / [milestone CSV](eeg_masked_300_assets/fixed_evaluations.csv) / [baseline CSV](eeg_masked_300_assets/baseline_summary.csv). Tests for fixed-mask alignment and exact continuation through diagnostic evaluation are in `tests/test_eeg_long_training.py`.
'''
    assets=Path('reports/eeg_masked_300_assets')
    assets.mkdir(parents=True,exist_ok=True)
    for filename in ['training_curves.png','summary.csv','fixed_evaluations.csv','baseline_summary.csv']:
        shutil.copyfile(output/filename,assets/filename)
    Path('reports/eeg_masked_300.md').write_text(text)
    (output/'report.md').write_text(text.replace('eeg_masked_300_assets/',''))
    save_json(output/'report_provenance.json',{'source_sha256':sha256(Path(__file__)),
              'summary_sha256':sha256(output/'summary.csv')})


def run(pilot,output,epochs=300,resume=False):
    pilot,output=Path(pilot),Path(output)
    cfg=json.loads((pilot/'config.json').read_text())
    start=cfg['training']['epochs']
    if start!=100 or epochs<=start:
        raise ValueError('Expected the completed 100-epoch sweep and a larger total budget')
    cfg['training']['epochs']=epochs
    partitions=json.loads((pilot/'participants.json').read_text())
    ids={k:{r['participant_id'] for r in rows} for k,rows in partitions.items()}
    if any(ids[a]&ids[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Participant partitions overlap')
    protocol={'pilot':str(pilot.resolve()),'epochs':epochs,'config':cfg,'evaluated_splits':['train','validation'],
        'source_last_sha256':{name:sha256(pilot/name/'last.pt') for name in NAMES},
        'normalization_sha256':sha256(pilot/'normalization.npz'),
        'partitions_sha256':sha256(pilot/'participants.json')}
    if output.exists():
        if not resume or json.loads((output/'protocol.json').read_text())!=protocol:
            raise ValueError('Use a new output directory or --resume with identical settings')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'protocol.json',protocol)
    save_json(output/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
              'source_sha256':{str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')}})
    with np.load(pilot/'normalization.npz') as values:
        norm=Normalizer(values['mean'],values['std'])
    # Explicitly materialize only train/validation; test cache files are never opened.
    datasets={k:EEGDataset(partitions[k],norm) for k in ('train','validation')}
    rows=pd.read_csv(output/'fixed_evaluations.csv').to_dict('records') if (output/'fixed_evaluations.csv').exists() else []
    baselines=json.loads((output/'baselines.json').read_text()) if (output/'baselines.json').exists() else {}
    started=time.perf_counter()
    device=torch.device(cfg['training']['device'])
    for name in NAMES:
        print(f'Starting {name}: 100 → {epochs} epochs',flush=True)
        path=output/name
        path.mkdir(exist_ok=True)
        if not (path/'last.pt').exists():
            shutil.copyfile(pilot/name/'last.pt',path/'last.pt')
        original=torch.load(pilot/name/'last.pt',map_location='cpu',weights_only=False)
        assert original['epoch']==99
        seed_everything(cfg['training']['seed'],cfg['training']['threads'])
        model=make_eeg_model(cfg,name,datasets['train'].x.shape[1:]).to(device)
        baselines[name]={}
        masks={}
        for split,dataset in datasets.items():
            folder=path/split
            folder.mkdir(exist_ok=True)
            baselines[name][split],masks[split]=fixed_baselines(dataset,model,cfg['training']['batch_size'],
                                                           604 if split=='train' else 603,folder)
        save_json(output/'baselines.json',baselines)
        def evaluate(epoch,kind,state):
            model.load_state_dict(state)
            for split,dataset in datasets.items():
                if any(r['model']==name and r['epoch']==epoch and r['checkpoint']==kind and r['split']==split for r in rows):
                    continue
                scores=evaluate_fixed(model,dataset,masks[split],cfg['training']['batch_size'],
                                      path/f'epoch_{epoch:03d}_{kind}_{split}')
                rows.append({'model':name,'epoch':epoch,'checkpoint':kind,'split':split,**scores})
                pd.DataFrame(rows).to_csv(output/'fixed_evaluations.csv',index=False)
                print(f'{name} epoch {epoch} {kind} {split}: MSE={scores["mean"]:.6f}',flush=True)
        evaluate(start,'last',original['model'])
        evaluate(start,'best',original['best_model'])
        for milestone in sorted(set(list(range(start+50,epochs+1,50))+[epochs])):
            chunk=copy.deepcopy(cfg)
            chunk['training']['epochs']=milestone
            train(model,datasets,chunk,path,device,resume=True)
            checkpoint=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
            if checkpoint['epoch']+1!=milestone:
                continue
            evaluate(milestone,'last',checkpoint['model'])
            if milestone==epochs:
                evaluate(milestone,'best',checkpoint['best_model'])
        assert sha256(pilot/name/'last.pt')==protocol['source_last_sha256'][name]
    save_json(output/'timing.json',{'seconds':time.perf_counter()-started})
    report(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',default='results/eeg_pilot_100')
    parser.add_argument('--output',default='results/eeg_masked_300')
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.pilot,args.output,args.epochs,args.resume)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
