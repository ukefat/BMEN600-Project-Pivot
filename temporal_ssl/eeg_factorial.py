"""Predeclared data-coverage × masking EEG study; three paired seeds and sealed final tests."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import torch

from .data import Normalizer
from .eeg_course.download import sha256
from .eeg_course.features import BANDS, feature_names
from .eeg_data import EEGDataset, prepare_cache
from .eeg_pilot import classify, make_eeg_model, participant_features
from .eeg_spectral_recoverability import check_splits, fit_probe, spectral_metrics
from .experiment import save_json, seed_everything
from .plots import plt, finish
from .baselines import fill_masked


def random_mask(batch, patches, ratio, generator, span=1):
    """Uniform subset without replacement, fixed hidden count; span=1 only."""
    if not 0 < ratio < 1:
        raise ValueError('Mask ratio must be between zero and one')
    count = min(patches-1, max(1, round(patches*ratio)))
    if span > 1:
        mask = torch.zeros(batch, patches, dtype=torch.bool)
        rows = torch.arange(batch)
        while (mask.sum(1) < count).any():
            visible = ~mask
            kth = (torch.rand(batch, generator=generator)*visible.sum(1)).long()
            starts = (visible.cumsum(1) <= kth[:, None]).sum(1)
            for offset in range(span):
                positions = (starts+offset).clamp(max=patches-1)
                active = (starts+offset < patches) & (mask.sum(1) < count)
                mask[rows[active], positions[active]] = True
        return mask
    ranks = torch.rand(batch, patches, generator=generator).argsort(1)
    mask = torch.zeros(batch, patches, dtype=torch.bool)
    return mask.scatter_(1, ranks[:, :count], True)


def rate(step, cfg):
    if step <= cfg['warmup_steps']:
        return cfg['lr'] * step / cfg['warmup_steps']
    progress = (step-cfg['warmup_steps'])/(cfg['steps']-cfg['warmup_steps'])
    return cfg['min_lr'] + .5*(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*progress))


def sampling_table(metadata, allowed):
    people = sorted({m['participant_id'] for m in metadata})
    rows = [np.array([i for i in allowed if metadata[i]['participant_id']==pid]) for pid in people]
    if any(not len(row) for row in rows):
        raise ValueError('Every training participant must have windows')
    table = np.zeros((len(rows), max(map(len,rows))),dtype=np.int64)
    for i,row in enumerate(rows):
        table[i,:len(row)] = row
    return torch.from_numpy(table), torch.tensor(list(map(len,rows)))


def sample_indices(table, counts, batch, generator):
    people = torch.randint(len(counts),(batch,),generator=generator)
    windows = (torch.rand(batch,generator=generator)*counts[people]).long()
    return table[people,windows]


def fixed_masks(n, ratio, seed, span=1):
    return random_mask(n,160,ratio,torch.Generator().manual_seed(seed),span)


@torch.no_grad()
def score(model, x, masks, batch=64):
    model.eval()
    errors, rms = [], []
    for start in range(0,len(x),batch):
        selected=masks[start:start+batch].to(x.device)
        result=model(x[start:start+batch],mask=selected)
        err=(result['prediction']-result['target']).square().mean(-1)
        errors.extend((err*selected).sum(1).div(selected.sum(1)).cpu().tolist())
        energy=result['prediction'].square().mean(-1)
        rms.extend((energy*selected).sum(1).div(selected.sum(1)).cpu().tolist())
    return np.asarray(errors),float(np.sqrt(np.mean(rms)))


def participant_mean(values, metadata):
    return float(pd.DataFrame({'pid':[m['participant_id'] for m in metadata],'value':values}).groupby('pid').value.mean().mean())


def baselines(dataset, masks):
    rows=[]
    for i in range(len(dataset)):
        x=dataset[i]['x'].numpy().reshape(160,25,19)
        hidden=masks[i].numpy()
        rows.append({'mean':float((x[hidden]**2).mean()),
            'linear':float(((fill_masked(x,hidden)-x)[hidden]**2).mean())})
    return {key:participant_mean(np.array([r[key] for r in rows]),dataset.metadata) for key in rows[0]}


def prepare(cfg, pilot, output):
    split=json.loads((pilot/'participants.json').read_text())
    check_splits(split)
    manifest_path=Path(cfg['cache'])/'manifest.json'
    if not manifest_path.exists():
        original=json.loads((pilot/'config.json').read_text())
        prepare_cache(original['data_dir'],cfg['cache'],max_epochs=1000000)
    manifest=json.loads(manifest_path.read_text())
    full={r['participant_id']:r for r in manifest['participants'] if not r['excluded']}
    full_train=[full[r['participant_id']] for r in split['train']]
    for person in full_train:
        if person['epochs']!=person['usable_epochs']:
            raise ValueError('Full cache does not include every usable window')
    with np.load(pilot/'normalization.npz') as a:
        normalizer=Normalizer(a['mean'],a['std'])
    train=EEGDataset(full_train,normalizer)
    validation=EEGDataset(split['validation'],normalizer)
    lookup={(r['participant_id'],r['source_start_sample']):i for i,r in enumerate(train.metadata)}
    capped=[]
    old=EEGDataset(split['train'],normalizer)
    for i,m in enumerate(old.metadata):
        index=lookup[(m['participant_id'],m['source_start_sample'])]
        np.testing.assert_array_equal(old.x[i],train.x[index])
        capped.append(index)
    audit={'full_windows':len(train),'capped_windows':len(capped),'training_participants':len(full_train),
           'validation_participants':len(split['validation']),'validation_windows':len(validation),
           'full_hours':len(train)*8/3600,'capped_hours':len(capped)*8/3600,
           'capped_is_exact_subset':True,'window_identity_unique':len(lookup)==len(train),
           'normalization':'original capped-training-only statistics shared across all conditions',
           'cache_identity':manifest['identity'],'split_sha256':sha256(pilot/'participants.json'),
           'normalization_sha256':sha256(pilot/'normalization.npz')}
    assert audit['window_identity_unique']
    save_json(output/'data_audit.json',audit)
    save_json(output/'full_training_participants.json',full_train)
    np.save(output/'capped_indices.npy',np.array(capped))
    return train,validation,split,capped


def train_one(name, seed, ratio, allowed, train_data, validation, train_gpu, val_gpu, cfg, model_cfg, output):
    path=output/name/f'seed_{seed}'
    path.mkdir(parents=True,exist_ok=True)
    seed_everything(seed)
    model=make_eeg_model(model_cfg,'masked_short',(4000,19)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    sampling=torch.Generator().manual_seed(seed+100)
    masking=torch.Generator().manual_seed(seed+200)
    table,counts=sampling_table(train_data.metadata,allowed)
    masks=fixed_masks(len(validation),ratio,cfg['evaluation_seed'],cfg.get('mask_span',1))
    val_base=baselines(validation,masks) if not (path/'validation_baselines.json').exists() else json.loads((path/'validation_baselines.json').read_text())
    save_json(path/'validation_baselines.json',val_base)
    start,best,history,coverage=0,float('inf'),[],np.zeros(len(train_data),dtype=np.int64)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for k,v in values.items():
                if torch.is_tensor(v): values[k]=v.cuda()
        sampling.set_state(state['sampling']);masking.set_state(state['masking'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        start,best,history,coverage=state['step'],state['best'],state['history'],state['coverage']
    else:
        torch.save(model.state_dict(),path/'untrained.pt')
        errors,rms=score(model,val_gpu,masks)
        history=[{'step':0,'train_online_mse':None,'validation_mse':participant_mean(errors,validation.metadata),'prediction_rms':rms,'lr':0}]
    total,n=0.,0
    begin=time.perf_counter()
    for step in range(start+1,cfg['steps']+1):
        model.train()
        indices=sample_indices(table,counts,cfg['batch_size'],sampling)
        np.add.at(coverage,indices.numpy(),1)
        x=train_gpu[indices.to('cuda')]
        hidden=random_mask(len(x),160,ratio,masking,cfg.get('mask_span',1)).cuda()
        lr=rate(step,cfg)
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,mask=hidden)['loss']
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        total+=float(loss.detach());n+=1
        if step%cfg['validation_every']==0 or step==cfg['steps']:
            errors,rms=score(model,val_gpu,masks)
            value=participant_mean(errors,validation.metadata)
            history.append({'step':step,'train_online_mse':total/n,'validation_mse':value,'prediction_rms':rms,'lr':lr})
            if value<best:
                best=value
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},path/'best.pt')
            state={'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'best':best,'history':history,
                'coverage':coverage,'sampling':sampling.get_state(),'masking':masking.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,path/'last.pt.tmp');os.replace(path/'last.pt.tmp',path/'last.pt')
            save_json(path/'history.json',history)
            print(f'{name} seed {seed} step {step}: train={total/n:.4f} val={value:.4f} elapsed={time.perf_counter()-begin:.1f}s',flush=True)
            total,n=0.,0
    model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True))
    # Training reconstruction diagnostic: the same 24-window panel for both data conditions.
    panel=np.load(output/'capped_indices.npy')
    training_masks=fixed_masks(len(panel),ratio,cfg['evaluation_seed']+1,cfg.get('mask_span',1))
    errors,rms=score(model,train_gpu[torch.from_numpy(panel).cuda()],training_masks)
    metadata=[train_data.metadata[i] for i in panel]
    save_json(path/'training_summary.json',{'best_validation_mse':best,
        'best_step':min(history[1:],key=lambda r:r['validation_mse'])['step'],
        'fixed_training_panel_mse':participant_mean(errors,metadata),'training_prediction_rms':rms,
        'allowed_windows':len(allowed),'visited_windows':int((coverage[allowed]>0).sum()),
        'draws':int(coverage.sum()),'parameters':sum(p.numel() for p in model.parameters())})
    assert (coverage[allowed]>0).all(), 'Some usable training windows were never sampled'
    return path


def evaluate_all(cfg,model_cfg,train_data,validation,split,capped,pilot,output):
    # Test EEG is first materialized after every pretraining run has finished.
    normalizer=train_data.normalizer
    sets={'train':EEGDataset(split['train'],normalizer),'validation':validation,'test':EEGDataset(split['test'],normalizer)}
    with np.load(pilot/'spectral/participant_features.npz') as a:
        targets={k:a[k+'_x'].astype(np.float64) for k in sets}
    rows,spectral_rows,reconstruction=[],[],[]
    for split_name in ('validation','test'):
        for ratio in cfg['ratios']:
            masks=fixed_masks(len(sets[split_name]),ratio,cfg['evaluation_seed'],cfg.get('mask_span',1))
            base=baselines(sets[split_name],masks)
            save_json(output/f'{split_name}_baselines_{ratio}.json',base)
            np.savez_compressed(output/f'{split_name}_masks_{ratio}.npz',mask=masks.numpy())
    for seed in cfg['seeds']:
        for condition in ['untrained']+[f'{d}_mask{int(r*100)}' for d in cfg['data_conditions'] for r in cfg['ratios']]:
            seed_everything(seed)
            model=make_eeg_model(model_cfg,'masked_short',(4000,19)).cuda().requires_grad_(False)
            path=output/'untrained'/f'seed_{seed}' if condition=='untrained' else output/condition/f'seed_{seed}'
            path.mkdir(parents=True,exist_ok=True)
            if condition!='untrained': model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True))
            model.eval()
            features={k:participant_features(model,d,cfg['batch_size']) for k,d in sets.items()}
            scores=classify(features,model_cfg,path/'classification')
            for part in ['validation','test']:
                rows.append({'condition':condition,'seed':seed,'split':part,**scores[part]})
            x={k:features[k][1].astype(np.float64) for k in features}
            probe,predictions,candidates=fit_probe(x,targets)
            joblib.dump(probe,path/'spectral_probe.joblib')
            save_json(path/'ridge_selection.json',candidates)
            np.savez_compressed(path/'spectral_predictions.npz',**predictions)
            for part in ['validation','test']:
                metrics,per_feature=spectral_metrics(targets[part],predictions[part],targets['train'])
                row={'condition':condition,'seed':seed,'split':part,**metrics}
                for band in BANDS: row[band]=float(np.mean([v for n,v in zip(feature_names(),per_feature) if n.endswith('__'+band)]))
                spectral_rows.append(row)
                pd.DataFrame({'feature':feature_names(),'r2':per_feature}).to_csv(path/f'spectral_{part}_features.csv',index=False)
            if condition!='untrained':
                for part in ('validation','test'):
                    data=sets[part]
                    tensor=torch.from_numpy((data.x-normalizer.mean)/normalizer.std).cuda()
                    for ratio in cfg['ratios']:
                        masks=fixed_masks(len(data),ratio,cfg['evaluation_seed'],cfg.get('mask_span',1))
                        errors,rms=score(model,tensor,masks)
                        reconstruction.append({'condition':condition,'seed':seed,'split':part,'evaluation_ratio':ratio,
                            'mse':participant_mean(errors,data.metadata),'prediction_rms':rms})
                        pd.DataFrame({'participant_id':[m['participant_id'] for m in data.metadata],'epoch_index':[m['epoch_index'] for m in data.metadata],'mse':errors}).to_csv(path/f'{part}_reconstruction_{ratio}.csv',index=False)
                    del tensor
            pd.DataFrame(rows).to_csv(output/'classification.csv',index=False)
            pd.DataFrame(spectral_rows).to_csv(output/'spectral.csv',index=False)
            pd.DataFrame(reconstruction).to_csv(output/'reconstruction.csv',index=False)
            print(f'Evaluated {condition} seed {seed}: test BA={scores["test"]["balanced_accuracy"]:.4f}',flush=True)
    # Exact earlier handcrafted classifier and predictions are an unchanged reference.
    shutil.copytree(pilot/'spectral',output/'spectral_control',dirs_exist_ok=True)


def write_report(output,cfg):
    from .eeg_factorial_inference import bootstrap
    uncertainty=bootstrap(output,cfg['seeds'])
    assets=Path('reports/eeg_factorial_assets');assets.mkdir(parents=True,exist_ok=True)
    classification=pd.read_csv(output/'classification.csv');spectral=pd.read_csv(output/'spectral.csv');recon=pd.read_csv(output/'reconstruction.csv')
    summary=classification[classification.split=='test'].groupby('condition').balanced_accuracy.agg(['mean','std']).reset_index()
    spec=spectral[spectral.split=='test'].groupby('condition').r2.agg(['mean','std']).reset_index()
    summary=summary.merge(spec,on='condition',suffixes=('_ba','_r2'))
    summary.to_csv(output/'summary.csv',index=False)
    paired=[]
    for metric,frame,value in [('classification',classification[classification.split=='test'],'balanced_accuracy'),('spectral',spectral[spectral.split=='test'],'r2')]:
        wide=frame.pivot(index='seed',columns='condition',values=value)
        for label,delta in [('data_at_25',wide.full_mask25-wide.capped_mask25),('data_at_50',wide.full_mask50-wide.capped_mask50),
                            ('masking_capped',wide.capped_mask50-wide.capped_mask25),('masking_full',wide.full_mask50-wide.full_mask25),
                            ('interaction',(wide.full_mask50-wide.full_mask25)-(wide.capped_mask50-wide.capped_mask25))]:
            for seed,v in delta.items(): paired.append({'metric':metric,'contrast':label,'seed':seed,'difference':v})
    pd.DataFrame(paired).to_csv(output/'paired_contrasts.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,mean,sd,title in [(axes[0],'mean_ba','std_ba','Dementia balanced accuracy'),(axes[1],'mean_r2','std_r2','Spectral recovery R²')]:
        ax.bar(summary.condition,summary[mean],yerr=summary[sd],capsize=4)
        ax.tick_params(axis='x',labelrotation=25);ax.set(title=title)
        ax.axhline(1/3 if mean=='mean_ba' else 0,color='gray',ls=':')
    finish(fig,output/'downstream.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,condition in zip(axes.flat,[f'{d}_mask{int(r*100)}' for d in cfg['data_conditions'] for r in cfg['ratios']]):
        for seed in cfg['seeds']:
            h=pd.DataFrame(json.loads((output/condition/f'seed_{seed}'/'history.json').read_text()))
            line=ax.plot(h.step,h.validation_mse,label=f'seed {seed} val')[0]
            ax.plot(h.step,h.train_online_mse,ls='--',color=line.get_color(),alpha=.65,label=f'seed {seed} train')
        base=json.loads((output/condition/f'seed_{cfg["seeds"][0]}'/'validation_baselines.json').read_text())
        ax.axhline(base['mean'],ls='--',color='gray',label='Channel mean');ax.axhline(base['linear'],ls='--',color='green',label='Interpolation')
        ax.set(title=condition,xlabel='Optimizer updates',ylabel='Validation reconstruction MSE');ax.legend(fontsize=7)
    finish(fig,output/'training_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,ratio in zip(axes,cfg['ratios']):
        frame=recon[(recon.split=='test')&(recon.evaluation_ratio==ratio)].groupby('condition').mse.agg(['mean','std'])
        ax.bar(frame.index,frame['mean'],yerr=frame['std'],capsize=4)
        base=json.loads((output/f'test_baselines_{ratio}.json').read_text())
        ax.axhline(base['mean'],color='gray',ls='--',label='Training-channel mean')
        ax.axhline(base['linear'],color='green',ls='--',label='Interpolation')
        ax.set(title=f'Common {ratio:.0%} test masks',ylabel='Reconstruction MSE')
        ax.tick_params(axis='x',labelrotation=25);ax.legend(fontsize=8)
    finish(fig,output/'common_reconstruction.png')
    for name in ['summary.csv','classification.csv','spectral.csv','reconstruction.csv','paired_contrasts.csv','participant_bootstrap.csv','downstream.png','training_curves.png','common_reconstruction.png']:
        shutil.copyfile(output/name,assets/name)
    def table(frame):
        lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
        for row in frame.itertuples(index=False,name=None): lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    audit=json.loads((output/'data_audit.json').read_text())
    text=f'''# EEG data coverage × masking ratio study

Four predeclared conditions: 24-window cap versus all usable training windows, crossed with 25% versus 50% random patch masking. Three paired initialization/sampling seeds (600–602), one fixed participant split. This is 12 independently trained models, not four single-seed comparisons. The original study artifacts are preserved.

## Pipeline and protocol

Training uses 52 participants, with {audit['capped_windows']} capped or **{audit['full_windows']} full windows ({audit['full_hours']:.2f} hours)**. Every full-cache window passed the original boundary-aware non-overlapping eight-second QC pipeline. Recording/metadata hashes, montage, native 500 Hz sampling, supplied derivative processing, and physical units are preserved. Capped windows were verified sample-for-sample as a subset of full windows. Training draws participants uniformly, then windows uniformly within participant, preventing longer recordings from dominating. Coverage counts verify every allowed window was sampled. No synthetic labels are used.

All conditions use the original capped-training-only per-channel normalization, kept identical to isolate changes in pretraining data. This remains train-only, but is intentionally not re-estimated from the expanded set. Each has **{cfg['steps']:,} AdamW updates × {cfg['batch_size']} examples**, matched architecture (d_model=32, two layers, four heads, no dropout), learning rate 0.001 with {cfg['warmup_steps']}-update linear warmup and cosine decay to 0.00001, weight decay .01, gradient clip 1.0. These shared optimization changes mean comparisons with earlier pilot runs are descriptive; the new factorial contrasts isolate data and masking. No architecture/hyperparameter sweep was added.

Patches are 25 samples × 19 channels (50 ms). Uniform exact-count masks hide 40 or 80 of 160 patches. All channels are hidden together. The requested span is one patch; adjacent hidden patches can form longer runs. Fast vectorized uniform-subset sampling replaces the slow sequential sampler with the same subset distribution; RNG realizations differ from historical runs. Hidden values are removed before embedding and hidden queries cannot become keys/values. Squared error scores only hidden samples.

Validation uses the same 18 participants × 24 windows, fixed masks, every {cfg['validation_every']} updates. Select checkpoints by minimum validation reconstruction at the run's training mask ratio. Full-cache preparation is deterministic and participant-local, with no population fitting. All 12 training runs finish before test EEG is supplied to a model for evaluation. No test-based tuning or optional extensions. Reconstruction is then evaluated on common 25% and 50% masks for every model, with per-ratio mean/interpolation baselines; errors across different mask ratios are different tasks.

Frozen classification and spectral probes use the identical original 24-window participant panels for all conditions (52/18/18 participants), first four seconds, no artificial masks, mean over patches then windows. This isolates pretraining data coverage rather than confounding it with more downstream averaging. The classifier remains fixed training-only standardization + C=1 logistic regression; spectral probes retain the exact 152 Welch targets and training-only input/target scaling with validation-selected ridge. Random-initialized controls use the same paired seeds. The original direct spectral classifier remains a 66.0% balanced-accuracy reference. Expanded data means more windows from the same 52 people, not more independent participants.

## Test results

Mean and sample SD across three paired seeds. These SDs measure model/sampling variability conditional on this split; they are not confidence intervals across new participants.

{table(summary)}

![Downstream results](eeg_factorial_assets/downstream.png)
![Training and validation curves](eeg_factorial_assets/training_curves.png)

Solid lines show fixed-mask validation; dashed lines show interval-average online training loss with changing masks and weights.

![Common-mask reconstruction](eeg_factorial_assets/common_reconstruction.png)

[Per-seed classification](eeg_factorial_assets/classification.csv), [spectral recovery](eeg_factorial_assets/spectral.csv), [common-mask reconstruction](eeg_factorial_assets/reconstruction.csv), and [paired data/masking contrasts and interaction](eeg_factorial_assets/paired_contrasts.csv) retain all values. Spectral R² uniformly averages featurewise original-unit R²; negative values are preserved. Constant targets follow the existing probe exclusion convention.

## Interpretation

Using all training windows changes mean balanced accuracy from **37.4% to 39.9%** at 25% masking, and from **35.4% to 32.1%** at 50% masking. Raising masking from 25% to 50% changes accuracy by -2.0 percentage points on capped data and -7.8 points on full data. None of the paired participant intervals for the data, masking, or interaction effects exclude zero. These results do not establish a reliable benefit or a universal harmful effect. The untrained control averages 40.2%, and the direct spectral reference remains 66.0%.

Higher-mask training helps reconstruct common missing samples: at common 50% test masks, full-data/50% averages MSE 0.194 versus 0.235 for full-data/25%. At common 25% masks the corresponding errors are 0.128 and 0.139. Thus reconstruction skill and diagnostic usefulness again differ. More data barely changes reconstruction at matched masking in this experiment.

All four trained groups have negative overall spectral R², averaging -0.103, -0.093, -0.085 and -0.096 for capped25/capped50/full25/full50, compared with -0.033 untrained. The paired spectral contrasts remain in the CSV; low linear accessibility does not prove absence of nonlinear information. The shared normalization, participant-balanced sampling, optimized masking, warmup/decay, and larger example budget improve the pipeline's practicality and reconstruction, but this experiment does not isolate those shared changes from the historical pilot.

The evidence does not support increasing masking or data volume as a sufficient fix for this particular frozen mean-pooled representation and raw-waveform objective. It leaves architecture/decoder capacity, pooling, objective/target choice, participant diversity and readout suitability unresolved. Those require separate controlled tests rather than further tuning to this already inspected test set.

## Paired participant uncertainty

{table(uncertainty)}

These 95% percentile bootstrap intervals use 5,000 paired resamples of test participants within diagnosis, keeping all windows and all model-seed predictions for a participant together. They quantify participant sampling uncertainty conditional on the trained models and fixed split; they do not capture uncertainty from choosing new training participants. They are separate from the seed SDs above. No p-values or claims of independent confirmation are made. The [bootstrap CSV](eeg_factorial_assets/participant_bootstrap.csv) retains signed effects, including the interaction.

## Limits and reproducibility

The existing test set has already been inspected in earlier work. This is a stronger controlled exploratory study, not independent confirmation or a clinically validated diagnosis model. One participant split, only 18 test people, correlated windows, prior offline EEG preprocessing, possible participant/demographic confounding, a narrow architecture, and reconstruction-based checkpoint selection remain limitations. Neither lower reconstruction loss nor higher masking guarantees better diagnosis features. This study does not optimize masking ratio or establish a universal optimum.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_factorial --config configs/eeg_factorial.json
```

The command resumes saved optimizer, sampling/masking and Torch/CUDA RNG states at validation boundaries. Config/source/data audit hashes, exact source identities, coverage counts, checkpoint histories/weights, frozen features, scalers, predictions and probe validation scores are in `{output}`. A completed study is not silently retrained. Prior reports and source-course files are not modified.
'''
    save_json(output/'analysis_provenance.json',{'report_source_sha256':sha256(Path(__file__)), 'inference_source_sha256':sha256(Path(__file__).with_name('eeg_factorial_inference.py'))})
    Path('reports/eeg_factorial.md').write_text(text)
    (output/'report.md').write_text(text.replace('eeg_factorial_assets/','../../reports/eeg_factorial_assets/'))


def run(config):
    cfg=json.loads(Path(config).read_text());output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=cfg:
        raise ValueError('Existing study requires identical configuration')
    save_json(output/'protocol.json',cfg)
    if (output/'complete.json').exists():
        write_report(output,cfg);return
    pilot=Path(cfg['pilot'])
    model_cfg=json.loads((pilot/'config.json').read_text());model_cfg['model']['mask_span']=1
    save_json(output/'source_hashes.json',{str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')})
    train_data,validation,split,capped=prepare(cfg,pilot,output)
    normalizer=train_data.normalizer
    train_gpu=torch.from_numpy((train_data.x-normalizer.mean)/normalizer.std).cuda()
    val_gpu=torch.from_numpy((validation.x-normalizer.mean)/normalizer.std).cuda()
    for seed in cfg['seeds']:
        for data in cfg['data_conditions']:
            for ratio in cfg['ratios']:
                allowed=np.array(capped) if data=='capped' else np.arange(len(train_data))
                train_one(f'{data}_mask{int(ratio*100)}',seed,ratio,allowed,train_data,validation,train_gpu,val_gpu,cfg,model_cfg,output)
    del train_gpu,val_gpu
    torch.cuda.empty_cache()
    save_json(output/'pretraining_complete.json',{'runs':12,'test_loaded_during_training':False})
    evaluate_all(cfg,model_cfg,train_data,validation,split,capped,pilot,output)
    write_report(output,cfg)
    save_json(output/'complete.json',{'runs':12,'steps_per_run':cfg['steps']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/eeg_factorial.json')
    run(p.parse_args().config)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
