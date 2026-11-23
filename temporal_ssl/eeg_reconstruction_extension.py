"""Add exact 3:1 forecasting and full-data 75% masking at three spans."""
import argparse,copy,json,os,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import eeg_factorial as masked
from . import eeg_reconstruction_study as causal
from .experiment import save_json,seed_everything
from .eeg_course.download import sha256
from .plots import plt,finish


def paired_uncertainty(root,cfg,provenance):
    lookup={(r['model'],r['seed']):Path(r['checkpoint']).parent/'classification/predictions.csv' for r in provenance}
    contrasts=[('causal_context120','causal_context80'),('causal_context120','causal_context107')]
    contrasts += [(f'masked_{span}ms_75pct',f'masked_{span}ms_50pct') for span in [50,100,200]]
    rows=[];rng=np.random.default_rng(4900)
    for left,right in contrasts:
        cube=[];ids=None;truth=None
        for model in [left,right]:
            values=[]
            for seed in cfg['seeds']:
                f=pd.read_csv(lookup[(model,seed)]);f=f[f.split=='test'].sort_values('participant_id')
                if ids is None:ids=f.participant_id.to_numpy();truth=f.truth.to_numpy()
                np.testing.assert_array_equal(ids,f.participant_id);np.testing.assert_array_equal(truth,f.truth)
                values.append((f.prediction.to_numpy()==truth).astype(float))
            cube.append(values)
        cube=np.asarray(cube);draws=np.zeros((2,5000));point=np.zeros(2)
        for group in ['AD','FTD','HC']:
            ix=np.flatnonzero(truth==group);sample=rng.choice(ix,(5000,len(ix)),replace=True)
            draws+=cube[:,:,sample].mean(axis=(1,3))/3;point+=cube[:,:,ix].mean(axis=(1,2))/3
        low,high=np.quantile(draws[0]-draws[1],[.025,.975])
        rows.append({'contrast':left+' minus '+right,'balanced_accuracy_difference':point[0]-point[1],'lower_95':low,'upper_95':high})
    pd.DataFrame(rows).to_csv(root/'participant_uncertainty.csv',index=False)


@torch.no_grad()
def common_mask_analysis(root,cfg,provenance):
    from .data import Normalizer
    from .eeg_data import EEGDataset
    from .eeg_pilot import make_eeg_model
    pilot=Path(cfg['pilot']);split=json.loads((pilot/'participants.json').read_text())
    with np.load(pilot/'normalization.npz') as a:norm=Normalizer(a['mean'],a['std'])
    data=EEGDataset(split['test'],norm);gpu=torch.from_numpy((data.x-norm.mean)/norm.std).cuda()
    model_cfg=json.loads((pilot/'config.json').read_text());rows=[];baselines=[]
    for span in [50,100,200]:
        masks={ratio:masked.fixed_masks(len(data),ratio,cfg['evaluation_seed'],span//50) for ratio in [.25,.5,.75]}
        for ratio,m in masks.items():baselines.append({'span_ms':span,'evaluation_ratio':ratio,**masked.baselines(data,m)})
        for item in [x for x in provenance if x['model'].startswith(f'masked_{span}ms_')]:
            model=make_eeg_model(model_cfg,'masked_short',(4000,19)).cuda().eval()
            model.load_state_dict(torch.load(item['checkpoint'],map_location='cuda',weights_only=True))
            for ratio,m in masks.items():
                errors,_=masked.score(model,gpu,m)
                rows.append({'model':item['model'],'seed':item['seed'],'span_ms':span,
                    'training_ratio':int(item['model'].split('_')[-1].removesuffix('pct'))/100,
                    'evaluation_ratio':ratio,'mse':masked.participant_mean(errors,data.metadata)})
    pd.DataFrame(rows).to_csv(root/'common_masks.csv',index=False)
    pd.DataFrame(baselines).to_csv(root/'common_mask_baselines.csv',index=False)


def consolidate(root,cfg):
    seed_everything(cfg['seeds'][0])
    if not (root/'causal/causal_horizons.csv').exists():
        from .data import Normalizer
        pilot=Path(cfg['pilot'])
        with np.load(pilot/'normalization.npz') as a:norm=Normalizer(a['mean'],a['std'])
        local=copy.deepcopy(cfg);local['contexts']=[120]
        causal.horizon_analysis(local,json.loads((pilot/'config.json').read_text()),json.loads((pilot/'participants.json').read_text()),norm,root/'causal')
    from .eeg_pilot import mask_runs
    geometry=[]
    for span in [50,100,200]:
        for ratio in [.25,.5,.75]:
            m=masked.fixed_masks(432,ratio,cfg['evaluation_seed'],span//50)
            runs=np.concatenate([mask_runs(row.numpy()) for row in m])*50
            geometry.append({'span_ms':span,'ratio':ratio,'gap_mean_ms':float(runs.mean()),'gap_median_ms':float(np.median(runs)),'gap_max_ms':int(runs.max())})
    pd.DataFrame(geometry).to_csv(root/'mask_geometry.csv',index=False)
    previous=Path(cfg['reference']);classes=[pd.read_csv(previous/'classification.csv')];recon=[pd.read_csv(previous/'reconstruction.csv')]
    provenance=json.loads((previous/'checkpoint_provenance.json').read_text())
    c=pd.read_csv(root/'causal/classification.csv');r=pd.read_csv(root/'causal/reconstruction.csv')
    classes.append(c[c.model=='causal_context120']);recon.append(r[r.model=='causal_context120'])
    provenance.extend([x for x in json.loads((root/'causal/checkpoint_provenance.json').read_text()) if x['model']=='causal_context120'])
    for span in [50,100,200]:
        path=root/f'span{span}';c=pd.read_csv(path/'classification.csv');r=pd.read_csv(path/'reconstruction.csv')
        c=c[c.condition=='full_mask75'].copy();c['model']=f'masked_{span}ms_75pct';classes.append(c.drop(columns='condition'))
        rows=[]
        for row in r[(r.condition=='full_mask75')&(r.evaluation_ratio==.75)].itertuples():
            base=json.loads((path/f'{row.split}_baselines_0.75.json').read_text())
            rows.append({'model':f'masked_{span}ms_75pct','seed':row.seed,'split':row.split,'neural_mse':row.mse,
                'mean_mse':base['mean'],'interpolation_mse':base['linear'],'persistence_mse':None,'linear_trend_mse':None})
        recon.append(pd.DataFrame(rows))
        for seed in cfg['seeds']:
            model_path=path/'full_mask75'/f'seed_{seed}'
            provenance.append({'model':f'masked_{span}ms_75pct','seed':seed,'checkpoint':str(model_path/'best.pt'),
                'sha256':sha256(model_path/'best.pt'),'selected_step':json.loads((model_path/'training_summary.json').read_text())['best_step']})
    paired_uncertainty(root,cfg,provenance)
    if not ((root/'common_masks.csv').exists() and (root/'common_mask_baselines.csv').exists()):common_mask_analysis(root,cfg,provenance)
    c=pd.concat(classes,ignore_index=True);r=pd.concat(recon,ignore_index=True)
    c.to_csv(root/'classification.csv',index=False);r.to_csv(root/'reconstruction.csv',index=False);save_json(root/'checkpoint_provenance.json',provenance)
    summary=c[c.split=='test'].groupby('model').agg(balanced_accuracy=('balanced_accuracy','mean'),ba_seed_sd=('balanced_accuracy','std'),macro_f1=('macro_f1','mean'),f1_seed_sd=('macro_f1','std')).reset_index()
    reconstruction=r[r.split=='test'].groupby('model').agg(neural_mse=('neural_mse','mean'),mse_seed_sd=('neural_mse','std'),mean_mse=('mean_mse','mean'),interpolation_mse=('interpolation_mse','mean'),persistence_mse=('persistence_mse','mean'),linear_trend_mse=('linear_trend_mse','mean')).reset_index()
    reconstruction['fractional_mse_reduction_vs_mean']=1-reconstruction.neural_mse/reconstruction.mean_mse
    summary.to_csv(root/'classification_summary.csv',index=False);reconstruction.to_csv(root/'reconstruction_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(13,5));ax.bar(summary.model,summary.balanced_accuracy,yerr=summary.ba_seed_sd.fillna(0),capsize=3)
    ax.axhline(1/3,color='gray',ls=':');ax.set(ylabel='Three-class test balanced accuracy',ylim=(0,1));ax.tick_params(axis='x',labelrotation=45);finish(fig,root/'classification.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,prefix in zip(axes,['masked','causal']):
        f=reconstruction[reconstruction.model.str.startswith(prefix)]
        x=np.arange(len(f));ax.bar(x-.18,f.neural_mse,.36,yerr=f.mse_seed_sd,capsize=3,label='Neural');ax.bar(x+.18,f.mean_mse,.36,label='Channel mean')
        ax.plot(x,f.interpolation_mse if prefix=='masked' else f.persistence_mse,'o',color='green',label='Interpolation' if prefix=='masked' else 'Persistence')
        ax.set(xticks=x,xticklabels=f.model,title=prefix,ylabel='Normalized test MSE');ax.tick_params(axis='x',labelrotation=45);ax.legend()
    finish(fig,root/'reconstruction.png')
    if (root/'causal/causal_horizons.csv').exists():
        horizons=pd.concat([pd.read_csv(previous/'causal_horizons.csv'),pd.read_csv(root/'causal/causal_horizons.csv')],ignore_index=True)
        horizons.to_csv(root/'causal_horizons.csv',index=False)
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        for ax,context in zip(axes,[80,107,120]):
            f=horizons[horizons.context_patches==context]
            for method in ['neural','mean','persistence','linear_trend']:
                stats=f[f.method==method].groupby('lead_seconds').mse.mean();ax.plot(stats.index,stats.values,label=method)
            ax.set(title=f'{context*.05:g} s history',xlabel='Forecast lead (seconds)',ylabel='Test patch MSE');ax.legend(fontsize=8)
        finish(fig,root/'causal_horizons.png')
    common=pd.read_csv(root/'common_masks.csv');common_base=pd.read_csv(root/'common_mask_baselines.csv')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,span in zip(axes,[50,100,200]):
        f=common[(common.span_ms==span)&(common.evaluation_ratio==.75)].groupby('training_ratio').mse.agg(['mean','std'])
        ax.errorbar(f.index*100,f['mean'],yerr=f['std'],marker='o',capsize=4,label='Neural')
        b=common_base[(common_base.span_ms==span)&(common_base.evaluation_ratio==.75)].iloc[0]
        ax.axhline(b['mean'],color='gray',ls='--',label='Mean');ax.axhline(b['linear'],color='green',ls='--',label='Interpolation')
        ax.set(title=f'{span} ms: common 75% test masks',xlabel='Training masking percentage',ylabel='MSE',xticks=[25,50,75]);ax.legend(fontsize=8)
    finish(fig,root/'common_masks.png')
    fig,axes=plt.subplots(3,3,figsize=(14,10))
    for row,ratio in enumerate([25,50,75]):
        for col,span in enumerate([50,100,200]):
            path=(Path('results/eeg_factorial') if span==50 else Path(f'results/eeg_span_study/span{span}')) if ratio!=75 else root/f'span{span}'
            ax=axes[row,col]
            for seed in cfg['seeds']:
                h=pd.DataFrame(json.loads((path/f'full_mask{ratio}'/f'seed_{seed}'/'history.json').read_text()))
                line=ax.plot(h.step,h.validation_mse,label=str(seed))[0];ax.plot(h.step,h.train_online_mse,ls='--',alpha=.5,color=line.get_color())
            ax.set(title=f'{span} ms / {ratio}%',xlabel='Updates',ylabel='MSE')
            if row==0 and col==0:ax.legend(fontsize=7)
    finish(fig,root/'masked_learning.png')
    def table(f):
        lines=['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']
        for row in f.itertuples(index=False,name=None):lines.append('| '+' | '.join('—' if pd.isna(v) else f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    text=f'''# Extended central EEG waveform baseline

Three-class AD/FTD/HC classification. Nine masked tasks (requested spans 50/100/200 ms × 25/50/75%) and three causal tasks (1:1, approximately 2:1, exact 3:1), three paired seeds each. **36 trained models total**: 24 compatible previous runs reused and 12 new runs (three causal, nine masked). Previous artifacts and binary results are preserved.

## Matched protocol

All 4,894 usable eight-second windows from the same 52 training participants; original 18/18 validation/test participants and 24-window evaluation panels. Same 19 channels/500 Hz, preprocessing/QC, original training-only channel scaling, 50 ms patches, width-32 two-layer/four-head encoder with 56,411 parameters, participant-balanced draws, 10,000 updates × batch 64, AdamW .001, 500-step warmup/cosine decay to .00001, weight decay .01, clip 1.0. The scaler was fitted on the original 24-window-per-training-participant panel and is held fixed across all full-data configurations, rather than silently refitted for this extension. No patch instance normalization, latent targets, architecture/readout changes or test-based tuning. Encoder selection minimizes the run's own validation reconstruction error. Budgets match examples/updates, not hidden-target counts or wall time.

Causal splits retain the full eight-second epoch: 4 s → 4 s; 5.35 s → 2.65 s (nearest 2:1 on the patch grid); and **6 s → 2 s, exactly 3:1** (120 context / 40 future patches). The 3:1 choice is fixed before evaluating results, rather than selecting 3:1 versus 4:1 using test performance. Source tokens use causal attention. Future queries contain no observations and cannot become keys/values; there is no teacher forcing/rollout. More context and shorter prediction change together, so this does not isolate context alone.

At 75% masking, exactly 120 of 160 patches (six seconds) are hidden across all channels, leaving two seconds visible. The existing visible-start block sampler and span=1 uniform subset sampler are reused. Blocks can merge/truncate; requested span is not every realized gap's length. Masked models can use visible context on both sides of gaps. Reconstructions are assessed on their own exact masks, with channel-mean and visible-sample interpolation baselines.

Every frozen classifier still observes **only the first four seconds**, without artificial masks, mean-pools patches/windows, and uses train-only StandardScaler + fixed C=1 logistic regression. Both untrained attention rules and the handcrafted spectral reference are retained. FTD is included in training and evaluation again. No new untrained masked control is needed for 75%: unmasked extraction is independent of mask ratio/span, and paired initial encoder weights are identical. All new pretraining runs finish before their test model evaluations begin.

The [mask geometry CSV](eeg_reconstruction_extended_assets/mask_geometry.csv) reports actual mean/median/maximum gap lengths for the fixed masks, including merged blocks.

## Classification

Mean and sample SD across three paired model/sampling seeds; spectral features have a single deterministic classifier. Seed SD is not independent-participant uncertainty.

{table(summary)}

![Classification](eeg_reconstruction_extended_assets/classification.png)

## Reconstruction

{table(reconstruction)}

![Reconstruction](eeg_reconstruction_extended_assets/reconstruction.png)
![Masked training/validation curves](eeg_reconstruction_extended_assets/masked_learning.png)

Solid curves show validation, dashed curves interval-average online training loss. Raw MSE across different tasks is **not a fair model ranking**. Hidden durations, accessible information, prediction lead and mask locations differ. Fractional reduction versus the training-channel mean contextualizes skill within each task; it is not latent-recovery R². Handcrafted spectral and untrained controls are frozen-feature classification references, not trained waveform decoders. Previous common-mask analyses and forecast-lead curves remain linked in the [earlier central report](eeg_reconstruction_study.md).

![Causal lead curves](eeg_reconstruction_extended_assets/causal_horizons.png)

![Common 75% reconstruction masks](eeg_reconstruction_extended_assets/common_masks.png)

Every masked encoder is additionally tested at all three masking percentages using common per-span masks. These comparisons keep target locations and visible information fixed while changing the training ratio. [Cross-ratio reconstruction CSV](eeg_reconstruction_extended_assets/common_masks.csv) and [baselines](eeg_reconstruction_extended_assets/common_mask_baselines.csv) retain all values.

## Paired diagnosis uncertainty

{table(pd.read_csv(root/'participant_uncertainty.csv'))}

95% percentile intervals use 5,000 paired resamples of test participants within diagnosis, retaining all seed predictions per participant. These are conditional on the fixed trained models/split, not new training-cohort uncertainty or multiple-comparison-adjusted confirmation.

## Findings

The 3:1 causal task averages MSE 0.849 versus 0.907 (1:1) and 0.862 (approximately 2:1). Its error reduction over its own channel-mean baseline is about 9.5%, compared with 4.8% and 8.8%. This changes both history and prediction horizon, so the easier task is not evidence that context alone caused the gain. Three-class balanced accuracy is 45.0% ± 11.4% across seeds, versus 38.7% ± 2.6% and 41.2% ± 8.6%. Both paired diagnosis intervals include zero.

At 75% masking, 50/100/200 ms spans average balanced accuracy 38.8% ± 16.6%, 30.4% ± 4.8%, and 35.7% ± 0.8%. Every paired 75%-minus-50% diagnosis interval includes zero. Thus this extension does not establish a consistent diagnostic benefit from 75% masking. The spectral reference remains 66.0%, untrained causal 39.6%, and untrained masked 40.2%.

Reconstruction does improve on the matching high-mask task. On common 75% masks, training at 25%/50%/75% gives MSE 0.511/0.391/0.343 for 50 ms spans, 0.560/0.453/0.397 for 100 ms, and 0.635/0.547/0.497 for 200 ms. All own-task 75% aggregates also beat linear interpolation (0.343 versus 0.374; 0.397 versus 0.431; 0.497 versus 0.582). Part of the shared-mask advantage can reflect matching the training and evaluation masking distributions. It does not establish a universally better representation or contradict the weaker diagnostic results; all cross-ratio values are retained in the CSV.

## Reproducibility and limits

One split with 18 repeatedly inspected test participants supports an exploratory baseline, not independent confirmation or a clinical claim. More windows add temporal coverage but not independent people. Offline provider preprocessing remains unchanged, so causality describes model access to supplied derivatives. Four-second feature extraction differs from eight-second training but is held constant across models. No adaptive budget extensions or classifier sweeps were performed.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_reconstruction_extension
```

The same configuration resumes saved optimizer/RNG states. [Per-seed classification](eeg_reconstruction_extended_assets/classification.csv), [reconstruction](eeg_reconstruction_extended_assets/reconstruction.csv), and [checkpoint provenance](eeg_reconstruction_extended_assets/checkpoint_provenance.json). All new checkpoints and detailed predictions are under `{root}`. The [original central study](eeg_reconstruction_study.md) remains intact.
'''
    assets=Path('reports/eeg_reconstruction_extended_assets');assets.mkdir(exist_ok=True)
    for file in ['classification.png','reconstruction.png','masked_learning.png','classification.csv','reconstruction.csv','classification_summary.csv','reconstruction_summary.csv','checkpoint_provenance.json','participant_uncertainty.csv','causal_horizons.csv','causal_horizons.png','common_masks.csv','common_mask_baselines.csv','common_masks.png','mask_geometry.csv']:shutil.copyfile(root/file,assets/file)
    old_report=Path('reports/eeg_reconstruction_study.md')
    notice='Latest expanded baseline: [75% masking and exact 3:1 causal task](eeg_reconstruction_extended.md). The original study is retained below.'
    old_text=old_report.read_text()
    if notice not in old_text:old_report.write_text(old_text.replace('# Central EEG waveform reconstruction comparison','# Central EEG waveform reconstruction comparison\n\n'+notice,1))
    Path('reports/eeg_reconstruction_extended.md').write_text(text);(root/'report.md').write_text(text.replace('eeg_reconstruction_extended_assets/','').replace('](eeg_reconstruction_study.md)','](../../reports/eeg_reconstruction_study.md)'))


def run(config):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=cfg:raise ValueError('Configuration mismatch')
    save_json(root/'protocol.json',cfg)
    if (root/'complete.json').exists():consolidate(root,cfg);return
    save_json(root/'source_hashes.json',{str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')})
    pilot=Path(cfg['pilot']);model_cfg=json.loads((pilot/'config.json').read_text())
    data,val,split,capped=masked.prepare(cfg,pilot,root);norm=data.normalizer
    gpu=torch.from_numpy((data.x-norm.mean)/norm.std).cuda();vgpu=torch.from_numpy((val.x-norm.mean)/norm.std).cuda()
    causal_root=root/'causal';causal_root.mkdir(exist_ok=True)
    for seed in cfg['seeds']:causal.train_one(120,seed,data,val,gpu,vgpu,cfg,model_cfg,causal_root)
    for span in [1,2,4]:
        local=copy.deepcopy(cfg);local['mask_span']=span;local['ratios']=[.75];local['data_conditions']=['full']
        path=root/f'span{span*50}';path.mkdir(exist_ok=True);save_json(path/'protocol.json',local);np.save(path/'capped_indices.npy',capped)
        for seed in cfg['seeds']:masked.train_one('full_mask75',seed,.75,np.arange(len(data)),data,val,gpu,vgpu,local,model_cfg,path)
    del gpu,vgpu;torch.cuda.empty_cache();save_json(root/'pretraining_complete.json',{'runs':12})
    causal_cfg=copy.deepcopy(cfg);causal_cfg['contexts']=[120];causal.evaluate(causal_cfg,model_cfg,data,val,split,causal_root);causal.horizon_analysis(causal_cfg,model_cfg,split,norm,causal_root)
    for span in [1,2,4]:
        local=copy.deepcopy(cfg);local['mask_span']=span;local['ratios']=[.75];local['data_conditions']=['full']
        masked.evaluate_all(local,model_cfg,data,val,split,capped,pilot,root/f'span{span*50}')
    consolidate(root,cfg);save_json(root/'complete.json',{'new_runs':12,'total_runs':36})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/eeg_reconstruction_extension.json');run(p.parse_args().config)
if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');main()
