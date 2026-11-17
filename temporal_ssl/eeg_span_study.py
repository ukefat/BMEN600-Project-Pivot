"""Full-data 100/200 ms mask-span comparison with the completed 50 ms reference."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import eeg_factorial as study
from .eeg_course.download import sha256
from .experiment import save_json
from .plots import plt,finish


def report(root,cfg):
    from .eeg_span_analysis import analyze
    if not (root/'common_reconstruction.csv').exists():analyze(root,cfg)
    classification=[];spectral=[]
    for span,path in [(1,Path(cfg['reference'])),(2,root/'span100'),(4,root/'span200')]:
        for dest,file in [(classification,'classification.csv'),(spectral,'spectral.csv')]:
            f=pd.read_csv(path/file)
            f=f[(f.split=='test')&f.condition.isin(['full_mask25','full_mask50','untrained'])].copy()
            f['span_ms']=span*50;dest.append(f)
    c=pd.concat(classification);s=pd.concat(spectral)
    c.to_csv(root/'classification.csv',index=False);s.to_csv(root/'spectral.csv',index=False)
    summary=c[c.condition!='untrained'].groupby(['span_ms','condition']).balanced_accuracy.agg(['mean','std']).reset_index()
    r=s[s.condition!='untrained'].groupby(['span_ms','condition']).r2.agg(['mean','std']).reset_index()
    summary=summary.merge(r,on=['span_ms','condition'],suffixes=('_ba','_r2'))
    summary.to_csv(root/'summary.csv',index=False)
    differences=[]
    for ratio in [25,50]:
        for metric,frame,value in [('classification',c,'balanced_accuracy'),('spectral',s,'r2')]:
            w=frame[frame.condition==f'full_mask{ratio}'].pivot(index='seed',columns='span_ms',values=value)
            for span in [100,200]:
                for seed,v in (w[span]-w[50]).items():differences.append({'ratio':ratio,'metric':metric,'span_ms':span,'seed':seed,'difference_vs_50ms':v})
    pd.DataFrame(differences).to_csv(root/'paired_changes.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,mean,sd,title in [(axes[0],'mean_ba','std_ba','Dementia balanced accuracy'),(axes[1],'mean_r2','std_r2','Spectral recovery R²')]:
        for condition in ['full_mask25','full_mask50']:
            f=summary[summary.condition==condition]
            ax.errorbar(f.span_ms,f[mean],yerr=f[sd],marker='o',capsize=4,label=condition)
        ax.set(title=title,xlabel='Requested masked span (ms)',xticks=[50,100,200]);ax.legend()
    control=c[(c.span_ms==50)&(c.condition=='untrained')].balanced_accuracy.mean()
    direct=json.loads((Path(cfg['reference'])/'spectral_control/classification.json').read_text())['test']['balanced_accuracy']
    axes[0].axhline(control,color='gray',ls='--',label='Untrained')
    axes[0].axhline(direct,color='black',ls=':',label='Direct spectral')
    axes[1].axhline(s[(s.span_ms==50)&(s.condition=='untrained')].r2.mean(),color='gray',ls='--',label='Untrained')
    for ax in axes:ax.legend(fontsize=8)
    finish(fig,root/'comparison.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,(span,ratio) in zip(axes.flat,[(100,25),(100,50),(200,25),(200,50)]):
        path=root/f'span{span}'
        for seed in cfg['seeds']:
            h=pd.DataFrame(json.loads((path/f'full_mask{ratio}'/f'seed_{seed}'/'history.json').read_text()))
            line=ax.plot(h.step,h.validation_mse,label=f'{seed} validation')[0]
            ax.plot(h.step,h.train_online_mse,ls='--',color=line.get_color(),alpha=.6)
        b=json.loads((path/f'full_mask{ratio}'/f'seed_{cfg["seeds"][0]}'/'validation_baselines.json').read_text())
        ax.axhline(b['mean'],color='gray',ls=':',label='Mean');ax.axhline(b['linear'],color='green',ls=':',label='Interpolation')
        ax.set(title=f'{span} ms, {ratio}% masked',xlabel='Updates',ylabel='Reconstruction MSE');ax.legend(fontsize=7)
    finish(fig,root/'training_curves.png')
    common=pd.read_csv(root/'common_reconstruction.csv')
    bases=pd.read_csv(root/'common_reconstruction_baselines.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for ax,(evaluation_span,evaluation_ratio) in zip(axes.flat,[(50,.25),(50,.5),(200,.25),(200,.5)]):
        f=common[(common.evaluation_span_ms==evaluation_span)&(common.evaluation_ratio==evaluation_ratio)]
        stats=f.groupby(['training_span_ms','training_ratio']).mse.agg(['mean','std'])
        labels=[f'{span} ms / {ratio:.0%}' for span,ratio in stats.index]
        ax.bar(labels,stats['mean'],yerr=stats['std'],capsize=3)
        b=bases[(bases.evaluation_span_ms==evaluation_span)&(bases.evaluation_ratio==evaluation_ratio)].iloc[0]
        ax.axhline(b['mean'],color='gray',ls='--',label='Mean');ax.axhline(b['linear'],color='green',ls='--',label='Interpolation')
        ax.set(title=f'Common {evaluation_span} ms / {evaluation_ratio:.0%} test masks',ylabel='Reconstruction MSE')
        ax.tick_params(axis='x',labelrotation=35);ax.legend(fontsize=7)
    finish(fig,root/'common_reconstruction.png')
    uncertainty=pd.read_csv(root/'participant_bootstrap.csv')
    def table(f):
        lines=['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']
        for row in f.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    from .eeg_pilot import mask_runs
    mask_rows=[]
    for span in [1,2,4]:
        for ratio in cfg['ratios']:
            masks=study.fixed_masks(432,ratio,cfg['evaluation_seed'],span)
            lengths=np.concatenate([mask_runs(m.numpy()) for m in masks])*50
            mask_rows.append({'requested_span_ms':span*50,'ratio':ratio,'median_ms':float(np.median(lengths)),
                'mean_ms':float(lengths.mean()),'max_ms':int(lengths.max()),'minimum_ms':int(lengths.min())})
    gap_summary=pd.DataFrame(mask_rows);gap_summary.to_csv(root/'mask_span_summary.csv',index=False)
    assets=Path('reports/eeg_span_study_assets');assets.mkdir(parents=True,exist_ok=True)
    for file in ['classification.csv','spectral.csv','summary.csv','paired_changes.csv','comparison.png','training_curves.png','participant_bootstrap.csv','common_reconstruction.csv','common_reconstruction_baselines.csv','mask_span_summary.csv','common_reconstruction.png']:
        shutil.copyfile(root/file,assets/file)
    values=summary.set_index(['span_ms','condition'])
    findings=[]
    for ratio in [25,50]:
        accuracies=[values.loc[(span,f'full_mask{ratio}'),'mean_ba'] for span in [50,100,200]]
        findings.append(f'- At {ratio}% masking, mean balanced accuracy is '+', '.join(f'{span} ms: {value:.1%}' for span,value in zip([50,100,200],accuracies))+'.')
    spectrum=values.loc[(100,'full_mask25')]
    findings.append(f'- At 100 ms / 25%, overall spectral R² averages {spectrum.mean_r2:.3f} (seed SD {spectrum.std_r2:.3f}); this is closer to the untrained reference, but remains negative and variable.')
    if ((uncertainty.lower_95<=0)&(uncertainty.upper_95>=0)).all():
        findings.append('- All four paired participant intervals for diagnosis changes relative to 50 ms include zero. This study does not establish a reliable span benefit.')
    findings.append('- Longer gaps can change reconstruction skill without making the mean-pooled features better for diagnosis. These results do not prove the network uses interpolation, or establish that useful information is absent nonlinearly.')
    text=f'''# EEG masked-span study

All usable training windows; requested spans **50, 100 and 200 ms**, crossed with **25% and 50% masking**. Patch size remains 50 ms. The 50 ms reference reuses the completed factorial study; 100/200 ms add twelve new runs (four conditions × paired seeds 600–602). No changes to data, architecture, normalization, batch size, 10,000-update budget, optimizer schedule, probe definitions or participant split. Each model receives 640,000 participant-balanced window draws and must visit all 4,894 usable training windows. Downstream probes retain the matched original 24-window panels and first-four-second observations.

Masked spans contain two/four consecutive patches; start positions are sampled uniformly among visible patches, with exact hidden counts of 40/80 out of 160. Blocks can merge or truncate; longer requested spans are not guarantees about every realized gap. The vectorized span sampler follows the previous visible-start/block-fill distribution, with new RNG realizations. It is identical to the original uniform-subset sampler at span=1. Hidden values are excluded before embedding and cannot become attention keys. Mean and visible-sample interpolation baselines use the exact scored masks. Reconstruction across different spans is a different task and should not be ranked directly.

All twelve new models finish training before test model evaluation begins. Validation checkpoint selection uses each condition's own fixed masking task, every 500 updates. Frozen logistic classifiers and train-scaled, validation-selected ridge spectral probes match the completed study. Neither test performance nor downstream validation labels select encoder checkpoints. Negative R² is retained.

## Realized gaps

{table(gap_summary)}

Gap statistics describe the fixed evaluation masks; starts can merge or truncate, so these are requested span comparisons rather than fixed-length-gap comparisons.

## Test results

Mean and sample SD over three paired seeds; SD is training/sampling variability conditional on the fixed split, not participant confidence intervals.

{table(summary)}

![Span comparison](eeg_span_study_assets/comparison.png)
![Training curves](eeg_span_study_assets/training_curves.png)

Solid lines show fixed-mask validation, dashed lines interval-average online training loss. CSVs include [per-seed classification](eeg_span_study_assets/classification.csv), [spectral recovery](eeg_span_study_assets/spectral.csv), and [paired changes relative to 50 ms](eeg_span_study_assets/paired_changes.csv). Per-condition reconstruction scores, exact masks and interpolation baselines are in the results directories.

This tests whether longer missing intervals improve diagnosis or spectral representations; reconstruction alone cannot establish an interpolation shortcut or prove useful representations. The models retain the same small bottleneck and pooling/readout. One split with 18 reused test participants limits conclusions; expanded windows are from the same 52 training participants. This is exploratory, not independent confirmation or a general optimum for masking duration.

![Matched reconstruction comparisons](eeg_span_study_assets/common_reconstruction.png)

## Interpretation

{chr(10).join(findings)}

## Paired participant uncertainty

{table(uncertainty)}

Paired participant-bootstrap intervals (5,000 resamples within diagnosis, preserving all seed predictions for each person) are saved in [participant uncertainty](eeg_span_study_assets/participant_bootstrap.csv). They are conditional on the fixed trained models and split, and do not capture new training-cohort uncertainty. All span models are additionally scored on identical 50 ms and 200 ms masks at both ratios: [common reconstruction](eeg_span_study_assets/common_reconstruction.csv) and [baselines](eeg_span_study_assets/common_reconstruction_baselines.csv). Those are comparisons of model ability on matched reconstruction tasks, unlike comparing each model's own-span loss.

## Reproduce

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_span_study
```

The command resumes identical configurations, preserving optimizer and sampling/masking/Torch/CUDA states. Config, source hashes, exact window identities, coverage counts, checkpoints, predictions, fitted probe scalers and validation scores are saved under `{root}`. Earlier results and original course files are preserved.
'''
    Path('reports/eeg_span_study.md').write_text(text)
    (root/'report.md').write_text(text.replace('eeg_span_study_assets/','../../reports/eeg_span_study_assets/'))
    save_json(root/'report_provenance.json',{'source_sha256':sha256(Path(__file__)),'summary_sha256':sha256(root/'summary.csv')})


def run(config):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=cfg:raise ValueError('Configuration mismatch')
    save_json(root/'protocol.json',cfg)
    if (root/'complete.json').exists():report(root,cfg);return
    save_json(root/'source_hashes.json',{str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')})
    pilot=Path(cfg['pilot']);model_cfg=json.loads((pilot/'config.json').read_text())
    data,val,split,capped=study.prepare(cfg,pilot,root)
    norm=data.normalizer
    gpu=torch.from_numpy((data.x-norm.mean)/norm.std).cuda();vgpu=torch.from_numpy((val.x-norm.mean)/norm.std).cuda()
    for span in [2,4]:
        local=copy.deepcopy(cfg);local['mask_span']=span;local['data_conditions']=['full']
        output=root/f'span{span*50}';output.mkdir(exist_ok=True)
        save_json(output/'protocol.json',local);np.save(output/'capped_indices.npy',capped)
        for seed in cfg['seeds']:
            for ratio in cfg['ratios']:
                study.train_one(f'full_mask{int(ratio*100)}',seed,ratio,np.arange(len(data)),data,val,gpu,vgpu,local,model_cfg,output)
    del gpu,vgpu;torch.cuda.empty_cache()
    save_json(root/'pretraining_complete.json',{'runs':12,'test_evaluation_started':False})
    for span in [2,4]:
        local=copy.deepcopy(cfg);local['mask_span']=span;local['data_conditions']=['full']
        study.evaluate_all(local,model_cfg,data,val,split,capped,pilot,root/f'span{span*50}')
    report(root,cfg);save_json(root/'complete.json',{'runs':12,'steps_per_run':cfg['steps']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/eeg_span_study.json')
    run(p.parse_args().config)
if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');main()
