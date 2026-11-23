"""Central matched EEG waveform study: reused masked runs and new causal ratios."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .eeg_models import FirstHalfPrediction
from .models import patchify
from . import eeg_factorial as base
from .eeg_data import EEGDataset
from .eeg_pilot import participant_features,classify
from .experiment import seed_everything,save_json
from .eeg_course.download import sha256
from .plots import plt,finish


class CausalRatioPrediction(FirstHalfPrediction):
    def __init__(self,channels,length,context_patches=80,**kwargs):
        super().__init__(channels,length,**kwargs)
        if not 1<=context_patches<length//self.patch_size:raise ValueError('Invalid causal split')
        self.half=context_patches

    def forward(self,x,mask=None,generator=None):
        if mask is not None:raise ValueError('Causal targets are a fixed future interval')
        patches=patchify(x,self.patch_size);b,n,_=patches.shape
        source=self.embedding(patches[:,:self.half])
        queries=self.mask_token.expand(b,n-self.half,-1)
        tokens=torch.cat([source,queries],1)+self.positions[:n]
        forbidden=torch.arange(n,device=x.device)[None,:].expand(b,-1)>=self.half
        encoded=self.encoder(tokens,mask=self.causal_attention_mask(n,x.device),src_key_padding_mask=forbidden)
        prediction=self.head(encoded[:,self.half:]);target=patches[:,self.half:]
        return {'loss':(prediction-target).square().mean(),'prediction':prediction,'target':target}


@torch.no_grad()
def errors(model,x,batch,per_patch=False):
    model.eval();values=[]
    for start in range(0,len(x),batch):
        r=model(x[start:start+batch]);squared=(r['prediction']-r['target']).square()
        values.extend((squared.mean(-1) if per_patch else squared.mean((1,2))).cpu().tolist())
    return np.array(values)


def make(cfg,context):return CausalRatioPrediction(19,4000,context_patches=context,**cfg['model']).cuda()


def train_one(context,seed,data,val,gpu,vgpu,cfg,model_cfg,root):
    path=root/f'context{context}'/f'seed_{seed}';path.mkdir(parents=True,exist_ok=True)
    seed_everything(seed);model=make(model_cfg,context)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    generator=torch.Generator().manual_seed(seed+100)
    table,counts=base.sampling_table(data.metadata,np.arange(len(data)))
    step0,best,h,coverage=0,float('inf'),[],np.zeros(len(data),dtype=np.int64)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for values in opt.state.values():
            for k,v in values.items():
                if torch.is_tensor(v):values[k]=v.cuda()
        generator.set_state(state['sampling']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        step0,best,h,coverage=state['step'],state['best'],state['history'],state['coverage']
    else:torch.save(model.state_dict(),path/'untrained.pt')
    total,n=0.,0;begin=time.perf_counter()
    for step in range(step0+1,cfg['steps']+1):
        model.train();ix=base.sample_indices(table,counts,cfg['batch_size'],generator);np.add.at(coverage,ix.numpy(),1)
        for group in opt.param_groups:group['lr']=base.rate(step,cfg)
        opt.zero_grad(set_to_none=True);loss=model(gpu[ix.cuda()])['loss']
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite causal loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach());n+=1
        if step%cfg['validation_every']==0 or step==cfg['steps']:
            score=base.participant_mean(errors(model,vgpu,cfg['batch_size']),val.metadata)
            h.append({'step':step,'train_mse':total/n,'validation_mse':score})
            if score<best:best=score;torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},path/'best.pt')
            state={'step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'sampling':generator.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best':best,'history':h,'coverage':coverage}
            torch.save(state,path/'last.pt.tmp');os.replace(path/'last.pt.tmp',path/'last.pt');save_json(path/'history.json',h)
            print(f'context {context} seed {seed} step {step}: train={total/n:.5f} val={score:.5f} elapsed={time.perf_counter()-begin:.1f}s',flush=True)
            total,n=0.,0
    assert (coverage>0).all() and coverage.sum()==cfg['steps']*cfg['batch_size']
    save_json(path/'audit.json',{'visited_windows':int((coverage>0).sum()),'draws':int(coverage.sum()),'parameters':sum(p.numel() for p in model.parameters()),'best_step':min(h,key=lambda r:r['validation_mse'])['step']})


def causal_baselines(dataset,context):
    values={k:[] for k in ['mean','persistence','linear_trend']};cut=context*25
    t=np.arange(cut,dtype=float)-(cut-1)
    design=np.column_stack([np.ones(cut),t]);future=np.column_stack([np.ones(4000-cut),np.arange(1,4000-cut+1)])
    for i in range(len(dataset)):
        x=dataset[i]['x'].numpy();history=x[:cut];target=x[cut:]
        predictions={'mean':np.zeros_like(target),'persistence':np.broadcast_to(history[-1],target.shape),
                     'linear_trend':future@np.linalg.lstsq(design,history,rcond=None)[0]}
        for name,p in predictions.items():values[name].append(float(np.square(p-target).mean()))
    return {k:base.participant_mean(np.array(v),dataset.metadata) for k,v in values.items()}


def evaluate(cfg,model_cfg,data,val,split,root):
    sets={'train':EEGDataset(split['train'],data.normalizer),'validation':val,'test':EEGDataset(split['test'],data.normalizer)}
    classification=[];reconstruction=[];sources=[]
    # Reuse only full-data, matched-budget masked runs, not capped/pilot studies.
    for span,source in [(50,Path('results/eeg_factorial')),(100,Path('results/eeg_span_study/span100')),(200,Path('results/eeg_span_study/span200'))]:
        c=pd.read_csv(source/'classification.csv');r=pd.read_csv(source/'reconstruction.csv')
        for ratio in [.25,.5]:
            condition=f'full_mask{int(ratio*100)}';name=f'masked_{span}ms_{int(ratio*100)}pct'
            for seed in cfg['seeds']:
                path=source/condition/f'seed_{seed}';audit=json.loads((path/'training_summary.json').read_text())
                assert audit['visited_windows']==4894 and audit['draws']==cfg['steps']*cfg['batch_size'] and audit['parameters']==56411
                sources.append({'model':name,'seed':seed,'checkpoint':str(path/'best.pt'),'sha256':sha256(path/'best.pt'),'selected_step':json.loads((path/('training_summary.json' if name.startswith('masked') else 'audit.json')).read_text())['best_step']})
                for part in ['validation','test']:
                    row=c[(c.condition==condition)&(c.seed==seed)&(c.split==part)].iloc[0]
                    classification.append({'model':name,'seed':seed,'split':part,'balanced_accuracy':row.balanced_accuracy,'macro_f1':row.macro_f1})
                    value=r[(r.condition==condition)&(r.seed==seed)&(r.split==part)&(r.evaluation_ratio==ratio)].mse.iloc[0]
                    baseline=json.loads((source/f'{part}_baselines_{ratio}.json').read_text())
                    reconstruction.append({'model':name,'seed':seed,'split':part,'neural_mse':value,'mean_mse':baseline['mean'],'interpolation_mse':baseline['linear'],'persistence_mse':None,'linear_trend_mse':None})
        for seed in cfg['seeds']:
            if span==50:
                for part in ['validation','test']:
                    row=c[(c.condition=='untrained')&(c.seed==seed)&(c.split==part)].iloc[0]
                    classification.append({'model':'untrained_masked','seed':seed,'split':part,'balanced_accuracy':row.balanced_accuracy,'macro_f1':row.macro_f1})
    for context in cfg['contexts']:
        name=f'causal_context{context}'
        baselines={k:causal_baselines(sets[k],context) for k in ['validation','test']}
        for seed in cfg['seeds']:
            path=root/f'context{context}'/f'seed_{seed}';seed_everything(seed);model=make(model_cfg,context)
            if context==80:
                model.load_state_dict(torch.load(path/'untrained.pt',map_location='cuda',weights_only=True))
                scores=classify({k:participant_features(model,d,cfg['batch_size']) for k,d in sets.items()},model_cfg,path/'untrained_classification')
                for part in ['validation','test']:classification.append({'model':'untrained_causal','seed':seed,'split':part,**scores[part]})
            model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True));model.eval()
            scores=classify({k:participant_features(model,d,cfg['batch_size']) for k,d in sets.items()},model_cfg,path/'classification')
            for part in ['validation','test']:
                classification.append({'model':name,'seed':seed,'split':part,**scores[part]})
                d=sets[part];gpu=torch.from_numpy((d.x-d.normalizer.mean)/d.normalizer.std).cuda()
                mse=errors(model,gpu,cfg['batch_size']);del gpu
                pd.DataFrame({'participant_id':[m['participant_id'] for m in d.metadata],'epoch_index':[m['epoch_index'] for m in d.metadata],'mse':mse}).to_csv(path/f'{part}_reconstruction.csv',index=False)
                b=baselines[part];reconstruction.append({'model':name,'seed':seed,'split':part,'neural_mse':base.participant_mean(mse,d.metadata),'mean_mse':b['mean'],'interpolation_mse':None,'persistence_mse':b['persistence'],'linear_trend_mse':b['linear_trend']})
            sources.append({'model':name,'seed':seed,'checkpoint':str(path/'best.pt'),'sha256':sha256(path/'best.pt'),'selected_step':json.loads((path/('training_summary.json' if name.startswith('masked') else 'audit.json')).read_text())['best_step']})
    spectral=json.loads((Path(cfg['pilot'])/'spectral/classification.json').read_text())
    for part in ['validation','test']:classification.append({'model':'handcrafted_spectral','seed':None,'split':part,**spectral[part]})
    pd.DataFrame(classification).to_csv(root/'classification.csv',index=False)
    pd.DataFrame(reconstruction).to_csv(root/'reconstruction.csv',index=False)
    save_json(root/'checkpoint_provenance.json',sources)



@torch.no_grad()
def horizon_analysis(cfg,model_cfg,split,norm,root):
    data=EEGDataset(split['test'],norm);x=torch.from_numpy((data.x-norm.mean)/norm.std).cuda();rows=[]
    for context in cfg['contexts']:
        cut=context*25;target_patches=160-context
        baseline={name:[] for name in ['mean','persistence','linear_trend']}
        t=np.arange(cut,dtype=float)-(cut-1);design=np.column_stack([np.ones(cut),t])
        future=np.column_stack([np.ones(4000-cut),np.arange(1,4000-cut+1)])
        for i in range(len(data)):
            values=data[i]['x'].numpy();history=values[:cut];target=values[cut:]
            for name,pred in [('mean',np.zeros_like(target)),('persistence',np.broadcast_to(history[-1],target.shape)),('linear_trend',future@np.linalg.lstsq(design,history,rcond=None)[0])]:
                baseline[name].append(np.square(pred-target).reshape(target_patches,25,19).mean((1,2)))
        for name,values in baseline.items():
            average=np.asarray(values).mean(0)
            for j,value in enumerate(average):rows.append({'context_patches':context,'seed':None,'method':name,'lead_seconds':(j+1)*.05,'mse':value})
        for seed in cfg['seeds']:
            model=make(model_cfg,context);model.load_state_dict(torch.load(root/f'context{context}'/f'seed_{seed}'/'best.pt',map_location='cuda',weights_only=True))
            average=errors(model,x,cfg['batch_size'],per_patch=True).mean(0)
            for j,value in enumerate(average):rows.append({'context_patches':context,'seed':seed,'method':'neural','lead_seconds':(j+1)*.05,'mse':value})
    frame=pd.DataFrame(rows);frame.to_csv(root/'causal_horizons.csv',index=False)
    fig,axes=plt.subplots(1,len(cfg['contexts']),figsize=(6*len(cfg['contexts']),4.5),squeeze=False)
    for ax,context in zip(axes.flat,cfg['contexts']):
        f=frame[frame.context_patches==context]
        for method in ['neural','mean','persistence','linear_trend']:
            stats=f[f.method==method].groupby('lead_seconds').mse.mean()
            ax.plot(stats.index,stats.values,label=method)
        ax.set(title=f'{context*.05:.2f} s history',xlabel='Prediction lead (seconds)',ylabel='Test patch MSE');ax.legend()
    finish(fig,root/'causal_horizons.png')

def causal_uncertainty(root,cfg):
    cube=[];ids=None;truth=None
    for context in cfg['contexts']:
        values=[]
        for seed in cfg['seeds']:
            f=pd.read_csv(root/f'context{context}'/f'seed_{seed}'/'classification/predictions.csv')
            f=f[f.split=='test'].sort_values('participant_id')
            if ids is None:ids=f.participant_id.to_numpy();truth=f.truth.to_numpy()
            np.testing.assert_array_equal(ids,f.participant_id);np.testing.assert_array_equal(truth,f.truth)
            values.append((f.prediction.to_numpy()==truth).astype(float))
        cube.append(values)
    cube=np.asarray(cube);rng=np.random.default_rng(4800);draws=np.zeros((2,5000));points=np.zeros(2)
    for group in ['AD','FTD','HC']:
        ix=np.flatnonzero(truth==group);resampled=rng.choice(ix,(5000,len(ix)),replace=True)
        draws+=cube[:,:,resampled].mean(axis=(1,3))/3;points+=cube[:,:,ix].mean(axis=(1,2))/3
    lower,upper=np.quantile(draws[1]-draws[0],[.025,.975])
    pd.DataFrame([{'contrast':'causal_approx2to1_minus_1to1','balanced_accuracy_difference':points[1]-points[0],
        'lower_95':lower,'upper_95':upper,'bootstrap_samples':5000}]).to_csv(root/'causal_ratio_uncertainty.csv',index=False)


def report(root,cfg):
    causal_uncertainty(root,cfg)
    c=pd.read_csv(root/'classification.csv');r=pd.read_csv(root/'reconstruction.csv')
    summary=c[c.split=='test'].groupby('model').agg(balanced_accuracy=('balanced_accuracy','mean'),ba_seed_sd=('balanced_accuracy','std'),macro_f1=('macro_f1','mean'),f1_seed_sd=('macro_f1','std')).reset_index()
    rec=r[r.split=='test'].groupby('model').agg(neural_mse=('neural_mse','mean'),mse_seed_sd=('neural_mse','std'),mean_mse=('mean_mse','mean'),interpolation_mse=('interpolation_mse','mean'),persistence_mse=('persistence_mse','mean'),linear_trend_mse=('linear_trend_mse','mean')).reset_index()
    rec['fractional_mse_reduction_vs_mean']=1-rec.neural_mse/rec.mean_mse
    summary.to_csv(root/'classification_summary.csv',index=False);rec.to_csv(root/'reconstruction_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(summary));ax.bar(x,summary.balanced_accuracy,yerr=summary.ba_seed_sd.fillna(0),capsize=3)
    ax.set(xticks=x,xticklabels=summary.model,ylabel='Test balanced accuracy',ylim=(0,1));ax.tick_params(axis='x',labelrotation=40);ax.axhline(1/3,ls=':',color='gray');finish(fig,root/'classification.png')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,prefix in zip(axes,['masked','causal']):
        f=rec[rec.model.str.startswith(prefix)];x=np.arange(len(f));ax.bar(x-.2,f.neural_mse,.4,yerr=f.mse_seed_sd,capsize=3,label='Neural');ax.bar(x+.2,f.mean_mse,.4,label='Channel mean')
        ax.plot(x,f.interpolation_mse if prefix=='masked' else f.persistence_mse,'o',color='green',label='Interpolation' if prefix=='masked' else 'Persistence')
        ax.set(xticks=x,xticklabels=f.model,title=prefix,ylabel='Normalized MSE');ax.tick_params(axis='x',labelrotation=35);ax.legend()
    finish(fig,root/'reconstruction.png')
    def table(f):
        lines=['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']
        for row in f.itertuples(index=False,name=None):lines.append('| '+' | '.join('—' if pd.isna(v) else f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    text=f'''# Central EEG waveform reconstruction comparison

Six masked configurations (requested spans 50/100/200 ms × 25/50% masking) and two causal forecasting tasks, three paired seeds (600–602). Full-data masked artifacts are reused after checking budget, capacity and coverage; causal models are retrained under the same full-data protocol. No latent-model results are mixed into this baseline.

## Controlled protocol

Same 52 training participants and all 4,894 usable eight-second windows; 18 validation and 18 test participants with the original 24-window panels. Same native 500 Hz/19 channels, preprocessing/QC, 25-sample patches, original training-only normalization, width-32 two-layer/four-head Transformer and linear waveform head (56,411 parameters). Same participant-balanced sampling, 10,000 updates × 64 windows = 640,000 draws, seed pairing, AdamW .001, 500-step warmup/cosine decay to .00001, weight decay .01 and clip 1.0. Checkpoints minimize their own validation reconstruction loss, never test or diagnostic labels. Matched examples/updates, not matched training wall time or number of loss-scored samples.

Causal tasks directly predict all remaining patches from the prefix, with no rollout or teacher forcing. Future tokens are position-only queries and never keys/values. **1:1:** 80 context patches / 80 target patches = 4 s → 4 s. **Approximately 2:1:** 107 / 53 patches = 5.35 s → 2.65 s (ratio 2.019:1). Exact 2:1 is impossible with 160 indivisible patches; this rounding retains the same eight-second windows and patch size. All 107 observed patches are genuinely available at training/prediction; no hidden future values enter the encoder. Masked tasks retain their existing visible-key-only bidirectional access and exact mask counts; blocks can merge/truncate.

**Frozen diagnosis observations are identical for every method:** first four seconds only, no masking, then mean patch and participant pooling. Causal locals use causal attention within that prefix; masked locals are bidirectional. Classifier is the fixed training-only StandardScaler + C=1 logistic regression. Untrained controls match each attention rule and paired initialization. Both causal splits have identical initial features on that common four-second prefix, so one untrained causal control is reported. The handcrafted spectral classifier uses the same four-second windows, participant pooling and split. It is a classification reference, not a waveform decoder. Untrained reconstruction is not compared to a trained readout; its frozen classification isolates representation changes.

## Frozen dementia classification

Mean and sample SD across three model/sampling seeds. These SDs are not independent-participant confidence intervals. Handcrafted spectral results are deterministic and have no seed SD.

{table(summary)}

![Classification](eeg_reconstruction_study_assets/classification.png)

## Waveform reconstruction

Each row uses its own hidden region/information access. **Do not rank causal versus masked methods by raw MSE**: masked models see both sides of gaps, whereas causal models forecast uncertain future waveforms. Changing a causal split changes both available history and target duration. Even masked rows with different spans/ratios have different targets. Mean/persistence/trend/interpolation are evaluated on exactly that row's targets, using only permitted observations. MSE first averages windows per participant, then participants, then seeds. No normalized patch-target losses are mixed in.

{table(rec)}

![Reconstruction](eeg_reconstruction_study_assets/reconstruction.png)

The existing matched-mask/cross-span reconstruction analyses remain under `results/eeg_factorial` and `results/eeg_span_study`. Per-window causal errors are saved here. Causal persistence repeats the last observed sample; trend extrapolates a least-squares line fitted only to the permitted history. Channel-mean predictions use zero under the common training normalization.

![Causal error versus forecast lead](eeg_reconstruction_study_assets/causal_horizons.png)

The neural curves approach the channel-mean baseline at longer forecast leads; much of the advantage is near the start. Persistence is stronger on the very earliest samples but deteriorates quickly. The horizon curves average patch errors across the same 18 × 24 test windows (equal window counts make this equal to participant averaging), then average paired seeds for neural models. Each lead is relative to its own context endpoint. They expose short-horizon behavior hidden by averaging several seconds of forecasts. [Horizon CSV](eeg_reconstruction_study_assets/causal_horizons.csv).

The fractional MSE reduction column is `1 - neural_MSE / channel_mean_MSE`, not latent recovery R². It contextualizes skill within each task and does not make different forecasting/masking tasks equivalent.

## Findings

The approximately-2:1 causal task has lower test error (0.862 versus 0.907) and a larger fractional reduction versus its own channel-mean baseline (8.8% versus 4.8%). It has both more context and a shorter forecast, so this is an easier-task comparison rather than an isolated context effect. Causal predictions still offer modest improvements over mean prediction when averaged across multi-second targets.

Diagnosis balanced accuracy averages 41.2% for approximately 2:1 and 38.7% for 1:1. The paired, diagnosis-stratified participant bootstrap gives a 2.5 percentage-point difference with a 95% percentile interval of roughly -13.8 to +18.9 points. This is not a reliable diagnosis improvement; the interval is conditional on these fixed trained models and this split, averaging the three seeds. [Uncertainty CSV](eeg_reconstruction_study_assets/causal_ratio_uncertainty.csv).

Masked neural errors increase with gap difficulty; interpolation remains better than the neural decoder in every own-task aggregate. Masked diagnosis accuracy ranges from 32.1% to 39.9%, while untrained masked features average 40.2%. Untrained causal features average 39.6%. The direct spectral classifier remains strongest at 66.0%. Neither lower reconstruction error nor the causal ratio point estimate establishes better diagnostic representations.

## Limits and reproducibility

One existing participant split, three seeds and 18 repeatedly inspected test participants support an exploratory baseline, not independent confirmation or a universal objective ranking. More windows do not add independent participants. Provider offline EEG processing can mix time information within recordings; model causality applies to supplied derivatives. First-four-second feature extraction differs from eight-second SSL training, equally documented across methods. No changes were chosen using new test results. Larger context/shorter horizon is a different forecasting task, not an isolated demonstration that context alone improves prediction.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_reconstruction_study
```

Identical configurations resume optimizer/sampling/Torch/CUDA RNG state. Old artifacts are preserved. [Classification CSV](eeg_reconstruction_study_assets/classification.csv), [reconstruction CSV](eeg_reconstruction_study_assets/reconstruction.csv), and [checkpoint provenance](eeg_reconstruction_study_assets/checkpoint_provenance.json) are portable companions. Details and checkpoints: `{root}`.
'''
    assets=Path('reports/eeg_reconstruction_study_assets');assets.mkdir(exist_ok=True)
    import shutil
    for name in ['classification.png','reconstruction.png','classification.csv','reconstruction.csv','classification_summary.csv','reconstruction_summary.csv','checkpoint_provenance.json','causal_horizons.csv','causal_horizons.png','causal_ratio_uncertainty.csv']:shutil.copyfile(root/name,assets/name)
    Path('reports/eeg_reconstruction_study.md').write_text(text);(root/'report.md').write_text(text.replace('eeg_reconstruction_study_assets/',''))


def run(config):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=cfg:raise ValueError('Configuration mismatch')
    save_json(root/'protocol.json',cfg)
    if (root/'complete.json').exists():report(root,cfg);return
    save_json(root/'source_hashes.json',{str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')})
    pilot=Path(cfg['pilot']);model_cfg=json.loads((pilot/'config.json').read_text())
    data,val,split,capped=base.prepare(cfg,pilot,root);norm=data.normalizer
    gpu=torch.from_numpy((data.x-norm.mean)/norm.std).cuda();vgpu=torch.from_numpy((val.x-norm.mean)/norm.std).cuda()
    for context in cfg['contexts']:
        for seed in cfg['seeds']:train_one(context,seed,data,val,gpu,vgpu,cfg,model_cfg,root)
    del gpu,vgpu;torch.cuda.empty_cache()
    save_json(root/'pretraining_complete.json',{'runs':6})
    evaluate(cfg,model_cfg,data,val,split,root);horizon_analysis(cfg,model_cfg,split,norm,root);report(root,cfg);save_json(root/'complete.json',{'runs':6})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/eeg_reconstruction_study.json');run(p.parse_args().config)
if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');main()
