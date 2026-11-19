"""Initial matched-budget latent versus waveform EEG baseline; no SSL sweep."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import Normalizer
from .eeg_latent import MaskedLatentPrediction,collapse_statistics
from . import eeg_factorial as base
from .eeg_course.download import sha256
from .eeg_data import EEGDataset
from .eeg_pilot import participant_features,classify,make_eeg_model
from .experiment import save_json,seed_everything
from .models import patchify
from .plots import plt,finish


def make_model(cfg,model_cfg):
    return MaskedLatentPrediction(19,4000,sigreg_weight=cfg['sigreg_weight'],sigreg_slices=cfg['sigreg_slices'],**model_cfg['model'])


@torch.no_grad()
def validate(model,x,masks,permutation,cfg):
    model.eval();g=torch.Generator().manual_seed(cfg['evaluation_seed']+10)
    sums={k:0. for k in ['loss','latent_loss','sigreg_loss']};diagnostics=None
    for start in range(0,len(x),cfg['batch_size']):
        ix=permutation[start:start+cfg['batch_size']]
        result=model(x[ix.cuda()],masks[ix].cuda(),g)
        for key in sums:sums[key]+=float(result[key])*len(ix)/len(x)
        if diagnostics is None:
            diagnostics=collapse_statistics(result['original'])
            diagnostics.update({'masked_'+k:v for k,v in collapse_statistics(result['masked']).items()})
    return {**sums,**diagnostics}


def diagnose_controls(root,cfg):
    pilot=Path(cfg['pilot']);model_cfg=json.loads((pilot/'config.json').read_text())
    with np.load(pilot/'normalization.npz') as a:norm=Normalizer(a['mean'],a['std'])
    split=json.loads((pilot/'participants.json').read_text())
    data=EEGDataset(split['validation'],norm)
    with np.load(root/'validation_protocol.npz') as a:order=a['order'][:64]
    x=torch.stack([data[int(i)]['x'] for i in order]).cuda()
    diagnostics={}
    for name,file in [('untrained','untrained.pt'),('latent','best.pt'),('latent_final','last.pt')]:
        seed_everything(cfg['seed']);m=make_model(cfg,model_cfg).cuda().eval()
        state=torch.load(root/file,map_location='cuda',weights_only=(file!='last.pt'))
        m.load_state_dict(state['model'] if file=='last.pt' else state)
        with torch.no_grad():z=m.encode(patchify(x,25))
        diagnostics[name]=collapse_statistics(z)
    raw=make_eeg_model(model_cfg,'masked_short',(4000,19)).cuda().eval()
    raw.load_state_dict(torch.load(Path(cfg['raw_reference'])/'best.pt',map_location='cuda',weights_only=True))
    with torch.no_grad():z=raw.encode(patchify(x,25))
    diagnostics['waveform']=collapse_statistics(z)
    save_json(root/'control_diagnostics.json',diagnostics)
    return diagnostics


def report(root,cfg):
    controls=diagnose_controls(root,cfg)
    h=pd.DataFrame(json.loads((root/'history.json').read_text()))
    rows=[]
    for name in ['untrained','latent','latent_final','waveform']:
        f=json.loads((root/name/'classification.json').read_text())
        for split in ['validation','test']:rows.append({'model':name,'split':split,**f[split]})
    table=pd.DataFrame(rows);table.to_csv(root/'comparison.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for key in ['latent_loss','sigreg_loss','loss']:axes[0,0].plot(h.step,h['validation_'+key],label=key)
    axes[0,0].set_yscale('log');axes[0,0].legend();axes[0,0].set(title='Validation objective components',xlabel='Updates',ylabel='Value (log scale)')
    for prefix in ['', 'masked_']:
        for level in ['patch','pooled']:
            axes[0,1].plot(h.step,h['validation_'+prefix+level+'_variance'],label=prefix+level)
            axes[1,0].plot(h.step,h['validation_'+prefix+level+'_effective_rank'],label=prefix+level)
    axes[0,1].set(title='Across-sample embedding variance',xlabel='Updates');axes[0,1].legend(fontsize=8)
    axes[1,0].set(title='Covariance entropy effective rank',xlabel='Updates');axes[1,0].legend(fontsize=8)
    test=table[table.split=='test'];axes[1,1].bar(test.model,test.balanced_accuracy)
    axes[1,1].axhline(1/3,ls=':',color='gray');axes[1,1].set(title='Frozen diagnosis classifier',ylabel='Balanced accuracy',ylim=(0,1))
    finish(fig,root/'diagnostics.png')
    def markdown(frame):
        lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
        for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    selected=json.loads((root/'selected.json').read_text())
    diagnostics=h[h.step==selected['step']].iloc[0]
    audit=json.loads((root/'training_audit.json').read_text())
    diagnostic_table=pd.DataFrame([{'model':k,**v} for k,v in controls.items()])
    text=f'''# Shared-encoder masked latent prediction EEG baseline

LeJEPA-inspired adaptation, not a reproduction of LeJEPA. One configuration and seed ({cfg['seed']}), with no sweep or adaptive regularization. Predict learned original-sequence patch embeddings rather than raw EEG samples. The existing waveform experiments and source course repository are unchanged.

## Architecture and gradient paths

The encoder is the existing two-layer, 32-dimensional, four-head Transformer, including its final affine LayerNorm, fixed sinusoidal positions and mean pooling. Original and masked sequences call the **same encoder object**. Hidden raw values are replaced before embedding and cannot become keys/values. A separate one-layer, width-32 Transformer plus linear 32→32 output maps the complete contextualized masked-branch sequence to latent predictions. It receives the masked-position contextualized vectors directly, without replacing them with fresh predictor mask tokens or adding positional encodings twice. Predictor keys/values are visible positions only; masked positions remain queries. The original-sequence embeddings never enter the predictor.

Loss is `masked_latent_MSE + {cfg['sigreg_weight']} * SIGReg(original_embeddings)`. Gradients flow through predictions, predictor, masked encoder branch **and original target branch**. No EMA, detach/stop-gradient, projector, sample-level penalty, or auxiliary objective. The waveform head is removed; encoder capacity stays identical. This model has 54,560 parameters including 13,824 predictor/output parameters, versus 56,411 for the waveform model; counts are also saved in the audit. Frozen features use the original/unmasked encoder only; predictor is discarded for diagnosis evaluation.

## SIGReg definition and scope

No existing implementation was found locally. The implementation follows the sliced Epps–Pulley statistic in the [official LeJEPA minimal example](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md) and [reference quadrature](https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/univariate/epps_pulley.py). This is an independent small implementation of that mathematical statistic; no package dependency or vendored upstream code is introduced.

For each of the 160 positions separately, B original patch embeddings are projected onto {cfg['sigreg_slices']} freshly sampled unit Gaussian directions. Empirical cosine/sine means across **B**, not across time, are compared to exp(-t²/2). Use 17 knots on [0,3], Gaussian weighting exp(-t²/2), trapezoidal endpoints, doubled weights for the negative half, and the reference statistic's factor B. Average over directions and positions. Do not center, standardize, or whiten the embeddings before SIGReg. This preserves penalties for shifted/scaled/non-Gaussian populations. Positions therefore cannot supply spurious diversity for a position-only collapsed encoder. Compared with the reference minimal example's 256 slices, 64 reduces compute in this initial patch-level baseline. The user's unweighted latent term plus λ-weighted regularizer is used, rather than a `(1-λ)` coefficient on prediction.

Patch positions are correlated and windows can share participants, so the independence assumptions of an ideal Gaussian goodness-of-fit test do not hold literally. Batch sampling balances participants. The existing final LayerNorm also constrains each embedding to an affine hyperplane (rank at most 31 of 32) and fixes pre-affine norm: exact full-dimensional Gaussianity is unattainable. It is intentionally retained to match the waveform encoder. SIGReg is an approximate anti-collapse pressure here, not a guarantee of exact Gaussian features or useful pooled representations.

## Data, budget and selection

All 4,894 usable eight-second windows from 52 training participants; same source hashes, preprocessing/QC, original training-only channel scaling and participant-balanced sampling as the full-window factorial study. Patch size 25 samples/50 ms, 25% hidden patches across all channels, span one patch (adjacent masks can merge). Same 10,000 updates × batch 64, AdamW .001, 500-step warmup/cosine decay to .00001, weight decay .01, clipping 1.0. Sampling and masking streams are paired with `{cfg['raw_reference']}`. Every allowed training window was visited ({audit['visited_windows']}), with {audit['draws']:,} total draws. Predictor/two branches and SIGReg increase compute, so this matches updates/examples rather than wall time.

Validation uses the original 18 participants/432 windows, fixed masks, and a fixed shuffled window order to mix participants within batches. SIGReg directions are fixed across validation calls via a separate RNG, including the final smaller batch. Combined loss is sample-weight averaged across batches. Select minimum **combined validation objective** among trained checkpoints, not latent MSE alone (a collapsed representation can trivially lower prediction error). Selected update: **{selected['step']}**. No labels or test features choose encoder checkpoints.

The training loop logs mean online latent MSE, SIGReg and combined loss each 500 steps. At the same intervals, fixed validation monitoring measures original and masked embedding variance/effective rank at patch and pooled levels. Patch statistics compute covariance across samples independently at each position, then average; pooled statistics compute covariance across mean-pooled windows. Effective rank is exp(entropy(normalized covariance eigenvalues)); zero covariance gives zero rank. These are descriptive collapse checks, not extra losses. The fixed first 64 shuffled validation windows are used throughout, avoiding a moving diagnostic population. Participant-level frozen features average the same 24 four-second observations as the waveform baseline, with original training-only scaling and fixed C=1 logistic regression. As in the waveform baseline, four-second unmasked feature extraction differs from eight-second SSL training; that shared observation protocol is retained rather than silently changed.

## Initial results

This run did **not** improve frozen diagnosis classification over its paired waveform or untrained controls. It avoided a constant embedding across all samples, but retained low effective dimensionality and nearly uniform representations across positions within each window. At the selected checkpoint, within-window variance fell from about 0.536 initially to 0.00266, while pooled variance rose from 0.00383 to 0.647. This is consistent with a window-level code broadcast across patch positions. It is a diagnostic interpretation, not proof of an exact mechanism.

SIGReg applied independently at each position constrains between-window distributions; it does not require different patches in the same window to contain different information. Thus temporal homogenization remains an admissible shortcut even without total across-sample collapse. The final checkpoint's poor classification further shows that low latent MSE or falling combined loss is not a surrogate for diagnostic utility here. No extra regularizer or objective was added in response to these results.

{markdown(table[['model','split','balanced_accuracy','macro_f1']])}

At the selected checkpoint, original patch variance is **{diagnostics.validation_patch_variance:.6f}**, patch effective rank **{diagnostics.validation_patch_effective_rank:.3f}**, pooled variance **{diagnostics.validation_pooled_variance:.6f}**, pooled effective rank **{diagnostics.validation_pooled_effective_rank:.3f}**. See the full traces rather than interpreting a low latent loss as success. `untrained` is this model's exactly paired initial encoder; `waveform` is the same seed/configuration's existing full-data/25% masked raw decoder. Raw-voltage and latent losses have different targets and units and are not directly ranked.

The same fixed 64-window validation panel also gives the following encoder-only diagnostics. Within-window variance is an extra **diagnostic**, not an added regularizer: small temporal variation combined with substantial pooled variance can reveal a window-level code broadcast across positions.

{markdown(diagnostic_table)}

![Objective and collapse diagnostics](eeg_latent_assets/diagnostics.png)

This is one exploratory seed on the already inspected 18-person test set. Nonzero patch variance does not guarantee useful sample-level features. No sample-level regularizer is added to repair pooled collapse in this baseline. The final checkpoint is also evaluated, so regression or a misleading minimum is visible. Untrained and waveform controls use matched access and observations. No spectral probe, nonlinear readout, or optimization sweep was added.

## Reproduce

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 \\
  .venv-gpu/bin/python -m temporal_ssl.eeg_latent_experiment --config configs/eeg_latent.json
.venv-gpu/bin/python -m pytest -q tests/test_eeg_latent.py
```

Checkpoints retain optimizer and independent sampling, masking, SIGReg, Torch and CUDA RNG states. Re-running resumes the identical protocol; config mismatches stop explicitly. Exact validation masks/order, untrained/best/final states, histories, frozen features, predictions, fitted classifiers and audit/source hashes are in `{root}`. [Objective/diagnostics CSV](eeg_latent_assets/history.csv), [classification CSV](eeg_latent_assets/comparison.csv), [training audit](eeg_latent_assets/training_audit.json).
'''
    assets=Path('reports/eeg_latent_assets');assets.mkdir(parents=True,exist_ok=True)
    h.to_csv(root/'history.csv',index=False)
    for file in ['diagnostics.png','history.csv','comparison.csv','training_audit.json','control_diagnostics.json']:
        shutil.copyfile(root/file,assets/file)
    Path('reports/eeg_latent.md').write_text(text);(root/'report.md').write_text(text.replace('eeg_latent_assets/',''))


def run(config):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=cfg:raise ValueError('Config mismatch')
    save_json(root/'protocol.json',cfg)
    if (root/'complete.json').exists():report(root,cfg);return
    save_json(root/'source_hashes.json',{str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')})
    pilot=Path(cfg['pilot']);model_cfg=json.loads((pilot/'config.json').read_text())
    model_cfg['model'].update(mask_ratio=cfg['mask_ratio'],mask_span=cfg['mask_span'])
    data,val,split,capped=base.prepare(cfg,pilot,root);norm=data.normalizer
    gpu=torch.from_numpy((data.x-norm.mean)/norm.std).cuda();vgpu=torch.from_numpy((val.x-norm.mean)/norm.std).cuda()
    seed_everything(cfg['seed']);model=make_model(cfg,model_cfg).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    sampling=torch.Generator().manual_seed(cfg['seed']+100);masking=torch.Generator().manual_seed(cfg['seed']+200)
    regularizing=torch.Generator().manual_seed(cfg['seed']+300)
    table,counts=base.sampling_table(data.metadata,np.arange(len(data)))
    masks=base.fixed_masks(len(val),cfg['mask_ratio'],cfg['evaluation_seed'],cfg['mask_span'])
    permutation=torch.randperm(len(val),generator=torch.Generator().manual_seed(cfg['evaluation_seed']+20))
    np.savez_compressed(root/'validation_protocol.npz',mask=masks.numpy(),order=permutation.numpy())
    start,best,history,coverage=0,float('inf'),[],np.zeros(len(data),dtype=np.int64)
    if (root/'last.pt').exists():
        state=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for k,v in values.items():
                if torch.is_tensor(v):values[k]=v.cuda()
        for generator,key in [(sampling,'sampling'),(masking,'masking'),(regularizing,'regularizing')]:generator.set_state(state[key])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        start,best,history,coverage=state['step'],state['best'],state['history'],state['coverage']
    else:
        torch.save(model.state_dict(),root/'untrained.pt')
        metrics=validate(model,vgpu,masks,permutation,cfg)
        history=[{'step':0,**{'validation_'+k:v for k,v in metrics.items()}}]
    sums={key:0. for key in ['loss','latent_loss','sigreg_loss']};n=0;begin=time.perf_counter()
    for step in range(start+1,cfg['steps']+1):
        model.train();ix=base.sample_indices(table,counts,cfg['batch_size'],sampling)
        np.add.at(coverage,ix.numpy(),1)
        mask=base.random_mask(len(ix),160,cfg['mask_ratio'],masking,cfg['mask_span']).cuda()
        for group in optimizer.param_groups:group['lr']=base.rate(step,cfg)
        optimizer.zero_grad(set_to_none=True)
        result=model(gpu[ix.cuda()],mask,regularizing)
        if not torch.isfinite(result['loss']):raise RuntimeError('Nonfinite latent objective')
        result['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        for key in sums:sums[key]+=float(result[key].detach())
        n+=1
        if step%cfg['validation_every']==0 or step==cfg['steps']:
            metrics=validate(model,vgpu,masks,permutation,cfg)
            history.append({'step':step,**{'train_'+k:v/n for k,v in sums.items()},
                **{'validation_'+k:v for k,v in metrics.items()},
                **{'train_'+k:v for k,v in collapse_statistics(result['original'].detach()).items()}})
            if metrics['loss']<best:
                best=metrics['loss'];torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},root/'best.pt')
            state={'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'best':best,'history':history,'coverage':coverage,
                'sampling':sampling.get_state(),'masking':masking.get_state(),'regularizing':regularizing.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,root/'last.pt.tmp');os.replace(root/'last.pt.tmp',root/'last.pt');save_json(root/'history.json',history)
            print(f'step {step}: latent={sums["latent_loss"]/n:.5f} SIGReg={sums["sigreg_loss"]/n:.5f} val_total={metrics["loss"]:.5f} patch_rank={metrics["patch_effective_rank"]:.2f} pooled_rank={metrics["pooled_effective_rank"]:.2f} elapsed={time.perf_counter()-begin:.1f}s',flush=True)
            sums={key:0. for key in sums};n=0
    assert (coverage>0).all()
    reference=Path(cfg['raw_reference']);raw_state=torch.load(reference/'last.pt',map_location='cpu',weights_only=False)
    np.testing.assert_array_equal(coverage,raw_state['coverage']);torch.testing.assert_close(sampling.get_state(),raw_state['sampling'],rtol=0,atol=0)
    initial=torch.load(root/'untrained.pt',map_location='cpu',weights_only=True);raw_initial=torch.load(reference/'untrained.pt',map_location='cpu',weights_only=True)
    for key,value in initial.items():
        if key.startswith(('embedding.','encoder.')):torch.testing.assert_close(value,raw_initial[key],rtol=0,atol=0)
    save_json(root/'training_audit.json',{'visited_windows':int((coverage>0).sum()),'draws':int(coverage.sum()),'updates':cfg['steps'],
        'parameters':sum(p.numel() for p in model.parameters()),'predictor_parameters':sum(p.numel() for p in model.predictor.parameters())+sum(p.numel() for p in model.prediction_head.parameters()),
        'initial_encoder_matches_waveform':True,'sampling_and_coverage_match_waveform':True,
        'raw_reference_best_sha256':sha256(reference/'best.pt'),'latent_training_seconds':time.perf_counter()-begin})
    selected=min(history[1:],key=lambda r:r['validation_loss']);save_json(root/'selected.json',{'step':selected['step'],'validation_loss':selected['validation_loss']})
    # Test EEG enters evaluation only after pretraining; train/validation probe panels unchanged.
    sets={'train':EEGDataset(split['train'],norm),'validation':val,'test':EEGDataset(split['test'],norm)}
    for name,file in [('untrained','untrained.pt'),('latent','best.pt')]:
        model.load_state_dict(torch.load(root/file,map_location='cuda',weights_only=True));model.eval()
        features={key:participant_features(model,d,cfg['batch_size']) for key,d in sets.items()}
        classify(features,model_cfg,root/name)
    model.load_state_dict(torch.load(root/'last.pt',map_location='cpu',weights_only=False)['model'])
    classify({k:participant_features(model,d,cfg['batch_size']) for k,d in sets.items()},model_cfg,root/'latent_final')
    shutil.copytree(reference/'classification',root/'waveform',dirs_exist_ok=True)
    # Exact initialization also implies exact original untrained feature/control scores.
    previous=json.loads((reference.parents[1]/'untrained'/f'seed_{cfg["seed"]}'/'classification/classification.json').read_text())
    assert previous==json.loads((root/'untrained/classification.json').read_text())
    report(root,cfg);save_json(root/'complete.json',{'selected_step':selected['step'],'updates':cfg['steps']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/eeg_latent.json');run(p.parse_args().config)
if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');main()
