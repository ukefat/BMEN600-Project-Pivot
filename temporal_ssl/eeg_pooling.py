"""Frozen patch pooling ablation on all centralized three-class waveform models."""
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import Normalizer
from .eeg_data import EEGDataset,aggregate_participants
from .eeg_pilot import make_eeg_model,classify
from .eeg_reconstruction_study import CausalRatioPrediction
from .eeg_course.download import sha256
from .experiment import loader,seed_everything,save_json
from .plots import plt,finish


def pool(local):
    mean=local.mean(1);std=local.std(1,unbiased=False)
    return {'mean':mean,'std':std,'mean_std':torch.cat([mean,std],-1)}


@torch.no_grad()
def features(model,dataset):
    model.eval();values={k:[] for k in ['mean','std','mean_std']}
    for batch in loader(dataset,64):
        local=model.representations(batch['x'][:,:2000].cuda())['local']
        for key,value in pool(local).items():values[key].append(value.cpu().numpy())
    return {key:aggregate_participants(np.concatenate(v),dataset.metadata) for key,v in values.items()}


def run():
    source=Path('results/eeg_reconstruction_extension');root=Path('results/eeg_pooling');root.mkdir(exist_ok=True)
    pilot=Path('results/eeg_pilot');cfg=json.loads((pilot/'config.json').read_text());split=json.loads((pilot/'participants.json').read_text())
    with np.load(pilot/'normalization.npz') as a:norm=Normalizer(a['mean'],a['std'])
    datasets={k:EEGDataset(rows,norm) for k,rows in split.items()}
    entries=json.loads((source/'checkpoint_provenance.json').read_text())
    for kind in ['masked','causal']:
        for seed in [600,601,602]:entries.append({'model':'untrained_'+kind,'seed':seed,'checkpoint':None})
    old=pd.read_csv(source/'classification.csv');rows=[];predictions=[]
    for entry in entries:
        name,seed=entry['model'],entry['seed'];seed_everything(seed)
        if name.startswith('causal_context') or name=='untrained_causal':
            context=int(name.removeprefix('causal_context')) if name!='untrained_causal' else 80
            model=CausalRatioPrediction(19,4000,context_patches=context,**cfg['model']).cuda()
        else:model=make_eeg_model(cfg,'masked_short',(4000,19)).cuda()
        if entry['checkpoint'] is not None:
            assert sha256(entry['checkpoint'])==entry['sha256']
            model.load_state_dict(torch.load(entry['checkpoint'],map_location='cuda',weights_only=True))
            previous=Path(entry['checkpoint']).parent/'classification/participant_features.npz'
        elif name=='untrained_masked':previous=Path(f'results/eeg_factorial/untrained/seed_{seed}/classification/participant_features.npz')
        else:previous=Path(f'results/eeg_reconstruction_study/context80/seed_{seed}/untrained_classification/participant_features.npz')
        path=root/name/f'seed_{seed}';path.mkdir(parents=True,exist_ok=True)
        if all((path/key/'classification.json').exists() for key in ['mean','std','mean_std']):
            extracted=None
        else:
            extracted={part:features(model,data) for part,data in datasets.items()}
            with np.load(previous) as a:
                for part in datasets:
                    np.testing.assert_array_equal(extracted[part]['mean'][0],a[part+'_ids'])
                    np.testing.assert_allclose(extracted[part]['mean'][1],a[part+'_x'],rtol=1e-5,atol=2e-6)
        for pooling in ['mean','std','mean_std']:
            if not (path/pooling/'classification.json').exists():
                score=classify({part:extracted[part][pooling] for part in datasets},cfg,path/pooling)
            else:score=json.loads((path/pooling/'classification.json').read_text())
            if pooling=='mean':
                for part in ['validation','test']:
                    expected=old[(old.model==name)&(old.seed==seed)&(old.split==part)].iloc[0]
                    np.testing.assert_allclose(score[part]['balanced_accuracy'],expected.balanced_accuracy)
                    np.testing.assert_allclose(score[part]['macro_f1'],expected.macro_f1)
            for part in ['validation','test']:rows.append({'model':name,'seed':seed,'pooling':pooling,'split':part,**score[part]})
            f=pd.read_csv(path/pooling/'predictions.csv');f['model']=name;f['seed']=seed;f['pooling']=pooling;predictions.append(f)
        pd.DataFrame(rows).to_csv(root/'scores.csv',index=False)
        print(name,seed, {p:json.loads((path/p/'classification.json').read_text())['test']['balanced_accuracy'] for p in ['mean','std','mean_std']},flush=True)
    f=pd.DataFrame(rows);pred=pd.concat(predictions,ignore_index=True);pred.to_csv(root/'predictions.csv',index=False)
    summary=f[f.split=='test'].groupby(['model','pooling']).agg(balanced_accuracy=('balanced_accuracy','mean'),ba_seed_sd=('balanced_accuracy','std'),macro_f1=('macro_f1','mean')).reset_index()
    summary.to_csv(root/'summary.csv',index=False)
    wide=summary.pivot(index='model',columns='pooling',values='balanced_accuracy').reset_index();wide['mean_std_minus_mean']=wide.mean_std-wide['mean'];wide.to_csv(root/'comparison.csv',index=False)
    changes=[]
    for model,frame in f[f.split=='test'].groupby('model'):
        w=frame.pivot(index='seed',columns='pooling',values='balanced_accuracy')
        for seed,value in (w.mean_std-w['mean']).items():changes.append({'model':model,'seed':seed,'difference':value})
    pd.DataFrame(changes).to_csv(root/'paired_changes.csv',index=False)
    # Paired participant uncertainty conditional on the trained models and split.
    rng=np.random.default_rng(5000);uncertainty=[]
    for model in wide.model:
        cubes=[];ids=None;truth=None
        for pooling in ['mean','mean_std']:
            correct=[]
            for seed in [600,601,602]:
                q=pred[(pred.model==model)&(pred.pooling==pooling)&(pred.seed==seed)&(pred.split=='test')].sort_values('participant_id')
                if ids is None:ids=q.participant_id.to_numpy();truth=q.truth.to_numpy()
                np.testing.assert_array_equal(ids,q.participant_id);np.testing.assert_array_equal(truth,q.truth)
                correct.append((q.prediction.to_numpy()==truth).astype(float))
            cubes.append(correct)
        cube=np.asarray(cubes);draws=np.zeros((2,5000))
        for group in ['AD','FTD','HC']:
            ix=np.flatnonzero(truth==group);sample=rng.choice(ix,(5000,len(ix)),replace=True)
            draws+=cube[:,:,sample].mean(axis=(1,3))/3
        low,high=np.quantile(draws[1]-draws[0],[.025,.975]);uncertainty.append({'model':model,'lower_95':low,'upper_95':high})
    pd.DataFrame(uncertainty).to_csv(root/'participant_uncertainty.csv',index=False)
    fig,ax=plt.subplots(figsize=(14,6));x=np.arange(len(wide));width=.25
    for offset,pooling in zip([-1,0,1],['mean','std','mean_std']):ax.bar(x+offset*width,wide[pooling],width,label=pooling)
    ax.axhline(1/3,color='gray',ls=':');ax.axhline(.6603174603,color='black',ls='--',label='Spectral baseline')
    ax.set(xticks=x,xticklabels=wide.model,ylabel='Three-class test balanced accuracy',ylim=(0,1));ax.tick_params(axis='x',labelrotation=45);ax.legend();finish(fig,root/'comparison.png')
    def table(frame):
        lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
        for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    text=f'''# Frozen EEG patch pooling ablation

No SSL models were retrained or selected again. All 36 trained waveform checkpoints (nine masked tasks and three causal tasks × three seeds), plus three untrained seeds for each attention rule, are evaluated on the same three-class AD/FTD/HC split. Same 52 training, 18 validation and 18 test participants, 24 windows per participant, same first-four-second unmasked EEG, original channel scaling, and frozen encoders. The existing handcrafted spectral classifier remains 66.0% balanced accuracy and is not recomputed with a different aggregation.

For each window, contextualized embeddings form an 80-patch × 32-feature matrix. Compare **mean** (32 features), **std** (32 features), and concatenated **mean + std** (64 features). Standard deviation is computed across patches independently for each embedding coordinate, with population denominator N (`unbiased=False`). Each window vector is then averaged over the same 24 windows per participant. This is the mean of within-window patch standard deviations, not a pooled standard deviation across a participant's recording; variation between windows is not added. These are embedding statistics, not raw-voltage features.

Every readout fits its own StandardScaler on training participants only and the identical fixed C=1 logistic regression, without tuning or test-based threshold/checkpoint selection. Mean+std doubles feature dimension and classifier coefficients; std-only is included as a same-dimensional diagnostic. A gain would reflect readout/pooling changes, not an improvement in the frozen encoder. C remains fixed, so effective regularization can change with the feature set. The original mean features and scores are checked against the saved study artifacts.

## Test balanced accuracy

Averages across three paired seeds. Full macro-F1 and seed SDs are in the summary CSV. Positive differences favor mean+std.

{table(wide)}

![Pooling comparison](eeg_pooling_assets/comparison.png)

[Detailed scores](eeg_pooling_assets/scores.csv), [summary with seed SD and macro-F1](eeg_pooling_assets/summary.csv), [paired seed changes](eeg_pooling_assets/paired_changes.csv), and [participant uncertainty](eeg_pooling_assets/participant_uncertainty.csv). The latter gives paired 95% percentile intervals from 5,000 diagnosis-stratified participant resamples, conditional on the fixed trained models/split. They are not training-cohort uncertainty or multiple-comparison-adjusted confirmation.

The approximately-2:1 causal model shows the largest point-estimate gain: 41.2% mean, 48.9% std-only, and 49.3% mean+std. Its conditional paired participant interval for mean+std minus mean is approximately +0.6 to +15.8 percentage points. This is exploratory and unadjusted across fourteen model/control comparisons, so it is not independent confirmation. The 3:1 causal model changes from 45.0% to 43.9%, and masked effects range from gains of about 4.7 points to losses of about 7.0 points. Pooling therefore affects individual results but is not a consistent solution. The 66.0% spectral reference remains stronger than every pooled learned mean result in this study.

One repeatedly inspected 18-person test set makes this exploratory. Standard deviation can include position/context effects as well as EEG variation; it does not establish direct preservation of spectral information. Untrained controls help distinguish generic embedding information from SSL gains. Mask ratios/spans describe pretraining only; classifier inputs are unmasked and identical. All original experiments and checkpoints are preserved.

Reproduce/resume: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m temporal_ssl.eeg_pooling`.
'''
    import shutil
    assets=Path('reports/eeg_pooling_assets');assets.mkdir(exist_ok=True)
    for file in ['scores.csv','summary.csv','comparison.csv','paired_changes.csv','participant_uncertainty.csv','comparison.png']:shutil.copyfile(root/file,assets/file)
    Path('reports/eeg_pooling.md').write_text(text);(root/'report.md').write_text(text.replace('eeg_pooling_assets/',''))
    save_json(root/'complete.json',{'encoders':42,'pooling_methods':3,'classifier_fits':126,'SSL_retrained':False,'mean_scores_match_original':True})
    print(wide.to_string(index=False))


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');run()
