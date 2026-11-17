"""Matched reconstruction tasks and paired participant uncertainty for the span study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import eeg_factorial as study
from .data import Normalizer
from .eeg_data import EEGDataset
from .eeg_pilot import make_eeg_model
from .experiment import seed_everything


def analyze(root,cfg):
    root=Path(root);pilot=Path(cfg['pilot'])
    sources={50:Path(cfg['reference']),100:root/'span100',200:root/'span200'}
    rng=np.random.default_rng(4700);rows=[]
    for ratio in [25,50]:
        cubes=[];truth=None;ids=None
        for span,path in sources.items():
            correct=[]
            for seed in cfg['seeds']:
                f=pd.read_csv(path/f'full_mask{ratio}'/f'seed_{seed}'/'classification/predictions.csv')
                f=f[f.split=='test'].sort_values('participant_id')
                if truth is None:truth=f.truth.to_numpy();ids=f.participant_id.to_numpy()
                np.testing.assert_array_equal(ids,f.participant_id);np.testing.assert_array_equal(truth,f.truth)
                correct.append((f.prediction==f.truth).to_numpy(dtype=float))
            cubes.append(np.stack(correct))
        cube=np.stack(cubes);draws=np.zeros((3,5000));points=np.zeros(3)
        for group in ['AD','FTD','HC']:
            ix=np.flatnonzero(truth==group);sample=rng.choice(ix,(5000,len(ix)),replace=True)
            draws+=cube[:,:,sample].mean(axis=(1,3))/3
            points+=cube[:,:,ix].mean(axis=(1,2))/3
        for index,span in [(1,100),(2,200)]:
            lo,hi=np.quantile(draws[index]-draws[0],[.025,.975])
            rows.append({'ratio':ratio,'span_ms':span,'difference_vs_50ms':points[index]-points[0],
                         'lower_95':lo,'upper_95':hi})
    pd.DataFrame(rows).to_csv(root/'participant_bootstrap.csv',index=False)
    with np.load(pilot/'normalization.npz') as a:norm=Normalizer(a['mean'],a['std'])
    split=json.loads((pilot/'participants.json').read_text());test=EEGDataset(split['test'],norm)
    gpu=torch.from_numpy((test.x-norm.mean)/norm.std).cuda()
    masks={(span,ratio):study.fixed_masks(len(test),ratio,cfg['evaluation_seed'],span//50) for span in [50,200] for ratio in cfg['ratios']}
    baseline=[]
    for (span,ratio),mask in masks.items():baseline.append({'evaluation_span_ms':span,'evaluation_ratio':ratio,**study.baselines(test,mask)})
    pd.DataFrame(baseline).to_csv(root/'common_reconstruction_baselines.csv',index=False)
    model_cfg=json.loads((pilot/'config.json').read_text());result=[]
    for span,path in sources.items():
        for ratio in cfg['ratios']:
            for seed in cfg['seeds']:
                seed_everything(seed);model=make_eeg_model(model_cfg,'masked_short',(4000,19)).cuda().eval()
                model.load_state_dict(torch.load(path/f'full_mask{int(ratio*100)}'/f'seed_{seed}'/'best.pt',map_location='cuda',weights_only=True))
                for (evaluation_span,evaluation_ratio),mask in masks.items():
                    error,rms=study.score(model,gpu,mask)
                    result.append({'training_span_ms':span,'training_ratio':ratio,'seed':seed,
                        'evaluation_span_ms':evaluation_span,'evaluation_ratio':evaluation_ratio,
                        'mse':study.participant_mean(error,test.metadata),'prediction_rms':rms})
    pd.DataFrame(result).to_csv(root/'common_reconstruction.csv',index=False)
    del gpu
