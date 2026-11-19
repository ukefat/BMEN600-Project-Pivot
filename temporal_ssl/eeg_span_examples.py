"""Deterministically selected held-out examples from the completed masked-span study."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import Normalizer
from .eeg_data import EEGDataset
from .eeg_factorial import fixed_masks
from .eeg_pilot import make_eeg_model
from .baselines import fill_masked
from .experiment import seed_everything
from .plots import plt,finish


def run(root,seed=600):
    root=Path(root);cfg=json.loads((root/'protocol.json').read_text());pilot=Path(cfg['pilot'])
    model_cfg=json.loads((pilot/'config.json').read_text());split=json.loads((pilot/'participants.json').read_text())
    with np.load(pilot/'normalization.npz') as a:norm=Normalizer(a['mean'],a['std'])
    data=EEGDataset(split['test'],norm)
    indices=[next(i for i,m in enumerate(data.metadata) if m['group']==group) for group in ['AD','FTD','HC']]
    x=torch.stack([data[i]['x'] for i in indices])
    output=root/'reconstruction_examples';output.mkdir(exist_ok=True)
    assets=Path('reports/eeg_span_reconstruction_assets');assets.mkdir(parents=True,exist_ok=True)
    rows=[]
    seed_everything(seed)
    for ratio in [.25,.5]:
        figures={view:plt.subplots(3,2,figsize=(14,9),sharex=True) for view in ['four_seconds','zoom']}
        for column,span in enumerate([100,200]):
            masks=fixed_masks(len(data),ratio,cfg['evaluation_seed'],span//50)[indices]
            model=make_eeg_model(model_cfg,'masked_short',(4000,19)).cuda().eval()
            path=root/f'span{span}'/f'full_mask{int(ratio*100)}'/f'seed_{seed}'
            model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True))
            with torch.no_grad():pred=model(x.cuda(),mask=masks.cuda())['prediction'].cpu().numpy().reshape(3,160,25,19)
            for row,index in enumerate(indices):
                target=x[row].numpy();mask=masks[row].numpy();metadata=data.metadata[index]
                interpolation=fill_masked(target.reshape(160,25,19),mask).reshape(4000,19)
                prediction=pred[row].reshape(4000,19);hidden=np.repeat(mask,25)
                mse=float(((prediction[hidden]-target[hidden])**2).mean())
                saved=pd.read_csv(path/f'test_reconstruction_{ratio}.csv')
                expected=saved[(saved.participant_id==metadata['participant_id'])&(saved.epoch_index==metadata['epoch_index'])].mse.iloc[0]
                np.testing.assert_allclose(mse,expected,rtol=1e-5)
                filename=f'span{span}_mask{int(ratio*100)}_{metadata["participant_id"]}_seed{seed}.npz'
                np.savez_compressed(output/filename,x=target,prediction=prediction,interpolation=interpolation,mask=mask,**metadata)
                rows.append({'span_ms':span,'ratio':ratio,'seed':seed,**metadata,'masked_mse':mse,
                    'interpolation_mse':float(((interpolation[hidden]-target[hidden])**2).mean())})
                for view,(fig,axes) in figures.items():
                    ax=axes[row,column];t=np.arange(4000)/500
                    neural=prediction[:,0].copy();linear=interpolation[:,0].copy()
                    neural[~hidden]=np.nan;linear[~hidden]=np.nan
                    ax.plot(t,target[:,0],color='black',lw=.9,label='Target EEG')
                    ax.plot(t,neural,color='#d95f02',lw=1.6,label='Model, hidden samples')
                    ax.plot(t,linear,color='#1b9e77',ls='--',lw=1.2,label='Interpolation')
                    edges=np.diff(np.r_[False,mask,False].astype(int))
                    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                        ax.axvspan(a*.05,b*.05,color='gray',alpha=.12,lw=0)
                    ax.set(xlim=(0,4 if view=='four_seconds' else 2),ylabel='Normalized Fp1',
                        title=f'{metadata["participant_id"]} ({metadata["group"]}), {span} ms; masked MSE {mse:.3f}')
                    if row==2:ax.set_xlabel('Seconds')
                    if row==0 and column==0:ax.legend(fontsize=8,loc='best')
        for view,(fig,axes) in figures.items():
            fig.suptitle(f'{int(ratio*100)}% masked; requested spans 100 vs 200 ms; paired seed {seed}\nGray: hidden intervals. Same first test participant/window per diagnosis; model input is full 8 s.',fontsize=12)
            filename=f'mask{int(ratio*100)}_{view}.png'
            finish(fig,assets/filename)
    pd.DataFrame(rows).to_csv(assets/'examples.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'examples.csv',index=False)
    text=f'''# Held-out EEG span reconstructions

Validation-selected checkpoints from the completed full-window span study, seed {seed}. These are **the first test participant and first cached window in each diagnosis**, selected deterministically without inspecting errors. Every model receives the full eight-second window with its original fixed evaluation masks. Panels display Fp1 over the first four seconds, with separate first-two-second enlargements. Gray intervals are hidden across all 19 channels. Orange predictions and green interpolation are drawn only inside hidden regions; black is the target. Units are the original fixed training-channel normalization, not per-patch normalization. Titles report masked MSE across all 19 channels and all eight seconds, checked against saved study scores.

## 25% masking

![25% masked examples](eeg_span_reconstruction_assets/mask25_four_seconds.png)

[Enlarge first two seconds](eeg_span_reconstruction_assets/mask25_zoom.png).

## 50% masking

![50% masked examples](eeg_span_reconstruction_assets/mask50_four_seconds.png)

[Enlarge first two seconds](eeg_span_reconstruction_assets/mask50_zoom.png).

Columns vary requested span; actual gaps can merge or truncate. Masks are therefore different across span conditions. The same subject/windows and seed are used throughout, but these panels compare different missing samples. Visual examples are not population performance estimates. [All example metrics](eeg_span_reconstruction_assets/examples.csv). Exact inputs, predictions, interpolation, masks and source identities are in `{output}`. No models were trained or modified.

Reproduce: `.venv-gpu/bin/python -m temporal_ssl.eeg_span_examples --seed {seed}`.
'''
    Path('reports/eeg_span_reconstructions.md').write_text(text)
    print(pd.DataFrame(rows)[['span_ms','ratio','participant_id','group','masked_mse','interpolation_mse']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='results/eeg_span_study');p.add_argument('--seed',type=int,default=600)
    a=p.parse_args();run(a.root,a.seed)
