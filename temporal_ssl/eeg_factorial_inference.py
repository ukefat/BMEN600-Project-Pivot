"""Paired participant bootstrap conditional on the fixed trained EEG models and split."""
from pathlib import Path
import numpy as np
import pandas as pd


def bootstrap(output,seeds,replicates=5000,seed=4600):
    output=Path(output)
    names=['untrained','capped_mask25','capped_mask50','full_mask25','full_mask50','spectral']
    cubes=[]
    identifiers,truth=None,None
    for name in names:
        correct=[]
        for model_seed in seeds:
            path=(output/'spectral_control/predictions.csv' if name=='spectral' else
                  output/name/f'seed_{model_seed}'/'classification/predictions.csv')
            frame=pd.read_csv(path).query("split == 'test'").sort_values('participant_id')
            if identifiers is None:
                identifiers=frame.participant_id.to_numpy();truth=frame.truth.to_numpy()
            np.testing.assert_array_equal(identifiers,frame.participant_id)
            np.testing.assert_array_equal(truth,frame.truth)
            correct.append((frame.prediction.to_numpy()==truth).astype(float))
        cubes.append(np.stack(correct))
    cube=np.stack(cubes)  # condition × model seed × independent participant
    rng=np.random.default_rng(seed)
    simulated=np.zeros((len(names),replicates))
    points=np.zeros(len(names))
    for group in ('AD','FTD','HC'):
        indices=np.flatnonzero(truth==group)
        selected=rng.choice(indices,size=(replicates,len(indices)),replace=True)
        simulated+=cube[:,:,selected].mean(axis=(1,3))/3
        points+=cube[:,:,indices].mean(axis=(1,2))/3
    contrasts={
        'data_at_25':np.array([0,-1,0,1,0,0]),
        'data_at_50':np.array([0,0,-1,0,1,0]),
        'masking_capped':np.array([0,-1,1,0,0,0]),
        'masking_full':np.array([0,0,0,-1,1,0]),
        'interaction':np.array([0,1,-1,-1,1,0]),
        'full50_minus_spectral':np.array([0,0,0,0,1,-1])}
    rows=[]
    for name,weights in contrasts.items():
        samples=weights@simulated
        lower,upper=np.quantile(samples,[.025,.975])
        rows.append({'contrast':name,'difference':float(weights@points),'lower_95':float(lower),
                     'upper_95':float(upper),'replicates':replicates})
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'participant_bootstrap.csv',index=False)
    return frame
