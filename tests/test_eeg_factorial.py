import numpy as np
import pytest
import torch
pytest.importorskip('sklearn')
from temporal_ssl.eeg_factorial import random_mask,sampling_table,sample_indices,rate
from temporal_ssl.models import TemporalSSL


def test_uniform_masks_and_exclusion():
    mask=random_mask(5000,160,.5,torch.Generator().manual_seed(2))
    assert (mask.sum(1)==80).all()
    assert (mask.float().mean(0)-.5).abs().max()<.035
    low=random_mask(5000,160,.25,torch.Generator().manual_seed(2))
    assert not (low&~mask).any()
    model=TemporalSSL(3,40,objective='masked',patch_size=5,d_model=8,nhead=2,layers=1).eval()
    x=torch.randn(2,40,3);hidden=random_mask(2,8,.5,torch.Generator().manual_seed(7))
    changed=x.reshape(2,8,5,3).clone();changed[hidden]=10000
    torch.testing.assert_close(model(x,mask=hidden)['prediction'],model(changed.reshape_as(x),mask=hidden)['prediction'])


def test_participant_balancing_and_allowed_windows():
    metadata=[{'participant_id':'a'}]*2+[{'participant_id':'b'}]*8
    table,counts=sampling_table(metadata,np.arange(10))
    generator=torch.Generator().manual_seed(4)
    indices=sample_indices(table,counts,50000,generator).numpy()
    assert abs(np.mean(indices<2)-.5)<.015
    assert set(indices)==set(range(10))
    table,counts=sampling_table(metadata,np.array([0,2]))
    assert set(sample_indices(table,counts,1000,generator).tolist())=={0,2}
    with pytest.raises(ValueError): sampling_table(metadata,np.array([0]))


def test_sampler_resume_and_schedule():
    table,counts=sampling_table([{'participant_id':str(i//3)} for i in range(9)],np.arange(9))
    g=torch.Generator().manual_seed(1)
    sample_indices(table,counts,64,g);state=g.get_state()
    expected=sample_indices(table,counts,64,g)
    restored=torch.Generator();restored.set_state(state)
    torch.testing.assert_close(expected,sample_indices(table,counts,64,restored))
    g=torch.Generator().manual_seed(3);random_mask(64,160,.5,g);state=g.get_state()
    expected=random_mask(64,160,.5,g);restored.set_state(state)
    torch.testing.assert_close(expected,random_mask(64,160,.5,restored))
    cfg={'steps':10000,'warmup_steps':500,'lr':.001,'min_lr':.00001}
    assert rate(1,cfg)<rate(500,cfg)
    assert rate(500,cfg)==pytest.approx(.001)
    assert rate(10000,cfg)==pytest.approx(.00001)


def test_paired_participant_bootstrap(tmp_path):
    import pandas as pd
    from temporal_ssl.eeg_factorial_inference import bootstrap
    names=['untrained','capped_mask25','capped_mask50','full_mask25','full_mask50','spectral']
    seeds=[600,601,602]
    for name in names:
        for seed in seeds:
            path=tmp_path/'spectral_control' if name=='spectral' else tmp_path/name/f'seed_{seed}'/'classification'
            path.mkdir(parents=True,exist_ok=True)
            truth=['AD']*7+['FTD']*5+['HC']*6
            pred=truth if name=='full_mask50' else ['FTD' if y!='FTD' else 'HC' for y in truth]
            pd.DataFrame({'participant_id':[f's-{i:02}' for i in range(18)],'split':'test','truth':truth,'prediction':pred}).to_csv(path/'predictions.csv',index=False)
    frame=bootstrap(tmp_path,seeds,replicates=200)
    row=frame.set_index('contrast').loc['masking_full']
    np.testing.assert_allclose([row.difference,row.lower_95,row.upper_95],1)


def test_span_mask_exact_count_edges_and_resume():
    from temporal_ssl.eeg_pilot import mask_runs
    for span in [2,4]:
        for ratio in [.25,.5]:
            g=torch.Generator().manual_seed(8)
            state=g.get_state();m=random_mask(128,160,ratio,g,span)
            restored=torch.Generator();restored.set_state(state)
            torch.testing.assert_close(m,random_mask(128,160,ratio,restored,span))
            assert (m.sum(1)==round(160*ratio)).all()
            lengths=np.concatenate([mask_runs(r.numpy()) for r in m])
            assert np.median(lengths)>=span
