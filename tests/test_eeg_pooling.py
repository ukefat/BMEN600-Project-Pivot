import torch
from temporal_ssl.eeg_pooling import pool
from temporal_ssl.eeg_data import aggregate_participants


def test_pooling_population_std_and_unchanged_two_stage_mean():
    x=torch.tensor([[[1.,2.],[3.,6.]],[[5.,10.],[9.,18.]]])
    f=pool(x)
    torch.testing.assert_close(f['mean'],torch.tensor([[2.,4.],[7.,14.]]))
    torch.testing.assert_close(f['std'],torch.tensor([[1.,2.],[2.,4.]]))
    torch.testing.assert_close(f['mean_std'],torch.cat([f['mean'],f['std']],-1))
    metadata=[{'participant_id':'a','group':'AD'}]*2
    _,v,_=aggregate_participants(f['mean_std'].numpy(),metadata)
    torch.testing.assert_close(torch.from_numpy(v),torch.tensor([[4.5,9.,1.5,3.]]))
    assert not torch.allclose(f['std'].mean(0),x.reshape(-1,2).std(0,unbiased=False))
    constant=pool(torch.ones(2,80,32))
    assert constant['std'].count_nonzero()==0
