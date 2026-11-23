import torch,pytest
from temporal_ssl.eeg_reconstruction_study import CausalRatioPrediction
from temporal_ssl.eeg_models import FirstHalfPrediction
from temporal_ssl.models import TemporalSSL


def test_causal_splits_exclude_future_and_preserve_features():
    for context in [4,5,6]:
        torch.manual_seed(600);m=CausalRatioPrediction(3,40,context_patches=context,patch_size=5,d_model=8,nhead=2,layers=2).eval()
        x=torch.randn(2,40,3);changed=x.clone();changed[:,context*5:]+=10000
        a=m(x);b=m(changed)
        torch.testing.assert_close(a['prediction'],b['prediction'])
        assert a['prediction'].shape==(2,8-context,15)
        torch.testing.assert_close(a['loss'],(a['prediction']-a['target']).square().mean())
        torch.testing.assert_close(m.representations(x[:,:20])['sequence'],m.representations(changed[:,:20])['sequence'])
        raw=TemporalSSL(3,40,objective='masked',patch_size=5,d_model=8,nhead=2,layers=2)
        assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in raw.parameters())
    torch.manual_seed(600);old=FirstHalfPrediction(3,40,patch_size=5,d_model=8,nhead=2,layers=2).eval()
    torch.manual_seed(600);new=CausalRatioPrediction(3,40,context_patches=4,patch_size=5,d_model=8,nhead=2,layers=2).eval()
    torch.testing.assert_close(old(x)['prediction'],new(x)['prediction'],rtol=0,atol=0)
    with pytest.raises(ValueError):CausalRatioPrediction(3,40,context_patches=8,patch_size=5)
