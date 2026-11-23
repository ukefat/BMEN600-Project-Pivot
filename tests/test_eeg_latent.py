import pytest
import torch
from temporal_ssl.eeg_latent import SIGReg,MaskedLatentPrediction,collapse_statistics
from temporal_ssl.models import TemporalSSL


def model():return MaskedLatentPrediction(3,40,patch_size=5,d_model=8,nhead=2,layers=1,dropout=0.)


def test_shared_initialization_gradients_and_mask_exclusion():
    torch.manual_seed(600);raw=TemporalSSL(3,40,objective='masked',patch_size=5,d_model=8,nhead=2,layers=1,dropout=0.)
    torch.manual_seed(600);m=model().eval()
    for key,value in m.state_dict().items():
        if key.startswith(('embedding.','encoder.')):torch.testing.assert_close(value,raw.state_dict()[key],rtol=0,atol=0)
    x=torch.randn(4,40,3);mask=torch.zeros(4,8,dtype=torch.bool);mask[:,2:4]=True
    altered=x.clone().reshape(4,8,5,3);altered[mask]=1000
    p,_=m.predict_masked(x,mask);q,_=m.predict_masked(altered.reshape_as(x),mask)
    torch.testing.assert_close(p,q)
    result=m(x,mask,torch.Generator().manual_seed(9))
    result['target'].retain_grad();result['masked'].retain_grad()
    torch.testing.assert_close(result['latent_loss'],(result['prediction']-result['target']).square().mean(-1)[mask].mean())
    result['latent_loss'].backward()
    assert result['target'].grad is not None and result['target'].grad.norm()>0
    assert result['masked'].grad is not None and result['masked'].grad.norm()>0
    assert m.embedding.weight.grad.norm()>0 and m.prediction_head.weight.grad.norm()>0
    # Separate branch sensitivity: masked branch has exactly zero hidden-input derivative.
    source=x.clone().requires_grad_();pred,_=m.predict_masked(source,mask)
    grad,=torch.autograd.grad(pred.square().mean(),source)
    assert grad.reshape(4,8,5,3)[mask].abs().max()==0


def test_sigreg_reference_quadrature_gaussian_and_position_collapse():
    torch.manual_seed(4);reg=SIGReg(slices=16)
    z=torch.randn(256,3,8,requires_grad=True)
    a=torch.randn(8,16);a=a/a.norm(dim=0)
    loss=reg(z,directions=a)
    projected=z.transpose(0,1)@a
    t=reg.t
    real=(projected.unsqueeze(-1)*t).cos().mean(1)
    imaginary=(projected.unsqueeze(-1)*t).sin().mean(1)
    # Independent full [-3,3] trapezoidal integration verifies factor 2 and N.
    grid=torch.linspace(-3,3,33)
    values=projected.unsqueeze(-1)*grid
    integrand=((values.cos().mean(1)-torch.exp(-grid.square()/2))**2+values.sin().mean(1)**2)*torch.exp(-grid.square()/2)
    expected=torch.trapezoid(integrand,grid,dim=-1).mean()*256
    torch.testing.assert_close(loss,expected)
    constant=torch.zeros_like(z)
    assert reg(constant,directions=a)>loss*5
    loss.backward();assert z.grad.norm()>0
    positional=torch.randn(1,3,8).expand(256,-1,-1)
    assert collapse_statistics(positional)['patch_variance']==0
    assert collapse_statistics(positional)['pooled_effective_rank']==0
    assert collapse_statistics(z.detach())['patch_effective_rank']>5
    with pytest.raises(ValueError):reg(torch.zeros(1,3,8))


def test_latent_optimizer_and_sigreg_rng_resume(tmp_path):
    import copy
    torch.manual_seed(5);x=torch.randn(4,40,3)
    mask=torch.zeros(4,8,dtype=torch.bool);mask[:,2:4]=True
    torch.manual_seed(600);full=model();initial=copy.deepcopy(full.state_dict())
    def advance(m,opt,g,count):
        for _ in range(count):
            opt.zero_grad(set_to_none=True);m(x,mask,g)['loss'].backward();opt.step()
    a=torch.optim.AdamW(full.parameters(),lr=.001);ga=torch.Generator().manual_seed(12)
    advance(full,a,ga,4)
    partial=model();partial.load_state_dict(initial);b=torch.optim.AdamW(partial.parameters(),lr=.001)
    gb=torch.Generator().manual_seed(12);advance(partial,b,gb,2)
    torch.save({'model':partial.state_dict(),'optimizer':b.state_dict(),'sigreg':gb.get_state()},tmp_path/'resume.pt')
    state=torch.load(tmp_path/'resume.pt',weights_only=False)
    resumed=model();resumed.load_state_dict(state['model']);c=torch.optim.AdamW(resumed.parameters(),lr=.001)
    c.load_state_dict(state['optimizer']);gc=torch.Generator();gc.set_state(state['sigreg'])
    advance(resumed,c,gc,2)
    for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    torch.testing.assert_close(ga.get_state(),gc.get_state(),rtol=0,atol=0)


def test_exact_target_neighborhood_and_local_gradients():
    for context in [1,5]:
        torch.manual_seed(600)
        m=MaskedLatentPrediction(3,40,patch_size=5,d_model=8,nhead=2,layers=2,target_context=context).eval()
        x=torch.randn(4,40,3);center=3;r=context//2
        changed=x.clone();changed[:,:max(0,center-r)*5]+=1000;changed[:,(center+r+1)*5:]-=1000
        torch.testing.assert_close(m.encode_targets(x)[:,center],m.encode_targets(changed)[:,center],rtol=0,atol=0)
        target=m.encode_targets(x)
        patches=x.reshape(4,8,15);tokens=m.embedding(patches)+m.positions[:8]
        expected=m.encoder(tokens[:,center-r:center+r+1])[:,r]
        torch.testing.assert_close(target[:,center],expected)
        mask=torch.zeros(4,8,dtype=torch.bool);mask[:,3]=True
        result=m(x,mask);result['target'].retain_grad();result['latent_loss'].backward()
        assert result['target'].grad.norm()>0 and m.embedding.weight.grad.norm()>0


def test_target_window_boundary_does_not_duplicate_observations():
    m=MaskedLatentPrediction(3,40,patch_size=5,d_model=8,nhead=2,layers=2,target_context=5).eval()
    x=torch.randn(2,40,3);tokens=m.embedding(x.reshape(2,8,15))+m.positions[:8]
    torch.testing.assert_close(m.encode_targets(x)[:,0],m.encoder(tokens[:,:3])[:,0])
    torch.testing.assert_close(m.encode_targets(x)[:,-1],m.encoder(tokens[:,-3:])[:,-1])
