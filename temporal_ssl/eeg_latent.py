"""Shared-weight masked latent prediction with patchwise, across-batch SIGReg."""
import torch
from torch import nn
from .models import TemporalSSL,patchify


class SIGReg(nn.Module):
    """Sliced Epps–Pulley statistic, positive-half Gaussian-weighted quadrature.

    Formula follows rbalestr-lab/lejepa's EppsPulley/SlicingUnivariateTest.
    Input [B,N,D]: N positions are separate populations, B observations each.
    No batch standardization, whitening, stop-gradient, or learned projector.
    """
    def __init__(self,slices=64,knots=17,t_max=3.):
        super().__init__()
        if slices<1 or knots<3 or t_max<=0:raise ValueError('Invalid SIGReg quadrature')
        self.slices=slices
        t=torch.linspace(0,t_max,knots)
        phi=torch.exp(-t.square()/2)
        weights=torch.full((knots,),2*t_max/(knots-1));weights[[0,-1]]/=2
        self.register_buffer('t',t);self.register_buffer('phi',phi);self.register_buffer('weights',weights*phi)

    def forward(self,z,generator=None,directions=None):
        if z.ndim!=3 or z.shape[0]<2:raise ValueError('SIGReg needs [B,N,D], B>=2')
        if directions is None:
            directions=torch.randn(z.shape[-1],self.slices,generator=generator,device='cpu').to(z.device)
            directions=directions/directions.norm(dim=0).clamp_min(1e-12)
        values=(z.float().transpose(0,1)@directions.float()).unsqueeze(-1)*self.t
        error=(values.cos().mean(1)-self.phi).square()+values.sin().mean(1).square()
        return ((error@self.weights)*z.shape[0]).mean()


class MaskedLatentPrediction(TemporalSSL):
    def __init__(self,channels,length,*,sigreg_weight=.02,sigreg_slices=64,target_context=None,**kwargs):
        # Initialize the exact existing backbone before allocating the predictor.
        super().__init__(channels,length,objective='masked',**kwargs)
        width=self.embedding.out_features
        self.head=nn.Identity() # No waveform decoder or unused waveform parameters.
        layer=nn.TransformerEncoderLayer(width,kwargs.get('nhead',4),4*width,
            dropout=kwargs.get('dropout',0.),activation='gelu',batch_first=True,norm_first=True)
        self.predictor=nn.TransformerEncoder(layer,1,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        self.prediction_head=nn.Linear(width,width)
        self.sigreg=SIGReg(sigreg_slices)
        self.sigreg_weight=sigreg_weight
        if target_context is not None and (target_context<1 or target_context%2==0):
            raise ValueError('Target context must be a positive odd number')
        self.target_context=target_context

    def encode_targets(self,x):
        patches=patchify(x,self.patch_size)
        if self.target_context is None:return self.encode(patches)
        b,n,_=patches.shape;r=self.target_context//2
        indices=torch.arange(n,device=x.device)[:,None]+torch.arange(-r,r+1,device=x.device)[None,:]
        padding=(indices<0)|(indices>=n)
        safe=indices.clamp(0,n-1)
        tokens=self.embedding(patches)+self.positions[:n]
        windows=tokens[:,safe].reshape(b*n,self.target_context,-1)
        forbidden=padding.unsqueeze(0).expand(b,-1,-1).reshape(b*n,self.target_context)
        local=self.encoder(windows,src_key_padding_mask=forbidden)
        return local[:,r].reshape(b,n,-1)

    def predict_masked(self,x,mask):
        local=self.encode(patchify(x,self.patch_size),mask)
        # Mask queries retain their contextualized states. Only visible keys/values
        # are read by the predictor, matching the backbone's access convention.
        return self.prediction_head(self.predictor(local,src_key_padding_mask=mask)),local

    def forward(self,x,mask,sigreg_generator=None):
        if not mask.any():raise ValueError('Need masked prediction targets')
        original=self.encode_targets(x)
        prediction,masked=self.predict_masked(x,mask)
        latent_loss=(prediction-original).square().mean(-1)[mask].mean()
        regularizer=self.sigreg(original,sigreg_generator)
        return {'loss':latent_loss+self.sigreg_weight*regularizer,'latent_loss':latent_loss,
            'sigreg_loss':regularizer,'prediction':prediction,'target':original,
            'original':original,'masked':masked,'selected':mask}


@torch.no_grad()
def collapse_statistics(z):
    def moments(x): # populations × observations × dimensions
        centered=x.double()-x.double().mean(-2,keepdim=True)
        covariance=centered.transpose(-2,-1)@centered/max(x.shape[-2]-1,1)
        eigen=torch.linalg.eigvalsh(covariance).clamp_min(0)
        total=eigen.sum(-1,keepdim=True)
        probabilities=eigen/total.clamp_min(1e-30)
        rank=torch.exp(-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1))
        rank=torch.where(total.squeeze(-1)>1e-20,rank,torch.zeros_like(rank))
        return float(covariance.diagonal(dim1=-2,dim2=-1).mean()),float(rank.mean())
    variance,rank=moments(z.transpose(0,1))
    pooled_variance,pooled_rank=moments(z.mean(1).unsqueeze(0))
    return {'patch_variance':variance,'patch_effective_rank':rank,
            'pooled_variance':pooled_variance,'pooled_effective_rank':pooled_rank,
            'within_window_variance':float(z.double().var(dim=1,unbiased=False).mean())}
