"""Fixed direct first-half → entire second-half reconstruction with the shared backbone."""
import torch

from .models import TemporalSSL, patchify


class FirstHalfPrediction(TemporalSSL):
    """All future patches are observation-free queries; they never act as attention keys.

    Source tokens use causal attention. Every future query reads only source tokens at
    every layer. There is no rollout, teacher forcing, or future-observation embedding.
    Allocated parameters exactly match the masked model, including the shared head.
    """
    def __init__(self, channels, length, **kwargs):
        super().__init__(channels, length, objective="causal", **kwargs)
        self.half = length//self.patch_size//2
        if length//self.patch_size % 2:
            raise ValueError("First-half prediction requires an even number of patches")

    def encode(self, patches, mask=None):
        if mask is not None or patches.shape[1] > self.half:
            raise ValueError("Causal EEG representations accept at most the first half, without a mask")
        n = patches.shape[1]
        return self.encoder(self.embedding(patches)+self.positions[:n],
                            mask=self.causal_attention_mask(n, patches.device))

    def forward(self, x, mask=None, generator=None):
        if mask is not None:
            raise ValueError("First-half prediction has a fixed target region")
        patches = patchify(x, self.patch_size)
        if patches.shape[1] != 2*self.half:
            raise ValueError("Training input must contain the full epoch for target extraction")
        b, n, _ = patches.shape
        source = self.embedding(patches[:, :self.half])
        queries = self.mask_token.expand(b, self.half, -1)
        tokens = torch.cat([source, queries], dim=1)+self.positions[:n]
        forbidden_keys = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        forbidden_keys[:, self.half:] = True
        encoded = self.encoder(tokens, mask=self.causal_attention_mask(n, x.device),
                               src_key_padding_mask=forbidden_keys)
        representation = encoded[:, self.half:]
        prediction, target = self.head(representation), patches[:, self.half:]
        return {"loss": (prediction-target).square().mean(), "prediction": prediction,
                "target": target, "selected": torch.ones(b, self.half, dtype=torch.bool, device=x.device),
                "indices": torch.arange(self.half, n, device=x.device), "representation": representation}

    @torch.no_grad()
    def representations(self, x, mask=None):
        local = self.encode(patchify(x, self.patch_size)[:, :self.half], mask)
        return {"local": local, "sequence": local.mean(1)}
