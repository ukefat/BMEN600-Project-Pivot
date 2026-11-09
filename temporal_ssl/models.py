"""Matched encoders with explicit observation-access rules."""
import math

import torch
from torch import nn


def patchify(x, patch_size):
    if x.shape[1] % patch_size:
        raise ValueError("Sequence length must be divisible by patch_size")
    return x.reshape(x.shape[0], x.shape[1]//patch_size, -1)


def span_mask(batch, patches, ratio, span, generator=None, device="cpu"):
    """Hide a fixed number of patches; sampled blocks may merge or truncate at edges."""
    if not 0 < ratio < 1 or span < 1 or patches < 2:
        raise ValueError("Require 0 < mask_ratio < 1, span >= 1, patches >= 2")
    count = min(patches-1, max(1, round(patches*ratio)))
    mask = torch.zeros(batch, patches, dtype=torch.bool)
    for row in mask:
        while int(row.sum()) < count:
            starts = (~row).nonzero().flatten()
            start = int(starts[torch.randint(len(starts), (), generator=generator)])
            for j in range(start, min(patches, start+span)):
                if not row[j] and int(row.sum()) < count:
                    row[j] = True
    return mask.to(device)


def positional_encoding(length, width):
    position = torch.arange(length).float()[:, None]
    frequency = torch.exp(torch.arange(0, width, 2).float()*(-math.log(10000)/width))
    out = torch.zeros(length, width)
    out[:, 0::2] = torch.sin(position*frequency)
    out[:, 1::2] = torch.cos(position*frequency[:out[:, 1::2].shape[1]])
    return out


class TemporalSSL(nn.Module):
    def __init__(self, channels, length, objective="causal", patch_size=4, d_model=32,
                 nhead=4, layers=2, dropout=0.0, horizon=1, context=8,
                 mask_ratio=0.5, mask_span=2):
        super().__init__()
        if objective not in {"causal", "masked"}:
            raise ValueError("objective must be causal or masked")
        if patch_size < 1 or length % patch_size or not 1 <= horizon < length//patch_size or context < 1:
            raise ValueError("Invalid patch_size, horizon, context, or sequence length")
        if not 0 < mask_ratio < 1 or mask_span < 1:
            raise ValueError("Require 0 < mask_ratio < 1 and mask_span >= 1")
        self.objective, self.patch_size = objective, patch_size
        self.horizon, self.context = horizon, context
        self.mask_ratio, self.mask_span = mask_ratio, mask_span
        self.embedding = nn.Linear(channels*patch_size, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model),
                                            enable_nested_tensor=False)
        self.head = nn.Linear(d_model, channels*patch_size)
        self.register_buffer("positions", positional_encoding(length//patch_size, d_model))

    def causal_attention_mask(self, size, device=None):
        return torch.ones(size, size, dtype=torch.bool, device=device).triu(1)

    def encode(self, patches, mask=None):
        b, n, _ = patches.shape
        if self.objective == "masked":
            if mask is not None:
                if mask.shape != (b, n) or mask.dtype != torch.bool or mask.all(-1).any():
                    raise ValueError("Mask must be boolean [B,N] with at least one visible patch per row")
                # Replace BEFORE embedding; masked values never enter any model computation.
                safe = patches.masked_fill(mask[..., None], 0)
                tokens = torch.where(mask[..., None], self.mask_token, self.embedding(safe))
            else:
                tokens = self.embedding(patches)
            # Masked queries may read visible keys; hidden queries are never keys at any layer.
            return self.encoder(tokens+self.positions[:n], src_key_padding_mask=mask)
        if mask is not None:
            raise ValueError("Causal encoder does not accept a reconstruction mask")
        tokens = self.embedding(patches)+self.positions[:n]
        # Separate windows prevent multi-layer attention from expanding context via earlier tokens.
        width = min(self.context, n)
        early = []
        for stop in range(1, width):
            encoded = self.encoder(tokens[:, :stop], mask=self.causal_attention_mask(stop, patches.device))
            early.append(encoded[:, -1:])
        indices = torch.arange(width-1, n, device=patches.device)[:, None] - torch.arange(width-1, -1, -1, device=patches.device)
        windows = tokens[:, indices].reshape(b*(n-width+1), width, -1)
        encoded = self.encoder(windows, mask=self.causal_attention_mask(width, patches.device))
        return torch.cat(early+[encoded[:, -1].reshape(b, n-width+1, -1)], dim=1)

    def forward(self, x, mask=None, generator=None):
        patches = patchify(x, self.patch_size)
        if self.objective == "causal":
            # For target j, source endpoint is j-horizon, including all patch samples there.
            representation = self.encode(patches[:, :-self.horizon])
            prediction = self.head(representation)
            target = patches[:, self.horizon:]
            selected = torch.ones(target.shape[:2], dtype=torch.bool, device=x.device)
            indices = torch.arange(self.horizon, patches.shape[1], device=x.device)
        else:
            if mask is None:
                mask = span_mask(len(x), patches.shape[1], self.mask_ratio, self.mask_span,
                                 generator, x.device)
            representation = self.encode(patches, mask)
            prediction, target, selected = self.head(representation), patches, mask
            indices = torch.arange(patches.shape[1], device=x.device)
        loss = (prediction-target).square().mean(-1)[selected].mean()
        return {"loss": loss, "prediction": prediction, "target": target,
                "selected": selected, "indices": indices, "representation": representation}

    @torch.no_grad()
    def representations(self, x, mask=None):
        """Local embedding is aligned to patch END; sequence pooling is its arithmetic mean.

        Call eval() first. Causal locals read <= context patches ending here. Masked locals
        default to the full uncorrupted sequence, including future; a mask can be supplied.
        """
        local = self.encode(patchify(x, self.patch_size), mask)
        return {"local": local, "sequence": local.mean(1)}
