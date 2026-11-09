import pytest
import torch

from temporal_ssl.models import TemporalSSL, patchify, span_mask


def model(objective, layers=2):
    torch.manual_seed(3)
    return TemporalSSL(3, 32, objective, patch_size=2, d_model=8, nhead=2,
                       layers=layers, context=3, horizon=2).eval()


@pytest.mark.parametrize("objective", ["causal", "masked"])
def test_forward_representations_and_loss(objective):
    net = model(objective)
    x = torch.randn(2, 32, 3)
    result = net(x)
    assert torch.isfinite(result["loss"])
    expected = (result["prediction"]-result["target"]).square().mean(-1)[result["selected"]].mean()
    torch.testing.assert_close(result["loss"], expected)
    result["loss"].backward()
    assert net.embedding.weight.grad is not None
    rep = net.representations(x)
    assert rep["local"].shape == (2, 16, 8)
    torch.testing.assert_close(rep["sequence"], rep["local"].mean(1))


def test_causal_future_and_context_exclusion_multilayer():
    net = model("causal", layers=3)
    x = torch.randn(2, 32, 3)
    rep = net.representations(x)["local"]
    future = x.clone()
    future[:, 16:] += 100
    torch.testing.assert_close(rep[:, :8], net.representations(future)["local"][:, :8])
    distant = x.clone()
    distant[:, :10] += 100
    torch.testing.assert_close(rep[:, 7:], net.representations(distant)["local"][:, 7:])
    result = net(x)
    # Last h patches never enter causal forward inputs, even with many layers.
    last_targets = x.clone()
    last_targets[:, -4:] += 100
    torch.testing.assert_close(result["prediction"], net(last_targets)["prediction"])
    mask = net.causal_attention_mask(4)
    assert mask[0, 1] and not mask[1, 0] and not mask.diagonal().any()


def test_masked_values_excluded_before_embedding():
    net = model("masked")
    x = torch.randn(2, 32, 3)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 3:8] = True
    changed = x.clone()
    changed.reshape(2, 16, 2, 3)[mask] = 10000
    torch.testing.assert_close(net(x, mask)["prediction"], net(changed, mask)["prediction"])
    # Visible values influence masked predictions bidirectionally.
    future = x.clone()
    future[:, -2:] += 50
    assert not torch.allclose(net(x, mask)["prediction"][:, 4], net(future, mask)["prediction"][:, 4])
    with pytest.raises(ValueError):
        net(x, torch.ones_like(mask))


def test_mask_sampling_and_capacity():
    a = span_mask(4, 16, 0.5, 3, torch.Generator().manual_seed(5))
    b = span_mask(4, 16, 0.5, 3, torch.Generator().manual_seed(5))
    assert torch.equal(a, b) and torch.all(a.sum(-1) == 8)
    causal, masked = model("causal"), model("masked")
    assert sum(p.numel() for p in causal.parameters()) == sum(p.numel() for p in masked.parameters())
    for key in causal.state_dict():
        torch.testing.assert_close(causal.state_dict()[key], masked.state_dict()[key])
