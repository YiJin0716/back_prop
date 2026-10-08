import torch

from back_prop.model_v4.compat_radiomics import DifferentiableRadiomics64


class LargeSurrogateExtractor(DifferentiableRadiomics64):
    def _exact(self, image_hu, probability):
        return torch.ones(64, device=probability.device)

    def _surrogate(self, image_hu, probability):
        return (1e8 + probability.mean()).expand(64)


def test_large_surrogate_preserves_exact_forward_and_backward():
    # Empty/near-uniform masks can produce enormous surrogate roughness.
    # Its magnitude must not erase the small exact feature via cancellation.
    logits = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    features = LargeSurrogateExtractor()(torch.zeros_like(logits), logits)
    torch.testing.assert_close(features, torch.ones_like(features), rtol=0, atol=0)
    features.sum().backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad.sum(), torch.tensor(16.0))
