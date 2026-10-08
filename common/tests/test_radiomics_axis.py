import torch

from back_prop.model_v4.compat_radiomics import DifferentiableRadiomics64


class CaptureAxialExtractor(DifferentiableRadiomics64):
    def _exact(self, image_hu, probability):
        flattened = image_hu.flatten()
        result = torch.zeros(64, device=image_hu.device)
        result[: flattened.numel()] = flattened
        return result

    def _surrogate(self, image_hu, probability):
        return torch.zeros(64, device=image_hu.device) + (
            image_hu.sum() + probability.sum()
        ) * 0.0


def test_online_radiomics_uses_axial_z_slice_and_transpose():
    x, y, z = torch.meshgrid(
        torch.arange(4), torch.arange(5), torch.arange(6), indexing="ij"
    )
    hu = (100 * z + 10 * x + y).float()
    ct = ((hu + 1024.0) / 2048.0)[None, None]
    mask_logits = torch.full_like(ct, -10.0)
    mask_logits[:, :, :, :, 3] = 10.0
    features = CaptureAxialExtractor()(ct, mask_logits)[0]
    expected = hu[:, :, 3].transpose(0, 1).flatten()
    assert torch.equal(features[: expected.numel()], expected)
