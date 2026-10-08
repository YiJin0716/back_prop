"""Parallel, differentiable diagnostic features; no malignancy semantic head."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from back_prop.common.base_model import MedicalNetEncoder, DEFAULT_MEDICALNET_CHECKPOINT

SEMANTIC_NAMES = ("lobulation", "margin", "sphericity", "spiculation", "subtlety", "texture")
SEMANTIC_INDICES_V2 = (0, 2, 3, 4, 5, 6)
RADIOMICS_NAMES = (
    "volume_mm3", "log1p_volume_mm3", "soft_surface_mm2", "equivalent_diameter_mm",
    "radius_gyration_mm", "sd_x_mm", "sd_y_mm", "sd_z_mm",
    "mean_hu", "sd_hu", "variance_hu2", "rms_hu", "mean_absolute_deviation_hu",
    "skewness", "excess_kurtosis", "energy_hu2_mm3",
    "fraction_above_minus_300_hu", "fraction_above_0_hu",
)
FEATURE_NAMES = RADIOMICS_NAMES + SEMANTIC_NAMES


class SoftRadiomics3D(nn.Module):
    """Fixed formulas on 1-mm CT and soft masks, with their actual derivatives.

    No threshold, hard component, selected slice, STE or learned coefficient.
    Mass is floored at one voxel for empty masks, and intensity variance at
    25 HU^2 only in standardized third/fourth moments. These floors define
    the formulas at degenerate masks rather than masking invalid gradients.
    """
    def forward(self, image_hu: Tensor, logits: Tensor, valid: Tensor) -> Tensor:
        with torch.autocast(device_type=image_hu.device.type, enabled=False):
            hu = image_hu.float()
            p = logits.float().sigmoid() * valid.float()
            dims = (1, 2, 3, 4)
            mass = p.sum(dims)
            denom = mass.clamp_min(1.0)
            mean = (p * hu).sum(dims) / denom
            centered = hu - mean[:, None, None, None, None]
            var = (p * centered.square()).sum(dims) / denom
            sd = (var + 1e-6).sqrt()
            rms = ((p * hu.square()).sum(dims) / denom + 1e-6).sqrt()
            mad = (p * centered.abs()).sum(dims) / denom
            # Scale HU first to keep moment intermediates well conditioned.
            normalized = centered / var.clamp_min(25.0).sqrt()[:, None, None, None, None]
            skew = (p * normalized.pow(3)).sum(dims) / denom
            kurtosis = (p * normalized.pow(4)).sum(dims) / denom - 3.0
            variances = []
            for axis in (2, 3, 4):
                marginal = p.sum(tuple(d for d in dims if d != axis))
                coordinates = torch.arange(p.shape[axis], device=p.device, dtype=p.dtype)
                centre = (marginal * coordinates).sum(1) / denom
                variances.append((marginal * (coordinates[None] - centre[:, None]).square()).sum(1) / denom)
            # Smoothed total variation including the external boundary.
            padded = F.pad(p, (1, 1, 1, 1, 1, 1))
            surface = sum(
                ((padded.diff(dim=axis).square() + 1e-6).sqrt() - 1e-3).sum(dims)
                for axis in (2, 3, 4)
            )
            values = [mass, torch.log1p(mass), surface,
                      (6.0 * mass.clamp_min(1e-6) / math.pi).pow(1.0 / 3.0),
                      (sum(variances) + 1e-6).sqrt()]
            values += [(v + 1e-6).sqrt() for v in variances]
            values += [mean, sd, var, rms, mad, skew, kurtosis,
                       (p * hu.square()).sum(dims)]
            # Smooth density thresholds (20 HU transition), no hard binning.
            values += [(p * ((hu - threshold) / 20.0).sigmoid()).sum(dims) / denom
                       for threshold in (-300.0, 0.0)]
            return torch.stack(values, dim=1)


class MedicalNetSemantics(nn.Module):
    """Pretrained 3-D ResNet50 -> six supervised five-level ordinal ratings."""
    def __init__(self, checkpoint_path=DEFAULT_MEDICALNET_CHECKPOINT, roi_size=64,
                 encoder_factory=None, use_checkpoint=True):
        super().__init__()
        self.encoder = MedicalNetEncoder(checkpoint_path) if encoder_factory is None else encoder_factory()
        self.roi_size = int(roi_size)
        self.use_checkpoint = use_checkpoint
        self.score = nn.Linear(self.encoder.output_dim, len(SEMANTIC_NAMES))
        nn.init.normal_(self.score.weight, std=0.001)
        nn.init.zeros_(self.score.bias)
        self.threshold_base = nn.Parameter(torch.full((6,), -1.5))
        self.threshold_steps = nn.Parameter(torch.full((6, 3), math.log(math.expm1(1.0))))
        self.register_buffer("classes", torch.arange(1, 6, dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        # One ROI per forward: retain pretrained batch statistics while
        # training all encoder weights, including BN affine parameters.
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, ct, mask_logits, valid):
        # The existing MedicalNet checkpoints were unstable under FP16.
        with torch.autocast(device_type=ct.device.type, enabled=False):
            masked_ct = ct.float() * mask_logits.float().sigmoid() * valid.float()
            roi = F.interpolate(masked_ct, size=(self.roi_size,) * 3,
                                mode="trilinear", align_corners=False)
            if self.training and self.use_checkpoint:
                embedding = checkpoint(self.encoder, roi, use_reentrant=False)
            else:
                embedding = self.encoder(roi)
            linear = self.score(embedding)
            thresholds = torch.cat((self.threshold_base[:, None],
                self.threshold_base[:, None] + F.softplus(self.threshold_steps).cumsum(1)), dim=1)
            cumulative = torch.sigmoid(thresholds[None] - linear[..., None])
            probabilities = torch.cat((cumulative[..., :1],
                cumulative[..., 1:] - cumulative[..., :-1], 1.0 - cumulative[..., -1:]), -1)
            probabilities = probabilities.clamp_min(1e-7)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            return probabilities, (probabilities * self.classes).sum(-1)
