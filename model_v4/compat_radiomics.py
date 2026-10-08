"""Legacy 64-feature compatibility implementation; not V4's 18-feature branch.

The forward values come from the bundled NumPy extractor in
``compat_radiomics64``.  A parameter-free torch approximation supplies a
straight-through gradient to the soft mask and CT; therefore the extraction
method itself is frozen while downstream loss can still reach Mask-DETR.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
import torch
from torch import Tensor, nn
import torch.nn.functional as F


from .compat_radiomics64 import FEATURE_NAMES, extract_radiomics64


def _gabor_bank() -> Tensor:
    coords = torch.arange(-4, 5, dtype=torch.float32)
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    kernels = []
    for theta in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
        for frequency in (0.3, 0.4, 0.5):
            wavelength = 1.0 / frequency
            sigma = 0.56 * wavelength
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            gaussian = torch.exp(-0.5 * (x_theta.square() + 0.25 * y_theta.square()) / sigma**2)
            kernels.append(gaussian * torch.sin(2.0 * math.pi * x_theta / wavelength))
    return torch.stack(kernels).unsqueeze(1)


def _largest_component(mask: np.ndarray, score: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        component = labels == int(np.argmax(sizes) + 1)
        if component.sum() >= 9:
            return component
    row, col = np.unravel_index(int(np.argmax(score)), score.shape)
    component = np.zeros_like(mask, dtype=bool)
    component[max(0, row - 2):row + 3, max(0, col - 2):col + 3] = True
    return component


class DifferentiableRadiomics64(nn.Module):
    """Extract the published 64 features without trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gabor_kernels", _gabor_bank(), persistent=False)

    @staticmethod
    def _exact(image_hu: Tensor, probability: Tensor) -> Tensor:
        image = image_hu.detach().float().cpu().numpy()
        score = probability.detach().float().cpu().numpy()
        hard = _largest_component(score >= 0.5, score)
        try:
            values = list(extract_radiomics64(image, hard).values())
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            # Degenerate early-training masks receive a deterministic local ROI.
            hard = _largest_component(np.zeros_like(hard), score)
            values = list(extract_radiomics64(image, hard).values())
        return torch.as_tensor(values, device=image_hu.device, dtype=torch.float32)

    def _surrogate(self, image_hu: Tensor, probability: Tensor) -> Tensor:
        eps = torch.tensor(1e-5, device=image_hu.device, dtype=torch.float32)
        image = image_hu.float() + 1024.0
        p = probability.float().clamp(1e-4, 1.0 - 1e-4)
        weight = p.sum().clamp_min(eps)
        height, width = p.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=p.device, dtype=p.dtype),
            torch.arange(width, device=p.device, dtype=p.dtype), indexing="ij",
        )
        cy, cx = (p * yy).sum() / weight, (p * xx).sum() / weight
        dy, dx = yy - cy, xx - cx
        cov_yy = (p * dy.square()).sum() / weight
        cov_xx = (p * dx.square()).sum() / weight
        cov_yx = (p * dy * dx).sum() / weight
        trace = cov_yy + cov_xx
        root = torch.sqrt(((cov_yy - cov_xx) * 0.5).square() + cov_yx.square() + eps)
        major_var, minor_var = trace * 0.5 + root, (trace * 0.5 - root).clamp_min(eps)
        major, minor = 4.0 * torch.sqrt(major_var), 4.0 * torch.sqrt(minor_var)
        perimeter = (
            (p[1:] - p[:-1]).abs().sum() + (p[:, 1:] - p[:, :-1]).abs().sum()
        ).clamp_min(eps)
        equiv = 2.0 * torch.sqrt(weight / math.pi)
        convex_area = (math.pi * major * minor * 0.25).clamp_min(weight)
        convex_perimeter = (math.pi * (3 * (major + minor) - torch.sqrt(
            (3 * major + minor) * (major + 3 * minor) + eps
        ))).clamp_min(eps)
        eccentricity = torch.sqrt(
            (1.0 - minor_var / major_var.clamp_min(eps)).clamp_min(0.0) + eps
        )
        radial = torch.sqrt(dx.square() + dy.square() + eps)
        radial_mean = (p * radial).sum() / weight
        radial_sd = torch.sqrt((p * (radial - radial_mean).square()).sum() / weight + eps)
        extent = (weight / (major * minor).clamp_min(weight)).clamp(max=1.0)
        shape_size = torch.stack((
            4 * math.pi * weight / convex_perimeter.square(),
            convex_perimeter / perimeter,
            major / minor.clamp_min(eps),
            perimeter.square() / (4 * math.pi * weight), eccentricity,
            weight / convex_area, extent, radial_sd, weight, convex_area,
            perimeter, convex_perimeter, equiv, major, minor,
        ))

        mean_fg = (p * image).sum() / weight
        var_fg = (p * (image - mean_fg).square()).sum() / weight
        bg = 1.0 - p
        bg_weight = bg.sum().clamp_min(eps)
        mean_bg = (bg * image).sum() / bg_weight
        var_bg = (bg * (image - mean_bg).square()).sum() / bg_weight
        temperature = 0.02
        fg_min = -torch.logsumexp(-image / temperature + torch.log(p), dim=(0, 1)) * temperature
        fg_max = torch.logsumexp(image / temperature + torch.log(p), dim=(0, 1)) * temperature
        bg_min = -torch.logsumexp(-image / temperature + torch.log(bg), dim=(0, 1)) * temperature
        bg_max = torch.logsumexp(image / temperature + torch.log(bg), dim=(0, 1)) * temperature
        intensity = torch.stack((
            fg_min, fg_max, mean_fg, torch.sqrt(var_fg + eps),
            bg_min, bg_max, mean_bg, torch.sqrt(var_bg + eps), (mean_fg - mean_bg).abs(),
        ))

        first, second = image[:, :-1], image[:, 1:]
        pair_weight = (p[:, :-1] * p[:, 1:]).clamp_min(1e-6)
        pair_weight = pair_weight / pair_weight.sum()
        delta = (first - second) / 23.375
        first_mean = (pair_weight * first).sum()
        second_mean = (pair_weight * second).sum()
        first_var = (pair_weight * (first - first_mean).square()).sum()
        second_var = (pair_weight * (second - second_mean).square()).sum()
        pair_probability = pair_weight.clamp_min(1e-12)
        haralick = torch.stack((
            (pair_weight * delta.square()).sum(),
            (pair_weight * (first - first_mean) * (second - second_mean)).sum()
            / torch.sqrt(first_var * second_var + eps),
            pair_weight.square().sum(),
            (pair_weight / (1.0 + delta.abs())).sum(),
            -(pair_probability * pair_probability.log()).sum(),
            (pair_weight * delta.pow(3)).sum(),
            (pair_weight / (delta.square() + 1.0)).sum(),
            (pair_weight * (first + second) * 0.5).sum() / 23.375,
            0.5 * (first_var + second_var) / 23.375**2,
            (pair_weight * ((first - first_mean) + (second - second_mean)).square()).sum() / 23.375**2,
            pair_weight.max(),
        ))

        response = F.conv2d(image[None, None], self.gabor_kernels.to(image), padding=4).abs()[0]
        response_mean = (response * p).flatten(1).sum(1) / weight
        response_var = (response - response_mean[:, None, None]).square()
        response_sd = torch.sqrt((response_var * p).flatten(1).sum(1) / weight + eps)
        gabor = torch.stack((response_mean, response_sd), dim=1).reshape(-1)

        padded = F.pad(image[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        neighbours = (
            padded[1:-1, :-2] + padded[1:-1, 2:],
            padded[:-2, 1:-1] + padded[2:, 1:-1],
            padded[:-2, :-2] + padded[2:, 2:],
            padded[:-2, 2:] + padded[2:, :-2],
        )
        residuals = [((image - 0.5 * value).square() * p).sum() / weight for value in neighbours]
        mean_neighbour = torch.stack(neighbours).mean(0) * 0.5
        residuals.append(((image - mean_neighbour).square() * p).sum() / weight)
        markov = torch.stack(residuals)
        result = torch.cat((shape_size, intensity, haralick, gabor, markov))
        if result.numel() != len(FEATURE_NAMES):
            raise RuntimeError(f"Expected 64 surrogate features, got {result.numel()}")
        return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    def forward(self, ct_normalized: Tensor, mask_logits: Tensor) -> Tensor:
        if ct_normalized.shape != mask_logits.shape or ct_normalized.ndim != 5:
            raise ValueError(f"Expected matching [1,1,D,H,W], got {ct_normalized.shape} and {mask_logits.shape}")
        probability_3d = mask_logits.sigmoid()[0, 0]
        # NIfTI arrays in this workspace are [x, y, z].  The paper-compatible
        # offline extractor selects the axial z section with maximum mask area
        # and transposes it to conventional image [row=y, column=x].  Keep the
        # online exact-forward path identical; selecting dimension 0 here would
        # silently use a sagittal section.
        section = int(probability_3d.detach().sum(dim=(0, 1)).argmax())
        probability = probability_3d[:, :, section].transpose(0, 1)
        image_hu = (
            ct_normalized[0, 0, :, :, section].transpose(0, 1) * 2048.0 - 1024.0
        )
        # The exact forward is always finite. Guard the fixed surrogate's
        # piecewise geometric derivatives at degenerate early-training masks;
        # invalid local entries contribute zero rather than poisoning the
        # detector or VISTA gradient.
        def safe_surrogate_gradient(gradient: Tensor) -> Tensor:
            return torch.nan_to_num(
                gradient, nan=0.0, posinf=0.0, neginf=0.0
            )

        if probability.requires_grad:
            probability.register_hook(safe_surrogate_gradient)
        if image_hu.requires_grad:
            image_hu.register_hook(safe_surrogate_gradient)
        exact = self._exact(image_hu, probability)
        surrogate = self._surrogate(image_hu, probability)
        # Subtract first: adding a huge surrogate (e.g. roughness for a
        # near-uniform mask) to a small exact feature can erase that feature.
        return (exact + (surrogate - surrogate.detach())).unsqueeze(0)
