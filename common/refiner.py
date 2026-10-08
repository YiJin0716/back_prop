"""Query-conditioned lightweight 3-D residual mask refiner."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FiLMResidualBlock3D(nn.Module):
    """Residual convolutional block modulated by one DETR query embedding."""

    def __init__(self, input_channels: int, output_channels: int, query_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, kernel_size=1, bias=False)
        )
        self.film = nn.Linear(query_dim, 2 * output_channels)
        # The pretrained/coarse prior should not be perturbed at construction.
        # FiLM starts as gamma=beta=0 and learns query conditioning thereafter.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, value: Tensor, query: Tensor) -> Tensor:
        residual = self.skip(value)
        hidden = F.silu(self.norm1(self.conv1(value)))
        hidden = self.norm2(self.conv2(hidden))
        gamma, beta = self.film(query).chunk(2, dim=1)
        broadcast = (query.shape[0], gamma.shape[1], 1, 1, 1)
        hidden = hidden * (1.0 + gamma.view(broadcast)) + beta.view(broadcast)
        return F.silu(hidden + residual)


class FineMaskRefiner3D(nn.Module):
    """Small fully-convolutional U-Net that returns residual mask logits.

    The expected production input is ``[B,12,X,Y,Z]``: normalized CT, VISTA
    probability, coarse-query probability, ROI-valid mask and eight optional
    context channels.  ``input_channels`` remains configurable for ablations.
    The spatial shape is not hard-coded; odd sizes are restored exactly by the
    interpolation-based decoder.

    The final convolution and every FiLM projection are zero initialized, so a
    newly constructed module returns an all-zero delta.  The caller should use
    ``fine_logits = sampled_coarse_logits + refiner(inputs, query)``.
    """

    def __init__(
        self,
        input_channels: int = 12,
        query_dim: int = 128,
        base_channels: int = 8,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or query_dim <= 0 or base_channels <= 0:
            raise ValueError("input_channels, query_dim and base_channels must be positive")
        self.input_channels = int(input_channels)
        self.query_dim = int(query_dim)
        self.base_channels = int(base_channels)
        self.use_checkpoint = bool(use_checkpoint)

        base = self.base_channels
        self.encoder_1 = FiLMResidualBlock3D(self.input_channels, base, self.query_dim)
        self.down_1 = nn.Conv3d(base, 2 * base, kernel_size=2, stride=2, bias=False)
        self.encoder_2 = FiLMResidualBlock3D(2 * base, 2 * base, self.query_dim)
        self.down_2 = nn.Conv3d(2 * base, 4 * base, kernel_size=2, stride=2, bias=False)
        self.bottleneck = FiLMResidualBlock3D(4 * base, 4 * base, self.query_dim)

        self.up_2 = nn.Conv3d(4 * base, 2 * base, kernel_size=3, padding=1, bias=False)
        self.decoder_2 = FiLMResidualBlock3D(4 * base, 2 * base, self.query_dim)
        self.up_1 = nn.Conv3d(2 * base, base, kernel_size=3, padding=1, bias=False)
        self.decoder_1 = FiLMResidualBlock3D(2 * base, base, self.query_dim)
        self.residual_head = nn.Conv3d(base, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _block(self, block: nn.Module, value: Tensor, query: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, value, query, use_reentrant=False)
        return block(value, query)

    def forward(self, x: Tensor, query: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must be [B,C,X,Y,Z], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channels, got {x.shape[1]}")
        if query.ndim != 2 or query.shape != (x.shape[0], self.query_dim):
            raise ValueError(
                f"query must be [{x.shape[0]},{self.query_dim}], got {tuple(query.shape)}"
            )
        if min(x.shape[-3:]) < 4:
            raise ValueError("each spatial dimension must be at least four voxels")

        skip_1 = self._block(self.encoder_1, x, query)
        skip_2 = self._block(self.encoder_2, self.down_1(skip_1), query)
        hidden = self._block(self.bottleneck, self.down_2(skip_2), query)

        hidden = F.interpolate(hidden, size=skip_2.shape[-3:], mode="trilinear", align_corners=False)
        hidden = self.up_2(hidden)
        hidden = self._block(self.decoder_2, torch.cat((hidden, skip_2), dim=1), query)
        hidden = F.interpolate(hidden, size=skip_1.shape[-3:], mode="trilinear", align_corners=False)
        hidden = self.up_1(hidden)
        hidden = self._block(self.decoder_1, torch.cat((hidden, skip_1), dim=1), query)
        return self.residual_head(hidden)


__all__ = ("FiLMResidualBlock3D", "FineMaskRefiner3D")
