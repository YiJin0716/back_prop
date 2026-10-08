"""V2 geometry with FP32 trilinear interpolation for PyTorch 2.0 BF16 support."""
from contextlib import nullcontext
import torch
from torch import Tensor
import torch.nn.functional as F
from back_prop.common.base_model import NoduleMaskDETR3D
from back_prop.common.refiner import FineMaskRefiner3D

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # PyTorch 2.0 environment used by the A5000 run.
    sdpa_kernel = None

class V3MaskDETR(NoduleMaskDETR3D):
    def forward(self, screening: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Torch 2.0.1's fused BF16 attention was observed to vary between
        # identical eval calls, enough to move rounded ROI centres. Use the
        # math backend for deterministic inference; training is unchanged.
        context = nullcontext()
        if screening.is_cuda and not self.training:
            # New Torch also has a cuDNN attention backend; selecting only
            # MATH explicitly excludes that backend as well.
            context = (sdpa_kernel(SDPBackend.MATH) if sdpa_kernel is not None else
                       torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                                       enable_mem_efficient=False))
        with context:
            return self._forward(screening)

    def _forward(self, screening: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        coarse = F.adaptive_avg_pool3d(screening, self.coarse_shape)
        encoded_grid = self.input_projection(coarse)
        features = encoded_grid.flatten(2).transpose(1, 2)
        memory = self.encoder(features + self.position_embedding)
        queries = self.query_embedding.expand(screening.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        object_logits = self.object_head(decoded).squeeze(-1)
        raw_boxes = self.box_head(decoded).float()
        centres = raw_boxes[..., :3].sigmoid()
        # Tight nodule boxes may be small, but never exactly zero-sized.
        sizes = 0.01 + 0.49 * raw_boxes[..., 3:].sigmoid()
        mask_features = self.mask_feature_projection(screening)
        context = self.context_projection(encoded_grid)
        context = F.interpolate(context.float(), size=screening.shape[-3:], mode="trilinear", align_corners=False)
        mask_features = mask_features + context
        mask_embeddings = self.mask_embedding(decoded)
        mask_logits = torch.einsum("bqc,bcdhw->bqdhw", mask_embeddings, mask_features)
        return object_logits.float(), torch.cat((centres, sizes), dim=-1), mask_logits.float(), decoded


class V3Refiner(FineMaskRefiner3D):
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

        hidden = F.interpolate(hidden.float(), size=skip_2.shape[-3:], mode="trilinear", align_corners=False)
        hidden = self.up_2(hidden)
        hidden = self._block(self.decoder_2, torch.cat((hidden, skip_2), dim=1), query)
        hidden = F.interpolate(hidden.float(), size=skip_1.shape[-3:], mode="trilinear", align_corners=False)
        hidden = self.up_1(hidden)
        hidden = self._block(self.decoder_1, torch.cat((hidden, skip_1), dim=1), query)
        return self.residual_head(hidden)
