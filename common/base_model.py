"""Whole-CT VISTA3D--3D Mask-DETR--MedicalNet joint model.

VISTA3D is evaluated exactly once per sliding window and its differentiable
soft logits plus decoder features are stitched over the complete CT. A 3D
Mask-DETR decoder converts that representation into a set of nodule instance
masks, object scores, and ROI boxes. Each soft instance mask weights its CT ROI
before MedicalNet feature extraction. VISTA3D is never called a second time.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from monai.networks.nets import resnet50


DEFAULT_VISTA_SOURCE = Path(__file__).parents[2] / "vista3D/VISTA/vista3d"
DEFAULT_VISTA_CHECKPOINT = Path(
    "/usr/project/rudinlab/datasets/LIDC_IDRI/vista3D_workspace/results/"
    "fold_0/checkpoints/best_metric_model.pt"
)
DEFAULT_MEDICALNET_CHECKPOINT = Path(
    "/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/pretrained/resnet_50_23dataset.pth"
)
SEMANTIC_NAMES = (
    "log1p_volume_mm3", "sphericity", "margin", "lobulation",
    "spiculation", "texture", "subtlety",
)


def _state_dict(path: str | Path) -> dict[str, Tensor]:
    value = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    return {str(key).removeprefix("module."): tensor for key, tensor in value.items()}


def build_vista3d(checkpoint_path: str | Path = DEFAULT_VISTA_CHECKPOINT) -> nn.Module:
    source = str(DEFAULT_VISTA_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    from vista3d.build_vista3d import build_vista3d_segresnet_decoder

    model = build_vista3d_segresnet_decoder(in_channels=1)
    incompatible = model.load_state_dict(_state_dict(checkpoint_path), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"VISTA3D checkpoint mismatch: {incompatible}")
    return model


class VistaNoduleSegmenter(nn.Module):
    def __init__(self, model: nn.Module, prompt_class: int = 23) -> None:
        super().__init__()
        self.model = model
        self.prompt_class = int(prompt_class)
        # Auto-decoder levels have 384/192/96/48 channels for the published
        # 48-filter VISTA3D model. Project each native scale to 12 channels.
        self.pyramid_projections = nn.ModuleList(
            nn.Conv3d(channels, 12, kernel_size=1)
            for channels in (384, 192, 96, 48)
        )

    def forward(self, image: Tensor) -> Tensor:
        logits, _ = self.forward_with_features(image)
        return logits

    def forward_with_features(self, image: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Return class logits and four native-resolution decoder feature maps."""
        if image.shape[0] != 1:
            raise ValueError("VISTA3D automatic class branch requires batch size one")
        prompt = torch.tensor([[self.prompt_class]], device=image.device, dtype=torch.long)
        image_encoder = self.model.image_encoder
        if image_encoder.preprocess is not None:
            image = image_encoder.preprocess(image)
        if not image_encoder.is_valid_shape(image):
            raise ValueError(f"Invalid VISTA3D window shape {tuple(image.shape)}")
        skip_features = image_encoder.encoder(image)
        skip_features.reverse()
        decoded = skip_features.pop(0)
        pyramid: list[Tensor] = []
        final_auto: Tensor | None = None
        for index, level in enumerate(image_encoder.up_layers_auto):
            decoded = level["upsample"](decoded)
            decoded = decoded + skip_features[index]
            decoded = level["blocks"](decoded)
            pyramid.append(self.pyramid_projections[index](decoded))
            if index + 1 == len(image_encoder.up_layers_auto):
                final_auto = level["head"](decoded)
        if final_auto is None:
            raise RuntimeError("VISTA3D auto decoder produced no feature map")
        logits, _ = self.model.class_head(final_auto, prompt)
        if logits.ndim == 6:
            logits = logits[:, 0]
        if logits.shape[:2] != (1, 1):
            raise RuntimeError(f"Unexpected VISTA3D output shape {tuple(logits.shape)}")
        return logits, tuple(pyramid)


class MedicalNetEncoder(nn.Module):
    output_dim = 2048

    def __init__(self, checkpoint_path: str | Path = DEFAULT_MEDICALNET_CHECKPOINT) -> None:
        super().__init__()
        self.network = resnet50(
            spatial_dims=3, n_input_channels=1, feed_forward=False, bias_downsample=False
        )
        incompatible = self.network.load_state_dict(_state_dict(checkpoint_path), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"MedicalNet checkpoint mismatch: {incompatible}")

    def forward(self, roi: Tensor) -> Tensor:
        return self.network(roi)


class WholeCTOutput(NamedTuple):
    boxes: Tensor
    sampling_boxes: Tensor
    candidate_logits: Tensor
    object_logits: Tensor
    semantic_features: Tensor
    nodule_logits: Tensor
    risk_logit: Tensor
    discovered_mask: Tensor
    window_count: int


def sliding_starts(length: int, window: int, overlap: float) -> list[int]:
    if length <= window:
        return [0]
    stride = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def extract_patch(volume: Tensor, centre: Tensor | np.ndarray, size: tuple[int, int, int]) -> Tensor:
    """Extract a centred CPU or GPU patch and pad outside the scan."""
    centre_array = np.asarray(centre.detach().cpu() if isinstance(centre, Tensor) else centre, dtype=int)
    spatial = np.asarray(volume.shape[-3:], dtype=int)
    requested = np.asarray(size, dtype=int)
    start = centre_array - requested // 2
    source_start = np.maximum(start, 0)
    source_end = np.minimum(start + requested, spatial)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_end))
    patch = volume[(...,) + slices]
    before = source_start - start
    after = start + requested - source_end
    padding: list[int] = []
    for left, right in zip(before[::-1], after[::-1]):
        padding.extend((int(left), int(right)))
    return F.pad(patch, padding, value=0.0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )

    def forward(self, value: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 != len(self.layers):
                value = F.gelu(value)
        return value


class NoduleMaskDETR3D(nn.Module):
    """3D mask-classification DETR over stitched VISTA3D soft features."""

    def __init__(
        self,
        num_queries: int = 24,
        hidden_dim: int = 128,
        coarse_shape: tuple[int, int, int] = (12, 12, 12),
        nheads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        input_channels: int = 50,
        mask_dim: int = 32,
    ) -> None:
        super().__init__()
        if hidden_dim % nheads:
            raise ValueError("DETR hidden_dim must be divisible by nheads")
        self.num_queries = int(num_queries)
        self.coarse_shape = tuple(int(value) for value in coarse_shape)
        self.input_projection = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU(),
        )
        tokens = int(np.prod(self.coarse_shape))
        self.position_embedding = nn.Parameter(torch.empty(1, tokens, hidden_dim))
        self.query_embedding = nn.Parameter(torch.empty(1, self.num_queries, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, nheads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, nheads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.object_head = nn.Linear(hidden_dim, 1)
        self.box_head = MLP(hidden_dim, hidden_dim, 6, 3)
        self.mask_feature_projection = nn.Sequential(
            nn.Conv3d(input_channels, mask_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, mask_dim), mask_dim),
            nn.GELU(),
        )
        self.context_projection = nn.Conv3d(hidden_dim, mask_dim, kernel_size=1)
        self.mask_embedding = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.query_embedding, std=0.02)
        nn.init.constant_(self.object_head.bias, -2.0)

    def forward(self, screening: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        coarse = F.adaptive_avg_pool3d(screening, self.coarse_shape)
        encoded_grid = self.input_projection(coarse)
        features = encoded_grid.flatten(2).transpose(1, 2)
        memory = self.encoder(features + self.position_embedding)
        queries = self.query_embedding.expand(screening.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        object_logits = self.object_head(decoded).squeeze(-1)
        raw_boxes = self.box_head(decoded)
        centres = raw_boxes[..., :3].sigmoid()
        # Tight nodule boxes may be small, but never exactly zero-sized.
        sizes = 0.01 + 0.49 * raw_boxes[..., 3:].sigmoid()
        mask_features = self.mask_feature_projection(screening)
        context = self.context_projection(encoded_grid)
        context = F.interpolate(context, size=screening.shape[-3:], mode="trilinear", align_corners=False)
        mask_features = mask_features + context
        mask_embeddings = self.mask_embedding(decoded)
        mask_logits = torch.einsum("bqc,bcdhw->bqdhw", mask_embeddings, mask_features)
        return object_logits, torch.cat((centres, sizes), dim=-1), mask_logits, decoded


def differentiable_box_sample(volume: Tensor, box: Tensor, output_size: tuple[int, int, int]) -> Tensor:
    """Sample a normalized (D,H,W,depth,height,width) box with box gradients."""
    if volume.ndim != 5 or volume.shape[0] != 1:
        raise ValueError(f"Expected volume [1,C,D,H,W], got {tuple(volume.shape)}")
    d = torch.linspace(-0.5, 0.5, output_size[0], device=box.device, dtype=box.dtype)
    h = torch.linspace(-0.5, 0.5, output_size[1], device=box.device, dtype=box.dtype)
    w = torch.linspace(-0.5, 0.5, output_size[2], device=box.device, dtype=box.dtype)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing="ij")
    coordinate_d = box[0] + grid_d * box[3]
    coordinate_h = box[1] + grid_h * box[4]
    coordinate_w = box[2] + grid_w * box[5]
    grid = torch.stack((coordinate_w, coordinate_h, coordinate_d), dim=-1)
    grid = (2.0 * grid - 1.0).unsqueeze(0)
    return F.grid_sample(
        volume, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


class WholeCTJointModel(nn.Module):
    """Scan a complete CT, then jointly characterize all selected nodules."""

    def __init__(
        self,
        *,
        window_size: tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.25,
        roi_size: tuple[int, int, int] = (32, 32, 32),
        num_queries: int = 24,
        detr_hidden_dim: int = 128,
        detr_coarse_shape: tuple[int, int, int] = (12, 12, 12),
        detr_nheads: int = 4,
        detr_encoder_layers: int = 2,
        detr_decoder_layers: int = 2,
        mask_shape: tuple[int, int, int] = (64, 64, 64),
        vista_feature_dim: int = 48,
        vista_checkpoint: str | Path = DEFAULT_VISTA_CHECKPOINT,
        medicalnet_checkpoint: str | Path = DEFAULT_MEDICALNET_CHECKPOINT,
        segmenter_factory: Callable[[], nn.Module] | None = None,
        encoder_factory: Callable[[], nn.Module] | None = None,
        encoder_dim: int | None = None,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        self.window_size = tuple(int(value) for value in window_size)
        self.overlap = float(overlap)
        self.roi_size = tuple(int(value) for value in roi_size)
        self.num_queries = int(num_queries)
        self.mask_shape = tuple(int(value) for value in mask_shape)
        self.vista_feature_dim = int(vista_feature_dim)
        self.use_checkpoint = bool(use_checkpoint)
        if segmenter_factory is None:
            self.segmenter = VistaNoduleSegmenter(build_vista3d(vista_checkpoint))
        else:
            self.segmenter = segmenter_factory()
        if encoder_factory is None:
            self.medicalnet = MedicalNetEncoder(medicalnet_checkpoint)
            feature_dim = MedicalNetEncoder.output_dim
        else:
            self.medicalnet = encoder_factory()
            if encoder_dim is None:
                raise ValueError("encoder_dim is required with encoder_factory")
            feature_dim = int(encoder_dim)
        self.detr = NoduleMaskDETR3D(
            num_queries=self.num_queries,
            hidden_dim=detr_hidden_dim,
            coarse_shape=detr_coarse_shape,
            nheads=detr_nheads,
            encoder_layers=detr_encoder_layers,
            decoder_layers=detr_decoder_layers,
            input_channels=self.vista_feature_dim + 2,
        )
        self.semantic_head = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 6)
        )
        self.object_head = nn.Linear(feature_dim, 1)
        self.malignancy_head = nn.Sequential(
            nn.Linear(feature_dim + 7, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _window(self, image: Tensor, starts: tuple[int, int, int]) -> Tensor:
        slices = tuple(slice(start, min(start + size, length)) for start, size, length in zip(starts, self.window_size, image.shape[-3:]))
        patch = image[(...,) + slices]
        padding: list[int] = []
        for current, wanted in zip(patch.shape[-3:][::-1], self.window_size[::-1]):
            padding.extend((0, wanted - current))
        return F.pad(patch, padding, value=0.0)

    def _vista_window_summary(
        self,
        patch: Tensor,
        valid_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        if hasattr(self.segmenter, "forward_with_features"):
            logits, feature_pyramid = self.segmenter.forward_with_features(patch)
        else:
            logits = self.segmenter(patch)
            feature_pyramid = (logits,)
        valid = tuple(slice(0, length) for length in valid_shape)
        logits = logits[(...,) + valid]
        probability = F.interpolate(
            logits.sigmoid(), size=output_shape, mode="trilinear", align_corners=False
        )
        summarized_features: list[Tensor] = []
        patch_shape = patch.shape[-3:]
        for features in feature_pyramid:
            feature_valid_shape = tuple(
                max(1, int(np.ceil(valid * current / original)))
                for valid, current, original in zip(
                    valid_shape, features.shape[-3:], patch_shape
                )
            )
            feature_valid = tuple(slice(0, length) for length in feature_valid_shape)
            summarized_features.append(
                F.interpolate(
                    features[(...,) + feature_valid],
                    size=output_shape,
                    mode="trilinear",
                    align_corners=False,
                )
            )
        features = torch.cat(summarized_features, dim=1)
        return probability.float(), features.float()

    def discover(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        """Stitch one differentiable VISTA3D pass into a whole-CT soft feature grid."""
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected whole CT [1,D,H,W], got {tuple(image.shape)}")
        shape = tuple(int(value) for value in image.shape[-3:])
        probability_sum = torch.zeros((1, 1, *self.mask_shape), device=self.device)
        feature_sum = torch.zeros(
            (1, self.vista_feature_dim, *self.mask_shape), device=self.device
        )
        count = torch.zeros((1, 1, *self.mask_shape), device=self.device)
        axes = [sliding_starts(length, window, self.overlap) for length, window in zip(shape, self.window_size)]
        window_count = 0
        for d in axes[0]:
            for h in axes[1]:
                for w in axes[2]:
                    starts = (d, h, w)
                    valid_shape = tuple(
                        min(window, length - start)
                        for start, window, length in zip(starts, self.window_size, shape)
                    )
                    lower = tuple(
                        int(np.floor(start * target / length))
                        for start, target, length in zip(starts, self.mask_shape, shape)
                    )
                    upper = tuple(
                        max(lo + 1, int(np.ceil((start + valid) * target / length)))
                        for start, valid, target, length, lo in zip(
                            starts, valid_shape, self.mask_shape, shape, lower
                        )
                    )
                    block_shape = tuple(hi - lo for lo, hi in zip(lower, upper))
                    destination = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
                    patch = self._window(image, starts).unsqueeze(0).to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        if self.use_checkpoint and self.training:
                            patch.requires_grad_(True)

                            def summarize(
                                value: Tensor,
                                fixed_valid_shape: tuple[int, int, int] = valid_shape,
                                fixed_block_shape: tuple[int, int, int] = block_shape,
                            ) -> tuple[Tensor, Tensor]:
                                return self._vista_window_summary(
                                    value, fixed_valid_shape, fixed_block_shape
                                )

                            probability, features = checkpoint(summarize, patch, use_reentrant=False)
                        else:
                            probability, features = self._vista_window_summary(
                                patch, valid_shape, block_shape
                            )
                    probability_sum[(...,) + destination] += probability
                    feature_sum[(...,) + destination] += features
                    count[(...,) + destination] += 1
                    window_count += 1
        probability = probability_sum / count.clamp_min(1)
        features = feature_sum / count.clamp_min(1)
        ct = F.interpolate(
            image.unsqueeze(0).to(self.device, non_blocking=True),
            size=self.mask_shape,
            mode="trilinear",
            align_corners=False,
        )
        screening = torch.cat((ct, probability, features), dim=1)
        discovered = F.interpolate(
            probability.detach().cpu(), size=shape, mode="trilinear", align_corners=False
        )[0, 0] >= 0.5
        return screening, probability, discovered, window_count

    def _sampling_boxes(self, boxes: Tensor, image_shape: tuple[int, int, int]) -> Tensor:
        minimum = boxes.new_tensor(self.window_size) / boxes.new_tensor(image_shape)
        context_size = torch.maximum(boxes[:, 3:] * 2.0, minimum).clamp(max=1.0)
        return torch.cat((boxes[:, :3], context_size), dim=1)

    def _medical_forward(self, patch: Tensor, mask_logits: Tensor) -> Tensor:
        roi = patch * mask_logits.sigmoid()
        roi = F.interpolate(roi, self.roi_size, mode="trilinear", align_corners=False)
        return self.medicalnet(roi)

    def forward(self, image: Tensor) -> WholeCTOutput:
        if image.device.type != "cpu":
            raise ValueError("Whole CT must remain on CPU; windows are moved to the model device internally")
        screening, _, discovered, window_count = self.discover(image)
        detr_object_logits, boxes, mask_logits, _ = self.detr(screening)
        boxes = boxes[0]
        candidate_logits = mask_logits[0]
        sampling_boxes = self._sampling_boxes(boxes, tuple(int(v) for v in image.shape[-3:]))
        gpu_image = image.unsqueeze(0).to(self.device, non_blocking=True)
        feature_list: list[Tensor] = []
        for sampling_box, global_mask_logits in zip(sampling_boxes, candidate_logits):
            patch = differentiable_box_sample(gpu_image, sampling_box, self.window_size)
            local_mask_logits = differentiable_box_sample(
                global_mask_logits[None, None], sampling_box, self.window_size
            )
            if self.use_checkpoint and self.training:
                patch.requires_grad_(True)
                features = checkpoint(
                    self._medical_forward, patch, local_mask_logits, use_reentrant=False
                )
            else:
                features = self._medical_forward(patch, local_mask_logits)
            feature_list.append(features.reshape(-1))
        features = torch.stack(feature_list)
        six_features = 1.0 + 4.0 * self.semantic_head(features).sigmoid()
        full_voxels = float(np.prod(image.shape[-3:]))
        # Full scans contain tens of millions of voxels, beyond FP16's finite
        # range; keep physical mask volume in FP32 under autocast.
        volume = candidate_logits.float().sigmoid().flatten(1).mean(1) * full_voxels
        semantic = torch.cat((torch.log1p(volume).unsqueeze(1), six_features), dim=1)
        # DETR is the object detector; this second score lets the MedicalNet
        # representation also reject non-nodule queries.
        object_logits = detr_object_logits[0] + self.object_head(features).squeeze(1)
        nodule_logits = self.malignancy_head(torch.cat((features, semantic), dim=1)).squeeze(1)
        probability_per_nodule = nodule_logits.sigmoid() * object_logits.sigmoid()
        risk_probability = 1.0 - torch.exp(torch.log1p(-probability_per_nodule.clamp(1e-6, 1 - 1e-6)).sum())
        risk_logit = torch.logit(risk_probability.clamp(1e-6, 1 - 1e-6)).reshape(1)
        return WholeCTOutput(
            boxes, sampling_boxes,
            candidate_logits, object_logits, semantic, nodule_logits, risk_logit,
            discovered, window_count,
        )
