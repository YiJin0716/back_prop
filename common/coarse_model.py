"""Coarse-to-fine, loss-connected whole-CT LIDC model (v2).

The complete once-resampled 1-mm CT is screened by one sliding VISTA3D pass.
A set-prediction detector produces coarse 64^3 proposals.  Each routed query
then receives an integer 1-mm ROI and an instance-conditioned residual mask
refinement.  Paper-compatible 2-D radiomics and seven ordinal heads convert
the soft fine mask into interpretable outputs and a scan-level reader-derived
malignancy surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from back_prop.common.base_model import (
    DEFAULT_VISTA_CHECKPOINT,
    NoduleMaskDETR3D,
    VistaNoduleSegmenter,
    build_vista3d,
    sliding_starts,
)
from back_prop.common.matcher import Match, identify_ignored_queries, match_coarse
from back_prop.common.refiner import FineMaskRefiner3D
from back_prop.common.roi import integer_crop, paste_compact_mask, sample_global_at_roi


ARCHITECTURE = "whole_ct_coarse_to_fine_radiomics_v2_m3_excluded"
DEFAULT_ORDINAL_INIT = Path(__file__).with_name("logistic_init.json")


@dataclass
class WholeCTOutput:
    boxes: Tensor
    object_logits: Tensor
    coarse_mask_logits: Tensor
    query_embeddings: Tensor
    matched_query_indices: Tensor
    matched_target_indices: Tensor
    refined_query_indices: Tensor
    refined_target_indices: Tensor
    crop_origins: Tensor
    roi_valid: Tensor
    roi_loss_valid: Tensor
    fine_supervision_valid: Tensor
    fine_mask_logits: Tensor
    fine_target_masks: Tensor
    semantic_probabilities: Tensor
    semantic_features: Tensor
    radiomics_features: Tensor
    nodule_logits: Tensor
    risk_logit: Tensor
    discovered_mask: Tensor
    window_count: int
    teacher_forced_count: int
    fallback_count: int
    scan_path_valid: bool


def _gaussian_weight(
    shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> Tensor:
    axes = [
        torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype)
        for length in shape
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    radius2 = sum(grid.square() for grid in grids)
    return torch.exp(-0.5 * radius2 / (0.5**2)).clamp_min(0.05)[None, None]


class WholeCTJointModelV2(nn.Module):
    """VISTA3D -> Mask-DETR -> native-grid fine mask -> radiomics -> ordinal."""

    def __init__(
        self,
        *,
        window_size: tuple[int, int, int] = (144, 160, 160),
        overlap: float = 0.25,
        screen_shape: tuple[int, int, int] = (64, 64, 64),
        roi_shape: tuple[int, int, int] = (128, 128, 128),
        num_queries: int = 24,
        detr_hidden_dim: int = 128,
        detr_coarse_shape: tuple[int, int, int] = (12, 12, 12),
        detr_nheads: int = 4,
        detr_encoder_layers: int = 2,
        detr_decoder_layers: int = 2,
        vista_feature_dim: int = 48,
        context_channels: int = 8,
        refiner_base_channels: int = 8,
        hard_negatives: int = 2,
        ignore_overlap_threshold: float = 0.1,
        teacher_full_epochs: int = 5,
        teacher_zero_epoch: int = 18,
        teacher_jitter: int = 8,
        minimum_roi_coverage: float = 0.8,
        inference_object_threshold: float = 0.5,
        vista_checkpoint: str | Path = DEFAULT_VISTA_CHECKPOINT,
        ordinal_init: str | Path = DEFAULT_ORDINAL_INIT,
        segmenter_factory: Callable[[], nn.Module] | None = None,
        radiomics_factory: Callable[[], nn.Module] | None = None,
        ordinal_heads_factory: Callable[[], nn.Module] | None = None,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0,1)")
        if teacher_zero_epoch <= teacher_full_epochs:
            raise ValueError("teacher_zero_epoch must exceed teacher_full_epochs")
        self.window_size = tuple(int(value) for value in window_size)
        self.overlap = float(overlap)
        self.screen_shape = tuple(int(value) for value in screen_shape)
        self.roi_shape = tuple(int(value) for value in roi_shape)
        self.num_queries = int(num_queries)
        self.vista_feature_dim = int(vista_feature_dim)
        self.context_channels = int(context_channels)
        self.hard_negatives = int(hard_negatives)
        self.ignore_overlap_threshold = float(ignore_overlap_threshold)
        self.teacher_full_epochs = int(teacher_full_epochs)
        self.teacher_zero_epoch = int(teacher_zero_epoch)
        self.teacher_jitter = int(teacher_jitter)
        self.minimum_roi_coverage = float(minimum_roi_coverage)
        self.inference_object_threshold = float(inference_object_threshold)
        self.use_checkpoint = bool(use_checkpoint)

        self.segmenter = (
            VistaNoduleSegmenter(build_vista3d(vista_checkpoint))
            if segmenter_factory is None
            else segmenter_factory()
        )
        screening_channels = self.vista_feature_dim + 2
        self.detr = NoduleMaskDETR3D(
            num_queries=self.num_queries,
            hidden_dim=detr_hidden_dim,
            coarse_shape=detr_coarse_shape,
            nheads=detr_nheads,
            encoder_layers=detr_encoder_layers,
            decoder_layers=detr_decoder_layers,
            input_channels=screening_channels,
        )
        # A fresh detector should initially expect roughly two objects among
        # 24 queries and nodule-sized boxes, rather than quarter-scan boxes.
        nn.init.constant_(self.detr.object_head.bias, -2.4)
        with torch.no_grad():
            self.detr.box_head.layers[-1].bias[3:].fill_(-2.7)
        self.context_compressor = nn.Conv3d(
            screening_channels, self.context_channels, kernel_size=1
        )
        fine_channels = 4 + self.context_channels  # CT, VISTA, coarse, valid, context
        self.refiner = FineMaskRefiner3D(
            input_channels=fine_channels,
            query_dim=detr_hidden_dim,
            base_channels=refiner_base_channels,
            use_checkpoint=use_checkpoint,
        )
        # V3/V4 supply their own diagnostic branches. Load the legacy
        # 64-feature implementation only when a V2 instance needs it.
        if radiomics_factory is None:
            from back_prop.common.radiomics import DifferentiableRadiomics64
            self.radiomics = DifferentiableRadiomics64()
        else:
            self.radiomics = radiomics_factory()
        if ordinal_heads_factory is None:
            from back_prop.common.ordinal import OrdinalRadiomicsHeads
            self.ordinal_heads = OrdinalRadiomicsHeads(ordinal_init)
        else:
            self.ordinal_heads = ordinal_heads_factory()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def teacher_probability(self, epoch: int) -> float:
        if epoch < self.teacher_full_epochs:
            return 1.0
        # ``teacher_zero_epoch`` is a human-facing one-based epoch count.  With
        # 18 training epochs, zero routing is therefore reached at index 17.
        zero_index = self.teacher_zero_epoch - 1
        if epoch >= zero_index:
            return 0.0
        span = zero_index - (self.teacher_full_epochs - 1)
        return float(zero_index - epoch) / float(span)

    def _window(self, image: Tensor, starts: tuple[int, int, int]) -> Tensor:
        slices = tuple(
            slice(start, min(start + size, length))
            for start, size, length in zip(starts, self.window_size, image.shape[-3:])
        )
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
            logits, pyramid = self.segmenter.forward_with_features(patch)
        else:
            logits = self.segmenter(patch)
            pyramid = (logits,)
        valid = tuple(slice(0, length) for length in valid_shape)
        logits = logits[(...,) + valid]
        probability = F.interpolate(
            logits.sigmoid(), size=output_shape, mode="trilinear", align_corners=False
        )
        summarized: list[Tensor] = []
        for features in pyramid:
            feature_valid_shape = tuple(
                max(1, int(np.ceil(valid_length * current / original)))
                for valid_length, current, original in zip(
                    valid_shape, features.shape[-3:], patch.shape[-3:]
                )
            )
            feature_valid = tuple(slice(0, length) for length in feature_valid_shape)
            summarized.append(
                F.interpolate(
                    features[(...,) + feature_valid],
                    size=output_shape,
                    mode="trilinear",
                    align_corners=False,
                )
            )
        features = torch.cat(summarized, dim=1)
        if features.shape[1] != self.vista_feature_dim:
            raise RuntimeError(
                f"VISTA feature channels {features.shape[1]} != {self.vista_feature_dim}"
            )
        return probability.float(), features.float()

    def discover(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        if image.ndim != 4 or image.shape[0] != 1 or image.device.type != "cpu":
            raise ValueError("Whole CT must be a CPU tensor [1,X,Y,Z]")
        shape = tuple(int(value) for value in image.shape[-3:])
        probability_sum = torch.zeros((1, 1, *self.screen_shape), device=self.device)
        feature_sum = torch.zeros(
            (1, self.vista_feature_dim, *self.screen_shape), device=self.device
        )
        weight_sum = torch.zeros((1, 1, *self.screen_shape), device=self.device)
        axes = [
            sliding_starts(length, window, self.overlap)
            for length, window in zip(shape, self.window_size)
        ]
        window_count = 0
        for x in axes[0]:
            for y in axes[1]:
                for z in axes[2]:
                    starts = (x, y, z)
                    valid_shape = tuple(
                        min(window, length - start)
                        for start, window, length in zip(starts, self.window_size, shape)
                    )
                    lower = tuple(
                        int(np.floor(start * target / length))
                        for start, target, length in zip(starts, self.screen_shape, shape)
                    )
                    upper = tuple(
                        max(lo + 1, int(np.ceil((start + valid) * target / length)))
                        for start, valid, target, length, lo in zip(
                            starts, valid_shape, self.screen_shape, shape, lower
                        )
                    )
                    block_shape = tuple(high - low for low, high in zip(lower, upper))
                    destination = tuple(slice(low, high) for low, high in zip(lower, upper))
                    patch = self._window(image, starts).unsqueeze(0).to(
                        self.device, non_blocking=True
                    )
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        backbone = getattr(self.segmenter, "model", self.segmenter)
                        backbone_trainable = any(
                            parameter.requires_grad for parameter in backbone.parameters()
                        )
                        if self.use_checkpoint and self.training and backbone_trainable:

                            def summarize(
                                value: Tensor,
                                fixed_valid_shape: tuple[int, int, int] = valid_shape,
                                fixed_block_shape: tuple[int, int, int] = block_shape,
                            ) -> tuple[Tensor, Tensor]:
                                return self._vista_window_summary(
                                    value, fixed_valid_shape, fixed_block_shape
                                )

                            probability, features = checkpoint(
                                summarize, patch, use_reentrant=False
                            )
                        else:
                            probability, features = self._vista_window_summary(
                                patch, valid_shape, block_shape
                            )
                    weight = _gaussian_weight(block_shape, self.device, probability.dtype)
                    probability_sum[(...,) + destination] += probability * weight
                    feature_sum[(...,) + destination] += features * weight
                    weight_sum[(...,) + destination] += weight
                    window_count += 1
        probability = probability_sum / weight_sum.clamp_min(1e-6)
        features = feature_sum / weight_sum.clamp_min(1e-6)
        ct = F.interpolate(
            image.unsqueeze(0).to(self.device, non_blocking=True),
            size=self.screen_shape,
            mode="trilinear",
            align_corners=False,
        )
        screening = torch.cat((ct, probability, features), dim=1)
        discovered = F.interpolate(
            probability.detach().cpu(), size=shape, mode="trilinear", align_corners=False
        )[0, 0] >= 0.5
        return screening, probability, discovered, window_count

    def _ignored_queries(
        self,
        boxes: Tensor,
        coarse_logits: Tensor,
        matched_queries: Tensor,
        batch: dict[str, object],
    ) -> Tensor:
        ignored_boxes = batch.get("ignored_target_boxes")
        ignored_masks = batch.get("ignored_target_masks")
        if not isinstance(ignored_boxes, Tensor) or not isinstance(ignored_masks, Tensor):
            return torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        return identify_ignored_queries(
            boxes,
            coarse_logits,
            matched_queries,
            ignored_boxes,
            ignored_masks,
            mask_dice_threshold=self.ignore_overlap_threshold,
        )

    def _route_training_queries(
        self,
        object_logits: Tensor,
        boxes: Tensor,
        coarse_logits: Tensor,
        match: Match,
        batch: dict[str, object],
    ) -> tuple[Tensor, Tensor]:
        selected_query = list(match.query_indices.tolist())
        selected_target = list(match.target_indices.tolist())
        unavailable = torch.zeros(len(object_logits), dtype=torch.bool, device=object_logits.device)
        unavailable[match.query_indices] = True
        unavailable |= self._ignored_queries(
            boxes, coarse_logits, match.query_indices, batch
        )
        candidates = (~unavailable).nonzero().flatten()
        count = min(self.hard_negatives, len(candidates))
        if count:
            order = object_logits[candidates].topk(count).indices
            selected_query.extend(candidates[order].tolist())
            selected_target.extend([-1] * count)
        return (
            torch.tensor(selected_query, dtype=torch.long, device=object_logits.device),
            torch.tensor(selected_target, dtype=torch.long, device=object_logits.device),
        )

    @staticmethod
    def _box_center_voxel(box: Tensor, scan_shape: tuple[int, int, int]) -> Tensor:
        shape = box.new_tensor(scan_shape)
        return box[:3] * shape - 0.5

    def _jittered_center(self, center: Tensor) -> Tensor:
        if self.teacher_jitter <= 0:
            return center.round().to(dtype=torch.long)
        jitter = torch.randint(
            -self.teacher_jitter,
            self.teacher_jitter + 1,
            (3,),
            device=center.device,
        )
        return center.round().to(dtype=torch.long) + jitter

    def _empty_downstream(self, reference: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        from back_prop.common.radiomics import FEATURE_NAMES
        features = reference.new_empty((0, len(FEATURE_NAMES)), dtype=torch.float32)
        probabilities = reference.new_empty((0, 7, 5), dtype=torch.float32)
        expected = reference.new_empty((0, 7), dtype=torch.float32)
        logits = reference.new_empty((0,), dtype=torch.float32)
        return features, probabilities, expected, logits

    def forward(
        self,
        image: Tensor,
        *,
        batch: dict[str, object] | None = None,
        epoch: int = 0,
        teacher_probability: float | None = None,
        object_threshold: float | None = None,
        compute_diagnostics: bool = True,
    ) -> WholeCTOutput:
        screening, vista_probability, discovered, window_count = self.discover(image)
        detr_object, boxes_batched, masks_batched, queries_batched = self.detr(screening)
        object_logits = detr_object[0]
        boxes = boxes_batched[0]
        coarse_logits = masks_batched[0]
        queries = queries_batched[0]
        context = self.context_compressor(screening)
        scan_shape = tuple(int(value) for value in image.shape[-3:])

        if batch is not None:
            target_boxes = batch["target_boxes"].to(self.device, non_blocking=True)
            target_masks = batch["target_masks"].to(self.device, non_blocking=True)
            match = match_coarse(
                object_logits, boxes, coarse_logits, target_boxes, target_masks
            )
            refined_query, refined_target = self._route_training_queries(
                object_logits, boxes, coarse_logits, match, batch
            )
            p_teacher = (
                self.teacher_probability(epoch)
                if teacher_probability is None
                else float(teacher_probability)
            )
        else:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            match = Match(empty, empty)
            threshold = (
                self.inference_object_threshold
                if object_threshold is None
                else float(object_threshold)
            )
            refined_query = (object_logits.sigmoid() >= threshold).nonzero().flatten()
            refined_target = torch.full_like(refined_query, -1)
            p_teacher = 0.0

        origins: list[Tensor] = []
        valid_masks: list[Tensor] = []
        loss_valid_masks: list[Tensor] = []
        fine_logits_list: list[Tensor] = []
        fine_targets: list[Tensor] = []
        supervision_validity: list[bool] = []
        radiomics_list: list[Tensor] = []
        teacher_count = 0
        fallback_count = 0

        for query_index, target_index in zip(
            refined_query.tolist(), refined_target.tolist()
        ):
            predicted_center = self._box_center_voxel(boxes[query_index], scan_shape)
            use_teacher = target_index >= 0 and bool(
                torch.rand((), device=self.device) < p_teacher
            )
            if use_teacher:
                target_box = batch["target_boxes"][target_index].to(self.device)
                center = self._jittered_center(
                    self._box_center_voxel(target_box, scan_shape)
                )
                teacher_count += 1
            else:
                center = predicted_center.detach().round().to(dtype=torch.long)

            ct_patch, origin, valid = integer_crop(image, center, self.roi_shape)
            loss_valid = valid.clone()
            ignored_crops = batch.get("ignored_target_mask_crops") if batch is not None else None
            ignored_origins = batch.get("ignored_target_mask_origins") if batch is not None else None
            if isinstance(ignored_crops, list) and isinstance(ignored_origins, Tensor):
                ignored_local = torch.zeros(self.roi_shape, dtype=torch.bool)
                for ignored_crop, ignored_origin in zip(ignored_crops, ignored_origins):
                    ignored_local |= paste_compact_mask(
                        ignored_crop, ignored_origin, origin, self.roi_shape
                    ).bool()
                # The forward path must never see a GT-derived hole.  Ignore
                # masks affect loss support only, avoiding label leakage and a
                # train/inference input shift.
                loss_valid = valid & ~ignored_local.unsqueeze(0)
            target = torch.zeros(self.roi_shape, dtype=torch.float32)
            supervision_valid = True
            if target_index >= 0:
                target = paste_compact_mask(
                    batch["target_mask_crops"][target_index],
                    batch["target_mask_origins"][target_index],
                    origin,
                    self.roi_shape,
                ).float()
                full_target_voxels = float(
                    batch["target_mask_crops"][target_index].sum().item()
                )
                coverage = float(target.sum().item()) / max(full_target_voxels, 1.0)
                if (
                    not use_teacher
                    and coverage < self.minimum_roi_coverage
                    and self.training
                    and p_teacher > 0.0
                ):
                    target_box = batch["target_boxes"][target_index].to(self.device)
                    center = self._jittered_center(
                        self._box_center_voxel(target_box, scan_shape)
                    )
                    ct_patch, origin, valid = integer_crop(image, center, self.roi_shape)
                    loss_valid = valid.clone()
                    if isinstance(ignored_crops, list) and isinstance(ignored_origins, Tensor):
                        ignored_local = torch.zeros(self.roi_shape, dtype=torch.bool)
                        for ignored_crop, ignored_origin in zip(ignored_crops, ignored_origins):
                            ignored_local |= paste_compact_mask(
                                ignored_crop, ignored_origin, origin, self.roi_shape
                            ).bool()
                        loss_valid = valid & ~ignored_local.unsqueeze(0)
                    target = paste_compact_mask(
                        batch["target_mask_crops"][target_index],
                        batch["target_mask_origins"][target_index],
                        origin,
                        self.roi_shape,
                    ).float()
                    fallback_count += 1
                    teacher_count += 1
                    coverage = float(target.sum().item()) / max(full_target_voxels, 1.0)
                supervision_valid = coverage >= self.minimum_roi_coverage

            coarse_roi = sample_global_at_roi(
                coarse_logits[query_index][None, None], origin, scan_shape, self.roi_shape
            )
            vista_roi = sample_global_at_roi(
                vista_probability, origin, scan_shape, self.roi_shape
            )
            context_roi = sample_global_at_roi(
                context, origin, scan_shape, self.roi_shape
            )
            ct_gpu = ct_patch.unsqueeze(0).to(self.device, non_blocking=True).float()
            valid_gpu = valid.unsqueeze(0).to(self.device, non_blocking=True).float()
            # grid_sample pads with numeric zero, but zero is a 0.5 mask
            # probability when interpreted as a logit.  Explicitly make
            # out-of-scan mask logits background before refinement/radiomics.
            coarse_roi = coarse_roi.float() * valid_gpu + (-12.0) * (1.0 - valid_gpu)
            fine_input = torch.cat(
                (ct_gpu, vista_roi, coarse_roi.sigmoid(), valid_gpu, context_roi), dim=1
            )
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                delta = self.refiner(fine_input, queries[query_index][None])
            fine_logit = (coarse_roi + delta.float()) * valid_gpu + (-12.0) * (
                1.0 - valid_gpu
            )
            # Do not activation-checkpoint this exact-forward extractor: doing
            # so would repeat its NumPy/CPU calculation during backward.
            radiomics = (
                self.radiomics(ct_gpu.float(), fine_logit)
                if compute_diagnostics else None
            )

            origins.append(origin.to(self.device))
            valid_masks.append(valid_gpu[0, 0].bool())
            loss_valid_masks.append(
                loss_valid.to(self.device, non_blocking=True)[0].bool()
            )
            fine_logits_list.append(fine_logit[0, 0])
            fine_targets.append(target.to(self.device, non_blocking=True))
            supervision_validity.append(supervision_valid)
            if radiomics is not None:
                radiomics_list.append(radiomics[0])

        if fine_logits_list:
            fine_mask_logits = torch.stack(fine_logits_list)
            fine_target_masks = torch.stack(fine_targets)
            roi_valid = torch.stack(valid_masks)
            roi_loss_valid = torch.stack(loss_valid_masks)
            fine_supervision_valid = torch.tensor(
                supervision_validity, dtype=torch.bool, device=self.device
            )
            crop_origins = torch.stack(origins).to(dtype=torch.long)
        else:
            fine_mask_logits = object_logits.new_empty((0, *self.roi_shape))
            fine_target_masks = object_logits.new_empty((0, *self.roi_shape))
            roi_valid = torch.empty(
                (0, *self.roi_shape), dtype=torch.bool, device=self.device
            )
            roi_loss_valid = torch.empty(
                (0, *self.roi_shape), dtype=torch.bool, device=self.device
            )
            fine_supervision_valid = torch.empty(
                (0,), dtype=torch.bool, device=self.device
            )
            crop_origins = torch.empty((0, 3), dtype=torch.long, device=self.device)

        if radiomics_list:
            radiomics_features = torch.stack(radiomics_list)
            semantic_probability, semantic = self.ordinal_heads(radiomics_features)
            malignant_probability = semantic_probability[:, 1, 3:].sum(dim=-1).clamp(
                1e-6, 1.0 - 1e-6
            )
            nodule_logits = torch.logit(malignant_probability)
            per_query = malignant_probability * object_logits[refined_query].sigmoid()
            log_survival = torch.log1p(-per_query.clamp(max=1.0 - 1e-6)).sum()
            risk_probability = (-torch.expm1(log_survival)).clamp(1e-6, 1.0 - 1e-6)
            risk_logit = torch.logit(risk_probability).reshape(1)
        else:
            radiomics_features, semantic_probability, semantic, nodule_logits = (
                self._empty_downstream(object_logits)
            )
            risk_logit = (object_logits.sum() * 0.0 - 13.8155).reshape(1)

        scan_path_valid = True
        if batch is not None:
            for routed_target, valid in zip(refined_target.tolist(), supervision_validity):
                # A scan-level diagnostic loss is meaningful only when every
                # retained matched nodule reaches its intended fine ROI.
                # Otherwise noisy-OR would train an unrelated crop as benign
                # or malignant even though integer routing has no gradient.
                if routed_target >= 0 and not valid:
                    scan_path_valid = False
                    break

        return WholeCTOutput(
            boxes=boxes,
            object_logits=object_logits,
            coarse_mask_logits=coarse_logits,
            query_embeddings=queries,
            matched_query_indices=match.query_indices,
            matched_target_indices=match.target_indices,
            refined_query_indices=refined_query,
            refined_target_indices=refined_target,
            crop_origins=crop_origins,
            roi_valid=roi_valid,
            roi_loss_valid=roi_loss_valid,
            fine_supervision_valid=fine_supervision_valid,
            fine_mask_logits=fine_mask_logits,
            fine_target_masks=fine_target_masks,
            semantic_probabilities=semantic_probability,
            semantic_features=semantic,
            radiomics_features=radiomics_features,
            nodule_logits=nodule_logits,
            risk_logit=risk_logit,
            discovered_mask=discovered,
            window_count=window_count,
            teacher_forced_count=teacher_count,
            fallback_count=fallback_count,
            scan_path_valid=scan_path_valid,
        )


__all__ = ("ARCHITECTURE", "WholeCTJointModelV2", "WholeCTOutput")
