"""V3 reuses v2 geometry; diagnostics are parallel and half-frozen."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from back_prop.common.coarse_model import WholeCTJointModelV2, WholeCTOutput, _gaussian_weight
from back_prop.common.matcher import Match, match_coarse
from back_prop.common.roi import integer_crop, paste_compact_mask, sample_global_at_roi
from back_prop.common.base_model import DEFAULT_MEDICALNET_CHECKPOINT, sliding_starts
from back_prop.common.features import SoftRadiomics3D, MedicalNetSemantics, FEATURE_NAMES, RADIOMICS_NAMES
from back_prop.common.geometry import V3MaskDETR, V3Refiner
from back_prop.common.rashomon import ContinuousRashomonBank, scan_logits_per_model

ARCHITECTURE = 'whole_ct_parallel_soft_radiomics_medicalnet_continuous_rashomon_v3'

@dataclass
class WholeCTOutputV3(WholeCTOutput):
    ensemble_logits: Tensor
    scan_logits: Tensor
    model_indices: Tensor
    diagnostic_features: Tensor
    bank_ready: bool

class WholeCTJointModelV3(WholeCTJointModelV2):
    def __init__(self, *, medicalnet_checkpoint=DEFAULT_MEDICALNET_CHECKPOINT,
                 medicalnet_roi_size=64, encoder_factory=None, bank_config=None, semantics_factory=None,
                 amp_dtype=torch.bfloat16, **kwargs):
        # Avoid constructing/loading the replaced v2 diagnostic modules.
        super().__init__(radiomics_factory=nn.Identity, ordinal_heads_factory=nn.Identity, **kwargs)
        # Keep identical initial state and checkpoint keys while providing
        # interpolation kernels supported for BF16 on the installed Torch.
        detector = V3MaskDETR(num_queries=self.num_queries,
            hidden_dim=kwargs.get('detr_hidden_dim',128), coarse_shape=kwargs.get('detr_coarse_shape',(12,12,12)),
            nheads=kwargs.get('detr_nheads',4), encoder_layers=kwargs.get('detr_encoder_layers',2),
            decoder_layers=kwargs.get('detr_decoder_layers',2), input_channels=self.vista_feature_dim+2)
        detector.load_state_dict(self.detr.state_dict())
        self.detr = detector
        refiner = V3Refiner(input_channels=4+self.context_channels,
            query_dim=kwargs.get('detr_hidden_dim',128), base_channels=kwargs.get('refiner_base_channels',8),
            use_checkpoint=False)
        refiner.load_state_dict(self.refiner.state_dict())
        self.refiner = refiner
        self.amp_dtype = amp_dtype
        del self.ordinal_heads
        self.radiomics = SoftRadiomics3D()
        self.semantics = (semantics_factory() if semantics_factory is not None else
                          MedicalNetSemantics(medicalnet_checkpoint, medicalnet_roi_size,
                                              encoder_factory, self.use_checkpoint))
        self.rashomon = ContinuousRashomonBank(FEATURE_NAMES, **(bank_config or {}))

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
            logits.float().sigmoid(), size=output_shape, mode="trilinear", align_corners=False
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
                    features[(...,) + feature_valid].float(),
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
                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda", dtype=self.amp_dtype):
                        # Pyramid projections remain trainable during warmup.
                        # Their weight gradients would otherwise retain every
                        # window's full-resolution decoder activations.
                        segmenter_trainable = any(
                            parameter.requires_grad for parameter in self.segmenter.parameters()
                        )
                        if self.use_checkpoint and self.training and segmenter_trainable:

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

    def forward(
        self,
        image: Tensor,
        *,
        batch: dict[str, object] | None = None,
        epoch: int = 0,
        teacher_probability: float | None = None,
        object_threshold: float | None = None,
        compute_diagnostics: bool = True,
        update_bank: bool = True,
        image_hu: Tensor | None = None,
    ) -> WholeCTOutputV3:
        image_hu = batch.get('image_hu') if image_hu is None and batch is not None else image_hu
        if image_hu is None or image_hu.shape != image.shape or image_hu.device.type != 'cpu':
            raise ValueError('Provide original image_hu on the same CPU 1-mm grid as image')
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
        probability_list: list[Tensor] = []
        semantic_list: list[Tensor] = []
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
                context.float(), origin, scan_shape, self.roi_shape
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
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda", dtype=self.amp_dtype):
                if self.training and self.use_checkpoint:
                    delta = checkpoint(self.refiner, fine_input, queries[query_index][None], use_reentrant=False)
                else:
                    delta = self.refiner(fine_input, queries[query_index][None])
            fine_logit = (coarse_roi + delta.float()) * valid_gpu + (-12.0) * (
                1.0 - valid_gpu
            )
            # Parallel diagnostics: use original HU for radiomics and the
            # windowed network CT for MedicalNet, both on the same soft mask.
            hu_patch, _, _ = integer_crop(image_hu, center, self.roi_shape)
            hu_gpu = hu_patch.unsqueeze(0).to(self.device, non_blocking=True)
            if self.training and self.use_checkpoint:
                radiomics = checkpoint(self.radiomics, hu_gpu, fine_logit, valid_gpu, use_reentrant=False)
            else:
                radiomics = self.radiomics(hu_gpu, fine_logit, valid_gpu)
            probabilities, semantic_values = self.semantics(ct_gpu, fine_logit, valid_gpu)
            probability_list.append(probabilities[0])
            semantic_list.append(semantic_values[0])

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
            semantic_probability = torch.stack(probability_list)
            semantic = torch.stack(semantic_list)
        else:
            radiomics_features = object_logits.new_empty((0, len(RADIOMICS_NAMES)), dtype=torch.float32)
            semantic_probability = object_logits.new_empty((0, 6, 5), dtype=torch.float32)
            semantic = object_logits.new_empty((0, 6), dtype=torch.float32)
        diagnostic_features = torch.cat((radiomics_features, semantic), dim=1)
        if self.training and batch is not None and update_bank:
            rows = ((refined_target >= 0) & fine_supervision_valid).nonzero().flatten()
            targets = refined_target[rows]
            labels = batch['malignancy_targets'].to(self.device)[targets]
            keys = [(batch['case_id'], int(batch['nodule_ids'][int(t)])) for t in targets]
            self.rashomon.observe_and_refit(diagnostic_features[rows], labels, keys,
                                            refit=compute_diagnostics)
        ready = bool(int(self.rashomon.count)) and compute_diagnostics
        if not self.training and not ready:
            raise RuntimeError('Fit or load a training-only Rashomon bank before inference')
        if ready:
            ensemble_logits, model_indices = self.rashomon(diagnostic_features)
            scan_logits = scan_logits_per_model(ensemble_logits, object_logits[refined_query])
            nodule_logits = torch.logit(ensemble_logits.sigmoid().mean(1).clamp(1e-6, 1-1e-6))
            risk_logit = torch.logit(scan_logits.sigmoid().mean().clamp(1e-6, 1-1e-6)).reshape(1)
        else:
            ensemble_logits = diagnostic_features.new_empty((len(refined_query), 0))
            scan_logits = diagnostic_features.new_empty((0,))
            model_indices = torch.empty(0, device=self.device, dtype=torch.long)
            nodule_logits = diagnostic_features.sum(1) * 0.0
            risk_logit = (object_logits.sum() * 0.0).reshape(1)

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

        return WholeCTOutputV3(
            ensemble_logits=ensemble_logits,
            scan_logits=scan_logits,
            model_indices=model_indices,
            diagnostic_features=diagnostic_features,
            bank_ready=ready,
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
