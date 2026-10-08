"""V3 geometry and MedicalNet; sequential radiomics then semantic residual risk."""
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils._pytree import register_dataclass
from back_prop.common.semantic_model import WholeCTJointModelV3, WholeCTOutputV3
from back_prop.common.features import RADIOMICS_NAMES
from back_prop.common.rashomon import scan_logits_per_model
from back_prop.common.matcher import Match, match_coarse
from back_prop.common.roi import integer_crop, paste_compact_mask, sample_global_at_roi
from .risk import ResidualRiskBank
from .features import MedicalNetSemantics

LEGACY_ARCHITECTURE = 'whole_ct_radiomics_offset_groupnorm_early_diagnostics_v4'
ARCHITECTURE = 'whole_ct_radiomics_offset_ws_groupnorm_early_diagnostics_v4'


@dataclass
class WholeCTOutputV4(WholeCTOutputV3):
    radiomics_logits: Tensor
    semantic_correction_logits: Tensor
    residual_targets: Tensor
    predicted_residuals: Tensor
    baseline_ready: bool


# DDP's unused-parameter search traverses dataclasses, but its output sink uses
# pytree flattening. Register all inherited fields so the sink can propagate
# undefined gradients when ROI coverage disables the semantic/risk losses.
# Otherwise DDP waits for MedicalNet gradients that will never be produced.
register_dataclass(WholeCTOutputV4)


class WholeCTJointModelV4(WholeCTJointModelV3):
    def __init__(self, *, bank_config=None, baseline_l2=1e-3,
                 semantic_weight_standardization=True, **kwargs):
        super().__init__(bank_config=bank_config, **kwargs)
        self.semantics = MedicalNetSemantics(self.semantics,
            weight_standardization=semantic_weight_standardization)
        self.rashomon = ResidualRiskBank(baseline_l2=baseline_l2, **(bank_config or {}))
        # V3's original sigmoid temperature is fixed for every epoch and saved.
        self.register_buffer('mask_temperature', torch.tensor(1.0))

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
        segmentation_only: bool = False,
    ) -> WholeCTOutputV4:
        if segmentation_only:
            if self.training or batch is not None:
                raise ValueError('Segmentation-only mining requires eval mode and no GT routing')
            compute_diagnostics = False
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
        semantic_inputs = []
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
            # Radiomics is a detached baseline only, never a segmentation objective.
            with torch.no_grad():
                if segmentation_only:
                    radiomics = fine_logit.new_zeros((1, len(RADIOMICS_NAMES)))
                else:
                    hu_patch, _, _ = integer_crop(image_hu, center, self.roi_shape)
                    hu_gpu = hu_patch.unsqueeze(0).to(self.device, non_blocking=True)
                    radiomics = self.radiomics(hu_gpu, fine_logit.detach(), valid_gpu)
            if not segmentation_only:
                semantic_inputs.append((ct_gpu, fine_logit, valid_gpu))

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

        radiomics_features = (torch.stack(radiomics_list) if radiomics_list else
            object_logits.new_empty((0, len(RADIOMICS_NAMES)), dtype=torch.float32))
        rows = ((refined_target >= 0) & fine_supervision_valid).nonzero().flatten()
        if self.training and batch is not None and update_bank:
            targets = refined_target[rows]
            labels = batch['malignancy_targets'].to(self.device)[targets]
            keys = [(batch['case_id'], int(batch['nodule_ids'][int(t)])) for t in targets]
            self.rashomon.observe_radiomics(radiomics_features[rows], labels, keys,
                                            refit=compute_diagnostics)
        baseline_ready = bool(int(self.rashomon.baseline_count)) and compute_diagnostics
        radiomics_logits = self.rashomon.baseline_logits(radiomics_features)
        residual_targets = torch.zeros_like(radiomics_logits)
        if baseline_ready and batch is not None and len(rows):
            labels = batch['malignancy_targets'].to(self.device)[refined_target[rows]]
            residual_targets[rows] = labels - radiomics_logits[rows].detach().sigmoid()

        # The second stage sees semantic features only and the fixed baseline;
        # radiomics descriptors are never eligible for its sparse support.
        for inputs in semantic_inputs:
            probabilities, semantic_values = self.semantics(*inputs)
            probability_list.append(probabilities[0])
            semantic_list.append(semantic_values[0])
        semantic_probability = (torch.stack(probability_list) if probability_list else
            object_logits.new_zeros((len(refined_query), 6, 5), dtype=torch.float32))
        semantic = (torch.stack(semantic_list) if semantic_list else
            object_logits.new_zeros((len(refined_query), 6), dtype=torch.float32))
        if self.training and batch is not None and update_bank:
            self.rashomon.observe_semantics(semantic[rows], labels, keys,
                                            refit=compute_diagnostics)
        diagnostic_features = torch.cat((radiomics_features, semantic), dim=1)
        ready = bool(int(self.rashomon.count)) and baseline_ready
        if not self.training and compute_diagnostics and not ready:
            raise RuntimeError('Fit or load both training-only V4 risk stages before inference')
        if ready:
            ensemble_logits, model_indices, semantic_correction_logits = self.rashomon(
                semantic, radiomics_logits)
            predicted_residuals = ensemble_logits.sigmoid() - radiomics_logits.detach().sigmoid()[:, None]
            scan_logits = scan_logits_per_model(ensemble_logits, object_logits[refined_query])
            nodule_logits = torch.logit(ensemble_logits.sigmoid().mean(1).clamp(1e-6, 1-1e-6))
            risk_logit = torch.logit(scan_logits.sigmoid().mean().clamp(1e-6, 1-1e-6)).reshape(1)
        else:
            ensemble_logits = diagnostic_features.new_empty((len(refined_query), 0))
            semantic_correction_logits = ensemble_logits
            predicted_residuals = ensemble_logits
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

        return WholeCTOutputV4(
            radiomics_logits=radiomics_logits,
            semantic_correction_logits=semantic_correction_logits,
            residual_targets=residual_targets,
            predicted_residuals=predicted_residuals,
            baseline_ready=baseline_ready,
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
