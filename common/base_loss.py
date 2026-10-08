"""Three-term objective for whole-CT joint training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from back_prop.common.base_model import WholeCTOutput


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-5) -> Tensor:
    probability = logits.float().sigmoid()
    target = target.to(probability)
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


@dataclass
class JointLoss:
    total: Tensor
    semantic: Tensor
    segmentation: Tensor
    malignancy: Tensor
    risk: Tensor
    candidates: int
    positive_candidates: int
    risk_valid: bool = True
    ignored_candidates: int = 0


class WholeCTCriterion(nn.Module):
    """Semantic, Mask-DETR Dice/no-object, and malignancy objective."""

    def __init__(
        self,
        window_size: tuple[int, int, int],
        weights: tuple[float, float, float] = (1, 1, 1),
        no_object_weight: float = 0.1,
        ignore_overlap_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("loss weights must be non-negative and not all zero")
        if no_object_weight <= 0:
            raise ValueError("no_object_weight must be positive")
        self.window_size = tuple(int(value) for value in window_size)
        self.no_object_weight = float(no_object_weight)
        if not 0.0 <= ignore_overlap_threshold <= 1.0:
            raise ValueError("ignore_overlap_threshold must be in [0, 1]")
        self.ignore_overlap_threshold = float(ignore_overlap_threshold)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    @staticmethod
    def _box_corners(boxes: Tensor) -> tuple[Tensor, Tensor]:
        half = boxes[..., 3:] * 0.5
        return boxes[..., :3] - half, boxes[..., :3] + half

    @classmethod
    def _generalized_iou(cls, first: Tensor, second: Tensor) -> Tensor:
        first_min, first_max = cls._box_corners(first)
        second_min, second_max = cls._box_corners(second)
        intersection_min = torch.maximum(first_min[:, None], second_min[None])
        intersection_max = torch.minimum(first_max[:, None], second_max[None])
        intersection = (intersection_max - intersection_min).clamp_min(0).prod(-1)
        first_volume = (first_max - first_min).clamp_min(0).prod(-1)[:, None]
        second_volume = (second_max - second_min).clamp_min(0).prod(-1)[None]
        union = first_volume + second_volume - intersection
        iou = intersection / union.clamp_min(1e-7)
        enclosing_min = torch.minimum(first_min[:, None], second_min[None])
        enclosing_max = torch.maximum(first_max[:, None], second_max[None])
        enclosing = (enclosing_max - enclosing_min).clamp_min(0).prod(-1)
        return iou - (enclosing - union) / enclosing.clamp_min(1e-7)

    @staticmethod
    def _pairwise_dice_cost(predicted_logits: Tensor, targets: Tensor) -> Tensor:
        predicted = predicted_logits.float().sigmoid().flatten(1)
        targets = targets.float().flatten(1)
        intersection = predicted @ targets.transpose(0, 1)
        denominator = predicted.sum(1, keepdim=True) + targets.sum(1).unsqueeze(0)
        return 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)

    def _match(
        self, output: WholeCTOutput, batch: dict[str, object]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = output.boxes.device
        target_boxes = batch["target_boxes"].to(device, non_blocking=True)
        target_masks = batch["target_masks"].to(device, non_blocking=True)
        if tuple(target_masks.shape[-3:]) != tuple(output.candidate_logits.shape[-3:]):
            raise ValueError(
                f"Target mask shape {tuple(target_masks.shape[-3:])} does not match "
                f"model output {tuple(output.candidate_logits.shape[-3:])}"
            )
        if not len(target_boxes):
            empty = torch.empty((0,), device=device, dtype=torch.long)
            return empty, empty, target_boxes, target_masks
        with torch.no_grad():
            class_cost = -output.object_logits.sigmoid()[:, None]
            l1_cost = torch.cdist(output.boxes.float(), target_boxes, p=1)
            mask_cost = self._pairwise_dice_cost(output.candidate_logits, target_masks)
            cost = class_cost + l1_cost + 2.0 * mask_cost
            query_numpy, target_numpy = linear_sum_assignment(cost.cpu().numpy())
        query_indices = torch.as_tensor(query_numpy, device=device, dtype=torch.long)
        target_indices = torch.as_tensor(target_numpy, device=device, dtype=torch.long)
        return query_indices, target_indices, target_boxes, target_masks

    def _ignored_query_mask(
        self,
        output: WholeCTOutput,
        batch: dict[str, object],
        retained_queries: Tensor,
    ) -> Tensor:
        """Shield at least one unmatched query for every excluded nodule.

        Soft Dice alone is nearly zero for diffuse early masks, so ignored
        nodules are assigned using box+mask Hungarian cost, supplemented by
        centre containment and the configured mask-overlap threshold.
        """
        device = output.boxes.device
        result = torch.zeros(len(output.object_logits), dtype=torch.bool, device=device)
        ignored_boxes = batch.get("ignored_target_boxes")
        ignored_masks = batch.get("ignored_target_masks")
        if (
            not isinstance(ignored_boxes, Tensor)
            or not isinstance(ignored_masks, Tensor)
            or not len(ignored_boxes)
        ):
            return result
        ignored_boxes = ignored_boxes.to(device=device, dtype=torch.float32)
        ignored_masks = ignored_masks.to(device, non_blocking=True)
        with torch.no_grad():
            dice = 1.0 - self._pairwise_dice_cost(
                output.candidate_logits, ignored_masks
            )
            ignored_min, ignored_max = self._box_corners(ignored_boxes)
            centres = output.boxes[:, :3]
            centre_inside = (
                (centres[:, None] >= ignored_min[None])
                & (centres[:, None] <= ignored_max[None])
            ).all(dim=-1).any(dim=1)
            result |= dice.max(dim=1).values >= self.ignore_overlap_threshold
            result |= centre_inside
            available_mask = torch.ones(len(result), dtype=torch.bool, device=device)
            available_mask[retained_queries] = False
            available = available_mask.nonzero().flatten()
            if len(available):
                l1 = torch.cdist(output.boxes[available].float(), ignored_boxes, p=1)
                giou = self._generalized_iou(
                    output.boxes[available].float(), ignored_boxes
                )
                cost = 5.0 * l1 + 2.0 * (1.0 - giou) + 2.0 * (
                    1.0 - dice[available]
                )
                forced_rows, _ = linear_sum_assignment(cost.cpu().numpy())
                result[available[torch.as_tensor(forced_rows, device=device)]] = True
            result[retained_queries] = False
        return result

    def forward(self, output: WholeCTOutput, batch: dict[str, object]) -> JointLoss:
        device = output.candidate_logits.device
        query_indices, target_indices, _, target_masks = self._match(output, batch)
        object_target = torch.zeros_like(output.object_logits, dtype=torch.float32)
        object_target[query_indices] = 1.0
        object_weights = torch.full_like(object_target, self.no_object_weight)
        object_weights[query_indices] = 1.0
        ignored_candidates = self._ignored_query_mask(output, batch, query_indices)
        if ignored_candidates.any():
            object_weights[ignored_candidates] = 0.0
        object_loss = F.binary_cross_entropy_with_logits(
            output.object_logits.float(), object_target, reduction="none"
        )
        object_loss = (object_loss * object_weights).sum() / object_weights.sum().clamp_min(1e-7)

        if len(query_indices):
            predicted_masks = output.candidate_logits[query_indices]
            masks = target_masks[target_indices]
            segmentation = soft_dice_loss(predicted_masks, masks) + object_loss

            semantic_target = batch["semantic_targets"].to(device)[target_indices]
            semantic_prediction = output.semantic_features[query_indices].float()
            valid_semantic = torch.isfinite(semantic_target)
            semantic = (
                F.smooth_l1_loss(semantic_prediction[valid_semantic], semantic_target[valid_semantic])
                if valid_semantic.any() else semantic_prediction.sum() * 0.0
            )
            malignancy_target = batch["malignancy_targets"].to(device)[target_indices]
            valid_malignancy = torch.isfinite(malignancy_target)
            nodule = (
                F.binary_cross_entropy_with_logits(
                    output.nodule_logits[query_indices][valid_malignancy].float(),
                    malignancy_target[valid_malignancy].float(),
                )
                if valid_malignancy.any() else output.nodule_logits.sum() * 0.0
            )
        else:
            segmentation = output.candidate_logits.sum() * 0.0 + object_loss
            semantic = output.semantic_features.sum() * 0.0
            nodule = output.nodule_logits.sum() * 0.0

        risk_target = batch["risk_target"].to(device).reshape_as(output.risk_logit)
        risk_valid_value = batch.get("risk_target_valid", True)
        risk_valid = (
            bool(risk_valid_value.item())
            if isinstance(risk_valid_value, Tensor) else bool(risk_valid_value)
        )
        risk = (
            F.binary_cross_entropy_with_logits(output.risk_logit.float(), risk_target.float())
            if risk_valid else output.risk_logit.float().sum() * 0.0
        )
        malignancy = 0.5 * (nodule + risk) if risk_valid else nodule
        parts = torch.stack((semantic, segmentation, malignancy))
        weights = self.weights.to(parts)
        total = (parts * weights).sum() / weights.sum()
        return JointLoss(
            total, semantic, segmentation, malignancy, risk,
            int(len(output.object_logits)), int(len(query_indices)),
            risk_valid, int(ignored_candidates.sum().item()),
        )
