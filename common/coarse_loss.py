"""Explicit coarse, fine, ordinal and scan objectives for model v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from back_prop.common.matcher import identify_ignored_queries, pairwise_generalized_iou
from back_prop.common.coarse_model import WholeCTOutput


def sigmoid_focal_loss(
    logits: Tensor,
    target: Tensor,
    *,
    valid: Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    logits = logits.float()
    target = target.to(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = logits.sigmoid()
    p_target = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_target = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_target * (1.0 - p_target).pow(gamma) * cross_entropy
    if valid is None:
        return loss.mean()
    weight = valid.to(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def masked_dice_loss(logits: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
    probability = logits.float().sigmoid()
    target = target.to(probability)
    if valid is not None:
        weight = valid.to(probability)
        probability = probability * weight
        target = target * weight
    intersection = (probability * target).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return (1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()


@dataclass
class V2Loss:
    total: Tensor
    object: Tensor
    box_l1: Tensor
    box_giou: Tensor
    coarse_dice: Tensor
    coarse_focal: Tensor
    fine_dice: Tensor
    fine_focal: Tensor
    ordinal: Tensor
    nodule: Tensor
    risk: Tensor
    anchor: Tensor
    positive_candidates: int
    refined_candidates: int
    ignored_candidates: int
    risk_valid: bool


class WholeCTCriterionV2(nn.Module):
    """Loss used after the model's single coarse Hungarian assignment."""

    def __init__(
        self,
        *,
        object_weight: float = 1.0,
        box_l1_weight: float = 5.0,
        box_giou_weight: float = 2.0,
        coarse_dice_weight: float = 1.0,
        coarse_focal_weight: float = 1.0,
        fine_dice_weight: float = 2.0,
        fine_focal_weight: float = 1.0,
        ordinal_weight: float = 1.0,
        nodule_weight: float = 0.5,
        risk_weight: float = 0.5,
        anchor_weight: float = 1e-4,
        no_object_weight: float = 0.1,
        hard_negative_fine_weight: float = 0.25,
        ignore_overlap_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.loss_weights = {
            "object": float(object_weight),
            "box_l1": float(box_l1_weight),
            "box_giou": float(box_giou_weight),
            "coarse_dice": float(coarse_dice_weight),
            "coarse_focal": float(coarse_focal_weight),
            "fine_dice": float(fine_dice_weight),
            "fine_focal": float(fine_focal_weight),
            "ordinal": float(ordinal_weight),
            "nodule": float(nodule_weight),
            "risk": float(risk_weight),
            "anchor": float(anchor_weight),
        }
        if any(value < 0 for value in self.loss_weights.values()):
            raise ValueError("Loss weights must be non-negative")
        self.no_object_weight = float(no_object_weight)
        self.hard_negative_fine_weight = float(hard_negative_fine_weight)
        self.ignore_overlap_threshold = float(ignore_overlap_threshold)

    @staticmethod
    def _zero(reference: Tensor) -> Tensor:
        return reference.float().sum() * 0.0

    def forward(
        self,
        output: WholeCTOutput,
        batch: dict[str, object],
        ordinal_anchor: Tensor | None = None,
        diagnostic_scale: float = 1.0,
    ) -> V2Loss:
        device = output.object_logits.device
        query = output.matched_query_indices
        target_index = output.matched_target_indices
        target_boxes = batch["target_boxes"].to(device, non_blocking=True).float()
        target_masks = batch["target_masks"].to(device, non_blocking=True).float()

        object_target = torch.zeros_like(output.object_logits, dtype=torch.float32)
        object_target[query] = 1.0
        object_weights = torch.full_like(object_target, self.no_object_weight)
        object_weights[query] = 1.0
        ignored = torch.zeros_like(object_target, dtype=torch.bool)
        ignored_boxes = batch.get("ignored_target_boxes")
        ignored_masks = batch.get("ignored_target_masks")
        if (
            isinstance(ignored_boxes, Tensor)
            and isinstance(ignored_masks, Tensor)
            and len(ignored_boxes)
        ):
            ignored = identify_ignored_queries(
                output.boxes,
                output.coarse_mask_logits,
                query,
                ignored_boxes,
                ignored_masks,
                mask_dice_threshold=self.ignore_overlap_threshold,
            )
            object_weights[ignored] = 0.0
        object_raw = F.binary_cross_entropy_with_logits(
            output.object_logits.float(), object_target, reduction="none"
        )
        object_loss = (object_raw * object_weights).sum() / object_weights.sum().clamp_min(1.0)

        if len(query):
            prediction_boxes = output.boxes[query].float()
            boxes = target_boxes[target_index]
            box_l1 = F.l1_loss(prediction_boxes, boxes)
            giou = pairwise_generalized_iou(prediction_boxes, boxes)
            box_giou = (1.0 - giou.diagonal()).mean()
            coarse_prediction = output.coarse_mask_logits[query]
            coarse_target = target_masks[target_index]
            coarse_dice = masked_dice_loss(coarse_prediction, coarse_target)
            coarse_focal = sigmoid_focal_loss(coarse_prediction, coarse_target)
        else:
            box_l1 = self._zero(output.boxes)
            box_giou = self._zero(output.boxes)
            coarse_dice = self._zero(output.coarse_mask_logits)
            coarse_focal = self._zero(output.coarse_mask_logits)

        positive_rows = (
            (output.refined_target_indices >= 0) & output.fine_supervision_valid
        ).nonzero().flatten()
        negative_rows = (output.refined_target_indices < 0).nonzero().flatten()
        if len(positive_rows):
            positive_fine = output.fine_mask_logits[positive_rows]
            positive_target = output.fine_target_masks[positive_rows]
            positive_valid = output.roi_loss_valid[positive_rows]
            fine_dice = masked_dice_loss(positive_fine, positive_target, positive_valid)
            positive_focal = sigmoid_focal_loss(
                positive_fine, positive_target, valid=positive_valid
            )

            semantic_target_indices = output.refined_target_indices[positive_rows]
            if diagnostic_scale > 0.0:
                histograms = batch["semantic_histograms"].to(device, non_blocking=True)[
                    semantic_target_indices
                ]
                probabilities = output.semantic_probabilities[positive_rows].float().clamp_min(1e-7)
                ordinal = -(histograms * probabilities.log()).sum(dim=-1).mean()
                binary_target = batch["malignancy_targets"].to(device, non_blocking=True)[
                    semantic_target_indices
                ]
                nodule = F.binary_cross_entropy_with_logits(
                    output.nodule_logits[positive_rows].float(), binary_target.float()
                )
            else:
                ordinal = self._zero(output.fine_mask_logits)
                nodule = self._zero(output.fine_mask_logits)
        else:
            fine_dice = self._zero(output.fine_mask_logits)
            positive_focal = self._zero(output.fine_mask_logits)
            ordinal = self._zero(output.semantic_probabilities)
            nodule = self._zero(output.nodule_logits)

        if len(negative_rows):
            negative_focal = sigmoid_focal_loss(
                output.fine_mask_logits[negative_rows],
                output.fine_target_masks[negative_rows],
                valid=output.roi_loss_valid[negative_rows],
            )
            fine_focal = positive_focal + self.hard_negative_fine_weight * negative_focal
        else:
            fine_focal = positive_focal

        risk_value = batch.get("risk_target_valid", True)
        risk_valid = (
            bool(risk_value.item()) if isinstance(risk_value, Tensor) else bool(risk_value)
        ) and output.scan_path_valid
        if risk_valid and diagnostic_scale > 0.0:
            risk_target = batch["risk_target"].to(device).reshape_as(output.risk_logit)
            risk = F.binary_cross_entropy_with_logits(
                output.risk_logit.float(), risk_target.float()
            )
        else:
            risk = self._zero(output.risk_logit)
        anchor = ordinal_anchor if ordinal_anchor is not None else self._zero(output.semantic_features)

        terms = {
            "object": object_loss,
            "box_l1": box_l1,
            "box_giou": box_giou,
            "coarse_dice": coarse_dice,
            "coarse_focal": coarse_focal,
            "fine_dice": fine_dice,
            "fine_focal": fine_focal,
            "ordinal": ordinal * float(diagnostic_scale),
            "nodule": nodule * float(diagnostic_scale),
            "risk": risk * float(diagnostic_scale),
            "anchor": anchor * float(diagnostic_scale),
        }
        total = sum(self.loss_weights[name] * value for name, value in terms.items())
        return V2Loss(
            total=total,
            object=object_loss,
            box_l1=box_l1,
            box_giou=box_giou,
            coarse_dice=coarse_dice,
            coarse_focal=coarse_focal,
            fine_dice=fine_dice,
            fine_focal=fine_focal,
            ordinal=ordinal,
            nodule=nodule,
            risk=risk,
            anchor=anchor,
            positive_candidates=int(len(query)),
            refined_candidates=int(len(output.refined_query_indices)),
            ignored_candidates=int(ignored.sum().item()),
            risk_valid=risk_valid,
        )


__all__ = (
    "V2Loss",
    "WholeCTCriterionV2",
    "masked_dice_loss",
    "sigmoid_focal_loss",
)
