"""Coarse-only set matching for model v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Match:
    query_indices: Tensor
    target_indices: Tensor


def box_corners(boxes: Tensor) -> tuple[Tensor, Tensor]:
    half = boxes[..., 3:] * 0.5
    return boxes[..., :3] - half, boxes[..., :3] + half


def pairwise_generalized_iou(first: Tensor, second: Tensor) -> Tensor:
    """Pairwise generalized IoU for normalized 3-D centre-size boxes."""
    first_min, first_max = box_corners(first)
    second_min, second_max = box_corners(second)
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


def pairwise_box_iou(first: Tensor, second: Tensor) -> Tensor:
    """Pairwise ordinary IoU for normalized 3-D centre-size boxes."""
    first_min, first_max = box_corners(first)
    second_min, second_max = box_corners(second)
    intersection = (
        torch.minimum(first_max[:, None], second_max[None])
        - torch.maximum(first_min[:, None], second_min[None])
    ).clamp_min(0).prod(-1)
    first_volume = (first_max - first_min).clamp_min(0).prod(-1)[:, None]
    second_volume = (second_max - second_min).clamp_min(0).prod(-1)[None]
    return intersection / (first_volume + second_volume - intersection).clamp_min(1e-7)


def pairwise_soft_dice(predicted_logits: Tensor, targets: Tensor) -> Tensor:
    predicted = predicted_logits.float().sigmoid().flatten(1)
    target = targets.float().flatten(1)
    intersection = predicted @ target.transpose(0, 1)
    denominator = predicted.sum(1, keepdim=True) + target.sum(1).unsqueeze(0)
    return (2.0 * intersection + 1e-5) / (denominator + 1e-5)


def match_coarse(
    object_logits: Tensor,
    boxes: Tensor,
    mask_logits: Tensor,
    target_boxes: Tensor,
    target_masks: Tensor,
) -> Match:
    """Hungarian match before any ROI routing or fine prediction.

    Cost = class + 5 box-L1 + 2 (1-GIoU) + 2 (1-coarse-Dice).
    The discrete assignment is deliberately detached from autograd; the
    matched predictions are subsequently trained by explicit differentiable
    loss terms.
    """
    device = boxes.device
    if len(target_boxes) == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return Match(empty, empty)
    with torch.no_grad():
        target_boxes = target_boxes.to(device=device, dtype=torch.float32)
        target_masks = target_masks.to(device=device)
        class_cost = -object_logits.float().sigmoid()[:, None]
        l1 = torch.cdist(boxes.float(), target_boxes, p=1)
        giou = pairwise_generalized_iou(boxes.float(), target_boxes)
        dice = pairwise_soft_dice(mask_logits, target_masks)
        cost = class_cost + 5.0 * l1 + 2.0 * (1.0 - giou) + 2.0 * (1.0 - dice)
        query, target = linear_sum_assignment(cost.cpu().numpy())
    return Match(
        torch.as_tensor(np.asarray(query), dtype=torch.long, device=device),
        torch.as_tensor(np.asarray(target), dtype=torch.long, device=device),
    )


def identify_ignored_queries(
    boxes: Tensor,
    mask_logits: Tensor,
    retained_query_indices: Tensor,
    ignored_boxes: Tensor,
    ignored_masks: Tensor,
    *,
    mask_dice_threshold: float = 0.1,
    box_iou_threshold: float = 0.05,
) -> Tensor:
    """Find queries that must not receive a no-object/background target.

    Every ignored physical nodule is assigned one otherwise-unmatched query by
    a box+mask Hungarian cost, even when early logits are diffuse. Additional
    queries are ignored when their centre, box, or coarse mask overlaps an
    ignored nodule. Retained positive matches always take precedence.
    """
    device = boxes.device
    result = torch.zeros(len(boxes), dtype=torch.bool, device=device)
    if len(ignored_boxes) == 0:
        return result
    ignored_boxes = ignored_boxes.to(device=device, dtype=torch.float32)
    ignored_masks = ignored_masks.to(device=device)
    available_mask = torch.ones(len(boxes), dtype=torch.bool, device=device)
    available_mask[retained_query_indices] = False
    available = available_mask.nonzero().flatten()
    with torch.no_grad():
        dice = pairwise_soft_dice(mask_logits, ignored_masks)
        box_iou = pairwise_box_iou(boxes.float(), ignored_boxes)
        ignored_min, ignored_max = box_corners(ignored_boxes)
        centres = boxes[:, :3]
        centre_inside = (
            (centres[:, None] >= ignored_min[None])
            & (centres[:, None] <= ignored_max[None])
        ).all(dim=-1).any(dim=1)
        result |= dice.max(dim=1).values >= mask_dice_threshold
        result |= box_iou.max(dim=1).values >= box_iou_threshold
        result |= centre_inside
        if len(available):
            l1 = torch.cdist(boxes[available].float(), ignored_boxes, p=1)
            giou = pairwise_generalized_iou(boxes[available].float(), ignored_boxes)
            forced_cost = 5.0 * l1 + 2.0 * (1.0 - giou) + 2.0 * (
                1.0 - dice[available]
            )
            query_numpy, _ = linear_sum_assignment(forced_cost.cpu().numpy())
            result[available[torch.as_tensor(query_numpy, device=device)]] = True
        result[retained_query_indices] = False
    return result


__all__ = (
    "Match",
    "box_corners",
    "match_coarse",
    "identify_ignored_queries",
    "pairwise_box_iou",
    "pairwise_generalized_iou",
    "pairwise_soft_dice",
)
