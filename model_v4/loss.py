"""V4 objectives: geometry, fine masks, semantics, residual risk and misses.

There is no object-classification loss or radiomics-to-mask loss.
"""
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from back_prop.common.coarse_loss import masked_dice_loss, sigmoid_focal_loss
from back_prop.common.matcher import pairwise_generalized_iou, identify_ignored_queries


def missing_nodule_loss(output, batch):
    """Mean negative log of object probability times GT foreground recall.

    One matched candidate must cover each retained physical GT. Unlike fine
    supervision this operates on whole-scan masks and never depends on ROI
    coverage, teacher forcing, an inference threshold, or diagnostic readiness.
    Background voxels are deliberately absent: V3 losses constrain false alarms.
    Log-space evaluation preserves gradients even for confidently missed GTs.
    """
    query, target = output.matched_query_indices, output.matched_target_indices
    masks = batch['target_masks'].to(output.coarse_mask_logits.device).float()
    if len(masks) == 0:
        return output.object_logits.float().sum() * 0.0
    if len(target) != len(masks) or len(torch.unique(target)) != len(masks):
        raise ValueError('Every retained GT needs a distinct query; increase num_queries')
    logits = output.coarse_mask_logits[query].float().flatten(1)
    foreground = masks[target].flatten(1)
    mass = foreground.sum(1)
    if (mass <= 0).any():
        raise ValueError('A retained GT has no foreground on the coarse grid')
    log_weights = foreground.clamp_min(torch.finfo(torch.float32).tiny).log()
    log_weights = log_weights.masked_fill(foreground <= 0, -torch.inf)
    log_recall = torch.logsumexp(F.logsigmoid(logits) + log_weights, dim=1) - mass.log()
    return (F.softplus(-output.object_logits[query].float()) - log_recall).mean()


@dataclass
class V4Loss:
    total: Tensor
    box_l1: Tensor
    box_giou: Tensor
    coarse_dice: Tensor
    coarse_focal: Tensor
    fine_dice: Tensor
    fine_focal: Tensor
    missing_nodule: Tensor
    semantic: Tensor
    nodule: Tensor
    risk: Tensor
    detr_loss: Tensor
    finetuner_loss: Tensor
    positive_candidates: int
    refined_candidates: int
    ignored_candidates: int
    risk_valid: bool


class WholeCTCriterionV4(nn.Module):
    def __init__(self, *, missing_nodule_weight=2.0, hard_negative_fine_weight=.25):
        super().__init__()
        self.loss_weights = dict(box_l1=5., box_giou=2., coarse_dice=1., coarse_focal=1.,
            fine_dice=2., fine_focal=1., missing_nodule=float(missing_nodule_weight),
            semantic=1., nodule=.5, risk=.5)
        if any(v < 0 for v in self.loss_weights.values()):
            raise ValueError('Loss weights must be nonnegative')
        self.hard_negative_fine_weight = hard_negative_fine_weight

    def forward(self, output, batch, *, diagnostic_scale=1.0):
        q, t = output.matched_query_indices, output.matched_target_indices
        device = output.object_logits.device
        zero = output.fine_mask_logits.float().sum() * 0.
        terms = {k: zero for k in self.loss_weights}
        if len(q):
            boxes = batch['target_boxes'].to(device)[t].float()
            terms['box_l1'] = F.l1_loss(output.boxes[q].float(), boxes)
            terms['box_giou'] = (1-pairwise_generalized_iou(output.boxes[q].float(), boxes).diagonal()).mean()
            masks = batch['target_masks'].to(device)[t].float()
            terms['coarse_dice'] = masked_dice_loss(output.coarse_mask_logits[q], masks)
            terms['coarse_focal'] = sigmoid_focal_loss(output.coarse_mask_logits[q], masks)
        positive = ((output.refined_target_indices >= 0) & output.fine_supervision_valid).nonzero().flatten()
        negative = (output.refined_target_indices < 0).nonzero().flatten()
        if len(positive):
            fine, gt = output.fine_mask_logits[positive], output.fine_target_masks[positive]
            valid = output.roi_loss_valid[positive]
            terms['fine_dice'] = masked_dice_loss(fine, gt, valid)
            terms['fine_focal'] = sigmoid_focal_loss(fine, gt, valid=valid)
            targets = output.refined_target_indices[positive]
            histogram = batch['semantic_histograms'].to(device)[targets]
            probability = output.semantic_probabilities[positive].float().clamp_min(1e-7)
            terms['semantic'] = -(histogram * probability.log()).sum(-1).mean()
            if output.bank_ready and diagnostic_scale > 0:
                logits = output.ensemble_logits[positive].float()
                labels = batch['malignancy_targets'].to(logits)[targets, None].expand_as(logits)
                terms['nodule'] = F.binary_cross_entropy_with_logits(logits, labels)
        if len(negative):
            terms['fine_focal'] = terms['fine_focal'] + self.hard_negative_fine_weight * sigmoid_focal_loss(
                output.fine_mask_logits[negative], output.fine_target_masks[negative],
                valid=output.roi_loss_valid[negative])
        risk_valid = bool(batch.get('risk_target_valid', True)) and output.scan_path_valid
        if output.bank_ready and risk_valid and diagnostic_scale > 0:
            terms['risk'] = F.binary_cross_entropy_with_logits(output.scan_logits.float(),
                                    batch['risk_target'].to(output.scan_logits).expand_as(output.scan_logits))
        terms['missing_nodule'] = missing_nodule_loss(output, batch)
        weighted = {k: v*self.loss_weights[k]*(diagnostic_scale if k in ('nodule','risk') else 1.)
                    for k,v in terms.items()}
        ignored = torch.zeros_like(output.object_logits, dtype=torch.bool)
        if 'ignored_target_boxes' in batch:
            ignored = identify_ignored_queries(output.boxes, output.coarse_mask_logits, q,
                        batch['ignored_target_boxes'].to(device), batch['ignored_target_masks'].to(device))
        return V4Loss(total=sum(weighted.values()), **terms,
            detr_loss=sum(weighted[k] for k in ('box_l1','box_giou','coarse_dice','coarse_focal')),
            finetuner_loss=weighted['fine_dice']+weighted['fine_focal'],
            positive_candidates=len(q), refined_candidates=len(output.refined_query_indices),
            ignored_candidates=int(ignored.sum()), risk_valid=risk_valid)
