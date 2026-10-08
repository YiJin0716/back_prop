"""Per-supervised-nodule losses, pooled correctly across DDP ranks and epochs."""
from dataclasses import dataclass
import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from back_prop.common.coarse_loss import masked_dice_loss, sigmoid_focal_loss
from back_prop.common.matcher import pairwise_generalized_iou
from back_prop.model_v4.loss import missing_nodule_loss

TERM_NAMES = ('box_l1', 'box_giou', 'coarse_dice', 'coarse_focal', 'fine_dice',
              'fine_focal_positive', 'fine_focal_negative', 'missing_nodule',
              'semantic', 'semantic_anchor', 'nodule', 'risk')


@dataclass
class V5Loss:
    total: Tensor
    numerators: dict[str, Tensor]
    counts: dict[str, int]
    positive_candidates: int
    refined_candidates: int
    supervised_candidates: int
    risk_valid: bool

    def mean(self, name):
        return (float(self.numerators[name].detach()) / self.counts[name]
                if self.counts[name] else None)


def pooled_objective(numerators, counts, weights, *, synchronize=True):
    """DDP averages gradients: scale local sums by world/global valid count.

    Ranks with no supervision still join the count collective, contribute zero
    numerator, and do not dilute gradients from ranks containing valid samples.
    """
    reference = next(iter(numerators.values()))
    denominator = reference.new_tensor([counts[k] for k in TERM_NAMES], dtype=torch.float64)
    world = dist.get_world_size() if synchronize and dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(denominator)
    return sum(numerators[k] * (world * weights[k] / denominator[i].clamp_min(1))
               for i, k in enumerate(TERM_NAMES))


class WholeCTCriterionV5(nn.Module):
    def __init__(self, *, missing_nodule_weight=2., hard_negative_fine_weight=.25,
                 semantic_anchor_weight=.5):
        super().__init__()
        self.loss_weights = dict(box_l1=5., box_giou=2., coarse_dice=1., coarse_focal=1.,
            fine_dice=2., fine_focal_positive=1., fine_focal_negative=float(hard_negative_fine_weight),
            missing_nodule=float(missing_nodule_weight), semantic=1.,
            semantic_anchor=float(semantic_anchor_weight), nodule=.5, risk=.5)
        if any(v < 0 for v in self.loss_weights.values()):
            raise ValueError('Loss weights must be nonnegative')

    def forward(self, output, batch, *, diagnostic_scale=1., semantic_scale=1.,
                anchor=None, synchronize=True):
        q, t = output.matched_query_indices, output.matched_target_indices
        device = output.object_logits.device
        zero = output.object_logits.float().sum() * 0.
        sums = {k: zero for k in TERM_NAMES}
        counts = {k: 0 for k in TERM_NAMES}

        def add(name, values):
            # One entry per supervised nodule (or per scan for scan risk).
            values = values.reshape(-1)
            sums[name], counts[name] = values.sum(), values.numel()

        if len(q):
            boxes = batch['target_boxes'].to(device)[t].float()
            add('box_l1', (output.boxes[q].float() - boxes).abs().mean(-1))
            add('box_giou', 1 - pairwise_generalized_iou(output.boxes[q].float(), boxes).diagonal())
            masks = batch['target_masks'].to(device)[t].float()
            for name, loss in [('coarse_dice', masked_dice_loss), ('coarse_focal', sigmoid_focal_loss)]:
                add(name, torch.stack([loss(output.coarse_mask_logits[query:query+1], masks[i:i+1])
                                       for i, query in enumerate(q.tolist())]))
            # Missing nodules retain whole-scan supervision regardless of routing.
            sums['missing_nodule'] = missing_nodule_loss(output, batch) * len(q)
            counts['missing_nodule'] = len(q)
        elif len(batch['target_boxes']):
            raise ValueError('Every retained GT needs a matched query')

        supported = output.roi_loss_valid.flatten(1).any(1)
        positive = ((output.refined_target_indices >= 0) & output.fine_supervision_valid & supported).nonzero().flatten()
        negative = ((output.refined_target_indices < 0) & supported).nonzero().flatten()
        if len(positive):
            for name, loss in [('fine_dice', masked_dice_loss), ('fine_focal_positive', sigmoid_focal_loss)]:
                add(name, torch.stack([loss(output.fine_mask_logits[i:i+1], output.fine_target_masks[i:i+1],
                                            valid=output.roi_loss_valid[i:i+1]) for i in positive.tolist()]))
        semantic_positive = positive[output.semantic_supervision_valid[positive]]
        if len(semantic_positive) and semantic_scale > 0:
            targets = output.refined_target_indices[semantic_positive]
            histogram = batch['semantic_histograms'].to(device)[targets]
            probability = output.semantic_probabilities[semantic_positive].float().clamp_min(1e-7)
            add('semantic', -(histogram * probability.log()).sum(-1).mean(-1))
            if output.bank_ready and diagnostic_scale > 0:
                logits = output.ensemble_logits[semantic_positive].float()
                labels = batch['malignancy_targets'].to(logits)[targets, None].expand_as(logits)
                add('nodule', F.binary_cross_entropy_with_logits(logits, labels, reduction='none').mean(-1))
        if anchor is not None and semantic_scale > 0:
            probability, histogram = anchor
            add('semantic_anchor', -(histogram.to(probability) * probability.float().clamp_min(1e-7).log()).sum(-1).mean(-1))
        if len(negative):
            add('fine_focal_negative', torch.stack([
                sigmoid_focal_loss(output.fine_mask_logits[i:i+1], output.fine_target_masks[i:i+1],
                                   valid=output.roi_loss_valid[i:i+1]) for i in negative.tolist()]))
        risk_valid = (output.bank_ready and diagnostic_scale > 0 and output.scan_path_valid
                      and bool(batch.get('risk_target_valid', True)) and bool(output.scan_logits.numel()))
        if risk_valid:
            add('risk', F.binary_cross_entropy_with_logits(output.scan_logits.float(),
                         batch['risk_target'].to(output.scan_logits).expand_as(output.scan_logits)).reshape(1))
        weights = {k: w * (diagnostic_scale if k in ('nodule', 'risk') else 1.)
                   for k, w in self.loss_weights.items()}
        for name in ('semantic', 'semantic_anchor'):
            weights[name] *= semantic_scale
        total = pooled_objective(sums, counts, weights, synchronize=synchronize)
        return V5Loss(total, sums, counts, len(q), len(output.refined_query_indices), len(positive), risk_valid)


class LossAccumulator:
    """Reduce sums/counts once per epoch; never divide a nodule loss by CT count."""
    def __init__(self, device='cpu'):
        self.values = torch.zeros((len(TERM_NAMES), 2), device=device, dtype=torch.float64)

    def update(self, losses):
        self.values[:, 0] += torch.stack([losses.numerators[k].detach().double() for k in TERM_NAMES])
        self.values[:, 1] += self.values.new_tensor([losses.counts[k] for k in TERM_NAMES])

    def compute(self, weights, *, diagnostic_scale=1., synchronize=True):
        values = self.values.clone()
        if synchronize and dist.is_initialized():
            dist.all_reduce(values)
        counts = {k: int(values[i, 1]) for i, k in enumerate(TERM_NAMES)}
        means = {k: float(values[i, 0] / values[i, 1]) if counts[k] else None
                 for i, k in enumerate(TERM_NAMES)}
        weighted = {k: v * weights[k] * (diagnostic_scale if k in ('nodule', 'risk') else 1.)
                    if v is not None else None for k, v in means.items()}
        # A wholly missing required objective has no comparable total. Report
        # null, not an apparent improvement caused by dropping that objective.
        required = [k for k in TERM_NAMES if k != 'fine_focal_negative' and weights[k] > 0
                    and (diagnostic_scale > 0 or k not in ('nodule', 'risk'))]
        total = sum(v for v in weighted.values() if v is not None)
        return {**means, 'total': total if all(counts[k] for k in required) else None,
                'active_total': total, 'loss_counts': counts,
                'loss_numerators': {k: float(values[i, 0]) for i, k in enumerate(TERM_NAMES)},
                'detr_loss': sum(weighted[k] for k in ('box_l1', 'box_giou', 'coarse_dice', 'coarse_focal'))
                    if counts['box_l1'] else None,
                'finetuner_loss': sum(weighted[k] or 0 for k in ('fine_dice', 'fine_focal_positive', 'fine_focal_negative'))
                    if counts['fine_dice'] or counts['fine_focal_negative'] else None}
