"""Per-supervised-nodule losses, pooled correctly across DDP ranks and epochs."""
from dataclasses import dataclass
import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from back_prop.common.coarse_loss import masked_dice_loss, sigmoid_focal_loss
from back_prop.common.matcher import pairwise_generalized_iou, identify_ignored_queries

LOSS_SCHEMA = 'v5_localization_v2'
TERM_NAMES = ('object', 'box_l1', 'box_giou', 'coarse_dice', 'coarse_focal', 'fine_dice',
              'fine_focal_positive', 'fine_focal_negative', 'missing_nodule',
              'semantic', 'semantic_anchor', 'nodule', 'risk')


def coarse_loss_support(batch, masks):
    ignored = batch.get('ignored_target_masks')
    if ignored is None or not len(ignored):
        return None
    # Retained positives take precedence if reader consensus masks overlap.
    return ~ignored.to(device=masks.device).bool().any(0)[None] | masks.bool()


def nodule_overlap_loss(logits, masks, *, valid=None):
    """Per-GT negative log soft Dice, evaluated in log space.

    Foreground recall alone rewards broad masks. This retains a useful gradient
    for completely missed nodules and also penalizes excess predicted volume.
    Object scores are supervised separately over positive and negative queries.
    """
    if valid is not None:
        logits=logits.float().masked_fill(~valid.bool(),-torch.inf)
        masks=masks*valid
    logits,masks=logits.float().flatten(1),masks.float().flatten(1)
    mass=masks.sum(1)
    if (mass<=0).any():
        raise ValueError('A retained GT has no foreground on the coarse grid')
    weights=masks.clamp_min(torch.finfo(torch.float32).tiny).log().masked_fill(masks<=0,-torch.inf)
    log_intersection=torch.logsumexp(F.logsigmoid(logits)+weights,dim=1)
    return (logits.sigmoid().sum(1)+mass).log()-log_intersection-logits.new_tensor(2.).log()


@dataclass
class V5Loss:
    total: Tensor
    numerators: dict[str, Tensor]
    counts: dict[str, int]
    positive_candidates: int
    refined_candidates: int
    supervised_candidates: int
    risk_valid: bool
    normalizers: dict[str, float] | None = None

    def mean(self, name):
        denominator = self.normalizers[name] if self.normalizers is not None else self.counts[name]
        return (float(self.numerators[name].detach()) / denominator
                if self.counts[name] else None)


def pooled_objective(numerators, counts, weights, *, synchronize=True, process_group=None, normalizers=None):
    """DDP averages gradients: scale local sums by world/global valid count.

    Ranks with no supervision still join the count collective, contribute zero
    numerator, and do not dilute gradients from ranks containing valid samples.
    """
    reference = next(iter(numerators.values()))
    normalizers = counts if normalizers is None else normalizers
    denominator = reference.new_tensor([normalizers[k] for k in TERM_NAMES], dtype=torch.float64)
    world = dist.get_world_size(process_group) if synchronize and dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(denominator, group=process_group)
    return sum(numerators[k] * (world * weights[k] / denominator[i].clamp_min(1e-12))
               for i, k in enumerate(TERM_NAMES))


class WholeCTCriterionV5(nn.Module):
    def __init__(self, *, missing_nodule_weight=2., hard_negative_fine_weight=.25,
                 semantic_anchor_weight=.5, object_background_weight=1.):
        super().__init__()
        if not 0 < object_background_weight <= 1:
            raise ValueError('Object background weight must be in (0,1]')
        self.object_background_weight = float(object_background_weight)
        self.loss_weights = dict(object=1., box_l1=5., box_giou=2., coarse_dice=1., coarse_focal=1.,
            fine_dice=2., fine_focal_positive=1., fine_focal_negative=float(hard_negative_fine_weight),
            missing_nodule=float(missing_nodule_weight), semantic=1.,
            semantic_anchor=float(semantic_anchor_weight), nodule=.5, risk=.5)
        if any(v < 0 for v in self.loss_weights.values()):
            raise ValueError('Loss weights must be nonnegative')

    def forward(self, output, batch, *, diagnostic_scale=1., semantic_scale=1.,
                anchor=None, synchronize=True, process_group=None):
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
            # Match Hungarian's six-coordinate L1 sum, then average nodules.
            add('box_l1', (output.boxes[q].float() - boxes).abs().sum(-1))
            add('box_giou', 1 - pairwise_generalized_iou(output.boxes[q].float(), boxes).diagonal())
            masks = batch['target_masks'].to(device)[t].float()
            support = coarse_loss_support(batch, masks)
            for name, loss in [('coarse_dice', masked_dice_loss), ('coarse_focal', sigmoid_focal_loss)]:
                add(name, torch.stack([loss(output.coarse_mask_logits[query:query+1], masks[i:i+1],
                                           valid=None if support is None else support[i:i+1])
                                       for i, query in enumerate(q.tolist())]))
            # Missing nodules retain whole-scan supervision regardless of routing.
            if len(q) != len(batch['target_boxes']) or len(t.unique()) != len(q):
                raise ValueError('Every retained GT needs a distinct matched query')
            add('missing_nodule', nodule_overlap_loss(output.coarse_mask_logits[q],masks,valid=support))
        elif len(batch['target_boxes']):
            raise ValueError('Every retained GT needs a matched query')

        # Calibrate confidence over all supervised queries. A positive-only
        # objective allowed every query to approach probability one.
        object_target = torch.zeros_like(output.object_logits, dtype=torch.float32)
        object_target[q] = 1.
        supervised_query = torch.ones_like(output.object_logits, dtype=torch.bool)
        ignored_boxes, ignored_masks = batch.get('ignored_target_boxes'), batch.get('ignored_target_masks')
        if ignored_boxes is not None and ignored_masks is not None and len(ignored_boxes):
            supervised_query &= ~identify_ignored_queries(output.boxes, output.coarse_mask_logits,
                q, ignored_boxes.to(device), ignored_masks.to(device))
        object_weights = torch.where(object_target.bool(),1.,self.object_background_weight)[supervised_query]
        add('object', F.binary_cross_entropy_with_logits(output.object_logits.float(),object_target,
            reduction='none')[supervised_query]*object_weights)

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
        normalizers = {k:float(v) for k,v in counts.items()}
        normalizers['object'] = float(object_weights.sum())
        total = pooled_objective(sums, counts, weights, synchronize=synchronize,
                                process_group=process_group,normalizers=normalizers)
        return V5Loss(total, sums, counts, len(q), len(output.refined_query_indices), len(positive), risk_valid,normalizers)


class LossAccumulator:
    """Reduce sums/counts once per epoch; never divide a nodule loss by CT count."""
    def __init__(self, device='cpu'):
        self.values = torch.zeros((len(TERM_NAMES), 3), device=device, dtype=torch.float64)

    def update(self, losses):
        self.values[:, 0] += torch.stack([losses.numerators[k].detach().double() for k in TERM_NAMES])
        self.values[:, 1] += self.values.new_tensor([losses.counts[k] for k in TERM_NAMES])
        normalizers = losses.counts if losses.normalizers is None else losses.normalizers
        self.values[:, 2] += self.values.new_tensor([normalizers[k] for k in TERM_NAMES])

    def compute(self, weights, *, diagnostic_scale=1., synchronize=True, process_group=None):
        values = self.values.clone()
        if synchronize and dist.is_initialized():
            dist.all_reduce(values, group=process_group)
        counts = {k: int(values[i, 1]) for i, k in enumerate(TERM_NAMES)}
        means = {k: float(values[i, 0] / values[i, 2]) if counts[k] else None
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
                'loss_normalizers': {k: float(values[i, 2]) for i, k in enumerate(TERM_NAMES)},
                'detr_loss': sum(weighted[k] or 0 for k in ('object', 'box_l1', 'box_giou', 'coarse_dice', 'coarse_focal'))
                    if counts['box_l1'] else None,
                'finetuner_loss': sum(weighted[k] or 0 for k in ('fine_dice', 'fine_focal_positive', 'fine_focal_negative'))
                    if counts['fine_dice'] or counts['fine_focal_negative'] else None}
