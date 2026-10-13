"""Readiness helpers and joint prediction-only geometry audit."""
import torch
import torch.distributed as dist
from back_prop.common.matcher import pairwise_box_iou
from .loss import coarse_loss_support

SCHEMA = 'v5_parallel_official_v3'


class WarmupReadiness:
    """Require a minimum duration and a sustained run of passing fixed probes."""
    def __init__(self, minimum_epochs, required_passes, consecutive_passes=0):
        self.minimum_epochs = minimum_epochs
        self.required_passes = required_passes
        self.consecutive_passes = consecutive_passes

    def update(self, completed_epochs, passed):
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        return (completed_epochs >= self.minimum_epochs
                and self.consecutive_passes >= self.required_passes)


@torch.no_grad()
def geometry_audit(model, dataset, args, rank, world):
    model.eval()
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed+731))
    order = order[:args.geometry_probe_cases]
    totals = torch.zeros(11, device=model.device, dtype=torch.float64)
    for index in order[rank::world]:
        batch = dataset[int(index)]
        out = model(batch['image'], batch=batch, geometry_only=True, update_bank=False)
        assert out.teacher_forced_count == out.fallback_count == 0
        rows = ((out.refined_target_indices >= 0) & out.fine_supervision_valid).nonzero().flatten()
        q,t = out.matched_query_indices,out.matched_target_indices
        p = out.coarse_mask_logits[q].float().sigmoid()
        g = batch['target_masks'].to(p)[t]
        support = coarse_loss_support(batch,g)
        if support is not None:
            p,g = p*support,g*support
        p,g = p.flatten(1),g.flatten(1)
        intersection = (p*g).sum(1)
        predicted_mass, target_mass = p.sum(1),g.sum(1)
        boxes = batch['target_boxes'].to(out.boxes)[t]
        # Uncovered GT has zero quality: no survivor-only Dice average.
        totals += totals.new_tensor([len(batch['nodule_ids']), len(rows),
            float(out.fine_soft_dice[rows].sum()), int(out.semantic_supervision_valid.sum()),
            float((2*intersection/(predicted_mass+target_mass).clamp_min(1e-8)).sum()),
            float(pairwise_box_iou(out.boxes[q],boxes).diag().sum()),
            float((intersection/predicted_mass.clamp_min(1e-8)).sum()),
            float((intersection/target_mass.clamp_min(1e-8)).sum()),
            float((predicted_mass/target_mass.clamp_min(1e-8)).sum()),
            len(out.refined_query_indices), 1])
    if world > 1:
        dist.all_reduce(totals)
    n = int(totals[0]); denominator = max(n, 1)
    result = dict(nodules=n, coverage=float(totals[1]/denominator),
                  dice=float(totals[2]/denominator), usable_fraction=float(totals[3]/denominator),
                  coarse_dice=float(totals[4]/denominator), box_iou=float(totals[5]/denominator),
                  coarse_precision=float(totals[6]/denominator), coarse_recall=float(totals[7]/denominator),
                  coarse_mass_ratio=float(totals[8]/denominator),
                  selected_queries_per_scan=float(totals[9]/totals[10].clamp_min(1)))
    result['passed'] = (n > 0 and result['coverage'] >= args.geometry_min_coverage
        and result['dice'] >= args.geometry_min_dice
        and result['coarse_dice'] >= args.geometry_min_coarse_dice
        and result['usable_fraction'] >= args.geometry_min_usable)
    return result
