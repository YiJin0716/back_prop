"""Average per-model losses, never a loss of an averaged prediction."""
import torch.nn.functional as F
from back_prop.common.coarse_loss import WholeCTCriterionV2


class WholeCTCriterionV3(WholeCTCriterionV2):
    def __init__(self, **kwargs):
        super().__init__(anchor_weight=0.0, **kwargs)

    def forward(self, output, batch, *, diagnostic_scale=1.0):
        # Reuse geometric/matching losses with all diagnostic terms disabled.
        losses = super().forward(output, batch, diagnostic_scale=0.0)
        rows = ((output.refined_target_indices >= 0) & output.fine_supervision_valid).nonzero().flatten()
        if len(rows):
            targets = output.refined_target_indices[rows]
            histograms = batch['semantic_histograms'].to(output.semantic_probabilities.device)[targets]
            probabilities = output.semantic_probabilities[rows].float().clamp_min(1e-7)
            losses.ordinal = -(histograms * probabilities.log()).sum(-1).mean()
            if output.bank_ready and diagnostic_scale > 0:
                logits = output.ensemble_logits[rows]
                labels = batch['malignancy_targets'].to(logits)[targets, None].expand_as(logits)
                losses.nodule = F.binary_cross_entropy_with_logits(logits, labels)
        if output.bank_ready and losses.risk_valid and diagnostic_scale > 0:
            labels = batch['risk_target'].to(output.scan_logits).expand_as(output.scan_logits)
            losses.risk = F.binary_cross_entropy_with_logits(output.scan_logits, labels)
        # Semantic supervision is on from the start; only malignancy is ramped.
        losses.total = losses.total + self.loss_weights['ordinal'] * losses.ordinal + diagnostic_scale * (
            self.loss_weights['nodule'] * losses.nodule + self.loss_weights['risk'] * losses.risk)
        return losses
