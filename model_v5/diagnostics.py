"""Streaming variation checks on actually supervised predicted fine ROIs."""
import torch
import torch.distributed as dist


class SemanticAccumulator:
    def __init__(self, device):
        # count, prediction sums/squares, target sums/squares
        self.stats = torch.zeros(25, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, output, batch):
        rows = ((output.refined_target_indices >= 0) & output.fine_supervision_valid
                & getattr(output, 'semantic_supervision_valid', output.fine_supervision_valid)
                & output.roi_loss_valid.flatten(1).any(1)).nonzero().flatten()
        prediction = output.semantic_features[rows].double()
        targets = batch['semantic_targets'].to(prediction)[output.refined_target_indices[rows]]
        self.stats += torch.cat((prediction.new_tensor([len(rows)]), prediction.sum(0),
                                prediction.square().sum(0), targets.sum(0), targets.square().sum(0)))

    def compute(self, *, enforce=True):
        stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(stats)
        n = int(stats[0]); denominator = max(n, 1)
        pstd = (stats[7:13] / denominator - (stats[1:7] / denominator).square()).clamp_min(0).sqrt()
        tstd = (stats[19:25] / denominator - (stats[13:19] / denominator).square()).clamp_min(0).sqrt()
        # Reconstruct moments only, not samples, to share the same guard rule.
        from back_prop.common.features import SEMANTIC_NAMES
        collapsed = [name for name, p, t in zip(SEMANTIC_NAMES, pstd, tstd)
                     if n >= 32 and t > .25 and p < .01]
        if collapsed and enforce:
            raise RuntimeError('Near-constant semantics on supervised fine ROIs: ' + ', '.join(collapsed))
        return dict(samples=n, checked=n >= 32, predicted_std=pstd.tolist(),
                    target_std=tstd.tolist(), near_constant_attributes=collapsed)
