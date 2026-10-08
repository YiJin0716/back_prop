"""MedicalNet semantics with standardized convolutions and GroupNorm."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from back_prop.common.features import MedicalNetSemantics as V3Semantics, SEMANTIC_NAMES


NORMALIZATION = dict(kind='GroupNorm', max_groups=32, min_channels_per_group=8,
                     affine_initialization='pretrained BatchNorm weight and bias')
CONVOLUTION = dict(kind='WeightStandardizedConv3d', eps=1e-5,
                   axes='input channels and spatial kernel, separately per output channel')


class WeightStandardizedConv3d(nn.Conv3d):
    """Keep trainable pretrained filters; normalize their forward-pass weights.

    Directly combining MedicalNet's BN-trained convolutions with GroupNorm
    suppresses differences between sparse nodule inputs across residual stages.
    Per-filter centering/scaling removes the common response responsible for
    that collapse without introducing batch-dependent running statistics.
    Stored weights and their gradients retain the original checkpoint layout.
    """
    def forward(self, x):
        dims = tuple(range(1, self.weight.ndim))
        variance, mean = torch.var_mean(self.weight, dim=dims, unbiased=False, keepdim=True)
        weight = (self.weight - mean) * torch.rsqrt(variance + CONVOLUTION['eps'])
        return self._conv_forward(x, weight, self.bias)


def standardize_convolutions(module):
    for name, child in module.named_children():
        if isinstance(child, WeightStandardizedConv3d):
            continue
        if isinstance(child, nn.Conv3d):
            # A one-coefficient filter has zero centered weight; leave it as is.
            if child.weight[0].numel() == 1:
                continue
            replacement = WeightStandardizedConv3d(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding, dilation=child.dilation,
                groups=child.groups, bias=child.bias is not None,
                padding_mode=child.padding_mode, device='meta')
            replacement.weight, replacement.bias = child.weight, child.bias
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            standardize_convolutions(child)


def check_semantic_spread(predicted_std, target_means):
    """Reject a collapsed warmup before spending a run on joint training.

    Only assess nontrivial cohorts and attributes with varying targets. This
    is a collapse check, not a substitute for held-out accuracy evaluation.
    """
    target_std = target_means.float().std(dim=0, unbiased=False)
    checked = len(target_means) >= 32
    collapsed = [name for name, p, t in zip(SEMANTIC_NAMES, predicted_std, target_std)
                 if checked and float(t) > .25 and float(p) < .01]
    if collapsed:
        raise RuntimeError('Semantic warmup produced near-constant predictions for '
                           + ', '.join(collapsed) + '; joint training was not started.')
    return dict(checked=checked, samples=len(target_means), target_std=target_std.tolist(),
                predicted_std=list(predicted_std), minimum_prediction_std=.01,
                minimum_target_std=.25)


def replace_batchnorm(module):
    """Keep pretrained convolutions/affine values; discard BN running statistics.

    Each group contains at least eight channels when the layer is large enough.
    Avoid one group per channel, which would recreate singleton BN statistics.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            channels = child.num_features
            groups = min(NORMALIZATION['max_groups'],
                         max(1, channels // NORMALIZATION['min_channels_per_group']))
            while channels % groups:
                groups -= 1
            norm = nn.GroupNorm(groups, channels, eps=child.eps, affine=child.affine)
            reference = child.weight if child.affine else child.running_mean
            if reference is not None:
                norm.to(device=reference.device, dtype=reference.dtype)
            if child.affine:
                with torch.no_grad():
                    norm.weight.copy_(child.weight)
                    norm.bias.copy_(child.bias)
                norm.weight.requires_grad_(child.weight.requires_grad)
                norm.bias.requires_grad_(child.bias.requires_grad)
            norm.train(child.training)
            setattr(module, name, norm)
        else:
            replace_batchnorm(child)


class MedicalNetSemantics(V3Semantics):
    def __init__(self, source, *, weight_standardization=True):
        nn.Module.__init__(self)
        self.encoder, self.score = source.encoder, source.score
        self.threshold_base, self.threshold_steps = source.threshold_base, source.threshold_steps
        self.register_buffer('classes', source.classes)
        self.roi_size, self.use_checkpoint = source.roi_size, source.use_checkpoint
        replace_batchnorm(self.encoder)
        self.weight_standardization = bool(weight_standardization)
        if self.weight_standardization:
            standardize_convolutions(self.encoder)

    def train(self, mode=True):
        return nn.Module.train(self, mode)

    def prepare_roi(self, ct, mask_logits, valid):
        return F.interpolate(ct.float() * mask_logits.float().sigmoid() * valid.float(),
                             size=(self.roi_size,) * 3, mode='trilinear', align_corners=False)

    def forward(self, ct, mask_logits=None, valid=None, *, prepared=False):
        with torch.autocast(device_type=ct.device.type, enabled=False):
            roi = ct.float() if prepared else self.prepare_roi(ct, mask_logits, valid)
            if self.training and self.use_checkpoint:
                embedding = checkpoint(self.encoder, roi, use_reentrant=False)
            else:
                embedding = self.encoder(roi)
            linear = self.score(embedding)
            thresholds = torch.cat((self.threshold_base[:, None], self.threshold_base[:, None]
                                    + F.softplus(self.threshold_steps).cumsum(1)), dim=1)
            cumulative = torch.sigmoid(thresholds[None] - linear[..., None])
            probability = torch.cat((cumulative[..., :1], cumulative[..., 1:] - cumulative[..., :-1],
                                     1.0 - cumulative[..., -1:]), -1).clamp_min(1e-7)
            probability = probability / probability.sum(-1, keepdim=True)
            return probability, (probability * self.classes).sum(-1)


@torch.no_grad()
def normalization_probe(semantics, rois, *, batch_size=4):
    """Check fixed-weight predictions across batch sizes and train/eval modes."""
    was_training, use_checkpoint = semantics.training, semantics.use_checkpoint
    semantics.use_checkpoint = False
    outputs = {}
    try:
        for name, training, size in [('train_single', True, 1),
                                     ('train_batched', True, batch_size),
                                     ('eval_batched', False, batch_size)]:
            semantics.train(training)
            # Compare normalization in IEEE FP32. cuDNN may choose different
            # TF32 convolution kernels for one ROI and a batch, introducing
            # rounding differences unrelated to train/eval normalization.
            with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                    benchmark=torch.backends.cudnn.benchmark,
                    benchmark_limit=torch.backends.cudnn.benchmark_limit,
                    deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
                rows = [semantics(chunk, prepared=True) for chunk in rois.split(size)]
            outputs[name] = tuple(torch.cat([row[i] for row in rows]) for i in (0, 1))
        reference = outputs['train_single']
        for name in ('train_batched', 'eval_batched'):
            for actual, expected in zip(outputs[name], reference):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-5)
        return dict(samples=len(rois), batched_size=batch_size, convolution_precision='IEEE FP32',
            batch_max_abs_difference=float((outputs['train_batched'][1]-reference[1]).abs().max()),
            train_eval_max_abs_difference=float((outputs['eval_batched'][1]-reference[1]).abs().max()),
            semantic_std=reference[1].std(dim=0, unbiased=False).cpu().tolist(),
            passed=True)
    finally:
        semantics.use_checkpoint = use_checkpoint
        semantics.train(was_training)
