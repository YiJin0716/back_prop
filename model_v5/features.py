"""Randomly initialized residual 3-D CNN for six reader-rated attributes."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from back_prop.common.features import SEMANTIC_NAMES

NORMALIZATION = dict(kind='GroupNorm', groups=4, initialization='random CNN; no running statistics')
CONVOLUTION = dict(kind='Conv3d', initialization='Kaiming; no pretrained weights')


class ResidualBlock3D(nn.Module):
    def __init__(self, inputs, channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(inputs, channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(4, channels), nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels))
        self.skip = (nn.Identity() if inputs == channels and stride == 1 else
                     nn.Conv3d(inputs, channels, 1, stride=stride, bias=False))

    def forward(self, x):
        return F.relu(self.layers(x) + self.skip(x))


class Encoder3D(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock3D(2, width, 2),
            ResidualBlock3D(width, width * 2, 2),
            ResidualBlock3D(width * 2, width * 4, 2),
            ResidualBlock3D(width * 4, width * 4))
        self.output_dim = width * 4 * 9

    def forward(self, x):
        x = self.layers(x)
        return torch.cat((F.adaptive_avg_pool3d(x, 2).flatten(1),
                          F.adaptive_max_pool3d(x, 1).flatten(1)), dim=1)


class CNNSemantics(nn.Module):
    """Separate masked CT and soft-mask channels preserve intensity and shape.

    Each attribute has a five-class distribution; its expected rating remains
    the six-dimensional input to the existing residual malignancy model.
    """
    def __init__(self, roi_size=64, width=16, use_checkpoint=True):
        super().__init__()
        if roi_size < 8 or width < 4 or width % 4:
            raise ValueError('Require ROI size >= 8 and a positive width divisible by four')
        self.roi_size, self.use_checkpoint = int(roi_size), bool(use_checkpoint)
        self.encoder = Encoder3D(width)
        self.score = nn.Sequential(nn.Linear(self.encoder.output_dim, 128), nn.ReLU(), nn.Linear(128, 30))
        self.register_buffer('classes', torch.arange(1, 6, dtype=torch.float32))
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    def prepare_roi(self, ct, mask_logits, valid):
        probability = mask_logits.float().sigmoid() * valid.float()
        return F.interpolate(torch.cat((ct.float() * probability, probability), dim=1),
                             size=(self.roi_size,) * 3, mode='trilinear', align_corners=False)

    def forward(self, ct, mask_logits=None, valid=None, *, prepared=False):
        # FP32 keeps small foreground differences and five-class probabilities
        # visible even when the whole-CT geometry runs under autocast.
        with torch.autocast(device_type=ct.device.type, enabled=False):
            roi = ct.float() if prepared else self.prepare_roi(ct, mask_logits, valid)
            if roi.ndim != 5 or roi.shape[1] != 2:
                raise ValueError('Prepared V5 ROIs require masked CT and mask channels')
            embedding = (checkpoint(self.encoder, roi, use_reentrant=False)
                         if self.training and self.use_checkpoint else self.encoder(roi))
            probabilities = self.score(embedding).reshape(-1, 6, 5).softmax(-1)
            return probabilities, (probabilities * self.classes).sum(-1)


def check_semantic_spread(predicted_std, target_means):
    """Fail visibly on collapse; variation alone does not establish accuracy."""
    target_std = target_means.float().std(dim=0, unbiased=False)
    checked = len(target_means) >= 32
    collapsed = [name for name, p, t in zip(SEMANTIC_NAMES, predicted_std, target_std)
                 if checked and float(t) > .25 and float(p) < .01]
    if collapsed:
        raise RuntimeError('Near-constant semantic predictions: ' + ', '.join(collapsed))
    return dict(checked=checked, samples=len(target_means), target_std=target_std.tolist(),
                predicted_std=list(predicted_std), minimum_prediction_std=.01)
