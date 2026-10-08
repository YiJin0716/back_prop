"""Official Sybil architecture, with an LIDC fine-tuning interface.

Adapted from upstream/sybil/models/sybil.py (MIT; see upstream/LICENSE.txt).
The pooling and cumulative head are loaded unchanged from the pinned upstream.
"""
import importlib.util
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

HERE = Path(__file__).resolve().parent


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pooling = source_module('baseline_sybil_pooling', HERE / 'upstream/sybil/models/pooling_layer.py')
cumulative = source_module('baseline_sybil_cumulative', HERE / 'upstream/sybil/models/cumulative_probability_layer.py')


class SybilNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_encoder = nn.Sequential(*list(torchvision.models.video.r3d_18(weights=None).children())[:-2])
        self.pool = pooling.MultiAttentionPool()
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(args.dropout)
        self.prob_of_failure_layer = cumulative.Cumulative_Probability_Layer(512, args, args.max_followup)

    def forward(self, image):
        activation = self.image_encoder(image)
        output = self.pool(activation)
        output['hidden'] = self.dropout(self.relu(output['hidden']))
        output['logit'] = self.prob_of_failure_layer(output['hidden'])
        output['activ'] = activation
        return output

    @classmethod
    def from_pretrained(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(checkpoint['args'])
        state = checkpoint['state_dict']
        assert all(k.startswith('model.') for k in state)
        model.load_state_dict({k[6:]: v for k, v in state.items()}, strict=True)
        return model

    def configure_finetuning(self):
        # Small fine-tune: last residual stage, attention pools and risk head.
        for stage in list(self.image_encoder.children())[:-1]:
            stage.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Whole CT batch size is one; retain NLST running BN statistics.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


def attention_loss(output, mask):
    """Spatial/volume KL terms from upstream losses, using consensus masks.

    No side-of-cancer term: a LIDC nodule annotation is not a cancer-side label.
    """
    target = F.interpolate(mask.float(), size=output['activ'].shape[-3:], mode='area')
    spatial = target[:, 0].flatten(2)
    area = spatial.sum(-1)
    distribution = spatial / area.unsqueeze(-1).clamp_min(1e-12)
    image_loss = F.kl_div(output['image_attention_1'].float(), distribution, reduction='none').sum()
    image_loss = image_loss / (area > 0).sum().clamp_min(1)
    volume_distribution = area / area.sum(-1, keepdim=True).clamp_min(1e-12)
    volume_loss = sum(F.kl_div(output[f'volume_attention_{i}'].float(), volume_distribution,
                               reduction='batchmean') for i in (1, 2))
    return image_loss + volume_loss
