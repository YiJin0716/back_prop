"""Paper's ResNet34 RetinaNet with device-safe loss and ignored LIDC boxes.

Network/FPN/anchors are reused from the cited yhenon implementation unchanged.
Loss formulas follow retinanet_upstream/retinanet/losses.py; -2 annotations mark
indeterminate nodules whose overlapping anchors must not be learned as negatives.
"""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'retinanet_upstream'))
from retinanet.model import ResNet, BasicBlock  # noqa: E402


class RetinaNet(ResNet):
    def __init__(self, pretrained=HERE / 'weights/resnet34-333f7ec4.pth'):
        super().__init__(1, BasicBlock, [3, 4, 6, 3])
        # Upstream stores these as plain tensors; make device moves work without
        # changing checkpoint keys from already-running training jobs.
        for name in ('mean', 'std'):
            value = getattr(self.regressBoxes, name)
            delattr(self.regressBoxes, name)
            self.regressBoxes.register_buffer(name, value, persistent=False)
        if pretrained:
            # Official download.pytorch.org legacy tar checkpoint, hash recorded
            # in sources.json. Its ImageNet classification head is not in FPN.
            weights = torch.load(pretrained, map_location='cpu', weights_only=False)
            weights = {k: v for k, v in weights.items() if not k.startswith('fc.')}
            result = self.load_state_dict(weights, strict=False)
            assert not result.unexpected_keys, result.unexpected_keys
            assert all(k.startswith(('fpn.', 'regressionModel.', 'classificationModel.'))
                       for k in result.missing_keys), result.missing_keys

    def raw_predictions(self, images):
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        x1 = self.layer1(x); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        pyramid = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressionModel(x) for x in pyramid], 1)
        classification = torch.cat([self.classificationModel(x) for x in pyramid], 1)
        anchors = self.anchors(images).to(images.device)
        return classification, regression, anchors

    def train(self, mode=True):
        super().train(mode)
        self.freeze_bn()
        return self

    def forward(self, images, annotations=None):
        classification, regression, anchors = self.raw_predictions(images)
        if annotations is not None:
            return detection_loss(classification, regression, anchors, annotations)
        boxes = self.clipBoxes(self.regressBoxes(anchors, regression), images)
        output = []
        for sample, candidate_boxes in zip(classification, boxes):
            scores = sample[:, 0]
            valid = (scores > .05) & torch.isfinite(candidate_boxes).all(-1)
            scores, candidate_boxes = scores[valid], candidate_boxes[valid]
            keep = nms(candidate_boxes.float(), scores.float(), .5)
            output.append({'scores': scores[keep], 'boxes': candidate_boxes[keep]})
        return output


def detection_loss(classifications, regressions, anchors, annotations):
    classification_losses, regression_losses = [], []
    anchor = anchors[0].float()
    widths = anchor[:, 2:] - anchor[:, :2]
    centers = (anchor[:, :2] + anchor[:, 2:]) / 2
    for probability, regression, annotation in zip(classifications, regressions, annotations):
        probability, regression = probability.float().clamp(1e-4, 1 - 1e-4), regression.float()
        truth = annotation[annotation[:, 4] == 0, :4].float()
        ignored = annotation[annotation[:, 4] == -2, :4].float()
        targets = torch.full_like(probability, -1)
        positive = torch.zeros(len(anchor), dtype=torch.bool, device=anchor.device)
        if len(truth):
            overlap, assigned = box_iou(anchor, truth).max(-1)
            targets[overlap < .4] = 0
            positive = overlap >= .5
            targets[positive] = 1
        else:
            targets[:] = 0
        if len(ignored):
            # Also ignore anchors whose center falls inside an ignored nodule;
            # small lesions can be much smaller than the default 32px anchor.
            overlap = box_iou(anchor, ignored).max(-1).values >= .02
            inside = ((centers[:, None] >= ignored[None, :, :2]) &
                      (centers[:, None] <= ignored[None, :, 2:])).all(-1).any(-1)
            targets[(overlap | inside) & ~positive] = -1
        alpha = torch.where(targets == 1, .25, .75)
        focal = torch.where(targets == 1, 1 - probability, probability).square() * alpha
        bce = -(targets * probability.log() + (1 - targets) * (1 - probability).log())
        classification_losses.append(torch.where(targets != -1, focal * bce, 0).sum() /
                                     positive.sum().clamp_min(1))
        if positive.any():
            gt = truth[assigned[positive]]
            gt_size = (gt[:, 2:] - gt[:, :2]).clamp_min(1)
            gt_center = (gt[:, :2] + gt[:, 2:]) / 2
            regression_target = torch.cat([(gt_center - centers[positive]) / widths[positive],
                                           (gt_size / widths[positive]).log()], -1)
            regression_target /= regression_target.new_tensor([.1, .1, .2, .2])
            # Upstream smooth-L1 beta=1/9.
            regression_losses.append(F.smooth_l1_loss(regression[positive], regression_target, beta=1 / 9))
        else:
            regression_losses.append(regression.sum() * 0)
    return torch.stack(classification_losses).mean(), torch.stack(regression_losses).mean()
