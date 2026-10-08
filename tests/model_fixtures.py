"""Small CPU model fixtures shared by model and evaluation tests."""
import torch
from torch import nn


class TinySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        return (image - 0.4) * self.scale



class TinyRadiomics(nn.Module):
    def forward(self, image, mask_logits):
        probability = mask_logits.sigmoid()
        value = (image * probability).mean() + probability.mean()
        return value.reshape(1, 1).expand(1, 64)



class TinyOrdinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(64, 7 * 5)
        self.register_buffer("classes", torch.arange(1, 6, dtype=torch.float32))

    def forward(self, features):
        probability = self.head(features).reshape(-1, 7, 5).softmax(dim=-1)
        return probability, (probability * self.classes).sum(dim=-1)



def make_batch():
    image = torch.zeros(1, 12, 12, 12)
    image[:, 4:7, 4:7, 4:7] = 1.0
    coarse = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
    coarse[:, 2:5, 2:5, 2:5] = 1
    histogram = torch.zeros(1, 7, 5)
    histogram[:, :, 3] = 1.0
    return {
        "image": image,
        "target_masks": coarse,
        "target_boxes": torch.tensor([[5.5 / 12, 5.5 / 12, 5.5 / 12, 3 / 12, 3 / 12, 3 / 12]]),
        "target_mask_crops": [torch.ones(3, 3, 3, dtype=torch.uint8)],
        "target_mask_origins": torch.tensor([[4, 4, 4]], dtype=torch.long),
        "semantic_histograms": histogram,
        "semantic_targets": torch.full((1, 7), 4.0),
        "malignancy_targets": torch.ones(1),
        "risk_target": torch.tensor(1.0),
        "risk_target_valid": torch.tensor(True),
        "ignored_target_masks": torch.empty(0, 8, 8, 8, dtype=torch.uint8),
    }



class TinyEncoder(nn.Module):
    output_dim = 4
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, 3, padding=1)
    def forward(self, x):
        return self.conv(x).tanh().mean((2, 3, 4))

