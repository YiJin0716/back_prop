"""PyTorch port of the cited Keras HSCNN, with EDICNet's three semantic tasks.

Copyright 2018 Shiwen Shen; Apache-2.0 (hscnn_upstream/LICENSE).
Derived from hscnn_upstream/model/cnn_model_module.py, build_no_direct_connection.
EDICNet changes: dropout .8; diameter/consistency/margin classes 3/2/2.
"""
import torch
from torch import nn


def dense_block(inputs, outputs, dropout=.8):
    return nn.Sequential(nn.Linear(inputs, outputs), nn.BatchNorm1d(outputs, eps=.001, momentum=.01),
                         nn.ReLU(), nn.Dropout(dropout))


class HSCNN(nn.Module):
    def __init__(self, patch_size=52):
        super().__init__()
        layers = []
        channels = 1
        for width in (16, 32):
            for _ in range(2):
                layers.extend([nn.Conv3d(channels, width, 3, padding=1),
                               nn.BatchNorm3d(width, eps=.001, momentum=.01), nn.ReLU()])
                channels = width
            layers.append(nn.MaxPool3d(2))
        self.features = nn.Sequential(*layers)
        flat = 32 * (patch_size // 4) ** 3
        self.flatten_dropout = nn.Dropout(.8)
        self.task_bases = nn.ModuleList([dense_block(flat, 256) for _ in range(3)])
        self.semantic_heads = nn.ModuleList([nn.Sequential(dense_block(256, 64), nn.Linear(64, n))
                                             for n in (3, 2, 2)])
        self.malignancy_head = nn.Sequential(dense_block(flat + 3 * 256, 256), nn.Linear(256, 2))
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, image):
        # Channels-last flatten matches the Keras connectivity ordering.
        features = self.features(image).permute(0, 2, 3, 4, 1).contiguous().flatten(1)
        features = self.flatten_dropout(features)
        bases = [base(features) for base in self.task_bases]
        semantic = [head(base) for head, base in zip(self.semantic_heads, bases)]
        malignancy = self.malignancy_head(torch.cat([*bases, features], dim=1))
        return malignancy, semantic
