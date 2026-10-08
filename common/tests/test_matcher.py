import torch

from back_prop.common.matcher import identify_ignored_queries


def test_tiny_ignored_nodule_is_assigned_even_when_soft_dice_is_tiny():
    boxes = torch.tensor([
        [0.2, 0.2, 0.2, 0.04, 0.04, 0.04],
        [0.5, 0.5, 0.5, 0.04, 0.04, 0.04],
        [0.8, 0.8, 0.8, 0.04, 0.04, 0.04],
    ])
    logits = torch.zeros(3, 64, 64, 64)
    ignored_mask = torch.zeros(1, 64, 64, 64, dtype=torch.uint8)
    ignored_mask[:, 31:34, 31:34, 31:34] = 1
    ignored_box = torch.tensor([[0.5, 0.5, 0.5, 0.03, 0.03, 0.03]])
    result = identify_ignored_queries(
        boxes, logits, torch.tensor([0]), ignored_box, ignored_mask
    )
    assert not result[0]
    assert result.sum() >= 1

