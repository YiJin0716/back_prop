import torch
from torch import nn

from back_prop.common.coarse_loss import WholeCTCriterionV2
from back_prop.common.coarse_model import WholeCTJointModelV2


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


def test_tiny_end_to_end_forward_loss_and_gradients():
    model = WholeCTJointModelV2(
        window_size=(8, 8, 8),
        overlap=0.25,
        screen_shape=(8, 8, 8),
        roi_shape=(8, 8, 8),
        num_queries=4,
        detr_hidden_dim=16,
        detr_coarse_shape=(4, 4, 4),
        detr_nheads=4,
        detr_encoder_layers=1,
        detr_decoder_layers=1,
        vista_feature_dim=1,
        context_channels=2,
        refiner_base_channels=2,
        hard_negatives=1,
        teacher_full_epochs=1,
        teacher_zero_epoch=2,
        teacher_jitter=0,
        segmenter_factory=TinySegmenter,
        radiomics_factory=TinyRadiomics,
        ordinal_heads_factory=TinyOrdinal,
        use_checkpoint=False,
    )
    batch = make_batch()
    model.train()
    output = model(batch["image"], batch=batch, epoch=0)
    assert output.coarse_mask_logits.shape == (4, 8, 8, 8)
    assert output.fine_mask_logits.shape == (2, 8, 8, 8)
    assert output.fine_target_masks.shape == output.fine_mask_logits.shape
    assert output.semantic_probabilities.shape == (2, 7, 5)
    assert output.matched_query_indices.numel() == 1
    losses = WholeCTCriterionV2()(output, batch)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.segmenter.scale.grad is not None
    assert model.segmenter.scale.grad.abs() > 0
    assert model.detr.box_head.layers[-1].weight.grad is not None
    assert model.detr.box_head.layers[-1].weight.grad.abs().sum() > 0
    assert model.refiner.residual_head.weight.grad is not None
    assert model.refiner.residual_head.weight.grad.abs().sum() > 0
    assert model.ordinal_heads.head.weight.grad is not None
    assert model.ordinal_heads.head.weight.grad.abs().sum() > 0


def test_inference_never_uses_teacher_routing():
    model = WholeCTJointModelV2(
        window_size=(8, 8, 8), screen_shape=(8, 8, 8), roi_shape=(8, 8, 8),
        num_queries=2, detr_hidden_dim=16, detr_coarse_shape=(4, 4, 4),
        detr_nheads=4, detr_encoder_layers=1, detr_decoder_layers=1,
        vista_feature_dim=1, context_channels=2, refiner_base_channels=2,
        teacher_full_epochs=1, teacher_zero_epoch=2,
        segmenter_factory=TinySegmenter, radiomics_factory=TinyRadiomics,
        ordinal_heads_factory=TinyOrdinal, use_checkpoint=False,
    )
    model.eval()
    image = torch.zeros(1, 8, 8, 8)
    output = model(image, object_threshold=0.0)
    assert output.refined_query_indices.numel() == 2
    assert output.teacher_forced_count == 0
    assert output.fallback_count == 0


def test_final_training_epoch_does_not_fallback_to_ground_truth_crop():
    model = WholeCTJointModelV2(
        window_size=(8, 8, 8), screen_shape=(8, 8, 8), roi_shape=(8, 8, 8),
        num_queries=2, detr_hidden_dim=16, detr_coarse_shape=(4, 4, 4),
        detr_nheads=4, detr_encoder_layers=1, detr_decoder_layers=1,
        vista_feature_dim=1, context_channels=2, refiner_base_channels=2,
        teacher_full_epochs=1, teacher_zero_epoch=2, teacher_jitter=0,
        segmenter_factory=TinySegmenter, radiomics_factory=TinyRadiomics,
        ordinal_heads_factory=TinyOrdinal, use_checkpoint=False,
    )
    batch = make_batch()
    batch["target_boxes"] = torch.tensor([[1 / 12, 1 / 12, 1 / 12, 2 / 12, 2 / 12, 2 / 12]])
    batch["target_mask_crops"] = [torch.ones(2, 2, 2, dtype=torch.uint8)]
    batch["target_mask_origins"] = torch.tensor([[0, 0, 0]])
    model.train()
    output = model(
        batch["image"], batch=batch, epoch=1, teacher_probability=0.0,
        compute_diagnostics=False,
    )
    assert output.teacher_forced_count == 0
    assert output.fallback_count == 0
    # A missed predicted ROI is recorded and excluded from fine/semantic
    # supervision rather than silently replaced by a GT-centred crop.
    assert not output.fine_supervision_valid[0]
    assert output.scan_path_valid is False
