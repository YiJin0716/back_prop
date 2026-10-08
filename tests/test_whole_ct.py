import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from back_prop.common.base_data import WholeCTLIDCDataset
from back_prop.common.base_loss import WholeCTCriterion
from back_prop.common.base_model import WholeCTJointModel, differentiable_box_sample, sliding_starts


class TinySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        return (image - 0.4) * self.scale


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(1, 4, 1)

    def forward(self, roi):
        return self.projection(roi).mean(dim=(-3, -2, -1))


def make_model():
    return WholeCTJointModel(
        window_size=(8, 8, 8), overlap=0.25, roi_size=(4, 4, 4),
        num_queries=4, detr_hidden_dim=16, detr_coarse_shape=(4, 4, 4),
        detr_nheads=4, detr_encoder_layers=1, detr_decoder_layers=1,
        mask_shape=(8, 8, 8), vista_feature_dim=1,
        segmenter_factory=TinySegmenter, encoder_factory=TinyEncoder,
        encoder_dim=4, use_checkpoint=False,
    )


def test_sliding_windows_cover_complete_volume():
    starts = sliding_starts(19, 8, 0.25)
    covered = np.zeros(19, dtype=bool)
    for start in starts:
        covered[start:start + 8] = True
    assert covered.all()
    assert starts[-1] == 11


def test_whole_ct_detr_queries_and_risk_gradient():
    model = make_model()
    image = torch.zeros(1, 18, 19, 20)
    image[:, 2:5, 3:6, 4:7] = 1
    image[:, 12:15, 13:16, 14:17] = 1
    output = model(image)
    assert output.window_count > 1
    assert output.discovered_mask.shape == image.shape[-3:]
    assert output.boxes.shape == (4, 6)
    assert output.object_logits.shape == (4,)
    assert output.candidate_logits.shape == (4, 8, 8, 8)
    F.binary_cross_entropy_with_logits(output.risk_logit, torch.ones_like(output.risk_logit)).backward()
    assert model.segmenter.scale.grad is not None
    assert torch.isfinite(model.segmenter.scale.grad)
    assert model.segmenter.scale.grad.abs() > 0
    assert model.detr.box_head.layers[-1].weight.grad is not None
    assert model.detr.box_head.layers[-1].weight.grad.abs().sum() > 0


def test_differentiable_3d_box_sampling_has_box_gradient():
    volume = torch.linspace(0, 1, 12 ** 3).reshape(1, 1, 12, 12, 12)
    box = torch.tensor([0.5, 0.5, 0.5, 0.6, 0.6, 0.6], requires_grad=True)
    differentiable_box_sample(volume, box, (6, 6, 6)).sum().backward()
    assert box.grad is not None
    assert torch.isfinite(box.grad).all()
    assert box.grad.abs().sum() > 0


def test_three_part_loss():
    model = make_model()
    image = torch.zeros(1, 12, 12, 12)
    image[:, 4:8, 4:8, 4:8] = 1
    target_mask = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
    target_mask[:, 3:6, 3:6, 3:6] = 1
    batch = {
        "image": image, "target_masks": target_mask,
        "target_boxes": torch.tensor([[0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3]]),
        "semantic_targets": torch.tensor([[4.1, 3, 3, 3, 3, 3, 3]], dtype=torch.float32),
        "malignancy_targets": torch.tensor([1.0]), "risk_target": torch.tensor(1.0),
    }
    output = model(image)
    losses = WholeCTCriterion((8, 8, 8), (2, 3, 5))(output, batch)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert losses.positive_candidates == 1
    assert model.detr.box_head.layers[-1].weight.grad is not None


def test_indeterminate_nodule_forces_one_query_out_of_no_object_loss():
    model = make_model()
    image = torch.zeros(1, 12, 12, 12)
    output = model(image)
    target = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
    target[:, 2:4, 2:4, 2:4] = 1
    ignored = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
    ignored[:, 6:7, 6:7, 6:7] = 1
    batch = {
        "target_masks": target,
        "target_boxes": torch.tensor([[0.35, 0.35, 0.35, 0.2, 0.2, 0.2]]),
        "semantic_targets": torch.full((1, 7), 3.0),
        "malignancy_targets": torch.tensor([0.0]),
        "risk_target": torch.tensor(0.0),
        "risk_target_valid": torch.tensor(False),
        "ignored_target_masks": ignored,
        "ignored_target_boxes": torch.tensor([[0.8, 0.8, 0.8, 0.05, 0.05, 0.05]]),
    }
    losses = WholeCTCriterion((8, 8, 8))(output, batch)
    assert losses.ignored_candidates >= 1
    assert losses.risk_valid is False
    assert losses.risk.item() == 0.0


def test_dataset_returns_complete_scan(tmp_path: Path):
    shape = (11, 13, 15)
    image = np.full(shape, -900, dtype=np.float32)
    first = np.zeros(shape, dtype=np.uint8)
    second = np.zeros(shape, dtype=np.uint8)
    first[2:5, 3:6, 4:7] = 1
    second[3:6, 4:7, 5:8] = 1
    affine = np.eye(4)
    image_path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(first, affine), tmp_path / "P__scan1__ann1_mask.nii.gz")
    nib.save(nib.Nifti1Image(second, affine), tmp_path / "P__scan1__ann2_mask.nii.gz")
    annotations = tmp_path / "annotations.csv"
    with annotations.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "patient_id", "scan_index", "annotation_index", "nodule_id",
            "sphericity", "margin", "lobulation", "spiculation", "texture", "subtlety",
            "malignancy",
        ])
        writer.writeheader()
        for annotation_id in (1, 2):
            writer.writerow({
                "patient_id": "P", "scan_index": 1, "annotation_index": annotation_id,
                "nodule_id": 7, "sphericity": 3, "margin": 4, "lobulation": 2,
                "spiculation": 1, "texture": 5, "subtlety": 4, "malignancy": 4,
            })
    dataset = WholeCTLIDCDataset(
        [{"patient_id": "P", "scan_id": "1", "image": str(image_path)}],
        annotations_csv=annotations, official_mask_dir=tmp_path, target_mask_shape=(8, 8, 8),
    )
    item = dataset[0]
    assert item["image"].shape == (1, *shape)
    assert item["target_masks"].shape == (1, 8, 8, 8)
    assert item["target_masks"].any()
    assert item["target_boxes"].shape == (1, 6)
    assert item["nodule_ids"].tolist() == [7]
    assert item["malignancy_targets"].tolist() == [1.0]
    assert item["isotropic_shape"] == shape


def test_dataset_excludes_physical_mean_three_but_keeps_ignore_region(tmp_path: Path):
    shape = (10, 10, 10)
    image_path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), image_path)
    annotations = tmp_path / "annotations.csv"
    fieldnames = [
        "patient_id", "scan_index", "annotation_index", "nodule_id",
        "sphericity", "margin", "lobulation", "spiculation", "texture",
        "subtlety", "malignancy",
    ]
    rows = []
    annotation_id = 0
    # [2,4] has physical mean 3 and must be excluded as one whole nodule;
    # [2,3] and [3,4] remain benign/malignant despite containing reader 3.
    for nodule_id, ratings, start in ((1, (2, 4), 1), (2, (2, 3), 4), (3, (3, 4), 7)):
        for rating in ratings:
            annotation_id += 1
            mask = np.zeros(shape, dtype=np.uint8)
            mask[start:start + 2, start:start + 2, start:start + 2] = 1
            nib.save(
                nib.Nifti1Image(mask, np.eye(4)),
                tmp_path / f"P__scan1__ann{annotation_id}_mask.nii.gz",
            )
            rows.append({
                "patient_id": "P", "scan_index": 1, "annotation_index": annotation_id,
                "nodule_id": nodule_id, "sphericity": 3, "margin": 3,
                "lobulation": 3, "spiculation": 3, "texture": 3,
                "subtlety": 3, "malignancy": rating,
            })
    with annotations.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    dataset = WholeCTLIDCDataset(
        [{"patient_id": "P", "scan_id": "1", "image": str(image_path)}],
        annotations_csv=annotations,
        official_mask_dir=tmp_path,
        target_mask_shape=(8, 8, 8),
    )
    item = dataset[0]
    assert item["nodule_ids"].tolist() == [2, 3]
    assert item["malignancy_targets"].tolist() == [0.0, 1.0]
    assert item["ignored_nodule_ids"].tolist() == [1]
    assert item["ignored_target_masks"].shape == (1, 8, 8, 8)
    assert item["risk_target"].item() == 1.0
    assert item["risk_target_valid"].item() is True


def test_negative_scan_with_indeterminate_nodule_has_no_binary_risk_target(tmp_path: Path):
    shape = (8, 8, 8)
    image_path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), image_path)
    annotations = tmp_path / "annotations.csv"
    fields = [
        "patient_id", "scan_index", "annotation_index", "nodule_id",
        "sphericity", "margin", "lobulation", "spiculation", "texture",
        "subtlety", "malignancy",
    ]
    rows = []
    for annotation_id, nodule_id, rating, start in (
        (1, 1, 2, 1), (2, 2, 2, 4), (3, 2, 4, 4)
    ):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[start:start + 2, start:start + 2, start:start + 2] = 1
        nib.save(
            nib.Nifti1Image(mask, np.eye(4)),
            tmp_path / f"P__scan1__ann{annotation_id}_mask.nii.gz",
        )
        rows.append({
            "patient_id": "P", "scan_index": 1, "annotation_index": annotation_id,
            "nodule_id": nodule_id, "sphericity": 3, "margin": 3,
            "lobulation": 3, "spiculation": 3, "texture": 3,
            "subtlety": 3, "malignancy": rating,
        })
    with annotations.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    dataset = WholeCTLIDCDataset(
        [{"patient_id": "P", "scan_id": "1", "image": str(image_path)}],
        annotations_csv=annotations, official_mask_dir=tmp_path,
        target_mask_shape=(8, 8, 8),
    )
    item = dataset[0]
    assert item["nodule_ids"].tolist() == [1]
    assert item["ignored_nodule_ids"].tolist() == [2]
    assert item["risk_target"].item() == 0.0
    assert item["risk_target_valid"].item() is False
