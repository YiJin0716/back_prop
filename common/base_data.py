"""Whole-volume LIDC input pipeline with nodule-identity-preserving targets."""

from __future__ import annotations

import csv
from collections import defaultdict
import errno
import json
import logging
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import Dataset

from back_prop.lidc_policy import (
    POLICY_NAME,
    binary_malignancy_target,
    is_indeterminate_malignancy,
    reader_mean_malignancy,
)


ROOT = Path(__file__).parents[2]
DEFAULT_MANIFEST = ROOT / "vista3D/data/folds/fold_0.json"
DEFAULT_ANNOTATIONS = ROOT / "nodule_iden.csv"
DEFAULT_OFFICIAL_MASK_DIR = Path(
    "/usr/project/rudinlab/datasets/LIDC_IDRI/seg_result/official_seg_result/annolvl"
)
FEATURE_COLUMNS = ("sphericity", "margin", "lobulation", "spiculation", "texture", "subtlety")


def _read_official_mask(path: Path, image_nii, *, attempts: int = 6,
                        initial_delay: float = 1.0) -> np.ndarray:
    """Materialize one mask, retrying bounded transient shared-filesystem errors.

    Include proxy reads in the retry: nib.load alone only loads the header.
    Geometry errors and permissions failures remain fatal. Persistent I/O errors
    also fail the run; no annotation is skipped or replaced by an empty mask.
    """
    if attempts < 1 or initial_delay < 0:
        raise ValueError("Invalid mask read retry settings")
    retryable_errno = {errno.ENOENT, errno.ESTALE, errno.EIO, errno.ETIMEDOUT,
                       errno.EAGAIN, errno.ECONNRESET}
    for attempt in range(1, attempts + 1):
        try:
            mask_nii = nib.as_closest_canonical(nib.load(path), enforce_diag=False)
            if mask_nii.shape != image_nii.shape or not np.allclose(mask_nii.affine, image_nii.affine):
                raise ValueError(f"Image/official-mask geometry mismatch: {path}")
            return np.asarray(mask_nii.dataobj, dtype=np.uint8) > 0
        except (OSError, EOFError) as exc:
            # nibabel can re-raise FileNotFoundError without preserving errno.
            retryable = (isinstance(exc, (FileNotFoundError, EOFError))
                         or getattr(exc, "errno", None) in retryable_errno)
            if not retryable or attempt == attempts:
                raise
            delay = initial_delay * 2 ** (attempt - 1)
            logging.getLogger(__name__).warning(
                "Official mask read failed; retrying path=%s attempt=%d/%d "
                "delay_seconds=%s error=%r errno=%s",
                path, attempt, attempts, delay, exc, getattr(exc, "errno", None),
            )
            time.sleep(delay)


def _case_key(case: dict[str, str]) -> tuple[str, str]:
    return case["patient_id"], str(int(float(case["scan_id"])))


def load_cases(manifest: str | Path, split: str = "training") -> list[dict[str, str]]:
    payload = json.loads(Path(manifest).read_text())
    cases = payload.get(split)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No {split!r} cases in {manifest}")
    return cases


def load_nodule_annotations(path: str | Path) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Group reader annotations into physical nodules using the CSV nodule mapping."""
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("nodule_id") or not row.get("annotation_index"):
                continue
            try:
                patient = row["patient_id"].strip()
                scan = str(int(float(row["scan_index"])))
                nodule_id = int(float(row["nodule_id"]))
                annotation_id = int(float(row["annotation_index"]))
                features = np.asarray([float(row[name]) for name in FEATURE_COLUMNS], dtype=np.float32)
                malignancy = float(row["malignancy"])
            except (ValueError, TypeError):
                continue
            grouped[(patient, scan, nodule_id)].append(
                {"annotation_id": annotation_id, "features": features, "malignancy": malignancy}
            )

    result: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for (patient, scan, nodule_id), rows in sorted(grouped.items()):
        reader_features = np.stack([row["features"] for row in rows]).astype(np.float32)
        reader_malignancy = np.asarray(
            [row["malignancy"] for row in rows], dtype=np.float32
        )
        result[(patient, scan)].append(
            {
                "nodule_id": nodule_id,
                "annotation_ids": tuple(int(row["annotation_id"]) for row in rows),
                "reader_features": reader_features,
                "reader_malignancy": reader_malignancy,
                "features": reader_features.mean(axis=0).astype(np.float32),
                "malignancy": reader_mean_malignancy(reader_malignancy),
            }
        )
    return dict(result)


def _occupancy_downsample(
    locations: np.ndarray,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Rasterize positive voxels onto a coarse grid without dropping tiny nodules."""
    target = np.zeros(output_shape, dtype=np.uint8)
    if not len(locations):
        return target
    scale = np.asarray(output_shape, dtype=np.float64) / np.asarray(input_shape, dtype=np.float64)
    coarse = np.floor(locations * scale).astype(np.int64)
    coarse = np.minimum(coarse, np.asarray(output_shape, dtype=np.int64) - 1)
    target[tuple(coarse.T)] = 1
    return target


class WholeCTLIDCDataset(Dataset):
    """Return a complete 1-mm CT and one consensus target per physical nodule.

    Reader-level masks are grouped by ``nodule_id`` before voting. No connected
    components or centroid-based reassignment is used.
    """

    def __init__(
        self,
        cases: list[dict[str, str]],
        annotations_csv: str | Path = DEFAULT_ANNOTATIONS,
        official_mask_dir: str | Path = DEFAULT_OFFICIAL_MASK_DIR,
        target_spacing: float = 1.0,
        target_mask_shape: tuple[int, int, int] = (64, 64, 64),
        min_votes: int = 2,
        exclude_indeterminate: bool = True,
        drop_cases_without_targets: bool = True,
        max_cases: int | None = None,
    ) -> None:
        self.annotations = load_nodule_annotations(annotations_csv)
        self.exclude_indeterminate = bool(exclude_indeterminate)
        self.drop_cases_without_targets = bool(drop_cases_without_targets)
        unfiltered_cases = list(cases)
        input_keys = {_case_key(case) for case in unfiltered_cases}
        input_nodules = [
            nodule for key, nodules in self.annotations.items() if key in input_keys
            for nodule in nodules
        ]
        if self.drop_cases_without_targets:
            unfiltered_cases = [
                case for case in unfiltered_cases
                if any(
                    not self.exclude_indeterminate
                    or not is_indeterminate_malignancy(float(nodule["malignancy"]))
                    for nodule in self.annotations.get(_case_key(case), [])
                )
            ]
        eligible_case_count = len(unfiltered_cases)
        if max_cases is not None:
            if max_cases < 1:
                raise ValueError("max_cases must be positive")
            unfiltered_cases = unfiltered_cases[:max_cases]
        if not unfiltered_cases:
            raise ValueError("No cases remain after applying the LIDC malignancy policy")
        self.cases = unfiltered_cases
        self.official_mask_dir = Path(official_mask_dir)
        self.target_spacing = float(target_spacing)
        self.target_mask_shape = tuple(int(value) for value in target_mask_shape)
        self.min_votes = int(min_votes)
        if self.min_votes < 1:
            raise ValueError("min_votes must be positive")
        selected = {_case_key(case) for case in self.cases}
        selected_nodules = [
            nodule for key, nodules in self.annotations.items() if key in selected
            for nodule in nodules
        ]
        self.policy_summary = {
            "name": POLICY_NAME,
            "exclude_indeterminate": self.exclude_indeterminate,
            "input_cases": len(cases),
            "eligible_cases_before_limit": eligible_case_count,
            "retained_cases": len(self.cases),
            "dropped_cases_without_targets": len(cases) - eligible_case_count,
            "truncated_cases_for_smoke_test": eligible_case_count - len(self.cases),
            "physical_nodules": len(input_nodules),
            "retained_physical_nodules": sum(
                not self.exclude_indeterminate
                or not is_indeterminate_malignancy(float(nodule["malignancy"]))
                for nodule in selected_nodules
            ),
            "excluded_indeterminate_nodules": sum(
                self.exclude_indeterminate
                and is_indeterminate_malignancy(float(nodule["malignancy"]))
                for nodule in input_nodules
            ),
        }

    def __len__(self) -> int:
        return len(self.cases)

    def _consensus_mask(
        self,
        patient: str,
        scan: str,
        annotation_ids: tuple[int, ...],
        image_nii: nib.Nifti1Image,
    ) -> np.ndarray:
        votes = np.zeros(image_nii.shape, dtype=np.uint8)
        for annotation_id in annotation_ids:
            path = self.official_mask_dir / f"{patient}__scan{scan}__ann{annotation_id}_mask.nii.gz"
            votes += _read_official_mask(path, image_nii)
        threshold = min(self.min_votes, len(annotation_ids))
        return votes >= threshold

    def __getitem__(self, index: int) -> dict[str, object]:
        case = self.cases[index]
        patient, scan = _case_key(case)
        image_nii = nib.as_closest_canonical(nib.load(case["image"]))
        spacing = nib.affines.voxel_sizes(image_nii.affine)
        zoom = np.asarray(spacing, dtype=np.float64) / self.target_spacing
        raw = np.asarray(image_nii.dataobj, dtype=np.float32)
        image_iso = ndimage.zoom(raw, zoom, order=1, mode="nearest", prefilter=False)
        del raw
        image_iso = ((np.clip(image_iso, -1024.0, 1024.0) + 1024.0) / 2048.0).astype(np.float32)

        target_masks: list[np.ndarray] = []
        target_boxes: list[np.ndarray] = []
        semantics: list[np.ndarray] = []
        malignancy: list[float] = []
        nodule_ids: list[int] = []
        target_mask_crops: list[torch.Tensor] = []
        target_mask_origins: list[np.ndarray] = []
        ignored_target_masks: list[np.ndarray] = []
        ignored_target_boxes: list[np.ndarray] = []
        ignored_target_mask_crops: list[torch.Tensor] = []
        ignored_target_mask_origins: list[np.ndarray] = []
        ignored_nodule_ids: list[int] = []
        spatial = np.asarray(image_iso.shape, dtype=np.float32)
        for nodule in self.annotations.get((patient, scan), []):
            consensus = self._consensus_mask(patient, scan, nodule["annotation_ids"], image_nii)
            consensus_iso = ndimage.zoom(
                consensus.astype(np.uint8), zoom, order=0, mode="nearest", prefilter=False
            ) > 0
            locations = np.argwhere(consensus_iso)
            if not len(locations):
                continue
            lower = locations.min(axis=0).astype(np.float32) / spatial
            upper = (locations.max(axis=0) + 1).astype(np.float32) / spatial
            box = np.concatenate(((lower + upper) * 0.5, upper - lower))
            coarse_mask = _occupancy_downsample(
                locations, image_iso.shape, self.target_mask_shape
            )
            if self.exclude_indeterminate and is_indeterminate_malignancy(
                float(nodule["malignancy"])
            ):
                ignored_target_boxes.append(box)
                ignored_target_masks.append(coarse_mask)
                crop_lower = locations.min(axis=0).astype(np.int64)
                crop_upper = locations.max(axis=0).astype(np.int64) + 1
                crop_slices = tuple(
                    slice(int(start), int(stop))
                    for start, stop in zip(crop_lower, crop_upper)
                )
                ignored_target_mask_crops.append(
                    torch.from_numpy(consensus_iso[crop_slices].astype(np.uint8, copy=True))
                )
                ignored_target_mask_origins.append(crop_lower)
                ignored_nodule_ids.append(int(nodule["nodule_id"]))
                continue
            target_boxes.append(box)
            target_masks.append(coarse_mask)
            crop_lower = locations.min(axis=0).astype(np.int64)
            crop_upper = locations.max(axis=0).astype(np.int64) + 1
            crop_slices = tuple(
                slice(int(start), int(stop)) for start, stop in zip(crop_lower, crop_upper)
            )
            target_mask_crops.append(
                torch.from_numpy(consensus_iso[crop_slices].astype(np.uint8, copy=True))
            )
            target_mask_origins.append(crop_lower)
            semantics.append(
                np.concatenate(
                    ([np.log1p(float(len(locations)))], np.asarray(nodule["features"], dtype=np.float32))
                ).astype(np.float32)
            )
            malignancy.append(binary_malignancy_target(float(nodule["malignancy"])))
            nodule_ids.append(int(nodule["nodule_id"]))

        count = len(target_masks)
        masks_array = (
            np.stack(target_masks) if count else np.empty((0, *self.target_mask_shape), dtype=np.uint8)
        )
        boxes_array = np.stack(target_boxes) if count else np.empty((0, 6), dtype=np.float32)
        semantics_array = np.stack(semantics) if count else np.empty((0, 7), dtype=np.float32)
        malignancy_array = np.asarray(malignancy, dtype=np.float32)
        scan_risk = float(malignancy_array.max()) if count else 0.0
        ignored_count = len(ignored_target_masks)
        ignored_masks_array = (
            np.stack(ignored_target_masks)
            if ignored_count else np.empty((0, *self.target_mask_shape), dtype=np.uint8)
        )
        ignored_boxes_array = (
            np.stack(ignored_target_boxes)
            if ignored_count else np.empty((0, 6), dtype=np.float32)
        )
        # A definitively malignant nodule makes the scan positive regardless
        # of any ignored nodule.  A nominally negative scan containing an
        # indeterminate nodule has no unambiguous binary scan target.
        risk_valid = bool(scan_risk > 0.0 or ignored_count == 0)
        return {
            "image": torch.from_numpy(image_iso).unsqueeze(0),
            "target_masks": torch.from_numpy(masks_array),
            "target_boxes": torch.from_numpy(boxes_array),
            "semantic_targets": torch.from_numpy(semantics_array),
            "malignancy_targets": torch.from_numpy(malignancy_array),
            "risk_target": torch.tensor(scan_risk, dtype=torch.float32),
            "risk_target_valid": torch.tensor(risk_valid, dtype=torch.bool),
            "nodule_ids": torch.tensor(nodule_ids, dtype=torch.long),
            "target_mask_crops": target_mask_crops,
            "target_mask_origins": torch.from_numpy(
                np.stack(target_mask_origins)
                if target_mask_origins else np.empty((0, 3), dtype=np.int64)
            ),
            "ignored_target_masks": torch.from_numpy(ignored_masks_array),
            "ignored_target_boxes": torch.from_numpy(ignored_boxes_array),
            "ignored_target_mask_crops": ignored_target_mask_crops,
            "ignored_target_mask_origins": torch.from_numpy(
                np.stack(ignored_target_mask_origins)
                if ignored_target_mask_origins else np.empty((0, 3), dtype=np.int64)
            ),
            "ignored_nodule_ids": torch.tensor(ignored_nodule_ids, dtype=torch.long),
            "excluded_indeterminate_nodules": ignored_count,
            "malignancy_policy": POLICY_NAME,
            "case_id": f"{patient}__scan{scan}",
            "original_shape": tuple(int(value) for value in image_nii.shape),
            "isotropic_shape": tuple(int(value) for value in image_iso.shape),
        }


def whole_ct_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("Whole-CT training requires batch_size=1 per process")
    return batch[0]
