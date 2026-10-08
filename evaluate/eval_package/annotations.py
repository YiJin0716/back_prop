"""Official reader scores joined by annotation ID, and reader-mask consensus."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .types import GroundTruthNodule, RATING_NAMES, compact_mask

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "vista3D/data/folds/fold_0.json"
DEFAULT_ANNOTATIONS = ROOT / "all_ct_annotations.csv"
DEFAULT_IDENTITIES = ROOT / "nodule_iden.csv"
DEFAULT_MASK_DIR = Path("/usr/project/rudinlab/datasets/LIDC_IDRI/seg_result/official_seg_result/annolvl")


def case_key(case):
    return f"{case['patient_id']}__scan{int(float(case['scan_id']))}"


def load_test_cases(manifest=DEFAULT_MANIFEST, split="testing"):
    """Read fold manifests or frozen cohort.json; reject patient split overlap."""
    manifest = Path(manifest)
    payload = json.loads(manifest.read_text())
    splits = payload.get("splits", payload)
    rows = splits.get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No cases for split {split!r} in {manifest}")
    result = []
    for row in rows:
        case = dict(row)
        case["scan_id"] = str(int(float(case.get("scan_id", case.get("scan_index")))))
        image = Path(case["image"])
        case["image"] = str(image if image.is_absolute() else manifest.parent / image)
        result.append(case)
    if len({case_key(c) for c in result}) != len(result):
        raise ValueError("Duplicate screening in evaluation split")
    if split != "training":
        overlap = {c["patient_id"] for c in result} & {
            c["patient_id"] for c in splits.get("training", [])}
        if overlap:
            raise ValueError(f"Training/evaluation patient overlap: {sorted(overlap)}")
    return result


class OfficialAnnotations:
    """Identity CSV supplies grouping only; numeric GT comes from official CSV.

    Rows without an annotation_index represent scans without reader nodules.
    Missing/duplicate joins for actual annotations fail instead of silently
    turning an unannotated case into a negative case.
    """

    def __init__(self, annotations_csv=DEFAULT_ANNOTATIONS, identity_csv=DEFAULT_IDENTITIES,
                 mask_dir=DEFAULT_MASK_DIR, *, cases=None, min_votes=2):
        if min_votes < 1:
            raise ValueError("min_votes must be positive")
        self.min_votes = int(min_votes)
        self.mask_dir = Path(mask_dir)
        self.annotations_csv = Path(annotations_csv)
        self.identity_csv = Path(identity_csv)
        allowed = None if cases is None else {case_key(c) for c in cases}
        identities = {}
        for key, row in self._rows(self.identity_csv, allowed):
            if not row.get("nodule_id", "").strip():
                raise ValueError(f"Missing physical nodule ID for {key}")
            if key in identities:
                raise ValueError(f"Duplicate annotation identity: {key}")
            identities[key] = int(float(row["nodule_id"]))
        self.grouped = {}
        self.known_cases = set()
        seen = set()
        for key, row in self._rows(self.annotations_csv, allowed, self.known_cases):
            if key in seen:
                raise ValueError(f"Duplicate official annotation: {key}")
            seen.add(key)
            if key not in identities:
                raise ValueError(f"Official annotation has no physical nodule identity: {key}")
            scores = {name: float(row[name]) for name in RATING_NAMES}
            if any(not np.isfinite(x) or x < 1 or x > 5 or x != int(x)
                   for x in scores.values()):
                raise ValueError(f"Invalid official reader ratings: {key}")
            self.grouped.setdefault(key[0], {}).setdefault(identities[key], []).append((key[1], scores))
        missing = set(identities) - seen
        if missing:
            raise ValueError(f"Identity references absent official annotations: {sorted(missing)[:5]}")
        if allowed is not None and allowed - self.known_cases:
            raise ValueError(f"Cases absent from official annotation CSV: {sorted(allowed-self.known_cases)}")

    @staticmethod
    def _rows(path, allowed, known_cases=None):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                scan = str(int(float(row["scan_index"])))
                case_id = f"{row['patient_id'].strip()}__scan{scan}"
                if allowed is not None and case_id not in allowed:
                    continue
                if known_cases is not None:
                    known_cases.add(case_id)
                annotation = row.get("annotation_index", "").strip()
                if not annotation or annotation.lower() == "nan":
                    continue
                yield (case_id, int(float(annotation))), row

    def load_nodules(self, case_id, canonical_image, zoom):
        from scipy.ndimage import zoom as resample
        from back_prop.common.base_data import _read_official_mask

        if case_id not in self.known_cases:
            raise ValueError(f"Case absent from official annotation CSV: {case_id}")
        result = []
        for nodule_id, readers in sorted(self.grouped.get(case_id, {}).items()):
            readers = sorted(readers)
            paths = [self.mask_dir / f"{case_id}__ann{ann}_mask.nii.gz" for ann, _ in readers]
            votes = np.zeros(canonical_image.shape, dtype=np.uint16)
            for path in paths:
                votes += _read_official_mask(path, canonical_image)
            consensus = votes >= min(self.min_votes, len(readers))
            mask = resample(consensus.astype(np.uint8), zoom, order=0,
                            mode="nearest", prefilter=False).astype(bool)
            result.append(GroundTruthNodule(
                nodule_id, compact_mask(mask), tuple(ann for ann, _ in readers),
                {name: [scores[name] for _, scores in readers] for name in RATING_NAMES},
                tuple(str(path) for path in paths)))
        return result


def load_image(case):
    """Exactly the canonical XYZ and scipy zoom convention used in V3 training."""
    import nibabel as nib
    from scipy.ndimage import zoom as resample

    source = nib.as_closest_canonical(nib.load(case["image"]))
    raw = np.asarray(source.dataobj, dtype=np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite 3D HU CT")
    zoom = nib.affines.voxel_sizes(source.affine)  # nominal 1-mm grid
    hu = resample(raw, zoom, order=1, mode="nearest", prefilter=False)
    # ndimage.zoom(grid_mode=False) aligns voxel-centre endpoints.
    ratio = np.divide(np.asarray(source.shape)-1, np.asarray(hu.shape)-1,
                      out=np.ones(3), where=np.asarray(hu.shape) > 1)
    transform = np.diag([*ratio, 1.0])
    affine = source.affine @ transform
    return hu, affine, source, zoom
