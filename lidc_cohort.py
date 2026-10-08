"""Fail-closed validation and provenance for LIDC physical-nodule cohorts.

This module deliberately does not trust a derived CSV's binary ``malignancy``
column.  It reconstructs physical nodules by joining reader annotations in
``all_ct_annotations.csv`` to ``nodule_iden.csv``, averages all available
reader malignancy ratings, and applies :mod:`back_prop.lidc_policy`.

The helpers are standard-library only so Slurm shell entry points can run the
policy gate before importing expensive imaging packages.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from back_prop.lidc_policy import (
    POLICY_NAME,
    is_indeterminate_malignancy,
    reader_mean_malignancy,
)


PROVENANCE_SCHEMA = "lidc_physical_nodule_cohort_policy_v1"
LUNA_MATCH_MAX_DISTANCE_MM = 6.0
_POLICY_PATH = Path(__file__).with_name("lidc_policy.py").resolve()


@dataclass(frozen=True, order=True)
class PhysicalNoduleKey:
    patient_id: str
    scan_index: int
    nodule_id: int

    def as_dict(self) -> dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "scan_index": self.scan_index,
            "nodule_id": self.nodule_id,
        }


@dataclass(frozen=True)
class ResolvedNodule:
    row_index: int
    key: PhysicalNoduleKey
    reader_mean_malignancy: float
    matched_annotation_index: int | None = None
    match_distance_mm: float | None = None


def _integer(value: object, *, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be a finite integer, got {value!r}")
    return int(number)


def _finite_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _require_fields(
    fieldnames: Sequence[str] | None,
    required: Iterable[str],
    *,
    path: Path,
) -> None:
    available = set(fieldnames or ())
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _literal_vector(value: object, *, field: str, length: int | None = None) -> list[float]:
    try:
        parsed = ast.literal_eval(str(value))
        result = [float(item) for item in parsed]
    except (SyntaxError, ValueError, TypeError) as error:
        raise ValueError(f"{field} is not a numeric vector: {value!r}") from error
    if length is not None and len(result) != length:
        raise ValueError(f"{field} must contain {length} values, got {len(result)}")
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} contains no values or non-finite values")
    return result


def _source_world_coordinate(row: Mapping[str, object]) -> tuple[float, float, float]:
    centroid = _literal_vector(row["centroid"], field="centroid", length=3)
    row_direction = _literal_vector(row["row_direction"], field="row_direction", length=3)
    col_direction = _literal_vector(row["col_direction"], field="col_direction", length=3)
    slice_zvals = _literal_vector(row["slice_zvals"], field="slice_zvals")
    origin = [
        _finite_float(row[f"origin_{axis}_mm"], field=f"origin_{axis}_mm")
        for axis in "xyz"
    ]
    pixel_spacing = _finite_float(row["pixel_spacing"], field="pixel_spacing")
    world = [
        origin[index]
        + centroid[1] * pixel_spacing * row_direction[index]
        + centroid[0] * pixel_spacing * col_direction[index]
        for index in range(3)
    ]
    lower = max(0, min(len(slice_zvals) - 1, math.floor(centroid[2])))
    upper = max(0, min(len(slice_zvals) - 1, math.ceil(centroid[2])))
    fraction = centroid[2] - math.floor(centroid[2])
    world[2] = slice_zvals[lower] + fraction * (
        slice_zvals[upper] - slice_zvals[lower]
    )
    return tuple(world)


class AuthoritativeLidcIndex:
    """Reader-derived physical-nodule means and LUNA coordinate lookup."""

    def __init__(self, source_csv: os.PathLike[str] | str, nodule_csv: os.PathLike[str] | str):
        self.source_csv = Path(source_csv).resolve()
        self.nodule_csv = Path(nodule_csv).resolve()
        if not self.source_csv.is_file():
            raise FileNotFoundError(self.source_csv)
        if not self.nodule_csv.is_file():
            raise FileNotFoundError(self.nodule_csv)

        annotation_to_key: dict[tuple[str, int, int], PhysicalNoduleKey] = {}
        annotation_rating: dict[tuple[str, int, int], float] = {}
        ratings: dict[PhysicalNoduleKey, list[float]] = defaultdict(list)
        first_annotation: dict[PhysicalNoduleKey, int] = {}
        with self.nodule_csv.open(newline="") as source:
            reader = csv.DictReader(source)
            _require_fields(
                reader.fieldnames,
                ("patient_id", "scan_index", "annotation_index", "nodule_id", "malignancy"),
                path=self.nodule_csv,
            )
            for row_number, row in enumerate(reader, start=2):
                # Rows without an annotation_index describe scans without an
                # annotated >=3 mm nodule and are not physical-nodule records.
                if not str(row["annotation_index"]).strip():
                    continue
                patient_id = str(row["patient_id"]).strip()
                if not patient_id:
                    raise ValueError(f"{self.nodule_csv}:{row_number}: empty patient_id")
                annotation_id = (
                    patient_id,
                    _integer(row["scan_index"], field="scan_index"),
                    _integer(row["annotation_index"], field="annotation_index"),
                )
                key = PhysicalNoduleKey(
                    patient_id=patient_id,
                    scan_index=annotation_id[1],
                    nodule_id=_integer(row["nodule_id"], field="nodule_id"),
                )
                rating = _finite_float(row["malignancy"], field="malignancy")
                if annotation_id in annotation_to_key:
                    raise ValueError(
                        f"{self.nodule_csv}:{row_number}: duplicate annotation identity {annotation_id}"
                    )
                annotation_to_key[annotation_id] = key
                annotation_rating[annotation_id] = rating
                ratings[key].append(rating)
                first_annotation[key] = min(
                    annotation_id[2], first_annotation.get(key, annotation_id[2])
                )

        means = {key: reader_mean_malignancy(values) for key, values in ratings.items()}
        source_by_series: dict[
            str, list[tuple[tuple[float, float, float], PhysicalNoduleKey, int]]
        ] = defaultdict(list)
        seen_annotations: set[tuple[str, int, int]] = set()
        series_to_scan: dict[str, tuple[str, int]] = {}
        source_required = (
            "patient_id",
            "scan_index",
            "annotation_index",
            "series_instance_uid",
            "malignancy",
            "centroid",
            "row_direction",
            "col_direction",
            "slice_zvals",
            "pixel_spacing",
            "origin_x_mm",
            "origin_y_mm",
            "origin_z_mm",
        )
        with self.source_csv.open(newline="") as source:
            reader = csv.DictReader(source)
            _require_fields(reader.fieldnames, source_required, path=self.source_csv)
            for row_number, row in enumerate(reader, start=2):
                if not str(row["annotation_index"]).strip():
                    continue
                patient_id = str(row["patient_id"]).strip()
                annotation_id = (
                    patient_id,
                    _integer(row["scan_index"], field="scan_index"),
                    _integer(row["annotation_index"], field="annotation_index"),
                )
                key = annotation_to_key.get(annotation_id)
                if key is None:
                    raise ValueError(
                        f"{self.source_csv}:{row_number}: annotation {annotation_id} "
                        f"is absent from {self.nodule_csv}"
                    )
                if annotation_id in seen_annotations:
                    raise ValueError(
                        f"{self.source_csv}:{row_number}: duplicate annotation identity {annotation_id}"
                    )
                source_rating = _finite_float(row["malignancy"], field="malignancy")
                if not math.isclose(
                    source_rating,
                    annotation_rating[annotation_id],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"Malignancy disagreement for annotation {annotation_id}: "
                        f"source={source_rating}, nodule map={annotation_rating[annotation_id]}"
                    )
                seen_annotations.add(annotation_id)
                uid = str(row["series_instance_uid"]).strip()
                if not uid:
                    raise ValueError(f"{self.source_csv}:{row_number}: empty series_instance_uid")
                scan_identity = (patient_id, annotation_id[1])
                previous_scan = series_to_scan.setdefault(uid, scan_identity)
                if previous_scan != scan_identity:
                    raise ValueError(
                        f"Series {uid} maps to both {previous_scan} and {scan_identity}"
                    )
                source_by_series[uid].append(
                    (_source_world_coordinate(row), key, annotation_id[2])
                )

        missing_source = set(annotation_to_key) - seen_annotations
        if missing_source:
            examples = sorted(missing_source)[:5]
            raise ValueError(
                f"{self.source_csv} is missing {len(missing_source)} mapped reader annotations; "
                f"examples={examples}"
            )
        self.means = means
        self.first_annotation = first_annotation
        self.source_by_series = dict(source_by_series)

    def mean_for(self, key: PhysicalNoduleKey) -> float:
        try:
            return self.means[key]
        except KeyError as error:
            raise ValueError(f"Unknown physical nodule key: {key}") from error

    def resolve_luna_row(
        self,
        row: Mapping[str, object],
        *,
        row_index: int,
        max_distance_mm: float = LUNA_MATCH_MAX_DISTANCE_MM,
    ) -> ResolvedNodule:
        uid = str(row.get("seriesuid", "")).strip()
        if not uid:
            raise ValueError(f"LUNA row {row_index}: missing seriesuid")
        candidates = self.source_by_series.get(uid)
        if not candidates:
            raise ValueError(f"LUNA row {row_index}: unknown seriesuid {uid}")
        target = tuple(
            _finite_float(row.get(f"coord{axis}"), field=f"coord{axis}")
            for axis in "XYZ"
        )
        distances = [math.dist(target, candidate[0]) for candidate in candidates]
        closest_index = min(range(len(distances)), key=distances.__getitem__)
        distance = distances[closest_index]
        _, key, annotation_index = candidates[closest_index]
        competing_keys = {
            candidate[1]
            for candidate, candidate_distance in zip(candidates, distances)
            if candidate[1] != key and math.isclose(
                candidate_distance, distance, rel_tol=0.0, abs_tol=1e-6
            )
        }
        if competing_keys:
            raise ValueError(
                f"LUNA row {row_index}: ambiguous nearest physical nodule at {distance:.6g} mm"
            )
        if distance > max_distance_mm:
            raise ValueError(
                f"LUNA row {row_index}: nearest authoritative annotation is "
                f"{distance:.3f} mm away (limit={max_distance_mm:.3f} mm)"
            )
        mean = self.mean_for(key)
        if is_indeterminate_malignancy(mean):
            raise ValueError(
                f"LUNA row {row_index}: {key} has physical reader mean malignancy==3"
            )
        return ResolvedNodule(
            row_index=row_index,
            key=key,
            reader_mean_malignancy=mean,
            matched_annotation_index=annotation_index,
            match_distance_mm=distance,
        )


def validate_luna_cohort(
    rows: Sequence[Mapping[str, object]],
    index: AuthoritativeLidcIndex,
    *,
    max_distance_mm: float = LUNA_MATCH_MAX_DISTANCE_MM,
) -> list[ResolvedNodule]:
    if not rows:
        raise ValueError("LUNA cohort is empty")
    required = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "malignancy"}
    resolved: list[ResolvedNodule] = []
    seen: dict[PhysicalNoduleKey, int] = {}
    for row_index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"LUNA row {row_index}: missing fields {missing}")
        diameter = _finite_float(row["diameter_mm"], field="diameter_mm")
        if diameter <= 0:
            raise ValueError(f"LUNA row {row_index}: diameter_mm must be positive")
        item = index.resolve_luna_row(
            row, row_index=row_index, max_distance_mm=max_distance_mm
        )
        label = _integer(row["malignancy"], field="malignancy")
        if label not in (0, 1):
            raise ValueError(
                f"LUNA row {row_index}: derived malignancy must be binary, got {label}"
            )
        expected_label = int(item.reader_mean_malignancy > 3.0)
        if label != expected_label:
            raise ValueError(
                f"LUNA row {row_index}: binary malignancy={label} disagrees with "
                f"authoritative reader mean={item.reader_mean_malignancy:g}"
            )
        if item.key in seen:
            raise ValueError(
                f"LUNA rows {seen[item.key]} and {row_index} duplicate physical nodule {item.key}"
            )
        seen[item.key] = row_index
        resolved.append(item)
    return resolved


def validate_keyed_cohort(
    rows: Sequence[Mapping[str, object]],
    index: AuthoritativeLidcIndex,
) -> list[ResolvedNodule]:
    """Validate rows carrying explicit patient/scan/nodule identifiers."""
    if not rows:
        raise ValueError("Physical-nodule cohort is empty")
    resolved: list[ResolvedNodule] = []
    seen: dict[PhysicalNoduleKey, int] = {}
    for row_index, row in enumerate(rows):
        missing = sorted({"patient_id", "scan_index", "nodule_id"} - set(row))
        if missing:
            raise ValueError(f"Cohort row {row_index}: missing fields {missing}")
        key = PhysicalNoduleKey(
            patient_id=str(row["patient_id"]).strip(),
            scan_index=_integer(row["scan_index"], field="scan_index"),
            nodule_id=_integer(row["nodule_id"], field="nodule_id"),
        )
        if not key.patient_id:
            raise ValueError(f"Cohort row {row_index}: empty patient_id")
        mean = index.mean_for(key)
        if is_indeterminate_malignancy(mean):
            raise ValueError(
                f"Cohort row {row_index}: {key} has physical reader mean malignancy==3"
            )
        if key in seen:
            raise ValueError(
                f"Cohort rows {seen[key]} and {row_index} duplicate physical nodule {key}"
            )
        seen[key] = row_index
        if str(row.get("physical_reader_mean_malignancy", "")).strip():
            supplied_mean = _finite_float(
                row["physical_reader_mean_malignancy"],
                field="physical_reader_mean_malignancy",
            )
            if not math.isclose(supplied_mean, mean, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"Cohort row {row_index}: supplied reader mean={supplied_mean:g} "
                    f"disagrees with authoritative mean={mean:g}"
                )
        if str(row.get("malignancy", "")).strip():
            label = _integer(row["malignancy"], field="malignancy")
            if label not in (0, 1) or label != int(mean > 3.0):
                raise ValueError(
                    f"Cohort row {row_index}: binary malignancy={label} disagrees "
                    f"with authoritative reader mean={mean:g}"
                )
        supplied_policy = str(row.get("malignancy_policy", "")).strip()
        if supplied_policy and supplied_policy != POLICY_NAME:
            raise ValueError(
                f"Cohort row {row_index}: malignancy_policy={supplied_policy!r}, "
                f"expected {POLICY_NAME!r}"
            )
        resolved.append(
            ResolvedNodule(
                row_index=row_index,
                key=key,
                reader_mean_malignancy=mean,
            )
        )
    return resolved


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_provenance(path: os.PathLike[str] | str) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def make_policy_provenance(
    *,
    cohort_kind: str,
    cohort_files: Sequence[os.PathLike[str] | str],
    source_csv: os.PathLike[str] | str,
    nodule_csv: os.PathLike[str] | str,
    resolved: Sequence[ResolvedNodule],
    extra_files: Sequence[os.PathLike[str] | str] = (),
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not resolved:
        raise ValueError("Cannot record provenance for an empty validated cohort")
    if any(is_indeterminate_malignancy(item.reader_mean_malignancy) for item in resolved):
        raise ValueError("Validated cohort unexpectedly contains reader mean malignancy==3")
    return {
        "schema": PROVENANCE_SCHEMA,
        "policy_name": POLICY_NAME,
        "policy_definition": {
            "unit": "physical_nodule",
            "aggregation": "arithmetic_mean_of_all_available_reader_malignancy_ratings",
            "excluded_when": "reader_mean_malignancy == 3 (absolute tolerance 1e-6)",
            "central_module": "back_prop.lidc_policy",
            "central_policy_file": file_provenance(_POLICY_PATH),
        },
        "validation": {
            "status": "passed",
            "cohort_kind": cohort_kind,
            "physical_nodule_count": len(resolved),
            "indeterminate_physical_nodule_count": 0,
            "unique_physical_nodule_count": len({item.key for item in resolved}),
        },
        "authoritative_inputs": {
            "reader_annotations_csv": file_provenance(source_csv),
            "annotation_to_physical_nodule_csv": file_provenance(nodule_csv),
        },
        "cohort_files": [file_provenance(path) for path in cohort_files],
        "extra_files": [file_provenance(path) for path in extra_files],
        "details": dict(details or {}),
    }


def load_policy_provenance(path: os.PathLike[str] | str) -> dict[str, object]:
    provenance_path = Path(path)
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Missing required malignancy-policy provenance sidecar: {provenance_path}"
        )
    with provenance_path.open() as source:
        payload = json.load(source)
    if payload.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(f"Unsupported policy provenance schema in {provenance_path}")
    if payload.get("policy_name") != POLICY_NAME:
        raise ValueError(f"Wrong malignancy policy in {provenance_path}")
    validation = payload.get("validation", {})
    if validation.get("status") != "passed":
        raise ValueError(f"Policy validation did not pass in {provenance_path}")
    if validation.get("indeterminate_physical_nodule_count") != 0:
        raise ValueError(f"Indeterminate nodules recorded in {provenance_path}")
    central = payload.get("policy_definition", {}).get("central_policy_file", {})
    if central.get("sha256") != sha256_file(_POLICY_PATH):
        raise ValueError(
            f"Central policy code changed since {provenance_path} was produced; "
            "start a new output directory and revalidate"
        )
    return payload


def write_policy_provenance(
    path: os.PathLike[str] | str,
    payload: Mapping[str, object],
) -> None:
    """Create an immutable JSON sidecar; an existing mismatch is an error."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = load_policy_provenance(output)
        if existing != dict(payload):
            raise FileExistsError(
                f"Existing provenance differs from the current inputs: {output}. "
                "Use a new output directory."
            )
        return
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = (
    "AuthoritativeLidcIndex",
    "LUNA_MATCH_MAX_DISTANCE_MM",
    "PROVENANCE_SCHEMA",
    "PhysicalNoduleKey",
    "ResolvedNodule",
    "file_provenance",
    "load_policy_provenance",
    "make_policy_provenance",
    "sha256_file",
    "validate_keyed_cohort",
    "validate_luna_cohort",
    "write_policy_provenance",
)
