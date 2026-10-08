"""Checkpoint loading and annotation-free joint-model inference."""
from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path

import numpy as np

from .annotations import OfficialAnnotations, case_key, load_image, load_test_cases
from .types import CaseEvaluation, MaskCrop, PredictedNodule, SEMANTIC_NAMES

ARCHITECTURES = {
    "whole_ct_radiomics_offset_ws_groupnorm_early_diagnostics_v4":
        ("back_prop.model_v4.model", "WholeCTJointModelV4"),
    "whole_ct_radiomics_offset_groupnorm_early_diagnostics_v4":
        ("back_prop.model_v4.model", "WholeCTJointModelV4"),
    "whole_ct_parallel_soft_radiomics_medicalnet_continuous_rashomon_v3":
        ("back_prop.common.semantic_model", "WholeCTJointModelV3"),
    "whole_ct_parallel_soft_radiomics_medicalnet_continuous_rashomon_v3_anneal":
        ("back_prop.common.annealed_model", "WholeCTJointModelV3"),
    "whole_ct_parallel_soft_radiomics_medicalnet_continuous_rashomon_v3_hard":
        ("back_prop.model_v3_hard.model", "WholeCTJointModelV3Hard"),
    "whole_ct_coarse_to_fine_radiomics_v2_m3_excluded":
        ("back_prop.common.coarse_model", "WholeCTJointModelV2"),
}


def _uninitialized_segmenter():
    import sys
    from back_prop.common.base_model import DEFAULT_VISTA_SOURCE, VistaNoduleSegmenter
    if str(DEFAULT_VISTA_SOURCE) not in sys.path:
        sys.path.insert(0, str(DEFAULT_VISTA_SOURCE))
    from vista3d.build_vista3d import build_vista3d_segresnet_decoder
    return VistaNoduleSegmenter(build_vista3d_segresnet_decoder(in_channels=1))


def _uninitialized_encoder():
    from torch import nn
    from monai.networks.nets import resnet50
    from back_prop.common.base_model import MedicalNetEncoder

    class Encoder(MedicalNetEncoder):
        def __init__(self):
            nn.Module.__init__(self)
            self.network = resnet50(spatial_dims=3, n_input_channels=1,
                                    feed_forward=False, bias_downsample=False)

    return Encoder()


def load_joint_model(checkpoint, *, device="cuda"):
    """Restore a trusted training checkpoint, strictly, without refitting.

    Supports V2, V3, V3_anneal, V3_hard and V4. Full weights are restored without
    needing the original VISTA/MedicalNet initialization checkpoint files.
    The returned model is in eval mode and carries checkpoint provenance.
    """
    import torch

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; allocate a GPU or select device='cpu'")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = saved.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported joint-model architecture: {architecture!r}")
    config = saved["config"]
    names = ("window_size", "overlap", "screen_shape", "roi_shape", "num_queries",
             "detr_hidden_dim", "detr_coarse_shape", "detr_nheads", "detr_encoder_layers",
             "detr_decoder_layers", "vista_feature_dim", "context_channels", "hard_negatives",
             "ignore_overlap_threshold", "teacher_full_epochs", "teacher_zero_epoch",
             "teacher_jitter", "minimum_roi_coverage", "inference_object_threshold")
    kwargs = {key: config[key] for key in names if key in config}
    kwargs.update(refiner_base_channels=config.get("refiner_base", 8),
                  segmenter_factory=_uninitialized_segmenter, use_checkpoint=False)
    is_v4 = architecture in {
        "whole_ct_radiomics_offset_groupnorm_early_diagnostics_v4",
        "whole_ct_radiomics_offset_ws_groupnorm_early_diagnostics_v4",
    }
    has_bank = "rashomon" in architecture or is_v4
    if has_bank:
        # The bank's own serialized schema is the authority, including modes
        # such as 'best', and any non-default fitting parameters.
        state = saved["model"]["rashomon._extra_state"]
        bank = dict(state["config"])
        bank.update({k: state[k] for k in ("fraction", "min_samples", "refresh_every", "mode")})
        kwargs.update(encoder_factory=_uninitialized_encoder,
                      medicalnet_roi_size=config.get("medicalnet_roi_size", 64),
                      amp_dtype=getattr(torch, config.get("amp_dtype", "bfloat16")),
                      bank_config=bank)
        if is_v4:
            kwargs["baseline_l2"] = state["baseline_l2"]
            # Old checkpoints must retain their original predictions. Weight
            # standardization requires a new warmup, not an inference patch.
            kwargs["semantic_weight_standardization"] = (
                architecture == "whole_ct_radiomics_offset_ws_groupnorm_early_diagnostics_v4")
    elif "ordinal_init" in config:
        kwargs["ordinal_init"] = config["ordinal_init"]
    if architecture.endswith("_hard"):
        kwargs["mask_temperature"] = config["mask_temperature"]
    module, name = ARCHITECTURES[architecture]
    model = getattr(importlib.import_module(module), name)(**kwargs)
    model.load_state_dict(saved["model"], strict=True)
    if has_bank and not int(model.rashomon.count):
        raise ValueError("Checkpoint has no fitted Rashomon bank; malignancy inference is unavailable")
    if is_v4 and not int(model.rashomon.baseline_count):
        raise ValueError("Checkpoint has no fitted V4 radiomics baseline")
    model.evaluation_metadata = {
        "checkpoint": str(Path(checkpoint).resolve()), "architecture": architecture,
        "epoch": int(saved["epoch"]) + 1,
        "training_manifest": str(config["manifest"]) if config.get("manifest") is not None else None,
        "amp_dtype": config.get("amp_dtype", "bfloat16" if has_bank else "float16"),
    }
    return model.to(device).eval()


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def evaluate_case(model, case, *, annotations=None, object_threshold=None,
                  mask_threshold=0.5, minimum_iou=0.1, amp=True):
    """Infer one whole CT, then attach official GT for all four comparisons.

    ``case`` is a manifest row. Prefer JointModelEvaluator for enforced split
    membership. No GT, target box, target mask or training batch is passed to
    forward. Results and crops are CPU arrays and can be reused by every plot.
    """
    import torch

    if not 0 < mask_threshold < 1 or not 0 <= minimum_iou <= 1:
        raise ValueError("Require 0 < mask_threshold < 1 and 0 <= minimum_iou <= 1")
    if object_threshold is not None and not 0 <= object_threshold <= 1:
        raise ValueError("object_threshold must be in [0, 1]")
    key = case_key(case)
    bank = getattr(model, "rashomon", None)
    if bank is not None:
        train_patients = {k.rsplit("__scan", 1)[0] for k in bank.allowed_cases}
        if case["patient_id"] in train_patients:
            raise ValueError(f"Evaluation patient was used to fit the model bank: {key}")
    hu, affine, source, zoom = load_image(case)
    image = torch.from_numpy(((np.clip(hu, -1024, 1024)+1024)/2048).astype(np.float32))[None]
    kwargs = dict(batch=None, teacher_probability=0.0, object_threshold=object_threshold)
    if bank is not None:
        kwargs.update(image_hu=torch.from_numpy(hu)[None], update_bank=False)
    model.eval()
    device = next(model.parameters()).device
    dtype = getattr(model, "amp_dtype", torch.bfloat16 if bank is not None else torch.float16)
    context = torch.autocast(device.type, dtype=dtype) if amp and device.type == "cuda" else nullcontext()
    before = None if bank is None else (
        {k: v.detach().cpu().clone() for k, v in bank.named_buffers()}, bank.updates, len(bank.memory))
    with torch.inference_mode(), context:
        output = model(image, **kwargs)
    if output.teacher_forced_count or output.fallback_count or (output.refined_target_indices >= 0).any():
        raise RuntimeError("Inference unexpectedly used GT routing")
    if before is not None and (bank.updates != before[1] or len(bank.memory) != before[2] or
            any(not torch.equal(v.detach().cpu(), before[0][k]) for k, v in bank.named_buffers())):
        raise RuntimeError("Rashomon bank changed during evaluation")
    features = _numpy(output.semantic_features)
    names = SEMANTIC_NAMES if features.shape[1] == 6 else (
        "lobulation", "malignancy", "margin", "sphericity", "spiculation", "subtlety", "texture")
    if features.shape[1] != len(names):
        raise ValueError(f"Unknown semantic output width: {features.shape}")
    probabilities = _numpy(output.semantic_probabilities)
    predictions = []
    for i, query in enumerate(output.refined_query_indices.tolist()):
        logits = output.fine_mask_logits[i].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite predicted segmentation logits")
        valid = output.roi_valid[i].detach().cpu().numpy().astype(bool)
        probability = _numpy(logits.sigmoid()) * valid
        origin = tuple(output.crop_origins[i].tolist())
        predictions.append(PredictedNodule(
            int(query), MaskCrop((probability >= mask_threshold) & valid, origin),
            MaskCrop(probability, origin), float(output.object_logits[query].float().sigmoid()),
            dict(zip(names, map(float, features[i]))),
            dict(zip(names, probabilities[i].tolist())),
            float(output.nodule_logits[i].float().sigmoid()),
            _numpy(output.ensemble_logits[i].sigmoid()).tolist() if bank is not None else []))
    numeric = [float(output.risk_logit.float().sigmoid().item())]
    for n in predictions:
        numeric += [n.object_probability, n.malignancy_probability, *n.semantic_features.values()]
        numeric += [v for values in n.semantic_probabilities.values() for v in values]
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite model diagnostic output")
    screening = output.discovered_mask.detach().cpu().numpy().astype(bool)
    if screening.shape != hu.shape:
        raise ValueError("Screening segmentation and CT geometry differ")
    # GT is loaded only after the annotation-free forward has completed.
    annotations = annotations or OfficialAnnotations(cases=[case])
    truth = annotations.load_nodules(key, source, zoom)
    temperature = getattr(model, "mask_temperature", getattr(model.radiomics, "mask_temperature", 1.0))
    metadata = dict(getattr(model, "evaluation_metadata", {}))
    metadata.update(
        image=str(case["image"]), annotation_csv=str(annotations.annotations_csv),
        identity_csv=str(annotations.identity_csv), official_mask_dir=str(annotations.mask_dir),
        consensus_min_votes=annotations.min_votes, ground_truth_used_for_prediction=False,
        bank_unchanged=True if bank is not None else None, mask_threshold=mask_threshold,
        diagnostic_mask_temperature=float(temperature),
        segmentation_probability="sigmoid(raw fine_mask_logits); padded voxels excluded",
        object_threshold=float(model.inference_object_threshold if object_threshold is None else object_threshold),
        window_count=int(output.window_count), axis_order="canonical XYZ",
        gt_includes_indeterminate=True, endpoint="LIDC reader-derived malignancy, not pathology",
    )
    return CaseEvaluation(key, hu, affine, screening, predictions, truth, numeric[0], metadata, minimum_iou)


class JointModelEvaluator:
    """Load once, select only cases from the requested split, infer once per case."""

    def __init__(self, model, manifest, *, split="testing", annotations=None):
        self.model = model
        self.manifest = Path(manifest)
        self.split = split
        self.cases = {case_key(c): c for c in load_test_cases(manifest, split)}
        self.annotations = annotations or OfficialAnnotations(cases=list(self.cases.values()))

    @classmethod
    def from_checkpoint(cls, checkpoint, manifest=None, *, device="cuda", split="testing",
                        annotations_csv=None, identity_csv=None, mask_dir=None, min_votes=2):
        model = load_joint_model(checkpoint, device=device)
        manifest = manifest or model.evaluation_metadata.get("training_manifest")
        if manifest is None:
            raise ValueError("Provide a test manifest; checkpoint has no manifest path")
        cases = load_test_cases(manifest, split)
        paths = {k: v for k, v in dict(annotations_csv=annotations_csv,
                 identity_csv=identity_csv, mask_dir=mask_dir).items() if v is not None}
        annotations = OfficialAnnotations(**paths, cases=cases, min_votes=min_votes)
        return cls(model, manifest, split=split, annotations=annotations)

    def evaluate(self, case_id, **kwargs):
        if case_id not in self.cases:
            raise ValueError(f"{case_id!r} is not in the {self.split!r} split")
        result = evaluate_case(self.model, self.cases[case_id], annotations=self.annotations, **kwargs)
        result.metadata.update(manifest=str(self.manifest.resolve()), split=self.split)
        return result

    def evaluate_test_set(self, case_ids=None, **kwargs):
        """Yield cases sequentially, so an entire test set never resides in RAM."""
        keys = list(self.cases) if case_ids is None else list(case_ids)
        unknown = set(keys) - self.cases.keys()
        if unknown:
            raise ValueError(f"Cases outside {self.split!r}: {sorted(unknown)}")
        for key in keys:
            yield self.evaluate(key, **kwargs)
