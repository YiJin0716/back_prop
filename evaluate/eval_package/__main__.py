"""python -m back_prop.evaluate.eval_package --checkpoint ... --case-id ... --output ..."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Joint-model prediction vs official LIDC GT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Default: manifest saved in checkpoint")
    parser.add_argument("--split", default="testing")
    parser.add_argument("--case-id", action="append", help="Repeatable; default: every case in split")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--identity-csv", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object-threshold", type=float)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-iou", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--nifti", action="store_true", help="Also export whole-CT prediction and GT NIfTI")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable outer autocast; V3 screening retains its internal checkpoint dtype")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (args.max_cases is not None and args.max_cases < 1) or args.threads < 1:
        parser.error("max-cases and threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
    from . import JointModelEvaluator, save_result
    from .comparisons import malignancy, nodule_segmentation, screening_segmentation
    evaluator = JointModelEvaluator.from_checkpoint(
        args.checkpoint, args.manifest, device=args.device, split=args.split,
        annotations_csv=args.annotations_csv, identity_csv=args.identity_csv,
        mask_dir=args.mask_dir, min_votes=args.min_votes)
    keys = list(evaluator.cases) if args.case_id is None else args.case_id
    if args.max_cases is not None:
        keys = keys[:args.max_cases]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for result in evaluator.evaluate_test_set(keys, object_threshold=args.object_threshold,
            mask_threshold=args.mask_threshold, minimum_iou=args.minimum_iou, amp=not args.no_amp):
        folder = save_result(result, args.output / result.case_id, plots=not args.no_plots, nifti=args.nifti)
        rows = nodule_segmentation(result)
        summary = dict(case_id=result.case_id, output=str(folder),
            segmentation=screening_segmentation(result)["metrics"], malignancy=malignancy(result)["scan"],
            nodule_counts={name: sum(row["status"] == name for row in rows)
                           for name in ("matched", "missed_gt", "unmatched_prediction")})
        summaries.append(summary)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
