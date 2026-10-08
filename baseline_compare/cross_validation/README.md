# V3 / Sybil / DeepLung: five-fold patient-level cross-validation

Frozen plan: [plan.json](plan.json). Submitted job graph: [jobs.json](jobs.json).
Run root: `/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/cross_validation_20260921`.
Outputs: `../plots/cross_validation_20260921/` (per-fold and pooled out-of-fold ROC PNG/PDF,
prediction CSV, AUC, paired AUC differences and patient-bootstrap intervals).
Fold 0 reuses the already completed V3, Sybil, DeepLung models and held-out predictions.
Only folds 1–4 are newly trained. The mistakenly submitted duplicate fold-0 jobs
were stopped; their IDs remain in `superseded_fold0_jobs.json` for history.
The aggregate job now depends on the four new evaluations. Fold-0 reuse is
verified in `fold0_reuse_audit.json`. Check Slurm and epoch logs for new training.

| Fold | Training CTs | Test CTs |
|---|---:|---:|
| 0 | 638 | 166 |
| 1 | 638 | 166 |
| 2 | 639 | 165 |
| 3 | 647 | 157 |
| 4 | 654 | 150 |

All 804 eligible CTs / 796 patients occur in exactly one test fold. The other
four folds form training. Indeterminate physical nodules (reader mean = 3) are
excluded from supervision. Scan-risk labels follow the actual mask-aware V3
loader; ambiguous scan labels are excluded identically for all models at test.
Existing validation subsets overlap training and are not used for joint-model or
baseline selection. No test calibration, threshold optimization or early stopping.

## Fixed training protocol

- **V3:** fresh joint model and Rashomon bank in each newly trained fold, initialized with that
  fold's segmentation-only VISTA3D checkpoint and the common MedicalNet checkpoint.
  Initial archived VISTA run and continuation configurations were checked against
  the corresponding fold manifest. The VISTA segmentation cohort includes cases
  without retained malignancy labels; it remains within the same training patients.
  Four A5000 GPUs with DDP; BF16; 18 epochs; 128³ windows. Original curriculum,
  teacher schedule, optimizer and loss weights are retained. No fold-0 joint-model
  checkpoint is transferred to another fold. All five folds
  use the same fixed training protocol; fold 0 reuses its completed run.
- **Sybil:** common public Sybil_1 checkpoint, same fine-tuning policy, three epochs.
- **DeepLung detector:** **each fold fine-tunes the same public `fd0066.ckpt` for
  five epochs**, SGD learning rate .001. It is not trained from scratch.
- **DeepLung malignancy classifier:** no public classifier weights are available in
  the existing release. Retain the prior DPN92 workflow: 700 epochs on training
  nodules, then 105 epochs adapting to that fold's fine-tuned detector proposals.
  GBMs and normalization statistics are fitted using training data only.

Training CT arrays are deterministic and can be shared across folds. Each training
cache directory exposes only its own training cases and has fold-specific metadata
and cohort hashes. Learned statistics, checkpoints, optimizer states, proposal
sets, and Rashomon banks are separate. Original fold-0 artifacts remain intact.

## Automatic job graph

`cache → Sybil / detector / initial classifier`; `detector → training proposals`;
`initial classifier + proposals → final classifier/GBM`.
V3 can train independently of the baseline cache.
`V3 + Sybil + final DeepLung → fold audit and test evaluation`;
`existing fold-0 predictions + four new fold evaluations → pooled predictions, ROC and AUC summary`.

Failed prerequisites prevent downstream evaluation. `submit.py` persists each job
ID immediately and skips already recorded jobs; it does not automatically retry
failed jobs. Training stages reject overwriting existing epoch logs. Any resumed
run requires an explicit checkpoint and corresponding dependency repair.

Evaluation independently checks final epochs, sample coverage, cohort hashes,
finite weights/optimizer state, optimizer or batch-normalization update counters,
and GBM/proposal provenance. Prediction receives CT only. V3 uses the complete
frozen Rashomon bank; no test case updates the bank. DeepLung scan risk is the
maximum nodule risk among up to 20 predicted candidates, or zero if none.

The summary reports each fold's AUC, mean and sample SD of fold AUCs, pooled
out-of-fold AUC, and paired V3-minus-baseline AUC differences. Patient-cluster
bootstrap resampling occurs within folds and uses the same samples for all
models. Its intervals are conditional on the trained models; they do not capture
training variability from retraining on new populations.

## Interpretation limits

Public DeepLung LUNA16 pretraining may include held-out LIDC patients. The user
explicitly chose public-weight fine-tuning; downstream train/test isolation does
not remove prior pretraining exposure. Fold 0 has already informed prior model
inspection, so this is not a wholly prospective untouched evaluation. These are
reader-derived malignancy labels, not confirmed future screening cancer incidence.

## Commands

From the repository root, with `back_prop/baseline_compare/.venv/bin/python`:

```bash
python -m back_prop.baseline_compare.cross_validation.build
python -m unittest back_prop.baseline_compare.cross_validation.test_protocol -v
python -m back_prop.baseline_compare.cross_validation.submit
python -m back_prop.baseline_compare.cross_validation.monitor
```

The separate late-curriculum V3 GPU smoke uses only four fold-1 training CTs and
writes no formal checkpoints. Its reduced bank-size requirement and teacher-forced training ROIs are confined to
the smoke job and does not alter any of the five full training runs.
