# DeepLung and Sybil comparison with V5

`plan.json` freezes the comparison with V5 run **12877415**, epoch **18**.
`jobs.json` records the submitted Slurm jobs and dependencies.

- Same 638 training CTs and physical-nodule targets as the V5 checkpoint; all
  178 held-out CTs are evaluated. The old 166-CT baseline test subset is not used.
- Sybil: official NLST initialization; 3 fixed fine-tuning epochs, last encoder
  stage LR `1e-5`, attention/risk-head LR `1e-4`. The first risk output is adapted
  to current LIDC reader malignancy, not interpreted as future cancer incidence.
- DeepLung: official public DPN26 detector, 5 fine-tuning epochs at LR `0.001`.
  Its DPN92 classifier starts from the existing 700-epoch checkpoint trained
  only on these same training CTs. New detector proposals from all training CTs
  drive 105 epochs of classifier adaptation and a training-only GBM fit.
- Final epochs are fixed in advance. Test data and the overlapping legacy
  validation split are not used for fitting, model selection or threshold tuning.
- Initialization and pretraining provenance are recorded in `plan.json`.

Successful training triggers annotation-free whole-CT inference, followed by a
CPU job updating the three-panel [ROC figure](../../evaluate/v5_joint_analysis/results_12877415_epoch18/volume_lvl/volume_auc_accuracy/malignancy_roc.png).
All three models use identical reader-derived labels under the exclude-3, ≥3
and >3 rules. CTs without detections remain in the evaluation. Missing or failed
predictions block the report rather than silently removing cases.

Large training files and job logs are stored under
`/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare/v5_12877415_20261015/`.
The maintained entry point is `python -m back_prop.baseline_compare.v5_comparison`;
the `prepare`, `submit` and individual stage commands share the frozen plan.
