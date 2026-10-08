# CURRENT CORRECTION — reuse existing fold 0

User clarified that one fold was already completed and only FOUR remain to train.
Reuse import PASSED: 166 CTs / 137 valid labels, all three AUCs unchanged.
Latest check: all four remaining V3 jobs12660730–12660733 are RUNNING.
Submission idempotence rechecked: no fold0 job recreated. Three protocol tests pass.
The previous choice to retrain all five was unnecessary. This supersedes historical
counts/status below. Original fold0 cohorts and hyperparameters match exactly;
config differences are only resume path, runtime, output directory and source audit.

- Canceled duplicate fold0 V3 12660729, detector12660754, classifier12660755,
  proposals12660756, adaptation12660757 and evaluation12660758. Duplicate Sybil
  12660753 had already completed; its new result is not used for the comparison.
- Original completed fold0 V3run12616464, Sybil12620985, DeepLung12622703/12623270,
  and original evaluation12632290 are reused, with weight hashes and case labels
  validated by reuse_fold0.py. See fold0_reuse_audit.json for successful import.
- Active jobs.json now excludes the seven duplicate fold0 stages. Full historical
  records are in superseded_fold0_jobs.json. Original plan archived separately.
- Aggregate12660783 dependency was changed BEFORE cancellation to only evaluations
  12660764,12660770,12660776,12660782. It combines those four folds with imported
  fold0 records/metrics. No other fold training was interrupted or restarted.
- submit.py and run_stage.py honor reused_folds=[0] and cannot accidentally
  resubmit fold0 training. Fold0 evaluation dispatch imports the old predictions.

--- Historical notes below (superseded by correction above) ---

# Five-fold CV continuation status — 2026-09-21

Goal active: submit all V3/Sybil/DeepLung folds and automatic test comparison, then
watch several rounds of real training progress. User explicitly changed DeepLung
to public-weight fine-tuning. Do not wait for all multi-day training to finish to
close the goal once the authorized monitoring milestone is actually satisfied.
Do not claim final test results exist before evaluation completion.

## Implemented and submitted

- Frozen plan/cross-fold audits: 804 CTs, 796 patients; train/test counts in README.
  All test patients occur once; both training and validation are patient-disjoint
  from that fold's test set. Matching VISTA archive/continuation configs audited.
- Generalized existing baseline scripts with optional `--cohort`, preserving
  fold-0 defaults, and propagated cohort hash through proposals and GBM fitting.
  Generalized existing final training audit counts; old Sybil audit re-passed.
- Shared deterministic cache reuses 638 original arrays and prepares 166 new CTs;
  each fold gets only its own training entries with separate provenance metadata.
- 37 formal jobs recorded in jobs.json. CPU cache **12660728**; V3 folds0–4
  **12660729–12660733**; downstream tasks **12660753–12660783**. Each DeepLung
  detector command explicitly loads public fd0066.ckpt (SHA147109...c2c) for5epochs.
  Initial classifier700epochs + proposal adaptation105epochs remains necessary
  because that classifier has no public checkpoint. No scratch detector was run.
- Fold evaluators depend on their V3, Sybil and final DeepLung jobs; aggregate
  **12660783** depends on all5 evaluations. Slurm afterok, kill-on-invalid-dep.
- Dedicated optional GPU smoke **12660787** uses fold1's first4trainingCTs,
  late curriculum index17, bank-min-samples2, no formal checkpoints. Check result.
- Three meaningful tests passed: known AUC and paired differences; exclusions,
  duplicate/cross-fold patient rejection; real frozen patient split verification.
  Targeted Python compilation passed. A first overly broad compilation included
  upstream's unexecuted Python2 reference code and reported its existing syntax;
  actual ported training/CV modules compile successfully.
- Regression of new three-model ROC/statistics against previous166-case fold0
  evaluation is running/should be checked. Outputs under validation/, not formal
  CV plots. No new test predictions or test-based tuning were performed.

## Latest observed live state (must recheck)

At ~00:51 EDT: cache64/166 newly needed CTs prepared; V3 fold0/fold1/fold2
running on4A5000 each, epoch1 step7/7/5, finite loss+gradients, optimizer updates
not skipped. Folds3/4 pending resource/priority. Baselines waiting for cache.
Monitor snapshots are persisted under monitoring/; latest_status.json summarizes
actual sacct and parsed step/epoch evidence. IDs alone are not health evidence.

## Remaining before authorized goal milestone

1. Check late-curriculum GPU smoke; investigate any errors (especially24GBVRAM).
2. Let cache finish and audit all5 actual retained-nodule counts/case coverage.
3. Verify live Sybil and public-weight DeepLung fine-tuning starts, multiple actual
   training progress records with finite gradients/losses; do not call queued jobs
   already trained. Watch/recheck several rounds, record exact progress and pending
   resource-dependent stages honestly. Fix failures and dependent jobs if needed.
4. Confirm complete dependency graph remains valid; save status and report IDs,
   output path, public-pretraining overlap caveat and what was actually observed.
5. Only then mark goal complete at the user's submission+monitoring milestone.

## Constraints and important interpretation

No changes to V3 model/train code or the18epoch curriculum. Fresh joint heads/bank
for each fold, own fold's VISTA pretraining. No cross-fold checkpoint warm starts.
Public DeepLung LUNA16 exposure may overlap test patients. Fold0 was previously
inspected; not wholly prospective evaluation. Current malignancy reader labels,
not prospective screening incidence. No subagents. Do not cancel unrelated jobs.

## Update: late-phase check and statistics PASS

- Replacement late-phase smoke12660794 COMPLETED0, four trainingCTs, teacher
  forcing1 confined to smoke. VISTA/DETR/refiner/MedicalNet gradients finite and
  nonzero; bank fitted3times, final19models, peak21.56GiB on A5000. No formal
  checkpoints saved. Earlier12660787 triggered bank-empty guard because a fresh
  random locator missed all nodules with teacher forcing0; not a formal failure.
- Three-model statistical regression reproduces prior fold0 AUCs exactly.
- All37 formal jobs and training/evaluation dependency edges audited.
- Last cache progress146/166; baseline startup still awaits successful cache.

## Update: cache metadata compatibility repaired

- CPUcache12660728 initially failed only at final metadata audit:637older
  entries lack optional skipped_empty_consensus_nodules. All166newCTarrays
  completed successfully. Fixed by deriving skippedIDs from source minuscache
  noduleIDs and validating any recorded skip list. Strict retainedcounts remain.
- Independently scanned all804metadata: only0909/scan746/nodule1 skipped, and
  that nodule is indeterminate/ignored. No retained targets are missing.
- Requeued SAMEcacheID12660728, refreshed all downstream afterokdependencies;
  no duplicate jobs. Reduced requeue request to2CPU16GB30min because remaining
  work is metadata audit/linking only (all804arrays already exist). Original
  submission command remains in registry; runtime override recorded there.

## Update: full cache passed; baseline runs started

- Cache12660728 successfully re-ran with804reused/0new, all5folds audited.
  Retainednodulecounts folds0–4:1599,1642,1609,1621,1617. No retained nodule lost.
- V3fold3 job12660732 started on4A5000 node17. Onlyfold4V3 still pending.
- Sybil/Deepdetector/initialclassifier folds0–2 started automatically aftercache.
  Detector configs independently checked: all use SAME public SHA147109...c2c,
  five epochs, and DIFFERENT corresponding foldcohort hashes.
- Keep monitoring actual baseline step/epoch logs for several rounds before
  ending the goal; other pending baseline folds remain resource-dependent.

## Authorized submission + early-monitoring milestone reached

- All37formaljobs remain COMPLETED/RUNNING/PENDING with valid dependencies.
- Multiple timestamped monitor rounds captured real progressing training, no
  nonfinite losses/gradients, skipped V3optimizer updates or unresolved errors.
- Latest: V3fold0–2 epoch1step33; fold3epoch1step6; fold4queued. Sybilfold0/1
  enteredepoch3; initialDeepclassifiersfold0–2 enteredepoch3. Deepdetectors
  folds0/1step201,fold2step101,fold3step51. All4runningdetectorspublichash
  matchesfd0066 and ownfoldcohort. These are INITIALhealth checks, NOT several
  completedV3epochs, trainingcompletion, or completedCVmetrics.
- `submission_monitoring_milestone.json` records exact scope/evidence. User
  authorized exit aftersubmission+severalmonitoringrounds; scheduled jobs keep
  training and automatically evaluate/aggregate later. No further edits needed
  unless an actual later job failure or a user change occurs.

## Fold 3 recovery — 2026-09-23

Original job `12660732` failed during epoch 14 when two annotation masks were temporarily invisible. Both files are present/readable now and had been successfully used in earlier epochs; transient NFS access remains the likely cause, not a proven storage diagnosis.

Added bounded retries around both mask header and voxel reads, without changing labels, vote thresholds or skipping missing annotations. Six fault-injection tests pass. Immutable epoch-13 checkpoint plus original config/metrics are preserved in `fold_3/v3/recovery_20260923/`.

Resume job `12687762` runs on 4 RTX 6000 Ada GPUs using the same environment, cohort and training hyperparameters, continuing epochs 14–18. Before training it verifies checkpoint/cohort hashes, reads all required annotation headers, and fully preprocesses both affected CTs. Logs are `logs/f3-v3_12687762.out` and `.err`. The original attempt is retained in `jobs.json` under `previous_attempts`.

The previously failed/cancelled evaluation jobs have not been resubmitted as part of this training-only recovery. Their path comparison and resume-provenance checks still need correction before evaluation can proceed.

Recovery startup verified: all 5304 required mask headers were readable, both previously failing CTs completed full preprocessing, 306 optimizer states and four RNG states restored. Epoch 14 step 1 completed with finite loss 7.5287 and gradient norm 14.5837; optimizer update was not skipped. Detailed evidence: `fold3_recovery_20260923.json`. Training remains RUNNING.
