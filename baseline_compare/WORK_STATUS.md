# Continuation notes — 2026-09-17

## LATEST STATE (supersedes the historical notes below)

### 2026-09-18: all baselines done; requested ROC job submitted

- All baseline components and both finalizers completed successfully. Ran
  `audit_training --require-all` again: all six stages PASS, including final
  DeepLung105epochs,92505additionalBNforwards,14087samples/epoch and fittedGBM.
  Final artifact manifests/weight hashes validated by the new evaluation runner.
- User explicitly requested submission of four-model test ROC after V3 finishes.
  Implemented evaluate_roc.py/sbatch. Formal job **12632290**, dependency
  afterok12616464, kill-on-invalid-dep, onePRO6000,64GB,24h limit. Outputs to
  baseline_compare/plots/test_roc.png + PDF and prediction/metric/provenance files.
- V3 final immutable epoch_018.pt required (saved epoch17 is zero-based).
  Baseline final artifacts required. Same test cases and mask-aware labels from
  V3 loader; ambiguous labels excluded identically before inference. Prediction
  receives only image tensors; no GT routing; no bank updates. Full-bank V3
  averaging, default single-ROI EDICNet, DeepLungmax20 fixed before test results.
- GPU smoke **12632286 COMPLETED**, 1m50s, all four predictions on trainingCT0001
  with V3epoch15 and final baseline weights. Bank unchanged. No test data used.
  Evidence: project baseline_compare/roc_smoke_v1/smoke_complete.json.
- Known-AUC fixture (20cases +1excluded), ties, common exclusions, cohort mismatch
  rejection, 2000 bootstrap replicates and PNG/PDF rendering passed. Initial4case
  fixture hit the deliberately strict90%valid-bootstrap guard; expanded fixture
  to20cases without changing implementation/guard. No unresolved test failure.
- Formal ROC remains pending V3 completion; submitting the job satisfies the
  user's current request to schedule it, but does not mean the figure exists.
  Preserve all V3/nonlinear model/userCPU jobs. No unrelated jobs were changed.

### 2026-09-17 17:27 EDT: EDICNet complete; DeepLung remains

- User asked when all three baselines finish. Sybil and EDICNet are finished.
- Slurm confirms Retina12620997 COMPLETED50epochs (1h43m32s), proposal12623264
  COMPLETED638cases (1h30m54s), EDIC finalizer12624666 COMPLETED (44s), all exit0.
- EDIC final_artifacts.json exists, artifact files exist, final training audits
  passed: Retina215100Adamsteps and HSCNN53400. Both final whole-CT inference
  variants saved; artifact links published in edicnet/.
- Deep classifier12622903 RUNNING, latest417/700epochs at15.29sec/epoch. Roughly
  1h12 remaining for CNN, then first GBM fit. Final classifier12623270 still
  pending afterok12622903; proposal dependency has completed.
- Read all638 proposal records using exact upcoming matching rule:12488 eligible
  detected samples,272ignored. Adding1599GT makes14087samples,881batches/epoch.
  Thus105adaptationepochs expected~3.5–4h at current GPU batch throughput,
  plus GBM/final checks and potential queue delay. Overall estimate~5–6h from
  17:27EDT, approximately22:30–23:30EDT, conditional on stable resources.
- Do not report only the remaining initial700-epoch stage as total completion.
- Goal active, no failure/blocker. Continue existing classifier and dependent
  final stage/finalizer12624667. No training restart was performed.

### Latest verified milestone: HSCNN finished

- **EDIC HSCNN12620991 COMPLETED**, exit0, elapsed38m12s. Independent verifier
  passed all200 epochs and1599 nodules each epoch, finite model/optimizer, exactly
  **53400 Adam updates and53400 BN training forwards**. Saved completion audit
  in its run directory; stdout in logs/audit_edicnet_classifier.log. Final weights
  SHA256 `e00014e00090249d04e34005d4e30ebd41e7ed0b3f85ad0682a5a5a380c48926`.
- Latest live snapshot: EDIC Retina12620997 completed26/50 epochs; Deep initial
  classifier12622903 completed26/700; proposal12623264 completed28/638 scans.
  All three confirmed RUNNING by squeue, no observed errors. Finalizers and
  final105-epoch classifier remain pending valid dependencies. No duplicate jobs.
- This continuation completed two independent audits (Deep detector and HSCNN),
  confirmed Deep classifier/proposal startup, and updated README. Goal incomplete:
  still requires remaining full training and final inference verification.
- Rough initial-classifier speed15 seconds/epoch gives~3 hours remaining before
  its GBM; subsequent105-epoch adaptation adds time. This is only an estimate.

### Update: detector completion and classifier startup

- DeepLung detector **12622703 COMPLETED**, exit 0, elapsed 33m18s. Independent
  `audit_training --only deeplung_detector --require-all` passed: all five epochs,
  all638 CTs each epoch,2284 samples per epoch,2855 BN training forwards, finite
  model/optimizer, detector head changed from public weights. Final checkpoint
  SHA256 `98434301a466f6b9e8dda59cf3710a439a8dd2116a62d8c0d3d6dbd8c342eaf9`.
- Proposal generation **12623264 RUNNING** on A6000 node06, automatically started
  after detector success. First two cases produced20 candidates each. Continue
  checking full638 completion; inference progress messages on stderr are normal.
- Initial DeepLung classifier **12622903 RUNNING** on PRO6000 node49. First eight
  epochs started with finite losses/gradients, roughly15-18 seconds per epoch
  including save. Full700 epochs and GBM still required before final105 stage.
- EDIC HSCNN last observed176/200; RetinaNet still running around25/50.
- Previous turn's public-weight question was answered separately; this resumed
  goal turn makes new progress via completed-detector audit and live startup
  verification. No blocker. Keep existing live jobs and dependencies.

### Latest continuation additions

- Previous turn made progress and verified live training waits. No blocker; do
  not mark goal blocked while these jobs are training or legitimately queued.
- `audit_training.py` now checks actual final epochs and sample coverage, hashes,
  finite model/optimizer state, Adam step counts or BN forward-count deltas, and
  final DeepLung GBM/proposal provenance. Sybil passes: **1914 optimizer steps**.
  This verifier was run successfully against real Sybil artifacts; remaining
  component-specific branches still await their actual completed checkpoints.
  If a counter check fails, investigate actual training/source evidence, rather
  than weakening it automatically or trusting a completion marker.
- `finalize.py`/`finalize.sbatch` publish artifact links only after audits and
  final-checkpoint CT inference. **EDICNet finalizer12624666** depends on
 12620991+12620997. **DeepLung finalizer12624667** depends on12622703+12623270.
  These run on A6000. Inspect their logs/results; do not call the goal complete
  just because finalization jobs have been scheduled.
- Sybil `final_artifacts.json` has been produced and verified using its existing
  final-checkpoint inference result (no repeat inference needed). It includes
  checksum, usable `finetuned.pt` link, training audit and inference evidence.
- EDICNet finalizer exercises default primary-ROI and explicit max-risk-over-20
  configurations. Both are prespecified, same trained weights, no test tuning.
- **12623264 proposal generation GRES changed to A6000 while pending**, retaining
  its dependency on12622703. This lets proposal generation run in parallel with
  the long PRO6000 classifier job, rather than waiting for that card too.
- `run_registry.json` now includes final verification job IDs. It remains a plan,
  not proof of training completion. README updated with the verification flow.
- Last live snapshot: Deep detector12622703 at epoch4/5 on A6000; EDIC HSCNN
 12620991 at116/200 on PRO6000; EDIC detector12620997 at20/50 on PRO6000. All
 three running with finite losses and no stderr errors. Deep initial classifier
 12622903 remains pending(PRO resources), final105-epoch classifier12623270 and
 proposal12623264 pending dependencies. Recheck actual state before acting.

### Earlier continuation state (superseded where noted above)

Previous goal turn made substantial progress; goal still ACTIVE. Recheck live
Slurm/files rather than trusting this timestamped note.

- **Sybil 12620985 COMPLETED 3/3 epochs**, each638 CTs. Saved checkpoint independently
  audited: epoch3, exact case keys/cohort hash, finite tensors, frozen encoder
  unchanged, attention-pool weights changed. `sybil/finetuned.pt` links final
  checkpoint. Run `training_audit.json` records evidence. Final whole-CT inference
  12623272 passed on A6000, `whole_ct_verification.json` written. Sybil now DONE
  for requested training and functional inference (no held-out metrics claimed).
- Shared cache initially12620984 failed on an **empty ignored consensus mask**
  for LIDC-IDRI-0909 scan746 nodule1. Actual V3 skips that empty mask and keeps
  scan/nodule2. Fixed prepare.py accordingly; resumed job **12623268 COMPLETED**,
  reusing637 existing records. All638 CTs and1599 retained nodules verified by
  `audit_cache.py`, saved `cache_audit.json`. There are442 nonempty ignored masks;
  effective valid scan-risk labels **513**, not CSV-only512. sybil/train.py now
  reads effective risk/mask from cache metadata. Exact original V3 comparison of
  the exception passed in logs/verify_cache_empty_consensus.log.
- EDIC native slice cache12620994 COMPLETED all638 CTs / **8603 slices**.
- Pending jobs' dependencies were updated IN PLACE to afterok12623268, retaining
  original job IDs. Do not restart these because old12620984 failed.
- **EDIC RetinaNet12620997 RUNNING**, last observed epoch11/50 on PRO6000 node46.
- **EDIC HSCNN12620991 RUNNING**, last observed epoch8/200 on PRO6000 node49. It
  spent~5min loading all1599 patches before logging steps; this was normal NFS
  I/O, not a hang. Epochs then~6sec plus saving large checkpoints.
- **DeepLung detector12622703 RUNNING**, epoch1/5 on A6000 node06. Its pending
  Slurm GRES was changed to gpu:a6000:1 because PRO6000 slots were full. Do not
  assume sbatch header reflects this runtime override. ~1.1sec/step,571steps/epoch
  during cold first epoch. An attempted same change to HSCNN was rejected because
  it had already started; HSCNN was left running unchanged on PRO6000.
- **DeepLung initial classifier12622903 PENDING(Resources)** for PRO6000,
  700epochs+depth1GBM. A6000 benchmark12623271 passed batch16, peak6.33GiB,
  .307sec/step (~6h compute for700epochs). Kept this long job on PRO6000 for
  throughput; it should become eligible when another PRO job finishes.
- **DeepLung proposal-generation12623264 PENDING(afterok12622703)**. It runs the
  fine-tuned detector over all638 training CTs and writes
  .../deeplung/proposals/12622703/proposals.json.
- **DeepLung final classifier12623270 PENDING(afterok12622903:12623264)**,
  105epochs of detected-proposal adaptation then depth2GBM, matching det2cls.py's
  default schedule(neptime=.3). This is the FINAL DeepLung classifier, not12622903.
  `train_classifier.py --proposals ... --init ...` implements the stage; all1599
  GT samples are retained to preserve all638 CTs if detector misses nodules.
  Nearest matches within max(16mm,d/2), ignored matches excluded, unmatched
  proposals negative. Candidate cap20 fixed before testing. A meaningful
  positive+negative GPU smoke12623269 passed (loss70.25, gradnorm1766 finite,
  update/inference). Earlier smoke12623267 used positive-only examples and had
  zero task gradient; superseded by12623269, not sufficient by itself.

### Whole-CT inference now implemented and verified for connectivity

- `predict.py` and `predict.sbatch`: --model sybil|deeplung|edicnet --image CT
  --checkpoint classifier.pt --output prediction.json; detection models also
  --detector-checkpoint; DeepLung requires --gbm. It reads image + models only,
  no annotation/cohort files as inputs. Uses shared geometry helpers only.
- DeepLung: full1mmCT tiles96³,64³cores/16context, global coordinate channels,
  decode anchors5/10/20 offset1.5, logit>-2, cubeNMS.1, cap20; DPN2560+17³pixels
  +diameter GBM; scan max risk,0 if none. Tile origins handle exact multiples of64.
- EDIC: every native slice, 608->640, coordinate restoration to1mm, preserves
  XY box extents and estimates depth using max XY diameter. HSCNN gets predicted
  box mask. Default highest-score single ROI (paper); --max-candidates enables
  explicit max-risk multi-ROI variant. No detection->risk0 documented.
- Whole-smoke12623237 passed Sybil+DeepLung (20 proposals through CNN+test-only
  GBM). Test GBM used36 nodules from10 trainingCTs, must NOT be finalartifact.
  Initial EDIC smoke hadzero proposals. Separate12623266 used partially trained
  RetinaNet12620997 and successfully ran one ROI through HSCNN; finite risk. Need
  repeat inference with FINAL trained detector+classifier+GBM before completion.
- EDIC detector mean/std tensors registered as nonpersistent buffers to support
  CPU/GPU device moves without changing existing checkpoint keys.
- `run_registry.json` lists planned paths, not completion evidence; model `runs/`
  symlinks point into project storage. README updated with important adaptations.

### Remaining work now

1. Monitor/repair actual training/proposal/GBM jobs to completion. No duplicates
   while a job is live. All model modules must finish their full configured
   epochs; final DeepLung includes105-epoch proposal stage. Do not mark goal done
   from schedules, smoke checks, or just Sybil completion.
2. Independently audit final EDIC and DeepLung checkpoints/epoch logs/sample
   coverage/cohort hashes, GBM fitting provenance. Verify final complete wholeCT
   inference using final artifacts, publish links inside model folders. Preserve
   official code/weight provenance and all material adaptations in final answer.
3. Need clear final inference paths/config registry and status documentation.
   Optional held-out evaluation is not yet run; no test-set tuning. User asked
   training, so do not invent performance claims from connectivity tests.
4. Preserve V3 live12616464 and unrelated user CPU12620043. No subagents.

---

## Historical checkpoint below (many items above are now resolved)

Goal remains active. The user requested actual training/fine-tuning of Sybil,
DeepLung and EDICNet, with the same training cohort as V3. Do not mark complete
on the basis of submissions or smoke runs. No subagents are authorized.

## Authoritative state to recheck

- User's live V3 job 12616464 must not be disturbed. User CPU job 12620043 is
  unrelated. Baseline environment is an overlay; do not pip install in pro6000.
- Full cache: CPU job 12620984, last observed running, 405/638 complete.
- EDIC slice cache: CPU job 12620994, last observed running, 479/638 complete.
- Sybil fine-tune 12620985 (3 epochs), EDIC HSCNN 12620991 (200), DeepLung
  detector 12622703 (5), DeepLung classifier 12622903 (700 + GBM) depend on
  successful full cache 12620984.
- EDIC RetinaNet 12620997 (50 epochs) depends on slice cache 12620994.
- Data/run root: /usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/baseline_compare
- Logs: this directory's logs/. Scripts contain exact output paths.
- Initial cache smoke 12620974 failed due to missing tqdm; fixed overlay deps,
  pip check passed. Replacement 12620981 passed. Early full cache 12620982 was
  canceled to correct EDIC margin coding; its 18 metadata records were migrated
  before 12620984 resumed. Do not restart live preparation because of these old
  failed/canceled jobs.

## Verified implementations

- Four official repositories cloned, pristine; sources.json has commits and
  hashes of 16 public checkpoints (5 Sybil, 10 DPN detectors, 1 ResNet34).
- Frozen cohort.json: exact V3 638 scans / 630 patients / 1599 valid physical
  nodules. 512 valid scan-level labels; the other 126 still supply nodule or
  Sybil attention supervision. Existing validation patients overlap training;
  validation is never used for selection. Test166 patients are disjoint.
- Real GPU one-step checks passed: Sybil12620983, HSCNN12620988,
  RetinaNet12620996, DPN detector12622312, DPN classifier12622704.
- Strict public state-dict load passed for Sybil and DPN26. RetinaNet ImageNet
  backbone loads after discarding only fc weights. Its official file uses legacy
  tar serialization, so trusted-source weights_only=False is required.
- verify_cache.py comparison with original V3 dataset passed 3 cases, including
  a case with ignored nodules and masked scan loss. Its displayed retained-count
  field was corrected after first passing run; rerun to update the log.
- DeepLung detection_data.py has a final additional neutral-anchor safeguard
  for centers inside large nodules; last formal jobs are still pending so it will
  be included at startup. No network/optimizer change since GPU smoke.

## Remaining work (do not narrow goal)

1. Confirm both preparations complete, inspect all 638 records, no silently
   missing cases, and monitor actual full-cohort training starts. Capture failures
   from authoritative Slurm state and logs, fix/resume as needed. Do not duplicate
   live jobs. `complete.json` indicates a finished configured module, but inspect
   epochs/checkpoints/sample coverage rather than relying on that marker alone.
2. Implement and verify annotation-free **whole-CT** inference for all three.
   Existing model smoke calls cover only network components, not complete scan
   systems. Need persistent inference CLI, CT preprocessing, candidate decode/
   merge/NMS for DeepLung/EDIC, nodule classifier/GBM calls, scan risk aggregation.
   Sybil first adapted logit is LIDC risk, never calibrated one-year incidence.
3. DeepLung classifier currently follows GT-nodule main_nodcls.py. Evaluate
   whether the official det2cls.py detected-proposal fine-tuning stage is needed
   for final system fidelity, and port if so. Do not substitute a shallow generic
   CNN or use GT boxes at inference. DPN92 + final GBM are the actual intended
   nodule classifier. Avoid original upstream test-set model-selection leakage.
4. Record important adaptations in final reporting: EDIC original clinical
   semantics replaced by documented LIDC proxies; EDIC manual mask replaced by
   training GT box/inference predicted box for an automatic system; DeepLung full
   CT adaptation uses no external lung mask. Public DeepLung LUNA pretraining may
   expose held-out LIDC patients; do not claim a leakage-free comparison without
   resolving this. Current request explicitly favors public pretrained weights.
5. Finish and verify all requested training, saved artifacts and functioning
   whole-CT risk output before goal completion. Test166 data can be prepared via
   prepare.py --split testing, but must never enter training/calibration/GBM fit.

## Useful implementation details

- Shared cache hu_1mm.npy is float32 canonical RAS xyz; metadata boxes are
  [xmin,ymin,zmin,xmax,ymax,zmax] in 1-mm array coordinates. No labels embedded
  in intensity inputs. Shared helper crop_cube pads safely.
- DeepLung detector expects ZYX DICOM-like orientation: canonical[::-1,::-1,:]
  .transpose(2,1,0). lps_nodule converts centers. Detector crop96, stride4,
  anchors5/10/20, grid offset1.5, predicted four offsets decode relative to anchor.
  Whole-scan sliding windows must supply coordinate_grid relative to whole CT.
- DPN classifier consumes 32³ canonical crops normalized with training-only
  pixel_mean/std saved in config. GBM features: DPN2560 + raw uint8 central
  patch[8:25]^3 /255 + diameter/training_max_diameter. gbm.pkl is written after
  final classifier epoch and includes the normalization stats.
- EDIC detection native slices use canonical raw[:, :, z] (rows=x, cols=y).
  Resize native to608, pad bottom/right to640, HU[-1000,500]→[0,1], repeat3.
  Detections are xy column/row boxes in resized coordinates; convert to native
  and physical/1mm coordinates before forming classifier proposals.
- EDIC HSCNN takes box_masked_patch(hu_1mm, center, box,52), normalized [0,1].
  Neural outputs: malignancy logits2 and three semantic heads3/2/2. Final scan
  aggregation and proposal handling must be explicitly documented adaptations.
- model loads are from trusted official sources or our own checkpoints. Never
  change the upstream repositories just to fix runners; ports are separate.
