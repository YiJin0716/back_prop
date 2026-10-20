# Per-CT HTML reports

`V5_12903757/<case_id>.html`: one self-contained interactive report for each of
the 178 test CTs, using V5 job 12903757, epoch 18. The layout shows CT/mask views,
semantic scores as plain numbers, risk, and segmentation agreement.

`V5_12955332/` uses the V5 hard/temperature-annealed model, epoch 18 (temperature
0.1), on the same 178 CTs and with the same layout. Each model directory has its
own `_job/plan.json`, submission record, logs, and shard status.

`pipeline_report.py` and `pipeline_report_assets/` render reports with roivue.
`run.py` loads the model once per GPU shard, infers each CT, and calls the
renderer in the `visual` environment. Temporary inference caches are deleted
after each case. Finished HTML files are written atomically and can be resumed.

`run.sbatch` submits four single-GPU shards. `_job/plan.json` records the full
test cohort and checkpoint; `_job/` also contains Slurm logs and per-shard status.
Completion requires all 178 `<case_id>.html` files and four completed shards.

From the `imaging_feature` workspace:

```bash
sbatch back_prop/evaluate/every_CT/run.sbatch
```

An optional positional argument selects another `_job/plan.json`; override the
Slurm job name and log paths when submitting a different model.

Reports can be downloaded and opened directly in a desktop browser or viewed
with VS Code Live Preview. Each HTML embeds its CT, masks and scores.
