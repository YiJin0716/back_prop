# back_prop Directory Guide

| Folder | Purpose |
|---|---|
| `model_v4/` | V4 model, warmup, losses, training, monitoring, and tests. |
| `model_v4_oversample/` | V4 training with extra sampling of training CTs containing missed nodules. |
| `common/` | Shared data loaders, model components, training utilities, and legacy checkpoint support used by V4. |
| `evaluate/` | Model evaluation, comparison with official annotations, per-reader rating analysis, and roivue segmentation visualization. |
| `baseline_compare/` | Comparisons with baselines such as Sybil, DeepLung, and EDICNet, including cross-validation and results. |

Model overview: `V4_model.pdf`. LaTeX source: `V4_model.tex`. Architecture diagram: `V4_structure.png`.

Clone into a directory named `back_prop` and run Python modules from its parent
directory (for example, `python -m back_prop.model_v4.train --help`). Initialize
baseline dependencies with `git submodule update --init --recursive` when needed.
CT data, pretrained weights, generated results, and Python environments are external.
The current paths and Slurm scripts target the original `imaging_feature` workspace;
adapt them and provide its sibling dependencies, including VISTA3D, on another machine.
