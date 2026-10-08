# back_prop Directory Guide

| Folder | Purpose |
|---|---|
| `model_v4/` | V4 model, warmup, losses, training, monitoring, and tests. |
| `model_v5/` | Ordinary 3D CNN semantics, valid-sample loss averaging, and earlier teacher withdrawal. |
| `model_v4_oversample/` | V4 training with extra sampling of training CTs containing missed nodules. |
| `common/` | Shared data loaders, model components, training utilities, and legacy checkpoint support used by V4. |
| `evaluate/` | Model evaluation, comparison with official annotations, per-reader rating analysis, and roivue segmentation visualization. |
| `baseline_compare/` | Comparisons with baselines such as Sybil, DeepLung, and EDICNet, including cross-validation and results. |

Model overview: `V4_model.pdf`. LaTeX source: `V4_model.tex`. Architecture diagram: `V4_structure.png`.
