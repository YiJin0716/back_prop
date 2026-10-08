# Joint model 评估与 LIDC GT 对照

支持训练 checkpoint：**V4、V3、V3_anneal、V3_hard、V2**。V1 / frozen_joint 的输出接口不同，当前不会自动转换。

对一个 test screening 只做一次整 CT 推理，随后可以查看：

| 数据函数 | 图像函数 | 模型输出与官方 GT |
|---|---|---|
| `screening_segmentation` | `plot_screening_segmentation` | 整 CT 分割 mask、GT consensus mask、Dice / IoU |
| `nodule_segmentation` | `plot_nodule_segmentation` | 每个预测结节与对应 GT 结节的分割、匹配状态、Dice / IoU |
| `semantic_features` | `plot_semantic_features` | semantic 期望评分、五档概率；GT 每个读者评分、均值、直方图 |
| `malignancy` | `plot_malignancy` | 结节和扫描级恶性概率；GT 读者评分与衍生二分类标签 |

四个数据函数都返回含 `prediction` 和 `ground_truth` 的字典或字典列表；四个绘图函数都返回 Matplotlib Figure，可以显示或保存。漏检的 prediction、额外预测的 GT 均为 `None`，不会被自动补成 0，也不会从结果里删除。

## Python / notebook

V4 的 roivue 交互对比见 [v4_segmentation_roivue.ipynb](../visualize_mask/v4_segmentation_roivue.ipynb)：使用 `visual` kernel，支持整幅 CT 叠加、逐结节对比、Dice/IoU 和独立 HTML 导出。V4 加载时同时恢复 GroupNorm MedicalNet、radiomics baseline 和 semantic residual bank，不重新拟合。

在仓库根目录运行，使用已有 `pro6000` 环境。完整模型的整 CT 推理建议在分配到 GPU 的节点执行。

已执行的随机单例示例见 [try.ipynb](try.ipynb)：当前用种子 422 从 testing split 抽取 `LIDC-IDRI-0516__scan522`，评估 V3 第 18 轮 checkpoint，并保存四类对照图与数值。Notebook 使用 `py311` kernel 展示结果，推理由 `pro6000` 环境执行；已有 `try_results/` 缓存时可直接在 CPU 上重新运行展示部分。之前 seed 42 的结果保留在独立目录中。

Cell 6 使用 `runner.ensure_case_result`：缺少缓存时检查推理环境是否能使用 CUDA；CPU 节点会自动提交 `rudin` 分区的单卡 A6000 Slurm 作业，显示状态并等待结果。本机已有 CUDA GPU 时直接运行。可在该单元格修改 `partition/account/gres`。更换 seed 后无需手动申请 notebook GPU kernel；中断等待后再次运行会继续等待已有作业，避免重复提交。失败时异常包含实际推理日志。2026-09-24 已用 seed 422 验证 CPU → Slurm GPU → 缓存的完整流程，作业 `12693960` 正常完成。

Notebook 的整个 screening 分割使用 `plot_screening_segmentation_3d(result, stage="final")`：与 `visual.ipynb` 一样采用 marching cubes + Plotly `Mesh3d`，红色预测、蓝色 GT，可旋转、缩放和点击图例切换显示。顶点按完整 affine 转换为物理坐标；空 mask 明确标注。传入 `save_path="final_3d.html"` 可导出独立 HTML。该函数额外需要 `plotly` 和 `scikit-image`（`py311` 环境已有）。

```python
from back_prop.evaluate.eval_package import (
    JointModelEvaluator,
    screening_segmentation, nodule_segmentation, semantic_features, malignancy,
    plot_screening_segmentation, plot_nodule_segmentation,
    plot_semantic_features, plot_malignancy,
    save_result, load_result,
)

evaluator = JointModelEvaluator.from_checkpoint(
    "/usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v3_runs/12616464/epoch_018.pt",
    manifest="vista3D/data/folds/fold_0.json",
    device="cuda",
)
print(list(evaluator.cases))
result = evaluator.evaluate("LIDC-IDRI-0005__scan16")

# 1. 整个 screening 的最终分割；另可查看初始 VISTA screening 分割。
fig = plot_screening_segmentation(result, stage="final")
fig_screening = plot_screening_segmentation(result, stage="screening")
print(screening_segmentation(result)["metrics"])

# 2. 按真实物理结节 ID 查看预测/GT；编号可从下表取得。
matches = nodule_segmentation(result)
print([(r["nodule_id"], r["query_id"], r["status"]) for r in matches])
if result.ground_truth:
    fig_nodule = plot_nodule_segmentation(result, nodule_id=result.ground_truth[0].nodule_id)
# 也可以按预测 query ID 查看，包括未匹配到 GT 的预测。
if result.predictions:
    fig_query = plot_nodule_segmentation(result, query_id=result.predictions[0].query_id)

# 3. 每个结节的 semantic features / 官方评分。
fig_semantics = plot_semantic_features(result)
semantic_rows = semantic_features(result)

# 4. 每个结节和整个 scan 的 malignancy / 官方评分衍生 GT。
fig_malignancy = plot_malignancy(result)
risk_comparison = malignancy(result)

# 保存所有对照图、数值与数组；可选 NIfTI 导出。
save_result(result, "back_prop/evaluate/eval_package/results/LIDC-IDRI-0005__scan16", nifti=True)
```

`axis=2` 是 axial，`axis=1` 是 coronal，`axis=0` 是 sagittal；可给分割绘图函数传入 `slice_index` 指定整个 CT 网格中的切片。默认切片取 GT 最大截面；没有 GT 时取预测最大截面，都没有则取中间切片。`plot_nodule_segmentation` 的标题还会注明显示区域的 origin，其显示切片编号为局部编号。图中红色是预测，蓝色是官方 GT。

如果已有加载好的模型，可以直接使用：

```python
from back_prop.evaluate.eval_package import evaluate_case, OfficialAnnotations
case = evaluator.cases["LIDC-IDRI-0005__scan16"]
result = evaluate_case(model, case, annotations=evaluator.annotations)
```

`evaluate_case` 接受一个 manifest row；`JointModelEvaluator.evaluate` 另外校验 case 属于指定 split。模型会切换到 `eval()`，使用 `torch.inference_mode()`，forward 只接收 CT 和 V3 所需的原始 HU，不传 GT batch、框或 mask，不更新 FasterRisk bank。V3 全模型库按 checkpoint 的 inference 模式运行；`best` 消融保留其单模型行为。空 bank 的 checkpoint 会报错。

## 命令行

```bash
cd /home/users/yj255/code/imaging_feature
/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python -m back_prop.evaluate.eval_package \
  --checkpoint /usr/project/rudinlab/datasets/LIDC_IDRI/joint_model/model_v3_runs/12616464/epoch_018.pt \
  --manifest vista3D/data/folds/fold_0.json \
  --case-id LIDC-IDRI-0005__scan16 \
  --output back_prop/evaluate/eval_package/results \
  --nifti
```

省略 `--case-id` 会顺序评估整个 test split；可以多次传入该参数，也可用 `--max-cases 2` 先检查少量 CT。省略 manifest 时读取 checkpoint 中的 manifest 路径。支持原始 fold JSON 和 `{"splits": {"testing": [...]}}` 格式的 cohort JSON。

每个 case 文件夹包含：

- `screening.png`、`final.png`、逐 nodule / 未匹配 query 的分割图、semantic 与 malignancy 对照图。
- `result.json`：四类对照的数值、每位读者评分、GT 来源路径、匹配信息、checkpoint 和阈值。
- `arrays.npz`：HU CT、affine、screening mask、逐预测 ROI 的 mask / 概率以及每个 GT mask。ROI 保存为紧凑数组和 origin，避免为每个结节分配整 CT。
- `--nifti` 时另导出 `screening_prediction.nii.gz`、`final_prediction.nii.gz`、`official_gt.nii.gz`。

输出根目录的 `summary.json` 包含每个已完成 case 的指标和匹配数量。全 test set 的 Python 入口是 `evaluator.evaluate_test_set()`，返回迭代器，便于逐例保存释放内存。结果文件夹不能并发写入同一 case。

## GT 与评估定义

- 评分来源是仓库根目录的 **`all_ct_annotations.csv`**，即本项目导出的 LIDC 读者标注。**`nodule_iden.csv` 只提供 annotation → physical nodule 的对应关系**，不作为数值评分来源。默认逐读者 mask 目录为 `/usr/project/rudinlab/datasets/LIDC_IDRI/seg_result/official_seg_result/annolvl`。可通过 `annotations_csv`、`identity_csv`、`mask_dir` 覆盖；CLI 参数见 `--help`。
- 每个物理结节使用至少 `min(min_votes, reader_count)` 票的 consensus；默认 `min_votes=2`，与训练数据集一致。这是由官方读者标注**合成的参考 mask**，不是 LIDC 额外提供的单一官方真值。结节分组使用物理 ID，不按连通域重新编号。
- CT 和官方 mask 先转 canonical XYZ，按训练流程重采样至名义 1 mm 网格：HU 线性插值，mask 最近邻。导出 affine 对齐 `scipy.ndimage.zoom(grid_mode=False)` 的端点，不能直接沿用原 CT affine。
- 匹配在最终细分割的 3D mask 上进行。默认 `minimum_iou=0.1`，先最大化有效一对一匹配数，再最大化总 IoU。设置 0 时仍要求正重叠。无重叠不会强行配对。相邻 GT 即使接触仍保留各自 ID；重复预测最多匹配一个 GT。`query_id` 是模型 query 编号，不等于 `nodule_id`。
- `stage="screening"` 对比初始 VISTA `discovered_mask`；`stage="final"` 对比按 objectness 自动选出的 refined mask 并集。默认 mask 阈值 0.5，objectness 阈值沿用模型配置。原始 fine logits 的 sigmoid 用于分割；V3_hard / V3_anneal 的诊断温度从 checkpoint 恢复，只影响模型原有的诊断计算，记录于 metadata。在 0.5 阈值下，正温度缩放不改变二值 mask。
- 六个属性顺序为 **lobulation、margin、sphericity、spiculation、subtlety、texture**。GT 保留每个读者的 1–5 评分、均值和经验分布。V2 额外输出的 malignancy ordinal 分布也保留在数据结果中。
- malignancy 的 GT 是**读者评分衍生标签**：平均分 <3 为 0，>3 为 1，等于 3 为 `None`。这些不确定结节仍出现在所有分割和属性对照里。scan 有任一明确恶性结节即为 1；否则存在不确定结节时为 `None`，全部明确良性或官方记录无结节时为 0。它不是病理诊断标签。
- 默认整 CT 分割指标包含所有已标注结节，包括 malignancy 均分为 3 的结节。因此它与排除这些结节的训练/历史评估指标可能不同。空 consensus 结节保留并标记 `gt_mask_empty`；不能参与正重叠匹配。预测和 GT 都为空时，体素 Dice / IoU 定义为 1。
- 缺失 annotation join、缺失官方 mask 或影像几何不一致会报错，不会把缺失标注当阴性。split 中与 training 重复的患者会被拒绝；V3 还检查 checkpoint bank 的训练患者。

## 在 CPU 重新查看

```python
from back_prop.evaluate.eval_package import load_result, plot_semantic_features, plot_nodule_segmentation
result = load_result("back_prop/evaluate/eval_package/results/LIDC-IDRI-0005__scan16")
fig = plot_semantic_features(result)
```

读取结果和绘图只需要 NumPy、SciPy、Matplotlib；NIfTI 导出还需要 nibabel。推理使用仓库现有的 PyTorch / MONAI / VISTA 环境。V3 checkpoint 包含完整权重，无需再次读取最初的 VISTA / MedicalNet 预训练权重；V2 构造仍使用配置中的 `ordinal_init` JSON。checkpoint 使用项目的完整训练格式，应加载可信的本地 checkpoint。

## 验证

```bash
/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python -m unittest back_prop.evaluate.eval_package.tests.test_evaluate -v
```

测试覆盖真实 V3 系列模型代码的小网络 checkpoint 加载/forward、V2 输出适配、GT 不进入 forward、bank 不变、负 ROI origin、轴翻转和重采样、官方读者评分来源、缺失/错位 GT 报错、一对一匹配、漏检、空结果、保存重载和所有绘图函数。

2026-09-24 验证：14 项测试通过；真实 V3 `epoch_018.pt` 严格加载成功（30 个 bank 模型）；真实 test case `LIDC-IDRI-0005__scan16` 的 HU 和 3 个物理结节 GT mask/origin 与现有训练 loader 逐元素一致。随后在 `rudin-01` A6000 上完成随机 test case `LIDC-IDRI-0908__scan747` 的完整 GPU 推理与结果导出（Slurm `12693742`）。真实 checkpoint 的 `Path` 类型 manifest 已转换为字符串，避免 JSON 导出失败；该情况已加入 checkpoint 加载测试。
