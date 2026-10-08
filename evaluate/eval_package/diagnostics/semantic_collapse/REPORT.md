# V3 semantic 输出近常数：成因与修复建议

检查对象：`model_v3_runs/12616464/epoch_018.pt`，notebook 当前 seed=4、`LIDC-IDRI-0301__scan301`。

**主因已定位到 MedicalNet 的输入与固定 BatchNorm 统计量严重失配。** `[0,1]` CT 乘稀疏 soft mask、在大 ROI 中缩放后，结节信号非常弱；第一层 BN 又按预训练时很大的方差归一化，进一步压低图像差异。后续编码器及全局平均池化输出主要由共同的背景响应决定，六个 ordinal heads 因而给出近似固定评分。

这是对当前 checkpoint、输入流程和受控样本的机制诊断；后面的方案尚未重新训练或证明泛化性能改善。原 checkpoint、生产模型代码、运行中的训练和 notebook 均未由这些实验修改。

## 1. 最直接的证据

使用当前 CT 中已匹配到 GT 的 query 7、23，保持 checkpoint 权重不变。CPU FP32 重建原语义输出，与已保存 GPU 结果最大差异为 `4.55e-5`。以下差异均指这两个结节六个 semantic 期望评分的最大绝对差；各行是独立的受控输入或归一化干预。

| 对照条件 | 两结节最大评分差 |
|---|---:|
| 原始流程：128³ masked ROI → 64³，固定 BN | 0.0000132 |
| **只让第一层 BN 使用当前两个 ROI 的统计量** | **0.948931** |
| 所有 BN 使用当前两个 ROI 的统计量 | 0.324596 |
| 同一 ROI 和 soft mask，改用原始 HU | 0.503500 |
| 以预测 mask 最大连通分量为中心，32³ crop，保留 CT 上下文 | 0.054319 |

这些实验恢复了输入敏感性，**没有证明评分正确**。例如只调整第一层 BN 时有评分接近 1 或 5，因此不能直接将该干预作为旧 checkpoint 的部署修补。

进一步排除单纯的漏检问题：

- 全零输入与 query 23 的真实 masked 输入，六项评分最大差只有 `0.0000163`。
- 换成两个官方 GT 的真实 mask，并以 GT 中心裁剪，仍只产生 `0.00000334` 的评分差。这里明确使用 GT 作为诊断对照，未计入正式预测。
- 预训练 encoder 接上 epoch-18 的 head，在相同稀疏输入上也只有 `0.00000191` 的差异。输入问题在 encoder 侧已有体现。

原始数据：[probe_results.json](probe_results.json)、[bn_controls.json](bn_controls.json)。

## 2. 信息在哪里被压低

输入 CT 的归一化为 `(clip(HU, -1024, 1024) + 1024) / 2048`，见 [model_v1/data.py](../../../../model_v1/data.py)。语义分支随后执行 `CT * sigmoid(mask_logits)` 并缩放至 64³，见 [model_v3/features.py](../../../../model_v3/features.py)。

对 query 7、23，送入 MedicalNet 的非微小信号体素（绝对值 >0.001）仅占 **0.0252% 和 0.0313%**；输入标准差约 **0.00287 和 0.00330**。

第一层 BN 的当前 checkpoint 数值：

| 指标 | 数值 |
|---|---:|
| 保存的各通道 running standard deviation 中位数 | 1135.13 |
| 当前卷积输入的各通道实际 standard deviation 中位数 | 0.33818 |
| 两结节在 `conv1` 输出处的差异 RMS | 0.86432 |
| 两结节在 `bn1` 输出处的差异 RMS | 0.00074451 |
| 第一层 BN 对结节差异的压缩倍数 | **1160.92 倍** |
| 最后 global-average-pooling 后的 embedding 差异 RMS | 0.00005049 |

固定统计量下，BN 对两输入差异的作用为：

`Δy = γ / sqrt(running_var + ε) × Δx`。

减均值和偏置不提供结节间差异；它们可产生共同的非零背景响应。这解释了为何 embedding 并非全零，输出却接近固定值。[PyTorch BatchNorm3d 官方定义](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm3d.html)。

训练代码 `MedicalNetSemantics.train()` 对所有 BN 强制调用 `eval()`。核对 **53 层 BN、106 个 running mean/variance buffer**，它们与原始预训练权重完全相同，最大差为 0；整个训练没有使这些统计量适应本任务。BN 的可训练 affine 参数仍参与训练，不能将此误解为“整个 MedicalNet 没有训练”。

![逐层信号与全队列变异对照](diagnostic_summary.png)

## 3. 不是单个 screening 或 DataFrame 的问题

复核已有 **166 个测试 CT、527 个 GT 框定位结节**的 V3 输出，并重新计算 checkpoint SHA256，确认与当前 epoch-18 权重一致：

- **437/527（82.9%）** 的结节，六项评分全部距对应队列中位数不足 0.001。
- 461/527（87.5%）全部不足 0.01。
- subtlety 预测标准差为 0.00617，官方读者均分标准差为 1.07506。
- texture 预测标准差为 0.03456，官方读者均分标准差为 1.22811。
- 仍有少数较大变化的结节，不能说模型在所有输入上完全输出同一常数。微小变化也可能提供排序信号，所以 AUC 不能独自验证语义评分是否学好。

数据来源是已完成的 GT 框定位实验，使用模型自己的 mask，不是本次重新运行全部 CT，也不是“全测试集原生检测”的统计。[完整审计](cohort_audit.json)。

另外，逐行比对训练用 `nodule_iden.csv` 与官方 `all_ct_annotations.csv`，6859 条读者标注的六项 semantic 和 malignancy 均无缺失对应、无评分差异。训练集中 1599 个符合 malignancy 筛选规则的物理结节，其各项 semantic 标准差为 0.78–1.04，标签并非常数。

相同的当前 ROI 输入第 3、6、9、12、15 epoch 的语义分支时，最大评分差也仅约 `6e-6–9e-5`。这只证明早期权重对这些固定输入已不敏感，不代表重跑了各早期 checkpoint 自身的分割。训练 ordinal loss 从 epoch 2 的 1.2439 至 epoch 17 的 1.2392 基本停滞；epoch 18 为 1.2264。它与长期学习近似群体先验的现象一致，但单靠 loss 不能确定因果。

## 4. 与 malignancy 差异的关系

V3 风险输入是 **18 项数学 radiomics + 6 项 semantic**，不是只看表里的六项。原始 HU 与 soft mask 计算的 radiomics 仍可随结节改变，所以 semantic 接近常数时 malignancy 仍会不同。修复语义分支后特征分布会改变，原 FasterRisk bank 的标准化和系数不能未经重拟合就视为仍适用。

## 5. 建议按以下顺序实施

1. **先修复归一化适配，并单独验证语义分支。** 从预训练 encoder 或受控对照的 epoch-18 encoder 开始，仅用训练集的代表性结节 ROI 重新估计 BN 统计量，并微调 encoder/head。删除或改造训练时无条件冻结 BN 的逻辑；推理时使用固定的训练统计量。另一实验分支可改为 GroupNorm，再充分微调。GroupNorm 在 train/eval 都从输入计算组内统计量，不依赖预训练 running buffers；它并不是旧 BN checkpoint 的等价替换。[PyTorch GroupNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html)。
2. **让 semantic 输入保留结节信号和上下文。** 使用围绕结节的较紧物理空间 crop（例如以 32/64 mm 多尺度为候选并由验证集选择）。保留 CT 主通道，将 soft mask 用作独立输入或软 attention，而不是把大 ROI 的 CT 几乎全部乘成零。为 MedicalNet 单独设计强度归一化，并与 BN 统计量一起适配。不能仅凭本次敏感性实验直接改成 raw HU 或乘固定的 2048。
3. **先证明模型能学到六项语义，再恢复联合训练。** 在训练集中的小样本子集做可过拟合检查；用独立 validation split 比较 feature-wise MAE、ordinal NLL、预测方差、与 GT 的相关性，并和训练集常量先验基线比较。语义学习稳定后，再逐步接回预测 ROI/mask 和 malignancy loss，同时从训练样本重新建立 FasterRisk 特征缓存、统计量和 bank。
4. **将异常检查纳入训练记录。** 分 feature 记录 prediction/GT 的标准差和 MAE，记录空输入与真实输入的差异、MedicalNet 输入占比、第一层 BN 的实际/运行方差比。只检查梯度有限、梯度非零或总 loss 下降，不能发现本次近常数问题。

32³ 紧裁剪对照还发现 query 7 的 mask 有多个远离的连通分量：直接取整体包围盒中心会裁到空白。因此紧裁剪方案需要明确结节/连通分量选择规则；本报告表格使用保留了最大连通分量的对照，不使用这一空白 crop 作为“修复成功”的证据。

**优先级建议：先做“训练集 BN 适配 + 紧 ROI 保留 CT 上下文”的语义专项实验。** 只增加训练轮数、调整学习率、降低 mask temperature 或提高 malignancy loss，均没有直接修复当前已证实的归一化失配。当前训练已有语义 warmup，因此新增 warmup 必须伴随输入/归一化修复并验证实际学习，而不只是延长现有流程。

## 6. 另一个需要审查但并非本次主因的兼容性问题

当前 `MedicalNetEncoder` 使用 MONAI 默认 ResNet50：conv1 stride=1，layer3/4 stride=2；[Tencent 原始 MedicalNet](https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py) 是 conv1 stride=2、layer3/4 stride=1 且 dilation=2/4。权重 shape 匹配不保证这些非参数设置一致。

本次仅恢复原始 stride/dilation、保持权重和稀疏输入不变，评分差仍只有 `0.0000148`，没有消除近常数现象。因此它是后续重建预训练 backbone 时应解决的兼容性问题，不能替代前面的输入/BN 主因。公开 MedicalNet 的 MRBrainS18 示例采用非零区域强度标准化，但那是迁移示例，不能据此断言本地 23-dataset 权重的全部原始训练预处理。

## 复现与文件

从仓库根目录执行（约数分钟，CPU，读取已有 checkpoint 和缓存）：

```bash
/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python back_prop/evaluate/eval_package/diagnostics/semantic_collapse/probe_semantics.py --output back_prop/evaluate/eval_package/diagnostics/semantic_collapse/probe_results.json
/home/users/yj255/.local/miniforge3/envs/pro6000/bin/python back_prop/evaluate/eval_package/diagnostics/semantic_collapse/probe_bn_controls.py --output back_prop/evaluate/eval_package/diagnostics/semantic_collapse/bn_controls.json
/home/users/yj255/.local/miniforge3/envs/py311/bin/python back_prop/evaluate/eval_package/diagnostics/semantic_collapse/audit_existing_results.py
/home/users/yj255/.local/miniforge3/envs/py311/bin/python back_prop/evaluate/eval_package/diagnostics/semantic_collapse/render_summary.py
```

JSON 记录源脚本 SHA256、实际数值、checkpoint 路径；队列审计另记录 checkpoint SHA256。输出图提供 [PNG](diagnostic_summary.png) 和 [PDF](diagnostic_summary.pdf)。
