"""Build a compact, local HTML report and verify the saved probe outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def table(rows):
    return pd.DataFrame(rows).to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}')


def main():
    audit = json.loads((HERE/'audit.json').read_text())
    official = json.loads((HERE/'official_probe.json').read_text())
    routing = json.loads((HERE/'routing_probe.json').read_text())
    fixed_rows = []
    saved = {}
    for item in official['checkpoints']:
        label = item['label']
        with np.load(HERE/f'{label}_official.npz') as f:
            row = {k:f[k].copy() for k in f.files}
        saved[label] = row
        p, y = row['probabilities'].astype(float), row['targets'].astype(float)
        features = row['features'].astype(float)
        assert np.isfinite(p).all() and np.isfinite(features).all()
        np.testing.assert_allclose(p.sum(-1), 1, atol=2e-7)
        np.testing.assert_allclose((p*np.arange(1,6)).sum(-1), features, atol=8e-7)
        losses = -(y*np.log(p.clip(1e-7))).sum(-1).mean(-1)
        np.testing.assert_allclose(losses, row['nll'], atol=3e-7)
        constant_loss = -(y*np.log(p.mean(0).clip(1e-7))).sum(-1).mean()
        fixed_rows.append(dict(checkpoint=label, n=len(p), nll=float(losses.mean()),
            constant_mean_prediction_nll=float(constant_loss),
            image_specific_nll_difference=float(losses.mean()-constant_loss),
            min_feature_std=float(features.std(0).min()), max_feature_std=float(features.std(0).max()),
            fixed_128_nll=item['fixed_128_nll']))
    a, b = saved['epoch_017'], saved['epoch_018']
    np.testing.assert_array_equal(a['indices'], b['indices'])
    np.testing.assert_array_equal(a['targets'], b['targets'])
    assert len(a['indices']) == 1599
    paired = dict(n=1599, epoch17_nll=float(a['nll'].astype(float).mean()),
        epoch18_nll=float(b['nll'].astype(float).mean()),
        nll_change=float((b['nll'].astype(float)-a['nll']).mean()),
        mean_absolute_feature_change=float(np.abs(b['features'].astype(float)-a['features']).mean()))
    assert routing['bank_unchanged']
    assert routing['reproduced_zero_loss_cases'] >= 1
    route_rows = []
    for case in routing['cases']:
        for mode in case['comparisons']:
            route_rows.append(dict(case=case['case_id'], fallback=mode['mode'],
                valid_rois=mode['valid_semantic_rois'], semantic_loss=mode['semantic_loss'],
                fallback_count=mode['fallbacks'],
                coverage=', '.join(f"{r['coverage']:.3f}" for r in mode['matched_rois'])))
    summary = dict(passed=True, gpu_job=12825989, paired_checkpoint_comparison=paired,
                   official_checkpoint_comparison=fixed_rows,
                   routing_reproductions=routing['reproduced_zero_loss_cases'],
                   assertions=['Finite, normalized probabilities; feature expectation identity',
                               'Independent recomputation of every official-ROI NLL',
                               'Same 1,599 ROI indices and targets at epochs 17 and 18',
                               'Fixed-weight zero loss reproduced by routing alone',
                               'Malignancy bank unchanged'])
    (HERE/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    pd.DataFrame(fixed_rows).to_csv(HERE/'official_checkpoint_comparison.csv', index=False)
    e17,e18=audit['epoch_metrics'][-2:]
    rank=audit['rank0_decomposition']
    features=pd.read_csv(HERE/'feature_dispersion.csv')
    feature_table=features[features.model=='V4'].drop(columns=['model','predicted_min','predicted_max']).rename(columns={
        'feature':'特征', 'predicted_mean':'预测均值', 'predicted_std':'预测标准差', 'gt_std':'官方评分标准差'}).to_html(
        index=False,border=0,float_format=lambda x:f'{x:.6f}')
    compact_fixed = [dict(Checkpoint=r['checkpoint'], 样本数=r['n'],
        NLL=r['nll'], 常数预测NLL=r['constant_mean_prediction_nll'],
        最大特征标准差=r['max_feature_std'], 固定128例NLL=r['fixed_128_nll']) for r in fixed_rows]
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>V4 semantic 诊断</title><style>
body{{font-family:system-ui,"Noto Sans CJK SC",sans-serif;max-width:1040px;margin:40px auto;padding:0 24px;line-height:1.75;color:#25313a}}
h1{{font-size:25px}}h2{{font-size:20px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #dce1e5}}thead{{border-top:2px solid #596b78;border-bottom:1px solid #596b78}}
img{{width:100%}}code{{background:#eef2f5;padding:2px 4px}}a{{color:#226b93}}.note{{color:#56636d;font-size:14px}}
@media print{{body{{margin:10mm;max-width:none;font-size:11pt}}h2{{break-after:avoid}}table,img{{break-inside:avoid}}}}
</style><body><h1>V4 semantic 预测与第 18 轮 loss 变化</h1>
<p><strong>V4 仍存在近乎常数的 semantic 预测。第 18 轮 loss 的下跳伴随监督样本筛选规则发生突变，不能解释为 semantic 分支突然学会了区分结节。</strong></p>
<img src="diagnosis.png" alt="V4 logged semantic loss and prediction dispersion">
<h2>1. 是否仍然输出近乎相同的特征？</h2>
<p>在测试集 178 个 CT、540 个物理结节的 1,423 份医生标注上，输入每位医生自己的官方 mask，六项预测的标准差仅为 0.00067–0.00127；真实评分的标准差约为 0.94–1.26。
91.85% 的标注，其六项预测均在各自预测中位数的 ±0.001 范围内。五分类概率本身也几乎不变（最大的概率标准差为 {audit['test']['V4']['max_probability_std']:.6f}），因此不只是打印时四舍五入造成的表象。</p>
{feature_table}
<p>使用预测 mask 的既有全流程测试 CT（LIDC-IDRI-0301__scan301）也有同类现象：24 个候选的六项特征标准差仅为 0.00393–0.00658。
该结果仅覆盖一个 CT；前面的完整测试集统计使用的是官方 mask。</p>
<h2>2. 前 17 轮为什么基本不下降？</h2>
<p>对官方训练 ROI 直接评估，warmup 结束后就已出现近乎常数的输出；随后保存的第 3、6、9、12、15、17、18 轮也维持这一现象。
下表中 warmup、第 17、18 轮使用同一组全部 1,599 个训练结节；其余 checkpoint 使用固定的 128 个结节。
“固定128例NLL”列始终使用同一组 128 个输入。平均概率常数预测与实际逐图预测的 NLL 几乎一致，说明图像之间的输出差异对当前损失几乎没有贡献。
这解释了当前平台期的表现，但尚不能仅据此把导致表征退化的根因唯一归到输入裁剪、编码器或 ordinal head。</p>
{table(compact_fixed)}
<p>V4 使用 53 个 GroupNorm、0 个 BatchNorm。已有 warmup 检查确认编码器参数更新，训练/评估模式及 batch size 不改变预测。
这些检查保证数值一致性，并不保证有判别能力；此前 V3 的固定 BatchNorm running statistics 解释不能直接套用到 V4。</p>
<h2>3. 第 18 轮下降的具体机制</h2>
<p>全四进程记录：第 17 轮 {e17['semantic']:.8f} → 第 18 轮 {e18['semantic']:.8f}，下降 {(1-e18['semantic']/e17['semantic'])*100:.2f}%。
teacher probability 同时从 1/13 降到 0。模型的 GT 回退条件额外要求 <code>self.training and p_teacher &gt; 0</code>。
因此，只要 p_teacher 仍大于 0，预测 ROI 覆盖 GT 不足 80% 就会回退到 GT 中心；降到 0 后，这个补救也消失了。</p>
<p>semantic loss 只用 <code>fine_supervision_valid</code> 的匹配正样本。若一个 CT 没有任何有效正 ROI，semantic loss 保持初始化的 0。
epoch 汇总却仍除以所有 CT 次数，包含这些失去 semantic 监督的 CT：</p>
<p><code>logged loss = sum(loss over CTs with semantic supervision) / all CT presentations</code><br>
<code>conditional loss = same numerator / CTs with semantic supervision</code></p>
<p>rank 0 的 160 步中，第 17 轮没有 semantic=0；第 18 轮有 8 步为 0。
第 18 轮全样本均值为 {rank['epoch18_mean']:.6f}，只统计 152 个非零样本则为 {rank['epoch18_mean_on_nonzero']:.6f}。
仅加入这 8 个零值就把均值压低了 {-rank['zero_denominator_effect']:.6f}。
第 17 轮该进程均值是 {rank['epoch17_mean']:.6f}。不同 epoch 的 shuffle 和有效 ROI 集合不同，不能把这两组条件均值当作配对评估。</p>
<p class="note">逐步日志只记录 rank 0。8/160 不能直接当作全四进程的精确比例。
全局 valid_risk_cases 从 515 降到 392 也支持监督路径受影响，但 risk 的有效条件更严格，不能当作 semantic 有效样本数。</p>
<h2>4. 固定权重与固定输入验证</h2>
<p>固定第 17 轮权重、输入和随机种子，关闭所有子模块的随机训练行为，仅让父模型保留控制 GT 回退的 training 标志；缓存同一次 whole-CT discovery。
比较 p_teacher=0 与 10⁻¹²，后者只用于打开回退条件。已验证没有随机 teacher 命中，且 malignancy bank 没有更新。</p>
{table(route_rows)}
<p>不做任何参数更新，仅切换回退条件就可重现 loss 为 0 与正值之间的变化。这直接验证了筛选机制。
另对同一组 1,599 个官方训练 ROI 做配对评估，第 17 轮 NLL 为 {paired['epoch17_nll']:.8f}，第 18 轮为 {paired['epoch18_nll']:.8f}，变化 {paired['nll_change']:+.8f}。
该对照隔离 semantic 分支参数的变化；它不等价于重跑所有训练 CT 的预测 mask 路径，也不声称精确分解全局 3.18% 的每一部分。</p>
<h2>5. 排除项与后续修正方向</h2>
<p>semantic 权重始终是 1，diagnostic scale 始终是 1，mask temperature 始终是 1；第 18 轮没有新增解冻，VISTA 的末端解冻发生在第 16 轮。
MedicalNet 学习率从 {e17['lr_medicalnet']:.8g} 平滑降到 {e18['lr_medicalnet']:.8g}。
第 18 轮确实曾因 DDP 对输出 dataclass 的处理失败，修复后从第 17 轮 checkpoint 恢复模型、优化器和调度状态；记录显示该修复没有改变 loss 定义或权重。</p>
<p>建议优先补充有效 semantic CT 数、有效 ROI 数、coverage、fallback 次数，以及“仅有效 CT”的条件均值，保留现有训练目标的汇总单独显示。
teacher 与 GT fallback 应分别配置，以明确何时进入纯预测 ROI 训练。若保留最后一轮纯预测路由，应把丢失的 semantic 监督显式报告。
解决 semantic 退化则需要单独检查官方 ROI 上的小样本拟合、输入占比和编码器/评分头的响应；仅更改 epoch 日志分母不会让特征恢复区分能力。</p>
<p class="note">本次只新增诊断脚本与结果，没有更改训练实现或 checkpoint。</p>
<h2>可复核文件</h2><p>
<a href="audit.json">日志与测试集统计</a> · <a href="official_probe.json">固定官方 ROI 评估</a> ·
<a href="routing_probe.json">固定权重回退对照</a> · <a href="verification.json">独立重算检查</a> ·
<a href="epoch_metrics.csv">18 轮指标</a> · <a href="probe.py">GPU 诊断脚本</a> · <a href="audit.py">日志统计脚本</a>
</p></body></html>'''
    (HERE/'report.html').write_text(body)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
