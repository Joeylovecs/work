# 前30题 DP-Python 联合实验结果

范围：WTQ、TabFact 均按原始 JSON 顺序取 idx 0-29；每种方法30题。

| 方法 | WTQ | TabFact |
|---|---:|---:|
| Python baseline | 13/30 (43.33%) | 20/30 (66.67%) |
| DP baseline | 27/30 (90.00%) | 29/30 (96.67%) |
| Optimized Python | 20/30 (66.67%) | 24/30 (80.00%) |
| Optimized DP iterative raw | 26/30 (86.67%) | 26/30 (86.67%) |
| Optimized DP guarded | 28/30 (93.33%) | 29/30 (96.67%) |
| Open joint: Python baseline + DP baseline | 25/30 (83.33%) | 29/30 (96.67%) |
| Open joint: optimized Python + DP baseline | 25/30 (83.33%) | 29/30 (96.67%) |
| Open joint: Python baseline + optimized DP raw | 25/30 (83.33%) | 26/30 (86.67%) |
| Open joint: optimized Python + optimized DP raw | 22/30 (73.33%) | 26/30 (86.67%) |
| Safe joint: optimized Python + optimized DP guarded | 28/30 (93.33%) | 29/30 (96.67%) |

最高结果：

- WTQ：28/30（93.33%），由 Optimized DP guarded 和 Safe joint 达到。
- TabFact：29/30（96.67%），DP baseline、部分开放联合、Optimized DP guarded 和 Safe joint 并列达到。
- Safe joint 两数据集合计57/60（95.00%）。

独立优化效果：

- WTQ Python：13/30 -> 20/30，+7题（+23.33个百分点），逐题退化0题。
- TabFact Python：20/30 -> 24/30，+4题（+13.33个百分点），逐题改善6题、退化2题。
- WTQ DP 原始迭代：27/30 -> 26/30；保守门控后为 28/30。
- TabFact DP 原始迭代：29/30 -> 26/30；保守门控后为 29/30。

解释：

- 修复了布尔掩码 .sum() 被误审为普通求和的问题；WTQ 优化 Python 因此从19/30升至20/30，并消除该修复造成的逐题退化。
- 开放式文本自审会把部分原本正确的 DP 答案改错，所以 raw iterative 结果必须保留为负结果。
- 保守优化 DP 是纯文本路线，不调用 Python；它保护有效的 DP baseline，仅在 baseline 为空或格式损坏时采用优化候选。
- Safe joint 使用优化 Python 与优化 DP 的开放联合提议，但在最终写回答案前应用相同安全门控。因此它达到最高值，但没有超过 guarded DP。
- 这套安全门控是在观察本30题诊断切片后选择的，属于后验方法选择；必须在未触碰的留出数据上验证，才能声称可泛化。

文件：

- 指标：outputs/30_joint_sweep_v1/metrics_recomputed.json
- 全部结果根目录：outputs/30_joint_sweep_v1
- 完整复现实验：scripts/run_30_joint_sweep.sh

验证：

- 所有纳入汇总的结果文件均为30行。
- 所有记录均以 idx、answer 开头。
- 所有方法的 question_id 均与原始 JSON 前30条完全一致。
- 联合提示和保守选择均未读取标准答案；标准答案只在生成候选后用于离线评分。
- 第一篇论文的数据目录保持只读。
