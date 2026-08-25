# 100 条实验总索引

- 样本区间：flat index `[0,100)`，end 为开区间。
- WTQ 和 TabFact 都按第一篇项目顺序运行，完整 sample ID 清单在每个方法目录的 `selected_samples.json`。
- 当前准确率最佳目录：`outputs/100_cycle1_clean/`。
- 第二轮迭代目录：`outputs/100_cycle2_clean/`。

## 当前最佳配置

| 数据集 | baseline | audit | joint | 推荐查看 |
|---|---:|---:|---:|---|
| WTQ | 67% | 66% | 83% | `100_cycle1_clean/wtq/joint/` |
| TabFact | 87% | 83% | 92% | `100_cycle1_clean/tabfact/joint/` |

## 第二轮迭代

| 数据集 | baseline | audit | joint | 主要结论 |
|---|---:|---:|---:|---|
| WTQ | 65% | 70% | 83% | audit 提升，joint 未提升 |
| TabFact | 84% | 81% | 91% | joint 和 audit 均未提升 |

每个方法目录包含 `config.json`、`selected_samples.json`、`result.jsonl`、`log/`、`summary.json` 和 `cache/`。
