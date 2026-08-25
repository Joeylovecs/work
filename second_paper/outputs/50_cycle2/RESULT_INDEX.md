# 50条实验结果索引（cycle2）

- 根目录：`outputs/50_cycle2`
- 区间：flat index `[0, 50)`，end 为开区间。
- 详细逐条结果：各方法目录下的 `result.jsonl`；逐条第一篇兼容日志：`log/<flat_index>.txt`。

## WTQ

### baseline
- accuracy: 70.00%; execution success: 82.00%; execution-success-but-wrong: 11; API calls: 59
- 文件：`outputs/50_cycle2/wtq/baseline/summary.json`、`outputs/50_cycle2/wtq/baseline/result.jsonl`、`outputs/50_cycle2/wtq/baseline/selected_samples.json`

### audit
- accuracy: 56.00%; execution success: 82.00%; execution-success-but-wrong: 18; API calls: 88
- semantic exceptions: 22; layers: {'OPERATOR': 27, 'PARAMETER': 13, 'GLOBAL': 13}; error types: {'WRONG_AGGREGATION': 4, 'WRONG_TARGET_COLUMN': 13, 'MISSING_OPERATOR': 12, 'MISSING_AGGREGATION': 6, 'ANSWER_TYPE_MISMATCH': 7, 'FINAL_RETURN_SEMANTICS_MISMATCH': 4, 'MISSING_RANKING': 4, 'CARDINALITY_MISMATCH': 2, 'WRONG_RANKING_DIRECTION': 1}
- 文件：`outputs/50_cycle2/wtq/audit/summary.json`、`outputs/50_cycle2/wtq/audit/result.jsonl`、`outputs/50_cycle2/wtq/audit/selected_samples.json`

### joint
- accuracy: 84.00%; execution success: 82.00%; execution-success-but-wrong: 7; API calls: 209
- 文件：`outputs/50_cycle2/wtq/joint/summary.json`、`outputs/50_cycle2/wtq/joint/result.jsonl`、`outputs/50_cycle2/wtq/joint/selected_samples.json`

### 配对审计统计
- baseline 静默错误候选：11
- auditor 检测到：7；检测率：63.64%
- semantic repair 后恢复正确：0；false positive flags：14；repair degradation：7

- silent IDs：nu-0, nu-1, nu-648, nu-2881, nu-3520, nu-2522, nu-4169, nu-1506, nu-14, nu-4258, nu-2675

## TABFACT

### baseline
- accuracy: 86.00%; execution success: 98.00%; execution-success-but-wrong: 7; API calls: 51
- 文件：`outputs/50_cycle2/tabfact/baseline/summary.json`、`outputs/50_cycle2/tabfact/baseline/result.jsonl`、`outputs/50_cycle2/tabfact/baseline/selected_samples.json`

### audit
- accuracy: 80.00%; execution success: 98.00%; execution-success-but-wrong: 10; API calls: 65
- semantic exceptions: 12; layers: {'OPERATOR': 16, 'GLOBAL': 2, 'PARAMETER': 9}; error types: {'MISSING_RANKING': 6, 'ANSWER_TYPE_MISMATCH': 2, 'MISSING_OPERATOR': 6, 'WRONG_TARGET_COLUMN': 3, 'UNKNOWN_COLUMN': 4, 'WRONG_AGGREGATION': 1, 'MISSING_FILTER': 2, 'MISSING_FILTER_PARAMETER': 2, 'MISSING_AGGREGATION': 1}
- 文件：`outputs/50_cycle2/tabfact/audit/summary.json`、`outputs/50_cycle2/tabfact/audit/result.jsonl`、`outputs/50_cycle2/tabfact/audit/selected_samples.json`

### joint
- accuracy: 90.00%; execution success: 98.00%; execution-success-but-wrong: 5; API calls: 201
- 文件：`outputs/50_cycle2/tabfact/joint/summary.json`、`outputs/50_cycle2/tabfact/joint/result.jsonl`、`outputs/50_cycle2/tabfact/joint/selected_samples.json`

### 配对审计统计
- baseline 静默错误候选：7
- auditor 检测到：3；检测率：42.86%
- semantic repair 后恢复正确：0；false positive flags：9；repair degradation：3

- silent IDs：1-2126093-3.html.csv_q0, 2-10826385-15.html.csv_q9, 2-11051845-5.html.csv_q1, 2-12206617-3.html.csv_q3, 2-1122485-2.html.csv_q3, 2-14101654-10.html.csv_q4, 1-16168849-1.html.csv_q3

