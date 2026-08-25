# 实验结果索引

- 根目录：`outputs/100_cycle2`
- 区间：由 `selected_samples.json` 确定；当前实验默认是 flat index `[0, 100)`，end 为开区间。
- 详细逐条结果：各方法目录下的 `result.jsonl`；逐条第一篇兼容日志：`log/<flat_index>.txt`。

## WTQ

### baseline
- accuracy: 65.00%; execution success: 83.00%; execution-success-but-wrong: 20; API calls: 117
- 文件：`outputs/100_cycle2/wtq/baseline/summary.json`、`outputs/100_cycle2/wtq/baseline/result.jsonl`、`outputs/100_cycle2/wtq/baseline/selected_samples.json`

### audit
- accuracy: 70.00%; execution success: 83.00%; execution-success-but-wrong: 18; API calls: 174
- semantic exceptions: 51; layers: {'OPERATOR': 54, 'PARAMETER': 42, 'GLOBAL': 34}; error types: {'MISSING_OPERATOR': 22, 'MISSING_AGGREGATION': 9, 'WRONG_TARGET_COLUMN': 38, 'WRONG_AGGREGATION': 10, 'ANSWER_TYPE_MISMATCH': 13, 'CARDINALITY_MISMATCH': 8, 'FINAL_RETURN_SEMANTICS_MISMATCH': 13, 'MISSING_RANKING': 13, 'UNKNOWN_COLUMN': 4}
- 文件：`outputs/100_cycle2/wtq/audit/summary.json`、`outputs/100_cycle2/wtq/audit/result.jsonl`、`outputs/100_cycle2/wtq/audit/selected_samples.json`

### joint
- accuracy: 83.00%; execution success: 80.00%; execution-success-but-wrong: 14; API calls: 420
- 文件：`outputs/100_cycle2/wtq/joint/summary.json`、`outputs/100_cycle2/wtq/joint/result.jsonl`、`outputs/100_cycle2/wtq/joint/selected_samples.json`

### 配对审计统计
- baseline 静默错误候选：20
- auditor 检测到：11；检测率：55.00%
- semantic repair 后恢复正确：4；false positive flags：36；repair degradation：4

- silent IDs：nu-0, nu-648, nu-695, nu-862, nu-3520, nu-2522, nu-2555, nu-14, nu-1362, nu-4258, nu-2675, nu-1202, nu-1911, nu-4053, nu-1363, nu-1406, nu-498, nu-3430, nu-2913, nu-2147

## TABFACT

### baseline
- accuracy: 84.00%; execution success: 97.00%; execution-success-but-wrong: 14; API calls: 103
- 文件：`outputs/100_cycle2/tabfact/baseline/summary.json`、`outputs/100_cycle2/tabfact/baseline/result.jsonl`、`outputs/100_cycle2/tabfact/baseline/selected_samples.json`

### audit
- accuracy: 81.00%; execution success: 98.00%; execution-success-but-wrong: 17; API calls: 134
- semantic exceptions: 25; layers: {'OPERATOR': 25, 'PARAMETER': 21, 'GLOBAL': 3}; error types: {'MISSING_RANKING': 9, 'WRONG_TARGET_COLUMN': 9, 'UNKNOWN_COLUMN': 6, 'MISSING_OPERATOR': 10, 'MISSING_AGGREGATION': 4, 'MISSING_FILTER': 1, 'MISSING_FILTER_PARAMETER': 3, 'WRONG_ENTITY': 2, 'ANSWER_TYPE_MISMATCH': 3, 'BOOLEAN_POLARITY_ERROR': 1, 'WRONG_AGGREGATION': 1}
- 文件：`outputs/100_cycle2/tabfact/audit/summary.json`、`outputs/100_cycle2/tabfact/audit/result.jsonl`、`outputs/100_cycle2/tabfact/audit/selected_samples.json`

### joint
- accuracy: 91.00%; execution success: 98.00%; execution-success-but-wrong: 7; API calls: 402
- 文件：`outputs/100_cycle2/tabfact/joint/summary.json`、`outputs/100_cycle2/tabfact/joint/result.jsonl`、`outputs/100_cycle2/tabfact/joint/selected_samples.json`

### 配对审计统计
- baseline 静默错误候选：14
- auditor 检测到：6；检测率：42.86%
- semantic repair 后恢复正确：2；false positive flags：18；repair degradation：6

- silent IDs：1-2126093-3.html.csv_q0, 2-12582968-1.html.csv_q8, 2-10826385-15.html.csv_q9, 2-11051845-5.html.csv_q1, 2-12206617-3.html.csv_q3, 2-1122485-2.html.csv_q3, 2-14101654-10.html.csv_q4, 2-1520559-1.html.csv_q8, 2-12206243-10.html.csv_q1, 1-16168849-1.html.csv_q3, 2-13135264-6.html.csv_q0, 2-11692087-1.html.csv_q1, 2-18332376-1.html.csv_q2, 2-14609295-5.html.csv_q8

