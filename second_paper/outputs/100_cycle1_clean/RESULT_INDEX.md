# 实验结果索引

- 根目录：`outputs/100_cycle1_clean`
- 区间：由 `selected_samples.json` 确定；当前实验默认是 flat index `[0, 100)`，end 为开区间。
- 详细逐条结果：各方法目录下的 `result.jsonl`；逐条第一篇兼容日志：`log/<flat_index>.txt`。

## WTQ

### baseline
- accuracy: 67.00%; execution success: 82.00%; execution-success-but-wrong: 22; API calls: 118
- 文件：`outputs/100_cycle1_clean/wtq/baseline/summary.json`、`outputs/100_cycle1_clean/wtq/baseline/result.jsonl`、`outputs/100_cycle1_clean/wtq/baseline/selected_samples.json`

### audit
- accuracy: 66.00%; execution success: 85.00%; execution-success-but-wrong: 23; API calls: 174
- semantic exceptions: 54; layers: {'OPERATOR': 54, 'PARAMETER': 35, 'GLOBAL': 28}; error types: {'MISSING_OPERATOR': 23, 'MISSING_AGGREGATION': 10, 'WRONG_TARGET_COLUMN': 31, 'WRONG_AGGREGATION': 9, 'ANSWER_TYPE_MISMATCH': 10, 'CARDINALITY_MISMATCH': 6, 'FINAL_RETURN_SEMANTICS_MISMATCH': 12, 'MISSING_RANKING': 12, 'UNKNOWN_COLUMN': 4}
- 文件：`outputs/100_cycle1_clean/wtq/audit/summary.json`、`outputs/100_cycle1_clean/wtq/audit/result.jsonl`、`outputs/100_cycle1_clean/wtq/audit/selected_samples.json`

### joint
- accuracy: 83.00%; execution success: 85.00%; execution-success-but-wrong: 14; API calls: 415
- 文件：`outputs/100_cycle1_clean/wtq/joint/summary.json`、`outputs/100_cycle1_clean/wtq/joint/result.jsonl`、`outputs/100_cycle1_clean/wtq/joint/selected_samples.json`

### 配对审计统计
- baseline 静默错误候选：22
- auditor 检测到：13；检测率：59.09%
- semantic repair 后恢复正确：1；false positive flags：38；repair degradation：8

- silent IDs：nu-0, nu-1, nu-648, nu-695, nu-2881, nu-2522, nu-4169, nu-2555, nu-14, nu-597, nu-2675, nu-1202, nu-1911, nu-4053, nu-1406, nu-263, nu-498, nu-3430, nu-3458, nu-2913, nu-2147, nu-2201

## TABFACT

### baseline
- accuracy: 87.00%; execution success: 98.00%; execution-success-but-wrong: 12; API calls: 102
- 文件：`outputs/100_cycle1_clean/tabfact/baseline/summary.json`、`outputs/100_cycle1_clean/tabfact/baseline/result.jsonl`、`outputs/100_cycle1_clean/tabfact/baseline/selected_samples.json`

### audit
- accuracy: 83.00%; execution success: 97.00%; execution-success-but-wrong: 15; API calls: 138
- semantic exceptions: 28; layers: {'OPERATOR': 20, 'PARAMETER': 22, 'GLOBAL': 3}; error types: {'MISSING_RANKING': 8, 'WRONG_TARGET_COLUMN': 7, 'UNKNOWN_COLUMN': 5, 'WRONG_AGGREGATION': 2, 'BOOLEAN_POLARITY_ERROR': 2, 'MISSING_FILTER': 1, 'MISSING_OPERATOR': 6, 'MISSING_FILTER_PARAMETER': 4, 'MISSING_AGGREGATION': 3, 'WRONG_ENTITY': 3, 'ANSWER_TYPE_MISMATCH': 3, 'WRONG_SORT_COLUMN': 1}
- 文件：`outputs/100_cycle1_clean/tabfact/audit/summary.json`、`outputs/100_cycle1_clean/tabfact/audit/result.jsonl`、`outputs/100_cycle1_clean/tabfact/audit/selected_samples.json`

### joint
- accuracy: 92.00%; execution success: 99.00%; execution-success-but-wrong: 8; API calls: 401
- 文件：`outputs/100_cycle1_clean/tabfact/joint/summary.json`、`outputs/100_cycle1_clean/tabfact/joint/result.jsonl`、`outputs/100_cycle1_clean/tabfact/joint/selected_samples.json`

### 配对审计统计
- baseline 静默错误候选：12
- auditor 检测到：4；检测率：33.33%
- semantic repair 后恢复正确：2；false positive flags：24；repair degradation：8

- silent IDs：2-12582968-1.html.csv_q8, 2-10826385-15.html.csv_q9, 2-11051845-5.html.csv_q1, 2-12206617-3.html.csv_q3, 2-12206617-3.html.csv_q8, 2-14101654-10.html.csv_q4, 1-16168849-1.html.csv_q3, 2-13135264-6.html.csv_q1, 2-13135264-6.html.csv_q0, 2-10167122-1.html.csv_q4, 2-16369528-1.html.csv_q3, 2-18332376-1.html.csv_q2

