# Codex Master Prompt — 第二篇 TableQA 工作：Multi-Granularity Semantic Logic Auditing

你现在负责我的第二篇 TableQA 论文代码项目。你的任务不是只给方案，而是：**阅读现有代码 → 复用第一篇可用模块 → 在第二篇 Linux 项目中实现 → 调 API 跑实验 → 查看日志 → 分析错误 → 迭代改进 → 保留最佳版本。**

## 0. 最高优先级约束

我在 VS Code 中有两个项目：

1. 第一篇论文完整项目（Windows 本地）：
   `D:\master 1\CS\VS code project\rethinking\wangjie`
2. 第二篇论文项目：当前 VS Code SSH 连接的 Linux workspace。

### 第一篇项目严格只读

允许：读取、搜索、理解、复制普通源码到第二篇项目。

禁止：修改、删除、重命名、格式化、生成缓存、安装依赖到该目录、改变 git 状态、写日志、写输出。

尤其不要复制 `.swp/.swo`、authorized_keys、`.env`、API key、私密配置。

### 只有第二篇 Linux workspace 可以增删改

每次开始先执行：

```bash
pwd
git status
conda activate work
python --version
```

必须确认当前位于第二篇 Linux 项目后才能写文件。

如果当前 SSH Codex 无法访问 Windows 的 `D:\...`，不要假装已经读过。明确说明无法访问，并继续分析第二篇已有代码；需要我手动复制时，只列出明确文件清单。

---

## 1. 数据格式已经确定，禁止重新设计

WTQ / TabFact 的数据加载、预处理、答案规范化和评估方式，第一篇论文已经确定。

**不要重新设计数据 schema，不要修改数据格式，不要为了第二篇方法改写数据集。**

WTQ 当前处理后的数据是“table-level object + questions/answers/ids”的既定 JSON 结构，包含类似字段：

```text
table_id
table.header
table.rows
table.name
questions
answers
ids
transposed_table
sampled_indices
title
row_shuffled_table
row_shuffled_transposed_table
```

示例中一个 table object 可以包含多条 question / answer / id。

要求：

- 优先直接复用第一篇已经验证过的 loader / evaluator / normalization。
- 不改变字段命名与含义。
- 不重新生成 transposed/shuffled 数据。
- 不改变 question-answer-id 对齐关系。
- 以第一篇现有代码定义的“单个评测实例”为准；如果第一篇已经把 table-level JSON flatten 成 question-level sample，则直接复用，禁止自行发明另一套 flatten 规则。
- 第二篇方法只改变**推理与审计流程**，不是数据处理流程。

---

## 2. 研究主线

第一篇研究的是 **Self-Verification Bias**：Reasoner 自己验证自己的错误推理时，容易继续相信原路径，因此使用独立 Analyst 做外部审计。

第二篇自然延伸为：

# Execution-Trust Bias

核心现象：

```text
Python execution success != semantic correctness
```

Python interpreter 能发现 syntax/runtime error，但无法发现大量“逻辑静默失败”：

- 题目要求 sum，代码用了 mean；
- 题目要求 after 2010，代码用了 >= 2010；
- 漏 filter；
- 选错列；
- 用错实体；
- sort direction 错误；
- highest 被实现成 sum；
- 返回类型与问题意图不一致。

这些代码可以完整运行并输出错误答案。

因此第二篇的核心创新必须放在：

# Successful-Execution Semantic Logic Auditing

即：**Python 成功执行之后，继续审计代码的实际语义是否忠实实现原始问题意图。**

不要把创新写成“让 LLM 写 Python”“报错后重试”“多跑几次投票”。

---

## 3. 核心方法结构

目标流程：

```text
Table + Question
      |
      v
Question Intent Extraction
      |
      v
Question Intent IR
      |
      v
Reasoner generates Python
      |
      v
Python execution
      |
 execution success
      |
      v
Code Semantic Recovery
(AST + runtime trace + table grounding + limited LLM semantic matching)
      |
      v
Code Intent IR
      |
      v
Multi-Granularity Semantic Alignment
      |
      +---- PASS ----> candidate answer
      |
      +---- FAIL ----> Semantic Logic Exception
                           |
                           v
                     targeted repair
                           |
                           v
                     re-execute + re-audit
```

重点：真正的审计发生在 **execution success 之后**。

---

## 4. 必须真实实现“三层多粒度审计”

这不是 prompt 中写三个标题，而必须反映在代码结构、IR、日志、异常类型和消融配置中。

### L1 — Global Intent Alignment

检查最终任务与返回目标：

- answer type：number/entity/boolean/string/list
- cardinality：single/multiple
- task type：lookup/aggregation/comparison/ranking/verification
- final return semantics

例如 Question 要 country/entity，代码最后返回 `df['score'].sum()` 数值：

```text
SemanticLogicException
level = GLOBAL
type = ANSWER_TYPE_MISMATCH
```

### L2 — Operator Logic Alignment

比较问题需要的操作链与代码实际执行的操作链：

- filter
- select
- sum/mean/count/max/min
- argmax/argmin
- comparison
- arithmetic/difference
- groupby
- sort/rank/top-k
- temporal filter
- join/merge（需要时）
- negation / boolean composition

例如 Question：`total sales of East after 2022`

Question Intent：

```text
FILTER(region = East)
FILTER(year > 2022)
AGGREGATE(sum, sales)
```

Code Intent 缺少第二个 filter：

```text
MISSING_OPERATOR
missing = FILTER(year > 2022)
```

### L3 — Parameter & Entity Alignment

检查：

- column
- cell/entity value
- constant
- year/date
- comparator
- target column
- sort column/direction
- alias / normalized string

例如：

```text
Question: after 2010
Code: year >= 2010
```

触发：

```text
COMPARATOR_MISMATCH
expected = >
actual = >=
```

L3 不得完全依赖 LLM。优先使用 table schema、真实 cell、类型解析、normalization、exact/fuzzy match；只有规则难以判断的同义/别名关系才调用 LLM。

---

## 5. Intent IR

实现稳定、可序列化的数据结构（dataclass / Pydantic / JSON 均可）。

Question Intent IR 至少支持：

```json
{
  "answer_type": "number",
  "cardinality": "single",
  "task_type": "aggregation",
  "target_columns": ["sales"],
  "filters": [
    {"column": "region", "op": "=", "value": "East"},
    {"column": "year", "op": ">", "value": 2022}
  ],
  "operations": ["filter", "filter", "aggregate"],
  "aggregation": "sum",
  "ranking": null,
  "arithmetic": null
}
```

Code Intent IR 至少支持：

```json
{
  "return_type": "number",
  "used_columns": ["region", "sales"],
  "filters": [
    {"column": "region", "op": "=", "value": "East"}
  ],
  "operations": ["filter", "aggregate"],
  "aggregation": "sum",
  "observed_output_type": "number"
}
```

AuditResult 至少包含：

```json
{
  "passed": false,
  "semantic_exception": true,
  "global": {"passed": true, "errors": []},
  "operator": {
    "passed": false,
    "errors": [{"type": "missing_filter", "expected": "year > 2022"}]
  },
  "parameter": {"passed": true, "errors": []},
  "repair_hint": "Add a strict filter year > 2022 before aggregation.",
  "confidence": 0.96
}
```

三个 level 的结果必须独立保存。

---

## 6. Code Intent IR 必须 Hybrid Recovery

不能只让 LLM 看代码后自由总结。

### A. Python AST / static analysis

优先识别 WTQ / TabFact 常见 Pandas pattern：

- `df[col]`
- `.loc/.iloc`
- boolean filter
- comparison operators
- constants
- `.sum/.mean/.count/.max/.min`
- `.idxmax/.idxmin`
- `.sort_values`
- `.nlargest/.nsmallest`
- `.groupby`
- `.merge`
- arithmetic expressions
- final expression / assigned answer

先覆盖高频模式，不要一开始写巨型 Python parser。

### B. Runtime trace

代码成功执行后保存：

- final object type
- final value
- relevant intermediate shape
- selected columns
- filter 前后 row count（可获得时）
- relevant intermediate values

### C. Table grounding

使用当前实例真实 table：

- schema
- cell values
- numeric/date/string parsing
- entity existence
- alias candidate

### D. Limited LLM semantic matching

只处理规则很难确定的自然语言映射，不得覆盖确定性的 AST/grounding 证据。

---

## 7. 两类异常必须分开

### Execution Failure

SyntaxError / KeyError / TypeError / ImportError 等，走传统执行错误修复。

### Semantic Failure

代码成功执行，但 semantic audit 失败：

```text
Semantic Logic Exception
```

必须记录：

- level
- error type
- expected
- actual
- evidence
- repair hint

默认 `max_semantic_repairs = 2`。

每次 repair 后必须：

```text
re-execute -> re-audit
```

不能只因为 repair 后代码跑通就直接接受。

---

## 8. 第一阶段只证明核心机制，不先做 5+5 投票

先做最干净的控制变量：

```text
B: DeepSeek-V3.2 + Python baseline
C: DeepSeek-V3.2 + Python + Multi-Granularity Semantic Audit
```

保持：

- 同模型
- 同一批 sample IDs
- 同 temperature
- 同 evaluator/normalization
- 同 initial Python-generation prompt（除为审计必要的结构化输出外尽量一致）
- 同 executor

B vs C 是第二篇最核心实验。

核心机制稳定后，再考虑：

```text
A: Text-only
D: Adaptive Text/Python + Semantic Audit
```

Adaptive routing 是外围增强，不得抢走论文主创新。

---

## 9. 测试策略：按用户要求使用测试集的小规模切片

数据格式不改，直接使用第一篇已确定的 test data/evaluator。

允许从测试集中取小规模样本进行工程调试。

### 样本规模逐级增加

优先：

```text
5 -> 10 -> 20 -> 30 -> 最多 40
```

不需要每次都跑到 40。

流程：

1. 先随机/截取 5 条，跑通 baseline 和 audit 全链路；
2. 无严重 bug 后扩到 10；
3. 再到 20；
4. 稳定后跑 30；
5. 只有需要进一步确认时最多跑 40。

### 必须保证公平

同一轮实验中，baseline 与 improved method 使用**完全相同的 sample IDs**。

如果要“随意截取 30 条”，请实现成可复现选择：

- 使用现有 test 实例顺序的一个固定 slice；或
- 固定 `random_seed` 后随机抽样，并把 sample IDs 写进 `selected_samples.json`。

不要每种方法重新随机抽样。

注意：如果第一篇 evaluator 已经有 question-level index，直接复用；不要因为 JSON 是 table-level object 就自行改变评测单位。

### 科研标记

这些反复查看日志后调整的方法属于开发/调试结果，输出目录标记为：

```text
debug_test_slice
```

禁止针对某个 test sample/gold answer 写 hard-code。

最终论文正式结果应在方法/prompt freeze 后重新运行并单独保存；调试 slice 的分数不要冒充未见测试集结果。

---

## 10. 每轮实验必须保存完整日志

Baseline 至少保存：

```text
dataset
sample_id/table_id/question_id
question
gold
model
prompt_version
raw_model_output
generated_python
execution_success
execution_error
execution_output
normalized_prediction
correct
latency
usage (如果 API 返回)
```

Audit method 额外保存：

```text
question_intent_ir
code_intent_ir
global_audit
operator_audit
parameter_audit
semantic_exception
repair_hint
repair_count
repaired_code
repaired_execution_output
final_audit
final_prediction
correct
```

推荐：

```text
outputs/
  <experiment_name>/
    config.json
    selected_samples.json
    predictions.jsonl
    summary.json
    audit_metrics.json
    logs/
    cases/
```

---

## 11. 不要只看 Accuracy

在当前 debug slice 上同时分析：

- baseline accuracy
- audited accuracy
- execution success rate
- execution-success-but-wrong 数量
- semantic exception trigger count
- successful repair count
- false positive：原代码答案正确但 auditor 拒绝
- repair degradation：原本正确，被修错
- detected silent failure
- missed silent failure
- Global / Operator / Parameter 各层触发数量
- API calls / tokens / latency

对每个错误 case 分类：

```text
wrong_column
missing_filter
wrong_comparator
wrong_aggregation
answer_type_mismatch
wrong_entity
wrong_sort/ranking
execution_failure
auditor_false_positive
auditor_false_negative
repair_failure
```

修改代码必须针对“重复错误模式”，禁止 sample-specific patch。

---

## 12. 多粒度消融必须提前留接口

代码必须能配置开关：

```text
Global only
Global + Operator
Global + Operator + Parameter
Full hybrid audit
LLM-only audit
AST/grounding + LLM hybrid audit
```

优先验证：

```text
Global
-> Global + Operator
-> Global + Operator + Parameter
```

并分别记录 accuracy、false positive、silent failure detection、repair success。

---

## 13. Synthetic semantic bug tests

大量 API 前先写最少 12 个单元测试：

1. sum vs mean
2. `>` vs `>=`
3. wrong target column
4. missing filter
5. wrong entity
6. wrong sort direction
7. highest -> sum
8. lowest -> max
9. answer type mismatch
10. single vs multiple cardinality
11. wrong arithmetic sign
12. boolean polarity mismatch

每个测试标注预期异常层：GLOBAL / OPERATOR / PARAMETER。

先 `pytest` 通过，再逐步扩大 API 测试。

---

## 14. API 配置

当前 Linux 环境名：

```text
work
```

模型：

```text
DeepSeek-V3.2
```

服务：

```text
https://ai.paratera.com/
```

OpenAI-compatible base URL：

```text
https://ai.paratera.com/v1/
```

调用方式：

```python
import os
import openai

client = openai.OpenAI(
    api_key=os.environ["PARATERA_API_KEY"],
    base_url=os.getenv("PARATERA_BASE_URL", "https://ai.paratera.com/v1/"),
)

response = client.chat.completions.create(
    model=os.getenv("MODEL_ID", "DeepSeek-V3.2"),
    messages=[
        {"role": "user", "content": "Hello world"}
    ],
)
```

API key 由我单独提供给当前 Codex/服务器环境。

**不要把 key 写进源码、README、JSON、YAML、日志或 git。**

如果 key 当前未设置，告诉我执行：

```bash
export PARATERA_API_KEY='...'
export PARATERA_BASE_URL='https://ai.paratera.com/v1/'
export MODEL_ID='DeepSeek-V3.2'
```

不要打印完整 key。

需要安装 Python package 时使用清华 PyPI 镜像，并且只安装确实缺少的包，不要无理由整体升级环境。

---

## 15. API client 工程要求

统一封装 client，支持：

- timeout
- retry
- exponential backoff
- rate-limit handling
- structured error log
- model/temperature config
- prompt version
- token usage（API 有则记录）
- response cache

Cache key 至少考虑：

```text
dataset
sample_id
model
method
prompt_version
question
table hash
```

绝不缓存 secret。

---

## 16. 优先复用第一篇代码

先只读检查第一篇项目，重点寻找：

- WTQ loader
- TabFact loader
- evaluation
- answer normalization
- table serialization
- prompt helper
- API retry/client
- experiment runner
- logging
- result analysis

有用则复制到第二篇后再修改。

不要推倒重写，不要动已经确定的数据流程。

---

## 17. 第二篇建议模块

根据当前 repo 实际情况复用/调整，不要为了目录漂亮而重复造轮子：

```text
semantic_audit/
    intent_schema.py
    question_intent.py
    code_intent.py
    ast_analyzer.py
    runtime_trace.py
    grounding.py
    auditor.py
    exceptions.py
    repair.py

routing/
    router.py              # 第二阶段再做

evaluation/
    audit_metrics.py
    error_analysis.py
```

---

## 18. 迭代实验流程

### Stage 0 — Repo/环境审计

```text
pwd
git status
work environment
repo inventory
找到 loader/evaluator/API wrapper
只读分析第一篇可复用模块
```

### Stage 1 — 5 条 baseline smoke test

DeepSeek-V3.2 + Python，只确认完整 pipeline 能跑通。

### Stage 2 — 5 条 audit smoke test

加入 Question IR / Code IR / 3-level audit / Semantic Logic Exception / repair。

### Stage 3 — synthetic tests + 修基础 bug

pytest。

### Stage 4 — 10/20 条

重点分析 silent semantic failures 和 false positives。

### Stage 5 — 30 条核心对比

同样本：baseline vs audit。

### Stage 6 — 最多 40 条

只有 30 条不足以判断某个重复错误模式时再扩大。

### Stage 7 — 核心方法稳定后再考虑 Adaptive Text/Python

不要提前做 5 text + 5 Python majority voting。

---

## 19. 每次修改必须记录原因

每个主要 cycle 在实验记录中写：

```text
Problem:
Observed cases:
Hypothesis:
General code change:
Expected effect:
Observed effect:
Regression / false positives:
Decision:
```

禁止：

```python
if sample_id == ...
if question == ...
if gold == ...
```

任何针对特定 test case 的 hard-code 都不允许。

---

## 20. 停止条件

目标不是无限修改，而是得到可解释的最佳通用方法。

最多连续 5 个主要 optimization cycles。

如果连续 2 个 cycle：

- accuracy 无提升；且
- silent failure detection / false positive / repair success 也无改善；

则停止盲目修改，保留当前最佳 commit/config，并输出剩余问题。

如果 improvement 只来自明显增加采样次数/API 调用次数，也要单独指出，不能伪装成 semantic audit 的收益。

---

## 21. 最终必须输出

最后执行：

```bash
git status
git diff --stat
```

然后汇报：

```text
First-project read-only status:
Files reused from paper 1:
Files added:
Files modified:
Data pipeline changes: NONE (除非发现原项目本身无法运行，并需先征得我确认)
Experiments run:
Selected sample IDs:
Best configuration:
WTQ baseline / audited result:
TabFact baseline / audited result:
Execution-success wrong cases:
Semantic failures detected:
Repairs successful:
False positives:
Repair degradations:
Global-level cases:
Operator-level cases:
Parameter-level cases:
Known remaining problems:
Recommended next experiment:
```

最终研究原则：

> 不要把“代码跑通”当成“代码忠实实现了问题”。第二篇工作的核心是发现并修复 successful-but-semantically-wrong 的 TableQA 工具推理，而且必须通过 Global / Operator / Parameter 三个粒度明确定位错误。
