# 上周四 TableQA 实验总结

## 结论先说

上周四（2026-08-20）最终冻结的配置是“受保护的 DP baseline + 选择性 Python/优化 DP/盲验证器证据”的联合推理：

| 数据集 | DP baseline | 最终 guarded joint | 绝对提升 | 相对正确数变化 |
|---|---:|---:|---:|---:|
| WTQ | 130/150 = 86.67% | 137/150 = **91.33%** | **+4.67 个百分点** | +7 |
| TabFact | 142/150 = 94.67% | 145/150 = **96.67%** | **+2.00 个百分点** | +3 |

单独优化的 Python 也有提升，但单独优化 DP 并不总是提升，因此最后没有采用“所有候选简单投票”，而是让 DP baseline 作为保护性主答案，只有通过严格规则的候选才可以替换它。

最终文件位于远程：

```text
/root/shared-nvme/wangjie/second_paper/outputs/150_final_v1/
```

本轮仅做了远程只读核实，并生成本地说明文档；没有修改远程代码。

## 一、上周四实际经历的思路和迭代

### 1. 初始基线和早期优化

先把问题拆成 Python 路线和 DP（文本直接推理）路线：

- Python baseline：模型生成 Python/Pandas，受控执行，直接评测，不做语义审计。
- DP baseline：模型直接根据问题和表格生成文本答案，不使用 Python。
- 优化 Python：增加问题意图、代码意图、运行时轨迹、表格 grounding，以及三层语义审计和有限修复。
- 优化 DP：对文本推理增加窄范围、类型感知的规则。

早期 100 条结果曾出现“audit 下降”和“joint 依赖文本候选”的问题，因此没有把旧的全局投票方案当作最终方法。具体早期结果为：

| 100 条阶段 | WTQ | TabFact | 处理结论 |
|---|---:|---:|---|
| 早期 baseline / audit / joint | 67% / 66% / 83% | 87% / 83% / 92% | audit 下降，不能作为最终方案 |
| 第二轮 cycle2 | 65% / 70% / 83% | 84% / 81% / 91% | WTQ audit 局部改善，但 joint 未改善；TabFact 下降 |
| 最终开发集 v6 | 49% Python / 87% DP / 93% joint | 73% Python / 93% DP / 97% joint | 采用保护性联合方案 |

### 2. 失败方案和为什么放弃

上周四没有继续盲目增加投票次数，而是逐个检查日志和错误样本，放弃了以下方案：

1. 简单多数投票：不同答案类型可以被错误地当作同一答案，且一个错误候选可能通过重复投票放大。
2. 候选可见的 verifier：验证器看到候选答案后容易被候选带偏，不能证明独立判断。
3. 全局语义提示词：对 WTQ/TabFact 的类型差异不够敏感，产生误报和 repair degradation。
4. 统一的“优化 DP 总是覆盖 baseline”：TabFact 上优化 DP 原始结果比 DP baseline 低，因此不能无条件替换。

### 3. 最终采用的 v6/frozen 方案

最终方案的核心是“非对称、受保护的联合选择”：

```text
原始顺序数据
   ├─ Python baseline
   ├─ DP baseline（主答案，默认保留）
   ├─ 优化 Python：意图/代码/运行时/表格三层审计 + 有限修复
   ├─ 优化 DP：窄范围语义规则
   └─ gold-blind 双验证器：看不到 gold，也不直接看到候选答案
             ↓
      guarded joint 选择器
      只有满足安全、类型、grounding、跨模态一致性规则时才替换 DP baseline
```

最终选择路线统计如下：

| 数据集 | DP baseline 保留 | blind verifier | optimized semantic consensus | 其他严格路线 |
|---|---:|---:|---:|---:|
| WTQ | 140 | 4 | 4 | duration 2 |
| TabFact | 145 | 3 | 2 | 无 |

选择过程中不读取 gold；最终 summary 也明确记录了 `gold_excluded_from_selection=true`。

## 二、如何运行实验

### 数据和顺序

使用第一篇项目的数据适配器和评测器，远程只读数据路径为：

```text
/root/shared-nvme/wangjie/Rethinking/data/wtq.json
/root/shared-nvme/wangjie/TabFact/data/tabfact.json
```

运行使用 adapter 的原始 `sampled_indices` 顺序和 `iter_range(0, N)`。最终 150 条均满足：

- `idx` 严格为 0 到 149；
- WTQ/TabFact 的 sample ID 与 adapter 的原始顺序逐条一致；
- `result.jsonl` 前两个字段为 `idx`、`answer`；
- 后续才写入 question、table、诊断字段等。

因此之前出现的 WTQ 第一条变成 `nu-2928` 的顺序问题，在最终结果中已经修正。

### 方法运行顺序

最终脚本为：

```bash
cd /root/shared-nvme/wangjie/second_paper
conda activate work
bash scripts/run_150_frozen.sh wtq
bash scripts/run_150_frozen.sh tabfact
```

该脚本实际依次运行：

1. Python baseline；
2. DP baseline；
3. 优化 Python；
4. 优化 DP raw；
5. 以 DP baseline 为首选的 guarded optimized DP；
6. gold-blind 双验证器；
7. guarded joint。

`run_150_frozen.sh` 会把前 100 条开发集的缓存复制到 final 目录，减少重复 API 调用。因此最终 150 条是“原始顺序前 150 条的冻结评估”，但前 100 条与开发迭代存在缓存/调参关系，不能称为完全独立测试集。

## 三、上周四最终结果和 baseline 对比

### WTQ 前 150 条

| 方法 | 正确数 / 150 | 准确率 | 相对 Python baseline | 相对 DP baseline |
|---|---:|---:|---:|---:|
| Python baseline | 86 | 57.33% | — | — |
| 优化 Python | 99 | 66.00% | **+13，+8.67 个百分点** | — |
| DP baseline | 130 | 86.67% | — | — |
| 优化 DP raw | 132 | 88.00% | — | **+2，+1.33 个百分点** |
| 优化 DP guarded | 131 | 87.33% | — | **+1，+0.67 个百分点** |
| 最终 guarded joint | **137** | **91.33%** | **+51，+34.00 个百分点** | **+7，+4.67 个百分点** |

### TabFact 前 150 条

| 方法 | 正确数 / 150 | 准确率 | 相对 Python baseline | 相对 DP baseline |
|---|---:|---:|---:|---:|
| Python baseline | 116 | 77.33% | — | — |
| 优化 Python | 124 | 82.67% | **+8，+5.33 个百分点** | — |
| DP baseline | 142 | 94.67% | — | — |
| 优化 DP raw | 140 | 93.33% | — | **-2，-1.33 个百分点** |
| 优化 DP guarded | 142 | 94.67% | — | **0，0 个百分点** |
| 最终 guarded joint | **145** | **96.67%** | **+29，+19.33 个百分点** | **+3，+2.00 个百分点** |

这组结果说明：你的“最终目标是 DP + 优化 Python 联合推理”是正确方向，但联合推理必须有保护机制；TabFact 的优化 DP raw 下降正好证明了不能把优化模块无条件当成更可靠答案。

## 四、项目文件用途

### 必须保留（最终运行和复现实验需要）

| 路径 | 用途 |
|---|---|
| `paper1_runtime/adapters.py` | WTQ/TabFact 数据适配、原始顺序、字段契约和评测入口 |
| `paper1_runtime/prompts.py` | Python、文本、选择、修复提示词 |
| `semantic_audit/intent_schema.py` | Question/Code Intent IR 和审计结构 |
| `semantic_audit/question_intent.py` | 问题意图解析 |
| `semantic_audit/ast_analyzer.py` | AST、Pandas 操作和结果分析 |
| `semantic_audit/runtime_trace.py` | 运行时证据和语义警告 |
| `semantic_audit/grounding.py` | 列、实体、数值、日期、字符串 grounding |
| `semantic_audit/auditor.py` | Global/Operator/Parameter 三层审计 |
| `semantic_audit/llm_audit.py`、`semantic_audit/prompts.py` | LLM 审计和相关提示词 |
| `semantic_audit/api_client.py` | API 调用和缓存接口 |
| `paper1_reuse/` | 第一篇项目的数据读取、表格序列化、执行器和评测复用代码 |
| `scripts/run_paper1.py` | 主实验执行入口 |
| `scripts/run_double_verifier.py` | gold-blind 双验证器 |
| `scripts/run_guarded_joint.py` | 最终受保护联合选择 |
| `scripts/run_conservative_fusion.py` | DP baseline 与优化 DP 的保护性融合 |
| `scripts/run_150_frozen.sh`、`scripts/_common_50.sh` | 最终 150 条冻结运行入口 |
| `evaluation/metrics.py`、`evaluation/paper1_compat.py` | 评测和兼容指标 |
| `tests/test_semantic_audit.py` | 12 个语义错误单元测试 |
| `outputs/150_final_v1/` | 最终 150 条结果、日志、summary 和 frozen code 快照 |
| `outputs/100_dev_v6/` | 最终开发集 v6 结果和缓存 |
| `README_SECOND_PAPER.md`、`EXPERIMENT_LOG.md`、`项目说明_中文.md` | 项目说明、历史实验和运行协议 |

### 可先压缩归档，再按需要手动删除

这些不是最终 `run_150_frozen.sh` 的必要输入，但对复现实验历史有价值：

- `outputs/100_cycle1*`、`outputs/100_cycle2*`：早期 audit/joint 迭代；
- `outputs/100_dev_v1` 到 `outputs/100_dev_v5`：最终 v6 前的 baseline、verifier、全局规则和失败实验；
- `outputs/20_recheck*`、`outputs/30*`、`outputs/50*`、`outputs/100_probe*`、`outputs/wrapper_probe`：小规模 smoke/recheck/历史试验；
- `scripts/run_20_three_way.sh`、`run_30_joint_sweep.sh`、`run_50_all.sh`、旧的 0_50/0_100 wrapper，以及 `run_consensus_joint.py`、`run_joint_from_results.py`、`run_structured_dp.py`：旧运行入口或已淘汰选择器；
- `.backups/`：周四修改过程的回滚副本；
- `.pytest_cache/`、各目录下的 `__pycache__/`：可重建缓存。

### 明确的临时文件候选

`.incoming/` 下的 `_remote_*.py` 是代码传输/临时副本，不是最终脚本的 import 依赖；如果你已经不需要保留传输痕迹，可以手动删除或先压缩归档。

`.secrets/paratera.env` 不是运行代码，但包含服务器 API 配置，不能当普通垃圾文件删除；如不再运行实验，应由你单独按凭据管理方式处理，不能把内容复制到报告或 Git。

建议顺序：先压缩 `outputs` 历史目录和 `.backups`，确认论文复现实验不再需要后再删除；本轮没有替你删除任何文件。

## 五、论文可主张的创新点

以下是可以形成论文方法贡献的候选表述，建议写成“方法贡献”，不要在未完成文献检索前直接写成“首次”：

1. **Execution-Trust Bias 的显式诊断。** 将“Python 成功执行”与“问题语义被正确实现”区分开，研究表格问答中静默错误，而不是只统计运行异常。
2. **问题意图—代码意图—运行结果的多粒度语义对齐。** 用 Question Intent IR 表示任务类型、答案类型、基数、目标列、过滤、聚合、排序和布尔极性；再从 AST/Pandas、运行时结果和表格真实值构造 Code Intent IR，并在 Global、Operator、Parameter 三个粒度审计。
3. **受控 Generate–Audit–Repair 闭环。** 审计发现问题后只给出定向 repair hint，限制修复次数，并要求修复代码重新执行且重新审计通过，避免“能运行就接受”。
4. **保护性非对称联合推理。** 不假设优化模块一定优于 baseline，以 DP baseline 为主答案，仅在类型安全、表格 grounding、跨模态一致性或独立盲验证满足条件时替换；这使方法能抵抗 TabFact 上优化 DP raw 的回退。
5. **gold-blind 的独立验证。** 盲验证器不读取 gold，也不直接看到候选答案，用于降低候选可见造成的确认偏差；选择逻辑保留 `gold_excluded_from_selection=true` 的审计证据。
6. **数据集感知的窄语义规则。** 例如持续时间单位、竞赛行数、电影奖项计数、TabFact 类别对比较和体育 score/points 别名等，避免用一个宽泛规则覆盖不同问题类型。
7. **可复现的证据链。** 固定原始 sample 顺序、`idx`/`answer` 字段顺序、逐条日志、缓存、配置、审计字段和 frozen code 哈希，使错误选择可以回溯到问题、表格、代码、运行结果和选择路线。

论文中应同时报告“检测率、修复成功率、误报率、repair degradation、API 成本和准确率”，不能只报告最后的 joint accuracy。

## 六、需要诚实写入论文的限制

- 最终 150 条包含前 100 条开发集缓存，不能当作完全独立测试集；下一步应冻结规则后在未参与调参的数据上复验。
- 当前只有一个主要模型/API 配置，尚未证明跨模型泛化。
- 150 条结果没有统计显著性检验，也没有完整 WTQ/TabFact 测试集结果。
- WTQ 仍有并列、时间参考、答案基数和实体解释歧义；TabFact 仍存在个别表格/标注不一致样本。
- 早期审计出现过 false positive 和 repair degradation；因此最终方案强调保护性选择，而不是声称所有审计修复都有效。
- 远程项目中未核实到用户此前提到的 `CODEX_MASTER_PROMPT.md`、`RESEARCH_METHOD_SPEC.md`、`EXPERIMENT_PROTOCOL.md`，本报告没有假设这些文件内容。

## 七、最终建议的论文实验表

正式论文建议至少保留以下消融：

1. Python baseline；
2. 优化 Python（完整三层审计和有限修复）；
3. DP baseline；
4. 优化 DP raw；
5. 优化 DP guarded；
6. blind verifier；
7. 最终 guarded joint。

同时在未参与开发的测试划分上报告：准确率、执行成功率、静默错误数、审计检测率、修复恢复数、误报数、repair degradation、API 调用数和 token 成本。这样才能证明提升来自“语义审计 + 保护性联合选择”，而不是来自更多 API 调用或样本缓存。

## 八、核实状态

- 远程 SSH：已连接到 `bingxing`，工作目录为 `/root/shared-nvme/wangjie/second_paper`。
- 远程代码：本轮未修改。
- 最终结果：WTQ 和 TabFact 前 150 条的结果文件、summary、顺序和字段位置已核对。
- 上周四时间证据：`outputs/100_dev_v6`、`outputs/150_final_v1/frozen_code/SHA256SUMS.txt` 及最终 summary 均为 2026-08-20 生成。
- 未验证项：完整测试集、跨模型泛化、统计显著性和 PDF 渲染视觉检查。
