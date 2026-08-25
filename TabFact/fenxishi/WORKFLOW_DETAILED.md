# 三轮迭代 + 并行分析系统 - 窗口与数据流详解

## 📋 系统概览

本系统使用 **3轮迭代 × 3条路径** = 最多9条推理路径，通过并行分析和多级投票机制得到最终答案。

---

## 🔄 完整流程详解

### 第一轮：初始推理 (Path 1, 2, 3)

#### 基座模型生成

```python
# Path 1
窗口状态: ✅ 全新窗口（无历史对话）
输入内容:
  - 表格 (Markdown格式)
  - 问题文本
  - CoT Prompt (引导逐步推理)
  
配置:
  temperature = 0.8      # 需要多样性
  max_tokens = 10240     # 足够生成完整推理
  
输出示例:
  Step 1: From the table, I identify rows where Year > 2000...
  Step 2: Calculate the sum: 10 + 20 + 30 = 60...
  Step 3: Compare with threshold...
  Final Answer: 60

# Path 2, Path 3
与 Path 1 完全独立，相同的输入，不同的随机种子
每个都是全新窗口，互不知道对方存在
```

#### 分析师模型并行分析

```python
# 分析 Path 1
窗口状态: ✅ 独立窗口A（无历史，无其他路径信息）
输入内容:
  - 表格
  - 问题
  - Path 1的完整推理文本
  - 验证Prompt（要求检查每步是否正确）
  
配置:
  temperature = 0.1      # 分析需要确定性
  max_tokens = 1024      # 判断+简短分析即可
  repetition_penalty = 1.5  # 避免重复废话
  
分析任务:
  1. ❌ 不自己计算答案
  2. ✅ 逐步验证Path 1中的每个声明
  3. ✅ 对照表格检查数据提取是否正确
  4. ✅ 检查数学计算是否正确
  5. ✅ 检查答案格式是否符合问题要求

输出格式A（全对）:
  **Overall Judgment: CORRECT**
  
  **Verification Summary:**
  I verified Step 1 correctly extracted rows 2-5. 
  Step 2's calculation is correct. 
  Final answer matches the question.

输出格式B（有错）:
  **Overall Judgment: INCORRECT**
  
  **First Error Step:** Step 2
  **Error Claim:** "10 + 20 + 30 = 100"
  **Table Shows:** Values are 10, 20, 30
  **Error Type:** Calculation error
  **Correction Suggestion:** Should be 60

# 分析 Path 2, Path 3
与分析 Path 1 完全独立
每个都是独立窗口，互不知道其他分析结果
```

#### 代码层面汇总

```python
# Python代码执行（非模型推理）
收集3个分析结果:
  Path 1: is_correct = True/False
  Path 2: is_correct = True/False
  Path 3: is_correct = True/False

检查提前退出条件:
  IF all(is_correct) AND all_answers_same:
    返回任意一条路径的答案
    结束推理
  ELSE:
    提取错误信息
    进入第二轮
```

---

### 第二轮：错误修正 (Path 4, 5, 6)

#### 错误信息提取

```python
从第一轮的分析结果中提取:
  error_infos = []
  
  FOR each analysis in [Path1_analysis, Path2_analysis, Path3_analysis]:
    IF analysis.is_correct == False:
      error_infos.append({
        'path_id': analysis.path_id,
        'first_error_step': "Step 2",  # 只记录第一个错误
        'error_analysis': "Error Claim: ... Table Shows: ..."  # 限制300字符
      })

# 最多保留3个错误（如果Path 1,2,3都错）
error_infos = error_infos[:3]
```

#### 基座模型生成（带错误提示）

```python
# Path 4
窗口状态: ✅ 全新窗口（无历史）
输入内容:
  [原始输入]
  - 表格
  - 问题  
  - CoT Prompt
  
  + [新增错误提示部分]
  + ═══════════════════════════
  + **PREVIOUS ERRORS TO AVOID:**
  + ═══════════════════════════
  + 
  + Error 1 (from Path 1):
  +   - First Error at: Step 2
  +   - Analysis: The reasoning claimed "10+20+30=100"
  +               but the correct sum is 60...
  + 
  + Error 2 (from Path 3):
  +   - First Error at: Step 1
  +   - Analysis: Extracted wrong rows...
  + 
  + ═══════════════════════════
  + **INSTRUCTIONS:**
  + 1. Carefully read the table
  + 2. Pay attention to errors above
  + 3. Generate NEW reasoning avoiding mistakes
  + 4. Verify each step
  + 5. End with Final Answer
  + ═══════════════════════════

配置: 同第一轮

关键点:
  ✅ 能看到: 错误在哪一步，错在哪里
  ❌ 看不到: 之前的完整推理路径（避免复制粘贴）

# Path 5, Path 6
相同的输入（相同的错误提示）
独立的窗口
```

#### 分析师模型并行分析

```python
# 分析 Path 4, 5, 6
窗口状态: ✅ 全新的3个独立窗口
  注意: 分析师不保留第一轮的分析历史！
  
输入: 表格 + 问题 + Path X推理 + 验证Prompt
配置: 同第一轮
输出: 同第一轮格式
```

#### 代码层面汇总

```python
检查提前退出条件:
  IF 有正确路径 AND 答案唯一:
    返回该答案
    结束推理
  ELSE:
    提取第二轮的错误信息
    进入第三轮
```

---

### 第三轮：最终推理 (Path 7, 8, 9)

流程与第二轮相同，但是:
- 错误提示来自第二轮的分析结果
- 分析师仍然是全新的独立窗口

---

### 最终决策

```python
# Python代码执行（非模型推理）

优先级 1: 第三轮有正确路径
  IF Round3中有被判CORRECT的路径:
    IF 只有1个正确:
      返回该路径答案
    ELSE:  # 多个正确
      在这些正确路径的答案中投票
      返回票数最多的（相同票数取最近的）

优先级 2: 前两轮有正确路径
  ELIF Round1或Round2中有被判CORRECT的路径:
    在这些"曾被判正确"的答案中投票
    返回票数最多的（相同票数取最近的）

优先级 3: 全部错误，强制投票
  ELSE:  # 9条路径全被判INCORRECT
    统计所有9个答案的词频
    返回出现次数最多的答案
    （票数相同取最近出现的）
```

---

## 📊 数据流总结

### 基座模型看到什么？

| 轮次 | 输入内容 | 历史信息 |
|------|---------|---------|
| 第1轮 | 表格 + 问题 + CoT Prompt | ❌ 无 |
| 第2轮 | 表格 + 问题 + CoT Prompt<br>+ 第1轮错误提示（最多3个错误） | ✅ 第1轮错误的**第一个错误步骤**和**错误分析**<br>❌ 看不到完整推理路径 |
| 第3轮 | 表格 + 问题 + CoT Prompt<br>+ 第2轮错误提示 | ✅ 第2轮错误的**第一个错误步骤**和**错误分析**<br>❌ 看不到第1轮信息<br>❌ 看不到完整推理路径 |

### 分析师模型看到什么？

| 轮次 | 输入内容 | 历史信息 |
|------|---------|---------|
| 任何轮 | 表格 + 问题 + 单条路径推理 + 验证Prompt | ❌ 无历史<br>❌ 不知道其他路径<br>❌ 不知道之前轮次的分析 |

---

## ⚙️ Token配置详解

### 基座模型（生成推理）

```python
max_tokens = 10240  # 当前设置

为什么这个值？
  ✅ 完整推理通常需要 2000-6000 tokens
  ✅ 留有余量避免截断
  ✅ 不至于太大浪费

建议范围: 8192 - 12288
  - 如果经常被截断 → 增加到 12288
  - 如果很少超过6000 → 可降到 8192 节省成本
```

### 分析师模型（验证分析）

```python
max_tokens = 1024  # 当前设置

为什么这个值？
  ✅ 判断 + 错误分析通常 200-600 tokens
  ✅ 留有余量
  ✅ 控制分析不要过于冗长

建议范围: 512 - 1536
  - 如果分析经常不完整 → 增加到 1536
  - 如果都很简短 → 可降到 768 节省成本
```

### Temperature设置

```python
基座模型:
  temperature = 0.8  # 需要多样性产生不同推理路径
  建议: 0.7 - 0.9
  
分析师模型:
  temperature = 0.1  # 验证应该确定，减少随机性
  建议: 0.0 - 0.2
```

### 估算Token消耗

```
单个问题完整流程（最坏情况，9条路径）:

基座模型输入:
  - 表格: ~1000 tokens (取决于表格大小)
  - 问题: ~50 tokens
  - CoT Prompt: ~200 tokens
  - 错误提示（第2,3轮）: ~500 tokens
  - 小计: ~1750 tokens/path × 9 paths = ~15,750 tokens

基座模型输出:
  - 推理: ~4000 tokens/path × 9 paths = ~36,000 tokens

分析师输入:
  - 表格 + 问题 + Prompt: ~1200 tokens
  - 路径推理: ~4000 tokens
  - 小计: ~5200 tokens/path × 9 paths = ~46,800 tokens

分析师输出:
  - 分析: ~400 tokens/path × 9 paths = ~3,600 tokens

总计（最坏情况）:
  输入: ~62,550 tokens
  输出: ~39,600 tokens
  合计: ~102,150 tokens/question

如果第1轮就成功（最好情况，3条路径）:
  输入: ~19,050 tokens
  输出: ~13,200 tokens  
  合计: ~32,250 tokens/question
```

---

## 🎯 优化建议

### 1. 错误信息传递优化 ✅ (已实现)

```python
# 限制每个错误分析最多300字符
if len(error_analysis) > 300:
    error_analysis = error_analysis[:300] + "..."

# 最多传递3个错误
error_infos = error_infos[:3]
```

### 2. 可选优化：提前截断

如果第2轮已经有2/3正确且答案一致，可以考虑提前结束：

```python
# 在第2轮后增加检查
if round_num == 2:
    correct_count = sum(1 for a in analyses if a.is_correct)
    if correct_count >= 2:
        # 检查这2个正确的答案是否一致
        correct_answers = [p.answer for p, a in zip(paths, analyses) if a.is_correct]
        if len(set(normalize_answer(a) for a in correct_answers)) == 1:
            # 提前结束
            return ...
```

### 3. 可选优化：动态调整温度

第3轮如果前两轮错误率高，可以降低温度提高稳定性：

```python
if round_num == 3:
    error_rate = (9 - stats["correct_paths"]) / 6  # 前6条的错误率
    if error_rate > 0.8:  # 80%以上都错
        temperature = 0.5  # 降低温度，提高稳定性
```

---

## ❓ 常见问题

### Q1: 为什么基座模型每次都是新窗口？
A: 
- ✅ 保证推理独立性，避免路径间污染
- ✅ 增加答案多样性，提高投票效果
- ✅ 每条路径都是"平等"的，无先后影响

### Q2: 为什么分析师每次都是新窗口？
A:
- ✅ 降低认知负载，专注分析单条路径
- ✅ 避免"锚定效应"（被之前的分析影响）
- ✅ 并行分析更公平

### Q3: 为什么只传第一个错误？
A:
- ✅ 第一个错误通常是根本原因
- ✅ 减少token消耗
- ✅ 避免信息过载
- ✅ 修正第一个错误后，后续错误可能自然消失

### Q4: 分析师会自己计算答案吗？
A: 
- ❌ 不会！分析师只验证基座模型的声明
- ✅ 只检查：数据提取是否正确、计算是否正确、逻辑是否合理
- ✅ 不独立求解问题

### Q5: 如果9条路径都错怎么办？
A:
- 使用优先级3：统计所有9个答案的词频
- 选出现次数最多的答案
- 这是一个"兜底"机制，依赖"大多数正确"原则

---

## 📝 总结

**窗口策略：全部独立，无历史保留**
- 基座：每条Path都是新窗口
- 分析师：每条Path都是新窗口，每轮都重置

**数据传递：精简高效**
- 第1轮→第2轮：只传错误步骤+分析（最多3个，每个300字符）
- 第2轮→第3轮：同上
- 不传递完整推理路径

**Token设置：合理平衡**
- 基座：10240（足够完整推理）
- 分析师：1024（判断+简短分析）
- Temperature: 0.8 vs 0.1（多样性 vs 确定性）

**三级决策：优先最新，兜底投票**
- P1: 第3轮正确 > P2: 历史正确投票 > P3: 全部投票
