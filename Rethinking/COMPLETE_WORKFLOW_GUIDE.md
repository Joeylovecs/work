# 完整实验运行指南

## 目录
1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [WTQ数据集完整流程](#wtq数据集完整流程)
4. [TabFact数据集完整流程](#tabfact数据集完整流程)
5. [路径隔离说明](#路径隔离说明)
6. [常见问题](#常见问题)

---

## 项目概述

### 研究思路回顾
----------------
```
步骤1: DeepSeek API 跑训练集 (0-3950) → 获得高质量推理步骤
步骤2: Qwen3-8B 跑训练集 (0-3950) → 获得基座模型推理步骤  
步骤3: 找出 DeepSeek正确 + Qwen错误 的样本
步骤4: 调用DeepSeek分析步骤3的推理步骤原因
步骤5: 构建微调数据集 (instruction: 表格+问题+推理步骤, response: 分析解释)
步骤6: LoRA微调Qwen3-8B → 得到分析师模型
步骤7-11: 三轮迭代推理验证 (基座模型 + 分析师模型)
```

### 目录结构

```
wangjie/
├── Rethinking Tabular DeepSeek new/   # WTQ数据集项目 (已完成)
│   ├── data/wtq.json                  # WTQ测试集
│   ├── data/train.json                # 训练集
│   ├── buzhou/                        # 步骤脚本
│   ├── fenxishi/                      # 分析师模型验证
│   └── output/                        # 输出结果
│
├── TabFact/                           # TabFact数据集项目 (新建)
│   ├── data/tabfact.json              # TabFact测试集
│   ├── buzhou/                        # 步骤脚本
│   ├── fenxishi/                      # 分析师模型验证
│   └── output/                        # 输出结果
│
├── Qwen3-8B/                          # Qwen3-8B模型权重
└── Llama-3.1-8B-Instruct/             # Llama模型权重
```

---

## 环境配置

```bash
# 激活conda环境
conda activate rethinking

# 设置环境变量
export LD_LIBRARY_PATH="/data/amax/home/E23101002/.conda/envs/rethinking/lib:$LD_LIBRARY_PATH"
export PATH="/data/amax/home/E23101002/.conda/envs/rethinking/bin:$PATH"
```

---

## WTQ数据集完整流程

### 阶段一：生成推理步骤 (对应研究步骤1-2)

#### 1.1 DeepSeek生成推理步骤

```bash 

cd /path/to/Rethinking\ Tabular\ DeepSeek\ new

# 运行DeepSeek API (0-3950)
bash scripts/all_dp.sh

# 或者直接运行:
python run_cot.py \
    --model DeepSeek-V3.1 --long_model DeepSeek-V3.1 \
    --provider openai --system "You are a helpful assistant" \
    --dataset train --sub_sample False \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 3950 --self_consistency 1 --temperature 0.8 \
    --log_dir output/train/dp --cache_dir cache/deepseek-V3.1
```

**输出文件**: `output/train/dp/result.jsonl`

#### 1.2 Qwen3-8B生成推理步骤

```bash
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new

# 运行Qwen3-8B (0-3950)
bash buzhou/qwen/all_dp_qwen.sh

# 或者直接运行:
python run_cot.py \
    --model "/path/to/Qwen3-8B" --long_model "/path/to/Qwen3-8B" \
    --provider huggingface --system "You are a helpful assistant..." \
    --dataset train --sub_sample False \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 3950 --self_consistency 1 --temperature 0.8 \
    --log_dir output/qwen3_8b_train --cache_dir cache/qwen3_8b
```

**输出文件**: `output/qwen3_8b_train/result.jsonl`

### 阶段二：分类正确/错误样本 (对应研究步骤3)

```bash
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new

# 分类DeepSeek结果
python scripts/split_correct_incorrect.py output/train/dp/result.jsonl
# 输出: output/train/dp/correct.jsonl, output/train/dp/incorrect.jsonl

# 分类Qwen结果
python scripts/split_correct_incorrect.py output/qwen3_8b_train/result.jsonl
# 输出: output/qwen3_8b_train/correct.jsonl, output/qwen3_8b_train/incorrect.jsonl
```

### 阶段三：构建训练数据集

```bash
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new

# 构建基于DeepSeek正确+Qwen错误的训练集
bash scripts/build_training_quick.sh

# 或者手动运行:
python scripts/build_training_dataset.py \
    --deepseek_correct "output/train/dp/correct.jsonl" \
    --qwen_incorrect "output/qwen3_8b_train/incorrect.jsonl" \
    --output_dir "training_data" \
    --positive_name "positive_samples.jsonl" \
    --negative_name "negative_samples.jsonl"
```

**输出文件**:
- `training_data/positive_samples.jsonl` (正向样本 - DeepSeek正确答案)
- `training_data/negative_samples.jsonl` (负向样本 - Qwen错误答案)

### 阶段四：DeepSeek分析推理步骤 (对应研究步骤4)

```bash
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new/buzhou/training_data_zhengti

# 标准化样本格式
python ../scripts/standardize_training_data_new.py

# 调用DeepSeek分析推理步骤
bash run_analysis.sh 0 2000    # 处理0-1999
bash run_analysis.sh 2000 4000 # 处理2000-3999
```

**输出文件**: `analysis_output_unified/` 目录下的分析结果

### 阶段五：构建微调数据集 (对应研究步骤5)

```bash
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new/buzhou/training_data_zhengti/weitiao

# 构建微调数据集
python build_finetuning_dataset.py

# 或者生成完整数据集
python generate_full_dataset.py
```

**输出文件**: `training_finetuning_dataset.jsonl`

### 阶段六：LoRA微调Qwen3-8B (对应研究步骤6)

```bash
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new/buzhou/training_data_zhengti/weitiao

# 运行微调
bash train_qwen3_lora_final.py

# 或者使用配置文件
bash train_qwen3_lora5090.sh
```

**输出文件**: `lora_final_best/` (LoRA权重)

#### 合并LoRA权重

```bash
# 合并LoRA到基座模型
bash merge_lora.sh

# 或者手动运行:
python merge_lora.py \
    --base_model /path/to/Qwen3-8B \
    --lora_path lora_final_best \
    --output_path qwen3_8b_merged_final
```

**输出文件**: `qwen3_8b_merged_final/` (合并后的分析师模型)

##### Llama-3.1-8B 合并示例

> 说明：LoRA 路径既可以给 `checkpoint-*`，也可以直接给训练输出目录（会自动选 best checkpoint）。

```bash
cd /root/shared-nvme/wangjie/Rethinking/buzhou/training_data_zhengti_llama/weitiao

# 你的最佳 LoRA 目录: lora_aggressive_v2Final
# 输出会在同层级生成: lora_aggressive_v2Final_merged/
LORA_RUN_DIR=./lora_aggressive_v2Final bash merge_lora.sh

# 或者手动运行:
python merge_lora.py \
    --base_model /root/shared-nvme/wangjie/Llama-3.1-8B-Instruct \
    --lora_path ./lora_aggressive_v2Final \
    --output_path ./lora_aggressive_v2Final_merged
```

### 阶段七：三轮迭代推理验证 (对应研究步骤7-11)

```bash
cd /path/to/Rethinking\

# 创建输出目录
mkdir -p fenxishi/output/parallel_voting_wtq

# 运行三轮迭代推理
nohup bash fenxishi/run_voting_wtq.sh > fenxishi/output/parallel_voting_wtq/nohup.log 2>&1 &

# 或者直接运行Python脚本:
python fenxishi/run_cot_with_voting.py \
    --model="/path/to/Qwen3-8B" \
    --analyst_model_path="buzhou/training_data_zhengti/weitiao/qwen3_8b_merged_final" \
    --provider="vllm" \
    --dataset="wtq" \
    --temperature=0.8 \
    --log_dir="fenxishi/output/parallel_voting_wtq"
```

**输出文件**: 
- `fenxishi/output/parallel_voting_wtq/result.jsonl`
- `fenxishi/output/parallel_voting_wtq/statistics.json`

### 阶段八：评估结果

```bash
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new

# 评估准确率
python evaluate.py fenxishi/output/parallel_voting_wtq/result.jsonl
```

---

## TabFact数据集完整流程

### 重要差异说明

| 项目 | WTQ | TabFact |
|------|-----|---------|
| 任务类型 | 表格问答 | 事实验证 |
| 答案格式 | 任意文本 | Yes/No |
| 数据集 | wtq.json | tabfact.json |
| 提示词 | 提取答案 | 验证声明 |

### 阶段一：生成推理步骤

#### 1.1 DeepSeek生成推理步骤

```bash
cd /path/to/TabFact

# 运行DeepSeek API
bash scripts/run_deepseek_tabfact.sh

# 或者直接运行:
python run_cot.py \
    --model DeepSeek-V3.1 --long_model DeepSeek-V3.1 \
    --provider openai --system "You are a helpful assistant" \
    --dataset tabfact --sub_sample False \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 4000 --self_consistency 1 --temperature 0.8 \
    --log_dir output/deepseek_tabfact --cache_dir cache/deepseek-V3.1
```

**输出文件**: `output/deepseek_tabfact/result.jsonl`

#### 1.2 Qwen3-8B生成推理步骤

```bash
cd /path/to/TabFact

# 运行Qwen3-8B
bash buzhou/qwen/all_dp_qwen.sh

# 或者直接运行:
python run_cot.py \
    --model "/path/to/Qwen3-8B" --long_model "/path/to/Qwen3-8B" \
    --provider huggingface --system "You are a helpful assistant..." \
    --dataset tabfact --sub_sample False \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 4000 --self_consistency 1 --temperature 0.8 \
    --log_dir output/qwen_tabfact --cache_dir cache/qwen3_8b
```

**输出文件**: `output/qwen_tabfact/result.jsonl`

### 阶段二：分类正确/错误样本

```bash
cd /path/to/TabFact

# 分类DeepSeek结果
python scripts/split_correct_incorrect.py output/deepseek_tabfact/result.jsonl

# 分类Qwen结果
python scripts/split_correct_incorrect.py output/qwen_tabfact/result.jsonl
```

### 阶段三：构建训练数据集

```bash
cd /path/to/TabFact

# 修改build_training_quick.sh中的路径后运行
bash scripts/build_training_quick.sh

# 或者手动运行:
python scripts/build_training_dataset.py \
    --deepseek_correct "output/deepseek_tabfact/correct.jsonl" \
    --qwen_incorrect "output/qwen_tabfact/incorrect.jsonl" \
    --output_dir "training_data" \
    --positive_name "positive_samples.jsonl" \
    --negative_name "negative_samples.jsonl"
```

### 阶段四：DeepSeek分析推理步骤

```bash
cd /path/to/TabFact/buzhou/training_data_zhengti

# 调用DeepSeek分析
bash run_analysis.sh 0 2000
```

### 阶段五：构建微调数据集

```bash
cd /path/to/TabFact/buzhou/training_data_zhengti/weitiao

python build_finetuning_dataset.py
```

### 阶段六：LoRA微调

```bash
cd /path/to/TabFact/buzhou/training_data_zhengti/weitiao

# 微调
bash train_qwen3_lora_final.py

# 合并
bash merge_lora.sh
```

### 阶段七：三轮迭代推理验证

```bash
cd /path/to/TabFact

# 创建输出目录
mkdir -p fenxishi/output/parallel_voting_tabfact

# 运行三轮迭代推理 (TabFact专用版本)
nohup bash fenxishi/run_voting_tabfact.sh > fenxishi/output/parallel_voting_tabfact/nohup.log 2>&1 &

# 或者直接运行:
python fenxishi/run_cot_with_voting_tabfact.py \
    --model="/path/to/Qwen3-8B" \
    --analyst_model_path="buzhou/training_data_zhengti/weitiao/qwen3_8b_merged_final" \
    --provider="vllm" \
    --dataset="tabfact" \
    --temperature=0.8 \
    --log_dir="fenxishi/output/parallel_voting_tabfact"
```

### 阶段八：评估结果

```bash
cd /path/to/TabFact

python evaluate.py fenxishi/output/parallel_voting_tabfact/result.jsonl
```

---

## 路径隔离说明

### ✅ 路径隔离已确认

运行 `TabFact/scripts/all_dp.sh` 时：
1. 脚本中的 `cd "$ROOT_DIR"` 会切换到 `TabFact/` 目录
2. 所有相对路径（如 `data/tabfact.json`, `output/xxx`）都基于 `TabFact/` 目录
3. **不会与 `Rethinking Tabular DeepSeek new/` 产生混淆**

### 路径配置检查清单

| 文件 | 需要检查的路径 | 说明 |
|------|---------------|------|
| `buzhou/qwen/all_dp_qwen.sh` | `ROOT_DIR`, `MODEL_PATH`, `log_dir` | 已修改为TabFact |
| `fenxishi/run_voting_tabfact.sh` | `MODEL_PATH`, `ANALYST_MODEL_PATH`, `log_dir` | 已创建 |
| `scripts/build_training_quick.sh` | 输入/输出文件路径 | 需根据实际情况修改 |

### 服务器路径映射

假设服务器路径为 `/data/amax/home/E23101002/wangjie/`:

```bash
# WTQ项目
/data/amax/home/E23101002/wangjie/Rethinking Tabular DeepSeek new/

# TabFact项目
/data/amax/home/E23101002/wangjie/TabFact/

# Qwen3-8B模型
/data/amax/home/E23101002/wangjie/Qwen3-8B/

# Llama模型
/data/amax/home/E23101002/wangjie/Llama-3.1-8B-Instruct/
```

---

## Llama基座模型实验

如果要使用Llama-3.1-8B作为基座模型：

### 1. 修改模型路径

在相应的shell脚本中修改：

```bash
# 原Qwen路径
MODEL_PATH="/data/amax/home/E23101002/wangjie/Qwen3-8B"

# 改为Llama路径
MODEL_PATH="/data/amax/home/E23101002/wangjie/Llama-3.1-8B-Instruct"
```

### 2. 完整流程

重复上述所有步骤，但使用Llama模型替代Qwen模型：

```bash
# 1. 用Llama生成推理步骤
# 2. 分类正确/错误
# 3. 构建训练数据
# 4. DeepSeek分析
# 5. 构建微调数据集
# 6. LoRA微调Llama
# 7. 三轮迭代验证
# 8. 评估
```

---

## 常见问题

### Q1: 如何断点续传？

使用 `--resume` 参数：

```bash
python run_cot.py ... --resume 1000 --stop_at 2000
```

### Q2: 如何查看运行日志？

```bash
# 查看nohup日志
tail -f fenxishi/output/parallel_voting_wtq/nohup.log

# 查看单条结果
cat fenxishi/output/parallel_voting_wtq/log/0.txt
```

### Q3: 如何评估中间结果？

```bash
python evaluate.py output/xxx/result.jsonl
```

### Q4: GPU显存不足怎么办？

1. 减少batch size
2. 使用更小的模型
3. 启用 `--provider vllm` 使用vLLM加速

### Q5: TabFact和WTQ的主要区别？

| 方面 | WTQ | TabFact |
|------|-----|---------|
| 任务 | 问答 | 事实验证 |
| 答案 | 任意文本 | Yes/No |
| 提示词 | 提取答案 | 验证声明 |
| 评估 | 精确匹配 | 二分类 |

---

## 文件修改清单 (TabFact)

以下是为TabFact项目已修改/创建的文件：

1. **`prompt/tabfact/cot.py`** - 修改提示词，添加详细推理步骤格式
2. **`buzhou/qwen/all_dp_qwen.sh`** - 修改dataset为tabfact
3. **`scripts/run_deepseek_tabfact.sh`** - 新建DeepSeek运行脚本
4. **`fenxishi/run_voting_tabfact.sh`** - 新建TabFact专用验证脚本
5. **`fenxishi/run_cot_with_voting_tabfact.py`** - 新建TabFact专用投票代码

---

## 快速参考命令

```bash
# 环境激活
conda activate rethinking

# WTQ完整流程
cd /path/to/Rethinking\ Tabular\ DeepSeek\ new
bash scripts/all_dp.sh                          # DeepSeek
bash buzhou/qwen/all_dp_qwen.sh                 # Qwen
python scripts/split_correct_incorrect.py xxx   # 分类
bash scripts/build_training_quick.sh            # 构建数据
bash buzhou/training_data_zhengti/run_analysis.sh # 分析
# ... 微调步骤 ...
bash fenxishi/run_voting_wtq.sh                 # 验证
python evaluate.py fenxishi/output/xxx/result.jsonl # 评估

# TabFact完整流程
cd /path/to/TabFact
bash scripts/run_deepseek_tabfact.sh            # DeepSeek
bash buzhou/qwen/all_dp_qwen.sh                 # Qwen
python scripts/split_correct_incorrect.py xxx   # 分类
bash scripts/build_training_quick.sh            # 构建数据
bash buzhou/training_data_zhengti/run_analysis.sh # 分析
# ... 微调步骤 ...
bash fenxishi/run_voting_tabfact.sh             # 验证
python evaluate.py fenxishi/output/xxx/result.jsonl # 评估
```

---

*文档创建时间: 2025年12月18日*
