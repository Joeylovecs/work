#!/bin/bash
#SBATCH --job-name=xiao_qwen_wtq_sikao
#SBATCH --partition=GPUFEE08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:2
#SBATCH --time=470:00:00
#SBATCH --constraint=Python
#SBATCH --output=logs/xiao_qwen_wtq_sikao_%j.out
#SBATCH --error=logs/xiao_qwen_wtq_sikao_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================
# 【消融实验】Qwen3-8B 基座 + 基座（无微调分析师）- WTQ - 思考模式 - HPC版本
# ============================================================
# 用途: 消融实验 - 验证微调分析师模型的贡献（思考模式）
# 将分析师模型替换为基座模型本身（基座+基座迭代）
# 基座模型 -> GPU 0 (HuggingFace)
# "分析师"模型 -> GPU 1 (HuggingFace，实际也是基座模型)
# 输出: fenxishi/output/xiao_final_sikao_wtq/

echo "=========================================="
echo "【消融实验】基座+基座 (无微调分析师) - 思考模式"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# ======== 环境配置 ========
cd $SLURM_SUBMIT_DIR
echo "Submit directory: $(pwd)"

# 激活conda环境
source ~/anaconda3/bin/activate
conda activate rethinking
echo "Conda environment: $CONDA_DEFAULT_ENV"

# 双卡环境变量配置
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 检查GPU
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# ======== 切换到项目根目录 ========
# 从 Rethinking/fenxishi 向上2级到 wangjie/
# 再进入 Rethinking/ 作为项目根目录
cd ..
ROOT_DIR=$(pwd)
echo "Project root: $ROOT_DIR"

# 验证关键文件存在
if [ ! -f "run_helper.py" ]; then
    echo "ERROR: run_helper.py not found in $ROOT_DIR"
    echo "Current directory contents:"
    ls -la
    exit 1
fi
echo "✓ run_helper.py found"

# 设置 PYTHONPATH
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# ======== 模型路径配置 ========
# 【消融实验】基座模型和分析师模型都用同一个基座模型
MODEL_PATH="/gpfs/home/E24301289/wangjie/Qwen3-8B"
ANALYST_MODEL_PATH="/gpfs/home/E24301289/wangjie/Qwen3-8B"

echo "Base Model: $MODEL_PATH"
echo "Analyst Model (消融-基座): $ANALYST_MODEL_PATH"

# ======== 参数配置 ========
RESUME=${1:-0}
STOP_AT=${2:-5000}

# ======== Thinking Mode 配置（思考模式开启）========
ENABLE_THINKING=true
ENABLE_THINKING_ANALYST=true

# ======== System Prompt 设置 ========
SYSTEM_PROMPT="You are a precise table question answering assistant. 

CRITICAL ANSWER FORMAT RULES:
1. For 'how many' questions: Answer with a NUMBER only (e.g., '3', not 'Spain, Italy, France')
2. For 'most/least/only/single' questions: Give ONE answer only, even if tied. Pick the one that appears FIRST in the table.
3. For comparison questions (which is more/less): Choose ONE option, not both.
4. For 'last/first' in a list: Identify by row position, NOT by label like 'Total'.
5. For 'after X year': The year immediately FOLLOWING X, not the same year.

VALIDATION:
- Before giving Final Answer, verify it matches the question type.
- If question asks 'how many', your answer MUST be a number.
- If question asks 'which one', your answer MUST be exactly ONE item.

Do NOT use <think> or any hidden/internal reasoning. Follow the requested step-by-step format and end with Final Answer only."

# ======== 创建输出目录 ========
mkdir -p fenxishi/output/xiao_final_sikao_wtq
mkdir -p fenxishi/output/xiao_final_sikao_wtq/log
mkdir -p cache/qwen3_8b
mkdir -p logs

echo "========================================="
echo "【消融实验】3-Round Iteration - 基座+基座 (无微调分析师)"
echo "【Qwen版本 - WTQ - 思考模式】"
echo "========================================="
echo "Base Model: $MODEL_PATH"
echo "Analyst Model (消融-基座): $ANALYST_MODEL_PATH"
echo "Dataset: WTQ"
echo "Resume: $RESUME  Stop at: $STOP_AT"
echo "Base Model Thinking Mode: $ENABLE_THINKING"
echo "Analyst Model Thinking Mode: $ENABLE_THINKING_ANALYST"
echo "Output: fenxishi/output/xiao_final_sikao_wtq"
echo "========================================="

# ======== 运行推理 ========
# 注意: HPC不支持VLLM，使用huggingface provider
# 使用 buzhou/.../HPCrun_cot_sikao.py 作为HPC入口（自动替换Model为HPCModel）
if [ ! -f "buzhou/training_data_zhengti_llama/fenxishi/2run_cot_sikao.py" ]; then
    echo "[WARN] 2run_cot_sikao.py not found on HPC."
    echo "[WARN] HPCrun_cot_sikao.py will fall back to legacy logic."
fi
python -u buzhou/training_data_zhengti_llama/fenxishi/HPCrun_cot_sikao.py \
    --model="$MODEL_PATH" \
    --long_model="$MODEL_PATH" \
    --analyst_model_path="$ANALYST_MODEL_PATH" \
    --provider="huggingface" \
    --system="$SYSTEM_PROMPT" \
    --dataset="wtq" \
    --sub_sample=False \
    --perturbation="none" \
    --norm=True \
    --disable_resort=False \
    --norm_cache=True \
    --resume=$RESUME \
    --stop_at=$STOP_AT \
    --temperature=0.3 \
    --enable_thinking="$ENABLE_THINKING" \
    --enable_thinking_analyst="$ENABLE_THINKING_ANALYST" \
    --log_dir="fenxishi/output/xiao_final_sikao_wtq" \
    --cache_dir="cache/qwen3_8b"

echo "=========================================="
echo "【消融实验】Job finished at: $(date)"
echo "Results saved in: fenxishi/output/xiao_final_sikao_wtq/result.jsonl"
echo "=========================================="
