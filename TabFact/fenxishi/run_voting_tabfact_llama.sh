#!/bin/bash
# ============================================================
# 3-Round Iteration + Parallel Analysis - TabFact (Llama版本)
# ============================================================
# 使用 Llama-3.1-8B 作为基座模型
# 使用微调后的 Llama 作为分析师模型
# 3轮迭代 + 多优先级投票

# ======== 模型路径配置 ========
# 基座模型 (原始 Llama-3.1-8B-Instruct)
MODEL_PATH="/data/amax/home/E23101002/wangjie/Llama-3.1-8B-Instruct"

# 分析师模型 (微调后的 Llama)
# 注意：需要先完成Llama的微调才能使用
ANALYST_MODEL_PATH="/data/amax/home/E23101002/wangjie/TabFact/buzhou/training_data_zhengti/weitiao/llama_merged_final"

# ======== Provider 配置 ========
# 可选: huggingface 或 vllm
PROVIDER=${PROVIDER:-vllm}

# ======== GPU配置 ========
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======== 切换到项目根目录 ========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $ROOT_DIR"
cd "$ROOT_DIR" || exit 1
echo "Current working directory: $(pwd)"

echo "========================================="
echo "3-Round Iteration + Parallel Analysis"
echo "Dataset: TabFact (Fact Verification)"
echo "Model: Llama-3.1-8B"
echo "========================================="
echo "Base Model: $MODEL_PATH"
echo "Analyst Model: $ANALYST_MODEL_PATH"
echo "Provider: $PROVIDER"
echo "========================================="
echo ""
echo "Workflow:"
echo "  Round 1: Base generates Path 1,2,3"
echo "  Round 2: Base generates Path 4,5,6 (with error analysis)"
echo "  Round 3: Base generates Path 7,8,9"
echo "  -> Final Decision with Priority Voting"
echo "========================================="

# ======== 系统提示设置 ========
SYSTEM_PROMPT="You are a precise table fact verification assistant.

CRITICAL RULES FOR TABFACT:
1. The task is to verify if a STATEMENT about the table is TRUE or FALSE
2. Your answer must be exactly 'Yes' (statement is true) or 'No' (statement is false)
3. Carefully read the statement and verify each claim against the table data
4. Pay attention to comparisons, aggregations, temporal references, and superlatives

VALIDATION:
- Before giving Final Answer, verify your reasoning matches the table data
- Your Final Answer must be exactly 'Yes' or 'No'

Follow the requested step-by-step format and end with Final Answer only."

# ======== 运行三轮迭代推理 ========
python fenxishi/run_cot_with_voting_tabfact.py \
    --model="$MODEL_PATH" \
    --long_model="$MODEL_PATH" \
    --analyst_model_path="$ANALYST_MODEL_PATH" \
    --provider="$PROVIDER" \
    --system="$SYSTEM_PROMPT" \
    --dataset="tabfact" \
    --sub_sample=False \
    --perturbation="none" \
    --norm=True \
    --disable_resort=True \
    --norm_cache=True \
    --resume=0 \
    --stop_at=1e6 \
    --temperature=0.8 \
    --enable_thinking=False \
    --log_dir="fenxishi/output/parallel_voting_tabfact_llama" \
    --cache_dir="cache/llama"
