#!/bin/bash
# 3-Round Iteration + Parallel Analysis + Multi-Priority Voting - TabFact Dataset
# Uses fine-tuned Qwen3-8B as analyst to evaluate reasoning steps
# Original Qwen3-8B as base model to generate reasoning steps
# 3 rounds iteration with multi-priority voting for final answer

# ======== Model Path Configuration ========
# Base reasoning model (original Qwen3-8B)
MODEL_PATH="/data/amax/home/E23101002/wangjie/Qwen3-8B"

# Analyst model (fine-tuned Qwen3-8B for TabFact)
# 注意：需要先训练TabFact专用的分析师模型
ANALYST_MODEL_PATH="/data/amax/home/E23101002/wangjie/TabFact/buzhou/training_data_zhengti/weitiao/qwen3_8b_merged_final"

# ======== GPU Configuration ========
# Use two 4090 GPUs for inference
# Base model -> GPU 0
# Analyst model -> GPU 1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======== Switch to project root directory ========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $ROOT_DIR"
cd "$ROOT_DIR" || exit 1
echo "Current working directory: $(pwd)"

# ======== Thinking Mode Control ========
ENABLE_THINKING=${ENABLE_THINKING:-false}
export QWEN_ENABLE_THINKING="$ENABLE_THINKING"

echo "========================================="
echo "3-Round Iteration + Parallel Analysis + Multi-Priority Voting"
echo "Dataset: TabFact (Fact Verification)"
echo "========================================="
echo "Base Model: $MODEL_PATH"
echo "Analyst Model: $ANALYST_MODEL_PATH"
echo "Thinking Mode: $ENABLE_THINKING"
echo "========================================="
echo ""
echo "Workflow:"
echo "  Round 1: Base generates Path 1,2,3 (3 independent windows)"
echo "           -> Analyst analyzes in 3 parallel windows"
echo "           -> All correct & same answer? -> YES: output, NO: continue"
echo ""
echo "  Round 2: Base generates Path 4,5,6 (with error analysis)"
echo "           -> Analyst analyzes in 3 parallel windows"
echo "           -> Has correct & unique? -> YES: output, NO: continue"
echo ""
echo "  Round 3: Base generates Path 7,8,9"
echo "           -> Analyst analyzes in 3 parallel windows"
echo "           -> Final Decision:"
echo "              Priority1: Round3 correct path"
echo "              Priority2: History correct vote"
echo "              Priority3: All 9 answers vote"
echo ""
echo "========================================="

# ======== System Prompt Setting ========
if [ "$ENABLE_THINKING" = "true" ]; then
    SYSTEM_PROMPT="You are a helpful assistant. Think step by step and show your reasoning process."
else
    SYSTEM_PROMPT="You are a precise table fact verification assistant.

CRITICAL RULES FOR TABFACT:
1. The task is to verify if a STATEMENT about the table is TRUE or FALSE
2. Your answer must be exactly 'Yes' (statement is true) or 'No' (statement is false)
3. Carefully read the statement and verify each claim against the table data
4. Pay attention to:
   - Comparisons (more than, less than, equal to)
   - Aggregations (total, average, count)
   - Temporal references (before, after, first, last)
   - Superlatives (most, least, highest, lowest)

VALIDATION:
- Before giving Final Answer, verify your reasoning matches the table data
- Check that you correctly interpreted the statement's meaning
- Your Final Answer must be exactly 'Yes' or 'No'

Do NOT use <think> or any hidden/internal reasoning. Follow the requested step-by-step format and end with Final Answer only."
fi

# ======== Run 3-Round + Parallel Analysis Inference ========
python fenxishi/run_cot_with_voting_tabfact.py \
    --model="$MODEL_PATH" \
    --long_model="$MODEL_PATH" \
    --analyst_model_path="$ANALYST_MODEL_PATH" \
    --provider="vllm" \
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
    --enable_thinking="$ENABLE_THINKING" \
    --log_dir="fenxishi/output/parallel_voting_tabfact" \
    --cache_dir="cache/qwen3_8b"
