#!/bin/bash
# ============================================================
# Qwen3-8B 跑 TabFact 训练集 (tabtrain.json)
# ============================================================
# 用途: 生成Qwen的推理步骤 (研究步骤2)
# 输出: output/qwen_tabtrain/result.jsonl

# ======== 模型路径配置 ========
# Qwen3-8B 模型路径
MODEL_PATH="/data/amax/home/E23101002/wangjie/Qwen3-8B"

# ======== 参数配置 ========
RESUME=${1:-0}           # 起始索引，可通过命令行参数传入
STOP_AT=${2:-10}       # 结束索引，可通过命令行参数传入

# ======== GPU配置 ========
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======== 切换到项目根目录 ========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# ======== 思考模式控制 ========
ENABLE_THINKING=${ENABLE_THINKING:-false}
export QWEN_ENABLE_THINKING="$ENABLE_THINKING"

# ======== 系统提示设置 ========
if [ "$ENABLE_THINKING" = "true" ]; then
    SYSTEM_PROMPT="You are a helpful assistant. Think step by step and show your reasoning process."
else
    SYSTEM_PROMPT="You are a helpful assistant. Do NOT use <think> or any hidden/internal reasoning. Follow the requested step-by-step format and end with Final Answer only."
fi

echo "========================================="
echo "Qwen3-8B 跑 TabFact 训练集"
echo "========================================="
echo "模型路径: $MODEL_PATH"
echo "数据集: tabtrain (TabFact训练集)"
echo "处理范围: $RESUME 到 $STOP_AT"
echo "思考模式: $ENABLE_THINKING"
echo "输出目录: output/qwen_tabtrain"
echo "========================================="

python run_cot.py \
    --model "$MODEL_PATH" \
    --long_model "$MODEL_PATH" \
    --provider huggingface \
    --system "$SYSTEM_PROMPT" \
    --dataset tabtrain \
    --sub_sample False \
    --perturbation none \
    --norm True \
    --disable_resort True \
    --norm_cache True \
    --resume $RESUME \
    --stop_at $STOP_AT \
    --self_consistency 1 \
    --temperature 0.8 \
    --use_strict_format True \
    --enable_thinking="$ENABLE_THINKING" \
    --log_dir output/qwen_tabtrain \
    --cache_dir cache/qwen3_8b

echo "完成! 结果保存在 output/qwen_tabtrain/result.jsonl"
