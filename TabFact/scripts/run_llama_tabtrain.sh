#!/bin/bash
# ============================================================
# Llama-3.1-8B 跑 TabFact 训练集 (tabtrain.json)
# ============================================================
# 用途: 生成Llama的推理步骤 (研究步骤2 - Llama版本)
# 输出: output/llama_tabtrain/result.jsonl

# ======== 模型路径配置 ========
# Llama-3.1-8B-Instruct 模型路径
MODEL_PATH="/data/amax/home/E23101002/wangjie/Llama-3.1-8B-Instruct"

# ======== 参数配置 ========
RESUME=${1:-0}           # 起始索引，可通过命令行参数传入
STOP_AT=${2:-4000}       # 结束索引，可通过命令行参数传入

# ======== Provider 配置 ========
# 可选: huggingface 或 vllm
PROVIDER=${PROVIDER:-huggingface}

# ======== GPU配置 ========
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======== 切换到项目根目录 ========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# ======== 系统提示设置 ========
SYSTEM_PROMPT="You are a helpful assistant. Follow the requested step-by-step format and end with Final Answer only."

echo "========================================="
echo "Llama-3.1-8B 跑 TabFact 训练集"
echo "========================================="
echo "模型路径: $MODEL_PATH"
echo "Provider: $PROVIDER"
echo "数据集: tabtrain (TabFact训练集)"
echo "处理范围: $RESUME 到 $STOP_AT"
echo "输出目录: output/llama_tabtrain"
echo "========================================="

python run_cot.py \
    --model "$MODEL_PATH" \
    --long_model "$MODEL_PATH" \
    --provider "$PROVIDER" \
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
    --enable_thinking=False \
    --log_dir output/llama_tabtrain \
    --cache_dir cache/llama

echo "完成! 结果保存在 output/llama_tabtrain/result.jsonl"
