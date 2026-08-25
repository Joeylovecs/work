#!/bin/bash
# 基线对比实验 - 原始Qwen推理(无分析师辅助)
# 用于与分析师辅助推理进行对比
# 注意: 此脚本仅运行基线实验,不包含自动对比评估

# ======== 模型路径配置 ========
MODEL_PATH="/root/shared-nvme/wangjie/Qwen3-8B"

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

echo "========================================="
echo "基线实验 - 原始Qwen推理(无分析师)"
echo "========================================="
echo "模型路径: $MODEL_PATH"
echo "数据集: WTQ"
echo "思考模式: $ENABLE_THINKING"
echo "========================================="

# ======== 系统提示设置 ========
if [ "$ENABLE_THINKING" = "true" ]; then
    SYSTEM_PROMPT="You are a helpful assistant. Think step by step and show your reasoning process."
else
    SYSTEM_PROMPT="You are a helpful assistant. Do NOT use <think> or any hidden/internal reasoning. Follow the requested step-by-step format and end with Final Answer only."
fi

# ======== 运行原始CoT推理 ========
python fenxishi/run_cot_fenxishi.py \
    --model="$MODEL_PATH" \
    --long_model="$MODEL_PATH" \
    --provider="huggingface" \
    --system="$SYSTEM_PROMPT" \
    --dataset="wtq" \
    --sub_sample=False \
    --perturbation="none" \
    --norm=True \
    --disable_resort=True \
    --norm_cache=True \
    --resume=0 \
    --stop_at=2203 \
    --self_consistency=1 \
    --temperature=0.1 \
    --use_strict_format=False \
    --enable_thinking="$ENABLE_THINKING" \
    --log_dir="fenxishi/output/baseline_qwen" \
    --cache_dir="cache/qwen3_8b"

echo ""
echo "========================================="
echo "基线实验完成!"
echo "========================================="
echo "结果文件: fenxishi/output/baseline_qwen/result.jsonl"
echo ""
echo "评估命令:"
echo "  python evaluate.py fenxishi/output/baseline_qwen/result.jsonl"
echo "========================================="
