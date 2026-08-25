#!/bin/bash
# 分析师建议增强模式 - WTQ数据集
# 使用微调后的Qwen3-8B作为分析师提供改进建议
# 原始Qwen3-8B作为推理者生成推理步骤
#
# 核心区别（vs Verifier模式）:
# 1. 分析师只提供建议，不做正误判断
# 2. 基座模型可以坚持原答案或修改
# 3. 避免"越改越错"的问题

# ======== 模型路径配置 ========
# 基础推理模型（原始Qwen3-8B）
MODEL_PATH="/root/shared-nvme/wangjie/Qwen3-8B"

# 分析师模型（微调后的Qwen3-8B）
ADVISOR_MODEL_PATH="/root/shared-nvme/wangjie/Rethinking Tabular DeepSeek new/buzhou/training_data_zhengti/weitiao/qwen3_8b_merged_final"

# ======== GPU配置 ========
# 使用两张4090 GPU进行推理
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======== 切换到项目根目录 ========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# ======== 思考模式控制 ========
# 对于分析师建议模式,建议关闭思考模式以提高准确性
ENABLE_THINKING=${ENABLE_THINKING:-false}
export QWEN_ENABLE_THINKING="$ENABLE_THINKING"

echo "========================================="
echo "分析师建议增强模式 - WTQ数据集"
echo "========================================="
echo "基础模型: $MODEL_PATH"
echo "分析师模型: $ADVISOR_MODEL_PATH"
echo "思考模式: $ENABLE_THINKING"
echo ""
echo "模式特点:"
echo "  - 分析师只给建议，不说对错"
echo "  - 基座模型被要求'再检查一下'"
echo "  - 可以坚持原答案，也可以修改"
echo "========================================="

# ======== 系统提示设置 ========
if [ "$ENABLE_THINKING" = "true" ]; then
    SYSTEM_PROMPT="You are a helpful assistant. Think step by step and show your reasoning process."
else
    SYSTEM_PROMPT="You are a helpful assistant. Do NOT use <think> or any hidden/internal reasoning. Follow the requested step-by-step format and end with Final Answer only."
fi

# ======== 运行分析师建议增强推理 ========
python fenxishi/run_cot_with_advisor.py \
    --model="$MODEL_PATH" \
    --long_model="$MODEL_PATH" \
    --advisor_model_path="$ADVISOR_MODEL_PATH" \
    --provider="huggingface" \
    --system="$SYSTEM_PROMPT" \
    --dataset="wtq" \
    --sub_sample=False \
    --perturbation="none" \
    --norm=True \
    --disable_resort=True \
    --norm_cache=True \
    --resume=0 \
    --stop_at=52 \
    --self_consistency=1 \
    --temperature=0.8 \
    --max_iterations=2 \
    --use_strict_format=False \
    --enable_thinking="$ENABLE_THINKING" \
    --log_dir="fenxishi/output/wtq_advisor" \
    --cache_dir="cache/qwen3_8b"

echo ""
echo "========================================="
echo "分析师建议增强推理完成!"
echo "========================================="
echo "结果文件: fenxishi/output/wtq_advisor/result.jsonl"
echo "统计信息: fenxishi/output/wtq_advisor/statistics.json"
echo ""
echo "评估命令:"
echo "  python evaluate.py fenxishi/output/wtq_advisor/result.jsonl"
echo "========================================="
