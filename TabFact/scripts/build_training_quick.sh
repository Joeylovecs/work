#!/bin/bash

# 构建训练集的快速运行脚本
# 使用默认路径：DeepSeek的output/train/dp/correct.json 和 Qwen的incorrect.jsonl

echo "🚀 开始构建基于DeepSeek和Qwen对比的训练集..."

# 检查必要文件是否存在
if [ ! -f "output/train/dp/correct.jsonl" ]; then
    echo "❌ 错误: DeepSeek正确答案文件不存在: output/train/dp/correct.jsonl"
    echo "请确保DeepSeek已经运行完成并生成了正确答案文件"
    exit 1
fi

if [ ! -f "output/qwen3_8b_train/incorrect.jsonl" ]; then
    echo "❌ 错误: Qwen错误答案文件不存在: output/qwen3_8b_train/incorrect.jsonl"
    echo "请确保Qwen已经运行完成并已分类错误答案"
    exit 1
fi

# 运行构建脚本
python scripts/build_training_dataset.py \
    --deepseek_correct "output/train/dp/correct.jsonl" \
    --qwen_incorrect "output/qwen3_8b_train/incorrect.jsonl" \
    --output_dir "training_data" \
    --positive_name "positive_samples.jsonl" \
    --negative_name "negative_samples.jsonl"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练集构建完成！"
    echo "📁 输出文件:"
    echo "   - 正向训练集: training_data/positive_samples.jsonl"
    echo "   - 负向训练集: training_data/negative_samples.jsonl"
    echo "   - 统计信息: training_data/training_stats.json"
    echo ""
    echo "📊 查看详细统计信息:"
    echo "   cat training_data/training_stats.json"
else
    echo "❌ 训练集构建失败！"
    exit 1
fi