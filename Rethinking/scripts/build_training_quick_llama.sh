#!/bin/bash

# 构建训练集的快速运行脚本
# 使用默认路径：DeepSeek的output/win_train/correct.jsonl 和 Llama的HPC/output/llama_wtqtrain/incorrect.jsonl

echo "🚀 开始构建基于DeepSeek和Llama对比的训练集..."

# 检查必要文件是否存在
if [ ! -f "output/win_train/correct.jsonl" ]; then
    echo "❌ 错误: DeepSeek正确答案文件不存在: output/win_train/correct.jsonl"
    echo "请确保DeepSeek已经运行完成并生成了正确答案文件"
    exit 1
fi

if [ ! -f "HPC/output/llama_wtqtrain/incorrect.jsonl" ]; then
    echo "❌ 错误: Llama错误答案文件不存在: HPC/output/llama_wtqtrain/incorrect.jsonl"
    echo "请确保Llama已经运行完成并已分类错误答案"
    exit 1
fi

# 运行构建脚本
python scripts/build_training_dataset.py \
    --deepseek_correct "output/win_train/correct.jsonl" \
    --qwen_incorrect "HPC/output/llama_wtqtrain/incorrect.jsonl" \
    --output_dir "output/win_train/training_data_llama" \
    --positive_name "positive_samples.jsonl" \
    --negative_name "negative_samples.jsonl"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练集构建完成！"
    echo "📁 输出文件:"
    echo "   - 正向训练集: output/win_train/training_data_llama/positive_samples.jsonl"
    echo "   - 负向训练集: output/win_train/training_data_llama/negative_samples.jsonl"
    echo "   - 统计信息: output/win_train/training_data_llama/training_stats.json"
    echo ""
    echo "📊 查看详细统计信息:"
    echo "   cat output/win_train/training_data_llama/training_stats.json"
else
    echo "❌ 训练集构建失败！"
    exit 1
fi