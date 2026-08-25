#!/bin/bash
# ============================================================
# DeepSeek 跑 TabFact 训练集 (tabtrain.json)
# ============================================================
# 用途: 生成DeepSeek的推理步骤 (研究步骤1)
# 输出: output/deepseek_tabtrain/result.jsonl

# ======== 参数配置 ========
RESUME=${1:-74974}           # 起始索引，可通过命令行参数传入   
STOP_AT=${2:-1e6}       # 结束索引，可通过命令行参数传入

echo "========================================="
echo "DeepSeek 跑 TabFact 训练集"
echo "========================================="
echo "数据集: tabtrain (TabFact训练集)"
echo "处理范围: $RESUME 到 $STOP_AT"
echo "输出目录: output/deepseek_tabtrain"
echo "========================================="

CUDA_VISIBLE_DEVICES=0 python run_cot.py \
    --model DeepSeek-V3.1 \
    --long_model DeepSeek-V3.1 \
    --provider openai \
    --system "You are a helpful assistant" \
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
    --log_dir output/deepseek_tabtrain \
    --cache_dir cache/deepseek-V3.1

echo "完成! 结果保存在 output/deepseek_tabtrain/result.jsonl"
