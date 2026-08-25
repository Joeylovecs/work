#!/bin/bash
# This script runs COT on TabFact dataset using DeepSeek API
# - tables are not perturbed
# - resorting stage in NORM is disabled

CUDA_VISIBLE_DEVICES=0 python run_cot.py \
    --model DeepSeek-V3.1 --long_model DeepSeek-V3.1 \
    --provider openai --system "You are a helpful assistant" --dataset tabfact --sub_sample False \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 20 --stop_at 25 --self_consistency 1 --temperature 0.8 \
    --log_dir output/deepseek_tabfact --cache_dir cache/deepseek-V3.1
