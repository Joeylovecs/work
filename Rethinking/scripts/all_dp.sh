#!/bin/bash
# This script runs dp on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled

CUDA_VISIBLE_DEVICES=0 python run_cot.py \
    --model DeepSeek-V3.1 --long_model DeepSeek-V3.1 \
    --provider openai --system "You are a helpful assistant" --dataset train --sub_sample False \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 3 --self_consistency 1 --temperature 0.8 \
    --log_dir output/win_train --cache_dir cache/deepseek-V3.1

 