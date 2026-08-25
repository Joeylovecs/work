#!/bin/bash
# Step 1: Generate wtq_agent_wo_norm results first

CUDA_VISIBLE_DEVICES=0 python run_agent.py \
    --model deepseek-chat\
    --provider openai --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 10 --self_consistency 1 --temperature 0.1 \
    --log_dir output/wtq_agent_wo_norm --cache_dir cache/deepseek-chat

