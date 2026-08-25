#!/bin/bash
# Run agent method with NORM processing - Generate detailed reasoning steps
CUDA_VISIBLE_DEVICES=0 python run_agent.py \
    --model DeepSeek-V3.1 \
    --provider openai --dataset shuili --sub_sample False \
    --perturbation none --use_full_table True --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 500 --self_consistency 1 --temperature 0.8 \
    --log_dir output/shuili/agent --cache_dir cache/DeepSeek-V3.1
