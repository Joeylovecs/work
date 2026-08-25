@echo off

cd ..

set CUDA_VISIBLE_DEVICES=0

python run_cot.py ^
    --model DeepSeek-V3.1 --long_model DeepSeek-V3.1 ^
    --provider openai --system "You are a helpful assistant" --dataset train --sub_sample False ^
    --perturbation none --norm True --disable_resort True --norm_cache True ^
    --resume 3950 --stop_at 1e6 --self_consistency 1 --temperature 0.8 ^
    --log_dir output/win_train --cache_dir cache/deepseek-V3.1

pause