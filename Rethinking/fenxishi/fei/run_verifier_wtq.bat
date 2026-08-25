@echo off
REM Windows批处理脚本 - 运行分析师辅助推理

echo ========================================
echo 分析师辅助推理框架 - WTQ数据集
echo ========================================

REM 设置模型路径 (使用绝对路径)
set BASE_MODEL=d:\master 1\CS\VS code project\rethinking\wangjie\Qwen3-8B
set VERIFIER_MODEL=buzhou\training_data_zhengti\weitiao\qwen3_8b_merged_final

echo.
echo 基础模型: %BASE_MODEL%
echo 分析师模型: %VERIFIER_MODEL%
echo.

REM 运行Python脚本
python fenxishi\run_cot_with_verifier.py ^
    --model="%BASE_MODEL%" ^
    --long_model="%BASE_MODEL%" ^
    --verifier_model_path="%VERIFIER_MODEL%" ^
    --provider="huggingface" ^
    --dataset="wtq" ^
    --perturbation="none" ^
    --norm=True ^
    --disable_resort=True ^
    --norm_cache=True ^
    --sub_sample=False ^
    --resume=0 ^
    --stop_at=100 ^
    --self_consistency=1 ^
    --temperature=0.1 ^
    --max_iterations=5 ^
    --log_dir="fenxishi\output\wtq_with_verifier" ^
    --cache_dir="cache" ^
    --system="You are a helpful assistant" ^
    --use_strict_format=False ^
    --enable_thinking=False

echo.
echo ========================================
echo 分析师辅助推理完成!
echo ========================================
pause
