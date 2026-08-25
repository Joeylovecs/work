@echo off
REM 对比实验: 原始Qwen vs 分析师辅助Qwen

echo ========================================
echo 对比实验: 原始Qwen vs 分析师辅助Qwen
echo ========================================
echo.

set BASE_MODEL=d:\master 1\CS\VS code project\rethinking\wangjie\Qwen3-8B
set VERIFIER_MODEL=buzhou\training_data_zhengti\weitiao\qwen3_8b_merged_final
set DATASET=wtq
set STOP_AT=100

echo 实验设置:
echo   基础模型: %BASE_MODEL%
echo   分析师模型: %VERIFIER_MODEL%
echo   数据集: %DATASET%
echo   测试数量: %STOP_AT%
echo.
echo ========================================
echo 实验1: 原始Qwen推理 (无分析师)
echo ========================================
echo.

python fenxishi\run_cot_fenxishi.py ^
    --model="%BASE_MODEL%" ^
    --long_model="%BASE_MODEL%" ^
    --provider="huggingface" ^
    --dataset="%DATASET%" ^
    --stop_at=%STOP_AT% ^
    --log_dir="fenxishi\output\baseline_qwen" ^
    --temperature=0.1 ^
    --self_consistency=1

echo.
echo ========================================
echo 实验2: 分析师辅助Qwen推理
echo ========================================
echo.

python fenxishi\run_cot_with_verifier.py ^
    --model="%BASE_MODEL%" ^
    --long_model="%BASE_MODEL%" ^
    --verifier_model_path="%VERIFIER_MODEL%" ^
    --provider="huggingface" ^
    --dataset="%DATASET%" ^
    --stop_at=%STOP_AT% ^
    --max_iterations=5 ^
    --log_dir="fenxishi\output\verifier_assisted_qwen" ^
    --temperature=0.1 ^
    --self_consistency=1

echo.
echo ========================================
echo 开始评估对比...
echo ========================================
echo.

REM 评估两个结果
echo 评估原始Qwen结果...
python evaluate.py fenxishi\output\baseline_qwen\result.jsonl

echo.
echo 评估分析师辅助Qwen结果...
python evaluate.py fenxishi\output\verifier_assisted_qwen\result.jsonl

echo.
echo ========================================
echo 对比实验完成!
echo ========================================
echo.
echo 结果文件位置:
echo   原始Qwen: fenxishi\output\baseline_qwen\
echo   分析师辅助: fenxishi\output\verifier_assisted_qwen\
echo.
echo 请查看evaluate.py的输出对比准确率提升情况
echo ========================================
pause
