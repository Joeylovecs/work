@echo off
call "%~dp0validate_environment.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0config.bat"

conda run -n "%CONDA_ENV%" --no-capture-output python "%PROJECT_DIR%\scripts\run_double_verifier.py" ^
  --dataset "%DATASET%" ^
  --primary "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\dp_baseline\result.jsonl" ^
  --primary-label dp_baseline ^
  --candidate "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\optimized_python\result.jsonl" ^
  --label optimized_python --hide-candidates --required-candidate-support 1 ^
  --confidence-threshold 0.9 --experiment "%EXPERIMENT_ROOT%/%DATASET%/blind_final_v1" --overwrite
exit /b %errorlevel%
