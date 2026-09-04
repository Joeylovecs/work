@echo off
call "%~dp0validate_environment.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0config.bat"

conda run -n "%CONDA_ENV%" --no-capture-output python "%PROJECT_DIR%\scripts\run_conservative_fusion.py" ^
  --dataset "%DATASET%" ^
  --preferred "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\dp_baseline\result.jsonl" ^
  --fallback "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\optimized_dp_raw\result.jsonl" ^
  --preferred-label dp_baseline --fallback-label optimized_dp_semantic ^
  --experiment "%EXPERIMENT_ROOT%/%DATASET%/optimized_dp_guarded" --overwrite
exit /b %errorlevel%
