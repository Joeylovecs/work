@echo off
call "%~dp0validate_environment.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0config.bat"

conda run -n "%CONDA_ENV%" --no-capture-output python "%PROJECT_DIR%\scripts\run_guarded_joint.py" ^
  --dataset "%DATASET%" ^
  --primary "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\dp_baseline\result.jsonl" ^
  --blind "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\blind_final_v1\result.jsonl" ^
  --python "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\optimized_python\result.jsonl" ^
  --optimized-dp "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\optimized_dp_raw\result.jsonl" ^
  --output "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\guarded_joint\result.jsonl"
exit /b %errorlevel%
