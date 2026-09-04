@echo off
call "%~dp0validate_environment.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0config.bat"

conda run -n "%CONDA_ENV%" --no-capture-output python "%PROJECT_DIR%\scripts\run_paper1.py" ^
  --dataset "%DATASET%" --mode dp_audit --experiment "%EXPERIMENT_ROOT%/%DATASET%/optimized_dp_raw" ^
  --start "%START_INDEX%" --end "%END_INDEX%" --all-questions ^
  --audit-level full --audit-mode hybrid --max-repairs 0 --max-execution-repairs 0 ^
  --dp-votes 1 --max-dp-repairs 0 --dp-revise-threshold 0.8 ^
  --model "%MODEL_ID%" --temperature 0.0 --timeout "%REQUEST_TIMEOUT%" --overwrite
exit /b %errorlevel%
