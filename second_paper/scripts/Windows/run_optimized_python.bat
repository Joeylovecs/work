@echo off
call "%~dp0validate_environment.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0config.bat"

conda run -n "%CONDA_ENV%" --no-capture-output python "%PROJECT_DIR%\scripts\run_paper1.py" ^
  --dataset "%DATASET%" --mode audit --experiment "%EXPERIMENT_ROOT%/%DATASET%/optimized_python" ^
  --start "%START_INDEX%" --end "%END_INDEX%" --all-questions ^
  --audit-level full --audit-mode hybrid --max-repairs 2 --max-execution-repairs 1 ^
  --dp-votes 1 --max-dp-repairs 2 --dp-revise-threshold 0.8 ^
  --model "%MODEL_ID%" --temperature 0.0 --timeout "%REQUEST_TIMEOUT%" --overwrite
exit /b %errorlevel%
