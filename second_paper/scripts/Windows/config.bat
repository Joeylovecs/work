@echo off
REM Windows configuration for the Linux run_150_frozen.sh pipeline.
REM END_INDEX is exclusive: 0 and 2 means indices 0 and 1.
set "DATASET=wtq"
set "START_INDEX=0"
set "END_INDEX=50"
set "EXPERIMENT_ROOT=windows_full_0_2"
set "MODEL_ID=DeepSeek-V3.2"
set "REQUEST_TIMEOUT=60"
set "CONDA_ENV=work2"

for %%I in ("%~dp0..\..") do set "PROJECT_DIR=%%~fI"
for %%I in ("%PROJECT_DIR%\..") do set "WORKSPACE_DIR=%%~fI"
set "PYTHONPATH=%WORKSPACE_DIR%;%PYTHONPATH%"
exit /b 0
