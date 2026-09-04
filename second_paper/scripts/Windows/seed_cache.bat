@echo off
REM Equivalent to the Linux seed_cache function: copy only missing cache files.
call "%~dp0config.bat"
if errorlevel 1 exit /b %errorlevel%

call :seed "%PROJECT_DIR%\outputs\100_dev_v1\%DATASET%\python_baseline" "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\python_baseline"
if errorlevel 8 exit /b %errorlevel%
call :seed "%PROJECT_DIR%\outputs\100_dev_v1\%DATASET%\dp_baseline" "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\dp_baseline"
if errorlevel 8 exit /b %errorlevel%
call :seed "%PROJECT_DIR%\outputs\100_dev_v6\%DATASET%\optimized_python_routed" "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\optimized_python"
if errorlevel 8 exit /b %errorlevel%
call :seed "%PROJECT_DIR%\outputs\100_dev_v6\%DATASET%\optimized_dp_routed" "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\optimized_dp_raw"
if errorlevel 8 exit /b %errorlevel%
call :seed "%PROJECT_DIR%\outputs\100_dev_v6\%DATASET%\blind_final_v1" "%PROJECT_DIR%\outputs\%EXPERIMENT_ROOT%\%DATASET%\blind_final_v1"
if errorlevel 8 exit /b %errorlevel%
exit /b 0

:seed
if not exist "%~1\cache\" exit /b 0
robocopy "%~1\cache" "%~2\cache" /E /XC /XN /XO /R:0 /W:0 /NFL /NDL /NJH /NJS >nul
exit /b %errorlevel%
