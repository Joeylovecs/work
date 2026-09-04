@echo off
call "%~dp0validate_environment.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0config.bat"

call "%~dp0seed_cache.bat"
if errorlevel 8 exit /b %errorlevel%
call "%~dp0run_python_baseline.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_dp_baseline.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_optimized_python.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_optimized_dp_raw.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_optimized_dp_guarded.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_blind_verifier.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_guarded_joint.bat"
if errorlevel 1 exit /b %errorlevel%

echo FINAL_PIPELINE_COMPLETE dataset=%DATASET% start=%START_INDEX% end=%END_INDEX%
