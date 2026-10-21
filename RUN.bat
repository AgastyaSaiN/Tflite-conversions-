@echo off
rem Starts the conversion of every .pt file in put_pt_files_here (Windows). Double-click this file.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Open the Anaconda / Miniconda prompt, activate the environment, and run:  python run.py
  pause
  exit /b 1
)
python run.py %*
echo.
pause
