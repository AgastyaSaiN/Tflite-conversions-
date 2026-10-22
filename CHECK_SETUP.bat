@echo off
rem Checks that this computer has everything needed (Windows). Double-click this file.
cd /d "%~dp0"
python run.py --check
echo.
pause
