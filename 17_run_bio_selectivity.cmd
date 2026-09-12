@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m fly_bio_selectivity
if errorlevel 1 goto failed
".venv\Scripts\python.exe" bio_tools\selectivity_report.py
if errorlevel 1 goto failed
echo Report: docs\BIO_SELECTIVITY_RESULT.md
pause
exit /b 0
:missing
echo Run 01_setup_windows.cmd first.
:failed
pause
exit /b 1
