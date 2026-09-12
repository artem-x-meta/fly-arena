@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe goto missing
.venv\Scripts\python.exe -m fly_arena run --mode connectome --seconds 0
pause
exit /b
:missing
echo Run launchers\01_setup_windows.cmd first.
pause
exit /b 1
