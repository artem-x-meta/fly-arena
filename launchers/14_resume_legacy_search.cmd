@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" goto missing
if exist "runs\compatibility\search\checkpoints\latest.npz" goto resume
".venv\Scripts\python.exe" -m fly_compat migrate --input "runs\search\checkpoints\latest.npz" --output "runs\compatibility\search\checkpoints\latest.npz"
if errorlevel 1 goto failed
:resume
".venv\Scripts\python.exe" -m fly_arena run --resume "runs\compatibility\search\checkpoints\latest.npz" --output "runs\compatibility\search" --seconds 0 %*
if errorlevel 1 goto failed
pause
exit /b 0
:missing
echo Run launchers\01_setup_windows.cmd first.
:failed
pause
exit /b 1
