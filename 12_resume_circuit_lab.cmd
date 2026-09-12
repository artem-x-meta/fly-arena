@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m fly_circuit_lab run --resume "runs\circuit-lab\live\checkpoints\latest.npz" --output "runs\circuit-lab\live" --seconds 0 %*
pause
exit /b
:missing
echo Run 01_setup_windows.cmd first.
pause
exit /b 1
