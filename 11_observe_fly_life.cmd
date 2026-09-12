@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m fly_circuit_lab run --control observe --config "configs\search-renewing.toml" --output "runs\circuit-lab\observer" --seconds 0 %*
pause
exit /b
:missing
echo Run 01_setup_windows.cmd first.
pause
exit /b 1
