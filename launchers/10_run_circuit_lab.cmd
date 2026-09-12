@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m fly_circuit_lab run --control paper --config "circuit_configs\ache2019.toml" --output "runs\circuit-lab\live" --seconds 4 %*
pause
exit /b
:missing
echo Run launchers\01_setup_windows.cmd first.
pause
exit /b 1
