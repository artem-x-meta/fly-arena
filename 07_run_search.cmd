@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m fly_arena run --mode ethology-hybrid --config "configs\search-renewing.toml" --output "runs\search" --seconds 0 %*
pause
exit /b
:missing
echo Run 01_setup_windows.cmd first.
pause
exit /b 1
