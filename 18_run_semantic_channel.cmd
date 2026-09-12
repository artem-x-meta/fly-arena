@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -X utf8 -m fly_semantic demo --semantic-config semantic_configs\need-food.toml --interactive %*
if errorlevel 1 goto failed
pause
exit /b 0
:missing
echo Run 01_setup_windows.cmd first.
:failed
pause
exit /b 1
