@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m fly_duel --social --scene encounter --single --seed 1 --seconds 12 --output "runs\food-fight\live" --viewer --video "runs\food-fight\demo.mp4" %*
set "duel_exit_code=%errorlevel%"
pause
exit /b %duel_exit_code%
:missing
echo Run launchers\01_setup_windows.cmd first.
pause
exit /b 1
