@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m fly_bio --output "runs\bio-visual-v1\controls"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m fly_bio --output "runs\bio-visual-v1\interventions" --case dark_loom:both:input_off --case dark_loom:both:retina --case dark_loom:both:lamina --case dark_loom:both:lc4_lplc2 --case dark_loom:both:gf --case dark_loom:left:none --case dark_loom:right:none
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m bio_tools.sensitivity
if errorlevel 1 goto failed
".venv\Scripts\python.exe" bio_tools\report.py
if errorlevel 1 goto failed
echo Report: runs\bio-visual-v1\RESULTS.md
pause
exit /b 0
:missing
echo Run 01_setup_windows.cmd first.
:failed
pause
exit /b 1
