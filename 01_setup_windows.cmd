@echo off
setlocal
cd /d "%~dp0"
py -3.12 --version
if errorlevel 1 goto python_missing
py -3.12 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 goto failed
echo Ready. Run 02_test_body.cmd next.
pause
exit /b 0
:python_missing
echo Install 64-bit Python 3.12 with the Python launcher from python.org first.
pause
exit /b 1
:failed
echo Setup failed. Please copy the error above.
pause
exit /b 1
