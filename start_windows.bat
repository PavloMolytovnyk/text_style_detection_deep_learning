@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
if not exist ".venv\textscope-ready" (
    ".venv\Scripts\python.exe" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    echo ready>".venv\textscope-ready"
)
".venv\Scripts\python.exe" run.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or launch failed. Check the error above. Python 3.12 is recommended.
pause
exit /b 1
