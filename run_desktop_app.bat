@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" main_qt.py
) else (
  python main_qt.py
)
pause
