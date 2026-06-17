@echo off
title Face Recognition System - Server Mode
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py --no-display
) else (
    python main.py --no-display
)
pause
