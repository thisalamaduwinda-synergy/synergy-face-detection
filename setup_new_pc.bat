@echo off
title Face Recognition System - New PC Setup
cd /d "%~dp0"

echo.
echo ============================================================
echo  Employee Face Recognition System - Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

:: Check Python version is 3.11
echo %PYVER% | findstr /b "3.11" >nul
if errorlevel 1 (
    echo [WARN] Python 3.11.x is recommended. Found: %PYVER%
    echo        Press any key to continue anyway, or Ctrl+C to cancel.
    pause >nul
)

:: Create virtual environment
echo.
echo [1/4] Creating virtual environment...
if exist ".venv" (
    echo       .venv already exists - skipping creation.
) else (
    python -m venv .venv
    echo       Virtual environment created.
)

:: Activate and upgrade pip
echo.
echo [2/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

:: Install requirements
echo.
echo [3/4] Installing requirements (this may take 5-10 minutes)...
".venv\Scripts\pip.exe" install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] requirements.txt install failed.
    pause
    exit /b 1
)
echo       Core requirements installed.

:: Install extra packages not in requirements.txt
echo.
echo [4/4] Installing extra packages (PyQt6, TTS, Audio)...
".venv\Scripts\pip.exe" install PyQt6 pyttsx3 edge-tts pygame pywin32 --quiet
if errorlevel 1 (
    echo [WARN] Some extra packages may have failed. Check above output.
)
echo       Extra packages installed.

:: Done
echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo  To run the desktop app:   double-click run_desktop_app.bat
echo  To run headless server:   run_server.bat
echo.
echo  IMPORTANT: Edit config\config.yaml before first run.
echo  - Set your camera RTSP URL under "cameras:"
echo  - Set camera_host, camera_user, camera_password under "greeting:"
echo.
pause
