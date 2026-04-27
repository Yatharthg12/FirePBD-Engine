@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title FirePBD Engine Launcher

set "PYTHON=%~dp0venv\Scripts\python.exe"
set "LOG_DIR=%~dp0logs"
set "LOG_FILE=%LOG_DIR%\launcher.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

echo ==================================================
echo FirePBD Engine launcher
echo ==================================================
echo Log file: "%LOG_FILE%"
echo.

if not exist "%PYTHON%" (
  echo [ERROR] Could not find venv\Scripts\python.exe.
  echo Install Python 3.10 or 3.11, then recreate the virtual environment.
  goto :fail
)

echo [1/3] Installing dependencies...
"%PYTHON%" -m pip install -r requirements.txt > "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Dependency install failed. See "%LOG_FILE%".
  goto :fail
)

echo [2/3] Running smoke test...
"%PYTHON%" smoke_test.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Smoke test failed. See "%LOG_FILE%".
  goto :fail
)

echo [3/3] Starting server at http://127.0.0.1:8000/app
echo Press Ctrl+C to stop the server.
echo.

"%PYTHON%" -m uvicorn backend.main:app --reload --port 8000 >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [INFO] Server stopped with exit code !EXIT_CODE!.
echo See "%LOG_FILE%" for details.
goto :fail

:fail
echo.
pause >nul
