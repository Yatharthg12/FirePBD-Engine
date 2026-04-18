@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Could not find venv\Scripts\python.exe.
  echo Install Python 3.10 or 3.11, then create the venv again.
  exit /b 1
)

echo Installing dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency install failed.
  exit /b 1
)

echo Running smoke test...
"venv\Scripts\python.exe" smoke_test.py
if errorlevel 1 (
  echo Smoke test failed.
  exit /b 1
)

echo Starting server at http://127.0.0.1:8000/app
"venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --port 8000
