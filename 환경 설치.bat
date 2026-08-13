@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title Transcriber - environment setup
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found on PATH.
  echo         Install Python 3.14 from python.org, then run this again.
  pause
  exit /b 1
)
python "%~dp0engine\setup_env.py"
echo.
pause
