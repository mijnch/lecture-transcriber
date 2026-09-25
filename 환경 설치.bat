@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title Transcriber - environment setup

rem Python 3.14 is required. Prefer the py launcher (installed by python.org even
rem when "Add to PATH" was not ticked), then fall back to python on PATH.
set "PY="
py -3.14 -c "import sys" >nul 2>nul && set "PY=py -3.14"
if not defined PY (
  python -c "import sys" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.14 not found.
  echo         Install Python 3.14 from python.org, then run this again.
  pause
  exit /b 1
)
%PY% "%~dp0engine\setup_env.py"
echo.
pause