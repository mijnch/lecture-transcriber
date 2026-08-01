@echo off
setlocal
chcp 65001 >nul
title Transcriber
set PYTHONIOENCODING=utf-8

set "PY=%~dp0engine\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo.
    echo  [오류] 전사 엔진을 찾을 수 없습니다.
    echo         %PY%
    echo.
    echo  Transcriber 폴더를 옮기거나 복사한 경우 engine 폴더가 함께 있는지 확인하고,
    echo  그래도 안 되면 사용법.md 의 "문제가 생겼을 때" 항목을 참고해주세요.
    echo.
    pause
    exit /b 1
)

rem python.exe 는 있어도 연결된 Python 본체가 사라졌으면 기동하지 못한다.
rem 그때 원인 불명 오류가 뜨는 대신 무엇이 문제인지 알려준다.
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [오류] 전사 엔진을 실행할 수 없습니다.
    echo         engine\venv 는 이 PC에 설치된 Python 3.14 를 참고하도록 되어 있습니다.
    echo         Python 을 지우거나 옮겼다면 engine\venv 를 다시 만들어야 합니다.
    echo.
    pause
    exit /b 1
)

"%PY%" "%~dp0engine\transcribe.py"
set RC=%ERRORLEVEL%

echo.
if %RC% neq 0 echo  일부 파일이 변환되지 않았습니다. 위의 실패 목록을 확인해주세요.
pause
exit /b %RC%
