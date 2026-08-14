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
    echo  이 폴더의 "환경 설치.bat" 을 실행하면 만들어집니다.
    echo  (다른 PC로 옮겨 온 경우에도 그것부터 실행하세요.)
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
    echo         engine\venv 는 만들어진 PC의 Python 3.14 를 참고하도록 되어 있어
    echo         다른 PC로 옮기면 그대로 쓸 수 없습니다.
    echo         이 폴더의 "환경 설치.bat" 을 실행하면 새로 만들어집니다.
    echo.
    pause
    exit /b 1
)

"%PY%" "%~dp0engine\transcribe.py" %*
set RC=%ERRORLEVEL%

echo.
if %RC% neq 0 echo  일부 파일이 변환되지 않았습니다. 위의 실패 목록을 확인해주세요.
pause
exit /b %RC%
