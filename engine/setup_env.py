# -*- coding: utf-8 -*-
"""다른 PC에서 이 도구를 쓸 수 있게 engine 폴더 안에 파이썬 환경을 만든다.

    python engine\\setup_env.py

무엇을 하는가
  1. 쓸 만한 Python 3.14 를 찾는다
  2. engine\\venv 를 만든다  (폴더 안이므로 도구를 옮기면 같이 간다)
  3. engine\\requirements.txt 로 패키지를 설치한다
  4. 핵심 import 와 외부 프로그램(FFmpeg·Tesseract) 을 검증한다

왜 필요한가
  venv 는 이식되지 않는다 — pyvenv.cfg 의 `home =` 에 만들어진 PC의 파이썬
  절대경로가 박히기 때문이다. 폴더를 다른 PC로 복사하면 venv 는 그대로
  못 쓰고 여기서 새로 만들어야 한다. ('Transcriber 실행.bat' 도 이 상황을
  감지해 안내하도록 되어 있다.)

★ 이 스크립트가 못 하는 것 (별도 프로그램이라 폴더에 넣을 성격이 아니다)
  - FFmpeg    : PATH 에 있어야 한다 (없으면 원본 직접 디코딩으로 폴백하지만 느리다)
  - Tesseract : 슬라이드 화면 글자 읽기에 쓴다. 없으면 그 기능만 꺼진다
  - Python 본체 : venv 를 만들려면 대상 PC에 Python 3.14 가 필요하다

★ 모델(faster-whisper) 은 첫 실행 때 자동으로 받아 engine\\models 에 넣는다.
  인터넷이 없는 PC로 옮긴다면 engine\\models 폴더도 함께 복사하라.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
BASE = ENGINE_DIR.parent
VENV_DIR = ENGINE_DIR / "venv"
REQ = ENGINE_DIR / "requirements.txt"
VENV_PY = VENV_DIR / "Scripts" / "python.exe"

REQUIRED_IMPORTS = ["faster_whisper", "ctranslate2", "av", "numpy", "pypdf", "onnxruntime"]


def say(msg: str) -> None:
    print(msg, flush=True)


def find_python() -> str | None:
    if sys.version_info[:2] == (3, 14):
        return sys.executable
    for cand in ("python", "python3", "py"):
        exe = shutil.which(cand)
        if not exe:
            continue
        try:
            out = subprocess.run([exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                                 capture_output=True, text=True, timeout=30)
            if out.stdout.strip() == "3.14":
                return exe
        except Exception:
            pass
    return None


def run(args: list[str], desc: str) -> bool:
    say(f"  → {desc}")
    try:
        return subprocess.run(args, timeout=3600).returncode == 0
    except Exception as e:
        say(f"    실패: {type(e).__name__}: {e}")
        return False


def main() -> int:
    say("=" * 62)
    say(" Transcriber 환경 설치")
    say("=" * 62)
    say(f" 도구 폴더 : {BASE}")

    if not REQ.is_file():
        say(f"\n[오류] 패키지 목록이 없습니다: {REQ}")
        return 1

    if VENV_PY.is_file():
        # 이 PC에서 실제로 기동되는지까지 확인한다 — 옮겨온 venv 는 파일은 있어도
        # pyvenv.cfg 가 가리키는 파이썬이 없어 실행되지 않는다.
        p = subprocess.run([str(VENV_PY), "-c", "import sys"], capture_output=True)
        if p.returncode == 0:
            say(f"\n이미 쓸 수 있는 환경이 있습니다: {VENV_DIR}")
            return verify()
        say(f"\nvenv 가 있지만 이 PC에서 기동되지 않습니다 (다른 PC에서 만들어진 것).")
        say("  지우고 다시 만듭니다...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    base = find_python()
    if not base:
        say("\n[오류] Python 3.14 를 찾지 못했습니다.")
        say("  python.org 에서 3.14 를 설치한 뒤 다시 실행하세요.")
        return 1
    say(f" 바탕 파이썬: {base}")

    say("\n[1/3] 가상환경 생성")
    if not run([base, "-m", "venv", str(VENV_DIR)], f"venv → {VENV_DIR}"):
        return 1

    say("\n[2/3] 패키지 설치 (수 분 걸립니다)")
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"], "pip 갱신")
    if not run([str(VENV_PY), "-m", "pip", "install", "-r", str(REQ)], "requirements.txt 설치"):
        say("    설치에 실패했습니다. 위의 오류를 확인하세요.")
        return 1

    say("\n[3/3] 검증")
    return verify()


def verify() -> int:
    if not VENV_PY.is_file():
        say(f"  [오류] {VENV_PY} 가 없습니다.")
        return 1

    ok = True
    for mod in REQUIRED_IMPORTS:
        p = subprocess.run([str(VENV_PY), "-c", f"import {mod}"], capture_output=True)
        if p.returncode != 0:
            ok = False
        say(f"  {'OK ' if p.returncode == 0 else '실패'}  import {mod}")

    say("")
    # 외부 프로그램 — 폴더 밖 의존물이라 여기서 만들어 줄 수 없다
    ff = shutil.which("ffmpeg")
    say(f"  {'OK ' if ff else '★  '} FFmpeg: {ff or '없음 — 원본 직접 디코딩으로 폴백(느림)'}")

    tess = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if tess.is_file():
        say(f"  OK   Tesseract: {tess}")
    else:
        say(f"  ★    Tesseract 없음 — 슬라이드 글자 읽기 기능만 꺼집니다")

    models = ENGINE_DIR / "models"
    n = len(list(models.glob("*"))) if models.is_dir() else 0
    say(f"  {'OK ' if n else '   '} 모델 폴더: {models} ({n}개 항목)"
        + ("" if n else " — 첫 실행 때 자동으로 받습니다(인터넷 필요)"))

    say("")
    say("=" * 62)
    say(" 준비 완료 — 'Transcriber 실행.bat' 으로 시작하세요."
        if ok else " 위의 실패 항목을 해결한 뒤 다시 실행하세요.")
    say("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
