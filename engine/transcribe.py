# -*- coding: utf-8 -*-
"""
Transcriber 엔진
"MP4 입력" 폴더의 영상/음성 파일을 전사하여 "MD 출력" 폴더에 Markdown으로 저장한다.

로컬 faster-whisper 기반이므로 파일 용량 제한(25MB 등)이 없다.
"""

import configparser
import csv
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ENGINE_DIR = Path(__file__).resolve().parent
BASE = ENGINE_DIR.parent
IN_DIR = BASE / "MP4 입력"
OUT_DIR = BASE / "MD 출력"
PDF_DIR = BASE / "강의자료 PDF"
MODELS_DIR = ENGINE_DIR / "models"
TESSDATA_DIR = ENGINE_DIR / "tessdata"
CONFIG_FILE = BASE / "설정.ini"
LOG_FILE = BASE / "실행기록.txt"

# 작업용 임시 폴더도 이 폴더 안에 둔다 — 기본값(%TEMP%)을 쓰면 전사 중 WAV·프레임
# 이미지가 수백 MB씩 폴더 밖에 생기고, 강제 종료되면 그대로 남는다.
# 여기 두면 흔적이 도구 폴더 안에서만 생기고 cleanup_stale_temp() 가 회수한다.
TMP_ROOT = BASE / ".tmp"


def tmp_root():
    """임시 폴더의 부모를 돌려준다(없으면 만든다).

    쓰기가 막힌 위치(읽기 전용 매체 등)에 도구가 놓였을 때까지 실패시키지는
    않는다 — 그 경우에만 None 을 돌려 tempfile 기본값(%TEMP%)으로 물러선다.
    """
    try:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        return TMP_ROOT
    except OSError:
        return None

MEDIA_EXTS = {".mp4", ".m4a", ".mp3", ".wav", ".mkv", ".mov", ".webm",
              ".avi", ".flac", ".ogg", ".aac", ".wma", ".mpeg", ".mpg", ".mts", ".wmv",
              ".m4v", ".ts", ".opus", ".3gp", ".amr", ".mp2"}

# 엔진 로직 판(版). 문단화·OCR·출력 형식을 바꿀 때마다 올린다.
# 이 값이 산출물 지문에 들어가므로, 올리면 기존 MD가 자동으로 다시 만들어진다.
ENGINE_REV = 15

# 강의자료 PDF 연동 — 화면에서 읽은 글자는 "몇 쪽인가"를 알아내는 열쇠로만 쓰고,
# 실을 내용은 PDF 원문을 그대로 가져온다. OCR 잡음이 사라지고 표·빈칸이 보존된다.
PDF_MATCH_RATIO = 0.35   # 화면 글자의 2연쇄 중 이만큼이 그 쪽에 있으면 같은 슬라이드
PDF_MIN_WORDS = 5        # 2연쇄가 이보다 적은 화면은 맞대볼 근거가 부족하다
PDF_SHORT_KEY = 12       # 이보다 짧은 화면은 근거가 적으므로 더 높은 일치를 요구한다
PDF_SHORT_RATIO = 0.60
PDF_FORWARD_BONUS = 0.04  # 방금 본 쪽 근처를 조금 더 쳐준다 (강의는 앞에서 뒤로)
PDF_FORWARD_SPAN = 12
PDF_NAME_RATIO = 0.50    # 파일 이름이 이만큼 겹치면 같은 강의의 자료로 본다
HOTWORD_MAX = 320        # 전사에 넣어 줄 전문용어 문자열의 최대 길이
# 쪽(화면)의 이 비율 이상에 되풀이되는 줄은 머리글·꼬리말(학교 배너, 저작권 문구)이다
BOILERPLATE_RATIO = 0.7
BOILERPLATE_MIN = 6

# 문단 분리 기준 — 상한에 도달해도 문장이 끝날 때까지 기다린다(문장 중간 절단 방지)
PARA_GAP_SEC = 2.0       # 이 이상 침묵하면 새 문단
PARA_SOFT_SEC = 20.0     # 이 길이를 넘으면 다음 문장 끝에서 문단을 닫는다
PARA_SOFT_CHARS = 300
PARA_HARD_SEC = 45.0     # 문장 끝이 끝내 안 나올 때의 강제 상한
PARA_HARD_CHARS = 650
PHRASE_GAP_SEC = 1.0     # 낱말 사이가 이만큼 비면 구절을 나눈다
# '다.' '요.' '까?' 는 '.' '?' 에 이미 포함되므로 두지 않는다
SENTENCE_END = ('.', '!', '?', '"', "'", '”', '…')

# 잘린 입력 판정 — 추출된 오디오가 원본 길이의 이 비율 미만이면 손상으로 본다
TRUNCATION_TOLERANCE = 0.98
MARKER = "<!-- transcriber:"

# 음성 전사 — 순차 경로(온도 폴백 + 품질 게이트)의 인자. 실측으로 정했다.
VAD_PARAMS = {"min_silence_duration_ms": 500}
QUALITY = dict(condition_on_previous_text=False, compression_ratio_threshold=2.4,
               log_prob_threshold=-1.0, no_speech_threshold=0.6)

# 신뢰할 수 없는 전사 구간 판정 — 이 구간은 산출물에 표식을 남긴다
SUSPECT_LOGPROB = -0.9   # 평균 확률이 이보다 낮으면 인식이 흔들린 것
SUSPECT_NO_SPEECH = 0.6  # 말이 아닐 확률이 이보다 높은데 글이 나왔으면 의심
SUSPECT_REPEAT = 3       # 같은 문장이 이 횟수 이상 반복되면 환각

# 빠진 말소리 복구 — 파일 전체를 한 언어로 읽으면 다른 언어로 말한 대목이 오류 없이
# 통째로 사라진다(한국어 강의 속 영어 설명 22초가 한 문장만 남은 것을 실측). 말소리는
# 있는데 낱말이 없는 구간을 찾아 그 구간만 언어를 새로 정해 다시 읽는다.
GAP_MIN_SEC = 3.0        # 이보다 짧은 빈틈은 숨·기침이다
GAP_JOIN_SEC = 2.0       # 이만큼 가까운 빈틈은 한 번에 다시 읽는다 (한 번에 ~10초 든다)
GAP_PAD_SEC = 0.3
REPAIR_MAX = 30          # 파일당 다시 읽는 구간 수의 상한
REPAIR_MIN_CPS = 3.0     # 초당 글자가 이보다 적으면 복구가 아니라 환각이다
LOST_MIN_SEC = 8.0       # 복구하지 못한 빈틈이 이보다 길면 산출물에 알린다
RELANG_MIN_PROB = 0.7    # ⚠ 구간을 다른 언어로 갈아끼우려면 그 언어라고 이만큼 확신해야 한다
VIDEO_JOIN_SEC = 30      # 재생 영상 구간 사이의 틈이 이보다 짧으면 같은 영상으로 잇는다
HALLUCINATIONS = ("시청해주셔서 감사합니다", "시청해 주셔서 감사합니다", "구독과 좋아요",
                  "thank you for watching", "thanks for watching", "subtitles by")

# 슬라이드 읽기(OCR)
OCR_MIN_CONF = 60        # 이보다 확신이 낮은 줄은 사진 속 잡음으로 버린다
OCR_MIN_ALNUM = 0.55     # 글자 비율이 이보다 낮으면 잡음
OCR_BARE_CONF = 75       # 한글 낱말도 영어 낱말도 없는 줄(숫자·기호뿐)은 이만큼 확신할 때만 싣는다 —
                         # 사진 슬라이드의 잡음('- 31 47 26 58', 'Qe 4 「')은 62~69, 진짜 숫자 표 줄은 85 이상
MIX_PENALTY = 6          # 판본 고르기에서 글자종이 섞인 조각(굵은 한글→HAS) 하나당 깎는 점수 —
                         # 12면 영어 줄 속 한글 이름(섞인 판본)이 숫자로 깨진 kor 판본에 졌다.
                         # 12·8·6 은 정답 PDF 대비 정확도가 같고, 4 부터 떨어진다(실측)
SLIDE_CROP = 0.66        # 화자가 곁들여진 화면에서 슬라이드가 차지하는 좌측 비율
EDGE_TOUCH = 0.97        # 잘라 읽은 줄의 오른쪽 끝이 이만큼 가면 가장자리에서 잘린 줄이다
SPEAKER_CELL = 0.05      # 슬라이드가 멈춘 동안 이만큼 자주 움직인 칸은 화자가 서 있는 곳
SLIDE_MERGE_RATIO = 0.55  # 낱말이 이만큼 겹치면 같은 슬라이드로 본다
SLIDE_MERGE_LOOKBACK = 3  # 직전 몇 장까지 견주어 볼지 (애니메이션 단계 대응)
SUBTITLE_MATCH_RATIO = 0.6   # 발화와 이만큼 겹치는 짧은 화면 글자는 영상 자막
SUBTITLE_LIVE = 0.2      # 화면이 이만큼 움직이고 있을 때만 자막으로 본다 (영상 재생 중)
# 수식·기호는 글자로 센다 — 'f(x) = ax + b' 같은 줄이 잡음으로 버려지는 것을 막는다
OCR_SYMBOLS = set("+-=*/%^<>()[]{}|~.,:;'\"₩$€£°±×÷≤≥≠→←↑↓∙·")
# Tesseract 는 이 PC에서 사실상 단일 스레드다. 코어 수만큼 동시에 돌리면 5.8배 빠르고
# 출력은 바이트 단위로 같다(실측: 1·3·4·6·8·12개 모두 동일).
OCR_WORKERS = os.cpu_count() or 4

# 화면 전환 검출 — 1초마다 축소 회색 화면을 받아 '바뀐 화소의 비율'을 본다.
# ffmpeg 의 scene 점수(평균 차이)는 흰 바탕에 글자만 바뀌는 전환을 못 잡았다:
# 합성 강의에서 전환 8번 중 1번만 잡혀 슬라이드 2장이 통째로 빠지고 나머지는
# 30초 안전 샘플에 걸려 최대 20초 늦게 찍혔다. 바뀐 화소 비율로는 글자 한 줄이
# 늘어나는 전환이 0.45%, 정지 화면이 0.000%로 뚜렷이 갈린다.
SCAN_W, SCAN_H = 320, 240
PIX_DELTA = 24           # 밝기가 이보다 크게 변한 화소만 센다 (압축 잡음 제외)
CHANGE_RATIO = 0.0015    # 움직이지 않는 영역의 이 비율 이상이 바뀌면 새 화면
MIN_GAP_SEC = 2          # 연달아 잡지 않는다 — 넘기는 도중의 화면은 다음 비교가 잡는다
SETTLE_SEC = 3           # 전환 직후 화면은 흐릿할 수 있어 이만큼 뒤에 한 번 더 본다
LIVE_CELL = 16           # 움직임을 따지는 칸의 크기(축소 화면 화소)
LIVE_EMA = 0.1
LIVE_ON = 0.25           # 이만큼 자주 움직이는 칸은 화자 창·동영상으로 보고 뺀다
VIDEO_LIVE = 0.6         # 화면 대부분이 움직이면 영상 재생 중 — 자막을 촘촘히 본다
VIDEO_GAP_SEC = 3


def log(msg: str, echo: bool = True):
    if echo:
        print(msg)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")
    except OSError:
        pass


def status(msg: str):
    """한 줄을 덮어쓰는 진행 표시."""
    sys.stdout.write("\r" + msg.ljust(60))
    sys.stdout.flush()


def acquire_lock():
    """중복 실행 방지 잠금. 이미 실행 중이면 None, 잠글 수 없는 환경이면 False."""
    import msvcrt
    try:
        f = open(ENGINE_DIR / ".lock", "w")
    except OSError:
        return False
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        return f
    except OSError:
        f.close()
        return None


def load_config():
    cfg = {"model": "large-v3-turbo", "language": "auto", "beam_size": 1,
           "슬라이드_읽기": True, "ocr_언어": "자동"}
    if not CONFIG_FILE.exists():
        return cfg

    parser = configparser.ConfigParser(interpolation=None)
    try:
        try:
            parser.read(CONFIG_FILE, encoding="utf-8-sig")
        except UnicodeDecodeError:
            parser.read(CONFIG_FILE, encoding="cp949")
    except (configparser.Error, UnicodeDecodeError, OSError) as e:
        log(f"⚠ 설정.ini를 읽을 수 없어 기본값으로 진행합니다. ({e})")
        return cfg
    # 섹션 이름의 대소문자·공백을 가리지 않는다 — 예전에는 [Settings]면 설정 전체가
    # 조용히 무시되고 기본값으로 돌아갔다.
    section = next((n for n in parser.sections()
                    if n.strip().lower() == "settings"), None)
    if section is None:
        log(f"⚠ 설정.ini에 [settings] 섹션이 없어 기본값으로 진행합니다. "
            f"(찾은 섹션: {', '.join(parser.sections()) or '없음'})")
        return cfg
    s = parser[section]

    valid_models = {"tiny", "base", "small", "medium", "large-v1", "large-v2",
                    "large-v3", "large", "large-v3-turbo", "turbo",
                    "distil-large-v3", "distil-large-v3.5"}
    model = s.get("model", cfg["model"]).strip()
    if model in valid_models or re.fullmatch(r"[\w.-]+/[\w.-]+", model):
        cfg["model"] = model
    elif model:
        log(f"⚠ 설정.ini의 model '{model}'을 알 수 없어 {cfg['model']}을 사용합니다.")

    lang = s.get("language", cfg["language"]).strip().lower()
    if lang in ("", "auto"):
        cfg["language"] = "auto"
    else:
        # 언어 목록은 무거운 전사 라이브러리에 들어 있다 — 필요할 때만 불러온다
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        if lang in _LANGUAGE_CODES:
            cfg["language"] = lang
        else:
            log(f"⚠ 설정.ini의 language '{lang}'은 올바른 언어 코드가 아닙니다. "
                f"auto / ko / en / ja / zh 중에서 골라주세요. 이번에는 auto로 진행합니다.")

    try:
        v = s.getint("beam_size", cfg["beam_size"])
        if not 1 <= v <= 10:
            log(f"⚠ 설정.ini의 beam_size={v}는 허용 범위(1~10)를 벗어나 {min(max(v, 1), 10)}로 조정합니다.")
        cfg["beam_size"] = min(max(v, 1), 10)
    except ValueError:
        log(f"⚠ 설정.ini의 beam_size 값이 숫자가 아니라 기본값({cfg['beam_size']})을 사용합니다.")

    # 알 수 없는 값을 조용히 무시하지 않는다 — 예전에는 '슬라이드_읽기 = false' 가
    # 경고 없이 '켬'으로 처리되어 사용자가 왜 안 꺼지는지 알 수 없었다.
    raw = s.get("슬라이드_읽기", "켬").strip().lower()
    if raw in ("켬", "on", "yes", "끔", "off", "no"):
        cfg["슬라이드_읽기"] = raw in ("켬", "on", "yes")
    else:
        log(f"⚠ 설정.ini의 슬라이드_읽기 '{raw}'를 알 수 없어 켬(으)로 진행합니다. "
            f"쓸 수 있는 값: 켬 / 끔")

    want = (s.get("슬라이드_언어", "자동").strip() or "자동")
    if want.lower() not in ("자동", "auto"):
        missing = [c for c in want.split("+")
                   if c and not (TESSDATA_DIR / f"{c}.traineddata").exists()]
        if missing:
            log(f"⚠ 슬라이드_언어 '{want}'의 언어 데이터가 없습니다 "
                f"({', '.join(m + '.traineddata' for m in missing)}). 자동으로 진행합니다.")
        else:
            cfg["ocr_언어"] = want
    return cfg


def config_fingerprint(cfg):
    slide = f"|slide{cfg['ocr_언어']}" if cfg["슬라이드_읽기"] else ""
    # 엔진 판을 함께 넣는다 — 로직을 고치면 기존 산출물이 자동으로 갱신된다
    return f"rev{ENGINE_REV}|{cfg['model']}|{cfg['language']}|beam{cfg['beam_size']}{slide}"


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def probe_media(path: Path):
    """ffprobe 한 번으로 원본 길이·오디오 길이·영상 유무를 얻는다.

    오디오 길이를 따로 보는 이유: 잘림 검사는 반드시 오디오 트랙과 견주어야 한다.
    컨테이너 길이는 영상 트랙을 따르므로, 끝에 무음 화면이 붙은 정상 녹화가
    '손상'으로 거부되는 일이 있었다. mp3에 붙은 앨범 표지 한 장은 영상이 아니다.
    """
    info = {"duration": None, "audio_duration": None, "video": False}
    exe = shutil.which("ffprobe")
    if not exe:
        return info
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,duration:stream_disposition=attached_pic",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, stdin=subprocess.DEVNULL)
        data = json.loads(r.stdout)
    except Exception:
        return info
    try:
        info["duration"] = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        pass
    for st in data.get("streams", []):
        kind = st.get("codec_type")
        if kind == "audio" and info["audio_duration"] is None:
            try:
                info["audio_duration"] = float(st["duration"])
            except (KeyError, TypeError, ValueError):
                pass
        elif kind == "video" and not (st.get("disposition") or {}).get("attached_pic"):
            info["video"] = True
    return info


def extract_audio(src: Path, dst: Path, media):
    """16kHz mono WAV로 추출.

    성공하면 True, ffmpeg가 없으면 None(원본 직접 디코딩으로 폴백).
    잘린 입력이나 추출 실패는 예외로 알려 조용한 절단을 막는다.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    timeout = max(600, (media["duration"] or 0) * 2)
    try:
        r = subprocess.run(
            [exe, "-nostdin", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("오디오 추출이 응답하지 않아 중단했습니다 (파일이 손상되었을 수 있습니다)")
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
        detail = (r.stderr or "").strip().splitlines()
        raise RuntimeError("이 파일에서 오디오를 읽을 수 없습니다"
                           + (f" — {detail[-1][:160]}" if detail else ""))

    got = (dst.stat().st_size - 44) / 32000        # 16kHz·mono·16bit
    # 오디오 트랙 길이와만 견준다. 영상이 더 긴 것은 정상이다(끝에 붙은 무음 화면 등).
    expect = media["audio_duration"]
    if expect and got < expect * TRUNCATION_TOLERANCE:
        raise RuntimeError(
            f"오디오가 {fmt_ts(expect)}인데 {fmt_ts(got)}까지만 읽혔습니다. "
            f"파일이 손상되었거나 복사가 끝나지 않았습니다")
    return True


def safe_stem(name: str, limit: int = 120) -> str:
    return name if len(name) <= limit else name[:limit].rstrip() + "~"


# ────────────────────────── 음성 전사 ──────────────────────────

def asr_threads():
    """전사 스레드 수. 물리 6코어/논리 12에서 실측한 값(10)을 코어 수에 맞춰 일반화한다.

    생산과 같은 인자로 240초 강의를 교차 6라운드 잰 결과 8스레드 56.72초 →
    10스레드 52.15초(+8.77%), 6/6 전승. 12는 다시 나빠진다 — 논리코어를 다 쓰면
    하이퍼스레드 경합으로 손해다. 코어가 적은 PC에서는 전부 쓴다.
    """
    n = os.cpu_count() or 4
    return n - 2 if n >= 8 else n


def segment_words(seg, offset=0.0):
    return [(w.start + offset, w.end + offset, w.word) for w in (seg.words or [])]


def is_suspect(seg) -> bool:
    """인식이 흔들린 구간. 매끄러운 문장으로 된 환각은 사람도 LLM도 알아볼 수 없으므로
    여기서 잡아 표식을 남긴다."""
    return ((getattr(seg, "avg_logprob", None) or 0) < SUSPECT_LOGPROB
            or (getattr(seg, "no_speech_prob", None) or 0) > SUSPECT_NO_SPEECH)


def reliable_words(words):
    """세그먼트 앞머리의 외톨이 짧은 낱말을 뺀다.

    Whisper 세그먼트가 빈 구간을 끼고 시작하면 첫 낱말('이 표는'의 '이')의 시각이
    수십 초 앞으로 끌려간다. 그 낱말을 믿으면 빈틈이 가려져 복구가 덜 된다.
    """
    i = 0
    while (i + 1 < len(words) and len(words[i][2].strip()) < 3
           and words[i + 1][0] - words[i][1] >= PHRASE_GAP_SEC):
        i += 1
    return words[i:]


def find_gaps(speech, spans, min_len=GAP_MIN_SEC, join=GAP_JOIN_SEC, pad=GAP_PAD_SEC):
    """말소리 구간(speech) 가운데 낱말(spans)이 덮지 않은 곳. 가까운 것은 합친다.

    둘 다 (시작초, 끝초) 목록이다.
    """
    covered = []
    for s, e in sorted((s - pad, e + pad) for s, e in spans):
        if covered and s <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], e)
        else:
            covered.append([s, e])
    holes, j = [], 0
    for a, b in sorted(speech):
        while j < len(covered) and covered[j][1] <= a:
            j += 1
        cur, k = a, j
        while k < len(covered) and covered[k][0] < b:
            if covered[k][0] > cur:
                holes.append((cur, covered[k][0]))
            cur = max(cur, covered[k][1])
            k += 1
        if cur < b:
            holes.append((cur, b))
    merged = []
    for a, b in holes:
        if b - a < 0.5:
            continue
        if merged and a - merged[-1][1] <= join:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def repair_gaps(model, audio, segments, cfg, sr=16000):
    """말소리는 있는데 전사가 비어 있는 구간을 언어를 새로 정해 다시 읽는다.

    돌려주는 값은 (보탤 세그먼트들, 복구한 구간들, 복구하지 못한 긴 구간들).
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    speech = [(c["start"] / sr, c["end"] / sr)
              for c in get_speech_timestamps(audio, VadOptions(**VAD_PARAMS))]
    # 낱말 시각이 없는 세그먼트는 세그먼트 전체를 덮은 것으로 본다
    spans = [span for seg in segments
             for span in ([(w[0], w[1]) for w in reliable_words(seg[4])] or [(seg[0], seg[1])])]
    gaps = find_gaps(speech, spans)
    added, fixed, lost = [], [], []
    for n, (a, b) in enumerate(gaps):
        if n >= REPAIR_MAX:
            lost += [g for g in gaps[n:] if g[1] - g[0] >= LOST_MIN_SEC]
            break
        status(f"  빠진 말소리 다시 읽는 중... [{fmt_ts(a)}] {n + 1}/{len(gaps)}")
        lo = max(0.0, a - GAP_PAD_SEC)
        clip = audio[int(lo * sr):int((b + GAP_PAD_SEC) * sr)]
        try:
            segs, info = model.transcribe(clip, beam_size=cfg["beam_size"], vad_filter=True,
                                          vad_parameters=VAD_PARAMS, word_timestamps=True,
                                          **QUALITY)
            segs = [s for s in segs if not is_suspect(s)]
        except Exception:
            segs, info = [], None
        got = []
        for s in segs:
            # 이미 전사된 낱말과 겹치는 가장자리는 버린다 (중복 방지)
            words = [w for w in segment_words(s, lo) if a <= (w[0] + w[1]) / 2 <= b]
            text = "".join(w[2] for w in words).strip()
            if text:
                got.append((words[0][0], words[-1][1], text, False, words))
        body = re.sub(r"\s", "", "".join(g[2] for g in got))
        ok = (info is not None and info.language_probability >= 0.6
              and len(body) >= REPAIR_MIN_CPS * (b - a)
              and not any(h in body.lower().replace(" ", "")
                          for h in (x.replace(" ", "") for x in HALLUCINATIONS)))
        if ok:
            added += got
            fixed.append((a, b))
        elif b - a >= LOST_MIN_SEC:
            lost.append((a, b))
    if gaps:
        status("")
        sys.stdout.write("\r")
    return added, fixed, lost


def suspect_spans(segments, join=GAP_JOIN_SEC, min_len=GAP_MIN_SEC):
    """⚠ 세그먼트가 이어진 구간들 (가까우면 합친다)."""
    spans = []
    for s in segments:
        if not s[3]:
            continue
        if spans and s[0] - spans[-1][1] <= join:
            spans[-1][1] = max(spans[-1][1], s[1])
        else:
            spans.append([s[0], s[1]])
    return [(a, b) for a, b in spans if b - a >= min_len]


def reread_suspects(model, audio, segments, main_lang, cfg, sr=16000):
    """⚠ 구간을 언어를 새로 정해 다시 읽고, 다른 언어로 확신 있게 읽히면 갈아끼운다.

    파일 언어로 고정해 읽으면 다른 언어로 말한 대목이 빠지기도 하지만(repair_gaps),
    그 언어의 **그럴듯한 환각으로 채워지기도** 한다 — 실강의(영어 강의 속 한국어 뉴스
    영상)에서 한국어 말소리 자리에 그럴듯한 영어 헛문장이 9분간
    이어졌다. 빈틈이 없으니 복구 장치가 못 잡는다. ⚠ 는 붙어 있으므로 그 구간을 다시 읽는다.
    같은 언어로 나오면 진짜 음질 문제이므로 그대로 둔다(⚠ 유지).

    돌려주는 값은 (새 세그먼트 목록, 바꾼 구간 [(시작, 끝, 언어)]).
    """
    swapped = []
    for a, b in suspect_spans(segments)[:REPAIR_MAX]:
        status(f"  다른 언어로 말한 대목인지 다시 읽는 중... [{fmt_ts(a)}]")
        lo = max(0.0, a - GAP_PAD_SEC)
        clip = audio[int(lo * sr):int((b + GAP_PAD_SEC) * sr)]
        try:
            segs, info = model.transcribe(clip, beam_size=cfg["beam_size"], vad_filter=True,
                                          vad_parameters=VAD_PARAMS, word_timestamps=True,
                                          **QUALITY)
            segs = list(segs)
        except Exception:
            continue
        good = [s for s in segs if not is_suspect(s)]
        if (info.language == main_lang or info.language_probability < RELANG_MIN_PROB
                or not segs or len(good) < 0.7 * len(segs)):
            continue
        new = []
        for s in good:
            words = [w for w in segment_words(s, lo) if a - 1 <= (w[0] + w[1]) / 2 <= b + 1]
            text = "".join(w[2] for w in words).strip()
            if text and not any(h in text.lower().replace(" ", "")
                                for h in (x.replace(" ", "") for x in HALLUCINATIONS)):
                new.append((words[0][0], words[-1][1], text, False, words))
        kept = [s for s in segments if not (a <= s[0] <= b and s[3])]
        # 구간 안에 남은 확실한 세그먼트(교수의 짧은 말 등)와 겹치는 낱말은 버린다
        inside = [(s[0], s[1]) for s in kept if a <= s[0] <= b]
        new = [n for n in new if not any(x <= (n[0] + n[1]) / 2 <= y for x, y in inside)]
        if not new:
            continue
        segments = kept + new
        swapped.append((a, b, info.language))
    status("")
    sys.stdout.write("\r")
    return sorted(segments, key=lambda s: s[0]), swapped


def reread_video_suspects(model, audio, segments, spans, foreign, cfg, sr=16000):
    """다른 언어 영상 안에 남은 ⚠ 구간을 그 영상의 언어로 고정해 다시 읽는다.

    reread_suspects 는 언어를 새로 정하는데, 음악이 깔린 짧은 조각은 교수의 언어로 판별되어
    재독이 버려졌다 — 영어 강의 속 한국어 영상 3분이 영어 헛문장(⚠)으로 남았다(실강의).
    그 영상이 어느 언어인지는 이미 안다(다른 언어로 읽힌 말이 있는 영상 구간). 그 안의 ⚠
    구간만 그 언어로 읽는다 — 교수가 영상 중간에 한 말은 대개 또렷해 ⚠ 가 아니다.
    돌려주는 값은 (새 세그먼트 목록, 바꾼 구간 [(시작, 끝, 언어)]).
    """
    done = []
    for a, b in spans:
        langs = Counter(lang for x, y, lang in foreign if x < b and a < y)
        if not langs:
            continue
        lang = langs.most_common(1)[0][0]
        for x, y in suspect_spans([s for s in segments if a <= s[0] < b]):
            if len(done) >= REPAIR_MAX:
                break
            status(f"  영상 속 흔들린 말을 영상의 언어({lang})로 다시 읽는 중... [{fmt_ts(x)}]")
            lo = max(0.0, x - GAP_PAD_SEC)
            clip = audio[int(lo * sr):int((y + GAP_PAD_SEC) * sr)]
            try:
                segs = list(model.transcribe(clip, language=lang, beam_size=cfg["beam_size"],
                                             vad_filter=True, vad_parameters=VAD_PARAMS,
                                             word_timestamps=True, **QUALITY)[0])
            except Exception:
                continue
            good = [s for s in segs if not is_suspect(s)]
            if not segs or len(good) < 0.7 * len(segs):
                continue
            new = []
            for s in good:
                words = [w for w in segment_words(s, lo) if x - 1 <= (w[0] + w[1]) / 2 <= y + 1]
                text = "".join(w[2] for w in words).strip()
                if text and not any(h in text.lower().replace(" ", "")
                                    for h in (z.replace(" ", "") for z in HALLUCINATIONS)):
                    new.append((words[0][0], words[-1][1], text, False, words))
            if not new:
                continue
            segments = [s for s in segments if not (x <= s[0] <= y and s[3])] + new
            done.append((x, y, lang))
    if done:
        status("")
        sys.stdout.write("\r")
    return sorted(segments, key=lambda s: s[0]), done


def split_phrases(segments, cuts=()):
    """세그먼트를 낱말 시각으로 짧은 구절로 나눈다. (시작, 끝, 글, 의심) 목록.

    전문용어 힌트를 넣으면 Whisper 세그먼트가 24초짜리로 뭉개진다(실측: 24개 → 7개).
    구절 단위로 나눠야 슬라이드가 바뀐 시점에서 문단을 끊을 수 있다.
    cuts(재생 영상이 시작·끝난 시각)에서도 끊는다 — 교수가 말을 마치자마자 영상이 시작되면
    쉼 없이 한 구절로 묶여, 영상의 첫마디가 교수의 말로 실리거나 영상 뒤 교수의 말이 📺 로
    실렸다(실강의). 글자가 하나도 없는 구절('... ...')은 버린다.
    """
    cuts = sorted(cuts)
    out = []

    def emit(s, e, text, bad):
        if any(ch.isalnum() for ch in text):
            out.append((s, e, text.strip(), bad))

    for seg in segments:
        s, e, text, bad = seg[0], seg[1], seg[2], seg[3]
        words = seg[4] if len(seg) > 4 else None
        if not words:
            if text.strip():
                emit(s, e, text, bad)
            continue
        cur, cs, ce = "", None, None
        for ws, we, w in words:
            crossed = cs is not None and any(cs < c <= ws for c in cuts)
            if cs is not None and not crossed and ws - ce >= PHRASE_GAP_SEC and len(cur.strip()) < 3:
                # 세그먼트 앞머리의 외톨이 낱말('이 표는'의 '이')은 시각이 부정확하다 —
                # 앞 문단에 붙지 않도록 뒤 낱말들과 한 구절로 묶고 그 시각을 쓴다
                cs = ws
            elif cs is not None and (crossed or ws - ce >= PHRASE_GAP_SEC or ends_sentence(cur)):
                emit(cs, ce, cur, bad)
                cur, cs = "", None
            if cs is None:
                cs = ws
            cur += w
            ce = we
        if cur.strip():
            emit(cs, ce, cur, bad)
    return out


# ────────────────────────── 슬라이드 읽기 (OCR) ──────────────────────────

def find_tesseract():
    # 도구 안에 언어 데이터를 따로 두었으면 그쪽을 쓰게 한다 (시스템 설치를 건드리지 않음)
    if (TESSDATA_DIR / "eng.traineddata").exists():
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(p).exists():
            return p
    return None


def ocr_lang_options(cfg, spoken_language: str):
    """슬라이드를 읽을 언어 후보들. 여러 개면 화면마다 잘 읽힌 쪽을 골라 쓴다.

    실측으로 정한 전략이다. 'kor+eng'로 읽으면 굵은 한글이 라틴 낱말로 오인된다
    (질문을→HAS, 관계를→AAS, 말을→SS, 지원내용→AMY). 같은 이미지를 'kor' 단독으로
    읽으면 본문 오류가 사라진다. 반대로 영어 슬라이드는 'eng'가 맞다.
    그래서 섞지 않고 따로 읽어 본 뒤 고른다 — 영어 강의에도 한국어 슬라이드가 섞여
    나오므로 발화 언어로 후보를 자르지 않는다.

    'kor+eng' 도 세 번째 후보로 읽는다. 한 줄에 두 언어가 섞이면 단독 판본은 어느 쪽이든
    반대 언어를 깨뜨린다 — 한국어 슬라이드 괄호 속 영어 용어는 숫자열이 되고, 영어 슬라이드의
    한국 이름은 기호가 됐다(실강의). 섞인 판본이 굵은 한글을 라틴으로 오인한 줄은 글자종 섞임
    벌점으로 떨어진다(실측: 정답 PDF 대비 재현율 0.875→0.894, 영어 강의에서 한글 이름·용어가
    든 줄 8개 복원).
    """
    want = cfg["ocr_언어"].strip()
    if want.lower() not in ("자동", "auto", ""):
        return [want]
    if not (TESSDATA_DIR / "kor.traineddata").exists():
        return ["eng"]
    return (["kor", "eng"] if spoken_language == "ko" else ["eng", "kor"]) + ["kor+eng"]


def script_mix_penalty(text: str) -> int:
    """한글 줄에 낀 라틴 조각(과 그 반대)의 개수. 오인식의 지표다."""
    han = sum(1 for ch in text if "가" <= ch <= "힣")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if han == lat:
        return 0
    bad = 0
    for tok in text.split():
        if not any(c.isalnum() for c in tok) or len(tok) > 4:
            continue
        t_han = any("가" <= c <= "힣" for c in tok)
        t_lat = any(c.isascii() and c.isalpha() for c in tok)
        if han > lat and t_lat and not t_han:
            bad += 1
        elif lat > han and t_han and not t_lat:
            bad += 1
    return bad


def ocr_score(lines) -> float:
    """읽어낸 글자의 양에서 글자종 섞임(오인식)을 벌점으로 뺀 점수."""
    return sum(len(t) - script_mix_penalty(t) * 12 for t in lines)


def bare_line(text: str) -> bool:
    """한글 두 음절 이상의 낱말도, 영어 세 글자 이상의 낱말도 없는 줄(숫자·기호뿐)인가."""
    return not re.search(r"[가-힣]{2}|[A-Za-z]{3}", text)


def ocr_lines(tess: str, img: Path, langs: str):
    """이미지에서 글자를 읽되, 확신이 낮은 줄(슬라이드 속 사진 등)은 버린다.

    한 번만 인식해서 두 가지 형식을 함께 받는다. 줄별 확신도는 tsv에서 가져오고,
    글자는 txt에서 가져온다 — 한글은 Tesseract가 음절 단위로 끊어 좌표만으로는
    띄어쓰기를 되살리기 어렵지만, txt 출력에는 이미 제대로 반영되어 있다.
    """
    with tempfile.TemporaryDirectory(prefix="ocr_", dir=tmp_root()) as td:
        base = Path(td) / "page"
        # 여러 개를 동시에 돌리므로 한 프로세스 안에서 스레드를 더 벌리지 않게 한다.
        # ★ 이 값은 Tesseract 에만 넘긴다. 우리 프로세스 전역에 걸면 음성 인식 엔진
        #   (CTranslate2)이 1스레드로 떨어지면서 결과까지 망가진다 — 실측: 40초 처리에
        #   111초, 한국어를 en 으로 오판하고 "Or or or" 를 쏟아냈다.
        r = subprocess.run([tess, str(img), str(base), "-l", langs, "--psm", "6",
                            "tsv", "txt"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", stdin=subprocess.DEVNULL, timeout=180,
                           env=dict(os.environ, OMP_THREAD_LIMIT="1"))
        # 실패를 "글자가 없었다"로 위장하지 않는다 — 언어 데이터가 없거나 설정이
        # 잘못되면 슬라이드가 통째로 빠진 MD가 정상처럼 저장되었다.
        if r.returncode != 0:
            detail = (r.stderr or "").strip().splitlines()
            raise RuntimeError(f"Tesseract 실패(-l {langs})"
                               + (f" — {detail[-1][:160]}" if detail else ""))
        try:
            tsv = base.with_suffix(".tsv").read_text(encoding="utf-8", errors="replace")
            txt = base.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise RuntimeError(f"Tesseract 결과 파일을 읽지 못했습니다 ({e})") from e

    order, confs, words, tops = [], defaultdict(list), defaultdict(list), defaultdict(list)
    rights, bottoms, page_w = defaultdict(list), defaultdict(list), 0
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") == "1":
            page_w = int(row.get("width") or 0)
        try:
            conf = float(row["conf"])
        except (ValueError, TypeError, KeyError):
            continue
        word = (row.get("text") or "").strip()
        if conf >= 0 and word:
            key = (row["block_num"], row["par_num"], row["line_num"])
            if key not in confs:
                order.append(key)
            confs[key].append(conf)
            words[key].append(word)
            try:
                tops[key].append(int(row["top"]))
                rights[key].append(int(row["left"]) + int(row["width"]))
                bottoms[key].append(int(row["top"]) + int(row["height"]))
            except (ValueError, TypeError, KeyError):
                pass

    txt_lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(txt_lines) != len(order):        # 어긋나면 tsv 낱말을 이어 붙여 쓴다
        txt_lines = [" ".join(words[k]) for k in order]

    out = []
    for key, line in zip(order, txt_lines):
        text = re.sub(r"\s{2,}", " ", line).strip()
        body = [ch for ch in text if not ch.isspace()]
        if not body:
            continue
        # 수식·기호도 내용이다. 공백은 분모에서 뺀다 — 예전에는 'f(x) = ax + b',
        # '10% -> 25%', 'GDP', 'Q&A' 같은 줄이 전부 잡음으로 버려졌다.
        content = sum(ch.isalnum() or ch in OCR_SYMBOLS for ch in body)
        conf = sum(confs[key]) / len(confs[key])
        if (conf >= (OCR_BARE_CONF if bare_line(text) else OCR_MIN_CONF)
                and len(text) >= 2 and content / len(body) >= OCR_MIN_ALNUM):
            # 넷째 값은 줄의 오른쪽 끝(이미지 폭 대비) — 잘라 읽은 가장자리에 닿았는지 본다.
            # 다섯째 값은 줄의 아랫변 — 판본끼리 같은 줄인지 세로로 겹치는 정도로 가린다
            right = max(rights[key]) / page_w if rights[key] and page_w else 0.0
            top = min(tops[key]) if tops[key] else 0
            out.append((top, text, conf, right, max(bottoms[key]) if bottoms[key] else top))
    out.sort(key=lambda x: x[0])
    return out


def line_score(line) -> float:
    """줄 하나의 품질 점수. (top, 글자, 확신도) 를 받는다.

    Tesseract의 확신도를 주로 본다 — 같은 영어 줄의 두 판본처럼 글자종이 같을 때는
    길이로는 좋고 나쁨을 가릴 수 없고(깨진 쪽이 더 길 수도 있다) 확신도만이 신호다.
    거기에 두 가지를 더한다.
      · 글자종 섞임은 벌점 — 굵은 한글이 라틴으로 오인된 판본을 떨어뜨린다.
      · 한 줄에 두 언어가 섞이면(예: '©2023. 가상마을신문. All rights reserved.')
        어느 판본을 골라도 반대 언어는 깨지므로 **한글을 살린다.** 고유명사·출처는
        내용이고 반대쪽에서 깨지는 것은 대개 정형 문구다. 영어 판본은 한글을 만들어
        내지 못하니, 한글이 두 자 이상 남았다는 건 실제로 읽어냈다는 뜻이다.
    """
    text, conf = line[1], (line[2] if len(line) > 2 else OCR_MIN_CONF)
    han = sum(1 for ch in text if "가" <= ch <= "힣")
    return (conf - script_mix_penalty(text) * MIX_PENALTY + (8 if han >= 2 else 0)
            + len(text) * 0.2)


def ocr_best(tess: str, img: Path, lang_options):
    """후보 언어로 각각 읽어 본 뒤 **줄 단위로** 잘 읽힌 쪽을 고른다.

    화면 하나를 통째로 한 언어에 맡기면, 한국어 슬라이드에 섞인 영어(출처 표기,
    'All rights reserved', 약어)가 함께 깨진다. 그래서 같은 높이에 있는 줄끼리
    맞대어 놓고 줄마다 더 나은 쪽을 뽑는다. 돌려주는 값은 고른 줄(ocr_lines 형식)이다.
    """
    return merge_ocr_passes([ocr_lines(tess, img, langs) for langs in lang_options], full=True)


def same_row(a, b, tol: int = 25) -> bool:
    """두 판본의 줄이 같은 줄인가 — 세로로 절반 넘게 겹치면 같다(아랫변이 없으면 윗변 차이)."""
    if len(a) > 4 and len(b) > 4 and a[4] > a[0] and b[4] > b[0]:
        overlap = min(a[4], b[4]) - max(a[0], b[0])
        return overlap >= 0.5 * min(a[4] - a[0], b[4] - b[0])
    return abs(a[0] - b[0]) <= tol


def merge_ocr_passes(passes, tol: int = 25, full: bool = False):
    """여러 번 읽은 결과를 줄 높이로 맞대어, 줄마다 잘 읽힌 쪽을 남긴다.
    full 이면 글자만이 아니라 고른 줄 전체(윗변, 글자, 확신도, 오른쪽 끝, 아랫변)를 돌려준다.

    같은 줄인지는 세로 겹침으로 본다 — 고정된 윗변 차이(25화소)는 2배로 키운 1080p 화면의
    큰 글자(높이 80화소 넘음)에는 좁다. 아랫변이 없는 입력은 예전처럼 윗변 차이로 본다."""
    passes = [p for p in passes if p]
    pick = (lambda ln: ln) if full else (lambda ln: ln[1])
    if not passes:
        return []
    if len(passes) == 1:
        return [pick(ln) for ln in passes[0]]

    merged, idx = [], [0] * len(passes)
    while True:
        live = [(passes[i][idx[i]][0], i)
                for i in range(len(passes)) if idx[i] < len(passes[i])]
        if not live:
            return merged
        base = passes[min(live)[1]][idx[min(live)[1]]]
        group = []
        for _top, i in live:
            if same_row(passes[i][idx[i]], base, tol):
                group.append(passes[i][idx[i]])
                idx[i] += 1
        merged.append(pick(max(group, key=line_score)))


def extend_cut_lines(lines, wide, edge: float = EDGE_TOUCH):
    """잘라 읽어 가장자리에서 끊긴 줄을, 넓게 읽은 판본의 같은 줄로 늘린다.

    화자가 슬라이드 위에 겹쳐 선 강의는 좌측만 잘라 읽으면 화자 위쪽을 지나는 긴 줄의
    끝이 잘렸다(실강의: 마지막 낱말이 반쯤 잘린 줄이 쪽마다 나왔다). 그렇다고
    화면 전체를 읽으면 어두운 제목 띠가 로고 쪽 흰 바탕과 한 줄로 묶여 제목이 통째로
    사라지고, 줄 높이로 두 판본을 섞으면 같은 줄이 두 번 들어갔다(실측: 정확도 0.87→0.82).
    그래서 **가장자리에 닿은 줄만**, 넓은 판본에서 그 줄로 시작하면서 더 긴 줄이 있으면
    **잘린 뒤꼬리만** 이어 붙인다 — 가장자리에 닿지 않은 줄에 로고·화자 조각이 붙는 일이 없고,
    앞부분은 좌측 판본을 그대로 둔다(넓은 판본으로 통째로 바꾸면 그 판본의 오독이 앞부분까지
    들어왔다). 경계에 걸려 반쯤 잘린 글자는 좌측 판본에서 틀리게 읽히므로, 두 판본이 마지막으로
    맞는 곳부터 넓은 판본을 쓴다.
    lines 는 잘라 읽은 줄(ocr_lines 형식), wide 는 넓게 읽은 줄의 글자 목록이다.
    """
    import difflib

    def squash(s):
        pos = [i for i, ch in enumerate(s) if not ch.isspace()]
        return "".join(s[i] for i in pos), pos

    out = []
    for ln in lines:
        text = ln[1]
        if len(ln) > 3 and ln[3] >= edge:
            head, head_pos = squash(text)
            best = None
            for cand in wide:
                body, body_pos = squash(cand)
                if (len(body) <= len(head)
                        or difflib.SequenceMatcher(None, head, body[:len(head)]).ratio() < 0.7):
                    continue
                blocks = [m for m in difflib.SequenceMatcher(None, head, body).get_matching_blocks()
                          if m.size >= 2]
                if not blocks:
                    continue
                a_end, b_end = blocks[-1].a + blocks[-1].size, blocks[-1].b + blocks[-1].size
                if sum(ch.isdigit() or (ch.isascii() and ch.isalpha()) or "가" <= ch <= "힣"
                       for ch in body[b_end:]) < 2:
                    continue                      # 더 읽힌 글자가 없거나 조각('ㅣ')뿐이다
                if best is None or len(body) - b_end > best[0]:
                    cut = head_pos[a_end - 1] + 1
                    start = body_pos[b_end]
                    glue = " " if start > 0 and cand[start - 1].isspace() else ""
                    best = (len(body) - b_end, text[:cut] + glue + cand[start:])
            if best:
                text = best[1]
        out.append(text)
    return out


def grab_frame(ff: str, src: Path, t: float, vf: str, png: Path) -> bool:
    """t초의 화면 한 장을 뽑는다. 여러 개를 동시에 돌리므로 프로세스마다 한 스레드."""
    subprocess.run([ff, "-nostdin", "-y", "-v", "error", "-threads", "1", "-ss", f"{t:.3f}",
                    "-i", str(src), "-frames:v", "1", "-vf", vf, str(png)],
                   capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
    return png.exists()


def read_screen(ff, tess, src, t, crop, png, langs, wide=""):
    """t초의 화면을 2배로 키워 읽는다(2배가 원본보다 낫고 3배와는 차이가 없다).

    좌측만 잘라 읽었는데 가장자리에서 끊긴 줄이 있으면, 화자만 가린 넓은 화면(wide)을
    한 번 더 읽어 그 줄을 늘린다 — 끊긴 줄이 없는 화면은 한 번만 읽는다.
    """
    try:
        if not grab_frame(ff, src, t, f"{crop}scale=iw*2:ih*2", png):
            return []
        lines = ocr_best(tess, png, langs)
        if crop and wide and any(ln[3] >= EDGE_TOUCH for ln in lines):
            png.unlink(missing_ok=True)
            if grab_frame(ff, src, t, f"{wide}scale=iw*2:ih*2", png):
                return extend_cut_lines(lines, [ln[1] for ln in ocr_best(tess, png, langs)])
        return [ln[1] for ln in lines]
    except subprocess.TimeoutExpired:      # 한 장이 느려도 전체를 포기하지는 않는다
        return []
    finally:
        png.unlink(missing_ok=True)


def background_gray(ff: str, src: Path, t: float) -> int:
    """t초 화면의 배경 밝기(작게 줄인 회색 화면의 중앙값). 못 구하면 흰색."""
    r = subprocess.run([ff, "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src),
                        "-frames:v", "1", "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    return sorted(r.stdout)[len(r.stdout) // 2] if r.stdout else 255


def speaker_mask(calm, cut_col: int, fill: int = 255) -> str:
    """화자가 서 있는 칸을 채워 가리는 ffmpeg 필터(없으면 빈 문자열).

    calm 은 슬라이드 쪽이 멈춘 동안 칸마다 움직인 비율이다. 오른쪽에서 가장 자주 움직인
    칸과 이어진 칸들을 화자로 보고, 줄마다 그 칸들을 상자로 덮는다 — 사각형 하나로
    덮으면 화자 머리 옆을 지나는 줄까지 가려졌다(화자는 위가 좁고 아래가 넓다).
    상자는 슬라이드 배경 밝기(fill)로 칠한다 — 흰색·검은색으로 칠하면 옆을 지나는 줄까지
    더 깨져 읽혔다(합성 강의: '즉시조치한다'→'즉시조지한나', 배경색이면 정확).
    """
    import numpy as np
    calm = np.asarray(calm)
    if calm.ndim != 2 or not calm.size:
        return ""
    gh, gw = calm.shape
    right = np.where(np.arange(gw) >= cut_col, calm, 0)
    seed = np.unravel_index(int(np.argmax(right)), calm.shape)
    if right[seed] < SPEAKER_CELL:
        return ""
    hot, body, todo = calm >= SPEAKER_CELL, np.zeros(calm.shape, bool), [seed]
    while todo:
        r, c = todo.pop()
        if 0 <= r < gh and 0 <= c < gw and hot[r, c] and not body[r, c]:
            body[r, c] = True
            todo += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    boxes = []
    for r in range(gh):
        cols = np.nonzero(body[r])[0]
        if len(cols):
            boxes.append(f"drawbox=x=iw*{cols[0] / gw:.4f}:y=ih*{r / gh:.4f}:"
                         f"w=iw*{(cols[-1] + 1 - cols[0]) / gw:.4f}:h=ih*{1 / gh:.4f}:"
                         f"color=0x{fill:02X}{fill:02X}{fill:02X}:t=fill,")
    return "".join(boxes)


def speaker_on_right(shares) -> bool:
    """슬라이드 쪽이 멈춰 있는 동안 오른쪽이 움직이는가 — 화자가 오른쪽에 있다는 뜻이다.

    '오른쪽이 한 번이라도 움직였나'로 보면, 슬라이드가 화면 전체인 강의에서 전체 화면
    영상이 재생될 때도 참이 되어 슬라이드를 잘라 읽었다(실강의: 모든 슬라이드의 오른쪽
    3분의 1이 사라졌다). 전체 시간의 좌우 비율로 보면, 영상을 19분 튼 좌우 분할 강의에서
    왼쪽도 많이 움직여 화자를 놓쳤다. 영상 재생 시간을 빼고 보면 뚜렷이 갈린다
    (실측: 화자 있음 0.23·0.25, 없음 0.001). 판단할 만큼 멈춘 시간이 없으면 자르지 않는다.
    shares 는 초마다 (화면 전체, 좌측 슬라이드 쪽) 움직인 칸의 비율이다.
    """
    gw = SCAN_W // LIVE_CELL
    cut = int(gw * SLIDE_CROP)
    calm = [((f * gw - l * cut) / (gw - cut), l) for f, l in shares[1:] if l < 0.05]
    if len(calm) < 60:
        return False
    right = sum(r for r, _l in calm) / len(calm)
    left = sum(l for _r, l in calm) / len(calm)
    return right >= 0.03 and right >= 5 * left


def pick_crop(ff, tess, src: Path, times, tmp: Path, langs, pool) -> str:
    """화자가 오른쪽에 있을 때, 좌측만 잘라 읽을지 정한다.

    화자 창이 따로 있는 녹화 강의는 잘라내야 목차·제목까지 읽히고(전체를 읽으면 제목을
    놓치고 화자 쪽 잡음이 줄 끝에 붙는다), 교수가 슬라이드 위에 겹쳐 선 강의는 자르면
    오른쪽 글자를 잃는다. 그래서 **오른쪽 띠만 따로 읽어** 확신 있게 읽힌 글자가 슬라이드
    쪽의 25% 이상이면 슬라이드가 오른쪽까지 이어진 것으로 보고 자르지 않는다.

    전체와 잘라 읽기를 통째로 견주면 안 된다 — 줄 수는 잘린 줄도 한 줄로 세고, 글자 수는
    화자 쪽 잡음이 붙어 늘고, 깨짐 판정은 글머리표(`=`)·URL 이 든 멀쩡한 줄까지 깎았다(실측).
    times 는 슬라이드가 멈춰 있던 화면들이다 — 길이의 25·50·75% 지점을 보던 때는 그 셋이
    재생 영상·표 화면에 걸려, 화자 무늬의 '| |'가 슬라이드 글자보다 많이 세어져 화자 쪽까지
    읽었다(합성 강의). 띠에서 여러 화면에 되풀이되는 줄(학교 로고)은 세지 않는다.
    """
    crop = f"crop=iw*{SLIDE_CROP}:ih:0:0,"
    strip = f"crop=iw*{1 - SLIDE_CROP:.2f}:ih:iw*{SLIDE_CROP}:0,"

    def sure_lines(t, region):
        png = tmp / f"probe_{t:.0f}_{len(region)}.png"
        try:
            if not grab_frame(ff, src, t, f"{region}scale=iw*2:ih*2", png):
                return []
            reads = [[ln[1] for ln in ocr_lines(tess, png, lg) if ln[2] >= 80 and not ocr_junk(ln[1])]
                     for lg in langs]
            return max(reads, key=lambda ls: sum(map(len, ls)), default=[])
        except subprocess.TimeoutExpired:
            return []
        finally:
            png.unlink(missing_ok=True)
    jobs = {pool.submit(sure_lines, t, r): r for t in times for r in (crop, strip)}
    found = {crop: [], strip: []}
    for fut in as_completed(jobs):
        found[jobs[fut]].append(fut.result())
    alnum = lambda s: sum(ch.isalnum() for ch in s)
    seen = Counter(re.sub(r"\s", "", ln) for page in found[strip] for ln in set(page))
    strip_chars = sum(alnum(ln) for page in found[strip] for ln in page if seen[re.sub(r"\s", "", ln)] < 2)
    crop_chars = sum(alnum(ln) for page in found[crop] for ln in page)
    return "" if strip_chars >= 0.25 * max(crop_chars, 1) else crop


def still_samples(picks, shares, duration, k: int = 5):
    """자르기 판단에 쓸, 슬라이드가 멈춰 있던 화면 k개의 시각(고르게 흩어서)."""
    still = [n for n in picks if live_share(shares, n, "crop") < 0.02]
    if len(still) < k:
        return [n + 1 for n in still] or [duration * f for f in (0.25, 0.5, 0.75)]
    return [still[round(i * (len(still) - 1) / (k - 1))] + 1 for i in range(k)]


def select_frames(frames):
    """(초, 축소 회색 화면) 흐름에서 읽어 볼 화면을 고른다.

    직전 초가 아니라 **마지막으로 고른 화면**과 견준다 — 조금씩 늘어나는 판서·항목도
    쌓이면 잡힌다. 계속 움직이는 칸(화자 창, 화면 속 동영상, 커서)은 스스로 알아내
    비교에서 뺀다. 그래서 30초 안전 샘플이 필요 없다.

    화면 전체 기준과 좌측 슬라이드 영역 기준을 **한 번의 디코딩으로 함께** 고른다.
    화자는 가끔씩만 움직여 '계속 움직이는 칸'으로 걸러지지 않는다 — 실강의 44분에서
    고른 화면 322장 중 170장이 화자 쪽 변화였다. 잘라 읽기로 정해지면 좌측 기준을 쓴다.
    돌려주는 값은 (전체 기준 고른 초, 좌측 기준 고른 초, 슬라이드 쪽이 멈춘 동안 칸마다
    움직인 비율(화자 자리), 초마다 움직이던 비율 [(화면 전체, 좌측 슬라이드 쪽)]).
    """
    import numpy as np
    chains, prev, ema, calm_sum, calm_n, shares = None, None, None, None, 0, []
    for n, f in frames:
        f = f.astype(np.int16)
        h, w = f.shape
        gh, gw = h // LIVE_CELL, w // LIVE_CELL
        cut = int(gw * SLIDE_CROP)
        if prev is None:
            ema, calm_sum = np.zeros((gh, gw)), np.zeros((gh, gw))
            # 영역마다 [고른 초, 마지막으로 고른 화면, 그 초, 재확인할 초]
            chains = {k: [[n], f, n, None] for k in ("full", "left")}
            shares.append((0.0, 0.0))
            prev = f
            continue
        moving = (np.abs(f - prev) > PIX_DELTA)[:gh * LIVE_CELL, :gw * LIVE_CELL]
        moving = moving.reshape(gh, LIVE_CELL, gw, LIVE_CELL).mean(axis=(1, 3)) > 0.02
        a = max(LIVE_EMA, 1 / n)       # 처음 몇 초는 누적 평균 — 화자 창을 빨리 알아본다
        ema = ema * (1 - a) + moving * a
        live = ema > LIVE_ON
        # 자막 판정용은 지수평균이 아니라 지금 이 순간의 움직임이다 — 지수평균은 영상이
        # 끝난 뒤에도 10초 넘게 남아 바로 다음 슬라이드까지 영상으로 묶었다
        shares.append((float(moving.mean()), float(moving[:, :cut].mean())))
        if shares[-1][1] < 0.05:        # 슬라이드 쪽이 멈춘 초 — 이때 움직이는 칸이 화자다
            calm_sum += moving
            calm_n += 1
        for key, chain in chains.items():
            picks, last, last_n, settle_at = chain
            region = ~live if key == "full" else ~live & (np.arange(gw) < cut)
            still = np.repeat(np.repeat(region, LIVE_CELL, 0), LIVE_CELL, 1)
            diff = np.abs(f - last)[:gh * LIVE_CELL, :gw * LIVE_CELL]
            ratio = (diff[still] > PIX_DELTA).mean() if still.any() else 0.0
            area = live if key == "full" else live[:, :cut]
            gap = n - last_n
            pick = False
            if ratio > CHANGE_RATIO and gap >= MIN_GAP_SEC:
                pick, settle_at = True, n + SETTLE_SEC
            elif settle_at is not None and n >= settle_at:
                settle_at = None
                pick = bool(still.any()) and diff[still].mean() > 0.3   # 흐릿했던 것이 선명해졌다
            elif area.mean() > VIDEO_LIVE and gap >= VIDEO_GAP_SEC:
                pick = True
            if pick:
                picks.append(n)
                last, last_n = f, n
            chain[1:] = [last, last_n, settle_at]
        prev = f
    if chains is None:
        return [], [], None, shares
    return chains["full"][0], chains["left"][0], calm_sum / max(calm_n, 1), shares

def live_share(shares, n: int, crop: str, ahead: int = 4) -> float:
    """n초 화면이 떠 있는 동안(직후 몇 초) 화면(잘라 읽으면 슬라이드 쪽)이 움직인 비율.

    n초 자체는 넣지 않는다 — 슬라이드가 넘어가는 순간도 움직임으로 잡히기 때문이다.
    전환 뒤 가만히 있는 슬라이드는 0이고, 재생 중인 영상은 매초 움직인다.
    평균이 아니라 (아래쪽) 중앙값이다 — 영상이 끝나며 슬라이드로 2초 동안 서서히 바뀌면
    뒤의 1초가 창에 들어와 평균이 0.25가 되어, 영상 뒤 슬라이드가 '움직이는 화면'으로
    영상에 묶였다(실강의: 교수 슬라이드 42초가 📺 로 실렸다).
    """
    k = 1 if crop else 0
    window = sorted(s[k] for s in shares[n + 1:n + 1 + ahead])
    return window[(len(window) - 1) // 2] if window else 0.0


def scan_screen_changes(ff, src: Path, tmp: Path):
    """1초마다 축소 회색 화면을 받아 전환 지점을 고른다. 디코딩은 한 번뿐이다.

    화면 전체를 본다 — 화자 창은 스스로 움직이는 칸으로 걸러지고, 어디가 움직이는지가
    잘라 읽을지 정하는 근거가 된다.
    round=up 이어야 n번째 화면이 '-ss n' 으로 뽑는 화면과 같다(실측: 16곳 모두 화소
    차이 0). 기본값(near)은 n+0.5초 가까이의 화면을 주어, 전환 직후를 잡고도 정작
    읽는 화면은 전환 직전 것이 되어 슬라이드가 한 장씩 밀려 찍혔다.
    """
    size = SCAN_W * SCAN_H
    err = tmp / "scan_err.txt"
    with err.open("wb") as fe:     # stderr 를 파이프로 받으면 오류가 쏟아질 때 멈춘다
        p = subprocess.Popen([ff, "-nostdin", "-v", "error", "-i", str(src), "-an", "-sn",
                              "-vf", f"fps=1:round=up,scale={SCAN_W}:{SCAN_H}:flags=area,"
                                     f"format=gray",
                              "-f", "rawvideo", "-"],
                             stdout=subprocess.PIPE, stderr=fe, stdin=subprocess.DEVNULL)

        def frames():
            import numpy as np
            n = 0
            while True:
                b = p.stdout.read(size)
                if len(b) < size:
                    return
                yield n, np.frombuffer(b, np.uint8).reshape(SCAN_H, SCAN_W)
                n += 1
        try:
            picks, picks_left, calm, shares = select_frames(frames())
        finally:
            p.stdout.close()
            p.wait()
    if not picks and p.returncode != 0:
        detail = err.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        raise RuntimeError("화면을 뽑아내지 못했습니다"
                           + (f" — {detail[-1][:160]}" if detail else ""))
    return picks, picks_left, calm, shares


def slide_key(lines):
    """슬라이드 비교용 낱말 집합. 한 글자 조각은 오인식 잡음이므로 뺀다."""
    words = re.sub(r"[^0-9a-z가-힣]+", " ", " ".join(lines).lower()).split()
    return {w for w in words if len(w) >= 2}


def merge_slides(found):
    """같은 슬라이드가 여러 번 잡힌 것을 하나로 합친다.

    발표 중 항목이 하나씩 나타나면 글자가 점점 늘어난다. 이때는 가장 완전한
    판본을 남기되 처음 등장한 시각을 쓴다.
    """
    merged = []
    for t, lines in found:
        if not lines:
            continue
        k = slide_key(lines)
        if not k:
            continue
        hit = None
        # 직전 한 장만 보지 않는다 — 오인식이 심하면 같은 슬라이드가 A A' A 처럼
        # 번갈아 나와 바로 앞과만 비교했을 때 병합에 실패했다.
        for j in range(len(merged) - 1, max(-1, len(merged) - 1 - SLIDE_MERGE_LOOKBACK), -1):
            pk = slide_key(merged[j][1])
            if pk and len(k & pk) >= SLIDE_MERGE_RATIO * min(len(k), len(pk)):
                hit = j
                break
        if hit is not None:
            prev_t, prev_lines = merged[hit]
            # 더 잘 읽힌 판본을 남기되 시각은 처음 잡힌 때를 쓴다
            if ocr_score(lines) > ocr_score(prev_lines):
                merged[hit] = (prev_t, lines)
            continue
        merged.append((t, lines))
    return merged


def extract_slides(src: Path, tmp: Path, tess: str, duration: float, langs):
    """화면이 바뀌는 지점을 찾아 그 슬라이드의 글자를 읽어 온다.

    돌려주는 값은 (슬라이드 목록, {시각: 그때 화면이 움직이던 비율}, 초마다 움직인 비율).
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다")
    shots = tmp / "slides"
    shots.mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(OCR_WORKERS)
    try:
        status("  화면이 바뀌는 지점을 찾는 중...")
        picks, picks_left, calm, shares = scan_screen_changes(ff, src, shots)
        # 화자가 오른쪽에 있으면 화자 쪽 변화로 고른 화면은 필요 없다 — 전체를 읽더라도
        side = speaker_on_right(shares)
        if side:
            picks = picks_left
            status("  화면 영역을 정하는 중...")
        samples = still_samples(picks, shares, duration or 600)
        crop = pick_crop(ff, tess, src, samples, shots, langs, pool) if side else ""
        # 잘라 읽다 끊긴 줄을 이어 읽을 넓은 화면 — 화자 자리만 슬라이드 배경색으로 가린다
        wide = (speaker_mask(calm, int(SCAN_W // LIVE_CELL * SLIDE_CROP),
                             background_gray(ff, src, samples[0])) if crop else "")
        # 넘어간 직후 화면은 인코딩이 덜 선명해 짧은 제목을 놓치기도 한다(합성 강의: 어두운 띠 위
        # 두 글자 제목이 첫 5초 동안 읽히지 않았다) — 8초 넘게 그대로인 슬라이드는 가운데쯤
        # (최대 10초 뒤) 화면도 읽어 더 잘 읽힌 쪽을 쓴다. 늦은 화면 하나만 읽으면 거기서 다른 줄이
        # 빠지기도 했다(합성 강의) — 프레임마다 인코딩이 달라 어느 쪽이 나을지 미리 알 수 없다.
        # 시각은 처음 그대로다
        mode = "crop" if side else ""
        ordered = sorted(picks)
        last = int(duration) if duration else len(shares)
        later = {n: n + min(10, (m - n) // 2)
                 for n, m in zip(ordered, ordered[1:] + [last])
                 if m - n >= 8 and live_share(shares, n, mode) < 0.02}
        # 재생 중인 영상의 화면(3초마다 고른다 — 실강의에서 읽을 화면의 절반 이상)은 자막을
        # 영상 판정과 📺 블록에만 쓰므로 단독 판본 둘로만 읽고, 잘린 줄을 이어 읽지 않는다.
        # 세 판본에 이어 읽기까지 하던 때는 화면 읽기가 rev12 의 2.5배(10분 26초)였다(실강의)
        playing = {n for n in picks if live_share(shares, n, mode) >= SUBTITLE_LIVE}
        jobs = {pool.submit(read_screen, ff, tess, src, at, crop, shots / f"f_{at:06d}.png",
                            langs[:2] if n in playing else langs, "" if n in playing else wide): n
                for n in picks for at in (n, later.get(n)) if at is not None}
        reads = defaultdict(list)
        for i, fut in enumerate(as_completed(jobs), 1):
            status(f"  슬라이드 읽는 중... {i}/{len(jobs)}")
            reads[jobs[fut]].append(fut.result())
        # 움직임은 화자 쪽을 뺀 슬라이드 쪽으로 잰다 (화자가 오른쪽에 있을 때)
        motion = [s[1 if side else 0] for s in shares]
        found = {}
        for n, rs in reads.items():            # 같은 시각으로 모인 화면은 잘 읽힌 쪽 하나만
            t = appeared_at(n, motion)
            found[t] = max(rs + ([found[t]] if t in found else []), key=ocr_score)
        found = list(found.items())
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        status("")
        sys.stdout.write("\r")
    found.sort(key=lambda x: x[0])
    live = {appeared_at(n, motion): live_share(shares, n, side) for n in picks}
    return merge_slides(found), live, motion


def appeared_at(n: int, motion) -> float:
    """n번째 초에 고른 화면이 실제로 뜬 시각.

    n번째 화면은 (n-1, n] 사이에 바뀐 것이므로 보통 n-0.5 초다. 그런데 영상이 끝난 직후의
    슬라이드는 영상 동안 '계속 움직이는 칸'으로 빠져 있던 영역이 비교에 다시 들어올 때(최대
    몇 초 뒤) 골라진다 — 그 사이 화면은 멈춰 있었다. 고른 초에 아무 변화가 없었으면, 멈춤을
    거슬러 올라가 마지막 큰 변화(전환)가 있던 초를 쓴다. 늦게 잡힌 시각이 영상 구간의 끝이
    되어, 영상 직후 교수의 첫마디가 📺 로 실렸다(합성 강의: 2.5초).
    """
    if not motion or n >= len(motion) or motion[n] >= 0.02:
        return max(0.0, n - 0.5)
    j = n
    while j > 0 and n - j < 6 and motion[j] < 0.02:
        j -= 1
    # 그 전환 직전까지 화면이 계속 움직이고 있었을 때(영상)만 — 슬라이드에 항목이 하나씩 늘어나는
    # 작은 변화를 앞 슬라이드가 넘어간 시각으로 끌어당기지 않는다
    was_video = sum(1 for m in motion[max(0, j - 4):j] if m >= 0.02) >= 3
    return max(0.0, j - 0.5) if motion[j] >= 0.3 and was_video else max(0.0, n - 0.5)


# ────────────────────────── 강의자료 PDF 연동 ──────────────────────────

def name_tokens(stem: str):
    return {w for w in re.sub(r"[^0-9a-z가-힣]+", " ", stem.lower()).split() if w}


def stem_week(stem: str):
    """파일 이름에서 주차를 읽는다. '7주차교재' → 7, '7표본설계' → 7."""
    m = re.search(r"(\d+)\s*주차", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else None


# 강의자료 한 건. name 은 표시·지문용("묶음.zip/교재.pdf"), stem 은 쪽 이름용,
# named 는 파일 이름에 이 강의가 적혀 있는지(아니면 주차만 같은 후보),
# container 는 같은 묶음(zip·폴더)끼리 알아보는 열쇠, load 는 PDF 바이트를 돌려준다.
Material = namedtuple("Material", "name stem week named container load")


def zip_member_name(info) -> str:
    """zip 안 이름. UTF-8 표시가 없는 옛 압축은 한국어 Windows 기준(cp949)으로 풀어 본다."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def iter_materials():
    """강의자료 폴더의 PDF — LMS 에서 받은 zip 묶음 안의 PDF 도 포함한다."""
    if not PDF_DIR.is_dir():
        return
    for p in sorted(PDF_DIR.rglob("*")):
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            yield p.name, p.stem, p.stem, str(p.parent), (lambda p=p: p.read_bytes())
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(p) as z:
                    members = [i for i in z.infolist()
                               if not i.is_dir() and i.filename.lower().endswith(".pdf")]
            except (zipfile.BadZipFile, OSError):
                log(f"  ⚠ 강의자료 {p.name}을(를) 열 수 없습니다 (손상된 압축 파일).")
                continue
            for i in members:
                inner = Path(zip_member_name(i))
                yield (f"{p.name}/{inner.name}", inner.stem, f"{p.stem} {inner.stem}", str(p),
                       lambda p=p, n=i.filename: zipfile.ZipFile(p).read(n))


def find_slide_pdfs(src: Path):
    """이 강의에 해당할 수 있는 강의자료를 **모두** 찾는다.

    한 주차에 슬라이드 교재와 실습지가 따로 있는 것이 보통이므로 하나만 고르지
    않는다. 주차 번호를 읽을 수 있으면 서로 다를 때 걸러낸다 — 과목 이름만 겹치면
    7주차 자료가 6주차 강의에 붙는 사고가 난다.

    LMS 에서 받은 그대로의 이름('6주차교재.pdf')에는 과목명이 없다. 그래서 주차만
    같은 자료도 **후보**(named=False)로 넣고, 화면과 실제로 맞춰 본 뒤 채택한다
    (pick_materials).
    """
    course, week, _period = parse_course(src.stem)
    course_key = re.sub(r"[^0-9a-z가-힣]+", "", course.lower())
    want = name_tokens(src.stem)
    hits = []
    for name, stem, match_text, container, load in iter_materials():
        have = name_tokens(match_text)
        if not have:
            continue
        flat = re.sub(r"[^0-9a-z가-힣]+", "", match_text.lower())
        named = bool(course_key and course_key in flat) or (
            len(want & have) / min(len(want), len(have)) >= PDF_NAME_RATIO)
        pw = stem_week(stem)
        if pw is None:
            pw = stem_week(match_text)
        if week is not None and pw is not None and pw != week:
            continue
        if named or (week is not None and pw == week):
            hits.append(Material(name, stem, pw, named, container, load))
    return hits


def drop_boilerplate(pages):
    """대부분의 쪽(화면)에 되풀이되는 줄(학교 배너, 저작권 꼬리말)을 뺀다.

    줄 목록의 목록을 받아 같은 모양으로 돌려준다. 이런 줄은 슬라이드마다 같은
    잡음 토큰을 싣고, 모든 쪽에 같은 2연쇄를 보태 쪽 사이의 구별을 흐린다.
    """
    if len(pages) < BOILERPLATE_MIN:
        return pages
    norm = lambda s: re.sub(r"[^0-9a-z가-힣]+", "", s.lower())
    seen = Counter(k for lines in pages for k in {norm(ln) for ln in lines} if k)
    common = {k for k, c in seen.items() if c >= BOILERPLATE_RATIO * len(pages)}
    if not common:
        return pages
    return [[ln for ln in lines if norm(ln) not in common] for lines in pages]


def load_slide_materials(src: Path):
    """후보 자료들을 읽어 (자료, 쪽 글자 목록) 목록과 지문용 이름 목록을 돌려준다.

    지문용 이름에는 글자를 못 읽은 자료도 넣는다 — "짝은 맞았다"로 기록해야
    같은 자료로 매번 다시 변환하는 일이 생기지 않는다.
    """
    found = find_slide_pdfs(src)
    loaded = []
    for m in found:
        try:
            pages = read_pdf_pages(m.load())
        except (OSError, zipfile.BadZipFile, KeyError):
            pages = []
        if pages and any(t.strip() for t in pages):
            loaded.append((m, pages))
        else:
            log(f"  ⚠ 강의자료 {m.name}에서 글자를 읽지 못했습니다"
                f" (그림으로 스캔된 PDF일 수 있습니다).")
    return loaded, [m.name for m in found]


def material_pages(chosen):
    """고른 자료들의 쪽 글자와 쪽 이름을 한 줄로 이어 붙인다. 되풀이되는 배너는 뺀다."""
    pages, labels, owner = [], [], []
    many = len(chosen) > 1
    stems = Counter(m.stem for m, _t in chosen)
    for k, (m, texts) in enumerate(chosen):
        tag = m.stem if stems[m.stem] == 1 else m.name     # 이름이 같은 자료끼리는 묶음까지 적는다
        for i, text in enumerate(texts, 1):
            pages.append(text)
            labels.append(f"{tag} {i}쪽" if many else f"{i}쪽")
            owner.append(k)
    lines = drop_boilerplate([[ln.rstrip() for ln in t.splitlines() if ln.strip()] for t in pages])
    return ["\n".join(x) for x in lines], labels, owner


def pick_materials(slides, loaded):
    """후보 자료 가운데 이 강의 화면과 **실제로 맞는** 것만 남긴다.

    이름에 강의가 적힌 자료는 그대로 쓴다. 주차만 같은 후보는 화면 2장 이상이 그
    자료의 쪽과 맞아야 채택한다 — 다른 과목의 같은 주차 자료가 섞여 있어도 걸러진다.
    쪽이 3쪽 이하인 작은 자료(실습지)는, 같은 묶음의 다른 자료가 채택됐으면 1장으로 족하다.
    """
    if not loaded or not slides:
        return [x for x in loaded if x[0].named]
    pages, labels, owner = material_pages(loaded)
    who = dict(zip(labels, owner))
    hits = Counter(who[a[2]] for a in align_slides_to_pdf(slides, pages, labels) if a[2])
    keep = {k for k, (m, _t) in enumerate(loaded) if m.named or hits[k] >= 2}
    kept_boxes = {loaded[k][0].container for k in keep}
    keep |= {k for k, (m, t) in enumerate(loaded)
             if hits[k] >= 1 and len(t) <= 3 and m.container in kept_boxes}
    return [x for k, x in enumerate(loaded) if k in keep]


def read_pdf_pages(data):
    """PDF(경로 또는 바이트)의 쪽별 글자. 읽지 못하면 빈 목록(부르는 쪽에서 알린다)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        pages = []
        for page in PdfReader(io.BytesIO(data) if isinstance(data, bytes) else str(data)).pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return pages
    except Exception:
        return []


# 낱말 끝의 조사 — 떼어 내야 '편향은'·'편향을'이 한 용어로 모인다
JOSA = ("에서는", "으로는", "으로서", "으로써", "에서", "으로", "에게", "까지", "부터",
        "보다", "처럼", "이란", "라는", "은", "는", "이", "가", "을", "를", "의", "에",
        "와", "과", "로", "도", "만")
COMMON_WORDS = {"그리고", "하지만", "그러나", "때문", "경우", "위한", "대한", "이상", "이하",
                "있는", "하는", "되는", "그래서", "이러한", "우리", "여러분", "지원", "다음",
                "주차", "교시", "학습", "목표", "내용"}


def pdf_hotwords(pages):
    """PDF에서 이 강의의 전문용어를 뽑아 전사 힌트로 쓴다.

    '신뢰수준'을 '실내수준'으로 잘못 듣던 오류가 줄어든다(합성 강의 실측).
    문장이 아니라 **쉼표로 나열한 용어 목록**으로 만든다 — 빈도가 같은 낱말을 PDF
    순서대로 이어 붙이면 PDF 문장이 그대로 프롬프트가 되어 Whisper가 문장부호를
    거의 찍지 않았다(합성 강의 실측: 문장부호 5개 → 쉼표 목록 12개, 정확도 동일).
    """
    terms = Counter()
    for w in re.findall(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9&-]{2,}", " ".join(pages)):
        if "가" <= w[0] <= "힣":
            if w.endswith("다"):                 # 서술어('추정한다', '전체이다')는 용어가 아니다
                continue
            j = next((j for j in JOSA if w.endswith(j)), "")
            if len(w) - len(j) >= 2:            # '값으로'에서 '로'만 떼어 '값으'가 되지 않게
                w = w[:len(w) - len(j)]
        elif not (w[0].isupper() or w.isupper()):   # 영어는 고유명사·약어만
            continue
        if w not in COMMON_WORDS:
            terms[w] += 1
    out = []
    for w in sorted(terms, key=lambda x: (-terms[x], -len(x))):
        if len(", ".join(out + [w])) > HOTWORD_MAX:
            break
        out.append(w)
    return ", ".join(out) + "." if out else ""


def page_key(text: str):
    """글자 2연쇄 집합.

    낱말 단위로 맞대면 한국어 조사 변화('표본' vs '표본의')와 OCR 잡음에
    너무 쉽게 어긋난다. 2연쇄는 두 가지 모두에 훨씬 강하다.
    """
    s = re.sub(r"[^0-9a-z가-힣]+", "", text.lower())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def align_slides_to_pdf(slides, pages, labels=None):
    """화면에서 읽은 슬라이드를 PDF 쪽에 맞춘다.

    쪽을 찾으면 본문을 **PDF 원문으로 바꾼다** — OCR 잡음이 사라지고 표·빈칸·기호가
    원본 그대로 남는다. 화면 글자는 "지금 몇 쪽인가"를 알아내는 열쇠로만 쓰인다.
    돌려주는 값은 (시각, 줄 목록, 쪽 이름 또는 None).
    """
    if labels is None:
        labels = [f"{i}쪽" for i in range(1, len(pages) + 1)]
    keys = [page_key(p) for p in pages]
    out, last = [], None
    for t, lines in slides:
        k = page_key(" ".join(lines))
        best, best_r, best_adj = None, 0.0, 0.0
        if len(k) >= PDF_MIN_WORDS:
            for i, pk in enumerate(keys):
                if not pk:
                    continue
                r = len(k & pk) / len(k)
                # 강의는 대체로 앞에서 뒤로 진행한다. 글자가 거의 같은 쪽이 여럿일 때
                # (교시마다 되풀이되는 학습목표 등) 방금 본 쪽 근처를 고르게 한다.
                adj = r + (PDF_FORWARD_BONUS
                           if last is not None and last <= i <= last + PDF_FORWARD_SPAN
                           else 0.0)
                if adj > best_adj:
                    best, best_r, best_adj = i, r, adj
        # 근거(2연쇄)가 적을수록 더 확실할 때만 인정한다 — 짧은 표제가 엉뚱한 쪽에
        # 붙으면 화면에 없던 내용이 통째로 실린다
        need = PDF_SHORT_RATIO if len(k) < PDF_SHORT_KEY else PDF_MATCH_RATIO
        if best is not None and best_r >= need:
            body = [ln.rstrip() for ln in pages[best].splitlines() if ln.strip()]
            # 같은 쪽이 잇따라 잡히면 한 번만 싣는다 (화면이 바뀌지 않은 것이다)
            last = best
            if out and out[-1][2] == labels[best]:
                continue
            out.append((t, body, labels[best]))
        else:
            out.append((t, lines, None))
    return out


def ocr_junk(line: str) -> bool:
    """그림·괘선을 읽은 잡음 줄인가.

    글자가 3자 미만이고 숫자도 없는 줄('NW', '| Sy \\'), 그리고 조각이 넷 이상인데 글자·숫자가
    두 자 넘게 붙은 조각이 30%도 안 되는 줄(표 괘선을 읽은 '1 | | | | | 1. | |.'). 숫자만
    있는 줄('31 47 26 58')은 표 내용일 수 있어 남긴다. 한글 두 음절('정리', '목차')은 제목이다 —
    글자 수만 세던 때는 마지막 정리 슬라이드의 제목이 잡음으로 버려졌다(합성 강의).
    """
    if (sum(ch.isalpha() for ch in line) < 3 and not any(ch.isdigit() for ch in line)
            and sum("가" <= ch <= "힣" for ch in line) < 2):
        return True
    toks = line.split()
    real = sum(1 for tok in toks if sum(ch.isalnum() for ch in tok) >= 2)
    return len(toks) >= 4 and real < 0.3 * len(toks)


def looks_garbled(s: str) -> bool:
    """읽다 만 글자인지. 차례에 올리기 전에 거른다."""
    toks = s.split()
    if not toks or sum(1 for x in toks if len(x) == 1) * 2 >= len(toks):
        return True                                   # 낱자가 절반 이상
    # 'CEO의', '6주차' 처럼 한글에 라틴·숫자가 붙는 것은 정상이다. 잡음의 신호는
    # 홑자모와 깨진 기호, 그리고 뒤죽박죽 대소문자다.
    for tok in toks:
        core = tok.strip(".,:;!?()[]'\"·’”“‘…")
        if not core:
            continue
        if any("ㄱ" <= c <= "ㅣ" for c in core):        # 홑자모(ㅅ, ㅁ, ㅇ, ㅣ …)
            return True
        if any(c in "|\\/[]{}<>~^_=＊" for c in core):  # 읽다 만 기호
            return True
        if re.search(r"[a-z][A-Z]", core):            # HItI 같은 뒤죽박죽 대소문자
            return True
    return False


def clean_moving_line(line: str) -> str:
    """움직이는 화면(영상 위 자막)에서 읽은 줄의 잡음 조각을 떼어 낸다(남을 게 없으면 빈 문자열).

    배경이 요란하면 자막 줄 끝에 '<”', '“=:', 'mes es' 같은 조각이 붙는다. 예전에는 조각 하나만
    있어도 줄 전체를 깨진 줄로 버려, 멀쩡히 읽힌 자막까지 사라져 영상을 놓쳤다(합성 강의).
    홑자모·읽다 만 기호가 든 조각, 한글 문장 속 3자 이하 라틴 조각(그 반대도)을 뗀다.
    """
    toks = line.split()
    han_line = any(sum("가" <= c <= "힣" for c in t) >= 2 for t in toks)
    lat_line = any(sum(c.isascii() and c.isalpha() for c in t) >= 4 for t in toks)
    keep = []
    for tok in toks:
        core = tok.strip(".,:;!?()[]'\"·’”“‘…")
        if not core or any("ㄱ" <= c <= "ㅣ" for c in core) or any(c in "|\\/[]{}<>~^_=＊" for c in core):
            continue
        is_lat = all(c.isascii() and c.isalpha() for c in core)
        is_han = all("가" <= c <= "힣" for c in core)
        if (han_line and is_lat and len(core) <= 3) or (lat_line and not han_line and is_han and len(core) <= 2):
            continue
        if len(core) == 1 and not ("가" <= core <= "힣" or core.isdigit()):
            continue
        keep.append(tok)
    text = " ".join(keep)
    return text if sum(ch.isalpha() for ch in text) >= 3 and not looks_garbled(text) else ""


def slide_title(lines):
    """차례에 쓸 만한 제목 한 줄을 고른다. 쓸 만한 것이 없으면 None.

    첫 줄을 그냥 쓰면 대학 배너('S A M P L E U N I V E R S I T Y'), 단독 절 번호
    ('01'), 읽다 만 글자('ㅅ 트 변 그 즈다')가 차례를 뒤덮는다.
    """
    for ln in lines[:4]:
        s = re.sub(r"\s+", " ", ln).strip()
        if len(s) < 4:
            continue
        toks = s.split()
        if len(toks) >= 4 and all(len(x) == 1 for x in toks):   # 자간 벌린 배너
            continue
        if re.fullmatch(r"[\d\W_]+", s):                        # 01, 02, ▪ …
            continue
        if looks_garbled(s):
            continue
        han = sum(1 for c in s if "가" <= c <= "힣")
        if han >= 2 or any(len(w) >= 3 and w.isalpha() for w in toks):
            return s[:60]
    return None


def parse_course(stem: str):
    """'재무관리 12-1강' → ('재무관리', 12, 1)."""
    m = re.match(r"^(.*?)\s*(\d+)\s*-\s*(\d+)\s*강?$", stem.strip())
    if m:
        return m.group(1).strip(), int(m.group(2)), int(m.group(3))
    return stem.strip(), None, None


def md_body_hash(text: str) -> str:
    """마커를 뗀 본문의 지문. 사용자가 손댔는지를 내용으로 판정한다.

    예전에는 파일 날짜(mtime)로 판정했는데, 백업·복사·동기화처럼 내용을 건드리지
    않는 작업만으로도 그 MD가 '편집됨'으로 굳어 영구히 갱신되지 않았다.
    """
    return hashlib.sha256(text.split(MARKER)[0].encode("utf-8")).hexdigest()[:16]


def read_marker(md: Path):
    """우리가 만든 MD인지 확인하고 기록된 메타데이터를 돌려준다. 아니면 None."""
    try:
        with md.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(MARKER):
                    return json.loads(line[len(MARKER):line.rindex("-->")].strip())
    except (OSError, ValueError):
        pass
    return None


_STAGED: list = []      # 끌어다 놓기로 우리가 만든 이름들 (종료 시 반드시 지운다)


def stage_dropped(paths):
    """바로가기에 끌어다 놓은 파일을 "MP4 입력" 에 잠시 걸어 둔다.

    돌려주는 값은 이번에 처리할 입력 폴더 안의 경로들이다. plan_targets 가
    relative_to(IN_DIR) 로 문서 이름을 만들기 때문에 임의 경로를 그대로 넘길 수 없다.

    하드링크를 먼저 쓴다 — 강의 영상은 수 GB라 복사하면 느리고 공간도 그만큼 든다.
    하드링크는 같은 파일에 이름을 하나 더 다는 것뿐이라 즉시 끝나고, 나중에 그 이름을
    지워도 원본은 남는다. 다른 볼륨이면 하드링크가 안 되므로 복사로 물러선다.
    """
    chosen = []
    for p in paths:
        if p.suffix.lower() not in MEDIA_EXTS:
            print(f"  건너뜀: {p.name} (지원하지 않는 형식)")
            continue
        if not p.is_file():
            print(f"  건너뜀: {p.name} (파일을 찾을 수 없습니다)")
            continue
        dest = IN_DIR / p.name
        if dest.exists():
            try:
                same = os.path.samefile(p, dest)
            except OSError:
                same = False
            if not same:   # 이름만 같은 다른 파일을 대신 전사하지 않는다
                print(f"  건너뜀: {p.name} (입력 폴더에 같은 이름의 다른 파일이 있습니다)")
                continue
            chosen.append(dest)
            continue
        try:
            os.link(p, dest)                   # 같은 볼륨: 즉시
        except OSError:
            try:
                shutil.copy2(p, dest)          # 다른 볼륨: 복사
            except OSError as e:
                print(f"  건너뜀: {p.name} ({e})")
                continue
        _STAGED.append(dest)   # 모듈 전역 — 어느 경로로 끝나든 __main__ 이 치울 수 있게 한다
        chosen.append(dest)
    return chosen


def unstage(staged):
    """우리가 만든 이름만 지운다. 원본 파일과 원래 입력 폴더 내용은 건드리지 않는다."""
    for p in staged:
        try:
            p.unlink()
        except OSError:
            pass


def plan_targets(cfg, only=None):
    """무엇을 전사할지 결정한다. 파일시스템을 변형하지 않는다.

    only 가 주어지면(끌어다 놓기) 그 파일들만 다룬다 — 입력 폴더에 쌓인 다른 파일까지
    몇 시간짜리 작업이 예고 없이 시작되지 않게 한다.
    """
    files, ignored = [], []
    for f in sorted(IN_DIR.rglob("*")):
        try:
            if not f.is_file():
                continue
            if f.suffix.lower() in MEDIA_EXTS:
                files.append(f)
            elif not f.name.startswith("~"):
                ignored.append(f)
        except OSError:
            continue

    def base_name(f):
        rel = f.relative_to(IN_DIR)
        return safe_stem(" - ".join(rel.parts[:-1] + (rel.stem,)))

    # 이름이 겹치는 파일은 모두 확장자를 붙여 구분한다 (먼저 온 쪽만 특별대우하지 않음)
    dup = {k for k, c in Counter(base_name(f).lower() for f in files).items() if c > 1}

    used, targets, skipped, blocked, rebuilt = set(), [], [], [], []
    fp = config_fingerprint(cfg)
    for f in files:
        try:
            stat = f.stat()
        except OSError:
            continue
        stem = base_name(f)
        name = f"{stem} ({f.suffix.lstrip('.').lower()})" if stem.lower() in dup else stem
        n, uniq = 2, name
        while uniq.lower() in used:                     # 그래도 겹치면 번호를 붙인다
            uniq = f"{name} ({n})"
            n += 1
        name = uniq
        used.add(name.lower())
        if only is not None and f not in only:          # 이름 배정은 모두에게 똑같이 한 뒤 거른다
            continue
        out = OUT_DIR / (name + ".md")

        if out.exists():
            mark = read_marker(out)
            if mark is None:
                blocked.append((f, out))
                continue
            if "sha" not in mark:
                # 기록 도중 중단된 미완성 파일 → 다시 만든다
                targets.append((f, out))
                continue
            try:
                cur = md_body_hash(out.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                targets.append((f, out))
                continue
            if cur != mark["sha"]:
                skipped.append((f, "직접 고친 것으로 보여 보존 — 새로 만들려면 이 .md를 지우세요"))
                continue
            pdf_tag = ", ".join(p.name for p in find_slide_pdfs(f))
            if (mark.get("bytes") == stat.st_size and mark.get("cfg") == fp
                    and mark.get("pdf", "") == pdf_tag):
                skipped.append((f, "이미 변환됨"))
                continue
            rebuilt.append(f)          # 이미 있는데 다시 만드는 것 (설정·자료·엔진 변경)
        targets.append((f, out))
    return targets, skipped, blocked, ignored, rebuilt


def ends_sentence(text: str) -> bool:
    """문장이 끝났는지. '소득세율은 3.' 처럼 숫자 뒤 마침표는 끝이 아니다."""
    t = text.rstrip()
    return bool(t) and t.endswith(SENTENCE_END) and not re.search(r"\d\.$", t)


def looks_hallucinated(text: str) -> bool:
    """같은 문장이 계속 되풀이되면 인식이 헛돈 것이다."""
    parts = [p.strip() for p in re.split(r"[.!?]", text) if len(p.strip()) >= 12]
    if not parts:
        return False
    top = max(parts.count(p) for p in set(parts))
    return top >= SUSPECT_REPEAT


def group_paragraphs(segments, breaks=()):
    """(start, end, text[, 의심]) 목록을 문단으로 묶는다. 문장 중간에서 끊지 않는다.

    breaks 는 슬라이드가 바뀐 시각들이다. 문단이 시작된 뒤 슬라이드가 바뀌었으면
    거기서 끊는다 — 그래야 다음 슬라이드 설명이 앞 슬라이드 밑에 붙지 않는다.
    돌려주는 값은 (시작시각, 본문, 의심여부) 이다.
    """
    paras, cur, start, end, bad = [], "", None, 0.0, False
    breaks, bi = sorted(breaks), 0

    def close():
        paras.append((start, cur.strip(), bad or looks_hallucinated(cur)))

    for seg in segments:
        s_start, s_end, text = seg[0], seg[1], seg[2]
        s_bad = bool(seg[3]) if len(seg) > 3 else False
        text = text.strip()
        if not text:
            continue
        crossed = False
        while bi < len(breaks) and breaks[bi] <= s_start:
            crossed = crossed or (start is not None and breaks[bi] > start)
            bi += 1
        over = (start is not None and
                (s_end - start >= PARA_SOFT_SEC or len(cur) >= PARA_SOFT_CHARS))
        gap = start is not None and s_start - end >= PARA_GAP_SEC
        forced = start is not None and (len(cur) >= PARA_HARD_CHARS
                                        or s_end - start >= PARA_HARD_SEC)
        # 상한을 넘었더라도 앞 문단이 문장으로 끝났을 때만 닫는다 (강제 상한 제외)
        if start is not None and (gap or forced or crossed or (over and ends_sentence(cur))):
            close()
            cur, start, bad = "", None, False
        if start is None:
            start = s_start
        cur = (cur + " " + text).strip()
        bad = bad or s_bad
        end = s_end
    if cur.strip():
        close()
    return paras


def label_slides(slides, paragraphs, live=None):
    """화면 글자를 '슬라이드'와 '영상 자막'으로 가른다. (시각, 줄, 쪽, 종류) 목록.

    강의 중 재생된 영상의 번인 자막이 슬라이드로 실리면, 그 표시가 "화면에 실제로
    있던 신뢰할 만한 원문"이라는 신호를 무의미하게 만든다. 짧은 한두 줄이 바로 옆
    발화와 대부분 겹치고 **그때 화면이 움직이고 있었으면** 자막으로 본다.

    움직임 조건이 없던 때는 교수가 제목 슬라이드를 소리 내 읽기만 해도 자막으로
    판정됐다. PDF 쪽과 맞춰진 화면은 당연히 슬라이드다.

    슬라이드 쪽이 움직이는 화면은 정지 슬라이드가 아니다. 그 화면의 깨진 줄(움직이는
    배경을 글자로 읽은 것)은 버리고, 자막으로 확정된 화면에 잇닿아 있으면 같은 영상으로
    묶는다 — 배경이 요란하면 자막 OCR이 깨져 발화와의 비교만으로는 절반을 놓쳤다.
    paragraphs 는 (시작, 글, ...) 목록, live 는 {시각: 움직이던 비율} 이다.
    """
    live = live or {}
    info = []
    for entry in slides:
        t, lines = entry[0], entry[1]
        page = entry[2] if len(entry) > 2 else None
        moving = page is None and live.get(t, 0.0) >= SUBTITLE_LIVE
        if moving:
            lines = [c for c in map(clean_moving_line, lines) if c]
        near = " ".join(p[1] for p in paragraphs if abs(p[0] - t) <= 25)
        # 글자 2연쇄로 견준다 — 움직이는 배경 위 자막은 OCR이 깨져 낱말로는 겹치지 않는다
        sw, nw = page_key(" ".join(lines)), page_key(near)
        # 2연쇄가 몇 개뿐인 조각('se an')은 발화와 우연히 겹친다(합성 강의: 영어 발화의 'response'
        # 에 전부 들어 있었다) — 근거로 삼으려면 여섯 개는 있어야 한다. 짧은 진짜 자막은 옆 자막
        # 화면에 잇닿아 있으면 아래에서 같은 영상으로 묶인다
        ratio = len(sw & nw) / len(sw) if len(sw) >= 6 else 0.0
        short = len(lines) <= 3 and sum(len(x) for x in lines) <= 90
        info.append((t, lines, page, moving, moving and short and ratio >= SUBTITLE_MATCH_RATIO))
    kinds = ["자막" if x[4] else "슬라이드" for x in info]
    grown = True
    while grown:                       # 자막 옆의 움직이는 화면은 같은 영상이다
        grown = False
        for i, x in enumerate(info):
            if kinds[i] != "자막" and x[3] and (
                    (i > 0 and kinds[i - 1] == "자막")
                    or (i + 1 < len(info) and kinds[i + 1] == "자막")):
                kinds[i], grown = "자막", True
    return [(x[0], x[1], x[2], k) for x, k in zip(info, kinds)
            if x[1] or k == "자막"]


def video_spans(labeled, ends=None, motion=None):
    """자막이 잇따라 나오는 구간 = 화면에서 영상이 재생된 구간.

    labeled 는 시간순이고, ends[i] 는 i번째 화면이 내려간 시각이다(없으면 마지막
    자막 화면이 뜬 시각까지로 본다). motion(초마다 화면이 움직인 비율)이 있으면
    시작을 움직임이 시작된 초까지 거슬러 올라간다 — 영상 첫 화면은 자막 OCR이 비어
    버려지는 일이 있어, 글자만 믿으면 구간 시작이 몇 초 늦었다.
    """
    spans, run = [], []
    for i, e in enumerate(labeled + [(None, None, None, "끝")]):
        if e[-1] == "자막":
            run.append(i)
            continue
        if len(run) >= 2:
            last = run[-1]
            a, b = labeled[run[0]][0], (ends[last] if ends else labeled[last][0])
            if motion:
                s = min(int(a) + 1, len(motion))
                while s > 0 and motion[s - 1] >= SUBTITLE_LIVE and (
                        not spans or s - 1 > spans[-1][1]):
                    s -= 1
                prev_end = max((labeled[j][0] for j in range(run[0])), default=0.0)
                a = max(min(a, s - 0.5), prev_end)   # 앞 화면이 뜬 때보다 앞서지는 않는다
            spans.append((a, b))
        run = []
    return spans


def screen_ends(screens, duration):
    """화면마다 언제까지 떠 있었는지 — 다음 화면이 뜬 시각(마지막은 영상 끝)."""
    order = sorted(range(len(screens)), key=lambda i: screens[i][0])
    ends = {}
    for n, i in enumerate(order):
        ends[i] = screens[order[n + 1]][0] if n + 1 < len(order) else duration
    return order, ends


# 영상 출처 표시 — 주소, 그리고 참고문헌 표기의 '[video file]'·'[동영상]'(주소가 줄 끝에서 잘려도 남는다)
VIDEO_URL = re.compile(r"youtu|watch\?v=|vimeo|tv\.naver|naver\.me|tvcast|dailymotion"
                       r"|\[\s*video(?: file)?\s*\]|\[\s*(?:동영상|영상)\s*\]", re.I)


def other_script(text: str, lang: str) -> bool:
    """글이 강의 언어와 다른 문자로 쓰였는가 — 영어 강의 속 한글, 한국어 강의 속 영어 문장."""
    han = sum("가" <= c <= "힣" for c in text)
    lat = sum(c.isascii() and c.isalpha() for c in text)
    return (lang == "en" and han >= 6 and han > lat) or (lang == "ko" and lat >= 20 and lat > 3 * han)


def motion_video_spans(motion, screens, win=20, lang=None):
    """슬라이드 쪽이 오래·크게 움직이고, 그 무렵 영상이라는 표시가 화면에 있으면 재생 영상이다.

    자막 글자도 없고 교수와 같은 언어인 영상(인터뷰 등)은 앞의 두 근거로는 못 찾는다
    (실강의: 94초 중 73초가 움직인 인터뷰가 교수의 말로 실렸다 — 보통 슬라이드는 70초 중 1초).
    움직임만 믿으면 판서·반복 애니메이션까지 영상이 되어 교수의 말이 📺 가 되므로 셋을 함께 본다:
      · 20초 넘게 움직임이 이어지고(4초 이하의 멈춤은 잇는다) 그중 60% 넘는 초가 움직이며
        평균 움직임이 0.08 이상(판서는 가는 선이라 훨씬 작다)
      · 그 구간이나 직전 30초에 뜬 화면에 영상 출처(youtube 주소, [video file] 표기)가 있거나,
        그 구간의 짧은 화면 글자가 2장 이상 강의 언어와 다른 문자다 — 영어 강의에서 튼
        영어 인터뷰의 한글 자막(실강의: 말도 교수와 같은 영어라 언어 근거가 없었다)
      · 강의의 40% 이상이 정지 슬라이드다 — 카메라만 비추는 강의는 움직임으로 가를 수 없다
    예전에는 20초 창을 밀며 보아서, 두 영상 사이에 7초 뜬 교수 슬라이드와 그 설명까지, 그리고
    영상 앞 슬라이드가 넘어간 순간(한 초의 움직임)부터 영상으로 묶였다(합성 강의).
    """
    if not motion or sum(1 for m in motion if m < 0.05) < 0.4 * len(motion):
        return []
    runs, first, last = [], None, None
    for n, m in enumerate(list(motion) + [0.0] * 6):
        if m > 0.02:
            if first is None:
                first = n
            last = n
        elif first is not None and n - last > 4:
            runs.append((first, last))
            first = None
    def video_like(k, w):
        return motion[k] >= 0.08 and sum(1 for m in w if m > 0.02) >= 4 and sum(w) / len(w) >= 0.08

    spans = []
    for s, e in runs:
        # 앞뒤에 이어 붙은 슬라이드 넘김·커서 움직임은 뗀다 — 영상다운 움직임(0.08 이상이 5초 중
        # 4초)이 시작된 초와 끝난 초로 다듬는다(실강의: 영상 직전 교수의 말 4초, 교수가 기사를 읽는
        # 12초가 커서 움직임과 이어져 📺 가 됐다)
        starts = [k for k in range(s, e + 1) if video_like(k, motion[k:k + 5])]
        ends = [k for k in range(s, e + 1) if video_like(k, motion[max(0, k - 4):k + 1])]
        if not starts or not ends or ends[-1] <= starts[0]:
            continue
        s, e = starts[0], ends[-1]
        seconds = motion[s:e + 1]
        if (len(seconds) >= win and sum(1 for m in seconds if m > 0.02) >= 0.6 * len(seconds)
                and sum(seconds) / len(seconds) >= 0.08):
            spans.append((max(0.0, s - 0.5), e + 0.5))

    def cue(a, b):
        # 출처 슬라이드는 영상 30초 전쯤 뜨기도 한다 — 교수가 대본 슬라이드 몇 장을 먼저 보여 준
        # 뒤 영상을 틀었다(실강의: 26초 전). 예전의 20초 창은 시작을 앞당겨 우연히 잡았었다
        near = [sc for sc in screens if a - 30 <= sc[0] < b]
        if any(VIDEO_URL.search(" ".join(sc[1])) for sc in near):
            return True
        subs = [sc for sc in near if a <= sc[0] and not sc[2] and 0 < len(sc[1]) <= 3
                and other_script(" ".join(sc[1]), lang)]
        return bool(lang) and len(subs) >= 2
    return [(a, b) for a, b in spans if cue(a, b)]


def mark_video(spans, screens, extra):
    """재생 영상 구간을 보태고, 그 안의 (강의자료 쪽이 아닌) 화면을 영상 화면으로 바꾼다."""
    if not extra:
        return list(spans), screens
    merged = []
    for a, b in sorted(list(spans) + list(extra)):
        if merged and a <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    screens = [(t, lines, page, "자막" if page is None and any(x - 0.5 <= t < y for x, y in extra)
                else kind) for t, lines, page, kind in screens]
    return merged, screens


def add_language_spans(spans, screens, swapped, motion):
    """다른 언어로 다시 읽힌 구간에서 슬라이드 쪽이 움직이고 있었으면 재생 영상이다.

    교수의 말과 언어가 다르고 화면이 움직인다 — 자막 글자가 없어도(뉴스 화면 등)
    재생된 영상이라고 볼 근거가 충분하다. 그 사이의 화면은 영상 화면으로 묶는다.
    시작은 화면이 계속 움직이기 시작한 초로 당긴다 — 영어 강의의 교수가 슬라이드의 한국어
    기사를 소리 내 읽은 뒤 영상을 틀면, 읽은 대목부터 영상으로 묶였다(실강의 12초).
    """
    moving = lambda a, b: (lambda s: bool(s) and sum(s) / len(s) >= SUBTITLE_LIVE)(
        motion[int(a):int(b) + 1])

    def settled_start(a, b):
        # 영상다운 움직임(0.08 이상)이어야 한다 — 교수의 커서·판서(0.03~0.05)가 이어진 초를
        # 영상 시작으로 잡아, 기사를 읽는 교수의 말이 여전히 📺 가 됐다(실강의)
        for k in range(int(a), int(b) + 1):
            w = motion[k:k + 5]
            if motion[k] >= 0.08 and sum(1 for m in w if m > 0.02) >= 4 and sum(w) / len(w) >= 0.08:
                return max(a, k - 0.5)
        return a

    extra = []
    for a, b, _lang in sorted(swapped):
        if not moving(a, b):
            continue
        a = settled_start(a, b)
        # 사이가 1분 이내이고 그동안에도 화면이 움직였으면 같은 영상이다 — 그 사이에는 영상 속
        # 노래나, 교수 언어로 '번역'되어 ⚠ 없이 남은 말이 있다(실강의). 교수의 말로 읽히면 안 된다.
        if extra and a - extra[-1][1] <= 60 and moving(extra[-1][1], a):
            extra[-1] = (extra[-1][0], b)
        else:
            extra.append((a, b))
    return mark_video(spans, screens, extra)


def foreign_spans(segments, main_lang):
    """교수의 언어와 다른 문자로 옮겨진 구간 — 영어 강의 속 한글 문장, 한국어 강의 속 영어 대목.

    재독으로 바뀐 구간만 보면, 처음부터 그 언어로 옮겨진 첫 문장(뉴스 앵커의 첫마디)이
    빠져 영상 구간이 2초 늦게 시작됐다(실강의). 한국어 강의의 영어는 용어가 흔하므로
    영어가 압도적인 긴 문장만 센다. 구간은 세그먼트가 아니라 **그 문자로 된 낱말**이 있는 곳까지다
    — 영어 세그먼트 하나가 영상 직후 교수의 한국어 첫마디까지 품어, 그 말이 📺 가 됐다(합성 강의).
    """
    def foreign_word(w):
        han = any("가" <= c <= "힣" for c in w)
        lat = any(c.isascii() and c.isalpha() for c in w)
        return han and not lat if main_lang == "en" else lat and not han

    out = []
    for s in segments:
        if not other_script(s[2], main_lang):
            continue
        a, b = s[0], s[1]
        words = [w for w in (s[4] if len(s) > 4 and s[4] else []) if foreign_word(w[2])]
        if words:
            a, b = words[0][0], words[-1][1]
        if out and a - out[-1][1] <= GAP_JOIN_SEC:
            out[-1] = (out[-1][0], max(out[-1][1], b), out[-1][2])
        else:
            out.append((a, b, "ko" if main_lang == "en" else "en"))
    return out


def join_video_spans(spans, screens, foreign=(), phrases=(), max_gap=VIDEO_JOIN_SEC):
    """짧은 틈으로 끊긴 재생 영상 구간을 잇는다.

    설명 영상은 움직이는 장면과 정지된 글자 카드가 번갈아 나온다 — 정지 카드에서 구간이
    끊기면 그동안의 영상 내레이션이 교수의 말로 읽혔다(실강의). 틈이 max_gap 이내면 잇고,
    90초 이내라도 그 사이 화면이 2장 이상 빠르게(평균 20초 미만) 바뀌었으면 글자 카드로 보고
    잇는다(실강의: 32초 동안 카드 4장). 교수의 슬라이드는 한 장이 오래 떠 있다. 잇지 않는 경우:
      · 틈에 강의자료 쪽이 떴다 — 교수가 슬라이드로 돌아온 것이다
      · 양쪽이 교수와 다른 언어의 영상인데 틈에 교수 언어의 말이 있다 — 두 클립 사이에
        교수가 논평한 것이다(실강의: 한국어 뉴스 두 클립 사이의 영어 논평 18초)
    """
    def is_foreign(a, b):
        return any(x < b and a < y for x, y, _l in foreign)

    joined = []
    for a, b in sorted(spans):
        cards = [s for s in screens if joined and joined[-1][1] <= s[0] < a]
        near = joined and (a - joined[-1][1] <= max_gap or (
            a - joined[-1][1] <= 90 and len(cards) >= 2 and (a - joined[-1][1]) / len(cards) < 20))
        if near:
            pa, pb = joined[-1]
            gap_pdf = any(s[2] for s in screens if pb <= s[0] < a)
            gap_talk = (is_foreign(pa, pb) or is_foreign(a, b)) and any(
                pb <= p[0] < a and not is_foreign(p[0], p[1]) for p in phrases)
            if not gap_pdf and not gap_talk:
                joined[-1] = (pa, max(pb, b))
                continue
        joined.append((a, b))
    screens = [(t, lines, page, "자막" if page is None and any(x - 0.5 <= t < y for x, y in joined)
                else kind) for t, lines, page, kind in screens]
    return joined, screens


def trim_still_tail(spans, screens, motion, duration):
    """재생 영상 구간의 꼬리에 붙은 멈춘 화면을 떼어 낸다.

    영상이 끝나고 슬라이드로 서서히 넘어가는 전환은 움직임으로 잡혀, 그 뒤 슬라이드가 영상
    화면으로 묶이고 구간이 그 슬라이드가 내려갈 때까지 늘어났다(실강의: 교수가 슬라이드를
    설명한 42초가 📺 로 실렸다). 구간 끝에 있는 화면이 뜬 뒤(전환 2초 제외) 3초 넘게 전혀
    움직이지 않았으면 영상이 아니다 — 구간을 그 화면이 뜬 때에서 끝낸다. 강의자료 쪽이면
    움직임과 상관없이 교수의 슬라이드다. 영상 속 정지 글자 카드는 구간 끝에 오지 않는 한
    그대로 둔다.
    """
    order, ends = screen_ends(screens, duration)

    def still(i):
        if screens[i][2]:
            return True
        # n초에 뜬 화면(t = n - 0.5)은 n+1초까지 전환이 이어질 수 있고, 다음 화면의 전환도
        # 1초 앞서 시작될 수 있다 — 그 사이만 본다. '전혀 안 움직임'을 요구하면 교수가 판서하거나
        # 커서를 움직인 몇 초(0.03~0.7) 때문에 영상 뒤 슬라이드를 떼지 못했다(실강의) — 영상처럼
        # 움직인(0.05 초과) 초가 20% 이하면 멈춘 화면이다
        seconds = motion[int(screens[i][0]) + 3:int(ends[i])]
        return len(seconds) >= 3 and sum(1 for m in seconds if m > 0.05) <= 0.2 * len(seconds)

    def frozen_since(a, b):
        """구간 끝까지 3초 넘게 완전히 멈춰 있고 그 직전에 전환(0.3 이상)이 있었으면 그 전환 시각.
        영상이 끝나 마지막 화면이 멈춘 채 다음 슬라이드가 몇 초 뒤에 뜨면, 그 사이 교수의 말이
        📺 로 실렸다(실강의: 5초). 글자가 없는 멈춘 화면은 화면 목록에 없어 위 규칙으로 못 뗀다."""
        k = min(int(b), len(motion)) - 1
        quiet = 0
        while k > a and motion[k] <= 0.02:
            k, quiet = k - 1, quiet + 1
        return k - 0.5 if quiet >= 3 and k > a and motion[k] >= 0.3 else None

    out, cut = [], []
    for a, b in spans:
        tail = [i for i in order if a < screens[i][0] < b]
        while tail and still(tail[-1]):
            i = tail.pop()
            cut.append((screens[i][0], b))
            b = screens[i][0]
        end = frozen_since(a, b) if motion else None
        if end is not None:
            cut.append((end, b))
            b = end
        if b > a:
            out.append((a, b))
    screens = [(t, lines, page, "슬라이드" if kind == "자막" and any(x <= t < y for x, y in cut)
                else kind) for t, lines, page, kind in screens]
    return out, screens


def write_markdown(out_path: Path, src: Path, info, paragraphs, cfg, screens=(),
                   pdf_name=None, pdf_tag=None, vad_lost=0.0, repaired=(), lost=(), spans=None):
    """screens 는 label_slides 가 돌려준 (시각, 줄, 쪽, 종류) 목록이다.
    spans(재생 영상 구간)를 주지 않으면 화면만 보고 정한다."""
    # 화면이 언제까지 떠 있었는지 — "이 발화가 어느 화면에 대한 것인가"를 확정한다
    order, ends = screen_ends(screens, info.duration)
    if spans is None:
        spans = video_spans([screens[i] for i in order], [ends[i] for i in order])
    real_slides = sum(1 for s in screens if s[3] == "슬라이드")
    from_pdf = sum(1 for s in screens if s[3] == "슬라이드" and s[2])
    suspect = [p[0] for p in paragraphs if len(p) > 2 and p[2]]

    def in_video(t):
        # 영상의 시작(움직임이 시작된 초)과 끝(다음 화면이 뜬 때)을 알므로 여유는 거의
        # 없어도 된다. 예전의 앞뒤 15초 여유는 영상 직후 교수가 "영상에서 보셨듯이…"
        # 라고 한 말까지 교수의 말이 아니라고 표시했다. 끝에는 여유를 두지 않는다 —
        # 0.5초 여유만으로도 슬라이드가 뜨자마자 시작한 교수의 말이 📺 가 되었다(실강의).
        return any(a - 0.5 <= t < b for a, b in spans)

    course, week, period = parse_course(src.stem)
    lines = ["---", f"과목: {course}"]
    if week is not None:
        lines += [f"주차: {week}", f"교시: {period}"]
    lines += [f"언어: {info.language}", f"길이: {fmt_ts(info.duration)}", f"원본: {src.name}"]
    if pdf_name:
        lines.append(f"강의자료: {pdf_name}")
    lines += [f"모델: {cfg['model']}", f"생성: {datetime.datetime.now():%Y-%m-%d %H:%M}",
              "---", "", f"# {src.stem}", ""]
    if screens:
        if from_pdf:
            lines.append(f"- **슬라이드**: {real_slides}장 중 **{from_pdf}장은 강의자료 "
                         f"PDF 원문**을 그대로 실었습니다(쪽번호 표시). 나머지는 화면에서 "
                         f"읽어낸 글자입니다.")
        else:
            lines.append(f"- **슬라이드**: 화면에서 {real_slides}장을 읽어 함께 실었습니다 "
                         f"(`> 🖵 슬라이드` 로 표시)")
    if spans:
        lines.append(f"- 📺 **강의 중 재생된 영상 {len(spans)}곳**: "
                     f"`📺` 표시가 붙은 화면 글자와 문단은 교수의 말이 아니라 "
                     f"재생된 영상의 자막·말소리입니다. 교수의 주장으로 읽지 마세요.")
    if suspect:
        lines.append(f"- ⚠ **인식이 흔들린 문단 {len(suspect)}개**(`⚠` 표시): "
                     f"원문과 다를 수 있습니다. 이 문단만으로 사실을 단정하지 마세요.")
    if repaired:
        lines.append(f"- 🔁 처음 인식에서 빠졌거나 다른 언어로 잘못 옮겨진 말소리 {len(repaired)}곳"
                     f"({sum(b - a for a, b in repaired):.0f}초)을 다시 읽어 채웠습니다.")
    if lost:
        lines.append(f"- ⚠ **말소리가 있었지만 옮기지 못한 구간 {len(lost)}곳**이 본문에 "
                     f"표시되어 있습니다(음악·잡음이거나 알아들을 수 없는 발화).")
    if vad_lost >= 0.15:
        lines.append(f"- ⚠ **무음으로 제외된 구간이 {vad_lost * 100:.0f}%**입니다. "
                     f"녹음 음량이 낮으면 일부 발화가 빠졌을 수 있습니다.")

    # 화면 차례 — 33,000자짜리 문서를 처음부터 훑지 않고 필요한 대목만 펼칠 수 있다
    toc, seen = [], set()
    for i in order:
        t, body, page, kind = screens[i]
        # 스쳐 지나간 화면(한 줄짜리)은 차례에 올리지 않는다 — 목차가 아니라 잡음이 된다
        if kind != "슬라이드" or not body or (len(body) < 2 and not page):
            continue
        title = slide_title(body)
        if not title or title in seen:
            continue
        seen.add(title)
        toc.append((t, page, title))
    if toc:
        lines += ["", "## 화면 차례", ""]
        for t, page, title in toc:
            lines.append(f"- `[{fmt_ts(t)}]`{f' · {page}' if page else ''} {title}")

    lines += ["", "---", ""]

    # 잇따른 자막 화면은 한 블록으로 묶는다 — 재생 영상 동안 2~3초마다 잡힌 화면이
    # 하나하나 실리면 같은 자막이 되풀이되고 깨진 줄이 문서를 덮는다
    blocks, run = [], []
    for i in order:
        if screens[i][3] == "자막":
            run.append(i)
            continue
        if run:
            blocks.append(run)
            run = []
        blocks.append([i])
    if run:
        blocks.append(run)

    def block_start(b):
        # 자막 블록은 영상이 실제로 시작된 때(움직임이 시작된 초)에 놓는다
        t = screens[b[0]][0]
        if screens[b[0]][3] == "자막":
            t = next((a for a, e in spans if a - 0.5 <= t <= e), t)
        return t

    # 말과 화면 글자를 시간순으로 엮는다 — 어떤 화면을 보며 한 말인지 드러난다
    timeline = ([(block_start(b), 0, b) for b in blocks]
                + [(a, 1, (a, b)) for a, b in lost]
                + [(p[0], 2, (p[1], len(p) > 2 and p[2])) for p in paragraphs])
    for start, kind, value in sorted(timeline, key=lambda x: (x[0], x[1])):
        if kind == 0:
            _t, body, page, what = screens[value[0]]
            span = f"[{fmt_ts(start)} – {fmt_ts(ends[value[-1]])}]"
            if what == "자막":
                lines.append(f"> **📺 영상 자막 {span}**")
                shown = []
                for i in value:
                    shown += [ln for ln in screens[i][1]
                              if ln not in shown and not looks_garbled(ln)]
                lines += [f"> {ln}" for ln in shown]
            else:
                lines.append(f"> **🖵 슬라이드{f' {page}' if page else ''} {span}**")
                lines += [f"> {ln}" for ln in body]
        elif kind == 1:
            a, b = value
            lines.append(f"**[{fmt_ts(a)}]** ⚠ *(여기서 {b - a:.0f}초 동안 말소리가 있었지만 "
                         f"옮기지 못했습니다)*")
        else:
            text, is_bad = value
            flags = ("📺 " if in_video(start) else "") + ("⚠ " if is_bad else "")
            lines.append(f"**[{fmt_ts(start)}]** {flags}{text}")
        lines.append("")

    body = "\n".join(lines) + "\n"
    mark = json.dumps({"src": src.name, "bytes": src.stat().st_size,
                       "cfg": config_fingerprint(cfg), "sha": md_body_hash(body),
                       "pdf": pdf_tag if pdf_tag is not None else (pdf_name or "")},
                      ensure_ascii=False)
    tmp = out_path.with_name(out_path.name + ".tmp")
    # 마커까지 한 번에 쓴다. 예전에는 두 번 나눠 써서 그 사이에 중단되면 그 MD가
    # '편집됨'으로 굳어 영구히 다시 만들어지지 않았다.
    tmp.write_text(body + f"{MARKER} {mark} -->\n", encoding="utf-8")
    try:
        tmp.replace(out_path)
    except OSError as e:
        raise RuntimeError(f"결과를 저장하지 못했습니다 ({e}). 전사 내용은 {tmp.name}에 남겨둡니다") from e


def transcribe_file(model, cfg, src: Path, out_path: Path, tmp_dir: Path, idx, total_n, tess):
    from faster_whisper import decode_audio

    size_mb = src.stat().st_size / 1024 / 1024
    print(f"\n[{idx}/{total_n}] ▶ {src.name} ({size_mb:.1f} MB)")

    media = probe_media(src)
    if media["duration"] and media["duration"] > 4 * 3600:
        print(f"  ⚠ {fmt_ts(media['duration'])}짜리 긴 파일입니다. 메모리를 많이 사용합니다.")

    wav = tmp_dir / (safe_stem(out_path.stem, 60) + ".wav")
    try:
        status("  오디오 추출 중...")
        used_ffmpeg = extract_audio(src, wav, media)
        status("")
        sys.stdout.write("\r")
        if used_ffmpeg is None:
            print("  (ffmpeg가 없어 원본을 직접 디코딩합니다)")
        # 한 번 풀어 둔 소리를 전사와 복구가 함께 쓴다
        audio = decode_audio(str(wav if used_ffmpeg else src), sampling_rate=16000)
    finally:
        wav.unlink(missing_ok=True)

    # 화면을 먼저 읽는다 — 강의자료 후보를 화면과 맞춰 본 뒤, 실제로 맞는 자료의
    # 전문용어만 음성 인식 힌트로 넘기기 위해서다(다른 과목 자료의 용어가 섞이면 안 된다).
    loaded, pdf_all = load_slide_materials(src)
    slides, live, motion = [], {}, []
    t_screen = time.monotonic()
    if cfg["슬라이드_읽기"] and tess and media["video"]:
        try:
            # 발화 언어는 아직 모르지만, 순서는 두 판본의 점수가 똑같을 때만 영향을 준다
            langs = ocr_lang_options(cfg, cfg["language"] if cfg["language"] != "auto" else "ko")
            slides, live, motion = extract_slides(src, tmp_dir, tess,
                                                  media["duration"] or len(audio) / 16000, langs)
            print(f"  슬라이드 {len(slides)}장을 읽었습니다." if slides
                  else "  (화면에서 읽을 만한 글자를 찾지 못했습니다)")
        except Exception as e:
            # print만 하면 창을 닫은 뒤 흔적이 없다. 기록에 남겨 사후 진단이 되게 한다.
            log(f"  ⚠ 슬라이드를 읽지 못했습니다: {out_path.name} — {e}"
                f" (음성 전사는 정상 저장)")
    screen_sec = time.monotonic() - t_screen

    chosen = pick_materials(slides, loaded)
    pdf_used = [m.name for m, _t in chosen]
    pdf_pages, pdf_labels, _owner = material_pages(chosen)
    for m, _t in loaded:
        if m.name not in pdf_used:
            print(f"  (강의자료 {m.name}은(는) 이 강의 화면과 맞지 않아 쓰지 않습니다)")
    if pdf_pages:
        print(f"  강의자료 {', '.join(pdf_used)} (총 {len(pdf_pages)}쪽)을 함께 씁니다.")
        if slides:
            slides = align_slides_to_pdf(slides, pdf_pages, pdf_labels)
            matched = sum(1 for s in slides if s[2])
            print(f"  그중 {matched}장을 강의자료 원문으로 바꿨습니다 "
                  f"(잡음 없이 표·빈칸까지 그대로).")
    hot = pdf_hotwords(pdf_pages) if pdf_pages else ""

    t0 = time.monotonic()
    # 순차 경로: 온도 폴백 + 반복/저신뢰 감지가 살아 있고 타임스탬프가 정밀하다.
    # 낱말 시각은 비용이 2.5%이고, 뭉개진 세그먼트를 구절로 나누고 빠진 구간을 찾는 데 쓴다.
    segments, info = model.transcribe(
        audio, language=None if cfg["language"] == "auto" else cfg["language"],
        beam_size=cfg["beam_size"], vad_filter=True, vad_parameters=VAD_PARAMS,
        word_timestamps=True, hotwords=hot or None, **QUALITY)
    print(f"  언어: {info.language} · 길이: {fmt_ts(info.duration)}")

    collected, total = [], max(info.duration, 0.01)
    for seg in segments:
        collected.append((seg.start, seg.end, seg.text, is_suspect(seg), segment_words(seg)))
        done = time.monotonic() - t0
        eta = done / max(seg.end, 1) * max(total - seg.end, 0)
        status(f"  진행률: {min(seg.end / total * 100, 100):5.1f}%  "
               f"[{fmt_ts(seg.end)}/{fmt_ts(total)}]  남은 시간 약 {fmt_ts(eta)}")
    status("")
    sys.stdout.write("\r")

    collected, swapped = reread_suspects(model, audio, collected, info.language, cfg)
    if swapped:
        print(f"  다른 언어로 말한 대목 {len(swapped)}곳을 그 언어로 다시 읽었습니다 "
              f"({', '.join(sorted({l for *_s, l in swapped}))}).")
    added, repaired, lost = repair_gaps(model, audio, collected, cfg)
    if repaired:
        print(f"  처음 인식에서 빠진 말소리 {len(repaired)}곳을 다시 읽어 채웠습니다.")
    collected = sorted(collected + added, key=lambda s: s[0])
    repaired = sorted(repaired + [(a, b) for a, b, _l in swapped])

    phrases = split_phrases(collected)
    if not phrases:
        raise RuntimeError("음성이 감지되지 않았습니다 (오디오 트랙이 없거나 무음일 수 있습니다)")

    # 화면에서 읽은 글자에도 되풀이되는 배너가 있으면 뺀다 (PDF 쪽은 이미 뺐다).
    # 그림 화면에서 나온 잡음 줄('NW', '| Sy \')도 뺀다 — 글자 3자 미만에 숫자도 없는 줄,
    # 그리고 표 괘선을 읽은 줄('1 | | | | 1. | |')처럼 낱말다운 조각이 거의 없는 줄.
    # 숫자만 있는 줄은 표 내용일 수 있어 남긴다.
    ocr_only = [i for i, s in enumerate(slides) if len(s) < 3 or not s[2]]
    cleaned = drop_boilerplate([[ln for ln in slides[i][1] if not ocr_junk(ln)] for i in ocr_only])
    for i, body in zip(ocr_only, cleaned):
        slides[i] = (slides[i][0], body) + tuple(slides[i][2:])
    slides = [s for s in slides if s[1]]

    screens = label_slides(slides, [(p[0], p[2]) for p in phrases], live)
    order, ends = screen_ends(screens, info.duration)
    spans = video_spans([screens[i] for i in order], [ends[i] for i in order], motion)
    foreign = sorted(set(swapped) | set(foreign_spans(collected, info.language)))
    spans, screens = add_language_spans(spans, screens, foreign, motion)
    spans, screens = mark_video(spans, screens, motion_video_spans(motion, screens, lang=info.language))
    spans, screens = join_video_spans(spans, screens, foreign, phrases)
    spans, screens = trim_still_tail(spans, screens, motion, info.duration)

    # 다른 언어 영상 안에 남은 ⚠ 를 그 언어로 다시 읽는다
    collected, forced = reread_video_suspects(model, audio, collected, spans, foreign, cfg)
    if forced:
        print(f"  영상 속 흔들린 말 {len(forced)}곳을 영상의 언어로 다시 읽었습니다.")
        repaired = sorted(repaired + [(a, b) for a, b, _l in forced])
    del audio
    elapsed = time.monotonic() - t0

    # 슬라이드가 바뀐 때와 영상이 시작·끝난 때 문단을 끊는다 — 구절도 그 시각에서 나눈다
    cuts = [a - 0.5 for a, _b in spans] + [b for _a, b in spans]
    phrases = split_phrases(collected, cuts)
    paragraphs = group_paragraphs(phrases, [s[0] for s in screens if s[3] == "슬라이드"] + cuts)
    vad_lost = 1 - (getattr(info, "duration_after_vad", info.duration) / total)
    write_markdown(out_path, src, info, paragraphs, cfg, screens,
                   ", ".join(pdf_used) if pdf_used else None, ", ".join(pdf_all),
                   vad_lost, repaired, lost, spans)
    speed = info.duration / elapsed if elapsed > 0 else 0
    log(f"  ✔ 완료: {out_path.name} (전사 {fmt_ts(elapsed)}, 실시간 대비 {speed:.1f}배"
        + (f", 복구 {len(repaired)}곳" if repaired else "")
        + (f", 슬라이드 {len(screens)}장 · 화면 읽기 {fmt_ts(screen_sec)}" if screens else "") + ")")


def cleanup_stale_temp():
    """지난 실행이 강제 종료되며 남긴 임시 파일을 치운다.

    창을 X로 닫으면 파이썬의 정리 코드가 돌지 못해 WAV·프레임 이미지가 수백 MB씩
    남는다. 강제 종료 자체는 막을 수 없으므로 다음 실행이 청소한다.
    도구 폴더 안(TMP_ROOT)과, 그곳에 쓸 수 없을 때 물러서는 %TEMP% 를 함께 훑는다.
    """
    freed = 0
    cutoff = time.time() - 6 * 3600
    stale = []
    # ocr_* 는 화면 한 장을 읽을 때 쓰는 폴더다 — 강제 종료되면 이것도 남는다(강의 화면 이미지).
    # 흔한 이름이라 도구 폴더 안에서만 치운다 — %TEMP% 에서는 다른 프로그램의 것일 수 있다
    for root, patterns in ((TMP_ROOT, ("transcriber_*", "ocr_*")),
                           (Path(tempfile.gettempdir()), ("transcriber_*",))):
        for pattern in patterns:
            try:
                stale += [d for d in root.glob(pattern) if d.is_dir()]
            except OSError:
                pass
    for d in stale:
        try:
            if d.stat().st_mtime > cutoff:
                continue
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                freed += size
        except OSError:
            continue
    for t in OUT_DIR.glob("*.md.tmp"):
        try:
            if t.stat().st_mtime < cutoff:
                freed += t.stat().st_size
                t.unlink()
        except OSError:
            continue
    if freed:
        print(f"지난 실행이 남긴 임시 파일 {freed / 1024 / 1024:.0f}MB를 정리했습니다.")


def check_ocr_langs(tess: str, options):
    """Tesseract가 실제로 가진 언어인지 미리 확인한다. 없으면 사유를 돌려준다."""
    try:
        r = subprocess.run([tess, "--list-langs"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           stdin=subprocess.DEVNULL, timeout=60)
    except Exception as e:
        return f"Tesseract를 실행할 수 없습니다 ({e})"
    have = {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
    want = {c for opt in options for c in opt.split("+") if c}
    missing = sorted(want - have)
    if missing:
        return (f"언어 데이터가 없습니다: {', '.join(missing)} "
                f"(가진 것: {', '.join(sorted(have)) or '없음'})")
    return None


_sleep_block = None   # (handle, reason_context) — 살려 둬야 사유 문자열이 유효하다


def prevent_sleep(enable) -> bool:
    """전사 중에는 시스템이 절전으로 들어가지 못하게 막는다. 걸었으면 True.

    Windows의 절전 타이머는 CPU 부하가 아니라 사용자 입력 유휴를 본다. 즉 몇 시간짜리
    무인 전사도 키보드를 안 건드리면 그냥 잠들어 버린다.

    화면은 일부러 막지 않는다(PowerRequestSystemRequired만 건다) —
    패널은 꺼지고 전사는 계속된다.

    SetThreadExecutionState가 아니라 PowerSetRequest를 쓰는 이유는
    `powercfg /requests`의 SYSTEM 칸에 아래 사유 문자열까지 찍혀서
    나중에 "무엇이 이 기계를 깨워 두는가"를 감사할 수 있기 때문이다.
    레거시 API는 그 목록에 아예 나타나지 않아 검증이 불가능하다.

    실패해도 전사 자체에는 영향이 없으므로 조용히 넘어간다.
    """
    global _sleep_block
    if os.name != "nt":
        return False

    POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x00000001
    # POWER_REQUEST_TYPE: 0=Display 1=System 2=AwayMode 3=Execution
    PowerRequestSystemRequired = 1
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        k32.SetThreadExecutionState.restype = wintypes.DWORD

        if enable:
            if _sleep_block is not None:
                return True

            class _Detailed(ctypes.Structure):
                _fields_ = [("LocalizedReasonModule", wintypes.HMODULE),
                            ("LocalizedReasonId", wintypes.ULONG),
                            ("ReasonStringCount", wintypes.ULONG),
                            ("ReasonStrings", ctypes.POINTER(wintypes.LPWSTR))]

            class _Reason(ctypes.Union):
                _fields_ = [("Detailed", _Detailed),
                            ("SimpleReasonString", wintypes.LPWSTR)]

            class _ReasonContext(ctypes.Structure):
                _fields_ = [("Version", wintypes.ULONG),
                            ("Flags", wintypes.DWORD),
                            ("Reason", _Reason)]

            k32.PowerCreateRequest.argtypes = [ctypes.POINTER(_ReasonContext)]
            k32.PowerCreateRequest.restype = wintypes.HANDLE
            k32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
            k32.PowerSetRequest.restype = wintypes.BOOL

            ctx = _ReasonContext()
            ctx.Version = 0
            ctx.Flags = POWER_REQUEST_CONTEXT_SIMPLE_STRING
            ctx.Reason.SimpleReasonString = "Transcriber: transcription in progress"

            h = k32.PowerCreateRequest(ctypes.byref(ctx))
            if h and k32.PowerSetRequest(h, PowerRequestSystemRequired):
                _sleep_block = (h, ctx)
                return True
            # 최신 API가 안 되면 레거시로 물러선다(감사는 안 되지만 동작은 한다)
            return k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
        if _sleep_block is not None:
            k32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            h = _sleep_block[0]
            k32.PowerClearRequest(h, PowerRequestSystemRequired)
            k32.CloseHandle(h)
            _sleep_block = None
        else:
            k32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass
    return False


def main():
    started = time.monotonic()
    print("=" * 58)
    print(" Transcriber — 영상·음성을 Markdown 전사문으로")
    print("=" * 58)

    lock = acquire_lock()
    if lock is None:
        print("\nTranscriber가 이미 실행 중입니다. 기존 창의 작업이 끝난 뒤 다시 실행해주세요.")
        return 0
    if lock is False:
        print("⚠ 중복 실행 방지 잠금을 걸 수 없습니다. 창을 두 개 열지 않도록 주의하세요.")

    for d in (IN_DIR, OUT_DIR, MODELS_DIR, PDF_DIR):
        d.mkdir(exist_ok=True)
    cleanup_stale_temp()

    # 바로가기에 영상을 끌어다 놓으면 입력 폴더를 찾아 들어가지 않아도 된다.
    only = None
    if sys.argv[1:]:
        only = set(stage_dropped([Path(a) for a in sys.argv[1:]]))
        print(f"\n끌어다 놓은 {len(only)}개를 처리합니다.")

    cfg = load_config()
    try:
        targets, skipped, blocked, ignored, rebuilt = plan_targets(cfg, only)
    except OSError as e:
        print(f"\n✘ \"{IN_DIR.name}\" 폴더를 읽을 수 없습니다: {e}")
        return 1

    if skipped:
        print(f"건너뜀 {len(skipped)}개:")
        for f, why in skipped[:10]:
            print(f"  - {f.name} ({why})")
        if len(skipped) > 10:
            print(f"  ... 외 {len(skipped) - 10}개")
    if blocked:
        print(f"⚠ 아래 파일은 같은 이름의 다른 문서가 이미 있어 건너뜁니다 "
              f"(덮어쓰지 않았습니다). 그 문서를 옮기거나 이름을 바꾼 뒤 다시 실행하세요:")
        for f, out in blocked:
            print(f"  - {f.name} → {out.name}")
    if ignored and only is None:
        print(f"무시된 파일 {len(ignored)}개 (지원하지 않는 형식): "
              f"{', '.join(f.name for f in ignored[:5])}"
              f"{' ...' if len(ignored) > 5 else ''}")
    if not targets:
        print(f"\n변환할 새 파일이 없습니다." if (skipped or blocked or ignored or only)
              else f"\n\"{IN_DIR.name}\" 폴더에 영상이나 음성 파일을 넣고 다시 실행해주세요.")
        return 0

    tess = find_tesseract() if cfg["슬라이드_읽기"] else None
    if tess:
        why = check_ocr_langs(tess, ocr_lang_options(cfg, "ko") + ocr_lang_options(cfg, "en"))
        if why:
            log(f"⚠ 슬라이드 읽기를 끕니다 — {why}")
            tess = None
    if cfg["슬라이드_읽기"] and not shutil.which("ffprobe"):
        print("⚠ ffprobe가 없어 파일 손상 검사와 슬라이드 읽기를 건너뜁니다.")
    slide_note = ("슬라이드 읽기 켬" if tess else
                  "슬라이드 읽기 끔" if not cfg["슬라이드_읽기"] else
                  "슬라이드 읽기 불가")
    print(f"\n변환 대상 {len(targets)}개 · {cfg['model']} · 언어 {cfg['language']} · {slide_note}")
    if rebuilt:
        # 오래 걸리는 작업이 예고 없이 시작되지 않도록 무엇을 왜 다시 만드는지 알린다
        mins = sum(probe_media(f)["duration"] or 0 for f in rebuilt) / 60 * 0.35
        print(f"  그중 {len(rebuilt)}개는 이미 만든 것을 다시 만듭니다 "
              f"(설정·강의자료·엔진이 바뀌었습니다). 이 몫만 대략 {mins:.0f}분입니다.")
        print("  기다릴 상황이 아니면 Ctrl+C 로 멈추세요. 기존 결과는 그대로 남습니다.")
    log(f"── 실행 시작 · 대상 {len(targets)}개 · 건너뜀 {len(skipped)}개 "
        f"· {config_fingerprint(cfg)}", echo=False)
    if prevent_sleep(True):
        print("작업이 끝날 때까지 PC가 절전으로 들어가지 않게 해두었습니다.")

    from faster_whisper import WhisperModel
    cached = any(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else False
    print("모델 로딩 중..." if cached else
          "음성 인식 모델을 내려받는 중입니다. 약 1.6GB이며 처음 한 번만 받습니다.\n"
          "  진행 표시가 없지만 정상 동작 중입니다. 회선에 따라 10~40분 걸릴 수 있습니다.")
    args = dict(device="cpu", compute_type="int8", cpu_threads=asr_threads(),
                download_root=str(MODELS_DIR))
    try:
        try:
            model = WhisperModel(cfg["model"], local_files_only=True, **args)
        except Exception:
            model = WhisperModel(cfg["model"], **args)
    except Exception as e:
        print(f"\n✘ 모델을 불러오지 못했습니다: {e}")
        print("  - 최초 실행이라면 인터넷 연결과 저장 공간(약 2GB)을 확인해주세요.")
        print(f"  - 설정.ini의 model 값(현재: {cfg['model']})이 올바른지 확인해주세요.")
        return 1

    ok, failures = 0, []
    with tempfile.TemporaryDirectory(prefix="transcriber_", dir=tmp_root(),
                                     ignore_cleanup_errors=True) as td:
        for i, (src, out_path) in enumerate(targets, 1):
            try:
                transcribe_file(model, cfg, src, out_path, Path(td), i, len(targets), tess)
                ok += 1
            except KeyboardInterrupt:
                log(f"  ⚠ 사용자가 중단했습니다 (진행 중이던 파일: {src.name})", echo=False)
                raise
            except Exception as e:
                msg = str(e) or type(e).__name__
                if isinstance(e, MemoryError):
                    msg = "메모리가 부족합니다. 영상을 나눠서 변환해주세요"
                elif type(e).__module__.startswith("av") or "Invalid data" in msg:
                    # 디스크 부족 같은 다른 OSError 까지 '파일 손상'으로 덮지 않는다
                    msg = "오디오를 읽을 수 없습니다 (형식이 잘못되었거나 파일이 손상됨)"
                failures.append((src.name, msg))
                log(f"  ✘ 실패: {src.name} ({src.stat().st_size / 1048576:.0f}MB) — {msg}")

    log(f"\n전체 완료: 성공 {ok}개 / 실패 {len(failures)}개", echo=True)
    if failures:
        print("실패 목록:")
        for name, err in failures:
            print(f"  ✘ {name}\n      {err}")
        print(f"\n기록은 {LOG_FILE.name}에 남아 있습니다.")
    print(f"결과 위치: {OUT_DIR}")

    # 결과를 보러 폴더를 찾아 들어가지 않아도 되도록 열어 준다.
    if ok:
        try:
            os.startfile(OUT_DIR)
        except Exception:
            pass

    # 강의 하나가 15분 넘게 걸린다 — 자리를 비웠어도 끝났음을 알 수 있게 소리로 알린다.
    # 짧은 파일까지 울리면 성가시므로 1분을 넘긴 경우만.
    if time.monotonic() - started > 60:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n중단했습니다. 완료된 파일은 저장되어 있고, "
              "진행 중이던 파일은 다음 실행에서 처음부터 다시 합니다.")
        sys.exit(1)
    finally:
        unstage(_STAGED)      # 끌어다 놓기로 만든 이름은 어떤 경우에도 남기지 않는다
        prevent_sleep(False)
