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
from collections import Counter, defaultdict
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
ENGINE_REV = 9

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
HALLUCINATIONS = ("시청해주셔서 감사합니다", "시청해 주셔서 감사합니다", "구독과 좋아요",
                  "thank you for watching", "thanks for watching", "subtitles by")

# 슬라이드 읽기(OCR)
OCR_MIN_CONF = 60        # 이보다 확신이 낮은 줄은 사진 속 잡음으로 버린다
OCR_MIN_ALNUM = 0.55     # 글자 비율이 이보다 낮으면 잡음
SLIDE_CROP = 0.66        # 화자가 곁들여진 화면에서 슬라이드가 차지하는 좌측 비율
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


def split_phrases(segments):
    """세그먼트를 낱말 시각으로 짧은 구절로 나눈다. (시작, 끝, 글, 의심) 목록.

    전문용어 힌트를 넣으면 Whisper 세그먼트가 24초짜리로 뭉개진다(실측: 24개 → 7개).
    구절 단위로 나눠야 슬라이드가 바뀐 시점에서 문단을 끊을 수 있다.
    """
    out = []
    for seg in segments:
        s, e, text, bad = seg[0], seg[1], seg[2], seg[3]
        words = seg[4] if len(seg) > 4 else None
        if not words:
            if text.strip():
                out.append((s, e, text.strip(), bad))
            continue
        cur, cs, ce = "", None, None
        for ws, we, w in words:
            if cs is not None and ws - ce >= PHRASE_GAP_SEC and len(cur.strip()) < 3:
                # 세그먼트 앞머리의 외톨이 낱말('이 표는'의 '이')은 시각이 부정확하다 —
                # 앞 문단에 붙지 않도록 뒤 낱말들과 한 구절로 묶고 그 시각을 쓴다
                cs = ws
            elif cs is not None and (ws - ce >= PHRASE_GAP_SEC or ends_sentence(cur)):
                out.append((cs, ce, cur.strip(), bad))
                cur, cs = "", None
            if cs is None:
                cs = ws
            cur += w
            ce = we
        if cur.strip():
            out.append((cs, ce, cur.strip(), bad))
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
    """
    want = cfg["ocr_언어"].strip()
    if want.lower() not in ("자동", "auto", ""):
        return [want]
    if not (TESSDATA_DIR / "kor.traineddata").exists():
        return ["eng"]
    return ["kor", "eng"] if spoken_language == "ko" else ["eng", "kor"]


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
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
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
        if (conf >= OCR_MIN_CONF
                and len(text) >= 2 and content / len(body) >= OCR_MIN_ALNUM):
            out.append((min(tops[key]) if tops[key] else 0, text, conf))
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
    return (conf - script_mix_penalty(text) * 12 + (8 if han >= 2 else 0)
            + len(text) * 0.2)


def ocr_best(tess: str, img: Path, lang_options):
    """후보 언어로 각각 읽어 본 뒤 **줄 단위로** 잘 읽힌 쪽을 고른다.

    화면 하나를 통째로 한 언어에 맡기면, 한국어 슬라이드에 섞인 영어(출처 표기,
    'All rights reserved', 약어)가 함께 깨진다. 그래서 같은 높이에 있는 줄끼리
    맞대어 놓고 줄마다 더 나은 쪽을 뽑는다. OCR 실행 횟수는 늘지 않는다.
    """
    return merge_ocr_passes([ocr_lines(tess, img, langs) for langs in lang_options])


def merge_ocr_passes(passes, tol: int = 25):
    """여러 번 읽은 결과를 줄 높이로 맞대어, 줄마다 잘 읽힌 쪽을 남긴다."""
    passes = [p for p in passes if p]
    if not passes:
        return []
    if len(passes) == 1:
        return [ln[1] for ln in passes[0]]

    merged, idx = [], [0] * len(passes)
    while True:
        live = [(passes[i][idx[i]][0], i)
                for i in range(len(passes)) if idx[i] < len(passes[i])]
        if not live:
            return merged
        base = min(live)[0]
        group = []
        for top, i in live:
            if abs(top - base) <= tol:
                group.append(passes[i][idx[i]])
                idx[i] += 1
        merged.append(max(group, key=line_score)[1])


def grab_frame(ff: str, src: Path, t: float, vf: str, png: Path) -> bool:
    """t초의 화면 한 장을 뽑는다. 여러 개를 동시에 돌리므로 프로세스마다 한 스레드."""
    subprocess.run([ff, "-nostdin", "-y", "-v", "error", "-threads", "1", "-ss", f"{t:.3f}",
                    "-i", str(src), "-frames:v", "1", "-vf", vf, str(png)],
                   capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
    return png.exists()


def read_screen(ff, tess, src, t, crop, png, langs):
    """t초의 화면을 2배로 키워 읽는다(2배가 원본보다 낫고 3배와는 차이가 없다)."""
    try:
        if not grab_frame(ff, src, t, f"{crop}scale=iw*2:ih*2", png):
            return []
        return ocr_best(tess, png, langs)
    except subprocess.TimeoutExpired:      # 한 장이 느려도 전체를 포기하지는 않는다
        return []
    finally:
        png.unlink(missing_ok=True)


def pick_crop(ff, tess, src: Path, duration: float, tmp: Path, langs, pool, live_time) -> str:
    """화면 전체와 좌측 일부를 견줘 글자가 더 잘 읽히는 쪽을 고른다.

    화자 얼굴이 곁들여진 녹화 강의는 잘라내야 목차·제목까지 읽히고,
    슬라이드만 꽉 찬 영상은 자르면 오른쪽 내용을 잃는다. 그래서 재보고 정한다.

    줄 수만 세면 화자 쪽 무늬(배경 포스터·이름 자막·옷 무늬)가 읽다 만 글자로 잡혀
    '전체 화면'이 이긴다(합성 강의 실측). 그래서 깨진 줄은 감점한다. 오른쪽이 한 번도
    움직이지 않았으면 화자 창이 없는 것이니 재 볼 필요도 없다.
    """
    cols = int(live_time.shape[1] * SLIDE_CROP) if live_time is not None else 0
    if not cols or live_time[:, cols:].mean() < 0.02:
        return ""
    crop = f"crop=iw*{SLIDE_CROP}:ih:0:0,"

    def score(frac, c):
        lines = read_screen(ff, tess, src, duration * frac, c,
                            tmp / f"probe_{frac}_{bool(c)}.png", langs)
        return sum(-0.5 if looks_garbled(ln) else 1 for ln in lines)
    jobs = {pool.submit(score, frac, c): c for frac in (0.25, 0.5, 0.75) for c in ("", crop)}
    total = Counter()
    for fut in as_completed(jobs):
        total[jobs[fut]] += fut.result()
    return crop if total[crop] > total[""] else ""


def select_frames(frames):
    """(초, 축소 회색 화면) 흐름에서 읽어 볼 화면을 고른다.

    직전 초가 아니라 **마지막으로 고른 화면**과 견준다 — 조금씩 늘어나는 판서·항목도
    쌓이면 잡힌다. 계속 움직이는 칸(화자 창, 화면 속 동영상, 커서)은 스스로 알아내
    비교에서 뺀다. 그래서 30초 안전 샘플이 필요 없다.
    돌려주는 값은 (고른 초 목록, 칸마다 움직이던 시간의 비율,
    초마다 움직이던 비율 [(화면 전체, 좌측 슬라이드 쪽)]).
    """
    import numpy as np
    picks, last, last_n, prev, ema, settle_at = [], None, None, None, None, None
    live_sum, shares = None, []
    for n, f in frames:
        f = f.astype(np.int16)
        h, w = f.shape
        gh, gw = h // LIVE_CELL, w // LIVE_CELL
        if prev is None:
            ema, live_sum = np.zeros((gh, gw)), np.zeros((gh, gw))
            picks.append(n)
            shares.append((0.0, 0.0))
            last, last_n, prev = f, n, f
            continue
        moving = (np.abs(f - prev) > PIX_DELTA)[:gh * LIVE_CELL, :gw * LIVE_CELL]
        moving = moving.reshape(gh, LIVE_CELL, gw, LIVE_CELL).mean(axis=(1, 3)) > 0.02
        a = max(LIVE_EMA, 1 / n)       # 처음 몇 초는 누적 평균 — 화자 창을 빨리 알아본다
        ema = ema * (1 - a) + moving * a
        live = ema > LIVE_ON
        live_sum += live
        # 자막 판정용은 지수평균이 아니라 지금 이 순간의 움직임이다 — 지수평균은 영상이
        # 끝난 뒤에도 10초 넘게 남아 바로 다음 슬라이드까지 영상으로 묶었다
        shares.append((float(moving.mean()), float(moving[:, :int(gw * SLIDE_CROP)].mean())))
        still = np.repeat(np.repeat(~live, LIVE_CELL, 0), LIVE_CELL, 1)
        diff = np.abs(f - last)[:gh * LIVE_CELL, :gw * LIVE_CELL]
        ratio = (diff[still] > PIX_DELTA).mean() if still.any() else 0.0
        gap = n - last_n
        pick = False
        if ratio > CHANGE_RATIO and gap >= MIN_GAP_SEC:
            pick, settle_at = True, n + SETTLE_SEC
        elif settle_at is not None and n >= settle_at:
            settle_at = None
            pick = bool(still.any()) and diff[still].mean() > 0.3   # 흐릿했던 것이 선명해졌다
        elif live.mean() > VIDEO_LIVE and gap >= VIDEO_GAP_SEC:
            pick = True
        if pick:
            picks.append(n)
            last, last_n = f, n
        prev = f
    live_time = live_sum / max(len(shares) - 1, 1) if live_sum is not None else None
    return picks, live_time, shares


def live_share(shares, n: int, crop: str, ahead: int = 4) -> float:
    """n초 화면이 떠 있는 동안(직후 몇 초) 화면(잘라 읽으면 슬라이드 쪽)이 움직인 비율.

    n초 자체는 넣지 않는다 — 슬라이드가 넘어가는 순간도 움직임으로 잡히기 때문이다.
    전환 뒤 가만히 있는 슬라이드는 0이고, 재생 중인 영상은 매초 움직인다.
    """
    k = 1 if crop else 0
    window = [s[k] for s in shares[n + 1:n + 1 + ahead]]
    return sum(window) / len(window) if window else 0.0


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
            picks, live_time, shares = select_frames(frames())
        finally:
            p.stdout.close()
            p.wait()
    if not picks and p.returncode != 0:
        detail = err.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        raise RuntimeError("화면을 뽑아내지 못했습니다"
                           + (f" — {detail[-1][:160]}" if detail else ""))
    return picks, live_time, shares


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
        picks, live_time, shares = scan_screen_changes(ff, src, shots)
        status("  화면 영역을 정하는 중...")
        crop = pick_crop(ff, tess, src, duration or 600, shots, langs, pool, live_time)
        jobs = {pool.submit(read_screen, ff, tess, src, n, crop,
                            shots / f"f_{n:06d}.png", langs): n for n in picks}
        found = []
        for i, fut in enumerate(as_completed(jobs), 1):
            status(f"  슬라이드 읽는 중... {i}/{len(jobs)}")
            # n번째 화면은 (n-1, n] 사이에 바뀐 것이다 — 가운데 값을 시각으로 쓴다
            found.append((max(0.0, jobs[fut] - 0.5), fut.result()))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        status("")
        sys.stdout.write("\r")
    found.sort(key=lambda x: x[0])
    live = {max(0.0, n - 0.5): live_share(shares, n, crop) for n in picks}
    motion = [s[1 if crop else 0] for s in shares]
    return merge_slides(found), live, motion


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


def find_slide_pdfs(src: Path):
    """이 강의에 해당하는 강의자료 PDF를 **모두** 찾는다.

    한 주차에 슬라이드 교재와 실습지가 따로 있는 것이 보통이므로 하나만 고르지
    않는다. 주차 번호를 읽을 수 있으면 서로 다를 때 걸러낸다 — 과목 이름만 겹치면
    7주차 자료가 6주차 강의에 붙는 사고가 난다.
    """
    if not PDF_DIR.is_dir():
        return []
    course, week, _period = parse_course(src.stem)
    course_key = re.sub(r"[^0-9a-z가-힣]+", "", course.lower())
    want = name_tokens(src.stem)
    hits = []
    for p in sorted(PDF_DIR.rglob("*.pdf")):
        have = name_tokens(p.stem)
        if not have:
            continue
        flat = re.sub(r"[^0-9a-z가-힣]+", "", p.stem.lower())
        ok = (course_key and course_key in flat) or (
            len(want & have) / min(len(want), len(have)) >= PDF_NAME_RATIO)
        if not ok:
            continue
        pw = stem_week(p.stem)
        if week is not None and pw is not None and pw != week:
            continue
        hits.append(p)
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
    """짝이 맞는 PDF들의 쪽 글자와 쪽 이름을 한 줄로 이어 붙인다.

    돌려주는 값은 (쪽 글자 목록, 쪽 이름 목록, 실제로 쓴 파일 이름, 짝이 맞은 파일 이름).
    마지막 값은 산출물 지문용이다 — 글자를 못 읽은 PDF도 "짝은 맞았다"로 기록해야
    같은 자료로 매번 다시 변환하는 일이 생기지 않는다.
    """
    found = find_slide_pdfs(src)
    if not found:
        return [], [], [], []
    pages, labels, used = [], [], []
    texts = {p: read_pdf_pages(p) for p in found}
    many = sum(1 for p in found if texts[p]) > 1
    for p in found:
        got = texts[p]
        if not got:
            log(f"  ⚠ 강의자료 {p.name}에서 글자를 읽지 못했습니다"
                f" (그림으로 스캔된 PDF일 수 있습니다).")
            continue
        used.append(p.name)
        for i, text in enumerate(got, 1):
            pages.append(text)
            labels.append(f"{p.stem} {i}쪽" if many else f"{i}쪽")
    lines = drop_boilerplate([[ln.rstrip() for ln in t.splitlines() if ln.strip()] for t in pages])
    return ["\n".join(x) for x in lines], labels, used, [p.name for p in found]


def read_pdf_pages(path: Path):
    """PDF의 쪽별 글자. 읽지 못하면 빈 목록(부르는 쪽에서 알린다)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        pages = []
        for page in PdfReader(str(path)).pages:
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
            lines = [ln for ln in lines if not looks_garbled(ln)]
        near = " ".join(p[1] for p in paragraphs if abs(p[0] - t) <= 25)
        # 글자 2연쇄로 견준다 — 움직이는 배경 위 자막은 OCR이 깨져 낱말로는 겹치지 않는다
        sw, nw = page_key(" ".join(lines)), page_key(near)
        ratio = len(sw & nw) / len(sw) if sw else 0.0
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
        # 라고 한 말까지 교수의 말이 아니라고 표시했다.
        return any(a - 0.5 <= t < b + 0.5 for a, b in spans)

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
        lines.append(f"- 🔁 처음 인식에서 빠졌던 말소리 {len(repaired)}곳"
                     f"({sum(b - a for a, b in repaired):.0f}초)을 다시 읽어 채웠습니다 "
                     f"(다른 언어로 말한 대목 등).")
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

    # 강의자료 PDF가 있으면 그 원문을 싣고, 전문용어는 전사 힌트로도 쓴다
    pdf_pages, pdf_labels, pdf_used, pdf_all = load_slide_materials(src)
    hot = pdf_hotwords(pdf_pages) if pdf_pages else ""
    if pdf_pages:
        print(f"  강의자료 {', '.join(pdf_used)} (총 {len(pdf_pages)}쪽)을 함께 씁니다.")

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

    added, repaired, lost = repair_gaps(model, audio, collected, cfg)
    if repaired:
        print(f"  처음 인식에서 빠진 말소리 {len(repaired)}곳을 다시 읽어 채웠습니다.")
    collected = sorted(collected + added, key=lambda s: s[0])
    del audio
    elapsed = time.monotonic() - t0

    phrases = split_phrases(collected)
    if not phrases:
        raise RuntimeError("음성이 감지되지 않았습니다 (오디오 트랙이 없거나 무음일 수 있습니다)")

    slides, live, motion = [], {}, []
    if cfg["슬라이드_읽기"] and tess and media["video"]:
        try:
            langs = ocr_lang_options(cfg, info.language)
            slides, live, motion = extract_slides(src, tmp_dir, tess,
                                                  media["duration"] or info.duration, langs)
            print(f"  슬라이드 {len(slides)}장을 읽었습니다." if slides
                  else "  (화면에서 읽을 만한 글자를 찾지 못했습니다)")
            if slides and pdf_pages:
                slides = align_slides_to_pdf(slides, pdf_pages, pdf_labels)
                matched = sum(1 for s in slides if s[2])
                print(f"  그중 {matched}장을 강의자료 원문으로 바꿨습니다 "
                      f"(잡음 없이 표·빈칸까지 그대로).")
        except Exception as e:
            # print만 하면 창을 닫은 뒤 흔적이 없다. 기록에 남겨 사후 진단이 되게 한다.
            log(f"  ⚠ 슬라이드를 읽지 못했습니다: {out_path.name} — {e}"
                f" (음성 전사는 정상 저장)")

    # 화면에서 읽은 글자에도 되풀이되는 배너가 있으면 뺀다 (PDF 쪽은 이미 뺐다)
    ocr_only = [i for i, s in enumerate(slides) if len(s) < 3 or not s[2]]
    cleaned = drop_boilerplate([slides[i][1] for i in ocr_only])
    for i, body in zip(ocr_only, cleaned):
        slides[i] = (slides[i][0], body) + tuple(slides[i][2:])
    slides = [s for s in slides if s[1]]

    screens = label_slides(slides, [(p[0], p[2]) for p in phrases], live)
    order, ends = screen_ends(screens, info.duration)
    spans = video_spans([screens[i] for i in order], [ends[i] for i in order], motion)
    # 슬라이드가 바뀐 때와 영상이 시작된 때 문단을 끊는다
    paragraphs = group_paragraphs(phrases, [s[0] for s in screens if s[3] == "슬라이드"]
                                  + [a for a, _b in spans])
    vad_lost = 1 - (getattr(info, "duration_after_vad", info.duration) / total)
    write_markdown(out_path, src, info, paragraphs, cfg, screens,
                   ", ".join(pdf_used) if pdf_used else None, ", ".join(pdf_all),
                   vad_lost, repaired, lost, spans)
    speed = info.duration / elapsed if elapsed > 0 else 0
    log(f"  ✔ 완료: {out_path.name} (전사 {fmt_ts(elapsed)}, 실시간 대비 {speed:.1f}배"
        + (f", 복구 {len(repaired)}곳" if repaired else "")
        + (f", 슬라이드 {len(screens)}장" if screens else "") + ")")


def cleanup_stale_temp():
    """지난 실행이 강제 종료되며 남긴 임시 파일을 치운다.

    창을 X로 닫으면 파이썬의 정리 코드가 돌지 못해 WAV·프레임 이미지가 수백 MB씩
    남는다. 강제 종료 자체는 막을 수 없으므로 다음 실행이 청소한다.
    도구 폴더 안(TMP_ROOT)과, 그곳에 쓸 수 없을 때 물러서는 %TEMP% 를 함께 훑는다.
    """
    freed = 0
    cutoff = time.time() - 6 * 3600
    stale = []
    for root in (TMP_ROOT, Path(tempfile.gettempdir())):
        try:
            stale += [d for d in root.glob("transcriber_*") if d.is_dir()]
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
