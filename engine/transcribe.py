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

MEDIA_EXTS = {".mp4", ".m4a", ".mp3", ".wav", ".mkv", ".mov", ".webm",
              ".avi", ".flac", ".ogg", ".aac", ".wma", ".mpeg", ".mpg", ".mts", ".wmv",
              ".m4v", ".ts", ".opus", ".3gp", ".amr", ".mp2"}

# 엔진 로직 판(版). 문단화·OCR·출력 형식을 바꿀 때마다 올린다.
# 이 값이 산출물 지문에 들어가므로, 올리면 기존 MD가 자동으로 다시 만들어진다.
ENGINE_REV = 8

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

# 문단 분리 기준 — 상한에 도달해도 문장이 끝날 때까지 기다린다(문장 중간 절단 방지)
PARA_GAP_SEC = 2.0       # 이 이상 침묵하면 새 문단
PARA_SOFT_SEC = 20.0     # 이 길이를 넘으면 다음 문장 끝에서 문단을 닫는다
PARA_SOFT_CHARS = 300
PARA_HARD_SEC = 45.0     # 문장 끝이 끝내 안 나올 때의 강제 상한
PARA_HARD_CHARS = 650
# '다.' '요.' '까?' 는 '.' '?' 에 이미 포함되므로 두지 않는다
SENTENCE_END = ('.', '!', '?', '"', "'", '”', '…')

# 잘린 입력 판정 — 추출된 오디오가 원본 길이의 이 비율 미만이면 손상으로 본다
TRUNCATION_TOLERANCE = 0.98
MARKER = "<!-- transcriber:"

# 신뢰할 수 없는 전사 구간 판정 — 이 구간은 산출물에 표식을 남긴다
SUSPECT_LOGPROB = -0.9   # 평균 확률이 이보다 낮으면 인식이 흔들린 것
SUSPECT_NO_SPEECH = 0.6  # 말이 아닐 확률이 이보다 높은데 글이 나왔으면 의심
SUSPECT_REPEAT = 3       # 같은 문장이 이 횟수 이상 반복되면 환각

# 슬라이드 읽기(OCR)
SCENE_THRESHOLD = 0.03   # 화면이 이만큼 바뀌면 슬라이드가 넘어간 것으로 본다
SLIDE_SAFETY_SEC = 30    # 변화가 안 잡혀도 최소 이 간격으로는 화면을 확인한다
OCR_MIN_CONF = 60        # 이보다 확신이 낮은 줄은 사진 속 잡음으로 버린다
OCR_MIN_ALNUM = 0.55     # 글자 비율이 이보다 낮으면 잡음
SLIDE_CROP = 0.66        # 화자가 곁들여진 화면에서 슬라이드가 차지하는 좌측 비율
SLIDE_MERGE_RATIO = 0.55  # 낱말이 이만큼 겹치면 같은 슬라이드로 본다
SLIDE_MERGE_LOOKBACK = 3  # 직전 몇 장까지 견주어 볼지 (애니메이션 단계 대응)
SUBTITLE_MATCH_RATIO = 0.6   # 발화와 이만큼 겹치는 짧은 화면 글자는 영상 자막
# 수식·기호는 글자로 센다 — 'f(x) = ax + b' 같은 줄이 잡음으로 버려지는 것을 막는다
OCR_SYMBOLS = set("+-=*/%^<>()[]{}|~.,:;'\"₩$€£°±×÷≤≥≠→←↑↓∙·")


def log(msg: str, echo: bool = True):
    if echo:
        print(msg)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")
    except OSError:
        pass


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
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    cfg = {"model": "large-v3-turbo", "language": "auto", "beam_size": 1,
           "batch_size": 1, "정확도_우선": True, "슬라이드_읽기": True, "ocr_언어": "자동"}
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
    if lang in ("", "auto") or lang in _LANGUAGE_CODES:
        cfg["language"] = lang or "auto"
    else:
        log(f"⚠ 설정.ini의 language '{lang}'은 올바른 언어 코드가 아닙니다. "
            f"auto / ko / en / ja / zh 중에서 골라주세요. 이번에는 auto로 진행합니다.")
        cfg["language"] = "auto"

    for key, lo, hi in (("beam_size", 1, 10), ("batch_size", 1, 32)):
        try:
            v = s.getint(key, cfg[key])
        except ValueError:
            log(f"⚠ 설정.ini의 {key} 값이 숫자가 아니라 기본값({cfg[key]})을 사용합니다.")
            continue
        if not lo <= v <= hi:
            log(f"⚠ 설정.ini의 {key}={v}는 허용 범위({lo}~{hi})를 벗어나 {min(max(v, lo), hi)}로 조정합니다.")
        cfg[key] = min(max(v, lo), hi)

    # 알 수 없는 값을 조용히 무시하지 않는다 — 예전에는 '슬라이드_읽기 = false' 가
    # 경고 없이 '켬'으로 처리되어 사용자가 왜 안 꺼지는지 알 수 없었다.
    def choice(key, default, options):
        raw = s.get(key, default).strip()
        for canon, words in options.items():
            if raw.lower() in words:
                return canon
        log(f"⚠ 설정.ini의 {key} '{raw}'를 알 수 없어 {default}(으)로 진행합니다. "
            f"쓸 수 있는 값: {' / '.join(w for ws in options.values() for w in ws)}")
        return default

    mode = choice("우선순위", "정확도",
                  {"정확도": {"정확도", "accuracy"}, "속도": {"속도", "speed"}})
    cfg["정확도_우선"] = mode == "정확도"
    if not cfg["정확도_우선"] and cfg["batch_size"] == 1:
        cfg["batch_size"] = 16

    cfg["슬라이드_읽기"] = choice("슬라이드_읽기", "켬",
                            {"켬": {"켬", "on", "yes"}, "끔": {"끔", "off", "no"}}) == "켬"

    want = (s.get("슬라이드_언어", "자동").strip() or "자동")
    if want.lower() in ("자동", "auto"):
        cfg["ocr_언어"] = "자동"
    else:
        missing = [c for c in want.split("+")
                   if c and not (TESSDATA_DIR / f"{c}.traineddata").exists()]
        if missing:
            log(f"⚠ 슬라이드_언어 '{want}'의 언어 데이터가 없습니다 "
                f"({', '.join(m + '.traineddata' for m in missing)}). 자동으로 진행합니다.")
            cfg["ocr_언어"] = "자동"
        else:
            cfg["ocr_언어"] = want
    return cfg


def config_fingerprint(cfg):
    mode = "정확도" if cfg["정확도_우선"] else f"속도b{cfg['batch_size']}"
    slide = f"|slide{cfg['ocr_언어']}" if cfg["슬라이드_읽기"] else ""
    # 엔진 판을 함께 넣는다 — 로직을 고치면 기존 산출물이 자동으로 갱신된다
    return (f"rev{ENGINE_REV}|{cfg['model']}|{cfg['language']}"
            f"|beam{cfg['beam_size']}|{mode}{slide}")


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def probe_duration(path: Path):
    """ffprobe로 원본 재생 길이(초)를 얻는다. 알 수 없으면 None."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, stdin=subprocess.DEVNULL)
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None


def probe_audio_duration(path: Path):
    """오디오 트랙 자체의 길이(초). 알 수 없으면 None.

    잘림 검사는 반드시 이 값과 견주어야 한다. 컨테이너 길이는 영상 트랙을 따르므로,
    끝에 무음 화면이 붙은 정상 녹화가 '손상'으로 거부되는 일이 있었다.
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, stdin=subprocess.DEVNULL)
        return float(json.loads(r.stdout)["streams"][0]["duration"])
    except Exception:
        return None


def extract_audio(src: Path, dst: Path, src_duration):
    """16kHz mono WAV로 추출.

    성공하면 True, ffmpeg가 없으면 None(원본 직접 디코딩으로 폴백).
    잘린 입력이나 추출 실패는 예외로 알려 조용한 절단을 막는다.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    timeout = max(600, (src_duration or 0) * 2)
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
    expect = probe_audio_duration(src)
    if expect and got < expect * TRUNCATION_TOLERANCE:
        raise RuntimeError(
            f"오디오가 {fmt_ts(expect)}인데 {fmt_ts(got)}까지만 읽혔습니다. "
            f"파일이 손상되었거나 복사가 끝나지 않았습니다")
    return True


def safe_stem(name: str, limit: int = 120) -> str:
    return name if len(name) <= limit else name[:limit].rstrip() + "~"


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


def has_video(src: Path) -> bool:
    """실제 영상 트랙이 있는지. mp3에 붙은 앨범 표지 한 장은 영상이 아니다."""
    exe = shutil.which("ffprobe")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v",
                            "-show_entries", "stream=codec_type,disposition",
                            "-of", "json", str(src)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", stdin=subprocess.DEVNULL, timeout=120)
        streams = json.loads(r.stdout).get("streams", [])
    except Exception:
        return False
    return any(s.get("codec_type") == "video"
               and not (s.get("disposition") or {}).get("attached_pic")
               for s in streams)


def ocr_lines(tess: str, img: Path, langs: str):
    """이미지에서 글자를 읽되, 확신이 낮은 줄(슬라이드 속 사진 등)은 버린다.

    한 번만 인식해서 두 가지 형식을 함께 받는다. 줄별 확신도는 tsv에서 가져오고,
    글자는 txt에서 가져온다 — 한글은 Tesseract가 음절 단위로 끊어 좌표만으로는
    띄어쓰기를 되살리기 어렵지만, txt 출력에는 이미 제대로 반영되어 있다.
    """
    with tempfile.TemporaryDirectory(prefix="ocr_") as td:
        base = Path(td) / "page"
        r = subprocess.run([tess, str(img), str(base), "-l", langs, "--psm", "6",
                            "tsv", "txt"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", stdin=subprocess.DEVNULL, timeout=180)
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
      · 한 줄에 두 언어가 섞이면(예: '©2023. 두꺼비마을신문. All rights reserved.')
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


def pick_crop(tess: str, src: Path, duration: float, tmp: Path, langs) -> str:
    """화면 전체와 좌측 일부를 견줘 글자가 더 많이 읽히는 쪽을 고른다.

    화자 얼굴이 곁들여진 녹화 강의는 잘라내야 목차·제목까지 읽히고,
    슬라이드만 꽉 찬 영상은 자르면 오른쪽 내용을 잃는다. 그래서 재보고 정한다.
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다")
    full_score = crop_score = 0
    for frac in (0.25, 0.5, 0.75):
        shot = tmp / f"probe_{frac}.png"
        subprocess.run([ff, "-nostdin", "-y", "-ss", str(duration * frac), "-i", str(src),
                        "-frames:v", "1", "-vf", "scale=iw*2:ih*2", str(shot)],
                       capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
        if not shot.exists():
            continue
        cropped = tmp / f"probe_{frac}_c.png"
        subprocess.run([ff, "-nostdin", "-y", "-i", str(shot),
                        "-vf", f"crop=iw*{SLIDE_CROP}:ih:0:0", str(cropped)],
                       capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
        try:        # 한 표본이 느려도 슬라이드 읽기 전체를 포기하지는 않는다
            full_score += len(ocr_best(tess, shot, langs))
            if cropped.exists():
                crop_score += len(ocr_best(tess, cropped, langs))
        except subprocess.TimeoutExpired:
            pass
        shot.unlink(missing_ok=True)
        cropped.unlink(missing_ok=True)
    return f"crop=iw*{SLIDE_CROP}:ih:0:0," if crop_score > full_score else ""


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


def extract_slides(src: Path, tmp: Path, tess: str, duration: float, langs: str):
    """화면이 바뀌는 지점을 찾아 그 슬라이드의 글자를 읽어 온다."""
    shots = tmp / "slides"
    shots.mkdir(exist_ok=True)
    crop = pick_crop(tess, src, duration or 600, tmp, langs)

    # 초당 1장만 검사한다 — 슬라이드는 그보다 빨리 넘어가지 않는다.
    # 화면 변화가 작아 놓치는 경우(흰 배경에 글자만 있는 슬라이드)를 막기 위해
    # 변화가 없어도 SLIDE_SAFETY_SEC 마다 한 번은 확인한다. 같은 화면이면 뒤에서 합쳐진다.
    vf = (f"fps=1,{crop}select='eq(n\\,0)+gt(scene\\,{SCENE_THRESHOLD})"
          f"+not(mod(n\\,{SLIDE_SAFETY_SEC}))',showinfo,scale=iw*2:ih*2")
    sys.stdout.write("\r  화면이 바뀌는 지점을 찾는 중...   ")
    sys.stdout.flush()
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다")
    r = subprocess.run([ff, "-nostdin", "-y", "-i", str(src), "-vf", vf,
                        "-fps_mode", "passthrough", "-an", str(shots / "s_%05d.png")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", stdin=subprocess.DEVNULL,
                       timeout=max(1800, (duration or 0) * 3))
    files = sorted(shots.glob("s_*.png"))
    if r.returncode != 0 and not files:
        detail = (r.stderr or "").strip().splitlines()
        raise RuntimeError("화면을 뽑아내지 못했습니다"
                           + (f" — {detail[-1][:160]}" if detail else ""))
    times = [float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    if not files:
        return []
    # 프레임과 시각이 어긋나면 이후 모든 슬라이드 시각이 밀린다. 조용히 넘기지 않는다.
    if len(times) != len(files):
        raise RuntimeError(f"화면 {len(files)}장과 시각 정보 {len(times)}개가 맞지 않습니다")

    found = []
    for i, (img, t) in enumerate(zip(files, times), 1):
        sys.stdout.write(f"\r  슬라이드 읽는 중... {i}/{len(files)}   ")
        sys.stdout.flush()
        try:
            found.append((t, ocr_best(tess, img, langs)))
        except subprocess.TimeoutExpired:
            pass
        img.unlink(missing_ok=True)
    sys.stdout.write("\r" + " " * 40 + "\r")
    return merge_slides(found)


# ────────────────────────── 강의자료 PDF 연동 ──────────────────────────

def name_tokens(stem: str):
    return {w for w in re.sub(r"[^0-9a-z가-힣]+", " ", stem.lower()).split() if w}


def stem_week(stem: str):
    """파일 이름에서 주차를 읽는다. '6주차교재' → 6, '6협상스타일' → 6."""
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
    return pages, labels, used, [p.name for p in found]


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


def pdf_hotwords(pages):
    """PDF에서 이 강의의 전문용어를 뽑아 전사 힌트로 쓴다.

    '전환사채'를 '전환사체'로, 'VC들을'을 'VCD를'로 잘못 듣던 오류가 줄어든다.
    """
    흔한말 = {"그리고", "하지만", "그러나", "때문", "경우", "위한", "대한", "이상", "이하",
            "있는", "있다", "한다", "등의", "또는", "그래서", "이러한", "우리", "여러분",
            "때문에", "하는", "되는", "것을", "것이", "수가", "지원", "다음"}
    words = re.findall(r"[가-힣]{2,}|[A-Z][A-Za-z]+", " ".join(pages))
    out = []
    for w, _n in Counter(words).most_common():
        if w in 흔한말 or w in out:
            continue
        out.append(w)
        if len(" ".join(out)) >= HOTWORD_MAX:
            break
    return " ".join(out)[:HOTWORD_MAX]


def page_key(text: str):
    """글자 2연쇄 집합.

    낱말 단위로 맞대면 한국어 조사 변화('벤처캐피탈' vs '벤처캐피탈의')와 OCR 잡음에
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

    첫 줄을 그냥 쓰면 대학 배너('H O N G I K U N I V E R S I T Y'), 단독 절 번호
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
    """'창업과실용법률 12-1강' → ('창업과실용법률', 12, 1)."""
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


def read_marker(md: Path, src_name: str):
    """우리가 만든 MD인지 확인하고 기록된 메타데이터를 돌려준다. 아니면 None.

    구버전이 만든 MD에는 마커가 없으므로 헤더의 원본 파일명으로 알아본다.
    이 경우 설정을 알 수 없으니 다시 변환하도록 빈 정보를 돌려준다.
    """
    legacy = None
    try:
        with md.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(MARKER):
                    return json.loads(line[len(MARKER):line.rindex("-->")].strip())
                if line.startswith("- **원본 파일**:") and line.split(":", 1)[1].strip() == src_name:
                    legacy = {"legacy": True}
    except (OSError, ValueError):
        pass
    return legacy


def plan_targets(cfg):
    """무엇을 전사할지 결정한다. 파일시스템을 변형하지 않는다."""
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
    from collections import Counter
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
        out = OUT_DIR / (name + ".md")

        if out.exists():
            mark = read_marker(out, f.name)
            if mark is None:
                blocked.append((f, out))
                continue
            if mark.get("legacy") or "sha" not in mark:
                # 구버전 결과, 또는 기록 도중 중단된 미완성 파일 → 다시 만든다
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


def group_paragraphs(segments):
    """(start, end, text[, 의심]) 목록을 문단으로 묶는다. 문장 중간에서 끊지 않는다.

    돌려주는 값은 (시작시각, 본문, 의심여부) 이다.
    """
    paras, cur, start, end, bad = [], "", None, 0.0, False

    def close():
        paras.append((start, cur.strip(), bad or looks_hallucinated(cur)))

    for seg in segments:
        s_start, s_end, text = seg[0], seg[1], seg[2]
        s_bad = bool(seg[3]) if len(seg) > 3 else False
        text = text.strip()
        if not text:
            continue
        over = (start is not None and
                (s_end - start >= PARA_SOFT_SEC or len(cur) >= PARA_SOFT_CHARS))
        gap = start is not None and s_start - end >= PARA_GAP_SEC
        forced = start is not None and (len(cur) >= PARA_HARD_CHARS
                                        or s_end - start >= PARA_HARD_SEC)
        # 상한을 넘었더라도 앞 문단이 문장으로 끝났을 때만 닫는다 (강제 상한 제외)
        if start is not None and (gap or forced or (over and ends_sentence(cur))):
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


def label_slides(slides, paragraphs):
    """화면 글자를 '슬라이드'와 '영상 자막'으로 가른다.

    강의 중 재생된 영상의 번인 자막이 슬라이드로 실리면, 그 표시가 "화면에 실제로
    있던 신뢰할 만한 원문"이라는 신호를 무의미하게 만든다. 짧은 한두 줄이 바로 옆
    발화와 대부분 겹치면 자막으로 본다.
    """
    labeled = []
    for entry in slides:
        t, lines = entry[0], entry[1]
        near = " ".join(p[1] for p in paragraphs if abs(p[0] - t) <= 25)
        sw, nw = slide_key(lines), slide_key([near])
        ratio = len(sw & nw) / len(sw) if sw else 0.0
        short = len(lines) <= 3 and sum(len(x) for x in lines) <= 90
        labeled.append((t, lines, "자막" if short and ratio >= SUBTITLE_MATCH_RATIO
                        else "슬라이드"))
    return labeled


def video_spans(labeled):
    """자막이 잇따라 나오는 구간 = 화면에서 영상이 재생된 구간."""
    spans, run = [], []
    for t, _lines, kind in labeled:
        if kind == "자막":
            run.append(t)
        else:
            if len(run) >= 2:
                spans.append((run[0], run[-1]))
            run = []
    if len(run) >= 2:
        spans.append((run[0], run[-1]))
    return spans


def write_markdown(out_path: Path, src: Path, info, paragraphs, cfg, elapsed, vad_lost,
                   slides=(), pdf_name=None, pdf_tag=None):
    labeled = label_slides(slides, paragraphs)
    spans = video_spans(labeled)
    # (시각, 줄, 쪽번호, 종류) 로 합친다
    screens = [(e[0], e[1], (e[2] if len(e) > 2 else None), k)
               for e, (_t, _l, k) in zip(slides, labeled)]
    real_slides = sum(1 for _t, _l, _p, k in screens if k == "슬라이드")
    from_pdf = sum(1 for _t, _l, p, k in screens if k == "슬라이드" and p)
    suspect = [p[0] for p in paragraphs if len(p) > 2 and p[2]]

    def in_video(t):
        return any(a - 15 <= t <= b + 15 for a, b in spans)

    course, week, period = parse_course(src.stem)
    mode = "정확도 우선" if cfg["정확도_우선"] else f"속도 우선(batch {cfg['batch_size']})"
    lines = ["---", f"과목: {course}"]
    if week is not None:
        lines += [f"주차: {week}", f"교시: {period}"]
    lines += [f"언어: {info.language}", f"길이: {fmt_ts(info.duration)}",
              f"원본: {src.name}"]
    if pdf_name:
        lines.append(f"강의자료: {pdf_name}")
    lines += ["---", "",
              f"# {src.stem}", "",
              f"- **원본 파일**: {src.name}",
              f"- **길이**: {fmt_ts(info.duration)}",
              f"- **언어**: {info.language}"
              + ("  (설정에서 지정)" if cfg["language"] not in ("", "auto")
                 else f"  (자동 감지, 확신도 {info.language_probability * 100:.0f}%)"),
              f"- **모델**: {cfg['model']} · {mode} · 처리 {fmt_ts(elapsed)}",
              f"- **생성 일시**: {datetime.datetime.now():%Y-%m-%d %H:%M}"]
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
    if vad_lost >= 0.15:
        lines.append(f"- ⚠ **무음으로 제외된 구간이 {vad_lost * 100:.0f}%**입니다. "
                     f"녹음 음량이 낮으면 일부 발화가 빠졌을 수 있습니다.")
    # 화면이 언제까지 떠 있었는지 — "이 발화가 어느 화면에 대한 것인가"를 확정한다
    order = sorted(range(len(screens)), key=lambda i: screens[i][0])
    ends = {}
    for n, i in enumerate(order):
        ends[i] = screens[order[n + 1]][0] if n + 1 < len(order) else info.duration

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

    # 말과 화면 글자를 시간순으로 엮는다 — 어떤 화면을 보며 한 말인지 드러난다
    timeline = ([(screens[i][0], "화면", i) for i in range(len(screens))]
                + [(p[0], "말", (p[1], len(p) > 2 and p[2])) for p in paragraphs])
    for start, kind, value in sorted(timeline, key=lambda x: (x[0], x[1] != "화면")):
        if kind == "화면":
            _t, body, page, what = screens[value]
            head = "📺 영상 자막" if what == "자막" else "🖵 슬라이드"
            span = f"[{fmt_ts(start)} – {fmt_ts(ends[value])}]"
            lines.append(f"> **{head}{f' {page}' if page else ''} {span}**")
            lines += [f"> {ln}" for ln in body]
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
    size_mb = src.stat().st_size / 1024 / 1024
    print(f"\n[{idx}/{total_n}] ▶ {src.name} ({size_mb:.1f} MB)")

    src_dur = probe_duration(src)
    if src_dur and src_dur > 4 * 3600:
        print(f"  ⚠ {fmt_ts(src_dur)}짜리 긴 파일입니다. 메모리를 많이 사용합니다.")

    # 강의자료 PDF가 있으면 그 원문을 싣고, 전문용어는 전사 힌트로도 쓴다
    pdf_pages, pdf_labels, pdf_used, pdf_all = load_slide_materials(src)
    hot = ""
    if pdf_pages:
        hot = pdf_hotwords(pdf_pages)
        print(f"  강의자료 {', '.join(pdf_used)} (총 {len(pdf_pages)}쪽)을 함께 씁니다.")

    wav = tmp_dir / (safe_stem(out_path.stem, 60) + ".wav")
    try:
        print("  오디오 추출 중...", end="", flush=True)
        used_ffmpeg = extract_audio(src, wav, src_dur)
        audio_input = wav if used_ffmpeg else src
        print("\r" + " " * 30 + "\r", end="")
        if used_ffmpeg is None:
            print("  (ffmpeg가 없어 원본을 직접 디코딩합니다)")

        common = dict(
            language=None if cfg["language"] in ("", "auto") else cfg["language"],
            beam_size=cfg["beam_size"],
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        t0 = time.monotonic()
        if cfg["정확도_우선"]:
            # 순차 경로: 온도 폴백 + 반복/저신뢰 감지가 살아 있고 타임스탬프가 정밀하다
            segments, info = model.model.transcribe(
                str(audio_input), condition_on_previous_text=False,
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                no_speech_threshold=0.6, hotwords=hot or None, **common)
        else:
            segments, info = model.transcribe(
                str(audio_input), batch_size=cfg["batch_size"], **common)
        print(f"  언어: {info.language} · 길이: {fmt_ts(info.duration)}")

        collected, total = [], max(info.duration, 0.01)
        for seg in segments:
            # 인식이 흔들린 구간은 표식을 달아 산출물에 남긴다. 매끄러운 문장으로
            # 된 환각은 사람도 LLM도 알아볼 수 없으므로 여기서 잡아야 한다.
            bad = ((getattr(seg, "avg_logprob", None) or 0) < SUSPECT_LOGPROB
                   or (getattr(seg, "no_speech_prob", None) or 0) > SUSPECT_NO_SPEECH)
            collected.append((seg.start, seg.end, seg.text, bad))
            pct = min(seg.end / total * 100, 100)
            done = time.monotonic() - t0
            eta = done / max(seg.end, 1) * max(total - seg.end, 0)
            sys.stdout.write(f"\r  진행률: {pct:5.1f}%  [{fmt_ts(seg.end)}/{fmt_ts(total)}]"
                             f"  남은 시간 약 {fmt_ts(eta)}   ")
            sys.stdout.flush()
        elapsed = time.monotonic() - t0
        sys.stdout.write("\r" + " " * 70 + "\r")
    finally:
        wav.unlink(missing_ok=True)

    paragraphs = group_paragraphs(collected)
    if not paragraphs:
        raise RuntimeError("음성이 감지되지 않았습니다 (오디오 트랙이 없거나 무음일 수 있습니다)")

    slides = []
    if cfg["슬라이드_읽기"] and tess and has_video(src):
        try:
            langs = ocr_lang_options(cfg, info.language)
            slides = extract_slides(src, tmp_dir, tess, src_dur or info.duration, langs)
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

    vad_lost = 1 - (getattr(info, "duration_after_vad", info.duration) / total)
    write_markdown(out_path, src, info, paragraphs, cfg, elapsed, vad_lost, slides,
                   ", ".join(pdf_used) if pdf_used else None,
                   ", ".join(pdf_all))
    speed = info.duration / elapsed if elapsed > 0 else 0
    log(f"  ✔ 완료: {out_path.name} (처리 {fmt_ts(elapsed)}, 실시간 대비 {speed:.1f}배)")


def cleanup_stale_temp():
    """지난 실행이 강제 종료되며 남긴 임시 파일을 치운다.

    창을 X로 닫으면 파이썬의 정리 코드가 돌지 못해 WAV·프레임 이미지가 수백 MB씩
    %TEMP%에 남는다. 강제 종료 자체는 막을 수 없으므로 다음 실행이 청소한다.
    """
    freed = 0
    cutoff = time.time() - 6 * 3600
    root = Path(tempfile.gettempdir())
    try:
        stale = [d for d in root.glob("transcriber_*") if d.is_dir()]
    except OSError:
        stale = []
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


def keep_awake():
    """작업이 끝날 때까지 PC가 잠들지 않게 한다.

    배터리로 쓰면 5분 뒤 절전에 들어가 무인 실행이 중간에 멈춘다.
    화면은 꺼져도 되므로 시스템만 붙잡는다.
    """
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        f = ctypes.windll.kernel32.SetThreadExecutionState
        f.restype = ctypes.c_ulong                      # 반환값을 잘라먹지 않도록 선언
        return f(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    except Exception:
        return False


def release_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


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


def main():
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

    cfg = load_config()
    try:
        targets, skipped, blocked, ignored, rebuilt = plan_targets(cfg)
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
    if ignored:
        print(f"무시된 파일 {len(ignored)}개 (지원하지 않는 형식): "
              f"{', '.join(f.name for f in ignored[:5])}"
              f"{' ...' if len(ignored) > 5 else ''}")
    if not targets:
        print(f"\n변환할 새 파일이 없습니다." if (skipped or blocked or ignored)
              else f"\n\"{IN_DIR.name}\" 폴더에 영상이나 음성 파일을 넣고 다시 실행해주세요.")
        return 0

    mode = "정확도 우선" if cfg["정확도_우선"] else f"속도 우선 (batch {cfg['batch_size']})"
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
    print(f"\n변환 대상 {len(targets)}개 · {cfg['model']} · 언어 {cfg['language']} · {mode} · {slide_note}")
    if rebuilt:
        # 오래 걸리는 작업이 예고 없이 시작되지 않도록 무엇을 왜 다시 만드는지 알린다
        mins = sum(probe_duration(f) or 0 for f in rebuilt) / 60 * 0.45
        print(f"  그중 {len(rebuilt)}개는 이미 만든 것을 다시 만듭니다 "
              f"(설정·강의자료·엔진이 바뀌었습니다). 이 몫만 대략 {mins:.0f}분입니다.")
        print("  기다릴 상황이 아니면 Ctrl+C 로 멈추세요. 기존 결과는 그대로 남습니다.")
    log(f"── 실행 시작 · 대상 {len(targets)}개 · 건너뜀 {len(skipped)}개 "
        f"· {config_fingerprint(cfg)}", echo=False)
    if keep_awake():
        print("작업이 끝날 때까지 PC가 절전으로 들어가지 않게 해두었습니다.")

    from faster_whisper import WhisperModel, BatchedInferencePipeline
    cached = any(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else False
    print("모델 로딩 중..." if cached else
          "음성 인식 모델을 내려받는 중입니다. 약 1.6GB이며 처음 한 번만 받습니다.\n"
          "  진행 표시가 없지만 정상 동작 중입니다. 회선에 따라 10~40분 걸릴 수 있습니다.")
    # cpu_threads=10 — 물리 6코어/논리 12에서 실측한 값이다. 생산과 동일한 인자
    # (정확도 경로 + vad_filter)로 240초 강의를 교차 6라운드 잰 결과 8스레드
    # 56.72초 → 10스레드 52.15초(+8.77%), 6/6 전승에 범위 겹침 없음(σ 0.16/0.04).
    # 4·6은 뚜렷이 느리고(87.6/71.3초) 12는 다시 나빠진다 — 논리코어를 다 쓰면
    # 하이퍼스레드 경합으로 손해다. 이 값을 근거 없이 바꾸지 말 것.
    try:
        try:
            base_model = WhisperModel(cfg["model"], device="cpu", compute_type="int8",
                                      cpu_threads=10, download_root=str(MODELS_DIR),
                                      local_files_only=True)
        except Exception:
            base_model = WhisperModel(cfg["model"], device="cpu", compute_type="int8",
                                      cpu_threads=10, download_root=str(MODELS_DIR))
    except Exception as e:
        print(f"\n✘ 모델을 불러오지 못했습니다: {e}")
        print("  - 최초 실행이라면 인터넷 연결과 저장 공간(약 2GB)을 확인해주세요.")
        print(f"  - 설정.ini의 model 값(현재: {cfg['model']})이 올바른지 확인해주세요.")
        return 1
    model = BatchedInferencePipeline(model=base_model)

    ok, failures = 0, []
    try:
        with tempfile.TemporaryDirectory(prefix="transcriber_",
                                         ignore_cleanup_errors=True) as td:
            for i, (src, out_path) in enumerate(targets, 1):
                try:
                    transcribe_file(model, cfg, src, out_path, Path(td), i, len(targets), tess)
                    ok += 1
                except KeyboardInterrupt:
                    log(f"  ⚠ 사용자가 중단했습니다 (진행 중이던 파일: {src.name})",
                        echo=False)
                    raise
                except Exception as e:
                    msg = str(e) or type(e).__name__
                    if "Invalid data" in msg or "Errno" in msg:
                        msg = "오디오를 읽을 수 없습니다 (형식이 잘못되었거나 파일이 손상됨)"
                    elif isinstance(e, MemoryError):
                        msg = "메모리가 부족합니다. 영상을 나눠서 변환해주세요"
                    failures.append((src.name, msg))
                    log(f"  ✘ 실패: {src.name} ({src.stat().st_size / 1048576:.0f}MB) — {msg}")
    finally:
        release_awake()

    log(f"\n전체 완료: 성공 {ok}개 / 실패 {len(failures)}개", echo=True)
    if failures:
        print("실패 목록:")
        for name, err in failures:
            print(f"  ✘ {name}\n      {err}")
        print(f"\n기록은 {LOG_FILE.name}에 남아 있습니다.")
    print(f"결과 위치: {OUT_DIR}")
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n중단했습니다. 완료된 파일은 저장되어 있고, "
              "진행 중이던 파일은 다음 실행에서 처음부터 다시 합니다.")
        sys.exit(1)
