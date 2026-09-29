# -*- coding: utf-8 -*-
"""
Transcriber 엔진
"MP4 입력" 폴더의 영상/음성 파일을 전사하여 "MD 출력" 폴더에 Markdown으로 저장한다.

로컬 faster-whisper 기반이므로 파일 용량 제한(25MB 등)이 없다.

이 파일은 흐름(설정·입력 고르기·전사·화면 읽기 실행·강의자료 찾기·출력 조립)과, 검증
스크립트가 바꿔 끼우는 경로 설정(IN_DIR·OUT_DIR·LOG_FILE 등)을 읽는 코드를 맡는다 — 그래야
그 교체가 계속 먹힌다. 단계별 판정은 speech_repair·screen_scan·slide_pdf·timeline·md_writer 에
있고, 이름은 여기서 모두 재수출하므로 transcribe.X 로 부를 수 있다.
"""

import configparser
import csv
import datetime
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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 단계별 하위 모듈 — 이름은 전부 여기서 재수출한다(검증 스크립트·기존 호출 호환).
from speech_repair import (  # noqa: F401
    GAP_JOIN_SEC,
    GAP_MIN_SEC,
    GAP_PAD_SEC,
    HALLUCINATIONS,
    LOST_MIN_SEC,
    PHRASE_GAP_SEC,
    QUALITY,
    RELANG_MIN_PROB,
    REPAIR_MAX,
    REPAIR_MIN_CPS,
    SENTENCE_END,
    SUSPECT_LOGPROB,
    SUSPECT_NO_SPEECH,
    VAD_PARAMS,
    ends_sentence,
    find_gaps,
    fmt_ts,
    is_suspect,
    reliable_words,
    repair_gaps,
    reread_suspects,
    reread_video_suspects,
    segment_words,
    split_phrases,
    status,
    suspect_spans,
)
from screen_scan import (  # noqa: F401
    CHANGE_RATIO,
    EDGE_TOUCH,
    LIVE_CELL,
    LIVE_EMA,
    LIVE_ON,
    MIN_GAP_SEC,
    MIX_PENALTY,
    OCR_MIN_CONF,
    PIX_DELTA,
    SCAN_H,
    SCAN_W,
    SETTLE_SEC,
    SLIDE_CROP,
    SLIDE_MERGE_LOOKBACK,
    SLIDE_MERGE_RATIO,
    SPEAKER_CELL,
    VIDEO_GAP_SEC,
    VIDEO_LIVE,
    appeared_at,
    background_gray,
    bare_line,
    clean_moving_line,
    extend_cut_lines,
    grab_frame,
    line_score,
    live_share,
    looks_garbled,
    merge_ocr_passes,
    merge_slides,
    ocr_junk,
    ocr_score,
    same_row,
    scan_screen_changes,
    script_mix_penalty,
    select_frames,
    slide_key,
    speaker_mask,
    speaker_on_right,
    still_samples,
)
from slide_pdf import (  # noqa: F401
    BOILERPLATE_MIN,
    BOILERPLATE_RATIO,
    COMMON_WORDS,
    HOTWORD_MAX,
    JOSA,
    PDF_FORWARD_BONUS,
    PDF_FORWARD_SPAN,
    PDF_MATCH_RATIO,
    PDF_MIN_WORDS,
    PDF_SHORT_KEY,
    PDF_SHORT_RATIO,
    align_slides_to_pdf,
    drop_boilerplate,
    Material,
    material_pages,
    name_tokens,
    page_key,
    pdf_hotwords,
    pick_materials,
    read_pdf_pages,
    stem_week,
    zip_member_name,
)
from timeline import (  # noqa: F401
    PARA_GAP_SEC,
    PARA_HARD_CHARS,
    PARA_HARD_SEC,
    PARA_SOFT_CHARS,
    PARA_SOFT_SEC,
    SUBTITLE_LIVE,
    SUBTITLE_MATCH_RATIO,
    SUSPECT_REPEAT,
    VIDEO_JOIN_SEC,
    VIDEO_URL,
    add_language_spans,
    foreign_spans,
    group_paragraphs,
    join_video_spans,
    label_slides,
    looks_hallucinated,
    mark_video,
    motion_video_spans,
    other_script,
    screen_ends,
    trim_still_tail,
    video_spans,
)
from md_writer import (  # noqa: F401
    ENGINE_REV,
    MARKER,
    config_fingerprint,
    md_body_hash,
    parse_course,
    read_marker,
    slide_title,
    write_markdown,
)

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

# 모델 내려받기(huggingface_hub·hf_xet)는 모델 파일 말고도 청크 캐시·로그를 둘 곳을
# HF_HOME(기본값은 사용자 홈의 .cache/huggingface)에서 정한다. 모델 자체는 download_root
# 로 engine/models 에 받지만 이 부가 캐시는 그 인자를 따르지 않는다. 실제로 무엇을
# 쓰는지는 확인하지 못했다(점검 환경에서 HuggingFace 에 닿을 수 없었다) — 그래서
# 예방으로 도구 폴더 안을 가리킨다. huggingface_hub 이 import 되기 전이어야 한다.
os.environ["HF_HOME"] = str(TMP_ROOT / "huggingface")


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
PDF_NAME_RATIO = 0.50    # 파일 이름이 이만큼 겹치면 같은 강의의 자료로 본다

# 잘린 입력 판정 — 추출된 오디오가 원본 길이의 이 비율 미만이면 손상으로 본다
TRUNCATION_TOLERANCE = 0.98
OCR_MIN_ALNUM = 0.55     # 글자 비율이 이보다 낮으면 잡음
OCR_BARE_CONF = 75       # 한글 낱말도 영어 낱말도 없는 줄(숫자·기호뿐)은 이만큼 확신할 때만 싣는다 —
# 수식·기호는 글자로 센다 — 'f(x) = ax + b' 같은 줄이 잡음으로 버려지는 것을 막는다
OCR_SYMBOLS = set("+-=*/%^<>()[]{}|~.,:;'\"₩$€£°±×÷≤≥≠→←↑↓∙·")
# Tesseract 는 이 PC에서 사실상 단일 스레드다. 코어 수만큼 동시에 돌리면 5.8배 빠르고
# 출력은 바이트 단위로 같다(실측: 1·3·4·6·8·12개 모두 동일).
OCR_WORKERS = os.cpu_count() or 4


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


def ocr_best(tess: str, img: Path, lang_options):
    """후보 언어로 각각 읽어 본 뒤 **줄 단위로** 잘 읽힌 쪽을 고른다.

    화면 하나를 통째로 한 언어에 맡기면, 한국어 슬라이드에 섞인 영어(출처 표기,
    'All rights reserved', 약어)가 함께 깨진다. 그래서 같은 높이에 있는 줄끼리
    맞대어 놓고 줄마다 더 나은 쪽을 뽑는다. 돌려주는 값은 고른 줄(ocr_lines 형식)이다.
    """
    return merge_ocr_passes([ocr_lines(tess, img, langs) for langs in lang_options], full=True)


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
