# -*- coding: utf-8 -*-
"""종단 검증 — 정답을 아는 합성 강의를 만들어 엔진을 끝까지 돌리고 산출물을 채점한다.

    engine\\venv\\Scripts\\python engine\\종단_검증.py [--keep]

--keep 을 주면 만든 영상과 산출물을 지우지 않고 남긴다(어긋난 곳을 직접 볼 때).
단위 검증(문단화_검증.py)은 함수 하나하나를 보지만, 이 도구의 결함은 대개 단계
사이에서 생겼다 — 전역 환경변수 하나가 음성 인식을 망가뜨리거나, 검출한 화면과
읽은 화면이 한 장씩 어긋나거나. 그런 것은 끝까지 돌려야만 보인다.

만드는 것 (전부 가상 자료, 약 2분 40초)
  · 슬라이드 7장 — 글자만 있는 흰 슬라이드, 굵은 한글, 영어 슬라이드, 표, 항목이
    하나씩 나타나는 슬라이드, 쪽마다 되풀이되는 학교 배너
  · 한국어 강의 속 20초짜리 영어 설명 (파일 전체를 한 언어로 읽으면 사라지던 대목)
  · 강의 중 재생된 영상 — 움직이는 화면 위 번인 자막과 그 말소리
  · 우측 34%에 계속 움직이는 화자 영역, 그리고 같은 슬라이드의 강의노트 PDF

필요한 것: Windows(Edge·음성 합성 Heami/Zira), ffmpeg, Tesseract, 전사 모델.
실제 입출력 폴더는 건드리지 않는다 — 도구 폴더의 .tmp 안에서 만들고 지운다.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import wave
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import transcribe as T

EDGE = next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                         r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")
             if Path(p).exists()), None)
FONT = "C\\:/Windows/Fonts/malgun.ttf"
BANNER = "<div class=band>S A M P L E &nbsp; U N I V E R S I T Y</div>"
CSS = """
@page { size: 1280px 720px; margin: 0 }
html,body { margin:0; padding:0 }
.s { width:1280px; height:720px; box-sizing:border-box; padding:48px 64px; position:relative;
     font-family:'Malgun Gothic'; background:#fff; color:#111; page-break-after:always; overflow:hidden }
.band { position:absolute; left:0; top:0; right:0; height:36px; background:#1d3b72; color:#fff;
        font-size:18px; letter-spacing:8px; padding:6px 64px }
h1 { font-size:46px; margin:40px 0 28px; color:#1d3b72 }
li { font-size:30px; margin:12px 0 }
b { color:#b00020 }
table { border-collapse:collapse; font-size:28px }
td,th { border:2px solid #333; padding:8px 20px }
"""

# (id, 본문, 발화 언어, 발화, 이 슬라이드 설명에만 나오는 낱말) — 전부 가상 자료
SLIDES = [
    ("s1", "<h1 style='margin-top:160px;font-size:60px'>표본통계학</h1>"
           "<li style='list-style:none;font-size:36px'>7주차 2교시 · 추정과 신뢰구간</li>",
     "ko", "안녕하세요. 표본통계학 칠 주차 두 번째 시간입니다. 오늘은 추정과 신뢰구간을 다룹니다.",
     "안녕하세요"),
    ("s2", "<h1>모집단과 표본의 관계</h1><ul><li><b>모집단</b>은 관심 대상 전체이다</li>"
           "<li><b>표본</b>은 모집단에서 뽑은 일부이다</li><li>추정에는 항상 <b>표본오차</b>가 따른다</li></ul>",
     "ko", "모집단은 우리가 알고 싶은 대상 전체이고 표본은 그중 실제로 관측한 일부입니다. "
           "표본에서 계산한 값으로 모수를 추정하는데 이때 표본오차가 반드시 생깁니다.",
     "관측한"),
    ("s3", "<h1>Confidence Interval</h1><ul><li>Point estimate plus or minus a margin of error</li>"
           "<li>A 95% interval uses z = 1.96</li><li>Larger samples give narrower intervals</li></ul>",
     "en", "Let me switch to English for this definition. A confidence interval is a point estimate "
           "plus or minus a margin of error. Larger samples give narrower intervals.",
     "margin of error"),
    ("s4", "<h1>신뢰수준별 임계값</h1><table><tr><th>신뢰수준</th><th>임계값 z</th></tr>"
           "<tr><td>90%</td><td>1.645</td></tr><tr><td>95%</td><td>1.960</td></tr>"
           "<tr><td>99%</td><td>2.576</td></tr></table>",
     "ko", "이 표는 신뢰수준별 임계값입니다. 신뢰수준을 높일수록 구간이 넓어집니다.",
     "넓어집니다"),
    ("s5a", "<h1>표본 크기의 결정</h1><ul><li>원하는 오차 한계를 먼저 정한다</li></ul>",
     "ko", "그러면 표본을 몇 개나 뽑아야 할까요. 먼저 허용할 오차 한계를 정합니다.",
     "뽑아야"),
    ("s5", "<h1>표본 크기의 결정</h1><ul><li>원하는 오차 한계를 먼저 정한다</li>"
           "<li>n = (z × σ / E)² 로 계산하고 올림한다</li></ul>",
     "ko", "공식에 넣어 계산하고 소수점이 나오면 반드시 올림합니다.",
     "올림합니다"),
    ("video", None, "ko", None, None),
    ("s6", "<h1>편향과 표본 크기</h1><ul><li>표본이 커져도 <b>편향</b>은 줄어들지 않는다</li>"
           "<li>무작위 추출이 편향을 막는 핵심이다</li></ul>",
     "ko", "영상에서 보셨듯이 표본을 크게 해도 편향은 줄어들지 않습니다. 무작위 추출이 핵심입니다.",
     "보셨듯이"),
    ("s7", "<h1>다음 시간 예고</h1><ul><li>가설검정의 논리</li><li>제1종 오류와 제2종 오류</li></ul>",
     "ko", "다음 시간에는 가설검정의 논리와 두 종류의 오류를 배웁니다. 수고하셨습니다.",
     "수고하셨습니다"),
]
SUBS = ["안녕하세요 저는 가상연구소의 김가상입니다",
        "현장에서는 응답률이 가장 큰 문제입니다",
        "그래서 저희는 추출틀을 매년 새로 만듭니다"]
SUB_SEC = 6.0


def run(args, **kw):
    r = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL, timeout=600, **kw)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"{Path(args[0]).name} 실패: {err[-1] if err else r.returncode}")


def ff(*args, cwd=None):
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], cwd=cwd)


def tts(text, lang, wav, work):
    voice = "Microsoft Heami Desktop" if lang == "ko" else "Microsoft Zira Desktop"
    js = work / "tts.json"
    js.write_text(json.dumps({"t": text, "v": voice, "o": str(wav)}, ensure_ascii=False),
                  encoding="utf-8")
    # 한글을 명령줄로 넘기면 코드 페이지에서 깨진다 — 파일로 넘긴다
    run(["powershell", "-NoProfile", "-Command",
         f"$j = Get-Content -Raw -Encoding UTF8 '{js}' | ConvertFrom-Json; "
         "Add-Type -AssemblyName System.Speech; "
         "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
         "$s.SelectVoice($j.v); $s.SetOutputToWaveFile($j.o); $s.Speak($j.t); $s.Dispose()"])
    fixed = wav.with_suffix(".n.wav")
    ff("-i", str(wav), "-ar", "22050", "-ac", "1", str(fixed))
    fixed.replace(wav)


def wav_len(p):
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def silence(path, sec):
    ff("-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", f"{sec:.3f}", str(path))


def edge(work, out, *args):
    """Edge 로 그리고 결과 파일이 다 써질 때까지 기다린다.

    Edge 는 명령이 끝난 **뒤에** 파일을 쓴다(실측: 즉시 없음, 3초 뒤 있음).
    """
    run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
         f"--user-data-dir={work / f'edge_{out.stem}'}", *args])
    size, deadline = -1, time.monotonic() + 30
    while time.monotonic() < deadline:
        if out.exists() and out.stat().st_size == size > 0:
            return
        size = out.stat().st_size if out.exists() else -1
        time.sleep(0.5)
    raise RuntimeError(f"Edge 가 {out.name} 을 만들지 못했습니다")


def build(work):
    """합성 강의 영상과 강의노트 PDF를 만들고 정답을 돌려준다."""
    truth, v_parts, a_parts, t = [], [], [], 0.0
    pages = []
    for sid, body, lang, speech, key in SLIDES:
        if sid == "video":
            for i, s in enumerate(SUBS):
                tts(s, "ko", work / f"sub{i}.wav", work)
                silence(work / f"subpad{i}.wav", max(0.2, SUB_SEC - wav_len(work / f"sub{i}.wav")))
                a_parts += [f"sub{i}.wav", f"subpad{i}.wav"]
            dur = SUB_SEC * len(SUBS)
            v_parts.append(("video", dur))
            truth.append({"id": "video", "start": t, "end": t + dur})
            t += dur
            continue
        html = work / f"{sid}.html"
        html.write_text(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
                        f"<div class=s>{BANNER}{body}</div>", encoding="utf-8")
        edge(work, work / f"{sid}.png", f"--screenshot={work / (sid + '.png')}", "--window-size=1280,720",
             html.as_uri())
        if sid != "s5a":                        # 강의노트에는 완성본만 있다
            pages.append(body)
        tts(speech, lang, work / f"{sid}.wav", work)
        silence(work / f"{sid}_pre.wav", 1.5)
        silence(work / f"{sid}_post.wav", 2.0)
        a_parts += [f"{sid}_pre.wav", f"{sid}.wav", f"{sid}_post.wav"]
        dur = 3.5 + wav_len(work / f"{sid}.wav")
        v_parts.append((sid, dur))
        truth.append({"id": sid, "start": t, "end": t + dur, "lang": lang, "key": key,
                      "page": len(pages) if sid != "s5a" else len(pages) + 1})
        t += dur

    deck = work / "deck.html"
    deck.write_text(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
                    + "".join(f"<div class=s>{BANNER}{b}</div>" for b in pages), encoding="utf-8")
    pdf = work / "표본통계학 7주차 강의노트.pdf"
    edge(work, pdf, f"--print-to-pdf={pdf}", "--no-pdf-header-footer", deck.as_uri())

    # 슬라이드 조각들을 이어 붙인다 — 재생 영상은 움직이는 무늬 위 번인 자막
    segs = []
    for i, (sid, dur) in enumerate(v_parts):
        seg = work / f"v{i}.mp4"
        if sid == "video":
            dt = ",".join(
                f"drawtext=fontfile='{FONT}':text='{s}':fontsize=34:fontcolor=white:box=1:"
                f"boxcolor=black@0.6:x=(w-tw)/2:y=h-90:enable='between(t,{k * SUB_SEC},{k * SUB_SEC + SUB_SEC - 0.1})'"
                for k, s in enumerate(SUBS))
            ff("-f", "lavfi", "-i", f"mandelbrot=s=1268x714:r=30", "-t", f"{dur:.3f}",
               "-vf", dt, "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", str(seg))
        else:
            ff("-loop", "1", "-framerate", "30", "-i", f"{sid}.png", "-t", f"{dur:.3f}",
               "-vf", "scale=1268:714,setsar=1", "-pix_fmt", "yuv420p", "-c:v", "libx264",
               "-preset", "veryfast", str(seg), cwd=work)
        segs.append(f"file '{seg.name}'")
    (work / "v.txt").write_text("\n".join(segs) + "\n", encoding="utf-8")
    (work / "a.txt").write_text("\n".join(f"file '{a}'" for a in a_parts) + "\n", encoding="utf-8")
    mp4 = work / "표본통계학 7-2강.mp4"
    fc = ("[0:v]scale=1267:713,setsar=1[sl];color=c=0x202020:s=1920x1080:r=30[bg];"
          "life=s=160x270:mold=10:r=30:ratio=0.1:death_color=#304050:life_color=#c09070,"
          "scale=653:1080:flags=neighbor[cam];"
          "[bg][sl]overlay=0:183:shortest=1[x];[x][cam]overlay=1267:0:shortest=1,format=yuv420p[v]")
    ff("-f", "concat", "-safe", "0", "-i", "v.txt", "-f", "concat", "-safe", "0", "-i", "a.txt",
       "-filter_complex", fc, "-map", "[v]", "-map", "1:a", "-t", f"{t:.2f}",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(mp4), cwd=work)
    return mp4, pdf, truth


def transcribe(work, mp4, pdf):
    """실제 입출력 폴더 대신 작업 폴더를 가리키게 하고 main()을 그대로 돌린다."""
    for d in ("in", "out", "pdf"):
        (work / d).mkdir()
    shutil.copy2(mp4, work / "in" / mp4.name)
    shutil.copy2(pdf, work / "pdf" / pdf.name)
    T.IN_DIR, T.OUT_DIR, T.PDF_DIR = work / "in", work / "out", work / "pdf"
    T.LOG_FILE, T.CONFIG_FILE = work / "log.txt", work / "설정.ini"
    T.ENGINE_DIR = work        # 중복 실행 잠금도 따로 — 실제 전사가 돌고 있어도 검증은 돈다
    sys.argv = sys.argv[:1]
    os.startfile = lambda *a, **k: None               # 결과 폴더를 열지 않는다
    sys.modules["winsound"] = types.SimpleNamespace(MessageBeep=lambda *a: None,
                                                    MB_ICONASTERISK=0)
    t0 = time.monotonic()
    try:
        rc = T.main()
    finally:
        T.prevent_sleep(False)
    return rc, time.monotonic() - t0, next((work / "out").glob("*.md"), None)


def grade(md: Path, truth):
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(name)

    text = md.read_text(encoding="utf-8")
    lines = text.splitlines()
    ts = lambda s: sum(int(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
    blocks = [(i, m.group(1), m.group(2), ts(m.group(3)), ts(m.group(4)))
              for i, ln in enumerate(lines)
              if (m := re.match(r"> \*\*(🖵 슬라이드|📺 영상 자막)\s*(\d+쪽)?\s*"
                                r"\[(\d\d:\d\d:\d\d) – (\d\d:\d\d:\d\d)\]", ln))]
    slides = [b for b in blocks if b[1] == "🖵 슬라이드"]
    pages = [int(b[2][:-1]) for b in slides if b[2]]
    want = sorted({s["page"] for s in truth if s["id"] != "video"})

    check("모든 쪽이 한 번씩, 순서대로", pages == want, f"{pages} (정답 {want})")
    for page in want:
        # 항목이 하나씩 나타나는 슬라이드는 첫 단계가 뜬 때가 그 쪽의 시작이다
        start = min(s["start"] for s in truth if s.get("page") == page)
        hit = [b for b in slides if b[2] == f"{page}쪽"]
        err = hit[0][3] - start if hit else None
        check(f"{page}쪽 전환 시각 ±1.5초", err is not None and abs(err) <= 1.5,
              f"{'없음' if err is None else f'{err:+.1f}초'}")
    check("PDF 쪽마다 되풀이되는 배너를 뺌", "S A M P L E" not in text and "SAMPLE" not in text)

    video = next(s for s in truth if s["id"] == "video")
    subs = [b for b in blocks if b[1] == "📺 영상 자막"]
    check("재생 영상이 한 블록으로", len(subs) == 1, f"{len(subs)}개")
    if subs:
        check("재생 영상 구간 ±3초", abs(subs[0][3] - video["start"]) <= 3
              and abs(subs[0][4] - video["end"]) <= 3,
              f"{subs[0][3]}–{subs[0][4]} (정답 {video['start']:.0f}–{video['end']:.0f})")
    paras = [(ts(m.group(1)), m.group(2)) for ln in lines
             if (m := re.match(r"\*\*\[(\d\d:\d\d:\d\d)\]\*\* (.*)", ln))]
    in_vid = [p for p in paras if video["start"] - 1 <= p[0] < video["end"]]
    out_vid = [p for p in paras if not (video["start"] - 16 <= p[0] <= video["end"] + 16)]
    check("영상 속 발화에 📺", in_vid and all(p[1].startswith("📺") for p in in_vid),
          f"{sum(p[1].startswith('📺') for p in in_vid)}/{len(in_vid)}")
    check("교수 발화에는 📺 없음", not any("📺" in p[1] for p in out_vid))
    check("제목 슬라이드를 영상 자막으로 오판하지 않음",
          not any(b[1] == "📺 영상 자막" and b[3] < 5 for b in blocks))

    body = " ".join(p[1] for p in paras).lower()
    check("한국어 강의 속 영어 설명이 살아 있음", "confidence interval" in body
          and "margin of error" in body)
    # 설명이 제 슬라이드 밑에 있어야 한다 — 다음 슬라이드 설명이 앞 슬라이드에 붙지 않게
    order = sorted(blocks, key=lambda b: b[0])
    for s in truth:
        if not s.get("key"):
            continue
        at = next((i for i, ln in enumerate(lines)
                   if ln.startswith("**[") and s["key"].lower() in ln.lower()), None)
        owner = max((b for b in order if at is not None and b[0] < at), key=lambda b: b[0],
                    default=None)
        page = f"{s['page']}쪽"
        check(f"{s['id']} 설명이 제 슬라이드 밑에", owner is not None and owner[2] == page,
              f"{owner[2] if owner else '없음'} (정답 {page})")
    return fails


def main():
    keep = "--keep" in sys.argv          # 엔진을 돌리며 sys.argv 를 비우므로 먼저 읽어 둔다
    missing = [n for n, ok in (("Edge", EDGE), ("ffmpeg", shutil.which("ffmpeg")),
                               ("Tesseract", T.find_tesseract())) if not ok]
    if missing:
        print(f"필요한 프로그램이 없습니다: {', '.join(missing)}")
        return 2
    # tempfile 의 임시 폴더는 쓰지 않는다 — Python 3.13부터 '현재 사용자·관리자 전용'
    # 권한이 붙어서, 관리자 권한에서 띄운 Edge 가 일반 권한으로 스스로를 재실행하면
    # 그 폴더에 쓰지 못하고 멈춘다(실측).
    work = (T.tmp_root() or Path(tempfile.gettempdir())) / f"e2e_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        print("합성 강의를 만드는 중... (약 30초)")
        t0 = time.monotonic()
        mp4, pdf, truth = build(work)
        print(f"  {mp4.name}: {truth[-1]['end']:.0f}초, 슬라이드 {len(truth) - 1}장 + 재생 영상 "
              f"({time.monotonic() - t0:.0f}초)")
        rc, took, md = transcribe(work, mp4, pdf)
        print(f"\n채점 (엔진 {took:.0f}초, 실시간 대비 {truth[-1]['end'] / took:.1f}배)")
        if rc != 0 or md is None:
            print(f"  FAIL  엔진이 실패했습니다 (종료 코드 {rc})")
            return 1
        fails = grade(md, truth)
        print(f"\n결과: {'전부 통과' if not fails else '실패 ' + ', '.join(fails)}")
        return 1 if fails else 0
    finally:
        if keep:
            print(f"\n작업 폴더를 남겨 둡니다: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
