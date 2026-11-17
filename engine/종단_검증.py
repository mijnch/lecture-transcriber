# -*- coding: utf-8 -*-
"""종단 검증 — 정답을 아는 합성 강의를 만들어 엔진을 끝까지 돌리고 산출물을 채점한다.

    engine\\venv\\Scripts\\python engine\\종단_검증.py [--keep]

--keep 을 주면 만든 영상과 산출물을 지우지 않고 남긴다(어긋난 곳을 직접 볼 때).
단위 검증(문단화_검증.py)은 함수 하나하나를 보지만, 이 도구의 결함은 대개 단계
사이에서 생겼다 — 전역 환경변수 하나가 음성 인식을 망가뜨리거나, 검출한 화면과
읽은 화면이 한 장씩 어긋나거나. 그런 것은 끝까지 돌려야만 보인다.

만드는 것 (전부 가상 자료)
  강의 A 「표본통계학 7-2강」 (약 3분 40초) — 슬라이드 오른쪽에 화자 창이 따로 있다
    · 슬라이드 9장 — 글자만 있는 흰 슬라이드, 굵은 한글, 영어 슬라이드, 표, 항목이
      하나씩 나타나는 슬라이드, 쪽마다 되풀이되는 학교 배너, 영상 출처 주소
    · 한국어 강의 속 20초짜리 영어 설명 (파일 전체를 한 언어로 읽으면 사라지던 대목)
    · 재생 영상 셋 — 번인 자막이 있는 영상(끝나며 다음 슬라이드로 서서히 바뀐다),
      자막 없이 교수와 같은 언어로 말하는 인터뷰(출처 주소만 있다), 자막 없는 영어 영상
    · 강의노트 PDF 는 이름에 과목이 없는 LMS 묶음 '7주차 강의자료.zip' 안에 있다
  강의 B 「품질관리 7-1강」 (약 1분 20초) — 화자가 슬라이드 위에 겹쳐 서 있다
    · 화자 위쪽을 지나 오른쪽까지 뻗는 긴 줄, 어두운 띠 위의 흰 제목, 한 줄에 섞인
      영어 용어. 같은 7주차지만 과목이 다른 A의 자료는 쓰지 말아야 한다

필요한 것: Windows(Edge·음성 합성 Heami/Zira), ffmpeg, Tesseract, 전사 모델.
실제 입출력 폴더는 건드리지 않는다 — 도구 폴더의 .tmp 안에서 만들고 지운다.
"""
import difflib
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
import zipfile
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
# 강의 B — 1920×1080 전체 화면 슬라이드. 제목은 화면 폭의 60%까지만 가는 어두운 띠 위에
# 있고 오른쪽 위에는 로고 글자가 있다(전체를 읽으면 제목이 로고와 묶여 사라지던 배치).
CSS_B = """
html,body { margin:0; padding:0 }
.s { width:1920px; height:1080px; box-sizing:border-box; position:relative;
     font-family:'Malgun Gothic'; background:#f4f4f1; color:#111; overflow:hidden }
.title { position:absolute; left:0; top:0; width:1150px; height:110px; background:#4a5a75; color:#fff;
         font-size:52px; font-weight:bold; padding:22px 80px; box-sizing:border-box }
.logo { position:absolute; right:90px; top:34px; font-size:40px; font-weight:bold; color:#333 }
.l { position:absolute; left:80px; font-size:44px; white-space:nowrap }
"""

# 강의 A — (id, 종류, 본문, 발화 언어, 발화, 이 대목에만 나오는 낱말, PDF에 있나). 전부 가상 자료
LECTURE_A = [
    ("s1", "slide", "<h1 style='margin-top:160px;font-size:60px'>표본통계학</h1>"
                    "<li style='list-style:none;font-size:36px'>7주차 2교시 · 추정과 신뢰구간</li>",
     "ko", "안녕하세요. 표본통계학 칠 주차 두 번째 시간입니다. 오늘은 추정과 신뢰구간을 다룹니다.",
     "안녕하세요", True),
    ("s2", "slide", "<h1>모집단과 표본의 관계</h1><ul><li><b>모집단</b>은 관심 대상 전체이다</li>"
                    "<li><b>표본</b>은 모집단에서 뽑은 일부이다</li><li>추정에는 항상 <b>표본오차</b>가 따른다</li></ul>",
     "ko", "모집단은 우리가 알고 싶은 대상 전체이고 표본은 그중 실제로 관측한 일부입니다. "
           "표본에서 계산한 값으로 모수를 추정하는데 이때 표본오차가 반드시 생깁니다.",
     "관측한", True),
    ("s3", "slide", "<h1>Confidence Interval</h1><ul><li>Point estimate plus or minus a margin of error</li>"
                    "<li>A 95% interval uses z = 1.96</li><li>Larger samples give narrower intervals</li></ul>",
     "en", "Let me switch to English for this definition. A confidence interval is a point estimate "
           "plus or minus a margin of error. Larger samples give narrower intervals.",
     "margin of error", True),
    ("s4", "slide", "<h1>신뢰수준별 임계값</h1><table><tr><th>신뢰수준</th><th>임계값 z</th></tr>"
                    "<tr><td>90%</td><td>1.645</td></tr><tr><td>95%</td><td>1.960</td></tr>"
                    "<tr><td>99%</td><td>2.576</td></tr></table>",
     "ko", "이 표는 신뢰수준별 임계값입니다. 신뢰수준을 높일수록 구간이 넓어집니다.",
     "넓어집니다", True),
    ("s5a", "slide", "<h1>표본 크기의 결정</h1><ul><li>원하는 오차 한계를 먼저 정한다</li></ul>",
     "ko", "그러면 표본을 몇 개나 뽑아야 할까요. 먼저 허용할 오차 한계를 정합니다.",
     "뽑아야", False),                       # 강의노트에는 완성본(s5)만 있다
    ("s5", "slide", "<h1>표본 크기의 결정</h1><ul><li>원하는 오차 한계를 먼저 정한다</li>"
                    "<li>n = (z × σ / E)² 로 계산하고 올림한다</li></ul>",
     "ko", "공식에 넣어 계산하고 소수점이 나오면 반드시 올림합니다.",
     "올림합니다", True),
    ("subs", "video", None, "ko", None, None, False),   # 번인 자막 영상 — 끝나며 s6 로 서서히 바뀐다
    ("s6", "slide", "<h1>편향과 표본 크기</h1><ul><li>표본이 커져도 <b>편향</b>은 줄어들지 않는다</li>"
                    "<li>무작위 추출이 편향을 막는 핵심이다</li></ul>",
     "ko", "영상에서 보셨듯이 표본을 크게 해도 편향은 줄어들지 않습니다. 무작위 추출이 핵심입니다.",
     "보셨듯이", True),
    ("s8", "slide", "<h1>현장 인터뷰</h1><ul><li>방문 조사의 실제</li></ul>"
                    "<p style='position:absolute;bottom:40px;font-size:22px'>"
                    "출처: https://www.youtube.com/watch?v=sample0001</p>",
     "ko", "이어서 조사 현장의 인터뷰 영상을 함께 보겠습니다.", "인터뷰 영상을", True),
    ("talk", "video", None, "ko",
     "저는 가상조사센터에서 방문 조사를 맡고 있습니다. 저희는 매년 이천 가구를 직접 찾아갑니다. "
     "집에 아무도 없으면 저녁에 다시 찾아가는 일이 가장 많은 시간을 차지합니다.",
     "이천 가구를", False),
    ("s9", "slide", "<h1>해외 사례</h1><ul><li>다시 찾아가기의 효과</li></ul>",
     "ko", "다음은 해외 조사팀이 들려주는 경험담입니다.", "경험담입니다", True),
    ("abroad", "video", None, "en",
     "Our survey team visits every household twice before we record a refusal. "
     "Evening visits doubled the response rate in large apartment complexes.",
     "apartment complexes", False),
    ("s7", "slide", "<h1>다음 시간 예고</h1><ul><li>가설검정의 논리</li><li>제1종 오류와 제2종 오류</li></ul>",
     "ko", "다음 시간에는 가설검정의 논리와 두 종류의 오류를 배웁니다. 수고하셨습니다.",
     "수고하셨습니다", True),
]
SUBS = ["안녕하세요 저는 가상연구소의 김가상입니다",
        "현장에서는 응답률이 가장 큰 문제입니다",
        "그래서 저희는 추출틀을 매년 새로 만듭니다"]
SUB_SEC = 6.0
FADE_SEC = 1.5          # 자막 영상이 끝나며 다음 슬라이드로 서서히 바뀌는 시간
TALK_SEC = {"talk": 26.0, "abroad": 20.0}
PATTERN = {"subs": "mandelbrot=s=1268x714:r=30",
           "talk": "life=s=317x179:mold=10:r=30:ratio=0.5:death_color=#203040:life_color=#e0c090,"
                   "scale=1268:714:flags=neighbor",
           "abroad": "cellauto=rule=110:s=634x357:r=30,scale=1268:714:flags=neighbor"}

# 강의 B — (id, 줄들[(y, 글)], 발화, 화자 위쪽을 지나 잘리던 줄 끝 낱말들)
LECTURE_B = [
    # 긴 줄은 화면 폭의 71~74%에서 끝난다 — 마지막 낱말이 좌측 66% 경계에 걸친다
    ("b1", "품질관리 7주차 · 관리도",
     [(200, "중심선은 공정 평균이고 한계선은 평균에서 떨어진 관리한계선이다"),
      (290, "점이 관리한계를 벗어나면 원인을 찾고 공정을 멈추어 즉시조치한다"),
      (560, "관리도(Control Chart)는 슈하트가 고안했다"),
      (650, "Source: 가상품질학회 (2024.3.2)")],
     "관리도는 공정이 안정되어 있는지 보는 도구입니다. 중심선과 관리한계선을 먼저 그립니다. "
     "점이 한계를 벗어나면 이상원인을 찾아서 조치합니다.",
     ["관리한계선이다", "즉시조치한다"]),
    ("b2", "공정능력지수",
     [(200, "공정능력지수는 규격의 폭을 공정 산포의 여섯 배로 나눈 비율값이다"),
      (560, "Cp와 Cpk를 함께 본다"),
      (650, "Reporter 김가상, Editor 이가상")],
     "공정능력지수는 규격의 폭을 공정 산포로 나눈 값입니다. 값이 클수록 불량이 적습니다. "
     "치우침까지 보려면 두 지수를 함께 봅니다.",
     ["비율값이다"]),
    ("b3", "정리",
     [(200, "관리도는 안정성을 공정능력지수는 공정의 능력을 보는 도구이다"),
      (560, "다음 시간: 샘플링검사")],
     "정리하면 관리도는 안정성을, 공정능력지수는 능력을 봅니다. 다음 시간에는 샘플링 검사를 배웁니다.",
     ["도구이다"]),
]
B_SEC = 27.0            # 강의 B 슬라이드 한 장의 길이 — 화자 판정에 멈춘 화면 60초가 필요하다


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


def shot(work, sid, html, size):
    page = work / f"{sid}.html"
    page.write_text(html, encoding="utf-8")
    edge(work, work / f"{sid}.png", f"--screenshot={work / (sid + '.png')}", f"--window-size={size}",
         page.as_uri())


def concat(work, v_parts, a_parts, fc, t, mp4):
    (work / f"{mp4.stem}_v.txt").write_text("\n".join(f"file '{v}'" for v in v_parts) + "\n",
                                            encoding="utf-8")
    (work / f"{mp4.stem}_a.txt").write_text("\n".join(f"file '{a}'" for a in a_parts) + "\n",
                                            encoding="utf-8")
    ff("-f", "concat", "-safe", "0", "-i", f"{mp4.stem}_v.txt", "-f", "concat", "-safe", "0",
       "-i", f"{mp4.stem}_a.txt", "-filter_complex", fc, "-map", "[v]", "-map", "1:a", "-t", f"{t:.2f}",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(mp4), cwd=work)


def build_a(work):
    """강의 A 영상과 강의노트 PDF(LMS 묶음 zip)를 만들고 정답을 돌려준다."""
    truth, v_parts, a_parts, t, pages = [], [], [], 0.0, []
    for sid, kind, body, *_ in LECTURE_A:
        if kind == "slide":
            shot(work, sid, f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
                            f"<div class=s>{BANNER}{body}</div>", "1280,720")
    for n, (sid, kind, body, lang, speech, key, in_pdf) in enumerate(LECTURE_A):
        seg = work / f"a{n}.mp4"
        if kind == "video":
            if sid == "subs":
                for i, s in enumerate(SUBS):
                    tts(s, "ko", work / f"sub{i}.wav", work)
                    silence(work / f"subpad{i}.wav", max(0.2, SUB_SEC - wav_len(work / f"sub{i}.wav")))
                    a_parts += [f"sub{i}.wav", f"subpad{i}.wav"]
                dur = SUB_SEC * len(SUBS)
                nxt = LECTURE_A[n + 1][0]
                dt = ",".join(
                    f"drawtext=fontfile='{FONT}':text='{s}':fontsize=34:fontcolor=white:box=1:"
                    f"boxcolor=black@0.6:x=(w-tw)/2:y=h-90:enable='between(t,{k * SUB_SEC},{k * SUB_SEC + SUB_SEC - 0.1})'"
                    for k, s in enumerate(SUBS))
                # 끝의 1.5초는 다음 슬라이드로 서서히 바뀐다 — 그 슬라이드가 영상에 묶이면 안 된다
                ff("-f", "lavfi", "-i", PATTERN["subs"], "-loop", "1", "-framerate", "30", "-i", f"{nxt}.png",
                   "-filter_complex", f"[0:v]trim=duration={dur:.3f},{dt},setpts=PTS-STARTPTS,"
                   f"fps=30,format=yuv420p[a];"
                   f"[1:v]scale=1268:714,setsar=1,trim=duration={FADE_SEC + 1},setpts=PTS-STARTPTS,"
                   f"fps=30,format=yuv420p[b];"
                   f"[a][b]xfade=transition=fade:duration={FADE_SEC}:offset={dur - FADE_SEC:.3f},"
                   f"trim=duration={dur:.3f},format=yuv420p[v]",
                   "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", str(seg), cwd=work)
            else:
                tts(speech, lang, work / f"{sid}.wav", work)
                dur = TALK_SEC[sid]
                silence(work / f"{sid}_pre.wav", 1.0)
                silence(work / f"{sid}_post.wav", max(0.5, dur - 1.0 - wav_len(work / f"{sid}.wav")))
                a_parts += [f"{sid}_pre.wav", f"{sid}.wav", f"{sid}_post.wav"]
                ff("-f", "lavfi", "-i", PATTERN[sid], "-t", f"{dur:.3f}", "-pix_fmt", "yuv420p",
                   "-c:v", "libx264", "-preset", "veryfast", str(seg))
            v_parts.append(seg.name)
            truth.append({"id": sid, "video": True, "start": t, "end": t + dur, "key": key})
            t += dur
            continue
        if in_pdf:
            pages.append(body)
        tts(speech, lang, work / f"{sid}.wav", work)
        silence(work / f"{sid}_pre.wav", 1.5)
        silence(work / f"{sid}_post.wav", 2.0)
        a_parts += [f"{sid}_pre.wav", f"{sid}.wav", f"{sid}_post.wav"]
        dur = 3.5 + wav_len(work / f"{sid}.wav")
        ff("-loop", "1", "-framerate", "30", "-i", f"{sid}.png", "-t", f"{dur:.3f}",
           "-vf", "scale=1268:714,setsar=1", "-pix_fmt", "yuv420p", "-c:v", "libx264",
           "-preset", "veryfast", str(seg), cwd=work)
        v_parts.append(seg.name)
        start = t - FADE_SEC if n and LECTURE_A[n - 1][0] == "subs" else t   # 서서히 나타나기 시작한 때
        truth.append({"id": sid, "start": start, "end": t + dur, "lang": lang, "key": key,
                      "page": len(pages) if in_pdf else len(pages) + 1,
                      "after_video": bool(n) and LECTURE_A[n - 1][1] == "video"})
        t += dur

    deck = work / "deck.html"
    deck.write_text(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
                    + "".join(f"<div class=s>{BANNER}{b}</div>" for b in pages), encoding="utf-8")
    pdf = work / "7주차교재.pdf"
    edge(work, pdf, f"--print-to-pdf={pdf}", "--no-pdf-header-footer", deck.as_uri())
    # LMS 에서 받은 그대로 — 묶음 이름에도, 안의 PDF 이름에도 과목이 없다(주차만 같다)
    bundle = work / "7주차 강의자료-20260925.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        z.write(pdf, pdf.name)

    mp4 = work / "표본통계학 7-2강.mp4"
    fc = ("[0:v]scale=1267:713,setsar=1[sl];color=c=0x202020:s=1920x1080:r=30[bg];"
          "life=s=160x270:mold=10:r=30:ratio=0.1:death_color=#304050:life_color=#c09070,"
          "scale=653:1080:flags=neighbor[cam];"
          "[bg][sl]overlay=0:183:shortest=1[x];[x][cam]overlay=1267:0:shortest=1,format=yuv420p[v]")
    concat(work, v_parts, a_parts, fc, t, mp4)
    return mp4, bundle, truth


def build_b(work):
    """강의 B — 화자가 슬라이드 오른쪽 아래에 겹쳐 선 전체 화면 강의. 강의자료는 없다."""
    truth, v_parts, a_parts, t = [], [], [], 0.0
    for sid, title, rows, speech, tails in LECTURE_B:
        body = (f"<div class=title>{title}</div><div class=logo>SAMPLE UNIV</div>"
                + "".join(f"<div class=l style='top:{y}px'>{txt}</div>" for y, txt in rows))
        shot(work, sid, f"<!doctype html><meta charset=utf-8><style>{CSS_B}</style>"
                        f"<div class=s>{body}</div>", "1920,1080")
        tts(speech, "ko", work / f"{sid}.wav", work)
        silence(work / f"{sid}_pre.wav", 1.5)
        silence(work / f"{sid}_post.wav", max(0.5, B_SEC - 1.5 - wav_len(work / f"{sid}.wav")))
        a_parts += [f"{sid}_pre.wav", f"{sid}.wav", f"{sid}_post.wav"]
        seg = work / f"{sid}.mp4"
        ff("-loop", "1", "-framerate", "30", "-i", f"{sid}.png", "-t", f"{B_SEC:.3f}",
           "-vf", "scale=1920:1080,setsar=1", "-pix_fmt", "yuv420p", "-c:v", "libx264",
           "-preset", "veryfast", str(seg), cwd=work)
        v_parts.append(seg.name)
        truth.append({"id": sid, "start": t, "end": t + B_SEC, "title": title, "tails": tails,
                      "rows": [txt for _y, txt in rows]})
        t += B_SEC
    mp4 = work / "품질관리 7-1강.mp4"
    # 화자: 오른쪽 아래(가로 71%~, 세로 44%~)에서 계속 움직이는 영역을 슬라이드 위에 겹친다
    fc = ("[0:v]setsar=1[sl];"
          "life=s=140x150:mold=10:r=30:ratio=0.3:death_color=#6a3030:life_color=#e8c8a0,"
          "scale=560:600:flags=neighbor[cam];"
          "[sl][cam]overlay=1360:480:shortest=1,format=yuv420p[v]")
    concat(work, v_parts, a_parts, fc, t, mp4)
    return mp4, truth


def transcribe(work, lectures, bundle):
    """실제 입출력 폴더 대신 작업 폴더를 가리키게 하고 main()을 그대로 돌린다."""
    for d in ("in", "out", "pdf"):
        (work / d).mkdir()
    for mp4 in lectures:
        shutil.copy2(mp4, work / "in" / mp4.name)
    shutil.copy2(bundle, work / "pdf" / bundle.name)
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
    return rc, time.monotonic() - t0, {p.stem: p for p in (work / "out").glob("*.md")}


def parse(md: Path):
    text = md.read_text(encoding="utf-8")
    lines = text.splitlines()
    ts = lambda s: sum(int(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
    blocks = []
    for i, ln in enumerate(lines):
        m = re.match(r"> \*\*(🖵 슬라이드|📺 영상 자막)\s*(\d+쪽)?\s*"
                     r"\[(\d\d:\d\d:\d\d) – (\d\d:\d\d:\d\d)\]", ln)
        if m:
            body = []
            for nxt in lines[i + 1:]:
                if not nxt.startswith("> "):
                    break
                body.append(nxt[2:])
            blocks.append((i, m.group(1), m.group(2), ts(m.group(3)), ts(m.group(4)), body))
    paras = [(ts(m.group(1)), m.group(2), i) for i, ln in enumerate(lines)
             if (m := re.match(r"\*\*\[(\d\d:\d\d:\d\d)\]\*\* (.*)", ln))]
    return text, lines, blocks, paras


class Grader:
    def __init__(self):
        self.fails = []

    def check(self, name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            self.fails.append(name)


def grade_a(g, md: Path, truth):
    text, lines, blocks, paras = parse(md)
    slides = [b for b in blocks if b[1] == "🖵 슬라이드"]
    pages = [int(b[2][:-1]) for b in slides if b[2]]
    want = sorted({s["page"] for s in truth if not s.get("video")})

    g.check("이름에 과목이 없는 LMS 묶음 속 자료를 화면과 맞춰 채택",
            "강의자료: 7주차 강의자료-20260925.zip/7주차교재.pdf" in text)
    g.check("모든 쪽이 한 번씩, 순서대로", pages == want, f"{pages} (정답 {want})")
    for page in want:
        # 항목이 하나씩 나타나는 슬라이드는 첫 단계가 뜬 때가 그 쪽의 시작이다
        start = min(s["start"] for s in truth if s.get("page") == page)
        # 영상 직후 슬라이드도 같은 기준이다 — 영상 동안 비교에서 빠져 있던 영역이 다시 들어올 때
        # 늦게 골라지지만, 실제로 바뀐 초로 되돌려 적는다(예전에는 최대 3초 늦었다)
        hit = [b for b in slides if b[2] == f"{page}쪽"]
        err = hit[0][3] - start if hit else None
        g.check(f"{page}쪽 전환 시각 ±1.5초", err is not None and abs(err) <= 1.5,
                f"{'없음' if err is None else f'{err:+.1f}초'}")
    g.check("PDF 쪽마다 되풀이되는 배너를 뺌", "S A M P L E" not in text and "SAMPLE" not in text)

    videos = [s for s in truth if s.get("video")]
    g.check("재생 영상 3곳을 모두 찾음", f"재생된 영상 {len(videos)}곳" in text,
            next((ln for ln in lines if "재생된 영상" in ln), "머리말에 영상 표시 없음")[:40])
    sub = next(s for s in videos if s["id"] == "subs")
    subs = [b for b in blocks if b[1] == "📺 영상 자막" and b[3] < sub["end"] + 5 and b[4] > sub["start"] - 5]
    g.check("자막 영상이 한 블록으로", len(subs) == 1, f"{len(subs)}개")
    if subs:
        g.check("자막 영상 구간 ±3초 (뒤 슬라이드로 서서히 바뀌어도)",
                abs(subs[0][3] - sub["start"]) <= 3 and abs(subs[0][4] - sub["end"]) <= 3,
                f"{subs[0][3]}–{subs[0][4]} (정답 {sub['start']:.0f}–{sub['end']:.0f})")
    for v in videos:
        inside = [p for p in paras if v["start"] - 1 <= p[0] < v["end"]]
        g.check(f"{v['id']} 영상 속 발화에 📺", inside and all(p[1].startswith("📺") for p in inside),
                f"{sum(p[1].startswith('📺') for p in inside)}/{len(inside)}")
        after = next((p for p in paras if p[0] >= v["end"]), None)
        g.check(f"{v['id']} 영상 직후 교수 발화에는 📺 없음", after is not None and "📺" not in after[1],
                after[1][:30] if after else "없음")
    far = [p for p in paras if not any(v["start"] - 16 <= p[0] <= v["end"] + 16 for v in videos)]
    g.check("영상과 먼 교수 발화에는 📺 없음", not any("📺" in p[1] for p in far))
    g.check("제목 슬라이드를 영상 자막으로 오판하지 않음",
            not any(b[1] == "📺 영상 자막" and b[3] < 5 for b in blocks))
    g.check("화면이 가만한 영어 설명은 영상이 아님",
            any("margin of error" in p[1].lower() and "📺" not in p[1] for p in paras))

    body = " ".join(p[1] for p in paras).lower()
    g.check("한국어 강의 속 영어 설명이 살아 있음", "confidence interval" in body
            and "margin of error" in body)
    # 설명이 제 슬라이드 밑에 있어야 한다 — 다음 슬라이드 설명이 앞 슬라이드에 붙지 않게
    for s in truth:
        if not s.get("key") or s.get("video"):
            continue
        at = next((p[2] for p in paras if s["key"].lower() in p[1].lower()), None)
        owner = max((b for b in blocks if at is not None and b[0] < at), key=lambda b: b[0],
                    default=None)
        page = f"{s['page']}쪽"
        g.check(f"{s['id']} 설명이 제 슬라이드 밑에", owner is not None and owner[2] == page,
                f"{owner[2] if owner else '없음'} (정답 {page})")


def grade_b(g, md: Path, truth):
    text, lines, blocks, paras = parse(md)
    slides = [b for b in blocks if b[1] == "🖵 슬라이드"]
    shown = "\n".join(ln for b in slides for ln in b[5])
    squash = lambda s: re.sub(r"[^0-9A-Za-z가-힣]", "", s)   # 띄어쓰기·가운뎃점 모양 차이는 보지 않는다
    g.check("같은 주차라도 과목이 다른 자료는 쓰지 않음", "강의자료:" not in text)
    g.check("화면 차례와 슬라이드가 있음", "## 화면 차례" in text and len(slides) >= len(truth),
            f"슬라이드 {len(slides)}장")
    out = [ln for b in slides for ln in b[5]]
    sim = lambda a, b: difflib.SequenceMatcher(None, squash(a), squash(b)).ratio()
    best = lambda want: max(out, key=lambda ln: sim(ln, want), default="")
    # OCR 한두 글자 오독은 이 검증의 대상이 아니다 — 줄이 있는지, 잘리지 않았는지를 본다
    for s in truth:
        got = best(s["title"])
        g.check(f"{s['id']} 어두운 띠 위 제목", sim(got, s["title"]) >= 0.8, f"{got!r}")
        for row, tail in zip([r for r in s["rows"] if any(r.endswith(t) for t in s["tails"])], s["tails"]):
            got = best(row)
            g.check(f"{s['id']} 화자 위를 지나는 긴 줄 끝까지 ('{tail}')",
                    sim(got, row) >= 0.85 and len(squash(got)) >= 0.95 * len(squash(row)), f"{got!r}")
    # 영어 줄 속 한글 이름은 섞인 판본으로 살아난다. 한글 줄 괄호 속 영어('Control Chart')는
    # 여전히 자주 깨지므로(README 한계) 채점하지 않는다
    g.check("영어 줄에 섞인 한글 이름", "가상품질학회" in shown and "김가상" in shown)
    rows = [r for s in truth for r in s["rows"] + [s["title"]]]
    stray = [ln for ln in out if len(squash(ln)) >= 4 and max(sim(ln, r) for r in rows) < 0.6]
    g.check("화자 영역·로고를 글자로 읽지 않음 (슬라이드에 없는 줄 없음)", not stray, str(stray[:3]))
    g.check("영상이 없는 강의에 📺 없음", "📺" not in text)


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
        print("합성 강의 2편을 만드는 중... (약 1분)")
        t0 = time.monotonic()
        mp4_a, bundle, truth_a = build_a(work)
        mp4_b, truth_b = build_b(work)
        total = truth_a[-1]["end"] + truth_b[-1]["end"]
        print(f"  {mp4_a.name}: {truth_a[-1]['end']:.0f}초 · {mp4_b.name}: {truth_b[-1]['end']:.0f}초 "
              f"({time.monotonic() - t0:.0f}초)")
        rc, took, mds = transcribe(work, [mp4_a, mp4_b], bundle)
        print(f"\n채점 (엔진 {took:.0f}초, 실시간 대비 {total / took:.1f}배)")
        if rc != 0 or set(mds) != {mp4_a.stem, mp4_b.stem}:
            print(f"  FAIL  엔진이 실패했습니다 (종료 코드 {rc}, 산출물 {sorted(mds)})")
            return 1
        g = Grader()
        print(f"\n[{mp4_a.stem}] 화자 창이 따로 있는 강의 · 재생 영상 셋 · LMS 묶음 자료")
        grade_a(g, mds[mp4_a.stem], truth_a)
        print(f"\n[{mp4_b.stem}] 화자가 슬라이드 위에 겹친 강의")
        grade_b(g, mds[mp4_b.stem], truth_b)
        print(f"\n결과: {'전부 통과' if not g.fails else '실패 ' + ', '.join(g.fails)}")
        return 1 if g.fails else 0
    finally:
        if keep:
            print(f"\n작업 폴더를 남겨 둡니다: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
