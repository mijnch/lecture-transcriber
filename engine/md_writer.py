# -*- coding: utf-8 -*-
"""산출물 쓰기 — 머리말·화면 차례·슬라이드/영상 블록·⚠ 표식을 갖춘 Markdown 을 쓰고,
다시 만들지 판정하는 마커(본문 SHA-256)를 읽는다.

transcribe 에서 떼어 낸 모듈이다 — transcribe 가 이름을 모두 재수출하므로
transcribe.X 로 부르던 곳은 그대로 된다.
"""

import datetime
import hashlib
import json
import re
from pathlib import Path

from speech_repair import fmt_ts
from screen_scan import looks_garbled
from timeline import screen_ends, video_spans


# 엔진 로직 판(版). 문단화·OCR·출력 형식을 바꿀 때마다 올린다.
# 이 값이 산출물 지문에 들어가므로, 올리면 기존 MD가 자동으로 다시 만들어진다.
ENGINE_REV = 15
MARKER = "<!-- transcriber:"


def config_fingerprint(cfg):
    slide = f"|slide{cfg['ocr_언어']}" if cfg["슬라이드_읽기"] else ""
    # 엔진 판을 함께 넣는다 — 로직을 고치면 기존 산출물이 자동으로 갱신된다
    return f"rev{ENGINE_REV}|{cfg['model']}|{cfg['language']}|beam{cfg['beam_size']}{slide}"


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
