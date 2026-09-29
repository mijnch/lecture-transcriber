# -*- coding: utf-8 -*-
"""시간축 조립 — 구절을 문단으로 묶고, 화면 글자와 발화를 맞대 재생 영상(📺)을 가르며,
영상 구간의 시작·끝을 화면 움직임과 언어로 다듬는다.

transcribe 에서 떼어 낸 모듈이다 — transcribe 가 이름을 모두 재수출하므로
transcribe.X 로 부르던 곳은 그대로 된다.
"""

import re

from speech_repair import GAP_JOIN_SEC, ends_sentence
from screen_scan import clean_moving_line
from slide_pdf import page_key


# 문단 분리 기준 — 상한에 도달해도 문장이 끝날 때까지 기다린다(문장 중간 절단 방지)
PARA_GAP_SEC = 2.0       # 이 이상 침묵하면 새 문단
PARA_SOFT_SEC = 20.0     # 이 길이를 넘으면 다음 문장 끝에서 문단을 닫는다
PARA_SOFT_CHARS = 300
PARA_HARD_SEC = 45.0     # 문장 끝이 끝내 안 나올 때의 강제 상한
PARA_HARD_CHARS = 650
SUSPECT_REPEAT = 3       # 같은 문장이 이 횟수 이상 반복되면 환각
VIDEO_JOIN_SEC = 30      # 재생 영상 구간 사이의 틈이 이보다 짧으면 같은 영상으로 잇는다
SUBTITLE_MATCH_RATIO = 0.6   # 발화와 이만큼 겹치는 짧은 화면 글자는 영상 자막
SUBTITLE_LIVE = 0.2      # 화면이 이만큼 움직이고 있을 때만 자막으로 본다 (영상 재생 중)


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
