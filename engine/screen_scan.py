# -*- coding: utf-8 -*-
"""화면 읽기의 판정부 — 화면 전환 검출(바뀐 화소 비율, 움직이는 칸 제외), 프레임 고르기,
화자 영역 판단, OCR 판본의 줄별 점수·합치기, 잘린 줄 이어 읽기, 잡음 줄 정리, 비슷한 슬라이드 병합.
Tesseract 를 실제로 부르는 실행부(ocr_lines·read_screen·extract_slides)는 transcribe 에 있다.

transcribe 에서 떼어 낸 모듈이다 — transcribe 가 이름을 모두 재수출하므로
transcribe.X 로 부르던 곳은 그대로 된다.
"""

import re
import subprocess
from pathlib import Path


# 슬라이드 읽기(OCR)
OCR_MIN_CONF = 60        # 이보다 확신이 낮은 줄은 사진 속 잡음으로 버린다
                         # 사진 슬라이드의 잡음('- 31 47 26 58', 'Qe 4 「')은 62~69, 진짜 숫자 표 줄은 85 이상
MIX_PENALTY = 6          # 판본 고르기에서 글자종이 섞인 조각(굵은 한글→HAS) 하나당 깎는 점수 —
                         # 12면 영어 줄 속 한글 이름(섞인 판본)이 숫자로 깨진 kor 판본에 졌다.
                         # 12·8·6 은 정답 PDF 대비 정확도가 같고, 4 부터 떨어진다(실측)
SLIDE_CROP = 0.66        # 화자가 곁들여진 화면에서 슬라이드가 차지하는 좌측 비율
EDGE_TOUCH = 0.97        # 잘라 읽은 줄의 오른쪽 끝이 이만큼 가면 가장자리에서 잘린 줄이다
SPEAKER_CELL = 0.05      # 슬라이드가 멈춘 동안 이만큼 자주 움직인 칸은 화자가 서 있는 곳
SLIDE_MERGE_RATIO = 0.55  # 낱말이 이만큼 겹치면 같은 슬라이드로 본다
SLIDE_MERGE_LOOKBACK = 3  # 직전 몇 장까지 견주어 볼지 (애니메이션 단계 대응)

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
