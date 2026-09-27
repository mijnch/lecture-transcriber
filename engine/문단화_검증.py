# -*- coding: utf-8 -*-
"""엔진 로직 단위 검증 — 전사 없이 가짜 입력으로 즉시 확인한다.

표준 라이브러리만으로 돈다. 화면 전환 검출 검증만 numpy 가 필요하다(없으면 건너뛴다).
예시 문장은 전부 가상 자료다 — 실제 강의의 조각은 어디에도 남기지 않는다.
"""
import sys
import tempfile
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import transcribe as T
from transcribe import (group_paragraphs, ends_sentence, looks_hallucinated,
                        script_mix_penalty, ocr_score, slide_key, merge_slides,
                        label_slides, video_spans, md_body_hash, merge_ocr_passes,
                        MARKER, PARA_HARD_CHARS, PARA_HARD_SEC)

fails, passed = [], 0


def check(name, cond, detail=""):
    global passed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if cond:
        passed += 1
    else:
        fails.append(name)


print("문단화")

# 1) 침묵이 길면 문단을 나눈다
p = group_paragraphs([(0, 3, "첫 문장입니다."), (10, 13, "긴 침묵 뒤 문장입니다.")])
check("긴 침묵에서 분리", len(p) == 2, f"문단 {len(p)}개")

# 2) 짧은 간격이면 이어붙인다
p = group_paragraphs([(0, 3, "앞 문장입니다."), (3.5, 6, "뒤 문장입니다.")])
check("짧은 간격은 병합", len(p) == 1, f"문단 {len(p)}개")

# 3) 문장이 끝날 때마다 닫히므로, 문단은 문장 중간에서 끊기지 않는다 (핵심 회귀 방지)
#    이전 판에서는 이 단정을 계산만 해두고 쓰지 않아 검증이 무력화되어 있었다.
segs = [(i * 6, i * 6 + 6, "이것은 제대로 끝나는 문장입니다. " * 2) for i in range(20)]
p = group_paragraphs(segs)
mid_cut = [t for _, t, _ in p if not ends_sentence(t)]
check("문장이 있으면 중간에서 절단하지 않음", not mid_cut,
      f"문단 {len(p)}개, 절단 {len(mid_cut)}개")

# 4) 문장 끝이 끝내 없으면 강제 상한에서 끊는다 (문단 폭주 방지)
segs = [(i * 5, i * 5 + 5, "끝나지 않는 말이 계속됩니다 " * 3) for i in range(20)]
p = group_paragraphs(segs)
check("문장 끝이 없으면 강제 상한에서 끊음",
      p and all(len(t) <= PARA_HARD_CHARS + 200 for _, t, _ in p),
      f"최대 {max(len(t) for _, t, _ in p)}자")

# 5) 강제 상한이 시각 기준으로도 작동한다
segs = [(i * 4, i * 4 + 4, "끝나지 않는 말이 계속됩니다 ") for i in range(40)]
p = group_paragraphs(segs)
spans = [p[i + 1][0] - p[i][0] for i in range(len(p) - 1)]
check("강제 상한으로 문단 폭주 차단", spans and max(spans) <= PARA_HARD_SEC + 5,
      f"문단 {len(p)}개, 최대 간격 {max(spans) if spans else 0:.0f}초")

# 6) 소프트 상한을 넘겨도 문장이 끝날 때까지 기다렸다가 닫는다
segs = [(i * 4, i * 4 + 4, "문장 조각이 이어집니다 ") for i in range(6)]
segs += [(24, 28, "여기서 문장이 끝납니다."), (28.5, 32, "새 문단의 첫 문장입니다.")]
p = group_paragraphs(segs)
check("소프트 상한 초과 후 문장 끝에서 분리",
      len(p) >= 2 and ends_sentence(p[0][1]), f"첫 문단 끝: ...{p[0][1][-12:]!r}")

# 7) 빈 텍스트는 무시하고 타임스탬프는 첫 발화 기준
p = group_paragraphs([(0, 1, "   "), (5, 8, "실제 발화입니다.")])
check("빈 세그먼트 무시 · 시작 시각 정확", len(p) == 1 and p[0][0] == 5, f"시작 {p[0][0]}")

# 8) 입력이 없으면 빈 결과
check("빈 입력 처리", group_paragraphs([]) == [])

# 9) 소수점·번호 뒤의 마침표는 문장 끝이 아니다
check("소수점을 문장 끝으로 보지 않음",
      not ends_sentence("유의수준은 0.") and not ends_sentence("항목 1.")
      and ends_sentence("문장입니다.") and ends_sentence("맞습니까?"))
p = group_paragraphs([(0, 22, "유의수준은 아주 길게 설명하면 이렇게 되는데 결론적으로 0."),
                      (22.5, 25, "05 입니다.")])
check("소수를 문단 경계로 쪼개지 않음", len(p) == 1, f"문단 {len(p)}개")

# 10) 슬라이드가 바뀌면 문단을 끊는다 — 다음 슬라이드 설명이 앞 슬라이드 밑에 붙지 않게
p = group_paragraphs([(10, 14, "첫 슬라이드 설명"), (15, 18, "다음 슬라이드 설명")], breaks=[14.5])
check("슬라이드 전환에서 문단 분리", len(p) == 2 and p[1][0] == 15, f"{[x[0] for x in p]}")
p = group_paragraphs([(10, 14, "첫 설명"), (15, 18, "이어지는 설명")], breaks=[9.0, 30.0])
check("문단 밖의 전환은 무시", len(p) == 1, f"문단 {len(p)}개")

print("\n구절 · 빠진 말소리")

W = lambda *ws: [(a, b, w) for a, b, w in ws]
# 11) 낱말 시각으로 긴 세그먼트를 구절로 나눈다 (핫워드를 넣으면 세그먼트가 24초로 뭉개진다)
seg = (0.0, 24.0, "앞부분 설명입니다. 뒤 설명", False,
       W((0.0, 0.5, " 앞부분"), (0.5, 1.2, " 설명입니다."), (6.0, 6.5, " 뒤"), (6.5, 7.0, " 설명")))
ph = T.split_phrases([seg])
check("낱말 시각으로 구절 분리", [x[0] for x in ph] == [0.0, 6.0], str([(x[0], x[2]) for x in ph]))
# 12) 세그먼트 앞머리의 외톨이 낱말은 뒤 낱말과 묶이고 뒤 낱말의 시각을 쓴다
seg = (38.5, 70.0, "이 표는 정리한 것입니다", False,
       W((62.0, 62.2, " 이"), (66.5, 66.9, " 표는"), (67.0, 68.0, " 정리한"), (68.0, 69.0, " 것입니다")))
ph = T.split_phrases([seg])
check("외톨이 앞머리 낱말을 뒤로 묶음",
      len(ph) == 1 and ph[0][0] == 66.5 and ph[0][2] == "이 표는 정리한 것입니다", str(ph))
check("외톨이 낱말은 빈틈 계산에서 뺌", T.reliable_words(seg[4])[0][2] == " 표는")
check("낱말 시각이 없으면 세그먼트 그대로",
      T.split_phrases([(1.0, 3.0, "그대로입니다", True)]) == [(1.0, 3.0, "그대로입니다", True)])

# 13) 말소리는 있는데 낱말이 없는 곳을 찾는다 (다른 언어로 말한 대목이 통째로 빠지는 문제)
gaps = T.find_gaps([(0, 10), (12, 30)], [(0, 8.5), (12, 15), (25, 30)])
check("빈틈 찾기 (숨 쉴 틈 같은 짧은 빈틈은 제외)",
      [(round(a, 1), round(b, 1)) for a, b in gaps] == [(15.3, 24.7)],
      str([(round(a, 1), round(b, 1)) for a, b in gaps]))
gaps = T.find_gaps([(0, 30)], [(0, 4), (8.5, 9), (13, 30)])
check("가까운 빈틈은 한 번에 다시 읽음",
      [(round(a, 1), round(b, 1)) for a, b in gaps] == [(4.3, 12.7)], str(gaps))
check("낱말이 다 덮으면 빈틈 없음", T.find_gaps([(0, 10)], [(0, 5), (5, 10)]) == [])

# 13-2) ⚠ 구간 묶기 — 다른 언어로 말한 대목이 환각으로 채워진 곳을 다시 읽기 위해
segs = [(0, 5, "정상", False), (5, 9, "환각1", True), (10, 14, "환각2", True),
        (20, 21, "짧은 의심", True), (30, 40, "정상", False)]
check("⚠ 구간 묶기 (가까운 것끼리, 짧은 것은 제외)", T.suspect_spans(segs) == [(5, 14)],
      str(T.suspect_spans(segs)))
# 13-3) 다른 언어로 다시 읽힌 구간에서 화면이 움직였으면 재생 영상 — 그 사이 화면도 영상으로
scr = [(100.0, ["뉴스 스튜디오"], None, "슬라이드"), (130.0, ["요약"], "3쪽", "슬라이드"),
       (200.0, ["다음 슬라이드"], None, "슬라이드")]
motion = [0.8 if 95 <= n <= 160 else 0.0 for n in range(300)]
sp, sc = T.add_language_spans([], scr, [(96.0, 160.0, "ko"), (250.0, 280.0, "ko")], motion)
check("언어가 바뀌고 화면이 움직인 구간은 영상", sp == [(96.0, 160.0)], str(sp))
check("그 사이 화면은 영상 화면으로 (PDF 쪽은 제외)",
      [k for *_x, k in sc] == ["자막", "슬라이드", "슬라이드"], str([k for *_x, k in sc]))
sp, _sc = T.add_language_spans([], scr, [(96.0, 110.0, "ko"), (150.0, 160.0, "ko")], motion)
check("영상이 계속 움직였으면 사이의 번역·노래 구간도 같은 영상", sp == [(96.0, 160.0)], str(sp))
still = [0.8 if 95 <= n <= 111 or 149 <= n <= 160 else 0.0 for n in range(300)]
sp, _sc = T.add_language_spans([], scr, [(96.0, 110.0, "ko"), (150.0, 160.0, "ko")], still)
check("사이에 화면이 멈췄으면 따로", sp == [(96.0, 110.0), (150.0, 160.0)], str(sp))

# 13-4) 오른쪽 화자 판정 — 슬라이드 쪽이 멈춘 동안 오른쪽이 움직여야 화자다.
#       전체 화면 영상이 재생되면 좌우가 함께 움직이므로 그 시간은 빼고 본다.
gw, cut = T.SCAN_W // T.LIVE_CELL, int(T.SCAN_W // T.LIVE_CELL * T.SLIDE_CROP)
def mix(right, left):          # 오른쪽·왼쪽 움직임 → (화면 전체, 좌측) 비율
    return ((right * (gw - cut) + left * cut) / gw, left)
split = [(0.0, 0.0)] + [mix(0.25, 0.0)] * 200 + [mix(0.6, 0.6)] * 200      # 좌우 분할 + 긴 영상
full = [(0.0, 0.0)] + [mix(0.0, 0.0)] * 300 + [mix(0.5, 0.5)] * 60          # 전체 화면 슬라이드 + 영상
check("영상을 오래 틀어도 좌우 분할 화자를 알아봄", T.speaker_on_right(split) is True)
check("전체 화면 영상은 화자로 보지 않음(자르지 않음)", T.speaker_on_right(full) is False)
check("판단할 만큼 멈춘 시간이 없으면 자르지 않음", T.speaker_on_right([mix(0.5, 0.5)] * 300) is False)
# 13-5) 교수와 다른 문자로 옮겨진 구간 — 영어 강의 속 한글 문장은 근거, 한국어 강의 속 용어는 아님
fs = T.foreign_spans([(0, 3, "Let me show you a clip.", False), (4, 8, "가상 스튜디오에 지금 비가 많이 옵니다", False),
                      (9, 12, "현장 연결이 잠시 끊겼습니다", False), (30, 33, "That was the anchor.", False)], "en")
check("영어 강의 속 한글 문장 구간", fs == [(4, 12, "ko")], str(fs))
check("한국어 강의 속 짧은 영어 용어는 근거가 아님",
      T.foreign_spans([(0, 3, "이것이 Supply Chain 입니다", False)], "ko") == [])

# 13-6) 영상 속 정지 글자 카드로 끊긴 구간은 잇되, 교수가 돌아온 틈은 잇지 않는다
scr = [(100.0, ["a"], None, "자막"), (131.0, ["정지 카드"], None, "슬라이드"), (140.0, ["b"], None, "자막"),
       (300.0, ["c"], None, "자막"), (320.0, ["강의 슬라이드"], "5쪽", "슬라이드"), (330.0, ["d"], None, "자막")]
sp, sc = T.join_video_spans([(100.0, 130.0), (140.0, 170.0), (300.0, 315.0), (330.0, 340.0)], scr)
check("정지 카드 틈은 같은 영상으로 잇기", sp[0] == (100.0, 170.0) and sc[1][3] == "자막", f"{sp} {sc[1][3]}")
check("틈에 강의자료 쪽이 뜨면 잇지 않음", (300.0, 315.0) in sp and (330.0, 340.0) in sp, str(sp))
sp, _sc = T.join_video_spans([(100.0, 130.0), (148.0, 170.0)], scr[:1],
                             foreign=[(100.0, 130.0, "ko"), (148.0, 170.0, "ko")],
                             phrases=[(135.0, 140.0, "That was a famous clip.", False)])
check("다른 언어 영상 사이에 교수가 논평하면 잇지 않음", sp == [(100.0, 130.0), (148.0, 170.0)], str(sp))

# 13-7) 자막도 언어 차이도 없는 영상 — 오래·크게 움직이고 출처 주소가 있으면 영상
talk = [0.0] * 600
for n in range(300, 394):
    talk[n] = 0.14 if n % 5 else 0.0          # 94초 중 약 80%가 움직이는 인터뷰
url = [(290.0, ["출처: https://www.youtube.com/watch?v=abc"], None, "슬라이드"),
       (400.0, ["다음 슬라이드"], None, "슬라이드")]
mv = T.motion_video_spans(talk, url)
check("움직임 + 출처 주소 = 재생 영상", len(mv) == 1 and 295 <= mv[0][0] <= 305 and mv[0][1] >= 390, str(mv))
check("출처 주소가 없으면 움직임만으로는 영상이 아님",
      T.motion_video_spans(talk, [(290.0, ["사진 설명"], None, "슬라이드")]) == [])
pen = [0.03 if 300 <= n < 400 else 0.0 for n in range(600)]    # 판서는 가는 선이라 움직임이 작다
check("판서 수준의 움직임은 영상이 아님", T.motion_video_spans(pen, url) == [])
cam = [0.2] * 600                                               # 카메라만 비추는 강의
check("정지 슬라이드가 거의 없는 강의는 움직임으로 가르지 않음", T.motion_video_spans(cam, url) == [])
# 13-8) 영상 속 글자 카드가 32초 동안 4장 — 30초를 넘어도 같은 영상으로 잇는다
cards = [(100.0, ["a"], None, "자막"), (131.0, ["카드1"], None, "슬라이드"), (134.0, ["카드2"], None, "슬라이드"),
         (137.0, ["카드3"], None, "슬라이드"), (150.0, ["카드4"], None, "슬라이드"), (163.0, ["b"], None, "자막")]
sp, _sc = T.join_video_spans([(100.0, 131.0), (163.0, 200.0)], cards)
check("빠르게 바뀌는 글자 카드 틈은 이음", sp == [(100.0, 200.0)], str(sp))
slow = [(100.0, ["a"], None, "자막"), (131.0, ["교수 슬라이드"], None, "슬라이드"), (180.0, ["b"], None, "자막")]
sp, _sc = T.join_video_spans([(100.0, 131.0), (180.0, 200.0)], slow)
check("한 장이 오래 뜬 틈(교수 슬라이드)은 잇지 않음", sp == [(100.0, 131.0), (180.0, 200.0)], str(sp))

print("\n신뢰할 수 없는 구간 표식")

# 14) 세그먼트의 의심 표식이 문단까지 전달된다
p = group_paragraphs([(0, 3, "정상 문장입니다.", False), (10, 13, "이상한 문장입니다.", True)])
check("의심 표식 전달", len(p) == 2 and p[0][2] is False and p[1][2] is True,
      f"{[x[2] for x in p]}")

# 15) 같은 문장이 되풀이되면 환각으로 표식한다
check("반복 환각 탐지",
      looks_hallucinated("I didn't do it. I didn't do it. I didn't do it. I didn't do it.")
      and not looks_hallucinated("첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다."))
p = group_paragraphs([(0, 20, "What is the new A. What is the new A. What is the new A.")])
check("반복 문단은 의심으로 표시", p and p[0][2] is True)

print("\n슬라이드 OCR")

# 16) 한글 줄에 낀 라틴 조각을 오인식으로 센다 (kor+eng 회귀 방지)
bad = "HAS 자주 던지는 편이고 상대의 SS 끝까지 듣는다"
good = "질문을 자주 던지는 편이고 상대의 말을 끝까지 듣는다"
check("글자종 섞임 벌점", script_mix_penalty(bad) == 2 and script_mix_penalty(good) == 0,
      f"섞임 {script_mix_penalty(bad)} / 정상 {script_mix_penalty(good)}")
check("더 잘 읽힌 판본을 고름", ocr_score([good]) > ocr_score([bad]),
      f"{ocr_score([good])} > {ocr_score([bad])}")
check("영문 줄은 벌점 없음", script_mix_penalty("An interval of 1.96 units") == 0)

# 17) 한 글자 조각은 비교에서 뺀다
check("비교 낱말 정리", slide_key(["표본분포 의 A ㄱ 추정"]) == {"표본분포", "추정"},
      str(sorted(slide_key(["표본분포 의 A ㄱ 추정"]))))

# 18) 오인식이 심해 잘 안 겹치는 같은 슬라이드도 합친다
a = ["표본추출의 기본 원리", "단순무작위추출과 층화추출", "비용 시간 정밀도"]
b = ["Oy 표본추출의 기본 원리", "단순무작위추출과 층화추출", "비용 시간"]
merged = merge_slides([(21.0, a), (120.0, b)])
check("오인식된 같은 슬라이드 병합", len(merged) == 1, f"{len(merged)}장")
check("병합 시 처음 시각 유지", merged and merged[0][0] == 21.0)
check("병합 시 잘 읽힌 판본 유지", merged and merged[0][1] == a)

# 19) 다른 슬라이드는 합치지 않는다
other = ["가설검정의 절차", "귀무가설과 대립가설", "유의수준 결정"]
check("다른 슬라이드는 유지", len(merge_slides([(0.0, a), (60.0, other)])) == 2)

# 20) 세 장 건너 되풀이되는 판본도 합친다 (직전 한 장만 보던 문제)
c = ["표본추출의 기본 원리", "단순무작위추출과 층화추출", "비용 시간 정밀도 등"]
check("건너뛴 중복도 병합",
      len(merge_slides([(0.0, a), (30.0, other), (60.0, c)])) == 2,
      f"{len(merge_slides([(0.0, a), (30.0, other), (60.0, c)]))}장")

# 21) 한 화면 안에서 줄마다 잘 읽힌 쪽을 고른다 (한국어 줄은 kor, 영어 줄은 eng)
#     입력은 (윗변 좌표, 글자, 확신도) 이다.
kor_pass = [(100, "표본의 크기를 먼저 정하고 계산한다", 92), (150, "11 [01115 [6561760.", 71)]
eng_pass = [(102, "SAS 크기를 HAS 계산한다", 88), (148, "All rights reserved.", 94)]
got = merge_ocr_passes([kor_pass, eng_pass])
check("줄 단위로 좋은 판본 선택",
      got == ["표본의 크기를 먼저 정하고 계산한다", "All rights reserved."], str(got))

# 22) 한 줄에 두 언어가 섞이면 한글을 살린다 (출처·고유명사가 정형 문구보다 중요)
mixed = merge_ocr_passes([[(100, "©2023. 가상마을신문. 11 [01115 [(6560060.", 78)],
                          [(101, "©2023. -HH|OtSA=. All rights reserved.", 80)]])
check("섞인 줄에서는 한글을 보존", "가상마을신문" in mixed[0], str(mixed))

# 23) 글자종이 같은 두 판본은 확신도로 가린다 (길이로는 깨진 쪽이 이길 수 있다)
eng_only = merge_ocr_passes([[(100, "A|| rights reserved by the puplisher", 62)],
                             [(100, "All rights reserved by the publisher", 93)]])
check("같은 글자종은 확신도로 판정",
      eng_only == ["All rights reserved by the publisher"], str(eng_only))

# 24) 한쪽에만 있는 줄은 버리지 않는다
only = merge_ocr_passes([[(100, "위쪽 줄입니다", 90)], [(400, "아래쪽 줄입니다", 90)]])
check("한쪽에만 잡힌 줄도 보존", only == ["위쪽 줄입니다", "아래쪽 줄입니다"], str(only))
check("빈 결과 처리", merge_ocr_passes([[], []]) == [])

print("\n화면 전환 검출")

try:
    import numpy as np
except ImportError:
    np = None
    print("  (numpy 가 없어 건너뜀)")
if np is not None:
    def frames(total, change_at=(), moving=None, video=()):
        """흰 바탕에 글자 한 줄씩 늘어나는 슬라이드 + 움직이는 칸 + 재생 영상 구간."""
        rng = np.random.default_rng(0)
        f = np.full((240, 320), 245, np.uint8)
        lines = 0
        for n in range(total):
            if n in change_at:
                lines += 1
                f = f.copy()
                f[20 + lines * 14:28 + lines * 14, 20:120] = 30     # 글자 한 줄(≈1%)
            g = f.copy()
            if moving:
                y0, y1, x0, x1 = moving
                g[y0:y1, x0:x1] = rng.integers(0, 255, (y1 - y0, x1 - x0))
            if any(a <= n < b for a, b in video):
                g[:, :210] = rng.integers(0, 255, (240, 210))
            yield n, g

    picks, _left, _live, _shares = T.select_frames(frames(60, change_at=(10, 30)))
    check("글자 한 줄 늘어난 전환을 잡음", picks == [0, 10, 30], str(picks))
    picks, _left, live_time, _ = T.select_frames(frames(60, change_at=(20, 40), moving=(0, 240, 220, 320)))
    check("움직이는 화자 창은 무시하고 전환만 잡음",
          20 in picks and 40 in picks and len([p for p in picks if p > 5]) == 2, str(picks))
    check("화자 창 쪽이 움직였다고 기록", live_time[:, 14:].mean() > 0.5 > live_time[:, :12].mean())

    def fidget(total, change_at):
        """화자가 가끔씩만 움직인다 — '계속 움직이는 칸'으로는 걸러지지 않는다."""
        for n, g in frames(total, change_at):
            if n % 9 == 0 and n:
                g = g.copy()
                g[60:200, 240:300] = 40 + (n * 37) % 180
            yield n, g
    full, left, _lt, _s = T.select_frames(fidget(60, (20, 40)))
    check("가끔 움직이는 화자는 좌측 기준에서 무시",
          left == [0, 20, 40] and len(full) > len(left), f"전체 {full} / 좌측 {left}")
    picks, _left, _l, shares = T.select_frames(frames(60, change_at=(5,), video=((20, 40),)))
    vids = [p for p in picks if 20 <= p < 40]
    check("영상 재생 중에는 촘촘히 봄", len(vids) >= 5, str(vids))
    check("정지 슬라이드는 움직임 0",
          T.live_share(shares, 5, "") == 0.0 and T.live_share(shares, 25, "crop") > 0.5)

print("\n영상 자막 구분")

# 25) 발화와 겹치는 짧은 화면 글자는, 화면이 움직이고 있을 때만 자막이다
paras = [(30.0, "안녕하세요 가상연구소의 김가상입니다 오늘은 현장 경험을 말씀드립니다")]
subs = [(29.0, ["안녕하세요 가상연구소의 김가상입니다"]),
        (33.0, ["오늘은 현장 경험을 말씀드립니다"]),
        (200.0, ["표본추출의 절차", "표집틀 작성", "표본 크기 결정", "추출과 조사"])]
lab = label_slides(subs, paras, {29.0: 0.8, 33.0: 0.9})
check("자막 판정", [x[3] for x in lab] == ["자막", "자막", "슬라이드"], str([x[3] for x in lab]))
lab = label_slides(subs[:1], paras, {29.0: 0.0})
check("제목을 소리 내 읽어도 화면이 가만하면 슬라이드", lab[0][3] == "슬라이드", lab[0][3])
lab = label_slides([(29.0, ["안녕하세요 가상연구소의 김가상입니다"], "1쪽")], paras, {29.0: 0.9})
check("PDF 쪽과 맞춰진 화면은 슬라이드", lab[0][3] == "슬라이드", lab[0][3])
# 26) 배경이 요란해 OCR이 깨진 자막 화면도 옆 자막과 같은 영상으로 묶고, 깨진 줄은 버린다
lab = label_slides([(29.0, ["안녕하세요 가상연구소의 김가상입니다"]),
                    (31.0, ["빌닝ㅎ세요 ㅅ는", "|| 스즈 ㅇㅇ"]),
                    (60.0, ["전혀 다른 정지 슬라이드 제목", "본문 내용"])],
                   paras, {29.0: 0.8, 31.0: 0.9, 60.0: 0.0})
check("자막 옆의 움직이는 화면도 영상", [x[3] for x in lab] == ["자막", "자막", "슬라이드"],
      str([(x[3], x[1]) for x in lab]))
check("움직이는 화면의 깨진 줄은 버림", lab[1][1] == [], str(lab[1][1]))

# 27) 자막이 잇따르면 영상 재생 구간으로 묶는다
seq = [(10.0, [], None, "자막"), (14.0, [], None, "자막"), (18.0, [], None, "자막"),
       (60.0, [], None, "슬라이드"), (90.0, [], None, "자막")]
check("영상 구간 검출", video_spans(seq) == [(10.0, 18.0)], str(video_spans(seq)))
check("영상 구간은 다음 화면이 뜰 때까지",
      video_spans(seq, [14.0, 18.0, 60.0, 90.0, 100.0]) == [(10.0, 60.0)])
# 영상 첫 화면의 자막 OCR이 비어 버려져도, 화면이 움직이기 시작한 초까지 거슬러 올라간다
vid = [(5.5, [], None, "슬라이드"), (79.5, [], None, "자막"), (83.5, [], None, "자막"),
       (90.5, [], None, "슬라이드")]
motion = [0.9 if 73 <= n <= 90 else 0.0 for n in range(100)]
got = video_spans(vid, [79.5, 83.5, 90.5, 100.0], motion)
check("영상 시작을 움직임으로 거슬러 찾음", got == [(72.5, 90.5)], str(got))

print("\n강의자료 PDF 연동")

from transcribe import (align_slides_to_pdf, pdf_hotwords, parse_course, slide_title,
                        stem_week, drop_boilerplate)

PAGES = ["표본추출의 흐름\n모집단 → 표집틀 → 표본 → 추정값\n오차 ← 편향 ← 비표본오차 · 무응답",
         "좋은 설문의 조건\n1. 질문을 짧게\n2. 유도 질문을 피할 것",
         "추정량의 평가 기준\n불편성\n효율성(분산, 일치성, 충분성)"]

# 28) 깨진 화면 글자로도 올바른 쪽을 찾아내고, 본문은 PDF 원문으로 바뀐다
noisy = [(1171.0, ["모집단 ao <= A 표집틀", "추정값 nay yoy 5", "비표본오차 나아"])]
got = align_slides_to_pdf(noisy, PAGES)
check("깨진 OCR로도 쪽을 찾음", got[0][2] == "1쪽", f"쪽={got[0][2]}")
check("본문이 PDF 원문으로 교체됨", "표집틀 → 표본" in " ".join(got[0][1]), str(got[0][1])[:60])
check("시각은 화면에서 잡은 그대로", got[0][0] == 1171.0)

# 29) 같은 쪽이 잇따라 잡히면 한 번만 싣는다 (영상 재생 중 중복 폭증 방지)
rep = align_slides_to_pdf([(10.0, ["모집단 표집틀 추정값 비표본오차"]),
                           (40.0, ["모집단 표집틀 편향 무응답"]),
                           (70.0, ["좋은 설문의 조건 유도 질문"])], PAGES)
check("같은 쪽 연속 중복 제거", [r[2] for r in rep] == ["1쪽", "2쪽"],
      str([r[2] for r in rep]))

# 30) 자료가 둘이면 쪽 이름에 어느 자료인지 함께 적는다
two = align_slides_to_pdf([(10.0, ["모집단 표집틀 추정값 비표본오차 무응답"])], PAGES,
                          ["7주차교재 1쪽", "7주차교재 2쪽", "실습지 1쪽"])
check("여러 자료의 쪽 이름 구분", two[0][2] == "7주차교재 1쪽", str(two[0][2]))

# 31) 어느 쪽과도 안 맞으면 화면에서 읽은 글자를 그대로 둔다
off = align_slides_to_pdf([(5.0, ["전혀 다른 화면 내용 광고 배너 문구"])], PAGES)
check("못 맞추면 OCR 글자 유지", off[0][2] is None and off[0][1][0].startswith("전혀"))

# 32) 전문용어를 쉼표 목록으로 뽑는다 (문장처럼 이으면 Whisper가 문장부호를 거의 안 찍었다)
hot = pdf_hotwords(PAGES)
check("전문용어 추출", "표집틀" in hot and ", " in hot and hot.endswith(".") and len(hot) <= 320,
      hot[:50])
hot = pdf_hotwords(["편향은 편향을 편향의 값으로 추정한다 Sampling frame"])
check("조사를 떼어 한 용어로 모음 · 서술어 제외",
      hot.startswith("편향,") and "값으로" in hot and "값으," not in hot
      and "추정한다" not in hot and "Sampling" in hot and "frame" not in hot, hot)

# 33) 대부분의 쪽에 되풀이되는 배너·꼬리말은 뺀다
pages = [["S A M P L E  U N I V E R S I T Y", f"{i}번째 쪽 제목", "공통 아님" if i < 3 else "본문"]
         for i in range(6)]
clean = drop_boilerplate(pages)
check("되풀이되는 배너 제거", all("S A M P L E" not in " ".join(p) for p in clean)
      and clean[0][0] == "0번째 쪽 제목", str(clean[0]))
check("절반만 나오는 줄은 유지", sum("공통 아님" in p for p in clean) == 3)
check("쪽이 적으면 건드리지 않음", drop_boilerplate(pages[:3]) == pages[:3])

# 33-2) 이름에 과목이 없는 LMS 자료(주차만 같은 후보)는 화면과 맞춰 본 뒤 채택한다
M = T.Material
book = [f"표본추출 {k}장\n모집단 표집틀 추정값 비표본오차 무응답 {k}번째 내용 설명" for k in range(1, 6)]
book[1] = "좋은 설문의 조건\n질문을 짧게 쓴다 유도 질문을 피한다 응답 선택지를 겹치지 않게"
sheet = ["실습지 층화추출 연습 문제 층별 표본 크기 배분 계산하기"]
other = ["재무제표의 구성 대차대조표 손익계산서 현금흐름표 자본변동표 주석",
         "유동비율 부채비율 자기자본이익률 총자산회전율 계산과 해석"]
loaded = [(M("묶음.zip/7주차교재.pdf", "7주차교재", 7, False, "묶음.zip", None), book),
          (M("묶음.zip/7실습지.pdf", "7실습지", 7, False, "묶음.zip", None), sheet),
          (M("다른과목 7주차.pdf", "다른과목 7주차", 7, False, "폴더", None), other),
          (M("표본통계학 보충.pdf", "표본통계학 보충", None, True, "폴더", None), ["보충 자료 쪽"])]
seen = [(10.0, ["표본추출 1장 모집단 표집틀 추정값 비표본오차"]),
        (60.0, ["좋은 설문의 조건 질문을 짧게 쓴다 유도 질문"]),
        (90.0, ["실습지 층화추출 연습 문제 층별 표본 크기 배분"]),
        (120.0, ["표본추출 3장 모집단 표집틀 추정값 무응답"])]
picked = [m.name for m, _t in T.pick_materials(seen, loaded)]
check("주차만 같은 자료는 화면과 맞을 때만 채택",
      picked == ["묶음.zip/7주차교재.pdf", "묶음.zip/7실습지.pdf", "표본통계학 보충.pdf"], str(picked))
check("화면을 못 읽었으면 이름에 강의가 적힌 자료만",
      [m.name for m, _t in T.pick_materials([], loaded)] == ["표본통계학 보충.pdf"])
info = types.SimpleNamespace(flag_bits=0, filename="7주차교재.pdf".encode("cp949").decode("cp437"))
check("옛 압축의 한글 이름을 풀어 읽음", T.zip_member_name(info) == "7주차교재.pdf",
      T.zip_member_name(info))
_pages, _labels, _o = T.material_pages(loaded[:2])
check("자료가 여럿이면 쪽 이름에 자료 이름", _labels[0] == "7주차교재 1쪽" and _labels[-1] == "7실습지 1쪽",
      str(_labels[:1] + _labels[-1:]))

# 34) 차례 제목은 배너·절번호·잡음을 건너뛰고 쓸 만한 줄을 고른다
_p1 = ["S A M P L E U N I V E R S I T Y", "담당교수ㅣ 가 상 인", "7주차 2교시", "표본통계학"]
check("배너·낱자 줄을 건너뛰고 쓸 만한 줄을 고름",
      slide_title(_p1) == "7주차 2교시", str(slide_title(_p1)))
check("단독 절번호를 건너뜀",
      slide_title(["01", "표본설계의 원칙"]) == "표본설계의 원칙")
check("잡음뿐이면 제목 없음", slide_title(["~ SS"]) is None and slide_title(["| Sy \\"]) is None)
check("정상 제목은 그대로", slide_title(["01 표본설계", "본문"]) == "01 표본설계")
check("영문 제목도 인정", slide_title(["Sampling in Practice"]) == "Sampling in Practice")
check("읽다 만 글자를 거름",
      slide_title(["ㅅ 트 변 그 즈다"]) is None
      and slide_title(["it t t t t t t t t tout"]) is None
      and slide_title(["\\(601ㅁ41[ㅇ표본의 크기 결정 방법"]) is None
      and slide_title(["거| 제!"]) is None
      and slide_title(["HItI ALO! 7 AO A"]) is None)
check("정상 한글 제목은 통과",
      slide_title(["추정량의 유형"]) == "추정량의 유형"
      and slide_title(["표본설계의 단계"]) == "표본설계의 단계"
      and slide_title(["학습 목표"]) == "학습 목표")
check("괄호·쉼표가 있어도 통과",
      slide_title(["CEO의 역할(기획, 실행, 평가)"]) == "CEO의 역할(기획, 실행, 평가)",
      str(slide_title(["CEO의 역할(기획, 실행, 평가)"])))

# 35) 자료 이름에서 주차를 읽어, 다른 주차 자료가 붙는 사고를 막는다
check("자료 이름에서 주차 파악",
      stem_week("7주차교재-표본통계학") == 7 and stem_week("7표본설계-표본통계학") == 7
      and stem_week("표본통계학") is None, str(stem_week("7주차교재-표본통계학")))

# 36) 파일 이름에서 과목·주차·교시를 읽는다
check("과목/주차/교시 파악",
      parse_course("재무관리 12-1강") == ("재무관리", 12, 1)
      and parse_course("DATA SCIENCE 9-3") == ("DATA SCIENCE", 9, 3)
      and parse_course("특강") == ("특강", None, None))

print("\n산출물 지문")

# 37) 본문이 같으면 같은 지문, 마커가 달라도 무관하다
body = "# 강의\n\n**[00:00:00]** 안녕하세요.\n"
h1 = md_body_hash(body + MARKER + ' {"a": 1} -->\n')
h2 = md_body_hash(body + MARKER + ' {"a": 2} -->\n')
check("마커는 지문에 영향 없음", h1 == h2)
check("본문이 바뀌면 지문도 바뀜",
      md_body_hash(body + "사용자 메모\n" + MARKER + " {} -->\n") != h1)

print("\n입력 고르기 · 산출물 생성")

with tempfile.TemporaryDirectory(prefix="mdcheck_", dir=T.tmp_root()) as td:
    td = Path(td)
    T.IN_DIR, T.OUT_DIR, T.PDF_DIR = td / "in", td / "out", td / "pdf"
    for d in (T.IN_DIR, T.OUT_DIR, T.PDF_DIR):
        d.mkdir()
    cfg = {"model": "large-v3-turbo", "language": "auto", "beam_size": 1,
           "슬라이드_읽기": True, "ocr_언어": "자동"}

    # 38) 끌어다 놓은 파일만 처리한다 — 입력 폴더의 다른 파일까지 몇 시간짜리 작업을 벌이지 않게
    (T.IN_DIR / "재무관리 1-1강.mp4").write_bytes(b"a")
    (T.IN_DIR / "재무관리 1-2강.mp4").write_bytes(b"b")
    outside = td / "밖"
    outside.mkdir()
    (outside / "재무관리 1-3강.mp4").write_bytes(b"c")
    (outside / "재무관리 1-1강.mp4").write_bytes(b"another file")
    chosen = T.stage_dropped([outside / "재무관리 1-3강.mp4", outside / "재무관리 1-1강.mp4"])
    check("이름만 같은 다른 파일은 대신 처리하지 않음",
          [p.name for p in chosen] == ["재무관리 1-3강.mp4"], str([p.name for p in chosen]))
    targets = T.plan_targets(cfg, set(chosen))[0]
    check("끌어다 놓은 것만 대상", [f.name for f, _ in targets] == ["재무관리 1-3강.mp4"],
          str([f.name for f, _ in targets]))
    check("인자 없이 실행하면 폴더 전체", len(T.plan_targets(cfg)[0]) == 3)
    T.unstage(T._STAGED)
    check("끌어다 놓기로 만든 이름은 치움", not (T.IN_DIR / "재무관리 1-3강.mp4").exists()
          and (outside / "재무관리 1-3강.mp4").exists())

    out = td / "재무관리 12-1강.md"
    src = td / "재무관리 12-1강.mp4"
    src.write_bytes(b"x" * 100)
    info = types.SimpleNamespace(duration=2828.0, language="ko", language_probability=1.0)
    paras = [(10.0, "오늘은 자본예산을 보겠습니다.", False),
             (1180.0, "이 도표를 보시면 현금흐름이 순환합니다.", False),
             (1500.0, "안녕하세요 가상은행의 김가상입니다", False),
             (1505.0, "오늘은 현장 경험을 말씀드립니다", False),
             (1512.0, "영상에서 보셨듯이 현장이 중요합니다.", False),
             (2000.0, "이상한 소리가 계속됩니다.", True)]
    screens = [(1171.0, ["자본예산의 절차", "투자안 → 현금흐름 → 할인"], "1쪽", "슬라이드"),
               (1495.0, ["안녕하세요 가상은행의 김가상입니다"], None, "자막"),
               (1502.0, ["오늘은 현장 경험을", "안녕하세요 가상은행의 김가상입니다"], None, "자막"),
               (1510.0, ["현장의 교훈", "표본은 매년 새로"], None, "슬라이드")]
    T.write_markdown(out, src, info, paras, cfg, screens, "재무관리 12주차.pdf",
                     repaired=[(300.0, 320.0)], lost=[(2400.0, 2412.0)])
    t = out.read_text(encoding="utf-8")

    check("머리말(frontmatter) 기록", t.startswith("---\n과목: 재무관리\n주차: 12\n교시: 1"))
    check("강의자료·모델 기록", "강의자료: 재무관리 12주차.pdf" in t and "모델: large-v3-turbo" in t)
    check("화면 차례 생성", "## 화면 차례" in t and "1쪽 자본예산의 절차" in t)
    check("슬라이드에 쪽번호와 유지 구간",
          "🖵 슬라이드 1쪽 [00:19:31 – 00:24:55]" in t,
          str([l for l in t.splitlines() if "🖵" in l][:1]))
    check("잇따른 자막은 한 블록으로, 줄은 한 번씩",
          t.count("📺 영상 자막") == 1 and "[00:24:55 – 00:25:10]" in t
          and t.count("> 안녕하세요 가상은행의 김가상입니다") == 1)
    check("영상 구간 발화에 📺", "**[00:25:00]** 📺 " in t and "**[00:25:05]** 📺 " in t,
          str([l for l in t.splitlines() if l.startswith("**[00:25:0")][:2]))
    check("영상이 끝난 뒤 교수의 말에는 📺 없음", "**[00:25:12]** 영상에서" in t)
    check("의심 문단에 ⚠", "**[00:33:20]** ⚠ 이상한" in t)
    check("옮기지 못한 말소리 구간 표시", "**[00:40:00]** ⚠ *(여기서 12초 동안" in t)
    check("머리말에 경고 요약", "재생된 영상 1곳" in t and "흔들린 문단 1개" in t
          and "잘못 옮겨진 말소리 1곳(20초)" in t and "옮기지 못한 구간 1곳" in t)

    mark = T.read_marker(out)
    check("마커 왕복", mark and mark.get("pdf") == "재무관리 12주차.pdf"
          and mark.get("sha") == md_body_hash(t), str(mark)[:80])

print(f"\n결과: {'전부 통과' if not fails else '실패 ' + ', '.join(fails)} ({passed}/{passed + len(fails)})")
sys.exit(1 if fails else 0)
