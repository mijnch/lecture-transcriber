# -*- coding: utf-8 -*-
"""엔진 로직 단위 검증 — 전사 없이 가짜 입력으로 즉시 확인한다."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from transcribe import (group_paragraphs, ends_sentence, looks_hallucinated,
                        script_mix_penalty, ocr_score, slide_key, merge_slides,
                        label_slides, video_spans, md_body_hash, merge_ocr_passes,
                        MARKER, PARA_HARD_CHARS, PARA_HARD_SEC)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
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
      not ends_sentence("소득세율은 3.") and not ends_sentence("항목 1.")
      and ends_sentence("문장입니다.") and ends_sentence("맞습니까?"))
p = group_paragraphs([(0, 22, "세율은 아주 길게 설명하면 이렇게 되는데 결론적으로 3."),
                      (22.5, 25, "5 퍼센트입니다.")])
check("소수를 문단 경계로 쪼개지 않음", len(p) == 1, f"문단 {len(p)}개")

print("\n신뢰할 수 없는 구간 표식")

# 10) 세그먼트의 의심 표식이 문단까지 전달된다
p = group_paragraphs([(0, 3, "정상 문장입니다.", False), (10, 13, "이상한 문장입니다.", True)])
check("의심 표식 전달", len(p) == 2 and p[0][2] is False and p[1][2] is True,
      f"{[x[2] for x in p]}")

# 11) 같은 문장이 되풀이되면 환각으로 표식한다
check("반복 환각 탐지",
      looks_hallucinated("I didn't do it. I didn't do it. I didn't do it. I didn't do it.")
      and not looks_hallucinated("첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다."))
p = group_paragraphs([(0, 20, "What is the new A. What is the new A. What is the new A.")])
check("반복 문단은 의심으로 표시", p and p[0][2] is True)

print("\n슬라이드 OCR")

# 12) 한글 줄에 낀 라틴 조각을 오인식으로 센다 (kor+eng 회귀 방지)
bad = "HAS 많이 하는 편이고 남의 SS 귀 기울여 들어준다"
good = "질문을 많이 하는 편이고 남의 말을 귀 기울여 들어준다"
check("글자종 섞임 벌점", script_mix_penalty(bad) == 2 and script_mix_penalty(good) == 0,
      f"섞임 {script_mix_penalty(bad)} / 정상 {script_mix_penalty(good)}")
check("더 잘 읽힌 판본을 고름", ocr_score([good]) > ocr_score([bad]),
      f"{ocr_score([good])} > {ocr_score([bad])}")
check("영문 줄은 벌점 없음", script_mix_penalty("An earthquake of 3.4 magnitude") == 0)

# 13) 한 글자 조각은 비교에서 뺀다
check("비교 낱말 정리", slide_key(["벤처캐피탈 의 A ㄱ 투자"]) == {"벤처캐피탈", "투자"},
      str(sorted(slide_key(["벤처캐피탈 의 A ㄱ 투자"]))))

# 14) 오인식이 심해 잘 안 겹치는 같은 슬라이드도 합친다
a = ["협상기반으로서의 인간관계", "개인갈등을 업무갈등으로 전환", "분노 성격차이 아집"]
b = ["Oy 협상기반으로서의 인간관계", "개인갈등을 업무갈등으로 전환", "분노 성격차이"]
merged = merge_slides([(21.0, a), (120.0, b)])
check("오인식된 같은 슬라이드 병합", len(merged) == 1, f"{len(merged)}장")
check("병합 시 처음 시각 유지", merged and merged[0][0] == 21.0)
check("병합 시 잘 읽힌 판본 유지", merged and merged[0][1] == a)

# 15) 다른 슬라이드는 합치지 않는다
other = ["벤처캐피탈 자금 조달 절차", "최초 접촉 사업개요서", "예비실사 투자 타당성"]
check("다른 슬라이드는 유지", len(merge_slides([(0.0, a), (60.0, other)])) == 2)

# 16) 세 장 건너 되풀이되는 판본도 합친다 (직전 한 장만 보던 문제)
c = ["협상기반으로서의 인간관계", "개인갈등을 업무갈등으로 전환", "분노 성격차이 아집 등"]
check("건너뛴 중복도 병합",
      len(merge_slides([(0.0, a), (30.0, other), (60.0, c)])) == 2,
      f"{len(merge_slides([(0.0, a), (30.0, other), (60.0, c)]))}장")

# 12-2) 한 화면 안에서 줄마다 잘 읽힌 쪽을 고른다 (한국어 줄은 kor, 영어 줄은 eng)
#        입력은 (윗변 좌표, 글자, 확신도) 이다.
kor_pass = [(100, "협상자의 스타일을 파악하여 대응한다", 92), (150, "11 [01115 [6561760.", 71)]
eng_pass = [(102, "SAS 스타일을 HAS 대응한다", 88), (148, "All rights reserved.", 94)]
got = merge_ocr_passes([kor_pass, eng_pass])
check("줄 단위로 좋은 판본 선택",
      got == ["협상자의 스타일을 파악하여 대응한다", "All rights reserved."], str(got))

# 12-3) 한 줄에 두 언어가 섞이면 한글을 살린다 (출처·고유명사가 정형 문구보다 중요)
mixed = merge_ocr_passes([[(100, "©2023. 두꺼비마을신문. 11 [01115 [(6560060.", 78)],
                          [(101, "©2023. -HH|OtSA=. All rights reserved.", 80)]])
check("섞인 줄에서는 한글을 보존", "두꺼비마을신문" in mixed[0], str(mixed))

# 12-4) 글자종이 같은 두 판본은 확신도로 가린다 (길이로는 깨진 쪽이 이길 수 있다)
eng_only = merge_ocr_passes([[(100, "A|| rights reserved by the puplisher", 62)],
                             [(100, "All rights reserved by the publisher", 93)]])
check("같은 글자종은 확신도로 판정",
      eng_only == ["All rights reserved by the publisher"], str(eng_only))

# 12-5) 한쪽에만 있는 줄은 버리지 않는다
only = merge_ocr_passes([[(100, "위쪽 줄입니다", 90)], [(400, "아래쪽 줄입니다", 90)]])
check("한쪽에만 잡힌 줄도 보존", only == ["위쪽 줄입니다", "아래쪽 줄입니다"], str(only))
check("빈 결과 처리", merge_ocr_passes([[], []]) == [])

print("\n영상 자막 구분")

# 17) 발화와 겹치는 짧은 화면 글자는 슬라이드가 아니라 자막
paras = [(30.0, "안녕하세요 카카오벤처스의 정신아입니다 아실만한 포트폴리오는", False)]
lab = label_slides([(29.0, ["안녕하세요 카카오벤처스의 정신아입니다"]),
                    (200.0, ["벤처캐피탈 자금 조달 절차", "최초 접촉 사업개요서",
                             "예비실사 및 투자 타당성 검토", "투자계약과 경영기술지원"])],
                   paras)
check("자막 판정", lab[0][2] == "자막", lab[0][2])
check("슬라이드 오판 없음", lab[1][2] == "슬라이드", lab[1][2])

# 18) 자막이 잇따르면 영상 재생 구간으로 묶는다
spans = video_spans([(10.0, [], "자막"), (14.0, [], "자막"), (18.0, [], "자막"),
                     (60.0, [], "슬라이드"), (90.0, [], "자막")])
check("영상 구간 검출", spans == [(10.0, 18.0)], str(spans))

print("\n강의자료 PDF 연동")

from transcribe import (align_slides_to_pdf, pdf_hotwords, parse_course,
                        page_key, name_tokens)

PAGES = ["벤처캐피탈의 업무 메커니즘\n투자자 → 출자 → Fund → 투자 → 벤처기업\n"
         "회수 ← 투자수익 ← 증권거래소 · 코스닥시장",
         "자금조달의 바람직한 자세\n1. 사업계획서를 충실히\n2. 군더더기를 줄일 것",
         "국내 벤처캐피탈 경영팀 평가항목\nCEO의 능력\nCEO의 인성(신뢰성, 성실성, 도덕성)"]

# 20) 깨진 화면 글자로도 올바른 쪽을 찾아내고, 본문은 PDF 원문으로 바뀐다
noisy = [(1171.0, ["투자자 ao <= A 벤처캐피탈", "주식인수 nay yoy 5", "코스닥시장 나아"])]
got = align_slides_to_pdf(noisy, PAGES)
check("깨진 OCR로도 쪽을 찾음", got[0][2] == "1쪽", f"쪽={got[0][2]}")
check("본문이 PDF 원문으로 교체됨", "Fund" in " ".join(got[0][1]), str(got[0][1])[:60])
check("시각은 화면에서 잡은 그대로", got[0][0] == 1171.0)

# 21) 같은 쪽이 잇따라 잡히면 한 번만 싣는다 (영상 재생 중 중복 폭증 방지)
rep = align_slides_to_pdf([(10.0, ["투자자 출자 Fund 회수"]),
                           (40.0, ["투자자 출자 Fund 투자수익"]),
                           (70.0, ["자금조달의 바람직한 자세 사업계획서"])], PAGES)
check("같은 쪽 연속 중복 제거", [r[2] for r in rep] == ["1쪽", "2쪽"],
      str([r[2] for r in rep]))

# 21-2) 자료가 둘이면 쪽 이름에 어느 자료인지 함께 적는다
two = align_slides_to_pdf([(10.0, ["투자자 출자 Fund 회수 벤처기업"])], PAGES,
                          ["6주차교재 1쪽", "6주차교재 2쪽", "실습지 1쪽"])
check("여러 자료의 쪽 이름 구분", two[0][2] == "6주차교재 1쪽", str(two[0][2]))

# 22) 어느 쪽과도 안 맞으면 화면에서 읽은 글자를 그대로 둔다
off = align_slides_to_pdf([(5.0, ["전혀 다른 화면 내용 광고 배너 문구"])], PAGES)
check("못 맞추면 OCR 글자 유지", off[0][2] is None and off[0][1][0].startswith("전혀"))

# 23) 전문용어를 뽑아 전사 힌트로 넘긴다
hot = pdf_hotwords(PAGES)
check("전문용어 추출", "벤처캐피탈" in hot and len(hot) <= 320, hot[:50])

# 22-2) 차례 제목은 배너·절번호·잡음을 건너뛰고 쓸 만한 줄을 고른다
from transcribe import slide_title
_p1 = ["H O N G I K U N I V E R S I T Y", "담당교수ㅣ 박 우 진", "6주차 1교시", "협상의기술"]
check("배너·낱자 줄을 건너뛰고 쓸 만한 줄을 고름",
      slide_title(_p1) == "6주차 1교시", str(slide_title(_p1)))
check("단독 절번호를 건너뜀",
      slide_title(["01", "협상스타일의 파악"]) == "협상스타일의 파악")
check("잡음뿐이면 제목 없음", slide_title(["~ SS"]) is None and slide_title(["| Sy \\"]) is None)
check("정상 제목은 그대로", slide_title(["01 협상스타일", "본문"]) == "01 협상스타일")
check("영문 제목도 인정", slide_title(["Newspapers in Korea"]) == "Newspapers in Korea")
check("읽다 만 글자를 거름",
      slide_title(["ㅅ 트 변 그 즈다"]) is None
      and slide_title(["it t t t t t t t t tout"]) is None
      and slide_title(["\\(601ㅁ41[ㅇ기업의 자금조달 방법"]) is None
      and slide_title(["거| 제!"]) is None
      and slide_title(["HItI ALO! 7 AO A"]) is None)
check("정상 한글 제목은 통과",
      slide_title(["엔젤투자자의 유형"]) == "엔젤투자자의 유형"
      and slide_title(["창업단계의 자금조달"]) == "창업단계의 자금조달"
      and slide_title(["학습 목표"]) == "학습 목표")
check("괄호·쉼표가 있어도 통과",
      slide_title(["CEO의 인성(신뢰성, 성실성, 도덕성)"]) == "CEO의 인성(신뢰성, 성실성, 도덕성)",
      str(slide_title(["CEO의 인성(신뢰성, 성실성, 도덕성)"])))

# 23-2) 자료 이름에서 주차를 읽어, 다른 주차 자료가 붙는 사고를 막는다
from transcribe import stem_week
check("자료 이름에서 주차 파악",
      stem_week("6주차교재-협상의기술") == 6 and stem_week("6협상스타일-협상의기술") == 6
      and stem_week("협상의기술") is None, str(stem_week("6주차교재-협상의기술")))

# 24) 파일 이름에서 과목·주차·교시를 읽는다
check("과목/주차/교시 파악",
      parse_course("창업과실용법률 12-1강") == ("창업과실용법률", 12, 1)
      and parse_course("MASS MEDIA SOCIETY 9-3") == ("MASS MEDIA SOCIETY", 9, 3)
      and parse_course("특강") == ("특강", None, None))

print("\n산출물 지문")

# 19) 본문이 같으면 같은 지문, 마커가 달라도 무관하다
body = "# 강의\n\n**[00:00:00]** 안녕하세요.\n"
h1 = md_body_hash(body + MARKER + ' {"a": 1} -->\n')
h2 = md_body_hash(body + MARKER + ' {"a": 2} -->\n')
check("마커는 지문에 영향 없음", h1 == h2)
check("본문이 바뀌면 지문도 바뀜",
      md_body_hash(body + "사용자 메모\n" + MARKER + " {} -->\n") != h1)

print("\n산출물 생성")

import tempfile, types
from pathlib import Path
from transcribe import write_markdown, read_marker

with tempfile.TemporaryDirectory(prefix="mdcheck_") as td:
    out = Path(td) / "창업과실용법률 12-1강.md"
    src = Path(td) / "창업과실용법률 12-1강.mp4"
    src.write_bytes(b"x" * 100)
    info = types.SimpleNamespace(duration=2828.0, language="ko", language_probability=1.0)
    paras = [(10.0, "오늘은 벤처캐피탈 자금조달을 보겠습니다.", False),
             (1180.0, "이 도표를 보시면 출자와 회수가 순환합니다.", False),
             (1500.0, "안녕하세요 카카오벤처스의 정신아입니다", False),
             (1505.0, "아실만한 포트폴리오는 카카오톡입니다", False),
             (2000.0, "이상한 소리가 계속됩니다.", True)]
    slides = [(1171.0, ["벤처캐피탈의 업무 메커니즘", "투자자 → 출자 → Fund"], "1쪽"),
              (1495.0, ["안녕하세요 카카오벤처스의 정신아입니다"], None),
              (1502.0, ["아실만한 포트폴리오는"], None)]
    cfg = {"model": "large-v3-turbo", "language": "auto", "beam_size": 1,
           "batch_size": 1, "정확도_우선": True, "슬라이드_읽기": True, "ocr_언어": "자동"}
    write_markdown(out, src, info, paras, cfg, 1200.0, 0.0, slides, "창업 12주차.pdf")
    t = out.read_text(encoding="utf-8")

    check("머리말(frontmatter) 기록", t.startswith("---\n과목: 창업과실용법률\n주차: 12\n교시: 1"))
    check("강의자료 이름 기록", "강의자료: 창업 12주차.pdf" in t)
    check("화면 차례 생성", "## 화면 차례" in t and "1쪽 벤처캐피탈의 업무 메커니즘" in t)
    check("슬라이드에 쪽번호와 유지 구간",
          "🖵 슬라이드 1쪽 [00:19:31 – 00:24:55]" in t,
          str([l for l in t.splitlines() if "🖵" in l][:1]))
    check("영상 자막 분리", "📺 영상 자막" in t)
    check("영상 구간 발화에 📺", "**[00:25:00]** 📺 " in t,
          str([l for l in t.splitlines() if l.startswith("**[00:25:00]")][:1]))
    check("의심 문단에 ⚠", "**[00:33:20]** ⚠ 이상한" in t)
    check("머리말에 경고 요약", "재생된 영상 1곳" in t and "흔들린 문단 1개" in t)

    mark = read_marker(out, src.name)
    check("마커 왕복", mark and mark.get("pdf") == "창업 12주차.pdf"
          and mark.get("sha") == md_body_hash(t), str(mark)[:80])

print(f"\n결과: {'전부 통과' if not fails else '실패 ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
