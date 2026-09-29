# -*- coding: utf-8 -*-
"""강의자료 PDF 연동 — 화면과 맞춰 볼 자료를 고르고(LMS 묶음 포함), 글자 2연쇄로 화면을
PDF 쪽에 정합하며, 전문용어 목록(hotwords)을 뽑는다. 자료 폴더를 뒤지는 부분은 transcribe 에 있다.

transcribe 에서 떼어 낸 모듈이다 — transcribe 가 이름을 모두 재수출하므로
transcribe.X 로 부르던 곳은 그대로 된다.
"""

import io
import re
from collections import Counter, namedtuple


# 강의자료 PDF 연동 — 화면에서 읽은 글자는 "몇 쪽인가"를 알아내는 열쇠로만 쓰고,
# 실을 내용은 PDF 원문을 그대로 가져온다. OCR 잡음이 사라지고 표·빈칸이 보존된다.
PDF_MATCH_RATIO = 0.35   # 화면 글자의 2연쇄 중 이만큼이 그 쪽에 있으면 같은 슬라이드
PDF_MIN_WORDS = 5        # 2연쇄가 이보다 적은 화면은 맞대볼 근거가 부족하다
PDF_SHORT_KEY = 12       # 이보다 짧은 화면은 근거가 적으므로 더 높은 일치를 요구한다
PDF_SHORT_RATIO = 0.60
PDF_FORWARD_BONUS = 0.04  # 방금 본 쪽 근처를 조금 더 쳐준다 (강의는 앞에서 뒤로)
PDF_FORWARD_SPAN = 12
HOTWORD_MAX = 320        # 전사에 넣어 줄 전문용어 문자열의 최대 길이
# 쪽(화면)의 이 비율 이상에 되풀이되는 줄은 머리글·꼬리말(학교 배너, 저작권 문구)이다
BOILERPLATE_RATIO = 0.7
BOILERPLATE_MIN = 6


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


# 강의자료 한 건. name 은 표시·지문용("묶음.zip/교재.pdf"), stem 은 쪽 이름용,
# named 는 파일 이름에 이 강의가 적혀 있는지(아니면 주차만 같은 후보),
# container 는 같은 묶음(zip·폴더)끼리 알아보는 열쇠, load 는 PDF 바이트를 돌려준다.
Material = namedtuple("Material", "name stem week named container load")


def zip_member_name(info) -> str:
    """zip 안 이름. UTF-8 표시가 없는 옛 압축은 한국어 Windows 기준(cp949)으로 풀어 본다."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


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


def material_pages(chosen):
    """고른 자료들의 쪽 글자와 쪽 이름을 한 줄로 이어 붙인다. 되풀이되는 배너는 뺀다."""
    pages, labels, owner = [], [], []
    many = len(chosen) > 1
    stems = Counter(m.stem for m, _t in chosen)
    for k, (m, texts) in enumerate(chosen):
        tag = m.stem if stems[m.stem] == 1 else m.name     # 이름이 같은 자료끼리는 묶음까지 적는다
        for i, text in enumerate(texts, 1):
            pages.append(text)
            labels.append(f"{tag} {i}쪽" if many else f"{i}쪽")
            owner.append(k)
    lines = drop_boilerplate([[ln.rstrip() for ln in t.splitlines() if ln.strip()] for t in pages])
    return ["\n".join(x) for x in lines], labels, owner


def pick_materials(slides, loaded):
    """후보 자료 가운데 이 강의 화면과 **실제로 맞는** 것만 남긴다.

    이름에 강의가 적힌 자료는 그대로 쓴다. 주차만 같은 후보는 화면 2장 이상이 그
    자료의 쪽과 맞아야 채택한다 — 다른 과목의 같은 주차 자료가 섞여 있어도 걸러진다.
    쪽이 3쪽 이하인 작은 자료(실습지)는, 같은 묶음의 다른 자료가 채택됐으면 1장으로 족하다.
    """
    if not loaded or not slides:
        return [x for x in loaded if x[0].named]
    pages, labels, owner = material_pages(loaded)
    who = dict(zip(labels, owner))
    hits = Counter(who[a[2]] for a in align_slides_to_pdf(slides, pages, labels) if a[2])
    keep = {k for k, (m, _t) in enumerate(loaded) if m.named or hits[k] >= 2}
    kept_boxes = {loaded[k][0].container for k in keep}
    keep |= {k for k, (m, t) in enumerate(loaded)
             if hits[k] >= 1 and len(t) <= 3 and m.container in kept_boxes}
    return [x for k, x in enumerate(loaded) if k in keep]


def read_pdf_pages(data):
    """PDF(경로 또는 바이트)의 쪽별 글자. 읽지 못하면 빈 목록(부르는 쪽에서 알린다)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        pages = []
        for page in PdfReader(io.BytesIO(data) if isinstance(data, bytes) else str(data)).pages:
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
