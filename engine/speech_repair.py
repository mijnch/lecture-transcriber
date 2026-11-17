# -*- coding: utf-8 -*-
"""음성 전사 보조 — 말소리는 있는데 낱말이 없는 구간과 흔들린(⚠) 구간을 찾아
언어를 새로 정해 다시 읽고, 낱말 시각과 문장 끝으로 구절을 나눈다.

transcribe 에서 떼어 낸 모듈이다 — transcribe 가 이름을 모두 재수출하므로
transcribe.X 로 부르던 곳은 그대로 된다.
"""

import re
import sys
from collections import Counter


PHRASE_GAP_SEC = 1.0     # 낱말 사이가 이만큼 비면 구절을 나눈다
# '다.' '요.' '까?' 는 '.' '?' 에 이미 포함되므로 두지 않는다
SENTENCE_END = ('.', '!', '?', '"', "'", '”', '…')

# 음성 전사 — 순차 경로(온도 폴백 + 품질 게이트)의 인자. 실측으로 정했다.
VAD_PARAMS = {"min_silence_duration_ms": 500}
QUALITY = dict(condition_on_previous_text=False, compression_ratio_threshold=2.4,
               log_prob_threshold=-1.0, no_speech_threshold=0.6)

# 신뢰할 수 없는 전사 구간 판정 — 이 구간은 산출물에 표식을 남긴다
SUSPECT_LOGPROB = -0.9   # 평균 확률이 이보다 낮으면 인식이 흔들린 것
SUSPECT_NO_SPEECH = 0.6  # 말이 아닐 확률이 이보다 높은데 글이 나왔으면 의심

# 빠진 말소리 복구 — 파일 전체를 한 언어로 읽으면 다른 언어로 말한 대목이 오류 없이
# 통째로 사라진다(한국어 강의 속 영어 설명 22초가 한 문장만 남은 것을 실측). 말소리는
# 있는데 낱말이 없는 구간을 찾아 그 구간만 언어를 새로 정해 다시 읽는다.
GAP_MIN_SEC = 3.0        # 이보다 짧은 빈틈은 숨·기침이다
GAP_JOIN_SEC = 2.0       # 이만큼 가까운 빈틈은 한 번에 다시 읽는다 (한 번에 ~10초 든다)
GAP_PAD_SEC = 0.3
REPAIR_MAX = 30          # 파일당 다시 읽는 구간 수의 상한
REPAIR_MIN_CPS = 3.0     # 초당 글자가 이보다 적으면 복구가 아니라 환각이다
LOST_MIN_SEC = 8.0       # 복구하지 못한 빈틈이 이보다 길면 산출물에 알린다
RELANG_MIN_PROB = 0.7    # ⚠ 구간을 다른 언어로 갈아끼우려면 그 언어라고 이만큼 확신해야 한다
HALLUCINATIONS = ("시청해주셔서 감사합니다", "시청해 주셔서 감사합니다", "구독과 좋아요",
                  "thank you for watching", "thanks for watching", "subtitles by")


def status(msg: str):
    """한 줄을 덮어쓰는 진행 표시."""
    sys.stdout.write("\r" + msg.ljust(60))
    sys.stdout.flush()


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def segment_words(seg, offset=0.0):
    return [(w.start + offset, w.end + offset, w.word) for w in (seg.words or [])]


def is_suspect(seg) -> bool:
    """인식이 흔들린 구간. 매끄러운 문장으로 된 환각은 사람도 LLM도 알아볼 수 없으므로
    여기서 잡아 표식을 남긴다."""
    return ((getattr(seg, "avg_logprob", None) or 0) < SUSPECT_LOGPROB
            or (getattr(seg, "no_speech_prob", None) or 0) > SUSPECT_NO_SPEECH)


def reliable_words(words):
    """세그먼트 앞머리의 외톨이 짧은 낱말을 뺀다.

    Whisper 세그먼트가 빈 구간을 끼고 시작하면 첫 낱말('이 표는'의 '이')의 시각이
    수십 초 앞으로 끌려간다. 그 낱말을 믿으면 빈틈이 가려져 복구가 덜 된다.
    """
    i = 0
    while (i + 1 < len(words) and len(words[i][2].strip()) < 3
           and words[i + 1][0] - words[i][1] >= PHRASE_GAP_SEC):
        i += 1
    return words[i:]


def find_gaps(speech, spans, min_len=GAP_MIN_SEC, join=GAP_JOIN_SEC, pad=GAP_PAD_SEC):
    """말소리 구간(speech) 가운데 낱말(spans)이 덮지 않은 곳. 가까운 것은 합친다.

    둘 다 (시작초, 끝초) 목록이다.
    """
    covered = []
    for s, e in sorted((s - pad, e + pad) for s, e in spans):
        if covered and s <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], e)
        else:
            covered.append([s, e])
    holes, j = [], 0
    for a, b in sorted(speech):
        while j < len(covered) and covered[j][1] <= a:
            j += 1
        cur, k = a, j
        while k < len(covered) and covered[k][0] < b:
            if covered[k][0] > cur:
                holes.append((cur, covered[k][0]))
            cur = max(cur, covered[k][1])
            k += 1
        if cur < b:
            holes.append((cur, b))
    merged = []
    for a, b in holes:
        if b - a < 0.5:
            continue
        if merged and a - merged[-1][1] <= join:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def repair_gaps(model, audio, segments, cfg, sr=16000):
    """말소리는 있는데 전사가 비어 있는 구간을 언어를 새로 정해 다시 읽는다.

    돌려주는 값은 (보탤 세그먼트들, 복구한 구간들, 복구하지 못한 긴 구간들).
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    speech = [(c["start"] / sr, c["end"] / sr)
              for c in get_speech_timestamps(audio, VadOptions(**VAD_PARAMS))]
    # 낱말 시각이 없는 세그먼트는 세그먼트 전체를 덮은 것으로 본다
    spans = [span for seg in segments
             for span in ([(w[0], w[1]) for w in reliable_words(seg[4])] or [(seg[0], seg[1])])]
    gaps = find_gaps(speech, spans)
    added, fixed, lost = [], [], []
    for n, (a, b) in enumerate(gaps):
        if n >= REPAIR_MAX:
            lost += [g for g in gaps[n:] if g[1] - g[0] >= LOST_MIN_SEC]
            break
        status(f"  빠진 말소리 다시 읽는 중... [{fmt_ts(a)}] {n + 1}/{len(gaps)}")
        lo = max(0.0, a - GAP_PAD_SEC)
        clip = audio[int(lo * sr):int((b + GAP_PAD_SEC) * sr)]
        try:
            segs, info = model.transcribe(clip, beam_size=cfg["beam_size"], vad_filter=True,
                                          vad_parameters=VAD_PARAMS, word_timestamps=True,
                                          **QUALITY)
            segs = [s for s in segs if not is_suspect(s)]
        except Exception:
            segs, info = [], None
        got = []
        for s in segs:
            # 이미 전사된 낱말과 겹치는 가장자리는 버린다 (중복 방지)
            words = [w for w in segment_words(s, lo) if a <= (w[0] + w[1]) / 2 <= b]
            text = "".join(w[2] for w in words).strip()
            if text:
                got.append((words[0][0], words[-1][1], text, False, words))
        body = re.sub(r"\s", "", "".join(g[2] for g in got))
        ok = (info is not None and info.language_probability >= 0.6
              and len(body) >= REPAIR_MIN_CPS * (b - a)
              and not any(h in body.lower().replace(" ", "")
                          for h in (x.replace(" ", "") for x in HALLUCINATIONS)))
        if ok:
            added += got
            fixed.append((a, b))
        elif b - a >= LOST_MIN_SEC:
            lost.append((a, b))
    if gaps:
        status("")
        sys.stdout.write("\r")
    return added, fixed, lost


def suspect_spans(segments, join=GAP_JOIN_SEC, min_len=GAP_MIN_SEC):
    """⚠ 세그먼트가 이어진 구간들 (가까우면 합친다)."""
    spans = []
    for s in segments:
        if not s[3]:
            continue
        if spans and s[0] - spans[-1][1] <= join:
            spans[-1][1] = max(spans[-1][1], s[1])
        else:
            spans.append([s[0], s[1]])
    return [(a, b) for a, b in spans if b - a >= min_len]


def reread_suspects(model, audio, segments, main_lang, cfg, sr=16000):
    """⚠ 구간을 언어를 새로 정해 다시 읽고, 다른 언어로 확신 있게 읽히면 갈아끼운다.

    파일 언어로 고정해 읽으면 다른 언어로 말한 대목이 빠지기도 하지만(repair_gaps),
    그 언어의 **그럴듯한 환각으로 채워지기도** 한다 — 실강의(영어 강의 속 한국어 뉴스
    영상)에서 한국어 말소리 자리에 그럴듯한 영어 헛문장이 9분간
    이어졌다. 빈틈이 없으니 복구 장치가 못 잡는다. ⚠ 는 붙어 있으므로 그 구간을 다시 읽는다.
    같은 언어로 나오면 진짜 음질 문제이므로 그대로 둔다(⚠ 유지).

    돌려주는 값은 (새 세그먼트 목록, 바꾼 구간 [(시작, 끝, 언어)]).
    """
    swapped = []
    for a, b in suspect_spans(segments)[:REPAIR_MAX]:
        status(f"  다른 언어로 말한 대목인지 다시 읽는 중... [{fmt_ts(a)}]")
        lo = max(0.0, a - GAP_PAD_SEC)
        clip = audio[int(lo * sr):int((b + GAP_PAD_SEC) * sr)]
        try:
            segs, info = model.transcribe(clip, beam_size=cfg["beam_size"], vad_filter=True,
                                          vad_parameters=VAD_PARAMS, word_timestamps=True,
                                          **QUALITY)
            segs = list(segs)
        except Exception:
            continue
        good = [s for s in segs if not is_suspect(s)]
        if (info.language == main_lang or info.language_probability < RELANG_MIN_PROB
                or not segs or len(good) < 0.7 * len(segs)):
            continue
        new = []
        for s in good:
            words = [w for w in segment_words(s, lo) if a - 1 <= (w[0] + w[1]) / 2 <= b + 1]
            text = "".join(w[2] for w in words).strip()
            if text and not any(h in text.lower().replace(" ", "")
                                for h in (x.replace(" ", "") for x in HALLUCINATIONS)):
                new.append((words[0][0], words[-1][1], text, False, words))
        kept = [s for s in segments if not (a <= s[0] <= b and s[3])]
        # 구간 안에 남은 확실한 세그먼트(교수의 짧은 말 등)와 겹치는 낱말은 버린다
        inside = [(s[0], s[1]) for s in kept if a <= s[0] <= b]
        new = [n for n in new if not any(x <= (n[0] + n[1]) / 2 <= y for x, y in inside)]
        if not new:
            continue
        segments = kept + new
        swapped.append((a, b, info.language))
    status("")
    sys.stdout.write("\r")
    return sorted(segments, key=lambda s: s[0]), swapped


def reread_video_suspects(model, audio, segments, spans, foreign, cfg, sr=16000):
    """다른 언어 영상 안에 남은 ⚠ 구간을 그 영상의 언어로 고정해 다시 읽는다.

    reread_suspects 는 언어를 새로 정하는데, 음악이 깔린 짧은 조각은 교수의 언어로 판별되어
    재독이 버려졌다 — 영어 강의 속 한국어 영상 3분이 영어 헛문장(⚠)으로 남았다(실강의).
    그 영상이 어느 언어인지는 이미 안다(다른 언어로 읽힌 말이 있는 영상 구간). 그 안의 ⚠
    구간만 그 언어로 읽는다 — 교수가 영상 중간에 한 말은 대개 또렷해 ⚠ 가 아니다.
    돌려주는 값은 (새 세그먼트 목록, 바꾼 구간 [(시작, 끝, 언어)]).
    """
    done = []
    for a, b in spans:
        langs = Counter(lang for x, y, lang in foreign if x < b and a < y)
        if not langs:
            continue
        lang = langs.most_common(1)[0][0]
        for x, y in suspect_spans([s for s in segments if a <= s[0] < b]):
            if len(done) >= REPAIR_MAX:
                break
            status(f"  영상 속 흔들린 말을 영상의 언어({lang})로 다시 읽는 중... [{fmt_ts(x)}]")
            lo = max(0.0, x - GAP_PAD_SEC)
            clip = audio[int(lo * sr):int((y + GAP_PAD_SEC) * sr)]
            try:
                segs = list(model.transcribe(clip, language=lang, beam_size=cfg["beam_size"],
                                             vad_filter=True, vad_parameters=VAD_PARAMS,
                                             word_timestamps=True, **QUALITY)[0])
            except Exception:
                continue
            good = [s for s in segs if not is_suspect(s)]
            if not segs or len(good) < 0.7 * len(segs):
                continue
            new = []
            for s in good:
                words = [w for w in segment_words(s, lo) if x - 1 <= (w[0] + w[1]) / 2 <= y + 1]
                text = "".join(w[2] for w in words).strip()
                if text and not any(h in text.lower().replace(" ", "")
                                    for h in (z.replace(" ", "") for z in HALLUCINATIONS)):
                    new.append((words[0][0], words[-1][1], text, False, words))
            if not new:
                continue
            segments = [s for s in segments if not (x <= s[0] <= y and s[3])] + new
            done.append((x, y, lang))
    if done:
        status("")
        sys.stdout.write("\r")
    return sorted(segments, key=lambda s: s[0]), done


def split_phrases(segments, cuts=()):
    """세그먼트를 낱말 시각으로 짧은 구절로 나눈다. (시작, 끝, 글, 의심) 목록.

    전문용어 힌트를 넣으면 Whisper 세그먼트가 24초짜리로 뭉개진다(실측: 24개 → 7개).
    구절 단위로 나눠야 슬라이드가 바뀐 시점에서 문단을 끊을 수 있다.
    cuts(재생 영상이 시작·끝난 시각)에서도 끊는다 — 교수가 말을 마치자마자 영상이 시작되면
    쉼 없이 한 구절로 묶여, 영상의 첫마디가 교수의 말로 실리거나 영상 뒤 교수의 말이 📺 로
    실렸다(실강의). 글자가 하나도 없는 구절('... ...')은 버린다.
    """
    cuts = sorted(cuts)
    out = []

    def emit(s, e, text, bad):
        if any(ch.isalnum() for ch in text):
            out.append((s, e, text.strip(), bad))

    for seg in segments:
        s, e, text, bad = seg[0], seg[1], seg[2], seg[3]
        words = seg[4] if len(seg) > 4 else None
        if not words:
            if text.strip():
                emit(s, e, text, bad)
            continue
        cur, cs, ce = "", None, None
        for ws, we, w in words:
            crossed = cs is not None and any(cs < c <= ws for c in cuts)
            if cs is not None and not crossed and ws - ce >= PHRASE_GAP_SEC and len(cur.strip()) < 3:
                # 세그먼트 앞머리의 외톨이 낱말('이 표는'의 '이')은 시각이 부정확하다 —
                # 앞 문단에 붙지 않도록 뒤 낱말들과 한 구절로 묶고 그 시각을 쓴다
                cs = ws
            elif cs is not None and (crossed or ws - ce >= PHRASE_GAP_SEC or ends_sentence(cur)):
                emit(cs, ce, cur, bad)
                cur, cs = "", None
            if cs is None:
                cs = ws
            cur += w
            ce = we
        if cur.strip():
            emit(cs, ce, cur, bad)
    return out


def ends_sentence(text: str) -> bool:
    """문장이 끝났는지. '소득세율은 3.' 처럼 숫자 뒤 마침표는 끝이 아니다."""
    t = text.rstrip()
    return bool(t) and t.endswith(SENTENCE_END) and not re.search(r"\d\.$", t)
