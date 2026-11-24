"""Numbers in dubbed lines: what the original says (digits, Chinese numerals),
what the dub says (digits, Khmer digits, Khmer number words), whether every
number of the original is in its dub line with its exact value, a repair
when that is clear, and Khmer words for the voice.

Why: on a 1-hour Chinese drama the AI wrote Khmer numbers in words and got
about one in five wrong: 681 as ប្រាំរយប៉ែតសិបមួយ (581 — six is "five-one",
ប្រាំមួយ, and the "one" went missing), 692 as 962, 56 as 55, 250 with a
misspelled "fifty" the voice could not say, letters of other scripts inside
the words. The translation now copies digits, this module checks every line,
and the voice gets the number words from code (km_say_numbers).

    km_words(n)                  681 -> ប្រាំមួយរយប៉ែតសិបមួយ
    km_say_numbers(text)         digits in a Khmer line -> words, for the voice
    km_digits(text)              0-9 -> ០-៩ (not inside Latin codes such as H2)
    source_numbers(text, lang)   the numbers an original line says
    dub_numbers(text)            [Span(value, start, end, clean)] in a dub line
    check(source, dub, lang)     (missing values, dub numbers matching none)
    repair(source, dub, lang, target) -> the dub with the number put right,
                                 or None when it is not clear which one
    keeps(source, old, new, lang)  a rewrite loses no number the line had

No Qt imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ZWSP = "​"
_KM_DIGITS = "០១២៣៤៥៦៧៨៩"
_TO_ASCII = str.maketrans(_KM_DIGITS, "0123456789")
_TO_KM = str.maketrans("0123456789", _KM_DIGITS)

# ---------------------------------------------------------------------------
# Khmer number words
# ---------------------------------------------------------------------------

_KM_ONES = ["សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ",
            "ប្រាំបី", "ប្រាំបួន"]
_KM_TENS = ["", "ដប់", "ម្ភៃ", "សាមសិប", "សែសិប", "ហាសិប", "ហុកសិប", "ចិតសិប",
            "ប៉ែតសិប", "កៅសិប"]
_KM_UNITS = {"រយ": 100, "ពាន់": 1000, "ម៉ឺន": 10_000, "សែន": 100_000, "លាន": 1_000_000}
_KM_VALUE = {w: k for k, w in enumerate(_KM_ONES)}
_KM_VALUE.update({w: 10 * k for k, w in enumerate(_KM_TENS) if w})
_KM_VALUE.update(_KM_UNITS)
_KM_TOKEN = "|".join(sorted(map(re.escape, _KM_VALUE), key=len, reverse=True))
_KM_TOKEN_RX = re.compile(_KM_TOKEN)
# a run of number words, a zero-width space allowed between them; "រយៈ"
# (period, as in រយៈពេល) is no hundred
_KM_RUN = re.compile(f"(?:{_KM_TOKEN})(?:{ZWSP}?(?:{_KM_TOKEN}))*(?!ៈ)")
# what may follow a number in a Khmer line (a word boundary): classifiers and
# units. Anything else glued to number words ("ហាស៊ីន", a misspelled
# "fifty") means the words are broken: not repaired by code.
_KM_AFTER = ("ពិន្ទុ", "នាក់", "ថ្ងៃ", "ឆ្នាំ", "ខែ", "ម៉ោង", "នាទី", "វិនាទី", "ដង",
             "យ័ន", "ដុល្លារ", "រៀល", "ភាគរយ", "គីឡូ", "ម៉ែត្រ", "គ្រាប់", "ក្បាល",
             "ផ្ទះ", "ជាន់", "លើក", "សប្តាហ៍", "អាទិត្យ", "ប្រអប់", "សន្លឹក", "ច្បាប់",
             "កញ្ចប់", "ដុំ", "ចាន", "កែវ", "ដប", "ទៀត", "ហើយ", "ទេ", "ដែរ", "បាន",
             "ក្នុង", "នៅ", "និង", "ឬ", "ជា", "គឺ", "នេះ", "នោះ", "ហ្នឹង", "ណា")


def _km_letter(c: str) -> bool:
    """A Khmer letter, vowel or sign (part of a word), not Khmer punctuation
    (។ ៕ ៖ ៗ…) or a Khmer digit."""
    return "ក" <= c <= "៓" or c == "៝"


def km_words(n: int) -> str:
    """A whole number in Khmer words, as a dub voice should say it (the
    traditional ម៉ឺន / សែន below a million): 681 -> ប្រាំមួយរយប៉ែតសិបមួយ."""
    n = int(n)
    if n < 0:
        return "ដក" + km_words(-n)
    if n == 0:
        return _KM_ONES[0]
    out = []
    millions, rest = divmod(n, 1_000_000)
    if millions:
        out.append(km_words(millions) + "លាន")
    for unit, word in ((100_000, "សែន"), (10_000, "ម៉ឺន"), (1000, "ពាន់"), (100, "រយ")):
        q, rest = divmod(rest, unit)
        if q:
            out.append(_KM_ONES[q] + word)
    tens, ones = divmod(rest, 10)
    if tens:
        out.append(_KM_TENS[tens])
    if ones:
        out.append(_KM_ONES[ones])
    return "".join(out)


def _km_run_value(run: str) -> int:
    """The value of a run of Khmer number words (ZWSP allowed between)."""
    total = section = cur = 0
    for tok in _KM_TOKEN_RX.findall(run.replace(ZWSP, "")):
        v = _KM_VALUE[tok]
        if tok == "លាន":
            total += ((section + cur) or 1) * 1_000_000
            section = cur = 0
        elif tok in _KM_UNITS:
            if section and section < v:          # ពីររយហាសិបពាន់ = 250 thousand
                section = (section + cur) * v
            else:
                section += (cur or 1) * v
            cur = 0
        else:
            cur += v
    return total + section + cur


# ---------------------------------------------------------------------------
# numbers in a dub line
# ---------------------------------------------------------------------------

@dataclass
class Span:
    value: float          # int when whole
    start: int
    end: int
    clean: bool = True    # digits, or number words with a word boundary after them
    words: bool = False


_DIGITS_RX = re.compile(r"[0-9០-៩]+(?:[,.][0-9០-៩]+)*")
_UNIT_AFTER_RX = re.compile(f"[ {ZWSP}]?(រយ|ពាន់|ម៉ឺន|សែន|លាន)")


def _num(text: str) -> float:
    t = text.translate(_TO_ASCII)
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", t):           # 1,000,000
        t = t.replace(",", "")
    t = t.replace(",", ".")
    try:
        v = float(t) if t.count(".") <= 1 else float(t.replace(".", ""))
    except ValueError:
        return float("nan")
    return int(v) if v == int(v) else v


def dub_numbers(text: str) -> list[Span]:
    """The numbers a dub line says: digits (0-9, ០-៩; "100 ម៉ឺន" is a
    million) and runs of Khmer number words with at least one digit word
    (a lone unit word such as សែន is also "very")."""
    text = text or ""
    spans: list[Span] = []
    for m in _DIGITS_RX.finditer(text):
        v, end = _num(m.group(0)), m.end()
        u = _UNIT_AFTER_RX.match(text, end)
        if u and v == v:
            v, end = v * _KM_UNITS[u.group(1)], u.end()
            v = int(v) if v == int(v) else v
        spans.append(Span(v, m.start(), end))
    for m in _KM_RUN.finditer(text):
        if any(s.start <= m.start() < s.end for s in spans):
            continue
        # one run can hold several numbers: two digit words in a row never
        # belong to one number (មួយពីរ "one or two", ហុកសិបចិតសិប "60 or 70")
        groups: list[list[re.Match]] = []
        prev = ""
        for t in _KM_TOKEN_RX.finditer(text, m.start(), m.end()):
            tok = t.group(0)
            kind = ("unit" if tok in _KM_UNITS else
                    "tens" if _KM_VALUE[tok] >= 10 else "ones")
            if not groups or (kind == "ones" and prev == "ones") or \
                    (kind == "tens" and prev in ("ones", "tens")):
                groups.append([])
            groups[-1].append(t)
            prev = kind
        for k, g in enumerate(groups):
            if all(t.group(0) in _KM_UNITS for t in g):
                continue                                 # សែន alone is also "very"
            start, end = g[0].start(), g[-1].end()
            after = text[end:] if k == len(groups) - 1 else ""
            # the words end at a boundary: a space, punctuation (។ too), a
            # digit, or a word that often follows a number
            clean = (not after or not _km_letter(after[0]) or after.startswith(_KM_AFTER))
            spans.append(Span(_km_run_value(text[start:end]), start, end, clean, words=True))
    spans.sort(key=lambda s: s.start)
    return spans


# ---------------------------------------------------------------------------
# numbers in the original
# ---------------------------------------------------------------------------

_ZH_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
_ZH_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10_000, "亿": 100_000_000}
_ZH_RUN = re.compile("[零〇一二两三四五六七八九十百千万亿]+")
# a counted thing after a single digit (六天, 两个, 三点, 五块): a number
_ZH_MEASURE = set("个人天年日号点分秒次岁块元位名本张条只件种遍倍层楼周斤米里杯瓶碗"
                  "道题门节课页届期集口份间部辆台场")    # not 句: 说两句 = "say a few words"
# words made of numerals that are no numbers: in case, one by one, "idiot"
_ZH_WORDS = {"万一", "一一", "二百五"}
_ZH_SCALE_AFTER = {"万": 10_000, "亿": 100_000_000, "千": 1000, "百": 100}


def _zh_value(run: str) -> int:
    if all(c in _ZH_DIGIT for c in run):                  # 二零二六 = 2026
        return int("".join(str(_ZH_DIGIT[c]) for c in run))
    total = section = num = 0
    last_unit = 0
    for c in run:
        if c in _ZH_DIGIT:
            num = _ZH_DIGIT[c]
        elif c in ("十", "百", "千"):
            section += (num or 1) * _ZH_UNIT[c]
            num = 0
            last_unit = _ZH_UNIT[c]
        else:                                           # 万 / 亿
            total += ((section + num) or 1) * _ZH_UNIT[c]
            section = num = 0
            last_unit = _ZH_UNIT[c]
    if num and last_unit >= 100 and len(run) >= 2 and run[-2] in _ZH_UNIT:
        num *= last_unit // 10                          # 三千八 = 3800, 两万五 = 25000
    return total + section + num


def _zh_is_number(run: str, before: str, after: str) -> bool:
    if run in _ZH_WORDS and not (after[:1] in _ZH_MEASURE):
        return False
    digits = [c for c in run if c in _ZH_DIGIT]
    if not digits:
        return False                                    # 十分 "very", 千万, 百姓
    if all(c in _ZH_DIGIT for c in run):
        if len(run) >= 4:
            return True                                 # 二零二六
        if len(run) == 1:
            # one digit: a number when something is counted (六天) or it is
            # ranked (第二); 一 is "a / one" in too many words; 一天两天 is
            # "a day or two"
            if len(before) >= 2 and before[-2] in _ZH_DIGIT and before[-1] == after[:1]:
                return False
            return run != "一" and (after[:1] in _ZH_MEASURE or before.endswith("第"))
        return False                                    # 三四 "three or four"
    # 三四十, 六七十: "thirty or forty" — no exact number
    for a, b in zip(run, run[1:]):
        if a in _ZH_DIGIT and b in _ZH_DIGIT and "零" not in (a, b) and "〇" not in (a, b):
            return False
    return True


_SRC_DIGITS_RX = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def source_numbers(text: str, lang: str = "") -> list:
    """The numbers an original line says, in order: digits ("100万" is a
    million; a long run of digits, 7 or more, is kept as a digit string),
    and for Chinese / Japanese the numerals that are exact numbers."""
    text = text or ""
    found: list[tuple[int, object]] = []
    for m in _SRC_DIGITS_RX.finditer(text):
        raw = m.group(0)
        if len(raw) >= 7 and raw.isdigit():
            found.append((m.start(), raw))              # a code, a list read as one
            continue
        v = _num(raw)
        if v != v:
            continue
        after = text[m.end():m.end() + 1]
        if after == "月" and isinstance(v, int) and 1 <= v <= 12:
            continue                                    # a month has a name in Khmer
        v = v * _ZH_SCALE_AFTER.get(after, 1)
        found.append((m.start(), int(v) if v == int(v) else v))
    base = (lang or "").split("-")[0].lower()
    if base in ("", "zh", "ja", "yue"):
        for m in _ZH_RUN.finditer(text):
            run = m.group(0)
            before, after = text[:m.start()], text[m.end():]
            if after[:1] == "月" and len(run) <= 2:
                continue                                # 三月: March
            if _zh_is_number(run, before, after):
                found.append((m.start(), _zh_value(run)))
    found.sort(key=lambda t: t[0])
    return [v for _p, v in found]


# ---------------------------------------------------------------------------
# checking and repairing a dub line
# ---------------------------------------------------------------------------

def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def check(source: str, dub: str, lang: str = "") -> tuple[list, list[Span]]:
    """(numbers of the original missing from the dub, dub numbers that match
    none of the original's). Every number of the original must be said."""
    want = source_numbers(source, lang)
    spans = dub_numbers(dub)
    used = [False] * len(spans)
    all_digits = "".join(c for c in (dub or "").translate(_TO_ASCII) if c.isdigit())
    missing = []
    for v in want:
        if isinstance(v, str):
            if v not in all_digits:
                missing.append(v)
            continue
        if isinstance(v, int) and v >= 100_000 and str(v) in all_digits:
            continue        # "692692!" on screen: the dub says ៦៩២ ៦៩២ — the same digits
        k = next((j for j, s in enumerate(spans) if not used[j] and _same(s.value, v)), None)
        if k is None:
            missing.append(v)
        else:
            used[k] = True
    return missing, [s for j, s in enumerate(spans) if not used[j]]


def digits_for(value, target: str = "km") -> str:
    """A number as digits for a line in `target` (Khmer digits for Khmer)."""
    text = value if isinstance(value, str) else (
        str(int(value)) if _same(value, int(value)) else repr(float(value)))
    return text.translate(_TO_KM) if (target or "").split("-")[0] == "km" else text


def repair(source: str, dub: str, lang: str = "", target: str = "km") -> str | None:
    """The dub with its one wrong number put right, when that is clear: one
    number of the original is missing, and the dub has exactly one other
    number of about its size (a tenth to ten times) written cleanly — it is
    written as digits instead. None when there is nothing to repair or it
    is not clear (the AI is asked again)."""
    missing, extra = check(source, dub, lang)
    if len(missing) != 1 or isinstance(missing[0], str):
        return None
    want = float(missing[0])
    near = [s for s in extra if s.clean and want and s.value == s.value
            and 0.1 <= float(s.value) / want <= 10]
    if len(near) != 1:
        return None
    s = near[0]
    fixed = dub[:s.start] + digits_for(missing[0], target) + dub[s.end:]
    return fixed if not check(source, fixed, lang)[0] else None


def keeps(source: str, old: str, new: str, lang: str = "") -> bool:
    """True when `new` (a rewording of `old`) loses or changes no number of
    the original that `old` says."""
    before = check(source, old, lang)[0]
    after = check(source, new, lang)[0]
    rest = list(before)
    for v in after:
        k = next((j for j, w in enumerate(rest) if _same(w, v) or w == v), None)
        if k is None:
            return False
        rest.pop(k)
    return True


# ---------------------------------------------------------------------------
# the voice and the subtitles
# ---------------------------------------------------------------------------

def _say_digit_by_digit(digits: str) -> str:
    return " ".join(_KM_ONES[int(d)] for d in digits)


def _say(match: re.Match) -> str:
    raw = match.group(0).translate(_TO_ASCII)
    if ":" in raw:                                       # 8:30
        h, m = raw.split(":", 1)
        return km_words(int(h)) + "ម៉ោង" + (km_words(int(m)) + "នាទី" if int(m) else "")
    pct = raw.endswith("%")
    raw = raw.rstrip("%")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
        raw = raw.replace(",", "")
    if raw.isdigit() and (len(raw) >= 8 or (len(raw) >= 2 and raw[0] == "0")):
        words = _say_digit_by_digit(raw)                 # a phone number, a code
    elif re.fullmatch(r"\d+[.,]\d+", raw):
        whole, frac = re.split(r"[.,]", raw)
        words = km_words(int(whole)) + "ចុច" + "".join(_KM_ONES[int(d)] for d in frac)
    elif raw.isdigit():
        words = km_words(int(raw))
    else:
        return match.group(0)
    return words + ("ភាគរយ" if pct else "")


_SAY_RX = re.compile(r"[0-9០-៩]{1,2}:[0-9០-៩]{2}(?![0-9០-៩])|[0-9០-៩]+(?:[,.][0-9០-៩]+)*%?")


def km_say_numbers(text: str) -> str:
    """The numbers of a Khmer line as words, for the voice: 681 or ៦៨១ ->
    ប្រាំមួយរយប៉ែតសិបមួយ; 3.5 -> បីចុចប្រាំ; 50% -> ហាសិបភាគរយ; 8:30 ->
    ប្រាំបីម៉ោងសាមសិបនាទី; a long number or one starting with 0 digit by
    digit. Subtitles keep the digits."""
    return _SAY_RX.sub(_say, text or "")


def km_digits(text: str) -> str:
    """0-9 written as Khmer digits ០-៩, except in Latin codes (H2, MP4)."""
    return re.sub(r"(?<![A-Za-z])[0-9]+(?:[.,][0-9]+)*(?![A-Za-z])",
                  lambda m: m.group(0).translate(_TO_KM), text or "")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    bad: list[str] = []

    def expect(name: str, got, want) -> None:
        ok = got == want
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"   got {got!r} want {want!r}"))
        if not ok:
            bad.append(name)

    expect("681 in words", km_words(681), "ប្រាំមួយរយប៉ែតសិបមួយ")
    expect("25000 in words", km_words(25000), "ពីរម៉ឺនប្រាំពាន់")
    expect("2026 in words", km_words(2026), "ពីរពាន់ម្ភៃប្រាំមួយ")
    expect("a billion in words", km_words(1_000_000_000), "មួយពាន់លាន")
    expect("words read back, 0 to 2,000,000", [n for n in list(range(0, 3000))
                                               + [9999, 10_001, 25_000, 99_999, 100_000,
                                                  250_000, 999_999, 1_000_000, 1_500_000,
                                                  2_470_768, 7_500]
                                               if _km_run_value(km_words(n)) != n], [])
    expect("the words the AI wrote", [_km_run_value(w) for w in (
        "ប្រាំរយប៉ែតសិបមួយ", "ពីររយហាសិបពាន់", "មួយលានប្រាំសែន", "ប្រាំបួនរយហុកសិបពីរ")],
        [581, 250_000, 1_500_000, 962])
    expect("Chinese numerals", [_zh_value(r) for r in ("六百八十一", "三千八", "两万五", "二零二六",
                                                       "十二", "一千零一", "一百")],
           [681, 3800, 25000, 2026, 12, 1001, 100])
    expect("numbers in Chinese lines", [source_numbers(s, "zh") for s in (
        "班长681，沈知夏", "他当场扣了我79分，全班56个人", "最后6天", "最后六天", "第二名",
        "万一明天错了呢", "三四个人", "一天两天", "六七十分", "100万", "十分好", "3月5号",
        "等一下", "二零二六年", "百分之八十", "一千零一夜", "247268291320360")],
           [[681], [79, 56], [6], [6], [2], [], [], [], [], [1_000_000], [], [5], [], [2026],
            [80], [1001], ["247268291320360"]])
    km = "ថ្នាក់ប្រធានបានប្រាំរយប៉ែតសិបមួយ សិន ជឺសៀ។"
    expect("the dub's 681 is 581", check("班长681，沈知夏", km, "zh")[0], [681])
    expect("…put right in code", repair("班长681，沈知夏", km, "zh"),
           "ថ្នាក់ប្រធានបាន៦៨១ សិន ជឺសៀ។")
    expect("56 for 55, 79 kept", repair("他当场扣了我79分，全班56个人",
                                         "កាត់ពិន្ទុចិតសិបប្រាំបួនពិន្ទុ សិស្សហាសិបប្រាំនាក់", "zh"),
           "កាត់ពិន្ទុចិតសិបប្រាំបួនពិន្ទុ សិស្ស៥៦នាក់")
    expect("a misspelled fifty is not patched (the AI redoes it)",
           repair("我不想只考250分", "ខ្ញុំមិនចង់បានពិន្ទុត្រឹមពីររយហាស៊ីនទេ!", "zh"), None)
    expect("a number left out is not patched", repair("7天前我还是那个头顶250的废物",
                                                       "ប្រាំពីរថ្ងៃមុនខ្ញុំជាមនុស្សឥតប្រយោជន៍។", "zh"), None)
    expect("digits are fine either way", check("692不是谁都能考到的", "មិនមែនគ្រប់គ្នាបាន ៦៩២ ទេ", "zh")[0],
           [])
    expect("a long digit run is matched digit by digit",
           check("247268291320360", "២៤៧ ២៦៨ ២៩១ ៣២០ ៣៦០", "zh")[0], [])
    expect("a rewording that keeps the number", keeps("班长681", "ប្រធានថ្នាក់បាន ៦៨១", "ប្រធាន ៦៨១", "zh"),
           True)
    expect("a rewording that loses it", keeps("班长681", "ប្រធានថ្នាក់បាន ៦៨១", "ប្រធានថ្នាក់", "zh"),
           False)
    expect("a rewording that repairs it", keeps("班长681", "បានប្រាំរយ", "បាន ៦៨១", "zh"), True)
    expect("រយៈពេល is no hundred", [s.value for s in dub_numbers("រយៈពេលពីរថ្ងៃ")], [2])
    expect("the voice says the digits", km_say_numbers("បាន ៦៨១ ពិន្ទុ, 50% នៅ 8:30, 3.5"),
           "បាន ប្រាំមួយរយប៉ែតសិបមួយ ពិន្ទុ, ហាសិបភាគរយ នៅ ប្រាំបីម៉ោងសាមសិបនាទី, បីចុចប្រាំ")
    expect("a phone number digit by digit", km_say_numbers("012"), "សូន្យ មួយ ពីរ")
    expect("Khmer digits, not in codes", km_digits("H2 បាន 681 ពិន្ទុ"), "H2 បាន ៦៨១ ពិន្ទុ")
    print("PASS" if not bad else f"FAILED: {bad}")
    raise SystemExit(1 if bad else 0)
