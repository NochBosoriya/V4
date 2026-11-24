"""Styled subtitles (.ass) with Khmer-friendly presets, word-by-word
karaoke highlighting and safe zones for vertical (9:16) video.

Cues come from an .srt, a Full Dub project or a recap montage. Word timings
are used when given (Whisper words); otherwise each cue's text is split into
words/phrases and timed proportionally to their length, which works for
Khmer text that has no spaces between words (it splits on spaces, zero-width
spaces and punctuation, falling back to fixed-size chunks).

No Qt imports.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace

from app.core.logger import get_logger

log = get_logger("subtitles")


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass
class SubStyle:
    name: str = "Clean"
    font: str = "Kantumruy Pro"
    size_ratio: float = 0.055        # font size as a fraction of video height
    primary: str = "#FFFFFF"
    highlight: str = "#FFD400"       # karaoke: colour a word turns to
    outline_color: str = "#000000"
    outline_ratio: float = 0.07      # outline width as a fraction of font size
    shadow: float = 1.0
    back_color: str = "#00000099"    # box / shadow colour (with alpha)
    boxed: bool = False              # opaque box behind the text
    bold: bool = False
    margin_v_ratio: float = 0.06     # bottom margin as a fraction of height
    karaoke: bool = False
    max_lines: int = 2


PRESETS: dict[str, SubStyle] = {
    "clean": SubStyle(name="Clean"),
    "bold": SubStyle(name="Bold outline", font="Kantumruy Pro", bold=True,
                     size_ratio=0.062, outline_ratio=0.11, shadow=2.0),
    "boxed": SubStyle(name="Boxed", boxed=True, back_color="#000000B0",
                      outline_ratio=0.0, shadow=0.0),
    "karaoke": SubStyle(name="Karaoke", bold=True, size_ratio=0.06,
                        outline_ratio=0.1, karaoke=True),
    "shorts": SubStyle(name="Shorts (9:16)", bold=True, size_ratio=0.045,
                       outline_ratio=0.12, karaoke=True, margin_v_ratio=0.22,
                       max_lines=2),
    "movie": SubStyle(name="Movie (Battambang)", font="Battambang",
                      size_ratio=0.05, outline_ratio=0.06, shadow=1.5),
}
PRESET_IDS = list(PRESETS)


def preset(style_id: str, vertical: bool = False) -> SubStyle:
    st = PRESETS.get((style_id or "clean").lower(), PRESETS["clean"])
    if vertical and st.margin_v_ratio < 0.15:
        # keep captions above the platform UI (titles, buttons) on Shorts/TikTok
        st = replace(st, margin_v_ratio=0.2, size_ratio=min(st.size_ratio, 0.048))
    return st


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ass_color(hexstr: str) -> str:
    """#RRGGBB[AA] -> &HAABBGGRR (ASS alpha: 00 opaque, FF transparent)."""
    h = (hexstr or "#FFFFFF").lstrip("#")
    if len(h) == 6:
        h += "FF"
    r, g, b, a = (int(h[i:i + 2], 16) for i in (0, 2, 4, 6))
    return f"&H{255 - a:02X}{b:02X}{g:02X}{r:02X}"


def _ts(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
            .replace("\n", "\\N"))


_KHMER = re.compile(r"[\u1780-\u17FF\u19E0-\u19FF]")
_CJK = re.compile(r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF\u0E00-\u0E7F]")


def _char_width(ch: str) -> float:
    """Rough glyph width in em, for line wrapping."""
    if ch == " ":
        return 0.3
    if _CJK.match(ch):
        return 1.0
    if _KHMER.match(ch):
        # combining vowels / subscripts take no horizontal space
        o = ord(ch)
        if 0x17B6 <= o <= 0x17D3 or o == 0x17DD:
            return 0.0
        return 0.62
    return 0.55


def text_width_em(text: str) -> float:
    return sum(_char_width(c) for c in text)


_COENG = "្"   # Khmer subscript sign: joins the next consonant below


def clusters(text: str) -> list[str]:
    """Grapheme-like clusters: a base character plus its combining marks;
    for Khmer, a coeng and the consonant after it stay in the same cluster."""
    out: list[str] = []
    join_next = False
    for ch in text:
        if out and (join_next or (_char_width(ch) == 0.0 and ch != " ")):
            out[-1] += ch
        else:
            out.append(ch)
        join_next = ch == _COENG
    return out


def split_words(text: str) -> list[str]:
    """Words (latin), or phrases (Khmer/Thai/CJK: split at spaces, ZWSP and
    punctuation; long unspaced runs are chunked between whole clusters).
    Joining the pieces gives back the text (ZWSP turned into spaces)."""
    text = re.sub(r"\s+", " ", (text or "").replace("​", " ")).strip()
    if not text:
        return []
    parts: list[str] = []
    for p in re.split(r"(?<=[ ។៕,，.!?！？、])", text):
        if not p:
            continue
        if not p.strip() and parts:
            parts[-1] += p          # keep the space with the previous word
        else:
            parts.append(p)
    out = []
    for p in parts:
        cl = clusters(p)
        if len(cl) > 10 and " " not in p.strip():
            # unspaced script: chunks of ~5 clusters
            chunks = ["".join(cl[i:i + 5]) for i in range(0, len(cl), 5)]
            if len(chunks) > 1 and len(clusters(chunks[-1])) < 3:
                tail = chunks.pop()     # not `chunks[-2] += chunks.pop()`:
                chunks[-1] += tail      # that writes to the first chunk
            out.extend(chunks)
        else:
            out.append(p)
    return out


def wrap(words: list[str], max_em: float, max_lines: int) -> list[list[str]]:
    """Balanced greedy wrap of words into at most max_lines lines (extra
    words go on the last line rather than being dropped)."""
    if not words:
        return []
    total = sum(text_width_em(w) for w in words)
    lines_needed = min(max_lines, max(1, int(total // max_em) + (total % max_em > 0)))
    target = total / lines_needed
    lines: list[list[str]] = [[]]
    width = 0.0
    for w in words:
        ww = text_width_em(w)
        if lines[-1] and width + ww > max(target * 1.15, 1) and len(lines) < lines_needed:
            lines.append([])
            width = 0.0
        lines[-1].append(w)
        width += ww
    return lines


def _word_timing(cue: Cue, words: list[str]) -> list[tuple[float, float, str]]:
    if cue.words and len(cue.words) == len(words):
        return list(cue.words)
    dur = max(0.2, cue.end - cue.start)
    weights = [max(0.5, text_width_em(w)) for w in words]
    tot = sum(weights) or 1.0
    t, out = cue.start, []
    for w, wt in zip(words, weights):
        d = dur * wt / tot
        out.append((t, t + d, w))
        t += d
    return out


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

def build_ass(cues: list[Cue], width: int, height: int, style: SubStyle) -> str:
    width, height = max(16, int(width)), max(16, int(height))
    size = max(12, int(round(height * style.size_ratio)))
    outline = round(size * style.outline_ratio, 1)
    margin_v = int(round(height * style.margin_v_ratio))
    margin_h = int(round(width * 0.06))
    max_em = (width - 2 * margin_h) / size
    border_style = 3 if style.boxed else 1
    back = _ass_color(style.back_color)
    outline_col = back if style.boxed else _ass_color(style.outline_color)
    # with karaoke, PrimaryColour is the "sung" colour, SecondaryColour the
    # colour before the word is reached
    primary = _ass_color(style.highlight if style.karaoke else style.primary)
    secondary = _ass_color(style.primary)
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{style.font},{size},{primary},{secondary},{outline_col},"
        f"{back},{-1 if style.bold else 0},0,0,0,100,100,0,0,{border_style},"
        f"{outline if not style.boxed else max(4, size * 0.18):g},{style.shadow:g},2,"
        f"{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    events = []
    for cue in cues:
        text = (cue.text or "").strip()
        if not text or cue.end <= cue.start:
            continue
        words = split_words(text)
        lines = wrap(words, max_em, style.max_lines)
        if style.karaoke:
            timed = _word_timing(cue, words)
            parts, k = [], 0
            for li, line in enumerate(lines):
                if li:
                    parts.append("\\N")
                for w in line:
                    s, e, _w = timed[k]
                    k += 1
                    cs = max(1, int(round((e - s) * 100)))
                    parts.append(f"{{\\kf{cs}}}{_esc(w)}")
            body = "".join(parts)
        else:
            body = "\\N".join(_esc("".join(l).strip()) for l in lines)
        events.append(f"Dialogue: 0,{_ts(cue.start)},{_ts(cue.end)},Default,,0,0,0,,"
                      f"{body}")
    return "\n".join(head + events) + "\n"


def write_ass(cues: list[Cue], path: str, width: int, height: int,
              style: SubStyle | str = "clean") -> str:
    st = preset(style, vertical=height > width) if isinstance(style, str) else style
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(build_ass(cues, width, height, st))
    return path


_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def read_srt(path: str) -> list[Cue]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.read()
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", content):
        m = _SRT_TIME.search(block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / (10 ** len(m.group(4)))
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / (10 ** len(m.group(8)))
        text = " ".join(l.strip() for l in block[m.end():].strip().splitlines() if l.strip())
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            cues.append(Cue(s, e, text))
    return cues


def srt_to_ass(srt_path: str, ass_path: str, width: int, height: int,
               style: str = "clean") -> str:
    return write_ass(read_srt(srt_path), ass_path, width, height, style)


def style_dict(style_id: str) -> dict:
    return asdict(preset(style_id))


if __name__ == "__main__":
    import os
    import tempfile

    assert _ass_color("#FFD400") == "&H0000D4FF", _ass_color("#FFD400")
    assert _ass_color("#00000099") == "&H66000000"
    assert _ts(3725.456) == "1:02:05.46"
    ws = split_words("Where are you going tonight?")
    assert ws[0] == "Where " and len(ws) == 5, ws
    line = "តើអ្នកនឹងទៅណា? ខ្ញុំរង់ចាំអ្នកយូរណាស់ហើយ។"
    km = split_words(line)
    assert len(km) >= 3 and "".join(km) == line, km
    assert not any(w[0] in "្ាិីុំ់" for w in km), \
        f"a chunk starts with a subscript/vowel: {km}"
    line2 = "សួស្ដី! ខ្ញុំស្រឡាញ់ភាពយន្តខ្មែរ ហើយអ្នកម្ដាយជើងខ្លាំងគឺជារឿងល្អបំផុត"
    assert "".join(split_words(line2)) == line2, split_words(line2)  # short-tail merge
    assert clusters("អ្នក") == ["អ្ន", "ក"], clusters("អ្នក")
    assert text_width_em("ក្ស") < text_width_em("កកក"), "subscripts are narrow"
    lines = wrap(split_words("one two three four five six seven eight nine ten"), 12, 2)
    assert len(lines) == 2 and abs(len(lines[0]) - len(lines[1])) <= 2, lines
    cues = [Cue(1.0, 3.0, "Hello there friend"),
            Cue(3.2, 5.0, "ចាស ខ្ញុំយល់ហើយ។", words=[])]
    ass = build_ass(cues, 1920, 1080, preset("karaoke"))
    assert "PlayResY: 1080" in ass and ass.count("Dialogue:") == 2
    assert "{\\kf" in ass and "Hello" in ass
    k = [int(x) for x in re.findall(r"\\kf(\d+)", ass.splitlines()[-2])]
    assert abs(sum(k) - 200) <= 3, k          # word durations fill the cue
    v = preset("clean", vertical=True)
    assert v.margin_v_ratio >= 0.2
    d = tempfile.mkdtemp()
    srt = os.path.join(d, "a.srt")
    with open(srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,500\n<i>Hi</i>\nthere\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nបាទ\n")
    rc = read_srt(srt)
    assert [c.text for c in rc] == ["Hi there", "បាទ"] and rc[0].end == 2.5
    out = srt_to_ass(srt, os.path.join(d, "a.ass"), 1080, 1920, "shorts")
    txt = open(out, encoding="utf-8-sig").read()
    assert "PlayResX: 1080" in txt and ",422,1" in txt.split("Style: Default")[1].splitlines()[0]
    print("PASS")
