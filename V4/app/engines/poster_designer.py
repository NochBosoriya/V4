"""Poster designer engine.

OCR title detection (PaddleOCR), region inpainting (LaMa onnx / cv2 TELEA /
PIL patch fallback), AI title translation + style-spec design, Khmer-correct
title rendering (RAQM -> uharfbuzz+freetype -> BASIC), and aspect-ratio export
variants.

Khmer note: subscript consonants (ជើងអក្សរ, formed with U+17D2 COENG) require
real complex-text shaping. Rendering per code point or per naive cluster with a
basic layout engine produces visibly wrong glyphs. This module therefore:
  1. uses Pillow's RAQM layout engine when available (libraqm = harfbuzz +
     fribidi, correct shaping),
  2. otherwise shapes manually with uharfbuzz (glyph ids + advances/offsets)
     and rasterizes each glyph with freetype-py,
  3. and only as a last resort falls back to Pillow BASIC layout with a logged
     bilingual warning.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

# Allow running this file directly (python app/engines/poster_designer.py)
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from app.core.errors import AppError, CancelledError, EngineNotAvailable
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import fonts_dir as _bundled_fonts_dir
from app.core.paths import models_dir, pip_install

if TYPE_CHECKING:  # no runtime import; ai_engine is only needed when passed in
    from app.core.ai_engine import AIEngine

log = get_logger("poster")

Progress = Optional[Callable[[float, str], None]]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _check_cancel(cancel: "threading.Event | None") -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


def _report(progress: Progress, fraction: float, text: str) -> None:
    if progress is not None:
        try:
            progress(max(0.0, min(1.0, float(fraction))), text)
        except Exception:
            pass


def _open_image(path: str) -> Image.Image:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as e:
        raise AppError(
            f"Cannot open image: {path} ({e})",
            f"មិនអាចបើករូបភាព {path} បានទេ ({e})") from e


def _save_image(img: Image.Image, out_path: str, quality: int = 95) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    try:
        if ext in (".jpg", ".jpeg"):
            img.convert("RGB").save(str(p), quality=quality)
        elif ext == ".webp":
            (img if img.mode in ("RGB", "RGBA") else img.convert("RGBA")).save(
                str(p), quality=quality)
        elif ext == ".bmp":
            img.convert("RGB").save(str(p))
        else:  # png and anything else
            img.save(str(p))
    except Exception as e:
        raise AppError(
            f"Failed to save image to {out_path}: {e}",
            f"មិនអាចរក្សាទុករូបភាពទៅ {out_path} បានទេ៖ {e}") from e


_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _parse_color(value: Any, default: tuple[int, int, int, int] = (255, 255, 255, 255)
                 ) -> tuple[int, int, int, int]:
    """Parse '#rgb'/'#rrggbb'/'#rrggbbaa' (leading '#' optional) or an RGB(A)
    sequence into an (r, g, b, a) tuple. Invalid input returns `default`."""
    if isinstance(value, (tuple, list)) and len(value) in (3, 4):
        try:
            vals = [max(0, min(255, int(round(float(v))))) for v in value]
            while len(vals) < 4:
                vals.append(255)
            return (vals[0], vals[1], vals[2], vals[3])
        except Exception:
            return default
    if not isinstance(value, str):
        return default
    m = _HEX_RE.match(value.strip())
    if not m:
        return default
    h = m.group(1)
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    a = int(h[6:8], 16) if len(h) == 8 else 255
    return (r, g, b, a)


def _is_hex_color(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX_RE.match(value.strip()))


def _norm_hex(value: Any, default: str) -> str:
    """Normalize a color into lowercase '#rrggbb' or '#rrggbbaa'."""
    src = value if (_is_hex_color(value) or isinstance(value, (tuple, list))) else default
    r, g, b, a = _parse_color(src, _parse_color(default))
    if a != 255:
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_json_obj(text: Any) -> Any:
    """Tolerant JSON extraction: strips code fences, then brace-scans for the
    first parseable object/array. Returns None on failure."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(t)):
                ch = t[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = in_str
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except Exception:
                            break
            start = t.find(opener, start + 1)
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OcrBox:
    x: int
    y: int
    w: int
    h: int
    text: str
    conf: float
    alts: tuple = ()        # other readings of the same text (read at other sizes)


def _co_float(v: Any, default: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return min(hi, max(lo, x))


def _co_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, x))


def _co_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    return default


def _co_pair_int(v: Any, default: tuple[int, int], lo: int, hi: int) -> tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (_co_int(v[0], default[0], lo, hi), _co_int(v[1], default[1], lo, hi))
    return default


def _co_pair_float(v: Any, default: tuple[float, float], lo: float, hi: float
                   ) -> tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (_co_float(v[0], default[0], lo, hi), _co_float(v[1], default[1], lo, hi))
    return default


@dataclass
class StyleSpec:
    """Title styling spec. All fields defaulted; `from_llm_json` is tolerant."""
    font: str = "Moul"
    size_ratio: float = 0.12                     # of image width
    fill: str = "#ffffff"
    gradient: list[str] | None = None            # 2 hex stops, vertical
    stroke_color: str = "#000000"
    stroke_width_ratio: float = 0.04             # of font size
    shadow_color: str = "#000000cc"
    shadow_offset: tuple[int, int] = (4, 4)
    shadow_blur: int = 8
    glow_color: str | None = None
    glow_radius: int = 0
    bevel: bool = False
    extrude_depth: int = 0
    extrude_color: str = "#222222"
    pos: tuple[float, float] = (0.5, 0.82)       # center, fraction of W/H
    letter_spacing: int = 0
    rotation: float = 0.0
    # a dark ink-brush stroke behind each line (like a movie-poster title)
    backdrop: bool = False
    backdrop_color: str = "#0b0805e8"
    sparkles: int = 0                            # small light stars around the title
    # line breaks: "" = as typed ("|" starts a new line); "colon" = also
    # after ":" / "៖"; "auto" = after a colon, else at the space nearest the
    # middle of a long title
    line_break: str = ""
    line_scales: list[float] | None = None       # size of each line (1.0 = full)
    align: str = "center"                        # the lines: "center" | "left" | "right"

    @classmethod
    def from_llm_json(cls, d: Any) -> "StyleSpec":
        """Build a StyleSpec from arbitrary LLM output: ignores unknown keys,
        coerces types, clamps ranges, never raises."""
        spec = cls()
        if isinstance(d, str):
            d = _extract_json_obj(d)
        if not isinstance(d, dict):
            return spec

        aliases = {
            "font_family": "font", "font_name": "font", "fontfamily": "font",
            "typeface": "font",
            "size": "size_ratio", "font_size_ratio": "size_ratio",
            "color": "fill", "fill_color": "fill", "text_color": "fill",
            "gradient_colors": "gradient", "gradient_stops": "gradient",
            "outline_color": "stroke_color", "stroke": "stroke_color",
            "outline_width_ratio": "stroke_width_ratio",
            "stroke_width": "stroke_width_ratio",
            "position": "pos", "letterspacing": "letter_spacing",
            "tracking": "letter_spacing", "angle": "rotation",
            "extrude": "extrude_depth", "glow": "glow_color",
        }
        data: dict[str, Any] = {}
        for k, v in d.items():
            if not isinstance(k, str):
                continue
            key = aliases.get(k.strip().lower(), k.strip().lower())
            if key not in data:
                data[key] = v

        if isinstance(data.get("font"), str) and data["font"].strip():
            spec.font = data["font"].strip()
        spec.size_ratio = _co_float(data.get("size_ratio", spec.size_ratio),
                                    spec.size_ratio, 0.02, 0.5)
        spec.fill = _norm_hex(data.get("fill", spec.fill), spec.fill)

        grad = data.get("gradient", spec.gradient)
        if isinstance(grad, (list, tuple)):
            stops = [_norm_hex(g, "#ffffff") for g in grad if _is_hex_color(g)]
            spec.gradient = stops[:4] if len(stops) >= 2 else None   # 2-4 stops, top to bottom
        else:
            spec.gradient = None if grad is None else spec.gradient

        spec.stroke_color = _norm_hex(data.get("stroke_color", spec.stroke_color),
                                      spec.stroke_color)
        spec.stroke_width_ratio = _co_float(
            data.get("stroke_width_ratio", spec.stroke_width_ratio),
            spec.stroke_width_ratio, 0.0, 0.3)
        spec.shadow_color = _norm_hex(data.get("shadow_color", spec.shadow_color),
                                      spec.shadow_color)
        spec.shadow_offset = _co_pair_int(data.get("shadow_offset", spec.shadow_offset),
                                          spec.shadow_offset, -100, 100)
        spec.shadow_blur = _co_int(data.get("shadow_blur", spec.shadow_blur),
                                   spec.shadow_blur, 0, 64)

        glow = data.get("glow_color", spec.glow_color)
        if glow is None or (isinstance(glow, str)
                            and glow.strip().lower() in ("", "none", "null", "false")):
            spec.glow_color = None
        elif _is_hex_color(glow) or isinstance(glow, (list, tuple)):
            spec.glow_color = _norm_hex(glow, "#ffffff")
        else:
            spec.glow_color = None
        spec.glow_radius = _co_int(data.get("glow_radius", spec.glow_radius),
                                   spec.glow_radius, 0, 64)

        spec.bevel = _co_bool(data.get("bevel", spec.bevel), spec.bevel)
        spec.extrude_depth = _co_int(data.get("extrude_depth", spec.extrude_depth),
                                     spec.extrude_depth, 0, 40)
        spec.extrude_color = _norm_hex(data.get("extrude_color", spec.extrude_color),
                                       spec.extrude_color)
        spec.pos = _co_pair_float(data.get("pos", spec.pos), spec.pos, 0.0, 1.0)
        spec.letter_spacing = _co_int(data.get("letter_spacing", spec.letter_spacing),
                                      spec.letter_spacing, -20, 200)
        spec.rotation = _co_float(data.get("rotation", spec.rotation),
                                  spec.rotation, -180.0, 180.0)
        spec.backdrop = _co_bool(data.get("backdrop", spec.backdrop), spec.backdrop)
        spec.backdrop_color = _norm_hex(data.get("backdrop_color", spec.backdrop_color),
                                        spec.backdrop_color)
        spec.sparkles = _co_int(data.get("sparkles", spec.sparkles), spec.sparkles, 0, 60)
        lb = str(data.get("line_break", spec.line_break) or "").strip().lower()
        spec.line_break = lb if lb in ("", "colon", "auto") else ""
        scales = data.get("line_scales", spec.line_scales)
        if isinstance(scales, (list, tuple)) and scales:
            spec.line_scales = [_co_float(s, 1.0, 0.2, 1.0) for s in list(scales)[:4]]
        else:
            spec.line_scales = None
        al = str(data.get("align", spec.align) or "").strip().lower()
        spec.align = al if al in ("center", "left", "right") else "center"
        return spec


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

_FONT_ALIASES: dict[str, list[str]] = {
    "moul": ["moul", "khmerosmuollight", "khmerosmuol", "muollight", "muol"],
    "khmeros": ["khmeros"],
    "khmerosmuollight": ["khmerosmuollight", "moul", "muol"],
    "khmerui": ["khmerui", "khmeruib"],
    "leelawadeeui": ["leelawui", "leelauib", "leelawadeeui"],
    "kantumruypro": ["kantumruypro", "kantumruy"],
    "kantumruy": ["kantumruy", "kantumruypro"],
    "battambang": ["battambang"],
    "nokora": ["nokora"],
    "koulen": ["koulen"],
    "hanuman": ["hanuman"],
    "dangrek": ["dangrek"],
}

_FONT_SCAN_CACHE: dict[str, list[tuple[str, set[str]]]] = {}
_FONT_CACHE_LOCK = threading.Lock()


def _norm_name(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _font_family_names(path: Path) -> set[str]:
    """Family names in a font file via fontTools (if importable) + filename."""
    names = {_norm_name(path.stem)}
    try:
        from fontTools.ttLib import TTFont
        try:
            from fontTools.ttLib import TTCollection
        except Exception:
            TTCollection = None
    except Exception:
        return names
    try:
        if path.suffix.lower() == ".ttc" and TTCollection is not None:
            fonts = list(TTCollection(str(path), lazy=True).fonts)
        else:
            fonts = [TTFont(str(path), lazy=True, fontNumber=0)]
        for f in fonts:
            try:
                name_table = f["name"]
                for nid in (1, 4, 16):
                    val = name_table.getDebugName(nid)
                    if val:
                        names.add(_norm_name(val))
            except Exception:
                pass
            finally:
                try:
                    f.close()
                except Exception:
                    pass
    except Exception:
        pass
    return names


def _scan_font_dir(d: Path, parse_names: bool) -> list[tuple[str, set[str]]]:
    key = f"{d}|{int(parse_names)}"
    with _FONT_CACHE_LOCK:
        if key in _FONT_SCAN_CACHE:
            return _FONT_SCAN_CACHE[key]
    entries: list[tuple[str, set[str]]] = []
    try:
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                    fams = (_font_family_names(p) if parse_names
                            else {_norm_name(p.stem)})
                    entries.append((str(p), fams))
    except Exception:
        pass
    with _FONT_CACHE_LOCK:
        _FONT_SCAN_CACHE[key] = entries
    return entries


_STYLE_WORDS = ("italic", "oblique", "thin", "hairline", "extralight", "light",
                "black", "heavy", "extrabold", "semibold", "bold", "medium",
                "condensed")


def _style_rank(path: str) -> int:
    """0 for Regular / upright variable files, higher for styled ones, so a
    family lookup picks e.g. Battambang-Regular over Battambang-Black."""
    stem = Path(path).stem.lower().split("[")[0]
    return sum(10 if w in ("italic", "oblique") else 1
               for w in _STYLE_WORDS if w in stem)


def find_font_file(name: str, fonts_dir: Path | None = None) -> str | None:
    """Find a font file by family name.

    Search order: bundled fonts dir (family names parsed via fontTools when
    available), then the Windows fonts folders (filename matching + a small
    alias map for common Khmer fonts). Returns an absolute path or None.
    """
    target = _norm_name(name)
    if not target:
        return None
    candidates = [target] + [c for c in _FONT_ALIASES.get(target, []) if c != target]

    bundled = Path(fonts_dir) if fonts_dir is not None else _bundled_fonts_dir()
    search: list[tuple[Path, bool]] = [(bundled, True)]
    windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    search.append((windir, False))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        search.append((Path(local) / "Microsoft" / "Windows" / "Fonts", False))

    for d, parse_names in search:
        entries = _scan_font_dir(d, parse_names)
        # exact family-name match first; the plainest style of the family
        exact = [path for path, fams in entries if any(c in fams for c in candidates)]
        if exact:
            return min(exact, key=_style_rank)
        # then substring match (both directions), min length 3 to avoid noise
        for path, fams in entries:
            for fam in fams:
                if len(fam) < 3:
                    continue
                for c in candidates:
                    if len(c) >= 3 and (c in fam or fam in c):
                        return path
    return None


def _resolve_font(spec_font: str, fonts_dir: Path | None) -> str:
    path = find_font_file(spec_font, fonts_dir)
    if path:
        return path
    for fallback in ("Moul", "Khmer OS Muol Light", "Kantumruy Pro",
                     "Battambang", "Khmer OS", "Khmer UI", "Leelawadee UI"):
        path = find_font_file(fallback, fonts_dir)
        if path:
            log.info("Font '%s' not found; using fallback '%s' (%s)",
                     spec_font, fallback, path)
            return path
    # any bundled font at all
    base = Path(fonts_dir) if fonts_dir is not None else _bundled_fonts_dir()
    entries = _scan_font_dir(base, False)
    if entries:
        log.info("Font '%s' not found; using first bundled font %s",
                 spec_font, entries[0][0])
        return entries[0][0]
    raise EngineNotAvailable(
        f"No suitable font found for '{spec_font}'. Place a Khmer font "
        f"(e.g. Moul.ttf) in app/assets/fonts.",
        f"រកមិនឃើញពុម្ពអក្សរសមរម្យសម្រាប់ '{spec_font}' ទេ។ សូមដាក់ពុម្ពអក្សរខ្មែរ "
        f"(ឧទាហរណ៍ Moul.ttf) ក្នុងថត app/assets/fonts។")


# ---------------------------------------------------------------------------
# Text shaping / rasterization
# ---------------------------------------------------------------------------

_KHMER_RE = re.compile(r"[\u1780-\u17ff\u19e0-\u19ff]")
_KHMER_COENG = "\u17d2"
_raqm_available_cache: bool | None = None
_basic_warned = False


def _raqm_available() -> bool:
    global _raqm_available_cache
    if _raqm_available_cache is None:
        ok = False
        try:
            from PIL import features
            ok = bool(features.check("raqm")) and hasattr(ImageFont, "Layout")
        except Exception:
            ok = False
        _raqm_available_cache = ok
        log.debug("RAQM layout available: %s", ok)
    return _raqm_available_cache


def _warn_basic_layout() -> None:
    global _basic_warned
    if _basic_warned:
        return
    _basic_warned = True
    log.warning(
        "RAQM/HarfBuzz unavailable - falling back to BASIC text layout; Khmer "
        "subscript consonants may render incorrectly. | RAQM/HarfBuzz មិនមានទេ "
        "- ប្រើការរៀបអក្សរធម្មតាជំនួសវិញ ដូច្នេះជើងអក្សរខ្មែរ (ជើងអក្សរ) "
        "អាចបង្ហាញមិនត្រឹមត្រូវ។")


def _split_clusters(text: str) -> list[str]:
    """Split text into orthographic-syllable-ish clusters. A combining mark,
    a COENG (U+17D2), the consonant following a COENG, and ZWJ/ZWNJ all stay
    attached to the preceding base character. Safe for letter-spacing Khmer."""
    clusters: list[str] = []
    cur = ""
    prev_coeng = False
    for ch in text:
        cat = unicodedata.category(ch)
        joiner = ch in ("\u200c", "\u200d")
        attach = bool(cur) and (prev_coeng or joiner or ch == _KHMER_COENG
                                or cat.startswith("M"))
        if attach:
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
        prev_coeng = (ch == _KHMER_COENG) or (joiner and prev_coeng)
    if cur:
        clusters.append(cur)
    return clusters


def _render_line_pil(text: str, font_path: str, size: int, letter_spacing: int,
                     layout: Any) -> Image.Image:
    """Render one line to an L-mode mask using Pillow. `layout` is an
    ImageFont.Layout value or None (Pillow default)."""
    if layout is None:
        font = ImageFont.truetype(font_path, size)
    else:
        font = ImageFont.truetype(font_path, size, layout_engine=layout)
    parts = _split_clusters(text) if letter_spacing else [text]
    widths = [font.getlength(p) for p in parts]
    total = sum(widths) + letter_spacing * max(0, len(parts) - 1)
    ascent, descent = font.getmetrics()
    margin = max(8, size // 2)
    w = int(math.ceil(max(1.0, total))) + margin * 2
    h = int(ascent + descent) + margin * 2
    canvas = Image.new("L", (max(w, 1), max(h, 1)), 0)
    drawer = ImageDraw.Draw(canvas)
    x = float(margin)
    for part, pw in zip(parts, widths):
        drawer.text((x, margin), part, fill=255, font=font)
        x += pw + letter_spacing
    bbox = canvas.getbbox()
    if bbox is None:
        return Image.new("L", (1, 1), 0)
    return canvas.crop(bbox)


def _ft_bitmap_to_image(bm: Any) -> Image.Image | None:
    """Convert a freetype-py glyph bitmap (gray or mono) to an L-mode image."""
    w, r = int(bm.width), int(bm.rows)
    if w <= 0 or r <= 0:
        return None
    pitch = int(bm.pitch)
    ap = abs(pitch) or w
    raw = bytes(bytearray(v & 0xFF for v in bm.buffer))
    rows = [raw[i * ap:i * ap + ap] for i in range(r)]
    if pitch < 0:
        rows.reverse()
    if int(getattr(bm, "pixel_mode", 2)) == 1:  # FT_PIXEL_MODE_MONO
        out = bytearray(w * r)
        for ri, row in enumerate(rows):
            for c in range(w):
                if c >> 3 < len(row) and (row[c >> 3] & (0x80 >> (c & 7))):
                    out[ri * w + c] = 255
        return Image.frombytes("L", (w, r), bytes(out))
    data = b"".join(row[:w].ljust(w, b"\0") for row in rows)
    return Image.frombytes("L", (w, r), data)


def _paste_max(canvas: Image.Image, glyph: Image.Image, x: int, y: int) -> None:
    """Paste an L glyph onto an L canvas using per-pixel max (no darkening
    where anti-aliased glyphs overlap)."""
    gw, gh = glyph.size
    cw, ch = canvas.size
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + gw, cw), min(y + gh, ch)
    if x1 <= x0 or y1 <= y0:
        return
    sub = glyph.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    region = canvas.crop((x0, y0, x1, y1))
    canvas.paste(ImageChops.lighter(region, sub), (x0, y0))


def _render_line_hb(text: str, font_path: str, size: int,
                    letter_spacing: int) -> Image.Image:
    """Shape with uharfbuzz (whole string, correct Khmer subscripts) and
    rasterize each shaped glyph with freetype-py at the shaped positions."""
    import uharfbuzz as hb
    import freetype

    data = Path(font_path).read_bytes()
    face = hb.Face(data)
    font = hb.Font(face)
    upem = int(face.upem) or 1000
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"ccmp": True, "kern": True, "liga": True,
                         "clig": True, "mark": True, "mkmk": True})
    infos = list(buf.glyph_infos)
    poss = list(buf.glyph_positions)
    scale = size / float(upem)

    ft_face = freetype.Face(str(font_path))
    ft_face.set_pixel_sizes(0, size)
    asc = ft_face.size.ascender / 64.0
    desc = ft_face.size.descender / 64.0  # negative
    margin = max(8, size // 2)
    n_breaks = sum(1 for i in range(1, len(infos))
                   if infos[i].cluster != infos[i - 1].cluster)
    total_adv = sum(p.x_advance for p in poss) * scale + letter_spacing * n_breaks
    w = int(math.ceil(max(1.0, abs(total_adv)))) + margin * 2
    h = int(math.ceil(max(1.0, asc - desc))) + margin * 2
    canvas = Image.new("L", (w, h), 0)
    pen_x = float(margin)
    baseline = margin + asc
    prev_cluster: int | None = None
    for info, pos in zip(infos, poss):
        if prev_cluster is not None and info.cluster != prev_cluster:
            pen_x += letter_spacing
        prev_cluster = info.cluster
        try:
            ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            slot = ft_face.glyph
            glyph_img = _ft_bitmap_to_image(slot.bitmap)
            if glyph_img is not None:
                gx = int(round(pen_x + pos.x_offset * scale + slot.bitmap_left))
                gy = int(round(baseline - pos.y_offset * scale - slot.bitmap_top))
                _paste_max(canvas, glyph_img, gx, gy)
        except Exception as e:
            log.debug("freetype glyph %s render failed: %s", info.codepoint, e)
        pen_x += pos.x_advance * scale
    bbox = canvas.getbbox()
    if bbox is None:
        return Image.new("L", (1, 1), 0)
    return canvas.crop(bbox)


def _render_line_mask(text: str, font_path: str, size: int,
                      letter_spacing: int) -> Image.Image:
    """One text line -> L mask, trying RAQM, then uharfbuzz+freetype, then
    BASIC layout with a bilingual warning."""
    if _raqm_available():
        try:
            return _render_line_pil(text, font_path, size, letter_spacing,
                                    ImageFont.Layout.RAQM)
        except Exception as e:
            log.debug("RAQM render failed, trying harfbuzz: %s", e)
    try:
        return _render_line_hb(text, font_path, size, letter_spacing)
    except CancelledError:
        raise
    except Exception as e:
        log.debug("uharfbuzz/freetype render failed, using BASIC: %s", e)
    if _KHMER_RE.search(text):
        _warn_basic_layout()
    layout = getattr(getattr(ImageFont, "Layout", None), "BASIC", None)
    return _render_line_pil(text, font_path, size, letter_spacing, layout)


def _draw_shaped_text(draw_target_size: tuple[int, int] | int, text: str,
                      font_path: str, size: int, letter_spacing: int = 0
                      ) -> tuple[Image.Image, int, int]:
    """Render (possibly multi-line) text as a tight L-mode mask.

    Returns (mask, width, height). `draw_target_size` bounds runaway sizes
    (mask width is capped at 4x the target image width)."""
    size = max(8, int(size))
    raw_lines = [ln.strip() for ln in str(text).replace("\r", "").split("\n")]
    lines = [ln for ln in raw_lines if ln]
    if not lines:
        blank = Image.new("L", (1, 1), 0)
        return blank, 1, 1
    masks = [_render_line_mask(ln, font_path, size, int(letter_spacing))
             for ln in lines]
    gap = max(2, int(size * 0.28))
    total_w = max(m.size[0] for m in masks)
    total_h = sum(m.size[1] for m in masks) + gap * (len(masks) - 1)
    canvas = Image.new("L", (max(total_w, 1), max(total_h, 1)), 0)
    y = 0
    for m in masks:
        canvas.paste(m, ((total_w - m.size[0]) // 2, y))
        y += m.size[1] + gap
    if isinstance(draw_target_size, (tuple, list)) and draw_target_size:
        target_w = int(draw_target_size[0])
    else:
        target_w = int(draw_target_size or 0)
    max_w = max(64, target_w * 4) if target_w > 0 else 0
    if max_w and canvas.size[0] > max_w:
        new_h = max(1, int(canvas.size[1] * max_w / canvas.size[0]))
        canvas = canvas.resize((max_w, new_h), Image.LANCZOS)
    return canvas, canvas.size[0], canvas.size[1]


# ---------------------------------------------------------------------------
# Title rendering with effects
# ---------------------------------------------------------------------------

def _tint(mask: Image.Image, rgba: tuple[int, int, int, int]) -> Image.Image:
    """Colored RGBA layer whose alpha is the mask scaled by the color alpha."""
    r, g, b, a = rgba
    img = Image.new("RGBA", mask.size, (r, g, b, 0))
    alpha = mask if a >= 255 else mask.point(lambda v: (v * a) // 255)
    img.putalpha(alpha)
    return img


def _stamp(layer: Image.Image, tinted: Image.Image, dx: int, dy: int) -> Image.Image:
    """Alpha-composite `tinted` onto `layer` at an (dx, dy) offset."""
    tmp = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    tmp.paste(tinted, (dx, dy), tinted)
    return Image.alpha_composite(layer, tmp)


def _alpha_paste(base: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """Proper 'over' composite of an RGBA layer onto an RGBA base at (x, y),
    clipping as needed (supports negative offsets)."""
    bw, bh = base.size
    lw, lh = layer.size
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + lw, bw), min(y + lh, bh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = layer.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    region = base.crop((x0, y0, x1, y1))
    base.paste(Image.alpha_composite(region, sub), (x0, y0))


def _vertical_gradient_fill(size_wh: tuple[int, int],
                            top: tuple[int, int, int, int],
                            bottom: tuple[int, int, int, int],
                            y0: int, y1: int, mask: Image.Image) -> Image.Image:
    """RGBA layer: vertical linear gradient (top->bottom over rows y0..y1),
    alpha taken from `mask`."""
    w, h = size_wh
    span = max(1, y1 - y0)
    t = np.clip((np.arange(h, dtype=np.float64) - y0) / span, 0.0, 1.0)
    top_rgb = np.array(top[:3], dtype=np.float64)
    bot_rgb = np.array(bottom[:3], dtype=np.float64)
    rows = (top_rgb[None, :] * (1.0 - t[:, None])
            + bot_rgb[None, :] * t[:, None])
    arr = np.repeat(rows[:, None, :], w, axis=1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").convert("RGBA")
    img.putalpha(mask)
    return img


_LINE_SPLIT_RE = re.compile(r"\n|\|")
_COLON_SPLIT_RE = re.compile(r"^(.+?[:៖])\s*(\S.*)$")


def title_lines(text: str, line_break: str = "") -> list[str]:
    """The title's lines. "|" (or a line break) always starts a new line;
    `line_break` "colon" also breaks after the first ":" / "៖", and "auto"
    after a colon, else a long title at the space nearest its middle."""
    parts = [p.strip() for p in _LINE_SPLIT_RE.split(str(text or "").replace("\r", ""))]
    lines = [p for p in parts if p]
    if len(lines) != 1 or line_break not in ("colon", "auto"):
        return lines
    one = lines[0]
    m = _COLON_SPLIT_RE.match(one)
    if m and len(m.group(2).strip()) >= 2:
        return [m.group(1).strip(), m.group(2).strip()]
    if line_break == "auto" and len(one) >= 18:
        spaces = [i for i, ch in enumerate(one) if ch == " "]
        if spaces:
            mid = len(one) / 2
            i = min(spaces, key=lambda s: abs(s - mid))
            if 0.25 * len(one) <= i <= 0.75 * len(one):
                return [one[:i].strip(), one[i + 1:].strip()]
    return lines


def _line_scales(lines: list[str], spec: StyleSpec) -> list[float]:
    """Size factor of each line (the largest is 1.0): spec.line_scales when
    it has one per line; else a lead-in line ending in a colon ("Reborn in
    the Qin Dynasty:") is smaller than the line after it."""
    sc = list(spec.line_scales or [])
    if len(sc) == len(lines) and sc:
        top = max(sc) or 1.0
        return [max(0.2, s / top) for s in sc]
    if len(lines) == 2 and lines[0].rstrip().endswith((":", "៖")):
        return [0.62, 1.0]
    return [1.0] * len(lines)


def _title_block(lines: list[str], font_path: str, sizes: list[int],
                 letter_spacing: int, align: str = "center"
                 ) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """The lines rendered and stacked — centred, or flush left / right as
    `align` says: (L mask, [(x, y, w, h) of each line in it])."""
    masks = [_render_line_mask(ln, font_path, s, int(letter_spacing))
             for ln, s in zip(lines, sizes)]
    gap = max(2, int(0.16 * max(sizes)))
    bw = max(m.size[0] for m in masks)
    bh = sum(m.size[1] for m in masks) + gap * (len(masks) - 1)
    block = Image.new("L", (max(1, bw), max(1, bh)), 0)
    rects: list[tuple[int, int, int, int]] = []
    y = 0
    for m in masks:
        x = (0 if align == "left" else bw - m.size[0] if align == "right"
             else (bw - m.size[0]) // 2)
        block.paste(m, (x, y))
        rects.append((x, y, m.size[0], m.size[1]))
        y += m.size[1] + gap
    return block, rects


def _gradient_layer(size_wh: tuple[int, int], stops: list[str],
                    spans: list[tuple[int, int]], mask: Image.Image) -> Image.Image:
    """RGBA fill: every line runs top to bottom through `stops` (2-4
    colours, evenly spaced; e.g. light / gold / amber / light = metallic);
    rows between lines take the nearest line's colours. Alpha = `mask`."""
    w, h = size_wh
    cols = np.array([_parse_color(s, (255, 255, 255, 255))[:3] for s in stops],
                    dtype=np.float64)
    at = np.linspace(0.0, 1.0, len(cols))
    ys = np.arange(h, dtype=np.float64)
    spans = spans or [(0, h)]
    centres = np.array([(a + b) / 2.0 for a, b in spans])
    idx = np.argmin(np.abs(ys[:, None] - centres[None, :]), axis=1)
    y0 = np.array([a for a, _b in spans], dtype=np.float64)[idx]
    y1 = np.array([b for _a, b in spans], dtype=np.float64)[idx]
    t = np.clip((ys - y0) / np.maximum(1.0, y1 - y0), 0.0, 1.0)
    rows = np.stack([np.interp(t, at, cols[:, c]) for c in range(3)], axis=1)
    arr = np.repeat(rows[:, None, :], w, axis=1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").convert("RGBA")
    img.putalpha(mask)
    return img


def _smooth_noise(shape: tuple[int, int], cell: tuple[float, float],
                  rng: np.random.Generator) -> np.ndarray:
    """Smooth random field in [-1, 1]: random values every `cell` (px down,
    px across), interpolated in between."""
    h, w = shape
    gh = max(2, int(math.ceil(h / max(1.0, cell[0]))) + 2)
    gw = max(2, int(math.ceil(w / max(1.0, cell[1]))) + 2)
    grid = rng.uniform(0, 255, (gh, gw)).astype(np.uint8)
    big = Image.fromarray(grid, "L").resize((w, h), Image.BICUBIC)
    return np.asarray(big, dtype=np.float32) / 127.5 - 1.0


def _ink_backdrop(size_wh: tuple[int, int], rects: list[tuple[int, int, int, int]],
                  pad: int, seed: int) -> Image.Image:
    """L mask of a dry ink-brush stroke behind each line, drawn the way a
    brush leaves it: many horizontal bristle lines overlapping into a solid
    body, shorter and ragged at the top and bottom edges, running dry
    (gaps, loose strands) toward the ends. Drawn at 2x, then shrunk."""
    w, h = size_wh
    rng = np.random.default_rng(seed)
    ss = 2
    big = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(big)
    for rx, ry, rw, rh in rects:
        cx, cy = pad + rx + rw / 2.0, pad + ry + rh / 2.0
        half_w = rw / 2.0 + 0.08 * rw + 0.35 * rh
        half_h = rh / 2.0 + 0.28 * rh
        n = max(24, int(2 * half_h / max(2.0, 0.035 * rh)))       # bristles
        step = 2 * half_h / n
        slope = float(rng.uniform(-0.03, 0.03))
        wave = max(1.0, 0.6 * rh)
        for k in range(n):
            v = -1.0 + (k + 0.5) * 2.0 / n                          # -1 top .. 1 bottom
            edge = abs(v)
            reach = 1.0 - 0.45 * edge ** 3 + float(rng.normal(0, 0.05))
            swipe = -0.10 * half_w * v                              # slanted ends, like a quick swipe
            left = cx + swipe - half_w * max(0.3, reach + float(rng.uniform(-0.04, 0.10)))
            right = cx + swipe + half_w * max(0.3, reach + float(rng.uniform(-0.02, 0.22)))
            y_mid = cy + v * half_h + float(rng.normal(0, step * 0.2))
            thick = max(1, int(step * float(rng.uniform(1.6, 2.6)) * ss))
            amp = step * float(rng.uniform(0.0, 0.5))
            phase = float(rng.uniform(0, 2 * math.pi))
            xs = np.linspace(left, right, 41)
            gap = 0                                                  # segments left to skip
            for xa, xb in zip(xs[:-1], xs[1:]):
                if gap:
                    gap -= 1
                    continue
                u = abs((xa + xb) / 2 - cx) / half_w                # 0 centre .. 1 end
                # the brush runs dry toward its ends (and a little at the
                # very top and bottom): long gaps there, a solid body
                dry = 0.6 * max(0.0, u - 0.7) + 0.04 * edge ** 4
                if rng.random() < dry:
                    gap = int(rng.integers(1, 5))
                    continue
                ya = y_mid + amp * math.sin(phase + xa / wave) + slope * (xa - cx)
                yb = y_mid + amp * math.sin(phase + xb / wave) + slope * (xb - cx)
                d.line([(xa * ss, ya * ss), (xb * ss, yb * ss)],
                       fill=int(rng.uniform(215, 255)), width=thick)
    small = big.resize((w, h), Image.LANCZOS)
    return small.filter(ImageFilter.GaussianBlur(1.0))


def _sparkle_layer(size_wh: tuple[int, int], rects: list[tuple[int, int, int, int]],
                   pad: int, count: int, color: str, seed: int) -> Image.Image:
    """RGBA: a few small glints (four-point stars with a soft gold halo),
    mostly on the upper part of the letters, some past the line ends."""
    w, h = size_wh
    rng = np.random.default_rng(seed + 7)
    core = Image.new("L", (w, h), 0)
    halo = Image.new("L", (w, h), 0)
    dc, dh = ImageDraw.Draw(core), ImageDraw.Draw(halo)
    for _ in range(min(12, int(count))):
        rx, ry, rw, rh = rects[int(rng.integers(len(rects)))]
        if rng.random() < 0.7:                                       # on the letters
            x = pad + rx + float(rng.uniform(0.02, 0.98)) * rw
            y = pad + ry + float(rng.uniform(-0.08, 0.35)) * rh
        else:                                                        # past an end
            x = pad + rx + (float(rng.uniform(-0.08, -0.01)) if rng.random() < 0.5
                            else float(rng.uniform(1.01, 1.08))) * rw
            y = pad + ry + float(rng.uniform(0.2, 0.8)) * rh
        arm = rh * float(rng.uniform(0.05, 0.14))
        t = max(0.8, arm * 0.08)
        dc.polygon([(x - arm, y), (x, y - t), (x + arm, y), (x, y + t)], fill=255)
        dc.polygon([(x, y - arm * 0.75), (x + t, y), (x, y + arm * 0.75), (x - t, y)],
                   fill=255)
        r = arm * 0.45
        dh.ellipse([x - r, y - r, x + r, y + r], fill=150)
    line_h = max(r[3] for r in rects)
    halo = halo.filter(ImageFilter.GaussianBlur(max(1.0, 0.025 * line_h)))
    core = core.filter(ImageFilter.GaussianBlur(0.6))
    gold = _parse_color(color, (255, 214, 120, 255))
    layer = _tint(halo, (gold[0], gold[1], gold[2], 255))
    return Image.alpha_composite(layer, _tint(core, (255, 248, 226, 255)))


def _glow_tone(color: str) -> str:
    """A warm, saturated version of a letter colour, for its glow."""
    import colorsys
    r, g, b, _a = _parse_color(color, (255, 210, 120, 255))
    hh, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    rr, gg, bb = colorsys.hsv_to_rgb(hh, 0.8, 1.0)
    return f"#{int(rr * 255):02x}{int(gg * 255):02x}{int(bb * 255):02x}"


def render_title(image_path: str, text: str, spec: StyleSpec, out_path: str,
                 fonts_dir: Path | None = None) -> str:
    """Render a styled (Khmer-shaping-correct) title onto a poster image.

    Lines: see title_lines() ("|" = new line) and _line_scales(). Effects
    order: ink backdrop -> shadow -> glow -> extrude -> stroke -> fill (solid,
    or a 2-4 stop gradient over each line) -> bevel highlight -> sparkles.
    Saves to out_path (same format as extension implies, quality 95)."""
    base = _open_image(image_path).convert("RGBA")
    img_w, img_h = base.size

    font_path = _resolve_font(spec.font, fonts_dir)
    size = max(10, int(round(spec.size_ratio * img_w)))
    lines = title_lines(text, spec.line_break) or [str(text or "").strip() or " "]
    scales = _line_scales(lines, spec)
    sizes = [max(8, int(round(size * k))) for k in scales]
    mask, rects = _title_block(lines, font_path, sizes, spec.letter_spacing, spec.align)
    mw, mh = mask.size
    max_w = max(64, img_w * 4)                 # a runaway size stays bounded
    if mw > max_w:
        f = max_w / mw
        mask = mask.resize((max_w, max(1, int(mh * f))), Image.LANCZOS)
        rects = [(int(x * f), int(y * f), max(1, int(rw * f)), max(1, int(rh * f)))
                 for x, y, rw, rh in rects]
        mw, mh = mask.size
        size = max(8, int(size * f))

    stroke_px = int(round(spec.stroke_width_ratio * size))
    stroke_px = max(0, min(stroke_px, 25, max(1, size // 2)))
    sdx, sdy = int(spec.shadow_offset[0]), int(spec.shadow_offset[1])
    glow_r = int(spec.glow_radius) if spec.glow_color else 0
    extrude = max(0, int(spec.extrude_depth))
    line_h = max(r[3] for r in rects)
    backdrop_room = int(0.12 * mw + 0.45 * line_h) if spec.backdrop else 0
    sparkle_room = int(0.4 * line_h) if spec.sparkles else 0

    pad = int(max(abs(sdx) + spec.shadow_blur * 2,
                  abs(sdy) + spec.shadow_blur * 2,
                  glow_r * 3, stroke_px + 2, extrude + 2, 8,
                  backdrop_room, sparkle_room)) + 4
    lw, lh = mw + pad * 2, mh + pad * 2
    mask_full = Image.new("L", (lw, lh), 0)
    mask_full.paste(mask, (pad, pad))
    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    seed = zlib.crc32(str(text).encode("utf-8")) & 0xFFFF

    # --- ink-brush backdrop (bottom-most) ---
    if spec.backdrop:
        ink = _ink_backdrop((lw, lh), rects, pad, seed)
        layer = Image.alpha_composite(
            layer, _tint(ink, _parse_color(spec.backdrop_color, (11, 8, 5, 232))))

    # --- shadow ---
    if spec.shadow_blur > 0 or sdx or sdy:
        shadow_rgba = _parse_color(spec.shadow_color, (0, 0, 0, 204))
        if shadow_rgba[3] > 0:
            src = mask_full
            if stroke_px > 0:
                src = mask_full.filter(ImageFilter.MaxFilter(2 * stroke_px + 1))
            sh = ImageChops.offset(src, sdx, sdy)
            if spec.shadow_blur > 0:
                sh = sh.filter(ImageFilter.GaussianBlur(spec.shadow_blur))
            layer = Image.alpha_composite(layer, _tint(sh, shadow_rgba))

    # --- glow ---
    if spec.glow_color and glow_r > 0:
        glow_rgba = _parse_color(spec.glow_color, (255, 255, 255, 255))
        gl = mask_full.filter(ImageFilter.GaussianBlur(glow_r))
        glow_tint = _tint(gl, glow_rgba)
        layer = Image.alpha_composite(layer, glow_tint)
        layer = Image.alpha_composite(layer, glow_tint)  # boost intensity

    # --- extrude (repeated offset stamps, deepest first) ---
    if extrude > 0:
        extrude_tint = _tint(mask_full, _parse_color(spec.extrude_color,
                                                     (34, 34, 34, 255)))
        for i in range(extrude, 0, -1):
            layer = _stamp(layer, extrude_tint, i, i)

    # --- stroke (dilated mask) ---
    if stroke_px > 0:
        dilated = mask_full.filter(ImageFilter.MaxFilter(2 * stroke_px + 1))
        layer = Image.alpha_composite(
            layer, _tint(dilated, _parse_color(spec.stroke_color, (0, 0, 0, 255))))

    # --- fill: a gradient over each line (2-4 stops), or solid ---
    if spec.gradient and len(spec.gradient) >= 2:
        spans = [(pad + ry, pad + ry + rh) for _rx, ry, _rw, rh in rects]
        fill_layer = _gradient_layer((lw, lh), list(spec.gradient), spans, mask_full)
    else:
        fill_layer = _tint(mask_full, _parse_color(spec.fill, (255, 255, 255, 255)))
    layer = Image.alpha_composite(layer, fill_layer)

    # --- bevel: emboss-style highlight/shade from shifted mask differences ---
    if spec.bevel:
        off = max(1, size // 40)
        hi = ImageChops.subtract(mask_full, ImageChops.offset(mask_full, off, off))
        hi = hi.filter(ImageFilter.GaussianBlur(max(1, off // 2)))
        hi = hi.point(lambda v: (v * 190) // 255)
        layer = Image.alpha_composite(layer, _tint(hi, (255, 255, 255, 255)))
        lo = ImageChops.subtract(mask_full, ImageChops.offset(mask_full, -off, -off))
        lo = lo.filter(ImageFilter.GaussianBlur(max(1, off // 2)))
        lo = lo.point(lambda v: (v * 120) // 255)
        layer = Image.alpha_composite(layer, _tint(lo, (0, 0, 0, 255)))

    # --- sparkles (top-most) ---
    if spec.sparkles > 0:
        tone = spec.gradient[0] if spec.gradient else spec.fill
        layer = Image.alpha_composite(
            layer, _sparkle_layer((lw, lh), rects, pad, spec.sparkles, _glow_tone(tone), seed))

    # --- rotation ---
    if abs(spec.rotation) > 0.01:
        layer = layer.rotate(spec.rotation, expand=True, resample=Image.BICUBIC)

    # --- composite at spec.pos (fractions of W/H, layer centered) ---
    lw, lh = layer.size
    cx = spec.pos[0] * img_w
    cy = spec.pos[1] * img_h
    _alpha_paste(base, layer, int(round(cx - lw / 2)), int(round(cy - lh / 2)))

    _save_image(base, out_path, quality=95)
    return out_path


def fit_size_ratio(text: str, spec: StyleSpec, box_w: float, box_h: float,
                   img_w: int, fonts_dir: Path | None = None) -> float:
    """The size_ratio at which `text`, laid out as `spec` says, just fills a
    box_w x box_h area (pixels) of a picture img_w wide."""
    font_path = _resolve_font(spec.font, fonts_dir)
    lines = title_lines(text, spec.line_break) or [str(text or "").strip() or " "]
    trial = 100
    sizes = [max(8, int(round(trial * k))) for k in _line_scales(lines, spec)]
    mask, _rects = _title_block(lines, font_path, sizes, spec.letter_spacing)
    bw, bh = mask.size
    f = min(max(1.0, box_w) / max(1, bw), max(1.0, box_h) / max(1, bh))
    return float(min(0.5, max(0.02, trial * f / max(1, img_w))))


def gold_glow_radius(size_ratio: float, img_w: int) -> int:
    """The gold brush glow for a title of this size: a soft rim (about a
    tenth of the letter size), so the ink stays black between the letters."""
    return int(max(4, min(40, round(0.08 * size_ratio * img_w))))


def gold_brush_spec() -> StyleSpec:
    """A cinematic drama-poster title (like the "Reborn in the Qin Dynasty"
    poster): metallic gold letters with a dark outline and a warm glow, on
    a black ink-brush stroke, with a few sparkles; a lead-in ending in ":"
    goes on a smaller first line."""
    return StyleSpec(
        font="Moul", size_ratio=0.085, fill="#ffd257",
        gradient=["#fff8d6", "#ffd257", "#c97c1c", "#ffe08f"],
        stroke_color="#2b1503", stroke_width_ratio=0.03,
        shadow_color="#000000b3", shadow_offset=(3, 5), shadow_blur=10,
        glow_color="#ff9d2e66", glow_radius=10, bevel=True,
        backdrop=True, backdrop_color="#0a0705eb", sparkles=12,
        pos=(0.5, 0.80), rotation=-3.0, line_break="auto")


# ---------------------------------------------------------------------------
# OCR title detection (PaddleOCR)
# ---------------------------------------------------------------------------

_PADDLE_OCR: Any = None
_PADDLE_LOCK = threading.Lock()


def _get_paddle_ocr() -> Any:
    global _PADDLE_OCR
    with _PADDLE_LOCK:
        if _PADDLE_OCR is not None:
            return _PADDLE_OCR
        cache = models_dir() / "paddleocr"
        cache.mkdir(parents=True, exist_ok=True)
        # Pin every home/cache env Paddle consults so models land under our
        # models dir instead of the user profile.
        os.environ["HOME"] = str(cache)
        os.environ["PADDLE_HOME"] = str(cache)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache)
        os.environ["PPOCR_HOME"] = str(cache)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        # no "checking connectivity to the model hosters" at every start
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise EngineNotAvailable(
                "PaddleOCR is not installed. Install it with: "
                f"{pip_install('paddleocr paddlepaddle')} "
                f"(import error: {e})",
                "PaddleOCR មិនទាន់បានដំឡើងទេ។ សូមដំឡើងជាមុនដោយប្រើពាក្យបញ្ជា៖ "
                f"{pip_install('paddleocr paddlepaddle')}") from e
        last_err: Exception | None = None
        # kwargs vary across paddleocr 2.x / 3.x — try richest first. 3.x:
        # no document straightening (UVDoc) or page turning: a poster is a
        # flat picture, those cost two more model downloads, and straightening
        # can give boxes that are not where the text is on the picture
        for kwargs in ({"lang": "ch", "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False, "use_textline_orientation": True},
                       {"lang": "ch", "use_angle_cls": True},
                       {"lang": "ch", "use_textline_orientation": True},
                       {"lang": "ch"}):
            try:
                _PADDLE_OCR = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError) as e:
                last_err = e
        if _PADDLE_OCR is None:
            raise AppError(
                f"Failed to initialize PaddleOCR: {last_err}",
                f"មិនអាចចាប់ផ្តើម PaddleOCR បានទេ៖ {last_err}")
        return _PADDLE_OCR


def _poly_to_rect(poly: Any) -> tuple[int, int, int, int]:
    arr = np.asarray(poly, dtype=np.float64).reshape(-1)
    if arr.size == 4:  # [x0, y0, x1, y1]
        x0, y0, x1, y1 = arr.tolist()
    else:
        pts = arr.reshape(-1, 2)
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return (int(round(min(x0, x1))), int(round(min(y0, y1))),
            max(1, int(round(abs(x1 - x0)))), max(1, int(round(abs(y1 - y0)))))


def _normalize_ocr_result(raw: Any) -> list[OcrBox]:
    """Normalize PaddleOCR output across 2.x (nested lists) and 3.x
    (dict-like OCRResult with rec_texts/rec_scores/rec_polys) shapes."""
    items: list[tuple[Any, str, float]] = []

    def from_dictlike(page: Any) -> bool:
        texts = polys = scores = None
        for key in ("rec_texts", "texts"):
            try:
                v = page[key]
                if v is not None:
                    texts = list(v)
                    break
            except Exception:
                pass
        if texts is None:
            return False
        for key in ("rec_polys", "dt_polys", "rec_boxes", "boxes"):
            try:
                v = page[key]
                if v is not None and len(v) > 0:
                    polys = list(v)
                    break
            except Exception:
                pass
        for key in ("rec_scores", "scores"):
            try:
                v = page[key]
                if v is not None:
                    scores = list(v)
                    break
            except Exception:
                pass
        for i, t in enumerate(texts):
            poly = polys[i] if polys is not None and i < len(polys) else None
            if poly is None:
                continue
            conf = 1.0
            if scores is not None and i < len(scores):
                try:
                    conf = float(scores[i])
                except (TypeError, ValueError):
                    conf = 1.0
            items.append((poly, str(t), conf))
        return True

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict) or (hasattr(node, "keys")
                                      and hasattr(node, "__getitem__")):
            from_dictlike(node)
            return
        if isinstance(node, (list, tuple)):
            # classic 2.x item: [poly_points, (text, conf)]
            if (len(node) == 2
                    and isinstance(node[0], (list, tuple, np.ndarray))
                    and isinstance(node[1], (list, tuple))
                    and len(node[1]) >= 1
                    and isinstance(node[1][0], str)):
                text = node[1][0]
                try:
                    conf = float(node[1][1]) if len(node[1]) > 1 else 1.0
                except (TypeError, ValueError):
                    conf = 1.0
                items.append((node[0], text, conf))
                return
            for child in node:
                walk(child)

    walk(raw)
    boxes: list[OcrBox] = []
    for poly, text, conf in items:
        text = (text or "").strip()
        if not text:
            continue
        try:
            x, y, w, h = _poly_to_rect(poly)
        except Exception:
            continue
        boxes.append(OcrBox(x=x, y=y, w=w, h=h, text=text,
                            conf=max(0.0, min(1.0, conf))))
    boxes.sort(key=lambda b: b.w * b.h * b.conf, reverse=True)
    return boxes


# PaddleOCR on this PC: "" = not known yet, "usable", or "unusable" (blocked
# by Smart App Control, not installed, or it loads but cannot read: "(Un-
# implemented) Convert..." of PaddlePaddle 3 on the CPU). Not known yet: the
# built-in reader answers at once and PaddleOCR is tried in the background
# (its first try downloads its models and took ~40 s here); from then on it
# is used when it works. The verdict is kept in models/paddleocr/
# unusable.json with PaddleOCR's version: a new version is tried again.
_PADDLE_STATE = ""
_PADDLE_UNUSABLE = ""                 # why, when "unusable"
_PADDLE_VERDICT_READ = False
_PADDLE_PROBE: "threading.Thread | None" = None
_PROBE_LOCK = threading.Lock()


def _paddle_versions() -> str:
    """The installed PaddleOCR / PaddlePaddle versions ("" = none), read
    without importing them (an import can be the blocked DLL load)."""
    from importlib import metadata
    out = []
    for dist in ("paddleocr", "paddlepaddle", "paddlepaddle-gpu"):
        try:
            out.append(f"{dist} {metadata.version(dist)}")
        except Exception:
            pass
    return ", ".join(out)


def _paddle_verdict_file() -> Path:
    return models_dir() / "paddleocr" / "unusable.json"


def _save_paddle_verdict(usable: bool, why: str = "") -> None:
    try:
        f = _paddle_verdict_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"versions": _paddle_versions(), "usable": usable,
                                 "why": why}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        log.debug("could not save the PaddleOCR verdict", exc_info=True)


def _mark_paddle_unusable(why: str) -> None:
    global _PADDLE_STATE, _PADDLE_UNUSABLE
    _PADDLE_STATE, _PADDLE_UNUSABLE = "unusable", why[:300] or "unusable"
    log.warning("PaddleOCR cannot be used here (%s): the built-in text reader reads "
                "the posters", _PADDLE_UNUSABLE)
    _save_paddle_verdict(False, _PADDLE_UNUSABLE)


def _mark_paddle_usable() -> None:
    global _PADDLE_STATE, _PADDLE_UNUSABLE
    _PADDLE_STATE, _PADDLE_UNUSABLE = "usable", ""
    log.info("PaddleOCR works here: it reads the posters from now on")
    _save_paddle_verdict(True)


def _paddle_state() -> str:
    """"usable", "unusable" or "" (not known yet): known this session, or
    saved earlier for the same PaddleOCR / PaddlePaddle versions."""
    global _PADDLE_STATE, _PADDLE_UNUSABLE, _PADDLE_VERDICT_READ
    if not _PADDLE_STATE and not _PADDLE_VERDICT_READ:
        _PADDLE_VERDICT_READ = True
        if not _paddle_versions():
            _PADDLE_STATE, _PADDLE_UNUSABLE = "unusable", "PaddleOCR is not installed"
            return _PADDLE_STATE
        try:
            saved = json.loads(_paddle_verdict_file().read_text(encoding="utf-8"))
            if saved.get("versions") == _paddle_versions():
                if saved.get("usable"):
                    _PADDLE_STATE = "usable"
                else:
                    _PADDLE_STATE = "unusable"
                    _PADDLE_UNUSABLE = str(saved.get("why") or "unusable")
        except Exception:
            pass
    return _PADDLE_STATE


def _paddle_unusable() -> str:
    """Why PaddleOCR is not used ("" = it is, or not known yet)."""
    return _PADDLE_UNUSABLE if _paddle_state() == "unusable" else ""


def _paddle_read(image_path: str) -> list[OcrBox]:
    """PaddleOCR's boxes for a picture; raises when it cannot load or read."""
    ocr = _get_paddle_ocr()
    try:
        try:
            raw = ocr.ocr(str(image_path), cls=True)
        except TypeError:
            raw = ocr.ocr(str(image_path))
    except CancelledError:
        raise
    except Exception as e:
        try:
            raw = ocr.predict(str(image_path))
        except Exception:
            # both ways fail: PaddleOCR itself does not work here (seen:
            # "(Unimplemented) Convert..." on every picture), not this picture
            raise AppError(f"loads, but fails to read: {e}",
                           f"ផ្ទុកបាន ប៉ុន្តែអានមិនបាន៖ {e}") from e
    return _normalize_ocr_result(raw)


def _start_paddle_probe(image_path: str) -> None:
    """Try PaddleOCR on a copy of the picture in the background, once: it
    is used from the next "Detect text" on if it works."""
    global _PADDLE_PROBE
    import shutil
    import tempfile
    with _PROBE_LOCK:
        if _PADDLE_PROBE is not None and _PADDLE_PROBE.is_alive():
            return
        tmp = tempfile.NamedTemporaryFile(prefix="rs_ocr_probe_",
                                          suffix=Path(image_path).suffix or ".png",
                                          delete=False)
        tmp.close()
        shutil.copy2(image_path, tmp.name)

        def probe() -> None:
            try:
                _paddle_read(tmp.name)
            except Exception as e:            # ImportError / "DLL load failed" / ...
                _mark_paddle_unusable(str(getattr(e, "en", "") or e) or type(e).__name__)
            else:
                _mark_paddle_usable()
            finally:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass

        LOG_BUS.publish("[ocr] checking PaddleOCR in the background (its first use downloads "
                        "its models); the built-in text reader answers meanwhile")
        _PADDLE_PROBE = threading.Thread(target=probe, daemon=True, name="paddleocr-probe")
        _PADDLE_PROBE.start()


def _detect_with_paddle(image_path: str, progress: Progress,
                        cancel: "threading.Event | None") -> list[OcrBox] | None:
    """PaddleOCR's boxes when it is known to work here; None otherwise (the
    caller then uses the built-in reader). Not known yet: a background try."""
    state = _paddle_state()
    if state == "":
        _start_paddle_probe(image_path)
        return None
    if state != "usable":
        return None
    _report(progress, 0.35, "Detecting text…")
    try:
        boxes = _paddle_read(image_path)
    except CancelledError:
        raise
    except Exception as e:
        _mark_paddle_unusable(str(getattr(e, "en", "") or e) or type(e).__name__)
        return None
    _check_cancel(cancel)
    return boxes


def _detect_with_reader(image_path: str, progress: Progress,
                        cancel: "threading.Event | None") -> list[OcrBox]:
    """The built-in text reader (PP-OCRv5 as ONNX, the hard-subtitle reader):
    onnxruntime + OpenCV only, so Smart App Control does not block it. Its
    21 MB of models are downloaded the first time."""
    from app.engines import subtitle_ocr as so
    if not so.models_present():
        _report(progress, 0.1, "Downloading the text reader (21 MB)…")
        from app.core import model_manager
        model_manager.download_model(so.MODEL_ID, cancel=cancel)
    _check_cancel(cancel)
    _report(progress, 0.35, "Detecting text (built-in reader)…")
    img = _load_bgr(image_path)
    found: list[OcrBox] = []
    for side in _reader_sides(img.shape[1], img.shape[0]):
        found += [OcrBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0), text=text,
                         conf=max(0.0, min(1.0, conf)))
                  for x0, y0, x1, y1, text, conf in so.read_image(img, max_side=side)]
        _check_cancel(cancel)
    return _merge_reads(found)


# the text is looked for on copies this big (longest side, px): the reader
# misses a title whose letters are much bigger than the text it learned on
# (a big brush title, "DATA ANALYSIS" over the whole width) on the full
# picture, and finds and reads it whole on the small copies
_READ_SIDES = (1280, 640, 352)


def _reader_sides(w: int, h: int) -> list[int]:
    sides = [min(_READ_SIDES[0], max(w, h))]
    for s in _READ_SIDES[1:]:
        if s < 0.85 * sides[-1]:
            sides.append(s)
    return sides


def _overlap(a: OcrBox, b: OcrBox) -> int:
    ix = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    iy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return ix * iy if ix > 0 and iy > 0 else 0


def _merge_reads(found: list[OcrBox]) -> list[OcrBox]:
    """One box per piece of text from the reads at several sizes. A box
    over two stacked lines (the small copies join a title's lines) gives
    way to the lines; of boxes on the same text the surer, bigger one
    stays, the others' readings kept as its alternatives."""
    lines: list[OcrBox] = []
    for b in found:
        # lines inside it (bits of a letter, a "-" read alone, are not lines)
        held = [o for o in found if o is not b and 0.3 * b.h <= o.h < 0.75 * b.h
                and _overlap(o, b) >= 0.7 * o.w * o.h]
        stacked = any(min(p.y + p.h, q.y + q.h) - max(p.y, q.y) < 0.3 * min(p.h, q.h)
                      and max(p.y + p.h, q.y + q.h) - min(p.y, q.y) >= 0.7 * b.h
                      for i, p in enumerate(held) for q in held[i + 1:])
        if not stacked:
            lines.append(b)
    lines.sort(key=lambda b: b.conf * math.sqrt(b.w * b.h), reverse=True)
    out: list[OcrBox] = []
    for b in lines:
        twin = next((k for k in out
                     if _overlap(b, k) >= 0.6 * min(b.w * b.h, k.w * k.h)), None)
        if twin is None:
            out.append(OcrBox(b.x, b.y, b.w, b.h, b.text, b.conf, tuple(b.alts)))
        elif b.text and b.text != twin.text and b.text not in twin.alts:
            twin.alts = twin.alts + (b.text,)
    return out


def _load_bgr(image_path: str) -> np.ndarray:
    """A picture as BGR uint8 (paths with Khmer / Chinese letters too)."""
    import cv2
    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        with Image.open(image_path) as im:
            img = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
    return img


def detect_title_boxes(image_path: str, progress: Progress = None,
                       cancel: "threading.Event | None" = None) -> list[OcrBox]:
    """Detect text boxes on a poster: PaddleOCR (lang='ch' covers zh+en)
    when it runs on this PC, else the built-in text reader (PaddleOCR can
    be blocked by Smart App Control: "DLL load failed"). Sorted by
    area*confidence descending (title candidates first)."""
    _check_cancel(cancel)
    if not Path(image_path).is_file():
        raise AppError(
            f"Image not found: {image_path}",
            f"រកមិនឃើញរូបភាព៖ {image_path}")
    _report(progress, 0.05, "Loading OCR engine…")
    boxes = _detect_with_paddle(image_path, progress, cancel)
    engine = "PaddleOCR"
    if boxes is None:
        engine = "built-in reader"
        try:
            boxes = _detect_with_reader(image_path, progress, cancel)
        except CancelledError:
            raise
        except Exception as e:
            why = f" PaddleOCR: {_PADDLE_UNUSABLE}." if _PADDLE_UNUSABLE else ""
            raise AppError(
                f"Text detection is not available on this PC.{why} The built-in text "
                f"reader could not run either: {e}",
                f"មិនអាចរកអក្សរនៅលើកុំព្យូទ័រនេះបានទេ។{why} កម្មវិធីអានអក្សរដែលភ្ជាប់មកជាមួយ "
                f"ក៏មិនអាចដំណើរការបានដែរ៖ {e}") from e
    boxes.sort(key=lambda b: b.w * b.h * b.conf, reverse=True)
    _report(progress, 1.0, f"Found {len(boxes)} text boxes")
    log.info("detect_title_boxes: %d boxes on %s (%s)", len(boxes), image_path, engine)
    return boxes


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------

_LAMA_SESSIONS: dict[str, Any] = {}
_LAMA_LOCK = threading.Lock()


def _lama_session(model_path: Path) -> Any:
    """The LaMa model, loaded once (200 MB: loading takes seconds)."""
    try:
        import onnxruntime as ort
    except Exception as e:
        raise EngineNotAvailable(
            f"onnxruntime is not installed (needed for LaMa inpainting): {e}",
            "onnxruntime មិនទាន់បានដំឡើងទេ (ត្រូវការសម្រាប់ការលុបអក្សរដោយ LaMa)។"
        ) from e
    key = str(model_path)
    with _LAMA_LOCK:
        sess = _LAMA_SESSIONS.get(key)
        if sess is None:
            # CUDA (NVIDIA) > DirectML (AMD / Intel / any DX12 GPU, needs the
            # onnxruntime-directml package) > CPU
            from app.core.gpu import onnx_providers
            providers = onnx_providers()
            try:
                sess = ort.InferenceSession(key, providers=providers)
            except Exception as e:
                if providers == ["CPUExecutionProvider"]:
                    raise
                log.warning("LaMa on %s failed (%s); using CPU", providers[0], e)
                sess = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
            _LAMA_SESSIONS[key] = sess
    return sess


def _lama_run(sess: Any, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One LaMa pass: RGB uint8 + mask of the same size -> RGB uint8."""
    names = [i.name for i in sess.get_inputs()]
    mask_name = next((n for n in names if "mask" in n.lower()),
                     names[-1] if len(names) > 1 else names[0])
    img_name = next((n for n in names if n != mask_name), names[0])
    mask_in = (mask > 0).astype(np.float32)[None, None]
    # the masked pixels blanked, as LaMa was trained: shown the old letters
    # under the mask it drew a faint copy of them back
    img_in = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None] * (1.0 - mask_in)
    out = np.asarray(sess.run(None, {img_name: img_in, mask_name: mask_in})[0])[0]
    if out.ndim == 3 and out.shape[0] in (1, 3):
        out = np.transpose(out, (1, 2, 0))
    if out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=-1)
    if float(out.max(initial=0.0)) <= 2.0:
        out = out * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _mask_regions(mask: np.ndarray, join: int = 40) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of each part of the mask; parts closer than `join`
    px share one."""
    import cv2
    near = cv2.dilate((mask > 0).astype(np.uint8), np.ones((join | 1, join | 1), np.uint8))
    _n, _lbl, stats, _c = cv2.connectedComponentsWithStats(near, 8)
    rects = [(int(x), int(y), int(x + bw), int(y + bh)) for x, y, bw, bh, _a in stats[1:]]
    merged = True
    while merged:
        merged = False
        out: list[tuple[int, int, int, int]] = []
        for r in rects:
            for i, o in enumerate(out):
                if r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3]:
                    out[i] = (min(r[0], o[0]), min(r[1], o[1]), max(r[2], o[2]), max(r[3], o[3]))
                    merged = True
                    break
            else:
                out.append(r)
        rects = out
    return rects


def _crop_square(arr: np.ndarray, x0: int, y0: int, side: int,
                 mirror: bool = True) -> np.ndarray:
    """arr[y0:y0+side, x0:x0+side]; where that leaves the picture it is
    mirrored (or zero when mirror is False)."""
    h, w = arr.shape[:2]
    sub = arr[max(0, y0):min(h, y0 + side), max(0, x0):min(w, x0 + side)]
    pads = ((max(0, -y0), max(0, y0 + side - h)), (max(0, -x0), max(0, x0 + side - w)))
    if not any(a or b for a, b in pads):
        return sub
    if arr.ndim == 3:
        pads = pads + ((0, 0),)
    return np.pad(sub, pads, mode="symmetric" if mirror else "constant")


def _inpaint_lama(arr: np.ndarray, mask: np.ndarray, model_path: Path,
                  cancel: "threading.Event | None" = None) -> np.ndarray:
    """Fill the masked pixels with LaMa. The usual LaMa export takes only
    512 x 512 pictures (fed a whole poster it refused, and the app silently
    used OpenCV's much weaker fill): each part of the mask is filled from a
    square around it with room on every side, shrunk to 512 when larger."""
    import cv2
    sess = _lama_session(model_path)
    _check_cancel(cancel)
    shape = sess.get_inputs()[0].shape
    side = shape[2] if (len(shape) == 4 and isinstance(shape[2], int)
                        and shape[2] == shape[3]) else None
    h, w = arr.shape[:2]
    if side is None:                    # a model that takes any size: all at once
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        img_p = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        mask_p = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="edge")
        out = _lama_run(sess, img_p, mask_p)[:h, :w]
        result = arr.copy()
        sel = mask > 0
        result[sel] = out[sel]
        return result
    result = arr.copy()
    todo = ((mask > 0).astype(np.uint8)) * 255      # holes not filled yet
    for x0, y0, x1, y1 in _mask_regions(mask):
        _check_cancel(cancel)
        rw, rh = x1 - x0, y1 - y0
        room = max(24, int(0.2 * max(rw, rh)))
        need = max(rw, rh) + 2 * room
        # how far into a hole LaMa must reach; its window must see around
        # every hole (filled at full size, a thick hole got a made-up gold
        # smear): shrunk only as far as that needs — thin letters at the
        # picture's own resolution
        thick = float(cv2.distanceTransform(
            np.pad((todo[y0:y1, x0:x1] > 0).astype(np.uint8), 1), cv2.DIST_L2, 5).max())
        scale = min(1.0, 0.12 * side / max(1.0, thick))
        if need <= side or need * scale <= side:
            # one square around the hole (shrunk to `side` when bigger)
            sq = max(side, need)
            _lama_square(sess, result, todo, (x0 + x1) // 2 - sq // 2,
                         (y0 + y1) // 2 - sq // 2, sq, side, (0, 0, w, h))
            continue
        # a big title at the picture's own resolution (shrunk to 512 a
        # 1500 px title area came back soft): squares of side/scale px
        # across it, each filling the holes in its middle part; the parts
        # filled first are what the next square sees around its holes
        tile = int(round(side / scale))
        step = max(32, int(0.8 * tile))
        inset = (tile - step) // 2
        for ky in range(max(1, -(-rh // step))):
            for kx in range(max(1, -(-rw // step))):
                _check_cancel(cancel)
                fx0, fy0 = x0 + kx * step, y0 + ky * step
                fill = (fx0, fy0, min(x1, fx0 + step), min(y1, fy0 + step))
                if not todo[fill[1]:fill[3], fill[0]:fill[2]].any():
                    continue
                _lama_square(sess, result, todo, fx0 - inset, fy0 - inset, tile, side, fill)
    return result


def _restore_detail(orig: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The filled holes sharpened to look like the rest of the poster. LaMa
    paints a large hole with soft shapes (at any size: its fine grain
    matches, its edges do not), and the erased area looked smudged next to
    the sharp picture. Each filled area is sharpened (unsharp mask) just
    enough that its edges are about as crisp as those around it (0.6 of
    their Laplacian energy; at most 2.5x)."""
    import cv2
    hole = (mask > 0).astype(np.uint8)
    if not hole.any():
        return filled
    H, W = hole.shape
    out = filled.copy()

    def crisp(a: np.ndarray, where: np.ndarray) -> float:
        g = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F)[where].var())

    for x0, y0, x1, y1 in _mask_regions(mask):
        pad = max(32, int(0.2 * max(x1 - x0, y1 - y0)))
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        hm = hole[Y0:Y1, X0:X1] > 0
        clear = cv2.dilate(hm.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
        if np.count_nonzero(clear) < 2000 or np.count_nonzero(hm) < 200:
            continue
        want = 0.6 * crisp(orig[Y0:Y1, X0:X1], clear)
        fl = out[Y0:Y1, X0:X1].astype(np.float32)
        if crisp(fl, hm) >= want:
            continue
        mid = fl - cv2.GaussianBlur(fl, (0, 0), 2.0)
        amount = 2.5
        for k in (0.5, 1.0, 1.5, 2.0, 2.5):
            if crisp(fl + k * mid, hm) >= want:
                amount = k
                break
        soft = cv2.GaussianBlur(hm.astype(np.float32), (0, 0), 2.0)[..., None] *             (cv2.dilate(hm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)[..., None]
        out[Y0:Y1, X0:X1] = np.clip(fl + amount * mid * soft, 0, 255).astype(np.uint8)
    return out


def _lama_square(sess: Any, result: np.ndarray, todo: np.ndarray, sx: int, sy: int,
                 sq: int, side: int, fill: tuple[int, int, int, int]) -> None:
    """LaMa on the sq x sq square at (sx, sy) (shrunk to side x side when
    bigger); the holes of `todo` inside `fill` (x0, y0, x1, y1) take its
    output (edges feathered) and are marked filled."""
    import cv2
    h, w = result.shape[:2]
    crop = _crop_square(result, sx, sy, sq)
    # the mask mirrored with the picture: past the picture's edge the crop
    # holds a mirror copy of the title, and left unmasked LaMa drew the old
    # letters back from it (a title at the edge)
    cmask = _crop_square(todo, sx, sy, sq)
    if sq != side:
        small = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
        # a small pixel is masked when any of its pixels is: picking one
        # pixel of each (INTER_NEAREST) skipped bits of thin strokes, and
        # they came back as a faint copy of the old letters
        smask = (cv2.resize(cmask, (side, side), interpolation=cv2.INTER_AREA) > 0
                 ).astype(np.uint8) * 255
        smask = cv2.dilate(smask, np.ones((3, 3), np.uint8))   # no thin rim left
        out = cv2.resize(_lama_run(sess, small, smask), (sq, sq),
                         interpolation=cv2.INTER_CUBIC)
    else:
        out = _lama_run(sess, crop, cmask)
    ix0, iy0 = max(0, sx, fill[0]), max(0, sy, fill[1])
    ix1, iy1 = min(w, sx + sq, fill[2]), min(h, sy + sq, fill[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return
    part = out[iy0 - sy:iy1 - sy, ix0 - sx:ix1 - sx].astype(np.float32)
    m = (todo[iy0:iy1, ix0:ix1] > 0).astype(np.float32)
    soft = np.maximum(m, cv2.GaussianBlur(m, (0, 0), 1.2))[..., None]
    here = result[iy0:iy1, ix0:ix1].astype(np.float32)
    result[iy0:iy1, ix0:ix1] = np.clip(here * (1.0 - soft) + part * soft,
                                       0, 255).astype(np.uint8)
    todo[iy0:iy1, ix0:ix1] = 0


def _inpaint_pil_fallback(arr: np.ndarray, mask: np.ndarray,
                          boxes: list[OcrBox]) -> np.ndarray:
    """LAST-RESORT inpaint when neither LaMa nor OpenCV is available: fill
    each box with the median color of a surrounding ring, then blend a blurred
    patch through the (feathered) mask. Crude but never fails."""
    h, w = arr.shape[:2]
    result = arr.copy()
    for b in boxes:
        x0 = max(0, int(b.x) - 6)
        y0 = max(0, int(b.y) - 6)
        x1 = min(w, int(b.x + b.w) + 6)
        y1 = min(h, int(b.y + b.h) + 6)
        if x1 <= x0 or y1 <= y0:
            continue
        rx0, ry0 = max(0, x0 - 14), max(0, y0 - 14)
        rx1, ry1 = min(w, x1 + 14), min(h, y1 + 14)
        ring_sel = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
        ring_sel[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = False
        ring = arr[ry0:ry1, rx0:rx1][ring_sel]
        if ring.size:
            fill = np.median(ring.reshape(-1, 3), axis=0)
        else:
            fill = np.array([127.0, 127.0, 127.0])
        result[y0:y1, x0:x1] = fill.astype(np.uint8)
    img = Image.fromarray(result)
    blurred = img.filter(ImageFilter.MedianFilter(5)).filter(
        ImageFilter.GaussianBlur(8))
    feathered = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(5))
    return np.asarray(Image.composite(blurred, img, feathered))


def inpaint_region(image_path: str, boxes: list[OcrBox], out_path: str,
                   method: str = "auto", progress: Progress = None,
                   cancel: "threading.Event | None" = None) -> str:
    """Remove text regions from a poster.

    Mask = boxes dilated by 6 px. method='auto' tries LaMa onnx
    (models_dir()/lama/lama_fp32.onnx via onnxruntime), then cv2.inpaint
    TELEA (radius 3), then a documented last-resort PIL median/blur patch.
    method may also be 'lama', 'telea', or 'blur' to force a backend."""
    _check_cancel(cancel)
    img = _open_image(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x0 = max(0, int(b.x) - 6)
        y0 = max(0, int(b.y) - 6)
        x1 = min(w, int(b.x + b.w) + 6)
        y1 = min(h, int(b.y + b.h) + 6)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if not mask.any():
        _save_image(img, out_path, quality=95)
        _report(progress, 1.0, "Nothing to inpaint")
        return out_path
    result = _inpaint_array(arr, mask, boxes, method, progress, cancel)
    _check_cancel(cancel)
    _save_image(Image.fromarray(result), out_path, quality=95)
    _report(progress, 1.0, "Inpaint complete")
    return out_path


def _inpaint_array(arr: np.ndarray, mask: np.ndarray, boxes: list[OcrBox],
                   method: str = "auto", progress: Progress = None,
                   cancel: "threading.Event | None" = None) -> np.ndarray:
    """The picture with the masked pixels filled: LaMa (method 'auto' or
    'lama'), else OpenCV TELEA ('telea'), else a basic patch ('blur';
    `boxes` say where)."""
    method = (method or "auto").strip().lower()
    result: np.ndarray | None = None
    soft_errors: list[str] = []

    if method in ("auto", "lama"):
        lama_path = models_dir() / "lama" / "lama_fp32.onnx"
        if lama_path.is_file():
            try:
                _report(progress, 0.2, "Inpainting with LaMa…")
                result = _restore_detail(arr, _inpaint_lama(arr, mask, lama_path, cancel), mask)
            except CancelledError:
                raise
            except EngineNotAvailable as e:
                if method == "lama":
                    raise
                soft_errors.append(e.en)
            except Exception as e:
                if method == "lama":
                    raise AppError(
                        f"LaMa inpainting failed: {e}",
                        f"ការលុបអក្សរដោយ LaMa បរាជ័យ៖ {e}") from e
                soft_errors.append(str(e))
        elif method == "lama":
            raise EngineNotAvailable(
                f"LaMa model not found at {lama_path}. Download it from "
                f"Settings → Models.",
                f"រកមិនឃើញម៉ូដែល LaMa នៅ {lama_path} ទេ។ សូមទាញយកតាម "
                f"ការកំណត់ → ម៉ូដែល។")

    if result is None and method in ("auto", "telea", "cv2"):
        try:
            import cv2
            _check_cancel(cancel)
            _report(progress, 0.55, "Inpainting (OpenCV TELEA)…")
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            out = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
            result = _restore_detail(arr, cv2.cvtColor(out, cv2.COLOR_BGR2RGB), mask)
        except CancelledError:
            raise
        except ImportError as e:
            if method != "auto":
                raise EngineNotAvailable(
                    "OpenCV is not installed. Install it with: "
                    f"{pip_install('opencv-python-headless')}",
                    "OpenCV មិនទាន់បានដំឡើងទេ។ សូមដំឡើងជាមុនដោយ៖ "
                    f"{pip_install('opencv-python-headless')}") from e
            soft_errors.append(str(e))
        except Exception as e:
            if method != "auto":
                raise AppError(
                    f"OpenCV inpainting failed: {e}",
                    f"ការលុបអក្សរដោយ OpenCV បរាជ័យ៖ {e}") from e
            soft_errors.append(str(e))

    if result is None:
        _check_cancel(cancel)
        _report(progress, 0.75, "Inpainting (basic patch fallback)…")
        result = _inpaint_pil_fallback(arr, mask, boxes)
        if soft_errors:
            log.warning(
                "inpaint_region fell back to basic PIL patch "
                "(LaMa/OpenCV unavailable): %s", "; ".join(soft_errors))

    if soft_errors:
        log.info("inpaint: LaMa not used (%s)", "; ".join(soft_errors))
    return result


# ---------------------------------------------------------------------------
# Replace the title, same style
# ---------------------------------------------------------------------------

def _scaled(progress: Progress, lo: float, hi: float) -> Progress:
    if progress is None:
        return None
    return lambda f, t: progress(lo + (hi - lo) * max(0.0, min(1.0, f)), t)


def _hex(c: Any) -> str:
    r, g, b = (int(max(0, min(255, round(float(v))))) for v in list(c)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _lum(c: Any) -> np.ndarray:
    a = np.asarray(c, dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def group_title_boxes(boxes: list[OcrBox], img_w: int, img_h: int) -> list[OcrBox]:
    """The boxes of the poster's main title: the most prominent text big
    enough to be a title, with the text right next to it of a similar size
    (a two-line title). [] when there is none (small print, a watermark)."""
    cands = [b for b in boxes
             if b.h >= 0.035 * img_h and b.w >= 0.05 * img_w and b.conf >= 0.3]
    if not cands:
        return []
    main = max(cands, key=lambda b: b.h * min(b.w, 8 * b.h) * max(0.3, b.conf))
    group = [main]
    grown = True
    while grown:
        grown = False
        for b in cands:
            if any(b is g for g in group) or b.h < 0.4 * main.h:
                continue
            for g in group:
                gap = 0.6 * max(g.h, b.h)
                if (b.x - gap < g.x + g.w and g.x - gap < b.x + b.w
                        and b.y - gap < g.y + g.h and g.y - gap < b.y + b.h):
                    group.append(b)
                    grown = True
                    break
    return group


def title_text(image_path: str, progress: Progress = None,
               cancel: "threading.Event | None" = None) -> str:
    """The poster's title as the text reader reads it (its lines joined by
    spaces); "" when the poster has none."""
    boxes = detect_title_boxes(image_path, progress=progress, cancel=cancel)
    w, h = _open_image(image_path).size
    rows = _box_rows(group_title_boxes(boxes, w, h))
    return " ".join(" ".join(b.text for b in sorted(r, key=lambda b: b.x)) for r in rows).strip()


def _box_rows(group: list[OcrBox]) -> list[list[OcrBox]]:
    """The title's boxes by line, top to bottom."""
    rows: list[list[OcrBox]] = []
    for b in sorted(group, key=lambda b: b.y + b.h / 2):
        cy = b.y + b.h / 2
        if rows:
            top = min(r.y for r in rows[-1])
            bottom = max(r.y + r.h for r in rows[-1])
            mid = (top + bottom) / 2
            if abs(cy - mid) < 0.35 * (bottom - top):
                rows[-1].append(b)
                continue
        rows.append([b])
    return rows


def _row_heights(text: np.ndarray, rows: list[list[OcrBox]]) -> list[float]:
    """Letter height of each line of a title: the letters' rows are split
    where the fewest letter pixels are, between the lines' middles."""
    on = text > 0
    prof = np.convolve(on.sum(axis=1).astype(np.float64), np.ones(9) / 9.0, mode="same")
    mids = []
    for row in rows:
        m = np.zeros(on.shape, bool)
        for b in row:
            m[b.y:b.y + b.h, b.x:b.x + b.w] = True
        ys, _xs = np.nonzero(on & m)
        mids.append(float(np.median(ys)) if len(ys) else float(np.mean([b.y + b.h / 2 for b in row])))
    cuts = [0]
    for a, b in zip(mids[:-1], mids[1:]):
        lo, hi = int(min(a, b)), int(max(a, b))
        cuts.append(lo + int(np.argmin(prof[lo:hi + 1])) if hi > lo else lo)
    cuts.append(on.shape[0])
    heights = []
    for top, bottom in zip(cuts[:-1], cuts[1:]):
        ys, _xs = np.nonzero(on[top:bottom])
        heights.append(float(np.percentile(ys, 95) - np.percentile(ys, 5)) if len(ys) > 20 else 0.0)
    return heights


def _mask_reading(mask: np.ndarray, group: list[OcrBox]) -> float:
    """How well `mask`, drawn black on white, reads as the title's text
    (0–1: likeness × the reader's sureness, over the boxes by width)."""
    import cv2

    from app.engines import subtitle_ocr as so
    ocr = so._ocr()
    total = weight = 0.0
    for b in group:
        crop = mask[b.y:b.y + b.h, b.x:b.x + b.w]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        if crop.shape[0] >= 1.5 * crop.shape[1]:
            crop = np.ascontiguousarray(np.rot90(crop))
        pic = cv2.copyMakeBorder(np.where(crop > 0, 0, 255).astype(np.uint8),
                                 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
        got, sure = ocr.recognize(cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR))
        n = len(_norm_text(b.text))
        refs = [b.text] + [a for a in b.alts if len(_norm_text(a)) >= 0.8 * n]
        total += max(_text_like(got, r) for r in refs) * sure * b.w
        weight += b.w
    return total / weight if weight else 0.0


def _edge_touch(mask: np.ndarray, group: list[OcrBox]) -> float:
    """Share of the boxes' edges the mask covers. Letters seldom reach the
    edge of their box; the space around them does (a mask of the space
    reads as the title too: the letters are its holes). A box edge on the
    picture's own edge does not count: a title over the whole width
    ("DATA ANALYSIS" at the top) reaches it with its letters."""
    H, W = mask.shape[:2]
    on = total = 0
    for b in group:
        m = mask[b.y:b.y + b.h, b.x:b.x + b.w] > 0
        if m.shape[0] < 8 or m.shape[1] < 8:
            continue
        k = max(2, int(0.04 * min(m.shape)))
        edge = np.zeros(m.shape, bool)
        if b.y > 2:
            edge[:k] = True
        if b.y + b.h < H - 2:
            edge[-k:] = True
        if b.x > 2:
            edge[:, :k] = True
        if b.x + b.w < W - 2:
            edge[:, -k:] = True
        on += int(np.count_nonzero(m & edge))
        total += int(np.count_nonzero(edge))
    return on / total if total else 0.0


def _letters_read_checked(lab: np.ndarray, group: list[OcrBox], core: np.ndarray,
                          text: np.ndarray, min_area: int) -> np.ndarray:
    """The title's letters, checked by reading them: drawn black on white,
    they must read as the title. When the colours around the title are the
    letters' colours too (white letters by a white hoodie and paper) the
    colour mask holds the scene instead (a navy jacket) and reads as
    nothing: then the colour groups of the boxes that read most like the
    title are the letters (white and gold letters: two groups together).
    The colour mask stays whenever it reads as the title."""
    import itertools

    import cv2

    from app.engines import subtitle_ocr as so
    try:
        if not so.models_present():
            return text
        now = _mask_reading(text, group)
    except Exception:
        log.debug("could not read the title's letters back", exc_info=True)
        return text
    # partly readable (DATA ANALYSIS: 0.49, Qin: 0.66) is the title's own
    # colours: kept (the erase grows over the rest); the scene instead of
    # the letters reads as nothing (七日逆襲 by the navy jacket: 0.00)
    if now >= 0.35:
        return text
    inside = core > 0
    pix = lab[inside].reshape(-1, 3).astype(np.float32)
    if len(pix) < 500:
        return text
    cands: list[np.ndarray] = []
    light = lab[..., 0]
    thr, _ = cv2.threshold(light[inside].reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands += [(light > thr) & inside, (light <= thr) & inside]
    k = 5
    cv2.setRNGSeed(7)
    sample = pix[np.random.default_rng(7).choice(len(pix), min(len(pix), 20000),
                                                 replace=False)]
    _c, _l, centers = cv2.kmeans(
        sample, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        3, cv2.KMEANS_PP_CENTERS)
    groups = np.full(inside.shape, -1, np.int32)
    groups[inside] = np.linalg.norm(pix[:, None, :] - centers[None], axis=2).argmin(1)
    for r in (1, 2, 3):
        for combo in itertools.combinations(range(k), r):
            cands.append(np.isin(groups, combo))
    best, best_score = text, now
    open3 = np.ones((3, 3), np.uint8)
    n_in = max(1, int(np.count_nonzero(inside)))
    for cand in cands:
        m = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, open3)
        if not 0.04 <= np.count_nonzero(m) / n_in <= 0.6 or _edge_touch(m, group) > 0.35:
            continue
        score = _mask_reading(m, group)
        if score > best_score + 0.05:
            best, best_score = m, score
    if best is text or best_score < max(0.3, now + 0.1):
        return text
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(best, 8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    best = np.isin(lbl, big).astype(np.uint8) * 255
    inner = _enclosed(best, group)
    if np.count_nonzero(inner) >= 0.5 * np.count_nonzero(best):
        # the letters' outline reads as well as the letters: what it
        # encloses is their fill (the colours the new title takes); the
        # erase reaches over the outline (stroke_px)
        best = inner
    log.info("title letters: the colour mask read %.2f; the colour groups that read best "
             "%.2f were taken%s", now, best_score,
             " (an outline: its inside)" if best is inner else "")
    return best


def _enclosed(mask: np.ndarray, group: list[OcrBox]) -> np.ndarray:
    """The holes of `mask` inside the boxes no bigger than a letter: the
    inside of outlined letters."""
    import cv2
    out = np.zeros_like(mask)
    for b in group:
        crop = (mask[b.y:b.y + b.h, b.x:b.x + b.w] > 0).astype(np.uint8)
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        free = np.pad(1 - crop, 1, constant_values=1)
        flood = free.copy()
        cv2.floodFill(flood, None, (0, 0), 2)           # the space around the letters
        holes = ((flood == 1)[1:-1, 1:-1]).astype(np.uint8)
        n, lbl, stats, _c = cv2.connectedComponentsWithStats(holes, 8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= b.h * b.h]
        part = out[b.y:b.y + b.h, b.x:b.x + b.w]
        part[np.isin(lbl, keep)] = 255
    return out


def _title_layer(rgb: np.ndarray, group: list[OcrBox]) -> dict:
    """The title's own pixels in and around its boxes: the letters (their
    colours told apart from what surrounds the title), the dark ink band
    behind them when there is one, and what to erase (the letters with
    their outline and glow, and the band). The scene inside the boxes (a
    hand, soldiers) is kept."""
    import cv2
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    line_h = float(np.median([b.h for b in group]))
    grow = max(4, int(0.12 * line_h))
    core = np.zeros((h, w), np.uint8)
    region = np.zeros((h, w), np.uint8)
    for b in group:
        cv2.rectangle(core, (b.x, b.y), (b.x + b.w - 1, b.y + b.h - 1), 255, -1)
        cv2.rectangle(region, (max(0, b.x - grow), max(0, b.y - grow)),
                      (min(w - 1, b.x + b.w + grow), min(h - 1, b.y + b.h + grow)), 255, -1)
    k = max(15, int(0.25 * line_h)) | 1
    ring = cv2.dilate(region, np.ones((k, k), np.uint8)) & (255 - region)
    if np.count_nonzero(ring) < 200:            # the title fills the picture
        ring = 255 - core
    bins = 16
    q = (lab // (256 // bins)).astype(np.intp)

    def hist(sel: np.ndarray) -> np.ndarray:
        hst = cv2.calcHist([lab], [0, 1, 2], sel, [bins] * 3, [0, 256] * 3)
        return hst / max(1.0, float(hst.sum()))

    med = (np.median(lab[ring > 0].reshape(-1, 3), axis=0) if ring.any()
           else np.array([128.0, 128.0, 128.0]))
    dist = np.linalg.norm(lab.astype(np.float32) - med.astype(np.float32)[None, None, :],
                          axis=2)
    inside = core > 0
    thr = float(np.percentile(dist[inside], 80)) if inside.any() else 0.0
    # ">=": letters of one flat colour (pure white) all sit exactly at the
    # 80th percentile when they fill a fifth of the box or more; with ">"
    # none of them was taken and the title stayed on the poster
    seed = ((dist >= max(thr, 8.0)) & inside).astype(np.uint8) * 255
    p_fg = hist(seed)[q[..., 0], q[..., 1], q[..., 2]]
    p_bg = hist(ring)[q[..., 0], q[..., 1], q[..., 2]]
    text = ((p_fg > 2.0 * p_bg + 1e-6) & (region > 0)).astype(np.uint8) * 255
    text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(text, 8)
    min_area = max(30, int(line_h * line_h * 0.002))
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    text = np.isin(lbl, big).astype(np.uint8) * 255
    if np.count_nonzero(text) < 0.01 * max(1, np.count_nonzero(core)):
        text = seed                             # the colours did not separate
    text = _letters_read_checked(lab, group, core, text, min_area)
    lum = lab[..., 0]
    near = cv2.dilate(text, np.ones((int(0.6 * line_h) | 1,) * 2, np.uint8))
    # an ink band is clearly darker than the scene around the title (on a
    # dark scene too: the threshold follows it). The scene is looked at half
    # to one line height away: right next to the letters a band would be
    # measured against itself
    k1, k2 = max(15, int(0.5 * line_h)) | 1, max(31, int(1.0 * line_h)) | 1
    outer = (cv2.dilate(region, np.ones((k2, k2), np.uint8))
             & (255 - cv2.dilate(region, np.ones((k1, k1), np.uint8))))
    if not outer.any():
        outer = ring
    ring_l = float(np.median(lum[outer > 0])) if outer.any() else 128.0
    dark = min(60.0, max(18.0, 0.6 * ring_l))
    band = ((lum < dark) & (region > 0) & (near > 0) & (text == 0)).astype(np.uint8) * 255
    band_l = float(np.median(lum[band > 0])) if band.any() else 255.0
    backdrop = bool(np.count_nonzero(band) > 0.8 * np.count_nonzero(text)
                    and ring_l >= band_l + max(10.0, 0.5 * ring_l))
    if backdrop:
        # an ink band is dark both in the gaps between the letters and a
        # little away from them (beyond any glow or outline). A drop shadow
        # over a dim scene is dark near the letters only; a glow lights the
        # gaps of a band. Measured (gaps + around): ink-band poster 0.86 +
        # 0.51, the gold preset 0.59 + 0.78, a red-outline title on a dim
        # scene 0.60 + 0.52
        def grown_px(r: float) -> np.ndarray:
            k = 2 * max(2, int(r)) + 1
            return cv2.dilate(text, np.ones((k, k), np.uint8)) > 0

        between = np.zeros((h, w), bool)
        for b in group:
            ys, xs = np.nonzero(text[b.y:b.y + b.h, b.x:b.x + b.w] > 0)
            if len(ys) >= 20:
                between[b.y + int(np.percentile(ys, 5)):b.y + int(np.percentile(ys, 95)),
                        b.x + int(xs.min()):b.x + int(xs.max())] = True
        between &= ~grown_px(max(3, 0.06 * line_h))
        around = grown_px(0.35 * line_h) & ~grown_px(0.15 * line_h) & (region > 0)
        backdrop = bool(between.any() and around.any()
                        and float(np.mean(lum[between] < dark))
                        + float(np.mean(lum[around] < dark)) >= 1.25)
    r = max(5, int(0.09 * line_h)) | 1
    erase = cv2.dilate(text, np.ones((r, r), np.uint8))
    if backdrop:
        erase |= cv2.dilate(band, np.ones((5, 5), np.uint8))
    erase &= cv2.dilate(region, np.ones((r, r), np.uint8))   # stays at the title
    return {"text": text, "band": band if backdrop else np.zeros_like(band),
            "backdrop": backdrop, "erase": erase, "region": region, "line_h": line_h,
            "ring_l": ring_l, "band_l": band_l}


def _title_style(rgb: np.ndarray, layer: dict, group: list[OcrBox], new_text: str,
                 img_w: int, img_h: int, fonts_dir: Path | None = None) -> StyleSpec:
    """The look of the poster's title as a StyleSpec for `new_text`: its
    colours top to bottom (a gradient when they change), the outline and
    its thickness, the ink band, the glow, sparkles, the angle, the size of
    each line, and the place and size of the old letters. The font is the
    bundled Khmer face nearest the old lettering (_title_face: the original
    typeface cannot be known, and a Chinese or English one has no Khmer
    letters)."""
    import cv2
    spec = StyleSpec(font="Moul")
    text = layer["text"]
    line_h = float(layer["line_h"])

    def grown(r: float) -> np.ndarray:
        """The letters grown by `r` px."""
        k = 2 * max(0, int(round(r))) + 1
        return cv2.dilate(text, np.ones((k, k), np.uint8)) > 0

    solid = cv2.erode(text, np.ones((3, 3), np.uint8))
    if np.count_nonzero(solid) < 50:
        solid = text
    main = max(group, key=lambda b: int(np.count_nonzero(
        text[b.y:b.y + b.h, b.x:b.x + b.w])))
    box = (slice(main.y, main.y + main.h), slice(main.x, main.x + main.w))
    spec.font = _title_face(text, main)

    # the letters' colours, top to bottom of the main line
    sel = solid[box] > 0
    ys, _xs = np.nonzero(sel)
    font_px = 0.8 * line_h
    if len(ys) >= 30:
        cols = rgb[box][sel].astype(np.float64)
        y_lo, y_hi = np.percentile(ys, 5), np.percentile(ys, 95)
        font_px = max(8.0, 1.25 * float(y_hi - y_lo))     # letter height -> font size
        t = np.clip((ys - y_lo) / max(1.0, y_hi - y_lo), 0.0, 1.0)
        bands = [cols[t < 1 / 3], cols[(t >= 1 / 3) & (t < 2 / 3)], cols[t >= 2 / 3]]
        stops = [np.median(b, axis=0) for b in bands if len(b) >= 10]
        spec.fill = _hex(np.median(cols, axis=0))
        if len(stops) == 3:
            top, mid, bot = stops
            spread = max(np.linalg.norm(top - bot), np.linalg.norm(top - mid),
                         np.linalg.norm(mid - bot))
            if spread > 40:
                spec.gradient = ([_hex(top), _hex(bot)]
                                 if np.linalg.norm(mid - (top + bot) / 2) < 18
                                 else [_hex(top), _hex(mid), _hex(bot)])
    fill_l = float(_lum(_parse_color(spec.fill)[:3]))

    # outline: the colour right around the letters when it stands out, and
    # how far it reaches
    stroke_px = 0
    rim = grown(2) & (text == 0)
    if rim.any():
        rc = np.median(rgb[rim], axis=0)
        if abs(float(_lum(rc)) - fill_l) > 70:
            for r in range(2, int(0.2 * font_px) + 3):
                ring_r = grown(r) & ~grown(r - 1)
                if not ring_r.any() or float(np.median(np.linalg.norm(
                        rgb[ring_r].astype(np.float64) - rc, axis=1))) > 60:
                    break
                stroke_px = r
            spec.stroke_color = _hex(rc)
            spec.stroke_width_ratio = round(float(min(0.15, max(0.015, stroke_px / font_px))), 3)
        else:
            spec.stroke_color = "#1a1208" if fill_l > 128 else "#f5f0e6"
            spec.stroke_width_ratio = 0.02
    layer["stroke_px"] = stroke_px          # the erase reaches over the outline too

    # the ink band behind the letters
    if layer["backdrop"]:
        spec.backdrop = True
        spec.backdrop_color = _hex(np.median(rgb[layer["band"] > 0], axis=0) * 0.9) + "eb"

    # glow: light beyond the outline, brighter than a little farther out
    near = grown(stroke_px + 0.06 * line_h) & ~grown(stroke_px + 2)
    far = grown(stroke_px + 0.2 * line_h) & ~grown(stroke_px + 0.1 * line_h)
    if near.any() and far.any():
        if float(np.median(_lum(rgb[near]))) > float(np.median(_lum(rgb[far]))) + 15:
            gc = np.clip(np.median(rgb[near], axis=0) * 1.5, 0, 255)
            spec.glow_color = _hex(gc) + "a0"
            spec.glow_radius = int(max(4, min(40, 0.06 * line_h)))
    if not spec.glow_color and spec.backdrop and fill_l > 150:
        # light letters on an ink band glow on posters like this; the haze
        # over the black ink is too faint to measure
        spec.glow_color = _glow_tone(spec.gradient[1] if spec.gradient else spec.fill) + "70"
        spec.glow_radius = int(max(4, min(30, 0.05 * line_h)))
    spec.shadow_color, spec.shadow_offset = "#000000a0", (2, 4)
    spec.shadow_blur = int(max(4, min(24, 0.04 * line_h)))

    # sparkles: small bright specks clear of the letters, on a styled title
    # (a glow or an ink band): elsewhere specks are the picture's own lights
    if spec.backdrop or spec.glow_color:
        specks = ((_lum(rgb) > 235) & grown(0.5 * line_h)
                  & ~grown(stroke_px + 0.1 * line_h)).astype(np.uint8)
        n, _l, st, _c = cv2.connectedComponentsWithStats(specks, 8)
        small = sum(1 for i in range(1, n)
                    if st[i, cv2.CC_STAT_AREA] <= 0.003 * line_h * line_h)
        if small >= 4:
            spec.sparkles = int(min(12, 4 + small // 2))

    # the angle of the main line (its long axis)
    yy, xx = np.nonzero(text[box] > 0)
    if len(xx) > 50:
        cov = np.cov(np.vstack([xx - xx.mean(), yy - yy.mean()]))
        evals, evecs = np.linalg.eigh(cov)
        vx, vy = evecs[:, int(np.argmax(evals))]
        ang = math.degrees(math.atan2(vy, vx))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        if 1.5 <= abs(ang) <= 25:
            spec.rotation = round(-ang, 1)

    # line sizes (a smaller lead-in line above, as on many posters), from
    # the letters of each line: the reader's boxes of two lines overlap
    rows = _box_rows(group)
    if len(rows) >= 2:
        hs = _row_heights(text, rows)
        if hs and max(hs) > 0:
            spec.line_scales = [round(max(0.3, v / max(hs)), 2) for v in hs]
        spec.line_break = "auto"

    # where the old letters were, and as big
    ty, tx = np.nonzero(text > 0)
    if len(tx):
        x0, x1 = np.percentile(tx, 1), np.percentile(tx, 99)
        y0, y1 = np.percentile(ty, 1), np.percentile(ty, 99)
    else:
        x0, x1 = min(b.x for b in group), max(b.x + b.w for b in group)
        y0, y1 = min(b.y for b in group), max(b.y + b.h for b in group)
    spec.pos = (round(float((x0 + x1) / 2 / img_w), 4), round(float((y0 + y1) / 2 / img_h), 4))
    layer["area"] = (float(x0), float(y0), float(x1), float(y1))
    spec.size_ratio = fit_size_ratio(new_text or "Title", spec, float(x1 - x0),
                                     float(y1 - y0), img_w, fonts_dir)
    return spec


def _smooth_on_plain(erase: np.ndarray, text: np.ndarray, rgb: np.ndarray,
                     group: list[OcrBox]) -> np.ndarray:
    """On a plain background (a flat or blurred band behind the line) each
    line's erase is one smooth area: LaMa fills a hole shaped like thin
    letters with a faint copy of them. Plain: the background's own detail
    (away from the letters, against a blurred copy) under 6 grey levels —
    the CLOSER band 1.8–2.4, scenes behind drama titles 13–22."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = erase.copy()
    for b in group:
        m = (text[b.y:b.y + b.h, b.x:b.x + b.w] > 0).astype(np.uint8)
        if m.shape[0] < 8 or m.shape[1] < 8 or not m.any():
            continue
        away = cv2.dilate(m, np.ones((15, 15), np.uint8)) == 0
        if np.count_nonzero(away) < 100:
            continue
        crop = gray[b.y:b.y + b.h, b.x:b.x + b.w]
        detail = float(np.mean(np.abs(crop - cv2.GaussianBlur(crop, (0, 0), 6))[away]))
        if detail >= 6.0:
            continue
        k = max(5, int(0.35 * b.h)) | 1
        part = out[b.y:b.y + b.h, b.x:b.x + b.w]
        out[b.y:b.y + b.h, b.x:b.x + b.w] = cv2.morphologyEx(
            part, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return out


def _title_face(text: np.ndarray, main: OcrBox) -> str:
    """The bundled Khmer face nearest the old title's lettering, from its
    strokes: Moul for strokes that swell and thin (brush, calligraphy,
    serif: most drama titles), Koulen for heavy even strokes (block letters
    such as CLOSER, FULL), Kantumruy Pro for light even ones. Stroke widths
    measured along the letters (90th / 10th percentile): block letters
    1.2–1.5, brush titles 4.6–6.5."""
    import cv2
    m = (text[main.y:main.y + main.h, main.x:main.x + main.w] > 0).astype(np.uint8)
    ys = np.nonzero(m)[0]
    if len(ys) < 200:
        return "Moul"
    letter_h = float(np.percentile(ys, 95) - np.percentile(ys, 5))
    if letter_h < 40:                       # too small to tell
        return "Moul"
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1.5)
    widths = 2.0 * dt[ridge]
    if len(widths) < 40:
        return "Moul"
    if float(np.percentile(widths, 90)) > 2.5 * max(1e-6, float(np.percentile(widths, 10))):
        return "Moul"
    return "Koulen" if float(np.median(widths)) >= 0.1 * letter_h else "Kantumruy Pro"


def replace_title(image_path: str, text: str, out_path: str,
                  boxes: list[OcrBox] | None = None, fonts_dir: Path | None = None,
                  as_title: bool = False, progress: Progress = None,
                  cancel: "threading.Event | None" = None,
                  grouped: bool = False, layout: bool = False) -> dict:
    """Take the poster's title off and give `text` its look.

    Returns {"found": bool, "clean": the picture without the old title (the
    original when none was found), "spec": a StyleSpec for `text`, "boxes":
    the old title's boxes, "area": (x0, y0, x1, y1) of its letters}. With
    no title on the poster the spec is gold_brush_spec(), sized for the
    lower middle. `boxes`: text boxes already found (else they are looked
    for now); `as_title`: they are the title (boxes the user drew: all of
    each box is erased), not boxes to pick the title from; `grouped`: they
    are the title's text boxes (read_headline): only its letters are
    erased. `layout`: `text` laid out in the old title's place like an
    automatic translation (_best_layout: its lines, line sizes, alignment;
    on the picture); the result's "text" has the lines ("|")."""
    _check_cancel(cancel)
    rgb = np.asarray(_open_image(image_path).convert("RGB"), dtype=np.uint8).copy()
    h, w = rgb.shape[:2]
    if boxes is None:
        _report(progress, 0.02, "Looking for the title…")
        boxes = detect_title_boxes(image_path, progress=_scaled(progress, 0.0, 0.4),
                                   cancel=cancel)
    group = (list(boxes) if (as_title or grouped) and boxes
             else group_title_boxes(list(boxes or []), w, h))
    if not group:
        spec = gold_brush_spec()
        spec.size_ratio = fit_size_ratio(text or "Title", spec, 0.70 * w, 0.24 * h, w,
                                         fonts_dir)
        spec.glow_radius = gold_glow_radius(spec.size_ratio, w)
        _report(progress, 1.0, "No title on the poster")
        return {"found": False, "clean": image_path, "spec": spec, "boxes": [],
                "area": None, "text": text}
    _report(progress, 0.45, "Reading the title's style…")
    layer = _title_layer(rgb, group)
    spec = _title_style(rgb, layer, group, text, w, h, fonts_dir)
    _check_cancel(cancel)
    _report(progress, 0.55, "Removing the old title…")
    erase = layer["erase"]
    if layer.get("stroke_px"):
        import cv2
        k = 2 * int(layer["stroke_px"]) + 1
        erase = cv2.dilate(erase, np.ones((k, k), np.uint8))
    erase = _smooth_on_plain(erase, layer["text"], rgb, group)
    if as_title:
        # the user drew the area: all of it goes (their box may hold parts of
        # the title in the colours around it), and nothing outside it
        erase = np.zeros_like(erase)
        for b in group:
            erase[max(0, b.y - 4):b.y + b.h + 4, max(0, b.x - 4):b.x + b.w + 4] = 255
    clean = _inpaint_array(rgb, erase, group, "auto", _scaled(progress, 0.55, 0.95), cancel)
    _check_cancel(cancel)
    _save_image(Image.fromarray(clean), out_path, quality=95)
    log.info("replace_title: %d box(es), band %s (ring L %.0f, band L %.0f), spec %s",
             len(group), layer["backdrop"], layer["ring_l"], layer["band_l"], spec)
    if layout and layer.get("area"):
        text = _best_layout(text, spec, layer["area"], group, w, h, fonts_dir)
    _report(progress, 1.0, "Title replaced")
    return {"found": True, "clean": out_path, "spec": spec, "boxes": group,
            "area": layer.get("area"), "text": text}


# ---------------------------------------------------------------------------
# AI: title translation + style spec
# ---------------------------------------------------------------------------

_LANG_NAMES = {
    "km": "Khmer", "en": "English", "zh": "Chinese", "th": "Thai",
    "vi": "Vietnamese", "ko": "Korean", "ja": "Japanese", "fr": "French",
    "es": "Spanish", "hi": "Hindi", "id": "Indonesian", "lo": "Lao",
    "my": "Burmese",
}


def translate_title(title: str, ai: "AIEngine", target_lang: str = "km",
                    genre: str = "",
                    cancel: "threading.Event | None" = None) -> list[str]:
    """Return exactly 3 catchy localized title options for a poster."""
    _check_cancel(cancel)
    lang_name = _LANG_NAMES.get((target_lang or "km").strip().lower(), target_lang)
    genre_line = f"\nGenre: {genre}" if genre else ""
    system = ("You are an expert movie-poster copywriter and localizer. "
              "You write short, punchy, cinematic titles that sound native, "
              "never literal word-for-word translations.")
    prompt = (
        f"Original poster title: {title}{genre_line}\n"
        f"Target language: {lang_name}\n\n"
        f"Create exactly 3 catchy cinematic title options in {lang_name} for "
        f"this poster. Keep each option short (poster-friendly), emotionally "
        f"striking, and natural in {lang_name}.\n"
        f'Respond with ONLY this JSON, nothing else:\n'
        f'{{"options": ["option 1", "option 2", "option 3"]}}')
    result = ai.chat(feature="poster", json_mode=True, system=system,
                     messages=[{"role": "user", "content": prompt}],
                     temperature=0.9, max_tokens=512, cancel=cancel)
    _check_cancel(cancel)
    data = _extract_json_obj(result.text)
    raw: Any = []
    if isinstance(data, dict):
        raw = data.get("options") or data.get("titles") or data.get("results") or []
    elif isinstance(data, list):
        raw = data
    options: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            s = str(item).strip().strip('"')
            if s and s not in options:
                options.append(s)
    if not options:
        log.warning("translate_title: could not parse options from model "
                    "reply; padding with the original title")
    while len(options) < 3:
        options.append(options[0] if options else title)
    return options[:3]


# ---------------------------------------------------------------------------
# Automatic title translation: the poster's main title read (the AI looks at
# the picture, helped by the text reader), translated, taken off and drawn
# back in its own look — one picture, or many
# ---------------------------------------------------------------------------

_LANG_LETTERS = {
    "km": "ក-៿᧠-᧿", "th": "฀-๿", "lo": "຀-໿",
    "my": "က-႟", "hi": "ऀ-ॿ", "ko": "가-힯ᄀ-ᇿ",
    "ja": "぀-ヿ㐀-鿿", "zh": "㐀-鿿豈-﫿",
}
_LATIN_LETTERS = "A-Za-zÀ-ɏ"
_CJK_LETTERS_RE = re.compile("[぀-ヿ㐀-鿿豈-﫿가-힯]")
_TITLE_QUOTES = "\"'“”‘’«»「」『』《》〈〉【】"

_HEADLINE_SYSTEM = (
    "You localize movie, drama and video posters. You find a poster's main title and "
    "translate it the way a native poster designer of the target language would write it: "
    "the same meaning, feeling and selling power, short and natural, never word for word.")


def output_name(image_path: str, target_lang: str = "km") -> str:
    """The translated picture's file name: <name>_KH.png for Khmer
    (<name>_EN.png, <name>_TH.png … for other languages)."""
    lang = (target_lang or "km").strip().lower()
    return f"{Path(image_path).stem}_{'KH' if lang == 'km' else lang.upper()}.png"


def _script_share(text: str, lang: str) -> float:
    """Share of the letters of `text` written in `lang`'s script."""
    letters = [ch for ch in str(text or "") if ch.isalpha()]
    if not letters:
        return 0.0
    rx = re.compile("[" + _LANG_LETTERS.get(lang, _LATIN_LETTERS) + "]")
    return sum(1 for ch in letters if rx.match(ch)) / len(letters)


def _poster_jpeg(image_path: str, max_side: int = 1280) -> bytes:
    """The picture as a JPEG for the AI to look at (at most max_side px)."""
    import io
    img = _open_image(image_path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _reading_hints(boxes: list[OcrBox], w: int, h: int, limit: int = 14) -> str:
    """What the text reader found, biggest first, for the AI."""
    out = []
    for b in sorted(boxes, key=lambda b: b.w * b.h * b.conf, reverse=True)[:limit]:
        alts = "".join(f' / "{a}"' for a in b.alts[:3])
        sure = (", read surely" if b.conf >= 0.9 else
                ", read unsurely" if b.conf < 0.7 else "")
        out.append(f'- "{b.text}"{alts}: {100 * b.x / w:.0f}–{100 * (b.x + b.w) / w:.0f}% '
                   f"across, {100 * b.y / h:.0f}–{100 * (b.y + b.h) / h:.0f}% down, letters "
                   f"{100 * b.h / h:.0f}% of the picture's height{sure}")
    return "\n".join(out) or "- (no text found)"


def _headline_prompt(hints: str, lang: str, seen: bool, src_lang: str, note: str = "") -> str:
    """The request: find the main title, read it, translate it (JSON)."""
    name = _LANG_NAMES.get(lang, lang)
    rules = [
        f"The same meaning, emotion and selling power as the original, in natural {name} "
        "that a poster would print; about as short as the original; no added words, no "
        "quotes, no explanations.",
        "One translated line for each line of the original; a single long line may become "
        "two balanced lines.",
        "Every number stays the same number, written in digits.",
        "A song, brand or app name printed in Latin letters (CLOSER, TikTok) stays as "
        f"printed; a person's or place's name is written in {name} letters by its sound.",
    ]
    if lang == "km":
        from app.engines import km_order
        rules.append("Khmer letters only: no Chinese, Japanese or Korean characters.")
        rules.append(km_order.rule(src_lang))
    if seen:
        intro = ("Look at this poster and find its MAIN TITLE: the biggest, most "
                 "eye-catching text — the name of the show or video, or its slogan. Not "
                 "small labels, badges, stickers, signs or screens in the scene, subtitles, "
                 "logos, or watermarks such as \"豆包AI生成\".\n"
                 "A text reader found these texts on it (its readings can have wrong "
                 "characters: trust what you see):\n")
        read = ('"headline": the main title as printed, one string per line, the exact '
                "characters you see")
    else:
        intro = ("A text reader read these texts on a poster (you cannot see the poster; "
                 "its readings can have wrong characters). Find the poster's MAIN TITLE: "
                 "usually the biggest, most eye-catching text — the name of the show or "
                 "video, or its slogan; not small labels, badges, signs in the scene, "
                 "subtitles or watermarks such as \"豆包AI生成\":\n")
        read = ('"headline": the main title, one string per line, with reading mistakes '
                "put right (from the other readings of the same text and from common "
                "phrases)")
    unread = ("" if seen else
              ' When the readings of the biggest text are random letters (a script the '
              'reader cannot read, such as Khmer or Thai), answer {"status": "unreadable"}.')
    return (
        intro + hints + "\n\n"
        f"Then translate the main title into {name}:\n"
        + "\n".join(f"- {r}" for r in rules) + "\n\n"
        "Answer with only this JSON:\n"
        '{"status": "ok", "headline": ["…"], "language": "<ISO 639-1 code of the title>", '
        '"parts": ["…"], ' + ('"box": [ymin, xmin, ymax, xmax], ' if seen else "")
        + '"meaning": "<what the title means, in English>", "translation": ["…"]}\n'
        f"{read}; \"parts\": the reader's texts above that belong to the main title, "
        "copied exactly from the list"
        + ('; "box": where the whole main title is, 0–1000 of the picture\'s height (y) '
           "and width (x)" if seen else "") + ".\n"
        f'When the main title is already in {name}, answer {{"status": "already"}}. '
        'When the poster has no title, answer {"status": "none"}.' + unread
        + (f"\n\n{note}" if note else ""))


def _ask_headline(ai: "AIEngine", image_path: str, hints: str, lang: str, src_lang: str,
                  seen: bool, cancel: "threading.Event | None", note: str = "") -> dict:
    """One AI request about the poster (the picture itself when `seen`)."""
    pics = {"images": [("image/jpeg", _poster_jpeg(image_path))]} if seen else {}
    result = ai.chat(feature="poster", json_mode=True, system=_HEADLINE_SYSTEM,
                     messages=[{"role": "user", "content": _headline_prompt(
                         hints, lang, seen, src_lang, note)}],
                     temperature=0.3, max_tokens=2048, cancel=cancel, **pics)
    _check_cancel(cancel)
    data = _extract_json_obj(result.text)
    if not isinstance(data, dict):
        raise AppError("The AI's answer about the title could not be read.",
                       "មិនអាចអានចម្លើយរបស់ AI អំពីចំណងជើងបានទេ។")
    return data


def _as_lines(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"\s*(?:\||\n)\s*", value)
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value if v is not None]
    else:
        parts = []
    return [p.strip() for p in parts if p and p.strip()]


def _ai_box(value: Any) -> tuple[float, float, float, float] | None:
    """The AI's [ymin, xmin, ymax, xmax] (0–1000) when it makes sense."""
    try:
        y0, x0, y1, x1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        return None
    if (y1 - y0) * (x1 - x0) < 2000:          # under 0.2 % of the picture
        return None
    return y0, x0, y1, x1


def _norm_text(s: str) -> str:
    return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())


def _text_like(a: str, b: str) -> float:
    import difflib
    a, b = _norm_text(a), _norm_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _box_like(part: str, b: OcrBox) -> float:
    return max([_text_like(part, b.text)] + [_text_like(part, a) for a in b.alts])


def _pick_title_boxes(boxes: list[OcrBox], parts: list[str],
                      box: tuple[float, float, float, float] | None,
                      w: int, h: int) -> list[OcrBox]:
    """The reader's boxes that make the main title, as the AI named them:
    by the reader's texts it picked out, else by the place it gave (a box
    at least half inside it); [] when neither matches a box."""
    picked: list[OcrBox] = []
    for part in parts:
        best = max(boxes, key=lambda b: _box_like(part, b), default=None)
        if best is not None and _box_like(part, best) >= 0.6 and best not in picked:
            picked.append(best)
    if picked or box is None:
        return picked
    y0, x0, y1, x1 = box
    area = OcrBox(int(x0 * w / 1000 - 0.02 * w), int(y0 * h / 1000 - 0.02 * h),
                  int((x1 - x0) * w / 1000 + 0.04 * w), int((y1 - y0) * h / 1000 + 0.04 * h),
                  "", 1.0)
    inside = [b for b in boxes if b.conf >= 0.3 and _overlap(b, area) >= 0.5 * b.w * b.h]
    if not inside:
        return []
    top = max(b.h for b in inside)
    return [b for b in inside if b.h >= 0.25 * top]


def _clean_title_text(text: str, lang: str) -> str:
    """The translation as the poster prints it: full-width ！？：，（） as
    ! ? : , ( ) (the Khmer faces have no full-width forms), no quotes
    around a line, "|" between lines; Khmer digits in a Khmer title."""
    out = []
    for ch in str(text or ""):
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            ch = chr(o - 0xFEE0)
        elif ch == "　":
            ch = " "
        elif ch == "、":
            ch = ","
        elif ch == "。":
            ch = ""
        out.append(ch)
    lines = []
    for line in _as_lines("".join(out)):
        line = re.sub(r"\s+", " ", line).strip().strip(_TITLE_QUOTES).strip()
        if line:
            lines.append(line)
    s = " | ".join(lines)
    if lang == "km":
        from app.engines import numbers
        s = numbers.km_digits(s)
    return s


_CMAPS: dict[str, set] = {}


def _missing_letters(text: str, font_path: str) -> str:
    """The characters of `text` that the font has no letter for."""
    if font_path not in _CMAPS:
        try:
            from fontTools.ttLib import TTFont
            kw = {"fontNumber": 0} if font_path.lower().endswith(".ttc") else {}
            f = TTFont(font_path, lazy=True, **kw)
            try:
                _CMAPS[font_path] = set(f.getBestCmap() or {})
            finally:
                f.close()
        except Exception:
            log.debug("could not read the font's letters: %s", font_path, exc_info=True)
            _CMAPS[font_path] = set()
    cmap = _CMAPS[font_path]
    if not cmap:
        return ""
    return "".join(sorted({ch for ch in str(text or "")
                           if ch not in " |​‌‍" and ord(ch) not in cmap}))


def read_headline(image_path: str, ai: "AIEngine", target_lang: str = "km",
                  boxes: list[OcrBox] | None = None, progress: Progress = None,
                  cancel: "threading.Event | None" = None) -> dict:
    """The poster's main title, read and translated (one AI request, two
    when the model cannot look at pictures or answers in the wrong letters).

    Returns {"status": "ok" | "none" (no title) | "already" (already in the
    target language) | "unreadable" (letters neither the reader nor the AI
    could read), "boxes": the title's text boxes, "source": the title as
    printed ("|" between lines), "lang": its language, "meaning": what it
    means (English), "text": the translation ("|" between lines), "seen":
    the AI looked at the picture itself (else only at the reader's texts)}."""
    _check_cancel(cancel)
    if ai is None:
        raise AppError("Translating the title needs the AI (Settings → API keys).",
                       "ការបកប្រែចំណងជើងត្រូវការ AI (ការកំណត់ → API keys)។")
    lang = (target_lang or "km").strip().lower()
    w, h = _open_image(image_path).size
    if boxes is None:
        _report(progress, 0.02, "Reading the poster's text…")
        boxes = detect_title_boxes(image_path, progress=_scaled(progress, 0.0, 0.45),
                                   cancel=cancel)
    hints = _reading_hints(boxes, w, h)
    src_lang = "zh" if any(_CJK_LETTERS_RE.search(b.text) for b in boxes) else "en"
    _report(progress, 0.5, "The AI reads and translates the title…")
    from app.core.ai_engine import ai_unavailable
    seen = True
    try:
        data = _ask_headline(ai, image_path, hints, lang, src_lang, True, cancel)
    except CancelledError:
        raise
    except Exception as e:
        if ai_unavailable(e):
            raise                     # no AI at all (no key, no credit): say so
        # a model that cannot see pictures refuses them: the reader's text
        log.warning("the AI could not look at the poster (%s); it gets the text "
                    "reader's readings instead", getattr(e, "en", "") or e)
        seen = False
        data = _ask_headline(ai, image_path, hints, lang, src_lang, False, cancel)
    out = {"status": "none", "boxes": [], "source": "", "lang": "", "meaning": "",
           "text": "", "seen": seen}
    status = str(data.get("status") or "ok").strip().lower()
    source = _as_lines(data.get("headline"))
    text = _clean_title_text(" | ".join(_as_lines(data.get("translation"))), lang)
    out.update(source=" | ".join(source), lang=str(data.get("language") or "").strip().lower(),
               meaning=str(data.get("meaning") or "").strip())
    if status in ("none", "already", "unreadable"):
        out["status"] = status
        return out
    if source and _script_share(" ".join(source), lang) >= 0.6:
        out["status"] = "already"
        return out
    if not text:
        return out
    if lang not in ("zh", "ja", "ko") and _CJK_LETTERS_RE.search(text):
        # Chinese characters left in a Khmer (Thai, English…) title: once more
        _report(progress, 0.75, "Asking the AI again (letters of the wrong script)…")
        again = _ask_headline(
            ai, image_path, hints, lang, src_lang, seen, cancel,
            note=(f"Your last translation was \"{text}\": it still has Chinese, Japanese or "
                  f"Korean characters. Write it in {_LANG_NAMES.get(lang, lang)} letters "
                  "only."))
        text2 = _clean_title_text(" | ".join(_as_lines(again.get("translation"))), lang)
        if text2:
            text = text2
    if lang == "km" and out["source"]:
        from app.engines import km_order
        text, _n = km_order.fix(out["source"], text)
    group = _pick_title_boxes(boxes, _as_lines(data.get("parts")),
                              _ai_box(data.get("box")) if seen else None, w, h)
    if not group:
        group = group_title_boxes(list(boxes), w, h)
    if not group:
        return out                    # the title is not on the picture where it can be found
    out.update(status="ok", boxes=group, text=text)
    log.info("read_headline %s: %r -> %r (%s, %d box(es), seen=%s)", Path(image_path).name,
             out["source"], text, out["lang"], len(group), seen)
    return out


def _ink_height(line: str, font_path: str, letter_spacing: int) -> float:
    """How tall a line's letters look at size 100 (10th–90th percentile of
    its ink): Khmer letters with their vowels look taller than Latin
    capitals at the same size."""
    m = np.asarray(_render_line_mask(line, font_path, 100, int(letter_spacing))) > 0
    ys = np.nonzero(m)[0]
    if len(ys) < 20:
        return 1.0
    return max(1.0, float(np.percentile(ys, 90) - np.percentile(ys, 10)))


def _matched_scales(lines: list[str], want: list[float], font_path: str,
                    letter_spacing: int) -> list[float]:
    """Line sizes at which the new lines look as big next to each other as
    the old ones did (`want`: the old lines' letter heights, largest 1)."""
    raw = [w / _ink_height(ln, font_path, letter_spacing) for ln, w in zip(lines, want)]
    top = max(raw) or 1.0
    return [round(max(0.2, r / top), 3) for r in raw]


def _title_align(group: list[OcrBox], img_w: int) -> str:
    """How the old title's lines sit: "left" (flush left, or one line at the
    left edge), "right", or "center"."""
    spans = [(min(b.x for b in r), max(b.x + b.w for b in r)) for r in _box_rows(group)]
    if not spans:
        return "center"
    x0, x1 = min(a for a, _b in spans), max(b for _a, b in spans)
    tol = 0.03 * img_w
    if len(spans) >= 2:
        lefts, rights = [a for a, _b in spans], [b for _a, b in spans]
        mids = [(a + b) / 2 for a, b in spans]
        if max(mids) - min(mids) > tol:
            if max(lefts) - min(lefts) <= tol:
                return "left"
            if max(rights) - min(rights) <= tol:
                return "right"
            return "center"
    mid = (x0 + x1) / 2
    if x0 < 0.12 * img_w and mid < 0.42 * img_w:
        return "left"
    if x1 > 0.88 * img_w and mid > 0.58 * img_w:
        return "right"
    return "center"


def _block_px(text: str, spec: StyleSpec, img_w: int, fonts_dir: Path | None
              ) -> tuple[float, float, float, float]:
    """The title on the picture (px): its letters' width and height, and the
    whole of it with outline, glow, shadow and tilt."""
    font_path = _resolve_font(spec.font, fonts_dir)
    lines = title_lines(text, spec.line_break) or [str(text or "").strip() or " "]
    size = max(10, int(round(spec.size_ratio * img_w)))
    sizes = [max(8, int(round(size * k))) for k in _line_scales(lines, spec)]
    mask, _rects = _title_block(lines, font_path, sizes, spec.letter_spacing, spec.align)
    w, h = mask.size
    extra = 2.0 * (spec.stroke_width_ratio * size + spec.glow_radius) + spec.shadow_blur
    fw, fh = w + extra, h + extra
    a = math.radians(abs(spec.rotation))
    return (float(w), float(h), fw * math.cos(a) + fh * math.sin(a),
            fw * math.sin(a) + fh * math.cos(a))


def _best_layout(text: str, spec: StyleSpec, area: tuple[float, float, float, float],
                 group: list[OcrBox], img_w: int, img_h: int,
                 fonts_dir: Path | None) -> str:
    """The translated title laid out where the old one was. Its lines: as
    the AI wrote them, on one line, or on two balanced lines — whichever has
    the biggest letters in the old title's place (a little taller allowed:
    Khmer stacks vowels and subscripts). The lines look as big next to each
    other as the old lines did; flush left / right when the old ones were;
    never off the picture. Sets spec's size, line sizes, alignment and
    place; returns the text with its lines ("|")."""
    import copy
    x0, y0, x1, y1 = area
    aw, ah = max(1.0, x1 - x0), max(1.0, y1 - y0)
    margin = 0.02 * img_w
    cy = (y0 + y1) / 2
    spec.align = _title_align(group, img_w)
    if spec.align == "left":
        room_w = min(1.1 * aw, img_w - x0 - margin)
    elif spec.align == "right":
        room_w = min(1.1 * aw, x1 - margin)
    else:
        room_w = min(1.1 * aw, 0.92 * img_w)
    room_w = max(room_w, 0.5 * aw)
    room_h = min(1.35 * ah, 0.45 * img_h, max(ah, 2 * min(cy, img_h - cy) - 2 * margin))
    font_path = _resolve_font(spec.font, fonts_dir)
    want = list(spec.line_scales or [])
    lines = title_lines(text) or [text]
    one = " ".join(lines)
    cands = [" | ".join(lines)]
    if len(lines) > 1:
        cands.append(one)
    spaces = [i for i, ch in enumerate(one) if ch == " "]
    if spaces:
        i = min(spaces, key=lambda s: abs(s - len(one) / 2))
        if 0.25 * len(one) <= i <= 0.75 * len(one):
            cands.append(one[:i].strip() + " | " + one[i + 1:].strip())
    spec.line_break = ""                       # the lines chosen here, as they are
    scored: list[tuple[float, str, list[float] | None]] = []
    for c in dict.fromkeys(cands):
        trial = copy.copy(spec)
        c_lines = title_lines(c)
        trial.line_scales = (_matched_scales(c_lines, want, font_path, spec.letter_spacing)
                             if want and len(want) == len(c_lines) else None)
        scored.append((fit_size_ratio(c, trial, room_w, room_h, img_w, fonts_dir), c,
                       trial.line_scales))
    choice = scored[0]
    best = max(scored, key=lambda t: t[0])
    if best[0] > 1.15 * choice[0]:
        choice = best
    spec.size_ratio, chosen, spec.line_scales = choice
    # the place: flush with the old title's left / right edge, else its
    # middle; all of it on the picture
    lw, _lh, fw, fh = _block_px(chosen, spec, img_w, fonts_dir)
    cx = (x0 + lw / 2 if spec.align == "left" else x1 - lw / 2 if spec.align == "right"
          else (x0 + x1) / 2)
    cx = (min(max(cx, fw / 2 + margin), img_w - fw / 2 - margin)
          if fw + 2 * margin < img_w else img_w / 2)
    cy = (min(max(cy, fh / 2 + margin), img_h - fh / 2 - margin)
          if fh + 2 * margin < img_h else img_h / 2)
    spec.pos = (round(cx / img_w, 4), round(cy / img_h, 4))
    return chosen


def _shaping_ok() -> bool:
    if _raqm_available():
        return True
    try:
        __import__("freetype")
        __import__("uharfbuzz")
        return True
    except Exception:
        return False


def _title_checks(image_path: str, clean_path: str, group: list[OcrBox], source: str,
                  src_lang: str, text: str, spec: StyleSpec, out_path: str, lang: str,
                  fonts_dir: Path | None) -> list[tuple[str, bool | None, str]]:
    """The checks before saving: [(name, ok — None when it could not be
    checked —, detail)]: "removed" (the reader finds nothing of the old
    title where it was), "letters" (no letters of another script, none the
    font lacks), "numbers" (the original's numbers are all there),
    "shaping" (complex text layout: Khmer subscripts drawn right), "size"
    (the saved picture is as big as the original)."""
    checks: list[tuple[str, bool | None, str]] = []
    from app.engines import subtitle_ocr as so
    left: list[str] = []
    try:
        if not so.models_present():
            raise EngineNotAvailable("the text reader is not downloaded", "")
        bgr = _load_bgr(clean_path)
        H, W = bgr.shape[:2]
        mx = int(0.08 * max(b.h for b in group))
        X0, Y0 = max(0, min(b.x for b in group) - mx), max(0, min(b.y for b in group) - mx)
        X1 = min(W, max(b.x + b.w for b in group) + mx)
        Y1 = min(H, max(b.y + b.h for b in group) + mx)
        crop = bgr[Y0:Y1, X0:X1]
        lines = [ln for ln in source.split("|") if ln.strip()] or [source]
        for side in _reader_sides(crop.shape[1], crop.shape[0]):
            for _a, _b, _c, _d, t, _conf in so.read_image(crop, min_conf=0.6, max_side=side):
                if (len(_norm_text(t)) >= 2 and t not in left
                        and max(_text_like(t, ln) for ln in lines) >= 0.6):
                    left.append(t)
        checks.append(("removed", not left, " / ".join(left)))
    except Exception as e:
        log.debug("could not check that the old title is gone: %s", e)
        checks.append(("removed", None, ""))

    bad = []
    if lang not in ("zh", "ja", "ko"):
        bad = sorted(set(_CJK_LETTERS_RE.findall(text)))
    missing = _missing_letters(text.replace("|", " "), _resolve_font(spec.font, fonts_dir))
    checks.append(("letters", not bad and not missing, "".join(bad) + missing))

    from app.engines import numbers as nb
    try:
        lost, _extra = nb.check(source.replace("|", " "), text.replace("|", " "), src_lang)
        checks.append(("numbers", not lost, ", ".join(str(v) for v in lost)))
    except Exception:
        log.debug("number check failed", exc_info=True)
        checks.append(("numbers", None, ""))

    checks.append(("shaping", _shaping_ok(), ""))
    if out_path and os.path.exists(out_path):
        with Image.open(out_path) as o, Image.open(image_path) as i:
            checks.append(("size", o.size == i.size, f"{o.size[0]}×{o.size[1]}"))
    return checks


def auto_translate_title(image_path: str, out_path: str, ai: "AIEngine",
                         target_lang: str = "km", fonts_dir: Path | None = None,
                         progress: Progress = None,
                         cancel: "threading.Event | None" = None) -> dict:
    """Read the poster's main title, translate it, take it off and draw the
    translation in its place in its own look: its colours, outline, glow,
    ink band, angle, size and the nearest Khmer face. Saved as `out_path`
    (PNG, the picture's own size; "" = not saved, the caller draws it).

    Returns read_headline()'s keys with "status" "done" when translated,
    plus "out", "clean" (the picture without the old title), "spec" (the
    StyleSpec of the new title, sized to its lines), "checks"
    (_title_checks)."""
    import tempfile
    hl = read_headline(image_path, ai, target_lang, progress=_scaled(progress, 0.0, 0.6),
                       cancel=cancel)
    result = dict(hl, out="", clean="", spec=None, checks=[])
    if hl["status"] != "ok":
        _report(progress, 1.0, "No title to translate")
        return result
    lang = (target_lang or "km").strip().lower()
    work = tempfile.mkdtemp(prefix="rs_autotitle_")
    res = replace_title(image_path, hl["text"], os.path.join(work, "clean.png"),
                        boxes=hl["boxes"], grouped=True, fonts_dir=fonts_dir,
                        progress=_scaled(progress, 0.6, 0.9), cancel=cancel, layout=True)
    spec: StyleSpec = res["spec"]
    text = res["text"]
    _check_cancel(cancel)
    if out_path:
        _report(progress, 0.92, "Drawing the translated title…")
        render_title(res["clean"], text, spec, out_path, fonts_dir=fonts_dir)
    _report(progress, 0.96, "Checking the result…")
    checks = _title_checks(image_path, res["clean"], res["boxes"], hl["source"], hl["lang"],
                           text, spec, out_path, lang, fonts_dir)
    result.update(status="done", text=text, out=out_path, clean=res["clean"], spec=spec,
                  checks=checks)
    _report(progress, 1.0, "Title translated")
    return result


def auto_translate_titles(paths: list[str], out_dir: str, ai: "AIEngine",
                          target_lang: str = "km", fonts_dir: Path | None = None,
                          overwrite: bool = True, progress: Progress = None,
                          cancel: "threading.Event | None" = None) -> list[dict]:
    """auto_translate_title for each picture on its own — its own title,
    translation and look — saved as <out_dir>/<name>_KH.png (out_dir "":
    beside the picture). One result per picture (with "path"); a picture
    that fails gets "status" "error" and the others go on, but when there
    is no AI at all (no key, no credit) the batch stops there: that
    picture's result has "stopped": True and the rest are not tried."""
    from app.core.ai_engine import ai_unavailable
    results: list[dict] = []
    n = max(1, len(paths))
    for i, path in enumerate(paths):
        _check_cancel(cancel)
        name = Path(path).name

        def sub(f: float, t: str, i: int = i, name: str = name) -> None:
            _report(progress, (i + max(0.0, min(1.0, f))) / n,
                    f"[{i + 1}/{len(paths)}] {name}: {t}")

        out = os.path.join(out_dir or str(Path(path).parent), output_name(path, target_lang))
        if not overwrite and os.path.exists(out):
            results.append({"path": path, "status": "kept", "out": out, "checks": []})
            continue
        try:
            r = auto_translate_title(path, out, ai, target_lang, fonts_dir, progress=sub,
                                     cancel=cancel)
        except CancelledError:
            raise
        except Exception as e:
            log.warning("translating the title of %s failed: %s", path,
                        getattr(e, "en", "") or e)
            r = {"status": "error", "out": "", "checks": [],
                 "error": str(getattr(e, "en", "") or e),
                 "error_km": str(getattr(e, "km", "") or getattr(e, "en", "") or e)}
            if ai_unavailable(e):
                r["stopped"] = True
                r["path"] = path
                results.append(r)
                break
        r["path"] = path
        results.append(r)
    _report(progress, 1.0, "Done")
    return results


# ---------------------------------------------------------------------------
# The title redesigned by an AI picture model (Gemini): no fonts — the model
# letters the new title in the old one's art style; only that part of its
# picture goes into the poster
# ---------------------------------------------------------------------------

_ASPECTS = {"1:1": 1.0, "2:3": 2 / 3, "3:2": 1.5, "3:4": 0.75, "4:3": 4 / 3, "4:5": 0.8,
            "5:4": 1.25, "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9}


def _nearest_aspect(w: int, h: int) -> str:
    r = w / max(1, h)
    return min(_ASPECTS, key=lambda k: abs(math.log(_ASPECTS[k] / r)))


def _edit_prompt(source: str, text: str, lang: str, where: str) -> str:
    """What the picture model is asked: the new title, letter for letter, in
    the old title's art style; nothing else changed."""
    name = _LANG_NAMES.get(lang, lang)
    lines = [ln.strip() for ln in str(text or "").split("|") if ln.strip()]
    shown = " / ".join(lines)
    layout = (f"in {len(lines)} lines, split where the ' / ' is" if len(lines) > 1
              else "on one line (two if the place needs it)")
    khmer = ("\n- Khmer script: the right letter shapes; every subscript (ជើង) under its "
             "consonant and every vowel sign above, below or beside it exactly as written; no "
             "letters added, dropped or swapped." if lang == "km" else "")
    return (f"Edit this poster. Replace its main title{f' «{source}»' if source else ''}"
            f"{f' ({where})' if where else ''} with this {name} title: «{shown}».\n"
            f"- Write the {name} text exactly as given, letter for letter, {layout}.{khmer}\n"
            "- Design it in the original title's own art style, as if the poster had been made "
            f"in {name}: the same place and size, the same colours, gradient, metal or 3D look, "
            "outline, shadow, glow, brush strokes and decorations, the same tilt and "
            "perspective — lettering made for this poster, not a plain typed font.\n"
            "- Take the original title off completely: none of its letters may remain.\n"
            "- Change nothing else: the same people, faces, hands, background, other texts, "
            "logos and watermark, the same framing.")


def _align_edit(orig: np.ndarray, edit: np.ndarray) -> np.ndarray:
    """The model's picture at the poster's size, lined up with it: it comes
    back about 1024 px and may be shifted or scaled a little (matched
    features + RANSAC; plain resize when they do not agree)."""
    import cv2
    H, W = orig.shape[:2]
    ed = cv2.resize(edit, (W, H), interpolation=cv2.INTER_CUBIC
                    if edit.shape[1] < W else cv2.INTER_AREA)
    try:
        g1 = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(ed, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(4000)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None:
            return ed
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d2, d1)
        if len(matches) < 30:
            return ed
        src = np.float32([k2[m.queryIdx].pt for m in matches])
        dst = np.float32([k1[m.trainIdx].pt for m in matches])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < 25:
            return ed
        scale = math.hypot(M[0, 0], M[1, 0])
        if not 0.9 < scale < 1.1 or math.hypot(M[0, 2], M[1, 2]) > 0.08 * max(W, H):
            return ed
        return cv2.warpAffine(ed, M, (W, H), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        log.debug("could not line the edited picture up", exc_info=True)
        return ed


def _merge_edit(orig: np.ndarray, ed: np.ndarray, boxes: list[OcrBox]
                ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The poster with the model's new title: the old title's place and what
    the model changed next to it come from its picture (colours matched to
    the poster around it, edges feathered); everything else stays the
    poster's own pixels at full resolution. (picture, (x0, y0, x1, y1))."""
    import cv2
    H, W = orig.shape[:2]
    o = orig.astype(np.float32)
    e = ed.astype(np.float32)
    diff = cv2.GaussianBlur(np.abs(e - o).mean(axis=2), (0, 0), 2)
    changed = cv2.morphologyEx((diff > 25).astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((15, 15), np.uint8))
    keep = np.zeros((H, W), np.uint8)
    if boxes:
        th = max(b.h for b in boxes)
        bx0, by0 = min(b.x for b in boxes), min(b.y for b in boxes)
        bx1, by1 = max(b.x + b.w for b in boxes), max(b.y + b.h for b in boxes)
        zone = (max(0, int(bx0 - 0.04 * W)), max(0, int(by0 - 0.3 * th)),
                min(W, int(bx1 + 0.04 * W)), min(H, int(by1 + 0.3 * th)))
        reach = (max(0, int(bx0 - 0.25 * W)), max(0, int(by0 - 1.0 * th)),
                 min(W, int(bx1 + 0.25 * W)), min(H, int(by1 + 1.0 * th)))
        keep[zone[1]:zone[3], zone[0]:zone[2]] = 1
        n, lbl, st, _c = cv2.connectedComponentsWithStats(changed, 8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area >= 50 and x < zone[2] and zone[0] < x + w and y < zone[3] and zone[1] < y + h:
                keep[lbl == i] = 1
        near = np.zeros_like(keep)
        near[reach[1]:reach[3], reach[0]:reach[2]] = 1
        keep &= near                    # the model's changes far away are not taken
    else:
        keep = changed
        if keep.mean() > 0.6:
            return ed.copy(), (0, 0, W, H)     # it redrew the whole picture
    if not keep.any():
        return ed.copy(), (0, 0, W, H)
    keep = cv2.dilate(keep, np.ones((11, 11), np.uint8))
    # the model's colours a little off from the poster's: matched on a band
    # around the part taken, where both show the same scene
    ring = (cv2.dilate(keep, np.ones((41, 41), np.uint8)) > 0) & (keep == 0)
    if np.count_nonzero(ring) > 500:
        for c in range(3):
            a, b = e[..., c][ring], o[..., c][ring]
            sa = float(a.std()) or 1.0
            gain = float(np.clip(b.std() / sa, 0.8, 1.25))
            e[..., c] = (e[..., c] - a.mean()) * gain + b.mean()
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 6)[..., None]
    out = np.clip(o * (1 - soft) + e * soft, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(keep)
    return out, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _read_title_back(ai: "AIEngine", image_path: str,
                     cancel: "threading.Event | None") -> str:
    """The new title as the AI reads it off the finished poster."""
    result = ai.chat(feature="poster", json_mode=True, temperature=0.0, max_tokens=512,
                     images=[("image/jpeg", _poster_jpeg(image_path))], cancel=cancel,
                     messages=[{"role": "user", "content": (
                         "Read the main title of this poster exactly as it is written, letter "
                         "for letter — keep any wrong or missing letters as they are; lines "
                         'separated by " / ". Answer only this JSON: {"title": "…"}')}])
    data = _extract_json_obj(result.text)
    return str((data or {}).get("title") or "").strip() if isinstance(data, dict) else ""


def _same_letters(a: str, b: str) -> bool:
    keep = lambda s: "".join(ch for ch in unicodedata.normalize("NFC", s or "")  # noqa: E731
                             if ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc"))
    return bool(keep(a)) and keep(a) == keep(b)


def ai_replace_title(image_path: str, text: str, out_path: str, ai: "AIEngine",
                     source: str = "", boxes: list[OcrBox] | None = None,
                     target_lang: str = "km", verify: bool = True,
                     progress: Progress = None,
                     cancel: "threading.Event | None" = None) -> dict:
    """The poster's title replaced by `text`, lettered by an AI picture model
    (Gemini, AIEngine.edit_image) in the original title's art style — no
    fonts. The model's picture is lined up with the poster and only the
    title's part of it is put in (the rest stays the poster's own pixels);
    saved as `out_path` at the poster's size. `boxes`: the old title (else
    looked for); `source`: its words. Returns {"out", "provider", "model",
    "note", "read" (the title as the AI reads it back), "checks", "region"}."""
    import io
    _check_cancel(cancel)
    lang = (target_lang or "km").strip().lower()
    orig = _open_image(image_path).convert("RGB")
    W, H = orig.size
    if boxes is None:
        _report(progress, 0.03, "Looking for the title…")
        boxes = group_title_boxes(detect_title_boxes(
            image_path, progress=_scaled(progress, 0.0, 0.15), cancel=cancel), W, H)
    rows = _box_rows(boxes) if boxes else []
    if not source and rows:
        source = " / ".join(" ".join(b.text for b in sorted(r, key=lambda b: b.x)) for r in rows)
    where = ""
    if boxes:
        x0, y0 = min(b.x for b in boxes), min(b.y for b in boxes)
        x1, y1 = max(b.x + b.w for b in boxes), max(b.y + b.h for b in boxes)
        where = (f"about {100 * x0 / W:.0f}–{100 * x1 / W:.0f}% across and "
                 f"{100 * y0 / H:.0f}–{100 * y1 / H:.0f}% down the picture")
    send = orig.copy()
    send.thumbnail((1536, 1536), Image.LANCZOS)
    buf = io.BytesIO()
    send.save(buf, "JPEG", quality=92)
    _report(progress, 0.2, "The AI redesigns the title…")
    res = ai.edit_image(_edit_prompt(source, text, lang, where), ("image/jpeg", buf.getvalue()),
                        aspect_ratio=_nearest_aspect(W, H), cancel=cancel)
    _check_cancel(cancel)
    _report(progress, 0.8, "Putting the new title into the poster…")
    got = np.asarray(Image.open(io.BytesIO(res.data)).convert("RGB"))
    base = np.asarray(orig)
    merged, region = _merge_edit(base, _align_edit(base, got), boxes or [])
    _save_image(Image.fromarray(merged), out_path)
    checks: list[tuple[str, bool | None, str]] = []
    # the old title gone: the reader finds nothing of it where it was
    from app.engines import subtitle_ocr as so
    if boxes and source and so.models_present():
        try:
            rx0, ry0, rx1, ry1 = region
            crop = _load_bgr(out_path)[ry0:ry1, rx0:rx1]
            lines = [ln for ln in source.split(" / ") if ln.strip()] or [source]
            left = [t for *_b, t, _c in so.read_image(crop, min_conf=0.6)
                    if len(_norm_text(t)) >= 2 and max(_text_like(t, ln) for ln in lines) >= 0.6]
            checks.append(("removed", not left, " / ".join(left)))
        except Exception:
            log.debug("could not check the old title is gone", exc_info=True)
    read = ""
    if verify:
        _report(progress, 0.9, "Reading the new title back…")
        try:
            read = _read_title_back(ai, out_path, cancel)
        except CancelledError:
            raise
        except Exception as e:
            log.info("could not read the new title back: %s", getattr(e, "en", "") or e)
        if read:
            checks.insert(0, ("spelling", _same_letters(read, text), read))
    checks.append(("size", True, f"{W}×{H}"))
    _report(progress, 1.0, "Title redesigned")
    log.info("ai_replace_title %s: %s/%s, region %s, read back %r", Path(image_path).name,
             res.provider, res.model, region, read)
    return {"out": out_path, "provider": res.provider, "model": res.model, "note": res.note,
            "read": read, "checks": checks, "region": region, "text": text}


def analyze_image_style(image_path: str) -> dict:
    """Pure-numpy image style stats: dominant colors via k-means (k=5,
    10 iterations), plus brightness/contrast/saturation. No LLM involved."""
    img = _open_image(image_path).convert("RGB")
    width, height = img.size
    small = img.copy()
    small.thumbnail((96, 96), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64).reshape(-1, 3)

    centers, counts = _kmeans(pixels, k=5, iters=10)
    order = np.argsort(-counts)
    total = float(counts.sum()) or 1.0
    colors: list[str] = []
    weights: list[float] = []
    for j in order:
        r, g, b = (int(max(0, min(255, round(c)))) for c in centers[j])
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
        weights.append(round(float(counts[j]) / total, 4))

    gray = pixels.mean(axis=1)
    mx = pixels.max(axis=1)
    mn = pixels.min(axis=1)
    saturation = float(((mx - mn) / np.maximum(mx, 1e-6)).mean())
    return {
        "width": int(width),
        "height": int(height),
        "dominant_colors": colors,
        "color_weights": weights,
        "brightness": round(float(gray.mean() / 255.0), 4),
        "contrast": round(float(gray.std() / 255.0), 4),
        "saturation": round(saturation, 4),
    }


def _kmeans(pixels: np.ndarray, k: int = 5, iters: int = 10
            ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic mini k-means (no sklearn): centers seeded at brightness
    quantiles. Returns (centers[k,3], counts[k])."""
    n = int(pixels.shape[0])
    if n == 0:
        return np.zeros((1, 3)), np.array([1])
    k = max(1, min(k, n))
    order = np.argsort(pixels.mean(axis=1))
    seed_idx = order[np.linspace(0, n - 1, k).astype(int)]
    centers = pixels[seed_idx].astype(np.float64).copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, int(iters))):
        dists = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        for j in range(k):
            sel = labels == j
            if sel.any():
                centers[j] = pixels[sel].mean(axis=0)
    counts = np.bincount(labels, minlength=k)
    return centers, counts


def design_spec(image_stats: dict, title: str, genre: str, ai: "AIEngine",
                cancel: "threading.Event | None" = None) -> StyleSpec:
    """Ask the poster-feature model for a StyleSpec matching the image mood.
    Parsing is tolerant (StyleSpec.from_llm_json); on unusable JSON we log and
    return sane defaults rather than failing."""
    _check_cancel(cancel)
    try:
        stats_json = json.dumps(image_stats, ensure_ascii=False, default=str)
    except Exception:
        stats_json = str(image_stats)
    system = ("You are a senior movie-poster art director. You pick title "
              "typography and effects that match a poster's color mood. "
              "You answer with strict JSON only.")
    prompt = (
        f"Poster image analysis: {stats_json}\n"
        f"Title text: {title}\n"
        f"Genre: {genre or 'unknown'}\n\n"
        "Design the title style. Respond with ONLY a JSON object with these "
        "keys (all optional, defaults exist):\n"
        '{\n'
        '  "font": "Moul",                     // font family name\n'
        '  "size_ratio": 0.12,                 // 0.02-0.5, of image width\n'
        '  "fill": "#ffffff",                  // hex\n'
        '  "gradient": ["#ffdd00", "#ff6600"], // 2 hex stops or null\n'
        '  "stroke_color": "#000000",\n'
        '  "stroke_width_ratio": 0.04,         // 0-0.3 of font size\n'
        '  "shadow_color": "#000000cc",\n'
        '  "shadow_offset": [4, 4],\n'
        '  "shadow_blur": 8,                   // 0-64\n'
        '  "glow_color": null,                 // hex or null\n'
        '  "glow_radius": 0,                   // 0-64\n'
        '  "bevel": false,\n'
        '  "extrude_depth": 0,                 // 0-40\n'
        '  "extrude_color": "#222222",\n'
        '  "pos": [0.5, 0.82],                 // center fractions of W/H\n'
        '  "letter_spacing": 0,                // -20..200 px\n'
        '  "rotation": 0.0                     // degrees, -180..180\n'
        '}\n'
        "Choose colors that contrast well with the dominant colors and match "
        "the genre mood.")
    result = ai.chat(feature="poster", json_mode=True, system=system,
                     messages=[{"role": "user", "content": prompt}],
                     temperature=0.6, max_tokens=800, cancel=cancel)
    _check_cancel(cancel)
    data = _extract_json_obj(result.text)
    if not isinstance(data, dict):
        log.warning("design_spec: model reply was not valid JSON; "
                    "using default StyleSpec")
        data = {}
    return StyleSpec.from_llm_json(data)


# ---------------------------------------------------------------------------
# Export variants
# ---------------------------------------------------------------------------

def _fit_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop toward the target aspect when it loses at most 10% of the
    pixels; otherwise blurred-background pad (scale-fill + blur 40, then a
    centered contain-fit of the original on top)."""
    w, h = img.size
    target_aspect = target_w / target_h
    src_aspect = w / h
    if src_aspect > target_aspect:
        crop_w, crop_h = int(round(h * target_aspect)), h
    else:
        crop_w, crop_h = w, int(round(w / target_aspect))
    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))
    loss = 1.0 - (crop_w * crop_h) / float(w * h)
    if loss <= 0.10:
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        cropped = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        return cropped.resize((target_w, target_h), Image.LANCZOS)
    # blurred pad
    bg_scale = max(target_w / w, target_h / h)
    bg_w = max(target_w, int(math.ceil(w * bg_scale)))
    bg_h = max(target_h, int(math.ceil(h * bg_scale)))
    bg = img.resize((bg_w, bg_h), Image.LANCZOS)
    bx = (bg_w - target_w) // 2
    by = (bg_h - target_h) // 2
    bg = bg.crop((bx, by, bx + target_w, by + target_h))
    bg = bg.filter(ImageFilter.GaussianBlur(40))
    fg_scale = min(target_w / w, target_h / h)
    fg_w = max(1, int(round(w * fg_scale)))
    fg_h = max(1, int(round(h * fg_scale)))
    fg = img.resize((fg_w, fg_h), Image.LANCZOS)
    bg.paste(fg, ((target_w - fg_w) // 2, (target_h - fg_h) // 2))
    return bg


def export_variants(image_path: str, out_dir: str, base_name: str) -> dict:
    """Export the poster as original + 9:16 (1080x1920) + 16:9 (1920x1080).
    Returns {"original": path, "9x16": path, "16x9": path}."""
    img = _open_image(image_path).convert("RGB")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(image_path).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        ext = ".jpg"
    results: dict[str, str] = {}
    original_path = out / f"{base_name}_original{ext}"
    _save_image(img, str(original_path), quality=95)
    results["original"] = str(original_path)
    for key, (tw, th) in (("9x16", (1080, 1920)), ("16x9", (1920, 1080))):
        variant = _fit_aspect(img, tw, th)
        p = out / f"{base_name}_{key}{ext}"
        _save_image(variant, str(p), quality=95)
        results[key] = str(p)
    log.info("export_variants: wrote %s", ", ".join(results.values()))
    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # --- StyleSpec.from_llm_json coercion ---
    spec = StyleSpec.from_llm_json({
        "font": "  Moul ",
        "size_ratio": "0.9",              # clamps to 0.5
        "fill": "fff",                    # normalizes to #ffffff
        "gradient": ["#ff0000", "#0000ff", "#00ff00"],  # 2-4 stops kept
        "stroke_width_ratio": 2,          # clamps to 0.3
        "shadow_offset": ["10", 4.7],     # coerces to (10, 5)
        "shadow_blur": 999,               # clamps to 64
        "glow_color": "#00ffff",
        "glow_radius": -5,                # clamps to 0
        "bevel": "true",
        "extrude_depth": "3",
        "pos": [1.5, "0.8"],              # clamps/coerces to (1.0, 0.8)
        "letter_spacing": "5",
        "rotation": "-720",               # clamps to -180
        "unknown_key": {"ignored": True},
        42: "non-string key ignored",
    })
    assert spec.font == "Moul", spec.font
    assert spec.size_ratio == 0.5, spec.size_ratio
    assert spec.fill == "#ffffff", spec.fill
    assert spec.gradient == ["#ff0000", "#0000ff", "#00ff00"], spec.gradient
    assert spec.stroke_width_ratio == 0.3, spec.stroke_width_ratio
    assert spec.shadow_offset == (10, 5), spec.shadow_offset
    assert spec.shadow_blur == 64, spec.shadow_blur
    assert spec.glow_color == "#00ffff", spec.glow_color
    assert spec.glow_radius == 0, spec.glow_radius
    assert spec.bevel is True
    assert spec.extrude_depth == 3, spec.extrude_depth
    assert spec.pos == (1.0, 0.8), spec.pos
    assert spec.letter_spacing == 5, spec.letter_spacing
    assert spec.rotation == -180.0, spec.rotation

    # garbage input -> pure defaults, never raises
    for garbage in (None, "not json", [1, 2, 3], 7, {"fill": object()}):
        s = StyleSpec.from_llm_json(garbage)
        assert s.font == "Moul" and s.size_ratio == 0.12 and s.pos == (0.5, 0.82)

    # alias + string-JSON input
    s2 = StyleSpec.from_llm_json('{"color": "#123456", "position": [0.25, 0.5]}')
    assert s2.fill == "#123456" and s2.pos == (0.25, 0.5)

    # --- title look: new fields, lines, the title's boxes ---
    five = StyleSpec.from_llm_json({"gradient": ["#000000", "#111111", "#222222",
                                                 "#333333", "#444444"]})
    assert five.gradient == ["#000000", "#111111", "#222222", "#333333"], five.gradient
    nf = StyleSpec.from_llm_json({"backdrop": "yes", "backdrop_color": "#000000cc",
                                  "sparkles": "99", "line_break": "AUTO",
                                  "line_scales": [0.5, "2", "x"]})
    assert (nf.backdrop is True and nf.backdrop_color == "#000000cc" and nf.sparkles == 60
            and nf.line_break == "auto" and nf.line_scales == [0.5, 1.0, 1.0]), nf
    assert title_lines("A | B") == ["A", "B"]
    assert title_lines("Reborn in the Qin Dynasty: Defying Fate") == \
        ["Reborn in the Qin Dynasty: Defying Fate"]
    assert title_lines("Reborn in the Qin Dynasty: Defying Fate", "colon") == \
        ["Reborn in the Qin Dynasty:", "Defying Fate"]
    assert title_lines("កើតជាថ្មីក្នុងរាជវង្សឈិន៖ ប្រឆាំងនឹងវាសនា", "auto") == \
        ["កើតជាថ្មីក្នុងរាជវង្សឈិន៖", "ប្រឆាំងនឹងវាសនា"]
    assert title_lines("The Long Road Home Again", "auto") == ["The Long Road", "Home Again"]
    assert title_lines("Short title", "auto") == ["Short title"]
    assert _line_scales(["Reborn in the Qin Dynasty:", "Defying Fate"], StyleSpec()) == [0.62, 1.0]
    assert _line_scales(["A", "B"], StyleSpec(line_scales=[0.5, 1.0])) == [0.5, 1.0]
    assert _line_scales(["A"], StyleSpec(line_scales=[0.5, 1.0])) == [1.0]
    sample = [OcrBox(289, 532, 1106, 222, "RebornitheQnas", 0.69),
              OcrBox(434, 674, 535, 231, "Defying", 0.98),
              OcrBox(947, 659, 311, 208, "Fate", 0.97),
              OcrBox(20, 20, 470, 30, "watermark", 0.9)]
    grp = group_title_boxes(sample, 1672, 940)
    assert len(grp) == 3 and all(b.text != "watermark" for b in grp), grp
    assert len(_box_rows(grp)) == 2
    assert group_title_boxes([OcrBox(20, 20, 470, 30, "small print", 0.9)], 1672, 940) == []
    print("ok: title look: new fields, lines, the title's boxes")

    # --- LaMa that takes only 512 x 512: filled from squares around the mask ---
    class _FakeInput:
        def __init__(self, name: str, shape: list):
            self.name, self.shape = name, shape

    class _FakeLama:
        def __init__(self):
            self.calls: list = []

        def get_inputs(self):
            return [_FakeInput("image", [1, 3, 512, 512]), _FakeInput("mask", [1, 1, 512, 512])]

        def run(self, _out, feeds):
            img = feeds["image"]
            self.calls.append(img.shape)
            assert img.shape == (1, 3, 512, 512) and feeds["mask"].shape == (1, 1, 512, 512)
            red = np.zeros_like(img)
            red[:, 0] = 1.0
            return [red]

    fake = _FakeLama()
    _LAMA_SESSIONS["fake-lama"] = fake
    pic = np.full((300, 900, 3), 100, np.uint8)
    hole = np.zeros((300, 900), np.uint8)
    hole[100:150, 100:800] = 255                     # a wide title
    hole[280:290, 10:20] = 255                       # and a speck far away
    filled = _inpaint_lama(pic, hole, Path("fake-lama"))
    # the 700 px title at its own resolution: two squares; the speck: one
    assert len(fake.calls) == 3 and all(c == (1, 3, 512, 512) for c in fake.calls), fake.calls
    assert tuple(filled[120, 400]) == (255, 0, 0) and tuple(filled[285, 15]) == (255, 0, 0)
    assert tuple(filled[30, 450]) == (100, 100, 100), "outside the mask stays as it was"
    del _LAMA_SESSIONS["fake-lama"]
    print("ok: LaMa 512 x 512: each part of the mask from a square around it")

    # --- the filled area sharpened like the picture around it ---
    import cv2 as _cv2
    _rng = np.random.default_rng(3)
    busy = (_rng.random((240, 480, 3)) * 255).astype(np.uint8)
    busy = _cv2.GaussianBlur(busy, (0, 0), 1.0)                  # a detailed scene
    hole = np.zeros((240, 480), np.uint8)
    hole[80:160, 120:360] = 255
    soft_fill = busy.copy()
    soft_fill[hole > 0] = _cv2.GaussianBlur(busy, (0, 0), 6.0)[hole > 0]   # LaMa-like: smooth

    def _crisp(a, where):
        return float(_cv2.Laplacian(_cv2.cvtColor(a, _cv2.COLOR_RGB2GRAY), _cv2.CV_64F)[where].var())

    sharp = _restore_detail(busy, soft_fill, hole)
    assert _crisp(sharp, hole > 0) > 2 * _crisp(soft_fill, hole > 0), "sharpened"
    assert np.array_equal(sharp[hole == 0][:1000], soft_fill[hole == 0][:1000]) or \
        np.abs(sharp.astype(int) - soft_fill.astype(int))[:40].max() == 0, "outside untouched"
    flat = np.full((240, 480, 3), 90, np.uint8)
    assert np.array_equal(_restore_detail(flat, flat, hole), flat), "a plain picture stays"
    print("ok: the erased area sharpened as much as the picture around it, nothing else")

    # --- the look of a title read back from a poster drawn in it ---
    with tempfile.TemporaryDirectory() as td2:
        scene = np.zeros((540, 960, 3), np.uint8)
        scene[..., 0] = np.linspace(40, 90, 540, dtype=np.float64)[:, None].astype(np.uint8)
        scene[..., 1] = 70
        scene[..., 2] = np.linspace(60, 160, 960, dtype=np.float64)[None, :].astype(np.uint8)
        Image.fromarray(scene).save(Path(td2) / "scene.png")
        gold = gold_brush_spec()
        gold.rotation, gold.sparkles, gold.pos, gold.size_ratio = 0.0, 0, (0.5, 0.7), 0.08
        gold.glow_radius = gold_glow_radius(0.08, 960)      # as the app sizes it
        render_title(str(Path(td2) / "scene.png"), "Defying Fate", gold,
                     str(Path(td2) / "poster.png"), fonts_dir=_bundled_fonts_dir())
        rgb = np.asarray(Image.open(Path(td2) / "poster.png").convert("RGB"))
        bright = (rgb.astype(np.int32).sum(axis=2) > 450) & (
            np.abs(rgb.astype(np.int32) - scene.astype(np.int32)).sum(axis=2) > 60)
        ys, xs = np.nonzero(bright)
        box = OcrBox(int(xs.min()), int(ys.min()), int(xs.max() - xs.min()),
                     int(ys.max() - ys.min()), "Defying Fate", 0.9)
        layer = _title_layer(rgb, [box])
        look = _title_style(rgb, layer, [box], "ថ្មី", 960, 540, _bundled_fonts_dir())
        assert layer["backdrop"] and look.backdrop, (layer["ring_l"], layer["band_l"])
        assert look.gradient and len(look.gradient) >= 2, look.gradient
        assert abs(look.pos[0] - 0.5) < 0.05 and abs(look.pos[1] - 0.7) < 0.06, look.pos
        erase = layer["erase"] > 0
        assert erase[int(ys.mean()), int(xs.mean())] and not erase[10, 10]
        print("ok: the title's look read back: gold gradient, ink band, place")

    # --- export_variants on a generated test image (PIL only) ---
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "poster.jpg"
        arr = np.zeros((600, 800, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(20, 220, 800, dtype=np.float64)[None, :].astype(np.uint8)
        arr[..., 1] = 90
        arr[..., 2] = np.linspace(200, 40, 600, dtype=np.float64)[:, None].astype(np.uint8)
        Image.fromarray(arr).save(str(src), quality=95)

        out = export_variants(str(src), td, "test")
        assert set(out.keys()) == {"original", "9x16", "16x9"}, out
        for p in out.values():
            assert Path(p).is_file(), p
        assert Image.open(out["original"]).size == (800, 600)
        assert Image.open(out["9x16"]).size == (1080, 1920)
        assert Image.open(out["16x9"]).size == (1920, 1080)

        # smoke: k-means style analysis
        stats = analyze_image_style(str(src))
        assert len(stats["dominant_colors"]) == 5, stats
        assert 0.0 <= stats["brightness"] <= 1.0
        assert 0.0 <= stats["contrast"] <= 1.0
        assert 0.0 <= stats["saturation"] <= 1.0

        # smoke: _fit_aspect crop branch (loss under 10%)
        near = Image.new("RGB", (1000, 540), (10, 20, 30))  # ~1.85:1 -> 16:9 crop
        fitted = _fit_aspect(near, 1920, 1080)
        assert fitted.size == (1920, 1080)

        # Detect text: PaddleOCR's state not known -> the built-in reader
        # answers at once and PaddleOCR is tried in the background; blocked
        # -> never tried again (for that version); working -> used next time
        import time as _time

        from app.engines import subtitle_ocr as _so
        real_models = models_dir()
        if _so.available():
            tries = []

            def _blocked():
                tries.append(1)
                raise ImportError("DLL load failed while importing paddle: blocked")

            class _FakePaddle:
                def ocr(self, path, cls=True):
                    return [[[[[10, 10], [300, 10], [300, 60], [10, 60]],
                              ("FAKE PADDLE", 0.99)]]]

            def _works():
                tries.append(1)
                _time.sleep(0.5)                    # loading takes a while
                return _FakePaddle()

            def _fresh_session(versions="paddleocr 9.9"):
                global _PADDLE_STATE, _PADDLE_UNUSABLE, _PADDLE_VERDICT_READ, _PADDLE_PROBE
                global _paddle_versions
                _PADDLE_STATE, _PADDLE_UNUSABLE, _PADDLE_VERDICT_READ = "", "", False
                _PADDLE_PROBE = None
                _paddle_versions = lambda: versions          # noqa: E731
                tries.clear()

            def _probe_done():
                if _PADDLE_PROBE is not None:
                    _PADDLE_PROBE.join(30)

            models_dir = lambda: Path(td) / "models"        # noqa: E731 (verdicts: temp)
            poster = Image.new("RGB", (900, 500), (20, 20, 40))
            ImageDraw.Draw(poster).text(
                (60, 180), "NOW PLAYING", fill=(255, 255, 255),
                font=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 90)
                if os.path.isfile("C:/Windows/Fonts/arialbd.ttf") else None)
            pp = Path(td) / "poster.png"
            poster.save(pp)

            # 1. blocked
            _fresh_session()
            _get_paddle_ocr = _blocked                      # noqa: F811
            t0 = _time.monotonic()
            found = detect_title_boxes(str(pp))
            assert found and "PLAYING" in found[0].text.replace(" ", ""), found
            assert _time.monotonic() - t0 < 10, "the first Detect does not wait for PaddleOCR"
            _probe_done()
            assert _paddle_state() == "unusable" and "DLL load failed" in _PADDLE_UNUSABLE
            saved = json.loads(_paddle_verdict_file().read_text(encoding="utf-8"))
            assert saved["versions"] == "paddleocr 9.9" and saved["usable"] is False
            detect_title_boxes(str(pp))
            assert len(tries) == 1, "PaddleOCR is not tried again once it failed"
            _fresh_session()                               # a new session, same version
            assert _paddle_unusable() and not tries
            _fresh_session("paddleocr 10.0")               # updated: tried again
            assert _paddle_state() == ""

            # 2. works (the updated version): the reader answers first,
            # PaddleOCR from then on
            _fresh_session("paddleocr 10.0")
            _get_paddle_ocr = _works                       # noqa: F811
            t0 = _time.monotonic()
            first = detect_title_boxes(str(pp))
            assert "PLAYING" in first[0].text.replace(" ", "") and _time.monotonic() - t0 < 10
            _probe_done()
            assert _paddle_state() == "usable" and len(tries) == 1
            second = detect_title_boxes(str(pp))
            assert second[0].text == "FAKE PADDLE" and (second[0].x, second[0].w) == (10, 290)
            _fresh_session("paddleocr 10.0")
            assert _paddle_state() == "usable", "the verdict is kept for the next session"

            # 3. not installed: no try at all
            _fresh_session("")
            detect_title_boxes(str(pp))
            assert _paddle_state() == "unusable" and not tries and _PADDLE_PROBE is None
            models_dir = lambda: real_models                # noqa: E731
            print("ok: Detect text: reader at once, PaddleOCR checked in the background, "
                  "used when it works")
        else:
            print("skipped: the built-in text reader's models are not downloaded")

    # --- automatic title translation -------------------------------------
    # reads at several sizes: a box over two stacked lines gives way to the
    # lines; the same text read twice is one box with the other reading kept;
    # bits of letters inside a big line box are not lines
    two = [OcrBox(100, 100, 800, 90, "英文歌分享 #6", 0.97), OcrBox(100, 200, 800, 110, "CLOSER", 0.95),
           OcrBox(95, 95, 815, 220, "英文歌 CLOSER", 0.9)]
    merged = _merge_reads(two)
    assert sorted(b.text for b in merged) == ["CLOSER", "英文歌分享 #6"], merged
    twice = _merge_reads([OcrBox(190, 470, 570, 230, "七日逆裝！", 0.83),
                          OcrBox(198, 478, 566, 216, "七日逆襲", 0.92)])
    assert len(twice) == 1 and twice[0].text == "七日逆襲" and twice[0].alts == ("七日逆裝！",), twice
    bits = _merge_reads([OcrBox(0, 0, 1280, 192, "DATA ANALYSIS", 0.98),
                         OcrBox(478, 54, 296, 110, "AN", 0.99), OcrBox(1156, 14, 100, 122, "S", 0.93),
                         OcrBox(762, 86, 69, 44, "-", 0.53), OcrBox(157, 53, 36, 13, "A", 0.69)])
    assert bits[0].text == "DATA ANALYSIS" and "AN" in bits[0].alts, bits
    assert _reader_sides(1280, 720) == [1280, 640, 352] and _reader_sides(480, 360) == [480, 352]
    assert output_name("C:/x/海报 1.jpg") == "海报 1_KH.png"
    assert output_name("/x/p.webp", "en") == "p_EN.png"
    assert _clean_title_text("“មីមួយចាន！” | ល្បី100 ។", "km") == "មីមួយចាន! | ល្បី១០០ ។"
    assert _clean_title_text("Hello、World。", "en") == "Hello,World"
    assert _ai_box([100, 50, 300, 950]) == (100.0, 50.0, 300.0, 950.0)
    assert _ai_box([1, 2, 3]) is None and _ai_box([500, 0, 400, 10]) is None
    assert _ai_box(["a", 1, 2, 3]) is None and _ai_box([0, 0, 10, 10]) is None
    reads = [OcrBox(167, 429, 1113, 291, "七日逆装震写全校！", 0.8, ("七日逆襲", "震鹭全校！")),
             OcrBox(1034, 210, 246, 134, "全校震！", 0.85, ("全校震露！",)),
             OcrBox(754, 338, 148, 132, "100", 0.95)]
    by_parts = _pick_title_boxes(reads, ["七日逆襲", "震驚全校！"], None, 1280, 720)
    assert [b.text for b in by_parts] == ["七日逆装震写全校！"], by_parts
    by_place = _pick_title_boxes(reads, [], (600, 130, 1000, 1000), 1280, 720)
    assert [b.text for b in by_place] == ["七日逆装震写全校！"], by_place
    assert _pick_title_boxes(reads, ["nothing like it"], None, 1280, 720) == []
    assert _script_share("មីមួយចាន", "km") == 1.0 and _script_share("CLOSER", "km") == 0
    assert 0 < _script_share("មីមួយចាន CLOSER", "km") < 1

    from app.core.errors import ProviderError as _PE

    class _StubAI:
        """Answers like the AI: what it was asked is kept."""
        def __init__(self, replies):
            self.replies, self.asked = list(replies), []

        def chat(self, **kw):
            self.asked.append(kw)
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return type("R", (), {"text": json.dumps(r, ensure_ascii=False)})()

    with tempfile.TemporaryDirectory() as td3:
        p3 = Path(td3) / "p.png"
        Image.new("RGB", (1280, 720), (30, 40, 60)).save(p3)
        ok_reply = {"status": "ok", "headline": ["七日逆襲！", "震驚全校！"], "language": "zh",
                    "parts": ["七日逆装震写全校！"], "box": [600, 130, 1000, 1000],
                    "meaning": "Seven-day comeback! The whole school is shocked!",
                    "translation": ["វាយបកក្នុង៧ថ្ងៃ!", "ធ្វើឲ្យសាលាទាំងមូលភ្ញាក់ផ្អើល!"]}
        ai = _StubAI([ok_reply])
        hl = read_headline(str(p3), ai, "km", boxes=reads)
        assert hl["status"] == "ok" and hl["seen"] and [b.text for b in hl["boxes"]] == \
            ["七日逆装震写全校！"], hl
        assert hl["text"] == "វាយបកក្នុង៧ថ្ងៃ! | ធ្វើឲ្យសាលាទាំងមូលភ្ញាក់ផ្អើល!", hl["text"]
        assert hl["source"] == "七日逆襲！ | 震驚全校！" and hl["lang"] == "zh"
        asked = ai.asked[0]
        assert asked["feature"] == "poster" and asked["json_mode"]
        assert asked["images"][0][0] == "image/jpeg" and asked["images"][0][1][:2] == b"\xff\xd8"
        assert "七日逆装震写全校！" in asked["messages"][0]["content"]
        assert "noun FIRST" in asked["messages"][0]["content"], "the Khmer noun order is asked"
        # a model that cannot see pictures: the reader's text instead
        ai = _StubAI([_PE("gemini HTTP 400: image input is not supported", provider="gemini",
                          status_code=400), dict(ok_reply, box=None)])
        hl = read_headline(str(p3), ai, "km", boxes=reads)
        assert hl["status"] == "ok" and not hl["seen"] and "images" not in ai.asked[1]
        assert "cannot see the poster" in ai.asked[1]["messages"][0]["content"]
        # Chinese left in the translation: asked once more; the noun order put right
        ai = _StubAI([dict(ok_reply, headline=["班长"], parts=["七日逆襲"],
                           translation=["班长ថ្នាក់ប្រធាន"]),
                      {"translation": ["ថ្នាក់ប្រធាន"]}])
        hl = read_headline(str(p3), ai, "km", boxes=reads)
        assert len(ai.asked) == 2 and "Chinese" in ai.asked[1]["messages"][0]["content"]
        assert hl["text"] == "ប្រធានថ្នាក់", hl["text"]
        # already Khmer / no title / no AI at all
        assert read_headline(str(p3), _StubAI([{"status": "already"}]), "km",
                             boxes=reads)["status"] == "already"
        assert read_headline(str(p3), _StubAI([dict(ok_reply, headline=["មីមួយចាន"])]), "km",
                             boxes=reads)["status"] == "already"
        assert read_headline(str(p3), _StubAI([{"status": "none"}]), "km",
                             boxes=reads)["status"] == "none"
        try:
            read_headline(str(p3), _StubAI([_PE("all gemini keys are rate-limited",
                                                provider="gemini", status_code=429)]),
                          "km", boxes=reads)
            raise AssertionError("no AI must stop the work")
        except _PE:
            pass
        print("ok: the title read and translated: picture + reader's texts, fallbacks, checks")

        from app.engines import subtitle_ocr as _so
        if _so.available():
            # a whole poster: a white block-letter title with a black outline
            scene = np.zeros((720, 1280, 3), np.uint8)
            scene[..., 0] = np.linspace(30, 120, 720, dtype=np.float64)[:, None].astype(np.uint8)
            scene[..., 1] = 60
            scene[..., 2] = np.linspace(90, 30, 1280, dtype=np.float64)[None, :].astype(np.uint8)
            poster = Image.fromarray(scene)
            arial = "C:/Windows/Fonts/arialbd.ttf"
            if os.path.isfile(arial):
                ImageDraw.Draw(poster).text((180, 420), "BIG SALE", fill=(255, 255, 255),
                                            font=ImageFont.truetype(arial, 170),
                                            stroke_width=8, stroke_fill=(0, 0, 0))
                src = Path(td3) / "海报 sale.jpg"
                poster.save(src, quality=95)
                ai = _StubAI([{"status": "ok", "headline": ["BIG SALE"], "language": "en",
                               "parts": ["BIG SALE"], "meaning": "A big sale",
                               "translation": ["ការលក់ធំ"]}])
                out = Path(td3) / output_name(str(src))
                r = auto_translate_title(str(src), str(out), ai, "km",
                                         fonts_dir=_bundled_fonts_dir())
                assert r["status"] == "done" and out.is_file(), r
                checks = dict((k, ok) for k, ok, _d in r["checks"])
                assert checks == {"removed": True, "letters": True, "numbers": True,
                                  "shaping": True, "size": True}, r["checks"]
                assert r["spec"].font == "Koulen", r["spec"].font    # block letters
                assert r["spec"].stroke_color.lower() in ("#000000", "#010101", "#020202") \
                    or _lum(_parse_color(r["spec"].stroke_color)[:3]) < 40, r["spec"]
                before = np.asarray(Image.open(src).convert("RGB")).astype(np.int32)
                after = np.asarray(Image.open(out).convert("RGB")).astype(np.int32)
                assert after.shape == before.shape
                assert np.abs(after - before)[420:620, 180:1100].mean() > 20, "redrawn"
                assert np.abs(after - before)[:250].mean() < 3, "the rest is kept"
                # many: each its own; a picture without a title is skipped;
                # no AI at all stops the batch
                plain = Path(td3) / "plain.png"
                Image.fromarray(scene).save(plain)
                ai = _StubAI([{"status": "ok", "headline": ["BIG SALE"], "language": "en",
                               "parts": ["BIG SALE"], "meaning": "A big sale",
                               "translation": ["ការលក់ធំ"]},
                              {"status": "none"},
                              _PE("no gemini key", provider="gemini", status_code=401)])
                outd = Path(td3) / "out"
                rs = auto_translate_titles([str(src), str(plain), str(src)], str(outd), ai, "km",
                                           fonts_dir=_bundled_fonts_dir())
                assert [x["status"] for x in rs] == ["done", "none", "error"], rs
                assert rs[2].get("stopped") and (outd / "海报 sale_KH.png").is_file()
                assert not (outd / "plain_KH.png").exists()
                kept = auto_translate_titles([str(src)], str(outd), _StubAI([]), "km",
                                             overwrite=False)
                assert kept[0]["status"] == "kept"
                print("ok: a poster's title translated and redrawn; many pictures, "
                      "each its own (_KH.png)")
        else:
            print("skipped: the built-in text reader's models are not downloaded")

    print("PASS")
