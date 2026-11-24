"""FFmpeg-based media probing, encoder detection, overlay compositing and export.

Public surface (see ARCHITECTURE.md, section app/engines/media_exporter.py):

    MediaInfo, probe(), detect_encoders(), OverlayConfig, ExportPreset,
    build_filtergraph(), overlay_inputs(), render(), cut_segments(),
    RenderJob, RenderQueue, save_preset(), load_presets(),
    atempo_chain(), plan_parts(), part_spans(), shift_subtitles()
    (speed changes, and episodes cut from one long video),
    output_size(), watermark_layer(), prepare_watermark(), preview_frame()
    (the watermark: a logo and/or text, at a 3x3 place or tiled, drawn the
    same in the Overlay tab's preview and in the export)

No Qt imports, no heavy deps.  Everything shells out to the bundled (or PATH)
ffmpeg/ffprobe with CREATE_NO_WINDOW and utf-8/replace decoding.  All ffmpeg
stderr lines are streamed to LOG_BUS so the UI console dock can show them.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Callable

from app.core.errors import AppError, CancelledError, EngineNotAvailable
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import app_data_dir, ffmpeg_path, ffprobe_path, fonts_dir, temp_dir

log = get_logger("media_exporter")

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Fraction of the main video width used for a corner-positioned chroma overlay.
CHROMA_WIDTH_FRAC = 0.4
# Margin in pixels for positioned overlays (watermark / chroma / ad).
OVERLAY_MARGIN = 16


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #

@dataclass
class MediaInfo:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str = ""
    acodec: str = ""
    bitrate: int = 0


def _parse_fraction(frac: str) -> float:
    """Parse an ffprobe rational like '30000/1001' (or '25') into a float."""
    try:
        if "/" in frac:
            num_s, den_s = frac.split("/", 1)
            num, den = float(num_s), float(den_s)
            if den == 0:
                return 0.0
            return num / den
        return float(frac)
    except (ValueError, TypeError):
        return 0.0


def probe(path: str) -> MediaInfo:
    """Probe a media file with ffprobe (-print_format json -show_format -show_streams)."""
    if not os.path.isfile(path):
        raise AppError(
            f"Media file not found: {path}",
            f"រកមិនឃើញឯកសារមេឌៀ៖ {path}",
        )
    cmd = [
        ffprobe_path(), "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
    ]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW, timeout=60,
        )
    except FileNotFoundError:
        raise EngineNotAvailable(
            "ffprobe not found. Place ffprobe.exe in the app's bin folder or install FFmpeg on PATH.",
            "រកមិនឃើញ ffprobe ទេ។ សូមដាក់ ffprobe.exe ក្នុងថត bin របស់កម្មវិធី ឬតំឡើង FFmpeg ក្នុង PATH។",
        )
    except subprocess.TimeoutExpired:
        raise AppError(
            f"ffprobe timed out while probing: {path}",
            f"ffprobe អស់ពេលកំណត់ ពេលកំពុងវិភាគ៖ {path}",
        )
    if cp.returncode != 0:
        tail = (cp.stderr or "").strip().splitlines()[-5:]
        raise AppError(
            f"ffprobe failed for {path}: " + " | ".join(tail),
            f"ffprobe បរាជ័យសម្រាប់ {path}៖ " + " | ".join(tail),
        )
    try:
        data = json.loads(cp.stdout or "{}")
    except json.JSONDecodeError:
        raise AppError(
            f"ffprobe returned invalid JSON for {path}",
            f"ffprobe បានផ្តល់ JSON មិនត្រឹមត្រូវ សម្រាប់ {path}",
        )

    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    info = MediaInfo()
    try:
        info.duration = float(fmt.get("duration") or 0.0)
    except (ValueError, TypeError):
        info.duration = 0.0
    try:
        info.bitrate = int(fmt.get("bit_rate") or 0)
    except (ValueError, TypeError):
        info.bitrate = 0

    for st in streams:
        if st.get("codec_type") == "video" and not info.vcodec:
            info.vcodec = st.get("codec_name") or ""
            info.width = int(st.get("width") or 0)
            info.height = int(st.get("height") or 0)
            info.fps = _parse_fraction(str(st.get("avg_frame_rate") or "0/0"))
            if info.fps <= 0:
                info.fps = _parse_fraction(str(st.get("r_frame_rate") or "0/0"))
            if info.duration <= 0:
                try:
                    info.duration = float(st.get("duration") or 0.0)
                except (ValueError, TypeError):
                    pass
        elif st.get("codec_type") == "audio" and not info.acodec:
            info.acodec = st.get("codec_name") or ""
    return info


# --------------------------------------------------------------------------- #
# Encoder detection
# --------------------------------------------------------------------------- #

_ENCODER_CACHE: list[str] | None = None
# NVIDIA NVENC, AMD AMF, Intel Quick Sync
_HW_CANDIDATES = ["h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf",
                  "h264_qsv", "hevc_qsv"]


def _probe_signature() -> str:
    """Changes when the candidate list or the installed GPUs change, so a
    persisted encoder list is re-probed after a GPU swap."""
    try:
        from app.core.gpu import list_gpus
        gpus = sorted(f"{g.vendor}:{g.name}" for g in list_gpus())
    except Exception:
        gpus = []
    return "|".join(_HW_CANDIDATES) + "#" + ",".join(gpus)


def best_h264_encoder(config=None) -> str:
    """The detected hardware H.264 encoder (NVENC > AMF > QSV), or libx264
    when there is none or Settings → "Prefer hardware encoding" is off."""
    try:
        if config is not None and not config.get("export.hw", True):
            return "libx264"
    except Exception:
        pass
    found = known_encoders(config)
    for enc in ("h264_nvenc", "h264_amf", "h264_qsv"):
        if enc in found:
            return enc
    return "libx264"


def known_encoders(config=None) -> list[str]:
    """Encoders already detected (module cache or persisted config), without
    probing; [] if detection has not run. Reads the old plain-list format too."""
    if _ENCODER_CACHE is not None:
        return list(_ENCODER_CACHE)
    try:
        cached = config.get("export.encoders") if config is not None else None
    except Exception:
        cached = None
    if isinstance(cached, dict):
        cached = cached.get("found")
    return [str(e) for e in cached] if isinstance(cached, list) else []


def detect_encoders(config=None) -> list[str]:
    """Detect usable video encoders by test-encoding one black frame each.

    Always includes libx264/libx265; probes NVENC, AMF and QSV (H.264 and
    HEVC). Result is cached in a module variable and, when a ConfigManager is
    passed, persisted under config key "export.encoders" as
    {"signature": ..., "found": [...]}.
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None:
        return list(_ENCODER_CACHE)
    sig = _probe_signature()
    if config is not None:
        try:
            cached = config.get("export.encoders")
        except Exception:
            cached = None
        if (isinstance(cached, dict) and cached.get("signature") == sig
                and isinstance(cached.get("found"), list) and cached["found"]):
            _ENCODER_CACHE = [str(e) for e in cached["found"]]
            return list(_ENCODER_CACHE)

    available: list[str] = []
    for enc in _HW_CANDIDATES:
        cmd = [
            ffmpeg_path(), "-hide_banner", "-v", "error",
            # hardware encoders reject tiny frames (NVENC min ~145x49)
            "-f", "lavfi", "-i", "color=black:s=320x240:d=0.1",
            "-c:v", enc, "-f", "null", "-",
        ]
        try:
            cp = subprocess.run(
                cmd, capture_output=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW, timeout=30,
            )
            if cp.returncode == 0:
                available.append(enc)
                log.info("Encoder available: %s", enc)
            else:
                log.info("Encoder not available: %s", enc)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            log.warning("Encoder test could not run for %s", enc)

    available.extend(["libx264", "libx265"])
    _ENCODER_CACHE = available
    if config is not None:
        try:
            config.set("export.encoders", {"signature": sig, "found": available})
        except Exception:
            log.warning("Could not persist encoder list to config", exc_info=True)
    return list(available)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class OverlayConfig:
    watermark: str | None = None          # logo picture (PNG with transparency, or JPG)
    watermark_opacity: float = 0.6
    watermark_pos: str = "top_right"      # one of GRID_POSITIONS, or "tile"
    watermark_scale: float = 0.15         # logo width (text: its height x 6) / frame width
    watermark_text: str = ""              # e.g. "@YourChannel": alone, or under the logo
    watermark_color: str = "auto"         # text colour; "auto" = white with a soft shadow
    watermark_font: str = "Kantumruy Pro"
    # internal: `watermark` is the finished full-frame layer already
    # (render() makes it with prepare_watermark())
    watermark_full: bool = False
    banner: str | None = None
    banner_height_ratio: float = 0.12
    chroma_video: str | None = None
    chroma_color: str = "0x00ff00"
    chroma_similarity: float = 0.15
    chroma_pos: str = "bottom_right"
    blur_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    blur_strength: int = 20
    ad_video: str | None = None
    ad_start: float = 0.0
    ad_pos: str = "top_left"
    ad_scale: float = 0.3

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blur_regions"] = [list(r) for r in self.blur_regions]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OverlayConfig":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        if (d or {}).get("watermark_on") is False:
            # the Overlay tab keeps a switched-off watermark's logo and text
            known["watermark"], known["watermark_text"] = None, ""
        regions = known.get("blur_regions") or []
        known["blur_regions"] = [
            (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            for r in regions if len(r) >= 4
        ]
        return cls(**known)


@dataclass
class ExportPreset:
    name: str = "Default"
    container: str = "mp4"
    vcodec: str = "libx264"
    bitrate: str = "6M"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    audio_bitrate: str = "192k"
    # when width x height has a different aspect than the source:
    # "" (stretch-scale, legacy) | "blur" | "pad" | "crop" | "smart"
    fit_mode: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExportPreset":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- #
# Filtergraph
# --------------------------------------------------------------------------- #

_POS_MAP = {
    "top_left": (str(OVERLAY_MARGIN), str(OVERLAY_MARGIN)),
    "top_right": (f"main_w-overlay_w-{OVERLAY_MARGIN}", str(OVERLAY_MARGIN)),
    "bottom_left": (str(OVERLAY_MARGIN), f"main_h-overlay_h-{OVERLAY_MARGIN}"),
    "bottom_right": (f"main_w-overlay_w-{OVERLAY_MARGIN}",
                     f"main_h-overlay_h-{OVERLAY_MARGIN}"),
    "center": ("(main_w-overlay_w)/2", "(main_h-overlay_h)/2"),
    "top_center": ("(main_w-overlay_w)/2", str(OVERLAY_MARGIN)),
    "bottom_center": ("(main_w-overlay_w)/2", f"main_h-overlay_h-{OVERLAY_MARGIN}"),
    "middle_left": (str(OVERLAY_MARGIN), "(main_h-overlay_h)/2"),
    "middle_right": (f"main_w-overlay_w-{OVERLAY_MARGIN}", "(main_h-overlay_h)/2"),
}

# the watermark's places, row by row (a 3x3 grid); "tile" repeats it
GRID_POSITIONS = ["top_left", "top_center", "top_right",
                  "middle_left", "center", "middle_right",
                  "bottom_left", "bottom_center", "bottom_right"]


def _pos_xy(pos: str) -> tuple[str, str]:
    return _POS_MAP.get((pos or "").strip().lower(), _POS_MAP["top_right"])


def _even(v: int) -> int:
    v = int(v)
    if v % 2:
        v -= 1
    return max(2, v)


def _clamp_box(region, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of a blur box cut to the frame; w/h <= 0 when nothing of
    it is inside. An unknown frame size (0) leaves the box as given."""
    x, y, w, h = (int(region[0]), int(region[1]), int(region[2]), int(region[3]))
    x, y = max(0, x), max(0, y)
    if frame_w > 0:
        w = min(w, frame_w - x)
    if frame_h > 0:
        h = min(h, frame_h - y)
    return x, y, w, h


def _sub_path_escape(path: str) -> str:
    r"""Escape a filesystem path for use inside the subtitles filter.

    Backslashes -> forward slashes; then escape single quotes and colons
    (drive letters) so the result can be wrapped in single quotes:
    C:\a\b.srt -> C\:/a/b.srt
    """
    p = str(path).replace("\\", "/")
    p = p.replace("'", r"\'")
    p = p.replace(":", r"\:")
    return p


def _plain_srt_to_ass(srt_path: str, font_name: str,
                      size: tuple[int, int] = (1920, 1080)) -> str:
    """Plain .srt -> .ass in temp_dir(): white text with a thin outline in
    `font_name` — the look ffmpeg gives an .srt — wrapped for `size`."""
    from app.engines.subtitles import SubStyle, read_srt, write_ass

    style = SubStyle(name="Plain", font=font_name, size_ratio=16 / 288,
                     outline_ratio=1 / 16, shadow=0.0, margin_v_ratio=10 / 288,
                     max_lines=3)
    out = str(temp_dir() / f"subs_{uuid.uuid4().hex[:8]}.ass")
    return write_ass(read_srt(srt_path), out, size[0], size[1], style)


def _subtitles_filter(srt_path: str, font_name: str,
                      size: tuple[int, int] | None = None) -> str:
    # Always the `ass` filter: ffmpeg's `subtitles` filter renders Khmer
    # without complex shaping (coeng drawn as a mark, vowels not reordered).
    # Styled .ass files carry their own fonts/colours; .srt is converted.
    if not srt_path.lower().endswith((".ass", ".ssa")):
        srt_path = _plain_srt_to_ass(srt_path, font_name, size or (1920, 1080))
    esc = _sub_path_escape(srt_path)
    fdir = _sub_path_escape(fonts_dir().as_posix())
    return f"ass=filename='{esc}':fontsdir='{fdir}':shaping=complex"


def overlay_inputs(overlays: OverlayConfig | None) -> list[str]:
    """Extra ffmpeg input files needed by build_filtergraph(), in fixed order.

    Order: chroma_video, watermark, banner, ad_video (only ones that are set).
    Input [0] is always the main video; these become inputs [1..N] in this
    exact order, matching the stream indices referenced by build_filtergraph().
    """
    if overlays is None:
        return []
    return [p for p in (overlays.chroma_video, overlays.watermark,
                        overlays.banner, overlays.ad_video) if p]


def _fit_parts(cur: str, src_w: int, src_h: int, W: int, H: int, mode: str,
               track: list | None, out_lbl: str) -> list[str]:
    """Filter parts that fit the source into W x H with `mode`."""
    mode = (mode or "").lower()
    if mode == "smart" and src_w * H > src_h * W:
        from app.engines.scenes import crop_x_expr
        sw = _even(round(src_w * H / src_h))
        x = crop_x_expr(track or [], sw, W)
        return [f"{cur}scale={sw}:{H},crop={W}:{H}:'{x}':0,setsar=1{out_lbl}"]
    if mode in ("crop", "smart"):
        return [f"{cur}scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},setsar=1{out_lbl}"]
    if mode == "pad":
        return [f"{cur}scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1{out_lbl}"]
    # blur: blurred, cropped copy of the frame behind the fitted frame
    return [f"{cur}split=2[fbga][ffga]",
            f"[fbga]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},gblur=sigma=24,eq=brightness=-0.08[fbg]",
            f"[ffga]scale={W}:{H}:force_original_aspect_ratio=decrease[ffg]",
            f"[fbg][ffg]overlay=(W-w)/2:(H-h)/2,setsar=1{out_lbl}"]


def output_size(src_info: MediaInfo, preset: "ExportPreset | None") -> tuple[int, int]:
    """The finished frame size: the preset's, else the source's (one side
    given: the other follows the source's shape)."""
    src_w = src_info.width or 1920
    src_h = src_info.height or 1080
    if preset is not None:
        if preset.width > 0 and preset.height > 0:
            return preset.width, preset.height
        if preset.width > 0:
            return preset.width, _even(round(src_h * preset.width / max(1, src_w)))
        if preset.height > 0:
            return _even(round(src_w * preset.height / max(1, src_h))), preset.height
    return src_w, src_h


def _wm_margin(w: int, h: int) -> int:
    return max(OVERLAY_MARGIN, round(0.02 * min(w, h)))


def _grid_xy(pos: str, size: tuple[int, int], item: tuple[int, int],
             margin: int) -> tuple[int, int]:
    w, h = size
    iw, ih = item
    k = GRID_POSITIONS.index(pos) if pos in GRID_POSITIONS else GRID_POSITIONS.index("top_right")
    row, col = divmod(k, 3)
    return ((margin, (w - iw) // 2, w - iw - margin)[col],
            (margin, (h - ih) // 2, h - ih - margin)[row])


def _hex_rgb(color: str) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})(?:[0-9a-fA-F]{2})?", (color or "").strip())
    if not m:
        return None
    v = m.group(1)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _watermark_item(ov: OverlayConfig, size: tuple[int, int]):
    """The watermark itself at full strength (the logo, the text, or the logo
    with the text under it), PIL RGBA; None when there is neither."""
    from PIL import Image, ImageFilter
    w, h = size
    parts = []
    if ov.watermark and os.path.isfile(ov.watermark):
        try:
            with Image.open(ov.watermark) as im:
                logo = im.convert("RGBA")
            lw = max(8, int(w * ov.watermark_scale))
            logo = logo.resize((lw, max(1, round(logo.height * lw / max(1, logo.width)))),
                               Image.LANCZOS)
            parts.append(logo)
        except Exception as exc:
            log.warning("watermark logo not usable (%s)", exc)
    text = (ov.watermark_text or "").strip()
    if text:
        from app.engines import poster_designer as pd
        px = max(10, int(w * ov.watermark_scale / 6))
        font = pd._resolve_font(ov.watermark_font or "Kantumruy Pro", fonts_dir())
        mask, mw, mh = pd._draw_shaped_text((w, h), text, font, px, 0)
        rgb = _hex_rgb(ov.watermark_color) or (255, 255, 255)        # "auto": white
        glyphs = Image.new("RGBA", (mw, mh), rgb + (255,))
        glyphs.putalpha(mask)
        # a soft shadow in the opposite tone keeps it readable on any picture
        lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
        shadow = (0, 0, 0) if lum > 0.4 else (255, 255, 255)
        pad = max(3, px // 10)
        smask = Image.new("L", (mw + 2 * pad, mh + 2 * pad), 0)
        smask.paste(mask, (pad + max(1, px // 24), pad + max(1, px // 24)))
        smask = smask.filter(ImageFilter.GaussianBlur(max(1.5, px / 18)))
        smask = smask.point(lambda v: int(v * 0.75))
        item = Image.new("RGBA", smask.size, shadow + (255,))
        item.putalpha(smask)
        item.alpha_composite(glyphs, (pad, pad))
        parts.append(item)
    if not parts:
        return None
    iw = max(p.width for p in parts)
    gap = max(2, int(h * 0.006))
    ih = sum(p.height for p in parts) + gap * (len(parts) - 1)
    item = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    y = 0
    for p in parts:
        item.alpha_composite(p, ((iw - p.width) // 2, y))
        y += p.height + gap
    return item


def watermark_layer(size: tuple[int, int], ov: OverlayConfig):
    """The watermark on a transparent frame of `size` (PIL RGBA), its opacity
    applied: at one of the GRID_POSITIONS (with a margin), or repeated at 30°
    across the whole frame ("tile"). None when there is no watermark."""
    from PIL import Image
    from app.engines import poster_designer as pd
    item = _watermark_item(ov, size)
    if item is None:
        return None
    op = max(0.0, min(1.0, float(ov.watermark_opacity)))
    item.putalpha(item.getchannel("A").point(lambda v: int(v * op)))
    w, h = size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if ov.watermark_pos == "tile":
        tile = item.rotate(30, expand=True, resample=Image.BICUBIC)
        sx, sy = max(8, int(tile.width * 1.8)), max(8, int(tile.height * 2.2))
        for row, y in enumerate(range(-tile.height // 2, h, sy)):
            off = sx // 2 if row % 2 else 0
            for x in range(-tile.width // 2 + off, w, sx):
                pd._alpha_paste(canvas, tile, x, y)
        return canvas
    x, y = _grid_xy(ov.watermark_pos, (w, h), item.size, _wm_margin(w, h))
    pd._alpha_paste(canvas, item, x, y)
    return canvas


def prepare_watermark(ov: OverlayConfig, size: tuple[int, int]) -> OverlayConfig:
    """`ov` with its watermark (logo and/or text) drawn as one full-frame PNG
    layer of the finished `size`, for build_filtergraph to lay over the
    video as it is; `ov` itself when there is no watermark or it is drawn."""
    if ov.watermark_full or not (ov.watermark or (ov.watermark_text or "").strip()):
        return ov
    layer = watermark_layer(size, ov)
    if layer is None:
        return _dc_replace(ov, watermark=None, watermark_text="")
    out = temp_dir() / f"wm_{uuid.uuid4().hex[:8]}.png"
    layer.save(out)
    return _dc_replace(ov, watermark=str(out), watermark_text="", watermark_full=True)


def _corner_box(pos: str, size: tuple[int, int], bw: int, bh: int) -> tuple[int, int, int, int]:
    """Where ffmpeg's _POS_MAP puts a bw x bh overlay (x, y, w, h)."""
    w, h = size
    m = OVERLAY_MARGIN
    xs = {"left": m, "center": (w - bw) // 2, "right": w - bw - m}
    ys = {"top": m, "middle": (h - bh) // 2, "bottom": h - bh - m}
    p = (pos or "").strip().lower()
    if p == "center":
        return xs["center"], ys["middle"], bw, bh
    row, _, col = p.partition("_")
    if row not in ys or col not in xs:
        row, col = "top", "right"
    return xs[col], ys[row], bw, bh


def preview_frame(frame, ov: OverlayConfig | None):
    """What the overlays do to one frame (a PIL picture at the source size,
    returned RGB): the blurred boxes, the banner and the watermark as the
    export draws them; the chroma-key and ad videos as tinted boxes where
    they will play (their own pictures are not read)."""
    from PIL import Image, ImageDraw, ImageFilter
    ov = ov or OverlayConfig()
    img = frame.convert("RGB")
    w, h = img.size
    for region in ov.blur_regions or []:
        if len(region) < 4:
            continue
        bx, by, bw, bh = _clamp_box(region, w, h)
        if bw <= 0 or bh <= 0:
            continue
        box = img.crop((bx, by, bx + bw, by + bh)).filter(
            ImageFilter.GaussianBlur(max(1, int(ov.blur_strength))))
        img.paste(box, (bx, by))
    canvas = img.convert("RGBA")
    if ov.banner and os.path.isfile(ov.banner):
        try:
            bh = max(2, int(h * ov.banner_height_ratio))
            with Image.open(ov.banner) as im:
                band = im.convert("RGBA").resize((w, bh), Image.LANCZOS)
            canvas.alpha_composite(band, (0, h - bh))
        except Exception as exc:
            log.info("banner not usable (%s)", exc)
    marks = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pen = ImageDraw.Draw(marks)
    line = max(2, w // 400)
    for path, pos, frac, rgb in ((ov.chroma_video, ov.chroma_pos, CHROMA_WIDTH_FRAC, (63, 185, 80)),
                                 (ov.ad_video, ov.ad_pos, ov.ad_scale, (240, 136, 62))):
        if not path:
            continue
        bw = max(8, int(w * frac))
        x, y, bw, bh = _corner_box(pos, (w, h), bw, max(8, bw * 9 // 16))
        pen.rectangle((x, y, x + bw, y + bh), fill=rgb + (70,), outline=rgb + (255,), width=line)
        cx, cy, r = x + bw // 2, y + bh // 2, max(6, bh // 6)
        pen.polygon([(cx - r // 2, cy - r), (cx - r // 2, cy + r), (cx + r, cy)],
                    fill=(255, 255, 255, 210))
    canvas.alpha_composite(marks)
    layer = watermark_layer((w, h), ov)
    if layer is not None:
        canvas.alpha_composite(layer)
    return canvas.convert("RGB")


def _intro_graph(graph: str, k: int, size: tuple[int, int], fps: float, seconds: float,
                 audio_in: str, speed: float) -> str:
    """`graph` (its picture ends in [vout], or none: input 0 as it is) with
    `seconds` of the still picture of input `k` before it, fitted over a
    blurred copy of itself as the DL tab's poster second is, and as much
    silence before the sound (`audio_in` -> [aout]; "" = no sound). The
    speed change stays on the video's own part."""
    W, H = _even(size[0]), _even(size[1])
    r = f"{fps:.6g}"
    parts: list[str] = []
    if graph:
        parts.append(graph[: -len("[vout]")] + "[vmain]")
        vin = "[vmain]"
    else:
        vin = "[0:v]"
    parts += [
        f"{vin}scale={W}:{H},setsar=1,fps={r},format=yuv420p[mv]",
        f"[{k}:v]split=2[ipa][ipb]",
        f"[ipa]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
        f"boxblur=20:2,eq=brightness=-0.08[ipbg]",
        f"[ipb]scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2[ipfg]",
        f"[ipbg][ipfg]overlay=(W-w)/2:(H-h)/2:shortest=1,setsar=1,fps={r},format=yuv420p[iv]",
        "[iv][mv]concat=n=2:v=1:a=0[vout]",
    ]
    if audio_in:
        fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
        chain = f"aresample=48000,{fmt}"
        if abs(speed - 1.0) > 1e-3:
            chain += "," + atempo_chain(speed)
        parts += [f"{audio_in}{chain}[ma]",
                  f"anullsrc=r=48000:cl=stereo,atrim=duration={seconds:.3f},{fmt}[sa]",
                  "[sa][ma]concat=n=2:v=0:a=1[aout]"]
    return ";".join(parts)


def build_filtergraph(src_info: MediaInfo, overlays: OverlayConfig,
                      srt_path: str | None, font_name: str = "Kantumruy Pro",
                      preset: "ExportPreset | None" = None,
                      crop_track: list | None = None) -> str:
    """Build the full -filter_complex string.

    Input layout: [0]=main video; [1..N]=files from overlay_inputs(overlays)
    in that fixed order (chroma, watermark, banner, ad).  The final video
    stream is labelled [vout].  Returns "" when no filtering is needed.

    Composition order: blur regions (in source pixels, so before anything
    rescales the frame), scale/fps normalization (from optional `preset`),
    chroma-key overlay, watermark, banner, ad video (time-windowed), and
    finally the subtitles filter (Windows-safe path escaping, bundled
    fontsdir, force_style FontName).
    """
    overlays = overlays or OverlayConfig()
    parts: list[str] = []
    cur = "[0:v]"
    n = 0

    def nxt() -> str:
        nonlocal n
        n += 1
        return f"[v{n}]"

    src_w = src_info.width or 1920
    src_h = src_info.height or 1080

    # 0) blur regions: picked on the source frame, so they are applied before
    #    scaling / reframing and clamped to it (a box wider than the scaled
    #    frame made crop fail; one past its edge blurred nothing)
    for i, region in enumerate(overlays.blur_regions):
        if len(region) < 4:
            continue
        bx, by, bw, bh = _clamp_box(region, src_info.width, src_info.height)
        if bw <= 0 or bh <= 0:
            continue
        parts.append(f"{cur}split=2[bs{i}a][bs{i}b]")
        parts.append(
            f"[bs{i}b]crop={bw}:{bh}:{bx}:{by},"
            f"gblur=sigma={max(1, overlays.blur_strength)}[bl{i}]"
        )
        lbl = nxt()
        parts.append(f"[bs{i}a][bl{i}]overlay={bx}:{by}{lbl}")
        cur = lbl

    # Effective output dimensions (after normalization) used to size overlays.
    eff_w, eff_h = output_size(src_info, preset)

    # 1) scale / fps normalization (or aspect fitting / reframing)
    norm: list[str] = []
    if (preset is not None and preset.fit_mode and preset.width > 0
            and preset.height > 0
            and src_w * preset.height != src_h * preset.width):
        lbl = nxt()
        parts.extend(_fit_parts(cur, src_w, src_h, _even(preset.width),
                                _even(preset.height), preset.fit_mode,
                                crop_track, lbl))
        cur = lbl
        if preset.fps > 0:
            norm.append(f"fps={preset.fps:g}")
    elif preset is not None:
        if preset.width > 0 or preset.height > 0:
            w = preset.width if preset.width > 0 else -2
            h = preset.height if preset.height > 0 else -2
            norm.append(f"scale={w}:{h}")
        if preset.fps > 0:
            norm.append(f"fps={preset.fps:g}")
    if norm:
        lbl = nxt()
        parts.append(f"{cur}{','.join(norm)}{lbl}")
        cur = lbl

    in_idx = 1  # next extra-input index, advances in overlay_inputs() order

    # 2) chroma-keyed video overlay
    if overlays.chroma_video:
        cw = _even(int(eff_w * CHROMA_WIDTH_FRAC))
        parts.append(
            f"[{in_idx}:v]colorkey={overlays.chroma_color}"
            f":{overlays.chroma_similarity:g}:0.05,scale={cw}:-2[ck]"
        )
        x, y = _pos_xy(overlays.chroma_pos)
        lbl = nxt()
        parts.append(f"{cur}[ck]overlay={x}:{y}{lbl}")
        cur = lbl
        in_idx += 1

    # 3) watermark: the finished full-frame layer (logo and/or text, placed or
    #    tiled by prepare_watermark), or a plain logo scaled and placed here
    if overlays.watermark and overlays.watermark_full:
        lbl = nxt()
        parts.append(f"{cur}[{in_idx}:v]overlay=0:0{lbl}")
        cur = lbl
        in_idx += 1
    elif overlays.watermark:
        ww = max(2, _even(int(eff_w * overlays.watermark_scale)))
        parts.append(
            f"[{in_idx}:v]format=rgba,"
            f"colorchannelmixer=aa={overlays.watermark_opacity:g},"
            f"scale={ww}:-1[wm]"
        )
        x, y = _pos_xy(overlays.watermark_pos)
        lbl = nxt()
        parts.append(f"{cur}[wm]overlay={x}:{y}{lbl}")
        cur = lbl
        in_idx += 1

    # 4) banner: full output width, fixed height ratio, glued to bottom
    if overlays.banner:
        bh = max(2, int(eff_h * overlays.banner_height_ratio))
        parts.append(f"[{in_idx}:v]scale={eff_w}:{bh}[bn]")
        lbl = nxt()
        parts.append(f"{cur}[bn]overlay=0:main_h-overlay_h{lbl}")
        cur = lbl
        in_idx += 1

    # 5) ad video overlay, enabled only for its own duration from ad_start
    if overlays.ad_video:
        aw = _even(int(eff_w * overlays.ad_scale))
        parts.append(f"[{in_idx}:v]scale={aw}:-2[ad]")
        ad_dur = 0.0
        try:
            ad_dur = probe(overlays.ad_video).duration
        except AppError:
            log.warning("Could not probe ad video duration: %s", overlays.ad_video)
        start = max(0.0, overlays.ad_start)
        if ad_dur > 0:
            enable = f"between(t,{start:g},{start + ad_dur:g})"
        else:
            enable = f"gte(t,{start:g})"
        x, y = _pos_xy(overlays.ad_pos)
        lbl = nxt()
        parts.append(f"{cur}[ad]overlay={x}:{y}:enable='{enable}'{lbl}")
        cur = lbl
        in_idx += 1

    # 6) subtitles — always last so text is burned over everything
    if srt_path:
        lbl = nxt()
        parts.append(f"{cur}{_subtitles_filter(srt_path, font_name, (eff_w, eff_h))}{lbl}")
        cur = lbl

    if not parts:
        return ""

    # Rename the final output label to [vout]
    last = parts[-1]
    assert last.endswith(cur)
    parts[-1] = last[: -len(cur)] + "[vout]"
    return ";".join(parts)


# --------------------------------------------------------------------------- #
# Subprocess execution with progress + cancel
# --------------------------------------------------------------------------- #

def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill an ffmpeg process and all of its children."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=15,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_ffmpeg(cmd: list[str], total_duration: float,
                progress: Callable[[float, str], None] | None,
                cancel: threading.Event | None,
                label: str = "ffmpeg",
                status_text: str = "Rendering") -> None:
    """Run an ffmpeg command that includes '-progress pipe:1 -nostats'.

    Parses out_time_us/out_time_ms key=value lines from stdout into progress
    fractions against total_duration; streams every stderr line to LOG_BUS;
    kills the process tree and raises CancelledError when cancel is set;
    raises AppError with the last 30 stderr lines on nonzero exit.
    """
    LOG_BUS.publish(f"[{label}] $ " + " ".join(shlex.quote(c) for c in cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError:
        raise EngineNotAvailable(
            "FFmpeg not found. Place ffmpeg.exe in the app's bin folder or install FFmpeg on PATH.",
            "រកមិនឃើញ FFmpeg ទេ។ សូមដាក់ ffmpeg.exe ក្នុងថត bin របស់កម្មវិធី ឬតំឡើង FFmpeg ក្នុង PATH។",
        )

    stderr_tail: deque[str] = deque(maxlen=60)

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                stderr_tail.append(line)
                LOG_BUS.publish(f"[{label}] {line}")

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    cancelled = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                cancelled = True
                _kill_tree(proc)
                break
            line = line.strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                # Both keys are microseconds in practice (ffmpeg quirk).
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                if total_duration > 0 and progress is not None and us >= 0:
                    frac = min(0.999, max(0.0, us / 1_000_000.0 / total_duration))
                    progress(frac, f"{status_text} {frac * 100:.0f}%")
            elif line == "progress=end" and progress is not None:
                progress(1.0, f"{status_text} 100%")
        # Wait for exit, still honouring cancel.
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                cancelled = True
                _kill_tree(proc)
            time.sleep(0.1)
    finally:
        t.join(timeout=5)
        try:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass

    if cancelled or (cancel is not None and cancel.is_set()):
        proc.wait()
        raise CancelledError("Operation cancelled")

    code = proc.wait()
    if code != 0:
        tail = "\n".join(list(stderr_tail)[-30:])
        raise AppError(
            f"FFmpeg failed (exit code {code}). Last output:\n{tail}",
            f"FFmpeg បានបរាជ័យ (កូដចាកចេញ {code})។ លទ្ធផលចុងក្រោយ៖\n{tail}",
        )


# --------------------------------------------------------------------------- #
# Speed, and episodes cut from one long video
# --------------------------------------------------------------------------- #

SPEED_MIN, SPEED_MAX = 0.25, 4.0
# a leftover shorter than this is added to the last episode without asking
# (rounding, or a few black frames at the end)
LEFTOVER_MIN = 1.0


def atempo_chain(speed: float) -> str:
    """The ffmpeg audio filter that plays `speed` times as fast with the
    pitch kept (voices stay natural). atempo takes 0.5-2.0 per step on older
    ffmpeg builds, so bigger changes are chained."""
    factors: list[float] = []
    t = float(speed)
    while t > 2.0:
        factors.append(2.0)
        t /= 2.0
    while t < 0.5:
        factors.append(0.5)
        t /= 0.5
    factors.append(t)
    return ",".join(f"atempo={f:.6g}" for f in factors)


def plan_parts(duration: float, part_seconds: float,
               speed: float = 1.0) -> tuple[int, float]:
    """How a video of `duration` s, played at `speed`, divides into episodes
    of `part_seconds` s of FINISHED video: (whole episodes, leftover seconds
    of finished video). A leftover within 0.5 s of a whole episode counts as
    one (rounding); one under LEFTOVER_MIN counts as none."""
    if duration <= 0 or part_seconds <= 0 or speed <= 0:
        return 0, 0.0
    total = duration / speed
    n = int(total // part_seconds)
    rem = total - n * part_seconds
    if part_seconds - rem < 0.5:
        n, rem = n + 1, 0.0
    if rem < LEFTOVER_MIN:
        rem = 0.0
    return n, rem


def part_spans(duration: float, part_seconds: float, speed: float = 1.0,
               leftover: str = "new") -> list[tuple[float, float]]:
    """(start, length) in SOURCE seconds of each episode. `leftover`: "new"
    makes the leftover an episode of its own, "join" adds it to the last
    whole episode. The last episode always runs to the end of the video."""
    n, rem = plan_parts(duration, part_seconds, speed)
    span = part_seconds * speed
    spans = [(i * span, span) for i in range(n)]
    if rem > 0 and (leftover != "join" or not spans):
        spans.append((n * span, duration - n * span))
    if spans:
        s, _ = spans[-1]
        spans[-1] = (s, duration - s)
    return spans


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_ASS_TIME_RE = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})(?:[.](\d{1,3}))?\s*$")


def _ass_seconds(text: str) -> float | None:
    m = _ASS_TIME_RE.match(text)
    if not m:
        return None
    frac = m.group(4) or "0"
    return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            + int(frac) / (10 ** len(frac)))


def shift_subtitles(path: str, start: float, length: float = 0.0) -> str:
    """A copy of a .srt / .ass / .ssa file for the part of the video that
    starts at `start` s and lasts `length` s (0 = to the end): lines outside
    it are dropped, the others moved so the part starts at 0 (a line running
    over an edge is cut there). Written to the temp folder; returns its path."""
    end = start + length if length > 0 else float("inf")
    ext = os.path.splitext(path)[1].lower()
    tag = uuid.uuid4().hex[:8]
    if ext in (".ass", ".ssa"):
        out = temp_dir() / f"part_subs_{tag}{ext}"
        kept: list[str] = []
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            for line in fh.read().splitlines():
                if line.startswith("Dialogue:"):
                    head, rest = line.split(":", 1)
                    f = rest.split(",", 9)
                    a = _ass_seconds(f[1]) if len(f) == 10 else None
                    b = _ass_seconds(f[2]) if len(f) == 10 else None
                    if a is not None and b is not None:
                        if b <= start or a >= end:
                            continue
                        f[1] = _ass_time(max(a, start) - start)
                        f[2] = _ass_time(min(b, end) - start)
                        line = head + ":" + ",".join(f)
                kept.append(line)
        out.write_text("\n".join(kept) + "\n", encoding="utf-8-sig")
        return str(out)
    from app.engines.subtitles import read_srt
    out = temp_dir() / f"part_subs_{tag}.srt"
    blocks: list[str] = []
    for c in read_srt(path):
        if c.end <= start or c.start >= end:
            continue
        blocks.append(f"{len(blocks) + 1}\n{_srt_time(max(c.start, start) - start)} --> "
                      f"{_srt_time(min(c.end, end) - start)}\n{c.text}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")
    return str(out)


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

def _vcodec_args(preset: ExportPreset) -> list[str]:
    vc = preset.vcodec or "libx264"
    args = ["-c:v", vc]
    if "nvenc" in vc:
        args += ["-rc", "vbr", "-cq", "23", "-preset", "p5"]
        if preset.bitrate:
            args += ["-b:v", preset.bitrate]
    elif "qsv" in vc:
        if preset.bitrate:
            args += ["-b:v", preset.bitrate]
        else:
            args += ["-global_quality", "23"]
    elif "amf" in vc:
        args += ["-quality", "quality"]
        if preset.bitrate:
            args += ["-rc", "vbr_peak", "-b:v", preset.bitrate]
        else:
            # constant QP roughly matching x264 CRF 20; hevc_amf has no B-frames
            args += ["-rc", "cqp", "-qp_i", "20", "-qp_p", "22"]
            if vc.startswith("h264"):
                args += ["-qp_b", "24"]
    else:  # libx264 / libx265 / anything software
        args += ["-preset", "medium"]
        if preset.bitrate:
            args += ["-b:v", preset.bitrate]
        else:
            args += ["-crf", "20"]
    return args


def render(video: str, out_path: str, preset: ExportPreset,
           overlays: OverlayConfig | None = None, srt_path: str | None = None,
           dub_audio: str | None = None,
           progress: Callable[[float, str], None] | None = None,
           cancel: threading.Event | None = None,
           subtitle_style: str = "", audio_lang: str = "",
           start: float = 0.0, duration: float = 0.0, speed: float = 1.0,
           poster: str = "", intro_image: str = "", intro_seconds: float = 0.0) -> str:
    """Single-pass export: overlays + subtitles + optional dub audio replace.

    Assembles one ffmpeg command with all inputs, the composed filtergraph,
    codec settings from the preset, faststart for mp4, and '-progress pipe:1'
    for live progress. audio_lang (ISO 639-2, e.g. "khm") labels the audio
    track; players show it in their audio menu. Returns out_path.

    `start` / `duration` (source seconds, 0 = to the end) export one part of
    the video, e.g. an episode cut from a long one: the video and the dub
    audio are read from `start`, and the subtitles and the smart-crop track
    are moved to the part's own clock, so it plays as a video of its own (a
    time-windowed ad overlay counts from the part's start). `speed` plays
    the result faster (> 1) or slower (< 1) with the voices' pitch kept; it
    is applied last, to the finished picture and sound.

    `poster`: a picture put into the finished file as its cover art (the
    thumbnail shown for the file). `intro_image` / `intro_seconds`: that
    long a still of the picture before the video, silent, as the DL tab's
    "Poster in the first second" starts a download. The file comment says
    what the result carries (cover_art.poster_comment).
    """
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")
    speed = float(speed or 1.0)
    if not SPEED_MIN <= speed <= SPEED_MAX:
        raise AppError(
            f"The speed must be between {SPEED_MIN:g}x and {SPEED_MAX:g}x (it is {speed:g}x).",
            f"ល្បឿនត្រូវនៅចន្លោះ {SPEED_MIN:g}x និង {SPEED_MAX:g}x (ឥឡូវគឺ {speed:g}x)។",
        )
    start = max(0.0, float(start or 0.0))
    duration = max(0.0, float(duration or 0.0))
    if progress:
        progress(0.0, "Probing source video")

    ov = overlays or OverlayConfig()
    info = probe(video)
    part = start > 0 or duration > 0
    if part and info.duration > 0 and start >= info.duration - 0.01:
        raise AppError(
            f"The part starts at {start:.1f} s, after the end of the video ({info.duration:.1f} s).",
            f"ផ្នែកនេះចាប់ផ្ដើមនៅ {start:.1f} វិនាទី ក្រោយចុងវីដេអូ ({info.duration:.1f} វិនាទី)។",
        )
    src_len = duration if duration > 0 else max(0.0, info.duration - start)
    if info.duration > 0:
        src_len = min(src_len, info.duration - start)
    if dub_audio and not os.path.isfile(dub_audio):
        raise AppError(
            f"Dub audio file not found: {dub_audio}",
            f"រកមិនឃើញឯកសារសំឡេងបញ្ចូល (dub)៖ {dub_audio}",
        )
    if srt_path and not os.path.isfile(srt_path):
        raise AppError(
            f"Subtitle file not found: {srt_path}",
            f"រកមិនឃើញឯកសារអត្ថបទរត់ (subtitle)៖ {srt_path}",
        )
    extra = overlay_inputs(ov)
    for p in extra:
        if not os.path.isfile(p):
            raise AppError(
                f"Overlay file not found: {p}",
                f"រកមិនឃើញឯកសារ overlay៖ {p}",
            )
    intro = bool(intro_image) and float(intro_seconds or 0.0) > 0
    if intro and not os.path.isfile(intro_image):
        raise AppError(
            f"Poster picture not found: {intro_image}",
            f"រកមិនឃើញរូបភាពប៉ុស្ទ័រ៖ {intro_image}",
        )
    out_w, out_h = output_size(info, preset)
    made: list[str] = []                      # temp files of this render
    prepared = prepare_watermark(ov, (out_w, out_h))
    if prepared is not ov and prepared.watermark_full and prepared.watermark:
        made.append(prepared.watermark)
    ov = prepared
    extra = overlay_inputs(ov)

    track = None
    if (preset.fit_mode or "").lower() == "smart" and preset.width > 0             and preset.height > 0:
        from app.engines.scenes import crop_track, detect_scenes
        if progress:
            progress(0.0, "Finding scene cuts for smart framing")
        cuts = detect_scenes(video, cancel=cancel)
        frame_note = (lambda f, t: progress(0.0, t)) if progress else None
        if part:
            # only this part's shots are framed (not the whole long video
            # for every episode), then moved to the part's clock
            end = start + src_len
            full = crop_track(video, [start] + [c for c in cuts if start < c < end] + [end],
                              info.duration, cancel=cancel, progress=frame_note)
            track = [(max(0.0, a - start), min(b, end) - start, cx)
                     for a, b, cx in full if b > start and a < end]
        else:
            track = crop_track(video, cuts, info.duration, cancel=cancel,
                               progress=frame_note)
    if srt_path and part:
        srt_path = shift_subtitles(srt_path, start, src_len)
    if srt_path and subtitle_style and srt_path.lower().endswith(".srt"):
        # styled captions are laid out for the final frame size
        from app.engines.subtitles import srt_to_ass
        srt_path = srt_to_ass(srt_path, str(temp_dir() / f"subs_{uuid.uuid4().hex[:8]}.ass"),
                              out_w, out_h, subtitle_style)
    graph = build_filtergraph(info, ov, srt_path, preset=preset, crop_track=track)
    fast = abs(speed - 1.0) > 1e-3
    if fast:
        # the finished picture, sped up; frames dropped / repeated to keep
        # the frame rate (setpts alone would make 30 fps play as 37.5 fps)
        fps = preset.fps if preset.fps > 0 else info.fps
        chain = f"setpts=PTS/{speed:.6g}" + (f",fps={fps:.6f}" if fps > 0 else "")
        graph = (graph[: -len("[vout]")] + f",{chain}[vout]") if graph \
            else f"[0:v]{chain}[vout]"

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # input options: read the video (and the dub audio) from the part's start
    seek = ((["-ss", f"{start:.3f}"] if start > 0 else [])
            + (["-t", f"{src_len:.3f}"] if duration > 0 else []))
    cmd = [ffmpeg_path(), "-y", "-hide_banner", *seek, "-i", video]
    for p in extra:
        cmd += ["-i", p]
    if dub_audio:
        cmd += [*seek, "-i", dub_audio]
        audio_map = f"{1 + len(extra)}:a:0"
        audio_in = f"[{1 + len(extra)}:a:0]"
    else:
        audio_map = "0:a:0?"
        audio_in = "[0:a:0]" if info.acodec else ""
    if intro:
        fps = preset.fps if preset.fps > 0 else (info.fps or 30.0)
        cmd += ["-loop", "1", "-framerate", f"{fps:.6g}", "-t", f"{float(intro_seconds):.3f}",
                "-i", intro_image]
        graph = _intro_graph(graph, 1 + len(extra) + (1 if dub_audio else 0),
                             (out_w, out_h), fps, float(intro_seconds), audio_in, speed)

    if graph:
        cmd += ["-filter_complex", graph, "-map", "[vout]"]
    else:
        cmd += ["-map", "0:v:0"]
    if intro and audio_in:
        cmd += ["-map", "[aout]"]
    else:
        cmd += ["-map", audio_map]
        if fast and (dub_audio or info.acodec):
            cmd += ["-filter:a", atempo_chain(speed)]

    cmd += _vcodec_args(preset)
    cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", preset.audio_bitrate or "192k"]
    if audio_lang:
        cmd += ["-metadata:s:a:0", f"language={audio_lang}"]
    # the file comment says what poster the result carries (a copied "poster"
    # comment of the source would claim one it does not have)
    from app.engines import cover_art
    src_note = cover_art.source_comment(video)
    starts_with_poster = intro or (start <= 0.01 and cover_art.POSTER_TAG_INTRO in src_note)
    if poster or intro or cover_art.is_poster_comment(src_note):
        cmd += ["-metadata",
                "comment=" + cover_art.poster_comment(bool(poster), starts_with_poster)]
    if (preset.container or "mp4").lower() == "mp4":
        if "hevc" in preset.vcodec or "265" in preset.vcodec:
            cmd += ["-tag:v", "hvc1"]   # Apple players refuse the default hev1 tag
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", out_path]

    total = src_len / speed + (float(intro_seconds) if intro else 0.0)
    try:
        _run_ffmpeg(cmd, total, progress, cancel, label="ffmpeg", status_text="Rendering")
    finally:
        for f in made:
            try:
                os.remove(f)
            except OSError:
                pass
    if poster:
        if progress:
            progress(0.995, "Adding the poster")
        if not cover_art.set_cover(out_path, poster, intro=starts_with_poster):
            LOG_BUS.publish(f"[export] could not put the poster into {os.path.basename(out_path)}")
    if progress:
        progress(1.0, "Render complete")
    return out_path


# --------------------------------------------------------------------------- #
# Cut segments
# --------------------------------------------------------------------------- #

def _concat_escape(path: str) -> str:
    """Escape a path for the concat demuxer list file (file 'p' directive)."""
    return str(path).replace("\\", "/").replace("'", "'\\''")


def cut_segments(video: str, segments: list[tuple[float, float]], out_path: str,
                 reencode: bool = True,
                 progress: Callable[[float, str], None] | None = None,
                 cancel: threading.Event | None = None) -> str:
    """Cut and join segments of a video.

    reencode=True  -> single accurate pass: trim/atrim + concat filter.
    reencode=False -> fast: per-segment '-ss/-to -c copy' parts joined with the
    concat demuxer (keyframe accuracy only).
    Returns out_path.
    """
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")
    segs = [(float(s), float(e)) for s, e in segments if float(e) > float(s)]
    if not segs:
        raise AppError(
            "No valid segments to cut (each segment needs end > start).",
            "គ្មានផ្នែកវីដេអូត្រឹមត្រូវសម្រាប់កាត់ទេ (ចុងបញ្ចប់ត្រូវធំជាងចំណុចចាប់ផ្តើម)។",
        )
    info = probe(video)
    has_audio = bool(info.acodec)
    total = sum(e - s for s, e in segs)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if reencode:
        parts: list[str] = []
        v_labels: list[str] = []
        a_labels: list[str] = []
        for i, (s, e) in enumerate(segs):
            parts.append(
                f"[0:v]trim=start={s:g}:end={e:g},setpts=PTS-STARTPTS[cv{i}]"
            )
            v_labels.append(f"[cv{i}]")
            if has_audio:
                parts.append(
                    f"[0:a]atrim=start={s:g}:end={e:g},asetpts=PTS-STARTPTS[ca{i}]"
                )
                a_labels.append(f"[ca{i}]")
        if has_audio:
            pairs = "".join(v + a for v, a in zip(v_labels, a_labels))
            parts.append(f"{pairs}concat=n={len(segs)}:v=1:a=1[vout][aout]")
        else:
            parts.append(f"{''.join(v_labels)}concat=n={len(segs)}:v=1:a=0[vout]")
        graph = ";".join(parts)

        cmd = [ffmpeg_path(), "-y", "-hide_banner", "-i", video,
               "-filter_complex", graph, "-map", "[vout]"]
        if has_audio:
            cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p"]
        if out_path.lower().endswith(".mp4"):
            cmd += ["-movflags", "+faststart"]
        cmd += ["-progress", "pipe:1", "-nostats", out_path]
        _run_ffmpeg(cmd, total, progress, cancel,
                    label="ffmpeg", status_text="Cutting")
        if progress:
            progress(1.0, "Cut complete")
        return out_path

    # ---- stream-copy path: per-segment copies + concat demuxer ----
    tmp = temp_dir() / f"cut_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(video)[1] or ".mp4"
    part_files: list[Path] = []
    try:
        for i, (s, e) in enumerate(segs):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Operation cancelled")
            part = tmp / f"part_{i:04d}{ext}"
            cmd = [ffmpeg_path(), "-y", "-hide_banner",
                   "-ss", f"{s:g}", "-to", f"{e:g}", "-i", video,
                   "-c", "copy", "-avoid_negative_ts", "make_zero",
                   "-progress", "pipe:1", "-nostats", str(part)]
            base = i / len(segs)
            span = 1.0 / len(segs)
            sub_progress = None
            if progress is not None:
                def sub_progress(f, _t, _b=base, _s=span, _i=i):
                    progress(min(0.99, _b + f * _s),
                             f"Copying segment {_i + 1}/{len(segs)}")
            _run_ffmpeg(cmd, e - s, sub_progress, cancel,
                        label="ffmpeg", status_text=f"Segment {i + 1}")
            part_files.append(part)

        list_file = tmp / "concat.txt"
        with open(list_file, "w", encoding="utf-8") as fh:
            for part in part_files:
                fh.write(f"file '{_concat_escape(str(part))}'\n")
        cmd = [ffmpeg_path(), "-y", "-hide_banner",
               "-f", "concat", "-safe", "0", "-i", str(list_file),
               "-c", "copy"]
        if out_path.lower().endswith(".mp4"):
            cmd += ["-movflags", "+faststart"]
        cmd += ["-progress", "pipe:1", "-nostats", out_path]
        _run_ffmpeg(cmd, total, progress, cancel,
                    label="ffmpeg", status_text="Joining")
        if progress:
            progress(1.0, "Cut complete")
        return out_path
    finally:
        try:
            for part in tmp.iterdir():
                part.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            log.warning("Could not clean temp cut dir: %s", tmp)


# --------------------------------------------------------------------------- #
# Render queue
# --------------------------------------------------------------------------- #

@dataclass
class RenderJob:
    id: str
    label: str
    kwargs: dict
    status: str = "pending"   # pending | running | done | error
    progress: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RenderJob":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        known.setdefault("id", uuid.uuid4().hex)
        known.setdefault("label", "")
        known.setdefault("kwargs", {})
        return cls(**known)


class RenderQueue:
    """Sequential, resumable render queue persisted to render_queue.json.

    Each job's kwargs carry an "op" key ("render" | "cut_segments"); the
    remaining kwargs are passed to that function.  "preset"/"overlays" values
    may be plain dicts (they are revived into their dataclasses).  State is
    saved after every status change, so a crashed/cancelled run resumes by
    skipping jobs already marked done.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else (app_data_dir() / "render_queue.json")
        self.jobs: list[RenderJob] = []
        self._lock = threading.Lock()
        self.load()

    # -- persistence ------------------------------------------------------- #

    def load(self) -> None:
        with self._lock:
            self.jobs = []
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    self.jobs = [RenderJob.from_dict(d) for d in data.get("jobs", [])]
                except (OSError, json.JSONDecodeError, TypeError) as e:
                    log.warning("Could not load render queue: %s", e)
                    self.jobs = []
            # A job left "running" by a crash resumes as pending.
            for job in self.jobs:
                if job.status == "running":
                    job.status = "pending"

    def save(self) -> None:
        with self._lock:
            data = {"jobs": [j.to_dict() for j in self.jobs]}
            tmp = self.path.with_suffix(".json.tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError as e:
                log.warning("Could not save render queue: %s", e)

    # -- queue ops --------------------------------------------------------- #

    def add(self, job: RenderJob) -> None:
        with self._lock:
            self.jobs.append(job)
        self.save()

    def remove(self, job_id: str) -> None:
        with self._lock:
            self.jobs = [j for j in self.jobs if j.id != job_id]
        self.save()

    def clear_done(self) -> None:
        with self._lock:
            self.jobs = [j for j in self.jobs if j.status != "done"]
        self.save()

    # -- execution --------------------------------------------------------- #

    @staticmethod
    def _revive_kwargs(kwargs: dict) -> tuple[str, dict]:
        kw = dict(kwargs)
        op = str(kw.pop("op", "render"))
        if isinstance(kw.get("preset"), dict):
            kw["preset"] = ExportPreset.from_dict(kw["preset"])
        if isinstance(kw.get("overlays"), dict):
            kw["overlays"] = OverlayConfig.from_dict(kw["overlays"])
        if isinstance(kw.get("segments"), list):
            kw["segments"] = [tuple(s) for s in kw["segments"]]
        return op, kw

    def run(self, on_job_progress: Callable[[str, float, str], None] | None = None,
            cancel: threading.Event | None = None) -> None:
        """Execute pending jobs sequentially. Resumable: done jobs are skipped;
        per-job failures are recorded in job.error and the queue continues;
        a set cancel event stops the current job and raises CancelledError."""
        for job in list(self.jobs):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Queue cancelled")
            if job.status == "done":
                continue
            job.status = "running"
            job.error = ""
            self.save()

            def _prog(frac: float, text: str, _job=job) -> None:
                _job.progress = float(frac)
                if on_job_progress is not None:
                    try:
                        on_job_progress(_job.id, float(frac), text)
                    except Exception:
                        pass

            try:
                op, kw = self._revive_kwargs(job.kwargs)
                if op == "cut_segments":
                    cut_segments(progress=_prog, cancel=cancel, **kw)
                elif op == "render":
                    render(progress=_prog, cancel=cancel, **kw)
                else:
                    raise AppError(
                        f"Unknown queue operation: {op}",
                        f"ប្រតិបត្តិការជួរមិនស្គាល់៖ {op}",
                    )
                job.status = "done"
                job.progress = 1.0
                self.save()
            except CancelledError:
                job.status = "pending"  # resumable next run
                self.save()
                raise
            except AppError as e:
                job.status = "error"
                job.error = e.en
                log.error("Render job %s failed: %s", job.id, e.en)
                self.save()
            except Exception as e:  # never let one bad job kill the queue
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                log.exception("Render job %s crashed", job.id)
                self.save()


# --------------------------------------------------------------------------- #
# Preset persistence (ConfigManager, key "export.presets")
# --------------------------------------------------------------------------- #

def save_preset(config, preset: ExportPreset) -> None:
    """Insert or update a preset (matched by name) under config "export.presets"."""
    raw = config.get("export.presets") or []
    if not isinstance(raw, list):
        raw = []
    out: list[dict] = []
    replaced = False
    for item in raw:
        if isinstance(item, dict) and item.get("name") == preset.name:
            out.append(preset.to_dict())
            replaced = True
        else:
            out.append(item)
    if not replaced:
        out.append(preset.to_dict())
    config.set("export.presets", out)


def load_presets(config) -> list[ExportPreset]:
    """Load presets from config; returns [ExportPreset()] when none stored."""
    raw = config.get("export.presets") or []
    presets: list[ExportPreset] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                try:
                    presets.append(ExportPreset.from_dict(item))
                except (TypeError, ValueError):
                    log.warning("Skipping malformed preset: %r", item)
    return presets or [ExportPreset()]


# --------------------------------------------------------------------------- #
# Offline self-test (no ffmpeg execution)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)
            print(f"FAIL: {name}")
        else:
            print(f"ok:   {name}")

    # 1) subtitle path escaping
    esc = _sub_path_escape(r"C:\Videos\my sub's file.srt")
    check("sub escape forward slashes", "\\" not in esc.replace("\\:", "").replace("\\'", ""))
    check("sub escape drive colon", esc.startswith("C\\:/"))
    check("sub escape quote", "\\'" in esc)
    check("sub escape full", esc == "C\\:/Videos/my sub\\'s file.srt")

    info = MediaInfo(duration=120.0, width=1920, height=1080, fps=25.0,
                     vcodec="h264", acodec="aac", bitrate=4_000_000)
    ov = OverlayConfig(
        watermark=r"C:\assets\logo.png", watermark_opacity=0.5,
        watermark_pos="bottom_right", watermark_scale=0.2,
        banner=r"C:\assets\banner.png", banner_height_ratio=0.1,
        chroma_video=r"C:\assets\host.mp4", chroma_color="0x00ff00",
        blur_regions=[(10, 20, 100, 50)], blur_strength=25,
    )
    preset = ExportPreset(name="YT 1080p", vcodec="h264_nvenc", bitrate="8M",
                          width=1920, height=1080, fps=30.0)
    test_srt = temp_dir() / "selftest ep 01.srt"
    test_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nសួស្ដី\n", encoding="utf-8")
    graph = build_filtergraph(info, ov, str(test_srt),
                              font_name="Kantumruy Pro", preset=preset)
    check("graph ends in [vout]", graph.endswith("[vout]"))
    check("graph normalization", "scale=1920:1080" in graph and "fps=30" in graph)
    check("graph chroma", "colorkey=0x00ff00" in graph)
    check("graph watermark alpha", "colorchannelmixer=aa=0.5" in graph)
    check("graph banner bottom", "overlay=0:main_h-overlay_h" in graph)
    check("graph blur", "gblur=sigma=25" in graph and "crop=100:50:10:20" in graph)
    check("graph subtitles via ass filter",
          "ass=filename='" in graph and ":shaping=complex" in graph)
    sub_ass = graph.split("ass=filename='")[1].split("'")[0].replace("\\:", ":")
    check("graph subtitles style",
          "Style: Default,Kantumruy Pro," in Path(sub_ass).read_text(encoding="utf-8-sig"))
    check("graph subtitles fontsdir", ":fontsdir='" in graph)
    check("graph subtitles last",
          graph.rindex("ass=filename=") > graph.rindex("gblur"))
    check("overlay input order",
          overlay_inputs(ov) == [ov.chroma_video, ov.watermark, ov.banner])
    check("empty graph", build_filtergraph(info, OverlayConfig(), None) == "")

    # 2) preset / overlay (de)serialization
    p2 = ExportPreset.from_dict(json.loads(json.dumps(preset.to_dict())))
    check("preset round-trip", p2 == preset)
    p3 = ExportPreset.from_dict({"name": "X", "unknown_field": 1})
    check("preset tolerant from_dict", p3.name == "X" and p3.vcodec == "libx264")
    ov2 = OverlayConfig.from_dict(json.loads(json.dumps(ov.to_dict())))
    check("overlay round-trip", ov2 == ov)
    check("overlay blur tuples", ov2.blur_regions == [(10, 20, 100, 50)])

    # 3) queue persistence round-trip
    with tempfile.TemporaryDirectory() as td:
        qpath = Path(td) / "render_queue.json"
        q = RenderQueue(path=qpath)
        job = RenderJob(
            id="job1", label="Episode 1",
            kwargs={"op": "render", "video": "in.mp4", "out_path": "out.mp4",
                    "preset": preset.to_dict(), "overlays": ov.to_dict()},
        )
        q.add(job)
        done_job = RenderJob(id="job2", label="Done already",
                             kwargs={"op": "cut_segments"}, status="done",
                             progress=1.0)
        q.add(done_job)
        running_job = RenderJob(id="job3", label="Was running",
                                kwargs={"op": "render"}, status="running",
                                progress=0.4)
        q.add(running_job)

        q2 = RenderQueue(path=qpath)
        check("queue count", len(q2.jobs) == 3)
        check("queue job fields",
              q2.jobs[0].id == "job1" and q2.jobs[0].kwargs["video"] == "in.mp4")
        check("queue done preserved", q2.jobs[1].status == "done")
        check("queue running reset", q2.jobs[2].status == "pending")
        op, kw = RenderQueue._revive_kwargs(q2.jobs[0].kwargs)
        check("queue kwargs revive",
              op == "render" and isinstance(kw["preset"], ExportPreset)
              and isinstance(kw["overlays"], OverlayConfig)
              and kw["preset"].name == "YT 1080p")

    # 4) misc helpers
    check("fraction parse", abs(_parse_fraction("30000/1001") - 29.97) < 0.01)
    check("fraction zero den", _parse_fraction("1/0") == 0.0)
    check("pos map fallback", _pos_xy("nonsense") == _pos_xy("top_right"))
    check("concat escape", _concat_escape(r"C:\a\it's.mp4") == "C:/a/it'\\''s.mp4")

    # 5) speed, and episodes cut from one long video
    check("atempo plain", atempo_chain(1.25) == "atempo=1.25")
    check("atempo chained past 2x / under 0.5x",
          atempo_chain(4.0) == "atempo=2,atempo=2"
          and atempo_chain(0.25) == "atempo=0.5,atempo=0.5")
    check("plan: 32:47 in 3:00 episodes = 10 + 2:47", plan_parts(1967.0, 180.0) == (10, 167.0))
    n_f, left_f = plan_parts(1967.0, 180.0, 1.25)
    check("plan: at 1.25x the finished length counts", n_f == 8 and abs(left_f - 133.6) < 1e-6)
    check("plan: a hair short of a whole episode is one", plan_parts(359.7, 180.0) == (2, 0.0))
    check("plan: under a second left over is none", plan_parts(360.6, 180.0) == (2, 0.0))
    check("plan: shorter than one episode", plan_parts(100.0, 180.0) == (0, 100.0))
    sp_new = part_spans(1967.0, 180.0, leftover="new")
    sp_join = part_spans(1967.0, 180.0, leftover="join")
    check("spans: the leftover as an episode of its own",
          len(sp_new) == 11 and sp_new[0] == (0.0, 180.0) and sp_new[-1] == (1800.0, 167.0))
    check("spans: the leftover added to the last episode",
          len(sp_join) == 10 and sp_join[-1] == (1620.0, 347.0))
    sp_fast = part_spans(1967.0, 180.0, speed=1.25)
    check("spans: at 1.25x each episode reads 1.25x as much video",
          sp_fast[1] == (225.0, 225.0) and abs(sum(ln for _s, ln in sp_fast) - 1967.0) < 1e-6)
    check("spans: one episode when shorter", part_spans(100.0, 180.0) == [(0.0, 100.0)])
    with tempfile.TemporaryDirectory() as td:
        from app.engines.subtitles import read_srt
        srt = Path(td) / "ep.srt"
        srt.write_text("1\n00:00:05,000 --> 00:00:08,000\nbefore\n\n"
                       "2\n00:03:01,500 --> 00:03:04,000\nសួស្ដី\n\n"
                       "3\n00:02:58,000 --> 00:03:02,000\nacross the cut\n\n"
                       "4\n00:06:00,500 --> 00:06:02,000\nafter\n", encoding="utf-8")
        out = shift_subtitles(str(srt), 180.0, 180.0)
        got = [(round(c.start, 3), round(c.end, 3), c.text) for c in read_srt(out)]
        Path(out).unlink(missing_ok=True)
        check("srt moved to the part's clock (outside dropped, edges cut)",
              got == [(1.5, 4.0, "សួស្ដី"), (0.0, 2.0, "across the cut")])
        ass = Path(td) / "ep.ass"
        ass.write_text("[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
                       "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
                       "Effect, Text\n"
                       "Dialogue: 0,0:00:05.00,0:00:08.00,Default,,0,0,0,,before\n"
                       "Dialogue: 0,0:03:01.50,0:03:04.00,Default,,0,0,0,,in, with a comma\n",
                       encoding="utf-8")
        out = shift_subtitles(str(ass), 180.0, 0.0)
        text = Path(out).read_text(encoding="utf-8-sig")
        Path(out).unlink(missing_ok=True)
        check("ass moved to the part's clock (styles and text kept)",
              "before" not in text and "[Script Info]" in text
              and "Dialogue: 0,0:00:01.50,0:00:04.00,Default,,0,0,0,,in, with a comma" in text)

    # 6) the watermark layer (logo and/or text, 3x3 place or tiled), the
    #    preview, and the poster second before an episode
    from PIL import Image
    wm_text = OverlayConfig(watermark_text="@ឆានែល Channel", watermark_pos="bottom_right",
                            watermark_opacity=0.5, watermark_scale=0.2)
    lay = watermark_layer((640, 360), wm_text)
    alpha = lay.getchannel("A")
    check("text watermark in the bottom right corner",
          alpha.crop((320, 180, 640, 360)).getbbox() is not None
          and alpha.crop((0, 0, 320, 180)).getbbox() is None)
    check("its opacity applied", 100 <= alpha.getextrema()[1] <= 130)
    from PIL import ImageChops
    solid = ImageChops.darker(lay.getchannel("R"), alpha.point(lambda v: 255 if v > 120 else 0))
    check("text colour 'auto' is white", solid.getextrema()[1] > 230)
    tiled = watermark_layer((640, 360), _dc_replace(wm_text, watermark_pos="tile"))
    cells = [tiled.getchannel("A").crop((x, y, x + 160, y + 120)).getbbox() is not None
             for x in range(0, 640, 160) for y in range(0, 360, 120)]
    check("tiled watermark across the frame", sum(cells) >= 9)
    with tempfile.TemporaryDirectory() as td:
        logo = Path(td) / "logo.png"
        Image.new("RGBA", (100, 50), (255, 0, 0, 255)).save(logo)
        both = OverlayConfig(watermark=str(logo), watermark_text="@me", watermark_pos="center",
                             watermark_opacity=1.0, watermark_scale=0.25)
        lay = watermark_layer((800, 400), both)
        bb = lay.getchannel("A").getbbox()
        check("logo with the text under it, centred",
              bb is not None and abs((bb[0] + bb[2]) / 2 - 400) <= 2
              and lay.getpixel((400, bb[1] + 5))[:3] == (255, 0, 0))
        prep = prepare_watermark(both, (800, 400))
        check("prepared as one full-frame layer",
              prep.watermark_full and prep.watermark != str(logo)
              and Image.open(prep.watermark).size == (800, 400) and prep.watermark_text == "")
        g = build_filtergraph(MediaInfo(width=800, height=400), prep, None)
        check("…laid over the video as it is", "[1:v]overlay=0:0" in g and "colorchannelmixer" not in g)
        Path(prep.watermark).unlink(missing_ok=True)
        frame = Image.new("RGB", (800, 400), (40, 40, 40))
        pv = preview_frame(frame, _dc_replace(both, blur_regions=[(0, 0, 50, 50)],
                                              ad_video="ad.mp4", ad_pos="top_left", ad_scale=0.2))
        check("preview: watermark and the ad's place drawn",
              pv.size == (800, 400) and pv.getpixel((400, bb[1] + 5)) == (255, 0, 0)
              and pv.getpixel((OVERLAY_MARGIN + 3, OVERLAY_MARGIN + 3)) != (40, 40, 40))
    off = OverlayConfig.from_dict({"watermark": "logo.png", "watermark_text": "@me",
                                   "watermark_on": False, "banner": "b.png"})
    check("a switched-off watermark is not used (the rest is)",
          off.watermark is None and off.watermark_text == "" and off.banner == "b.png")
    check("no watermark, no layer", watermark_layer((64, 36), OverlayConfig()) is None
          and prepare_watermark(OverlayConfig(), (64, 36)) == OverlayConfig())
    check("grid: middle right", _grid_xy("middle_right", (100, 100), (10, 10), 5) == (85, 45))
    check("output size follows the preset",
          output_size(MediaInfo(width=1920, height=1080), ExportPreset(width=1280)) == (1280, 720)
          and output_size(MediaInfo(width=1920, height=1078), None) == (1920, 1078))
    ig = _intro_graph("[0:v]scale=1280:720[vout]", 2, (1280, 720), 30.0, 1.0, "[0:a:0]", 1.25)
    check("poster second: picture before the video",
          "[0:v]scale=1280:720[vmain]" in ig and "[iv][mv]concat=n=2:v=1:a=0[vout]" in ig
          and "[2:v]split=2" in ig)
    check("…silence before the sound, the speed on the video's own part",
          "atrim=duration=1.000" in ig and "[sa][ma]concat=n=2:v=0:a=1[aout]" in ig
          and "[0:a:0]aresample=48000" in ig and "atempo=1.25[ma]" in ig)
    check("…no sound, no audio chain", "[aout]" not in _intro_graph("", 1, (64, 36), 25.0,
                                                                     1.0, "", 1.0))

    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        raise SystemExit(1)
    print("PASS")
