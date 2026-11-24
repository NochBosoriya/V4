"""Thumbnail maker: pick the strongest frames of a video and compose a
1280x720 (or 1080x1920) thumbnail with a Khmer-correct title.

find_frames()     candidate frames at the middle of shots, scored by
                  sharpness, faces (count / size), colourfulness and
                  exposure, spread over the video
compose()         cover-crop around the faces, punch up contrast/colour,
                  darken the bottom, draw a 1–2 line title (+ optional
                  tag line) with poster_designer.render_title
suggest_titles()  short click-worthy titles from the AI

No Qt imports; cv2 is lazy-imported.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.core.errors import AppError, CancelledError
from app.core.logger import get_logger
from app.engines.poster_designer import (StyleSpec, _draw_shaped_text,
                                         _resolve_font, render_title)

log = get_logger("thumbnail")

ProgressFn = Callable[[float, str], None]


@dataclass
class FrameCandidate:
    time: float
    path: str
    score: float
    faces: int = 0
    face_x: float = 0.5           # weighted face centre (0..1)
    face_y: float = 0.4
    sharpness: float = 0.0
    colorfulness: float = 0.0
    brightness: float = 0.0


STYLES: dict[str, StyleSpec] = {
    "bold": StyleSpec(font="Moul", size_ratio=0.085, fill="#FFE14D",
                      gradient=["#FFF27A", "#FFB800"], stroke_color="#000000",
                      stroke_width_ratio=0.10, shadow_color="#000000dd",
                      shadow_offset=(6, 6), shadow_blur=10, pos=(0.5, 0.80)),
    "clean": StyleSpec(font="Kantumruy Pro", size_ratio=0.075, fill="#FFFFFF",
                       stroke_color="#000000", stroke_width_ratio=0.08,
                       shadow_color="#000000aa", shadow_offset=(4, 4),
                       shadow_blur=8, pos=(0.5, 0.82)),
    "drama": StyleSpec(font="Moul", size_ratio=0.085, fill="#FFFFFF",
                       gradient=["#FFFFFF", "#FFC2C2"], stroke_color="#B00000",
                       stroke_width_ratio=0.09, glow_color="#ff3030",
                       glow_radius=10, shadow_color="#000000dd",
                       shadow_offset=(5, 5), shadow_blur=10, pos=(0.5, 0.80)),
    "neon": StyleSpec(font="Battambang", size_ratio=0.08, fill="#E8FFFF",
                      stroke_color="#003344", stroke_width_ratio=0.06,
                      glow_color="#00E5FF", glow_radius=14,
                      shadow_color="#00000099", shadow_offset=(3, 3),
                      shadow_blur=6, pos=(0.5, 0.80)),
}
STYLE_IDS = list(STYLES)


# ---------------------------------------------------------------------------
# frame selection
# ---------------------------------------------------------------------------

def _colorfulness(bgr: np.ndarray) -> float:
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _score_frame(bgr: np.ndarray, detector) -> dict:
    import cv2  # lazy heavy import
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (480, max(1, int(h * 480 / max(1, w)))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=6,
                                      minSize=(28, 28))
    area = gray.shape[0] * gray.shape[1]
    fx, fy, fw = 0.5, 0.4, []
    if len(faces):
        weights = [float(a * b) for (_x, _y, a, b) in faces]
        fx = float(np.average([(x + a / 2) / gray.shape[1] for (x, _y, a, _b) in faces],
                              weights=weights))
        fy = float(np.average([(y + b / 2) / gray.shape[0] for (_x, y, _a, b) in faces],
                              weights=weights))
        fw = weights
    face_frac = (max(fw) / area) if fw else 0.0
    # one or two big faces is what makes a thumbnail; crowds score less
    n = len(fw)
    face_score = (min(1.0, face_frac / 0.06) * (1.0 if n <= 2 else 0.7)) if n else 0.0
    return {"sharp": sharp, "bright": bright, "color": _colorfulness(small),
            "faces": n, "face_score": face_score, "fx": fx, "fy": fy}


def find_frames(video: str, out_dir: str, count: int = 8,
                cuts: list[float] | None = None,
                progress: ProgressFn | None = None,
                cancel: threading.Event | None = None) -> list[FrameCandidate]:
    """Best `count` frames (saved as JPG in out_dir), best first."""
    import cv2  # lazy heavy import
    from app.engines.audio_pipeline import get_duration
    from app.engines.scenes import _face_detector, detect_scenes, shots
    if not os.path.isfile(video):
        raise AppError(f"Video not found: {video}", f"រកមិនឃើញវីដេអូ៖ {video}")
    duration = get_duration(video)
    if duration <= 0:
        raise AppError("Could not read the video duration.",
                       "មិនអាចអានរយៈពេលវីដេអូបានទេ។")
    if cuts is None:
        try:
            cuts = detect_scenes(video, progress=None, cancel=cancel)
        except CancelledError:
            raise
        except Exception:
            cuts = []
    lo, hi = duration * 0.03, duration * 0.97       # skip logos / credits
    times = [(a + b) / 2 for a, b in shots(cuts, duration)
             if b - a >= 1.2 and lo <= (a + b) / 2 <= hi]
    if len(times) < count * 3:
        times += list(np.linspace(lo, hi, count * 5))
    times = sorted(set(round(t, 2) for t in times))
    if len(times) > 60:
        times = [times[int(i)] for i in np.linspace(0, len(times) - 1, 60)]
    det = _face_detector()
    cap = cv2.VideoCapture(video)
    raw: list[tuple[float, dict, np.ndarray]] = []
    try:
        for i, t in enumerate(times):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Operation cancelled")
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            m = _score_frame(fr, det)
            if m["bright"] < 18:             # black / fade frames
                continue
            raw.append((t, m, fr))
            if progress:
                progress(0.9 * (i + 1) / len(times), f"Scoring frame {i + 1}/{len(times)}")
    finally:
        cap.release()
    if not raw:
        raise AppError("No usable frames found in this video.",
                       "រកមិនឃើញរូបភាពដែលអាចប្រើបានក្នុងវីដេអូនេះទេ។")

    def norm(vals):
        a = np.log1p(np.asarray(vals, dtype=np.float64))
        return (a - a.min()) / (np.ptp(a) or 1.0)

    sharp_n = norm([m["sharp"] for _t, m, _f in raw])
    color_n = norm([m["color"] for _t, m, _f in raw])
    scored = []
    for (t, m, fr), s, c in zip(raw, sharp_n, color_n):
        exposure = 1.0 - min(1.0, abs(m["bright"] - 120.0) / 110.0)
        score = 0.30 * s + 0.40 * m["face_score"] + 0.20 * c + 0.10 * exposure
        scored.append((score, t, m, fr))
    scored.sort(key=lambda x: -x[0])
    gap = duration / max(1, count * 3)
    chosen = []
    for item in scored:
        if all(abs(item[1] - c[1]) >= gap for c in chosen):
            chosen.append(item)
        if len(chosen) >= count:
            break
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for score, t, m, fr in chosen:
        path = os.path.join(out_dir, f"frame_{t:09.2f}.jpg")
        # cv2.imwrite cannot write to a folder with non-ASCII letters on
        # Windows (a Chinese video name): it saved nothing, without an error,
        # and the preview then failed with FileNotFoundError
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            continue
        with open(path, "wb") as fh:
            fh.write(buf.tobytes())
        out.append(FrameCandidate(time=float(t), path=path, score=round(float(score), 3),
                                  faces=int(m["faces"]), face_x=float(m["fx"]),
                                  face_y=float(m["fy"]),
                                  sharpness=round(m["sharp"], 1),
                                  colorfulness=round(m["color"], 1),
                                  brightness=round(m["bright"], 1)))
    if progress:
        progress(1.0, f"{len(out)} candidate frames")
    return out


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def _cover(img: Image.Image, size: tuple[int, int], cx: float, cy: float) -> Image.Image:
    W, H = size
    s = max(W / img.width, H / img.height)
    im = img.resize((max(W, round(img.width * s)), max(H, round(img.height * s))),
                    Image.LANCZOS)
    x = int(max(0, min(im.width - W, cx * im.width - W / 2)))
    y = int(max(0, min(im.height - H, cy * im.height - H * 0.4)))
    return im.crop((x, y, x + W, y + H))


def _bottom_shade(img: Image.Image, start: float = 0.5, strength: float = 0.75) -> Image.Image:
    W, H = img.size
    ramp = np.clip((np.arange(H) / H - start) / (1 - start), 0, 1) ** 1.4
    alpha = (ramp * 255 * strength).astype(np.uint8)
    mask = Image.fromarray(np.repeat(alpha[:, None], W, axis=1), "L")
    return Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, mask)


def _wrap_title(text: str, font_path: str, size: int, max_w: int,
                img_size: tuple[int, int]) -> tuple[list[str], int]:
    """Up to two lines that fit max_w, shrinking the font if needed."""
    from app.engines.subtitles import split_words

    def width(s: str, sz: int) -> int:
        _m, w, _h = _draw_shaped_text(img_size, s, font_path, sz, 0)
        return w

    words = split_words(text) or [text]
    while size > 12:
        if width(text, size) <= max_w:
            return [text], size
        # best two-line split by width balance
        best = None
        for k in range(1, len(words)):
            a, b = "".join(words[:k]).strip(), "".join(words[k:]).strip()
            wa, wb = width(a, size), width(b, size)
            if max(wa, wb) <= max_w and (best is None or max(wa, wb) < best[0]):
                best = (max(wa, wb), [a, b])
        if best:
            return best[1], size
        size = int(size * 0.9)
    return [text], size


def compose(frame_path: str, title: str, out_path: str,
            size: tuple[int, int] = (1280, 720), style: str | StyleSpec = "bold",
            tagline: str = "", face_x: float = 0.5, face_y: float = 0.4,
            enhance: bool = True, fonts_dir: Path | None = None) -> str:
    """Compose the finished thumbnail and save it (JPG/PNG by extension)."""
    spec = STYLES.get(style, STYLES["bold"]) if isinstance(style, str) else style
    img = Image.open(frame_path).convert("RGB")
    img = _cover(img, size, face_x, face_y)
    if enhance:
        img = ImageEnhance.Contrast(img).enhance(1.12)
        img = ImageEnhance.Color(img).enhance(1.22)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))
    if title.strip():
        img = _bottom_shade(img)
    work = str(Path(out_path).with_suffix(".base.png"))
    img.save(work)
    try:
        cur = work
        W, H = size
        if title.strip():
            font_path = _resolve_font(spec.font, fonts_dir)
            fsize = max(12, int(round(spec.size_ratio * W)))
            lines, fsize = _wrap_title(title.strip(), font_path, fsize, int(W * 0.92), size)
            line_h = fsize * 1.45 / H
            y_last = min(0.9, spec.pos[1] + (0.04 if len(lines) == 1 else 0.0))
            for i, line in enumerate(lines):
                y = y_last - (len(lines) - 1 - i) * line_h
                st = replace(spec, size_ratio=fsize / W, pos=(spec.pos[0], y))
                nxt = str(Path(out_path).with_suffix(f".l{i}.png"))
                render_title(cur, line, st, nxt, fonts_dir)
                if cur != work:
                    os.remove(cur)
                cur = nxt
        if tagline.strip():
            st = StyleSpec(font="Kantumruy Pro", size_ratio=0.032, fill="#FFFFFF",
                           stroke_color="#000000", stroke_width_ratio=0.12,
                           shadow_color="#000000aa", shadow_offset=(2, 2),
                           shadow_blur=4, pos=(0.5, 0.08))
            nxt = str(Path(out_path).with_suffix(".tag.png"))
            render_title(cur, tagline.strip(), st, nxt, fonts_dir)
            if cur != work:
                os.remove(cur)
            cur = nxt
        final = Image.open(cur).convert("RGB")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.lower().endswith((".jpg", ".jpeg")):
            final.save(out_path, quality=92, optimize=True)
        else:
            final.save(out_path)
        if cur != work and os.path.isfile(cur):
            os.remove(cur)
        return out_path
    finally:
        if os.path.isfile(work):
            os.remove(work)


def suggest_titles(ai, context: str, lang: str = "km", n: int = 5,
                   cancel: threading.Event | None = None) -> list[str]:
    """Short, click-worthy thumbnail titles in `lang` (no clickbait lies)."""
    from app.engines.recap_engine import _lang_name
    system = ("You write YouTube/Facebook thumbnail titles for drama recap "
              "channels. Respond with ONLY a JSON object {\"titles\": [..]}. "
              f"Write {n} titles in {_lang_name(lang)}: 2-6 words each, "
              "emotional and curiosity-raising, true to the story, no emojis, "
              "no hashtags, no quotes.")
    res = ai.chat(feature="poster", system=system, json_mode=True,
                  temperature=0.9, max_tokens=800, cancel=cancel,
                  messages=[{"role": "user", "content":
                             f"Story / episode context:\n{context[:6000]}"}])
    try:
        from app.core.ai_engine import extract_json
        data = extract_json(res.text)
        titles = data.get("titles") if isinstance(data, dict) else data
        return [str(t).strip().strip('"“”') for t in titles or [] if str(t).strip()][:n]
    except Exception as exc:
        raise AppError(f"Could not read the AI title ideas: {exc}",
                       f"មិនអាចអានគំនិតចំណងជើងពី AI បានទេ៖ {exc}") from exc


def save_meta(cands: list[FrameCandidate], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in cands], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    import subprocess
    import tempfile
    from types import SimpleNamespace as NS

    from app.core.paths import ffmpeg_path

    d = tempfile.mkdtemp(prefix="缩略图_ថតរូប_")      # non-ASCII, like a Chinese video's name
    v = os.path.join(d, "v.mp4")
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=black:s=640x360:d=1:r=25",
                    "-f", "lavfi", "-i", "testsrc2=s=640x360:d=3:r=25",
                    "-f", "lavfi", "-i", "mandelbrot=s=640x360:r=25",
                    "-filter_complex",
                    "[2]trim=duration=3,setpts=PTS-STARTPTS[m];[0][1][m]concat=n=3:v=1:a=0",
                    "-pix_fmt", "yuv420p", v], check=True)
    cands = find_frames(v, os.path.join(d, "frames"), count=3)
    assert cands and all(os.path.isfile(c.path) for c in cands)
    assert all(c.brightness >= 18 for c in cands), "black frames are skipped"
    assert cands == sorted(cands, key=lambda c: -c.score)
    out = compose(cands[0].path, "ខ្ញុំរង់ចាំអ្នកយូរណាស់ហើយ ស្នេហ៍ក្បត់ដ៏ឈឺចាប់",
                  os.path.join(d, "thumb.jpg"), tagline="ភាគ ១")
    im = Image.open(out)
    assert im.size == (1280, 720)
    out2 = compose(cands[0].path, "Short", os.path.join(d, "thumb_v.png"),
                   size=(1080, 1920), style="neon")
    assert Image.open(out2).size == (1080, 1920)
    assert not [f for f in os.listdir(d) if ".l0" in f or ".base" in f], "temp files cleaned"
    fake = NS(chat=lambda **k: NS(text='{"titles": ["ស្នេហ៍ក្បត់", "\\"Twist\\""]}'))
    assert suggest_titles(fake, "x") == ["ស្នេហ៍ក្បត់", "Twist"]
    print("frames:", [(c.time, c.score) for c in cands])
    print("thumb:", out)
    print("PASS")
