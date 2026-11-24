"""Overlay tab: two sides, Poster and Overlay.

Poster: a video's thumbnail and poster pictures. The picture comes from the
video itself (the poster it carries, e.g. from the DL tab: what the file
shows as its thumbnail), from its best frames, or from an image file. Then
the full poster localization flow: OCR title detection, inpainting,
"Replace title — same style", 3-option AI title translation or title ideas,
AI style design plus a manual override panel and style presets, a tag line,
thumbnail sizes, drag-to-position rendering with uharfbuzz-shaped Khmer
text shown live, undo/redo, exports, putting the result into the video as
its thumbnail, and batch templates.

Overlay: builds a ``media_exporter.OverlayConfig`` (watermark: text and/or a
logo, 3x3 place or tiled; banner, chroma-keyed video, ad video, blur
regions, subtitles), drawn live on a frame of the video the way the export
draws it, and offers a 10-second preview render.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap,
                           QShortcut)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.errors import AppError, EngineNotAvailable
from app.core.i18n import pick, tr
from app.core.logger import get_logger
from app.core.paths import fonts_dir
from app.ui.workers import FnWorker

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.overlay_tab")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]
CORNER_LABELS = {
    "top_left": "Top left", "top_right": "Top right",
    "bottom_left": "Bottom left", "bottom_right": "Bottom right",
}
GRID_LABELS = {      # the watermark's 3x3 places (media_exporter.GRID_POSITIONS)
    "top_left": "Top left", "top_center": "Top centre", "top_right": "Top right",
    "middle_left": "Middle left", "center": "Centre", "middle_right": "Middle right",
    "bottom_left": "Bottom left", "bottom_center": "Bottom centre",
    "bottom_right": "Bottom right",
}
THUMB_SIZES = [("16x9", "YouTube / Facebook 1280x720", (1280, 720)),
               ("9x16", "Shorts / TikTok 1080x1920", (1080, 1920)),
               ("1x1", "Square 1080x1080", (1080, 1080))]
THUMB_STYLE_LABELS = {"bold": "Bold yellow (Moul)", "clean": "Clean white",
                      "drama": "Drama red glow (Moul)", "neon": "Neon blue (Battambang)"}
VIDEO_FILTER = "(*.mp4 *.mkv *.mov *.webm *.avi *.ts *.m4v)"
SUB_FONTS = ["Kantumruy Pro", "Battambang", "Hanuman", "Moul"]
POSTER_FONT_CANDIDATES = [
    "Moul", "Koulen", "Kantumruy Pro", "Battambang", "Hanuman", "Khmer OS",
    "Khmer OS Muol Light", "Impact", "Arial Black", "Arial",
    "Segoe UI", "Times New Roman",
]
TEMPLATES_KEY = "poster.templates"
OVERLAY_LAST_KEY = "overlay.last"

LANGS = [
    ("km", "ខ្មែរ (Khmer)"), ("en", "English"), ("zh", "中文 (Chinese)"),
    ("th", "ไทย (Thai)"), ("vi", "Tiếng Việt (Vietnamese)"),
    ("ko", "한국어 (Korean)"), ("ja", "日本語 (Japanese)"),
    ("fr", "Français (French)"), ("es", "Español (Spanish)"),
    ("hi", "हिन्दी (Hindi)"),
]


def _msg(en: str, km: str = "") -> str:
    """The message in the UI language (English when there is no Khmer)."""
    return pick(en, km)


# --- defensive ProgressRow adapters ------------------------------------------

def _make_progress_row(label: str) -> QWidget:
    from app.ui import widgets as W
    try:
        row = W.ProgressRow(label)
    except TypeError:
        row = W.ProgressRow()
        lab = row.findChild(QLabel)
        if lab is not None:
            lab.setText(label)
    row.setVisible(False)
    return row


def _row_progress(row: QWidget, frac: float, text: str) -> None:
    fn = getattr(row, "set_progress", None)
    if callable(fn):
        try:
            fn(frac, text)
            return
        except Exception:
            pass
    bar = row.findChild(QProgressBar)
    if bar is not None:
        bar.setRange(0, 1000)
        bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
    if text:
        lab = row.findChild(QLabel)
        if lab is not None:
            lab.setText(text)


def _wire_cancel(row: QWidget, worker) -> None:
    row._rs_worker = worker
    if getattr(row, "_rs_cancel_wired", False):
        return
    sig = None
    for name in ("cancel_requested", "cancelled", "canceled", "cancel_clicked"):
        cand = getattr(row, name, None)
        if cand is not None and hasattr(cand, "connect"):
            sig = cand
            break
    if sig is None:
        btn = row.findChild(QToolButton) or row.findChild(QPushButton)
        if btn is not None:
            sig = btn.clicked

    def _do_cancel(*_a):
        w = getattr(row, "_rs_worker", None)
        if w is not None:
            try:
                w.cancel()
            except Exception:
                pass

    if sig is not None:
        try:
            sig.connect(_do_cancel)
            row._rs_cancel_wired = True
        except Exception:
            pass


# --- worker functions ---------------------------------------------------------

def _frame_image_fn(video: str, at: float, progress=None, cancel=None):
    """One frame of `video` at `at` s, as a PIL picture (RGB) at the
    video's own size (the size blur boxes are measured in)."""
    import subprocess

    from PIL import Image

    from app.core.paths import ffmpeg_path

    folder = tempfile.mkdtemp(prefix="rs_ovframe_")
    out = os.path.join(folder, "frame.png")
    try:
        for t in (max(0.0, at), max(0.0, at - 1.5), 0.0):    # past the last frame: earlier
            cp = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-v", "error", "-y", "-ss", f"{t:.3f}",
                 "-i", video, "-map", "0:V:0", "-frames:v", "1", out],
                capture_output=True, encoding="utf-8", errors="replace", timeout=120,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if cp.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
                with Image.open(out) as im:
                    return im.convert("RGB")
        raise AppError("Could not read a frame of this video.",
                       "មិនអាចអានរូបភាពពីវីដេអូនេះបានទេ។")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _compose_overlay_fn(frame, cfg_dict: dict, progress=None, cancel=None):
    """The frame with the overlays drawn as the export draws them."""
    from app.engines import media_exporter as me
    return me.preview_frame(frame, me.OverlayConfig.from_dict(cfg_dict))


def _qimage(img) -> QImage:
    rgb = img.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    return QImage(data, rgb.width, rgb.height, 3 * rgb.width,
                  QImage.Format.Format_RGB888).copy()


def _preview_render_fn(video: str, cfg_dict: dict, srt_path: str | None,
                       start: float = 0.0, progress=None, cancel=None):
    """10 s of the video from `start`, rendered with the overlays."""
    from app.engines import media_exporter as me

    info = me.probe(video)
    total = info.duration or 10.0
    start = max(0.0, min(float(start), total - 1.0))
    length = min(10.0, max(1.0, total - start))
    tmp = tempfile.mkdtemp(prefix="rs_ovl_prev_")
    cut = os.path.join(tmp, "cut.mp4")
    out = os.path.join(tmp, "preview.mp4")

    def sub(base, span):
        def _p(f, t):
            if progress:
                progress(base + span * max(0.0, min(1.0, f)), t)
        return _p

    me.cut_segments(video, [(start, start + length)], cut, reencode=True,
                    progress=sub(0.0, 0.45), cancel=cancel)
    if srt_path and start > 0:
        srt_path = me.shift_subtitles(srt_path, start, length)
    cfg = me.OverlayConfig.from_dict(cfg_dict)
    me.render(cut, out, me.ExportPreset(), overlays=cfg, srt_path=srt_path,
              progress=sub(0.45, 0.55), cancel=cancel)
    return out


def _video_poster_fn(video: str, progress=None, cancel=None) -> str:
    """The poster the video carries (or carried before a dub), "" if none."""
    from app.engines import cover_art
    return cover_art.find_poster(video)


def _find_frames_fn(video: str, count: int, progress=None, cancel=None):
    import hashlib

    from app.core.paths import temp_dir
    from app.engines import thumbnail as th
    key = hashlib.sha1(os.path.abspath(video).encode("utf-8", "replace")).hexdigest()[:12]
    return th.find_frames(video, str(temp_dir() / "thumb_frames" / key), count=count,
                          progress=progress, cancel=cancel)


def _sized_picture_fn(path: str, size, fx: float, fy: float, progress=None, cancel=None) -> str:
    """The picture cut to a thumbnail size around its faces, its colours
    lifted as the thumbnail maker does; a PNG in a temp folder."""
    from PIL import Image, ImageEnhance, ImageFilter

    from app.engines import thumbnail as th
    with Image.open(path) as im:
        img = im.convert("RGB")
    img = th._cover(img, (int(size[0]), int(size[1])), fx, fy)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Color(img).enhance(1.22)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))
    out = os.path.join(tempfile.mkdtemp(prefix="rs_thumb_"), "picture.png")
    img.save(out)
    return out


def _set_cover_fn(video: str, picture: str, progress=None, cancel=None) -> bool:
    from app.engines import cover_art
    if progress:
        progress(0.3, tr("poster.setting_thumb", "Putting the picture into the video…"))
    return cover_art.set_cover(video, picture)


def _ideas_fn(ai, context: str, lang: str, progress=None, cancel=None):
    from app.engines import thumbnail as th
    if progress:
        progress(0.2, tr("poster.ideas_running", "Writing title ideas…"))
    return th.suggest_titles(ai, context, lang, n=3, cancel=cancel)


def _tag_style():
    """The small tag line at the top of a thumbnail ("ភាគ ១ / EP 1")."""
    from app.engines.poster_designer import StyleSpec
    return StyleSpec(font="Kantumruy Pro", size_ratio=0.032, fill="#FFFFFF",
                     stroke_color="#000000", stroke_width_ratio=0.12,
                     shadow_color="#000000aa", shadow_offset=(2, 2), shadow_blur=4,
                     pos=(0.5, 0.08))


def _replace_title_fn(image_path: str, text: str, boxes_data: list | None,
                      as_title: bool, grouped: bool = False, progress=None, cancel=None):
    """Take the poster's title off and give `text` its look, laid out in its
    place (see poster_designer.replace_title, layout=True)."""
    from app.engines import poster_designer as pd
    boxes = None
    if boxes_data:
        boxes = [pd.OcrBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"],
                           text=b.get("text", ""), conf=float(b.get("conf", 1.0)),
                           alts=tuple(b.get("alts") or ()))
                 for b in boxes_data]
    out = os.path.join(tempfile.mkdtemp(prefix="rs_title_"), "clean.png")
    return pd.replace_title(image_path, text, out, boxes=boxes, fonts_dir=fonts_dir(),
                            as_title=as_title, grouped=grouped, layout=True,
                            progress=progress, cancel=cancel)


def _ai_replace_fn(image_path: str, text: str, ai, lang: str, boxes_data: list | None,
                   source: str, progress=None, cancel=None):
    """The title redesigned by the AI picture model in the original title's
    art style (poster_designer.ai_replace_title)."""
    from app.engines import poster_designer as pd
    boxes = None
    if boxes_data:
        boxes = [pd.OcrBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"],
                           text=b.get("text", ""), conf=float(b.get("conf", 1.0)))
                 for b in boxes_data]
    out = os.path.join(tempfile.mkdtemp(prefix="rs_title_"), "designed.png")
    return pd.ai_replace_title(image_path, text, out, ai, source=source, boxes=boxes,
                               target_lang=lang, progress=progress, cancel=cancel)


def _translate_title_fn(title: str, ai, target_lang: str, image_path: str = "",
                        progress=None, cancel=None):
    """Three translations of the poster's title; with no title given, the
    title is read off the picture first."""
    from app.engines import poster_designer as pd
    if not title and image_path:
        if progress:
            progress(0.05, tr("poster.reading_title", "Reading the poster's title…"))
        title = pd.title_text(image_path, cancel=cancel)
        if not title:
            raise AppError("No title found on the poster: type the title to translate in "
                           "the title box.",
                           "រកមិនឃើញចំណងជើងនៅលើប៉ុស្ទ័រទេ៖ សូមវាយចំណងជើងដែលត្រូវបកប្រែ"
                           "ក្នុងប្រអប់ចំណងជើង។")
    if progress:
        progress(0.2, tr("poster.translating", "Translating title…"))
    return pd.translate_title(title, ai, target_lang=target_lang, cancel=cancel)


def _design_fn(image_path: str, title: str, ai, progress=None, cancel=None):
    from app.engines import poster_designer as pd
    if progress:
        progress(0.1, tr("poster.analyzing", "Analyzing image style…"))
    stats = pd.analyze_image_style(image_path)
    if progress:
        progress(0.5, tr("poster.designing", "Designing title style…"))
    return pd.design_spec(stats, title, "", ai, cancel=cancel)


def _render_title_fn(image_path: str, text: str, spec_dict: dict, tagline: str = "",
                     progress=None, cancel=None):
    """The picture with the title (and the small tag line at the top)."""
    from app.engines import poster_designer as pd
    folder = tempfile.mkdtemp(prefix="rs_title_")
    cur = image_path
    if text.strip():
        spec = pd.StyleSpec.from_llm_json(spec_dict)
        try:
            # never wider than the picture: a long title (Khmer runs long) at
            # the chosen size was cut off at both edges
            from PIL import Image
            with Image.open(image_path) as im:
                w, h = im.size
            room = pd.fit_size_ratio(text, spec, 0.92 * w, 0.45 * h, w, fonts_dir())
            spec.size_ratio = min(spec.size_ratio, room)
        except Exception:
            log.debug("could not fit the title to the picture", exc_info=True)
        cur = pd.render_title(image_path, text, spec, os.path.join(folder, "render.png"),
                              fonts_dir=fonts_dir())
    if tagline.strip():
        cur = pd.render_title(cur, tagline.strip(), _tag_style(),
                              os.path.join(folder, "render_tag.png"), fonts_dir=fonts_dir())
    return cur


def _export_variants_fn(image_path: str, out_dir: str, base_name: str,
                        progress=None, cancel=None):
    from app.engines import poster_designer as pd
    if progress:
        progress(0.2, tr("poster.exporting", "Exporting variants…"))
    return pd.export_variants(image_path, out_dir, base_name)


def _batch_posters_fn(ai, folder: str, spec_dict: dict, text: str,
                      target_lang: str, auto_per_poster: bool, out_dir: str,
                      progress=None, cancel=None):
    from app.core.errors import CancelledError
    from app.engines import poster_designer as pd

    spec = pd.StyleSpec.from_llm_json(spec_dict)
    images = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS))
    results = []
    n = max(1, len(images))
    for i, img in enumerate(images):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        name = os.path.basename(img)

        def sub(f, t, i=i, name=name):
            if progress:
                progress((i + max(0.0, min(1.0, f))) / n, f"[{name}] {t}")

        try:
            base = img
            title = text
            if auto_per_poster:
                boxes = []
                try:
                    boxes = pd.detect_title_boxes(img, cancel=cancel)
                except EngineNotAvailable:
                    boxes = []
                if boxes:
                    sub(0.2, tr("poster.inpainting", "Removing title…"))
                    clean = os.path.join(
                        tempfile.mkdtemp(prefix="rs_batch_"), "clean.png")
                    base = pd.inpaint_region(img, boxes[:1], clean, cancel=cancel)
                    src_title = boxes[0].text.strip()
                    if src_title:
                        sub(0.5, tr("poster.translating", "Translating title…"))
                        options = pd.translate_title(
                            src_title, ai, target_lang=target_lang, cancel=cancel)
                        if options:
                            title = options[0]
            if not title:
                results.append((img, "", "no title text"))
                continue
            sub(0.8, tr("poster.rendering", "Rendering title…"))
            out = os.path.join(
                out_dir, os.path.splitext(name)[0] + "_titled.png")
            pd.render_title(base, title, spec, out, fonts_dir=fonts_dir())
            results.append((img, out, ""))
        except CancelledError:
            raise
        except AppError as e:
            results.append((img, "", e.en))
        except Exception as e:
            results.append((img, "", str(e)))
    return results


def _auto_title_fn(image_path: str, ai, lang: str, progress=None, cancel=None):
    """The poster's main title read, translated and taken off; the editor
    draws the translation (poster_designer.auto_translate_title)."""
    from app.engines import poster_designer as pd
    return pd.auto_translate_title(image_path, "", ai, target_lang=lang,
                                   fonts_dir=fonts_dir(), progress=progress, cancel=cancel)


def _auto_titles_fn(paths: list, out_dir: str, ai, lang: str, overwrite: bool,
                    progress=None, cancel=None):
    """Each picture's title translated and redrawn, saved as <name>_KH.png."""
    from app.engines import poster_designer as pd
    return pd.auto_translate_titles(list(paths), out_dir, ai, target_lang=lang,
                                    fonts_dir=fonts_dir(), overwrite=overwrite,
                                    progress=progress, cancel=cancel)


def _font_families() -> list[str]:
    """Poster font candidates that actually resolve to a file, else all."""
    found: list[str] = []
    try:
        from app.engines.poster_designer import find_font_file
        for name in POSTER_FONT_CANDIDATES:
            try:
                if find_font_file(name, fonts_dir()):
                    found.append(name)
            except Exception:
                continue
    except Exception:
        log.exception("could not scan fonts")
    return found or list(POSTER_FONT_CANDIDATES)


# --- canvases -----------------------------------------------------------------

class _ImageCanvas(QLabel):
    """QLabel showing a scaled pixmap with box overlays and interactions.

    Modes: "select" (click OCR boxes / drag title position),
    "manual" (rubber-band a new box).
    Coordinates emitted are in original-image pixels or fractions.
    """

    box_clicked = Signal(int)                 # index of clicked box
    manual_box = Signal(int, int, int, int)   # x, y, w, h (image coords)
    pos_dragged = Signal(float, float)        # fractions of W/H
    drag_finished = Signal()

    def __init__(self):
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._boxes: list[tuple[int, int, int, int]] = []
        self._selected = -1
        self.mode = "select"
        self.drag_enabled = True
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin = QPoint()
        self._dragging = False
        self.setMinimumSize(320, 320)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #0d1117; border: 1px solid #30363d;")

    # geometry -------------------------------------------------------------

    def set_image(self, pixmap: QPixmap | None):
        self._pixmap = pixmap
        self.update()

    def set_boxes(self, boxes: list[tuple[int, int, int, int]], selected: int = -1):
        self._boxes = list(boxes)
        self._selected = selected
        self.update()

    def _fit(self) -> tuple[float, int, int]:
        """(scale, offset_x, offset_y) of the drawn pixmap inside the label."""
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0, 0, 0
        iw, ih = self._pixmap.width(), self._pixmap.height()
        lw, lh = max(1, self.width()), max(1, self.height())
        scale = min(lw / iw, lh / ih)
        sw, sh = int(iw * scale), int(ih * scale)
        return scale, (lw - sw) // 2, (lh - sh) // 2

    def _to_image(self, pt: QPoint) -> tuple[int, int] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        scale, ox, oy = self._fit()
        x = (pt.x() - ox) / scale
        y = (pt.y() - oy) / scale
        if x < 0 or y < 0 or x > self._pixmap.width() or y > self._pixmap.height():
            return None
        return int(x), int(y)

    # painting -------------------------------------------------------------

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        if self._pixmap is None or self._pixmap.isNull():
            return
        p = QPainter(self)
        scale, ox, oy = self._fit()
        target = QRect(ox, oy, int(self._pixmap.width() * scale),
                       int(self._pixmap.height() * scale))
        p.drawPixmap(target, self._pixmap)
        for i, (x, y, w, h) in enumerate(self._boxes):
            rect = QRect(int(ox + x * scale), int(oy + y * scale),
                         int(w * scale), int(h * scale))
            pen = QPen(QColor("#58a6ff") if i == self._selected
                       else QColor("#f85149"))
            pen.setWidth(3 if i == self._selected else 2)
            p.setPen(pen)
            p.drawRect(rect)
        p.end()

    # interaction ----------------------------------------------------------

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)
        if self.mode == "manual":
            self._origin = ev.position().toPoint()
            self._rubber.setGeometry(QRect(self._origin, QSize()))
            self._rubber.show()
            return
        img_pt = self._to_image(ev.position().toPoint())
        if img_pt is None:
            return
        x, y = img_pt
        for i, (bx, by, bw, bh) in enumerate(self._boxes):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.box_clicked.emit(i)
                return
        if self.drag_enabled and self._pixmap is not None:
            self._dragging = True
            self.pos_dragged.emit(x / self._pixmap.width(),
                                  y / self._pixmap.height())

    def mouseMoveEvent(self, ev):  # noqa: N802
        if self.mode == "manual" and self._rubber.isVisible():
            self._rubber.setGeometry(
                QRect(self._origin, ev.position().toPoint()).normalized())
            return
        if self._dragging and self._pixmap is not None:
            img_pt = self._to_image(ev.position().toPoint())
            if img_pt is not None:
                self.pos_dragged.emit(img_pt[0] / self._pixmap.width(),
                                      img_pt[1] / self._pixmap.height())

    def mouseReleaseEvent(self, ev):  # noqa: N802
        if self.mode == "manual" and self._rubber.isVisible():
            self._rubber.hide()
            a = self._to_image(self._origin)
            b = self._to_image(ev.position().toPoint())
            if a and b:
                x1, y1 = a
                x2, y2 = b
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                if w > 4 and h > 4:
                    self.manual_box.emit(x, y, w, h)
            return
        if self._dragging:
            self._dragging = False
            self.drag_finished.emit()


# --- color button --------------------------------------------------------------

class _ColorButton(QPushButton):
    color_changed = Signal(str)

    def __init__(self, color: str = "#ffffff", allow_alpha: bool = False):
        super().__init__()
        self._color = color
        self._alpha = allow_alpha
        self.setFixedSize(46, 24)
        self.clicked.connect(self._pick)
        self._paint()

    def color(self) -> str:
        return self._color

    def set_color(self, color: str):
        self._color = color or "#ffffff"
        self._paint()

    def _paint(self):
        self.setStyleSheet(
            f"QPushButton {{ background: {self._color[:7]};"
            " border: 1px solid #30363d; border-radius: 4px; }")
        self.setToolTip(self._color)

    def _pick(self):
        opts = QColorDialog.ColorDialogOption.ShowAlphaChannel if self._alpha \
            else QColorDialog.ColorDialogOption(0)
        c = QColorDialog.getColor(QColor(self._color), self, "", opts)
        if c.isValid():
            self._color = c.name(QColor.NameFormat.HexArgb) if self._alpha \
                else c.name()
            # PIL wants #rrggbbaa; Qt gives #aarrggbb — reorder
            if self._alpha and len(self._color) == 9:
                a, rgb = self._color[1:3], self._color[3:]
                self._color = f"#{rgb}{a}"
            self._paint()
            self.color_changed.emit(self._color)


# ==============================================================================
# Tab
# ==============================================================================

class OverlayTab(QWidget):
    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self._workers: list = []

        # poster side
        self._image_path: str | None = None    # as loaded
        self._work_path: str | None = None     # after inpaint
        self._rendered_path: str | None = None
        self._boxes: list = []                 # OcrBox list
        self._selected_box = -1
        self._spec_dict: dict = self._default_spec_dict()
        self._undo: list[tuple[str, str]] = []
        self._redo: list[tuple[str, str]] = []
        self._render_pending = False
        self._render_running = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(300)
        self._render_timer.timeout.connect(self._start_render)
        self._pv_video = ""            # the video the thumbnail is for
        self._pv_pick: tuple[str, float, float] | None = None   # picture before sizing
        self._pv_pick_key = ""         # which strip picture is on the canvas
        self._pv_from_strip = False    # the canvas shows the video's own picture / frame
        self._auto_out = ""            # translated title: where Export saves it (<name>_KH.png)
        self._auto_boxes: list = []    # the original title's boxes the translation found
        self._auto_source = ""         # the original title's text (as the AI read it)
        self._ai_designed = False      # the picture's title was lettered by the AI model
        self._pre_ai_work: str | None = None   # the editor's picture before that

        # overlay side
        self._ov_video = ""
        self._ov_duration = 0.0
        self._ov_frame = None          # PIL frame under the overlays
        self._ov_frame_busy = self._ov_frame_again = False
        self._ov_busy = self._ov_again = False
        self._ov_frame_timer = QTimer(self)
        self._ov_frame_timer.setSingleShot(True)
        self._ov_frame_timer.setInterval(250)
        self._ov_frame_timer.timeout.connect(self._load_ov_frame)
        self._ov_compose_timer = QTimer(self)
        self._ov_compose_timer.setSingleShot(True)
        self._ov_compose_timer.setInterval(120)
        self._ov_compose_timer.timeout.connect(self._compose_ov_preview)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self.tabs.addTab(self._build_poster_tab(), tr("overlay.side_poster", "Poster"))
        self.tabs.addTab(self._build_video_tab(), tr("overlay.side_overlay", "Overlay"))
        self.tabs.currentChanged.connect(lambda _i: self._on_side_shown())

        self._load_last_overlay_config()

        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._on_redo)

    # ------------------------------------------------------------- worker glue

    def _launch(self, worker) -> None:
        self._workers = [w for w in self._workers if w.isRunning()]
        self._workers.append(worker)
        pool = getattr(self.ctx, "pool", None)
        for name in ("add", "start", "submit", "track"):
            fn = getattr(pool, name, None)
            if callable(fn):
                try:
                    fn(worker)
                    break
                except Exception:
                    continue
        try:
            if not worker.isRunning():
                worker.start()
        except Exception:
            log.exception("could not start worker")

    def _run_task(self, row, fn, *args, on_ok=None, on_fail=None,
                  running_text="", **kwargs):
        worker = FnWorker(fn, *args, **kwargs)
        row.setVisible(True)
        _row_progress(row, 0.0, running_text or tr("common.working", "Working…"))
        _wire_cancel(row, worker)
        worker.progress.connect(lambda f, t, r=row: _row_progress(r, f, t))

        def _ok(result):
            _row_progress(row, 1.0, tr("common.done", "Done"))
            if on_ok:
                try:
                    on_ok(result)
                except Exception as e:
                    log.exception("post-task handler failed")
                    QMessageBox.warning(self, tr("err.title", "Error"), str(e))

        def _fail(en, km):
            _row_progress(row, 0.0, tr("common.failed", "Failed"))
            if on_fail:
                try:
                    on_fail(en, km)
                    return
                except Exception:
                    log.exception("failure handler failed")
            QMessageBox.warning(self, tr("err.title", "Error"), _msg(en, km))

        worker.finished_ok.connect(_ok)
        worker.failed.connect(_fail)
        self._launch(worker)
        return worker

    # =========================================================================
    # Overlay side: the video's overlays, drawn live on one of its frames
    # =========================================================================

    def _build_video_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 6, 0, 0)
        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)

        def file_row(line: QLineEdit, filt: str) -> QWidget:
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(line, 1)
            btn = QPushButton(tr("common.browse", "Browse…"))

            def pick():
                p, _ = QFileDialog.getOpenFileName(self, "", "", filt)
                if p:
                    line.setText(p)
            btn.clicked.connect(pick)
            h.addWidget(btn)
            return holder

        def corner_combo() -> QComboBox:
            cb = QComboBox()
            for c in CORNERS:
                cb.addItem(tr(f"overlay.pos.{c}", CORNER_LABELS[c]), c)
            return cb

        def slider_row(slider: QSlider, label: QLabel) -> QWidget:
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(slider, 1)
            label.setMinimumWidth(44)
            h.addWidget(label)
            return holder

        img_filt = tr("overlay.filter_image", "Images") + " (*.png *.jpg *.jpeg *.webp);;All (*.*)"
        vid_filt = tr("overlay.filter_video", "Videos") + " (*.mp4 *.mkv *.mov *.webm *.avi);;All (*.*)"

        # ---- left: the result on a frame of the video ----------------------
        left = QVBoxLayout()
        src = QHBoxLayout()
        src.addWidget(QLabel(tr("overlay.ov_video", "Video:")))
        self.ov_video_lbl = QLabel(tr("overlay.ov_no_video", "No video chosen"))
        self.ov_video_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.ov_video_lbl.setStyleSheet("color: #8b949e;")
        src.addWidget(self.ov_video_lbl, 1)
        b = QPushButton(tr("thumb.use_current", "Use current video"))
        b.clicked.connect(lambda: self._set_ov_video(self.ctx.current_video or "", ask=True))
        src.addWidget(b)
        b = QPushButton(tr("common.browse", "Browse…"))
        b.clicked.connect(self._on_ov_browse)
        src.addWidget(b)
        left.addLayout(src)

        self.ov_canvas = _ImageCanvas()
        self.ov_canvas.drag_enabled = False
        self.ov_canvas.setWordWrap(True)
        self.ov_canvas.setText(tr("overlay.ov_empty", "Choose a video: one of its frames shows "
                                                      "the overlays as the export draws them."))
        self.ov_canvas.manual_box.connect(self._on_blur_drawn)
        self.ov_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self.ov_canvas, 1)

        fr = QHBoxLayout()
        fr.addWidget(QLabel(tr("overlay.ov_frame", "Frame:")))
        self.ov_slider = QSlider(Qt.Orientation.Horizontal)
        self.ov_slider.setRange(0, 1000)
        self.ov_slider.setValue(100)
        self.ov_slider.setEnabled(False)
        self.ov_slider.setToolTip(tr("overlay.ov_frame_tip", "The frame of the video the preview shows"))
        self.ov_slider.valueChanged.connect(self._on_ov_slider)
        fr.addWidget(self.ov_slider, 1)
        self.ov_time = QLabel("0:00")
        self.ov_time.setMinimumWidth(56)
        fr.addWidget(self.ov_time)
        left.addLayout(fr)
        self.ov_note = QLabel(tr(
            "overlay.ov_note", "The watermark, the banner and the blurred boxes are drawn as the "
                               "export draws them; the green and orange boxes show where the "
                               "chroma-key and ad videos will play. The export adds the subtitles."))
        self.ov_note.setWordWrap(True)
        self.ov_note.setStyleSheet("color: #8b949e;")
        left.addWidget(self.ov_note)
        self.frame_row = _make_progress_row(tr("overlay.extracting_frame", "Extracting frame…"))
        left.addWidget(self.frame_row)
        body.addLayout(left, 3)

        # ---- right: the controls -------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(470)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 8, 0)

        # watermark: text and/or logo, opacity, size, 3x3 place or tiled
        self.wm_box = QGroupBox(tr("overlay.watermark", "Watermark"))
        self.wm_box.setCheckable(True)
        self.wm_box.setChecked(True)
        self.wm_box.setToolTip(tr("overlay.wm_on", "Burn the watermark into the video"))
        self.wm_box.setStyleSheet(
            "QToolButton[cell=\"true\"] { background: #21262d; border: 1px solid #30363d;"
            " border-radius: 3px; min-width: 30px; min-height: 22px; }"
            "QToolButton[cell=\"true\"]:checked { background: #58a6ff; border-color: #58a6ff; }"
            "QToolButton[cell=\"true\"]:hover { border-color: #58a6ff; }")
        wf = QFormLayout(self.wm_box)
        self.wm_text = QLineEdit()
        self.wm_text.setPlaceholderText(tr("overlay.wm_text_ph", "e.g. @YourChannel"))
        wf.addRow(tr("overlay.wm_text", "Text:"), self.wm_text)
        self.wm_file = QLineEdit()
        self.wm_file.setPlaceholderText(tr("overlay.wm_logo_ph", "PNG with transparency, or JPG"))
        wf.addRow(tr("overlay.wm_logo", "Logo:"), file_row(self.wm_file, img_filt))
        crow = QWidget()
        ch = QHBoxLayout(crow)
        ch.setContentsMargins(0, 0, 0, 0)
        self.wm_color_auto = QCheckBox(tr("overlay.wm_auto", "Auto (white, soft shadow)"))
        self.wm_color_auto.setChecked(True)
        ch.addWidget(self.wm_color_auto)
        self.wm_color = _ColorButton("#ffffff")
        self.wm_color.setEnabled(False)
        self.wm_color_auto.toggled.connect(lambda on: self.wm_color.setEnabled(not on))
        ch.addWidget(self.wm_color)
        ch.addStretch(1)
        wf.addRow(tr("overlay.wm_color", "Text colour:"), crow)
        self.wm_opacity = QSlider(Qt.Orientation.Horizontal)
        self.wm_opacity.setRange(5, 100)
        self.wm_opacity.setValue(60)
        self.wm_opacity_lab = QLabel("60%")
        self.wm_opacity.valueChanged.connect(lambda v: self.wm_opacity_lab.setText(f"{v}%"))
        wf.addRow(tr("overlay.opacity", "Opacity:"), slider_row(self.wm_opacity, self.wm_opacity_lab))
        self.wm_size = QSlider(Qt.Orientation.Horizontal)
        self.wm_size.setRange(2, 40)
        self.wm_size.setValue(15)
        self.wm_size.setToolTip(tr("overlay.wm_size_tip", "The logo's width, in % of the video's "
                                                          "width (the text follows it)"))
        self.wm_size_lab = QLabel("15%")
        self.wm_size.valueChanged.connect(lambda v: self.wm_size_lab.setText(f"{v}%"))
        wf.addRow(tr("overlay.wm_size", "Size:"), slider_row(self.wm_size, self.wm_size_lab))
        pos = QWidget()
        pl = QHBoxLayout(pos)
        pl.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        grid.setSpacing(3)
        self.wm_pos_group = QButtonGroup(self)
        self.wm_pos_group.setExclusive(True)
        from app.engines.media_exporter import GRID_POSITIONS
        for k, gid in enumerate(GRID_POSITIONS):
            cell = QToolButton()
            cell.setProperty("cell", True)
            cell.setCheckable(True)
            cell.setToolTip(tr(f"overlay.pos.{gid}", GRID_LABELS[gid]))
            self.wm_pos_group.addButton(cell, k)
            grid.addWidget(cell, k // 3, k % 3)
        pl.addLayout(grid)
        self.wm_tile = QToolButton()
        self.wm_tile.setText(tr("overlay.wm_tile", "Tile"))
        self.wm_tile.setCheckable(True)
        self.wm_tile.setToolTip(tr("overlay.wm_tile_tip", "Repeat the watermark across the "
                                                          "whole picture, tilted"))
        self.wm_pos_group.addButton(self.wm_tile, len(GRID_POSITIONS))
        pl.addSpacing(10)
        pl.addWidget(self.wm_tile, 0, Qt.AlignmentFlag.AlignVCenter)
        pl.addStretch(1)
        self.wm_pos_group.button(GRID_POSITIONS.index("top_right")).setChecked(True)
        wf.addRow(tr("overlay.position", "Position:"), pos)
        lay.addWidget(self.wm_box)

        # banner
        bn_box = QGroupBox(tr("overlay.banner", "Banner"))
        bf = QFormLayout(bn_box)
        self.banner_file = QLineEdit()
        bf.addRow(tr("overlay.file", "File:"), file_row(self.banner_file, img_filt))
        self.banner_ratio = QDoubleSpinBox()
        self.banner_ratio.setRange(0.02, 0.5)
        self.banner_ratio.setSingleStep(0.01)
        self.banner_ratio.setValue(0.12)
        bf.addRow(tr("overlay.height_ratio", "Height ratio:"), self.banner_ratio)
        lay.addWidget(bn_box)

        # chroma
        ck_box = QGroupBox(tr("overlay.chroma", "Chroma-key video"))
        cf = QFormLayout(ck_box)
        self.chroma_file = QLineEdit()
        cf.addRow(tr("overlay.file", "File:"), file_row(self.chroma_file, vid_filt))
        self.chroma_color = _ColorButton("#00ff00")
        cf.addRow(tr("overlay.key_color", "Key color:"), self.chroma_color)
        self.chroma_sim = QDoubleSpinBox()
        self.chroma_sim.setRange(0.01, 1.0)
        self.chroma_sim.setSingleStep(0.01)
        self.chroma_sim.setValue(0.15)
        cf.addRow(tr("overlay.similarity", "Similarity:"), self.chroma_sim)
        self.chroma_pos = corner_combo()
        self.chroma_pos.setCurrentIndex(CORNERS.index("bottom_right"))
        cf.addRow(tr("overlay.position", "Position:"), self.chroma_pos)
        lay.addWidget(ck_box)

        # ad video
        ad_box = QGroupBox(tr("overlay.ad", "Ad video"))
        af = QFormLayout(ad_box)
        self.ad_file = QLineEdit()
        af.addRow(tr("overlay.file", "File:"), file_row(self.ad_file, vid_filt))
        self.ad_start = QDoubleSpinBox()
        self.ad_start.setRange(0.0, 100000.0)
        self.ad_start.setValue(0.0)
        self.ad_start.setSuffix(" s")
        af.addRow(tr("overlay.start_time", "Start time:"), self.ad_start)
        self.ad_pos = corner_combo()
        af.addRow(tr("overlay.position", "Position:"), self.ad_pos)
        self.ad_scale = QDoubleSpinBox()
        self.ad_scale.setRange(0.05, 1.0)
        self.ad_scale.setSingleStep(0.05)
        self.ad_scale.setValue(0.3)
        af.addRow(tr("overlay.scale", "Scale:"), self.ad_scale)
        lay.addWidget(ad_box)

        # blur regions
        bl_box = QGroupBox(tr("overlay.blur", "Blur regions"))
        bl = QVBoxLayout(bl_box)
        self.blur_table = QTableWidget(0, 4)
        self.blur_table.setHorizontalHeaderLabels(["x", "y", "w", "h"])
        self.blur_table.setMaximumHeight(140)
        bl.addWidget(self.blur_table)
        blr = QHBoxLayout()
        pick_btn = QPushButton(tr("overlay.draw_blur", "Draw on the preview"))
        pick_btn.setToolTip(tr("overlay.draw_blur_tip", "Then drag a box over the area to blur"))
        pick_btn.clicked.connect(self._on_pick_blur_region)
        blr.addWidget(pick_btn)
        add_btn = QPushButton(tr("overlay.add_region", "Add row"))
        add_btn.clicked.connect(lambda: self._add_blur_row(0, 0, 100, 100))
        blr.addWidget(add_btn)
        del_btn = QPushButton(tr("overlay.del_region", "Remove selected"))
        del_btn.clicked.connect(self._del_blur_row)
        blr.addWidget(del_btn)
        blr.addStretch(1)
        bl.addLayout(blr)
        srow = QHBoxLayout()
        srow.addWidget(QLabel(tr("overlay.strength", "Strength:")))
        self.blur_strength = QSpinBox()
        self.blur_strength.setRange(1, 60)
        self.blur_strength.setValue(20)
        srow.addWidget(self.blur_strength)
        srow.addStretch(1)
        bl.addLayout(srow)
        lay.addWidget(bl_box)

        # subtitles
        sub_box = QGroupBox(tr("overlay.subtitles", "Subtitles"))
        sf = QFormLayout(sub_box)
        self.srt_file = QLineEdit()
        sf.addRow(tr("overlay.srt_file", "SRT file:"),
                  file_row(self.srt_file, "SubRip (*.srt);;All (*.*)"))
        self.sub_font = QComboBox()
        self.sub_font.addItems(SUB_FONTS)
        sf.addRow(tr("overlay.sub_font", "Font:"), self.sub_font)
        lay.addWidget(sub_box)
        lay.addStretch(1)
        scroll.setWidget(inner)
        body.addWidget(scroll, 2)

        # ---- footer ----------------------------------------------------------
        act = QHBoxLayout()
        save_cfg_btn = QPushButton(tr("overlay.apply_cfg", "Apply config"))
        save_cfg_btn.setToolTip(tr("overlay.apply_cfg_tip", "The Export tab and the series "
                                                            "pipeline use these overlays"))
        save_cfg_btn.clicked.connect(self._on_apply_config)
        act.addWidget(save_cfg_btn)
        prev_btn = QPushButton(tr("overlay.preview10", "Preview 10s render"))
        prev_btn.setToolTip(tr("overlay.preview10_tip", "Renders 10 seconds from the frame shown, "
                                                        "with every overlay and the subtitles"))
        prev_btn.clicked.connect(self._on_preview_10s)
        act.addWidget(prev_btn)
        self.ov_status = QLabel("")
        self.ov_status.setStyleSheet("color: #8b949e;")
        act.addWidget(self.ov_status, 1)
        outer.addLayout(act)
        self.preview_row = _make_progress_row(tr("overlay.previewing", "Rendering preview…"))
        outer.addWidget(self.preview_row)

        # every change redraws the preview
        for sig in (self.wm_box.toggled, self.wm_text.textChanged, self.wm_file.textChanged,
                    self.wm_color_auto.toggled, self.wm_color.color_changed,
                    self.wm_opacity.valueChanged, self.wm_size.valueChanged,
                    self.wm_pos_group.idClicked, self.banner_file.textChanged,
                    self.banner_ratio.valueChanged, self.chroma_file.textChanged,
                    self.chroma_pos.currentIndexChanged, self.ad_file.textChanged,
                    self.ad_pos.currentIndexChanged, self.ad_scale.valueChanged,
                    self.blur_table.cellChanged, self.blur_strength.valueChanged):
            sig.connect(self._schedule_ov_preview)
        return page

    def _add_blur_row(self, x: int, y: int, w: int, h: int):
        r = self.blur_table.rowCount()
        self.blur_table.blockSignals(True)
        self.blur_table.insertRow(r)
        for c, v in enumerate((x, y, w, h)):
            self.blur_table.setItem(r, c, QTableWidgetItem(str(int(v))))
        self.blur_table.blockSignals(False)
        self._schedule_ov_preview()

    def _del_blur_row(self):
        r = self.blur_table.currentRow()
        if r >= 0:
            self.blur_table.removeRow(r)
            self._schedule_ov_preview()

    def _blur_regions(self) -> list:
        out = []
        for r in range(self.blur_table.rowCount()):
            vals = []
            for c in range(4):
                it = self.blur_table.item(r, c)
                try:
                    vals.append(int(float(it.text())) if it else 0)
                except ValueError:
                    vals.append(0)
            if vals[2] > 0 and vals[3] > 0:
                out.append(tuple(vals))
        return out

    def _wm_position(self) -> str:
        from app.engines.media_exporter import GRID_POSITIONS
        k = self.wm_pos_group.checkedId()
        if k == len(GRID_POSITIONS):
            return "tile"
        return GRID_POSITIONS[k] if 0 <= k < len(GRID_POSITIONS) else "top_right"

    def _collect_overlay_dict(self) -> dict:
        """The overlays as the export uses them (a switched-off watermark: none)."""
        def opt(line: QLineEdit):
            t = line.text().strip()
            return t if t else None

        on = self.wm_box.isChecked()
        chroma = self.chroma_color.color().lstrip("#")[:6]
        return {
            "watermark": opt(self.wm_file) if on else None,
            "watermark_opacity": self.wm_opacity.value() / 100.0,
            "watermark_pos": self._wm_position(),
            "watermark_scale": self.wm_size.value() / 100.0,
            "watermark_text": self.wm_text.text().strip() if on else "",
            "watermark_color": "auto" if self.wm_color_auto.isChecked() else self.wm_color.color(),
            "watermark_font": "Kantumruy Pro",
            "banner": opt(self.banner_file),
            "banner_height_ratio": float(self.banner_ratio.value()),
            "chroma_video": opt(self.chroma_file),
            "chroma_color": f"0x{chroma}",
            "chroma_similarity": float(self.chroma_sim.value()),
            "chroma_pos": str(self.chroma_pos.currentData()),
            "blur_regions": self._blur_regions(),
            "blur_strength": int(self.blur_strength.value()),
            "ad_video": opt(self.ad_file),
            "ad_start": float(self.ad_start.value()),
            "ad_pos": str(self.ad_pos.currentData()),
            "ad_scale": float(self.ad_scale.value()),
        }

    def overlay_config(self):
        """The overlays as currently set in this tab (also saved, so the
        Export tab and the series pipeline see the same settings)."""
        self._save_overlay_config()
        return getattr(self.ctx, "overlay_config", None)

    def _save_overlay_config(self):
        d = self._collect_overlay_dict()
        try:
            from app.engines.media_exporter import OverlayConfig
            setattr(self.ctx, "overlay_config", OverlayConfig(**d))
        except Exception:
            log.exception("could not build OverlayConfig")
            setattr(self.ctx, "overlay_config", None)
        stored = dict(d)
        stored["blur_regions"] = [list(t) for t in stored["blur_regions"]]
        # a switched-off watermark keeps its text and logo for next time
        stored["watermark_on"] = self.wm_box.isChecked()
        stored["watermark"] = self.wm_file.text().strip() or None
        stored["watermark_text"] = self.wm_text.text().strip()
        stored["subtitle_srt"] = self.srt_file.text().strip()
        stored["subtitle_font"] = self.sub_font.currentText()
        try:
            self.ctx.config.set(OVERLAY_LAST_KEY, stored)
        except Exception:
            log.exception("could not persist overlay config")

    def _on_apply_config(self):
        self._save_overlay_config()
        self.ov_status.setText(tr("overlay.applied", "Saved: the Export tab and the series "
                                                     "pipeline use these overlays."))

    def _load_last_overlay_config(self):
        try:
            d = self.ctx.config.get(OVERLAY_LAST_KEY, None)
        except Exception:
            d = None
        if not isinstance(d, dict):
            return
        try:
            from app.engines.media_exporter import GRID_POSITIONS
            self.wm_box.setChecked(d.get("watermark_on", True) is not False)
            self.wm_file.setText(d.get("watermark") or "")
            self.wm_text.setText(d.get("watermark_text") or "")
            color = str(d.get("watermark_color") or "auto")
            self.wm_color_auto.setChecked(not color.startswith("#"))
            if color.startswith("#"):
                self.wm_color.set_color(color)
            self.wm_opacity.setValue(
                max(5, min(100, int(round(float(d.get("watermark_opacity", 0.6)) * 100)))))
            self.wm_size.setValue(
                max(2, min(40, int(round(float(d.get("watermark_scale", 0.15)) * 100)))))
            pos = str(d.get("watermark_pos", "top_right"))
            k = len(GRID_POSITIONS) if pos == "tile" else (
                GRID_POSITIONS.index(pos) if pos in GRID_POSITIONS else 2)
            self.wm_pos_group.button(k).setChecked(True)
            self.banner_file.setText(d.get("banner") or "")
            self.banner_ratio.setValue(float(d.get("banner_height_ratio", 0.12)))
            self.chroma_file.setText(d.get("chroma_video") or "")
            cc = str(d.get("chroma_color", "0x00ff00")).replace("0x", "#")
            self.chroma_color.set_color(cc if cc.startswith("#") else "#00ff00")
            self.chroma_sim.setValue(float(d.get("chroma_similarity", 0.15)))
            i = self.chroma_pos.findData(d.get("chroma_pos", "bottom_right"))
            if i >= 0:
                self.chroma_pos.setCurrentIndex(i)
            self.ad_file.setText(d.get("ad_video") or "")
            self.ad_start.setValue(float(d.get("ad_start", 0.0)))
            i = self.ad_pos.findData(d.get("ad_pos", "top_left"))
            if i >= 0:
                self.ad_pos.setCurrentIndex(i)
            self.ad_scale.setValue(float(d.get("ad_scale", 0.3)))
            self.blur_strength.setValue(int(d.get("blur_strength", 20)))
            for reg in d.get("blur_regions", []) or []:
                if isinstance(reg, (list, tuple)) and len(reg) == 4:
                    self._add_blur_row(*[int(v) for v in reg])
            self.srt_file.setText(d.get("subtitle_srt", ""))
            f = d.get("subtitle_font", "")
            if f in SUB_FONTS:
                self.sub_font.setCurrentText(f)
        except Exception:
            log.exception("could not restore overlay config")

    # -- the live preview ------------------------------------------------------

    def _set_ov_video(self, path: str, ask: bool = False) -> None:
        """Show `path`'s frames under the overlays."""
        if not path or not os.path.isfile(path):
            if ask:
                QMessageBox.information(
                    self, tr("overlay.title", "Overlay"),
                    _msg("Select a source video first (Import tab).",
                         "សូមជ្រើសរើសវីដេអូប្រភពជាមុនសិន (ផ្ទាំង Import)។"))
            return
        self._ov_video = path
        self.ov_video_lbl.setText(os.path.basename(path))
        self.ov_video_lbl.setToolTip(path)
        try:
            from app.engines.media_exporter import probe
            self._ov_duration = float(probe(path).duration or 0.0)
        except Exception:
            self._ov_duration = 0.0
        self.ov_slider.setEnabled(self._ov_duration > 0)
        self._on_ov_slider(self.ov_slider.value(), now=True)

    def _on_ov_browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, tr("thumb.pick", "Choose video"), "",
            tr("overlay.filter_video", "Videos") + " (*.mp4 *.mkv *.mov *.webm *.avi *.m4v);;All (*.*)")
        if p:
            self._set_ov_video(p, ask=True)

    def _ov_frame_time(self) -> float:
        return self._ov_duration * self.ov_slider.value() / 1000.0

    def _on_ov_slider(self, _v: int = 0, now: bool = False) -> None:
        t = self._ov_frame_time()
        self.ov_time.setText(f"{int(t // 60)}:{int(t % 60):02d}")
        if now:
            self._load_ov_frame()
        else:
            self._ov_frame_timer.start()

    def _load_ov_frame(self) -> None:
        if not self._ov_video:
            return
        if self._ov_frame_busy:
            self._ov_frame_again = True
            return
        self._ov_frame_busy = True
        video, t = self._ov_video, self._ov_frame_time()

        def ok(img):
            self._ov_frame_busy = False
            if video == self._ov_video:
                self._ov_frame = img
                self._compose_ov_preview()
            if self._ov_frame_again:
                self._ov_frame_again = False
                self._load_ov_frame()

        def failed(en, km):
            self._ov_frame_busy = False
            self.ov_status.setText(_msg(en, km))

        w = FnWorker(_frame_image_fn, video, t)
        w.finished_ok.connect(ok)
        w.failed.connect(failed)
        self._launch(w)

    def _schedule_ov_preview(self, *_a) -> None:
        self._ov_compose_timer.start()

    def _compose_ov_preview(self) -> None:
        if self._ov_frame is None:
            return
        if self._ov_busy:
            self._ov_again = True
            return
        self._ov_busy = True

        def ok(img):
            self._ov_busy = False
            self.ov_canvas.setText("")
            self.ov_canvas.set_image(QPixmap.fromImage(_qimage(img)))
            if self._ov_again:
                self._ov_again = False
                self._compose_ov_preview()

        def failed(en, km):
            self._ov_busy = False
            self.ov_status.setText(_msg(en, km))

        w = FnWorker(_compose_overlay_fn, self._ov_frame, self._collect_overlay_dict())
        w.finished_ok.connect(ok)
        w.failed.connect(failed)
        self._launch(w)

    def _on_pick_blur_region(self):
        if self._ov_frame is None:
            if not self._ov_video:
                self._set_ov_video(self.ctx.current_video or "", ask=True)
            if not self._ov_video:
                return
        self.ov_canvas.mode = "manual"
        self.ov_status.setText(tr("overlay.pick_hint", "Drag a rectangle over the area to blur:"))

    def _on_blur_drawn(self, x: int, y: int, w: int, h: int):
        self.ov_canvas.mode = "select"
        self.ov_status.setText("")
        self._add_blur_row(x, y, w, h)

    def _on_preview_10s(self):
        video = self._ov_video or self.ctx.current_video
        if not video or not os.path.exists(video):
            self._set_ov_video(video or "", ask=True)
            return
        self._save_overlay_config()
        srt = self.srt_file.text().strip() or None
        if srt and not os.path.exists(srt):
            srt = None
        start = self._ov_frame_time() if video == self._ov_video else 0.0

        def done(path):
            try:
                self.ctx.open_in_preview(path)
            except Exception:
                log.exception("open_in_preview failed")
                QMessageBox.information(
                    self, tr("overlay.title", "Overlay"),
                    _msg(f"Preview rendered: {path}",
                         f"បានបង្កើតវីដេអូសាកល្បង៖ {path}"))

        self._run_task(
            self.preview_row, _preview_render_fn, video,
            self._collect_overlay_dict(), srt, start,
            on_ok=done,
            running_text=tr("overlay.previewing", "Rendering preview…"))

    # =========================================================================
    # Poster editor sub-tab
    # =========================================================================

    def _default_spec_dict(self) -> dict:
        try:
            from app.engines.poster_designer import StyleSpec
            return dataclasses.asdict(StyleSpec())
        except Exception:
            return {
                "font": "Moul", "size_ratio": 0.12, "fill": "#ffffff",
                "gradient": None, "stroke_color": "#000000",
                "stroke_width_ratio": 0.04, "shadow_color": "#000000cc",
                "shadow_offset": (4, 4), "shadow_blur": 8, "glow_color": None,
                "glow_radius": 0, "bevel": False, "extrude_depth": 0,
                "extrude_color": "#222222", "pos": (0.5, 0.82),
                "letter_spacing": 0, "rotation": 0.0,
            }

    def _build_poster_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 6, 0, 0)

        # ---- the video this thumbnail is for: the poster it carries (what the
        #      file shows, e.g. from the DL tab) or its best frames
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel(tr("thumb.video", "Video:")))
        self.pv_video = QLineEdit()
        self.pv_video.setPlaceholderText(tr(
            "poster.video_ph", "The video this thumbnail is for: its own thumbnail opens here"))
        self.pv_video.editingFinished.connect(
            lambda: self.pv_video.text().strip()
            and self._set_poster_video(self.pv_video.text().strip(), ask=True))
        vrow.addWidget(self.pv_video, 1)
        b = QPushButton(tr("common.browse", "Browse…"))
        b.clicked.connect(self._on_poster_video_browse)
        vrow.addWidget(b)
        b = QPushButton(tr("thumb.use_current", "Use current video"))
        b.clicked.connect(lambda: self._set_poster_video(self.ctx.current_video or "", ask=True))
        vrow.addWidget(b)
        vrow.addSpacing(8)
        vrow.addWidget(QLabel(tr("thumb.count", "Frames:")))
        self.pv_count = QSpinBox()
        self.pv_count.setRange(4, 16)
        self.pv_count.setValue(8)
        vrow.addWidget(self.pv_count)
        self.pv_find = QPushButton(tr("thumb.find", "Find best frames"))
        self.pv_find.setProperty("class", "accent")
        self.pv_find.setToolTip(tr("poster.find_tip", "Pictures of the video with clear faces "
                                                      "and strong colours, best first"))
        self.pv_find.clicked.connect(self._on_find_frames)
        vrow.addWidget(self.pv_find)
        outer.addLayout(vrow)
        self.pv_row = _make_progress_row(tr("thumb.working", "Thumbnail"))
        outer.addWidget(self.pv_row)
        self.pv_strip = QListWidget()
        self.pv_strip.setViewMode(QListWidget.ViewMode.IconMode)
        self.pv_strip.setFlow(QListWidget.Flow.LeftToRight)
        self.pv_strip.setWrapping(False)
        self.pv_strip.setMovement(QListWidget.Movement.Static)
        self.pv_strip.setIconSize(QSize(160, 90))
        self.pv_strip.setSpacing(4)
        self.pv_strip.setFixedHeight(146)
        self.pv_strip.setToolTip(tr("poster.strip_tip", "Click a picture to design the "
                                                        "thumbnail on it"))
        self.pv_strip.itemClicked.connect(self._on_strip_pick)
        self.pv_strip.setVisible(False)
        outer.addWidget(self.pv_strip)
        lay = QHBoxLayout()
        outer.addLayout(lay, 1)

        # ---- left: canvas + load
        left = QVBoxLayout()
        load_row = QHBoxLayout()
        self.poster_browse_btn = QPushButton(tr("poster.load", "Load image…"))
        self.poster_browse_btn.clicked.connect(self._on_poster_browse)
        load_row.addWidget(self.poster_browse_btn)
        self.manual_box_check = QCheckBox(tr("poster.manual_box", "Manual box"))
        self.manual_box_check.toggled.connect(self._on_manual_toggle)
        load_row.addWidget(self.manual_box_check)
        load_row.addStretch(1)
        left.addLayout(load_row)

        self.canvas = _ImageCanvas()
        self.canvas.setAcceptDrops(True)
        self.canvas.box_clicked.connect(self._on_box_clicked)
        self.canvas.manual_box.connect(self._on_manual_box)
        self.canvas.pos_dragged.connect(self._on_pos_dragged)
        self.canvas.drag_finished.connect(self._on_drag_finished)
        # drag&drop straight onto the canvas
        self.canvas.dragEnterEvent = self._canvas_drag_enter  # type: ignore[method-assign]
        self.canvas.dropEvent = self._canvas_drop  # type: ignore[method-assign]
        left.addWidget(self.canvas, 1)

        self.poster_status = QLabel(tr(
            "poster.hint", "Load a poster, then Translate the title automatically — or "
            "pick an AI idea or type your own title and press Replace title — same style."))
        self.poster_status.setWordWrap(True)
        self.poster_status.setStyleSheet("color: #8b949e;")
        left.addWidget(self.poster_status)
        lay.addLayout(left, 3)

        # ---- right: controls (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        rw = QWidget()
        right = QVBoxLayout(rw)

        # step buttons
        steps = QGroupBox(tr("poster.steps", "Steps"))
        sv = QVBoxLayout(steps)
        # automatic: the main title read, translated and drawn back in its look
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel(tr("poster.lang", "Translate into:")))
        self.poster_lang = QComboBox()
        for code, native in LANGS:
            self.poster_lang.addItem(native, code)
        lang_row.addWidget(self.poster_lang, 1)
        sv.addLayout(lang_row)
        self.auto_btn = QPushButton(tr("poster.auto", "Translate the title automatically"))
        self.auto_btn.setProperty("class", "accent")
        self.auto_btn.setToolTip(tr(
            "poster.auto_tip",
            "Reads the poster's main title, translates it into the language above, takes "
            "the old title off and draws the translation in its place with the same look: "
            "colours, outline, glow, ink band, angle, size and the nearest Khmer font. "
            "One AI request (the AI looks at the picture)."))
        self.auto_btn.clicked.connect(self._on_auto_title)
        sv.addWidget(self.auto_btn)
        self.auto_row = _make_progress_row(tr("poster.auto_running",
                                              "Reading and translating the title…"))
        sv.addWidget(self.auto_row)
        self.auto_batch_btn = QPushButton(tr("poster.auto_batch",
                                             "Translate the titles of many pictures…"))
        self.auto_batch_btn.setToolTip(tr(
            "poster.auto_batch_tip",
            "Each picture on its own: its own title, translation and look. Each is saved "
            "as a PNG of the same size named <name>_KH.png."))
        self.auto_batch_btn.clicked.connect(self._on_auto_titles)
        sv.addWidget(self.auto_batch_btn)
        self.auto_batch_row = _make_progress_row(tr("poster.auto_batch_running",
                                                    "Translating the titles…"))
        sv.addWidget(self.auto_batch_row)
        self.auto_list = QListWidget()
        self.auto_list.setVisible(False)
        self.auto_list.setMaximumHeight(150)
        self.auto_list.setToolTip(tr("poster.auto_list_tip",
                                     "Click a picture to check or change it here"))
        self.auto_list.itemClicked.connect(self._on_auto_item)
        sv.addWidget(self.auto_list)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #30363d;")
        sv.addWidget(line)

        self.translate_btn = QPushButton(tr("poster.translate", "Translate title (3 options)"))
        self.translate_btn.clicked.connect(self._on_translate_title)
        sv.addWidget(self.translate_btn)
        self.translate_row = _make_progress_row(tr("poster.translating", "Translating title…"))
        sv.addWidget(self.translate_row)
        self.ideas_btn = QPushButton(tr("poster.ideas", "AI title ideas (3 options)"))
        self.ideas_btn.setToolTip(tr(
            "poster.ideas_tip", "Three short, click-worthy titles in the language above, "
                                "written by the AI from the story (the Recap tab's story, "
                                "else the video's name)."))
        self.ideas_btn.clicked.connect(self._on_ideas)
        sv.addWidget(self.ideas_btn)
        self.ideas_row = _make_progress_row(tr("poster.ideas_running", "Writing title ideas…"))
        sv.addWidget(self.ideas_row)

        self.option_group = QButtonGroup(self)
        self.option_radios: list[QRadioButton] = []
        for i in range(3):
            rb = QRadioButton("—")
            rb.setVisible(False)
            self.option_group.addButton(rb, i)
            self.option_radios.append(rb)
            sv.addWidget(rb)
        self.option_group.idClicked.connect(self._on_option_picked)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText(tr("poster.title_text2",
                                              "Your title…  ( | starts a new line)"))
        self.title_edit.editingFinished.connect(self._on_title_edited)
        # drawn while typing (after a short pause), so the thumbnail is seen as it grows
        self.title_edit.textChanged.connect(lambda _t: self._schedule_render())
        sv.addWidget(self.title_edit)

        self.replace_btn = QPushButton(tr("poster.replace", "Replace title — same style"))
        self.replace_btn.setToolTip(tr(
            "poster.replace_tip",
            "The AI (a Gemini picture model) redesigns the poster's title with the text in "
            "the box (an AI idea or your own) in the original title's own art style — no "
            "fonts. Needs a Gemini key with billing on (about US$0.04 a picture) or "
            "OpenRouter credit; without one you can draw it with the app's fonts."))
        self.replace_btn.setStyleSheet("font-weight: 600;")
        self.replace_btn.clicked.connect(self._on_replace_title)
        sv.addWidget(self.replace_btn)
        self.replace_row = _make_progress_row(tr("poster.replacing", "Replacing the title…"))
        sv.addWidget(self.replace_row)

        self.design_btn = QPushButton(tr("poster.ai_design", "AI design"))
        self.design_btn.clicked.connect(self._on_ai_design)
        sv.addWidget(self.design_btn)
        self.design_row = _make_progress_row(tr("poster.designing", "Designing title style…"))
        sv.addWidget(self.design_row)
        right.addWidget(steps)

        # thumbnail: size and the small top line
        thumb_box = QGroupBox(tr("poster.thumb_box", "Thumbnail"))
        tf = QFormLayout(thumb_box)
        self.pv_size = QComboBox()
        self.pv_size.addItem(tr("poster.size_original", "Original size"), None)
        for key, label, size in THUMB_SIZES:
            self.pv_size.addItem(tr(f"thumb.size.{key}", label), size)
        self.pv_size.setToolTip(tr("poster.size_tip", "A picture of the video is cut to this "
                                                      "size around the faces"))
        self.pv_size.currentIndexChanged.connect(self._on_thumb_size)
        tf.addRow(tr("thumb.size", "Size:"), self.pv_size)
        self.tag_edit = QLineEdit()
        self.tag_edit.setPlaceholderText(tr("thumb.tag.ph", "Small top line, e.g. ភាគ ១ / EP 1"))
        self.tag_edit.editingFinished.connect(self._on_title_edited)
        self.tag_edit.textChanged.connect(lambda _t: self._schedule_render())
        tf.addRow(tr("thumb.tag", "Tag line:"), self.tag_edit)
        right.addWidget(thumb_box)

        # manual style panel
        style_box = QGroupBox(tr("poster.style", "Style overrides"))
        form = QFormLayout(style_box)

        self.gold_btn = QPushButton(tr("poster.gold", "Gold brush style"))
        self.gold_btn.setToolTip(tr(
            "poster.gold_tip", "A drama-poster title: metallic gold letters with a dark "
                               "outline and a warm glow on a black ink-brush stroke, with "
                               "a few sparkles."))
        self.gold_btn.clicked.connect(self._on_gold_preset)
        preset_row = QWidget()
        prl = QHBoxLayout(preset_row)
        prl.setContentsMargins(0, 0, 0, 0)
        prl.addWidget(self.gold_btn)
        self.quick_style = QComboBox()
        self.quick_style.addItem(tr("poster.quick_style", "Thumbnail styles…"), "")
        for sid, label in THUMB_STYLE_LABELS.items():
            self.quick_style.addItem(tr(f"thumb.style.{sid}", label), sid)
        self.quick_style.activated.connect(self._on_quick_style)
        prl.addWidget(self.quick_style, 1)
        form.addRow(tr("poster.preset", "Preset:"), preset_row)

        self.font_combo = QComboBox()
        self.font_combo.addItems(_font_families())
        self.font_combo.currentTextChanged.connect(
            lambda v: self._spec_change("font", v))
        form.addRow(tr("poster.font", "Font:"), self.font_combo)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(2, 40)
        self.size_slider.setValue(12)
        self.size_slider.valueChanged.connect(
            lambda v: self._spec_change("size_ratio", v / 100.0))
        form.addRow(tr("poster.size", "Size (% width):"), self.size_slider)

        self.fill_btn = _ColorButton("#ffffff")
        self.fill_btn.color_changed.connect(
            lambda c: self._spec_change("fill", c))
        form.addRow(tr("poster.fill", "Fill:"), self.fill_btn)

        grad_holder = QWidget()
        gh = QHBoxLayout(grad_holder)
        gh.setContentsMargins(0, 0, 0, 0)
        self.grad_check = QCheckBox(tr("poster.gradient", "Gradient"))
        self.grad_check.toggled.connect(self._on_gradient_changed)
        gh.addWidget(self.grad_check)
        self.grad_c1 = _ColorButton("#ffdd55")
        self.grad_c1.color_changed.connect(lambda _c: self._on_gradient_changed())
        gh.addWidget(self.grad_c1)
        self.grad_c2 = _ColorButton("#ff5533")
        self.grad_c2.color_changed.connect(lambda _c: self._on_gradient_changed())
        gh.addWidget(self.grad_c2)
        gh.addStretch(1)
        form.addRow("", grad_holder)

        stroke_holder = QWidget()
        sh = QHBoxLayout(stroke_holder)
        sh.setContentsMargins(0, 0, 0, 0)
        self.stroke_btn = _ColorButton("#000000")
        self.stroke_btn.color_changed.connect(
            lambda c: self._spec_change("stroke_color", c))
        sh.addWidget(self.stroke_btn)
        self.stroke_width = QSpinBox()
        self.stroke_width.setRange(0, 25)
        self.stroke_width.setValue(4)
        self.stroke_width.setSuffix(" %")
        self.stroke_width.valueChanged.connect(
            lambda v: self._spec_change("stroke_width_ratio", v / 100.0))
        sh.addWidget(self.stroke_width)
        sh.addStretch(1)
        form.addRow(tr("poster.stroke", "Stroke:"), stroke_holder)

        shadow_holder = QWidget()
        shh = QHBoxLayout(shadow_holder)
        shh.setContentsMargins(0, 0, 0, 0)
        self.shadow_btn = _ColorButton("#000000cc", allow_alpha=True)
        self.shadow_btn.color_changed.connect(
            lambda c: self._spec_change("shadow_color", c))
        shh.addWidget(self.shadow_btn)
        self.shadow_dx = QSpinBox(); self.shadow_dx.setRange(-40, 40); self.shadow_dx.setValue(4)
        self.shadow_dy = QSpinBox(); self.shadow_dy.setRange(-40, 40); self.shadow_dy.setValue(4)
        self.shadow_blur = QSpinBox(); self.shadow_blur.setRange(0, 60); self.shadow_blur.setValue(8)
        for wdg in (self.shadow_dx, self.shadow_dy):
            wdg.valueChanged.connect(self._on_shadow_offset)
        self.shadow_blur.valueChanged.connect(
            lambda v: self._spec_change("shadow_blur", int(v)))
        shh.addWidget(QLabel("dx")); shh.addWidget(self.shadow_dx)
        shh.addWidget(QLabel("dy")); shh.addWidget(self.shadow_dy)
        shh.addWidget(QLabel(tr("poster.blur", "blur"))); shh.addWidget(self.shadow_blur)
        form.addRow(tr("poster.shadow", "Shadow:"), shadow_holder)

        glow_holder = QWidget()
        gl = QHBoxLayout(glow_holder)
        gl.setContentsMargins(0, 0, 0, 0)
        self.glow_check = QCheckBox("")
        self.glow_check.toggled.connect(self._on_glow_changed)
        gl.addWidget(self.glow_check)
        self.glow_btn = _ColorButton("#58a6ff")
        self.glow_btn.color_changed.connect(lambda _c: self._on_glow_changed())
        gl.addWidget(self.glow_btn)
        self.glow_radius = QSpinBox()
        self.glow_radius.setRange(0, 60)
        self.glow_radius.setValue(12)
        self.glow_radius.valueChanged.connect(lambda _v: self._on_glow_changed())
        gl.addWidget(self.glow_radius)
        gl.addStretch(1)
        form.addRow(tr("poster.glow", "Glow:"), glow_holder)

        b_e_holder = QWidget()
        be = QHBoxLayout(b_e_holder)
        be.setContentsMargins(0, 0, 0, 0)
        self.bevel_check = QCheckBox(tr("poster.bevel", "Bevel"))
        self.bevel_check.toggled.connect(
            lambda v: self._spec_change("bevel", bool(v)))
        be.addWidget(self.bevel_check)
        be.addWidget(QLabel(tr("poster.extrude", "Extrude:")))
        self.extrude_depth = QSpinBox()
        self.extrude_depth.setRange(0, 40)
        self.extrude_depth.valueChanged.connect(
            lambda v: self._spec_change("extrude_depth", int(v)))
        be.addWidget(self.extrude_depth)
        self.extrude_btn = _ColorButton("#222222")
        self.extrude_btn.color_changed.connect(
            lambda c: self._spec_change("extrude_color", c))
        be.addWidget(self.extrude_btn)
        be.addStretch(1)
        form.addRow("", b_e_holder)

        ink_holder = QWidget()
        ih = QHBoxLayout(ink_holder)
        ih.setContentsMargins(0, 0, 0, 0)
        self.backdrop_check = QCheckBox(tr("poster.backdrop", "Ink band"))
        self.backdrop_check.setToolTip(tr(
            "poster.backdrop_tip", "A dark ink-brush stroke behind each line."))
        self.backdrop_check.toggled.connect(
            lambda v: self._spec_change("backdrop", bool(v)))
        ih.addWidget(self.backdrop_check)
        self.backdrop_btn = _ColorButton("#0a0705eb", allow_alpha=True)
        self.backdrop_btn.color_changed.connect(
            lambda c: self._spec_change("backdrop_color", c))
        ih.addWidget(self.backdrop_btn)
        self.sparkle_check = QCheckBox(tr("poster.sparkles", "Sparkles"))
        self.sparkle_check.toggled.connect(
            lambda v: self._spec_change("sparkles", 12 if v else 0))
        ih.addWidget(self.sparkle_check)
        ih.addStretch(1)
        form.addRow("", ink_holder)

        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(-20, 80)
        self.spacing_spin.valueChanged.connect(
            lambda v: self._spec_change("letter_spacing", int(v)))
        form.addRow(tr("poster.spacing", "Letter spacing:"), self.spacing_spin)

        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-45.0, 45.0)
        self.rotation_spin.setSingleStep(0.5)
        self.rotation_spin.valueChanged.connect(
            lambda v: self._spec_change("rotation", float(v)))
        form.addRow(tr("poster.rotation", "Rotation:"), self.rotation_spin)
        right.addWidget(style_box)

        self.render_row = _make_progress_row(tr("poster.rendering", "Rendering title…"))
        right.addWidget(self.render_row)

        # exports
        exp_box = QGroupBox(tr("poster.export", "Export"))
        ev = QVBoxLayout(exp_box)
        self.set_thumb_btn = QPushButton(tr("poster.set_thumb", "Put into the video as its thumbnail"))
        self.set_thumb_btn.setProperty("class", "accent")
        self.set_thumb_btn.setToolTip(tr(
            "poster.set_thumb_tip", "The finished picture becomes the thumbnail the video "
                                    "file shows (Windows Explorer, phones, players). The "
                                    "video's picture and sound are not changed."))
        self.set_thumb_btn.clicked.connect(self._on_set_thumbnail)
        ev.addWidget(self.set_thumb_btn)
        self.set_thumb_row = _make_progress_row(
            tr("poster.setting_thumb", "Putting the picture into the video…"))
        ev.addWidget(self.set_thumb_row)
        er1 = QHBoxLayout()
        png_btn = QPushButton(tr("poster.export_png", "Export PNG/JPG"))
        png_btn.clicked.connect(self._on_export_image)
        er1.addWidget(png_btn)
        var_btn = QPushButton(tr("poster.export_variants", "Export 9:16 && 16:9"))
        var_btn.clicked.connect(self._on_export_variants)
        er1.addWidget(var_btn)
        ev.addLayout(er1)
        self.variants_row = _make_progress_row(tr("poster.exporting", "Exporting variants…"))
        ev.addWidget(self.variants_row)
        er2 = QHBoxLayout()
        save_tpl_btn = QPushButton(tr("poster.save_tpl", "Save style template"))
        save_tpl_btn.clicked.connect(self._on_save_template)
        er2.addWidget(save_tpl_btn)
        apply_tpl_btn = QPushButton(tr("poster.apply_tpl", "Apply template to folder…"))
        apply_tpl_btn.clicked.connect(self._on_apply_template_folder)
        er2.addWidget(apply_tpl_btn)
        ev.addLayout(er2)
        self.auto_per_check = QCheckBox(
            tr("poster.auto_per", "Auto per poster (OCR + translate each)"))
        ev.addWidget(self.auto_per_check)
        self.batch_row = _make_progress_row(tr("poster.batch_running", "Batch running…"))
        ev.addWidget(self.batch_row)
        self.batch_list = QListWidget()
        self.batch_list.setVisible(False)
        self.batch_list.setMaximumHeight(120)
        ev.addWidget(self.batch_list)
        right.addWidget(exp_box)
        right.addStretch(1)

        scroll.setWidget(rw)
        lay.addWidget(scroll, 2)
        return page

    # ------------------------------------------------------------ image load

    def showEvent(self, ev):  # noqa: N802
        p = getattr(self.ctx, "poster_path", None)
        if p and os.path.exists(p) and p != getattr(self, "_image_path", None):
            self._use_file_picture(p)        # an image sent from the Import tab
            self.tabs.setCurrentIndex(0)
        super().showEvent(ev)
        self._on_side_shown()

    def _on_side_shown(self) -> None:
        """A side opens on the current video when it has none yet: the Poster
        side on the video's own thumbnail, the Overlay side on one of its frames."""
        cur = self.ctx.current_video or ""
        if not cur or not os.path.isfile(cur) or not self.isVisible():
            return
        if self.tabs.currentIndex() == 0 and not self._pv_video and not self._work_path:
            self._set_poster_video(cur)
        elif self.tabs.currentIndex() == 1 and not self._ov_video:
            self._set_ov_video(cur)

    def _canvas_drag_enter(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def _canvas_drop(self, ev):
        files = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        images = [f for f in files if f.lower().endswith(IMAGE_EXTS)]
        videos = [f for f in files if f.lower().endswith(
            tuple(VIDEO_FILTER.strip("()").replace("*", "").split()))]
        if images:
            self._use_file_picture(images[0])
        elif videos:
            self._set_poster_video(videos[0], ask=True)
        ev.acceptProposedAction()

    def _on_poster_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("poster.load", "Load image…"), "",
            tr("overlay.filter_image", "Images") + " (*.png *.jpg *.jpeg *.webp)")
        if path:
            self._use_file_picture(path)

    # -- the video's own thumbnail and its best frames -------------------------

    def _on_poster_video_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("thumb.pick", "Choose video"), "",
            tr("overlay.filter_video", "Videos") + f" {VIDEO_FILTER};;All (*.*)")
        if path:
            self._set_poster_video(path, ask=True)

    def _set_poster_video(self, path: str, ask: bool = False) -> None:
        """The video the thumbnail is for: the strip shows the poster it
        carries (opened at once when the picture on the canvas is not a file
        of the user's) and, on request, its best frames."""
        if not path or not os.path.isfile(path):
            if ask:
                QMessageBox.information(self, tr("thumb.title_box", "Thumbnail"),
                                        tr("thumb.no_video", "Choose a video first."))
            return
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                os.path.abspath(self._pv_video or "~")) and self.pv_strip.count():
            return
        self._pv_video = path
        self.pv_video.setText(path)
        self.pv_strip.clear()
        self.pv_strip.setVisible(True)
        self.poster_status.setText(tr("poster.reading_thumb", "Reading the video's thumbnail…"))

        def ok(poster):
            if path != self._pv_video:
                return
            if poster:
                self._strip_add(poster, tr("poster.own_thumb", "Video's thumbnail"),
                                {"kind": "cover", "path": poster, "fx": 0.5, "fy": 0.4}, first=True)
                if not self._work_path or self._pv_from_strip:
                    self._on_strip_pick(self.pv_strip.item(0))
                self.poster_status.setText(tr(
                    "poster.own_thumb_hint", "The video's own thumbnail is open: type a title, "
                                             "or press Find best frames for other pictures."))
            else:
                self.poster_status.setText(tr(
                    "poster.no_own_thumb", "This video carries no thumbnail: press Find best "
                                           "frames to choose one of its pictures."))

        def failed(en, km):
            self.poster_status.setText(_msg(en, km))

        w = FnWorker(_video_poster_fn, path)
        w.finished_ok.connect(ok)
        w.failed.connect(failed)
        self._launch(w)

    def _strip_add(self, path: str, text: str, data: dict, first: bool = False) -> None:
        item = QListWidgetItem(QIcon(QPixmap(path)), text)
        item.setData(Qt.ItemDataRole.UserRole, data)
        item.setToolTip(text)
        if first:
            self.pv_strip.insertItem(0, item)
        else:
            self.pv_strip.addItem(item)

    def _on_find_frames(self) -> None:
        video = self.pv_video.text().strip()
        if video and video != self._pv_video:
            self._set_poster_video(video, ask=True)
        video = self._pv_video
        if not video or not os.path.isfile(video):
            QMessageBox.information(self, tr("thumb.title_box", "Thumbnail"),
                                    tr("thumb.no_video", "Choose a video first."))
            return

        def done(frames):
            if video != self._pv_video:
                return
            for i in reversed(range(self.pv_strip.count())):       # the old frames go
                d = self.pv_strip.item(i).data(Qt.ItemDataRole.UserRole) or {}
                if d.get("kind") == "frame":
                    self.pv_strip.takeItem(i)
            frames = list(frames or [])
            for c in frames:
                if not os.path.isfile(c.path):
                    continue
                label = (f"{int(c.time // 60)}:{c.time % 60:04.1f}  ·  "
                         + tr("thumb.faces", "faces") + f" {c.faces}")
                self._strip_add(c.path, label, {"kind": "frame", "path": c.path,
                                                "fx": c.face_x, "fy": c.face_y})
            self.pv_strip.setVisible(True)
            self.poster_status.setText(tr("thumb.found", "Best frames first") + f": {len(frames)}")
            if frames and (not self._work_path or not self._pv_from_strip):
                for i in range(self.pv_strip.count()):
                    d = self.pv_strip.item(i).data(Qt.ItemDataRole.UserRole) or {}
                    if d.get("kind") == "frame":
                        self._on_strip_pick(self.pv_strip.item(i))
                        break

        self._run_task(self.pv_row, _find_frames_fn, video, int(self.pv_count.value()),
                       on_ok=done, running_text=tr("thumb.find", "Find best frames"))

    def _on_strip_pick(self, item) -> None:
        if item is None:
            return
        d = item.data(Qt.ItemDataRole.UserRole) or {}
        path = str(d.get("path") or "")
        if not path or not os.path.isfile(path):
            return
        self.pv_strip.setCurrentItem(item)
        key = f"{path}|{self.pv_size.currentIndex()}"
        if key == self._pv_pick_key and self._work_path:
            return                          # the same picture: keep the edits on it
        self._use_picture(path, float(d.get("fx", 0.5)), float(d.get("fy", 0.4)),
                          from_strip=True)

    def _use_file_picture(self, path: str) -> None:
        """An image file of the user's (Load image…, a drop, the Import tab)."""
        self.pv_strip.clearSelection()
        self._use_picture(path, 0.5, 0.4, from_strip=False)

    def _use_picture(self, path: str, fx: float, fy: float, from_strip: bool) -> None:
        """Design on `path`: cut to the thumbnail size when one is chosen; the
        title and tag line typed so far are drawn on it at once."""
        self._pv_pick = (path, fx, fy)
        self._pv_from_strip = from_strip
        self._pv_pick_key = f"{path}|{self.pv_size.currentIndex()}" if from_strip else ""
        size = self.pv_size.currentData()

        def show(pic: str) -> None:
            if self._pv_pick != (path, fx, fy):
                return                      # another picture was chosen meanwhile
            self._load_poster(pic)
            if self.title_edit.text().strip() or self.tag_edit.text().strip():
                self._schedule_render()

        if not size:
            show(path)
            return
        self._run_task(self.pv_row, _sized_picture_fn, path, size, fx, fy, on_ok=show,
                       running_text=tr("thumb.working", "Thumbnail"))

    def _on_thumb_size(self, _i: int = 0) -> None:
        if self._pv_pick:
            path, fx, fy = self._pv_pick
            self._use_picture(path, fx, fy, self._pv_from_strip)

    def _on_quick_style(self, index: int) -> None:
        """A thumbnail style (Bold yellow, Clean white, Drama red, Neon blue),
        sized so the title fits the picture."""
        sid = self.quick_style.itemData(index)
        self.quick_style.setCurrentIndex(0)
        if not sid:
            return
        from PySide6.QtGui import QImageReader

        from app.engines import poster_designer as pd
        from app.engines import thumbnail as th
        spec = dataclasses.replace(th.STYLES[sid])
        text = self.title_edit.text().strip()
        img = self._work_path
        if img and os.path.exists(img) and text:
            size = QImageReader(img).size()
            if size.isValid() and size.width() > 0:
                try:
                    spec.size_ratio = min(spec.size_ratio, pd.fit_size_ratio(
                        text, spec, 0.92 * size.width(), 0.30 * size.height(),
                        size.width(), fonts_dir()))
                except Exception:
                    log.exception("could not size the title")
        if self._rendered_path and isinstance(self._spec_dict.get("pos"), (list, tuple)):
            spec.pos = tuple(self._spec_dict["pos"])
        self._push_snapshot()
        self._spec_dict = dataclasses.asdict(spec)
        self._apply_spec_to_controls()
        self._schedule_render()

    def _on_ideas(self) -> None:
        """Three title ideas from the story, in the language chosen above."""
        rows = getattr(self.ctx, "recap_rows", None)
        if rows:
            context = " ".join(str(getattr(r, "narration", "")) for r in rows)[:5000]
        else:
            video = self._pv_video or self.ctx.current_video or ""
            context = (self.title_edit.text().strip()
                       or os.path.splitext(os.path.basename(video))[0])
        if not context.strip():
            QMessageBox.information(self, tr("poster.title", "Poster"), tr(
                "poster.ideas_none", "Choose the video (or write the title) first: the ideas "
                                     "come from its story."))
            return
        lang = str(self.poster_lang.currentData() or "km")

        def done(options):
            options = list(options or [])[:3]
            for i, rb in enumerate(self.option_radios):
                rb.setVisible(i < len(options))
                if i < len(options):
                    rb.setText(options[i])
            if options:
                self.option_radios[0].setChecked(True)
                self._push_snapshot()
                self.title_edit.setText(options[0])
                self._schedule_render()

        self._run_task(self.ideas_row, _ideas_fn, self.ctx.ai, context, lang, on_ok=done,
                       running_text=tr("poster.ideas_running", "Writing title ideas…"))

    def _on_set_thumbnail(self) -> None:
        """The finished picture into the video file, as the thumbnail it shows."""
        title = tr("poster.set_thumb", "Put into the video as its thumbnail")
        video = self._pv_video or self.pv_video.text().strip()
        if not video or not os.path.isfile(video):
            QMessageBox.information(self, title, tr(
                "poster.set_thumb_no_video", "Choose the video at the top first."))
            return
        if not video.lower().endswith((".mp4", ".m4v", ".mov")):
            QMessageBox.information(self, title, tr(
                "poster.set_thumb_mp4", "Only MP4 and MOV videos carry a thumbnail picture."))
            return
        pic = self._export_source()
        if not pic:
            return
        name = os.path.basename(video)
        if QMessageBox.question(self, title, tr(
                "poster.set_thumb_q", "Put this picture into “{name}” as its thumbnail? The "
                                      "video's picture and sound stay as they are.")
                .replace("{name}", name)) != QMessageBox.StandardButton.Yes:
            return

        def done(ok):
            if ok:
                self.poster_status.setText(tr(
                    "poster.set_thumb_done", "Done: “{name}” now shows this picture as its "
                                             "thumbnail.").replace("{name}", name))
            else:
                QMessageBox.warning(self, title, tr(
                    "poster.set_thumb_failed", "Could not put the picture into the video. Is "
                                               "it open in a player? Close it and try again."))

        self._run_task(self.set_thumb_row, _set_cover_fn, video, pic, on_ok=done,
                       running_text=tr("poster.setting_thumb",
                                       "Putting the picture into the video…"))

    def _load_poster(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(
                self, tr("err.title", "Error"),
                _msg("Could not load this image.",
                     "មិនអាចផ្ទុករូបភាពនេះបានទេ។"))
            return
        self._image_path = path
        self._work_path = path
        self._rendered_path = None
        self._auto_out = ""
        self._auto_boxes = []
        self._auto_source = ""
        self._ai_designed = False
        self._boxes = []
        self._selected_box = -1
        self._undo.clear()
        self._redo.clear()
        setattr(self.ctx, "poster_path", path)
        self.canvas.set_boxes([])
        self.canvas.set_image(pix)
        for rb in self.option_radios:
            rb.setVisible(False)
        self.poster_status.setText(os.path.basename(path)
                                   + f"  ({pix.width()}×{pix.height()})")

    def _require_image(self) -> str | None:
        if not self._work_path or not os.path.exists(self._work_path):
            QMessageBox.information(
                self, tr("poster.title", "Poster"),
                _msg("Load a poster image first.",
                     "សូមផ្ទុករូបភាពប៉ុស្ទ័រជាមុនសិន។"))
            return None
        return self._work_path

    # ------------------------------------------------------------------ boxes

    def _on_manual_toggle(self, on: bool):
        self.canvas.mode = "manual" if on else "select"

    def _boxes_as_rects(self) -> list:
        return [(b.x, b.y, b.w, b.h) for b in self._boxes]

    def _on_box_clicked(self, index: int):
        self._selected_box = index
        self.canvas.set_boxes(self._boxes_as_rects(), index)
        if 0 <= index < len(self._boxes):
            txt = self._boxes[index].text.strip()
            if txt:
                self.title_edit.setText(txt)

    def _on_manual_box(self, x: int, y: int, w: int, h: int):
        try:
            from app.engines.poster_designer import OcrBox
            box = OcrBox(x=x, y=y, w=w, h=h, text="", conf=1.0)
        except Exception:
            log.exception("could not create OcrBox")
            return
        self._boxes.append(box)
        self._selected_box = len(self._boxes) - 1
        self.canvas.set_boxes(self._boxes_as_rects(), self._selected_box)
        self.manual_box_check.setChecked(False)

    # ---------------------------------------------------------- replace title

    def _replace_inputs(self):
        """(original picture, title text, boxes data, drawn by hand) for a
        replace; None after telling the user what is missing."""
        if not self._require_image():
            return None
        img = (self._image_path if self._image_path and os.path.exists(self._image_path)
               else self._work_path)
        text = self.title_edit.text().strip()
        if not text:
            QMessageBox.information(
                self, tr("poster.title", "Poster"),
                tr("poster.need_title", "Type your title in the title box first."))
            self.title_edit.setFocus()
            return None
        drawn = [b for b in self._boxes if not b.text and b.conf >= 1.0]
        boxes = drawn or list(self._auto_boxes)
        boxes_data = [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "text": b.text,
                       "conf": b.conf, "alts": list(getattr(b, "alts", ()) or ())}
                      for b in boxes] or None
        return img, text, boxes_data, bool(drawn)

    def _on_replace_title(self):
        """The title in the title box (an AI idea or your own) put in place of
        the original poster's title, lettered by the AI picture model
        (Gemini) in that title's own art style — no fonts, like the Doubao
        picture the user showed. When no key can edit pictures (free Gemini
        keys have no picture quota), the app's fonts can draw it instead."""
        got = self._replace_inputs()
        if got is None:
            return
        img, text, boxes_data, _drawn = got
        lang = self._auto_lang()
        source = self._auto_source

        def done(r):
            self._push_snapshot()
            if not self._ai_designed:
                self._pre_ai_work = self._work_path
            self._ai_designed = True
            self._work_path = r["out"]
            self._boxes = []
            self._selected_box = -1
            self.canvas.set_boxes([])
            self.canvas.set_image(QPixmap(self._work_path))
            self._rendered_path = None
            self._schedule_render()
            if not self._auto_out and lang == "km":
                from app.engines.poster_designer import output_name
                original = self._image_path or img
                self._auto_out = os.path.join(os.path.dirname(original),
                                              output_name(original, lang))
            lines = [tr("poster.ai_done", "Designed by {model}: «{text}»").format(
                model=r.get("model", ""), text=str(r.get("text") or "").replace(" | ", " / "))]
            checks = self._check_lines(r.get("checks"))
            if checks:
                lines.append(" · ".join(checks))
            lines.append(tr("poster.ai_next",
                            "To change it, edit the title and press Replace title again: each "
                            "press is a new design (a paid picture)."))
            self.poster_status.setText("\n".join(lines))

        def failed(en, km):
            msg = _msg(en, km)
            ans = QMessageBox.question(
                self, tr("poster.replace", "Replace title — same style"),
                tr("poster.ai_failed", "The AI could not redesign the title:") + "\n\n"
                + msg + "\n\n"
                + tr("poster.ai_font_q", "Draw the title with the app's Khmer fonts in the "
                                         "same style instead?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if ans == QMessageBox.StandardButton.Yes:
                self._replace_with_font()
            else:
                self.poster_status.setText(msg)

        self._run_task(
            self.replace_row, _ai_replace_fn, img, text, self.ctx.ai, lang, boxes_data,
            source, on_ok=done, on_fail=failed,
            running_text=tr("poster.ai_designing", "The AI is redesigning the title…"))

    def _replace_with_font(self):
        """The title in the title box put in place of the ORIGINAL poster's
        title with the app's Khmer fonts, in that title's own art style: always
        worked out from the picture as loaded (also after the automatic
        translation, whose picture has no title any more — before, the
        title was looked for there, not found, and drawn in the gold brush
        style). The title the automatic translation found is used again;
        a box you drew is the title. No title on the poster: gold brush."""
        got = self._replace_inputs()
        if got is None:
            return
        # a box drawn by hand is the title itself; else the title the automatic
        # translation found; else it is looked for on the original picture
        img, text, boxes_data, drawn = got

        def done(res):
            self._push_snapshot()
            self._ai_designed = False
            found = bool(res.get("found"))
            if found:
                self._work_path = res["clean"]
                self.canvas.set_image(QPixmap(self._work_path))
                laid = str(res.get("text") or "")
                if laid and laid != self.title_edit.text():
                    self.title_edit.setText(laid)
            self._boxes = []
            self._selected_box = -1
            self.canvas.set_boxes([])
            try:
                self._spec_dict = dataclasses.asdict(res["spec"])
            except Exception:
                self._spec_dict = self._default_spec_dict()
            self._apply_spec_to_controls()
            self._rendered_path = None
            self._schedule_render()
            self.poster_status.setText(
                tr("poster.replaced", "Old title removed: your title has its look (colours, "
                                      "outline, glow, ink band, angle and size). Change "
                                      "anything on the right; drag on the picture to move it.")
                if found else
                tr("poster.no_title", "No title found on this poster: your title was added in "
                                      "the gold brush style. Drag on the picture to move it."))

        self._run_task(
            self.replace_row, _replace_title_fn, img, text, boxes_data, drawn,
            bool(boxes_data and not drawn), on_ok=done,
            running_text=tr("poster.replacing", "Replacing the title…"))

    # ------------------------------------------------------- automatic title

    def _auto_lang(self) -> str:
        return str(self.poster_lang.currentData() or "km")

    @staticmethod
    def _check_lines(checks, problems_only: bool = False) -> list[str]:
        """The checks before saving, one short line each (✓ / ⚠); those
        that could not be made are left out."""
        names = {
            "spelling": (tr("poster.chk.spelling", "spelled as written"),
                         tr("poster.chk.spelling_bad", "the AI wrote «{detail}»: press "
                                                       "Replace title again for a new try")),
            "removed": (tr("poster.chk.removed", "old title removed"),
                        tr("poster.chk.removed_bad", "the old title may still show: {detail}")),
            "letters": (tr("poster.chk.letters", "letters right"),
                        tr("poster.chk.letters_bad", "letters to check: {detail}")),
            "numbers": (tr("poster.chk.numbers", "numbers kept"),
                        tr("poster.chk.numbers_bad", "numbers missing: {detail}")),
            "shaping": (tr("poster.chk.shaping", "Khmer subscripts drawn right"),
                        tr("poster.chk.shaping_bad",
                           "plain text layout: Khmer subscripts may look wrong")),
            "size": (tr("poster.chk.size", "same size as the original ({detail})"),
                     tr("poster.chk.size_bad", "the size changed ({detail})")),
        }
        out = []
        for key, ok, detail in checks or []:
            if ok is None or (problems_only and ok):
                continue
            good, bad = names.get(key, (key, key))
            out.append(("✓ " + good if ok else "⚠ " + bad).format(detail=detail))
        return out

    @staticmethod
    def _auto_status(r: dict) -> str:
        """Why a picture's title was not translated."""
        st = r.get("status")
        if st == "none":
            return tr("poster.auto_none", "No main title found on this picture.")
        if st == "already":
            return tr("poster.auto_already", "Its title is already in that language.")
        if st == "unreadable":
            return tr("poster.auto_unreadable",
                      "Its title could not be read: type the translation in the title box "
                      "and press Replace title — same style.")
        if st == "kept":
            return tr("poster.auto_kept", "translated before: kept as it was")
        if st == "error":
            return _msg(str(r.get("error") or ""), str(r.get("error_km") or ""))
        return ""

    def _auto_summary(self, r: dict, out_name: str) -> str:
        """What was read and written, the checks, and what to do next."""
        lines = [tr("poster.auto_read", "Read: «{source}» — {meaning}").format(
                     source=str(r.get("source") or "").replace(" | ", " / "),
                     meaning=str(r.get("meaning") or "")),
                 tr("poster.auto_wrote", "Translation: {text}").format(
                     text=str(r.get("text") or "").replace(" | ", " / "))]
        if not r.get("seen"):
            lines.append(tr("poster.auto_unseen",
                            "The AI model could not look at the picture: it worked from the "
                            "text reader's reading. Check the words."))
        checks = self._check_lines(r.get("checks"))
        if checks:
            lines.append(" · ".join(checks))
        lines.append(tr("poster.auto_next",
                        "Change the words or the look on the right, then Export PNG/JPG "
                        "(saved as {name}).").format(name=out_name))
        return "\n".join(lines)

    def _show_auto(self, r: dict, out_path: str) -> None:
        """A translated title in the editor: the picture without the old
        title, the translation drawn in its look (editable)."""
        self._push_snapshot()
        self._ai_designed = False
        self._work_path = r["clean"]
        self.canvas.set_image(QPixmap(self._work_path))
        self._boxes = []
        self._selected_box = -1
        self.canvas.set_boxes([])
        try:
            self._spec_dict = dataclasses.asdict(r["spec"])
        except Exception:
            self._spec_dict = self._default_spec_dict()
        self._apply_spec_to_controls()
        self.title_edit.setText(str(r.get("text") or ""))
        for rb in self.option_radios:
            rb.setVisible(False)
        self._rendered_path = None
        self._schedule_render()
        self._auto_out = out_path
        self._auto_boxes = list(r.get("boxes") or [])
        self._auto_source = str(r.get("source") or "")
        self.poster_status.setText(self._auto_summary(r, os.path.basename(out_path)))

    def _on_auto_title(self):
        """The loaded poster's main title translated and redrawn in its look."""
        img = self._require_image()
        if not img:
            return
        if self._image_path and os.path.exists(self._image_path):
            img = self._image_path          # the picture as loaded, with its title
        lang = self._auto_lang()
        from app.engines.poster_designer import output_name
        original = self._image_path or img
        out_path = os.path.join(os.path.dirname(original), output_name(original, lang))

        def done(r):
            if r.get("status") != "done":
                self.poster_status.setText(self._auto_status(r))
                return
            self._show_auto(r, out_path)

        self._run_task(self.auto_row, _auto_title_fn, img, self.ctx.ai, lang, on_ok=done,
                       running_text=tr("poster.auto_running",
                                       "Reading and translating the title…"))

    def _on_auto_titles(self):
        """Many pictures, each its own title, translation and look:
        <name>_KH.png in the folder chosen."""
        start = os.path.dirname(self._image_path) if self._image_path else ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("poster.auto_pick", "Choose the pictures"), start,
            "Images (*.jpg *.jpeg *.png *.webp)")
        if not paths:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, tr("poster.auto_out", "Save the translated pictures in…"),
            os.path.dirname(paths[0]))
        if not out_dir:
            return
        lang = self._auto_lang()
        from app.engines.poster_designer import output_name
        before = [p for p in paths if os.path.exists(os.path.join(out_dir, output_name(p, lang)))]
        overwrite = True
        if before:
            ans = QMessageBox.question(
                self, tr("poster.auto_batch", "Translate the titles of many pictures…"),
                tr("poster.auto_exists",
                   "{n} of these pictures were translated before (for example {name}). "
                   "Replace those files? (No keeps them as they are.)").format(
                    n=len(before), name=output_name(before[0], lang)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes)
            if ans == QMessageBox.StandardButton.Cancel:
                return
            overwrite = ans == QMessageBox.StandardButton.Yes
        self.auto_list.clear()
        self.auto_list.setVisible(True)

        def done(results):
            self.auto_list.clear()
            ok_n = 0
            for r in results:
                name = os.path.basename(str(r.get("path") or ""))
                st = r.get("status")
                if st == "done":
                    ok_n += 1
                    probs = [p[2:] for p in self._check_lines(r.get("checks"), True)]
                    text = (("⚠ " if probs else "✓ ") + f"{name} → {os.path.basename(r['out'])}"
                            + (": " + "; ".join(probs) if probs else ""))
                else:
                    text = ("✗ " if st == "error" else "– ") + f"{name}: {self._auto_status(r)}"
                item = QListWidgetItem(text)
                if st == "done":
                    item.setData(Qt.ItemDataRole.UserRole, r)
                    item.setToolTip(self._auto_summary(r, os.path.basename(r["out"])))
                self.auto_list.addItem(item)
            note = tr("poster.auto_batch_done",
                      "Translated {ok} of {n} pictures → {folder}. Click a picture in the list "
                      "to check or change it.").format(ok=ok_n, n=len(paths), folder=out_dir)
            if any(r.get("stopped") for r in results):
                note += "\n" + tr("poster.auto_stopped",
                                  "Stopped: no AI could answer (see the last line of the list).")
            self.poster_status.setText(note)

        self._run_task(self.auto_batch_row, _auto_titles_fn, list(paths), out_dir,
                       self.ctx.ai, lang, overwrite, on_ok=done,
                       running_text=tr("poster.auto_batch_running", "Translating the titles…"))

    def _on_auto_item(self, item) -> None:
        """A translated picture of the list in the editor, to check or change."""
        r = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not r or not os.path.exists(str(r.get("clean") or "")):
            return
        self._load_poster(r["path"])
        self._show_auto(r, r["out"])

    def _on_gold_preset(self):
        """The gold brush look for the title, sized for the picture (where an
        earlier title was placed, it stays)."""
        from PySide6.QtGui import QImageReader

        from app.engines import poster_designer as pd
        spec = pd.gold_brush_spec()
        if self._ai_designed:
            self._ai_designed = False
            if self._pre_ai_work and os.path.exists(self._pre_ai_work):
                self._work_path = self._pre_ai_work
                self.canvas.set_image(QPixmap(self._work_path))
        text = self.title_edit.text().strip()
        img = self._work_path
        if img and os.path.exists(img) and text:
            size = QImageReader(img).size()
            if size.isValid() and size.width() > 0:
                try:
                    spec.size_ratio = pd.fit_size_ratio(
                        text, spec, 0.70 * size.width(), 0.24 * size.height(),
                        size.width(), fonts_dir())
                    spec.glow_radius = pd.gold_glow_radius(spec.size_ratio, size.width())
                except Exception:
                    log.exception("could not size the title")
        if self._rendered_path and isinstance(self._spec_dict.get("pos"), (list, tuple)):
            spec.pos = tuple(self._spec_dict["pos"])
        self._push_snapshot()
        self._spec_dict = dataclasses.asdict(spec)
        self._apply_spec_to_controls()
        self._schedule_render()

    # -------------------------------------------------------------- translate

    def _source_title(self) -> str:
        """The original title to translate: as the automatic translation read
        it, else a title typed in another language than Khmer; "" = read it
        off the picture."""
        if self._auto_source:
            return self._auto_source.replace(" | ", " ")
        typed = self.title_edit.text().strip()
        if typed and not any("\u1780" <= ch <= "\u17ff" for ch in typed):
            return typed
        return ""

    def _on_translate_title(self):
        title = self._source_title()
        if not title and not self._require_image():
            return
        original = (self._image_path if self._image_path and os.path.exists(self._image_path)
                    else self._work_path or "")
        lang = str(self.poster_lang.currentData() or "km")

        def done(options):
            options = list(options or [])[:3]
            for i, rb in enumerate(self.option_radios):
                if i < len(options):
                    rb.setText(options[i])
                    rb.setVisible(True)
                else:
                    rb.setVisible(False)
            if options:
                self.option_radios[0].setChecked(True)
                self._push_snapshot()
                self.title_edit.setText(options[0])
                self._schedule_render()

        self._run_task(
            self.translate_row, _translate_title_fn, title, self.ctx.ai, lang, original,
            on_ok=done,
            running_text=tr("poster.translating", "Translating title…"))

    def _on_option_picked(self, idx: int):
        rb = self.option_radios[idx] if 0 <= idx < len(self.option_radios) else None
        if rb is None or not rb.isVisible():
            return
        self._push_snapshot()
        self.title_edit.setText(rb.text())
        self._schedule_render()

    def _on_title_edited(self):
        self._push_snapshot()
        self._schedule_render()

    # ---------------------------------------------------------------- design

    def _on_ai_design(self):
        img = self._require_image()
        if not img:
            return
        title = self.title_edit.text().strip() or self._source_title()

        def done(spec):
            self._push_snapshot()
            try:
                self._spec_dict = dataclasses.asdict(spec)
            except Exception:
                self._spec_dict = self._default_spec_dict()
            self._apply_spec_to_controls()
            self._schedule_render()

        self._run_task(
            self.design_row, _design_fn, img, title, self.ctx.ai,
            on_ok=done,
            running_text=tr("poster.designing", "Designing title style…"))

    def _apply_spec_to_controls(self):
        """Reflect self._spec_dict into the manual panel without feedback loops."""
        d = self._spec_dict
        widgets = [
            self.font_combo, self.size_slider, self.fill_btn, self.grad_check,
            self.grad_c1, self.grad_c2, self.stroke_btn, self.stroke_width,
            self.shadow_btn, self.shadow_dx, self.shadow_dy, self.shadow_blur,
            self.glow_check, self.glow_btn, self.glow_radius, self.bevel_check,
            self.extrude_depth, self.extrude_btn, self.spacing_spin,
            self.rotation_spin, self.backdrop_check, self.backdrop_btn,
            self.sparkle_check,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            font = str(d.get("font", "Moul"))
            if self.font_combo.findText(font) < 0:
                self.font_combo.addItem(font)
            self.font_combo.setCurrentText(font)
            self.size_slider.setValue(
                max(2, min(40, int(round(float(d.get("size_ratio", 0.12)) * 100)))))
            self.fill_btn.set_color(str(d.get("fill", "#ffffff")))
            grad = d.get("gradient")
            if isinstance(grad, (list, tuple)) and len(grad) >= 2:
                self.grad_check.setChecked(True)
                self.grad_c1.set_color(str(grad[0]))        # top
                self.grad_c2.set_color(str(grad[-1]))       # bottom (middle stops kept)
            else:
                self.grad_check.setChecked(False)
            self.stroke_btn.set_color(str(d.get("stroke_color", "#000000")))
            self.stroke_width.setValue(
                max(0, min(25, int(round(float(d.get("stroke_width_ratio", 0.04)) * 100)))))
            self.shadow_btn.set_color(str(d.get("shadow_color", "#000000cc")))
            off = d.get("shadow_offset", (4, 4))
            if isinstance(off, (list, tuple)) and len(off) >= 2:
                self.shadow_dx.setValue(int(off[0]))
                self.shadow_dy.setValue(int(off[1]))
            self.shadow_blur.setValue(int(d.get("shadow_blur", 8)))
            glow = d.get("glow_color")
            self.glow_check.setChecked(bool(glow))
            if glow:
                self.glow_btn.set_color(str(glow))
            self.glow_radius.setValue(int(d.get("glow_radius", 0)) or 12)
            self.bevel_check.setChecked(bool(d.get("bevel", False)))
            self.extrude_depth.setValue(int(d.get("extrude_depth", 0)))
            self.extrude_btn.set_color(str(d.get("extrude_color", "#222222")))
            self.spacing_spin.setValue(int(d.get("letter_spacing", 0)))
            self.rotation_spin.setValue(float(d.get("rotation", 0.0)))
            self.backdrop_check.setChecked(bool(d.get("backdrop", False)))
            self.backdrop_btn.set_color(str(d.get("backdrop_color") or "#0a0705eb"))
            self.sparkle_check.setChecked(int(d.get("sparkles", 0) or 0) > 0)
        except Exception:
            log.exception("could not apply spec to controls")
        finally:
            for w in widgets:
                w.blockSignals(False)

    # ------------------------------------------------------------- spec edits

    def _spec_change(self, key: str, value):
        self._push_snapshot()
        self._spec_dict[key] = value
        self._schedule_render()

    def _on_gradient_changed(self, *_a):
        self._push_snapshot()
        if self.grad_check.isChecked():
            # the top and bottom colours change; a metallic gradient's middle
            # stops (gold preset, a matched title) stay
            old = self._spec_dict.get("gradient")
            mid = list(old[1:-1]) if isinstance(old, (list, tuple)) and len(old) > 2 else []
            self._spec_dict["gradient"] = [self.grad_c1.color(), *mid, self.grad_c2.color()]
        else:
            self._spec_dict["gradient"] = None
        self._schedule_render()

    def _on_glow_changed(self, *_a):
        self._push_snapshot()
        if self.glow_check.isChecked():
            self._spec_dict["glow_color"] = self.glow_btn.color()
            self._spec_dict["glow_radius"] = int(self.glow_radius.value())
        else:
            self._spec_dict["glow_color"] = None
            self._spec_dict["glow_radius"] = 0
        self._schedule_render()

    def _on_shadow_offset(self, *_a):
        self._push_snapshot()
        self._spec_dict["shadow_offset"] = (
            int(self.shadow_dx.value()), int(self.shadow_dy.value()))
        self._schedule_render()

    def _on_pos_dragged(self, fx: float, fy: float):
        self._spec_dict["pos"] = (round(float(fx), 4), round(float(fy), 4))
        self._schedule_render()

    def _on_drag_finished(self):
        self._push_snapshot()

    # --------------------------------------------------------------- snapshot

    def _snapshot(self) -> tuple[str, str]:
        return json.dumps(self._spec_dict), self.title_edit.text()

    def _push_snapshot(self):
        snap = self._snapshot()
        if self._undo and self._undo[-1] == snap:
            return
        self._undo.append(snap)
        if len(self._undo) > 100:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, snap: tuple[str, str]):
        spec_json, text = snap
        try:
            self._spec_dict = json.loads(spec_json)
        except Exception:
            self._spec_dict = self._default_spec_dict()
        self.title_edit.blockSignals(True)
        self.title_edit.setText(text)
        self.title_edit.blockSignals(False)
        self._apply_spec_to_controls()
        self._schedule_render()

    def _on_undo(self):
        if not self._undo:
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())

    def _on_redo(self):
        if not self._redo:
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())

    # ----------------------------------------------------------------- render

    def _schedule_render(self):
        self._render_timer.start()

    def _start_render(self):
        img = self._work_path
        # the AI lettered the title into the picture: no second title on it
        text = "" if self._ai_designed else self.title_edit.text().strip()
        tag = self.tag_edit.text().strip()
        if not img or not os.path.exists(img):
            return
        if not text and not tag:
            if self._rendered_path and not self._render_running:
                self._rendered_path = None          # nothing on it: the picture as it is
                self.canvas.set_image(QPixmap(img))
            return
        if self._render_running:
            self._render_pending = True
            return
        self._render_running = True

        def done(out_path):
            self._render_running = False
            self._rendered_path = out_path
            pix = QPixmap(out_path)
            if not pix.isNull():
                self.canvas.set_image(pix)
            if self._render_pending:
                self._render_pending = False
                self._schedule_render()

        def failed(en, km):
            self._render_running = False
            self.poster_status.setText(
                tr("poster.render_failed", "Render failed:") + " " + en)
            log.error("render_title failed: %s", en)
            if self._render_pending:
                self._render_pending = False
                self._schedule_render()

        self._run_task(
            self.render_row, _render_title_fn, img, text, dict(self._spec_dict), tag,
            on_ok=done, on_fail=failed,
            running_text=tr("poster.rendering", "Rendering title…"))

    # ---------------------------------------------------------------- exports

    def _export_source(self) -> str | None:
        p = self._rendered_path or self._work_path
        if not p or not os.path.exists(p):
            QMessageBox.information(
                self, tr("poster.title", "Poster"),
                _msg("Nothing to export yet.", "មិនទាន់មានអ្វីត្រូវនាំចេញទេ។"))
            return None
        return p

    def _on_export_image(self):
        src = self._export_source()
        if not src:
            return
        if self._auto_out:
            # a translated title: <name>_KH.png beside the picture (or where
            # the batch saved it)
            start = self._auto_out
        elif self._pv_from_strip and self._pv_video:
            # a picture of the video: named after the video, saved beside it
            start = os.path.splitext(self._pv_video)[0] + ".thumb.jpg"
        else:
            start = os.path.splitext(os.path.basename(self._image_path or src))[0] + "_titled.png"
        out, _ = QFileDialog.getSaveFileName(
            self, tr("poster.export_png", "Export PNG/JPG"),
            start, "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not out:
            return
        try:
            if out.lower().endswith((".jpg", ".jpeg")):
                from PIL import Image
                Image.open(src).convert("RGB").save(out, quality=95)
            else:
                shutil.copyfile(src, out)
            self.poster_status.setText(
                tr("poster.exported", "Exported:") + " " + out)
        except Exception as e:
            log.exception("poster export failed")
            QMessageBox.warning(self, tr("err.title", "Error"), str(e))

    def _on_export_variants(self):
        src = self._export_source()
        if not src:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, tr("poster.export_variants", "Export 9:16 && 16:9").replace("&&", "&"))
        if not out_dir:
            return
        base = os.path.splitext(os.path.basename(self._image_path or src))[0]

        def done(paths):
            self.poster_status.setText(
                tr("poster.exported", "Exported:") + " "
                + ", ".join(str(v) for v in (paths or {}).values()))

        self._run_task(
            self.variants_row, _export_variants_fn, src, out_dir, base,
            on_ok=done,
            running_text=tr("poster.exporting", "Exporting variants…"))

    # -------------------------------------------------------------- templates

    def _templates(self) -> dict:
        try:
            d = self.ctx.config.get(TEMPLATES_KEY, {})
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _on_save_template(self):
        name, ok = QInputDialog.getText(
            self, tr("poster.save_tpl", "Save style template"),
            tr("poster.tpl_name", "Template name:"))
        if not ok or not name.strip():
            return
        tpls = self._templates()
        tpls[name.strip()] = {
            "spec": dict(self._spec_dict),
            "text": self.title_edit.text(),
        }
        try:
            self.ctx.config.set(TEMPLATES_KEY, tpls)
            self.poster_status.setText(
                tr("poster.tpl_saved", "Template saved:") + " " + name.strip())
        except Exception as e:
            log.exception("template save failed")
            QMessageBox.warning(self, tr("err.title", "Error"), str(e))

    def _on_apply_template_folder(self):
        tpls = self._templates()
        if not tpls:
            QMessageBox.information(
                self, tr("poster.title", "Poster"),
                _msg("No saved templates yet — use 'Save style template' first.",
                     "មិនទាន់មានគំរូរក្សាទុកទេ — សូមប្រើ 'Save style template' ជាមុនសិន។"))
            return
        names = sorted(tpls.keys())
        name, ok = QInputDialog.getItem(
            self, tr("poster.apply_tpl", "Apply template to folder…"),
            tr("poster.pick_tpl", "Template:"), names, 0, False)
        if not ok or not name:
            return
        folder = QFileDialog.getExistingDirectory(
            self, tr("poster.pick_folder", "Choose folder of poster images"))
        if not folder:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, tr("poster.pick_out", "Choose output folder"))
        if not out_dir:
            return
        tpl = tpls.get(name) or {}
        spec = tpl.get("spec") or dict(self._spec_dict)
        text = str(tpl.get("text") or self.title_edit.text())
        auto = self.auto_per_check.isChecked()
        lang = str(self.poster_lang.currentData() or "km")
        if not auto and not text:
            QMessageBox.information(
                self, tr("poster.title", "Poster"),
                _msg("The template has no title text; enable 'Auto per poster' "
                     "or set a title.",
                     "គំរូនេះគ្មានអត្ថបទចំណងជើងទេ — សូមបើក 'Auto per poster' "
                     "ឬកំណត់ចំណងជើង។"))
            return
        self.batch_list.clear()
        self.batch_list.setVisible(True)

        def done(results):
            self.batch_list.clear()
            ok_n = 0
            for img, out, err in results:
                if out:
                    ok_n += 1
                    self.batch_list.addItem("✓ " + os.path.basename(img))
                else:
                    self.batch_list.addItem(
                        "✗ " + os.path.basename(img) + " — " + err)
            self.poster_status.setText(
                tr("poster.batch_done", "Batch finished") +
                f": {ok_n}/{len(results)} → {out_dir}")

        self._run_task(
            self.batch_row, _batch_posters_fn, self.ctx.ai, folder, spec, text,
            lang, auto, out_dir,
            on_ok=done,
            running_text=tr("poster.batch_running", "Batch running…"))
