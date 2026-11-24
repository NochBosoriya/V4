"""Scan card of the Download tab.

What a scanned link holds: one video, or a playlist / series whose videos
show as episode cards (poster + length), with the quality, episode-range
shortcuts, the save folder and the Download button. The card is a view:
DlTab runs the scans and starts the downloads.
"""
from __future__ import annotations

import html
import queue
import threading
from urllib.parse import urlsplit

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, QRectF, QSize, Qt,
                            QTimer, Signal)
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QKeySequence, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QListView, QProgressBar, QPushButton,
                               QSizePolicy, QSpinBox, QStackedWidget, QStyle,
                               QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget)

from app.core.downloader import (ScanEntry, ScanResult, apply_details, fetch_entry_details,
                                 fetch_image, prune_thumb_cache, url_key)
from app.core.i18n import tr
from app.core.logger import get_logger
from app.ui import theme

log = get_logger("ui.dl_card")

QUALITY_CHOICES = (
    ("best", "dl.q.best", "Best available"),
    ("1080", "dl.q.1080", "1080p (Full HD)"),
    ("720", "dl.q.720", "720p (HD)"),
    ("480", "dl.q.480", "480p"),
    ("360", "dl.q.360", "360p"),
)
_Q_HEIGHT = {"1080": 1080, "720": 720, "480": 480, "360": 360}

BLUE = "#388bfd"
BAR_BG = "#111923"
CARD_BG = "#0f141b"
MIN_CARD_W, MAX_CARD_W, SPACING = 176, 236, 10
PAD, CAPTION_H = 6, 26


def fmt_duration(sec) -> str:
    try:
        s = int(float(sec or 0))
    except (TypeError, ValueError):
        return ""
    if s <= 0:
        return ""
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def fmt_size(n: float) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{max(1, int(n // 1024))} KB"


def _card_image_url(url: str) -> str:
    """A card-sized copy where the image host makes one (Bilibili's CDN
    resizes on request), so a 64-episode list does not fetch 64 full posters."""
    parts = urlsplit(url)
    if parts.netloc.endswith("hdslb.com") and "@" not in parts.path.rsplit("/", 1)[-1]:
        return url + "@336w_189h_1c.jpg"
    return url


def compose_poster(img: QImage, size: QSize, dpr: float) -> QPixmap:
    """The poster filling a 16:9 box: cropped when it is close to 16:9, else
    whole (portrait short-drama posters) over a blurred, dimmed copy."""
    w, h = max(1, round(size.width() * dpr)), max(1, round(size.height() * dpr))
    out = QPixmap(w, h)
    out.fill(QColor(theme.BG))
    p = QPainter(out)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    ratio, box = img.width() / max(1, img.height()), w / h
    if abs(ratio - box) / box < 0.12:
        s = img.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        p.drawImage((w - s.width()) // 2, (h - s.height()) // 2, s)
    else:
        tiny = img.scaled(32, 18, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        tiny = tiny.copy((tiny.width() - 32) // 2, (tiny.height() - 18) // 2, 32, 18)
        p.drawImage(0, 0, tiny.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        p.fillRect(0, 0, w, h, QColor(0, 0, 0, 110))
        fg = img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p.drawImage((w - fg.width()) // 2, (h - fg.height()) // 2, fg)
    p.end()
    out.setDevicePixelRatio(dpr)
    return out


def _draw_badge(p: QPainter, rect: QRectF, text: str, font: QFont,
                bg: QColor, fg: QColor = QColor("#ffffff")) -> None:
    path = QPainterPath()
    path.addRoundedRect(rect, 4, 4)
    p.fillPath(path, bg)
    p.setFont(font)
    p.setPen(fg)
    p.drawText(rect, Qt.AlignCenter, text)


class _DaemonPool:
    """Tiny worker pool on daemon threads: a poster request stuck on the
    network can never keep the app from closing."""

    def __init__(self, workers: int, name: str):
        self._q: queue.Queue = queue.Queue()
        for i in range(workers):
            threading.Thread(target=self._run, name=f"{name}-{i}", daemon=True).start()

    def submit(self, fn, *args) -> None:
        self._q.put((fn, args))

    def clear(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _run(self) -> None:
        while True:
            fn, args = self._q.get()
            try:
                fn(*args)
            except Exception:
                log.exception("card loader task failed")


class _Loader(QObject):
    """Poster images and missing episode details, read on worker threads;
    results come back on the UI thread through the signals."""

    image = Signal(str, QImage)          # url, image (at most 480 px wide)
    details = Signal(int, object)        # generation, (url, details dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images = _DaemonPool(6, "poster")
        self._infos = _DaemonPool(3, "details")
        self._asked: set[str] = set()
        self._lock = threading.Lock()
        self.cookies = ""
        self.gen = 0
        self._images.submit(prune_thumb_cache)

    def reset(self) -> None:
        """A new scan: results still on the way for the old one are dropped."""
        self.gen += 1
        self._images.clear()
        self._infos.clear()
        with self._lock:
            self._asked.clear()

    def _first_ask(self, key: str) -> bool:
        with self._lock:
            if key in self._asked:
                return False
            self._asked.add(key)
            return True

    def want_image(self, url: str, referer: str = "") -> None:
        if url and self._first_ask("img:" + url):
            self._images.submit(self._load_image, url, referer)

    def want_details(self, url: str) -> None:
        if url and self._first_ask("info:" + url):
            self._infos.submit(self._load_details, url, self.gen)

    def _load_image(self, url: str, referer: str) -> None:
        img = QImage()
        for u in dict.fromkeys((_card_image_url(url), url)):
            try:
                if img.loadFromData(fetch_image(u, self.cookies, referer)):
                    break
            except Exception as exc:
                log.debug("poster %s: %s", u, exc)
        if img.isNull():
            return                       # the card keeps its placeholder
        if img.width() > 480:
            img = img.scaledToWidth(480, Qt.SmoothTransformation)
        self.image.emit(url, img)

    def _load_details(self, url: str, gen: int) -> None:
        if gen != self.gen:
            return
        try:
            d = fetch_entry_details(url, self.cookies)
        except Exception as exc:
            log.debug("details %s: %s", url, exc)
            return
        self.details.emit(gen, (url, d))

    def shutdown(self) -> None:
        self.gen += 1
        self._images.clear()
        self._infos.clear()


class EpisodeModel(QAbstractListModel):
    """The scanned entries; `checked` holds the rows chosen for download."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: list[ScanEntry] = []
        self.keys: list[str] = []
        self.checked: set[int] = set()
        self.marks: dict[str, str] = {}      # url key -> "list" | "done"

    def set_entries(self, entries: list[ScanEntry]) -> None:
        self.beginResetModel()
        self.entries = list(entries)
        self.keys = [url_key(e.url) for e in self.entries]
        self.checked = set()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.entries)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.entries):
            return None
        e = self.entries[index.row()]
        if role == Qt.DisplayRole:
            return e.title
        if role == Qt.ToolTipRole:
            return "\n".join(x for x in (e.title or e.url, fmt_duration(e.duration)) if x)
        return None

    def changed(self, rows=None) -> None:
        if not self.entries:
            return
        rows = [r for r in (rows if rows is not None else (0, len(self.entries) - 1))
                if 0 <= r < len(self.entries)]
        if rows:
            self.dataChanged.emit(self.index(min(rows)), self.index(max(rows)))

    def set_rows(self, rows, on: bool) -> None:
        rows = list(rows)
        if on:
            self.checked.update(rows)
        else:
            self.checked.difference_update(rows)
        self.changed(rows)

    def select_only(self, rows) -> None:
        self.checked = {r for r in rows if 0 <= r < len(self.entries)}
        self.changed()


class _CardDelegate(QStyledItemDelegate):
    """Paints one episode card: poster with its length, EP label and title,
    blue border + check when chosen, tags for the link's own video / already
    in the list / saved."""

    def __init__(self, card: "ScanCard"):
        super().__init__(card)
        self._card = card
        self.card_w = 196
        self._pix: dict[str, QPixmap] = {}

    def set_card_width(self, w: int) -> bool:
        if w == self.card_w:
            return False
        self.card_w = w
        self._pix.clear()
        return True

    def clear(self) -> None:
        self._pix.clear()

    def thumb_size(self) -> QSize:
        tw = self.card_w - 2 * PAD
        return QSize(tw, round(tw * 9 / 16))

    def sizeHint(self, option, index) -> QSize:
        return QSize(self.card_w, self.thumb_size().height() + 2 * PAD + CAPTION_H)

    def _poster(self, url: str) -> QPixmap | None:
        pix = self._pix.get(url)
        if pix is not None:
            return pix
        img = self._card.images.get(url)
        if img is None:
            self._card.loader.want_image(url)
            return None
        if len(self._pix) > 600:
            self._pix.clear()
        pix = self._pix[url] = compose_poster(img, self.thumb_size(), self._card.devicePixelRatioF())
        return pix

    def paint(self, p: QPainter, opt, index) -> None:
        card, m = self._card, self._card.model
        row = index.row()
        if row >= len(m.entries):
            return
        e = m.entries[row]
        if not e.thumb or not e.title:
            card.loader.want_details(e.url)          # the list page did not say
        checked = row in m.checked
        hover = bool(opt.state & QStyle.State_MouseOver)
        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        outline = QPainterPath()
        outline.addRoundedRect(r, 9, 9)
        p.fillPath(outline, QColor("#16233a") if checked else QColor(CARD_BG))

        ts = self.thumb_size()
        tr_ = QRectF(r.x() + PAD - 0.5, r.y() + PAD - 0.5, ts.width(), ts.height())
        clip = QPainterPath()
        clip.addRoundedRect(tr_, 6, 6)
        p.save()
        p.setClipPath(clip)
        pix = self._poster(e.thumb) if e.thumb else None
        if pix is not None:
            p.drawPixmap(tr_.toRect(), pix)
        else:
            g = QLinearGradient(tr_.topLeft(), tr_.bottomRight())
            g.setColorAt(0, QColor("#1c2633"))
            g.setColorAt(1, QColor("#121820"))
            p.fillRect(tr_, g)
            big = QFont(opt.font)
            big.setPointSizeF(big.pointSizeF() + 5)
            big.setBold(True)
            p.setFont(big)
            p.setPen(QColor("#3b4654"))
            p.drawText(tr_, Qt.AlignCenter, card.ep_label(e))
        if checked:
            p.fillRect(tr_, QColor(31, 111, 235, 56))
        p.restore()

        small = QFont(opt.font)
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.5))
        small.setBold(True)
        fm = QFontMetrics(small)
        if e.duration:                                   # length, bottom right
            text = fmt_duration(e.duration)
            bw, bh = fm.horizontalAdvance(text) + 10, fm.height() + 2
            _draw_badge(p, QRectF(tr_.right() - bw - 5, tr_.bottom() - bh - 5, bw, bh),
                        text, small, QColor(0, 0, 0, 190))
        tags = []
        if e.matched and card.is_playlist:
            tags.append((tr("dl.ep.this", "This link"), QColor(theme.SELECTION)))
        mark = m.marks.get(m.keys[row]) if row < len(m.keys) else None
        if mark == "done":
            tags.append((tr("dl.ep.saved", "Saved"), QColor("#238636")))
        elif mark == "list":
            tags.append((tr("dl.ep.inlist", "In list"), QColor(80, 90, 104, 230)))
        x = tr_.x() + 5
        for text, color in tags:                         # tags, top left
            bw, bh = fm.horizontalAdvance(text) + 10, fm.height() + 2
            _draw_badge(p, QRectF(x, tr_.y() + 5, bw, bh), text, small, color)
            x += bw + 4

        cr = QRectF(tr_.right() - 27, tr_.y() + 5, 22, 22)   # check, top right
        if checked:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.SELECTION))
            p.drawEllipse(cr)
            p.setPen(QPen(QColor("#ffffff"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            tick = QPainterPath()
            tick.moveTo(cr.x() + 6, cr.y() + 11.5)
            tick.lineTo(cr.x() + 9.5, cr.y() + 15)
            tick.lineTo(cr.x() + 16, cr.y() + 8)
            p.drawPath(tick)
        elif hover:
            p.setPen(QPen(QColor(255, 255, 255, 210), 1.6))
            p.setBrush(QColor(0, 0, 0, 100))
            p.drawEllipse(cr)

        cap = QRectF(r.x() + PAD, tr_.bottom() + 2, r.width() - 2 * PAD, CAPTION_H - 2)
        label = card.ep_label(e)
        bold = QFont(opt.font)
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(QColor(theme.ACCENT if checked else theme.TEXT))
        p.drawText(cap, Qt.AlignVCenter | Qt.AlignLeft, label)
        rest = cap.adjusted(QFontMetrics(bold).horizontalAdvance(label) + 8, 0, 0, 0)
        if rest.width() > 28 and e.title:
            p.setFont(opt.font)
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(rest, Qt.AlignVCenter | Qt.AlignLeft,
                       QFontMetrics(opt.font).elidedText(e.title, Qt.ElideRight, int(rest.width())))

        p.setBrush(Qt.NoBrush)
        if checked:
            p.setPen(QPen(QColor(theme.ACCENT), 2))
        elif hover:
            p.setPen(QPen(QColor(theme.TEXT_DIM), 1))
        else:
            p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawPath(outline)
        p.restore()


class EpisodeGrid(QListView):
    """Episode cards in a grid that fills its width. Click toggles a card,
    drag paints the same choice over more cards, Shift+click picks a range,
    Ctrl+A / Space work from the keyboard."""

    toggled = Signal()

    def __init__(self, card: "ScanCard"):
        super().__init__(card)
        self._card = card
        self.setObjectName("dlGrid")
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setUniformItemSizes(True)
        self.setSpacing(SPACING)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setMouseTracking(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(28)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.StrongFocus)
        self._anchor = -1
        self._paint_on: bool | None = None

    def _row_at(self, ev) -> int:
        idx = self.indexAt(ev.position().toPoint())
        return idx.row() if idx.isValid() else -1

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        self.setFocus()
        row = self._row_at(ev)
        if row < 0:
            return
        m = self.model()
        if ev.modifiers() & Qt.ShiftModifier and self._anchor >= 0:
            lo, hi = sorted((self._anchor, row))
            m.set_rows(range(lo, hi + 1), True)
        else:
            on = row not in m.checked
            m.set_rows([row], on)
            self._anchor, self._paint_on = row, on
        self.setCurrentIndex(m.index(row))
        self.toggled.emit()

    def mouseMoveEvent(self, ev) -> None:
        if self._paint_on is not None and ev.buttons() & Qt.LeftButton:
            row = self._row_at(ev)
            m = self.model()
            if row >= 0 and (row in m.checked) != self._paint_on:
                m.set_rows([row], self._paint_on)
                self.toggled.emit()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        self._paint_on = None
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        self.mousePressEvent(ev)       # a fast second click toggles again

    def keyPressEvent(self, ev) -> None:
        m = self.model()
        if ev.matches(QKeySequence.SelectAll):
            m.select_only(range(len(m.entries)))
            self.toggled.emit()
            return
        if ev.key() == Qt.Key_Space and self.currentIndex().isValid():
            row = self.currentIndex().row()
            m.set_rows([row], row not in m.checked)
            self.toggled.emit()
            return
        super().keyPressEvent(ev)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.relayout()

    def relayout(self) -> None:
        w = self.viewport().width()
        cols = max(1, (w - SPACING) // (MIN_CARD_W + SPACING))
        card_w = max(MIN_CARD_W, min(MAX_CARD_W, (w - SPACING * (cols + 1)) // cols))
        delegate = self.itemDelegate()
        if isinstance(delegate, _CardDelegate) and delegate.set_card_width(card_w):
            self.scheduleDelayedItemsLayout()


_QSS = f"""
QFrame#dlCard {{ background: {theme.PANEL}; border: 1px solid {BLUE}; border-radius: 12px; }}
QFrame#dlCardHeader {{
    border: none; border-top-left-radius: 11px; border-top-right-radius: 11px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2b6fe3, stop:1 #163a85);
}}
QFrame#dlCardHeader QLabel {{ background: transparent; color: #ffffff; }}
QLabel#dlBadge {{
    background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.55);
    border-radius: 9px; padding: 1px 9px; font-size: 7.5pt; font-weight: 700;
}}
QLabel#dlEst {{ color: rgba(255,255,255,0.80); font-size: 8.5pt; }}
QLabel#dlQLabel {{ color: rgba(255,255,255,0.85); font-size: 8pt; font-weight: 700; }}
QComboBox#dlQuality {{
    background: rgba(13,17,23,0.35); color: #ffffff; border: 1px solid rgba(255,255,255,0.45);
    border-radius: 8px; padding: 4px 10px; min-width: 190px; font-weight: 600;
}}
QComboBox#dlQuality:hover {{ border-color: #ffffff; }}
QComboBox#dlQuality QAbstractItemView {{
    background: {theme.PANEL}; color: {theme.TEXT}; selection-background-color: {theme.SELECTION};
}}
QLineEdit#dlTitle {{
    background: transparent; color: #ffffff; border: 1px solid transparent; border-radius: 6px;
    padding: 0px 4px; font-size: 15pt; font-weight: 700;
}}
QLineEdit#dlTitle:hover {{ border-color: rgba(255,255,255,0.35); }}
QLineEdit#dlTitle:focus {{ border-color: rgba(255,255,255,0.75); background: rgba(13,17,23,0.25); }}
QLabel#dlSub {{ color: rgba(255,255,255,0.82); font-size: 9pt; padding-left: 5px; }}
QFrame#dlSelBar {{ background: {BAR_BG}; border: none; border-bottom: 1px solid {theme.BORDER}; }}
QFrame#dlSelBar QLabel {{ background: transparent; }}
QLabel#dlSelTitle {{ color: {theme.ACCENT}; font-weight: 700; }}
QFrame#dlPill {{ border: 1px solid {BLUE}; border-radius: 8px; background: transparent; }}
QToolButton#dlPillBtn {{
    color: {theme.ACCENT}; background: transparent; border: none; padding: 4px 8px; font-weight: 600;
}}
QToolButton#dlPillBtn:hover {{ background: rgba(56,139,253,0.16); border-radius: 7px; }}
QSpinBox#dlPillSpin {{ background: {theme.BG}; min-width: 56px; padding: 2px 4px; }}
QPushButton#dlOutline {{
    color: {theme.ACCENT}; background: transparent; border: 1px solid {BLUE};
    border-radius: 8px; padding: 5px 14px; font-weight: 600;
}}
QPushButton#dlOutline:hover {{ background: rgba(56,139,253,0.16); }}
QPushButton#dlGhost {{
    color: {theme.TEXT_DIM}; background: transparent; border: 1px solid {theme.BORDER};
    border-radius: 8px; padding: 5px 14px;
}}
QPushButton#dlGhost:hover {{ color: {theme.TEXT}; border-color: {theme.TEXT_DIM}; }}
QListView#dlGrid {{ background: {theme.PANEL}; border: none; border-radius: 0px; }}
QFrame#dlCardFooter {{
    background: {BAR_BG}; border: none; border-top: 1px solid {theme.BORDER};
    border-bottom-left-radius: 11px; border-bottom-right-radius: 11px;
}}
QFrame#dlCardFooter QLabel, QFrame#dlCardFooter QCheckBox {{ background: transparent; }}
QLabel#dlFolder {{ color: {theme.TEXT}; }}
QLabel#dlNote {{ color: #3fb950; font-weight: 600; }}
QPushButton#dlLink {{
    color: {theme.ACCENT}; background: transparent; border: none; padding: 2px 4px; font-weight: 600;
}}
QPushButton#dlLink:hover {{ text-decoration: underline; }}
QPushButton#dlDownload {{ padding: 8px 22px; font-size: 10.5pt; border-radius: 8px; min-width: 210px; }}
QLabel#dlEmptyIcon {{ color: {theme.ACCENT}; font-size: 30pt; background: transparent; }}
QLabel#dlEmptyText {{ color: {theme.TEXT}; font-size: 11pt; font-weight: 600; background: transparent; }}
QLabel#dlHint {{ color: {theme.TEXT_DIM}; background: transparent; }}
QLabel#dlError {{ color: #f85149; font-weight: 600; background: transparent; }}
QLabel#dlPoster {{ background: transparent; }}
"""


class ScanCard(QFrame):
    """The card: empty / scanning / error states, a playlist's episode grid
    or one video's poster, quality, ranges, save folder and Download."""

    download_clicked = Signal()
    change_folder = Signal()
    ask_toggled = Signal(bool)
    poster_toggled = Signal(bool)
    quality_changed = Signal(str)
    cancel_clicked = Signal()
    retry_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dlCard")
        self.setStyleSheet(_QSS)
        self.result: ScanResult | None = None
        self.images: dict[str, QImage] = {}
        self.sizes: dict[str, int] = {}       # quality key -> bytes of one video
        self.size_duration = 0.0              # length of the video `sizes` is for
        self.heights: list[int] = []
        self._lo = self._hi = 1
        self._user_quality = True
        self.loader = _Loader(self)
        self.loader.image.connect(self._on_image)
        self.loader.details.connect(self._on_details)
        self.model = EpisodeModel(self)
        self._note_timer = QTimer(self)
        self._note_timer.setSingleShot(True)
        self._note_timer.timeout.connect(lambda: self.note_label.setText(""))

        root = QVBoxLayout(self)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._build_empty())        # 0
        self.stack.addWidget(self._build_busy())         # 1
        self.stack.addWidget(self._build_error())        # 2
        self.stack.addWidget(self._build_playlist())     # 3
        self.stack.addWidget(self._build_single())       # 4
        root.addWidget(self.stack, 1)
        root.addWidget(self._build_footer())
        self.show_empty()

    # -- building -------------------------------------------------------------
    def _build_header(self) -> QFrame:
        self.header = QFrame(self)
        self.header.setObjectName("dlCardHeader")
        lay = QVBoxLayout(self.header)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.badge = QLabel("", self.header)
        self.badge.setObjectName("dlBadge")
        self.est = QLabel("", self.header)
        self.est.setObjectName("dlEst")
        top.addWidget(self.badge)
        top.addWidget(self.est)
        top.addStretch(1)
        q = QLabel(tr("dl.quality.label", "QUALITY:"), self.header)
        q.setObjectName("dlQLabel")
        self.quality = QComboBox(self.header)
        self.quality.setObjectName("dlQuality")
        for key, k, default in QUALITY_CHOICES:
            self.quality.addItem(tr(k, default), key)
        self.quality.currentIndexChanged.connect(self._on_quality)
        top.addWidget(q)
        top.addWidget(self.quality)
        lay.addLayout(top)
        self.title_edit = QLineEdit(self.header)
        self.title_edit.setObjectName("dlTitle")
        self.title_edit.setPlaceholderText(tr("dl.title.ph", "Series name"))
        lay.addWidget(self.title_edit)
        self.subtitle = QLabel("", self.header)
        self.subtitle.setObjectName("dlSub")
        lay.addWidget(self.subtitle)
        return self.header

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(8)
        lay.addStretch(1)
        return w, lay

    def _build_empty(self) -> QWidget:
        w, lay = self._page()
        icon = QLabel("⬇", w)
        icon.setObjectName("dlEmptyIcon")
        icon.setAlignment(Qt.AlignCenter)
        text = QLabel(tr("dl.card.empty", "Paste a video, playlist or series link above, "
                                          "then press Scan."), w)
        text.setObjectName("dlEmptyText")
        text.setAlignment(Qt.AlignCenter)
        text.setWordWrap(True)
        hint = QLabel(tr("dl.card.empty2", "A playlist shows every episode with its poster "
                                           "and length, so you can pick a range (EP 1 – 50, "
                                           "51 – end …) or single episodes."), w)
        hint.setObjectName("dlHint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        for x in (icon, text, hint):
            lay.addWidget(x)
        lay.addStretch(1)
        return w

    def _build_busy(self) -> QWidget:
        w, lay = self._page()
        self.busy_text = QLabel(tr("dl.card.scanning", "Scanning the link…"), w)
        self.busy_text.setObjectName("dlEmptyText")
        self.busy_text.setAlignment(Qt.AlignCenter)
        self.busy_detail = QLabel("", w)
        self.busy_detail.setObjectName("dlHint")
        self.busy_detail.setAlignment(Qt.AlignCenter)
        bar = QProgressBar(w)
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        bar.setFixedWidth(320)
        bar.setMaximumHeight(8)
        cancel = QPushButton(tr("dl.card.cancel", "Cancel"), w)
        cancel.setObjectName("dlGhost")
        cancel.clicked.connect(self.cancel_clicked.emit)
        lay.addWidget(self.busy_text)
        lay.addWidget(self.busy_detail)
        lay.addWidget(bar, 0, Qt.AlignHCenter)
        lay.addSpacing(6)
        lay.addWidget(cancel, 0, Qt.AlignHCenter)
        lay.addStretch(1)
        return w

    def _build_error(self) -> QWidget:
        w, lay = self._page()
        self.error_title = QLabel(tr("dl.card.error", "Could not read this link"), w)
        self.error_title.setObjectName("dlError")
        self.error_title.setAlignment(Qt.AlignCenter)
        self.error_text = QLabel("", w)
        self.error_text.setObjectName("dlHint")
        self.error_text.setAlignment(Qt.AlignCenter)
        self.error_text.setWordWrap(True)
        self.error_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        retry = QPushButton(tr("dl.card.retry", "Try again"), w)
        retry.setObjectName("dlOutline")
        retry.clicked.connect(self.retry_clicked.emit)
        lay.addWidget(self.error_title)
        lay.addWidget(self.error_text)
        lay.addSpacing(6)
        lay.addWidget(retry, 0, Qt.AlignHCenter)
        lay.addStretch(1)
        return w

    def _pill(self, *widgets) -> QFrame:
        f = QFrame(self)
        f.setObjectName("dlPill")
        lay = QHBoxLayout(f)
        lay.setContentsMargins(2, 2, 4, 2)
        lay.setSpacing(2)
        for x in widgets:
            lay.addWidget(x)
        return f

    def _build_playlist(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QFrame(w)
        bar.setObjectName("dlSelBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 8, 16, 8)
        bl.setSpacing(8)
        self.sel_title = QLabel("", bar)
        self.sel_title.setObjectName("dlSelTitle")
        bl.addWidget(self.sel_title)
        bl.addStretch(1)

        def spin() -> QSpinBox:
            s = QSpinBox(bar)
            s.setObjectName("dlPillSpin")
            s.setKeyboardTracking(False)        # typing 50 does not pick 1–5 first
            s.setRange(1, 1)
            return s

        def pill_btn(tip: str) -> QToolButton:
            b = QToolButton(bar)
            b.setObjectName("dlPillBtn")
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            return b

        self.first_btn = pill_btn(tr("dl.sel.first.tip", "Select from the first episode up to "
                                                         "the number you set"))
        self.first_spin = spin()
        self.first_btn.clicked.connect(self._apply_first)
        self.first_spin.valueChanged.connect(self._apply_first)
        self.last_spin = spin()
        self.last_btn = pill_btn(tr("dl.sel.last.tip", "Select from the number you set to "
                                                       "the last episode"))
        self.last_btn.clicked.connect(self._apply_last)
        self.last_spin.valueChanged.connect(self._apply_last)
        self.last_spin.setToolTip(self.last_btn.toolTip())
        self.first_spin.setToolTip(self.first_btn.toolTip())
        bl.addWidget(self._pill(self.first_btn, self.first_spin))
        bl.addWidget(self._pill(self.last_spin, self.last_btn))
        all_btn = QPushButton(tr("dl.sel.all", "Select all"), bar)
        all_btn.setObjectName("dlOutline")
        all_btn.clicked.connect(self._select_all)
        clear_btn = QPushButton(tr("dl.sel.clear", "Clear"), bar)
        clear_btn.setObjectName("dlGhost")
        clear_btn.clicked.connect(self._clear)
        bl.addWidget(all_btn)
        bl.addWidget(clear_btn)
        lay.addWidget(bar)
        self.grid = EpisodeGrid(self)
        self.delegate = _CardDelegate(self)
        self.grid.setItemDelegate(self.delegate)
        self.grid.setModel(self.model)
        self.grid.toggled.connect(self._update_footer)
        lay.addWidget(self.grid, 1)
        return w

    def _build_single(self) -> QWidget:
        w = QWidget(self)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(24, 18, 24, 18)
        lay.setSpacing(28)
        lay.addStretch(1)
        self.poster = QLabel(w)
        self.poster.setObjectName("dlPoster")
        self.poster.setFixedSize(448, 252)
        self.poster.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.poster, 0, Qt.AlignVCenter)
        self.single_info = QLabel("", w)
        self.single_info.setObjectName("dlHint")
        self.single_info.setWordWrap(True)
        self.single_info.setTextFormat(Qt.RichText)
        self.single_info.setMinimumWidth(260)
        self.single_info.setMaximumWidth(420)
        lay.addWidget(self.single_info, 0, Qt.AlignVCenter)
        lay.addStretch(1)
        return w

    def _build_footer(self) -> QFrame:
        self.footer = QFrame(self)
        self.footer.setObjectName("dlCardFooter")
        lay = QVBoxLayout(self.footer)
        lay.setContentsMargins(16, 10, 16, 12)
        lay.setSpacing(8)
        row1 = QHBoxLayout()
        self.count_label = QLabel("", self.footer)
        self.count_label.setTextFormat(Qt.RichText)
        self.note_label = QLabel("", self.footer)
        self.note_label.setObjectName("dlNote")
        self.download_btn = QPushButton("", self.footer)
        self.download_btn.setObjectName("dlDownload")
        self.download_btn.setProperty("class", "accent")
        self.download_btn.setCursor(Qt.PointingHandCursor)
        self.download_btn.clicked.connect(self.download_clicked.emit)
        row1.addWidget(self.count_label)
        row1.addSpacing(14)
        row1.addWidget(self.note_label, 1)
        row1.addWidget(self.download_btn)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        save = QLabel(tr("dl.save.to", "Save to:"), self.footer)
        save.setObjectName("dlHint")
        self.folder_label = QLabel("", self.footer)
        self.folder_label.setObjectName("dlFolder")
        self.folder_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        change = QPushButton(tr("dl.save.change", "Change…"), self.footer)
        change.setObjectName("dlLink")
        change.setCursor(Qt.PointingHandCursor)
        change.clicked.connect(self.change_folder.emit)
        self.ask_check = QCheckBox(tr("dl.save.ask", "Ask where to save each time"), self.footer)
        self.ask_check.toggled.connect(self.ask_toggled.emit)
        self.poster_check = QCheckBox(tr("dl.poster.intro", "Poster in the first second"),
                                      self.footer)
        self.poster_check.setToolTip(tr(
            "dl.poster.intro.tip",
            "Every download keeps its poster inside the video, as the file's cover (no "
            "separate picture file). With this on, the poster is also shown for the "
            "video's first second (the video is re-encoded once, which takes extra time)."))
        self.poster_check.toggled.connect(self.poster_toggled.emit)
        row2.addWidget(save)
        row2.addWidget(self.folder_label, 1)
        row2.addWidget(change)
        row2.addSpacing(10)
        row2.addWidget(self.ask_check)
        row2.addSpacing(18)
        row2.addWidget(self.poster_check)
        lay.addLayout(row2)
        return self.footer

    # -- states ---------------------------------------------------------------
    @property
    def is_playlist(self) -> bool:
        return self.result is not None and self.result.kind == "playlist"

    def _chrome(self, on: bool) -> None:
        self.header.setVisible(on)
        self.footer.setVisible(on)

    def show_empty(self) -> None:
        self.result = None
        self.loader.reset()
        self.model.set_entries([])
        self._chrome(False)
        self.stack.setCurrentIndex(0)

    def show_busy(self, url: str) -> None:
        self._chrome(False)
        self.busy_text.setText(tr("dl.card.scanning", "Scanning the link…"))
        self.busy_detail.setText(QFontMetrics(self.busy_detail.font()).elidedText(
            url, Qt.ElideMiddle, 640))
        self.stack.setCurrentIndex(1)

    def set_busy_text(self, text: str) -> None:
        if text:
            self.busy_detail.setText(text)

    def show_error(self, text: str) -> None:
        self._chrome(False)
        self.error_text.setText(text)
        self.stack.setCurrentIndex(2)

    def show_result(self, res: ScanResult) -> None:
        self.result = res
        self.loader.reset()
        self.images.clear()
        self.delegate.clear()
        self.sizes, self.heights = dict(res.sizes), list(res.heights)
        self.size_duration = res.entries[0].duration if res.kind == "video" and res.entries else 0.0
        self.model.set_entries(res.entries)
        first = res.entries[0] if res.entries else None
        title = res.title or (first.title if first else "") or urlsplit(res.url).netloc
        self.title_edit.setText(title)
        self.title_edit.setCursorPosition(0)
        if self.is_playlist:
            self.badge.setText(tr("dl.badge.series", "SERIES") if res.numbering == "titles"
                               else tr("dl.badge.playlist", "PLAYLIST"))
            self.title_edit.setReadOnly(False)
            self.title_edit.setToolTip(tr("dl.title.tip", "Series name: also the name of the "
                                                          "folder the episodes go in. Click to "
                                                          "rename."))
            eps = [e.ep for e in res.entries if e.ep]
            self._lo, self._hi = (min(eps), max(eps)) if eps else (1, 1)
            mid = self._lo + (self._hi - self._lo) // 2
            for s, v in ((self.first_spin, mid), (self.last_spin, min(self._hi, mid + 1))):
                s.blockSignals(True)
                s.setRange(self._lo, self._hi)
                s.setValue(v)
                s.blockSignals(False)
            self.first_btn.setText(tr("dl.sel.from", "EP {n} –").replace("{n}", str(self._lo)))
            self.last_btn.setText(tr("dl.sel.to", "– EP {n}").replace("{n}", str(self._hi)))
            self.sel_title.setText("● " + tr("dl.sel.title", "Select episodes (EP {lo} – EP {hi})")
                                   .replace("{lo}", str(self._lo)).replace("{hi}", str(self._hi)))
            matched = [i for i, e in enumerate(res.entries) if e.matched]
            self.model.select_only(matched)          # the link's own episode
            self.stack.setCurrentIndex(3)
            self.grid.relayout()
            if matched:
                QTimer.singleShot(0, lambda r=matched[0]: self.grid.scrollTo(
                    self.model.index(r), QAbstractItemView.PositionAtCenter))
            self.grid.setFocus()
        else:
            self.badge.setText(tr("dl.badge.video", "VIDEO"))
            self.title_edit.setReadOnly(True)
            self.title_edit.setToolTip(title)
            self.model.select_only([0])
            self.poster.setPixmap(QPixmap())
            if first and first.thumb:
                self.loader.want_image(first.thumb)
            self.stack.setCurrentIndex(4)
            self._paint_single()
        self._update_subtitle()
        self._update_quality_items()
        self._update_footer()
        self._chrome(True)

    def set_sample(self, probe: dict) -> None:
        """Sizes / heights of one episode (the first), for the estimates."""
        if not self.is_playlist or not probe:
            return
        self.sizes = dict(probe.get("sizes") or {})
        self.heights = list(probe.get("heights") or [])
        self.size_duration = float(probe.get("duration") or 0)
        self._update_quality_items()
        self._update_footer()

    # -- inputs from the tab ----------------------------------------------------
    def set_quality(self, key: str) -> None:
        i = self.quality.findData(key)
        self._user_quality = False
        self.quality.setCurrentIndex(max(0, i))
        self._user_quality = True

    def set_folder(self, path: str) -> None:
        self._folder = path
        self.folder_label.setToolTip(path)
        self._elide_folder()

    def _elide_folder(self) -> None:
        path = getattr(self, "_folder", "")
        w = max(120, self.folder_label.width())
        self.folder_label.setText(QFontMetrics(self.folder_label.font()).elidedText(
            path, Qt.ElideMiddle, w))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._elide_folder)

    def set_ask(self, on: bool) -> None:
        self.ask_check.blockSignals(True)
        self.ask_check.setChecked(on)
        self.ask_check.blockSignals(False)

    def set_poster(self, on: bool) -> None:
        self.poster_check.blockSignals(True)
        self.poster_check.setChecked(on)
        self.poster_check.blockSignals(False)

    def set_cookies(self, cookies: str) -> None:
        self.loader.cookies = cookies

    def set_marks(self, marks: dict[str, str]) -> None:
        self.model.marks = marks
        self.model.changed()

    def note(self, text: str) -> None:
        self.note_label.setText(text)
        self._note_timer.start(7000)

    # -- outputs ------------------------------------------------------------------
    def quality_key(self) -> str:
        return self.quality.currentData() or "best"

    def series_name(self) -> str:
        return self.title_edit.text().strip()

    def poster_intro(self) -> bool:
        return self.poster_check.isChecked()

    def selected_entries(self) -> list[ScanEntry]:
        return [e for i, e in enumerate(self.model.entries) if i in self.model.checked]

    def ep_label(self, e: ScanEntry) -> str:
        if not e.ep:
            return tr("dl.ep.extra", "Extra")
        text = tr("dl.ep.label", "EP {n}").replace("{n}", str(e.ep))
        return text + (f" ({e.dup + 1})" if e.dup else "")

    # -- range shortcuts -------------------------------------------------------------
    def _rows_where(self, test) -> list[int]:
        return [i for i, e in enumerate(self.model.entries) if e.ep and test(e.ep)]

    def _pick(self, rows: list[int]) -> None:
        self.model.select_only(rows)
        if rows:
            self.grid.scrollTo(self.model.index(rows[0]), QAbstractItemView.PositionAtTop)
        self._update_footer()

    def _apply_first(self, *_a) -> None:
        hi = self.first_spin.value()
        self._pick(self._rows_where(lambda n: self._lo <= n <= hi))

    def _apply_last(self, *_a) -> None:
        lo = self.last_spin.value()
        self._pick(self._rows_where(lambda n: n >= lo))

    def _select_all(self) -> None:
        self._pick(list(range(len(self.model.entries))))

    def _clear(self) -> None:
        self.model.select_only([])
        self._update_footer()

    # -- updates ---------------------------------------------------------------------
    def _on_quality(self, _i: int) -> None:
        self._update_footer()
        self._paint_single()
        if self._user_quality:
            self.quality_changed.emit(self.quality_key())

    def _on_image(self, url: str, img: QImage) -> None:
        self.images[url] = img
        if self.is_playlist:
            self.grid.viewport().update()
        else:
            self._paint_single()

    def _on_details(self, gen: int, payload) -> None:
        if gen != self.loader.gen or not self.is_playlist:
            return
        url, d = payload
        rows = [i for i, e in enumerate(self.model.entries) if e.url == url]
        for i in rows:
            apply_details(self.model.entries[i], d)
        self.model.changed(rows)

    def _update_subtitle(self) -> None:
        res = self.result
        if res is None:
            return
        bits = []
        if res.kind == "playlist":
            bits.append(tr("dl.sub.eps", "{n} episodes").replace("{n}", str(len(res.entries))))
            if res.truncated:
                bits.append(tr("dl.sub.more", "first {n} shown").replace("{n}", str(len(res.entries))))
            if res.uploader:
                bits.append(res.uploader)
            bits.append(tr("dl.sub.titles", "numbered from the titles") if res.numbering == "titles"
                        else tr("dl.sub.order", "in playlist order"))
        else:
            e = res.entries[0]
            if res.uploader:
                bits.append(res.uploader)
            if e.duration:
                bits.append(fmt_duration(e.duration))
            if self.heights:
                bits.append(tr("dl.sub.upto", "up to {h}p").replace("{h}", str(max(self.heights))))
        self.subtitle.setText("  ·  ".join(bits))

    def _update_quality_items(self) -> None:
        """Grey out heights the video does not have; one video also shows
        the size of each choice."""
        top = max(self.heights) if self.heights else 0
        model = self.quality.model()
        for i, (key, k, default) in enumerate(QUALITY_CHOICES):
            text = tr(k, default)
            if not self.is_playlist and key in self.sizes:
                text += f"  ·  {fmt_size(self.sizes[key])}"
            self.quality.setItemText(i, text)
            item = model.item(i)
            if item is not None:
                # 1080p on a 720p video would give the 720p file: say so by greying it
                item.setEnabled(key == "best" or not top or _Q_HEIGHT[key] <= top
                                or _Q_HEIGHT[key] == min(h for h in _Q_HEIGHT.values() if h >= top))

    def _bytes_for(self, e: ScanEntry | None) -> float:
        size = self.sizes.get(self.quality_key()) or 0
        if size and e is not None and e.duration and self.size_duration and self.is_playlist:
            return size * e.duration / self.size_duration
        return float(size)

    def _update_footer(self, *_a) -> None:
        res = self.result
        if res is None:
            return
        n, total = len(self.model.checked), len(self.model.entries)
        blue = f"<b style='color:{theme.ACCENT}'>{{n}}</b>"
        if self.is_playlist:
            text = html.escape(tr("dl.sel.count", "Selected: {n} / {total} episodes")
                               .replace("{total}", str(total)))
            text = text.replace("{n}", blue.replace("{n}", str(n)))
            per = [self._bytes_for(e) for e in self.selected_entries()]
            if n and all(per):
                text += "&nbsp;&nbsp;·&nbsp;&nbsp;" + html.escape(
                    tr("dl.sel.total", "≈ {size} in total").replace("{size}", fmt_size(sum(per))))
            avg = self._bytes_for(None)
            durs = [e.duration for e in self.model.entries if e.duration]
            if avg and durs and self.size_duration:
                avg = avg * (sum(durs) / len(durs)) / self.size_duration
            self.est.setText(tr("dl.est.ep", "Est. ~{size} / EP").replace("{size}", fmt_size(avg))
                             if avg else "")
            self.download_btn.setEnabled(n > 0)
            self.download_btn.setText(
                "⬇  " + (tr("dl.download.n", "Download {n} episodes").replace("{n}", str(n))
                         if n else tr("dl.download.pick", "Select episodes to download")))
        else:
            size = self._bytes_for(None)
            e = res.entries[0]
            text = html.escape(tr("dl.single.one", "1 video")) + (
                "&nbsp;&nbsp;·&nbsp;&nbsp;" + fmt_duration(e.duration) if e.duration else "")
            if size:
                text += "&nbsp;&nbsp;·&nbsp;&nbsp;" + html.escape("≈ " + fmt_size(size))
            self.est.setText(tr("dl.est.one", "Est. ~{size}").replace("{size}", fmt_size(size))
                             if size else "")
            self.download_btn.setEnabled(res.warning != "drm")
            self.download_btn.setText("⬇  " + tr("dl.download.one", "Download video"))
        self.count_label.setText(f"<span style='color:{theme.TEXT_DIM}'>{text}</span>")

    def _paint_single(self) -> None:
        res = self.result
        if res is None or res.kind != "video" or not res.entries:
            return
        e = res.entries[0]
        dpr = self.devicePixelRatioF()
        size = self.poster.size()
        img = self.images.get(e.thumb)
        if img is not None:
            pix = compose_poster(img, size, dpr)
        else:
            pix = QPixmap(round(size.width() * dpr), round(size.height() * dpr))
            pix.setDevicePixelRatio(dpr)
            pix.fill(QColor("#1c2633"))
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        if e.duration:
            f = QFont(self.font())
            f.setBold(True)
            fm = QFontMetrics(f)
            text = fmt_duration(e.duration)
            bw, bh = fm.horizontalAdvance(text) + 14, fm.height() + 6
            _draw_badge(p, QRectF(size.width() - bw - 10, size.height() - bh - 10, bw, bh),
                        text, f, QColor(0, 0, 0, 190))
        p.end()
        rounded = QPixmap(pix.size())
        rounded.setDevicePixelRatio(dpr)
        rounded.fill(Qt.transparent)
        q = QPainter(rounded)
        q.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, size.width(), size.height()), 10, 10)
        q.setClipPath(clip)
        q.drawPixmap(0, 0, pix)
        q.end()
        self.poster.setPixmap(rounded)
        rows = []
        if res.warning == "drm":
            rows.append(f"<span style='color:#f85149'><b>{html.escape(tr('dl.drm', 'This video is DRM-protected: the site does not allow downloading it.'))}</b></span>")
        elif res.warning == "preview":
            rows.append(f"<span style='color:#d29922'><b>{html.escape(tr('dl.hongguo', 'Hongguo (红果短剧) plays its full episodes only in its own protected player, so they cannot be downloaded. This is the short public preview.'))}</b></span>")
        if e.duration:
            rows.append(f"{html.escape(tr('dl.single.length', 'Length'))}: "
                        f"<b style='color:{theme.TEXT}'>{fmt_duration(e.duration)}</b>")
        if res.uploader:
            rows.append(f"{html.escape(tr('dl.single.by', 'Uploader'))}: "
                        f"<b style='color:{theme.TEXT}'>{html.escape(res.uploader)}</b>")
        if self.heights:
            rows.append(f"{html.escape(tr('dl.single.best', 'Best quality'))}: "
                        f"<b style='color:{theme.TEXT}'>{max(self.heights)}p</b>")
        size_b = self._bytes_for(None)
        if size_b:
            rows.append(f"{html.escape(tr('dl.single.size', 'Size at this quality'))}: "
                        f"<b style='color:{theme.TEXT}'>≈ {fmt_size(size_b)}</b>")
        self.single_info.setText("<br><br>".join(rows))

    def shutdown(self) -> None:
        self.loader.shutdown()
