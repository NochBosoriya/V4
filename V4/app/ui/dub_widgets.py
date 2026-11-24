"""Building blocks of the Dub tab's monitor (the user's design, 2026-11-11):
glowing navy cards, round icons, the hardware tiles and check rows of
"System Readiness Check", collapsible section headers, a live indicator and
the task list's two-line rows (task name, its note dimmed below).

Icons are painted with QPainter on a 24-unit grid (Feather-style strokes):
sharp at any display scale, and no icon font is needed (the offscreen test
platform and older Windows have no Segoe Fluent Icons)."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui import theme

GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BLUE = theme.ACCENT
GREY = theme.TEXT_DIM
CARD_BG = "#0e1522"          # navy, close to the window so inner widgets blend
TILE_BG = "#131d30"
TILE_BORDER = "#22314d"
STATE_COLOR = {"ok": GREEN, "warn": AMBER, "bad": RED, "none": GREY, "busy": BLUE}
STATE_ICON = {"ok": "check", "warn": "warn", "bad": "cross", "none": "dot", "busy": "clock"}


# ---------------------------------------------------------------------------
# icons
# ---------------------------------------------------------------------------

def _dpr() -> float:
    app = QGuiApplication.instance()
    scr = app.primaryScreen() if app is not None else None
    return float(scr.devicePixelRatio()) if scr is not None else 1.0


def _pen(color: QColor, width: float = 2.0) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _poly(p: QPainter, pts, close: bool = False) -> None:
    path = QPainterPath(QPointF(*pts[0]))
    for x, y in pts[1:]:
        path.lineTo(x, y)
    if close:
        path.closeSubpath()
    p.drawPath(path)


def _glyph(p: QPainter, name: str, color: str) -> None:
    """Draw icon `name` on a 24 x 24 grid (the painter is scaled to it)."""
    c = QColor(color)
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    if name == "pulse":
        _poly(p, [(2, 12), (6, 12), (9, 4.5), (15, 19.5), (18, 12), (22, 12)])
    elif name == "check":
        _poly(p, [(5, 12.5), (10, 17.5), (19, 7)])
    elif name == "cross":
        _poly(p, [(6.5, 6.5), (17.5, 17.5)])
        _poly(p, [(17.5, 6.5), (6.5, 17.5)])
    elif name == "warn":
        _poly(p, [(12, 3.5), (21.5, 20), (2.5, 20)], close=True)
        _poly(p, [(12, 9.5), (12, 13.8)])
        p.setBrush(c)
        p.drawEllipse(QPointF(12, 16.9), 0.8, 0.8)
    elif name == "info":
        p.drawEllipse(QPointF(12, 12), 9.5, 9.5)
        _poly(p, [(12, 11), (12, 16.5)])
        p.setBrush(c)
        p.drawEllipse(QPointF(12, 7.6), 0.8, 0.8)
    elif name == "gear":
        p.drawEllipse(QPointF(12, 12), 3.0, 3.0)
        p.drawEllipse(QPointF(12, 12), 7.0, 7.0)
        for k in range(8):
            a = k * math.pi / 4
            _poly(p, [(12 + 7.0 * math.cos(a), 12 + 7.0 * math.sin(a)),
                      (12 + 9.8 * math.cos(a), 12 + 9.8 * math.sin(a))])
    elif name == "chip":
        p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
        p.drawRect(QRectF(9.5, 9.5, 5, 5))
        for v in (9, 12, 15):
            _poly(p, [(v, 2.5), (v, 6)])
            _poly(p, [(v, 18), (v, 21.5)])
            _poly(p, [(2.5, v), (6, v)])
            _poly(p, [(18, v), (21.5, v)])
    elif name == "bolt":
        fill = QColor(c)
        fill.setAlphaF(0.25)
        p.setBrush(fill)
        _poly(p, [(13, 2.5), (4.5, 13.5), (11.5, 13.5), (10.5, 21.5), (19.5, 10.5),
                  (12.5, 10.5)], close=True)
    elif name == "memory":
        p.drawRoundedRect(QRectF(3, 7, 18, 10), 1.5, 1.5)
        for x in (7.5, 12, 16.5):
            _poly(p, [(x, 10), (x, 14)])
        for x in (6, 10, 14, 18):
            _poly(p, [(x, 17), (x, 20.5)])
    elif name == "terminal":
        p.drawRoundedRect(QRectF(3, 4.5, 18, 15), 2, 2)
        _poly(p, [(7, 9.5), (10, 12), (7, 14.5)])
        _poly(p, [(12, 15), (17, 15)])
    elif name == "clock":
        p.drawEllipse(QPointF(12, 12), 9.5, 9.5)
        _poly(p, [(12, 6.5), (12, 12), (15.5, 14)])
    elif name == "pending":
        p.drawEllipse(QPointF(12, 12), 8, 8)
    elif name == "play":
        p.setBrush(c)
        _poly(p, [(8.5, 5.5), (18.5, 12), (8.5, 18.5)], close=True)
    elif name == "stop":
        p.setBrush(c)
        p.drawRoundedRect(QRectF(7, 7, 10, 10), 2, 2)
    elif name == "pause":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(7, 6.5, 3.6, 11), 1.2, 1.2)
        p.drawRoundedRect(QRectF(13.4, 6.5, 3.6, 11), 1.2, 1.2)
    elif name == "chevron_down":
        _poly(p, [(6, 9), (12, 15), (18, 9)])
    elif name == "chevron_right":
        _poly(p, [(9, 6), (15, 12), (9, 18)])
    elif name == "dot":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawEllipse(QPointF(12, 12), 5, 5)
    elif name == "refresh":
        p.drawArc(QRectF(4, 4, 16, 16), 40 * 16, 290 * 16)
        _poly(p, [(19.5, 3.5), (19.5, 9), (14, 9)])
    elif name == "list":
        for y in (6, 12, 18):
            _poly(p, [(9, y), (20, y)])
            p.drawEllipse(QPointF(4.5, y), 0.9, 0.9)
    elif name == "shield":
        _poly(p, [(12, 2.5), (20, 5.5), (20, 11.5), (12, 21.5), (4, 11.5), (4, 5.5)], close=True)
        _poly(p, [(8.5, 12), (11, 14.5), (15.5, 9.5)])


def _canvas(size: int) -> tuple[QPixmap, QPainter]:
    dpr = _dpr()
    pm = QPixmap(max(1, int(round(size * dpr))), max(1, int(round(size * dpr))))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    return pm, p


def icon_pixmap(name: str, color: str, size: int = 18) -> QPixmap:
    """A painted line icon of `size` logical pixels."""
    pm, p = _canvas(size)
    p.scale(size / 24.0, size / 24.0)
    _glyph(p, name, color)
    p.end()
    return pm


def round_pixmap(name: str, color: str, size: int = 34, solid: bool = False) -> QPixmap:
    """An icon in a circle: tinted with `color` (solid=False) or a filled
    disc with a soft halo and the icon in white (the monitor's own icon)."""
    pm, p = _canvas(size)
    c = QColor(color)
    if solid:
        halo = QColor(c)
        halo.setAlphaF(0.22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(QRectF(0, 0, size, size))
        p.setBrush(c)
        p.drawEllipse(QRectF(size * 0.12, size * 0.12, size * 0.76, size * 0.76))
        inner, glyph = size * 0.44, "#ffffff"
    else:
        fill, ring = QColor(c), QColor(c)
        fill.setAlphaF(0.14)
        ring.setAlphaF(0.55)
        p.setPen(QPen(ring, 1.4))
        p.setBrush(fill)
        p.drawEllipse(QRectF(1, 1, size - 2, size - 2))
        inner, glyph = size * 0.54, color
    off = (size - inner) / 2.0
    p.translate(off, off)
    p.scale(inner / 24.0, inner / 24.0)
    _glyph(p, name, glyph)
    p.end()
    return pm


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------

def clear_widget(parent: QWidget | None = None) -> QWidget:
    """A layout holder that lets the card behind it show (the app's style
    sheet paints every plain QWidget with the window colour)."""
    w = QWidget(parent)
    w.setObjectName("clear")
    return w


class GlowCard(QFrame):
    """A rounded navy card with a soft glowing edge; its contents sit
    inside the glow margin."""

    def __init__(self, parent: QWidget | None = None, color: str = BLUE,
                 radius: int = 12, glow: int = 5):
        super().__init__(parent)
        self._color, self._radius, self._glow = QColor(color), radius, glow
        self.setContentsMargins(glow, glow, glow, glow)
        self.setStyleSheet("QWidget#clear { background: transparent; }")

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _ev):  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = self._glow
        r = QRectF(self.rect()).adjusted(g, g, -g, -g)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k in range(g, 0, -1):          # the glow: fainter further out
            c = QColor(self._color)
            c.setAlphaF(0.045 * (g - k + 1) / g * 2.2)
            p.setPen(QPen(c, 1.0))
            p.drawRoundedRect(r.adjusted(-k + 0.5, -k + 0.5, k - 0.5, k - 0.5),
                              self._radius + k, self._radius + k)
        edge = QColor(self._color)
        edge.setAlphaF(0.85)
        p.setPen(QPen(edge, 1.3))
        p.setBrush(QColor(CARD_BG))
        p.drawRoundedRect(r.adjusted(0.6, 0.6, -0.6, -0.6), self._radius, self._radius)


# ---------------------------------------------------------------------------
# small widgets
# ---------------------------------------------------------------------------

class LivePill(QFrame):
    """'● Real-time monitoring' while a job runs, '● Idle' otherwise."""

    def __init__(self, live_text: str, idle_text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("livePill")
        self._texts = [live_text, idle_text]
        self.dot = QLabel()
        self.text = QLabel()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 3, 12, 3)
        lay.setSpacing(6)
        lay.addWidget(self.dot)
        lay.addWidget(self.text)
        self.live = False
        self.set_live(False)

    def set_texts(self, live_text: str, idle_text: str) -> None:
        self._texts = [live_text, idle_text]
        self.set_live(self.live)

    def set_live(self, live: bool) -> None:
        self.live = bool(live)
        color = GREEN if self.live else GREY
        self.dot.setPixmap(icon_pixmap("dot", color, 12))
        self.text.setText(self._texts[0] if self.live else self._texts[1])
        self.setStyleSheet(
            f"QFrame#livePill {{ background: {TILE_BG}; border: 1px solid "
            f"{'#2c5a3a' if self.live else TILE_BORDER}; border-radius: 12px; }}"
            f"QFrame#livePill QLabel {{ color: {'#c9f1d2' if self.live else GREY}; }}")


class ElideLabel(QLabel):
    """A one-line label that keeps its whole text (text() returns it) and
    draws it shortened with "…" to fit the room it is given."""

    def __init__(self, text: str = "", parent: QWidget | None = None,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight):
        super().__init__(text, parent)
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(20, super().minimumSizeHint().height())

    def paintEvent(self, _ev):  # noqa: N802 (Qt override)
        p = QPainter(self)
        r = self.contentsRect()
        p.setPen(self.palette().color(self.foregroundRole()))
        p.setFont(self.font())
        p.drawText(r, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self.fontMetrics().elidedText(self.text(), self._mode, r.width()))


class Counter(QWidget):
    """An icon and a short count text ("Running: 1")."""

    def __init__(self, icon: str, color: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("clear")
        self.icon = QLabel()
        self.icon.setPixmap(icon_pixmap(icon, color, 14))
        self.label = QLabel()
        self.label.setStyleSheet(f"color: {color};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        lay.addWidget(self.icon)
        lay.addWidget(self.label)

    def set_text(self, text: str) -> None:
        self.label.setText(text)


class HwTile(QFrame):
    """A hardware tile of the readiness check: a round icon, a title and a
    status line ("✓ Ready")."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("hwTile")
        self.setStyleSheet(f"QFrame#hwTile {{ background: {TILE_BG}; border: 1px solid "
                           f"{TILE_BORDER}; border-radius: 10px; }}")
        self.icon = QLabel()
        self.title = ElideLabel()
        self.title.setStyleSheet("font-weight: 600;")
        self.status_icon = QLabel()
        self.status = ElideLabel()
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(self.title)
        st = QHBoxLayout()
        st.setSpacing(4)
        st.addWidget(self.status_icon)
        st.addWidget(self.status, 1)
        col.addLayout(st)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(10)
        lay.addWidget(self.icon)
        lay.addLayout(col, 1)
        self.setMinimumWidth(130)

    def set(self, icon: str, title: str, state: str, status: str, tip: str = "") -> None:
        color = STATE_COLOR.get(state, GREY)
        self.icon.setPixmap(round_pixmap(icon, color, 34))
        self.title.setText(title)
        self.status_icon.setPixmap(icon_pixmap(STATE_ICON.get(state, "dot"), color, 13))
        self.status.setText(status)
        self.status.setStyleSheet(f"color: {color};")
        self.setToolTip(tip or title)


class CheckRow(QFrame):
    """One readiness check: status icon, name and its short state; click
    to select (the fix shows below), double-click to fix."""

    clicked = Signal(str)
    double_clicked = Signal(str)

    def __init__(self, cid: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.cid = cid
        self.setObjectName("checkRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.icon = QLabel()
        self.name = ElideLabel()
        self.state = ElideLabel()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)
        lay.addWidget(self.icon)
        lay.addWidget(self.name, 3)
        lay.addWidget(self.state, 2)
        self._color = GREY
        self.selected = False
        self._style()

    def set(self, state: str, name: str, short: str, tip: str) -> None:
        self._color = STATE_COLOR.get(state, GREY)
        self.icon.setPixmap(icon_pixmap(STATE_ICON.get(state, "dot"), self._color, 15))
        self.name.setText(name)
        self.state.setText(short)
        self.state.setStyleSheet(f"color: {self._color if state != 'ok' else GREY};")
        self.setToolTip(tip)

    def set_selected(self, on: bool) -> None:
        self.selected = bool(on)
        self._style()

    def _style(self) -> None:
        border = BLUE if self.selected else "transparent"
        bg = "#16233b" if self.selected else "transparent"
        self.setStyleSheet(f"QFrame#checkRow {{ background: {bg}; border: 1px solid {border};"
                           f" border-radius: 6px; }}")

    def mousePressEvent(self, ev):  # noqa: N802 (Qt override)
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.cid)
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):  # noqa: N802 (Qt override)
        if ev.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.cid)
        super().mouseDoubleClickEvent(ev)


class AutoHideLabel(QLabel):
    """A label that takes no room while it has nothing to say."""

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        super().setText(text)
        self.setVisible(bool(text))


class ClickLabel(QLabel):
    """A label that can be clicked (a section title folds its section)."""

    clicked = Signal()

    def mousePressEvent(self, ev):  # noqa: N802 (Qt override)
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class SectionHeader(QWidget):
    """A section's title row: a fold arrow, an optional round state icon,
    the title, a dimmed summary and widgets on the right. Clicking the
    arrow or the title folds the section (`toggled(open)`)."""

    toggled = Signal(bool)

    def __init__(self, title: str, icon: str = "", parent: QWidget | None = None,
                 foldable: bool = True):
        super().__init__(parent)
        self.setObjectName("clear")
        self.open = True
        self.fold = QToolButton()
        self.fold.setAutoRaise(True)
        self.fold.setVisible(foldable)
        self.fold.clicked.connect(lambda: self.set_open(not self.open, user=True))
        self.state_icon = QLabel()
        self.state_icon.setVisible(bool(icon))
        self._icon = icon
        self.title = ClickLabel(title)
        self.title.setObjectName("sectionTitle")
        self.title.setStyleSheet("font-size: 11pt; font-weight: 600;")
        if foldable:
            self.title.setCursor(Qt.CursorShape.PointingHandCursor)
            self.title.clicked.connect(lambda: self.set_open(not self.open, user=True))
        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {GREY};")
        self.right = QHBoxLayout()
        self.right.setSpacing(12)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)
        lay.addWidget(self.fold)
        lay.addWidget(self.state_icon)
        lay.addWidget(self.title)
        lay.addWidget(self.summary)
        lay.addStretch(1)
        lay.addLayout(self.right)
        self.set_state("ok" if icon else "none")
        self._arrow()

    def set_state(self, state: str, icon: str = "") -> None:
        if self._icon or icon:
            name = icon or STATE_ICON.get(state, self._icon)
            self.state_icon.setPixmap(round_pixmap(name, STATE_COLOR.get(state, BLUE), 26))
            self.state_icon.setVisible(True)

    def set_open(self, on: bool, user: bool = False) -> None:
        self.open = bool(on)
        self._arrow()
        if user:
            self.toggled.emit(self.open)

    def _arrow(self) -> None:
        self.fold.setIcon(QIcon(icon_pixmap("chevron_down" if self.open else "chevron_right",
                                            GREY, 16)))
        self.fold.setIconSize(QSize(16, 16))


# ---------------------------------------------------------------------------
# the task list
# ---------------------------------------------------------------------------

STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 1     # a task row's status


class TaskDelegate(QStyledItemDelegate):
    """Draws a task row's first column: its icon, the task name and, on a
    second dimmed line, the note it finished with (the tooltip has it all),
    centred in the row."""

    _GAP = 3            # px between the name and the note

    @staticmethod
    def _lines(index) -> tuple[str, str]:
        # the raw text: the view's own text turns the line break into
        # U+2028, and the two lines were then drawn as one block
        raw = str(index.data(Qt.ItemDataRole.DisplayRole) or "").replace("\u2028", "\n")
        title, _, note = raw.partition("\n")
        return title, note.replace("\n", " ").strip()

    @staticmethod
    def _note_font(font: QFont) -> QFont:
        nf = QFont(font)
        if nf.pointSizeF() > 0:
            nf.setPointSizeF(max(7.0, nf.pointSizeF() - 0.5))
        return nf

    def paint(self, p: QPainter, option, index) -> None:  # noqa: D401
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QIcon()
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, p, opt.widget)
        title, note = self._lines(index)
        f = QFont(opt.font)
        if index.data(STATUS_ROLE) == "running":
            f.setBold(True)
        nf = self._note_font(opt.font)
        th, nh = QFontMetrics(f).height(), QFontMetrics(nf).height()
        r = QRect(opt.rect).adjusted(8, 2, -8, -2)
        block = th + (self._GAP + nh if note else 0)
        top = r.top() + max(0, (r.height() - block) // 2)
        x = r.left()
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None and not icon.isNull():
            size = 20
            p.drawPixmap(x, top + max(0, (block - size) // 2), icon.pixmap(QSize(size, size)))
            x += size + 10
        p.save()
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        title_r = QRect(x, top, r.right() - x, th)
        p.drawText(title_r, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   _elide(f, title, title_r.width()))
        if note:
            p.setFont(nf)
            p.setPen(QColor(GREY))
            note_r = QRect(x, top + th + self._GAP, r.right() - x, nh)
            p.drawText(note_r, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                       _elide(nf, note, note_r.width()))
        p.restore()

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 (Qt override)
        title, note = self._lines(index)
        h = QFontMetrics(option.font).height()
        if note:
            h += self._GAP + QFontMetrics(self._note_font(option.font)).height()
        return QSize(200, max(32, h + 14))


def _elide(font: QFont, text: str, width: int) -> str:
    return QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, max(10, width))
