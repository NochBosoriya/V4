"""Dark slate design system for RecapStudio.

Palette:
    background  #0d1117
    cards/panel #161b22
    accent      #58a6ff
    borders     #30363d
    text        #e6edf3   secondary #8b949e
Radii: 6px everywhere.
"""
from __future__ import annotations

BG = "#0d1117"
PANEL = "#161b22"
ACCENT = "#58a6ff"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
HOVER = "#21262d"
PRESSED = "#1c2129"
DISABLED_TEXT = "#484f58"
SELECTION = "#1f6feb"

QSS: str = f"""
/* ============ base ============ */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Kantumruy Pro", "Khmer OS", sans-serif;
    font-size: 9.5pt;
    selection-background-color: {SELECTION};
    selection-color: #ffffff;
}}
QMainWindow, QDialog {{
    background-color: {BG};
}}
QLabel {{
    background: transparent;
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ /* H/V lines */
    color: {BORDER};
    background-color: {BORDER};
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:hover {{
    background-color: {ACCENT};
}}

/* ============ group boxes / panels ============ */
QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_DIM};
    background-color: transparent;
    font-weight: 600;
}}

/* ============ tab bar (header nav look) ============ */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {BORDER};
    background-color: {BG};
    top: -1px;
}}
QTabBar {{
    background: transparent;
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: {TEXT_DIM};
    padding: 9px 18px 7px 18px;
    margin: 0px 2px;
    min-width: 40px;
    font-weight: 600;
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:disabled {{
    color: {DISABLED_TEXT};
}}

/* ============ buttons ============ */
QPushButton {{
    background-color: {HOVER};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: #2b313a;
    border-color: #8b949e;
}}
QPushButton:pressed {{
    background-color: {PRESSED};
}}
QPushButton:disabled {{
    color: {DISABLED_TEXT};
    background-color: {PANEL};
    border-color: {BORDER};
}}
QPushButton:focus {{
    border-color: {ACCENT};
}}
QPushButton[class="accent"] {{
    background-color: {ACCENT};
    color: #0d1117;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton[class="accent"]:hover {{
    background-color: #79b8ff;
    border-color: #79b8ff;
}}
QPushButton[class="accent"]:pressed {{
    background-color: #388bfd;
    border-color: #388bfd;
}}
QPushButton[class="accent"]:disabled {{
    background-color: #1f3a5f;
    border-color: #1f3a5f;
    color: #6e7681;
}}
QToolButton {{
    background-color: transparent;
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}}
QToolButton:hover {{
    background-color: {HOVER};
    border-color: {BORDER};
}}
QToolButton:pressed {{
    background-color: {PRESSED};
}}

/* ============ inputs ============ */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QTimeEdit, QDateEdit, QDateTimeEdit {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {DISABLED_TEXT};
    background-color: {BG};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: none;
}}
/* QComboBox::down-arrow: painted images, see _arrow_qss() */
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {SELECTION};
    selection-color: #ffffff;
    outline: none;
    padding: 4px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button, QTimeEdit::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button, QTimeEdit::down-button {{
    background-color: {HOVER};
    border: 1px solid {BORDER};
    border-radius: 3px;
    width: 16px;
    margin: 1px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QTimeEdit::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover,
QTimeEdit::down-button:hover {{
    background-color: #2b313a;
}}
/* QSpinBox::up-arrow / ::down-arrow: painted images, see _arrow_qss() */

/* ============ tables / lists / trees ============ */
QTableWidget, QTableView, QListWidget, QListView, QTreeWidget, QTreeView {{
    background-color: {PANEL};
    alternate-background-color: #1b2129;
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    gridline-color: {BORDER};
    outline: none;
}}
QTableWidget::item, QTableView::item, QListWidget::item, QTreeWidget::item {{
    padding: 3px 6px;
}}
QTableWidget::item:selected, QTableView::item:selected,
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {SELECTION};
    color: #ffffff;
}}
QHeaderView {{
    background-color: {PANEL};
    border: none;
}}
QHeaderView::section {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-weight: 600;
}}
QHeaderView::section:hover {{
    color: {TEXT};
}}
QTableCornerButton::section {{
    background-color: {PANEL};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
}}

/* ============ progress bar ============ */
QProgressBar {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    color: {TEXT};
    min-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
    margin: 1px;
}}

/* ============ scrollbars (slim, rounded) ============ */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background-color: {BORDER};
    border-radius: 4px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: #6e7681;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background-color: {BORDER};
    border-radius: 4px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: #6e7681;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
    background: none;
    border: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ============ tooltip ============ */
QToolTip {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
}}

/* ============ dock widgets ============ */
QDockWidget {{
    background-color: {BG};
    color: {TEXT};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    text-align: left;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 2px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {HOVER};
    border-radius: 4px;
}}

/* ============ menus ============ */
QMenuBar {{
    background-color: {BG};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 6px;
}}
QMenuBar::item:selected {{
    background-color: {HOVER};
}}
QMenu {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px;
}}
QMenu::item {{
    background: transparent;
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {SELECTION};
    color: #ffffff;
}}
QMenu::item:disabled {{
    color: {DISABLED_TEXT};
}}
QMenu::separator {{
    height: 1px;
    background-color: {BORDER};
    margin: 5px 8px;
}}
QMenu::indicator {{
    width: 14px;
    height: 14px;
    margin-left: 4px;
}}

/* ============ checkboxes / radios ============ */
QCheckBox, QRadioButton {{
    background: transparent;
    spacing: 7px;
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: {DISABLED_TEXT};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    background-color: {PANEL};
    border: 1px solid {BORDER};
}}
QCheckBox::indicator {{
    border-radius: 4px;
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}
QRadioButton::indicator:checked {{
    background-color: {PANEL};
    border: 5px solid {ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {BG};
    border-color: {BORDER};
}}

/* ============ sliders ============ */
QSlider {{
    background: transparent;
    min-height: 22px;
}}
QSlider::groove:horizontal {{
    background-color: {BORDER};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background-color: {ACCENT};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: {ACCENT};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background-color: #79b8ff;
}}
QSlider::groove:vertical {{
    background-color: {BORDER};
    width: 4px;
    border-radius: 2px;
}}
QSlider::handle:vertical {{
    background-color: {ACCENT};
    border: none;
    width: 14px;
    height: 14px;
    margin: 0 -5px;
    border-radius: 7px;
}}

/* ============ status bar ============ */
QStatusBar {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}
QStatusBar::item {{
    border: none;
}}
QStatusBar QLabel {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 0 8px;
}}

/* ============ misc ============ */
QMessageBox {{
    background-color: {PANEL};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
"""


# The arrows of drop-down lists and number boxes. They used to be CSS border
# triangles (a 0x0 box with one coloured border), which Qt draws as a flat
# grey bar; now they are small painted images (name: shape, width, height in
# px, colour), written once to the app's cache at 1x and 2x for sharp
# high-DPI screens.
_ARROWS = {
    "combo": ("down", 10, 6, TEXT_DIM),
    "combo_hover": ("down", 10, 6, TEXT),
    "combo_off": ("down", 10, 6, DISABLED_TEXT),
    "up": ("up", 8, 5, TEXT_DIM),
    "up_hover": ("up", 8, 5, TEXT),
    "up_off": ("up", 8, 5, DISABLED_TEXT),
    "down": ("down", 8, 5, TEXT_DIM),
    "down_hover": ("down", 8, 5, TEXT),
    "down_off": ("down", 8, 5, DISABLED_TEXT),
}


def _paint_arrow(path, shape: str, w: int, h: int, colour: str, scale: int) -> bool:
    """A filled triangle pointing `shape` ("up"/"down"), saved as a PNG."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPolygonF

    W, H = w * scale, h * scale
    img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(colour))
    pts = ([QPointF(0, 0), QPointF(W, 0), QPointF(W / 2, H)] if shape == "down"
           else [QPointF(0, H), QPointF(W, H), QPointF(W / 2, 0)])
    p.drawPolygon(QPolygonF(pts))
    p.end()
    return img.save(str(path), "PNG")


def _arrow_files(folder) -> dict[str, str]:
    """{name: path} of the arrow images in `folder`, painted when missing."""
    folder.mkdir(parents=True, exist_ok=True)
    url: dict[str, str] = {}
    for name, (shape, w, h, colour) in _ARROWS.items():
        stem = f"arrow-{shape}-{w}x{h}-{colour.lstrip('#')}"
        # QIcon (which loads style-sheet images) picks "<name>@2x.png" by
        # itself on a high-DPI screen
        for scale, suffix in ((1, ""), (2, "@2x")):
            f = folder / f"{stem}{suffix}.png"
            if not f.is_file() and not _paint_arrow(f, shape, w, h, colour, scale):
                raise OSError(f"could not write {f}")
        url[name] = (folder / f"{stem}.png").as_posix()
    return url


def _arrow_qss() -> str:
    """The arrow rules, pointing at the painted images (made when missing),
    in the app's data folder or else the system's temp folder. "" if
    neither can be written (the boxes then show no arrow at all)."""
    import tempfile
    from pathlib import Path

    from app.core.paths import app_data_dir
    url: dict[str, str] = {}
    for where in (lambda: app_data_dir() / "cache" / "ui",
                  lambda: Path(tempfile.gettempdir()) / "RecapStudio-ui"):
        try:
            url = _arrow_files(where())
            break
        except Exception as exc:  # noqa: BLE001 - a look, never a reason to fail
            from app.core.logger import get_logger
            get_logger("theme").warning("arrow images not made (%s)", exc)
    if not url:
        return ""
    return f"""
QComboBox::down-arrow {{
    image: url("{url['combo']}");
    width: 10px; height: 6px;
    margin-right: 6px;
}}
QComboBox::down-arrow:hover, QComboBox::down-arrow:on {{
    image: url("{url['combo_hover']}");
}}
QComboBox::down-arrow:disabled {{
    image: url("{url['combo_off']}");
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow, QTimeEdit::up-arrow {{
    image: url("{url['up']}");
    width: 8px; height: 5px;
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover,
QTimeEdit::up-arrow:hover {{
    image: url("{url['up_hover']}");
}}
QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off,
QTimeEdit::up-arrow:disabled, QTimeEdit::up-arrow:off {{
    image: url("{url['up_off']}");
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow, QTimeEdit::down-arrow {{
    image: url("{url['down']}");
    width: 8px; height: 5px;
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover,
QTimeEdit::down-arrow:hover {{
    image: url("{url['down_hover']}");
}}
QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off,
QTimeEdit::down-arrow:disabled, QTimeEdit::down-arrow:off {{
    image: url("{url['down_off']}");
}}
"""


def apply(app) -> None:
    """Apply Fusion style, a dark QPalette and the QSS to a QApplication."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")

    def c(hexstr: str) -> QColor:
        return QColor(hexstr)

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, c(BG))
    pal.setColor(QPalette.ColorRole.WindowText, c(TEXT))
    pal.setColor(QPalette.ColorRole.Base, c(PANEL))
    pal.setColor(QPalette.ColorRole.AlternateBase, c("#1b2129"))
    pal.setColor(QPalette.ColorRole.Text, c(TEXT))
    pal.setColor(QPalette.ColorRole.Button, c(HOVER))
    pal.setColor(QPalette.ColorRole.ButtonText, c(TEXT))
    pal.setColor(QPalette.ColorRole.BrightText, c("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight, c(SELECTION))
    pal.setColor(QPalette.ColorRole.HighlightedText, c("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, c(ACCENT))
    pal.setColor(QPalette.ColorRole.LinkVisited, c(ACCENT))
    pal.setColor(QPalette.ColorRole.ToolTipBase, c(PANEL))
    pal.setColor(QPalette.ColorRole.ToolTipText, c(TEXT))
    pal.setColor(QPalette.ColorRole.PlaceholderText, c(TEXT_DIM))
    for group in (QPalette.ColorGroup.Disabled,):
        pal.setColor(group, QPalette.ColorRole.WindowText, c(DISABLED_TEXT))
        pal.setColor(group, QPalette.ColorRole.Text, c(DISABLED_TEXT))
        pal.setColor(group, QPalette.ColorRole.ButtonText, c(DISABLED_TEXT))
        pal.setColor(group, QPalette.ColorRole.Highlight, c(BORDER))
        pal.setColor(group, QPalette.ColorRole.HighlightedText, c(TEXT_DIM))
    app.setPalette(pal)
    app.setStyleSheet(QSS + _arrow_qss())
