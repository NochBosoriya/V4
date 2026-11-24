"""Main application window: header tab bar, lazily built tab pages,
console dock, language switcher, status bar with GPU / activity info."""
from __future__ import annotations

import traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import get_language, set_language, tr
from app.core.logger import get_logger
from app.ui.context import AppContext
from app.ui.widgets import ConsoleDock, VideoPreview
from app.ui.workers import FnWorker
from app.version import BUILD_DATE, __version__

log = get_logger("ui.main_window")

# (tab key, tr key, english default, module name, class name) — order is fixed.
_TAB_SPECS: list[tuple[str, str, str, str]] = [
    ("tab.import", "Import", "app.ui.tabs.import_tab", "ImportTab"),
    # the Translate tab (SRT / narration -> voice track) was removed on
    # 2026-11-19: nothing of the series work used it (backup: Ai Tool\Backup\
    # translate-tab-2026-11-19)
    ("tab.dub", "Translate && Dub", "app.ui.tabs.dub_tab", "DubTab"),
    ("tab.recap", "Recap", "app.ui.tabs.recap_tab", "RecapTab"),
    ("tab.overlay", "Overlay", "app.ui.tabs.overlay_tab", "OverlayTab"),
    ("tab.dl", "DL", "app.ui.tabs.dl_tab", "DlTab"),
    ("tab.library", "Drama MG", "app.ui.tabs.library_tab", "LibraryTab"),
    ("tab.export", "Export", "app.ui.tabs.export_tab", "ExportTab"),
    ("tab.publish", "Publish", "app.ui.tabs.publish_tab", "PublishTab"),
    ("tab.settings", "Settings", "app.ui.tabs.settings_tab", "SettingsTab"),
]


class _ErrorPage(QWidget):
    """Shown in place of a tab whose module failed to import/construct."""

    def __init__(self, tab_label: str, error_text: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        title = QLabel(
            tr("err.tab_failed", "This tab failed to load:") + f" {tab_label}",
            self)
        title.setStyleSheet("font-size: 12pt; font-weight: 600; color: #f85149;")
        detail = QLabel(error_text, self)
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        detail.setStyleSheet("color: #8b949e; font-family: Consolas, monospace;")
        lay.addWidget(title)
        lay.addWidget(detail)
        lay.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        ctx.main = self

        self.setWindowTitle(f"RecapStudio {__version__}")
        self.resize(1440, 860)
        self.setMinimumSize(1100, 700)

        # Shared preview player, routed to by AppContext.open_in_preview.
        # Kept unparented-to-layout until a tab adopts it; hidden by default.
        self.preview = VideoPreview(self)
        self.preview.hide()

        # ---------------- tabs ----------------
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self._built: set[int] = set()
        for tr_key, default, _mod, _cls in _TAB_SPECS:
            placeholder = QWidget(self.tabs)
            self.tabs.addTab(placeholder, tr(tr_key, default))
        self.tabs.currentChanged.connect(self._ensure_tab_built)
        self.setCentralWidget(self.tabs)

        # ---------------- language switcher (tab-bar corner) ----------------
        corner = QWidget(self.tabs)
        corner_lay = QVBoxLayout(corner)
        corner_lay.setContentsMargins(4, 2, 8, 2)
        self.lang_combo = QComboBox(corner)
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("ខ្មែរ", "km")  # ខ្មែរ
        current = get_language()
        self.lang_combo.setCurrentIndex(1 if current == "km" else 0)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        corner_lay.addWidget(self.lang_combo)
        self.tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        # ---------------- console dock ----------------
        self.console = ConsoleDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console)
        toggle = QAction(tr("view.console", "Toggle Console"), self)
        toggle.setShortcut(QKeySequence("Ctrl+`"))
        toggle.triggered.connect(
            lambda: self.console.setVisible(not self.console.isVisible()))
        self.addAction(toggle)

        # ---------------- status bar ----------------
        sb = self.statusBar()
        self._gpu_lbl = QLabel(tr("status.gpu_detecting", "GPU: detecting..."), sb)
        self._activity_lbl = QLabel("", sb)
        sb.addWidget(self._gpu_lbl)
        sb.addPermanentWidget(self._activity_lbl)
        # the version, at the far right: a click opens About
        self._about_btn = QPushButton(f"v{__version__}", sb)
        self._about_btn.setFlat(True)
        self._about_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._about_btn.setStyleSheet("QPushButton { color: #8b949e; border: none;"
                                      " padding: 0 6px; }"
                                      "QPushButton:hover { color: #58a6ff; }")
        self._about_btn.setToolTip(tr("about.title", "About RecapStudio"))
        self._about_btn.clicked.connect(self.show_about)
        sb.addPermanentWidget(self._about_btn)

        self._gpu_worker: FnWorker | None = None
        self._startup_report: dict | None = None
        self._report_waiters: list = []
        self._start_gpu_probe()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_activity)
        self._status_timer.start()

        # Build the first tab now.
        self._ensure_tab_built(0)

    def show_about(self) -> None:
        """The version, build date, Python / Qt and where the data is."""
        import html
        import sys

        from PySide6 import __version__ as qt_version

        from app.core import paths
        try:
            data = str(paths.app_data_dir())
        except Exception:
            data = "?"
        text = (f"<h3>RecapStudio {__version__}</h3>"
                f"<p>{html.escape(tr('about.built', 'Built {date}').replace('{date}', BUILD_DATE))}</p>"
                f"<p>{html.escape(tr('about.desc', 'Recap, translate and dub videos.'))}</p>"
                f"<p>Python {sys.version.split()[0]} · PySide6 {qt_version}</p>"
                f"<p>{html.escape(tr('about.data', 'Your data (settings, projects, library):'))}"
                f"<br>{html.escape(data)}</p>")
        QMessageBox.about(self, tr("about.title", "About RecapStudio"), text)

    # ------------------------------------------------------------------
    # lazy tab construction
    # ------------------------------------------------------------------
    def _ensure_tab_built(self, index: int) -> None:
        if index < 0 or index in self._built or index >= len(_TAB_SPECS):
            return
        self._built.add(index)
        tr_key, default, mod_name, cls_name = _TAB_SPECS[index]
        label = tr(tr_key, default)
        try:
            import importlib
            module = importlib.import_module(mod_name)
            page_cls = getattr(module, cls_name)
            page = page_cls(self.ctx)
        except Exception:
            err = traceback.format_exc()
            log.error("tab %s failed to load:\n%s", cls_name, err)
            page = _ErrorPage(label, err, self.tabs)
        old = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, page, label)
        self.tabs.setCurrentIndex(index)
        if old is not None:
            old.deleteLater()

    def select_tab_by_key(self, key: str) -> bool:
        """Switch to a tab by short key ("dub") or tr key ("tab.dub")."""
        want = key if key.startswith("tab.") else f"tab.{key}"
        for i, (tr_key, _default, _mod, _cls) in enumerate(_TAB_SPECS):
            if tr_key == want:
                self._ensure_tab_built(i)
                self.tabs.setCurrentIndex(i)
                return True
        log.warning("select_tab_by_key: unknown tab %r", key)
        return False

    def tab_widget_by_key(self, key: str) -> QWidget | None:
        """Return the (built) page widget for a tab key, or None."""
        want = key if key.startswith("tab.") else f"tab.{key}"
        for i, (tr_key, _default, _mod, _cls) in enumerate(_TAB_SPECS):
            if tr_key == want:
                current = self.tabs.currentIndex()
                self._ensure_tab_built(i)
                self.tabs.setCurrentIndex(current)
                return self.tabs.widget(i)
        return None

    # ------------------------------------------------------------------
    # language
    # ------------------------------------------------------------------
    def _on_language_changed(self, _index: int) -> None:
        code = self.lang_combo.currentData()
        old = get_language()
        if code == old:
            return
        set_language(code)
        self.retranslate_ui(old, code)

    def retranslate_ui(self, old: str, new: str) -> None:
        """Show the whole interface in `new` now — no restart. Every open
        window's texts are converted; tabs that build texts from data
        (task lists, status panels) re-fill them in retranslate()."""
        from PySide6.QtWidgets import QApplication

        from app.ui.retranslate import retranslate_tree
        self.setUpdatesEnabled(False)
        try:
            n = 0
            for top in QApplication.topLevelWidgets():
                n += retranslate_tree(top, old, new)
            for i, (tr_key, default, _mod, _cls) in enumerate(_TAB_SPECS):
                if i < self.tabs.count():
                    self.tabs.setTabText(i, tr(tr_key, default))
            for i in range(self.tabs.count()):
                hook = getattr(self.tabs.widget(i), "retranslate", None)
                if callable(hook):
                    try:
                        hook()
                    except Exception:
                        log.exception("retranslate of tab %d failed", i)
            if self._startup_report is not None:
                self._on_gpu_report(self._startup_report)
            else:
                self._gpu_lbl.setText(tr("status.gpu_detecting", "GPU: detecting..."))
            self._refresh_activity()
        finally:
            self.setUpdatesEnabled(True)
        log.info("interface language: %s (%d texts switched, no restart)", new, n)

    # ------------------------------------------------------------------
    # status bar helpers
    # ------------------------------------------------------------------
    def _start_gpu_probe(self) -> None:
        def probe() -> dict:
            from app.core.model_manager import startup_report
            return startup_report()

        def failed(en: str, _km: str) -> None:
            log.warning("startup report failed: %s", en)
            self._gpu_lbl.setText(tr("status.gpu_unknown", "GPU: unknown"))
            self._report_waiters.clear()

        worker = FnWorker(probe)
        worker.finished_ok.connect(self._on_gpu_report)
        worker.failed.connect(failed)
        self._gpu_worker = worker
        self.ctx.pool.start(worker)

    def when_report(self, fn) -> None:
        """Call fn(report) with the startup environment report (GPU, ffmpeg,
        models) once it is ready — now if it already is. One probe serves the
        status bar and the startup check dialog."""
        if self._startup_report is not None:
            fn(self._startup_report)
        else:
            self._report_waiters.append(fn)

    def _on_gpu_report(self, report: object) -> None:
        try:
            rep = dict(report) if isinstance(report, dict) else {}
        except Exception:
            rep = {}
        self._startup_report = rep
        waiters, self._report_waiters = self._report_waiters, []
        for fn in waiters:
            try:
                fn(rep)
            except Exception:
                log.exception("startup report consumer failed")
        gpu = rep.get("gpu")
        if gpu:
            vram = rep.get("vram_gb")
            text = f"GPU: {gpu} ({vram:.0f} GB)" if vram else f"GPU: {gpu}"
            acc = rep.get("accel") or {}
            if acc:
                ai = ("CUDA" if "CUDA" in (acc.get("torch"), acc.get("whisper"))
                      else tr("status.ai_cpu", "CPU"))
                text += "  ·  " + tr("status.ai", "AI") + f": {ai}"
            self._gpu_lbl.setText(text)
            names = [g.get("name", "") for g in rep.get("gpus") or []]
            tip = "\n".join(names)
            if rep.get("accel_hints"):
                tip += "\n\n" + "\n".join(rep["accel_hints"])
            self._gpu_lbl.setToolTip(tip)
        else:
            self._gpu_lbl.setText(tr("status.gpu_none", "GPU: none (CPU mode)"))

    def _refresh_activity(self) -> None:
        parts: list[str] = []
        try:
            items = getattr(self.ctx.downloads, "items", None)
            if callable(items):
                items = items()
            if isinstance(items, dict):
                # {id: DownloadItem}; copied because download threads add to it
                items = list(items.values())
            if items:
                active = sum(
                    1 for it in items
                    if getattr(it, "status", "") in
                    ("queued", "fetching_meta", "downloading"))
                if active:
                    parts.append(
                        tr("status.downloads", "Downloads") + f": {active}")
        except Exception:
            pass
        try:
            n = self.ctx.pool.active_count()
            if n:
                parts.append(tr("status.tasks", "Tasks") + f": {n}")
        except Exception:
            pass
        self._activity_lbl.setText("   ".join(parts) if parts
                                   else tr("status.idle", "Idle"))

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not getattr(self, "_docks_sized", False):
            # a compact console to start with (four log lines; drag to
            # resize): the Dub tab's monitor needs the height
            self._docks_sized = True
            QTimer.singleShot(0, lambda: self.resizeDocks([self.console], [120],
                                                          Qt.Orientation.Vertical))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # a long dub must not end on one stray click: ask first (its
        # finished work is saved either way)
        for i in range(self.tabs.count()):
            busy = getattr(self.tabs.widget(i), "busy_reason", None)
            reason = busy() if callable(busy) else ""
            if reason:
                ans = QMessageBox.question(
                    self, "RecapStudio",
                    reason + "\n\n" + tr("main.close_busy",
                                         "Close anyway? Everything finished so far is saved; "
                                         "you can continue later."),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if ans != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                break
        try:
            self._status_timer.stop()
            # stop yt-dlp too: its worker threads would otherwise keep the
            # process alive (window gone) until every download finished
            self.ctx.downloads.shutdown()
            self.ctx.pool.cancel_all()
            self.ctx.pool.wait_all(3000)
        except Exception:
            pass
        super().closeEvent(event)
