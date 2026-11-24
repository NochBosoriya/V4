"""Import tab: drop or browse videos/images, probe media info, add to library.

Videos become ``ctx.current_video`` (probed via media_exporter.probe in a
worker, opened in the main preview). Images become ``ctx.poster_path`` with a
shortcut button to the Overlay tab's Poster Editor.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.errors import AppError
from app.core.i18n import pick, tr
from app.core.logger import get_logger
from app.ui.workers import FnWorker

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.import_tab")

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
ALL_EXTS = VIDEO_EXTS + IMAGE_EXTS

RECENT_KEY = "recent.files"
RECENT_MAX = 12


# ---------------------------------------------------------------------------
# Shared defensive helpers (ProgressRow / DropArea APIs are built in parallel)
# ---------------------------------------------------------------------------

def _msg(en: str, km: str = "") -> str:
    """The message in the UI language (English when there is no Khmer)."""
    return pick(en, km)


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
        except TypeError:
            pass
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
    row._rs_worker = worker  # latest worker owns the cancel button
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


class _FallbackDrop(QLabel):
    """The drop zone: a dashed box with the hint text; emits the dropped files."""

    dropped = Signal(list)

    def __init__(self, text: str, exts: tuple):
        super().__init__(text)
        self._exts = tuple(e.lower() for e in exts)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(140)
        self.setStyleSheet(
            "QLabel { border: 2px dashed #30363d; border-radius: 6px;"
            " color: #8b949e; background: #161b22; font-size: 14px; }"
        )

    def dragEnterEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):  # noqa: N802
        files = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        files = [f for f in files if f.lower().endswith(self._exts)]
        if files:
            self.dropped.emit(files)
        ev.acceptProposedAction()


def _make_drop_area(text: str, exts: tuple, on_files: Callable[[list], None]) -> QWidget:
    """The drop zone: a dashed box that says what to drop. (widgets.DropArea
    draws nothing and has no text, so the zone used to be an invisible band.)"""
    fb = _FallbackDrop(text, exts)
    fb.dropped.connect(lambda files: on_files(list(files)))
    return fb


def _fmt_duration(seconds: float) -> str:
    try:
        s = int(round(float(seconds)))
    except Exception:
        s = 0
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _probe_media(path: str):
    """Worker fn: probe media info (lazy engine import)."""
    from app.engines import media_exporter
    return media_exporter.probe(path)


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class ImportTab(QWidget):
    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self._workers: list = []
        self._last_info = None  # MediaInfo of the current video

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --- hero drop area -------------------------------------------------
        self.drop = _make_drop_area(
            tr("import.drop_hint",
               "Drop a video (mp4 / mkv / mov / webm / avi) or a poster image "
               "(jpg / png / webp) here"),
            ALL_EXTS,
            self._on_files,
        )
        root.addWidget(self.drop)

        browse_row = QHBoxLayout()
        self.browse_btn = QPushButton(tr("import.browse", "Browse…"))
        self.browse_btn.clicked.connect(self._on_browse)
        browse_row.addWidget(self.browse_btn)
        browse_row.addStretch(1)
        root.addLayout(browse_row)

        # --- probe progress -------------------------------------------------
        self.probe_row = _make_progress_row(tr("import.probing", "Probing media…"))
        root.addWidget(self.probe_row)

        # --- info card ------------------------------------------------------
        self.info_card = QFrame()
        self.info_card.setObjectName("card")
        self.info_card.setStyleSheet(
            "QFrame#card { background: #161b22; border: 1px solid #30363d;"
            " border-radius: 6px; }"
        )
        grid = QGridLayout(self.info_card)
        grid.setContentsMargins(12, 10, 12, 10)
        self.info_title = QLabel("")
        self.info_title.setStyleSheet("font-weight: 600; color: #58a6ff;")
        grid.addWidget(self.info_title, 0, 0, 1, 4)
        self._info_labels: dict[str, QLabel] = {}
        fields = [
            ("duration", tr("import.duration", "Duration")),
            ("resolution", tr("import.resolution", "Resolution")),
            ("fps", tr("import.fps", "FPS")),
            ("vcodec", tr("import.vcodec", "Video codec")),
            ("acodec", tr("import.acodec", "Audio codec")),
            ("bitrate", tr("import.bitrate", "Bitrate")),
        ]
        for i, (key, label) in enumerate(fields):
            r, c = divmod(i, 2)
            cap = QLabel(label + ":")
            cap.setStyleSheet("color: #8b949e;")
            val = QLabel("—")
            grid.addWidget(cap, r + 1, c * 2)
            grid.addWidget(val, r + 1, c * 2 + 1)
            self._info_labels[key] = val
        self.info_card.setVisible(False)
        root.addWidget(self.info_card)

        # --- add to library -------------------------------------------------
        self.lib_group = QGroupBox(tr("import.add_library", "Add to Library"))
        lib_row = QHBoxLayout(self.lib_group)
        lib_row.addWidget(QLabel(tr("import.series", "Series:")))
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(160)
        lib_row.addWidget(self.series_combo)
        lib_row.addWidget(QLabel(tr("import.or_new", "or new:")))
        self.series_edit = QLineEdit()
        self.series_edit.setPlaceholderText(tr("import.new_series", "New series name"))
        lib_row.addWidget(self.series_edit, 1)
        self.add_lib_btn = QPushButton(tr("import.add_btn", "Add to Library"))
        self.add_lib_btn.clicked.connect(self._on_add_to_library)
        lib_row.addWidget(self.add_lib_btn)
        self.lib_group.setVisible(False)
        root.addWidget(self.lib_group)

        # --- image / poster hint --------------------------------------------
        self.poster_frame = QFrame()
        self.poster_frame.setStyleSheet(
            "QFrame { background: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
        )
        p_row = QHBoxLayout(self.poster_frame)
        p_row.setContentsMargins(12, 8, 12, 8)
        self.poster_label = QLabel("")
        p_row.addWidget(self.poster_label, 1)
        self.poster_btn = QPushButton(
            tr("import.open_poster_editor", "Open in Overlay > Poster Editor"))
        self.poster_btn.clicked.connect(self._on_open_poster_editor)
        p_row.addWidget(self.poster_btn)
        self.poster_frame.setVisible(False)
        root.addWidget(self.poster_frame)

        # --- recent files ---------------------------------------------------
        recent_cap = QLabel(tr("import.recent", "Recent files"))
        recent_cap.setStyleSheet("color: #8b949e; font-weight: 600;")
        root.addWidget(recent_cap)
        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._on_recent_open)
        root.addWidget(self.recent_list, 1)

        self._reload_series()
        self._reload_recent()

    # ------------------------------------------------------------------ util

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

    def _run_task(self, row, fn, *args, on_ok=None, running_text="", **kwargs):
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
                except Exception as e:  # never crash the UI thread
                    log.exception("post-task handler failed")
                    QMessageBox.warning(self, tr("err.title", "Error"), str(e))

        def _fail(en, km):
            _row_progress(row, 0.0, tr("common.failed", "Failed"))
            QMessageBox.warning(self, tr("err.title", "Error"), _msg(en, km))

        worker.finished_ok.connect(_ok)
        worker.failed.connect(_fail)
        self._launch(worker)
        return worker

    # ------------------------------------------------------------- file flow

    def _on_browse(self):
        filt = (
            tr("import.filter_media", "Media files")
            + " (*.mp4 *.mkv *.mov *.webm *.avi *.jpg *.jpeg *.png *.webp);;"
            + tr("import.filter_video", "Videos") + " (*.mp4 *.mkv *.mov *.webm *.avi);;"
            + tr("import.filter_image", "Images") + " (*.jpg *.jpeg *.png *.webp)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, tr("import.pick_file", "Choose a video or image"), "", filt)
        if path:
            self._on_files([path])

    def _on_files(self, files: list):
        if not files:
            return
        path = str(files[0])
        low = path.lower()
        if low.endswith(VIDEO_EXTS):
            self._handle_video(path)
        elif low.endswith(IMAGE_EXTS):
            self._handle_image(path)
        else:
            QMessageBox.information(
                self, tr("import.title", "Import"),
                _msg("Unsupported file type.",
                     "ប្រភេទឯកសារនេះមិនត្រូវបានគាំទ្រទេ។"))

    def _handle_video(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(
                self, tr("err.title", "Error"),
                _msg(f"File not found: {path}",
                     f"រកមិនឃើញឯកសារ៖ {path}"))
            return
        self.ctx.current_video = path
        self.poster_frame.setVisible(False)
        self.info_title.setText(os.path.basename(path))
        self.info_card.setVisible(True)
        for lab in self._info_labels.values():
            lab.setText("…")
        self.lib_group.setVisible(True)
        self._push_recent(path)
        try:
            self.ctx.open_in_preview(path)
        except Exception:
            log.exception("open_in_preview failed")
        self._run_task(
            self.probe_row, _probe_media, path,
            on_ok=self._show_info,
            running_text=tr("import.probing", "Probing media…"),
        )

    def _show_info(self, info):
        self._last_info = info
        try:
            self._info_labels["duration"].setText(_fmt_duration(info.duration))
            self._info_labels["resolution"].setText(f"{info.width} × {info.height}")
            self._info_labels["fps"].setText(f"{info.fps:.3f}".rstrip("0").rstrip("."))
            self._info_labels["vcodec"].setText(info.vcodec or "—")
            self._info_labels["acodec"].setText(info.acodec or "—")
            br = getattr(info, "bitrate", 0) or 0
            self._info_labels["bitrate"].setText(
                f"{br / 1_000_000:.2f} Mbps" if br else "—")
        except Exception:
            log.exception("could not display media info")

    def _handle_image(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(
                self, tr("err.title", "Error"),
                _msg(f"File not found: {path}",
                     f"រកមិនឃើញឯកសារ៖ {path}"))
            return
        setattr(self.ctx, "poster_path", path)
        self.poster_label.setText(
            tr("import.poster_loaded", "Poster image loaded:")
            + " " + os.path.basename(path))
        self.poster_frame.setVisible(True)
        self._push_recent(path)

    def _on_open_poster_editor(self):
        main = getattr(self.ctx, "main", None)
        fn = getattr(main, "select_tab_by_key", None) if main is not None else None
        if callable(fn):
            try:
                fn("overlay")
                return
            except Exception:
                log.exception("select_tab_by_key failed")
        QMessageBox.information(
            self, tr("import.poster", "Poster"),
            _msg("Open the Overlay tab and use the Poster Editor sub-tab "
                 "to edit this image.",
                 "សូមបើកផ្ទាំង Overlay ហើយប្រើផ្នែក Poster Editor "
                 "ដើម្បីកែរូបភាពនេះ។"))

    # ---------------------------------------------------------------- library

    def _reload_series(self):
        self.series_combo.clear()
        self.series_combo.addItem(tr("import.select_series", "— select —"), None)
        try:
            for s in self.ctx.library.get_series():
                sid = s.get("id", s.get("series_id"))
                name = s.get("name", s.get("title", str(sid)))
                self.series_combo.addItem(str(name), sid)
        except Exception:
            log.exception("could not load series list")

    def _on_add_to_library(self):
        video = self.ctx.current_video
        if not video:
            QMessageBox.information(
                self, tr("import.title", "Import"),
                _msg("No video selected.", "មិនទាន់បានជ្រើសរើសវីដេអូទេ។"))
            return
        new_name = self.series_edit.text().strip()
        try:
            if new_name:
                sid = self.ctx.library.add_series(new_name)
            else:
                sid = self.series_combo.currentData()
            if sid is None:
                QMessageBox.information(
                    self, tr("import.title", "Import"),
                    _msg("Choose a series or type a new series name.",
                         "សូមជ្រើសរើសរឿង ឬវាយឈ្មោះរឿងថ្មី។"))
                return
            info = self._last_info
            self.ctx.library.add_episode(
                int(sid),
                os.path.splitext(os.path.basename(video))[0],
                video,
                duration=float(getattr(info, "duration", 0.0) or 0.0),
                width=int(getattr(info, "width", 0) or 0),
                height=int(getattr(info, "height", 0) or 0),
            )
        except AppError as e:
            QMessageBox.warning(self, tr("err.title", "Error"), _msg(e.en, e.km))
            return
        except Exception as e:
            log.exception("add to library failed")
            QMessageBox.warning(self, tr("err.title", "Error"), str(e))
            return
        self.series_edit.clear()
        self._reload_series()
        QMessageBox.information(
            self, tr("import.title", "Import"),
            _msg("Episode added to the library.",
                 "បានបញ្ចូលភាគចូលបណ្ណាល័យរួចរាល់។"))

    # ----------------------------------------------------------------- recent

    def _get_recent(self) -> list:
        try:
            val = self.ctx.config.get(RECENT_KEY, [])
            if isinstance(val, list):
                return [str(p) for p in val]
        except Exception:
            log.exception("could not read recent files")
        return []

    def _push_recent(self, path: str):
        items = self._get_recent()
        items = [p for p in items if p != path]
        items.insert(0, path)
        items = items[:RECENT_MAX]
        try:
            self.ctx.config.set(RECENT_KEY, items)
        except Exception:
            log.exception("could not save recent files")
        self._reload_recent(items)

    def _reload_recent(self, items: list | None = None):
        if items is None:
            items = self._get_recent()
        self.recent_list.clear()
        for p in items:
            it = QListWidgetItem(os.path.basename(p))
            it.setToolTip(p)
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.recent_list.addItem(it)

    def _on_recent_open(self, item: QListWidgetItem):
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        if not os.path.exists(path):
            items = [p for p in self._get_recent() if p != path]
            try:
                self.ctx.config.set(RECENT_KEY, items)
            except Exception:
                pass
            self._reload_recent(items)
            QMessageBox.information(
                self, tr("import.title", "Import"),
                _msg("File no longer exists; removed from recent list.",
                     "ឯកសារនេះលែងមានទៀតហើយ ត្រូវបានដកចេញពីបញ្ជីថ្មីៗ។"))
            return
        self._on_files([path])
