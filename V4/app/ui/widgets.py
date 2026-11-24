"""Reusable UI widgets: console dock, video preview, timeline grid,
drag-and-drop area and progress rows.

Only this layer touches PySide6. Engine modules are imported lazily inside
methods so a missing heavy dependency degrades gracefully instead of killing
the UI at import time.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import pick, tr, tr_text
from app.core.logger import LOG_BUS, get_logger

log = get_logger("ui.widgets")

WARN_BG = "#f8514966"  # translucent red for overlong timeline rows


def scroll_page(page: QWidget) -> QVBoxLayout:
    """Give `page` a frameless body that scrolls up and down, and return the
    body's layout: in a short window the page scrolls instead of squeezing
    its fields until they overlap (Export, Publish at 1100x700)."""
    from PySide6.QtWidgets import QFrame, QScrollArea
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea(page)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QWidget()
    scroll.setWidget(body)
    outer.addWidget(scroll)
    return QVBoxLayout(body)


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------

def fmt_mmss(seconds: float) -> str:
    """Format seconds as mm:ss (rounded down)."""
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def fmt_mmss_d(seconds: float) -> str:
    """Format seconds as mm:ss.d (one decimal)."""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:04.1f}"


def parse_mmss_d(text: str) -> float:
    """Parse 'mm:ss', 'mm:ss.d', 'h:mm:ss' or plain seconds into seconds.

    Raises ValueError on garbage.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty time")
    parts = text.split(":")
    if len(parts) == 1:
        return max(0.0, float(parts[0].replace(",", ".")))
    total = 0.0
    for p in parts:
        total = total * 60.0 + float(p.strip().replace(",", ".") or 0.0)
    return max(0.0, total)


# ---------------------------------------------------------------------------
# ConsoleDock
# ---------------------------------------------------------------------------

class ConsoleDock(QDockWidget):
    """Bottom dock showing LOG_BUS lines. LOG_BUS publishes from arbitrary
    threads, so lines are bridged through a queued Signal to the UI thread."""

    _line_received = Signal(str)

    MAX_BLOCKS = 5000

    def __init__(self, parent: QWidget | None = None):
        super().__init__(tr("console.title", "Console"), parent)
        self.setObjectName("ConsoleDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        body = QWidget(self)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        self._clear_btn = QPushButton(tr("console.clear", "Clear"), body)
        self._clear_btn.setMaximumWidth(90)
        self._autoscroll = QCheckBox(tr("console.autoscroll", "Auto-scroll"), body)
        self._autoscroll.setChecked(True)
        bar.addWidget(self._clear_btn)
        bar.addWidget(self._autoscroll)
        bar.addStretch(1)
        lay.addLayout(bar)

        self._text = QPlainTextEdit(body)
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self.MAX_BLOCKS)
        self._text.setMinimumHeight(40)     # the dock can start small (MainWindow)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self._text.setFont(font)
        self._text.setStyleSheet(
            "QPlainTextEdit { background-color: #0d1117; color: #e6edf3;"
            " border: 1px solid #30363d; border-radius: 6px; }"
        )
        lay.addWidget(self._text, 1)
        self.setWidget(body)

        self._clear_btn.clicked.connect(self._text.clear)
        # Queued connection: publisher may be any thread.
        self._line_received.connect(self._append_line,
                                    Qt.ConnectionType.QueuedConnection)
        LOG_BUS.subscribe(self._on_bus_line)
        for line in LOG_BUS.tail(500):
            self._text.appendPlainText(line)
        self.destroyed.connect(lambda *_: LOG_BUS.unsubscribe(self._on_bus_line))

    # called from arbitrary threads
    def _on_bus_line(self, line: str) -> None:
        self._line_received.emit(line)

    @Slot(str)
    def _append_line(self, line: str) -> None:
        self._text.appendPlainText(line)
        if self._autoscroll.isChecked():
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    # No closeEvent unsubscribe: the dock's X only hides it (Ctrl+` shows it
    # again), so it must keep receiving lines; `destroyed` unsubscribes.


# ---------------------------------------------------------------------------
# VideoPreview
# ---------------------------------------------------------------------------

class VideoPreview(QWidget):
    """Small embedded media player: video widget, play/pause, seek slider,
    duration label (mm:ss / mm:ss)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setVolume(0.8)
        self._player.setAudioOutput(self._audio)

        self._video = QVideoWidget(self)
        self._video.setMinimumHeight(180)
        self._player.setVideoOutput(self._video)

        self._play_btn = QPushButton(tr("preview.play", "Play"), self)
        self._play_btn.setMaximumWidth(80)
        self._play_btn.setEnabled(False)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)

        self._time_lbl = QLabel("00:00 / 00:00", self)
        self._time_lbl.setMinimumWidth(96)
        self._time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._video, 1)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._play_btn)
        row.addWidget(self._slider, 1)
        row.addWidget(self._time_lbl)
        lay.addLayout(row)

        self._play_btn.clicked.connect(self.toggle_play)
        self._slider.sliderMoved.connect(self._player.setPosition)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.errorOccurred.connect(self._on_error)

    # ------------------------------------------------------------------
    def open(self, path: str) -> None:
        """Load a media file and show its first frame."""
        if not path:
            return
        if not os.path.exists(path):
            log.warning("preview: file not found: %s", path)
            QMessageBox.warning(
                self, tr("preview.title", "Preview"),
                tr("preview.not_found", "File not found:") + f"\n{path}")
            return
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(path))
        self._play_btn.setEnabled(True)
        self._slider.setEnabled(True)
        self._player.pause()  # decode first frame

    def toggle_play(self) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def stop(self) -> None:
        self._player.stop()

    def current_path(self) -> str:
        return self._player.source().toLocalFile()

    def seek(self, seconds: float, play: bool = False) -> None:
        """Jump to `seconds`; optionally start playback."""
        self._player.setPosition(max(0, int(float(seconds) * 1000)))
        if play:
            self._player.play()

    # ------------------------------------------------------------------
    def _on_state(self, state) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText(tr("preview.pause", "Pause") if playing
                               else tr("preview.play", "Play"))

    def _on_position(self, pos_ms: int) -> None:
        if not self._slider.isSliderDown():
            self._slider.setValue(pos_ms)
        self._update_time(pos_ms, self._player.duration())

    def _on_duration(self, dur_ms: int) -> None:
        self._slider.setRange(0, max(0, dur_ms))
        self._update_time(self._player.position(), dur_ms)

    def _update_time(self, pos_ms: int, dur_ms: int) -> None:
        self._time_lbl.setText(
            f"{fmt_mmss(pos_ms / 1000.0)} / {fmt_mmss(max(0, dur_ms) / 1000.0)}")

    def _on_error(self, _error, error_string: str) -> None:
        if error_string:
            log.error("media player error: %s", error_string)
            LOG_BUS.publish(f"[preview] {error_string}")


# ---------------------------------------------------------------------------
# TimelineGrid
# ---------------------------------------------------------------------------

class TimelineGrid(QTableWidget):
    """Editable table of recap TimelineRows.

    Columns: Start | End | Title | Narration | Visual Plan | On-screen text.
    Start/End render as mm:ss.d and stay editable. After any edit the speech
    timing is re-checked via recap_engine.check_row_timing and overlong rows
    get a translucent red background + tooltip.
    """

    rowsChanged = Signal()

    COL_START, COL_END, COL_TITLE, COL_NARR, COL_VISUAL, COL_ONSCREEN = range(6)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(0, 6, parent)
        self.setHorizontalHeaderLabels([
            tr("timeline.start", "Start"),
            tr("timeline.end", "End"),
            tr("timeline.title", "Title"),
            tr("timeline.narration", "Narration"),
            tr("timeline.visual_plan", "Visual Plan"),
            tr("timeline.onscreen", "On-screen text"),
        ])
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.setWordWrap(True)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self.COL_NARR, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(self.COL_START, 80)
        self.setColumnWidth(self.COL_END, 80)
        self.setColumnWidth(self.COL_TITLE, 160)
        self.setColumnWidth(self.COL_VISUAL, 200)
        self.setColumnWidth(self.COL_ONSCREEN, 160)
        self.verticalHeader().setDefaultSectionSize(28)

        self._loading = False
        self.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    def set_rows(self, rows: list) -> None:
        """Populate from list[app.engines.recap_engine.TimelineRow]."""
        self._loading = True
        try:
            self.setRowCount(0)
            self.setRowCount(len(rows))
            for i, row in enumerate(rows):
                self._set_cell(i, self.COL_START, fmt_mmss_d(getattr(row, "start", 0.0)))
                self._set_cell(i, self.COL_END, fmt_mmss_d(getattr(row, "end", 0.0)))
                self._set_cell(i, self.COL_TITLE, getattr(row, "title", ""))
                self._set_cell(i, self.COL_NARR, getattr(row, "narration", ""))
                self._set_cell(i, self.COL_VISUAL, getattr(row, "visual_plan", ""))
                self._set_cell(i, self.COL_ONSCREEN, getattr(row, "onscreen_text", ""))
            for i in range(self.rowCount()):
                self._recheck_row(i)
        finally:
            self._loading = False
        self.rowsChanged.emit()

    def _set_cell(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(str(text))
        self.setItem(row, col, item)

    def rows(self) -> list:
        """Convert the table contents back to list[TimelineRow]."""
        from app.engines.recap_engine import TimelineRow
        out: list = []
        for i in range(self.rowCount()):
            out.append(TimelineRow(
                start=self._cell_seconds(i, self.COL_START),
                end=self._cell_seconds(i, self.COL_END),
                title=self._cell_text(i, self.COL_TITLE),
                narration=self._cell_text(i, self.COL_NARR),
                visual_plan=self._cell_text(i, self.COL_VISUAL),
                onscreen_text=self._cell_text(i, self.COL_ONSCREEN),
            ))
        return out

    def _cell_text(self, row: int, col: int) -> str:
        item = self.item(row, col)
        return item.text() if item is not None else ""

    def _cell_seconds(self, row: int, col: int) -> float:
        try:
            return parse_mmss_d(self._cell_text(row, col))
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        row = item.row()
        col = item.column()
        if col in (self.COL_START, self.COL_END):
            # normalize the time text back to mm:ss.d
            try:
                secs = parse_mmss_d(item.text())
            except (ValueError, TypeError):
                secs = 0.0
            self._loading = True
            try:
                item.setText(fmt_mmss_d(secs))
            finally:
                self._loading = False
        self._recheck_row(row)
        self.rowsChanged.emit()

    def _recheck_row(self, row: int) -> None:
        """Recompute speech timing for a row; paint warning bg if overlong."""
        fits, needed = True, 0.0
        try:
            from app.core.i18n import get_language
            from app.engines.recap_engine import TimelineRow, check_row_timing
            tl_row = TimelineRow(
                start=self._cell_seconds(row, self.COL_START),
                end=self._cell_seconds(row, self.COL_END),
                title=self._cell_text(row, self.COL_TITLE),
                narration=self._cell_text(row, self.COL_NARR),
                visual_plan=self._cell_text(row, self.COL_VISUAL),
                onscreen_text=self._cell_text(row, self.COL_ONSCREEN),
            )
            fits, needed = check_row_timing(tl_row, get_language())
        except Exception as e:  # engine not built / import issue: no warning paint
            log.debug("timing check unavailable: %r", e)
            return

        slot = max(0.0, self._cell_seconds(row, self.COL_END)
                   - self._cell_seconds(row, self.COL_START))
        warn = not fits
        brush = QBrush(QColor(WARN_BG)) if warn else QBrush(Qt.BrushStyle.NoBrush)
        tip = ""
        if warn:
            tip = (tr("timeline.overlong",
                      "Narration too long: needs {needed}s, slot is {slot}s")
                   .replace("{needed}", f"{needed:.1f}")
                   .replace("{slot}", f"{slot:.1f}"))
        self._loading = True
        try:
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item is None:
                    item = QTableWidgetItem("")
                    self.setItem(row, col, item)
                item.setBackground(brush)
                item.setToolTip(tip)
        finally:
            self._loading = False


# ---------------------------------------------------------------------------
# DropArea
# ---------------------------------------------------------------------------

class DropArea(QWidget):
    """Widget that accepts file drops filtered by extension.

    Use directly, subclass, or embed. Emits filesDropped(list[str]).
    ``extensions`` like [".mp4", ".mkv"]; empty list accepts any file.
    """

    filesDropped = Signal(list)

    def __init__(self, extensions: list[str] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.extensions = [e.lower() if e.startswith(".") else "." + e.lower()
                           for e in (extensions or [])]
        self.setAcceptDrops(True)

    def set_extensions(self, extensions: list[str]) -> None:
        self.extensions = [e.lower() if e.startswith(".") else "." + e.lower()
                           for e in (extensions or [])]

    def _accepted_paths(self, mime) -> list[str]:
        if not mime.hasUrls():
            return []
        paths: list[str] = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if not p:
                continue
            if os.path.isdir(p):
                paths.append(p)
                continue
            if not self.extensions or os.path.splitext(p)[1].lower() in self.extensions:
                paths.append(p)
        return paths

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._accepted_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._accepted_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = self._accepted_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.filesDropped.emit(paths)
        else:
            event.ignore()


# ---------------------------------------------------------------------------
# ProgressRow
# ---------------------------------------------------------------------------

class ProgressRow(QWidget):
    """A single running task: label + progress bar + cancel button.

    Call set_worker(FnWorker) BEFORE starting it (or right after); the row
    auto-connects to progress/finished_ok/failed/cancelled, hides itself when
    done and pops a QMessageBox (in the UI language) on failure.
    """

    done = Signal()

    def __init__(self, label: str = "", parent: QWidget | None = None,
                 hide_on_done: bool = True):
        super().__init__(parent)
        self._hide_on_done = hide_on_done
        self._worker = None

        self._label = QLabel(label, self)
        self._label.setMinimumWidth(120)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._status = QLabel("", self)
        self._status.setStyleSheet("color: #8b949e;")
        self._cancel_btn = QPushButton(tr("common.cancel", "Cancel"), self)
        self._cancel_btn.setMaximumWidth(90)
        self._cancel_btn.setEnabled(False)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)
        lay.addWidget(self._label)
        lay.addWidget(self._bar, 2)
        lay.addWidget(self._status, 1)
        lay.addWidget(self._cancel_btn)

        self._cancel_btn.clicked.connect(self._on_cancel_clicked)

    # ------------------------------------------------------------------
    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def set_worker(self, worker) -> None:
        """Wire an FnWorker's signals to this row."""
        self._worker = worker
        self._cancel_btn.setEnabled(True)
        self._bar.setValue(0)
        self._status.setText("")
        self.show()
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_finished_ok)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)

    def worker(self):
        return self._worker

    # ------------------------------------------------------------------
    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            self._cancel_btn.setEnabled(False)
            self._status.setText(tr("common.cancelling", "Cancelling..."))
            try:
                self._worker.cancel()
            except Exception:
                pass

    @Slot(float, str)
    def _on_progress(self, fraction: float, text: str) -> None:
        self._bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        if text:
            self._status.setText(tr_text(text))   # engine messages in the UI language

    @Slot(object)
    def _on_finished_ok(self, _result) -> None:
        self._bar.setValue(1000)
        self._finish(tr("common.done", "Done"))

    @Slot()
    def _on_cancelled(self) -> None:
        self._finish(tr("common.cancelled", "Cancelled"))

    @Slot(str, str)
    def _on_failed(self, en: str, km: str) -> None:
        self._finish(tr("common.failed", "Failed"))
        text = pick(en, km)              # the UI language only, not both
        QMessageBox.critical(
            self.window() if self.window() is not None else self,
            tr("common.error", "Error"), text)

    def _finish(self, status: str) -> None:
        self._status.setText(status)
        self._cancel_btn.setEnabled(False)
        self._worker = None
        if self._hide_on_done:
            self.hide()
        self.done.emit()
