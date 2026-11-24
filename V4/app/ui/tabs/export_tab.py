"""Export tab: source picker, preset editor, toggles, and a render queue."""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTime, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import get_language, tr
from app.core.logger import get_logger
from app.ui.widgets import scroll_page

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.export_tab")

_ENCODER_LABELS = {
    "libx264": "libx264 (CPU)",
    "libx265": "libx265 (CPU HEVC)",
    "h264_nvenc": "h264_nvenc (NVIDIA)",
    "hevc_nvenc": "hevc_nvenc (NVIDIA HEVC)",
    "h264_amf": "h264_amf (AMD)",
    "hevc_amf": "hevc_amf (AMD HEVC)",
    "h264_qsv": "h264_qsv (Intel QSV)",
    "hevc_qsv": "hevc_qsv (Intel QSV HEVC)",
}

_RESOLUTIONS = [
    ("keep", "Keep", 0, 0),
    ("1080p", "1080p (1920x1080)", 1920, 1080),
    ("720p", "720p (1280x720)", 1280, 720),
    ("4k", "4K (3840x2160)", 3840, 2160),
    ("9x16", "9:16 (1080x1920)", 1080, 1920),
]

_FIT_MODES = [
    ("blur", "Blurred background (whole picture)"),
    ("smart", "Smart crop — follows faces, shot by shot"),
    ("crop", "Crop the centre"),
    ("pad", "Black bars"),
    ("", "Stretch (old behaviour)"),
]

_JOB_STATUS_KEYS = {
    "pending": ("export.job.pending", "Pending"),
    "running": ("export.job.running", "Running"),
    "done": ("export.job.done", "Done"),
    "error": ("export.job.error", "Error"),
    "cancelled": ("export.job.cancelled", "Cancelled"),
}

JCOL_LABEL, JCOL_STATUS, JCOL_PROGRESS = range(3)


def _err_text(exc: Exception) -> str:
    en = getattr(exc, "en", str(exc))
    km = getattr(exc, "km", "")
    return km if (get_language() == "km" and km) else en


def _clock(seconds: float) -> str:
    """3:05, or 1:02:03 from an hour on."""
    s = int(round(max(0.0, seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _QueueBridge(QObject):
    """RenderQueue.run on_job_progress (worker thread) -> UI thread."""

    job_progress = Signal(str, float, str)


class ExportTab(QWidget):
    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self._queue = None            # RenderQueue, created lazily
        self._queue_worker = None
        self._encoders: list[str] = ["libx264", "libx265"]
        self._job_rows: dict[str, int] = {}
        self._presets: list = []

        root = scroll_page(self)          # a short window scrolls, nothing overlaps
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ---------------- source ----------------
        src_box = QGroupBox(tr("export.source", "Source"), self)
        src_lay = QHBoxLayout(src_box)
        self.rb_current = QRadioButton(tr("export.src.current", "Current video"), src_box)
        self.rb_file = QRadioButton(tr("export.src.pick", "Pick file"), src_box)
        self.rb_file.setChecked(True)
        src_lay.addWidget(self.rb_current)
        src_lay.addWidget(self.rb_file)
        self.src_edit = QLineEdit(src_box)
        self.src_edit.setPlaceholderText(tr("export.src.ph", "Video file to export"))
        src_lay.addWidget(self.src_edit, 1)
        btn_browse = QPushButton(tr("export.browse", "Browse..."), src_box)
        btn_browse.clicked.connect(self._pick_source)
        src_lay.addWidget(btn_browse)
        btn_from_lib = QPushButton(tr("export.src.library", "From library..."), src_box)
        btn_from_lib.clicked.connect(self._pick_from_library)
        src_lay.addWidget(btn_from_lib)
        root.addWidget(src_box)
        self.rb_current.toggled.connect(self._current_toggled)

        # ---------------- preset editor ----------------
        preset_box = QGroupBox(tr("export.preset", "Export preset"), self)
        grid = QGridLayout(preset_box)

        grid.addWidget(QLabel(tr("export.container", "Container")), 0, 0)
        self.container_combo = QComboBox(preset_box)
        self.container_combo.addItems(["mp4", "mkv"])
        grid.addWidget(self.container_combo, 0, 1)

        grid.addWidget(QLabel(tr("export.encoder", "Encoder")), 0, 2)
        self.encoder_combo = QComboBox(preset_box)
        self._fill_encoders(self._encoders)
        grid.addWidget(self.encoder_combo, 0, 3)

        grid.addWidget(QLabel(tr("export.bitrate", "Video bitrate")), 1, 0)
        self.bitrate_combo = QComboBox(preset_box)
        self.bitrate_combo.setEditable(True)
        self.bitrate_combo.addItems(["2M", "4M", "6M", "8M", "10M", "12M", "16M", "20M"])
        self.bitrate_combo.setCurrentText("6M")
        grid.addWidget(self.bitrate_combo, 1, 1)

        grid.addWidget(QLabel(tr("export.resolution", "Resolution")), 1, 2)
        self.res_combo = QComboBox(preset_box)
        for key, label, w, h in _RESOLUTIONS:
            self.res_combo.addItem(tr(f"export.res.{key}", label), (w, h))
        grid.addWidget(self.res_combo, 1, 3)

        grid.addWidget(QLabel(tr("export.fps", "FPS")), 2, 0)
        self.fps_combo = QComboBox(preset_box)
        self.fps_combo.addItem(tr("export.fps.keep", "Keep"), 0.0)
        for f in (24, 30, 60):
            self.fps_combo.addItem(str(f), float(f))
        grid.addWidget(self.fps_combo, 2, 1)

        grid.addWidget(QLabel(tr("export.abitrate", "Audio bitrate")), 2, 2)
        self.abitrate_combo = QComboBox(preset_box)
        self.abitrate_combo.setEditable(True)
        self.abitrate_combo.addItems(["128k", "160k", "192k", "256k", "320k"])
        self.abitrate_combo.setCurrentText("192k")
        grid.addWidget(self.abitrate_combo, 2, 3)

        grid.addWidget(QLabel(tr("export.fit", "Aspect change")), 3, 0)
        self.fit_combo = QComboBox(preset_box)
        for key, label in _FIT_MODES:
            self.fit_combo.addItem(tr(f"export.fit.{key or 'stretch'}", label), key)
        self.fit_combo.setCurrentIndex(max(0, self.fit_combo.findData("blur")))
        self.fit_combo.setToolTip(tr(
            "export.fit.tip", "Used when the output size has a different shape than "
                              "the source, e.g. 9:16 from a 16:9 episode."))
        grid.addWidget(self.fit_combo, 3, 1, 1, 3)

        prow = QHBoxLayout()
        prow.addWidget(QLabel(tr("export.presets", "Presets")))
        self.preset_combo = QComboBox(preset_box)
        self.preset_combo.setMinimumWidth(180)
        prow.addWidget(self.preset_combo, 1)
        btn_load = QPushButton(tr("export.preset.load", "Load"), preset_box)
        btn_load.clicked.connect(self._load_selected_preset)
        prow.addWidget(btn_load)
        btn_save = QPushButton(tr("export.preset.save", "Save preset..."), preset_box)
        btn_save.clicked.connect(self._save_preset)
        prow.addWidget(btn_save)
        btn_del = QPushButton(tr("export.preset.delete", "Delete"), preset_box)
        btn_del.clicked.connect(self._delete_preset)
        prow.addWidget(btn_del)
        grid.addLayout(prow, 4, 0, 1, 4)      # under "Aspect change" (row 3), not on top of it
        root.addWidget(preset_box)

        # ---------------- toggles ----------------
        tog_box = QGroupBox(tr("export.options", "Options"), self)
        tgrid = QGridLayout(tog_box)

        self.chk_subs = QCheckBox(tr("export.burn_subs", "Burn subtitles"), tog_box)
        tgrid.addWidget(self.chk_subs, 0, 0)
        self.srt_edit = QLineEdit(tog_box)
        self.srt_edit.setPlaceholderText(tr("export.srt.ph", ".srt file"))
        self.srt_edit.setEnabled(False)
        tgrid.addWidget(self.srt_edit, 0, 1)
        self.btn_srt = QPushButton(tr("export.srt.pick", "Pick .srt"), tog_box)
        self.btn_srt.setEnabled(False)
        self.btn_srt.clicked.connect(self._pick_srt)
        tgrid.addWidget(self.btn_srt, 0, 2)
        self.sub_style_combo = QComboBox(tog_box)
        self.sub_style_combo.addItem(tr("export.sub_style.plain", "Plain"), "")
        from app.engines.subtitles import PRESETS
        for sid, st in PRESETS.items():
            self.sub_style_combo.addItem(st.name, sid)
        self.sub_style_combo.setToolTip(tr(
            "export.sub_style.tip", "Style for burned .srt subtitles (laid out for the "
                                    "output size; Khmer-ready fonts, karaoke, Shorts)."))
        self.sub_style_combo.setEnabled(False)
        self.chk_subs.toggled.connect(self.sub_style_combo.setEnabled)
        tgrid.addWidget(self.sub_style_combo, 0, 3)
        self.chk_subs.toggled.connect(self.srt_edit.setEnabled)
        self.chk_subs.toggled.connect(self.btn_srt.setEnabled)

        self.chk_overlay = QCheckBox(
            tr("export.apply_overlay", "Apply overlay config from Overlay tab"), tog_box)
        tgrid.addWidget(self.chk_overlay, 1, 0, 1, 3)

        self.chk_dub = QCheckBox(tr("export.use_dub", "Use dubbed audio"), tog_box)
        tgrid.addWidget(self.chk_dub, 2, 0)
        self.dub_edit = QLineEdit(tog_box)
        self.dub_edit.setPlaceholderText(tr("export.dub.ph", "Dubbed audio file (.wav/.mp3)"))
        self.dub_edit.setEnabled(False)
        tgrid.addWidget(self.dub_edit, 2, 1)
        self.btn_dub = QPushButton(tr("export.dub.pick", "Pick audio"), tog_box)
        self.btn_dub.setEnabled(False)
        self.btn_dub.clicked.connect(self._pick_dub)
        tgrid.addWidget(self.btn_dub, 2, 2)
        self.chk_dub.toggled.connect(self._dub_toggled)

        # speed: the whole export (or every episode) plays faster / slower
        tgrid.addWidget(QLabel(tr("export.speed", "Speed")), 3, 0)
        self.speed_spin = QDoubleSpinBox(tog_box)
        from app.engines.media_exporter import SPEED_MAX, SPEED_MIN
        self.speed_spin.setRange(SPEED_MIN, SPEED_MAX)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("x")
        self.speed_spin.setToolTip(tr(
            "export.speed.tip", "Plays the exported video faster (above 1.00x) or slower "
                                "(below 1.00x). Voices keep their natural pitch."))
        self.speed_spin.setMinimumWidth(110)
        self.speed_info = QLabel("", tog_box)
        self.speed_info.setStyleSheet("color: #8b949e;")
        speed_row = QHBoxLayout()
        speed_row.setContentsMargins(0, 0, 0, 0)
        speed_row.addWidget(self.speed_spin)
        speed_row.addSpacing(8)
        speed_row.addWidget(self.speed_info)
        speed_row.addStretch(1)
        tgrid.addLayout(speed_row, 3, 1, 1, 3)
        root.addWidget(tog_box)

        # ---------------- split into episodes ----------------
        split_box = QGroupBox(tr("export.split", "Split into episodes"), self)
        sgrid = QGridLayout(split_box)
        self.chk_split = QCheckBox(tr("export.split.on", "Cut into episodes of"), split_box)
        sgrid.addWidget(self.chk_split, 0, 0)
        self.split_len = QTimeEdit(split_box)
        self.split_len.setDisplayFormat("H:mm:ss")
        self.split_len.setMinimumTime(QTime(0, 0, 10))
        saved = int(self.ctx.config.get("export.split_seconds") or 180)
        self.split_len.setTime(QTime(0, 0, 0).addSecs(max(10, min(saved, 86_399))))
        self.split_len.setToolTip(tr(
            "export.split.len_tip", "The length of each finished episode (hours:minutes:"
                                    "seconds), after the speed change."))
        self.split_len.setEnabled(False)
        sgrid.addWidget(self.split_len, 0, 1)
        sgrid.setColumnStretch(2, 1)
        self.split_info = QLabel("", split_box)
        self.split_info.setWordWrap(True)
        sgrid.addWidget(self.split_info, 1, 0, 1, 3)
        hint = QLabel(tr(
            "export.split.hint", "Each episode is saved as “<video name>_ep1”, “_ep2”… in a "
                                 "folder you choose, with the settings above (size, speed, "
                                 "subtitles, overlays, dubbed audio)."), split_box)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8b949e;")
        sgrid.addWidget(hint, 2, 0, 1, 3)
        # the poster a downloaded video carries (its thumbnail, maybe its
        # first second too): in every episode, or in none
        self.chk_keep_poster = QCheckBox(
            tr("export.split.poster", "Keep the original video's poster"), split_box)
        self.chk_keep_poster.setToolTip(tr(
            "export.split.poster_tip",
            "On: every episode carries the original video's poster as its thumbnail (the "
            "picture shown for the file). A video that starts with its poster (the DL tab's "
            "“Poster in the first second”) gives every episode that first second too.\n"
            "Off: the episodes have no poster, and episode 1 starts after the poster second."))
        keep = self.ctx.config.get("export.split_keep_poster")
        self.chk_keep_poster.setChecked(True if keep is None else bool(keep))
        self.chk_keep_poster.setEnabled(False)
        self.chk_keep_poster.toggled.connect(self._remember_keep_poster)
        sgrid.addWidget(self.chk_keep_poster, 3, 0, 1, 2)
        self.poster_info = QLabel("", split_box)
        self.poster_info.setWordWrap(True)
        self.poster_info.setStyleSheet("color: #8b949e;")
        sgrid.addWidget(self.poster_info, 4, 0, 1, 3)
        root.addWidget(split_box)
        self.chk_split.toggled.connect(self.split_len.setEnabled)
        self.chk_split.toggled.connect(self.chk_keep_poster.setEnabled)

        # the lines that say how long the result is / how it splits: after
        # a short pause in typing (reading the length runs ffprobe)
        self._durations: dict[tuple[str, float], float] = {}
        self._posters: dict[tuple[str, float], tuple[str, float]] = {}
        self._info_timer = QTimer(self)
        self._info_timer.setSingleShot(True)
        self._info_timer.setInterval(350)
        self._info_timer.timeout.connect(self._update_info)
        for sig in (self.src_edit.textChanged, self.rb_current.toggled,
                    self.speed_spin.valueChanged, self.split_len.timeChanged,
                    self.chk_split.toggled):
            sig.connect(lambda *_: self._info_timer.start())

        # ---------------- queue ----------------
        q_box = QGroupBox(tr("export.queue", "Render queue"), self)
        q_lay = QVBoxLayout(q_box)
        q_btns = QHBoxLayout()
        btn_add = QPushButton(tr("export.queue.add", "Add to queue"), q_box)
        btn_add.clicked.connect(self._add_to_queue)
        q_btns.addWidget(btn_add)
        self.btn_start = QPushButton(tr("export.queue.start", "Start Queue"), q_box)
        self.btn_start.clicked.connect(self._start_queue)
        q_btns.addWidget(self.btn_start)
        self.btn_cancel = QPushButton(tr("export.queue.cancel", "Cancel"), q_box)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_queue)
        q_btns.addWidget(self.btn_cancel)
        self.btn_resume = QPushButton(tr("export.queue.resume", "Resume interrupted"), q_box)
        self.btn_resume.clicked.connect(self._resume_interrupted)
        q_btns.addWidget(self.btn_resume)
        q_btns.addStretch(1)
        self.queue_status = QLabel("", q_box)
        q_btns.addWidget(self.queue_status)
        q_lay.addLayout(q_btns)

        self.job_table = QTableWidget(0, 3, q_box)
        self.job_table.setHorizontalHeaderLabels([
            tr("export.job.label", "Job"),
            tr("export.job.status", "Status"),
            tr("export.job.progress", "Progress"),
        ])
        jhdr = self.job_table.horizontalHeader()
        jhdr.setSectionResizeMode(JCOL_LABEL, QHeaderView.Stretch)
        jhdr.setSectionResizeMode(JCOL_STATUS, QHeaderView.ResizeToContents)
        jhdr.setSectionResizeMode(JCOL_PROGRESS, QHeaderView.Fixed)
        self.job_table.setColumnWidth(JCOL_PROGRESS, 200)
        self.job_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.job_table.verticalHeader().setVisible(False)
        self.job_table.setMinimumHeight(130)      # a few jobs stay visible (the page scrolls)
        q_lay.addWidget(self.job_table, 1)
        root.addWidget(q_box, 1)

        # bridge for queue progress
        self._qbridge = _QueueBridge(self)
        self._qbridge.job_progress.connect(self._on_job_progress)

        # detect encoders once in a worker; load presets and persisted queue
        self._detect_encoders_async()
        self._reload_presets()
        self._init_queue_from_disk()
        self._update_info()

    def showEvent(self, ev):  # noqa: N802 - Qt override
        super().showEvent(ev)
        if self.rb_current.isChecked():          # the current video may have changed
            self.src_edit.setText(self.ctx.current_video or "")
        self._info_timer.start()

    # ------------------------------------------------------------------
    # length lines: speed and split
    # ------------------------------------------------------------------
    def _source_path(self) -> str:
        if self.rb_current.isChecked():
            return self.ctx.current_video or ""
        return self.src_edit.text().strip()

    def _source_length(self, path: str) -> float:
        """The video's length in seconds (read once per file); 0 = unknown."""
        try:
            key = (os.path.normcase(os.path.abspath(path)), os.path.getmtime(path))
        except (OSError, TypeError, ValueError):
            return 0.0
        if key not in self._durations:
            try:
                from app.engines.media_exporter import probe
                self._durations[key] = float(probe(path).duration or 0.0)
            except Exception as exc:
                log.info("could not read the length of %s: %s", path, exc)
                self._durations[key] = 0.0
        return self._durations[key]

    def _source_poster(self, path: str) -> tuple[str, float]:
        """(the video's poster as a JPEG or "", seconds of the poster shown at
        its start or 0), read once per file."""
        try:
            key = (os.path.normcase(os.path.abspath(path)), os.path.getmtime(path))
        except (OSError, TypeError, ValueError):
            return "", 0.0
        if key not in self._posters:
            poster, intro = "", 0.0
            try:
                from app.engines import cover_art
                poster = cover_art.find_poster(path)
                intro = cover_art.intro_seconds(path) if poster else 0.0
            except Exception as exc:
                log.info("could not look for the poster of %s: %s", path, exc)
            self._posters[key] = (poster, intro)
        return self._posters[key]

    def _remember_keep_poster(self, on: bool) -> None:
        try:
            self.ctx.config.set("export.split_keep_poster", bool(on))
        except Exception:
            log.debug("could not remember the poster choice", exc_info=True)
        self._info_timer.start()

    def _split_seconds(self) -> int:
        return QTime(0, 0, 0).secsTo(self.split_len.time())

    def _update_info(self) -> None:
        """'32:47 becomes 26:14' beside the speed; how the video splits."""
        from app.engines.media_exporter import plan_parts
        src = self._source_path()
        length = self._source_length(src) if src and os.path.isfile(src) else 0.0
        speed = float(self.speed_spin.value())
        if length > 0 and abs(speed - 1.0) > 1e-3:
            self.speed_info.setText(tr("export.speed.length", "{old} becomes {new}")
                                    .replace("{old}", _clock(length))
                                    .replace("{new}", _clock(length / speed)))
        else:
            self.speed_info.setText(tr("export.speed.normal", "1.00x = normal speed"))
        self.split_info.setVisible(self.chk_split.isChecked())
        self.poster_info.setVisible(self.chk_split.isChecked())
        if not self.chk_split.isChecked():
            return
        poster, intro = (self._source_poster(src) if src and os.path.isfile(src)
                         else ("", 0.0))
        if not src or not os.path.isfile(src):
            self.poster_info.setText("")
        elif not poster:
            self.poster_info.setText(tr("export.split.poster_none",
                                        "This video has no poster to keep."))
        elif intro > 0:
            self.poster_info.setText(tr(
                "export.split.poster_intro", "Poster found: the video carries it as its "
                                             "thumbnail and shows it for its first second."))
        else:
            self.poster_info.setText(tr("export.split.poster_found",
                                        "Poster found: the video carries it as its thumbnail."))
        if not src or not os.path.isfile(src):
            text = tr("export.split.no_source", "Pick a video to see how it splits.")
        elif length <= 0:
            text = tr("export.split.no_length", "Could not read the length of this video.")
        else:
            part = self._split_seconds()
            n, left = plan_parts(max(0.0, length - intro), part, speed)
            if n == 0 or (n == 1 and left == 0):
                text = tr("export.split.short", "The video ({total}) is not longer than one "
                                                "episode: there is nothing to split.")
            elif left > 0:
                text = tr("export.split.info_left",
                          "{total} → {n} episode(s) of {len}, and {left} left over (you will "
                          "be asked what to do with it).")
            else:
                text = tr("export.split.info_even", "{total} → {n} episode(s) of {len}.")
            text = (text.replace("{total}", _clock(max(0.0, length - intro) / speed))
                    .replace("{n}", str(n)).replace("{len}", _clock(part))
                    .replace("{left}", _clock(left)))
        self.split_info.setText(text)

    # ------------------------------------------------------------------
    # source
    # ------------------------------------------------------------------
    def set_source_file(self, path: str) -> None:
        """Called by LibraryTab 'Export this' to preload a file."""
        self.rb_file.setChecked(True)
        self.src_edit.setText(path)

    def _current_toggled(self, on: bool) -> None:
        if on:
            cur = self.ctx.current_video or ""
            self.src_edit.setText(cur)
            self.src_edit.setEnabled(False)
        else:
            self.src_edit.setEnabled(True)

    def _pick_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("export.pick_video", "Pick video"), "",
            "Video (*.mp4 *.mkv *.mov *.avi *.webm *.ts *.flv);;All files (*.*)")
        if path:
            self.rb_file.setChecked(True)
            self.src_edit.setText(path)

    def _pick_from_library(self) -> None:
        try:
            series = self.ctx.library.get_series()
        except Exception as exc:
            QMessageBox.warning(self, tr("export.error", "Export error"), _err_text(exc))
            return
        entries: list[tuple[str, str]] = []
        for s in series:
            try:
                for ep in self.ctx.library.get_episodes(s.get("id")):
                    fp = ep.get("filepath", "")
                    if fp:
                        entries.append((f"{s.get('name', '?')} / {ep.get('title', '?')}", fp))
            except Exception:
                continue
        if not entries:
            QMessageBox.information(
                self, tr("export.library", "Library"),
                tr("export.library.empty", "No episodes in the library yet."))
            return
        labels = [e[0] for e in entries]
        label, ok = QInputDialog.getItem(
            self, tr("export.src.library", "From library..."),
            tr("export.pick_episode", "Pick an episode:"), labels, 0, False)
        if ok and label:
            self.set_source_file(entries[labels.index(label)][1])

    def _pick_srt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("export.srt.pick", "Pick .srt"), "",
            "Subtitles (*.srt *.ass);;All files (*.*)")
        if path:
            self.srt_edit.setText(path)

    def _dub_toggled(self, on: bool) -> None:
        self.dub_edit.setEnabled(on)
        self.btn_dub.setEnabled(on)
        if on and not self.dub_edit.text().strip():
            auto = getattr(self.ctx, "dubbed_audio", None) or self.ctx.config.get("session.dubbed_audio")
            if auto and os.path.isfile(str(auto)):
                self.dub_edit.setText(str(auto))

    def _pick_dub(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("export.dub.pick", "Pick audio"), "",
            "Audio (*.wav *.mp3 *.m4a *.flac *.aac);;All files (*.*)")
        if path:
            self.dub_edit.setText(path)

    # ------------------------------------------------------------------
    # encoders / presets
    # ------------------------------------------------------------------
    def _fill_encoders(self, encoders: list[str]) -> None:
        current = self.encoder_combo.currentData() if self.encoder_combo.count() else None
        self.encoder_combo.clear()
        seen = []
        for enc in encoders:
            if enc in seen:
                continue
            seen.append(enc)
            self.encoder_combo.addItem(_ENCODER_LABELS.get(enc, enc), enc)
        if "libx264" not in seen:
            self.encoder_combo.addItem(_ENCODER_LABELS["libx264"], "libx264")
        if current:
            idx = self.encoder_combo.findData(current)
            if idx >= 0:
                self.encoder_combo.setCurrentIndex(idx)

    def _detect_encoders_async(self) -> None:
        def detect():
            from app.engines.media_exporter import detect_encoders
            # with the config the result is saved, so Full Dub and the series
            # pipeline know the GPU encoder without opening this tab
            return detect_encoders(self.ctx.config)

        def done(result) -> None:
            try:
                encoders = list(result or [])
            except Exception:
                encoders = []
            if encoders:
                self._encoders = encoders
                self._fill_encoders(encoders)

        try:
            from app.ui.workers import FnWorker
            worker = FnWorker(detect)
            worker.finished_ok.connect(done)
            worker.failed.connect(lambda en, km: log.warning("detect_encoders failed: %s", en))
            self.ctx.pool.start(worker) if hasattr(self.ctx.pool, "start") else worker.start()
            self._keep(worker)
        except Exception as exc:
            log.warning("encoder detection worker failed: %s", exc)

    def _keep(self, worker) -> None:
        for meth in ("add", "keep", "track"):
            fn = getattr(self.ctx.pool, meth, None)
            if callable(fn):
                try:
                    fn(worker)
                    return
                except Exception:
                    pass
        if not hasattr(self, "_workers"):
            self._workers = []
        self._workers.append(worker)

    def _current_preset(self):
        from app.engines.media_exporter import ExportPreset
        w, h = self.res_combo.currentData() or (0, 0)
        return ExportPreset(
            name=self.preset_combo.currentText().strip() or "Default",
            container=self.container_combo.currentText(),
            vcodec=self.encoder_combo.currentData() or "libx264",
            bitrate=self.bitrate_combo.currentText().strip() or "6M",
            width=int(w), height=int(h),
            fps=float(self.fps_combo.currentData() or 0.0),
            audio_bitrate=self.abitrate_combo.currentText().strip() or "192k",
            fit_mode=self.fit_combo.currentData() or "",
        )

    def _apply_preset(self, p) -> None:
        idx = self.container_combo.findText(getattr(p, "container", "mp4"))
        if idx >= 0:
            self.container_combo.setCurrentIndex(idx)
        eidx = self.encoder_combo.findData(getattr(p, "vcodec", "libx264"))
        if eidx < 0:
            self.encoder_combo.addItem(getattr(p, "vcodec", "libx264"), getattr(p, "vcodec", "libx264"))
            eidx = self.encoder_combo.count() - 1
        self.encoder_combo.setCurrentIndex(eidx)
        self.bitrate_combo.setCurrentText(getattr(p, "bitrate", "6M"))
        w, h = int(getattr(p, "width", 0)), int(getattr(p, "height", 0))
        ridx = 0
        for i in range(self.res_combo.count()):
            if self.res_combo.itemData(i) == (w, h):
                ridx = i
                break
        self.res_combo.setCurrentIndex(ridx)
        fps = float(getattr(p, "fps", 0.0))
        fidx = self.fps_combo.findData(fps)
        self.fps_combo.setCurrentIndex(fidx if fidx >= 0 else 0)
        self.abitrate_combo.setCurrentText(getattr(p, "audio_bitrate", "192k"))
        fidx = self.fit_combo.findData(getattr(p, "fit_mode", "blur") or "")
        self.fit_combo.setCurrentIndex(max(0, fidx))

    def _reload_presets(self) -> None:
        try:
            from app.engines.media_exporter import load_presets
            self._presets = load_presets(self.ctx.config) or []
        except Exception as exc:
            log.warning("load_presets failed: %s", exc)
            self._presets = []
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for p in self._presets:
            self.preset_combo.addItem(getattr(p, "name", "?"))
        self.preset_combo.blockSignals(False)

    def _load_selected_preset(self) -> None:
        i = self.preset_combo.currentIndex()
        if 0 <= i < len(self._presets):
            self._apply_preset(self._presets[i])

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self, tr("export.preset.save", "Save preset..."),
            tr("export.preset.name", "Preset name:"),
            text=self.preset_combo.currentText().strip() or "Default")
        if not ok or not name.strip():
            return
        preset = self._current_preset()
        preset.name = name.strip()
        try:
            from app.engines.media_exporter import save_preset
            save_preset(self.ctx.config, preset)
        except Exception as exc:
            QMessageBox.warning(self, tr("export.error", "Export error"), _err_text(exc))
            return
        self._reload_presets()
        idx = self.preset_combo.findText(preset.name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    def _delete_preset(self) -> None:
        i = self.preset_combo.currentIndex()
        if not (0 <= i < len(self._presets)):
            return
        name = getattr(self._presets[i], "name", "")
        try:
            raw = self.ctx.config.get("export.presets") or []
            raw = [p for p in raw if (p.get("name") if isinstance(p, dict) else getattr(p, "name", None)) != name]
            self.ctx.config.set("export.presets", raw)
        except Exception as exc:
            QMessageBox.warning(self, tr("export.error", "Export error"), _err_text(exc))
            return
        self._reload_presets()

    # ------------------------------------------------------------------
    # queue
    # ------------------------------------------------------------------
    def _ensure_queue(self):
        if self._queue is None:
            from app.engines.media_exporter import RenderQueue
            self._queue = RenderQueue()
            try:
                self._queue.load()
            except Exception as exc:
                log.warning("queue load failed: %s", exc)
        return self._queue

    def _init_queue_from_disk(self) -> None:
        try:
            q = self._ensure_queue()
        except Exception as exc:
            log.warning("RenderQueue init failed: %s", exc)
            return
        pending = 0
        for job in getattr(q, "jobs", []) or []:
            self._add_job_row(job)
            if getattr(job, "status", "") in ("pending", "running"):
                pending += 1
        if pending:
            self.queue_status.setText(
                tr("export.queue.pending", "{n} interrupted/pending job(s) — press Resume")
                .replace("{n}", str(pending)))

    def _overlay_config(self):
        """Overlays from the Overlay tab's current fields when it is open,
        else the settings it saved (they survive a restart)."""
        tabs = getattr(getattr(self.ctx, "main", None), "tabs", None)
        if tabs is not None:
            from app.ui.tabs.overlay_tab import OverlayTab
            for i in range(tabs.count()):
                tab = tabs.widget(i)
                if isinstance(tab, OverlayTab):
                    try:
                        return tab.overlay_config()
                    except Exception as exc:
                        log.warning("could not read the Overlay tab: %s", exc)
        ov = getattr(self.ctx, "overlay_config", None)
        if ov is not None:
            return ov
        try:
            from app.engines.media_exporter import OverlayConfig
            saved = self.ctx.config.get("overlay.last")
            if isinstance(saved, dict):
                return OverlayConfig.from_dict(saved)
        except Exception as exc:
            log.warning("could not load the saved overlay settings: %s", exc)
        return None

    def _ask_out_path(self, src: str, container: str) -> str:
        """Where the export goes, asked every time. The dialog starts at the
        next free "<name>_export.<ext>" in the folder used last (else next
        to the source). "" = the user backed out."""
        ext = (container or "mp4").lower().lstrip(".")
        name = os.path.splitext(os.path.basename(src))[0]
        last = str(self.ctx.config.get("export.out_dir") or "")
        folder = last if last and os.path.isdir(last) else os.path.dirname(src)
        start, n = os.path.join(folder, f"{name}_export.{ext}"), 1
        while os.path.exists(start):
            start, n = os.path.join(folder, f"{name}_export_{n}.{ext}"), n + 1
        while True:
            path, _ = QFileDialog.getSaveFileName(
                self, tr("export.save_as", "Save the exported video as"), start,
                f"{ext.upper()} (*.{ext})")
            if not path:
                return ""
            if os.path.splitext(path)[1].lower() != f".{ext}":
                path += f".{ext}"
            path = os.path.normpath(path)
            if os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(src)):
                break
            QMessageBox.warning(self, tr("export.error", "Export error"), tr(
                "export.same_as_source", "That is the source video itself: choose another "
                "name, or the export would replace it."))
            start = path
        try:
            self.ctx.config.set("export.out_dir", os.path.dirname(path))
        except Exception:
            log.debug("could not remember the export folder", exc_info=True)
        return path

    def _overlays_dict(self):
        """The Overlay tab's settings as a dict for a queue job, or None."""
        if not self.chk_overlay.isChecked():
            return None
        ov = self._overlay_config()
        if ov is None:
            return None
        try:
            return asdict(ov)
        except Exception:
            return getattr(ov, "__dict__", None)

    def _job_kwargs(self, src: str, out_path: str, preset, srt_path, dub, overlays) -> dict:
        return {
            "op": "render",
            "video": src,
            "out_path": out_path,
            "preset": asdict(preset),
            "overlays": overlays,
            "srt_path": srt_path,
            "dub_audio": dub,
            "subtitle_style": (self.sub_style_combo.currentData() or "") if srt_path else "",
            "speed": round(float(self.speed_spin.value()), 4),
        }

    def _ask_out_folder(self, src: str) -> str:
        """The folder for the episodes, asked every time (starts at the folder
        used last, else the source's). "" = the user backed out."""
        last = str(self.ctx.config.get("export.out_dir") or "")
        start = last if last and os.path.isdir(last) else os.path.dirname(src)
        folder = QFileDialog.getExistingDirectory(
            self, tr("export.split.folder", "Save the episodes in"), start)
        if not folder:
            return ""
        folder = os.path.normpath(folder)
        try:
            self.ctx.config.set("export.out_dir", folder)
        except Exception:
            log.debug("could not remember the export folder", exc_info=True)
        return folder

    def _ask_leftover(self, n: int, part: int, left: float) -> str:
        """The time after the last whole episode: "join" (added to episode
        n), "new" (episode n + 1 of its own) or "" (cancel)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr("export.split", "Split into episodes"))
        box.setText(tr("export.split.left_q", "The last {left} does not fill a whole episode.")
                    .replace("{left}", _clock(left)))
        box.setInformativeText(
            tr("export.split.left_info", "Add it to episode {n} (it becomes {joined}), or make "
                                         "it episode {next} on its own ({left})?")
            .replace("{n}", str(n)).replace("{joined}", _clock(part + left))
            .replace("{next}", str(n + 1)).replace("{left}", _clock(left)))
        b_join = box.addButton(tr("export.split.join", "Add to episode {n}")
                               .replace("{n}", str(n)), QMessageBox.ButtonRole.AcceptRole)
        b_new = box.addButton(tr("export.split.new", "Make episode {next}")
                              .replace("{next}", str(n + 1)), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(tr("common.cancel", "Cancel"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_new)
        box.exec()
        hit = box.clickedButton()
        return "join" if hit is b_join else "new" if hit is b_new else ""

    def _add_split_jobs(self, src: str, preset, srt_path, dub) -> None:
        """One queue job per episode: <name>_ep1, _ep2 … in a folder asked
        now, each the same length of finished video (after the speed); the
        time left over goes where the user says. With "Keep the original
        video's poster" every episode carries the poster as its thumbnail
        (and starts with the poster second when the video does); without
        it, none does and episode 1 starts after the poster second."""
        from app.engines.media_exporter import RenderJob, part_spans, plan_parts
        title = tr("export.split", "Split into episodes")
        speed = float(self.speed_spin.value())
        part = self._split_seconds()
        length = self._source_length(src)
        if length <= 0:
            QMessageBox.warning(self, title, tr("export.split.no_length",
                                                "Could not read the length of this video."))
            return
        poster, intro = self._source_poster(src)
        keep = self.chk_keep_poster.isChecked() and bool(poster)
        content = max(0.0, length - intro)          # the video after its poster second
        n, left = plan_parts(content, part, speed)
        if n == 0 or (n == 1 and left == 0):
            QMessageBox.information(self, title, tr(
                "export.split.short", "The video ({total}) is not longer than one episode: "
                                      "there is nothing to split.")
                .replace("{total}", _clock(content / speed)))
            return
        leftover = self._ask_leftover(n, part, left) if left > 0 else "new"
        if not leftover:
            return
        spans = [(intro + s, ln) for s, ln in part_spans(content, part, speed, leftover)]
        folder = self._ask_out_folder(src)
        if not folder:
            return
        ext = (preset.container or "mp4").lower()
        name = os.path.splitext(os.path.basename(src))[0]
        outs = [os.path.join(folder, f"{name}_ep{i}.{ext}") for i in range(1, len(spans) + 1)]
        there = [o for o in outs if os.path.exists(o)]
        if there and QMessageBox.question(self, title, tr(
                "export.split.exists", "{n} file(s) with these names are already in this "
                                       "folder (for example “{name}”). Replace them?")
                .replace("{n}", str(len(there))).replace("{name}", os.path.basename(there[0]))
                ) != QMessageBox.StandardButton.Yes:
            return
        overlays = self._overlays_dict()
        jobs = []
        for i, ((s, ln), out) in enumerate(zip(spans, outs)):
            kw = self._job_kwargs(src, out, preset, srt_path, dub, overlays)
            if keep:
                kw["poster"] = poster
                if intro > 0:
                    # every episode starts with the poster second, episode 1
                    # too (made the same way, so it stays one second at any speed)
                    kw["intro_image"], kw["intro_seconds"] = poster, round(intro, 4)
            kw["start"] = round(s, 3)
            # the last episode runs to the very end (no frame lost to rounding)
            kw["duration"] = 0.0 if i == len(spans) - 1 else round(ln, 3)
            jobs.append(RenderJob(
                id=uuid.uuid4().hex[:12],
                label=f"{os.path.basename(src)} {_clock(s)}–{_clock(s + ln)} -> "
                      f"{os.path.basename(out)}",
                kwargs=kw))
        try:
            q = self._ensure_queue()
            for job in jobs:
                q.add(job)
        except Exception as exc:
            QMessageBox.warning(self, tr("export.error", "Export error"), _err_text(exc))
            return
        for job in jobs:
            self._add_job_row(job)
        try:
            self.ctx.config.set("export.split_seconds", part)
        except Exception:
            log.debug("could not remember the episode length", exc_info=True)
        self.queue_status.setText(
            tr("export.split.added", "{n} episodes added to the queue: press Start Queue.")
            .replace("{n}", str(len(jobs))))

    def _add_to_queue(self) -> None:
        src = self._source_path()
        if not src or not os.path.isfile(src):
            QMessageBox.warning(self, tr("export.error", "Export error"),
                                tr("export.no_source", "Select a valid source video file first."))
            return

        preset = self._current_preset()
        srt_path = self.srt_edit.text().strip() if self.chk_subs.isChecked() else None
        if self.chk_subs.isChecked() and (not srt_path or not os.path.isfile(srt_path)):
            QMessageBox.warning(self, tr("export.error", "Export error"), tr(
                "export.no_srt", "Burn subtitles is enabled but no valid .srt file was selected."))
            return

        dub = self.dub_edit.text().strip() if self.chk_dub.isChecked() else None
        if self.chk_dub.isChecked() and (not dub or not os.path.isfile(dub)):
            QMessageBox.warning(self, tr("export.error", "Export error"), tr(
                "export.no_dub", "Use dubbed audio is enabled but no valid audio file was selected."))
            return

        if self.chk_split.isChecked():
            self._add_split_jobs(src, preset, srt_path, dub)
            return

        out_path = self._ask_out_path(src, preset.container)
        if not out_path:                        # backed out of the save dialog
            return

        kwargs = self._job_kwargs(src, out_path, preset, srt_path, dub, self._overlays_dict())
        from app.engines.media_exporter import RenderJob
        job = RenderJob(id=uuid.uuid4().hex[:12],
                        label=f"{os.path.basename(src)} -> {os.path.basename(out_path)}",
                        kwargs=kwargs)
        try:
            q = self._ensure_queue()
            q.add(job)
        except Exception as exc:
            QMessageBox.warning(self, tr("export.error", "Export error"), _err_text(exc))
            return
        self._add_job_row(job)
        self.queue_status.setText(tr("export.queue.added", "Job added."))

    def _add_job_row(self, job) -> None:
        jid = getattr(job, "id", "")
        if jid in self._job_rows:
            return
        row = self.job_table.rowCount()
        self.job_table.insertRow(row)
        self._job_rows[jid] = row
        label = QTableWidgetItem(getattr(job, "label", jid))
        out = str((getattr(job, "kwargs", None) or {}).get("out_path") or "")
        if out:                                  # where it is saved (the label has the name only)
            label.setToolTip(out)
        self.job_table.setItem(row, JCOL_LABEL, label)
        status = getattr(job, "status", "pending")
        key, default = _JOB_STATUS_KEYS.get(status, ("export.job.unknown", status))
        st = QTableWidgetItem(tr(key, default))
        st.setToolTip(getattr(job, "error", ""))
        self.job_table.setItem(row, JCOL_STATUS, st)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(int(float(getattr(job, "progress", 0.0) or 0.0) * 1000))
        self.job_table.setCellWidget(row, JCOL_PROGRESS, bar)
        self.job_table.setRowHeight(row, 30)

    def _on_job_progress(self, job_id: str, fraction: float, text: str) -> None:
        row = self._job_rows.get(job_id)
        if row is None:
            # a job we did not add in this session (e.g. resumed) — find it
            q = self._queue
            if q is not None:
                for job in getattr(q, "jobs", []) or []:
                    if getattr(job, "id", "") == job_id:
                        self._add_job_row(job)
                        row = self._job_rows.get(job_id)
                        break
        if row is None:
            return
        bar = self.job_table.cellWidget(row, JCOL_PROGRESS)
        if isinstance(bar, QProgressBar):
            bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        st = self.job_table.item(row, JCOL_STATUS)
        if st is not None and text:
            st.setText(text)

    def _start_queue(self) -> None:
        if self._queue_worker is not None and getattr(self._queue_worker, "isRunning", lambda: False)():
            return
        try:
            q = self._ensure_queue()
        except Exception as exc:
            QMessageBox.warning(self, tr("export.error", "Export error"), _err_text(exc))
            return
        bridge = self._qbridge

        def run_queue(progress=None, cancel: threading.Event | None = None):
            q.run(on_job_progress=bridge.job_progress.emit, cancel=cancel)
            return True

        from app.ui.workers import FnWorker
        worker = FnWorker(run_queue)
        worker.finished_ok.connect(self._queue_finished)
        worker.failed.connect(self._queue_failed)
        self._queue_worker = worker
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.queue_status.setText(tr("export.queue.running", "Queue running..."))
        try:
            self.ctx.pool.start(worker) if hasattr(self.ctx.pool, "start") else worker.start()
        except Exception:
            worker.start()
        self._keep(worker)

    def _queue_finished(self, _result=None) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.queue_status.setText(tr("export.queue.done", "Queue finished."))
        self._sync_rows_from_queue()

    def _queue_failed(self, en: str, km: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.queue_status.setText(tr("export.queue.failed", "Queue stopped."))
        self._sync_rows_from_queue()
        QMessageBox.warning(self, tr("export.error", "Export error"),
                            km if (get_language() == "km" and km) else en)

    def _sync_rows_from_queue(self) -> None:
        q = self._queue
        if q is None:
            return
        for job in getattr(q, "jobs", []) or []:
            row = self._job_rows.get(getattr(job, "id", ""))
            if row is None:
                continue
            status = getattr(job, "status", "pending")
            key, default = _JOB_STATUS_KEYS.get(status, ("export.job.unknown", status))
            st = self.job_table.item(row, JCOL_STATUS)
            if st is not None:
                st.setText(tr(key, default))
                st.setToolTip(getattr(job, "error", ""))
            bar = self.job_table.cellWidget(row, JCOL_PROGRESS)
            if isinstance(bar, QProgressBar):
                bar.setValue(int(float(getattr(job, "progress", 0.0) or 0.0) * 1000))

    def _cancel_queue(self) -> None:
        if self._queue_worker is not None:
            try:
                self._queue_worker.cancel()
            except Exception as exc:
                log.warning("queue cancel failed: %s", exc)
        self.queue_status.setText(tr("export.queue.cancelling", "Cancelling..."))

    def _resume_interrupted(self) -> None:
        try:
            q = self._ensure_queue()
            q.load()
        except Exception as exc:
            QMessageBox.warning(self, tr("export.error", "Export error"), _err_text(exc))
            return
        for job in getattr(q, "jobs", []) or []:
            self._add_job_row(job)
        self._sync_rows_from_queue()
        pending = [j for j in getattr(q, "jobs", []) or []
                   if getattr(j, "status", "") in ("pending", "running")]
        if not pending:
            self.queue_status.setText(tr("export.queue.nothing", "Nothing to resume."))
            return
        self._start_queue()
