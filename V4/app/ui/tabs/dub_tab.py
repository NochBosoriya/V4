"""Full Dub tab: translate a whole video with one cloned voice per character.

Left (about 60 % of the width): "Task Tracking & Problem Detection" — the
run panel (progress, time left, Continue / Cancel), "System Readiness Check"
(hardware tiles, every check with its one-click fix) and the task list
(counters, each task's note and time, a bar on the running task). Right:
Characters (voice sample, engine, preset voice, test), Lines (speaker,
original, translation, timing fit, preview), Glossary + story context, and
Output. Every step runs in an FnWorker via app.engines.dub_engine.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.ai_engine import AI_ADVICE
from app.core.i18n import get_language, tr, tr_text
from app.core.logger import get_logger
from app.engines import dub_engine as de
from app.ui import dub_widgets as dw
from app.ui import theme
from app.ui.run_panel import RunPanel, _fmt_bytes, _fmt_clock, _keep_awake  # noqa: F401
from app.ui.run_panel import FIX_IN_SETTINGS, ask_no_key, no_ai_key, open_settings
from app.ui.langs import LANGS
from app.ui.widgets import ProgressRow
from app.ui.workers import FnWorker

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.dub_tab")

WHISPER_SIZES = ["auto", "tiny", "base", "small", "medium", "large-v3"]
VIDEO_FILTER = "Video (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.ts *.m4v);;All files (*)"
AUDIO_FILTER = "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.aac);;All files (*)"

_FIT_COLOR = {"overlong": "#d2992255", "overrun": "#f8514955",
              "failed": "#f8514966", "late": "#d2992233"}

C_ID, C_NAME, C_GENDER, C_LINES, C_TALK, C_ENGINE, C_VOICE, C_PITCH, \
    C_SAMPLE, C_REFTEXT = range(10)
L_NUM, L_START, L_END, L_SPK, L_VOICE, L_SRC, L_TXT, L_FIT, L_PLAY = range(9)
_UNSURE_COLOR = "#d2992255"     # gender to check
_SWITCHED_COLOR = "#58a6ff33"   # line voiced with the other gender
_REVIEW_COLOR = "#d2992244"     # line marked ⚑ for a person to check
_SHORT_COLOR = "#a371f733"      # dub line shorter than its whole translation


def _fmt_t(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def _ro(text: str) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it


# --- worker functions (run off the UI thread) --------------------------------

def _readiness_fn(config, options):
    return de.readiness(config, options), _hardware_fn()


def _hardware_fn() -> dict:
    """What the readiness tiles show: the main graphics card, its memory,
    whether speech recognition can use CUDA, and the Python version. The
    GPU list is cached by app.core.gpu (the startup check made it)."""
    import sys
    info: dict = {"python": ".".join(str(v) for v in sys.version_info[:3])}
    try:
        from app.core import gpu as g
        p = g.primary_gpu(g.list_gpus())
        if p is not None:
            info.update(gpu=p.name, vendor=p.vendor, vram_gb=p.vram_gb)
            if p.vendor == "nvidia":
                info["cuda"] = g.ct2_cuda()
                try:
                    from importlib.metadata import version
                    info["cuda_ver"] = ".".join(version("nvidia-cublas-cu12").split(".")[:2])
                except Exception:
                    pass
    except Exception:
        log.debug("hardware info unavailable", exc_info=True)
    return info


def _fix_fn(check: de.Check, progress=None, cancel=None):
    if check.pip:
        de.install_packages(check.pip, progress=progress, cancel=cancel)
    if check.download == "openvoice":
        de.download_openvoice(progress=progress, cancel=cancel)
    elif check.download:
        from app.core.model_manager import download_model
        download_model(check.download, progress=progress, cancel=cancel)
    return check.id


def _import_sample_fn(src: str, out: str, cancel=None):
    from app.engines import audio_pipeline as ap
    return ap._to_24k_mono_wav(src, out, cancel=cancel)


DubRunPanel = RunPanel
# why no AI answered (ai_engine.ai_failure_kind) when the fix is in Settings:
# the paused task's panel gets an "Open Settings" button
_FIX_IN_SETTINGS = FIX_IN_SETTINGS
# below this tab height the dubbing settings fold away by themselves (they
# take ~170 px; with them open under 640 px the Characters / Lines table
# keeps fewer than 3 rows)
SHORT_TAB_H = 640


class DubTab(QWidget):
    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self.engine = de.DubEngine(ctx.ai, ctx.config)
        self.project: de.DubProject | None = None
        self._checks: list[de.Check] = []
        self._busy = False
        self._filling = False
        self._workers: list[FnWorker] = []
        self._players: list = []
        self._checked_once = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)

        # ---------------- source -------------------------------------------
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel(tr("dub.video", "Video:")))
        self.src_edit = QLineEdit()
        self.src_edit.setReadOnly(True)
        self.src_edit.setPlaceholderText(
            tr("dub.video.ph", "Pick a video to dub (a saved project reopens automatically)"))
        src_row.addWidget(self.src_edit, 1)
        b = QPushButton(tr("common.browse", "Browse…"))
        b.clicked.connect(self._on_browse)
        src_row.addWidget(b)
        b = QPushButton(tr("dub.use_current", "Use current video"))
        b.clicked.connect(self._on_use_current)
        src_row.addWidget(b)
        self.folder_btn = QPushButton(tr("dub.open_folder", "Project folder"))
        self.folder_btn.clicked.connect(self._open_project_folder)
        src_row.addWidget(self.folder_btn)
        root.addLayout(src_row)

        # ---------------- options (fold away for more room below) ----------
        self.opt_toggle = QToolButton()
        self.opt_toggle.setText(tr("dub.options", "Dubbing settings"))
        self.opt_toggle.setIcon(QIcon(dw.icon_pixmap("gear", dw.BLUE, 18)))
        self.opt_toggle.setIconSize(QSize(18, 18))
        self.opt_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.opt_toggle.setStyleSheet("QToolButton { font-weight: 600; padding: 3px 8px; }")
        self.opt_toggle.setToolTip(tr("dub.options.fold", "Show or hide the dubbing settings"))
        self.opt_chev = QToolButton()
        self.opt_chev.setAutoRaise(True)
        self.opt_chev.setToolTip(self.opt_toggle.toolTip())
        for b in (self.opt_toggle, self.opt_chev):
            b.clicked.connect(lambda: self._set_options_open(not self.opt_box.isVisible(),
                                                             save=True))
        oh = QHBoxLayout()
        oh.setSpacing(0)
        oh.addWidget(self.opt_toggle)
        oh.addWidget(self.opt_chev)
        oh.addStretch(1)
        root.addLayout(oh)
        opt_box = QFrame()
        opt_box.setObjectName("optBox")
        opt_box.setStyleSheet(f"QFrame#optBox {{ background: {theme.PANEL}; border: 1px solid "
                              f"{theme.BORDER}; border-radius: 8px; }}")
        self.opt_box = opt_box
        grid = QGridLayout(opt_box)
        grid.setContentsMargins(12, 8, 12, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(5)
        self.tgt_combo = QComboBox()
        for code, native in LANGS:
            self.tgt_combo.addItem(native, code)
        self.src_combo = QComboBox()
        self.src_combo.addItem(tr("dub.auto", "Auto-detect"), "")
        for code, native in LANGS:
            self.src_combo.addItem(native, code)
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(WHISPER_SIZES)
        self.whisper_combo.setCurrentText("auto")
        self.whisper_combo.setToolTip(tr("dub.whisper.tip", "auto = medium on an NVIDIA GPU, small on CPU"))
        self.spk_spin = QSpinBox()
        self.spk_spin.setRange(0, 12)
        self.spk_spin.setSpecialValueText(tr("dub.auto", "Auto-detect"))
        self.voice_mode = QComboBox()
        self.voice_mode.addItem(tr("dub.mode.pair", "Male / female voice (Khmer: Piseth / Sreymom)"),
                                "pair")
        self.voice_mode.addItem(tr("dub.mode.clone", "Clone each character's voice"), "clone")
        self.voice_mode.addItem(tr("dub.mode.preset", "Preset voices"), "preset")
        self.voice_mode.setToolTip(tr(
            "dub.mode.tip", "Male / female: every man speaks with the male voice and every "
            "woman with the female voice, switching line by line."))
        self.vary_check = QCheckBox(tr("dub.vary", "Make characters sound different"))
        self.vary_check.setToolTip(tr(
            "dub.vary.tip", "Shift the pitch a little for each character of the same gender. "
            "Off = the plain Piseth / Sreymom voices."))
        self.bg_combo = QComboBox()
        self.bg_combo.addItem(tr("dub.bg.keep", "Keep music & effects"), "keep")
        self.bg_combo.addItem(tr("dub.bg.none", "Voices only (no background)"), "none")
        self.bg_combo.setToolTip(tr(
            "dub.bg.tip", "Voices only exports just the new voices; it is faster because the "
            "music does not need to be separated."))
        self.quality_combo = QComboBox()
        self.quality_combo.addItem(tr("dub.q.high", "Best: translate + AI review"), "high")
        self.quality_combo.addItem(tr("dub.q.fast", "Fast: single pass"), "fast")
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(1.0, 2.5)
        self.tempo_spin.setSingleStep(0.1)
        self.tempo_spin.setValue(de.DubOptions().max_tempo)
        self.tempo_spin.setSuffix("×")
        self.tempo_spin.setToolTip(tr(
            "dub.tempo.tip2", "Lines are fitted at up to 1.4× speed; a longer one is first said "
            "more briefly with the same meaning. Only what still does not fit is spoken "
            "faster — just enough to end before the next line — up to this."))
        self.cuts_check = QCheckBox(tr("dub.cuts", "Allow cutting words to keep lip-sync"))
        self.cuts_check.setChecked(de.DubOptions().allow_cuts)
        self.cuts_check.setToolTip(tr(
            "dub.cuts.tip", "Off: a line too long for its time is only worded more tightly — "
            "every sentence and question stays, and the voice speaks it faster or a "
            "little late. On: the AI may cut it down to its main point to stay in sync "
            "(details and whole sentences can be lost)."))
        self.demucs_check = QCheckBox(tr("dub.demucs", "Separate voices from music (Demucs)"))
        self.demucs_check.setChecked(True)
        self.series_combo = QComboBox()
        self.series_combo.setEditable(True)
        self.series_combo.setToolTip(tr(
            "dub.series.tip",
            "Episodes of the same series share a glossary and a voice cast: "
            "known characters are recognised by voice and keep their name and voice."))
        self.tm_check = QCheckBox(tr("dub.tm", "Reuse translation memory"))
        self.tm_check.setChecked(True)
        self.tm_check.setToolTip(tr(
            "dub.tm.tip", "Lines translated before (or fixed by you) are reused "
                          "instead of asking the AI again."))
        self.style_edit = QLineEdit()
        self.style_edit.setPlaceholderText(
            tr("dub.style.ph", "Optional style, e.g. 'casual, young people' or 'royal court drama'"))

        grid.addWidget(QLabel(tr("dub.target", "Dub into:")), 0, 0)
        grid.addWidget(self.tgt_combo, 0, 1)
        grid.addWidget(QLabel(tr("dub.source", "Spoken language:")), 0, 2)
        grid.addWidget(self.src_combo, 0, 3)
        grid.addWidget(QLabel(tr("dub.whisper", "Whisper model:")), 0, 4)
        grid.addWidget(self.whisper_combo, 0, 5)
        grid.addWidget(QLabel(tr("dub.speakers", "Characters:")), 0, 6)
        grid.addWidget(self.spk_spin, 0, 7)
        grid.addWidget(QLabel(tr("dub.voices", "Voices:")), 1, 0)
        grid.addWidget(self.voice_mode, 1, 1)
        grid.addWidget(QLabel(tr("dub.quality", "Translation:")), 1, 2)
        grid.addWidget(self.quality_combo, 1, 3)
        grid.addWidget(QLabel(tr("dub.tempo", "Max speed-up:")), 1, 4)
        grid.addWidget(self.tempo_spin, 1, 5)
        grid.addWidget(self.demucs_check, 1, 6, 1, 2)
        grid.addWidget(QLabel(tr("dub.series", "Series:")), 2, 0)
        grid.addWidget(self.series_combo, 2, 1)
        grid.addWidget(self.tm_check, 2, 2, 1, 2)
        grid.addWidget(QLabel(tr("dub.style", "Style:")), 2, 4)
        grid.addWidget(self.style_edit, 2, 5, 1, 3)
        grid.addWidget(QLabel(tr("dub.background", "Background:")), 3, 0)
        grid.addWidget(self.bg_combo, 3, 1)
        grid.addWidget(self.vary_check, 3, 2, 1, 2)
        grid.addWidget(self.cuts_check, 3, 4, 1, 4)
        root.addWidget(opt_box)

        # ---------------- actions ------------------------------------------
        act = QHBoxLayout()
        self.run_all_btn = QPushButton(tr("dub.run_all2", "▶  Translate && dub whole video"))
        self.run_all_btn.setToolTip(tr(
            "dub.run_all.tip", "Runs every task below in order: listen, find the speakers "
            "and their gender, translate, speak, mix and export. A stopped run continues "
            "where it left off."))
        self.run_all_btn.setProperty("class", "accent")
        self.run_all_btn.clicked.connect(self._on_run_all)
        act.addWidget(self.run_all_btn)
        self.run_step_btn = QPushButton(tr("dub.run_step", "Run selected task"))
        self.run_step_btn.clicked.connect(self._on_run_selected)
        act.addWidget(self.run_step_btn)
        self.redo_btn = QPushButton(tr("dub.redo", "Redo from selected task"))
        self.redo_btn.clicked.connect(self._on_redo_from)
        act.addWidget(self.redo_btn)
        self.save_btn = QPushButton(tr("dub.save", "Save edits"))
        self.save_btn.clicked.connect(self._save_project)
        act.addWidget(self.save_btn)
        act.addStretch(1)
        root.addLayout(act)
        # installs / sample imports: a plain progress row; the dub itself:
        # the run panel, which stays on screen when it ends
        self.progress_row = ProgressRow(tr("dub.working", "Dubbing"))
        self.progress_row.hide()
        root.addWidget(self.progress_row)
        self.run_panel = DubRunPanel()
        self.run_panel.continue_clicked.connect(self._on_continue)
        self.run_panel.cancel_clicked.connect(self._on_cancel_run)
        self.run_panel.dismiss_clicked.connect(self._on_dismiss_error)
        self.run_panel.play_clicked.connect(self._on_play_output)
        self.run_panel.show_clicked.connect(self._on_show_output)
        self.run_panel.change_clicked.connect(lambda: self._ask_output(force=True))
        self.run_panel.settings_clicked.connect(self._open_settings)
        self.run_panel.state_changed.connect(lambda _s: self._update_idle_hint())
        self._run_worker: FnWorker | None = None
        self._last_run: tuple | None = None       # (fn, args) to Continue
        self._run_label = ""
        self._p_base, self._p_span = 0.0, 1.0     # run progress -> whole dub
        self._run_seconds = 0.0                   # earlier runs of this video
        self._todo_at_start: set[str] = set()     # tasks not done when the run began
        self._cur_step, self._cur_step_t0 = "", 0.0
        self._last_overall = 0.0
        self._last_error: tuple[str, str] | None = None

        # ---------------- body: the monitor (left) | the details (right) ----
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(8)
        split.setStyleSheet("QSplitter::handle { background: transparent; }")
        self.monitor = dw.GlowCard()
        outer = QVBoxLayout(self.monitor)
        outer.setContentsMargins(2, 2, 2, 2)
        # on a short window the monitor scrolls instead of squeezing its parts
        mon_scroll = QScrollArea()
        mon_scroll.setWidgetResizable(True)
        mon_scroll.setFrameShape(QFrame.Shape.NoFrame)
        mon_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mon_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget { background: transparent; }")
        mon_body = dw.clear_widget()
        mon_scroll.setWidget(mon_body)
        outer.addWidget(mon_scroll)
        mon = QVBoxLayout(mon_body)
        mon.setContentsMargins(14, 10, 14, 10)
        mon.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(12)
        mon_icon = QLabel()
        mon_icon.setPixmap(dw.round_pixmap("pulse", dw.BLUE, 38, solid=True))
        head.addWidget(mon_icon)
        mon_title = QLabel(tr("dub.mon.title", "Task Tracking & Problem Detection"))
        mon_title.setStyleSheet("font-size: 13pt; font-weight: 700;")
        head.addWidget(mon_title, 1)
        self.live_pill = dw.LivePill(tr("dub.mon.live", "Real-time monitoring"),
                                     tr("dub.mon.idle", "Idle"))
        head.addWidget(self.live_pill, 0, Qt.AlignmentFlag.AlignVCenter)
        mon.addLayout(head)
        mon.addWidget(self.run_panel)
        self.idle_hint = QLabel("")
        self.idle_hint.setWordWrap(True)
        self.idle_hint.setObjectName("idleHint")
        self.idle_hint.setStyleSheet(
            f"QLabel#idleHint {{ color: {dw.GREY}; background: {dw.TILE_BG}; border: 1px dashed "
            f"{dw.TILE_BORDER}; border-radius: 8px; padding: 9px 12px; }}")
        mon.addWidget(self.idle_hint)

        # System Readiness Check: hardware tiles, then every check (folds)
        self.ready_head = dw.SectionHeader(tr("dub.ready.title", "System Readiness Check"),
                                           icon="check")
        self.ready_head.fold.setToolTip(tr("dub.ready.fold", "Show or hide every check"))
        self.ready_head.toggled.connect(lambda on: self._set_ready_open(on, user=True))
        recheck = QPushButton(tr("dub.recheck", "Recheck"))
        recheck.setIcon(QIcon(dw.icon_pixmap("refresh", theme.TEXT, 14)))
        recheck.clicked.connect(self._run_readiness)
        self.ready_head.right.addWidget(recheck)
        mon.addWidget(self.ready_head)
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.hw_tiles: dict[str, dw.HwTile] = {}
        for key in ("gpu", "cuda", "vram", "python"):
            self.hw_tiles[key] = dw.HwTile()
            tiles.addWidget(self.hw_tiles[key], 1)
        mon.addLayout(tiles)
        self.ready_body = dw.clear_widget()
        rb = QVBoxLayout(self.ready_body)
        rb.setContentsMargins(0, 0, 0, 0)
        rb.setSpacing(6)
        self.check_grid = QGridLayout()
        self.check_grid.setHorizontalSpacing(10)
        self.check_grid.setVerticalSpacing(1)
        rb.addLayout(self.check_grid)
        drow = QHBoxLayout()
        self.need_detail = QLabel("")
        self.need_detail.setWordWrap(True)
        self.need_detail.setStyleSheet(f"color: {dw.GREY};")
        # a long note (AI routes, a pip command) is cut off after 4 lines;
        # the whole note is the tooltip
        self.need_detail.setMaximumHeight(self.need_detail.fontMetrics().lineSpacing() * 4 + 6)
        self.need_detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # a fix with a long path (pip commands) has no spaces to wrap at: it
        # must not widen the monitor (it pushed its right edge out of view)
        self.need_detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        drow.addWidget(self.need_detail, 1)
        self.fix_btn = QPushButton(tr("dub.fix", "Fix selected"))
        self.fix_btn.clicked.connect(self._on_fix)
        drow.addWidget(self.fix_btn, 0, Qt.AlignmentFlag.AlignTop)
        rb.addLayout(drow)
        mon.addWidget(self.ready_body)
        self._check_rows: list[dw.CheckRow] = []
        self._sel_check = ""
        self._ready_user_set = False             # the user folded / opened it
        self._hw: dict = {}

        # Task List: counters, then each task with its note, time and bar
        self.task_head = dw.SectionHeader(self._task_list_title())
        self.task_head.fold.setToolTip(tr("dub.tasks.fold", "Show or hide the task list"))
        self.task_head.toggled.connect(lambda on: self.task_tree.setVisible(on))
        self.task_counts: dict[str, dw.Counter] = {}
        for key, icon, color in (("running", "clock", dw.BLUE), ("pending", "clock", dw.GREY),
                                 ("done", "check", dw.GREEN), ("warn", "warn", dw.AMBER),
                                 ("failed", "cross", dw.RED)):
            self.task_counts[key] = dw.Counter(icon, color)
            self.task_head.right.addWidget(self.task_counts[key])
        mon.addWidget(self.task_head)
        self.task_tree = QTreeWidget()
        self.task_tree.setColumnCount(3)
        self.task_tree.setHeaderHidden(True)
        self.task_tree.setRootIsDecorated(False)
        self.task_tree.setUniformRowHeights(False)
        self.task_tree.setItemDelegateForColumn(0, dw.TaskDelegate(self.task_tree))
        th = self.task_tree.header()
        th.setStretchLastSection(False)
        th.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        th.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        th.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.task_tree.setColumnWidth(1, 170)
        self.task_tree.setColumnWidth(2, 170)
        self.task_tree.setStyleSheet(
            "QTreeWidget { background: transparent; border: none; }"
            "QTreeWidget::item { border-bottom: 1px solid #1a2740; }"
            "QTreeWidget::item:hover { background: rgba(88,166,255,0.07); }"
            "QTreeWidget::item:selected { background: rgba(88,166,255,0.18); color: #e6edf3; }")
        for i, (sid, label) in enumerate(de.STEPS):
            it = QTreeWidgetItem([f"{i + 1}. {tr('dub.step.' + sid, label)}", "", ""])
            it.setData(0, Qt.ItemDataRole.UserRole, sid)
            self.task_tree.addTopLevelItem(it)
        self.task_tree.setCurrentItem(self.task_tree.topLevelItem(0))
        self.task_tree.setMinimumHeight(120)          # three or four tasks at least
        self._task_bar: QProgressBar | None = None
        self._task_bar_row = -1
        mon.addWidget(self.task_tree, 1)
        split.addWidget(self.monitor)

        self.side = dw.GlowCard()
        sv = QVBoxLayout(self.side)
        sv.setContentsMargins(12, 8, 12, 12)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_characters(), tr("dub.tab.chars", "Characters"))
        self.tabs.addTab(self._build_lines(), tr("dub.tab.lines", "Lines"))
        self.tabs.addTab(self._build_glossary(), tr("dub.tab.glossary", "Glossary && context"))
        self.tabs.addTab(self._build_output(), tr("dub.tab.output", "Output"))
        sv.addWidget(self.tabs)
        split.addWidget(self.side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([840, 560])
        root.addWidget(split, 1)

        self.status_label = dw.AutoHideLabel("")
        self.status_label.setStyleSheet("color: #8b949e;")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        root.addWidget(self.status_label)

        self._poll = QTimer(self)
        self._poll.setInterval(600)
        self._poll.timeout.connect(self._refresh_tasks)
        self._fill_series_combo()
        opened = self.ctx.config.get("ui.dub.settings_open", True) if self.ctx.config else True
        self._settings_auto_folded = False       # folded for a run: opened again after it
        self._short_folded = False               # folded for a short window: opened when taller
        self._was_short = False
        self._set_options_open(opened is not False)
        self._set_ready_open(False)
        self._render_hw()
        self._refresh_tasks()
        self._set_enabled()
        self._update_idle_hint()

    # ======================================================================
    # page builders
    # ======================================================================
    def _build_characters(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        hint = QLabel(tr("dub.chars.hint",
                         "Each detected speaker is a character. Name them, check the "
                         "gender, listen to the voice sample used for cloning and "
                         "replace it with a cleaner clip if needed."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8b949e;")
        info = QLabel()
        info.setPixmap(dw.icon_pixmap("info", dw.BLUE, 18))
        hint_row = QHBoxLayout()
        hint_row.setSpacing(8)
        hint_row.addWidget(info, 0, Qt.AlignmentFlag.AlignTop)
        hint_row.addWidget(hint, 1)
        v.addLayout(hint_row)
        self.char_table = QTableWidget(0, 10)
        self.char_table.setHorizontalHeaderLabels([
            "ID", tr("dub.c.name", "Name"), tr("dub.c.gender", "Gender"),
            tr("dub.c.lines", "Lines"), tr("dub.c.talk", "Talk"),
            tr("dub.c.engine", "Engine"), tr("dub.c.voice", "Preset voice"),
            tr("dub.c.pitch", "Pitch"), tr("dub.c.sample", "Voice sample"),
            tr("dub.c.reftext", "Sample transcript")])
        hh = self.char_table.horizontalHeader()
        hh.setSectionResizeMode(C_REFTEXT, QHeaderView.ResizeMode.Stretch)
        for c, wdt in ((C_ID, 44), (C_NAME, 140), (C_GENDER, 92), (C_LINES, 50),
                       (C_TALK, 60), (C_ENGINE, 110), (C_VOICE, 170),
                       (C_PITCH, 60), (C_SAMPLE, 270)):
            self.char_table.setColumnWidth(c, wdt)
        self.char_table.cellChanged.connect(self._on_char_cell)
        v.addWidget(self.char_table, 1)
        row = QHBoxLayout()
        self.merge_btn = QPushButton(tr("dub.merge", "Merge selected into first"))
        self.merge_btn.setToolTip(tr("dub.merge.tip",
                                     "Select two or more rows: their lines become one character."))
        self.merge_btn.clicked.connect(self._on_merge_chars)
        row.addWidget(self.merge_btn)
        row.addStretch(1)
        v.addLayout(row)
        return w

    def _build_lines(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.problems_check = QCheckBox(tr("dub.only_problems2",
                                           "Only lines to check (timing, ⚑, shortened)"))
        self.problems_check.toggled.connect(self._fill_lines)
        top.addWidget(self.problems_check)
        top.addStretch(1)
        self.lines_info = QLabel("")
        self.lines_info.setStyleSheet("color: #8b949e;")
        top.addWidget(self.lines_info)
        v.addLayout(top)
        self.line_table = QTableWidget(0, 9)
        self.line_table.setHorizontalHeaderLabels([
            "#", tr("dub.l.start", "Start"), tr("dub.l.end", "End"),
            tr("dub.l.speaker", "Speaker"), tr("dub.l.voice", "Voice"),
            tr("dub.l.src", "Original"),
            tr("dub.l.txt", "Translation (editable)"), tr("dub.l.fit", "Fit"), "▶"])
        hh = self.line_table.horizontalHeader()
        hh.setSectionResizeMode(L_SRC, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(L_TXT, QHeaderView.ResizeMode.Stretch)
        for c, wdt in ((L_NUM, 44), (L_START, 76), (L_END, 76), (L_SPK, 160),
                       (L_VOICE, 120), (L_FIT, 116), (L_PLAY, 34)):
            self.line_table.setColumnWidth(c, wdt)
        self.line_table.setWordWrap(True)
        self.line_table.cellChanged.connect(self._on_line_cell)
        self.line_table.cellDoubleClicked.connect(self._on_line_dclick)
        self.line_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.line_table.customContextMenuRequested.connect(self._on_lines_menu)
        v.addWidget(self.line_table, 1)
        hint = QLabel(tr("dub.l.voice_hint",
                         "Voice: blue = voiced with the other gender than its character; "
                         "yellow = sounds like the other gender — check it. Right-click "
                         "selected lines to set Male / Female."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8b949e;")
        v.addWidget(hint)
        return w

    def _voice_combo(self, i: int) -> QComboBox:
        p = self.project
        ln = p.lines[i]
        ch = p.character(ln.speaker)
        cg = ch.gender if ch else "F"
        cb = QComboBox()
        cb.addItem(tr("dub.l.voice.char", "Like character") + (" ♂" if cg == "M" else " ♀"), "")
        cb.addItem(tr("dub.male", "Male") + " ♂", "M")
        cb.addItem(tr("dub.female", "Female") + " ♀", "F")
        cb.setCurrentIndex(max(0, cb.findData(ln.gender)))
        tip = tr("dub.l.voice.tip", "Voice gender of this line")
        if ln.voice_gender:
            tip += (f"\n{tr('dub.l.measured', 'Sounds')}: "
                    + (tr("dub.male", "Male") if ln.voice_gender == "M" else tr("dub.female", "Female"))
                    + f" ({ln.voice_conf:.0%})")
        if ln.gender_src == "voice":
            tip += "\n" + tr("dub.l.auto_switched",
                             "Switched automatically: this character mixes two voices.")
            cb.setStyleSheet(f"QComboBox {{ background-color: {_SWITCHED_COLOR}; }}")
        elif (not ln.gender and ln.voice_gender and ln.voice_gender != cg
              and ln.voice_conf >= 0.6):
            tip += "\n" + tr("dub.l.check", "Sounds like the other gender — check it.")
            cb.setStyleSheet(f"QComboBox {{ background-color: {_UNSURE_COLOR}; }}")
        cb.setToolTip(tip)
        cb.currentIndexChanged.connect(
            lambda _x, idx=i, c=cb: self._set_line_gender([idx], c.currentData()))
        return cb

    def _selected_line_indexes(self) -> list[int]:
        out = []
        for r in sorted({ix.row() for ix in self.line_table.selectedIndexes()}):
            num = self.line_table.item(r, L_NUM)
            if num is not None:
                out.append(int(num.data(Qt.ItemDataRole.UserRole)))
        return out

    def _on_lines_menu(self, pos) -> None:
        if self.project is None:
            return
        rows = self._selected_line_indexes()
        if not rows:
            r = self.line_table.rowAt(pos.y())
            num = self.line_table.item(r, L_NUM) if r >= 0 else None
            rows = [int(num.data(Qt.ItemDataRole.UserRole))] if num else []
        if not rows:
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        n = f" ({len(rows)})" if len(rows) > 1 else ""
        acts = {menu.addAction(tr("dub.l.set_m", "Voice: Male") + n): "M",
                menu.addAction(tr("dub.l.set_f", "Voice: Female") + n): "F",
                menu.addAction(tr("dub.l.set_char", "Voice: like the character") + n): ""}
        lines = self.project.lines
        menu.addSeparator()
        flagged = [i for i in rows if lines[i].review]
        checked_act = (menu.addAction(tr("dub.l.checked", "Checked: remove ⚑") + n)
                       if flagged else None)
        live = [i for i in rows if not lines[i].skip]
        gone = [i for i in rows if lines[i].skip]
        skip_act = (menu.addAction(tr("dub.l.skip", "Nobody says this: leave the line out") + n)
                    if live else None)
        back_act = (menu.addAction(tr("dub.l.unskip", "Put the line back") + n)
                    if gone else None)
        chosen = menu.exec(self.line_table.viewport().mapToGlobal(pos))
        if chosen in acts:
            self._set_line_gender(rows, acts[chosen])
        elif chosen is not None and chosen is checked_act:
            for i in flagged:
                lines[i].review = ""
            self._fill_lines()
        elif chosen is not None and chosen is skip_act:
            self._set_line_skip(live, True)
        elif chosen is not None and chosen is back_act:
            self._set_line_skip(gone, False)

    def _set_line_skip(self, rows: list[int], skip: bool) -> None:
        """Leave lines out of the dub and subtitles (a sentence the
        recogniser invented), or put them back — they keep their place, so
        the other lines' numbers and voice files stay as they are."""
        p = self.project
        for i in rows:
            ln = p.lines[i]
            ln.skip = skip
            ln.review = ""
            if skip:
                ln.text, ln.full_text, ln.text_src, ln.tts_key = "", "", "", ""
                ln.status = ""
        # a line put back needs its translation (the memory and caches make
        # that one AI call); a line left out only changes the voice track
        p.invalidate_from("synthesize" if skip else "translate")
        self._fill_lines()
        self._refresh_tasks()

    def _build_glossary(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(tr("dub.context", "Story context (sent with every translation request):")))
        self.context_edit = QPlainTextEdit()
        self.context_edit.setMaximumHeight(120)
        self.context_edit.textChanged.connect(self._on_context_changed)
        v.addWidget(self.context_edit)
        v.addWidget(QLabel(tr("dub.glossary", "Glossary — names and terms are always written this way:")))
        self.gloss_table = QTableWidget(0, 3)
        self.gloss_table.setHorizontalHeaderLabels([
            tr("dub.g.src", "Original"), tr("dub.g.tgt", "Translation"),
            tr("dub.g.note", "Note")])
        self.gloss_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.gloss_table.cellChanged.connect(self._on_gloss_cell)
        v.addWidget(self.gloss_table, 1)
        row = QHBoxLayout()
        b = QPushButton(tr("dub.g.add", "Add term"))
        b.clicked.connect(self._on_gloss_add)
        row.addWidget(b)
        b = QPushButton(tr("dub.g.del", "Remove selected"))
        b.clicked.connect(self._on_gloss_del)
        row.addWidget(b)
        row.addStretch(1)
        self.save_series_btn = QPushButton(tr("dub.g.save_series", "Save cast && glossary to series"))
        self.save_series_btn.setToolTip(tr(
            "dub.g.save_series.tip",
            "Named characters (voice sample + voice fingerprint) and these terms are "
            "remembered for the next episodes. Also happens automatically on export."))
        self.save_series_btn.clicked.connect(self._on_save_series)
        row.addWidget(self.save_series_btn)
        note = QLabel(tr("dub.g.rerun", "After editing, run the Translate task again."))
        note.setStyleSheet("color: #8b949e;")
        row.addWidget(note)
        v.addLayout(row)
        return w

    def _build_output(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.music_spin = QDoubleSpinBox()
        self.music_spin.setRange(-30.0, 12.0)
        self.music_spin.setSuffix(" dB")
        f.addRow(tr("dub.music_gain", "Music && effects level:"), self.music_spin)
        self.voice_spin = QDoubleSpinBox()
        self.voice_spin.setRange(-30.0, 12.0)
        self.voice_spin.setSuffix(" dB")
        f.addRow(tr("dub.voice_gain", "New voices level:"), self.voice_spin)
        self.soft_check = QCheckBox(tr("dub.soft_subs", "Add subtitles as a selectable track"))
        self.soft_check.setChecked(True)
        f.addRow("", self.soft_check)
        self.burn_check = QCheckBox(tr("dub.burn_subs", "Burn subtitles into the picture (re-encodes)"))
        f.addRow("", self.burn_check)
        self.subfull_check = QCheckBox(tr("dub.sub_full", "Subtitles show the whole translation"))
        self.subfull_check.setChecked(de.DubOptions().subtitle_full)
        self.subfull_check.setToolTip(tr(
            "dub.sub_full.tip", "On: each subtitle shows the line's complete translation, even "
            "where the dub voice says a shorter version to fit the timing. Off: the "
            "subtitles say exactly what the voice says."))
        f.addRow("", self.subfull_check)
        self.sub_style = QComboBox()
        self.sub_style.addItem(tr("dub.sub_style.plain", "Plain (.srt only)"), "")
        from app.engines.subtitles import PRESETS
        for sid, st in PRESETS.items():
            self.sub_style.addItem(st.name, sid)
        self.sub_style.setToolTip(tr(
            "dub.sub_style.tip", "Also writes a styled .ass file; burned subtitles use it."))
        f.addRow(tr("dub.sub_style", "Subtitle style:"), self.sub_style)
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText(tr("dub.out.ph2", "Asked when you start: suggested <name>.dub.<lang>.mp4"))
        out_row.addWidget(self.out_edit, 1)
        b = QPushButton(tr("common.browse", "Browse…"))
        b.clicked.connect(self._on_pick_output)
        out_row.addWidget(b)
        holder = QWidget()
        holder.setLayout(out_row)
        f.addRow(tr("dub.output", "Output file:"), holder)
        btns = QHBoxLayout()
        self.play_out_btn = QPushButton(tr("dub.play_out", "Play dubbed video"))
        self.play_out_btn.clicked.connect(self._on_play_output)
        btns.addWidget(self.play_out_btn)
        self.show_out_btn = QPushButton(tr("dub.show_out", "Show in folder"))
        self.show_out_btn.clicked.connect(self._on_show_output)
        btns.addWidget(self.show_out_btn)
        self.srt_btn = QPushButton(tr("dub.export_srt", "Export .srt only"))
        self.srt_btn.clicked.connect(self._on_export_srt)
        btns.addWidget(self.srt_btn)
        btns.addStretch(1)
        bh = QWidget()
        bh.setLayout(btns)
        f.addRow("", bh)
        self.out_info = QLabel("")
        self.out_info.setWordWrap(True)
        self.out_info.setStyleSheet("color: #8b949e;")
        f.addRow("", self.out_info)
        return w

    # ======================================================================
    # lifecycle
    # ======================================================================
    def showEvent(self, ev):  # noqa: N802 (Qt override)
        if not self._checked_once:
            self._checked_once = True
            QTimer.singleShot(0, self._run_readiness)
        if self.project is None and self.ctx.current_video \
                and os.path.isfile(self.ctx.current_video):
            self._open_video(self.ctx.current_video)
        super().showEvent(ev)

    def _start(self, worker: FnWorker, *, row: bool = True) -> FnWorker:
        self._workers = [w for w in self._workers if w.isRunning()]
        self._workers.append(worker)
        if row:
            self.progress_row.set_worker(worker)
        self.ctx.pool.start(worker)
        return worker

    def busy_reason(self) -> str:
        """Why closing the app now would interrupt work ("" = it would not)."""
        if not self._busy:
            return ""
        return tr("dub.busy_close", "A dub is running ({where}).").replace("{where}", self._where())

    def _set_enabled(self) -> None:
        has = self.project is not None
        idle = not self._busy
        for wdg in (self.run_all_btn, self.run_step_btn, self.redo_btn,
                    self.save_btn, self.merge_btn, self.srt_btn,
                    self.save_series_btn):
            wdg.setEnabled(has and idle)
        self.tabs.setEnabled(idle)
        self.folder_btn.setEnabled(has)
        for wdg in (self.tgt_combo, self.src_combo, self.whisper_combo,
                    self.spk_spin, self.voice_mode, self.quality_combo,
                    self.tempo_spin, self.demucs_check, self.style_edit,
                    self.series_combo, self.tm_check, self.bg_combo, self.vary_check,
                    self.cuts_check):
            wdg.setEnabled(idle)
        self.live_pill.set_live(self._busy)

    def resizeEvent(self, ev):  # noqa: N802 (Qt override)
        """A short window (the 1100x700 minimum with the console open: the
        tab is ~525 px high) left the Characters / Lines table no room for
        a single row: below SHORT_TAB_H the settings fold away by
        themselves, and open again when the window is tall enough."""
        super().resizeEvent(ev)
        if not hasattr(self, "_was_short"):       # still being built
            return
        short = self.height() < SHORT_TAB_H
        if short == self._was_short:
            return
        self._was_short = short
        if short:
            # isHidden, not isVisible: the first resize comes before the tab
            # is on screen, when every child counts as not visible
            if not self.opt_box.isHidden() and not self._busy:
                self._set_options_open(False)
                self._short_folded = True
        elif self._short_folded:
            self._short_folded = False
            if not self._busy:
                self._set_options_open(True)

    def _set_options_open(self, on: bool, save: bool = False) -> None:
        """Fold the dubbing settings away (more room for the monitor).
        `save`: the user did it — remembered, and a fold made for a run is
        no longer undone when the run ends."""
        self.opt_box.setVisible(bool(on))
        self.opt_chev.setIcon(QIcon(dw.icon_pixmap("chevron_down" if on else "chevron_right",
                                                   dw.GREY, 16)))
        if save:
            self._settings_auto_folded = False
            self._short_folded = False           # the user's choice wins over the height
            if self.ctx.config is not None:
                try:
                    self.ctx.config.set("ui.dub.settings_open", bool(on))
                except Exception:
                    log.debug("could not remember the settings fold", exc_info=True)

    def _update_idle_hint(self) -> None:
        """What the monitor says when no run is on its panel."""
        if self.run_panel.state:
            self.idle_hint.hide()
            return
        p = self.project
        if p is None:
            text = tr("dub.mon.pick", "Pick a video above to start.")
        else:
            nxt = p.next_step()
            started = any(p.step_status(s) != "pending" for s in de.STEP_IDS)
            if nxt is None:
                text = tr("dub.mon.finished", "Every task is done.")
            elif started:
                text = (tr("dub.mon.paused", "Paused before {task}. Press the blue button above "
                                             "to continue.")
                        .replace("{task}", self._task_text(de.STEP_IDS.index(nxt))))
            else:
                text = tr("dub.mon.ready", "Ready. Press the blue button above to dub the whole "
                                           "video, or run the tasks one by one.")
        self.idle_hint.setText(text)
        self.idle_hint.show()

    # ======================================================================
    # readiness
    # ======================================================================
    def _run_readiness(self) -> None:
        self.ready_head.set_state("busy", "clock")
        self.ready_head.summary.setText(tr("dub.checking", "Checking…"))
        w = FnWorker(_readiness_fn, self.ctx.config, self._collect_options())
        w.finished_ok.connect(self._on_readiness_result)
        w.failed.connect(lambda en, km: self._show_need_error(en))
        self._start(w, row=False)

    def _on_readiness_result(self, result) -> None:
        checks, hw = result if isinstance(result, tuple) else (result, None)
        if hw is not None:
            self._hw = dict(hw)
        self._on_readiness(checks)

    def _show_need_error(self, en: str) -> None:
        self.ready_head.set_state("bad")
        self.ready_head.summary.setText(en)
        self.need_detail.setText(en)
        self._set_ready_open(True)

    @staticmethod
    def _check_label(c: de.Check) -> str:
        label = tr("dub.need." + c.id, c.label)
        if "{size}" in label:              # the Whisper model in use
            size = c.label.split("'")[1] if c.label.count("'") >= 2 else ""
            label = label.replace("{size}", size)
        return label

    def _on_readiness(self, checks) -> None:
        self._checks = list(checks or [])
        for row in self._check_rows:
            self.check_grid.removeWidget(row)
            row.deleteLater()
        self._check_rows = []
        for k, c in enumerate(self._checks):
            row = dw.CheckRow(c.id)
            fixable = bool(c.pip or c.download)
            tip = c.detail + ("" if c.ok else f"\n\n{tr('dub.howfix', 'How to fix')}: {c.fix}"
                              + (f"\n({tr('dub.dblclick', 'double-click or press Fix selected')})"
                                 if fixable else ""))
            row.set("ok" if c.ok else "bad" if c.required else "warn", self._check_label(c),
                    tr_text(c.detail.split("\n")[0]), tip)
            row.clicked.connect(self._select_check)
            row.double_clicked.connect(self._fix_check)
            self.check_grid.addWidget(row, k // 2, k % 2)
            self._check_rows.append(row)
        blockers = sum(1 for c in self._checks if not c.ok and c.required)
        optional = sum(1 for c in self._checks if not c.ok and not c.required)
        n, ok = len(self._checks), sum(1 for c in self._checks if c.ok)
        self.ready_head.set_state("bad" if blockers else "warn" if optional else "ok")
        summary = (tr("dub.ready.all", "All {n} checks ready").replace("{n}", str(n)) if ok == n
                   else tr("dub.ready.some", "{ok} of {n} ready").replace("{ok}", str(ok))
                   .replace("{n}", str(n)))
        if blockers:
            summary += " · " + tr("dub.ready.blockers", "{k} must be fixed").replace(
                "{k}", str(blockers))
        if optional:
            summary += " · " + tr("dub.ready.optional", "{k} optional").replace(
                "{k}", str(optional))
        self.ready_head.summary.setText(summary)
        first = (next((c for c in self._checks if not c.ok and c.required), None)
                 or next((c for c in self._checks if not c.ok), None)
                 or (self._checks[0] if self._checks else None))
        self._select_check(first.id if first else "")
        if not self._ready_user_set:          # a problem opens the list by itself
            self._set_ready_open(bool(blockers))
        self._render_hw()
        if blockers:
            self.status_label.setText(
                tr("dub.blockers", "Fix the red items before dubbing") + f" ({blockers}).")

    def _set_ready_open(self, on: bool, user: bool = False) -> None:
        self.ready_body.setVisible(bool(on))
        self.ready_head.set_open(bool(on))
        if user:
            self._ready_user_set = True

    def _select_check(self, cid: str) -> None:
        self._sel_check = cid
        for row in self._check_rows:
            row.set_selected(row.cid == cid)
        self._on_need_selected()

    def _fix_check(self, cid: str) -> None:
        self._select_check(cid)
        self._on_fix()

    def _render_hw(self) -> None:
        """The four hardware tiles (the GPU, CUDA for speech recognition,
        graphics memory, Python) from the last readiness check."""
        t, hw = self.hw_tiles, self._hw
        ready = tr("dub.hw.ready", "Ready")
        if not hw:
            for key, icon, name in (("gpu", "chip", "GPU"), ("cuda", "bolt", "CUDA"),
                                    ("vram", "memory", "VRAM"), ("python", "terminal", "Python")):
                t[key].set(icon, name, "none", tr("dub.checking", "Checking…"))
            return
        gpu = str(hw.get("gpu") or "")
        if gpu:
            t["gpu"].set("chip", re.sub(r"\s+(Laptop\s+)?GPU$", "", gpu), "ok", ready, gpu)
        else:
            t["gpu"].set("chip", "GPU", "warn", tr("dub.hw.no_gpu", "No graphics card found"))
        if hw.get("vendor") == "nvidia":
            title = "CUDA " + str(hw["cuda_ver"]) if hw.get("cuda_ver") else "CUDA"
            if hw.get("cuda"):
                t["cuda"].set("bolt", title, "ok", ready, tr(
                    "dub.hw.cuda.tip", "Speech recognition (Whisper) runs on the NVIDIA GPU: "
                    "several times faster than on the CPU."))
            else:
                t["cuda"].set("bolt", title, "warn", tr("dub.hw.cpu_only", "CPU only"), tr(
                    "dub.hw.cuda.off", "Speech recognition runs on the CPU. The check "
                    "“Speech recognition on the NVIDIA GPU” below can fix it."))
        else:
            t["cuda"].set("bolt", "CUDA", "none", tr("dub.hw.none", "Not available"),
                          tr("dub.hw.cuda.none", "CUDA needs an NVIDIA graphics card."))
        gb = hw.get("vram_gb")
        tip = tr("dub.hw.vram.tip", "Graphics memory. The most accurate speech model "
                                    "(Whisper large-v3) needs about 4.5 GB; 6 GB or more is best.")
        if gb:
            title = tr("dub.hw.vram", "{gb} GB VRAM").replace("{gb}", f"{float(gb):.0f}")
            if float(gb) >= 6:
                t["vram"].set("memory", title, "ok", ready, tip)
            else:
                t["vram"].set("memory", title, "warn", tr("dub.hw.low", "Low"), tip)
        else:
            t["vram"].set("memory", "VRAM", "none", tr("dub.hw.none", "Not available"), tip)
        t["python"].set("terminal", f"Python {hw.get('python', '')}".strip(), "ok", ready,
                        tr("dub.hw.python.tip", "The Python the app runs on, with its packages."))

    def _on_need_selected(self, *_a) -> None:
        c = self._selected_check()
        if c is None:
            self.need_detail.setText("")
            self.fix_btn.setEnabled(False)
            return
        text = tr_text(c.detail) if c.ok else f"{tr_text(c.detail)}\n→ {c.fix}"
        if not c.ok and (c.pip or c.download):
            text += "\n" + tr("dub.fix_hint", "Press “Fix selected” to install it.")
        # paths may break after \ and / (a zero-width space there)
        self.need_detail.setText(text.replace("\\", "\\​").replace("/", "/​"))
        self.need_detail.setToolTip(text)
        self.fix_btn.setEnabled(not c.ok)

    def _selected_check(self) -> de.Check | None:
        return next((c for c in self._checks if c.id == self._sel_check), None)

    def _on_fix(self) -> None:
        c = self._selected_check()
        if c is None:
            return
        if c.ok:
            QMessageBox.information(self, tr("dub.title", "Full Dub"),
                                    tr("dub.already_ok", "This item is already OK."))
            return
        if c.id == "ai" and c.required:          # no AI key at all: where to add one
            self._open_settings()
            return
        if not (c.pip or c.download):
            QMessageBox.information(self, tr("dub.title", "Full Dub"), c.fix)
            return
        steps = [f"pip install {' '.join(a)}" for a in c.pip]
        if c.download:
            steps.append(tr("dub.download_model", "download model") + f": {c.download}")
        ok = QMessageBox.question(
            self, tr("dub.title", "Full Dub"),
            tr("dub.fix_confirm", "Run these to fix it?") + "\n\n" + "\n".join(steps))
        if ok != QMessageBox.StandardButton.Yes:
            return
        w = FnWorker(_fix_fn, c)
        w.finished_ok.connect(lambda _r: self._run_readiness())
        self._start(w)

    # ======================================================================
    # project open / options
    # ======================================================================
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("dub.pick", "Choose video"),
                                              "", VIDEO_FILTER)
        if path:
            self._open_video(path)

    def _on_use_current(self) -> None:
        v = self.ctx.current_video
        if v and os.path.isfile(v):
            self._open_video(v)
        else:
            QMessageBox.information(self, tr("dub.title", "Full Dub"),
                                    tr("dub.no_current", "No video is open yet — use Browse or the Import tab."))

    def _open_video(self, path: str) -> None:
        if self._busy:
            QMessageBox.information(
                self, tr("dub.title", "Full Dub"),
                tr("dub.busy_open", "A dub is running. Wait for it to finish, or press "
                                    "Cancel on the progress panel, then open the other video."))
            return
        try:
            self.project = self._project_for(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("err.title", "Error"), str(exc))
            return
        self._last_run = None
        # the previous run's error belongs to that run: a language switch
        # redrew it over the newly opened video's state
        self._last_error = None
        self._run_seconds = 0.0
        self.run_panel.reset()
        stale = [s for s in de.STEP_IDS if self.project.step_status(s) == "running"]
        for sid in stale:            # the app was closed (or crashed) during it
            self.project.set_step(sid, "pending", "interrupted")
        if stale:
            try:
                self.project.save()
            except Exception:
                log.debug("could not save the interrupted state", exc_info=True)
        self.src_edit.setText(path)
        self._fill_series_combo()
        if not self.project.options.series:
            self.project.options.series = self._library_series(path)
        self._apply_options(self.project.options)
        self._fill_all()
        self._set_enabled()
        self._show_saved_state()
        self._update_idle_hint()        # says where the dub stands (ready / paused / done)
        self.status_label.setText("")

    def _project_for(self, path: str) -> de.DubProject:
        """The video's project. None yet, but an earlier dub of it was made
        before the video was moved (same name and length, its old place
        empty): offer to go on with that one instead of starting again."""
        p = de.DubProject.for_video(path)
        if os.path.isfile(p.path):
            return p
        try:
            moved = de.find_moved_project(path)
            old = de.DubProject.load(moved) if moved else None
        except Exception:
            log.debug("could not look for a moved project", exc_info=True)
            old = None
        if old is None:
            return p
        done = sum(1 for s in de.STEP_IDS if old.step_done(s))
        ans = QMessageBox.question(
            self, tr("dub.title", "Full Dub"),
            tr("dub.moved.q", "An earlier dub of this video was found ({done} of {n} tasks "
                              "done). It was made when the video was at:\n{old}\n\nGo on "
                              "with that dub, with the video at its new place?")
            .replace("{done}", str(done)).replace("{n}", str(len(de.STEP_IDS)))
            .replace("{old}", old.video))
        if ans != QMessageBox.StandardButton.Yes:
            return p
        return de.DubProject.relink(moved, path)

    def _library_series(self, path: str) -> str:
        try:
            ep = self.ctx.library.find_episode_by_path(path)
            if ep:
                for s in self.ctx.library.get_series():
                    if s.get("id") == ep.get("series_id"):
                        return str(s.get("name") or "")
        except Exception:
            log.debug("library series lookup failed", exc_info=True)
        return ""

    def _fill_series_combo(self) -> None:
        names: list[str] = []
        try:
            names += [str(s.get("name") or "") for s in self.ctx.library.get_series()]
        except Exception:
            pass
        try:
            from app.core.series_memory import list_series_memories
            names += list_series_memories()
        except Exception:
            pass
        cur = self.series_combo.currentText()
        self.series_combo.clear()
        self.series_combo.addItem("")
        for n in sorted({n for n in names if n}, key=str.lower):
            self.series_combo.addItem(n)
        self.series_combo.setCurrentText(cur)

    def _on_save_series(self) -> None:
        p = self.project
        if p is None:
            return
        self._sync_options()
        if not p.options.series:
            QMessageBox.information(self, tr("dub.title", "Full Dub"),
                                    tr("dub.series.need", "Choose or type a series name first."))
            return
        self._pull_glossary()
        try:
            msg = self.engine.save_to_series(p, manual=True)
            p.save()
        except Exception as exc:
            QMessageBox.warning(self, tr("err.title", "Error"), str(exc))
            return
        self.status_label.setText(tr("dub.series.saved", "Saved") + f": {msg}")

    def _apply_options(self, o: de.DubOptions) -> None:
        def sel(combo: QComboBox, data) -> None:
            i = combo.findData(data)
            if i >= 0:
                combo.setCurrentIndex(i)
        sel(self.tgt_combo, o.target_lang)
        sel(self.src_combo, o.source_lang)
        self.whisper_combo.setCurrentText(o.whisper_size)
        self.spk_spin.setValue(int(o.num_speakers))
        sel(self.voice_mode, o.voice_mode)
        sel(self.quality_combo, o.quality)
        self.tempo_spin.setValue(float(o.max_tempo))
        self.demucs_check.setChecked(bool(o.use_demucs))
        self.style_edit.setText(o.style)
        self.music_spin.setValue(float(o.music_gain_db))
        self.voice_spin.setValue(float(o.voice_gain_db))
        self.soft_check.setChecked(bool(o.soft_subtitles))
        self.burn_check.setChecked(bool(o.burn_subtitles))
        self.subfull_check.setChecked(bool(o.subtitle_full))
        self.cuts_check.setChecked(bool(o.allow_cuts))
        self.out_edit.setText(o.output_path)
        self.series_combo.setCurrentText(o.series)
        self.tm_check.setChecked(bool(o.use_tm))
        sel(self.sub_style, o.subtitle_style)
        sel(self.bg_combo, o.background)
        self.vary_check.setChecked(bool(o.vary_pitch))

    def _collect_options(self) -> de.DubOptions:
        return de.DubOptions(
            target_lang=self.tgt_combo.currentData() or "km",
            source_lang=self.src_combo.currentData() or "",
            whisper_size=self.whisper_combo.currentText(),
            use_demucs=self.demucs_check.isChecked(),
            num_speakers=self.spk_spin.value(),
            voice_mode=self.voice_mode.currentData() or "pair",
            vary_pitch=self.vary_check.isChecked(),
            background=self.bg_combo.currentData() or "keep",
            quality=self.quality_combo.currentData() or "high",
            style=self.style_edit.text().strip(),
            max_tempo=self.tempo_spin.value(),
            music_gain_db=self.music_spin.value(),
            voice_gain_db=self.voice_spin.value(),
            soft_subtitles=self.soft_check.isChecked(),
            burn_subtitles=self.burn_check.isChecked(),
            subtitle_full=self.subfull_check.isChecked(),
            allow_cuts=self.cuts_check.isChecked(),
            output_path=self.out_edit.text().strip(),
            series=self.series_combo.currentText().strip(),
            use_tm=self.tm_check.isChecked(),
            subtitle_style=self.sub_style.currentData() or "",
        )

    def _sync_options(self) -> None:
        """Push UI options into the project, invalidating what they affect."""
        p = self.project
        if p is None:
            return
        new, old = self._collect_options(), p.options
        if new.use_demucs != old.use_demucs or (
                new.background != old.background and new.background == "keep"
                and not (p.music_wav and os.path.isfile(p.music_wav))):
            p.invalidate_from("prepare")      # the music must be separated first
        elif (new.whisper_size, new.source_lang) != (old.whisper_size, old.source_lang):
            p.invalidate_from("transcribe")
        elif (new.num_speakers, new.series) != (old.num_speakers, old.series):
            p.invalidate_from("speakers")
        elif new.target_lang != old.target_lang:
            p.invalidate_from("glossary")
        elif (new.quality, new.style, new.use_tm) != (old.quality, old.style, old.use_tm):
            p.invalidate_from("translate")
        elif (new.max_tempo, new.allow_cuts) != (old.max_tempo, old.allow_cuts):
            # from Translate, not Fit: lines Fit had reworded or cut at the
            # old speed get their whole translation back first (translation
            # memory and the caches make that quick and free)
            p.invalidate_from("translate")
        elif (new.voice_mode, new.vary_pitch) != (old.voice_mode, old.vary_pitch):
            p.invalidate_from("synthesize")
        elif (new.music_gain_db, new.voice_gain_db, new.background) != \
                (old.music_gain_db, old.voice_gain_db, old.background):
            p.invalidate_from("mix")
        elif (new.soft_subtitles, new.burn_subtitles, new.output_path,
              new.subtitle_style, new.subtitle_full) != (
                old.soft_subtitles, old.burn_subtitles, old.output_path,
                old.subtitle_style, old.subtitle_full):
            p.invalidate_from("export")
        p.options = new

    def _save_project(self) -> None:
        if self.project is None:
            return
        self._sync_options()
        try:
            self.project.save()
            self.status_label.setText(tr("dub.saved", "Project saved") + f": {self.project.path}")
        except Exception as exc:
            QMessageBox.warning(self, tr("err.title", "Error"), str(exc))
        self._refresh_tasks()

    def _open_project_folder(self) -> None:
        if self.project is not None:
            os.makedirs(self.project.work_dir, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.project.work_dir))

    # ======================================================================
    # running tasks
    # ======================================================================
    def _selected_step(self) -> str | None:
        it = self.task_tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def _blocking_checks(self) -> list[de.Check]:
        return [c for c in self._checks if c.required and not c.ok]

    def _confirm_blockers(self) -> bool:
        # no AI key: asked on its own, with what can be done (_ask_no_key)
        bad = [c for c in self._blocking_checks() if c.id != "ai"]
        if not bad:
            return True
        names = "\n".join(f"• {c.label}: {c.fix}" for c in bad)
        ans = QMessageBox.question(
            self, tr("dub.title", "Full Dub"),
            tr("dub.blockers_q", "Some requirements are missing:") + f"\n\n{names}\n\n"
            + tr("dub.continue_q", "Continue anyway?"))
        return ans == QMessageBox.StandardButton.Yes

    # -- no AI key yet ------------------------------------------------------
    def _no_ai_key(self) -> bool:
        """No provider has an API key (asked now: a key added in Settings
        a moment ago counts, whatever the last readiness check said)."""
        return no_ai_key(self.ctx.config)

    def _ai_steps_ahead(self, fn, args) -> bool:
        """The run `fn(*args)` would reach a task that asks the AI."""
        p = self.project
        if p is None or fn == self.engine.run_until_ai:
            return False
        if fn == self.engine.run_step:
            return args[1] in de.AI_STEPS
        return any(not p.step_done(s) for s in de.AI_STEPS)

    def _ask_no_key(self, can_transcribe: bool) -> str:
        """No AI key yet: "settings" (open Settings), "transcribe" (the tasks
        that need no AI) or "" (cancel). (A method: tests answer it.)"""
        return ask_no_key(self, can_transcribe)

    def _open_settings(self) -> None:
        open_settings(self.ctx)

    def _failed_ai_kind(self) -> str:
        """Why the failed task found no AI ("no_key", "budget"…; "" = not that)."""
        p = self.project
        if p is None:
            return ""
        failed = next((s for s in de.STEP_IDS if p.step_status(s) == "failed"), None)
        return str(p.steps[failed].get("ai") or "") if failed else ""

    # -- a finished dub whose video is gone ------------------------------
    def _output_missing(self) -> bool:
        """Every task is done, but the dubbed video is not where it was saved
        (moved, renamed or deleted)."""
        p = self.project
        return (p is not None and p.next_step() is None
                and not (p.output and os.path.isfile(p.output)))

    def _remake_video(self) -> None:
        """Save the dubbed video again from the finished work: the export
        (and the mix, if its file is gone too) — no AI, no voices."""
        p = self.project
        if p is None or self._busy:
            return
        if not os.path.isfile(p.video):
            self._show_saved_state()
            return
        if not self._ask_output(force=True):
            return
        p.invalidate_from("mix" if not (p.mix_wav and os.path.isfile(p.mix_wav)) else "export")
        self._begin(self.engine.run_all, p)

    # -- where the dubbed video goes -------------------------------------
    def _planned_output(self) -> str:
        p = self.project
        if p is None:
            return ""
        return self.out_edit.text().strip() or p.options.output_path or self.engine.default_output(p)

    def _ask_output(self, force: bool = False) -> bool:
        """Ask where the dubbed video goes (once per video; `force` = the
        Change… button). The folder must be writable and roomy enough — a
        2-hour dub must not fail at its last step for want of either.
        Returns False if the user backed out."""
        p = self.project
        if p is None:
            return False
        if not force and (self.out_edit.text().strip() or p.options.output_path):
            return True
        name = os.path.basename(self.engine.default_output(p))
        last_dir = str(self.ctx.config.get("dub.out_dir") or "")
        current = self.out_edit.text().strip() or p.options.output_path
        start = current or os.path.join(
            last_dir if os.path.isdir(last_dir) else os.path.dirname(p.video), name)
        while True:
            path, _ = QFileDialog.getSaveFileName(
                self, tr("dub.save_as", "Save dubbed video as"), start,
                "MP4 (*.mp4);;MKV (*.mkv)")
            if not path:
                return False
            if os.path.splitext(path)[1].lower() not in (".mp4", ".mkv"):
                path += ".mp4"
            path = os.path.normpath(path)
            folder = os.path.dirname(path)
            problem = self._output_problem(path)
            if problem:
                ans = QMessageBox.warning(
                    self, tr("dub.title", "Full Dub"), problem,
                    QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Ignore
                    | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Retry)
                if ans == QMessageBox.StandardButton.Retry:
                    start = path
                    continue
                if ans == QMessageBox.StandardButton.Cancel:
                    return False
            break
        self.out_edit.setText(path)
        try:
            self.ctx.config.set("dub.out_dir", folder)
        except Exception:
            log.debug("could not remember the output folder", exc_info=True)
        if self.run_panel.state in ("error", "stopped", "partial"):
            self.run_panel.set_path(tr("dub.run.save_to", "Save to") + f": {path}")
        if self.project is not None and not self._busy:
            self._sync_options()
            self.project.save()
            self._refresh_tasks()
        return True

    def _output_problem(self, path: str) -> str:
        """Why `path` cannot take the dubbed video ("" = fine)."""
        p = self.project
        folder = os.path.dirname(path) or "."
        if p is not None and os.path.normcase(os.path.abspath(path)) == \
                os.path.normcase(os.path.abspath(p.video)):
            return tr("dub.out.same", "That is the original video — choose another name, "
                                      "or the dub would replace it.")
        try:
            os.makedirs(folder, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".rs_write_test_"):
                pass
        except OSError as exc:
            return tr("dub.out.no_write", "Cannot save in this folder") + f":\n{folder}\n\n{exc}"
        try:
            need = os.path.getsize(p.video) * 1.1 + 50e6 if p is not None else 0
            free = shutil.disk_usage(folder).free
            if need and free < need:
                return (tr("dub.out.no_space", "Not enough free space there: {free} free, "
                                               "the dubbed video needs about {need}.")
                        .replace("{free}", _fmt_bytes(free)).replace("{need}", _fmt_bytes(need)))
        except OSError:
            pass
        return ""

    # -- running ------------------------------------------------------------
    def _begin(self, fn, *args) -> None:
        if self.project is None:
            return
        self._sync_options()
        self.project.save()
        self._busy = True
        if self.opt_box.isVisible():
            # the settings cannot change during a run: their room goes to
            # the monitor until it ends
            self._set_options_open(False)
            self._settings_auto_folded = True
        self._set_enabled()
        self._last_run = (fn, args)
        self._todo_at_start = {s for s in de.STEP_IDS if not self.project.step_done(s)}
        self._cur_step, self._cur_step_t0, self._last_error = "", 0.0, None
        self._run_label = self._step_name(args[1]) if len(args) > 1 else ""
        # the bar shows the whole dub: run_all reports over the tasks left,
        # run_step within its one task — both by the same task weights
        p, total = self.project, float(sum(de.STEP_WEIGHTS.values()))
        if len(args) > 1:
            self._p_base = de.overall_progress(p, running=args[1], frac=0.0)
            self._p_span = de.STEP_WEIGHTS.get(args[1], 0) / total
        else:
            self._p_base = de.overall_progress(p)
            self._p_span = 1.0 - self._p_base
        w = FnWorker(fn, *args)
        w.progress.connect(self._on_run_progress)
        w.finished_ok.connect(self._on_task_ok)
        w.failed.connect(self._on_task_failed)
        w.cancelled.connect(self._on_task_cancelled)
        self._run_worker = w
        # "Transcribe only" saves no video: no "Save to:" line for it
        self.run_panel.start("" if fn == self.engine.run_until_ai else self._planned_output(),
                             self._p_base, self._run_seconds)
        self.status_label.setText("")
        _keep_awake(True)
        self._start(w, row=False)
        self._poll.start()

    def _on_run_all(self) -> None:
        p = self.project
        if p is None:
            return
        # a setting changed since the dub finished (a faster "Max speed-up")
        # marks the tasks it affects to be done again
        self._sync_options()
        if p.next_step() is None:              # nothing left to do
            if self._output_missing():         # …but the video was moved / deleted
                self._remake_video()
                return
            self._show_saved_state()
            self.status_label.setText(tr("dub.all_done", "Everything is already done. "
                                         "Use “Redo from selected task” to make it again."))
            return
        if not self._confirm_blockers():
            return
        fn = self.engine.run_all
        if self._no_ai_key() and self._ai_steps_ahead(fn, (p,)):
            first_ai = de.STEP_IDS.index(de.FIRST_AI_STEP)
            choice = self._ask_no_key(de.STEP_IDS.index(p.next_step()) < first_ai)
            if choice == "settings":
                self._open_settings()
            if choice != "transcribe":
                return
            fn = self.engine.run_until_ai      # the dubbed video is asked for later
        elif not self._ask_output():
            return
        self._begin(fn, p)

    def _on_run_selected(self) -> None:
        step = self._selected_step()
        if self.project is None or not step:
            return
        if step == "export" and not self._ask_output():
            return
        self._begin(self.engine.run_step, self.project, step)

    def _on_redo_from(self) -> None:
        step = self._selected_step()
        if self.project is None or not step:
            return
        label = de.dict_label(step)
        ans = QMessageBox.question(
            self, tr("dub.title", "Full Dub"),
            tr("dub.redo_q", "Redo this task and everything after it?") + f"\n\n{label}")
        if ans != QMessageBox.StandardButton.Yes:
            return
        if not self._ask_output():
            return
        self.project.invalidate_from(step)
        self._begin(self.engine.run_all, self.project)

    def _on_continue(self) -> None:
        """Continue after an error / a stop: the same run again — finished
        tasks are skipped and finished parts of the stopped one reused."""
        if self.project is None or self._busy:
            return
        if self._output_missing():             # "Make it again"
            self._remake_video()
            return
        fn, args = self._last_run or (self.engine.run_all, (self.project,))
        if fn == self.engine.run_step and self.project.step_done(args[1]):
            fn, args = self.engine.run_all, (self.project,)      # that task is done now
        if (fn == self.engine.run_until_ai and self.project.next_step()
                and de.STEP_IDS.index(self.project.next_step())
                >= de.STEP_IDS.index(de.FIRST_AI_STEP)):
            fn, args = self.engine.run_all, (self.project,)      # transcribed: now the rest
        if fn == self.engine.run_all and self.project.next_step() is None:
            self._show_saved_state()
            return
        if self._no_ai_key() and self._ai_steps_ahead(fn, args):
            if self._ask_no_key(False) == "settings":
                self._open_settings()
            return
        if (fn == self.engine.run_all or args[1:] == ("export",)) and not self._ask_output():
            return
        self._begin(fn, *((self.project,) + tuple(args[1:])))

    def _on_cancel_run(self) -> None:
        w = self._run_worker
        if w is not None and w.isRunning():
            self.run_panel.set_stopping()
            w.cancel()

    def _on_dismiss_error(self) -> None:
        """Cancel on an error: stop here (nothing is undone)."""
        self.run_panel.set_stopped(self._where(), tr("dub.run.cancelled_note",
                                                     "Cancelled after the error."))

    def _on_run_progress(self, frac: float, text: str) -> None:
        # run_all prefixes the task's English name ("Translate…: lines 1-30");
        # the panel title already names the task (in the UI language)
        for _sid, label in de.STEPS:
            if text.startswith(label + ": "):
                text = text[len(label) + 2:]
                break
        self._last_overall = self._p_base + self._p_span * frac
        self.run_panel.set_progress(self._last_overall, tr_text(text))
        if self._task_bar is not None and 0 <= self._task_bar_row < len(de.STEP_IDS):
            within = self._task_fraction(de.STEP_IDS[self._task_bar_row])
            self._task_bar.setValue(max(self._task_bar.value(), int(within * 1000)))

    def _where(self) -> str:
        """'task 6 of 10: Translate dialogue' for the failed / stopped /
        running task (from the project's task states)."""
        p = self.project
        if p is None:
            return ""
        for want in ("running", "failed"):
            for i, sid in enumerate(de.STEP_IDS):
                if p.step_status(sid) == want:
                    return self._task_text(i)
        nxt = p.next_step()
        return self._task_text(de.STEP_IDS.index(nxt)) if nxt else ""

    @staticmethod
    def _step_name(sid: str) -> str:
        return tr("dub.step." + sid, de.dict_label(sid))

    def _task_text(self, i: int) -> str:
        sid, label = de.STEPS[i]
        return (tr("dub.run.task_of", "task {i} of {n}: {task}")
                .replace("{i}", str(i + 1)).replace("{n}", str(len(de.STEPS)))
                .replace("{task}", tr("dub.step." + sid, label)))

    def _run_ended(self) -> None:
        self._run_seconds += self.run_panel.segment_seconds()
        if self.project is not None and self._todo_at_start:
            try:
                de.learn_speeds(self.ctx.config, self.project, sorted(self._todo_at_start))
            except Exception:
                log.debug("could not learn task timings", exc_info=True)
        self._todo_at_start = set()
        self._poll.stop()
        self._busy = False
        self._run_worker = None
        _keep_awake(False)
        if self._settings_auto_folded:
            self._settings_auto_folded = False
            if self._was_short:                  # still a short window: stay folded
                self._short_folded = True
            else:
                self._set_options_open(True)
        self._set_enabled()
        self._fill_all()
        self._update_idle_hint()
        win = self.window()
        if win is not None and not win.isActiveWindow():
            QApplication.alert(win)            # flash the taskbar button

    def _on_task_ok(self, _result) -> None:
        self._run_ended()
        p = self.project
        if p is None:
            return
        if p.next_step() is None and p.output:
            self._show_saved_state()
            self.status_label.setText(tr("dub.saved_to", "Saved to") + f": {p.output}")
        elif self._last_run and self._last_run[0] == self.engine.run_until_ai:
            self._show_transcribed()
        else:
            nxt = p.next_step()
            done = self._run_label or tr("dub.run.some", "the tasks you ran")
            self.run_panel.set_partial(done, self._task_text(de.STEP_IDS.index(nxt)) if nxt else "")
            self.run_panel.set_fraction(de.overall_progress(p))

    def _show_transcribed(self) -> None:
        """After "Transcribe only": what was done, and what is next — an
        AI key (with Open Settings) or, once there is one, Continue."""
        p = self.project
        if p is None:
            return
        title = (tr("dub.run.transcribed", "Transcribed: {n} lines, {k} speakers")
                 .replace("{n}", str(len(p.lines))).replace("{k}", str(len(p.characters))))
        if self._no_ai_key():
            detail = tr("dub.run.transcribed_next", "Next: add an AI key in Settings → AI "
                                                    "Providers, then press Continue to translate "
                                                    "and dub the rest.")
            self.run_panel.set_partial("", "", title=title, detail=detail, settings=True)
        else:
            nxt = p.next_step()
            self.run_panel.set_partial("", self._task_text(de.STEP_IDS.index(nxt)) if nxt else "",
                                       title=title)
        self.run_panel.set_fraction(de.overall_progress(p))

    def _on_task_failed(self, en: str, km: str) -> None:
        self._run_ended()
        self._last_error = (en, km)
        self._show_error()

    def _show_error(self) -> None:
        en, km = self._last_error or ("", "")
        msg = km if get_language() == "km" and km else en
        tip = en if not km or km == en else f"{en}\n\n{km}"
        kind = self._failed_ai_kind()        # set: no AI could answer (a pause)
        self.run_panel.set_error(self._where(), msg, tip,
                                 settings=kind in _FIX_IN_SETTINGS, paused=bool(kind))

    def _on_task_cancelled(self) -> None:
        self._run_ended()
        self.run_panel.set_stopped(self._where(), tr("dub.run.you_stopped",
                                                     "You stopped the dub."))

    def _show_saved_state(self) -> None:
        """What the panel shows for an opened / finished project: the saved
        video when everything is done, the error where it stopped, or
        nothing for a project that is simply not finished yet."""
        p = self.project
        if p is None:
            self.run_panel.reset()
            return
        if p.next_step() is None and p.output and os.path.isfile(p.output):
            info = []
            try:
                info.append(_fmt_bytes(os.path.getsize(p.output)))
            except OSError:
                pass
            if p.duration:
                info.append(_fmt_clock(p.duration))
            notes = [s for s in de.STEP_IDS if p.step_status(s) == "warn"]
            note = (tr("dub.run.notes", "{n} task(s) finished with notes — see Tasks on the left.")
                    .replace("{n}", str(len(notes)))) if notes else ""
            self.run_panel.set_done(p.output, " · ".join(info), note)
            self.out_info.setText(tr("dub.saved_to", "Saved to") + f": {p.output}"
                                  + (f"\n{p.srt_path}" if p.srt_path else ""))
            return
        if p.next_step() is None:
            # every task done, but the video is not where it was saved: say
            # so, and offer to save it again (export only — no AI, no voices)
            gone = p.output or self._planned_output()
            title = tr("dub.run.missing", "The dubbed video is missing")
            if os.path.isfile(p.video):
                self.run_panel.set_stopped(
                    "", tr("dub.run.missing_note", "It is no longer at {path}: moved, renamed "
                           "or deleted. “Make it again” saves it once more from the finished "
                           "work (only the last task runs, no AI).").replace("{path}", gone),
                    title=title, change=False, hint="",
                    action=tr("dub.run.remake", "Make it again"))
            else:
                self.run_panel.set_stopped(
                    "", tr("dub.run.missing_both", "It is no longer at {path}, and the original "
                           "video is missing too ({video}), so it cannot be made again. Put the "
                           "original video back there to make it again.")
                    .replace("{path}", gone).replace("{video}", p.video),
                    title=title, change=False, hint="", cont=False)
            self.run_panel.set_fraction(1.0)
            self.run_panel.set_path("")
            self.out_info.setText("")
            return
        failed = next((s for s in de.STEP_IDS if p.step_status(s) == "failed"), None)
        if failed:
            st = p.steps[failed]
            note = str(st.get("note") or "")
            kind = str(st.get("ai") or "")
            msg = note
            if get_language() == "km":
                # the engine's Khmer text, else the advice for an AI pause
                # (older projects), else a known message; English when
                # there is no Khmer for it
                msg = (str(st.get("note_km") or "")
                       or (AI_ADVICE.get(kind, AI_ADVICE[""])[1] if kind else "")
                       or tr_text(note) or note)
            self.run_panel.set_error(self._where(), msg, note if msg != note else "",
                                     settings=kind in _FIX_IN_SETTINGS, paused=bool(kind))
            self.run_panel.set_fraction(de.overall_progress(p))
            self.run_panel.set_path(tr("dub.run.save_to", "Save to") + f": {self._planned_output()}")
            return
        stopped = next((s for s in de.STEP_IDS
                        if p.steps[s].get("note") in ("cancelled", "interrupted")), None)
        if stopped:
            self.run_panel.set_stopped(
                self._where(), tr("dub.run.interrupted", "The app was closed during this task.")
                if p.steps[stopped].get("note") == "interrupted" else "")
            self.run_panel.set_fraction(de.overall_progress(p))
            self.run_panel.set_path(tr("dub.run.save_to", "Save to") + f": {self._planned_output()}")
            return
        self.run_panel.reset()

    def _update_run_button(self) -> None:
        p = self.project
        nxt = p.next_step() if p is not None else None
        started = p is not None and any(p.step_status(s) != "pending" for s in de.STEP_IDS)
        if p is not None and nxt and started:
            i = de.STEP_IDS.index(nxt)
            self.run_all_btn.setText(
                tr("dub.run.continue_btn", "▶  Continue dubbing (task {i} of {n})")
                .replace("{i}", str(i + 1)).replace("{n}", str(len(de.STEPS))))
        elif p is not None and self._output_missing():
            self.run_all_btn.setText(tr("dub.run.remake_btn", "▶  Make the dubbed video again"))
        else:
            self.run_all_btn.setText(tr("dub.run_all2", "▶  Translate && dub whole video"))

    # ======================================================================
    # filling the views from the project
    # ======================================================================
    def _fill_all(self) -> None:
        self._refresh_tasks()
        self._fill_chars()
        self._fill_lines()
        self._fill_glossary()
        p = self.project
        if p is not None and p.output and os.path.isfile(p.output):
            self.out_info.setText(tr("dub.saved_to", "Saved to") + f": {p.output}")

    def _task_list_title(self) -> str:
        return tr("dub.tasks.title", "Task List ({n} tasks)").replace("{n}", str(len(de.STEPS)))

    def _refresh_tasks(self) -> None:
        p = self.project
        running = None
        if p is not None and self._busy:
            running = next((i for i, s in enumerate(de.STEP_IDS)
                            if p.step_status(s) == "running"), None)
            if running is not None:
                sid = de.STEP_IDS[running]
                self.run_panel.set_task(running + 1, len(de.STEPS), self._step_name(sid))
                self._update_eta(sid)
        counts = dict.fromkeys(("running", "pending", "done", "warn", "failed"), 0)
        for i, (sid, label) in enumerate(de.STEPS):
            it = self.task_tree.topLevelItem(i)
            st = p.steps[sid] if p else {"status": "pending", "note": ""}
            status = st.get("status", "pending")
            raw = str(st.get("note") or "")
            stopped = status == "pending" and raw in ("cancelled", "interrupted")
            counts[status if status in counts else "pending"] += 1
            note = "" if stopped else "; ".join(tr_text(part) for part in raw.split("; ") if part)
            if status == "failed" and get_language() == "km" and st.get("note_km"):
                note = str(st["note_km"])            # the engine's own Khmer text
            it.setText(0, f"{i + 1}. {tr('dub.step.' + sid, label)}" + (f"\n{note}" if note else ""))
            it.setToolTip(0, raw or label)
            it.setData(0, dw.STATUS_ROLE, status)
            kind = "stopped" if stopped else status
            it.setIcon(0, self._task_icon(kind))
            text, color = self._task_status(kind, st, i == running)
            it.setText(2, text)
            it.setForeground(2, QColor(color))
            it.setTextAlignment(2, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            it.setToolTip(2, raw or text)
        self._place_task_bar(running)
        self.task_head.title.setText(self._task_list_title())
        cnt = self.task_counts
        cnt["running"].set_text(tr("dub.tasks.running", "Running: {n}").replace(
            "{n}", str(counts["running"])))
        cnt["pending"].set_text(tr("dub.tasks.pending", "Pending: {n}").replace(
            "{n}", str(counts["pending"])))
        cnt["done"].set_text(tr("dub.tasks.completed", "Completed: {n}").replace(
            "{n}", str(counts["done"] + counts["warn"])))
        cnt["warn"].set_text(tr("dub.tasks.notes", "With notes: {n}").replace(
            "{n}", str(counts["warn"])))
        cnt["failed"].set_text(tr("dub.tasks.failed", "Failed: {n}").replace(
            "{n}", str(counts["failed"])))
        cnt["warn"].setVisible(counts["warn"] > 0)
        cnt["failed"].setVisible(counts["failed"] > 0)
        if p is not None:
            done = sum(1 for s in de.STEP_IDS if p.step_done(s))
            self.lines_info.setText(
                f"{len(p.lines)} " + tr("dub.lines_n", "lines") + f" · {done}/{len(de.STEPS)} "
                + tr("dub.tasks_done", "tasks done"))
        if not self._busy:
            self._update_run_button()

    _TASK_ICONS: dict = {}

    def _task_icon(self, kind: str) -> QIcon:
        """The round icon of a task row (painted once per kind)."""
        if kind not in DubTab._TASK_ICONS:
            spec = {"running": ("play", dw.BLUE, True), "done": ("check", dw.GREEN, False),
                    "warn": ("warn", dw.AMBER, False), "failed": ("cross", dw.RED, False),
                    "stopped": ("stop", dw.AMBER, False)}.get(kind, ("pending", dw.GREY, False))
            DubTab._TASK_ICONS[kind] = QIcon(dw.round_pixmap(spec[0], spec[1], 20, solid=spec[2]))
        return DubTab._TASK_ICONS[kind]

    def _task_status(self, kind: str, st: dict, live: bool) -> tuple[str, str]:
        """The right-hand column of a task row: its state and time."""
        secs = st.get("secs")
        took = f" · {_fmt_clock(float(secs))}" if secs and float(secs) >= 1 else ""
        if kind == "running":
            spent = time.monotonic() - self._cur_step_t0 if live and self._cur_step_t0 else 0.0
            return (tr("dub.st.running", "Running…")
                    + (f" {_fmt_clock(spent)}" if spent >= 1 else ""), dw.BLUE)
        if kind == "done":
            return tr("dub.st.done", "Done") + took, dw.GREEN
        if kind == "warn":
            return tr("dub.st.warn", "Done, with notes") + took, dw.AMBER
        if kind == "failed":
            return tr("dub.st.failed", "Failed"), dw.RED
        if kind == "stopped":
            return tr("dub.st.stopped", "Stopped"), dw.AMBER
        return tr("dub.st.pending", "Pending"), dw.GREY

    def _place_task_bar(self, row: int | None) -> None:
        """A thin progress bar beside the running task (its own progress)."""
        if row is not None and row == self._task_bar_row and self._task_bar is not None:
            return
        if self._task_bar_row >= 0:
            old = self.task_tree.topLevelItem(self._task_bar_row)
            if old is not None:
                self.task_tree.removeItemWidget(old, 1)
        self._task_bar, self._task_bar_row = None, -1
        if row is None:
            return
        holder = dw.clear_widget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(4, 0, 14, 0)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        bar.setStyleSheet("QProgressBar { background: #1a2740; border: none; border-radius: 4px; }"
                          f"QProgressBar::chunk {{ background: {dw.BLUE}; border-radius: 4px; }}")
        lay.addWidget(bar)
        self.task_tree.setItemWidget(self.task_tree.topLevelItem(row), 1, holder)
        self._task_bar, self._task_bar_row = bar, row
        bar.setValue(int(self._task_fraction(de.STEP_IDS[row]) * 1000))

    def _task_fraction(self, sid: str) -> float:
        """How far the running task itself is (from the whole dub's bar)."""
        p = self.project
        if p is None:
            return 0.0
        total = float(sum(de.STEP_WEIGHTS.values()))
        done_w = sum(de.STEP_WEIGHTS[s] for s in de.STEP_IDS if p.step_done(s) and s != sid)
        w = de.STEP_WEIGHTS.get(sid, 1)
        return max(0.0, min(1.0, (self._last_overall * total - done_w) / w)) if w else 0.0

    def _update_eta(self, sid: str) -> None:
        """Time left from how long each task took on earlier dubs (learned
        per minute of video), so it no longer swings with the fixed task
        weights; the panel goes by the run's own pace until every task left
        has been timed once."""
        p = self.project
        if p is None:
            return
        now = time.monotonic()
        if sid != self._cur_step:
            self._cur_step, self._cur_step_t0 = sid, now
        within = self._task_fraction(sid)
        fn = self._last_run[0] if self._last_run else None
        after = ([s for s in de.STEP_IDS[de.STEP_IDS.index(sid) + 1:] if not p.step_done(s)]
                 if fn == self.engine.run_all else [])
        try:
            eta = de.estimate_left(self.ctx.config, p, sid, within, now - self._cur_step_t0, after)
        except Exception:
            eta = None
        self.run_panel.set_eta(eta)

    def retranslate(self) -> None:
        """The interface language changed (no restart): texts built from
        the project are built again; app.ui.retranslate converted the rest."""
        self._refresh_tasks()
        if self._checks:
            keep = self._sel_check
            self._on_readiness(self._checks)
            if keep:
                self._select_check(keep)
        self._render_hw()
        self.live_pill.set_texts(tr("dub.mon.live", "Real-time monitoring"),
                                 tr("dub.mon.idle", "Idle"))
        self._update_run_button()
        self.run_panel.retranslate()
        if not self._busy:
            if self.run_panel.state == "error" and self._last_error:
                self._show_error()
            elif self.run_panel.state in ("done", "error") or (
                    self.run_panel.state == "stopped" and self._output_missing()):
                self._show_saved_state()       # (an error of a reopened project too)
            elif (self.run_panel.state == "partial" and self._last_run
                  and self._last_run[0] == self.engine.run_until_ai):
                self._show_transcribed()
        for i, (_sid, _label) in enumerate(de.STEPS):     # task list tooltips
            it = self.task_tree.topLevelItem(i)
            if it is not None and not (self.project and self.project.steps[_sid].get("note")):
                it.setToolTip(0, tr("dub.step." + _sid, _label))
        self._update_idle_hint()

    def _engine_combo(self, value: str) -> QComboBox:
        cb = QComboBox()
        for e in de.CHAR_ENGINES:
            cb.addItem(tr("dub.engine." + e.replace("+", "_"), e), e)
        i = cb.findData(value or "auto")
        cb.setCurrentIndex(max(0, i))
        return cb

    def _fill_chars(self) -> None:
        p = self.project
        self._filling = True
        try:
            self.char_table.setRowCount(0)
            if p is None:
                return
            p.recount()
            for r, ch in enumerate(p.characters):
                self.char_table.insertRow(r)
                self.char_table.setItem(r, C_ID, _ro(ch.id))
                name_item = QTableWidgetItem(ch.name)
                if ch.cast_match:
                    name_item.setToolTip(tr("dub.c.matched", "Recognised from the series as")
                                         + f" {ch.cast_match}")
                    name_item.setForeground(QColor("#58a6ff"))
                self.char_table.setItem(r, C_NAME, name_item)
                g = QComboBox()
                g.addItem(tr("dub.male", "Male"), "M")
                g.addItem(tr("dub.female", "Female"), "F")
                g.setCurrentIndex(0 if ch.gender == "M" else 1)
                g.currentIndexChanged.connect(
                    lambda _i, cid=ch.id, cb=g: self._set_char_gender(cid, cb.currentData()))
                unsure = (ch.gender_source == "voice" and ch.gender_conf < de.GENDER_SURE)
                src = {"voice": tr("dub.g.src.voice", "measured from the voice"),
                       "ai": tr("dub.g.src.ai", "set by the AI from the dialogue"),
                       "user": tr("dub.g.src.user", "set by you")}.get(ch.gender_source, "")
                g.setToolTip((f"{src} — " if src else "")
                             + tr("dub.g.sure", "sure") + f" {ch.gender_conf:.0%}"
                             + ("\n" + tr("dub.g.check", "Not sure: play the sample and check.")
                                if unsure else ""))
                if unsure:
                    g.setStyleSheet(f"QComboBox {{ background-color: {_UNSURE_COLOR}; }}")
                self.char_table.setCellWidget(r, C_GENDER, g)
                self.char_table.setItem(r, C_LINES, _ro(str(ch.line_count)))
                self.char_table.setItem(r, C_TALK, _ro(f"{ch.talk_seconds:.0f}s"))
                e = self._engine_combo(ch.engine)
                resolved = self.engine.voice_params(p, ch)
                e.setToolTip(tr("dub.resolved", "Will use") + f": {resolved['engine']}"
                             + (" + OpenVoice" if resolved.get("clone_ref") else ""))
                e.currentIndexChanged.connect(
                    lambda _i, cid=ch.id, cb=e: self._set_char(cid, "engine", cb.currentData()))
                self.char_table.setCellWidget(r, C_ENGINE, e)
                vi = QTableWidgetItem(ch.voice)
                vi.setToolTip(tr("dub.voice.tip",
                                 "Empty = automatic. Edge voice name (e.g. km-KH-PisethNeural), "
                                 "OpenAI voice or ElevenLabs voice id."))
                self.char_table.setItem(r, C_VOICE, vi)
                self.char_table.setItem(r, C_PITCH, QTableWidgetItem(ch.pitch))
                self.char_table.setCellWidget(r, C_SAMPLE, self._sample_cell(ch))
                self.char_table.setItem(r, C_REFTEXT, QTableWidgetItem(ch.ref_text))
        finally:
            self._filling = False

    def _sample_cell(self, ch: de.Character) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(4)
        has = bool(ch.ref_wav and os.path.isfile(ch.ref_wav))
        play = QToolButton()
        play.setText("▶")
        play.setToolTip(tr("dub.play_sample", "Play the voice sample (original voice)"))
        play.setEnabled(has)
        play.clicked.connect(lambda _=False, p=ch.ref_wav: self._play(p))
        h.addWidget(play)
        rep = QToolButton()
        rep.setText(tr("dub.replace", "Replace…"))
        rep.clicked.connect(lambda _=False, cid=ch.id: self._on_replace_sample(cid))
        h.addWidget(rep)
        test = QToolButton()
        test.setText(tr("dub.test", "Test voice"))
        test.setToolTip(tr("dub.test.tip", "Speak one of this character's lines with the dub voice"))
        test.clicked.connect(lambda _=False, cid=ch.id: self._on_test_voice(cid))
        h.addWidget(test)
        h.addStretch(1)
        return w

    def _speaker_combo(self, current: str) -> QComboBox:
        cb = QComboBox()
        p = self.project
        for ch in (p.characters if p else []):
            cb.addItem(f"{ch.label()} ({ch.id})", ch.id)
        i = cb.findData(current)
        if i < 0:
            cb.addItem(current, current)
            i = cb.count() - 1
        cb.setCurrentIndex(i)
        return cb

    def _fill_lines(self, *_a) -> None:
        p = self.project
        self._filling = True
        try:
            self.line_table.setRowCount(0)
            if p is None:
                return
            only = self.problems_check.isChecked()
            r = 0
            for i, ln in enumerate(p.lines):
                shortened = self._shortened(ln)
                if only and ln.status not in _FIT_COLOR and not ln.review and not shortened:
                    continue
                self.line_table.insertRow(r)
                num = _ro(("⚑ " if ln.review else "") + str(i + 1))
                num.setData(Qt.ItemDataRole.UserRole, i)
                if ln.review:
                    num.setToolTip(self._review_text(ln.review))
                    num.setBackground(QColor(_REVIEW_COLOR))
                self.line_table.setItem(r, L_NUM, num)
                self.line_table.setItem(r, L_START, _ro(_fmt_t(ln.start)))
                self.line_table.setItem(r, L_END, _ro(_fmt_t(ln.end)))
                cb = self._speaker_combo(ln.speaker)
                cb.currentIndexChanged.connect(
                    lambda _x, idx=i, c=cb: self._set_line_speaker(idx, c.currentData()))
                self.line_table.setCellWidget(r, L_SPK, cb)
                self.line_table.setCellWidget(r, L_VOICE, self._voice_combo(i))
                src = _ro(ln.source)
                tip = []
                if ln.heard:
                    # a misheard word was corrected before translating
                    tip.append(tr("dub.l.heard", "Recognised as: {heard}")
                               .replace("{heard}", ln.heard))
                if getattr(ln, "on_screen", False):
                    tip.append(tr("dub.l.on_screen",
                                  "Words taken from the video's own subtitle"))
                if ln.review:
                    tip.append(self._review_text(ln.review))
                if ln.skip:
                    # left out by the user: struck through, not dubbed
                    f = src.font()
                    f.setStrikeOut(True)
                    src.setFont(f)
                    tip.append(tr("dub.l.skipped", "Left out: nobody says this "
                                                   "(right-click → Put the line back)"))
                if tip:
                    src.setToolTip("\n".join(tip))
                self.line_table.setItem(r, L_SRC, src)
                txt = QTableWidgetItem(ln.text)
                if shortened:
                    # the dub says less than the translation: show what is missing
                    txt.setToolTip(tr("dub.l.shortened",
                                      "Shortened to fit the timing. Whole translation "
                                      "(the subtitles show it):") + f"\n{ln.full_text}")
                    txt.setBackground(QColor(_SHORT_COLOR))
                self.line_table.setItem(r, L_TXT, txt)
                self.line_table.setItem(r, L_FIT, self._fit_item(i))
                play = QToolButton()
                play.setText("▶")
                play.setToolTip(tr("dub.l.preview", "Hear this line in the character's dub voice"))
                play.clicked.connect(lambda _=False, idx=i: self._on_preview_line(idx))
                self.line_table.setCellWidget(r, L_PLAY, play)
                r += 1
            self.line_table.resizeRowsToContents()
        finally:
            self._filling = False

    @staticmethod
    def _shortened(ln) -> bool:
        """The dub line says less than the line's whole translation."""
        return bool(ln.full_text and ln.text and ln.text_src != "user"
                    and ln.text.strip() != ln.full_text.strip())

    @staticmethod
    def _review_text(reason: str) -> str:
        """Why a line is marked ⚑ (DubLine.review), for its tooltip."""
        return {
            "recovered": tr("dub.l.review.recovered",
                            "⚑ Missed by the first listen and found by a second one: check "
                            "that it was really said (and by whom)."),
            "low_confidence": tr("dub.l.review.low",
                                 "⚑ The speech recogniser was unsure of these words: check "
                                 "them against the video."),
            "maybe_prompt": tr("dub.l.review.prompt",
                               "⚑ Only a character's name: check that someone really says it "
                               "here (the recogniser sometimes writes out the names it was given)."),
            "foreign_script": tr("dub.l.review.script",
                                 "⚑ The translation still has words the dub voice cannot read: "
                                 "write them in the target script."),
            "glossary": tr("dub.l.review.glossary",
                           "⚑ A name or term in this line is not written as in the Glossary: "
                           "check the translation."),
            "number": tr("dub.l.review.number",
                         "⚑ A number of the original is missing or different in this line: "
                         "check it against the video and write it with digits (the voice "
                         "says digits correctly)."),
        }.get(reason, "⚑ " + reason)

    def _fit_item(self, i: int) -> QTableWidgetItem:
        p = self.project
        ln = p.lines[i]
        slot = p.slot(i)
        if ln.status == "late":
            text = tr("dub.fit.late", "late") + f" +{ln.delay:.1f}s"
        elif ln.status == "overrun":
            text = tr("dub.fit.overrun", "overlaps")
        elif ln.status == "overlong":
            text = f"⚠ {ln.needed:.1f}/{slot:.1f}s"
        elif ln.status == "failed":
            text = tr("dub.fit.failed", "TTS failed")
        elif ln.status == "ok":
            text = f"✔ {ln.needed:.1f}/{slot:.1f}s" if ln.needed else "✔"
        else:
            text = ""
        fast = ln.status in ("ok", "late") and ln.speed > 1.02
        if fast:
            text += f" · {ln.speed:.1f}×"       # how fast the voice says it to fit
        it = _ro(text)
        it.setToolTip(tr("dub.fit.tip", "Estimated speech seconds / available seconds")
                      + f" (max {p.max_chars(i)} " + tr("dub.chars", "chars") + ")"
                      + ("\n" + tr("dub.fit.speed", "Spoken {x}× faster to fit its time")
                         .replace("{x}", f"{ln.speed:.2f}") if fast else ""))
        if ln.status in _FIT_COLOR:
            it.setBackground(QColor(_FIT_COLOR[ln.status]))
        return it

    def _fill_glossary(self) -> None:
        p = self.project
        self._filling = True
        try:
            self.context_edit.setPlainText(p.story_context if p else "")
            self.gloss_table.setRowCount(0)
            for g in (p.glossary if p else []):
                r = self.gloss_table.rowCount()
                self.gloss_table.insertRow(r)
                self.gloss_table.setItem(r, 0, QTableWidgetItem(g.source))
                self.gloss_table.setItem(r, 1, QTableWidgetItem(g.target))
                self.gloss_table.setItem(r, 2, QTableWidgetItem(g.note))
        finally:
            self._filling = False

    # ======================================================================
    # edits -> project
    # ======================================================================
    def _set_char(self, cid: str, attr: str, value) -> None:
        p = self.project
        ch = p.character(cid) if p else None
        if ch is None or self._filling:
            return
        if getattr(ch, attr) == value:
            return
        setattr(ch, attr, value)
        p.invalidate_from("synthesize")
        if attr == "engine":
            # rebuild after the emitting combo box has finished its signal
            QTimer.singleShot(0, self._fill_chars)
        self._refresh_tasks()

    def _set_char_gender(self, cid: str, gender: str) -> None:
        p = self.project
        ch = p.character(cid) if p else None
        if ch is None or self._filling or ch.gender == gender:
            return
        ch.gender, ch.gender_source, ch.gender_conf = gender, "user", 1.0
        # lines switched to the other gender were relative to the old guess;
        # voicing decides them again against the new gender
        ch.gender_recheck = True
        for ln in p.lines:
            if ln.speaker == cid and ln.gender_src == "voice":
                ln.gender, ln.gender_src, ln.tts_key = "", "", ""
        p.invalidate_from("synthesize")    # only lines whose voice changes are redone
        QTimer.singleShot(0, self._fill_all)

    def _set_line_gender(self, rows: list[int], gender: str) -> None:
        """Voice these lines as male / female ("" = like their character)."""
        p = self.project
        if p is None or self._filling:
            return
        changed = False
        for i in rows:
            ln = p.lines[i]
            new_src = "user" if gender else ""
            if (ln.gender, ln.gender_src) != (gender, new_src):
                ln.gender, ln.gender_src, ln.tts_key = gender, new_src, ""
                changed = True
        if changed:
            p.invalidate_from("synthesize")   # only these lines are re-voiced
            QTimer.singleShot(0, self._fill_lines)
            self._refresh_tasks()

    def _on_char_cell(self, r: int, c: int) -> None:
        if self._filling or self.project is None:
            return
        cid_item = self.char_table.item(r, C_ID)
        item = self.char_table.item(r, c)
        if cid_item is None or item is None:
            return
        attr = {C_NAME: "name", C_VOICE: "voice", C_PITCH: "pitch",
                C_REFTEXT: "ref_text"}.get(c)
        if attr:
            text = item.text().strip()
            if attr == "pitch" and text and not text.endswith("Hz"):
                text = f"{int(float(text)):+d}Hz" if _is_number(text) else "+0Hz"
            self._set_char(cid_item.text(), attr, text)
            if attr == "name":
                self._fill_lines()

    def _on_merge_chars(self) -> None:
        p = self.project
        rows = sorted({i.row() for i in self.char_table.selectedIndexes()})
        if p is None or len(rows) < 2:
            QMessageBox.information(self, tr("dub.title", "Full Dub"),
                                    tr("dub.merge.need", "Select two or more character rows first."))
            return
        ids = [self.char_table.item(r, C_ID).text() for r in rows]
        keep, drop = ids[0], set(ids[1:])
        for ln in p.lines:
            if ln.speaker in drop:
                ln.speaker = keep
        p.characters = [c for c in p.characters if c.id not in drop]
        p.recount()
        p.invalidate_from("voices")
        self._fill_all()

    def _set_line_speaker(self, i: int, cid: str) -> None:
        p = self.project
        if p is None or self._filling or not cid:
            return
        ln = p.lines[i]
        if ln.speaker != cid:
            ln.speaker, ln.tts_key = cid, ""
            p.recount()
            p.invalidate_from("synthesize")
            self._refresh_tasks()

    def _on_line_cell(self, r: int, c: int) -> None:
        p = self.project
        if self._filling or p is None or c != L_TXT:
            return
        num = self.line_table.item(r, L_NUM)
        item = self.line_table.item(r, L_TXT)
        if num is None or item is None:
            return
        i = int(num.data(Qt.ItemDataRole.UserRole))
        text = item.text().strip()
        ln = p.lines[i]
        if text == ln.text:
            return
        ln.text, ln.tts_key = text, ""
        # kept when the translation is redone; an emptied line goes back to the AI
        ln.text_src = "user" if text else ""
        if text:
            ln.skip = False          # typing a line in puts it back
            if ln.review == "foreign_script":
                ln.review = ""       # the user wrote it: theirs to judge
        if text and p.options.use_tm:
            try:
                from app.core.translation_memory import get_tm
                ch = p.character(ln.speaker)
                get_tm().add(de._base_lang(p.options.target_lang), ln.source, text,
                             gender=ch.gender if ch else "", series=p.options.series,
                             origin="human", src_lang=p.src_lang)
            except Exception:
                log.debug("translation memory update failed", exc_info=True)
        # the Fit step's own measure (the measured Khmer voice model), so
        # an edited line is judged as Fit judges it
        ln.needed = round(de.speech_seconds(text, p.options.target_lang), 2)
        ln.status = "ok" if ln.needed <= p.slot(i) * max(1.0, p.options.max_tempo) else "overlong"
        if p.step_done("synthesize") or p.step_done("fit"):
            p.invalidate_from("synthesize")
        self._filling = True
        try:
            self.line_table.setItem(r, L_FIT, self._fit_item(i))
        finally:
            self._filling = False
        self._refresh_tasks()

    def _on_line_dclick(self, r: int, c: int) -> None:
        num = self.line_table.item(r, L_NUM)
        if num is None or self.project is None or c in (L_TXT, L_SPK):
            return
        ln = self.project.lines[int(num.data(Qt.ItemDataRole.UserRole))]
        main = self.ctx.main
        preview = getattr(main, "preview", None) if main else None
        if preview is not None:
            try:
                if preview.current_path() != self.project.video:
                    self.ctx.open_in_preview(self.project.video)
                preview.seek(ln.start, play=True)
            except Exception:
                log.exception("preview seek failed")

    def _on_context_changed(self) -> None:
        if not self._filling and self.project is not None:
            self.project.story_context = self.context_edit.toPlainText().strip()

    def _pull_glossary(self) -> None:
        p = self.project
        if p is None:
            return
        # a term left as it was keeps where it came from; one typed or
        # changed here is the user's (the AI never replaces it)
        old = {(g.source, g.target, g.note): g.origin for g in p.glossary}
        out = []
        for r in range(self.gloss_table.rowCount()):
            vals = [(self.gloss_table.item(r, c).text().strip()
                     if self.gloss_table.item(r, c) else "") for c in range(3)]
            if vals[0]:
                origin = old.get(tuple(vals))
                out.append(de.GlossaryTerm(*vals, origin="user" if origin is None else origin))
        p.glossary = out

    def _on_gloss_cell(self, _r: int, _c: int) -> None:
        if not self._filling:
            self._pull_glossary()

    def _on_gloss_add(self) -> None:
        r = self.gloss_table.rowCount()
        self.gloss_table.insertRow(r)
        self.gloss_table.setCurrentCell(r, 0)
        self.gloss_table.editItem(self.gloss_table.item(r, 0) or self._new_gloss_item(r))

    def _new_gloss_item(self, r: int) -> QTableWidgetItem:
        it = QTableWidgetItem("")
        self.gloss_table.setItem(r, 0, it)
        return it

    def _on_gloss_del(self) -> None:
        for r in sorted({i.row() for i in self.gloss_table.selectedIndexes()}, reverse=True):
            self.gloss_table.removeRow(r)
        self._pull_glossary()

    # ======================================================================
    # samples, previews, output
    # ======================================================================
    def _on_replace_sample(self, cid: str) -> None:
        p = self.project
        ch = p.character(cid) if p else None
        if ch is None:
            return
        src, _ = QFileDialog.getOpenFileName(
            self, tr("dub.pick_sample", "Choose a clean 5–12 s clip of this voice"), "", AUDIO_FILTER)
        if not src:
            return
        out = p.sub("voices", f"{cid}_custom.wav")

        def done(path) -> None:
            ch.ref_wav, ch.ref_custom = path, True
            ch.ref_text = ""
            p.invalidate_from("synthesize")
            p.save()
            self._fill_chars()
            self._refresh_tasks()
            QMessageBox.information(
                self, tr("dub.title", "Full Dub"),
                tr("dub.sample_set", "Sample replaced. Type what is said in it into "
                   "“Sample transcript” for the best cloning (VoxCPM / CosyVoice need it)."))

        w = FnWorker(_import_sample_fn, src, out)
        w.finished_ok.connect(done)
        self._start(w)

    def _on_test_voice(self, cid: str) -> None:
        p = self.project
        ch = p.character(cid) if p else None
        if ch is None:
            return
        line = next((l for l in p.lines if l.speaker == cid and l.text.strip()), None)
        text = line.text if line else {
            "km": "សួស្តី នេះគឺជាសំឡេងរបស់ខ្ញុំ សម្រាប់ការបញ្ចូលសំឡេងភាពយន្ត។",
        }.get(p.options.target_lang, "Hello, this is my voice for the dub.")
        self._speak(ch, text, p.sub("preview", f"test_{cid}.wav"))

    def _on_preview_line(self, i: int) -> None:
        p = self.project
        if p is None:
            return
        ln = p.lines[i]
        if not ln.text.strip():
            QMessageBox.information(self, tr("dub.title", "Full Dub"),
                                    tr("dub.no_translation", "This line has no translation yet."))
            return
        ch = p.character(ln.speaker) or de.Character(id=ln.speaker)
        self._speak(ch, ln.text, p.sub("preview", f"line_{i:05d}.wav"), ln)

    def _speak(self, ch: de.Character, text: str, out: str, ln=None) -> None:
        self._sync_options()
        w = FnWorker(self.engine.synth_text, self.project, ch, text, out, ln=ln)
        w.finished_ok.connect(lambda path: self._play(path))
        self._start(w)

    def _play(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            return
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            for old in self._players:
                old.stop()
            player = QMediaPlayer(self)
            out = QAudioOutput(self)
            player.setAudioOutput(out)
            player._rs_audio_out = out  # keep a reference
            player.setSource(QUrl.fromLocalFile(path))
            self._players = [player]
            player.play()
        except Exception:
            log.exception("could not play %s", path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_pick_output(self) -> None:
        self._ask_output(force=True)

    def _output_path(self) -> str:
        p = self.project
        if p is None:
            return ""
        return p.output or self.out_edit.text().strip() or self.engine.default_output(p)

    def _on_play_output(self) -> None:
        path = self._output_path()
        if path and os.path.isfile(path):
            self.ctx.open_in_preview(path)
        else:
            QMessageBox.information(self, tr("dub.title", "Full Dub"),
                                    tr("dub.no_output", "No dubbed video yet — run the Export task."))

    def _on_show_output(self) -> None:
        path = self._output_path()
        folder = os.path.dirname(path) if path else ""
        if folder and os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_export_srt(self) -> None:
        p = self.project
        if p is None or not any(l.text.strip() for l in p.lines):
            QMessageBox.information(self, tr("dub.title", "Full Dub"),
                                    tr("dub.no_translation_all", "Translate the lines first."))
            return
        start = os.path.splitext(self._output_path())[0] + f".{p.options.target_lang}.srt"
        path, _ = QFileDialog.getSaveFileName(self, tr("dub.save_srt", "Save subtitles"),
                                              start, "SubRip (*.srt)")
        if path:
            self.engine.write_srt(p, path)
            self.status_label.setText(tr("dub.srt_saved", "Subtitles saved") + f": {path}")


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
