"""RunPanel: the state of a long job kept on screen (the Dub tab's dub, the
Library's batch of episodes): running (progress, what it is doing, time
spent / left, Cancel), done (where the result is, Play / Show in folder),
error (where it stopped and why, Continue / Cancel — no pop-up) or stopped.
It goes away only on reset() (the Dub tab: another video is opened)."""
from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import tr
from app.core.logger import get_logger

log = get_logger("ui.run_panel")


def _fmt_clock(sec: float) -> str:
    """12:31 or 1:02:11 (elapsed / remaining time on the run panel)."""
    sec = int(max(0.0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _keep_awake(on: bool) -> None:
    """While a dub runs (hours for a long video) Windows must not go to
    sleep: that stops the run. The screen may still turn off."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        log.debug("SetThreadExecutionState failed", exc_info=True)


_PANEL_COLORS = {          # border / bar colour per state (or look)
    "running": "#58a6ff", "done": "#3fb950", "partial": "#3fb950",
    "error": "#f85149", "stopped": "#d29922", "stopping": "#d29922", "paused": "#d29922",
}
_STATE_ICONS = {           # the round icon at the panel's left, per state (or look)
    "running": "pulse", "done": "check", "partial": "check", "error": "cross",
    "stopped": "stop", "stopping": "stop", "paused": "pause",
}

# why no AI answered (ai_engine.ai_failure_kind) when the fix is in Settings:
# the panel then gets an "Open Settings" button
FIX_IN_SETTINGS = ("no_key", "bad_key", "no_credit", "budget")


def no_ai_key(config) -> bool:
    """No provider has an API key (asked now, not from an older check)."""
    try:
        return not config.providers_with_keys()
    except Exception:
        return False


def ask_no_key(parent: QWidget, can_transcribe: bool) -> str:
    """No AI key yet: "settings" (open Settings), "transcribe" (only the
    tasks that need no AI) or "" (cancel). The Dub tab and the Library's
    batch ask the same question."""
    from PySide6.QtWidgets import QMessageBox
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(tr("dub.title", "Full Dub"))
    box.setText(tr("dub.nokey.q", "There is no AI key yet, so the dub cannot translate."))
    box.setInformativeText(
        tr("dub.nokey.info", "Add a key in Settings → AI Providers (for example an "
                             "OpenRouter key). Or transcribe now: the speech is turned "
                             "into text and the speakers are found, without AI; "
                             "translate after adding a key.")
        if can_transcribe else
        tr("dub.nokey.info2", "Add a key in Settings → AI Providers (for example an "
                              "OpenRouter key), then press Continue."))
    b_set = box.addButton(tr("dub.run.open_settings", "Open Settings"),
                          QMessageBox.ButtonRole.AcceptRole)
    b_tr = (box.addButton(tr("dub.nokey.transcribe", "Transcribe only"),
                          QMessageBox.ButtonRole.ActionRole) if can_transcribe else None)
    box.addButton(tr("common.cancel", "Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(b_set)
    box.exec()
    hit = box.clickedButton()
    return "settings" if hit is b_set else "transcribe" if b_tr is not None and hit is b_tr else ""


def open_settings(ctx) -> None:
    """Switch the main window to the Settings tab."""
    main = getattr(ctx, "main", None)
    try:
        if main is not None:
            main.select_tab_by_key("settings")
    except Exception:
        log.debug("could not switch to Settings", exc_info=True)


class RunPanel(QFrame):
    """The state of the current job, kept on screen until reset()."""

    continue_clicked = Signal()
    cancel_clicked = Signal()        # stop the running job
    dismiss_clicked = Signal()       # "Cancel" on an error: stop here
    play_clicked = Signal()
    show_clicked = Signal()
    change_clicked = Signal()        # choose another save location
    settings_clicked = Signal()      # open Settings (no AI key / key refused)
    state_changed = Signal(str)      # "" when reset (the panel hides)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("dubRunPanel")
        self.state = ""
        self._t0 = 0.0
        self._t_end = 0.0
        self._frac = 0.0
        self._f0 = 0.0
        self._before = 0.0
        self._eta: tuple[float, float] | None = None   # (seconds left, when said)

        self.icon = QLabel()
        self.icon.setFixedSize(46, 46)
        self.icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.title = QLabel("")
        self.title.setObjectName("dubRunTitle")
        self.title.setWordWrap(True)
        self.time_lbl = QLabel("")
        self.time_lbl.setStyleSheet("color: #8b949e;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(14)
        self.pct = QLabel("0%")
        self.pct.setObjectName("dubRunPct")
        self.pct.setMinimumWidth(44)
        self.pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bar.valueChanged.connect(lambda v: self.pct.setText(f"{v // 10}%"))
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_lbl = QLabel("")
        self.path_lbl.setStyleSheet("color: #8b949e;")
        self.path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # one line, the middle of a long path left out (the tooltip has it all):
        # a path has no spaces to wrap at and made the panel too wide
        self.path_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._path_full = ""

        self.change_btn = QPushButton(tr("dub.run.change", "Change…"))
        self.change_btn.clicked.connect(self.change_clicked)
        self.play_btn = QPushButton(tr("dub.play_out", "Play dubbed video"))
        self.play_btn.clicked.connect(self.play_clicked)
        self.show_btn = QPushButton(tr("dub.show_out", "Show in folder"))
        self.show_btn.clicked.connect(self.show_clicked)
        self.settings_btn = QPushButton(tr("dub.run.open_settings", "Open Settings"))
        self.settings_btn.clicked.connect(self.settings_clicked)
        self.cont_btn = QPushButton(tr("dub.run.continue", "Continue"))
        self.cont_btn.setProperty("class", "accent")
        self.cont_btn.clicked.connect(self.continue_clicked)
        self.cancel_btn = QPushButton(tr("common.cancel", "Cancel"))
        self.cancel_btn.clicked.connect(self._on_cancel)

        top = QHBoxLayout()                 # the title and the buttons on one row
        top.setSpacing(8)
        top.addWidget(self.title, 1)
        for b in self._all_buttons():
            top.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        mid = QHBoxLayout()                 # the bar, its % and the time
        mid.setSpacing(10)
        mid.addWidget(self.bar, 1)
        mid.addWidget(self.pct)
        mid.addWidget(self.time_lbl)
        col = QVBoxLayout()
        col.setSpacing(5)
        col.addLayout(top)
        col.addLayout(mid)
        col.addWidget(self.detail)
        col.addWidget(self.path_lbl)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 14, 10)
        lay.setSpacing(12)
        lay.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(col, 1)
        # never squeezed below its own height (the texts overlapped when the
        # Dub tab's monitor was short of room)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        self.hide()

    # -- look -------------------------------------------------------------
    def _paint(self, state: str) -> None:
        from app.ui.dub_widgets import round_pixmap
        c = _PANEL_COLORS.get(state, "#30363d")
        self.icon.setPixmap(round_pixmap(_STATE_ICONS.get(state, "pulse"), c, 46, solid=True))
        self.setStyleSheet(
            f"QFrame#dubRunPanel {{ border: 1px solid {c}; border-radius: 10px;"
            f" background-color: #131d30; }}"
            f"QLabel#dubRunTitle {{ font-size: 15px; font-weight: 600; color: {c}; }}"
            f"QLabel#dubRunPct {{ font-size: 14px; font-weight: 600; color: {c}; }}"
            f"QProgressBar {{ background-color: #0e1522; border: 1px solid #22314d;"
            f" border-radius: 7px; }}"
            f"QProgressBar::chunk {{ background-color: {c}; border-radius: 6px; margin: 1px; }}")

    def _all_buttons(self) -> tuple[QPushButton, ...]:
        return (self.change_btn, self.play_btn, self.show_btn, self.settings_btn,
                self.cont_btn, self.cancel_btn)

    def _buttons(self, *shown: QPushButton) -> None:
        for b in self._all_buttons():
            b.setVisible(b in shown)
            b.setEnabled(True)
        self.cont_btn.setText(tr("dub.run.continue", "Continue"))   # set_stopped may rename it

    def _set(self, state: str, title: str, detail: str = "", look: str = "") -> None:
        self.state = state
        self._paint(look or state)
        self.title.setText(title)
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.show()
        self.state_changed.emit(state)

    # -- states -----------------------------------------------------------
    def start(self, save_to: str, frac0: float = 0.0, spent_before: float = 0.0,
              title: str = "") -> None:
        """Running. `frac0`: how far the whole job already is (the bar goes
        on from there after Continue instead of restarting at 0 %);
        `spent_before`: seconds the earlier runs took."""
        self._t0, self._t_end = time.monotonic(), 0.0
        self._before = max(0.0, spent_before)
        self._frac = self._f0 = max(0.0, min(1.0, frac0))
        self._eta = None
        self.bar.setValue(int(self._frac * 1000))
        self._set("running", title or tr("dub.run.starting", "Starting…"))
        self.set_path(tr("dub.run.save_to", "Save to") + f": {save_to}" if save_to else "")
        self._buttons(self.cancel_btn)
        self._clock.start()
        self._tick()

    def set_running_title(self, text: str) -> None:
        if self.state == "running":
            self.title.setText(text)

    def set_task(self, i: int, n: int, task: str) -> None:
        self.set_running_title(tr("dub.run.running", "Dubbing — task {i} of {n}: {task}")
                               .replace("{i}", str(i)).replace("{n}", str(n))
                               .replace("{task}", task))

    def set_progress(self, frac: float, text: str) -> None:
        if self.state not in ("running", "stopping"):
            return
        # forward only: a task's own progress can dip (retry rounds) and the
        # bar slid back from 85 % to 84 % on a real dub
        self._frac = max(self._frac, min(1.0, float(frac)))
        self.bar.setValue(int(self._frac * 1000))
        if text and self.state == "running":
            self.detail.setText(text)
            self.detail.setVisible(True)

    def set_eta(self, seconds: float | None) -> None:
        """Time left as worked out by the caller (e.g. from past runs);
        None = estimate from this run's own pace."""
        self._eta = (max(0.0, seconds), time.monotonic()) if seconds is not None else None

    def set_stopping(self) -> None:
        self.state = "stopping"
        self._paint("stopping")
        self.title.setText(tr("dub.run.stopping", "Stopping… (finishing the current step safely)"))
        self.cancel_btn.setEnabled(False)
        self.state_changed.emit("stopping")

    def set_done(self, saved: str, info: str, notes: str = "", title: str = "",
                 play: bool = True) -> None:
        self._stop_clock()
        self.bar.setValue(1000)
        self._set("done", title or tr("dub.run.done", "Done — the dubbed video is ready"),
                  notes)
        self.set_path((tr("dub.saved_to", "Saved to") + f": {saved}" if saved else "")
                      + (f"  ({info})" if info else ""))
        self._buttons(*((self.play_btn, self.show_btn) if play else (self.show_btn,)))

    def set_fraction(self, frac: float) -> None:
        """Show how far the whole job is (a saved project, a finished task)."""
        self._frac = max(0.0, min(1.0, frac))
        self.bar.setValue(int(self._frac * 1000))

    def set_partial(self, finished: str, nxt: str, title: str = "", detail: str = "",
                    settings: bool = False, change: bool = True) -> None:
        """Some tasks done, the rest to come. `title` / `detail` replace the
        "Task finished: …" / "Next: …" texts; `settings`: an Open Settings
        button too (e.g. transcribed without an AI key)."""
        self._stop_clock()
        self._set("partial", title or tr("dub.run.task_done", "Task finished: {task}")
                  .replace("{task}", finished),
                  detail or tr("dub.run.next", "Next: {task}. Press Continue to dub the rest.")
                  .replace("{task}", nxt))
        self._buttons(*(((self.change_btn,) if change else ())
                        + ((self.settings_btn,) if settings else ()) + (self.cont_btn,)))

    def set_error(self, where: str, message: str, tooltip: str = "", title: str = "",
                  change: bool = True, hint: str = "", settings: bool = False,
                  paused: bool = False) -> None:
        """`settings`: an "Open Settings" button too (the error is fixed
        there: no AI key, a refused key, no credit, the budget). `paused`:
        nothing broke, the AI could not answer: amber and "Paused", not a
        red "Error"."""
        self._stop_clock()
        said_saved = any(w in message.lower() for w in ("is saved", "រក្សាទុក"))
        hint = hint or (
            tr("dub.run.error_hint2", "Continue tries again from this task; Cancel stops here.")
            if said_saved else tr(
                "dub.run.error_hint", "Everything finished so far is saved. Continue tries "
                "again from this task; Cancel stops here."))
        head = title or (tr("dub.run.paused", "Paused") if paused else tr("dub.run.error", "Error"))
        self._set("error", head + (f" — {where}" if where else ""),
                  (message + "\n" if message else "") + hint, look="paused" if paused else "")
        self.detail.setToolTip(tooltip)
        self._buttons(*(((self.change_btn,) if change else ())
                        + ((self.settings_btn,) if settings else ())
                        + (self.cont_btn, self.cancel_btn)))

    def set_stopped(self, where: str, note: str = "", title: str = "",
                    change: bool = True, hint: str | None = None, action: str = "",
                    cont: bool = True) -> None:
        """`hint`: replaces "Everything finished so far is saved…" ("" = no
        hint); `action`: another text for the Continue button; `cont`: show
        that button at all."""
        self._stop_clock()
        if hint is None:
            hint = tr("dub.run.stopped_hint", "Everything finished so far is saved. "
                      "Press Continue any time to go on from here.")
        self._set("stopped", (title or tr("dub.run.stopped", "Stopped"))
                  + (f" — {where}" if where else ""),
                  "\n".join(t for t in (note, hint) if t))
        self.detail.setToolTip("")
        self._buttons(*(((self.change_btn,) if change else ())
                        + ((self.cont_btn,) if cont else ())))
        if action:
            self.cont_btn.setText(action)

    def set_path(self, text: str) -> None:
        self._path_full = text
        self._elide_path()
        self.path_lbl.setToolTip(text)
        self.path_lbl.setVisible(bool(text))

    def path_text(self) -> str:
        """The whole "Save to: …" line (the label may show it shortened)."""
        return self._path_full

    def _elide_path(self) -> None:
        width = max(80, self.path_lbl.width())
        self.path_lbl.setText(self.path_lbl.fontMetrics().elidedText(
            self._path_full, Qt.TextElideMode.ElideMiddle, width))

    def resizeEvent(self, ev):  # noqa: N802 (Qt override)
        super().resizeEvent(ev)
        self._elide_path()

    def reset(self) -> None:
        self._stop_clock()
        self.state = ""
        self._eta = None
        # another video / job: its time starts from nothing ("Took 0:33" of
        # the previous video stayed on the next one's panel)
        self._t0 = self._t_end = self._before = 0.0
        self._frac = self._f0 = 0.0
        self.time_lbl.setText("")
        self.bar.setValue(0)
        self.title.setText("")
        self.detail.setText("")
        self.detail.setToolTip("")
        self.set_path("")
        self.hide()
        self.state_changed.emit("")

    def retranslate(self) -> None:
        """After a language switch (the texts themselves are converted by
        app.ui.retranslate): the time line is rebuilt."""
        self._tick()

    # -- time ---------------------------------------------------------------
    def _stop_clock(self) -> None:
        if self._clock.isActive():
            self._clock.stop()
            self._t_end = time.monotonic()
        self._tick()

    def segment_seconds(self) -> float:
        """Seconds of the current (or last) run."""
        if not self._t0:
            return 0.0
        end = time.monotonic() if self._clock.isActive() else (self._t_end or time.monotonic())
        return max(0.0, end - self._t0)

    def _tick(self) -> None:
        if not self._t0:
            self.time_lbl.setText("")
            return
        running = self._clock.isActive()
        spent = self.segment_seconds()
        # the time shown covers every run (Continue adds on); the time left
        # comes from the caller's estimate, else from this run's own pace
        text = (tr("dub.run.elapsed", "Elapsed") if running
                else tr("dub.run.took", "Took")) + f" {_fmt_clock(self._before + spent)}"
        f = self._frac
        left = None
        if running and self._eta is not None:
            left = max(0.0, self._eta[0] - (time.monotonic() - self._eta[1]))
        elif running and f - self._f0 >= 0.02 and spent >= 20:
            left = spent * (1.0 - f) / (f - self._f0)
        if left is not None:
            text += " · " + tr("dub.run.left", "about {t} left").replace(
                "{t}", _fmt_clock(left) if left >= 60 else tr("dub.run.under_min", "<1 min"))
        self.time_lbl.setText(text)

    def _on_cancel(self) -> None:
        if self.state == "running":
            self.cancel_clicked.emit()
        else:
            self.dismiss_clicked.emit()
