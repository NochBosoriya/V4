"""Dialog + runner for the one-click series pipeline (app.engines.auto_pipeline)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import pick, tr, tr_text
from app.core.logger import get_logger
from app.engines import auto_pipeline as autop
from app.engines.dub_engine import STEPS
from app.ui.run_panel import (
    FIX_IN_SETTINGS,
    RunPanel,
    _keep_awake,
    ask_no_key,
    no_ai_key,
    open_settings,
)
from app.ui.langs import LANGS
from app.ui.workers import FnWorker

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.auto_dialog")

OPTIONS_KEY = "auto.last"


class AutoDialog(QDialog):
    """Pick options, then run; shows progress and a per-episode log."""

    def __init__(self, ctx: "AppContext", episodes: list[dict], series: str,
                 parent=None):
        super().__init__(parent)
        self.ctx, self.episodes = ctx, list(episodes)
        self.setWindowTitle(tr("auto.title", "Process episodes"))
        self.resize(640, 0)
        saved = autop.AutoOptions(**{
            k: v for k, v in dict(ctx.config.get(OPTIONS_KEY) or {}).items()
            if k in autop.AutoOptions.__dataclass_fields__})
        saved.series = series or saved.series
        self._worker: FnWorker | None = None

        lay = QVBoxLayout(self)
        head = QLabel(tr("auto.head", "{n} episode(s) of “{s}” — each becomes a finished video.")
                      .replace("{n}", str(len(self.episodes))).replace("{s}", series or "—"))
        head.setWordWrap(True)
        lay.addWidget(head)
        f = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem(tr("auto.mode.dub", "Full dub — every line in the characters' voices"), "dub")
        self.mode.addItem(tr("auto.mode.recap", "Recap — narrated short version"), "recap")
        self.mode.setCurrentIndex(max(0, self.mode.findData(saved.mode)))
        f.addRow(tr("auto.mode", "Make:"), self.mode)
        self.lang = QComboBox()
        for code, native in LANGS:
            self.lang.addItem(native, code)
        self.lang.setCurrentIndex(max(0, self.lang.findData(saved.target_lang)))
        f.addRow(tr("auto.lang", "Language:"), self.lang)
        self.minutes = QDoubleSpinBox()
        self.minutes.setRange(0.5, 30.0)
        self.minutes.setSuffix(" min")
        self.minutes.setValue(saved.recap_minutes)
        f.addRow(tr("auto.minutes", "Recap length:"), self.minutes)
        self.whisper = QComboBox()
        self.whisper.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.whisper.setCurrentText(saved.whisper_size)
        f.addRow(tr("auto.whisper", "Whisper model:"), self.whisper)
        self.encoder = QComboBox()
        from app.engines.media_exporter import known_encoders
        for e in known_encoders(ctx.config) or ["libx264", "libx265"]:
            self.encoder.addItem(e)
        self.encoder.setCurrentText(saved.vcodec)
        f.addRow(tr("auto.encoder", "Encoder:"), self.encoder)
        self.overlays = QCheckBox(tr("auto.overlays", "Apply the Overlay tab's saved watermark / banner"))
        self.overlays.setChecked(saved.use_overlays)
        f.addRow("", self.overlays)
        self.vertical = QCheckBox(tr("auto.vertical", "Also make a 9:16 Shorts version (smart crop + captions)"))
        self.vertical.setChecked(saved.vertical)
        f.addRow("", self.vertical)
        self.thumb = QCheckBox(tr("auto.thumb", "Make a thumbnail with an AI title"))
        self.thumb.setChecked(saved.thumbnail)
        f.addRow("", self.thumb)
        self.overwrite = QCheckBox(tr("auto.overwrite", "Redo episodes that are already finished"))
        self.overwrite.setChecked(False)
        f.addRow("", self.overwrite)
        orow = QHBoxLayout()
        self.out = QLineEdit(saved.out_dir)
        self.out.setPlaceholderText(tr("auto.out.ph", "Default: “RecapStudio out” next to each video"))
        orow.addWidget(self.out, 1)
        b = QPushButton(tr("common.browse", "Browse…"))
        b.clicked.connect(self._pick_out)
        orow.addWidget(b)
        ow = QWidget()
        ow.setLayout(orow)
        f.addRow(tr("auto.out", "Save to:"), ow)
        lay.addLayout(f)
        self.series = series
        # the same panel as the Dub tab: stays on screen with the result,
        # an error offers Continue (finished episodes are skipped) / Cancel
        self.panel = RunPanel()
        self.panel.cancel_clicked.connect(self._cancel_run)
        self.panel.continue_clicked.connect(self._start)
        self.panel.dismiss_clicked.connect(
            lambda: self.panel.set_stopped("", tr("auto.run.cancelled", "Stopped after the error."),
                                           change=False))
        self.panel.show_clicked.connect(self._show_folder)
        self.panel.settings_clicked.connect(self._open_settings)
        lay.addWidget(self.panel)
        self._spent = 0.0
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.log.hide()
        lay.addWidget(self.log)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.start_btn = self.buttons.addButton(tr("auto.start", "Start"),
                                                QDialogButtonBox.ButtonRole.AcceptRole)
        self.start_btn.setProperty("class", "accent")
        self.start_btn.clicked.connect(self._start)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _inputs(self) -> list:
        out = []
        for t in (QComboBox, QCheckBox, QLineEdit, QDoubleSpinBox):
            out += self.findChildren(t)
        return out

    def _mode_changed(self, *_a) -> None:
        self.minutes.setEnabled(self.mode.currentData() == "recap")

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, tr("auto.out", "Save to:"), self.out.text())
        if d:
            self.out.setText(d)

    def options(self) -> autop.AutoOptions:
        return autop.AutoOptions(
            mode=self.mode.currentData(), target_lang=self.lang.currentData(),
            series=self.series, whisper_size=self.whisper.currentText(),
            recap_minutes=self.minutes.value(), use_overlays=self.overlays.isChecked(),
            vertical=self.vertical.isChecked(), thumbnail=self.thumb.isChecked(),
            out_dir=self.out.text().strip(), vcodec=self.encoder.currentText(),
            overwrite=self.overwrite.isChecked())

    def _start(self) -> None:
        opts = self.options()
        saved = autop.options_dict(opts)
        saved.pop("overwrite", None)
        saved.pop("transcribe_only", None)
        self.ctx.config.set(OPTIONS_KEY, saved)
        if opts.mode == "dub":
            self._relink_moved()
            from app.engines.dub_engine import readiness
            # no AI key: asked on its own below, with what can be done
            bad = [c for c in readiness(self.ctx.config)
                   if c.required and not c.ok and c.id != "ai"]
            if bad:
                ans = QMessageBox.question(
                    self, self.windowTitle(),
                    tr("dub.blockers_q", "Some requirements are missing:") + "\n\n"
                    + "\n".join(f"• {c.label}: {c.fix}" for c in bad) + "\n\n"
                    + tr("dub.continue_q", "Continue anyway?"))
                if ans != QMessageBox.StandardButton.Yes:
                    return
        if no_ai_key(self.ctx.config):
            # both modes need the AI (a recap writes its script with it); a
            # dub can still transcribe every episode now
            choice = self._ask_no_key(opts.mode == "dub")
            if choice == "settings":
                self._open_settings()
                return
            if choice != "transcribe":
                return
            opts.transcribe_only = True
        self._opts = opts
        engine = autop.AutoPipeline(self.ctx.ai, self.ctx.config, self.ctx.library)
        self._worker = w = FnWorker(engine.run, self.episodes, opts)
        w.finished_ok.connect(self._done)
        w.failed.connect(self._failed)
        w.cancelled.connect(self._cancelled)
        self.log.show()
        self.log.appendPlainText(tr("auto.started", "Started") + f": {len(self.episodes)} "
                                 + tr("auto.episodes", "episode(s)"))
        self.start_btn.setEnabled(False)
        for wdg in self._inputs():
            wdg.setEnabled(False)
        self.panel.start("" if opts.transcribe_only else self._out_folder(opts), 0.0,
                         self._spent, title=tr("dub.run.starting", "Starting…"))
        self._fit_height()
        w.progress.connect(self._on_progress)
        _keep_awake(True)
        self.ctx.pool.start(w)

    def _ask_no_key(self, can_transcribe: bool) -> str:
        """No AI key yet: "settings", "transcribe" or "" (cancel); the Dub
        tab's question. (A method: tests answer it.)"""
        return ask_no_key(self, can_transcribe)

    def _open_settings(self) -> None:
        """Settings is in the main window: this dialog closes to reach it."""
        open_settings(self.ctx)
        self.reject()

    def _relink_moved(self) -> None:
        """Episodes dubbed before their video was moved (same name and
        length, the old place empty): ask once, then go on with those dubs
        instead of starting each again."""
        from app.engines import dub_engine as de
        found = []
        for ep in self.episodes:
            video = str(ep.get("filepath") or "")
            if not video or not os.path.isfile(video) or de.project_json_for(video):
                continue
            try:
                moved = de.find_moved_project(video)
            except Exception:
                log.debug("could not look for a moved project", exc_info=True)
                moved = ""
            if moved:
                found.append((str(ep.get("title") or Path(video).stem), video, moved))
        if not found:
            return
        names = "\n".join(f"• {title}" for title, _v, _m in found[:8])
        if len(found) > 8:
            names += "\n" + tr("auto.run.more", "… and {n} more").replace("{n}", str(len(found) - 8))
        ans = QMessageBox.question(
            self, self.windowTitle(),
            tr("auto.moved.q", "Earlier dubs were found for {n} of these episodes, made before "
                               "their videos were moved:\n{list}\n\nGo on with those dubs?")
            .replace("{n}", str(len(found))).replace("{list}", names))
        if ans != QMessageBox.StandardButton.Yes:
            return
        for _title, video, moved in found:
            try:
                de.DubProject.relink(moved, video)
            except Exception:
                log.warning("could not go on with %s for %s", moved, video, exc_info=True)

    def _fit_height(self) -> None:
        """Grow with the panel and the log (the form rows were squeezed
        flat when they appeared in a dialog sized for the form alone)."""
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.resize(self.width(),
                                                 max(self.height(), self.sizeHint().height())))

    # -- progress / result ---------------------------------------------------
    _STAGES = ("Dub: ", "Recap: ", "Assemble: ", "Overlays: ", "9:16: ")

    def _on_progress(self, frac: float, text: str) -> None:
        """"[2/5] Ep 2: Dub: Translate dialogue with context: Translating
        lines 1-30 of 46" -> title "episode 2 of 5: Ep 2", detail "Translate
        dialogue with context: Translating lines…" (in the interface language)."""
        m = re.match(r"^\[(\d+)/(\d+)\] (.*)$", text or "", re.S)
        detail = text
        if m:
            rest, title = m.group(3), m.group(3)
            for stage in self._STAGES:
                j = rest.find(": " + stage)
                if j >= 0:
                    title, detail = rest[:j], rest[j + 2 + len(stage):]
                    break
            self.panel.set_running_title(
                tr("auto.run.title", "Processing — episode {i} of {n}: {title}")
                .replace("{i}", m.group(1)).replace("{n}", m.group(2)).replace("{title}", title))
        for sid, label in STEPS:
            if detail.startswith(label + ": "):
                detail = (tr("dub.step." + sid, label) + ": "
                          + tr_text(detail[len(label) + 2:]))
                break
        else:
            detail = tr_text(detail)
        self.panel.set_progress(frac, detail)

    def _out_folder(self, opts=None) -> str:
        opts = opts or self.options()
        if opts.out_dir.strip():
            return opts.out_dir.strip()
        first = next((e.get("filepath") for e in self.episodes if e.get("filepath")), "")
        return str(Path(first).parent / "RecapStudio out") if first else ""

    def _show_folder(self) -> None:
        folder = self._out_folder()
        if folder and os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _cancel_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.panel.set_stopping()
            self._worker.cancel()

    def _ended(self) -> None:
        self._spent += self.panel.segment_seconds()
        _keep_awake(False)
        if not self.isActiveWindow():
            QApplication.alert(self)

    def _done(self, results) -> None:
        self._ended()
        results = list(results or [])
        bad = [r for r in results if r.status != "done"]
        n = len(results)
        opts = getattr(self, "_opts", None)
        if bad:
            lines = "\n".join(f"• {r.title}: {pick(tr_text(r.message), r.message_km) or r.status}"
                              for r in bad[:6])
            more = len(bad) - 6
            if more > 0:
                lines += "\n" + tr("auto.run.more", "… and {n} more").replace("{n}", str(more))
            # every one stopped because no AI answered: a pause, not a breakage
            paused = all(r.ai for r in bad)
            where = (tr("auto.run.waiting", "{k} of {n} episode(s) wait for the AI") if paused
                     else tr("auto.run.failed", "{k} of {n} episode(s) failed"))
            self.panel.set_error(
                where.replace("{k}", str(len(bad))).replace("{n}", str(n)), lines, change=False,
                hint=tr("auto.run.error_hint", "Continue runs them again — finished episodes "
                                               "are skipped; Cancel stops here."),
                settings=any(r.ai in FIX_IN_SETTINGS for r in bad), paused=paused)
        elif opts is not None and opts.transcribe_only:
            self.panel.set_partial(
                "", "", change=False, settings=no_ai_key(self.ctx.config),
                title=tr("auto.run.transcribed", "Transcribed: {n} episode(s)")
                .replace("{n}", str(n)),
                detail=tr("auto.run.transcribed_next", "Next: add an AI key in Settings → AI "
                                                       "Providers, then press Continue to "
                                                       "translate and dub them."))
        else:
            self.panel.set_done(self._out_folder(), "", "", play=False,
                                title=tr("auto.run.done", "Done — {n} episode(s) finished")
                                .replace("{n}", str(n)))
        self._finish(autop.summary(results))

    def _failed(self, en: str, km: str) -> None:
        self._ended()
        from app.core.i18n import get_language
        msg = km if get_language() == "km" and km else en
        self.panel.set_error("", msg, en, change=False,
                             hint=tr("auto.run.error_hint", "Continue runs them again — finished "
                                                            "episodes are skipped; Cancel stops here."))
        self._finish(en)

    def _cancelled(self) -> None:
        self._ended()
        self.panel.set_stopped("", tr("auto.run.you_stopped", "You stopped processing."),
                               change=False)
        self._finish(tr("common.cancelled", "Cancelled"))

    def _finish(self, text: str) -> None:
        self._fit_height()
        self.log.appendPlainText(text)
        self.start_btn.setEnabled(True)
        for wdg in self._inputs():
            wdg.setEnabled(True)
        self._mode_changed()

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            ans = QMessageBox.question(self, self.windowTitle(),
                                       tr("auto.stop_q", "Stop processing? Finished episodes are kept "
                                                         "and the next run resumes."))
            if ans != QMessageBox.StandardButton.Yes:
                return
            self._worker.cancel()
        super().reject()
