"""Download tab: scan a link into the card (one video, or a playlist /
series as episode cards), download what is chosen, and a live queue table."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.downloader import (DuplicateUrlError, cookie_file_missing,
                                 sanitize_dirname, url_key)
from app.core.i18n import get_language, tr
from app.core.logger import get_logger
from app.ui.tabs.dl_card import ScanCard

if TYPE_CHECKING:  # avoid import cycles at runtime; ctx is duck-typed
    from app.ui.context import AppContext

log = get_logger("ui.dl_tab")

COL_TITLE, COL_SERIES, COL_STATUS, COL_PROGRESS, COL_SPEED, COL_ETA, COL_ACTIONS = range(7)

_STATUS_KEYS = {
    "queued": ("dl.status.queued", "Queued"),
    "fetching_meta": ("dl.status.fetching", "Fetching info"),
    "downloading": ("dl.status.downloading", "Downloading"),
    "processing": ("dl.status.processing", "Adding poster…"),
    "paused": ("dl.status.paused", "Paused"),
    "done": ("dl.status.done", "Done"),
    "error": ("dl.status.error", "Error"),
    # UI-only: X was clicked on a running item that has not stopped yet
    "cancelling": ("dl.status.cancelling", "Cancelling…"),
}

_STATUS_COLORS = {
    "done": "#3fb950",
    "error": "#f85149",
    "paused": "#d29922",
    "cancelling": "#d29922",
    "downloading": "#58a6ff",
    "processing": "#a5d6ff",
}


def _err_text(en: str, km: str) -> str:
    return km if (get_language() == "km" and km) else en


def _open_folder(path: str) -> None:
    try:
        if path and os.path.isdir(path):
            os.startfile(path)  # noqa: S606 - Windows desktop app
    except Exception as exc:  # pragma: no cover - shell integration
        log.warning("open folder failed: %s", exc)


def _reveal_file(filepath: str) -> None:
    try:
        if filepath and os.path.isfile(filepath):
            subprocess.Popen(
                ["explorer", "/select,", os.path.normpath(filepath)],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            _open_folder(os.path.dirname(filepath) if filepath else "")
    except Exception as exc:  # pragma: no cover
        log.warning("reveal file failed: %s", exc)


class _Bridge(QObject):
    """Thread-safe bridge: DownloadQueue callback (worker thread) -> UI slot."""

    item_changed = Signal(object)


class _BatchDialog(QDialog):
    """Multi-line URL input dialog (one URL per line) + optional series name."""

    def __init__(self, parent: QWidget | None = None, series: str = ""):
        super().__init__(parent)
        self.setWindowTitle(tr("dl.batch.title", "Add batch of URLs"))
        self.resize(560, 380)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(tr("dl.batch.hint", "One URL per line:")))
        self.edit = QPlainTextEdit(self)
        self.edit.setPlaceholderText("https://...\nhttps://...")
        lay.addWidget(self.edit, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("dl.series", "Series")))
        self.series_edit = QLineEdit(series, self)
        self.series_edit.setPlaceholderText(tr("dl.series.ph", "Series name (optional)"))
        row.addWidget(self.series_edit, 1)
        lay.addLayout(row)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def urls(self) -> list[str]:
        return [ln.strip() for ln in self.edit.toPlainText().splitlines() if ln.strip()]

    def series(self) -> str:
        return self.series_edit.text().strip()


class _RowActions(QWidget):
    """Per-row action buttons; visibility follows the item status."""

    def __init__(self, tab: "DlTab", item_id: str):
        super().__init__()
        self._tab = tab
        self._id = item_id
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(2)

        def mk(text: str, tip: str) -> QToolButton:
            b = QToolButton(self)
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            lay.addWidget(b)
            return b

        self.btn_pause = mk("II", tr("dl.act.pause", "Pause"))
        self.btn_resume = mk(">", tr("dl.act.resume", "Resume"))
        self.btn_retry = mk("R", tr("dl.act.retry", "Retry"))
        self.btn_cancel = mk("X", tr("dl.act.remove", "Cancel / remove from the list"))
        self.btn_folder = mk("[]", tr("dl.act.open_folder", "Open folder"))
        lay.addStretch(1)

        self.btn_pause.clicked.connect(lambda: tab._queue_call("pause", self._id))
        self.btn_resume.clicked.connect(lambda: tab._queue_call("resume", self._id))
        self.btn_retry.clicked.connect(lambda: tab._queue_call("retry", self._id))
        self.btn_cancel.clicked.connect(lambda: tab._queue_call("cancel", self._id))
        self.btn_folder.clicked.connect(lambda: _reveal_file(tab._filepath_for(self._id)))
        self.set_status("queued")

    def set_status(self, status: str) -> None:
        # "processing" (adding the poster) cannot pause: the video is complete
        self.btn_pause.setVisible(status in ("queued", "fetching_meta", "downloading"))
        self.btn_resume.setVisible(status == "paused")
        self.btn_retry.setVisible(status == "error")
        # done rows can be taken off the list too (the file stays on disk)
        self.btn_cancel.setVisible(status != "cancelling")
        self.btn_folder.setVisible(status == "done")


class DlTab(QWidget):
    """Downloads tab: the scan card feeds ctx.downloads; the table mirrors
    every item's state."""

    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self._rows: dict[str, int] = {}          # item id -> table row
        self._filepaths: dict[str, str] = {}     # item id -> last known filepath
        self._thumb_set: set[str] = set()        # ids whose thumbnail icon is applied
        self._removed: set[str] = set()          # ids taken off the list (late updates ignored)
        self._cancelling: set[str] = set()       # X clicked, worker not stopped yet
        self._marks: dict[str, str] = {}         # url key -> "list" | "done" (card tags)
        self._scan_worker = None
        self._scan_gen = 0
        self._last_scan_url = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- link row --------------------------------------------------
        top = QHBoxLayout()
        self.url_edit = QLineEdit(self)
        self.url_edit.setPlaceholderText(tr("dl.url.ph", "Paste a video, playlist or series link…"))
        self.url_edit.setClearButtonEnabled(True)
        self.url_edit.returnPressed.connect(self._scan_clicked)
        top.addWidget(self.url_edit, 1)
        btn_paste = QPushButton(tr("dl.paste", "Paste"), self)
        btn_paste.setToolTip(tr("dl.paste.tip", "Paste the link from the clipboard and scan it"))
        btn_paste.clicked.connect(self._paste_clicked)
        top.addWidget(btn_paste)
        self.scan_btn = QPushButton(tr("dl.scan", "Scan"), self)
        self.scan_btn.setProperty("class", "accent")
        self.scan_btn.setToolTip(tr("dl.scan.tip", "Read the link: one video, or every episode "
                                                   "of a playlist / series"))
        self.scan_btn.clicked.connect(self._scan_clicked)
        top.addWidget(self.scan_btn)
        btn_batch = QPushButton(tr("dl.add_batch", "Add batch..."), self)
        btn_batch.clicked.connect(self._batch_clicked)
        top.addWidget(btn_batch)
        root.addLayout(top)

        # --- login row -------------------------------------------------
        cfg = self.ctx.config
        opt = QHBoxLayout()
        opt.addWidget(QLabel(tr("dl.cookies", "Login:"), self))
        self.cookie_combo = QComboBox(self)
        self.cookie_combo.addItem(tr("dl.cookies.none", "None (public videos)"), "")
        self.cookie_combo.addItem(tr("dl.cookies.file", "cookies.txt file…"), "file")
        for b in ("firefox", "chrome", "edge"):
            self.cookie_combo.addItem(tr("dl.cookies.browser", "Cookies from") + f" {b.title()}",
                                      f"browser:{b}")
        self.cookie_combo.setToolTip(tr(
            "dl.cookies.tip", "For sites that need you to be logged in (and age-limited "
                              "videos). Chrome / Edge must be closed while reading their "
                              "cookies and may still refuse; a cookies.txt file or Firefox "
                              "works best."))
        saved = str(cfg.get("downloads.cookies", "") or "")
        self._cookie_file = saved[5:] if saved.startswith("file:") else ""
        i = self.cookie_combo.findData("file" if saved.startswith("file:") else saved)
        self.cookie_combo.setCurrentIndex(max(0, i))
        self.cookie_combo.activated.connect(self._cookie_picked)
        opt.addWidget(self.cookie_combo)
        self.cookie_label = QLabel("", self)
        self.cookie_label.setStyleSheet("color: #8b949e;")
        opt.addWidget(self.cookie_label, 1)
        sites = QLabel(tr("dl.sites", "YouTube, Bilibili, TikTok, Facebook, Dailymotion and "
                                      "1000+ more sites (yt-dlp)"), self)
        sites.setStyleSheet("color: #8b949e;")
        opt.addWidget(sites)
        root.addLayout(opt)

        # --- scan card ---------------------------------------------------
        self.card = ScanCard(self)
        self.card.set_quality(str(cfg.get("downloads.quality", "best") or "best"))
        self.card.set_poster(bool(cfg.get("downloads.poster_intro", True)))
        self.card.set_ask(bool(cfg.get("downloads.ask_folder", True)))
        self.card.set_folder(self._downloads_dir())
        self.card.download_clicked.connect(self._download_from_card)
        self.card.change_folder.connect(lambda: self._choose_folder(force=True))
        self.card.ask_toggled.connect(lambda on: self.ctx.config.set("downloads.ask_folder", on))
        self.card.poster_toggled.connect(self._poster_toggled)
        self.card.quality_changed.connect(self._quality_changed)
        self.card.cancel_clicked.connect(self._cancel_scan)
        self.card.retry_clicked.connect(lambda: self._scan(self._last_scan_url))
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.card.shutdown)

        # --- download list -------------------------------------------------
        queue_box = QWidget(self)
        ql = QVBoxLayout(queue_box)
        ql.setContentsMargins(0, 6, 0, 0)
        ql.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel(tr("dl.queue.title", "Download list"), queue_box)
        title.setStyleSheet("font-weight: 600; color: #8b949e;")
        head.addWidget(title)
        head.addStretch(1)
        btn_open = QPushButton(tr("dl.open_folder", "Open downloads folder"), queue_box)
        btn_open.clicked.connect(lambda: _open_folder(self._downloads_dir()))
        head.addWidget(btn_open)
        ql.addLayout(head)

        self.table = QTableWidget(0, 7, queue_box)
        self.table.setHorizontalHeaderLabels([
            tr("dl.col.title", "Title"),
            tr("dl.col.series", "Series"),
            tr("dl.col.status", "Status"),
            tr("dl.col.progress", "Progress"),
            tr("dl.col.speed", "Speed"),
            tr("dl.col.eta", "ETA"),
            tr("dl.col.actions", "Actions"),
        ])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(COL_TITLE, QHeaderView.Stretch)
        hdr.setSectionResizeMode(COL_SERIES, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_PROGRESS, QHeaderView.Fixed)
        self.table.setColumnWidth(COL_PROGRESS, 160)
        hdr.setSectionResizeMode(COL_SPEED, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_ETA, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_ACTIONS, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setIconSize(QSize(48, 28))
        ql.addWidget(self.table, 1)

        split = QSplitter(Qt.Vertical, self)
        split.setChildrenCollapsible(False)
        split.addWidget(self.card)
        split.addWidget(queue_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([560, 260])
        root.addWidget(split, 1)

        # the queue saves where the card says (unless an item has its own folder)
        try:
            self.ctx.downloads.set_out_dir(self._downloads_dir())
        except Exception as exc:
            log.warning("set_out_dir failed: %s", exc)
        self._options_changed()

        # --- callback bridge (worker thread -> queued signal -> slot) ---
        self._bridge = _Bridge(self)
        self._bridge.item_changed.connect(self._on_item_changed)
        try:
            self.ctx.downloads.set_callback(self._bridge.item_changed.emit)
        except Exception as exc:
            log.warning("set_callback failed: %s", exc)

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------
    def _paste_clicked(self) -> None:
        text = QApplication.clipboard().text().strip()
        if text:
            self.url_edit.setText(text.splitlines()[0].strip())
            self._scan_clicked()

    def _scan_clicked(self) -> None:
        url = self.url_edit.text().strip()
        if url:
            self._scan(url)

    def _scan(self, url: str) -> None:
        from app.core.downloader import looks_like_url, scan_url
        from app.ui.workers import FnWorker
        url = (url or "").strip()
        if not url:
            return
        self._last_scan_url = url
        if not looks_like_url(url):
            self.card.show_error(tr(
                "dl.err.not_url", "That doesn't look like a link. Copy the address of a video, "
                                  "playlist or series page (it starts with https://)."))
            return
        self._cancel_scan(silent=True)
        gen = self._scan_gen
        self.card.show_busy(url)
        w = FnWorker(scan_url, url, self._cookie_value())
        w.progress.connect(lambda _f, text, g=gen: g == self._scan_gen and self.card.set_busy_text(text))
        w.finished_ok.connect(lambda res, g=gen: self._on_scanned(g, res))
        w.failed.connect(lambda en, km, g=gen: self._on_scan_failed(g, en, km))
        self._scan_worker = w
        self.ctx.pool.start(w)

    def _cancel_scan(self, silent: bool = False) -> None:
        if self._scan_worker is not None:
            try:
                self._scan_worker.cancel()
            except Exception:
                pass
        self._scan_worker = None
        self._scan_gen += 1          # whatever the old scan still returns is ignored
        if not silent:
            self.card.show_empty()

    def _on_scanned(self, gen: int, res) -> None:
        if gen != self._scan_gen:
            return
        self._scan_worker = None
        self.card.show_result(res)
        self.card.set_marks(self._marks)
        if res.kind == "playlist" and res.entries:
            # one episode's sizes stand in for the others (the estimates)
            from app.core.downloader import probe_video
            from app.ui.workers import FnWorker
            sample = next((e for e in res.entries if e.matched), res.entries[0])
            w = FnWorker(probe_video, sample.url, self._cookie_value())
            w.finished_ok.connect(lambda d, g=gen: g == self._scan_gen and self.card.set_sample(d))
            self.ctx.pool.start(w)

    def _on_scan_failed(self, gen: int, en: str, km: str) -> None:
        if gen != self._scan_gen:
            return
        self._scan_worker = None
        self.card.show_error(_err_text(en, km))

    # ------------------------------------------------------------------
    # downloading
    # ------------------------------------------------------------------
    def _download_from_card(self) -> None:
        res = self.card.result
        entries = self.card.selected_entries()
        if res is None or not entries:
            return
        base = self._choose_folder()
        if not base:
            return
        playlist = res.kind == "playlist"
        series = self.card.series_name() if playlist else ""
        out_dir = Path(base)
        if playlist:
            # episodes go in a folder named after the show, unless that is
            # the folder the user just picked
            sub = sanitize_dirname(series or res.title or "Playlist")
            if out_dir.name.casefold() != sub.casefold():
                out_dir = out_dir / sub
        quality, poster = self.card.quality_key(), self.card.poster_intro()
        added = skipped = 0
        try:
            for e in entries:
                try:
                    item = self.ctx.downloads.add(
                        e.url, series, index=e.ep if playlist else 0, title=e.title,
                        quality=quality, out_dir=str(out_dir), poster_intro=poster,
                        thumb_hint=e.thumb)
                except DuplicateUrlError as exc:
                    if not playlist:
                        self._show_duplicate(exc.item)     # point at its row
                        return
                    skipped += 1
                    continue
                self._on_item_changed(item)
                added += 1
            self._start_queue()
        except Exception as exc:
            self._show_error(exc)
        if added:
            self.card.note("✓ " + tr("dl.added", "{n} added to the download list")
                           .replace("{n}", str(added)))
        self._show_skipped(skipped)

    def _batch_clicked(self) -> None:
        from app.core.downloader import looks_like_url
        dlg = _BatchDialog(self, self.card.series_name() if self.card.is_playlist else "")
        if dlg.exec() != QDialog.Accepted:
            return
        lines = dlg.urls()
        urls = [u for u in lines if looks_like_url(u)]
        bad = len(lines) - len(urls)
        if bad:
            QMessageBox.information(
                self, tr("dl.batch.title", "Add batch of URLs"),
                tr("dl.batch.bad", "{n} line(s) were not links and were skipped.")
                .replace("{n}", str(bad)))
        base = self._choose_folder() if urls else ""
        if not base:
            return
        # straight into the picked folder; a series name makes a folder in it
        series = dlg.series()
        out_dir = str(Path(base) / sanitize_dirname(series)) if series else base
        try:
            items = self.ctx.downloads.add_many(urls, series, out_dir=out_dir)
            for it in items:
                self._on_item_changed(it)
            self._start_queue()
            self._show_skipped(len(urls) - len(items))
        except Exception as exc:
            self._show_error(exc)

    def _choose_folder(self, force: bool = False) -> str:
        """Where downloads go: asks (starting at the last folder) when "Ask
        where to save each time" is on or force; "" if the user cancels."""
        current = self._downloads_dir()
        if not (force or self.ctx.config.get("downloads.ask_folder", True)):
            return current
        path = QFileDialog.getExistingDirectory(
            self, tr("dl.save.pick", "Choose where to save the videos"), current)
        if not path:
            return ""
        path = os.path.normpath(path)
        self.ctx.config.set("downloads.dir", path)
        try:
            self.ctx.downloads.set_out_dir(path)
        except Exception as exc:
            log.warning("set_out_dir failed: %s", exc)
        self.card.set_folder(path)
        return path

    def _cookie_value(self) -> str:
        data = self.cookie_combo.currentData() or ""
        if data == "file":
            return f"file:{self._cookie_file}" if self._cookie_file else ""
        return data

    def _cookie_picked(self, _i: int) -> None:
        if self.cookie_combo.currentData() == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, tr("dl.cookies.pick", "Choose a cookies.txt file"),
                os.path.dirname(self._cookie_file) if self._cookie_file else "",
                "cookies (*.txt);;All files (*)")
            if path:
                self._cookie_file = path
            elif not self._cookie_file:
                self.cookie_combo.setCurrentIndex(0)
        self._options_changed()

    def _quality_changed(self, key: str) -> None:
        self.ctx.config.set("downloads.quality", key)
        self._options_changed()

    def _poster_toggled(self, on: bool) -> None:
        self.ctx.config.set("downloads.poster_intro", on)
        self._options_changed()

    def _encoder(self) -> str:
        """The hardware H.264 encoder Settings found (NVENC / AMF / QSV), else x264."""
        try:
            from app.engines.media_exporter import best_h264_encoder
            return best_h264_encoder(self.ctx.config)
        except Exception:
            return "libx264"

    def _options_changed(self, *_a) -> None:
        cookies = self._cookie_value()
        self.ctx.config.set("downloads.cookies", cookies)
        self.ctx.downloads.set_options(quality=self.card.quality_key(), cookies=cookies,
                                       poster_intro=self.card.poster_intro(),
                                       encoder=self._encoder())
        self.card.set_cookies(cookies)
        name = os.path.basename(cookies[5:]) if cookies.startswith("file:") else ""
        if cookie_file_missing(cookies):
            # moved or deleted: downloads would quietly run without the login
            self.cookie_label.setStyleSheet("color: #f85149;")
            self.cookie_label.setText("⚠ " + tr(
                "dl.cookies.missing", "{name} not found (moved or deleted?): downloads run "
                "without your login. Pick the file again.").replace("{name}", name))
        else:
            self.cookie_label.setStyleSheet("color: #8b949e;")
            self.cookie_label.setText(name)

    def _start_queue(self) -> None:
        try:
            self.ctx.downloads.start()
        except Exception as exc:
            log.warning("downloads.start failed: %s", exc)

    def _queue_call(self, method: str, item_id: str) -> None:
        try:
            fn = getattr(self.ctx.downloads, method, None)
            if callable(fn):
                fn(item_id)
            if method == "retry":
                self._start_queue()
            elif method == "cancel":
                item = self.ctx.downloads.items.get(item_id)
                if item is None:
                    # gone from the queue (normally the "removed" update
                    # already dropped the row): never leave a dead row
                    self._remove_row(item_id)
                else:
                    # still running: yt-dlp only stops at its next check,
                    # seconds away while it reads the video info — say so now
                    self._cancelling.add(item_id)
                    self._on_item_changed(item)
        except Exception as exc:
            self._show_error(exc)

    def _show_error(self, exc: Exception) -> None:
        en = getattr(exc, "en", str(exc))
        km = getattr(exc, "km", "")
        QMessageBox.warning(self, tr("dl.error", "Download error"), _err_text(en, km))

    def _show_duplicate(self, item) -> None:
        """The link is already in the list: point at its row instead."""
        row = self._rows.get(item.id)
        if row is not None:
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, COL_TITLE))
        key, default = _STATUS_KEYS.get(item.status, ("dl.status.unknown", item.status))
        QMessageBox.information(
            self, tr("dl.dup.title", "Already in the list"),
            tr("dl.dup.one", "This link is already in the list (status: {status}).\n"
                             "Use R to retry it, or X to remove it and add it again.")
            .replace("{status}", tr(key, default)) + f"\n\n{item.url}")

    def _show_skipped(self, count: int) -> None:
        if count > 0:
            QMessageBox.information(
                self, tr("dl.dup.title", "Already in the list"),
                tr("dl.dup.some", "{n} link(s) were already in the list and were skipped.")
                .replace("{n}", str(count)))

    # ------------------------------------------------------------------
    # queue state -> table (and the card's "In list" / "Saved" tags)
    # ------------------------------------------------------------------
    def _set_mark(self, url: str, mark: str | None) -> None:
        key = url_key(url)
        if self._marks.get(key) == mark:
            return
        if mark is None:
            self._marks.pop(key, None)
        else:
            self._marks[key] = mark
        self.card.set_marks(self._marks)

    def _on_item_changed(self, item) -> None:
        if item is None:
            return
        item_id = getattr(item, "id", "")
        if not item_id or item_id in self._removed:
            return
        status = getattr(item, "status", "queued")
        if status == "removed":
            self._set_mark(getattr(item, "url", ""), None)
            self._remove_row(item_id)
            return
        self._set_mark(getattr(item, "url", ""), "done" if status == "done" else "list")
        if item_id in self._cancelling:
            if status in ("done", "error", "paused"):
                # it stopped (or finished) before it saw the cancel: take it
                # off the list now; a finished video stays on disk
                self.ctx.downloads.cancel(item_id)
                self._remove_row(item_id)
                return
            status = "cancelling"
        row = self._rows.get(item_id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows[item_id] = row
            title_cell = QTableWidgetItem("")
            title_cell.setData(Qt.UserRole, item_id)
            self.table.setItem(row, COL_TITLE, title_cell)
            self.table.setItem(row, COL_SERIES, QTableWidgetItem(""))
            self.table.setItem(row, COL_STATUS, QTableWidgetItem(""))
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(True)
            self.table.setCellWidget(row, COL_PROGRESS, bar)
            self.table.setItem(row, COL_SPEED, QTableWidgetItem(""))
            self.table.setItem(row, COL_ETA, QTableWidgetItem(""))
            self.table.setCellWidget(row, COL_ACTIONS, _RowActions(self, item_id))
            self.table.setRowHeight(row, 34)

        title = getattr(item, "title", "") or getattr(item, "url", "")
        error = getattr(item, "error", "")
        self._filepaths[item_id] = getattr(item, "filepath", "") or self._filepaths.get(item_id, "")

        title_item = self.table.item(row, COL_TITLE)
        title_item.setText(title)
        title_item.setToolTip(getattr(item, "url", ""))
        thumb = getattr(item, "thumbnail", "")
        if thumb and item_id not in self._thumb_set and os.path.isfile(thumb):
            pix = QPixmap(thumb)
            if not pix.isNull():
                title_item.setIcon(QIcon(pix.scaled(48, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                self._thumb_set.add(item_id)

        self.table.item(row, COL_SERIES).setText(getattr(item, "series", ""))

        key, default = _STATUS_KEYS.get(status, ("dl.status.unknown", status))
        st_item = self.table.item(row, COL_STATUS)
        st_item.setText(tr(key, default))
        st_item.setToolTip(error)
        color = _STATUS_COLORS.get(status)
        if color:
            from PySide6.QtGui import QColor
            st_item.setForeground(QColor(color))
        else:
            # back to the normal text colour: after R a row would otherwise
            # keep the red of "Error" while it is queued / fetching again
            st_item.setData(Qt.ForegroundRole, None)

        bar = self.table.cellWidget(row, COL_PROGRESS)
        if isinstance(bar, QProgressBar):
            frac = float(getattr(item, "progress", 0.0) or 0.0)
            frac = 1.0 if status == "done" else max(0.0, min(1.0, frac))
            bar.setValue(int(frac * 1000))
            bar.setFormat(f"{frac * 100:.1f}%")

        self.table.item(row, COL_SPEED).setText(getattr(item, "speed", "") or "")
        self.table.item(row, COL_ETA).setText(getattr(item, "eta", "") or "")

        actions = self.table.cellWidget(row, COL_ACTIONS)
        if isinstance(actions, _RowActions):
            actions.set_status(status)

    def _remove_row(self, item_id: str) -> None:
        self._removed.add(item_id)
        self._cancelling.discard(item_id)
        self._filepaths.pop(item_id, None)
        self._thumb_set.discard(item_id)
        row = self._rows.pop(item_id, None)
        if row is None:
            return
        self.table.removeRow(row)
        # rows below shifted up: rebuild id -> row from the stored ids
        self._rows = {}
        for r in range(self.table.rowCount()):
            cell = self.table.item(r, COL_TITLE)
            rid = cell.data(Qt.UserRole) if cell is not None else None
            if rid:
                self._rows[rid] = r

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _filepath_for(self, item_id: str) -> str:
        return self._filepaths.get(item_id, "")

    def _downloads_dir(self) -> str:
        """The folder the user picked last, else Videos\\RecapStudio."""
        v = self.ctx.config.get("downloads.dir")
        if v and os.path.isdir(str(v)):
            return str(v)
        d = Path.home() / "Videos" / "RecapStudio"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(d)
