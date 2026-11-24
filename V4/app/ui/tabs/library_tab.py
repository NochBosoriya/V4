"""Library tab: series list on the left, episodes table on the right."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import tr
from app.core.logger import get_logger

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.library_tab")

(
    COL_TITLE,
    COL_DURATION,
    COL_RES,
    COL_TRANSCRIBED,
    COL_TRANSLATED,
    COL_DUBBED,
    COL_RENDERED,
    COL_ACTIONS,
) = range(8)

_FLAG_COLS = {
    COL_TRANSCRIBED: "transcribed",
    COL_TRANSLATED: "translated",
    COL_DUBBED: "dubbed",
    COL_RENDERED: "rendered",
}


def _fmt_duration(seconds: float) -> str:
    try:
        s = int(round(float(seconds or 0)))
    except Exception:
        s = 0
    if s <= 0:
        return "-"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class _RefreshBridge(QObject):
    """Bridges downloads-finished callbacks (worker threads) to the UI thread."""

    changed = Signal(object)


class _EpisodeActions(QWidget):
    def __init__(self, tab: "LibraryTab", episode: dict):
        super().__init__()
        self._tab = tab
        self._ep = dict(episode)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(2)

        def mk(text: str, tip: str, slot) -> QToolButton:
            b = QToolButton(self)
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        mk("▶", tr("lib.act.preview", "Open in preview"), self._preview)
        mk("✂", tr("lib.act.recap", "Recap this"), self._recap)
        mk("🎙", tr("lib.act.dub", "Full dub this"), self._dub)
        mk("⇪", tr("lib.act.export", "Export this"), self._export)
        mk("✕", tr("lib.act.delete", "Delete"), self._delete)
        lay.addStretch(1)

    def _preview(self) -> None:
        self._tab.open_episode_preview(self._ep)

    def _recap(self) -> None:
        self._tab.recap_episode(self._ep)

    def _export(self) -> None:
        self._tab.export_episode(self._ep)

    def _dub(self) -> None:
        self._tab.dub_episode(self._ep)

    def _delete(self) -> None:
        self._tab.delete_episode(self._ep)


class LibraryTab(QWidget):
    """Two-pane drama library backed by ctx.library."""

    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self._current_series_id: int | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel(tr("lib.title", "Drama Library")))
        top.addStretch(1)
        btn_sel = QPushButton(tr("lib.process_sel", "Process selected…"), self)
        btn_sel.setToolTip(tr("lib.process_sel.tip",
                              "Dub or recap the selected episodes into finished videos."))
        btn_sel.clicked.connect(lambda: self.process_episodes(selected_only=True))
        top.addWidget(btn_sel)
        btn_all = QPushButton(tr("lib.process_all", "Process series…"), self)
        btn_all.setProperty("class", "accent")
        btn_all.setToolTip(tr("lib.process_all.tip",
                              "One click: every episode of this series becomes a "
                              "finished video (+ 9:16 version and thumbnail)."))
        btn_all.clicked.connect(lambda: self.process_episodes(selected_only=False))
        top.addWidget(btn_all)
        btn_refresh = QPushButton(tr("lib.refresh", "Refresh"), self)
        btn_refresh.clicked.connect(self.refresh)
        top.addWidget(btn_refresh)
        root.addLayout(top)

        split = QSplitter(Qt.Horizontal, self)
        root.addWidget(split, 1)

        # --- left: series list -------------------------------------------
        self.series_list = QListWidget(split)
        self.series_list.setIconSize(QSize(48, 64))
        self.series_list.currentItemChanged.connect(self._series_selected)
        self.series_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.series_list.customContextMenuRequested.connect(self._series_menu)

        # --- right: episodes table ----------------------------------------
        self.table = QTableWidget(0, 8, split)
        self.table.setHorizontalHeaderLabels([
            tr("lib.col.title", "Title"),
            tr("lib.col.duration", "Duration"),
            tr("lib.col.resolution", "Resolution"),
            tr("lib.col.transcribed", "Transcribed"),
            tr("lib.col.translated", "Translated"),
            tr("lib.col.dubbed", "Dubbed"),
            tr("lib.col.rendered", "Rendered"),
            tr("lib.col.actions", "Actions"),
        ])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(COL_TITLE, QHeaderView.Stretch)
        for c in (COL_DURATION, COL_RES, COL_TRANSCRIBED, COL_TRANSLATED,
                  COL_DUBBED, COL_RENDERED, COL_ACTIONS):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._row_double_clicked)

        split.addWidget(self.series_list)
        split.addWidget(self.table)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        # auto refresh: downloads callback bridge + fallback timer
        self._bridge = _RefreshBridge(self)
        self._bridge.changed.connect(self._on_download_change)
        self._timer = QTimer(self)
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self._timer_tick)
        self._timer.start()
        self._dirty = False

        self.refresh()

    # ------------------------------------------------------------------
    # refresh logic
    # ------------------------------------------------------------------
    def notify_download_done(self, item) -> None:
        """May be called from any thread (e.g. main window fans out DL callback)."""
        self._bridge.changed.emit(item)

    def _on_download_change(self, item) -> None:
        if getattr(item, "status", "") == "done":
            self.refresh()
        else:
            self._dirty = True

    def _timer_tick(self) -> None:
        if self._dirty and self.isVisible():
            self._dirty = False
            self.refresh()

    def refresh(self) -> None:
        try:
            series = self.ctx.library.get_series()
        except Exception as exc:
            log.warning("get_series failed: %s", exc)
            series = []
        prev = self._current_series_id
        self.series_list.blockSignals(True)
        self.series_list.clear()
        select_row = 0
        for i, s in enumerate(series):
            name = s.get("name", "?")
            count = s.get("episode_count", s.get("episodes", ""))
            if count in ("", None):
                try:
                    count = len(self.ctx.library.get_episodes(s.get("id")))
                except Exception:
                    count = 0
            item = QListWidgetItem(f"{name}  ({count})")
            item.setData(Qt.UserRole, s.get("id"))
            poster = s.get("poster", "")
            if poster and os.path.isfile(poster):
                pix = QPixmap(poster)
                if not pix.isNull():
                    item.setIcon(QIcon(pix.scaled(48, 64, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)))
            self.series_list.addItem(item)
            if s.get("id") == prev:
                select_row = i
        self.series_list.blockSignals(False)
        if self.series_list.count():
            self.series_list.setCurrentRow(select_row)
        else:
            self._current_series_id = None
            self.table.setRowCount(0)

    def _series_selected(self, current: QListWidgetItem | None, _prev=None) -> None:
        if current is None:
            self._current_series_id = None
            self.table.setRowCount(0)
            return
        self._current_series_id = current.data(Qt.UserRole)
        self._load_episodes()

    def _load_episodes(self) -> None:
        self.table.setRowCount(0)
        if self._current_series_id is None:
            return
        try:
            episodes = self.ctx.library.get_episodes(self._current_series_id)
        except Exception as exc:
            log.warning("get_episodes failed: %s", exc)
            episodes = []
        for ep in episodes:
            row = self.table.rowCount()
            self.table.insertRow(row)
            title_item = QTableWidgetItem(ep.get("title", "?"))
            title_item.setData(Qt.UserRole, dict(ep))
            title_item.setToolTip(ep.get("filepath", ""))
            self.table.setItem(row, COL_TITLE, title_item)
            self.table.setItem(row, COL_DURATION,
                               QTableWidgetItem(_fmt_duration(ep.get("duration", 0))))
            w, h = ep.get("width", 0) or 0, ep.get("height", 0) or 0
            self.table.setItem(row, COL_RES,
                               QTableWidgetItem(f"{w}x{h}" if w and h else "-"))
            for col, flag in _FLAG_COLS.items():
                done = bool(ep.get(flag))
                it = QTableWidgetItem("✓" if done else "—")
                it.setTextAlignment(Qt.AlignCenter)
                it.setForeground(QColor("#3fb950" if done else "#8b949e"))
                self.table.setItem(row, col, it)
            self.table.setCellWidget(row, COL_ACTIONS, _EpisodeActions(self, ep))
            self.table.setRowHeight(row, 32)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def _episode_at(self, row: int) -> dict | None:
        it = self.table.item(row, COL_TITLE)
        return it.data(Qt.UserRole) if it else None

    def _row_double_clicked(self, row: int, _col: int) -> None:
        ep = self._episode_at(row)
        if ep:
            self.open_episode_preview(ep)

    def open_episode_preview(self, ep: dict) -> None:
        path = ep.get("filepath", "")
        if not path or not os.path.isfile(path):
            QMessageBox.warning(
                self, tr("lib.missing", "File missing"),
                tr("lib.missing.msg", "The video file was not found on disk."))
            return
        try:
            self.ctx.open_in_preview(path)
        except Exception as exc:
            log.warning("open_in_preview failed: %s", exc)

    def recap_episode(self, ep: dict) -> None:
        path = ep.get("filepath", "")
        if path:
            self.ctx.current_video = path
        self._switch_tab("recap")

    def export_episode(self, ep: dict) -> None:
        path = ep.get("filepath", "")
        if path:
            self.ctx.current_video = path
        target = self._switch_tab("export")
        if target is not None and hasattr(target, "set_source_file") and path:
            try:
                target.set_source_file(path)
            except Exception as exc:
                log.warning("export preload failed: %s", exc)

    def _switch_tab(self, name: str):
        """Switch the main window to tab `name` ("recap", "export", "dub"...)
        and return that tab's page widget (or None)."""
        main = getattr(self.ctx, "main", None)
        if main is None:
            return None
        try:
            if main.select_tab_by_key(name):
                return main.tab_widget_by_key(name)
        except Exception as exc:
            log.warning("tab switch to %s failed: %s", name, exc)
        return None

    def dub_episode(self, ep: dict) -> None:
        path = ep.get("filepath", "")
        if not path or not os.path.isfile(path):
            self.open_episode_preview(ep)      # shows the "file missing" message
            return
        self.ctx.current_video = path
        page = self._switch_tab("dub")
        if page is not None and hasattr(page, "_open_video"):
            try:
                page._open_video(path)
            except Exception as exc:
                log.warning("open in Full Dub failed: %s", exc)

    def process_episodes(self, selected_only: bool) -> None:
        if self._current_series_id is None:
            QMessageBox.information(self, tr("lib.title", "Drama Library"),
                                    tr("lib.no_series", "Pick a series first."))
            return
        if selected_only:
            rows = sorted({i.row() for i in self.table.selectedIndexes()})
            episodes = [e for e in (self._episode_at(r) for r in rows) if e]
        else:
            episodes = list(self.ctx.library.get_episodes(self._current_series_id))
        if not episodes:
            QMessageBox.information(self, tr("lib.title", "Drama Library"),
                                    tr("lib.no_episodes", "Select one or more episodes first."))
            return
        item = self.series_list.currentItem()
        series = item.text().rsplit("  (", 1)[0] if item else ""
        from app.ui.tabs.auto_dialog import AutoDialog
        dlg = AutoDialog(self.ctx, episodes, series, self)
        dlg.exec()
        self.refresh()

    def delete_episode(self, ep: dict) -> None:
        title = ep.get("title", "?")
        q = tr("lib.delete.confirm", "Delete episode '{t}' from the library? The video file is kept on disk.")
        if QMessageBox.question(self, tr("lib.delete", "Delete"),
                                q.replace("{t}", title)) != QMessageBox.Yes:
            return
        try:
            self.ctx.library.delete_episode(ep.get("id"))
        except Exception as exc:
            log.warning("delete_episode failed: %s", exc)
        self.refresh()

    def _series_menu(self, pos) -> None:
        item = self.series_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_del = menu.addAction(tr("lib.series.delete", "Delete series"))
        chosen = menu.exec(self.series_list.mapToGlobal(pos))
        if chosen is act_del:
            sid = item.data(Qt.UserRole)
            q = tr("lib.series.delete.confirm",
                   "Delete this series and all its episodes from the library? Files on disk are kept.")
            if QMessageBox.question(self, tr("lib.series.delete", "Delete series"),
                                    q) != QMessageBox.Yes:
                return
            try:
                self.ctx.library.delete_series(sid)
            except Exception as exc:
                log.warning("delete_series failed: %s", exc)
            self.refresh()
