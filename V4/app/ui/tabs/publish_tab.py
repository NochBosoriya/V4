"""Publish tab: AI metadata, then upload to YouTube and/or a Facebook Page.

Uploads never start without a confirmation dialog listing every destination
and its visibility. "When to post": Post immediately (YouTube with the
visibility chosen, Facebook published at once) or Post on schedule (uploaded
now; YouTube private until the time, then public; Facebook publishes it at
the time — the PC need not be on). Engine: app.core.publisher.
"""
from __future__ import annotations

import glob
import json
import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QDateTime, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QButtonGroup,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.core import publisher as pub
from app.core.i18n import tr
from app.core.logger import get_logger
from app.core.paths import projects_dir
from app.ui.langs import LANGS
from app.ui.widgets import ProgressRow, scroll_page
from app.ui.workers import FnWorker

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.publish_tab")

_YT_HELP = (
    "One-time setup (free):\n"
    "1. console.cloud.google.com → create a project → enable “YouTube Data API v3”.\n"
    "2. OAuth consent screen → External → add your Google account as a test user.\n"
    "3. Credentials → Create credentials → OAuth client ID → type “Desktop app”.\n"
    "4. Paste the client ID and secret here, then press Connect and sign in.")
_FB_HELP = (
    "Use a Page access token with pages_manage_posts + pages_read_engagement "
    "+ publish_video (Meta for Developers → Graph API Explorer → your app → "
    "Get Page Access Token). The Page ID is on the Page's About section.")


def _upload_fn(cfg, video, meta, yt, fb, thumb, lang, progress=None, cancel=None):
    links = []
    n = int(bool(yt)) + int(bool(fb))
    k = 0

    def sub(f, t):
        if progress:
            progress((k + f) / max(1, n), t)

    if yt:
        links.append(("YouTube", pub.YouTube(cfg).upload(
            video, meta, privacy=yt["privacy"], made_for_kids=yt["kids"],
            thumbnail=thumb, language=lang, progress=sub, cancel=cancel,
            publish_at=yt.get("at"))))
        k += 1
    if fb:
        links.append(("Facebook", pub.FacebookPage(cfg).upload(
            video, meta, publish_now=fb["publish_now"], thumbnail=thumb,
            progress=sub, cancel=cancel, scheduled_at=fb.get("at"))))
    return links


class _CredDialog(QDialog):
    def __init__(self, parent, title: str, help_text: str, fields: list[tuple[str, str, bool]]):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 0)
        lay = QVBoxLayout(self)
        h = QLabel(help_text)
        h.setWordWrap(True)
        h.setStyleSheet("color: #8b949e;")
        h.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(h)
        f = QFormLayout()
        self.edits = []
        for label, value, secret in fields:
            e = QLineEdit(value)
            if secret:
                e.setEchoMode(QLineEdit.EchoMode.Password)
            f.addRow(label, e)
            self.edits.append(e)
        lay.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self) -> list[str]:
        return [e.text().strip() for e in self.edits]


class PublishTab(QWidget):
    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self.yt = pub.YouTube(ctx.config)
        self.fb = pub.FacebookPage(ctx.config)
        self._workers: list = []

        root = scroll_page(self)          # a short window scrolls, nothing overlaps
        root.setContentsMargins(16, 16, 16, 16)
        files = QGroupBox(tr("pub.files", "What to publish"))
        ff = QFormLayout(files)
        self.video_edit = QLineEdit()
        ff.addRow(tr("pub.video", "Video:"), self._file_row(self.video_edit, self._pick_video))
        self.thumb_edit = QLineEdit()
        self.thumb_edit.setPlaceholderText(tr("pub.thumb.ph", "Optional .jpg (Overlay → Thumbnail)"))
        self.thumb_edit.textChanged.connect(self._show_thumb)
        ff.addRow(tr("pub.thumb", "Thumbnail:"), self._file_row(self.thumb_edit, self._pick_thumb))
        self.thumb_view = QLabel()
        self.thumb_view.setFixedHeight(90)
        ff.addRow("", self.thumb_view)
        root.addWidget(files)

        meta = QGroupBox(tr("pub.meta", "Title, description and tags"))
        mf = QFormLayout(meta)
        mrow = QHBoxLayout()
        self.lang_combo = QComboBox()
        for code, native in LANGS:
            self.lang_combo.addItem(native, code)
        mrow.addWidget(self.lang_combo)
        self.ai_btn = QPushButton(tr("pub.ai", "AI write it"))
        self.ai_btn.setProperty("class", "accent")
        self.ai_btn.clicked.connect(self._on_ai)
        mrow.addWidget(self.ai_btn)
        mrow.addStretch(1)
        mh = QWidget()
        mh.setLayout(mrow)
        mf.addRow(tr("pub.lang", "Language:"), mh)
        self.context_edit = QPlainTextEdit()
        self.context_edit.setPlaceholderText(tr(
            "pub.context.ph", "Story context for the AI (filled from the Full Dub project when found)"))
        self.context_edit.setMaximumHeight(70)
        mf.addRow(tr("pub.context", "Context:"), self.context_edit)
        self.title_edit = QLineEdit()
        self.title_edit.setMaxLength(100)
        mf.addRow(tr("pub.title", "Title:"), self.title_edit)
        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setMinimumHeight(110)
        mf.addRow(tr("pub.desc", "Description:"), self.desc_edit)
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("tag one, tag two, …")
        mf.addRow(tr("pub.tags", "Tags:"), self.tags_edit)
        self.hash_edit = QLineEdit()
        self.hash_edit.setPlaceholderText("#khmerdrama #recap")
        mf.addRow(tr("pub.hashtags", "Hashtags:"), self.hash_edit)
        root.addWidget(meta, 1)

        when = QGroupBox(tr("pub.when", "When to post"))
        wl = QHBoxLayout(when)
        self.post_now = QRadioButton(tr("pub.when.now", "Post immediately"))
        self.post_later = QRadioButton(tr("pub.when.later", "Post on schedule:"))
        self.post_now.setChecked(True)
        self._when_group = QButtonGroup(self)
        for rb in (self.post_now, self.post_later):
            self._when_group.addButton(rb)
            # the theme's round button is wider than Qt counts: the words were cut
            rb.setMinimumWidth(rb.sizeHint().width() + 16)
            wl.addWidget(rb)
        self.post_at = QDateTimeEdit()
        self.post_at.setCalendarPopup(True)
        self.post_at.setDisplayFormat("yyyy-MM-dd  HH:mm")
        self.post_at.setDateTime(self._next_slot())
        self.post_at.setToolTip(tr("pub.when.tip", "Your PC's time. The video uploads now; "
                                                   "YouTube and Facebook post it at this time "
                                                   "(your PC can be off)."))
        wl.addWidget(self.post_at)
        wl.addStretch(1)
        self.post_now.toggled.connect(self._on_when)
        root.addWidget(when)

        dest = QHBoxLayout()
        ybox = QGroupBox("YouTube")
        yf = QFormLayout(ybox)
        self.yt_check = QCheckBox(tr("pub.yt.use", "Upload to YouTube"))
        yf.addRow(self.yt_check)
        self.yt_status = QLabel()
        yf.addRow(tr("pub.account", "Account:"), self.yt_status)
        self.yt_privacy = QComboBox()
        for p, label in (("private", "Private"), ("unlisted", "Unlisted"), ("public", "Public")):
            self.yt_privacy.addItem(tr(f"pub.privacy.{p}", label), p)
        yf.addRow(tr("pub.privacy", "Visibility:"), self.yt_privacy)
        self.yt_kids = QCheckBox(tr("pub.kids", "Made for kids"))
        yf.addRow("", self.yt_kids)
        yb = QHBoxLayout()
        for text, slot in ((tr("pub.yt.client", "OAuth client…"), self._yt_client),
                           (tr("pub.connect", "Connect"), self._yt_connect),
                           (tr("pub.disconnect", "Disconnect"), self._yt_disconnect)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            yb.addWidget(b)
        yw = QWidget()
        yw.setLayout(yb)
        yf.addRow(yw)
        dest.addWidget(ybox)

        fbox = QGroupBox(tr("pub.fb", "Facebook Page"))
        fbf = QFormLayout(fbox)
        self.fb_check = QCheckBox(tr("pub.fb.use", "Upload to the Facebook Page"))
        fbf.addRow(self.fb_check)
        self.fb_status = QLabel()
        fbf.addRow(tr("pub.page", "Page:"), self.fb_status)
        fb_btn = QPushButton(tr("pub.fb.set", "Page ID && token…"))
        fb_btn.clicked.connect(self._fb_set)
        fbf.addRow(fb_btn)
        dest.addWidget(fbox)
        root.addLayout(dest)

        act = QHBoxLayout()
        self.upload_btn = QPushButton(tr("pub.upload", "Upload…"))
        self.upload_btn.setProperty("class", "accent")
        self.upload_btn.clicked.connect(self._on_upload)
        act.addWidget(self.upload_btn)
        act.addStretch(1)
        root.addLayout(act)
        self.progress_row = ProgressRow(tr("pub.working", "Publishing"))
        self.progress_row.hide()
        root.addWidget(self.progress_row)
        self.result_label = QLabel("")
        self.result_label.setOpenExternalLinks(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        root.addWidget(self.result_label)
        self._on_when()
        self._refresh_accounts()

    @staticmethod
    def _next_slot() -> QDateTime:
        """The next whole hour at least 30 minutes away."""
        t = QDateTime.currentDateTime().addSecs(30 * 60)
        t = t.addSecs(3600 - t.time().minute() * 60 - t.time().second())
        return QDateTime(t.date(), t.time().addMSecs(-t.time().msec()))

    def _on_when(self, *_a) -> None:
        later = self.post_later.isChecked()
        self.post_at.setEnabled(later)
        # a scheduled YouTube video is private until the time, then public
        self.yt_privacy.setEnabled(not later)
        self.yt_privacy.setToolTip(tr("pub.when.yt_tip", "Scheduled: private until the time, "
                                                         "then public.") if later else "")

    # ------------------------------------------------------------------
    def _file_row(self, edit: QLineEdit, slot) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        b = QPushButton(tr("common.browse", "Browse…"))
        b.clicked.connect(slot)
        h.addWidget(b)
        return w

    def _start(self, worker: FnWorker) -> None:
        self._workers = [w for w in self._workers if w.isRunning()]
        self._workers.append(worker)
        self.progress_row.set_worker(worker)
        self.ctx.pool.start(worker)

    def showEvent(self, ev):  # noqa: N802 (Qt override)
        if not self.video_edit.text() and self.ctx.current_video:
            self.set_video(self.ctx.current_video)
        super().showEvent(ev)

    def set_video(self, path: str) -> None:
        self.video_edit.setText(path)
        stem = os.path.splitext(path)[0]
        for cand in (stem + ".thumb.jpg", stem + ".jpg"):
            if os.path.isfile(cand) and not self.thumb_edit.text():
                self.thumb_edit.setText(cand)
        if not self.context_edit.toPlainText().strip():
            ctx = self._project_context(path)
            if ctx:
                self.context_edit.setPlainText(ctx)

    def _project_context(self, video: str) -> str:
        """Story context from the Full Dub project that produced (or uses) this video."""
        target = os.path.normcase(os.path.abspath(video))
        for pj in glob.glob(str(projects_dir() / "dub" / "*" / "project.json")):
            try:
                d = json.load(open(pj, encoding="utf-8"))
            except Exception:
                continue
            paths = {os.path.normcase(os.path.abspath(x)) for x in
                     (d.get("output") or "", d.get("video") or "") if x}
            if target in paths:
                lines = " ".join(l.get("text", "") for l in (d.get("lines") or [])[:60])
                return ((d.get("story_context") or "") + "\n" + lines).strip()
        return ""

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("pub.video", "Video:"), "",
                                              "Video (*.mp4 *.mov *.mkv);;All files (*)")
        if path:
            self.set_video(path)

    def _pick_thumb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("pub.thumb", "Thumbnail:"), "",
                                              "Images (*.jpg *.jpeg *.png)")
        if path:
            self.thumb_edit.setText(path)

    def _show_thumb(self, path: str) -> None:
        pm = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        self.thumb_view.setPixmap(pm.scaledToHeight(90, Qt.TransformationMode.SmoothTransformation)
                                  if not pm.isNull() else QPixmap())

    # accounts -----------------------------------------------------------
    def _refresh_accounts(self) -> None:
        if self.yt.connected():
            self.yt_status.setText("✔ " + (self.yt.channel_name() or "connected"))
        elif self.yt.client()[0]:
            self.yt_status.setText(tr("pub.yt.not_connected", "Client set — press Connect"))
        else:
            self.yt_status.setText(tr("pub.yt.no_client", "Not set up — press “OAuth client…”"))
        self.yt_check.setEnabled(self.yt.connected())
        if not self.yt.connected():
            self.yt_check.setChecked(False)
        pid, _tok = self.fb.page()
        self.fb_status.setText(("✔ " + pid) if self.fb.connected()
                               else tr("pub.fb.none", "Not set up"))
        self.fb_check.setEnabled(self.fb.connected())
        if not self.fb.connected():
            self.fb_check.setChecked(False)

    def _yt_client(self) -> None:
        cid, sec = self.yt.client()
        dlg = _CredDialog(self, "YouTube OAuth client", _YT_HELP,
                          [("Client ID", cid, False), ("Client secret", sec, True)])
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cid, sec = dlg.values()
            self.yt.set_client(cid, sec)
            self._refresh_accounts()

    def _yt_connect(self) -> None:
        w = FnWorker(self.yt.connect)
        w.finished_ok.connect(lambda _n: self._refresh_accounts())
        self._start(w)

    def _yt_disconnect(self) -> None:
        self.yt.disconnect()
        self._refresh_accounts()

    def _fb_set(self) -> None:
        pid, tok = self.fb.page()
        dlg = _CredDialog(self, "Facebook Page", _FB_HELP,
                          [("Page ID", pid, False), ("Page access token", tok, True)])
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        pid, tok = dlg.values()
        self.fb.set_page(pid, tok)
        w = FnWorker(self.fb.page_name)
        w.finished_ok.connect(lambda name: (self._refresh_accounts(),
                                            self.fb_status.setText(f"✔ {name}")))
        w.failed.connect(lambda en, km: QMessageBox.warning(self, "Facebook", en))
        self._start(w)

    # metadata / upload ----------------------------------------------------
    def _meta(self) -> pub.Metadata:
        return pub.Metadata(
            title=self.title_edit.text().strip(),
            description=self.desc_edit.toPlainText().strip(),
            tags=[t.strip() for t in self.tags_edit.text().split(",") if t.strip()],
            hashtags=[h for h in self.hash_edit.text().split() if h.strip()])

    def _on_ai(self) -> None:
        video = self.video_edit.text().strip()
        context = self.context_edit.toPlainText().strip() or os.path.basename(video)
        w = FnWorker(pub.generate_metadata, self.ctx.ai, context,
                     self.lang_combo.currentData(), "YouTube",
                     self.title_edit.text().strip() or os.path.splitext(os.path.basename(video))[0])
        w.finished_ok.connect(self._fill_meta)
        self._start(w)

    def _fill_meta(self, m: pub.Metadata) -> None:
        self.title_edit.setText(m.title)
        self.desc_edit.setPlainText(m.description)
        self.tags_edit.setText(", ".join(m.tags))
        self.hash_edit.setText(" ".join(m.hashtags))

    def _on_upload(self) -> None:
        video = self.video_edit.text().strip()
        meta = self._meta()
        if not os.path.isfile(video):
            QMessageBox.information(self, tr("pub.title_box", "Publish"),
                                    tr("pub.no_video", "Choose the video file first."))
            return
        if not meta.title:
            QMessageBox.information(self, tr("pub.title_box", "Publish"),
                                    tr("pub.no_title", "Write a title (or press “AI write it”)."))
            return
        at = None
        if self.post_later.isChecked():
            at = self.post_at.dateTime().toPython().astimezone()     # the PC's local time
            why = pub.schedule_problem(at, facebook=self.fb_check.isChecked())
            if why:
                QMessageBox.information(
                    self, tr("pub.title_box", "Publish"),
                    tr("pub.when.too_far", "Facebook can schedule a post at most 6 months "
                                           "ahead. Pick an earlier time.")
                    if why == "too_far" else
                    tr("pub.when.too_soon", "Pick a time at least 15 minutes from now."))
                return
        yt = ({"privacy": self.yt_privacy.currentData(), "kids": self.yt_kids.isChecked(),
               "at": at} if self.yt_check.isChecked() else None)
        fb = ({"publish_now": at is None, "at": at} if self.fb_check.isChecked() else None)
        if not (yt or fb):
            QMessageBox.information(self, tr("pub.title_box", "Publish"),
                                    tr("pub.no_dest", "Tick YouTube and/or Facebook first "
                                                      "(connect the account if it is greyed out)."))
            return
        size_mb = os.path.getsize(video) / 1024 ** 2
        dests = []
        when_txt = (at.strftime("%Y-%m-%d %H:%M") if at else "")
        if yt:
            dests.append(f"YouTube ({self.yt.channel_name() or 'channel'}) — " + (
                tr("pub.when.yt_at", "private, public at {time}").format(time=when_txt) if at
                else self.yt_privacy.currentText()))
        if fb:
            dests.append(f"Facebook Page {self.fb.page()[0]} — " + (
                tr("pub.when.fb_at", "posted at {time}").format(time=when_txt) if at
                else tr("pub.when.fb_now", "posted now")))
        ans = QMessageBox.question(
            self, tr("pub.confirm", "Confirm upload"),
            tr("pub.confirm_q", "Upload this video?") +
            f"\n\n{os.path.basename(video)} ({size_mb:,.0f} MB)\n“{meta.title}”\n\n"
            + "\n".join(f"→ {d}" for d in dests))
        if ans != QMessageBox.StandardButton.Yes:
            return
        w = FnWorker(_upload_fn, self.ctx.config, video, meta, yt, fb,
                     self.thumb_edit.text().strip(), self.lang_combo.currentData())
        w.finished_ok.connect(self._uploaded)
        self._start(w)

    def _uploaded(self, links) -> None:
        html = "<br>".join(f"✔ {name}: <a href='{url}'>{url}</a>" for name, url in links or [])
        self.result_label.setText(html)
