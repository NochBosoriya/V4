"""Application bootstrap: QApplication, theme, singletons, main window,
and the non-blocking startup environment check with model downloads."""
from __future__ import annotations

import os
import sys

# OpenCV must not start OpenCL (no part of the app uses it — all OpenCV work is
# on numpy arrays): once the startup probe has initialised CUDA in one worker
# thread and OpenCV loads the NVIDIA OpenCL driver in another (scene
# detection, thumbnails, the subtitle reader…), the process hangs in the driver when it exits
# and stays in memory (~1 GB) until a reboot. Measured 2026-11-16: same run
# with OpenCL off exits at once. Set before anything imports cv2.
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

from app.core.i18n import set_language, tr  # noqa: E402
from app.core.logger import get_logger  # noqa: E402

log = get_logger("ui_main")


def _build_context():
    """Create the shared singletons and wrap them in an AppContext."""
    from app.core.ai_engine import AIEngine
    from app.core.config_manager import get_default_config
    from app.core.downloader import DownloadQueue, Library
    from app.ui.context import AppContext
    from app.ui.workers import WorkerPool

    config = get_default_config()
    set_language(config.get("ui.language", "en"))

    library = Library()
    downloads = DownloadQueue(library, out_dir=config.get("downloads.dir") or None)
    downloads.set_options(quality=config.get("downloads.quality") or "best",
                          cookies=config.get("downloads.cookies") or "")
    ai = AIEngine(config)
    pool = WorkerPool()
    return AppContext(config=config, ai=ai, library=library,
                      downloads=downloads, pool=pool)


def _register_bundled_fonts() -> None:
    """Make app/assets/fonts usable by Qt (theme font, font pickers) without
    installing them into Windows."""
    from PySide6.QtGui import QFontDatabase

    from app.core.paths import fonts_dir

    d = fonts_dir()
    if not d.is_dir():
        return
    loaded = 0
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
            if QFontDatabase.addApplicationFont(str(p)) >= 0:
                loaded += 1
            else:
                log.warning("Qt could not load font %s", p.name)
    log.info("registered %d bundled font(s)", loaded)


_DISMISSED_KEY = "startup.dismissed_models"
# package a model needs before downloading its weights is any use
_MODEL_NEEDS = {"whisper-small": "faster_whisper", "demucs-htdemucs": "demucs",
                "lama": "onnxruntime", "paddleocr-ch": "paddleocr",
                "mdx-vocals": "onnxruntime"}


def _installed(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _models_worth_offering(missing: list[str], config) -> list[str]:
    """The missing models this install can actually use: one Whisper size
    ("small", the default everywhere) and only while no Whisper model is
    present; Demucs / LaMa / PaddleOCR only when the package that runs them
    is installed (the .exe build has no Demucs or PaddleOCR); none the user
    said not to ask about again."""
    from app.core.model_manager import MODELS

    dismissed = set(config.get(_DISMISSED_KEY) or [])
    any_whisper = not all(m in missing for m in MODELS if m.startswith("whisper-"))
    out = []
    for mid in missing:
        if mid in dismissed:
            continue
        if mid.startswith("whisper-") and (mid != "whisper-small" or any_whisper):
            continue
        need = _MODEL_NEEDS.get(mid)
        if need and not _installed(need):
            continue
        if mid == "mdx-vocals" and _installed("demucs"):
            continue        # Demucs separates voices here already
        out.append(mid)
    return out


def _show_startup_check_dialog(report: dict, ctx, parent) -> None:
    """Non-blocking dialog listing missing models / ffmpeg with per-item
    'Download now' handling via FnWorker + ProgressRow, or 'Later'."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    )

    from app.core.i18n import get_language
    from app.ui.widgets import ProgressRow
    from app.ui.workers import FnWorker

    missing = _models_worth_offering(list(report.get("missing_models") or []),
                                     ctx.config)
    ffmpeg_ok = bool(report.get("ffmpeg_ok", True))
    if ffmpeg_ok and not missing:
        return

    dlg = QDialog(parent)
    dlg.setWindowTitle(tr("startup.check_title", "Environment check"))
    dlg.setModal(False)
    dlg.setMinimumWidth(560)
    lay = QVBoxLayout(dlg)
    lay.setSpacing(8)

    header = QLabel(tr("startup.missing_header",
                       "Some components are missing:"), dlg)
    header.setStyleSheet("font-weight: 600; font-size: 11pt;")
    lay.addWidget(header)

    if not ffmpeg_ok:
        text = tr("startup.no_ffmpeg",
                  "ffmpeg was not found. Place ffmpeg.exe in the 'bin' folder "
                  "next to the app, or install it and add it to PATH.")
        if get_language() != "km":   # in Khmer the line above is already Khmer
            text += "\n" + tr("startup.no_ffmpeg_km",
                              "រកមិនឃើញ ffmpeg ទេ។ "
                              "សូមដាក់ ffmpeg.exe "
                              "ក្នុងថត bin ក្បែរកម្មវិធី "
                              "ឬដំឡើងវាហើយបញ្ចូល"
                              "ទៅក្នុង PATH។")
        ff = QLabel(text, dlg)
        ff.setWordWrap(True)
        ff.setStyleSheet("color: #f8b849;")
        lay.addWidget(ff)

    # one row per missing model
    rows: list[tuple[str, ProgressRow, QPushButton]] = []
    for model_id in missing:
        row_widget = QHBoxLayout()
        prow = ProgressRow(model_id, dlg, hide_on_done=False)
        btn = QPushButton(tr("startup.download_now", "Download now"), dlg)
        btn.setProperty("class", "accent")
        btn.setMaximumWidth(140)
        row_widget.addWidget(prow, 1)
        row_widget.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(row_widget)
        rows.append((model_id, prow, btn))

    def make_start(model_id: str, prow: ProgressRow, btn: QPushButton):
        def start() -> None:
            btn.setEnabled(False)

            def download(progress=None, cancel=None):
                from app.core.model_manager import download_model
                download_model(model_id, progress=progress, cancel=cancel)
                return model_id

            worker = FnWorker(download)
            prow.set_worker(worker)
            worker.finished_ok.connect(
                lambda _r: prow.set_label(model_id + "  ✓"))
            worker.failed.connect(lambda *_: btn.setEnabled(True))
            worker.cancelled.connect(lambda: btn.setEnabled(True))
            ctx.pool.start(worker)
        return start

    for model_id, prow, btn in rows:
        btn.clicked.connect(make_start(model_id, prow, btn))

    footer = QHBoxLayout()
    dont_ask = QCheckBox(tr("startup.dont_ask", "Don't ask about these models again"), dlg)
    footer.addWidget(dont_ask)
    footer.addStretch(1)
    dl_all = QPushButton(tr("startup.download_all", "Download all"), dlg)
    dl_all.setProperty("class", "accent")
    later = QPushButton(tr("startup.later", "Later"), dlg)
    footer.addWidget(dl_all)
    footer.addWidget(later)
    lay.addLayout(footer)

    def on_download_all() -> None:
        for _mid, _prow, btn in rows:
            if btn.isEnabled():
                btn.click()
    dl_all.clicked.connect(on_download_all)
    if not rows:
        dl_all.hide()
        dont_ask.hide()
    later.clicked.connect(dlg.close)

    def on_finished(_result: int) -> None:
        if dont_ask.isChecked():
            old = set(ctx.config.get(_DISMISSED_KEY) or [])
            ctx.config.set(_DISMISSED_KEY, sorted(old | set(missing)))
    dlg.finished.connect(on_finished)

    # The dialog pops up (and takes focus) while the user may be typing
    # elsewhere: a stray Enter/Space must not start gigabytes of downloads.
    for b in [dl_all, later] + [r[2] for r in rows]:
        b.setAutoDefault(False)
    dont_ask.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    later.setDefault(True)

    dlg.show()
    later.setFocus()
    # keep a reference on the parent so the dialog is not GC'd
    parent._startup_dialog = dlg


def run() -> int:
    """RecapStudio entry point. Returns the process exit code."""
    from PySide6.QtWidgets import QApplication

    from app.ui import theme

    from app.version import BUILD_DATE, __version__

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("RecapStudio")
    app.setOrganizationName("RecapStudio")
    app.setApplicationVersion(__version__)
    log.info("RecapStudio %s (%s), Python %s", __version__, BUILD_DATE,
             sys.version.split()[0])
    _register_bundled_fonts()
    theme.apply(app)

    try:
        ctx = _build_context()
    except Exception as e:
        # Anything failing this early deserves a visible message, not a crash.
        from PySide6.QtWidgets import QMessageBox
        log.exception("startup failed: %r", e)
        QMessageBox.critical(
            None, "RecapStudio",
            tr("startup.fatal", "RecapStudio failed to start:") + f"\n{e!r}")
        return 1

    from app.ui.main_window import MainWindow

    win = MainWindow(ctx)
    win.show()

    # The main window probes the environment in the background (never on
    # the UI thread); show the dialog only if something is missing.
    def on_report(rep: dict) -> None:
        try:
            _show_startup_check_dialog(rep, ctx, win)
        except Exception as e:
            log.error("startup check dialog failed: %r", e)

    win.when_report(on_report)

    rc = app.exec()
    _exit_if_threads_linger(ctx, rc)
    return rc


def _exit_if_threads_linger(ctx, rc: int) -> None:
    """End the process now if work is still running after the window closed.

    A task that cannot stop mid-call (Whisper inside CTranslate2, an HTTP
    read) or a download still unwinding would otherwise either keep
    RecapStudio.exe alive with no window, or abort with "QThread: Destroyed
    while thread is still running" when Python tears the workers down.
    Everything is already saved (SQLite commits per change; projects and
    .part files are written as they go), so only the log needs flushing.
    """
    import logging
    import os
    import threading

    busy = ctx.pool.active_count() > 0 or any(
        t.is_alive() and t.name.startswith("dl") for t in threading.enumerate())
    if not busy:
        return
    log.info("exiting with background work still running")
    logging.shutdown()
    os._exit(rc)
