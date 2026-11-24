"""Shared application context passed to every tab and worker owner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.logger import get_logger

if TYPE_CHECKING:
    from app.core.ai_engine import AIEngine
    from app.core.config_manager import ConfigManager
    from app.core.downloader import DownloadQueue, Library
    from app.ui.main_window import MainWindow
    from app.ui.workers import WorkerPool

log = get_logger("ui.context")


@dataclass
class AppContext:
    """Singletons and cross-tab state shared by the whole UI."""

    config: "ConfigManager"
    ai: "AIEngine"
    library: "Library"
    downloads: "DownloadQueue"
    pool: "WorkerPool"
    main: "MainWindow | None" = None
    current_video: str | None = None

    def open_in_preview(self, path: str) -> None:
        """Route a media file into the main window's preview player."""
        self.current_video = path
        if self.main is None:
            log.warning("open_in_preview: no main window yet (%s)", path)
            return
        preview = getattr(self.main, "preview", None)
        if preview is None:
            log.warning("open_in_preview: main window has no preview widget")
            return
        try:
            preview.open(path)
        except Exception as e:  # never let a preview problem crash the caller
            log.error("open_in_preview failed: %r", e)
