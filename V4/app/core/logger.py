"""Rotating file logging + a thread-safe log bus feeding the UI console dock."""
from __future__ import annotations

import logging
import logging.handlers
import threading
from collections import deque
from typing import Callable

from app.core.paths import app_data_dir

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_lock = threading.Lock()
_configured = False


class LogBus:
    """Thread-safe pub/sub of log lines for the UI console."""

    def __init__(self, maxlen: int = 5000):
        self._subs: list[Callable[[str], None]] = []
        self._buffer: deque[str] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[str], None]) -> None:
        with self._lock:
            if fn not in self._subs:
                self._subs.append(fn)

    def unsubscribe(self, fn: Callable[[str], None]) -> None:
        with self._lock:
            if fn in self._subs:
                self._subs.remove(fn)

    def publish(self, line: str) -> None:
        line = line.rstrip("\r\n")
        if not line:
            return
        with self._lock:
            self._buffer.append(line)
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(line)
            except Exception:
                pass

    def tail(self, n: int = 500) -> list[str]:
        with self._lock:
            return list(self._buffer)[-n:]


LOG_BUS = LogBus()


class _BusHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            LOG_BUS.publish(self.format(record))
        except Exception:
            pass


def _self_test_log_dir():
    """A module's self-test (`python -m app.engines.dub_engine`) logs to a
    temp folder: its fake keys and failures ("Key marked invalid", "pop from
    empty list") used to land in the user's app.log among real runs."""
    import sys
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if str(getattr(spec, "name", "") or "").startswith("app."):
        import tempfile
        from pathlib import Path
        return Path(tempfile.gettempdir()) / "RecapStudio-selftest"
    return None


def _configure_root() -> None:
    global _configured
    with _lock:
        if _configured:
            return
        root = logging.getLogger("recapstudio")
        root.setLevel(logging.DEBUG)
        root.propagate = False

        log_dir = _self_test_log_dir() or app_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_dir / "app.log", maxBytes=5 * 1024 * 1024,
                backupCount=5, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            fh.setLevel(logging.DEBUG)
            root.addHandler(fh)
        except Exception:
            pass

        bh = _BusHandler()
        bh.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        bh.setLevel(logging.INFO)
        root.addHandler(bh)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"recapstudio.{name}")
