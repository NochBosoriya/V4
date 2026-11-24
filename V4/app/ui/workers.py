"""Qt worker threads wrapping the blocking engine functions.

FnWorker runs a plain function on a QThread, injecting ``progress=`` and
``cancel=`` keyword arguments only when the target function accepts them
(detected via inspect.signature). Results/errors are delivered through
signals so the UI thread stays responsive.
"""
from __future__ import annotations

import inspect
import threading

from PySide6.QtCore import QThread, Signal

from app.core.errors import AppError, CancelledError
from app.core.logger import get_logger

log = get_logger("ui.workers")


def _accepts_kwarg(fn, name: str) -> bool:
    """True if fn accepts keyword argument `name` (explicitly or via **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        p = params[name]
        return p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


class FnWorker(QThread):
    """Run ``fn(*args, **kwargs)`` on a thread with progress/cancel plumbing.

    Signals:
        progress(float, str)  -- fraction 0..1 and status text
        finished_ok(object)   -- the function's return value
        failed(str, str)      -- (english, khmer) error messages
        cancelled()           -- the operation raised CancelledError
    """

    progress = Signal(float, str)
    finished_ok = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = dict(kwargs)
        self._cancel_event = threading.Event()

        if _accepts_kwarg(fn, "progress") and "progress" not in self._kwargs:
            self._kwargs["progress"] = self._emit_progress
        if _accepts_kwarg(fn, "cancel") and "cancel" not in self._kwargs:
            self._kwargs["cancel"] = self._cancel_event

    # ------------------------------------------------------------------
    def _emit_progress(self, fraction: float, text: str = "") -> None:
        """Called from the worker thread; Signal.emit is thread-safe in Qt."""
        try:
            f = float(fraction)
        except (TypeError, ValueError):
            f = 0.0
        f = max(0.0, min(1.0, f))
        self.progress.emit(f, str(text))

    def cancel(self) -> None:
        """Request cancellation. The wrapped function polls the event."""
        self._cancel_event.set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    # ------------------------------------------------------------------
    def run(self) -> None:  # executed on the worker thread
        try:
            result = self._fn(*self._args, **self._kwargs)
        except CancelledError:
            log.info("worker cancelled: %r", getattr(self._fn, "__name__", self._fn))
            self.cancelled.emit()
        except AppError as e:
            log.error("worker failed: %s", e.en)
            self.failed.emit(e.en, e.km)
        except Exception as e:  # noqa: BLE001 - report anything to the UI
            log.exception("worker crashed: %r", e)
            self.failed.emit(repr(e), "")
        else:
            self.finished_ok.emit(result)


class WorkerPool:
    """Keeps strong references to running FnWorkers so Qt/GC never drops them."""

    def __init__(self):
        self._workers: list[FnWorker] = []
        self._lock = threading.Lock()

    def start(self, worker: FnWorker) -> FnWorker:
        """Register and start a worker; it is auto-removed when finished."""
        self.add(worker)
        worker.start()
        return worker

    def add(self, worker: FnWorker) -> None:
        with self._lock:
            if worker not in self._workers:
                self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._remove(w))

    def _remove(self, worker: FnWorker) -> None:
        with self._lock:
            try:
                self._workers.remove(worker)
            except ValueError:
                pass

    def cancel_all(self) -> None:
        with self._lock:
            workers = list(self._workers)
        for w in workers:
            try:
                w.cancel()
            except Exception:
                pass

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for w in self._workers if w.isRunning())

    def wait_all(self, timeout_ms: int = 5000) -> None:
        """Best-effort wait for all workers (used at shutdown)."""
        with self._lock:
            workers = list(self._workers)
        for w in workers:
            try:
                w.wait(timeout_ms)
            except Exception:
                pass
