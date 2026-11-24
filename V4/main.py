"""RecapStudio entry point."""
import os
import sys


def _no_console_streams() -> None:
    """Started without a console (the exe, or pythonw from a start file),
    sys.stdout / sys.stderr are None and every progress bar (tqdm: the
    Whisper model download) fails on .write: give them somewhere to go."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


if __name__ == "__main__":
    _no_console_streams()
    from app.ui_main import run
    sys.exit(run())
