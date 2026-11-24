"""Application paths. Everything user-writable lives under %LOCALAPPDATA%/RecapStudio."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "RecapStudio"


def app_root() -> Path:
    """Repo root in dev; the executable's directory when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # this file: <root>/app/core/paths.py
    return Path(__file__).resolve().parent.parent.parent


def _store_python_local_cache() -> Path | None:
    """Physical LocalCache dir when running under Microsoft Store Python.

    Store-packaged Python virtualizes writes to %LOCALAPPDATA% into
    %LOCALAPPDATA%/Packages/<family>/LocalCache/Local. Python itself sees the
    redirected files, but external processes (ffmpeg, demucs, yt-dlp) do not —
    so temp files handed to them "don't exist". Using the physical path
    directly is visible to everyone and contains the same data.

    A virtual environment made from Store Python (VS Code makes one in the
    project folder) reports its own python.exe, outside WindowsApps, but runs
    the Store interpreter (`sys._base_executable`) with the same redirection:
    a file it wrote was "No such file or directory" for ffmpeg (2026-11-16).
    It uses the same physical folder.
    """
    if getattr(sys, "frozen", False) or os.name != "nt":
        return None
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    for exe in (sys.executable, getattr(sys, "_base_executable", "") or ""):
        parts = Path(exe).parts
        if "WindowsApps" not in parts:
            continue
        for part in parts:
            if part.startswith("PythonSoftwareFoundation.Python."):
                cache = Path(base) / "Packages" / part / "LocalCache" / "Local"
                if cache.is_dir():
                    return cache
    return None


def app_data_dir() -> Path:
    cache = _store_python_local_cache()
    if cache is not None:
        p = cache / APP_NAME
    else:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        p = Path(base) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def bin_dir() -> Path:
    return app_root() / "bin"


def _bundled_or_system(exe: str) -> str:
    bundled = bin_dir() / exe
    if bundled.exists():
        return str(bundled)
    found = shutil.which(exe)
    return found or exe


def ffmpeg_path() -> str:
    return _bundled_or_system("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def ffprobe_path() -> str:
    return _bundled_or_system("ffprobe.exe" if os.name == "nt" else "ffprobe")


def pip_install(args: str) -> str:
    """What to type to install `args` into the Python this app runs on. A
    bare "pip install" can reach another Python, or none: a portable copy's
    Python is not on PATH. The exe cannot install packages; it keeps the
    short form."""
    if getattr(sys, "frozen", False):
        return f"pip install {args}"
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":       # no console: type it with python.exe
        exe = exe.with_name("python.exe")
    on_path = shutil.which("python")
    if on_path and os.path.normcase(os.path.abspath(on_path)) == os.path.normcase(str(exe)):
        return f"python -m pip install {args}"
    return f'"{exe}" -m pip install {args}'


def fonts_dir() -> Path:
    return app_root() / "app" / "assets" / "fonts"


def locales_dir() -> Path:
    return app_root() / "app" / "locales"


def projects_dir() -> Path:
    p = app_data_dir() / "projects"
    p.mkdir(parents=True, exist_ok=True)
    return p


def temp_dir() -> Path:
    p = app_data_dir() / "temp"
    p.mkdir(parents=True, exist_ok=True)
    return p
