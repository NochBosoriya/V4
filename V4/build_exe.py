"""Build dist/RecapStudio/ with PyInstaller (one-dir) from recapstudio.spec.

    python -m pip install pyinstaller==6.22.3
    python fetch_assets.py      # once: fonts + ffmpeg
    python build_exe.py

Model weights are not bundled — the first-run downloader fetches them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_assets() -> list[str]:
    problems = []
    for exe in ("ffmpeg.exe", "ffprobe.exe"):
        if not (ROOT / "bin" / exe).is_file():
            problems.append(f"bin/{exe} missing")
    fonts = ROOT / "app" / "assets" / "fonts"
    if not any(fonts.glob("*.ttf")):
        problems.append("no fonts in app/assets/fonts")
    km = ROOT / "app" / "locales" / "km.json"
    if km.is_file() and km.stat().st_size < 100:
        problems.append("app/locales/km.json is empty")
    return problems


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed:  python -m pip install pyinstaller==6.22.3")
        return 1
    problems = check_assets()
    if problems:
        print("Missing assets — run  python fetch_assets.py  first:")
        for p in problems:
            print("  -", p)
        return 1
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "pyinstaller"),
           str(ROOT / "recapstudio.spec")]
    print(" ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc == 0:
        print(f"\nBuilt: {ROOT / 'dist' / 'RecapStudio' / 'RecapStudio.exe'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
