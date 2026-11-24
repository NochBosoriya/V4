"""Fetch the files RecapStudio ships next to its code but does not keep in
source: Khmer fonts (app/assets/fonts) and ffmpeg/ffprobe (bin/).

    python fetch_assets.py                    # fonts + copy ffmpeg from PATH
    python fetch_assets.py --download-ffmpeg  # fonts + ffmpeg from gyan.dev
    python fetch_assets.py --ffmpeg-from C:\\ffmpeg\\bin
    python fetch_assets.py --fonts-only

Fonts come from the google/fonts repository (all SIL Open Font License);
each family's OFL.txt is saved under app/assets/licenses/fonts/. Font files
go flat into app/assets/fonts, and nothing else does, because the
poster/thumbnail font lookup and ffmpeg's ``fontsdir`` load every entry of
that one folder.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
FONTS_DIR = ROOT / "app" / "assets" / "fonts"
FONT_LICENSES_DIR = ROOT / "app" / "assets" / "licenses" / "fonts"
BIN_DIR = ROOT / "bin"

# google/fonts directory -> why the app needs it
FONT_FAMILIES = {
    "kantumruypro": "UI, subtitles and clean thumbnails (default subtitle font)",
    "moul": "poster and thumbnail titles (default poster font)",
    "battambang": "'Movie' subtitle style, neon thumbnails",
    "hanuman": "subtitle font option",
    "koulen": "bold display titles",
    "notosanskhmer": "fallback that covers every Khmer character",
}
GITHUB_API = "https://api.github.com/repos/google/fonts/contents/ofl/{family}"

FFMPEG_ZIP = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_EXES = ("ffmpeg.exe", "ffprobe.exe")

_session = requests.Session()
_session.headers["User-Agent"] = "RecapStudio-fetch-assets"


def _download(url: str, timeout: int = 60) -> bytes:
    r = _session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def fetch_fonts(force: bool = False) -> int:
    """Download every .ttf of FONT_FAMILIES. Returns the number of new files."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    licenses = FONT_LICENSES_DIR
    licenses.mkdir(parents=True, exist_ok=True)
    added = 0
    for family, purpose in FONT_FAMILIES.items():
        print(f"[fonts] {family}: {purpose}")
        r = _session.get(GITHUB_API.format(family=family), timeout=30)
        if r.status_code == 403:
            raise SystemExit("GitHub API rate limit hit (60 requests/hour "
                             "without a token). Try again later.")
        r.raise_for_status()
        for entry in r.json():
            name = entry.get("name", "")
            url = entry.get("download_url")
            if not url:
                continue
            if name.lower().endswith((".ttf", ".otf")):
                out = FONTS_DIR / name
            elif name == "OFL.txt":
                out = licenses / f"{family}-OFL.txt"
            else:
                continue
            if out.exists() and not force:
                print(f"         = {name} (already there)")
                continue
            out.write_bytes(_download(url))
            added += 1
            print(f"         + {out.relative_to(ROOT)}")
    return added


def _find_system_ffmpeg() -> Path | None:
    found = shutil.which("ffmpeg")
    if not found:
        return None
    folder = Path(found).resolve().parent
    if folder == BIN_DIR.resolve():
        return None
    return folder


def copy_ffmpeg(src_dir: Path, force: bool = False) -> None:
    """Copy ffmpeg.exe/ffprobe.exe (and the build's LICENSE) into bin/."""
    BIN_DIR.mkdir(exist_ok=True)
    for exe in FFMPEG_EXES:
        src = src_dir / exe
        if not src.is_file():
            raise SystemExit(f"{src} not found.")
        dst = BIN_DIR / exe
        if dst.exists() and not force:
            print(f"[ffmpeg] = bin/{exe} (already there)")
            continue
        shutil.copy2(src, dst)
        print(f"[ffmpeg] + bin/{exe}  ({dst.stat().st_size / 1e6:.0f} MB) from {src_dir}")
    for lic in (src_dir / "LICENSE", src_dir.parent / "LICENSE"):
        if lic.is_file():
            shutil.copy2(lic, BIN_DIR / "FFMPEG-LICENSE.txt")
            break


def download_ffmpeg(force: bool = False) -> None:
    """Download the gyan.dev release 'essentials' build (has libass, NVENC,
    QSV and AMF) and extract ffmpeg.exe, ffprobe.exe and LICENSE into bin/."""
    BIN_DIR.mkdir(exist_ok=True)
    if all((BIN_DIR / e).exists() for e in FFMPEG_EXES) and not force:
        print("[ffmpeg] = bin/ already has ffmpeg.exe and ffprobe.exe")
        return
    print(f"[ffmpeg] downloading {FFMPEG_ZIP} (~100 MB)…")
    data = _download(FFMPEG_ZIP, timeout=600)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name in FFMPEG_EXES or (name == "LICENSE"
                                       and info.filename.count("/") == 1):
                out = BIN_DIR / ("FFMPEG-LICENSE.txt" if name == "LICENSE" else name)
                out.write_bytes(zf.read(info))
                print(f"[ffmpeg] + bin/{out.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fonts-only", action="store_true")
    ap.add_argument("--ffmpeg-only", action="store_true")
    ap.add_argument("--ffmpeg-from", type=Path, metavar="DIR",
                    help="folder that contains ffmpeg.exe and ffprobe.exe")
    ap.add_argument("--download-ffmpeg", action="store_true",
                    help="download ffmpeg instead of copying the one on PATH")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args(argv)

    if not args.ffmpeg_only:
        n = fetch_fonts(force=args.force)
        print(f"[fonts] {n} new file(s) in {FONTS_DIR.relative_to(ROOT)}")

    if not args.fonts_only:
        if args.download_ffmpeg:
            download_ffmpeg(force=args.force)
        else:
            src = args.ffmpeg_from or _find_system_ffmpeg()
            if src is None:
                print("[ffmpeg] not found on PATH — rerun with --download-ffmpeg "
                      "or --ffmpeg-from DIR")
                return 1
            copy_ffmpeg(src, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
