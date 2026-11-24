"""A video file's poster: the picture the file carries as its thumbnail (MP4
cover art, what Windows Explorer, phones and players show for the file).

Downloads from the DL tab carry their poster inside the file: as cover art
and, with "Poster in the first second", as the video's first second too
(app.core.downloader.add_poster). Files the app writes later (a dub, an
episode cut by the Export tab) are new files; these helpers keep the poster:

    find_poster(video)        a JPEG of the video's poster ("" = none known)
    read_cover(video)         the cover art inside the file, as a JPEG
    set_cover(video, image)   put a picture into the file as its cover
    intro_seconds(video)      length of the poster second at its start (0 = none)
    poster_comment(...)       the file comment that says what the file carries

No Qt imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

from app.core.logger import get_logger
from app.core.paths import app_data_dir, ffmpeg_path, ffprobe_path, temp_dir

log = get_logger("cover_art")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# containers that hold cover art as an "attached picture" video track
COVER_EXTS = (".mp4", ".m4v", ".mov")
# the same words as the DL tab writes (app.core.downloader), so a file the
# dub or the Export tab wrote is not given a second poster when downloaded again
POSTER_TAG = "RecapStudio poster"
POSTER_TAG_INTRO = "RecapStudio poster + 1s intro"
POSTER_INTRO_S = 1.0


def _probe(video: str) -> dict:
    try:
        cp = subprocess.run(
            [ffprobe_path(), "-v", "error", "-show_streams", "-show_format", "-of", "json", video],
            capture_output=True, encoding="utf-8", errors="replace", timeout=60,
            creationflags=_NO_WINDOW)
        return json.loads(cp.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def _is_cover(stream: dict) -> bool:
    return (stream.get("codec_type") == "video"
            and bool((stream.get("disposition") or {}).get("attached_pic")))


def _main_videos(info: dict) -> list[dict]:
    return [s for s in info.get("streams") or []
            if s.get("codec_type") == "video" and not _is_cover(s)]


def _comment(info: dict) -> str:
    tags = (info.get("format") or {}).get("tags") or {}
    return str({str(k).lower(): v for k, v in tags.items()}.get("comment", ""))


def _cache_dir() -> Path:
    d = app_data_dir() / "cache" / "posters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_key(path: str) -> str:
    """Changes when the file does (a new cover is a new file)."""
    st = os.stat(path)
    raw = f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:20]


def has_cover(video: str) -> bool:
    return any(_is_cover(s) for s in _probe(video).get("streams") or [])


def read_cover(video: str, out_path: str = "") -> str:
    """The cover art inside `video` as a JPEG (kept in the app's cache unless
    `out_path` is given); "" when the file carries none."""
    if not video or not os.path.isfile(video):
        return ""
    info = _probe(video)
    idx = next((int(s.get("index", 0)) for s in info.get("streams") or [] if _is_cover(s)), None)
    if idx is None:
        return ""
    out = Path(out_path) if out_path else _cache_dir() / f"cover_{_file_key(video)}.jpg"
    if not out_path and out.is_file() and out.stat().st_size > 0:
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"~cover-{uuid.uuid4().hex[:8]}.jpg")
    try:
        cp = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-v", "error", "-y", "-i", video,
             "-map", f"0:{idx}", "-frames:v", "1", "-q:v", "2", "-update", "1", str(tmp)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=120,
            creationflags=_NO_WINDOW)
        if cp.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            os.replace(tmp, out)
            return str(out)
        log.info("could not read the cover of %s: %s", video, (cp.stderr or "").strip()[-200:])
        return ""
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("could not read the cover of %s: %s", video, exc)
        return ""
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _download_poster(video: str) -> str:
    """The poster the DL tab saved in its cache when it downloaded `video`."""
    try:
        from app.core.downloader import poster_cache_path
        p = poster_cache_path(video)
        return str(p) if p.is_file() and p.stat().st_size > 0 else ""
    except Exception:
        return ""


def dub_sources(video: str) -> list[str]:
    """The videos that dub projects made `video` from: the file is a dub's
    output under the name the dub gave it, or (renamed or moved since) a
    video of 30 s or more exactly as long as a dubbed video was (a dub keeps
    its source's length to the hundredth of a second)."""
    out: list[str] = []
    want = os.path.normcase(os.path.abspath(video))
    root = app_data_dir() / "projects" / "dub"
    try:
        projects = list(root.glob("*/project.json"))
    except OSError:
        return out
    by_length: list[tuple[float, str]] = []
    for pj in projects:
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        src = str(data.get("video") or "")
        if not src:
            continue
        names = [str(data.get("output") or ""),
                 str((data.get("options") or {}).get("output_path") or "")]
        if any(n and os.path.normcase(os.path.abspath(n)) == want for n in names):
            if src not in out:
                out.append(src)
        elif any(names):
            try:
                by_length.append((float(data.get("duration") or 0), src))
            except (TypeError, ValueError):
                pass
    if not out and by_length:
        try:
            length = float((_probe(video).get("format") or {}).get("duration") or 0)
        except ValueError:
            length = 0.0
        if length >= 30:
            for dur, src in by_length:
                if abs(dur - length) <= 0.05 and src not in out:
                    out.append(src)
    return out


def find_poster(video: str, sources: tuple[str, ...] | list[str] = ()) -> str:
    """A JPEG of `video`'s poster, "" when none is known: the cover art in the
    file; else the poster the DL tab saved when it downloaded the file; else
    the same for each of `sources`, then for the video a dub project made
    this file from (dubs made before the dub kept the poster)."""
    seen: set[str] = set()

    def look(path: str) -> str:
        key = os.path.normcase(os.path.abspath(path)) if path else ""
        if not key or key in seen:
            return ""
        seen.add(key)
        return read_cover(path) or _download_poster(path)

    for cand in (video, *sources):
        hit = look(cand)
        if hit:
            return hit
    for cand in dub_sources(video) if video else []:
        hit = look(cand)
        if hit:
            return hit
    return ""


def intro_seconds(video: str) -> float:
    """How long the poster shown at the start of `video` lasts (the DL tab's
    "Poster in the first second": one second, in whole frames), or 0 when
    the file does not start with it."""
    info = _probe(video)
    if POSTER_TAG_INTRO not in _comment(info):
        return 0.0
    fps = 0.0
    for s in _main_videos(info)[:1]:
        for key in ("avg_frame_rate", "r_frame_rate"):
            num, _, den = str(s.get(key) or "").partition("/")
            try:
                fps = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                continue
            if 5 <= fps <= 120:
                break
            fps = 0.0
    if not fps:
        return POSTER_INTRO_S
    return max(1, round(fps * POSTER_INTRO_S)) / fps


def is_poster_comment(comment: str) -> bool:
    return comment.startswith(POSTER_TAG)


def poster_comment(cover: bool, intro: bool) -> str:
    """The file comment for a file the app wrote: what poster it carries."""
    if intro:
        return POSTER_TAG_INTRO
    return POSTER_TAG if cover else ""


def source_comment(video: str) -> str:
    return _comment(_probe(video))


def _as_jpeg(image: str, out: str, long_side: int = 1920) -> bool:
    try:
        from PIL import Image
        with Image.open(image) as im:
            pic = im.convert("RGB")
        if max(pic.size) > long_side:
            scale = long_side / max(pic.size)
            pic = pic.resize((max(1, round(pic.width * scale)), max(1, round(pic.height * scale))),
                             Image.LANCZOS)
        pic.save(out, quality=92, subsampling=0)
        return True
    except Exception as exc:
        log.info("poster picture %s not usable: %s", image, exc)
        return False


def set_cover(video: str, image: str, intro: bool | None = None) -> bool:
    """Put `image` into `video` as its cover art (replacing the one it has):
    a stream copy, so the picture and sound are not re-encoded and every
    other track (audio, subtitles) and the file's details are kept. Written
    beside the video and swapped in at the end: on any failure the file
    stays as it was. `intro`: whether the video starts with the poster second
    (recorded in the file comment; None = keep what the comment says).
    Returns True when the file now carries the picture."""
    ext = os.path.splitext(video)[1].lower()
    if ext not in COVER_EXTS or not os.path.isfile(video) or not os.path.isfile(image):
        return False
    info = _probe(video)
    mains = _main_videos(info)
    if not mains:
        return False
    try:
        duration = float((info.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    tag = uuid.uuid4().hex[:8]
    jpg = str(temp_dir() / f"cover_in_{tag}.jpg")
    tmp = os.path.join(os.path.dirname(os.path.abspath(video)), f"~cover-{tag}{ext}")
    try:
        if not _as_jpeg(image, jpg):
            return False
        old = _comment(info)
        if intro is None:
            intro = POSTER_TAG_INTRO in old
        cmd = [ffmpeg_path(), "-hide_banner", "-v", "error", "-y", "-i", video, "-i", jpg,
               "-map", "0:V", "-map", "0:a?", "-map", "0:s?", "-map", "1:v:0", "-c", "copy",
               f"-disposition:v:{len(mains)}", "attached_pic", "-map_metadata", "0"]
        if not old or is_poster_comment(old):
            cmd += ["-metadata", f"comment={poster_comment(True, intro)}"]
        cmd += ["-movflags", "+faststart", tmp]
        cp = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                            timeout=3600, creationflags=_NO_WINDOW)
        if cp.returncode != 0 or not os.path.isfile(tmp):
            log.warning("could not put the poster into %s: %s", video,
                        (cp.stderr or "").strip()[-300:])
            return False
        check = _probe(tmp)
        try:
            got = float((check.get("format") or {}).get("duration") or 0)
        except ValueError:
            got = 0.0
        if not any(_is_cover(s) for s in check.get("streams") or []) or \
                (duration and got < duration - 0.6):
            log.warning("the copy of %s with its poster looks incomplete; kept the file", video)
            return False
        os.replace(tmp, video)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not put the poster into %s: %s", video, exc)
        return False
    finally:
        for f in (tmp, jpg):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    import sys
    import tempfile

    from PIL import Image

    sys.stdout.reconfigure(encoding="utf-8")
    bad: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(("ok    " if ok else "FAIL  ") + name)
        if not ok:
            bad.append(name)

    d = tempfile.mkdtemp(prefix="封面_ប៉ុស្ទ័រ_")
    side = Path(tempfile.mkdtemp(prefix="rs_cover_data_"))   # never the app's real data folder
    globals().update(app_data_dir=lambda: side, temp_dir=lambda: side)
    v = os.path.join(d, "v.mp4")
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:d=3:r=25",
                    "-f", "lavfi", "-i", "sine=f=440:d=3", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", v], check=True)
    pic = os.path.join(d, "poster.png")
    Image.new("RGB", (400, 600), (200, 30, 30)).save(pic)
    check("a new video has no cover", not has_cover(v) and read_cover(v) == "")
    check("set_cover puts the poster in", set_cover(v, pic))
    check("the file carries it", has_cover(v))
    got = read_cover(v)
    check("read_cover gives it back", bool(got) and Image.open(got).size == (400, 600)
          and Image.open(got).convert("RGB").getpixel((200, 300))[0] > 150)
    check("the comment says so", source_comment(v) == POSTER_TAG)
    info = _probe(v)
    check("picture and sound kept", len(_main_videos(info)) == 1
          and any(s.get("codec_type") == "audio" for s in info.get("streams") or []))
    Image.new("RGB", (640, 360), (20, 160, 40)).save(pic)
    check("a second set_cover replaces it", set_cover(v, pic)
          and sum(_is_cover(s) for s in _probe(v).get("streams") or []) == 1)
    green = read_cover(v)
    check("…and the new one is read", Image.open(green).convert("RGB").getpixel((5, 5))[1] > 120)
    check("find_poster finds the file's own", find_poster(v) == green)
    check("no intro without the tag", intro_seconds(v) == 0.0)
    check("set_cover can record an intro", set_cover(v, pic, intro=True)
          and abs(intro_seconds(v) - 1.0) < 0.05)
    check("a file it cannot hold is refused", not set_cover(os.path.join(d, "x.webm"), pic))
    check("nothing left beside the video", sorted(os.listdir(d)) == ["poster.png", "v.mp4"])
    # a dub renamed after it was made: found by its exact length (30 s and more)
    long_src = os.path.join(d, "source.mp4")
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=160x90:d=31:r=10", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    long_src], check=True)
    set_cover(long_src, pic)
    renamed = os.path.join(d, "ឈ្មោះថ្មី.mp4")
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-i", long_src, "-map", "0:V",
                    "-c", "copy", renamed], check=True)
    proj = side / "projects" / "dub" / "p1"
    proj.mkdir(parents=True)
    length = float((_probe(renamed).get("format") or {}).get("duration") or 0)
    (proj / "project.json").write_text(json.dumps(
        {"video": long_src, "output": os.path.join(d, "old name.dub.km.mp4"),
         "duration": length, "options": {"output_path": ""}}), encoding="utf-8")
    check("a renamed dub is traced to its source by its length",
          dub_sources(renamed) == [long_src] and bool(find_poster(renamed)))
    check("a short clip is not matched by length", dub_sources(v) == [])
    print("PASS" if not bad else f"FAILED: {bad}")
    raise SystemExit(1 if bad else 0)
