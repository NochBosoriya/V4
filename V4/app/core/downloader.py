"""Video download queue (yt-dlp) + local media Library (SQLite).

Contract: app/core/downloader.py — DownloadItem, DownloadQueue, Library.
No Qt imports here. yt_dlp is lazy-imported inside worker methods.
"""
from __future__ import annotations

import gc
import glob
import hashlib
import html
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit

from app.core.errors import AppError, CancelledError
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import app_data_dir, bin_dir, ffmpeg_path, ffprobe_path, pip_install

log = get_logger("downloader")


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    poster    TEXT NOT NULL DEFAULT '',
    meta_json TEXT NOT NULL DEFAULT '{}',
    created   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id   INTEGER NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    filepath    TEXT NOT NULL DEFAULT '',
    duration    REAL NOT NULL DEFAULT 0,
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    transcribed INTEGER NOT NULL DEFAULT 0,
    translated  INTEGER NOT NULL DEFAULT 0,
    dubbed      INTEGER NOT NULL DEFAULT 0,
    rendered    INTEGER NOT NULL DEFAULT 0,
    thumbnail   TEXT NOT NULL DEFAULT '',
    created     REAL NOT NULL,
    FOREIGN KEY (series_id) REFERENCES series (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_episodes_series ON episodes (series_id);
CREATE INDEX IF NOT EXISTS idx_episodes_path   ON episodes (filepath);
"""

_STATUS_FLAGS = ("transcribed", "translated", "dubbed", "rendered")
_EPISODE_FIELDS = ("title", "filepath", "duration", "width", "height",
                   "thumbnail") + _STATUS_FLAGS


class Library:
    """Thread-safe SQLite media library at app_data_dir()/library.db (WAL)."""

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path else app_data_dir() / "library.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- series ------------------------------------------------------------

    def add_series(self, name: str, poster: str = "", meta: dict | None = None) -> int:
        name = (name or "").strip() or "Misc"
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM series WHERE name = ?", (name,)).fetchone()
            if row:
                sid = int(row["id"])
                if poster or meta:
                    sets, vals = [], []
                    if poster:
                        sets.append("poster = ?")
                        vals.append(poster)
                    if meta:
                        sets.append("meta_json = ?")
                        vals.append(json.dumps(meta, ensure_ascii=False))
                    vals.append(sid)
                    self._conn.execute(
                        f"UPDATE series SET {', '.join(sets)} WHERE id = ?", vals)
                    self._conn.commit()
                return sid
            cur = self._conn.execute(
                "INSERT INTO series (name, poster, meta_json, created) "
                "VALUES (?, ?, ?, ?)",
                (name, poster, json.dumps(meta or {}, ensure_ascii=False),
                 time.time()))
            self._conn.commit()
            return int(cur.lastrowid)

    def get_series(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.*, COUNT(e.id) AS episode_count FROM series s "
                "LEFT JOIN episodes e ON e.series_id = s.id "
                "GROUP BY s.id ORDER BY s.created DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["meta"] = json.loads(d.get("meta_json") or "{}")
            except (TypeError, ValueError):
                d["meta"] = {}
            out.append(d)
        return out

    def delete_series(self, series_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM episodes WHERE series_id = ?",
                               (series_id,))
            self._conn.execute("DELETE FROM series WHERE id = ?", (series_id,))
            self._conn.commit()

    # -- episodes ----------------------------------------------------------

    def add_episode(self, series_id: int, title: str, filepath: str,
                    duration: float = 0, width: int = 0, height: int = 0) -> int:
        filepath = str(filepath)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM episodes WHERE filepath = ?",
                (filepath,)).fetchone()
            if row:
                eid = int(row["id"])
                self._conn.execute(
                    "UPDATE episodes SET series_id = ?, title = ?, duration = ?, "
                    "width = ?, height = ? WHERE id = ?",
                    (series_id, title, float(duration), int(width), int(height),
                     eid))
                self._conn.commit()
                return eid
            cur = self._conn.execute(
                "INSERT INTO episodes (series_id, title, filepath, duration, "
                "width, height, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (series_id, title, filepath, float(duration), int(width),
                 int(height), time.time()))
            self._conn.commit()
            return int(cur.lastrowid)

    def set_status(self, episode_id: int, **flags) -> None:
        """Update the bool workflow columns transcribed/translated/dubbed/rendered."""
        sets, vals = [], []
        for k, v in flags.items():
            if k not in _STATUS_FLAGS:
                raise AppError(
                    f"Unknown episode status flag: {k}",
                    f"មិនស្គាល់ស្ថានភាពភាគ: {k}")
            sets.append(f"{k} = ?")
            vals.append(1 if v else 0)
        if not sets:
            return
        vals.append(episode_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE episodes SET {', '.join(sets)} WHERE id = ?", vals)
            self._conn.commit()

    def update_episode(self, episode_id: int, **fields) -> None:
        """Internal-ish helper: update any writable episode column."""
        sets, vals = [], []
        for k, v in fields.items():
            if k not in _EPISODE_FIELDS:
                continue
            sets.append(f"{k} = ?")
            if k in _STATUS_FLAGS:
                vals.append(1 if v else 0)
            else:
                vals.append(v)
        if not sets:
            return
        vals.append(episode_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE episodes SET {', '.join(sets)} WHERE id = ?", vals)
            self._conn.commit()

    def get_episodes(self, series_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE series_id = ? ORDER BY created ASC",
                (series_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_episode(self, episode_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM episodes WHERE id = ?",
                               (episode_id,))
            self._conn.commit()

    def find_episode_by_path(self, path: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE filepath = ?",
                (str(path),)).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


# --------------------------------------------------------------------------
# Download queue
# --------------------------------------------------------------------------

@dataclass
class DownloadItem:
    id: str
    url: str
    title: str = ""
    # queued|fetching_meta|downloading|paused|done|error, and "removed" once,
    # when it leaves the queue (cancelled / taken off the list)
    status: str = "queued"
    progress: float = 0.0
    speed: str = ""
    eta: str = ""
    filepath: str = ""
    thumbnail: str = ""
    error: str = ""
    series: str = ""
    index: int = 0          # episode number: file name starts "EP001 - "
    quality: str = ""       # key of QUALITIES, fixed when the item is added
    out_dir: str = ""       # save exactly here ("" = <queue folder>/<series>)
    poster_intro: bool = False   # show the poster for the video's first second
    thumb_hint: str = ""    # poster URL the scan showed (fallback poster)


# image "formats" some sites list are never the video: Weibo offers its
# 320x180 preview .jpg, which 480p / 360p used to download instead
_NO_IMAGES = "[ext!=jpg][ext!=jpeg][ext!=png][ext!=webp][ext!=gif]"


def _quality_spec(cap: int | None) -> str:
    """yt-dlp format selector: the best video up to `cap` pixels high; when
    the site has nothing that small, the smallest it has (closest to cap)."""
    n = _NO_IMAGES
    if cap is None:
        return f"bv*{n}+ba/b{n}"
    return f"bv*[height<={cap}]{n}+ba/b[height<={cap}]{n}/wv*{n}+ba/w{n}"


# quality choice -> yt-dlp format selector
QUALITIES = {
    "best": _quality_spec(None),
    "1080": _quality_spec(1080),
    "720": _quality_spec(720),
    "480": _quality_spec(480),
    "360": _quality_spec(360),
}
# at the same resolution prefer H.264 / AAC: plays in the preview player and
# edits faster than AV1 / VP9
FORMAT_SORT = ("res", "vcodec:h264", "acodec:aac")
COOKIE_BROWSERS = ("firefox", "chrome", "edge", "brave", "opera", "vivaldi")

# youtu.be/ID, youtube.com/watch?v=ID, /shorts/ID, /embed/ID, /live/ID
_YT_VIDEO_ID = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/|v/)|youtu\.be/)"
    r"([\w-]{11})")
# share / tracking query parameters that do not change which video a link is
_TRACKING_PARAMS = ("si", "feature", "pp", "fbclid", "gclid", "igsh", "igshid",
                    "is_from_webapp", "sender_device", "spm_id_from", "vd_source")


def _url_key(url: str) -> str:
    """Same key for links to the same video: youtu.be/ID?si=… and
    youtube.com/watch?v=ID match; tracking parameters are ignored."""
    url = url.strip()
    m = _YT_VIDEO_ID.search(url)
    if m:
        return "youtube:" + m.group(1)
    parts = urlsplit(url if "://" in url else "https://" + url)
    host = parts.netloc.lower()
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                   if k.lower() not in _TRACKING_PARAMS
                   and not k.lower().startswith(("utm_", "share_")))
    return f"{host}{parts.path.rstrip('/')}?{urlencode(query)}"


url_key = _url_key          # public: the UI marks cards already in the list


# files a download leaves behind until it finishes: name.mp4.part,
# name.mp4.part-Frag7, name.mp4.ytdl, and the separate video / audio streams
# "…[id].f137.mp4" that are merged at the end. A finished "…[id].mp4" never
# matches, so a cancel cannot delete a completed video.
_PARTIAL_FILE = re.compile(r"(\.part(-Frag\d+)?|\.ytdl|\]\.f[^.\\/\]]+\.\w+)$")


def _delete_partial_files(paths: set[str]) -> None:
    """Delete what a cancelled download wrote. yt-dlp leaves its .part file
    open when a progress hook aborts it; the handle goes once the aborted
    frames are collected, so a file still in use is retried after gc."""
    candidates = set()
    for p in paths:
        candidates.update((p, p + ".part", p + ".ytdl"))
        candidates.update(glob.glob(glob.escape(p) + "-Frag*"))
        candidates.update(glob.glob(glob.escape(p) + ".part-Frag*"))
    for path in candidates:
        if not (_PARTIAL_FILE.search(path) and os.path.isfile(path)):
            continue
        for attempt in (1, 2):
            try:
                os.remove(path)
                break
            except PermissionError:
                if attempt == 1:
                    gc.collect()
                    time.sleep(0.2)
                else:
                    log.warning("could not delete partial file: %s", path)
            except OSError as exc:
                log.warning("could not delete partial file %s: %s", path, exc)
                break


class DuplicateUrlError(AppError):
    """add(): the link is already in the queue; .item is the existing entry."""

    def __init__(self, item: "DownloadItem"):
        super().__init__(f"This link is already in the list:\n{item.url}",
                         f"តំណនេះមាននៅក្នុងបញ្ជីរួចហើយ:\n{item.url}")
        self.item = item


def _js_runtime_opts() -> dict:
    """YouTube needs a JavaScript runtime for some videos: use deno / node /
    bun from the app's bin folder or PATH, plus yt-dlp's official helper
    scripts (fetched from GitHub when needed)."""
    import shutil
    for name in ("deno", "node", "bun"):
        exe = bin_dir() / f"{name}.exe"
        path = str(exe) if exe.is_file() else shutil.which(name)
        if path:
            return {"js_runtimes": {name: {"path": path}},
                    "remote_components": ["ejs:github"]}
    return {}


_COOKIE_WARNED: set[str] = set()      # missing cookies files already reported


def cookie_file_missing(cookies: str) -> bool:
    """True when the login setting is a cookies.txt that no longer exists
    (moved or deleted since it was picked)."""
    return cookies.startswith("file:") and not Path(cookies[5:]).is_file()


def _cookie_opts(cookies: str) -> dict:
    """"file:<path>" (cookies.txt) or "browser:<name>" -> yt-dlp options, so
    sites that need a login (and age-restricted videos) can be downloaded."""
    if cookie_file_missing(cookies):
        # said once: without it the site sees no login, and its error
        # ("sign in", "private video") would not tell why
        if cookies not in _COOKIE_WARNED:
            _COOKIE_WARNED.add(cookies)
            log.warning("cookies file not found: %s", cookies[5:])
            LOG_BUS.publish(f"[DL] the cookies file is gone (moved or deleted?): {cookies[5:]}"
                            " — downloading without your login. Pick it again under Login.")
        return {}
    if cookies.startswith("file:"):
        return {"cookiefile": cookies[5:]}
    if cookies.startswith("browser:") and cookies[8:] in COOKIE_BROWSERS:
        return {"cookiesfrombrowser": (cookies[8:], None, None, None)}
    return {}


def _import_ytdlp():
    try:
        import yt_dlp  # lazy heavy dep
    except ImportError as exc:
        raise AppError(f"yt-dlp is not installed. Run: {pip_install('yt-dlp')}",
                       f"yt-dlp មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ: {pip_install('yt-dlp')}") from exc
    return yt_dlp


# --------------------------------------------------------------------------
# Scanning a link (the Download tab's card)
# --------------------------------------------------------------------------

MAX_SCAN_ENTRIES = 1000      # bigger playlists / channels: the first 1000
_DETAIL_LIMIT = 300          # read missing titles before showing up to this many
_DETAIL_WORKERS = 4
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_URL_RE = re.compile(r"^(?:https?://)?(?:[\w-]+\.)+[a-z][a-z0-9-]{1,23}(?::\d{2,5})?"
                     r"(?:[/?#]\S*)?$", re.I)


def looks_like_url(text: str) -> bool:
    """True for something that can be a web link ("youtu.be/x", "https://…");
    plain words such as "hello world" are not."""
    t = (text or "").strip()
    return bool(t) and not any(c.isspace() for c in t) and bool(_URL_RE.match(t))


def _ydl_error(exc: BaseException) -> str:
    """yt-dlp's message for a failed link, without colour codes / 'ERROR:'."""
    msg = _ANSI.sub("", str(exc)).strip()
    msg = re.sub(r"^(?:ERROR:\s*)+", "", msg)
    return (msg.splitlines()[0] if msg else "")[:300] or exc.__class__.__name__


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Scan cancelled")


def _progress(progress, frac: float, text: str) -> None:
    if progress is not None:
        try:
            progress(frac, text)
        except Exception:
            pass


# -- episode numbers from titles ---------------------------------------------

_DIGITS = str.maketrans("０１２３４５６７８９០១២៣៤៥៦៧៨៩๐๑๒๓๔๕๖๗๘๙",
                        "012345678901234567890123456789")
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_NUM = r"(\d{1,4}|[零〇一二两兩三四五六七八九十百千]{1,7})"
# a number followed by a unit is a count, not an episode: "5场BOSS战", "50小时"
_UNIT = (r"(?!\s*(?:场|場|小时|小時|分钟|分鐘|分|秒|年|月|日|天|个|個|位|万|萬|千|亿|億|倍|人|次"
         r"|岁|歲|号|號|%|％|[kKpP](?![a-zA-Z])))")
# strongest first; the first pattern that matches gives the number
_EP_PATTERNS = [re.compile(p) for p in (
    r"第\s*" + _NUM + r"\s*[-~～至到]\s*" + _NUM + r"\s*[集话話回期]",   # 第1-3集: the first
    r"第\s*" + _NUM + r"\s*[集话話回期章篇部]",                         # 第12集 第十二集 第3話
    r"(?i)(?<![a-z])(?:episode|episod|eps|epi|ep)\s*\.?\s*[#:：\-]?\s*(\d{1,4})(?!\d)",
    r"(?<![A-Za-z])E(\d{1,4})(?!\d)",                                   # E12, S01E05
    r"ភាគ\s*(?:ទី)?\s*(\d{1,4})",                                       # Khmer ភាគទី១២
    r"(?i)(?<![a-z])tập\s*(\d{1,4})",                                   # Vietnamese
    r"ตอน(?:ที่)?\s*(\d{1,4})",                                         # Thai
    r"(\d{1,4})\s*[-~～]\s*\d{1,4}\s*[集话話화회]",                      # 1-3集
    r"(\d{1,4})\s*[集话話화회]",                                         # 12集 12话 12화
    r"(?i)(?<![a-z])(?:part|pt)\s*\.?\s*(\d{1,4})(?!\d)",               # Part 3
    r"^\s*[Pp](\d{1,4})(?!\d)",                                         # Bilibili "P3 …"
    r"(?i)(?<![a-z])p(\d{1,3})(?=\s)",                                  # "… p04 恋爱循环"
    r"[》」』]\s*(\d{1,3})(?!\d)" + _UNIT,                               # 《不烧心》12
    r"[#＃]\s*(\d{1,4})(?!\d)",                                          # #12
    r"[（(\[【]\s*(\d{1,3})\s*[)）\]】]",                                 # (12) 【12】
    r"^\s*(\d{1,3})" + _UNIT + r"\s*(?:[.、:：\-–]|\s)",                 # "12. title"
    r"(?<!\d)(\d{1,3})\s*$",                                            # "title 12"
)]
_NOT_EPISODE = [re.compile(p) for p in (
    r"[全共]\s*" + _NUM + r"\s*[集话話]",                                # 全36集 = how many
    r"\d{2,4}\s*[-./年]\s*\d{1,2}\s*[-./月]\s*\d{1,2}\s*日?",            # dates
    r"\d{1,2}:\d{2}(?::\d{2})?",                                        # times
    r"(?i)\b\d{3,4}\s*[pk]\b",                                          # 1080p 4K
)]


def _to_int(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    if not any(ch in _CN_UNIT for ch in s):           # 一二 = 12, digit by digit
        try:
            return int("".join(str(_CN_DIGIT[ch]) for ch in s))
        except (KeyError, ValueError):
            return None
    total = num = 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            total += (num or 1) * _CN_UNIT[ch]
            num = 0
        else:
            return None
    return total + num


def parse_episode_number(title: str) -> int | None:
    """The episode a video title names: "第12集", "第十二集", "EP12",
    "Episode 12", "ភាគទី១២", "Tập 12", "12話", "P3" … (None if none)."""
    t = (title or "").translate(_DIGITS)
    for rx in _NOT_EPISODE:
        t = rx.sub(" ", t)
    for rx in _EP_PATTERNS:
        m = rx.search(t)
        if m:
            n = _to_int(m.group(1))
            if n is not None and 0 < n < 10000:
                return n
    return None


@dataclass
class ScanEntry:
    """One video of a scanned link."""
    url: str
    title: str = ""
    duration: float = 0.0
    thumb: str = ""            # poster image URL shown on its card
    id: str = ""
    pl_index: int = 0          # position in the playlist (1 = first)
    ep: int = 0                # episode number (0 = an extra: trailer, behind the scenes…)
    dup: int = 0               # 1, 2 … for more videos with the same number
    ep_hint: int = 0           # episode number the site itself gives
    matched: bool = False      # the video the scanned link itself points at


@dataclass
class ScanResult:
    url: str
    kind: str                  # "video" | "playlist"
    title: str = ""
    uploader: str = ""
    entries: list[ScanEntry] = field(default_factory=list)
    numbering: str = ""        # playlist: "titles" (from the titles) | "order"
    truncated: bool = False    # the playlist has more than MAX_SCAN_ENTRIES
    extractor: str = ""
    warning: str = ""          # "drm": the site locks this video; "preview": a short preview only
    # one video: the heights it has and the size per QUALITIES key
    heights: list[int] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)


def number_episodes(entries: list[ScanEntry]) -> str:
    """Give each entry its episode number and put them in episode order.
    The titles' numbers are used when most titles have one and they rarely
    repeat ("titles"); else the playlist order is the numbering ("order")."""
    nums = [e.ep_hint or parse_episode_number(e.title) or 0 for e in entries]
    found = [n for n in nums if n]
    if entries and len(found) >= max(1.0, 0.6 * len(entries)) \
            and len(set(found)) >= 0.8 * len(found):
        for e, n in zip(entries, nums):
            e.ep = n
        numbered = sorted((e for e in entries if e.ep), key=lambda e: (e.ep, e.pl_index))
        seen: dict[int, int] = {}
        for e in numbered:
            e.dup = seen.get(e.ep, 0)
            seen[e.ep] = e.dup + 1
        entries[:] = numbered + [e for e in entries if not e.ep]
        return "titles"
    entries.sort(key=lambda e: e.pl_index)
    for k, e in enumerate(entries, start=1):
        e.ep, e.dup = k, 0
    return "order"


def _thumb_near(info: dict, want: int) -> str:
    """The thumbnail URL closest to `want` pixels wide (yt-dlp lists them
    worst to best; without sizes the last one)."""
    thumbs = [t for t in (info.get("thumbnails") or [])
              if isinstance(t, dict) and t.get("url")]
    sized = [t for t in thumbs if t.get("width")]

    def ratio(t: dict) -> float:
        return int(t.get("width") or 0) / max(1, int(t.get("height") or 0)) if t.get("height") else 0

    # YouTube's hq / sd thumbnails are 4:3 with black bars baked in: take a
    # 16:9 one when the site has one that is big enough
    wide = [t for t in sized if 1.7 <= ratio(t) <= 1.8 and int(t["width"]) >= 0.4 * want]
    if wide and any(1.3 <= ratio(t) <= 1.36 for t in sized):
        sized = wide
    if sized:
        def cost(t: dict) -> float:
            w = int(t.get("width") or 0)
            return abs(w - want) + (want if w < 0.66 * want else 0)
        return str(min(sized, key=cost)["url"])
    if thumbs:
        return str(thumbs[-1]["url"])
    return str(info.get("thumbnail") or "")


def _uploader(info: dict) -> str:
    return str(info.get("uploader") or info.get("channel") or info.get("uploader_id") or "")


def _scan_target(url: str) -> tuple[str, str, int]:
    """(link to scan, YouTube video id it points at, Bilibili part it points
    at). "watch?v=ID&list=…" scans the list and marks ID; a Bilibili part
    link (?p=N) scans every part and marks part N."""
    vid = ""
    m = _YT_VIDEO_ID.search(url)
    if m and "list=" in url:
        vid = m.group(1)
    part = 0
    parts = urlsplit(url)
    if (parts.netloc.lower().endswith("bilibili.com")
            and re.match(r"/video/(?:BV\w+|av\d+)", parts.path, re.I)):
        query = parse_qsl(parts.query, keep_blank_values=True)
        p = [v for k, v in query if k == "p"]
        if p and p[0].isdigit():
            part = int(p[0])
            url = parts._replace(query=urlencode([(k, v) for k, v in query if k != "p"])).geturl()
    return url, vid, part


def _bili_part(url: str) -> int:
    for k, v in parse_qsl(urlsplit(url).query):
        if k == "p" and v.isdigit():
            return int(v)
    return 0


def _scan_opts(cookies: str) -> dict:
    return {"quiet": True, "no_warnings": True, "logger": _YdlLogger(),
            "extract_flat": "in_playlist", "skip_download": True,
            "playlistend": MAX_SCAN_ENTRIES, "socket_timeout": 20,
            "color": "no_color", "format_sort": list(FORMAT_SORT),
            **_js_runtime_opts(), **_cookie_opts(cookies)}


# Hongguo (红果短剧) plays full episodes only in its protected player; a page
# carries at most a short public preview (checked 2026-11-01)
HONGGUO_NOTE = (
    "Hongguo (红果短剧) plays its full episodes only in its own protected player, "
    "so they cannot be downloaded. Only the short public preview on an episode "
    "page can be.",
    "Hongguo (红果短剧) ចាក់ភាគពេញតែក្នុងកម្មវិធីចាក់ដែលការពាររបស់ខ្លួនប៉ុណ្ណោះ "
    "ដូច្នេះមិនអាចទាញយកបានទេ។ អាចទាញយកបានតែវីដេអូបង្ហាញខ្លីសាធារណៈនៅលើទំព័រភាគប៉ុណ្ណោះ។")


def _is_hongguo(url: str) -> bool:
    return "hongguo" in urlsplit(url if "://" in url else "https://" + url).netloc.lower()


def _scan_error(url: str, exc: BaseException) -> AppError:
    msg = _ydl_error(exc)
    if _is_hongguo(url):
        # yt-dlp only says "Unsupported URL"
        return AppError(f"{HONGGUO_NOTE[0]} (yt-dlp: {msg})", HONGGUO_NOTE[1])
    return AppError(f"Could not read this link: {msg}",
                    f"មិនអាចអានតំណនេះបានទេ៖ {msg}")


def scan_url(url: str, cookies: str = "", progress=None,
             cancel: threading.Event | None = None) -> ScanResult:
    """Read a link without downloading: one video (its qualities and sizes),
    or a playlist / series page (its videos, numbered as episodes)."""
    yt_dlp = _import_ytdlp()
    raw = (url or "").strip()
    if not looks_like_url(raw):
        raise AppError(
            "That doesn't look like a link. Copy the address of a video, playlist "
            "or series page (it starts with https://).",
            "នេះមិនមែនជាតំណទេ។ សូមចម្លងអាសយដ្ឋាននៃទំព័រវីដេអូ បញ្ជីចាក់ ឬរឿងភាគ "
            "(ចាប់ផ្ដើមដោយ https://)។")
    if "://" not in raw:
        raw = "https://" + raw
    target, want_id, want_part = _scan_target(raw)
    _progress(progress, 0.05, "Reading the link")
    with yt_dlp.YoutubeDL(_scan_opts(cookies)) as ydl:
        try:
            info = ydl.extract_info(target, download=False)
        except Exception as exc:
            if not want_id:
                raise _scan_error(raw, exc) from exc
            # the video's list is private or gone: show just the video
            video_url, want_id = f"https://www.youtube.com/watch?v={want_id}", ""
            try:
                info = ydl.extract_info(video_url, download=False)
            except Exception as exc2:
                raise _scan_error(raw, exc2) from exc2
        _check_cancel(cancel)
        if not info:
            raise AppError(f"Could not read this link: {raw}",
                           f"មិនអាចអានតំណនេះបានទេ៖ {raw}")
        if info.get("entries") is None:
            return _video_result(raw, ydl, info)
    res = _playlist_result(raw, info)
    missing = [e for e in res.entries if not e.title]
    if missing and len(missing) <= _DETAIL_LIMIT:
        # the playlist page only listed links (Bilibili parts / collections,
        # Dailymotion…): read each video's title, length and poster
        _read_details(missing, cookies, progress, cancel)
    res.numbering = number_episodes(res.entries)
    for e in res.entries:
        if (want_id and e.id == want_id) or (want_part and _bili_part(e.url) == want_part):
            e.matched = True
    _progress(progress, 1.0, "Done")
    return res


def _video_result(url: str, ydl, info: dict) -> ScanResult:
    title = str(info.get("title") or "")
    e = ScanEntry(url=str(info.get("webpage_url") or url), title=title,
                  duration=float(info.get("duration") or 0), thumb=_thumb_near(info, 720),
                  id=str(info.get("id") or ""), pl_index=1, matched=True)
    e.ep = parse_episode_number(title) or 0
    formats = info.get("formats") or []
    drm = bool(formats) and all(f.get("has_drm") for f in formats)
    # "preview": Hongguo pages carry only a short public preview
    warning = "drm" if drm else "preview" if _is_hongguo(url) else ""
    return ScanResult(url=url, kind="video", title=title, uploader=_uploader(info),
                      entries=[e], extractor=str(info.get("extractor_key") or ""),
                      warning=warning, heights=_heights_of(info),
                      sizes=_sizes_by_quality(ydl, info))


def _playlist_result(url: str, info: dict) -> ScanResult:
    from yt_dlp.utils import unsmuggle_url
    entries: list[ScanEntry] = []
    for k, e in enumerate((e for e in info.get("entries") or [] if e), start=1):
        if len(entries) >= MAX_SCAN_ENTRIES:
            break
        u = str(e.get("url") or e.get("webpage_url") or "")
        if "://" not in u:
            u = str(e.get("webpage_url") or "")
        if not u:
            continue
        smug = unsmuggle_url(u, {})[1]          # Bilibili.tv hides details here
        smug = smug if isinstance(smug, dict) else {}
        vid = str(e.get("id") or "")
        thumb = _thumb_near(e, 360) or str(smug.get("thumbnail") or "")
        if not thumb and vid and str(e.get("ie_key") or "").startswith("Youtube") \
                and len(vid) == 11:
            thumb = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"
        try:
            hint = int(e.get("episode_number") or smug.get("episode_number") or 0)
        except (TypeError, ValueError):
            hint = 0
        entries.append(ScanEntry(
            url=u, title=str(e.get("title") or smug.get("title") or "").strip(),
            duration=float(e.get("duration") or 0), thumb=thumb, id=vid,
            pl_index=int(e.get("playlist_index") or k), ep_hint=hint))
    total = info.get("playlist_count")
    return ScanResult(
        url=url, kind="playlist",
        title=str(info.get("title") or info.get("playlist_title") or "").strip(),
        uploader=_uploader(info), entries=entries,
        truncated=len(entries) >= MAX_SCAN_ENTRIES
        or (isinstance(total, int) and total > len(entries)),
        extractor=str(info.get("extractor_key") or ""))


def _read_details(entries: list[ScanEntry], cookies: str, progress,
                  cancel: threading.Event | None) -> None:
    n = len(entries)
    ex = ThreadPoolExecutor(_DETAIL_WORKERS, thread_name_prefix="scan")
    try:
        futures = {ex.submit(fetch_entry_details, e.url, cookies): e for e in entries}
        for done, fut in enumerate(as_completed(futures), start=1):
            _check_cancel(cancel)
            try:
                apply_details(futures[fut], fut.result())
            except Exception as exc:     # one unreadable episode keeps its link
                log.info("episode details failed: %s", _ydl_error(exc))
            _progress(progress, 0.1 + 0.85 * done / n,
                      f"Reading episode details {done}/{n}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def apply_details(e: ScanEntry, d: dict) -> None:
    """Fill what fetch_entry_details found into an entry (keeps what it had)."""
    e.title = e.title or d.get("title") or ""
    e.duration = e.duration or float(d.get("duration") or 0)
    e.thumb = e.thumb or d.get("thumb") or ""
    e.id = e.id or d.get("id") or ""
    e.ep_hint = e.ep_hint or int(d.get("ep") or 0)


# -- per-thread yt-dlp for small requests ---------------------------------------

class _QuietLogger:
    def debug(self, msg: str) -> None:
        pass

    info = warning = debug

    def error(self, msg: str) -> None:
        LOG_BUS.publish(f"[scan][error] {_ANSI.sub('', str(msg))}")


_TLS = threading.local()


def _thread_ydl(cookies: str):
    """A YoutubeDL kept per thread and cookie setting: one per request would
    read the browser's cookies again every time."""
    cache = getattr(_TLS, "ydls", None)
    if cache is None:
        cache = _TLS.ydls = {}
    ydl = cache.get(cookies)
    if ydl is None:
        yt_dlp = _import_ytdlp()
        ydl = cache[cookies] = yt_dlp.YoutubeDL({
            "quiet": True, "no_warnings": True, "logger": _QuietLogger(),
            "skip_download": True, "noplaylist": True, "socket_timeout": 20,
            "color": "no_color", **_js_runtime_opts(), **_cookie_opts(cookies)})
    return ydl


def fetch_entry_details(url: str, cookies: str = "") -> dict:
    """Title, length and poster of one video a playlist page listed without
    them: {"title", "duration", "thumb", "id", "ep"}."""
    ydl = _thread_ydl(cookies)
    info = ydl.extract_info(url, download=False, process=False) or {}
    for _ in range(2):          # a link result points at the real video page
        if info.get("_type") in ("url", "url_transparent") and info.get("url") \
                and not info.get("title"):
            info = ydl.extract_info(info["url"], ie_key=info.get("ie_key"),
                                    download=False, process=False) or {}
    try:
        ep = int(info.get("episode_number") or 0)
    except (TypeError, ValueError):
        ep = 0
    return {"title": str(info.get("title") or "").strip(),
            "duration": float(info.get("duration") or 0),
            "thumb": _thumb_near(info, 360), "id": str(info.get("id") or ""), "ep": ep}


def _download_bytes(url: str, cookies: str = "", headers: dict | None = None,
                    limit: int = 15 * 1024 * 1024) -> bytes:
    from yt_dlp.networking import Request
    resp = _thread_ydl(cookies).urlopen(Request(url, headers=dict(headers or {})))
    try:
        data = resp.read(limit + 1)
    finally:
        resp.close()
    if len(data) > limit:
        raise AppError(f"File too big: {url}", f"ឯកសារធំពេក៖ {url}")
    return data


def _looks_like_image(data: bytes) -> bool:
    return (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"
            or data[:4] == b"GIF8" or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
            or data[4:8] == b"ftyp" or data[:2] == b"BM")


def _thumb_cache_dir() -> Path:
    d = app_data_dir() / "cache" / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_image(url: str, cookies: str = "", referer: str = "") -> bytes:
    """A poster / thumbnail image's bytes (kept in a disk cache)."""
    key = hashlib.sha1(url.encode("utf-8", "replace")).hexdigest()
    path = _thumb_cache_dir() / key[:2] / f"{key}.img"
    try:
        return path.read_bytes()
    except OSError:
        pass
    data = _download_bytes(url, cookies, {"Referer": referer} if referer else None)
    if not _looks_like_image(data):
        raise AppError(f"Not an image: {url}", f"មិនមែនជារូបភាពទេ៖ {url}")
    try:
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_name(f"{key}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as exc:
        log.info("thumbnail cache write failed: %s", exc)
    return data


def prune_thumb_cache(max_files: int = 4000) -> None:
    """Keep the newest max_files cached card images and video posters each."""
    for files in (lambda: _thumb_cache_dir().glob("*/*.img"),
                  lambda: app_data_dir().joinpath("cache", "posters").glob("*.jpg")):
        try:
            found = list(files())
            if len(found) <= max_files:
                continue
            found.sort(key=lambda p: p.stat().st_mtime)
            for p in found[:len(found) - int(max_files * 0.8)]:
                p.unlink(missing_ok=True)
        except OSError as exc:
            log.info("cache prune failed: %s", exc)


# -- qualities and sizes --------------------------------------------------------

_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp", "gif")


def _heights_of(info: dict) -> list[int]:
    return sorted({int(f["height"]) for f in info.get("formats") or []
                   if f.get("height") and f.get("vcodec") != "none"
                   and str(f.get("ext") or "") not in _IMAGE_EXTS})


def _format_bytes(f: dict, duration: float) -> float:
    size = f.get("filesize") or f.get("filesize_approx")
    if not size and f.get("tbr") and duration:
        size = float(f["tbr"]) * 1000 / 8 * duration
    return float(size or 0)


def _sizes_by_quality(ydl, info: dict) -> dict[str, int]:
    """Download size per QUALITIES key, picked the way a download picks."""
    formats = info.get("formats") or []
    duration = float(info.get("duration") or 0)
    out: dict[str, int] = {}
    for key, spec in QUALITIES.items():
        try:
            picked = ydl._select_formats(formats, ydl.build_format_selector(spec))
        except Exception:
            continue
        if picked:
            best = picked[-1]
            size = sum(_format_bytes(f, duration)
                       for f in (best.get("requested_formats") or [best]))
            if size > 0:
                out[key] = int(size)
    return out


def probe_video(url: str, cookies: str = "") -> dict:
    """Heights, size per quality and length of one video (the first episode
    of a playlist stands in for the others): {"heights", "sizes", "duration"}."""
    yt_dlp = _import_ytdlp()
    opts = {**_scan_opts(cookies), "noplaylist": True, "logger": _QuietLogger()}
    opts.pop("extract_flat", None)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False) or {}
        return {"heights": _heights_of(info), "sizes": _sizes_by_quality(ydl, info),
                "duration": float(info.get("duration") or 0)}


# -- the poster: saved, embedded as cover art, shown for the first second -------

_META_IMAGE = [re.compile(p, re.I) for p in (
    r'<meta[^>]+(?:property|name)=["\'](?:og:image(?::secure_url)?|twitter:image(?::src)?)'
    r'["\'][^>]*content=["\']([^"\']+)',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\'](?:og:image|twitter:image)',
    r'"thumbnailUrl"\s*:\s*\[?\s*"([^"]+)"',
    r'<video[^>]+poster=["\']([^"\']+)',
)]


def _page_poster_url(page_url: str, cookies: str = "") -> str:
    """The poster a video page names for link previews (og:image …)."""
    try:
        data = _download_bytes(page_url, cookies, limit=3 * 1024 * 1024)
    except Exception as exc:
        log.info("page poster: could not read %s: %s", page_url, exc)
        return ""
    text = data.decode("utf-8", "replace")
    for rx in _META_IMAGE:
        m = rx.search(text)
        if m:
            u = html.unescape(m.group(1)).replace("\\/", "/").strip()
            if u:
                return urljoin(page_url, u)
    return ""


def _probe_av(path: str) -> dict | None:
    """The main video stream and the first audio stream (ffprobe), or None
    when there is no video."""
    try:
        cp = subprocess.run(
            [ffprobe_path(), "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
            capture_output=True, encoding="utf-8", errors="replace", timeout=60,
            creationflags=_NO_WINDOW)
        data = json.loads(cp.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")), None)
    if not v or not v.get("width") or not v.get("height"):
        return None
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps, fpsf = "30", 30.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        fr = str(v.get(key) or "")
        num, _, den = fr.partition("/")
        try:
            val = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if 5 <= val <= 120:
            fps, fpsf = fr, val
            break
    sar = 1.0
    sn, _, sd = str(v.get("sample_aspect_ratio") or "1:1").partition(":")
    try:
        sar = float(sn) / float(sd) if float(sn) > 0 and float(sd) > 0 else 1.0
    except ValueError:
        pass
    w = int(round(int(v["width"]) * sar / 2)) * 2
    h = int(v["height"]) // 2 * 2
    try:
        duration = float((data.get("format") or {}).get("duration") or v.get("duration") or 0)
    except ValueError:
        duration = 0.0
    tags = {str(k).lower(): str(val) for k, val in
            ((data.get("format") or {}).get("tags") or {}).items()}
    comment = tags.get("comment", "")
    has_cover = any((s.get("disposition") or {}).get("attached_pic") for s in streams)
    if POSTER_TAG_INTRO in comment:
        poster = "intro"
    elif POSTER_TAG in comment:
        poster = "cover"
    else:
        # a cover without our tag was added by an older RecapStudio, which
        # may already have put the intro in: never risk a second one
        poster = "unknown" if has_cover else ""
    out = {"v_index": int(v.get("index", 0)), "w": w, "h": h, "fps": fps, "fpsf": fpsf,
           "duration": duration, "a": None, "poster": poster}
    if a:
        ch = int(a.get("channels") or 2)
        out["a"] = {"index": int(a.get("index", 1)), "rate": int(a.get("sample_rate") or 48000),
                    "channels": ch, "layout": str(a.get("channel_layout") or "")
                    or {1: "mono", 6: "5.1", 8: "7.1"}.get(ch, "stereo")}
    return out


def _image_to_jpeg(data: bytes, out: Path) -> bool:
    """Save image bytes (jpg / png / webp / avif …) as a real JPEG; refuses
    tiny images (tracking pixels, sprite strips)."""
    tag = uuid.uuid4().hex[:8]
    src, tmp = out.with_name(f"~img-{tag}.src"), out.with_name(f"~img-{tag}.jpg")
    try:
        src.write_bytes(data)
        cp = subprocess.run([ffmpeg_path(), "-hide_banner", "-v", "error", "-y", "-i", str(src),
                             "-frames:v", "1", "-q:v", "2", "-update", "1", str(tmp)],
                            capture_output=True, timeout=60, creationflags=_NO_WINDOW)
        if cp.returncode != 0 or not tmp.is_file():
            return False
        size = subprocess.run([ffprobe_path(), "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=width,height", "-of", "csv=p=0", str(tmp)],
                              capture_output=True, encoding="utf-8", errors="replace",
                              timeout=30, creationflags=_NO_WINDOW).stdout.strip()
        w, _, h = size.partition(",")
        if not (w.isdigit() and h.isdigit() and int(w) >= 100 and int(h) >= 60):
            return False
        os.replace(tmp, out)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        for f in (src, tmp):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass


def _frame_poster(video: str, out: Path, duration: float) -> bool:
    """No poster anywhere: the most typical frame of the video's opening."""
    t = max(0.0, min(5.0, duration * 0.1)) if duration else 1.0
    tmp = out.with_name(f"~img-{uuid.uuid4().hex[:8]}.jpg")
    try:
        cp = subprocess.run([ffmpeg_path(), "-hide_banner", "-v", "error", "-y", "-ss", f"{t:.2f}",
                             "-i", video, "-vf", "thumbnail=50", "-frames:v", "1", "-q:v", "2",
                             "-update", "1", str(tmp)],
                            capture_output=True, timeout=120, creationflags=_NO_WINDOW)
        if cp.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            os.replace(tmp, out)
            return True
        return False
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _poster_cache_dir() -> Path:
    d = app_data_dir() / "cache" / "posters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def poster_cache_path(video: str) -> Path:
    key = hashlib.sha1(os.path.abspath(video).encode("utf-8", "replace")).hexdigest()[:20]
    return _poster_cache_dir() / f"{key}.jpg"


def make_poster(info: dict, video: str, cookies: str = "", hint: str = "") -> str:
    """The video's poster as a JPEG in the app's cache (never a picture file
    in the user's folder: the poster goes inside the video): the site's
    poster (largest first), else the one the scan showed, else the page's
    link-preview image, else a frame of the video. Returns its path, "" if
    nothing worked. The cached copy is what the Library shows."""
    out = poster_cache_path(video)
    thumbs = [t for t in reversed(info.get("thumbnails") or [])
              if isinstance(t, dict) and t.get("url")]
    if not thumbs and info.get("thumbnail"):
        thumbs = [{"url": info["thumbnail"]}]
    tried: set[str] = set()
    candidates = [(t["url"], t.get("http_headers")) for t in thumbs[:4]]
    if hint:
        candidates.append((hint, None))
    for url, headers in candidates:
        if url in tried:
            continue
        tried.add(url)
        try:
            data = _download_bytes(url, cookies, headers)
        except Exception as exc:
            log.info("poster %s failed: %s", url, _ydl_error(exc))
            continue
        if _looks_like_image(data) and _image_to_jpeg(data, out):
            return str(out)
    page = str(info.get("webpage_url") or "")
    url = _page_poster_url(page, cookies) if page else ""
    if url and url not in tried:
        try:
            data = _download_bytes(url, cookies, {"Referer": page})
            if _looks_like_image(data) and _image_to_jpeg(data, out):
                LOG_BUS.publish("[poster] used the page's preview image")
                return str(out)
        except Exception as exc:
            log.info("page poster %s failed: %s", url, _ydl_error(exc))
    if _frame_poster(video, out, float(info.get("duration") or 0)):
        LOG_BUS.publish("[poster] the site has no poster: used a frame of the video")
        return str(out)
    return ""


POSTER_INTRO_S = 1.0
# written into the video's "comment" when its poster is added, so the same
# file downloaded again (yt-dlp: "already downloaded") is not given a second
# poster second
POSTER_TAG = "RecapStudio poster"
POSTER_TAG_INTRO = "RecapStudio poster + 1s intro"


def poster_done(video: str, intro: bool) -> bool:
    """True when RecapStudio already put the poster into this video (with
    the intro too, when `intro` is wanted)."""
    av = _probe_av(video)
    state = av["poster"] if av else ""
    return state in ("intro", "unknown") or (state == "cover" and not intro)


def _encoder_args(encoder: str) -> list[str]:
    """Video encoder options for the poster pass (stream 0 only)."""
    if encoder == "libx264" or not encoder:
        return ["-c:v:0", "libx264", "-preset", "veryfast", "-crf", "19"]
    from app.engines.media_exporter import ExportPreset, _vcodec_args
    args = _vcodec_args(ExportPreset(vcodec=encoder, bitrate=""))
    return ["-c:v:0" if a == "-c:v" else a for a in args]


def _intro_command(video: str, poster: str, out: str, av: dict, encoder: str) -> tuple[list[str], float]:
    """ffmpeg command: POSTER_INTRO_S of the poster (fitted over a blurred
    copy of itself, silent) + the whole video, re-encoded once; the poster
    is also embedded as cover art. Returns (command, output duration)."""
    w, h, fps = av["w"], av["h"], av["fps"]
    frames = max(1, round(av["fpsf"] * POSTER_INTRO_S))
    d = frames / av["fpsf"]
    a = av["a"]
    cmd = [ffmpeg_path(), "-hide_banner", "-y",
           "-loop", "1", "-framerate", fps, "-t", f"{d:.4f}", "-i", poster]
    if a:
        cmd += ["-f", "lavfi", "-t", f"{d:.4f}", "-i",
                f"anullsrc=r={a['rate']}:cl={a['layout']}"]
    vi = 2 if a else 1
    cmd += ["-i", video, "-i", poster]
    g = (f"[0:v]split=2[pa][pb];"
         f"[pa]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
         f"boxblur=20:2,eq=brightness=-0.08[bg];"
         f"[pb]scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2[fg];"
         f"[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,setsar=1,fps={fps},format=yuv420p[iv];"
         f"[{vi}:{av['v_index']}]scale={w}:{h},setsar=1,fps={fps},format=yuv420p[mv];")
    if a:
        fmt = f"aformat=sample_rates={a['rate']}:channel_layouts={a['layout']}"
        g += (f"[1:a]{fmt}[ia];[{vi}:{a['index']}]{fmt}[ma];"
              f"[iv][ia][mv][ma]concat=n=2:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        g += "[iv][mv]concat=n=2:v=1:a=0[v]"
        maps = ["-map", "[v]"]
    cmd += ["-filter_complex", g, *maps, "-map", f"{vi + 1}:v:0",
            *_encoder_args(encoder), "-c:v:1", "copy", "-disposition:v:1", "attached_pic",
            "-c:a", "aac", "-b:a", "192k" if not a or a["channels"] <= 2 else "384k",
            "-map_metadata", str(vi), "-metadata", f"comment={POSTER_TAG_INTRO}",
            "-max_muxing_queue_size", "9999",
            "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out]
    return cmd, d + av["duration"]


def add_poster(video: str, poster: str, intro: bool, encoder: str = "libx264",
               progress=None, cancel: threading.Event | None = None) -> str:
    """Embed the poster as the video's cover art and, with intro, show it for
    the first second too (the video is re-encoded once for that). Returns
    the final path (.mp4); on any failure the original file stays as it was.
    Raises CancelledError when cancel is set."""
    av = _probe_av(video)
    src = Path(video)
    if av is None or not os.path.isfile(poster):
        return video
    if av["poster"] in ("intro", "unknown") or (av["poster"] == "cover" and not intro):
        return video            # it has its poster already (downloaded before)
    # every result is an .mp4: a .webm / .flv / .mkv is repackaged (no
    # re-encode) in the same step that puts the poster in; if MP4 cannot hold
    # its video or audio format, the file stays as it was, without a poster
    final = src.with_suffix(".mp4")
    tag = uuid.uuid4().hex[:8]
    tmp = final.with_name(f"~poster-{tag}.mp4")
    # ffmpeg reads "%" in an image name as a pattern: such a poster gets a
    # plain-named copy in the cache (never a picture next to the video)
    pic = Path(poster)
    if "%" in pic.name:
        pic = _poster_cache_dir() / f"~in-{tag}.jpg"
    try:
        if pic != Path(poster):
            pic.write_bytes(Path(poster).read_bytes())
        if intro:
            from app.engines.media_exporter import _run_ffmpeg
            ok = False
            for enc in dict.fromkeys((encoder or "libx264", "libx264")):
                cmd, total = _intro_command(str(src), str(pic), str(tmp), av, enc)
                try:
                    _run_ffmpeg(cmd, total, progress, cancel, label="poster",
                                status_text="Adding poster")
                    ok = True
                    break
                except CancelledError:
                    raise
                except Exception as exc:
                    LOG_BUS.publish(f"[poster] {enc} failed, trying the next encoder: "
                                    f"{str(exc).splitlines()[0][:200]}")
            if not ok:
                return video
            want = av["duration"] + POSTER_INTRO_S
        else:
            cp = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-v", "error", "-y", "-i", str(src), "-i", str(pic),
                 "-map", f"0:{av['v_index']}", "-map", "0:a?", "-map", "1:v:0", "-c", "copy",
                 "-disposition:v:1", "attached_pic", "-map_metadata", "0",
                 "-metadata", f"comment={POSTER_TAG}", "-movflags", "+faststart", str(tmp)],
                capture_output=True, encoding="utf-8", errors="replace", timeout=1800,
                creationflags=_NO_WINDOW)
            if cp.returncode != 0:
                why = (f"MP4 cannot hold this {src.suffix} file's video / audio format; "
                       "kept it as it was, without a poster"
                       if src.suffix.lower() not in (".mp4", ".m4v", ".mov")
                       else "could not embed the cover")
                LOG_BUS.publish(f"[poster] {why}: {cp.stderr.strip()[-300:]}")
                return video
            want = av["duration"]
        check = _probe_av(str(tmp))
        if check is None or (want and check["duration"] < want - 0.6):
            LOG_BUS.publish("[poster] the new file looks incomplete; kept the original")
            return video
        os.replace(tmp, final)
        if final != src:
            src.unlink(missing_ok=True)      # e.g. a .webm now re-encoded to .mp4
        return str(final)
    except CancelledError:
        raise
    except OSError as exc:
        LOG_BUS.publish(f"[poster] could not replace the video file ({exc}); kept the original")
        return video
    finally:
        for f in (tmp, pic if pic != Path(poster) else None):
            try:
                if f is not None:
                    f.unlink(missing_ok=True)
            except OSError:
                pass


class _Abort(Exception):
    """Internal: raised from a progress hook to stop yt-dlp mid-download."""


class _YdlLogger:
    """Routes yt-dlp log lines to the LOG_BUS console."""

    def debug(self, msg: str) -> None:
        msg = str(msg)
        # yt-dlp sends its own info-level output through debug() prefixed [..]
        if msg.startswith("[debug]"):
            return
        LOG_BUS.publish(f"[yt-dlp] {msg}")

    def info(self, msg: str) -> None:
        LOG_BUS.publish(f"[yt-dlp] {msg}")

    def warning(self, msg: str) -> None:
        LOG_BUS.publish(f"[yt-dlp][warn] {msg}")

    def error(self, msg: str) -> None:
        LOG_BUS.publish(f"[yt-dlp][error] {msg}")


def _sanitize_dirname(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip())
    name = name.strip(". ")
    return name[:120] or "Misc"


sanitize_dirname = _sanitize_dirname     # public: the UI names a series folder


def _fmt_speed(bps: float | None) -> str:
    if not bps or bps <= 0:
        return ""
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if bps < 1024 or unit == "GB/s":
            return f"{bps:.1f} {unit}"
        bps /= 1024.0
    return ""


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


@dataclass
class _ItemFlags:
    pause: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)


class DownloadQueue:
    """yt-dlp based download queue with 2 concurrent worker threads."""

    def __init__(self, library: Library, out_dir: str | None = None):
        self.library = library
        self.out_dir = Path(out_dir) if out_dir else app_data_dir() / "downloads"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, DownloadItem] = {}
        self._flags: dict[str, _ItemFlags] = {}
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._started = False
        self._pending: list[str] = []   # ids queued before start()
        self._written: dict[str, set[str]] = {}  # id -> files yt-dlp wrote (unfinished)
        self._closing = False           # shutdown(): app exit, not a user cancel
        self._cb: Callable[[DownloadItem], None] | None = None
        self.quality = "best"           # key of QUALITIES
        self.cookies = ""               # "" | "file:<cookies.txt>" | "browser:<name>"
        self.poster_intro = False       # new items: poster as the first second
        self.encoder = "libx264"        # video encoder for that (hardware if any)

    # -- public API ---------------------------------------------------------

    def set_callback(self, fn: Callable[[DownloadItem], None]) -> None:
        self._cb = fn

    def set_options(self, quality: str | None = None, cookies: str | None = None,
                    poster_intro: bool | None = None, encoder: str | None = None) -> None:
        """Quality and poster choice for items added after this; login cookies
        and the encoder for downloads that start after this."""
        if quality is not None:
            self.quality = quality if quality in QUALITIES else "best"
        if cookies is not None:
            self.cookies = cookies or ""
        if poster_intro is not None:
            self.poster_intro = bool(poster_intro)
        if encoder is not None:
            self.encoder = encoder or "libx264"

    def set_out_dir(self, out_dir: str | None) -> None:
        """Change the download folder ("" / None = the default under
        app_data_dir()); applies to downloads that start after this."""
        self.out_dir = Path(out_dir) if out_dir else app_data_dir() / "downloads"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def find_duplicate(self, url: str) -> DownloadItem | None:
        """The queued item that points at the same video as url, if any."""
        key = _url_key(url)
        with self._lock:
            for item in self.items.values():
                if _url_key(item.url) == key:
                    return item
        return None

    def add(self, url: str, series: str = "", index: int = 0, *, title: str = "",
            quality: str = "", out_dir: str = "", poster_intro: bool | None = None,
            thumb_hint: str = "") -> DownloadItem:
        """Queue one link. Raises DuplicateUrlError if it is already in the
        list (in any state, including done / error). quality / poster_intro
        default to the queue's current options; out_dir "" saves into
        <queue folder>/<series>."""
        item = DownloadItem(
            id=uuid.uuid4().hex, url=url.strip(), title=title, series=series,
            index=int(index or 0),
            quality=quality if quality in QUALITIES else self.quality,
            out_dir=out_dir or "",
            poster_intro=self.poster_intro if poster_intro is None else bool(poster_intro),
            thumb_hint=thumb_hint or "")
        with self._lock:
            existing = self.find_duplicate(item.url)
            if existing is not None:
                raise DuplicateUrlError(existing)
            self.items[item.id] = item
            self._flags[item.id] = _ItemFlags()
        self._emit(item)
        self._submit(item.id)
        return item

    def add_many(self, urls: list[str], series: str = "", numbered: bool = False,
                 like: DownloadItem | None = None,
                 out_dir: str | None = None) -> list[DownloadItem]:
        """Queue several links; numbered=True names them EP001, EP002, …
        in the given order (one series folder). `like`: copy its quality,
        folder and poster choice (the videos of a playlist link); `out_dir`
        overrides the folder. Links already in the list are skipped: the
        result holds only the newly added items."""
        urls = [u for u in urls if u and u.strip()]
        extra = {} if like is None else {"quality": like.quality, "out_dir": like.out_dir,
                                         "poster_intro": like.poster_intro}
        if out_dir is not None:
            extra["out_dir"] = out_dir
        added = []
        for k, u in enumerate(urls, start=1):
            try:
                added.append(self.add(u, series, index=(k if numbered else 0), **extra))
            except DuplicateUrlError:
                pass
        return added

    def start(self) -> None:
        with self._lock:
            self._started = True
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="dl")
            pending, self._pending = self._pending, []
        for item_id in pending:
            self._submit(item_id)

    def pause(self, item_id: str) -> None:
        with self._lock:
            item = self.items.get(item_id)
            flags = self._flags.get(item_id)
        if not item or not flags:
            return
        # "processing" (adding the poster) is not paused: the video is complete
        if item.status == "downloading" or item.status == "fetching_meta":
            flags.pause.set()   # hook raises _Abort; .part file kept
        elif item.status == "queued":
            item.status = "paused"
            self._emit(item)

    def resume(self, item_id: str) -> None:
        self._reenqueue(item_id)

    def retry(self, item_id: str) -> None:
        with self._lock:
            item = self.items.get(item_id)
        if item:
            item.error = ""
        self._reenqueue(item_id)

    def cancel(self, item_id: str) -> None:
        with self._lock:
            item = self.items.get(item_id)
            flags = self._flags.get(item_id)
        if not item or not flags:
            return
        flags.cancel.set()
        if item.status in ("queued", "paused", "error", "done"):
            # not actively running: remove immediately
            self._remove(item_id)
        # if downloading, the hook aborts and the worker removes it

    def shutdown(self) -> None:
        """Stop workers (does not wait for running downloads to abort).
        Partial files are kept: the same link added after a restart resumes."""
        with self._lock:
            self._closing = True
            for flags in self._flags.values():
                flags.cancel.set()
            ex, self._executor = self._executor, None
            self._started = False
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)

    # -- internals ----------------------------------------------------------

    def _emit(self, item: DownloadItem) -> None:
        cb = self._cb
        if cb is not None:
            try:
                cb(item)
            except Exception:
                log.exception("download callback failed")

    def _remove(self, item_id: str) -> None:
        with self._lock:
            item = self.items.pop(item_id, None)
            self._flags.pop(item_id, None)
            written = self._written.pop(item_id, set())
            if item_id in self._pending:
                self._pending.remove(item_id)
        if item is not None:
            if item.status != "done" and not self._closing:
                # cancelled before it finished: its .part / stream files
                # are useless now (a finished video is never touched)
                _delete_partial_files(written)
            # "removed" (not "error"): the UI drops the row; an "error" row
            # would stay behind with dead R / X buttons, since the queue no
            # longer knows this id
            item.status = "removed"
            item.speed = ""
            item.eta = ""
            self._emit(item)

    def _reenqueue(self, item_id: str) -> None:
        with self._lock:
            item = self.items.get(item_id)
            if item is None:
                return
            # "queued" too: a second R / resume must not submit it twice, or
            # swap the flags a waiting worker is about to use
            if item.status in ("queued", "downloading", "fetching_meta", "done"):
                return
            self._flags[item_id] = _ItemFlags()  # fresh pause/cancel flags
            item.status = "queued"
            item.speed = ""
            item.eta = ""
        self._emit(item)
        self._submit(item_id)

    def _submit(self, item_id: str) -> None:
        with self._lock:
            if not self._started or self._executor is None:
                if item_id not in self._pending:
                    self._pending.append(item_id)
                return
            ex = self._executor
        try:
            ex.submit(self._process, item_id)
        except RuntimeError:
            with self._lock:
                if item_id not in self._pending:
                    self._pending.append(item_id)

    def _process(self, item_id: str) -> None:
        with self._lock:
            item = self.items.get(item_id)
            flags = self._flags.get(item_id)
            if item is None or flags is None or item.status != "queued":
                return
            # claim it: another submit of the same id (pause + resume while
            # it waited for a worker) now finds it taken and skips
            item.status = "fetching_meta"
        if flags.cancel.is_set():
            self._remove(item_id)
            return
        aborted = False
        try:
            self._run_item(item, flags)
        except _Abort:
            # handled below: until this block is left, the aborted yt-dlp
            # frames keep the .part file open and Windows cannot delete it
            aborted = True
        except Exception as exc:  # noqa: BLE001 - captured into item.error
            log.exception("download failed: %s", item.url)
            item.status = "error"
            why = HONGGUO_NOTE if _is_hongguo(item.url) else (exc, exc)
            item.error = (f"Download failed: {why[0]} / "
                          f"ការទាញយកបានបរាជ័យ: {why[1]}")
            item.speed = ""
            item.eta = ""
            self._emit(item)
        if aborted:
            if flags.cancel.is_set():
                self._remove(item.id)
            else:
                item.status = "paused"
                item.speed = ""
                item.eta = ""
                self._emit(item)

    def _run_item(self, item: DownloadItem, flags: _ItemFlags) -> None:
        try:
            import yt_dlp  # lazy heavy dep
        except ImportError as exc:
            raise AppError(
                f"yt-dlp is not installed. Run: {pip_install('yt-dlp')}",
                f"yt-dlp មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ: {pip_install('yt-dlp')}",
            ) from exc

        item.status = "fetching_meta"
        self._emit(item)

        info = self._fetch_meta(yt_dlp, item, flags)
        if info is None:
            return  # playlist was expanded (or item aborted)

        item.title = info.get("title") or item.title or item.url
        if not item.series:
            item.series = (info.get("series") or info.get("playlist_title")
                           or info.get("uploader") or "")
        self._emit(item)

        # the folder the user picked for this item, else <queue folder>/<series>
        target_dir = (Path(item.out_dir) if item.out_dir
                      else self.out_dir / _sanitize_dirname(item.series or "Misc"))
        target_dir.mkdir(parents=True, exist_ok=True)

        final_path: dict[str, str] = {}
        last_emit = [0.0]
        with self._lock:
            # kept across pause / retry: the .part files are resumed later
            written = self._written.setdefault(item.id, set())

        def hook(d: dict) -> None:
            if flags.cancel.is_set() or flags.pause.is_set():
                raise _Abort()
            for p in (d.get("tmpfilename"), d.get("filename")):
                if p and p not in written:
                    with self._lock:
                        written.add(p)
            st = d.get("status")
            if st == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                done = d.get("downloaded_bytes") or 0
                if total:
                    item.progress = max(0.0, min(1.0, done / total))
                item.speed = _fmt_speed(d.get("speed"))
                item.eta = _fmt_eta(d.get("eta"))
                item.status = "downloading"
                now = time.monotonic()
                if now - last_emit[0] >= 0.25:
                    last_emit[0] = now
                    self._emit(item)
            elif st == "finished":
                fn = d.get("filename") or ""
                if fn:
                    final_path["file"] = fn
                item.progress = 1.0
                item.speed = ""
                item.eta = ""
                self._emit(item)

        ffmpeg = ffmpeg_path()
        # the id keeps same-titled videos (common on Douyin / Shorts) from
        # overwriting each other; .120B keeps the path under MAX_PATH; an
        # episode number sorts a series in order
        name = "%(title).120B [%(id)s].%(ext)s"
        if item.index:
            name = f"EP{item.index:03d} - " + name.replace(".120B", ".100B")
        opts = {
            "outtmpl": str(target_dir / name),
            "format": QUALITIES.get(item.quality or self.quality, QUALITIES["best"]),
            "format_sort": list(FORMAT_SORT),
            **_js_runtime_opts(), **_cookie_opts(self.cookies),
            "continuedl": True,
            "noprogress": True,
            "quiet": True,
            "no_warnings": False,
            "color": "no_color",       # no "[0;31m" codes in error messages
            "logger": _YdlLogger(),
            "progress_hooks": [hook],
            "windowsfilenames": True,
            "retries": 3,
            "fragment_retries": 5,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "ignoreerrors": False,
        }
        # point yt-dlp at the bundled ffmpeg if present
        if Path(ffmpeg).is_absolute() and Path(ffmpeg).exists():
            opts["ffmpeg_location"] = str(bin_dir())

        item.status = "downloading"
        self._emit(item)
        LOG_BUS.publish(f"[yt-dlp] downloading: {item.url}")

        with yt_dlp.YoutubeDL(opts) as ydl:
            dl_info = self._download_from(yt_dlp, ydl, item, info, flags)
            # the progress hook sees each part (video .f137.mp4, audio
            # .f251.webm); the merged file is in requested_downloads
            if dl_info is not None:
                reqs = dl_info.get("requested_downloads") or []
                merged_path = reqs[0].get("filepath") if reqs else ""
                if merged_path and Path(merged_path).exists():
                    final_path["file"] = merged_path
                elif not final_path.get("file"):
                    try:
                        final_path["file"] = ydl.prepare_filename(dl_info)
                    except Exception:
                        pass

        filepath = final_path.get("file", "")
        # merge step may have changed the extension
        if filepath and not Path(filepath).exists():
            merged = Path(filepath).with_suffix(".mp4")
            if merged.exists():
                filepath = str(merged)
        item.filepath = filepath
        item.progress = 1.0

        if flags.cancel.is_set():
            self._remove(item.id)
            return

        # the poster goes inside the video: embedded as its cover art and,
        # if chosen, shown for the first second (no picture file beside it)
        if filepath:
            filepath = self._poster_step(item, dl_info or info, filepath, flags)
            item.filepath = filepath

        # probe duration / resolution (tolerate any failure)
        duration, width, height = 0.0, 0, 0
        if filepath:
            try:
                from app.engines.media_exporter import probe  # lazy
                mi = probe(filepath)
                duration, width, height = mi.duration, mi.width, mi.height
            except Exception as exc:
                log.warning("probe failed for %s: %s", filepath, exc)

        # register into the library
        if filepath:
            try:
                sid = self.library.add_series(item.series or "Misc")
                eid = self.library.add_episode(
                    sid, item.title or Path(filepath).stem, filepath,
                    duration=duration, width=width, height=height)
                if item.thumbnail:
                    self.library.update_episode(eid, thumbnail=item.thumbnail)
            except Exception as exc:
                log.warning("library registration failed: %s", exc)

        with self._lock:
            self._written.pop(item.id, None)  # finished: nothing to clean up
        item.status = "done"
        item.speed = ""
        item.eta = ""
        self._emit(item)
        LOG_BUS.publish(f"[yt-dlp] done: {item.title}")

    # sites whose stream links work only for the read that made them: YouTube
    # refused reused links (HTTP 403) in 4 of 5 tries (2026-11-03)
    _REREAD_EXTRACTORS = {"youtube"}

    @staticmethod
    def _download_from(yt_dlp, ydl, item: DownloadItem, info: dict, flags: _ItemFlags):
        """Download from the info _fetch_meta has just read, as yt-dlp's
        --load-info-json does, instead of reading the video page a second
        time (Bilibili: one read instead of two). A site that refuses those
        links gets one fresh read (yt-dlp's own fallback)."""
        if not (info.get("formats") or info.get("url")) or str(
                info.get("extractor_key") or "").lower() in DownloadQueue._REREAD_EXTRACTORS:
            return ydl.extract_info(item.url, download=True)
        utils = yt_dlp.utils
        try:
            return ydl.process_ie_result(info, download=True)
        except (utils.DownloadError, utils.ReExtractInfo) as exc:
            if flags.cancel.is_set() or flags.pause.is_set():
                raise
            if not isinstance(exc, utils.ReExtractInfo) and not re.search(
                    r"HTTP Error (401|403|404|410)|Forbidden|expired", str(exc), re.I):
                raise
            LOG_BUS.publish(f"[yt-dlp] the site refused the links read a moment ago "
                            f"({_ydl_error(exc)}); reading the video again")
            return ydl.extract_info(item.url, download=True)

    def _fetch_meta(self, yt_dlp, item: DownloadItem, flags: _ItemFlags):
        """extract_info(download=False). Returns info dict for a single video,
        or None if the URL was a playlist that got expanded into child items."""
        opts = {
            "quiet": True,
            "no_warnings": True,
            "logger": _YdlLogger(),
            "extract_flat": "in_playlist",
            "skip_download": True,
            # a video link that also carries &list= means that video; a
            # playlist link is still expanded
            "noplaylist": True,
            "color": "no_color",
            **_js_runtime_opts(), **_cookie_opts(self.cookies),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(item.url, download=False)
        if flags.cancel.is_set() or flags.pause.is_set():
            raise _Abort()
        if info is None:
            raise AppError(
                f"Could not read video info: {item.url}",
                f"មិនអាចអានព័ត៌មានវីដេអូបានទេ: {item.url}")

        entries = info.get("entries")
        if entries is not None:
            entries = [e for e in entries if e]
            playlist_title = info.get("title") or "Playlist"
            series = item.series or playlist_title
            child_urls = []
            for e in entries:
                u = e.get("webpage_url") or e.get("url") or ""
                if u:
                    child_urls.append(u)
            LOG_BUS.publish(
                f"[yt-dlp] playlist '{playlist_title}': "
                f"{len(child_urls)} videos queued")
            item.title = f"[Playlist] {playlist_title} ({len(child_urls)})"
            item.status = "done"
            item.progress = 1.0
            self._emit(item)
            # one folder, files numbered EP001… in playlist order, with the
            # playlist item's quality / poster choice. Picked folder without a
            # series name: the playlist gets its own folder in it (as the card)
            child_dir = (str(Path(item.out_dir) / _sanitize_dirname(series))
                         if item.out_dir and not item.series else None)
            self.add_many(child_urls, series=series, numbered=True, like=item,
                          out_dir=child_dir)
            return None
        return info

    def _poster_step(self, item: DownloadItem, info: dict, filepath: str,
                     flags: _ItemFlags) -> str:
        """Embed the poster as cover art and, with item.poster_intro, show it
        for the first second. The poster itself stays in the app's cache
        (the Library shows it), never beside the video. Returns the video's
        path afterwards. X during this stops it and leaves the downloaded
        video as it is."""
        item.status = "processing"
        item.progress = 0.0
        self._emit(item)
        if poster_done(filepath, item.poster_intro):
            # the same file downloaded before already has its poster inside
            LOG_BUS.publish(f"[poster] already in the video, not added again: {item.title}")
            cached = poster_cache_path(filepath)
            item.thumbnail = str(cached) if cached.is_file() else ""
            return filepath
        try:
            poster = make_poster(info, filepath, self.cookies, item.thumb_hint)
        except Exception as exc:     # never lose a finished download over its poster
            log.warning("poster failed for %s: %s", filepath, exc)
            poster = ""
        item.thumbnail = poster
        if not poster:
            LOG_BUS.publish(f"[poster] no poster found for: {item.title}")
            return filepath
        last = [0.0]

        def prog(frac: float, _text: str = "") -> None:
            item.progress = max(0.0, min(1.0, frac))
            now = time.monotonic()
            if now - last[0] >= 0.25:
                last[0] = now
                self._emit(item)

        try:
            return add_poster(filepath, poster, item.poster_intro, self.encoder,
                              progress=prog, cancel=flags.cancel)
        except CancelledError:
            raise _Abort() from None
        except Exception as exc:
            log.warning("adding the poster failed for %s: %s", filepath, exc)
            return filepath


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    db = tmp / "library_test.db"
    lib = Library(db)

    sid = lib.add_series("Test Drama", poster="p.jpg", meta={"year": 2026})
    sid2 = lib.add_series("Test Drama")  # dedup by name
    assert sid == sid2, "series dedup failed"

    eid = lib.add_episode(sid, "Ep 1", str(tmp / "ep1.mp4"),
                          duration=61.5, width=1920, height=1080)
    eid2 = lib.add_episode(sid, "Ep 1 (renamed)", str(tmp / "ep1.mp4"))
    assert eid == eid2, "episode dedup by path failed"

    lib.set_status(eid, transcribed=True, dubbed=True)
    eps = lib.get_episodes(sid)
    assert len(eps) == 1
    assert eps[0]["transcribed"] == 1 and eps[0]["dubbed"] == 1
    assert eps[0]["translated"] == 0

    found = lib.find_episode_by_path(str(tmp / "ep1.mp4"))
    assert found is not None and found["id"] == eid
    assert lib.find_episode_by_path("nope.mp4") is None

    series = lib.get_series()
    assert len(series) == 1 and series[0]["name"] == "Test Drama"
    assert series[0]["meta"]["year"] == 2026
    assert series[0]["episode_count"] == 1

    lib.update_episode(eid, thumbnail="thumb.jpg", duration=90.0)
    assert lib.get_episodes(sid)[0]["thumbnail"] == "thumb.jpg"

    lib.delete_episode(eid)
    assert lib.get_episodes(sid) == []
    lib.delete_series(sid)
    assert lib.get_series() == []
    lib.close()

    q = DownloadQueue(Library(tmp / "lib2.db"))
    it = q.add("https://example.com/video", series="S1")
    assert it.id in q.items and it.status == "queued"
    assert it.quality == "best" and not it.poster_intro and not it.out_dir
    q.cancel(it.id)
    assert it.id not in q.items
    q.set_options(quality="720", poster_intro=True)
    it = q.add("https://example.com/v2", out_dir=str(tmp / "pick"))
    assert it.quality == "720" and it.poster_intro and it.out_dir == str(tmp / "pick")
    kids = q.add_many(["https://example.com/a", "https://example.com/b"], "S", True, like=it)
    assert [k.index for k in kids] == [1, 2] and all(k.out_dir == it.out_dir for k in kids)
    kids = q.add_many(["https://example.com/c"], like=it, out_dir=str(tmp / "pick" / "PL"))
    assert kids[0].out_dir == str(tmp / "pick" / "PL") and kids[0].poster_intro
    q.shutdown()

    # links
    for good in ("https://www.youtube.com/watch?v=abc", "youtu.be/abcdefghijk",
                 "bilibili.com/video/BV1x", "http://x.co:8080/a?b=c#d"):
        assert looks_like_url(good), good
    for bad in ("hello world", "hello", "", "   ", "file.mp4 now"):
        assert not looks_like_url(bad), bad

    # episode numbers from titles
    cases = {
        "《不烧心》第12集": 12, "不烧心 第十二集": 12, "第一百零五集 大结局": 105,
        "第3話 出会い": 3, "EP12 The Return": 12, "Ep.05 - start": 5,
        "Episode 7: Night": 7, "S01E05 Pilot": 5, "ភាគទី១២ រឿងស្នេហ៍": 12,
        "Tập 8 - Phim": 8, "ตอนที่ 4": 4, "12화": 12, "P3 第三部分": 3,
        "全36集 第5集": 5, "【全36集】不烧心": None, "第1-3集 合集": 1,
        "不烧心 #14": 14, "不烧心 (9)": 9, "10. Title": 10, "Show name 23": 23,
        "直播回放 2024-05-12": None, "Trailer 4K": None, "Behind the scenes": None,
        "１２集": 12, "第二十集": 20, "DEEP 3 sleep": None,
        "老戴《黑神话 悟空》01 序章 【全流程攻略解说】": 1, "《不烧心》12": 12,
        "物语中的人物是如何吐槽自己的OP的 p04 恋爱循环/抚子+咩咩": 4,
        "老戴Vlog《黑神话悟空 艺术展》DLC 在做了": None, "Help 3D map": None,
        "老戴《 黑神话 悟空》 5场BOSS战解说": None, "【无剧透】老戴《黑神话：悟空》50小时游玩感受": None,
        "《不烧心》 3 大结局": 3, "5 个技巧": None,
    }
    for title, want in cases.items():
        got = parse_episode_number(title)
        assert got == want, (title, got, want)

    def mk(titles):
        return [ScanEntry(url=f"https://x/{k}", title=t, pl_index=k)
                for k, t in enumerate(titles, start=1)]

    ents = mk(["第3集", "第1集", "预告", "第2集", "第2集 (重制)", "第5集", "第4集"])
    assert number_episodes(ents) == "titles"                          # shuffled + extra + repeat
    assert [(e.ep, e.dup) for e in ents] == [(1, 0), (2, 0), (2, 1), (3, 0), (4, 0), (5, 0), (0, 0)]
    ents = mk(["Vlog", "Cooking", "Travel"])                         # no numbers
    assert number_episodes(ents) == "order" and [e.ep for e in ents] == [1, 2, 3]
    ents = mk(["Day 5", "Day 5 again", "Day 5 b", "x"])               # mostly the same number
    assert number_episodes(ents) == "order"

    # quality selectors: an image "format" is never picked (Weibo, 480p)
    import yt_dlp
    fake = [{"format_id": "scrubber", "ext": "jpg", "url": "u0", "width": 320, "height": 180},
            {"format_id": "720", "ext": "mp4", "url": "u1", "height": 720,
             "vcodec": "avc1", "acodec": "mp4a", "filesize": 1000},
            {"format_id": "1080", "ext": "mp4", "url": "u2", "height": 1080,
             "vcodec": "avc1", "acodec": "mp4a", "filesize": 2000}]
    with yt_dlp.YoutubeDL({"quiet": True, "format_sort": list(FORMAT_SORT)}) as y:
        for key, want in (("best", "1080"), ("1080", "1080"), ("720", "720"),
                          ("480", "720"), ("360", "720")):
            got = y._select_formats(fake, y.build_format_selector(QUALITIES[key]))
            assert got and got[-1]["format_id"] == want, (key, got)
        sizes = _sizes_by_quality(y, {"formats": fake, "duration": 10})
        assert sizes["480"] == 1000 and sizes["best"] == 2000, sizes

    # a moved cookies.txt: no cookie option, and one warning (not one per call)
    seen = []
    LOG_BUS.subscribe(seen.append)
    gone = "file:" + str(tmp / "moved" / "cookies.txt")
    assert cookie_file_missing(gone) and not cookie_file_missing("") \
        and not cookie_file_missing("browser:firefox")
    assert _cookie_opts(gone) == {} and _cookie_opts(gone) == {}
    assert len([s for s in seen if "cookies file is gone" in s]) == 1, seen
    real = tmp / "cookies.txt"
    real.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    assert _cookie_opts("file:" + str(real)) == {"cookiefile": str(real)}

    # Hongguo: the reason instead of yt-dlp's "Unsupported URL"
    assert _is_hongguo("https://www.hongguoduanju.com/detail?series_id=1")
    assert _is_hongguo("hongguoduanju.com/x") and not _is_hongguo("https://youtu.be/x")
    err = _scan_error("https://www.hongguoduanju.com/detail?series_id=1",
                      Exception("ERROR: Unsupported URL: https://www.hongguoduanju.com/"))
    assert "protected player" in err.en and "Unsupported URL" in err.en and "ភាគពេញ" in err.km

    # download from the info just read (no second read); a refused link: one fresh read
    from types import SimpleNamespace as NS

    class FakeYdl:
        def __init__(self, fail: str = ""):
            self.calls, self.fail = [], fail

        def process_ie_result(self, info, download=True):
            self.calls.append("reuse")
            if self.fail:
                raise yt_dlp.utils.DownloadError(self.fail)
            return dict(info, requested_downloads=[])

        def extract_info(self, url, download=True):
            self.calls.append("read again")
            return {"id": "x", "requested_downloads": []}

    it = DownloadItem(id="t", url="https://example.com/v")
    fl = NS(cancel=threading.Event(), pause=threading.Event())
    info = {"id": "x", "formats": [{"url": "u"}]}
    y = FakeYdl()
    DownloadQueue._download_from(yt_dlp, y, it, info, fl)
    assert y.calls == ["reuse"], y.calls
    y = FakeYdl("ERROR: unable to download video data: HTTP Error 403: Forbidden")
    DownloadQueue._download_from(yt_dlp, y, it, info, fl)
    assert y.calls == ["reuse", "read again"], y.calls
    y = FakeYdl("ERROR: Requested format is not available")
    try:
        DownloadQueue._download_from(yt_dlp, y, it, info, fl)
        raise AssertionError("other errors must not be retried")
    except yt_dlp.utils.DownloadError:
        assert y.calls == ["reuse"], y.calls
    y = FakeYdl()
    DownloadQueue._download_from(yt_dlp, y, it, {"id": "x"}, fl)   # no formats read: read it
    assert y.calls == ["read again"], y.calls
    y = FakeYdl()                                                   # YouTube: links tied to the read
    DownloadQueue._download_from(yt_dlp, y, it, dict(info, extractor_key="Youtube"), fl)
    assert y.calls == ["read again"], y.calls

    print("PASS")
