"""Shot / scene-cut detection and subject-aware framing.

detect_scenes()  ffmpeg's scene score on a down-scaled copy (fast), cached
                 per file in app_data_dir()/scenes
snap_start()     move a clip start onto a nearby cut so recaps never open
                 mid-shot or flash the tail of the previous shot
subject_x()      horizontal centre of the people in a time range (OpenCV
                 face detection, motion as a fallback) for 9:16 smart crops
crop_track()     one centre per shot, smoothed, for long-form reframing

No Qt imports; cv2 is lazy-imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.errors import CancelledError
from app.core.logger import get_logger
from app.core.paths import app_data_dir, ffmpeg_path

log = get_logger("scenes")

ProgressFn = Callable[[float, str], None]

_PTS = re.compile(r"pts_time:([0-9.]+)")


def _cache_file(video: str, threshold: float) -> Path:
    st = os.stat(video)
    key = hashlib.sha1(f"{os.path.abspath(video).lower()}|{st.st_size}|"
                       f"{int(st.st_mtime)}|{threshold:.3f}".encode()).hexdigest()[:16]
    d = app_data_dir() / "scenes"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def detect_scenes(video: str, threshold: float = 0.32, min_shot: float = 0.6,
                  progress: ProgressFn | None = None,
                  cancel: threading.Event | None = None) -> list[float]:
    """Sorted cut times (seconds). Shots shorter than `min_shot` are merged."""
    from app.engines.audio_pipeline import _stream_subprocess, get_duration
    try:
        cache = _cache_file(video, threshold)
        if cache.is_file():
            return [float(t) for t in json.loads(cache.read_text())]
    except OSError:
        cache = None
    try:
        duration = get_duration(video)
    except Exception:
        duration = 0.0
    cuts: list[float] = []

    def on_line(line: str) -> None:
        m = _PTS.search(line)
        if m and "showinfo" in line:
            t = float(m.group(1))
            cuts.append(t)
            if progress and duration > 0:
                progress(min(0.99, t / duration), f"Finding scene cuts… {len(cuts)}")

    cmd = [ffmpeg_path(), "-hide_banner", "-nostats", "-i", video, "-an", "-sn",
           "-vf", f"scale=320:-2,select='gt(scene,{threshold:.3f})',showinfo",
           "-f", "null", "-"]
    # _stream_subprocess publishes every line; scene runs are noisy, so keep
    # the label short
    rc = _stream_subprocess(cmd, label="scenes", cancel=cancel, on_line=on_line)
    if rc != 0 and not cuts:
        log.warning("scene detection failed (code %s) for %s", rc, video)
        return []
    out: list[float] = []
    for t in sorted(cuts):
        if not out or t - out[-1] >= min_shot:
            out.append(round(t, 3))
    if cache is not None:
        try:
            cache.write_text(json.dumps(out))
        except OSError:
            pass
    if progress:
        progress(1.0, f"{len(out)} scene cuts")
    return out


def shots(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    """[(start, end)] shots covering [0, duration]."""
    edges = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b - a > 1e-3]


def snap_start(start: float, length: float, cuts: list[float],
               before: float = 0.7, after: float = 1.0,
               tail_guard: float = 0.5) -> float:
    """Move a clip start onto a cut close to it:
    - a cut up to `after` s after the start (we would open on the end of the
      previous shot) or up to `before` s before it -> start at that cut;
    - otherwise, if a cut falls in the clip's last `tail_guard` s (a flash of
      the next shot), shift the clip earlier so it ends on that cut."""
    if not cuts:
        return start
    arr = np.asarray(cuts)
    near = arr[(arr >= start - before) & (arr <= start + after)]
    if near.size:
        return float(near[np.argmin(np.abs(near - start))]) + 0.04
    end = start + length
    tail = arr[(arr > end - tail_guard) & (arr < end)]
    if tail.size:
        shift = end - float(tail.min()) + 0.04
        prev = arr[arr <= start]
        floor = float(prev.max()) if prev.size else 0.0
        if start - shift >= floor:
            return start - shift
    return start


# ---------------------------------------------------------------------------
# subject-aware framing
# ---------------------------------------------------------------------------

_CASCADE = None
_CASCADE_LOCK = threading.Lock()


def _face_detector():
    global _CASCADE
    import cv2  # lazy heavy import
    with _CASCADE_LOCK:
        if _CASCADE is None:
            path = os.path.join(cv2.data.haarcascades,
                                "haarcascade_frontalface_default.xml")
            _CASCADE = cv2.CascadeClassifier(path)
        return _CASCADE


def _frames(cap, times: list[float]):
    import cv2  # lazy heavy import
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, fr = cap.read()
        if ok and fr is not None:
            yield t, fr


def subject_x(video: str, start: float, end: float, samples: int = 4,
              cap=None) -> float:
    """Horizontal centre (0..1) of the faces in [start, end]; weighted by face
    size. Falls back to the centroid of motion, then 0.5."""
    import cv2  # lazy heavy import
    own = cap is None
    if own:
        cap = cv2.VideoCapture(video)
    try:
        span = max(0.1, end - start)
        times = [start + span * (k + 0.5) / samples for k in range(samples)]
        det = _face_detector()
        xs, ws = [], []
        prev_gray, motion_x, motion_w = None, [], []
        for _t, fr in _frames(cap, times):
            h, w = fr.shape[:2]
            scale = 480.0 / max(1, w)
            small = cv2.resize(fr, (480, max(1, int(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = det.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                         minSize=(24, 24))
            for (fx, fy, fw, fh) in faces:
                xs.append((fx + fw / 2) / gray.shape[1])
                ws.append(float(fw * fh))
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
                col = diff.sum(axis=0)
                if col.sum() > 0:
                    motion_x.append(float((col * np.arange(col.size)).sum()
                                          / col.sum() / col.size))
                    motion_w.append(float(col.sum()))
            prev_gray = gray
        if xs:
            return float(np.average(xs, weights=ws))
        if motion_x:
            return float(np.average(motion_x, weights=motion_w))
        return 0.5
    except Exception as exc:
        log.debug("subject_x failed: %s", exc)
        return 0.5
    finally:
        if own:
            cap.release()


def crop_track(video: str, cuts: list[float], duration: float,
               progress: ProgressFn | None = None,
               cancel: threading.Event | None = None,
               max_shots: int = 400) -> list[tuple[float, float, float]]:
    """[(start, end, centre_x)] per shot. Very long lists are coarsened
    (neighbouring shots merged) so the crop expression stays small."""
    import cv2  # lazy heavy import
    sh = shots(cuts, duration)
    while len(sh) > max_shots:
        sh = [(sh[i][0], sh[min(i + 1, len(sh) - 1)][1]) for i in range(0, len(sh), 2)]
    cap = cv2.VideoCapture(video)
    out = []
    try:
        for i, (a, b) in enumerate(sh):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Operation cancelled")
            out.append((a, b, subject_x(video, a, b, samples=3, cap=cap)))
            if progress and i % 5 == 0:
                progress(i / max(1, len(sh)), f"Framing shot {i + 1}/{len(sh)}")
    finally:
        cap.release()
    # damp jitter: tiny moves between neighbouring shots are not worth a jump
    for i in range(1, len(out)):
        if abs(out[i][2] - out[i - 1][2]) < 0.04:
            out[i] = (out[i][0], out[i][1], out[i - 1][2])
    return out


def crop_x_expr(track: list[tuple[float, float, float]], src_w: int, crop_w: int) -> str:
    """ffmpeg crop `x` expression stepping through the shot centres."""
    def px(cx: float) -> int:
        return int(max(0, min(src_w - crop_w, cx * src_w - crop_w / 2)))
    if not track:
        return str(px(0.5))
    expr = str(px(track[-1][2]))
    for a, b, cx in reversed(track[:-1]):
        expr = f"if(lt(t,{b:.3f}),{px(cx)},{expr})"
    return expr


if __name__ == "__main__":
    # snapping
    cuts = [5.0, 9.0, 20.0]
    assert abs(snap_start(4.5, 3.0, cuts) - 5.04) < 1e-9, "cut just after start"
    assert abs(snap_start(5.5, 3.0, cuts) - 5.04) < 1e-9, "cut just before start"
    s = snap_start(6.5, 2.7, cuts)          # ends 9.2: flashes the next shot
    assert abs((s + 2.7) - 8.96) < 1e-9, s   # now ends just before the cut
    assert snap_start(12.0, 3.0, cuts) == 12.0, "no cut nearby"
    assert snap_start(3.0, 2.0, []) == 3.0
    assert shots([5.0, 9.0], 12.0) == [(0.0, 5.0), (5.0, 9.0), (9.0, 12.0)]
    e = crop_x_expr([(0, 5, 0.2), (5, 9, 0.8), (9, 12, 0.5)], 1920, 608)
    assert e.startswith("if(lt(t,5.000),80,if(lt(t,9.000),1232,656))"), e

    # real ffmpeg run on a synthetic clip with two hard cuts (at 2 s and 4 s)
    import subprocess
    import tempfile
    from app.engines.audio_pipeline import get_duration
    d = tempfile.mkdtemp()
    v = os.path.join(d, "cuts.mp4")
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=red:s=320x240:d=2:r=25",
                    "-f", "lavfi", "-i", "testsrc=s=320x240:d=2:r=25",
                    "-f", "lavfi", "-i", "color=blue:s=320x240:d=2:r=25",
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
                    "-pix_fmt", "yuv420p", v], check=True)
    found = detect_scenes(v)
    assert len(found) == 2 and abs(found[0] - 2.0) < 0.1 and abs(found[1] - 4.0) < 0.1, found
    assert detect_scenes(v) == found, "cached"
    x = subject_x(v, 2.0, 4.0)
    assert 0.0 <= x <= 1.0
    tr = crop_track(v, found, get_duration(v))
    assert len(tr) == 3 and tr[0][0] == 0.0
    print("PASS")
