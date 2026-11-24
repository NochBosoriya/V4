"""Burned-in (hard) subtitles: find the subtitle row of a video, read every
subtitle with its time on screen, and correct a speech-recognition
transcript from them.

Chinese short dramas nearly always carry their script as hard subtitles.
Speech recognition mishears homophones and near-homophones there (摩迪 for
摩的, 他 for 她, 领配 for 顶配, 静哥 for 镜头); the subtitles have the words
the actors say, so the transcribe step takes the words from them and keeps
the recogniser's timing.

OCR: PP-OCRv5 mobile (text detection + recognition; simplified and
traditional Chinese, Japanese, Latin letters) as ONNX on the CPU —
onnxruntime + OpenCV + numpy only. PaddleOCR's own package pulls in
scikit-learn, which Windows Smart App Control blocks on some PCs.

No Qt imports."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.errors import CancelledError
from app.core.logger import get_logger
from app.core.paths import ffmpeg_path, models_dir

log = get_logger("subtitle_ocr")

ProgressFn = Callable[[float, str], None]

# source languages whose subtitles this reader handles (the recogniser reads
# Chinese characters and kana; the transcript alignment is per character)
LANGS = {"zh", "ja"}

MODEL_ID = "subtitle-ocr"
_DIR = "subtitle_ocr"
# name: (bytes, sha256, mirrors)
FILES: dict[str, tuple[int, str, list[str]]] = {
    "det.onnx": (4819576, "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae", [
        "https://huggingface.co/breezedeus/cnstd-ppocr-ch_PP-OCRv5_det/resolve/main/"
        "ch_PP-OCRv5_det_infer.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/"
        "det/ch_PP-OCRv5_mobile_det.onnx"]),
    "rec.onnx": (16631306, "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5", [
        "https://huggingface.co/breezedeus/cnocr-ppocr-ch_PP-OCRv5/resolve/main/"
        "ch_PP-OCRv5_rec_infer.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/"
        "rec/ch_PP-OCRv5_rec_mobile_infer.onnx"]),
    "dict.txt": (74012, "d1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b", [
        "https://huggingface.co/webnn/PP-OCRv5-ONNX/resolve/main/ch_PP-OCRv5_dict.txt",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/paddle/PP-OCRv5/"
        "rec/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt"]),
}
SIZE_MB = 21
# bump when reading changes: cached subtitle lists of an older version are read again
_READ_VERSION = 1


def model_dir() -> Path:
    return models_dir() / _DIR


def models_present() -> bool:
    d = model_dir()
    return all((d / n).is_file() and (d / n).stat().st_size == size
               for n, (size, _h, _u) in FILES.items())


def available() -> bool:
    """The reader can run: models downloaded and OpenCV + onnxruntime load."""
    if not models_present():
        return False
    try:
        import cv2  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class SubEvent:
    start: float
    end: float
    text: str
    conf: float = 1.0


# ---------------------------------------------------------------------------
# OCR (PP-OCRv5: DB text detection + CTC recognition)
# ---------------------------------------------------------------------------

_OCR: "_Ocr | None" = None
_OCR_LOCK = threading.Lock()


class _Ocr:
    def __init__(self, threads: int = 0) -> None:
        import onnxruntime as ort
        d = model_dir()
        so = ort.SessionOptions()
        # more threads per reader did not help (12 read slower than 4 on a
        # 32-thread CPU); several readers side by side do (read_subtitles)
        so.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 2) // 2))
        so.log_severity_level = 3
        self.det = ort.InferenceSession(str(d / "det.onnx"), so,
                                        providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(d / "rec.onnx"), so,
                                        providers=["CPUExecutionProvider"])
        with open(d / "dict.txt", encoding="utf-8") as f:
            self.chars = ["<blank>"] + [ln.rstrip("\r\n") for ln in f] + [" "]
        self.lock = threading.Lock()

    def detect(self, img: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Text boxes (x0, y0, x1, y1, score) in `img` (BGR uint8)."""
        import cv2
        hh, ww = img.shape[:2]
        H, W = max(32, (hh + 31) // 32 * 32), max(32, (ww + 31) // 32 * 32)
        x = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        with self.lock:
            prob = self.det.run(None, {self.det.get_inputs()[0].name:
                                       x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        bm = (prob > 0.3).astype(np.uint8)
        cnts, _ = cv2.findContours(bm, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = ww / W, hh / H
        out = []
        for c in cnts:
            x0, y0, bw, bh = cv2.boundingRect(c)
            if bw < 4 or bh < 4:
                continue
            region = bm[y0:y0 + bh, x0:x0 + bw] > 0
            score = float(prob[y0:y0 + bh, x0:x0 + bw][region].mean())
            if score < 0.6:
                continue
            # DB "unclip": the map marks a shrunk text core
            dd = bw * bh * 1.5 / (2.0 * (bw + bh))
            out.append((int(max(0, x0 - dd) * sx), int(max(0, y0 - dd) * sy),
                        int(min(W, x0 + bw + dd) * sx), int(min(H, y0 + bh + dd) * sy), score))
        return out

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        import cv2
        hh, ww = crop.shape[:2]
        if hh < 4 or ww < 4:
            return "", 0.0
        nw = max(16, min(1600, int(round(ww * 48.0 / hh))))
        x = cv2.resize(crop, (nw, 48)).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        with self.lock:
            out = self.rec.run(None, {self.rec.get_inputs()[0].name:
                                      x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        idx = out.argmax(1)
        best = out.max(1)
        text, probs, last = [], [], 0
        for k, pr in zip(idx.tolist(), best.tolist()):
            if k != last and k != 0 and k < len(self.chars):
                text.append(self.chars[k])
                probs.append(pr)
            last = k
        return "".join(text), (float(np.mean(probs)) if probs else 0.0)


def _ocr() -> _Ocr:
    global _OCR
    with _OCR_LOCK:
        if _OCR is None:
            _OCR = _Ocr()
        return _OCR


def read_image(img: np.ndarray, min_conf: float = 0.5,
               max_side: int = 1280) -> list[tuple[int, int, int, int, str, float]]:
    """Every text box in a picture (BGR uint8), anywhere in it: (x0, y0, x1,
    y1, text, conf) in the picture's own pixels, the surest first. The
    posters' "Detect text" uses it when PaddleOCR cannot run (its package
    can be blocked by Smart App Control); this reader only needs
    onnxruntime + OpenCV. Text is found on a copy at most `max_side` px
    and read from the full-size picture."""
    import cv2
    hh, ww = img.shape[:2]
    s = min(1.0, max_side / float(max(hh, ww)))
    small = img if s >= 1.0 else cv2.resize(img, (max(32, int(ww * s)), max(32, int(hh * s))),
                                            interpolation=cv2.INTER_AREA)
    ocr = _ocr()
    out = []
    for x0, y0, x1, y1, _score in ocr.detect(small):
        X0, Y0 = max(0, int(x0 / s)), max(0, int(y0 / s))
        X1, Y1 = min(ww, int(round(x1 / s))), min(hh, int(round(y1 / s)))
        if X1 - X0 < 6 or Y1 - Y0 < 6:
            continue
        crop = img[Y0:Y1, X0:X1]
        if crop.shape[0] >= 1.5 * crop.shape[1]:
            # a vertical column of text (CJK posters): read it turned, as
            # PaddleOCR does
            crop = np.ascontiguousarray(np.rot90(crop))
        txt, conf = ocr.recognize(crop)
        txt = txt.strip()
        if txt and conf >= min_conf:
            out.append((X0, Y0, X1, Y1, txt, conf))
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * b[5], reverse=True)
    return out


# ---------------------------------------------------------------------------
# reading a video
# ---------------------------------------------------------------------------

_BAND_TOP = 0.5          # the subtitle search band: the lower half of the picture
_BAND_W = 960            # band width the OCR sees


def _probe_size(video: str) -> tuple[int, int]:
    from app.core.paths import ffprobe_path
    r = subprocess.run([ffprobe_path(), "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", video],
                       capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    w, h = (int(v) for v in r.stdout.strip().split(",")[:2])
    return w, h


def _frames(video: str, fps: float, bw: int, bh: int, cancel,
            start: float = 0.0, length: float = 0.0):
    """Yield (time, BGR band image) for the lower half of the picture; from
    `start` s for `length` s (0 = to the end)."""
    span = ((["-ss", f"{start:.3f}"] if start > 0 else [])
            + (["-t", f"{length:.3f}"] if length > 0 else []))
    cmd = [ffmpeg_path(), "-v", "error", "-nostdin", *span, "-i", video, "-an", "-sn", "-dn",
           "-vf", f"fps={fps},crop=iw:ih*{1 - _BAND_TOP:.3f}:0:ih*{_BAND_TOP:.3f},"
                  f"scale={bw}:{bh}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    size = bw * bh * 3
    k = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise CancelledError("cancelled")
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(bh, bw, 3)
            k += 1
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait()


@dataclass
class _Box:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    conf: float


def _line_text(boxes: list[_Box], bw: int) -> str:
    """Boxes of one text line, left to right; a wide gap becomes a space."""
    boxes = sorted(boxes, key=lambda b: b.x0)
    out = boxes[0].text
    for a, b in zip(boxes, boxes[1:]):
        gap = b.x0 - a.x1
        out += (" " if gap > 0.8 * (a.y1 - a.y0) else "") + b.text
    return out


def _rows(boxes: list[_Box]) -> list[list[_Box]]:
    """Group boxes into text lines (vertical overlap)."""
    rows: list[list[_Box]] = []
    for b in sorted(boxes, key=lambda b: (b.y0 + b.y1) / 2):
        cy = (b.y0 + b.y1) / 2
        for r in rows:
            r0 = min(x.y0 for x in r)
            r1 = max(x.y1 for x in r)
            if r0 <= cy <= r1:
                r.append(b)
                break
        else:
            rows.append([b])
    return rows


_CALIB_S = 90.0          # s of video used to find the subtitle row
_PARALLEL_MIN_S = 240.0  # a shorter video is read by one reader
_PART_MIN_S = 60.0       # each reader of a long video gets at least this much


def _readers() -> int:
    """Readers working side by side on a long video (each with 2 threads).
    60 s of a drama on a 32-thread CPU: one reader 17.5 s, 4 readers 8.4 s,
    8 readers 6.1 s."""
    return max(1, min(8, (os.cpu_count() or 4) // 4))


class _AnyOf:
    """Set when any of the events is: the caller's Cancel, or a reader that
    failed (the others stop instead of reading on for nothing)."""

    def __init__(self, *events) -> None:
        self.events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(e.is_set() for e in self.events)


def _part_spans(duration: float, fps: float, readers: int) -> list[tuple[float, float]]:
    """(start, length) of the parts read side by side after the first
    _CALIB_S, on the frame grid (no frame read twice or left out); the last
    part runs to the end (length 0). [] = one reader for the whole video."""
    if readers < 2 or duration < _PARALLEL_MIN_S:
        return []
    rest = duration - _CALIB_S
    n = int(min(readers, rest // _PART_MIN_S))
    if n < 2:
        return []
    frames = int(round(rest * fps))
    per = frames // n
    head = int(round(_CALIB_S * fps))
    spans = [((head + k * per) / fps, per / fps) for k in range(n)]
    spans[-1] = (spans[-1][0], 0.0)
    return spans


def _scan(frames, ocr: "_Ocr", bw: int, bh: int,
          strip: tuple[int, int] | None = None, end: float = 0.0, tick=None):
    """Read the subtitle band of each frame: (found, strip). found: per frame
    the centred text lines (y centre, height, text, conf) in band px; strip:
    the band rows of the subtitle row once known (found from the first
    _CALIB_S of video, then only that part of the band is read). `end` > 0:
    frames from there on belong to the next reader."""
    found: list[tuple[float, list[tuple[float, float, str, float]]]] = []
    prev_small = None
    prev_lines: list[tuple[float, float, str, float]] = []
    for t, img in frames:
        if end and t >= end - 1e-6:
            break
        if tick is not None:
            tick(t)
        y_off = 0
        view = img
        if strip is not None:
            y_off = strip[0]
            view = img[strip[0]:strip[1]]
        # unchanged picture in the subtitle area: same text as before
        small = view[:, ::4][::4].astype(np.int16)
        if prev_small is not None and small.shape == prev_small.shape and \
                float(np.abs(small - prev_small).mean()) < 1.5:
            found.append((t, prev_lines))
            continue
        prev_small = small
        boxes = []
        for x0, y0, x1, y1, _s in ocr.detect(view):
            if y1 - y0 < 8 or (x1 - x0) < (y1 - y0) * 0.8:
                continue
            txt, conf = ocr.recognize(view[y0:y1, x0:x1])
            txt = txt.strip()
            if txt and conf >= 0.6:
                boxes.append(_Box(x0, y0 + y_off, x1, y1 + y_off, txt, conf))
        lines = []
        for r in _rows(boxes):
            x0 = min(b.x0 for b in r)
            x1 = max(b.x1 for b in r)
            cx = (x0 + x1) / 2 / bw
            if abs(cx - 0.5) > 0.12:          # subtitles are centred
                continue
            y0 = min(b.y0 for b in r)
            y1 = max(b.y1 for b in r)
            conf = min(b.conf for b in r)
            lines.append(((y0 + y1) / 2, float(y1 - y0), _line_text(r, bw), conf))
        found.append((t, lines))
        prev_lines = lines
        if strip is None and t >= _CALIB_S:
            row = _subtitle_row(found)
            if row is not None:
                cy, rh = row
                strip = (max(0, int(cy - 2.6 * rh)), min(bh, int(cy + 1.4 * rh)))
                prev_small = None
    return found, strip


def read_subtitles(video: str, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None,
                   fps: float = 5.0, duration: float = 0.0,
                   readers: int = 0) -> list[SubEvent]:
    """Every hard subtitle of `video` with its time on screen; [] when the
    video has none (no steady text row near the bottom of the picture).

    A video of _PARALLEL_MIN_S or more (its `duration` given) is read by
    several readers at once (`readers`, 0 = _readers()): the first _CALIB_S
    by one, which finds the subtitle row, then the rest in parts side by
    side, each with its own OCR model; the frames come out the same."""
    w, h = _probe_size(video)
    bw = _BAND_W if w >= _BAND_W else w - w % 2
    bh = max(32, int(round(h * (1 - _BAND_TOP) * bw / w / 2)) * 2)
    spans = _part_spans(duration, fps, readers or _readers())
    lock = threading.Lock()
    read = [0]

    def tick(t: float) -> None:
        if progress is None or duration <= 0:
            return
        with lock:
            read[0] += 1
            n = read[0]
        if n % 25 == 0:
            done = n / fps if spans else t
            progress(min(0.99, done / duration),
                     f"Reading on-screen subtitles {done:.0f}/{duration:.0f} s")

    if not spans:
        found, _strip = _scan(_frames(video, fps, bw, bh, cancel), _ocr(), bw, bh, tick=tick)
    else:
        head = spans[0][0]
        found, strip = _scan(_frames(video, fps, bw, bh, cancel, 0.0, head + 1.0), _ocr(),
                             bw, bh, end=head, tick=tick)
        if strip is None:
            # one reader finds the row at its first frame after _CALIB_S, from
            # the frames before it: the same frames, so the same row
            row = _subtitle_row(found)
            if row is not None:
                cy, rh = row
                strip = (max(0, int(cy - 2.6 * rh)), min(bh, int(cy + 1.4 * rh)))
        stop = threading.Event()
        either = _AnyOf(cancel, stop)

        def part(span: tuple[float, float]):
            start, length = span
            try:
                got, _s = _scan(_frames(video, fps, bw, bh, either, start,
                                        length + 1.0 if length else 0.0),
                                _Ocr(threads=2), bw, bh, strip=strip,
                                end=start + length if length else 0.0, tick=tick)
                return got
            except BaseException:
                stop.set()
                raise

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(spans)) as ex:
            futures = [ex.submit(part, s) for s in spans]
        errors = [f.exception() for f in futures if f.exception() is not None]
        if errors:
            # the reader that failed, not the ones it stopped
            raise next((e for e in errors if not isinstance(e, CancelledError)), errors[0])
        for f in futures:
            found.extend(f.result())
        log.info("on-screen subtitles: %d readers side by side after the first %.0f s",
                 len(spans), head)
    row = _subtitle_row(found)
    if row is None:
        log.info("no hard subtitles found in %s", os.path.basename(video))
        return []
    cy, rh = row
    per_frame: list[tuple[float, str, float]] = []
    for t, lines in found:
        keep = [ln for ln in lines if cy - 1.6 * rh <= ln[0] <= cy + 0.7 * rh
                and 0.6 * rh <= ln[1] <= 1.6 * rh]
        keep.sort(key=lambda ln: ln[0])               # a two-line subtitle: top line first
        text = "".join(ln[2] for ln in keep)
        conf = min((ln[3] for ln in keep), default=0.0)
        per_frame.append((t, text, conf))
    # a sign or a caption read on the row by chance reads less sure than
    # subtitle lettering
    events = [e for e in _events(per_frame, fps) if e.conf >= 0.8]
    # a channel name / watermark sitting on the subtitle row the whole time
    on = sum(e.end - e.start for e in events)
    stay = Counter()
    for e in events:
        stay[e.text] += e.end - e.start
    events = [e for e in events if not (on > 60 and stay[e.text] > 0.3 * on)]
    log.info("%d hard subtitles read (row at %.0f%% of the picture height)", len(events),
             100 * (_BAND_TOP + (1 - _BAND_TOP) * cy / bh))
    return events


def _subtitle_row(found) -> tuple[float, float] | None:
    """(y centre, height) in band px of the row where changing, centred text
    keeps appearing; None when there is no such row."""
    items = [(ln[0], ln[1], ln[2]) for _t, lines in found for ln in lines
             if len(re.sub(r"\s", "", ln[2])) >= 2]
    if len(items) < 8:
        return None
    count = Counter(it[2] for it in items)
    frames = max(1, len(found))
    # text that never changes (a logo, a channel name) is not a subtitle
    items = [it for it in items if count[it[2]] < 0.5 * frames]
    if len(items) < 8:
        return None
    hs = np.array([it[1] for it in items])
    ys = np.array([it[0] for it in items])
    step = max(2.0, float(np.median(hs)) * 0.5)
    best, best_n = None, 0
    for y in ys:
        near = np.abs(ys - y) <= step
        if near.sum() > best_n:
            best, best_n = y, int(near.sum())
    near = np.abs(ys - best) <= step
    texts = {items[k][2] for k in np.nonzero(near)[0]}
    if len(texts) < 4 or best_n < 0.05 * frames:
        return None
    return float(np.median(ys[near])), float(np.median(hs[near]))


def _events(per_frame: list[tuple[float, str, float]], fps: float) -> list[SubEvent]:
    """Frames showing (nearly) the same text become one subtitle; its text
    is the reading seen most (OCR flickers by a character now and then)."""
    half = 0.5 / fps
    out: list[SubEvent] = []
    run: list[tuple[float, str, float]] = []

    def close() -> None:
        if not run:
            return
        votes: Counter = Counter()
        for _t, txt, c in run:
            votes[txt] += c
        text = votes.most_common(1)[0][0]
        start, end = run[0][0] - half, run[-1][0] + half
        if end - start >= 0.25 or len(run) >= 2:
            out.append(SubEvent(round(max(0.0, start), 2), round(end, 2), text,
                                round(max(c for _t, txt, c in run if txt == text), 3)))
        run.clear()

    for t, text, conf in per_frame:
        norm = _norm(text)
        if not norm:
            close()
            continue
        if run:
            last = _norm(run[-1][1])
            if SequenceMatcher(None, last, norm, autojunk=False).ratio() < 0.7:
                close()
        run.append((t, text, conf))
    close()
    return out


def _norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text or "", flags=re.UNICODE).lower()


def cached_read(video: str, cache_path: str, progress: ProgressFn | None = None,
                cancel: threading.Event | None = None,
                duration: float = 0.0) -> list[SubEvent]:
    """read_subtitles, kept in `cache_path` for the same video file."""
    st = os.stat(video)
    key = hashlib.sha1(f"{os.path.abspath(video).lower()}|{st.st_size}|{int(st.st_mtime)}|"
                       f"{_READ_VERSION}".encode("utf-8")).hexdigest()
    try:
        with open(cache_path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("key") == key:
            return [SubEvent(**e) for e in d.get("events") or []]
    except (OSError, ValueError, TypeError):
        pass
    events = read_subtitles(video, progress, cancel, duration=duration)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"key": key, "events": [asdict(e) for e in events]}, f,
                  ensure_ascii=False, indent=0)
    os.replace(tmp, cache_path)
    return events


# ---------------------------------------------------------------------------
# correcting a transcript from the subtitles
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[\s\W_]", flags=re.UNICODE)


def _units(text: str) -> list[str]:
    """Characters of `text` with trailing punctuation attached (\"的？\");
    a Latin word or number stays one unit (\"H2\")."""
    out: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text or ""):
        tok = m.group(0)
        if _PUNCT.fullmatch(tok):
            if out:
                out[-1] += tok
            continue
        out.append(tok)
    return out


# The reader takes the numeral 一 (one flat stroke) at the start of a
# subtitle for a dash ("-个小时前" for 一个小时前, "-瓶30万"); it counts as 一
# only where the speech has one (a dash inside a line is punctuation:
# "我靠-你怎么全区第一了").
_LEAD_DASH = re.compile(r"^\s*[-－—–](?![-－—–])(?=[一-鿿])")


def _ocr_text(text: str, lang: str) -> tuple[str, bool]:
    """(text to align, whether its first character is a dash read as 一)."""
    m = _LEAD_DASH.match(text or "") if lang == "zh" else None
    return ("一" + text[m.end():], True) if m else (text or "", False)


def _capped(start: float, end: float, text: str, first: bool) -> tuple[float, float]:
    """A recognised word's time without a stretch across silence or music
    (Deepgram gave one 了 7.7 s, too far from its subtitle to be matched): at
    most 0.8 s + 0.3 s a character; the first word of a segment loses its
    start, any other word its end (dub_engine._unstretch's rule)."""
    cap = 0.8 + 0.3 * len(text.strip())
    if end - start > cap:
        return (end - cap, end) if first else (start, start + cap)
    return start, end


_PY_CACHE: dict[str, str] = {}


def _sound(u: str, lang: str) -> str:
    """Pinyin without tone for a Chinese character (traditional too); the
    lower-case unit otherwise."""
    core = _PUNCT.sub("", u).lower()
    if lang != "zh" or not core or not ("一" <= core[0] <= "鿿"):
        return core
    if core in _PY_CACHE:
        return _PY_CACHE[core]
    try:
        from pypinyin import lazy_pinyin
        s = "".join(lazy_pinyin(core))
    except Exception:
        s = core
    _PY_CACHE[core] = s
    return s


def _initial_final(py: str) -> tuple[str, str]:
    for ini in ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
                "j", "q", "x", "r", "z", "c", "s", "y", "w"):
        if py.startswith(ini):
            return ini, py[len(ini):]
    return "", py


def _sub_cost(a: str, b: str, lang: str) -> float:
    ca, cb = _PUNCT.sub("", a).lower(), _PUNCT.sub("", b).lower()
    if ca == cb:
        return 0.0
    sa, sb = _sound(a, lang), _sound(b, lang)
    if sa and sa == sb:
        return 0.15                    # homophone (他/她, 迪/的) or the same char written another way
    if lang == "zh" and sa and sb:
        ia, fa = _initial_final(sa)
        ib, fb = _initial_final(sb)
        flat = {"zh": "z", "ch": "c", "sh": "s"}
        if fa == fb or flat.get(ia, ia) == flat.get(ib, ib) and \
                fa.rstrip("g") == fb.rstrip("g"):
            return 0.55                # near-homophone (领/顶, 开/拍)
    return 0.85


_GAP = 0.6
_BAND = 2.5        # s: an OCR character is matched only with speech this close in time


@dataclass
class _U:
    text: str
    start: float
    end: float
    seg: int = -1          # ASR segment
    ev: int = -1           # subtitle event
    speaker: Any = None
    dash: bool = False     # a subtitle's leading dash, maybe the numeral 一


def _align(asr: list[_U], ocr: list[_U], lang: str) -> list[tuple[int | None, int | None]]:
    """Banded edit-distance alignment of recognised characters with
    subtitle characters (times within _BAND seconds). Returns pairs
    (asr index | None, ocr index | None) in order."""
    n, m = len(asr), len(ocr)
    to = [(u.start + u.end) / 2 for u in ocr]
    ta = [(u.start + u.end) / 2 for u in asr]
    import bisect
    # row i = the first i recognised characters done: subtitle characters
    # well before the last of them must be done too, and any up to the next
    # one (plus the band) may be
    lo = [0] * (n + 1)
    hi = [0] * (n + 1)
    for i in range(n + 1):
        lo[i] = bisect.bisect_left(to, ta[i - 1] - _BAND) if i > 0 else 0
        hi[i] = bisect.bisect_right(to, ta[i] + _BAND) if i < n else m
    for i in range(1, n + 1):
        hi[i] = max(hi[i], hi[i - 1])
        lo[i] = min(max(lo[i], lo[i - 1]), hi[i - 1])
    INF = float("inf")
    cost: list[dict[int, float]] = []
    back: list[dict[int, int]] = []           # 0 diag, 1 up (asr only), 2 left (ocr only)
    for i in range(n + 1):
        row: dict[int, float] = {}
        bk: dict[int, int] = {}
        prev = cost[i - 1] if i > 0 else None
        for j in range(lo[i], hi[i] + 1):
            best, how = INF, -1
            if i == 0 and j == 0:
                best, how = 0.0, 0
            if i > 0 and j > 0 and prev is not None and (j - 1) in prev:
                dt = abs(ta[i - 1] - to[j - 1])
                if dt <= _BAND:
                    # far apart in time: rather not the same character
                    c = prev[j - 1] + _sub_cost(asr[i - 1].text, ocr[j - 1].text, lang) \
                        + 0.3 * max(0.0, dt - 0.6)
                    if c < best:
                        best, how = c, 0
            if i > 0 and prev is not None and j in prev:
                c = prev[j] + _GAP
                if c < best:
                    best, how = c, 1
            if j > 0 and (j - 1) in row:
                c = row[j - 1] + _GAP
                if c < best:
                    best, how = c, 2
            if how >= 0:
                row[j] = best
                bk[j] = how
        cost.append(row)
        back.append(bk)
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    if j not in back[i]:
        j = max(back[i])
        pairs.extend((None, k) for k in range(m - 1, j - 1, -1))
    while i > 0 or j > 0:
        how = back[i].get(j, 1 if i > 0 else 2)
        if how == 0 and i > 0 and j > 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif how == 1 and i > 0:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


@dataclass
class ApplyStats:
    matched: int = 0           # recognised characters confirmed by the subtitles
    changed: int = 0           # recognised characters replaced by the subtitle's
    added: int = 0             # subtitle characters the recogniser missed
    dropped: int = 0           # recognised characters not in the subtitles
    coverage: float = 0.0      # share of subtitle characters matched with speech
    used: bool = False         # the subtitles were applied
    examples: list[str] | None = None


def apply_to_segments(segments: list, events: list[SubEvent], lang: str,
                      voiced: Callable[[float, float], float] | None = None,
                      word_cls: Any = None, seg_cls: Any = None) -> tuple[list, ApplyStats]:
    """Segments whose words carry the subtitle text and the recogniser's
    timing; a segment also ends where one subtitle gives way to the next
    (a turn of speakers nearly always does). Unchanged when the subtitles do
    not transcribe this speech (too little of them matches).

    A subtitle nobody was heard saying becomes a line of its own at its
    on-screen time when `voiced(start, end)` (the share of that time with a
    voice in it) says someone speaks there — the recogniser missed it; a
    title card or a caption over silence is left out."""
    lang = (lang or "").split("-")[0].lower()
    st = ApplyStats(examples=[])
    if not events or not segments:
        return segments, st
    if word_cls is None or seg_cls is None:
        from app.engines.audio_pipeline import Segment, Word
        word_cls, seg_cls = word_cls or Word, seg_cls or Segment
    asr: list[_U] = []
    for si, seg in enumerate(segments):
        words = [w for w in (getattr(seg, "words", None) or []) if str(w.text).strip()]
        if not words:
            us = _units(str(seg.text))
            dur = (float(seg.end) - float(seg.start)) / max(1, len(us))
            asr += [_U(u, float(seg.start) + k * dur, float(seg.start) + (k + 1) * dur, si,
                       speaker=getattr(seg, "speaker", None)) for k, u in enumerate(us)]
            continue
        for wi, w in enumerate(words):
            us = _units(str(w.text))
            if not us:
                if asr and asr[-1].seg == si:      # punctuation word: onto the last unit
                    asr[-1].text += str(w.text).strip()
                continue
            ws, we = _capped(float(w.start), float(w.end), str(w.text), wi == 0)
            dur = (we - ws) / len(us)
            asr += [_U(u, ws + k * dur, ws + (k + 1) * dur, si,
                       speaker=getattr(w, "speaker", None)) for k, u in enumerate(us)]
    ocr: list[_U] = []
    for ei, e in enumerate(events):
        text, dash = _ocr_text(e.text, lang)
        us = _units(text)
        dur = (e.end - e.start) / max(1, len(us))
        ocr += [_U(u, e.start + k * dur, e.start + (k + 1) * dur, ev=ei, dash=dash and k == 0)
                for k, u in enumerate(us)]
    if not asr or not ocr:
        return segments, st
    pairs = _align(asr, ocr, lang)
    good = sum(1 for a, o in pairs if a is not None and o is not None
               and _sub_cost(asr[a].text, ocr[o].text, lang) <= 0.15)
    st.coverage = round(good / max(1, len(ocr)), 3)
    if st.coverage < 0.45:
        log.info("subtitles do not transcribe the speech (%.0f%% match); transcript kept",
                 100 * st.coverage)
        return segments, st
    # which subtitle events have speech matched to them at all; one nobody
    # was heard saying counts only with a voice under it
    ev_spoken = Counter(ocr[o].ev for a, o in pairs if a is not None and o is not None)
    silent = {ei for ei, e in enumerate(events) if not ev_spoken.get(ei)
              and (voiced is None or voiced(e.start, e.end) < 0.35)}
    # the same subtitle read twice (下注 then 下註, or half of it while it
    # fades in): nobody said the copy, so it is no line of its own
    sounds = [[_sound(u, lang) for u in _units(_ocr_text(e.text, lang)[0])] for e in events]

    def within(part: list[str], whole: list[str]) -> bool:
        n = len(part)
        return 0 < n <= len(whole) and any(whole[k:k + n] == part for k in range(len(whole) - n + 1))

    for ei, e in enumerate(events):
        if ev_spoken.get(ei) or ei in silent:
            continue
        for nj in (ei - 1, ei + 1):
            o = events[nj] if 0 <= nj < len(events) else None
            if (o is not None and ev_spoken.get(nj) and max(o.start, e.start) - min(o.end, e.end) <= 0.3
                    and within(sounds[ei], sounds[nj])):
                silent.add(ei)
                log.info("subtitle %r read again as %r: the copy is left out", o.text, e.text)
                break
    if silent:
        pairs = [(a, o) for a, o in pairs if o is None or ocr[o].ev not in silent]
    out: list[_U] = []
    pending: list[_U] = []                   # subtitle characters waiting for a time

    def on_screen(t: float) -> bool:
        k = _event_at(events, t)
        return k is not None and k not in silent

    for a, o in pairs:
        if o is not None and ocr[o].dash and (a is None or _sound(asr[a].text, lang) != "yi"):
            # the leading dash was punctuation (nobody said 一 there); a word
            # heard in its place is under the subtitle: dropped like others
            if a is not None:
                st.dropped += 1
            continue
        if a is not None and o is not None:
            ua, uo = asr[a], ocr[o]
            u = _U(uo.text, ua.start, ua.end, ua.seg, uo.ev, ua.speaker)
            if _PUNCT.sub("", ua.text).lower() == _PUNCT.sub("", uo.text).lower():
                st.matched += 1
            else:
                st.changed += 1
                if len(st.examples) < 8:
                    st.examples.append(f"{_PUNCT.sub('', ua.text)}→{_PUNCT.sub('', uo.text)}")
            if pending:
                _place(pending, out, u, lang)
                pending = []
            out.append(u)
        elif a is not None:
            ua = asr[a]
            mid = (ua.start + ua.end) / 2
            if on_screen(mid):
                st.dropped += 1                   # the subtitle has the words said here
                continue
            u = _U(ua.text, ua.start, ua.end, ua.seg, -1, ua.speaker)
            if pending:
                _place(pending, out, u, lang)
                pending = []
            out.append(u)
        else:
            uo = ocr[o]
            st.added += 1
            pending.append(_U(uo.text, uo.start, uo.end, -1, uo.ev, None))
    if pending:
        _place(pending, out, None, lang)
    # a subtitle nobody was heard saying keeps its on-screen time
    for ei, e in enumerate(events):
        if not ev_spoken.get(ei):
            for u in out:
                if u.ev == ei and u.seg < 0:
                    u.seg = -2 - ei
    st.used = True
    segs = _rebuild(out, word_cls, seg_cls)
    if lang == "zh":
        segs = _drop_lone_particles(segs)
    return segs, st


# sentence-final particles that are never a line by themselves
_LONE = set("了吗呢吧嘛呗咯嘞啦")


def _drop_lone_particles(segs: list) -> list:
    """Recognised words outside the subtitles that are only a particle,
    next to a subtitle (≤ 0.5 s): that subtitle's sentence end heard again
    (傻了吧 heard as 傻了罢了 left a line "了")."""
    subs = [s for s in segs if getattr(s, "subtitle", False)]
    out = []
    for s in segs:
        core = _PUNCT.sub("", str(s.text))
        if (not getattr(s, "subtitle", False) and core and all(c in _LONE for c in core)
                and any(max(t.start - s.end, s.start - t.end) <= 0.5 for t in subs)):
            log.info("lone particle %r at %.1f s next to a subtitle: left out", s.text, s.start)
            continue
        out.append(s)
    return out


def _event_at(events: list[SubEvent], t: float) -> int | None:
    for k, e in enumerate(events):
        if e.start - 0.15 <= t <= e.end + 0.15:
            return k
    return None


_CHAR_S = 0.18      # s a character the recogniser missed is given, when there is room


_JOIN = {"zh": "，", "ja": "、"}
_ENDS_PUNCT = re.compile(r"[\s\W_]$", flags=re.UNICODE)


def _place(pending: list[_U], out: list[_U], nxt: _U | None, lang: str = "") -> None:
    """Give subtitle characters the recogniser missed a time. Those of the
    subtitle just said go right after it (into the pause when there is one,
    else sharing that character's time); those of the subtitle about to be
    said go right before it. A subtitle the recogniser missed altogether
    keeps its on-screen time; a short one (≤ 4 characters) with no time of
    its own there joins the words its on-screen time overlaps most ("呵",
    sorted by time alone, had landed between 当 and 然; "肖少" + "何必这么大的
    火气": one line, "肖少，何必这么大的火气")."""
    groups: list[list[_U]] = []
    for u in pending:
        if groups and groups[-1][0].ev == u.ev:
            groups[-1].append(u)
        else:
            groups.append([u])
    for g in groups:
        prev = out[-1] if out else None
        k = len(g)
        if prev is not None and prev.ev == g[0].ev and prev.seg >= 0:
            room_end = nxt.start if nxt is not None else prev.end + k * _CHAR_S
            if room_end - prev.end >= k * _CHAR_S * 0.6:
                t0 = prev.end
                dur = min(_CHAR_S, (room_end - prev.end) / k)
            else:                               # share the last character's time
                dur = (prev.end - prev.start) / (k + 1)
                t0 = prev.start + dur
                prev.end = t0
            for n_, u in enumerate(g):
                u.start, u.end = t0 + n_ * dur, t0 + (n_ + 1) * dur
                u.seg, u.speaker = prev.seg, prev.speaker
        elif nxt is not None and nxt.ev == g[0].ev and nxt.seg >= 0:
            room_start = prev.end if prev is not None else nxt.start - k * _CHAR_S
            if nxt.start - room_start >= k * _CHAR_S * 0.6:
                dur = min(_CHAR_S, (nxt.start - room_start) / k)
                t0 = nxt.start - k * dur
            else:                               # share the next character's time
                dur = (nxt.end - nxt.start) / (k + 1)
                t0 = nxt.start
                nxt.start = t0 + k * dur
            for n_, u in enumerate(g):
                u.start, u.end = t0 + n_ * dur, t0 + (n_ + 1) * dur
                u.seg, u.speaker = nxt.seg, nxt.speaker
        else:
            os_, oe_ = g[0].start, g[-1].end            # its on-screen time
            # the words heard around it (unless reading order put them on
            # the other side of it in time)
            lo = prev.end if prev is not None and prev.start < oe_ else 0.0
            hi = nxt.start if nxt is not None and nxt.end > os_ else float("inf")

            def overlap(v: _U | None) -> float:
                return -1.0 if v is None else min(oe_, v.end) - max(os_, v.start)

            joint = _JOIN.get(lang, " ")
            # with room of its own it keeps its on-screen time (a little
            # overlap is harmless: a subtitle's characters stay together)
            free = min(oe_, hi) - max(os_, lo)
            if free < k * _CHAR_S * 0.6 and k <= 4 and max(overlap(prev), overlap(nxt)) > 0:
                # a short one (a name, "呵") said over the words next to it
                if nxt is not None and overlap(nxt) >= overlap(prev):
                    # said with the next words: shares their first character's time
                    t0 = max(lo, nxt.start - k * _CHAR_S)
                    dur = max(0.01, (nxt.end - t0) / (k + 1))
                    for n_, u in enumerate(g):
                        u.start, u.end = t0 + n_ * dur, t0 + (n_ + 1) * dur
                        u.ev, u.seg, u.speaker = nxt.ev, nxt.seg, nxt.speaker
                    nxt.start = t0 + k * dur
                    if not _ENDS_PUNCT.search(g[-1].text):
                        g[-1].text += joint
                elif prev is not None:
                    # said with the words before: shares their last character's time
                    t1 = min(hi, prev.end + k * _CHAR_S)
                    dur = max(0.01, (t1 - prev.start) / (k + 1))
                    prev.end = prev.start + dur
                    for n_, u in enumerate(g):
                        u.start, u.end = prev.end + n_ * dur, prev.end + (n_ + 1) * dur
                        u.ev, u.seg, u.speaker = prev.ev, prev.seg, prev.speaker
                    if not _ENDS_PUNCT.search(prev.text):
                        prev.text += joint
        out.extend(g)


def _rebuild(units: list[_U], word_cls: Any, seg_cls: Any) -> list:
    """Units (in reading order) → segments: one segment per subtitle — a
    drama's subtitle is one speaker's words (the recogniser cut "阿姨" |
    "坐我这" apart and the halves went to two people), its characters kept
    together in their order and marked `subtitle` — and, outside subtitles,
    the recogniser's own segments. Segments are in time order."""
    groups: list[list[_U]] = []
    by_ev: dict[int, list[_U]] = {}
    for u in units:
        if not u.text:
            continue
        if u.ev >= 0:
            if u.ev not in by_ev:
                by_ev[u.ev] = []
                groups.append(by_ev[u.ev])
            by_ev[u.ev].append(u)
        elif groups and groups[-1][0].ev < 0 and groups[-1][-1].seg == u.seg:
            groups[-1].append(u)
        else:
            groups.append([u])
    groups.sort(key=lambda g: (min(u.start for u in g), max(u.end for u in g)))
    segs = []
    for g in groups:
        words = [word_cls(start=round(u.start, 3), end=round(max(u.end, u.start + 0.01), 3),
                          text=u.text, speaker=u.speaker) for u in g]
        spk = next((u.speaker for u in g if u.speaker), None)
        segs.append(seg_cls(start=min(w.start for w in words), end=max(w.end for w in words),
                            text="".join(u.text for u in g), words=words, speaker=spk,
                            subtitle=g[0].ev >= 0))
    return segs
