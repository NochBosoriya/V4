"""Audio pipeline: extraction, vocal separation, transcription, gender
detection, multi-engine TTS, voice-track assembly and ducked mixing.

Contract module — see ARCHITECTURE.md ("app/engines/audio_pipeline.py").
No Qt imports. Heavy deps (faster_whisper, demucs, edge_tts, torch, local TTS
engines) are lazy-imported inside functions/methods.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

from app.core.errors import AppError, CancelledError, EngineNotAvailable, ProviderError
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import app_data_dir, ffmpeg_path, ffprobe_path, models_dir, pip_install

log = get_logger("audio_pipeline")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

TTS_ENGINES = ["edge", "voxcpm", "f5tts", "cosyvoice", "openai", "elevenlabs"]

_DEFAULT_EDGE_VOICE = "km-KH-SreymomNeural"
_TTS_SR = 24000  # canonical sample rate for all synthesized speech


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str | None = None   # speaker label from the recogniser (Deepgram)

    def to_json(self) -> dict:
        d = {"start": self.start, "end": self.end, "text": self.text}
        if self.speaker:
            d["speaker"] = self.speaker
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Word":
        return cls(start=float(d.get("start", 0.0)),
                   end=float(d.get("end", 0.0)),
                   text=str(d.get("text", "")),
                   speaker=d.get("speaker") or None)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]
    speaker: str | None = None
    gender: str | None = None  # "M" | "F"
    # one hard subtitle's words (subtitle_ocr): one line, one speaker — the
    # dub keeps it whole; not saved
    subtitle: bool = False
    # why a person should check this segment ("" = nothing to check):
    # "recovered" — heard only by the second listen over a gap the first
    # pass left (recover_gaps); "low_confidence" — Whisper itself doubted it
    review: str = ""

    def to_json(self) -> dict:
        d = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_json() for w in self.words],
            "speaker": self.speaker,
            "gender": self.gender,
        }
        if self.review:
            d["review"] = self.review
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Segment":
        return cls(
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            text=str(d.get("text", "")),
            words=[Word.from_json(w) for w in (d.get("words") or [])],
            speaker=d.get("speaker"),
            gender=d.get("gender"),
            review=str(d.get("review") or ""),
        )


def _fmt_srt_time(t: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    if t < 0:
        t = 0.0
    ms_total = int(round(t * 1000.0))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    duration: float
    # second listen (recover_gaps): segments it added, and the stretches of
    # speech nobody could recognise — (start, end, reason, text heard) — so a
    # sentence the recogniser lost is reported, never silently gone
    recovered: int = 0
    unheard: list = field(default_factory=list)

    def to_srt(self) -> str:
        blocks: list[str] = []
        for i, seg in enumerate(self.segments, start=1):
            text = (seg.text or "").strip()
            blocks.append(
                f"{i}\n{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}\n{text}\n"
            )
        return "\n".join(blocks)

    def to_json(self) -> dict:
        d = {
            "language": self.language,
            "duration": self.duration,
            "segments": [s.to_json() for s in self.segments],
        }
        if self.recovered or self.unheard:
            d["recovered"] = self.recovered
            d["unheard"] = [list(u) for u in self.unheard]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Transcript":
        return cls(
            segments=[Segment.from_json(s) for s in (d.get("segments") or [])],
            language=str(d.get("language", "")),
            duration=float(d.get("duration", 0.0)),
            recovered=int(d.get("recovered") or 0),
            unheard=[tuple(u) for u in (d.get("unheard") or [])],
        )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, creationflags=_CREATE_NO_WINDOW,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _stream_subprocess(
    cmd: list[str],
    *,
    label: str,
    env: dict | None = None,
    cancel: threading.Event | None = None,
    on_line: Callable[[str], None] | None = None,
    tail: deque | None = None,
) -> int:
    """Run a subprocess, streaming every stdout/stderr line (split on both
    \\n and \\r so tqdm-style progress is captured) to LOG_BUS. Returns the
    exit code; raises CancelledError (after killing the process tree) if the
    cancel event fires."""
    log.debug("run [%s]: %s", label, " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            creationflags=_CREATE_NO_WINDOW,
            env=env,
        )
    except FileNotFoundError as e:
        raise AppError(
            f"Executable not found: {cmd[0]}",
            f"រកមិនឃើញកម្មវិធី៖ {cmd[0]}",
        ) from e

    def _handle(line: str) -> None:
        line = line.strip()
        if not line:
            return
        LOG_BUS.publish(f"[{label}] {line}")
        if tail is not None:
            tail.append(line)
        if on_line is not None:
            try:
                on_line(line)
            except Exception:
                pass

    buf = ""
    try:
        assert proc.stdout is not None
        while True:
            if cancel is not None and cancel.is_set():
                _kill_process_tree(proc)
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
                raise CancelledError("Operation cancelled")
            ch = proc.stdout.read(1)
            if ch == "":
                break
            if ch in ("\n", "\r"):
                if buf:
                    _handle(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            _handle(buf)
        return proc.wait()
    finally:
        if proc.poll() is None:
            _kill_process_tree(proc)


_TIME_RE = re.compile(r"(?:out_time|time)=(\d+):(\d+):(\d+(?:\.\d+)?)")
_OUT_TIME_US_RE = re.compile(r"out_time_us=(\d+)")


def _ffmpeg_progress_handler(
    duration: float,
    progress: Callable[[float, str], None] | None,
    text: str,
) -> Callable[[str], None]:
    def on_line(line: str) -> None:
        if progress is None or duration <= 0:
            return
        secs = -1.0
        m = _OUT_TIME_US_RE.search(line)
        if m:
            secs = int(m.group(1)) / 1_000_000.0
        else:
            m = _TIME_RE.search(line)
            if m:
                secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        if secs >= 0:
            progress(min(0.99, secs / duration), text)
    return on_line


def _run_ffmpeg(
    args: list[str],
    *,
    duration: float = 0.0,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
    label: str = "ffmpeg",
    status_text: str = "Processing…",
) -> None:
    cmd = [ffmpeg_path(), "-y", "-hide_banner", "-nostats",
           "-progress", "pipe:1"] + args
    tail: deque = deque(maxlen=15)
    rc = _stream_subprocess(
        cmd, label=label, cancel=cancel, tail=tail,
        on_line=_ffmpeg_progress_handler(duration, progress, status_text),
    )
    if rc != 0:
        detail = " | ".join(list(tail)[-4:])
        raise AppError(
            f"ffmpeg failed (code {rc}): {detail}",
            f"ffmpeg បរាជ័យ (កូដ {rc})។ សូមពិនិត្យកំណត់ហេតុ (log) សម្រាប់ព័ត៌មានលម្អិត។",
        )


# ---------------------------------------------------------------------------
# ffprobe / duration
# ---------------------------------------------------------------------------

def get_duration(path: str) -> float:
    """Media duration in seconds via ffprobe."""
    try:
        res = subprocess.run(
            [ffprobe_path(), "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, encoding="utf-8", errors="replace",
            creationflags=_CREATE_NO_WINDOW, timeout=60,
        )
        out = (res.stdout or "").strip()
        if res.returncode == 0 and out and out.lower() != "n/a":
            return float(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    raise AppError(
        f"Could not read media duration: {path}",
        f"មិនអាចអានរយៈពេលនៃឯកសារមេឌាបានទេ៖ {path}",
    )


# ---------------------------------------------------------------------------
# Audio extraction / vocal separation
# ---------------------------------------------------------------------------

def extract_audio(video: str, out_wav: str, sr: int = 16000, mono: bool = True,
                  progress: Callable[[float, str], None] | None = None,
                  cancel: threading.Event | None = None,
                  gain_db: float = 0.0) -> str:
    """Extract the audio track of a video to PCM WAV via ffmpeg.

    gain_db < 0 leaves headroom: loudly mastered dramas decode above full
    scale (true peaks of +3 dBFS are common) and 16-bit PCM would hard-clip
    every such peak into audible distortion."""
    _check_cancel(cancel)
    if not os.path.isfile(video):
        raise AppError(
            f"Video file not found: {video}",
            f"រកមិនឃើញឯកសារវីដេអូទេ៖ {video}",
        )
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    try:
        duration = get_duration(video)
    except AppError:
        duration = 0.0
    if progress:
        progress(0.0, "Extracting audio…")
    args = ["-i", video, "-vn"]
    if gain_db:
        args += ["-af", f"volume={gain_db:.2f}dB"]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(int(sr)), "-acodec", "pcm_s16le", out_wav]
    _run_ffmpeg(args, duration=duration, progress=progress, cancel=cancel,
                label="ffmpeg", status_text="Extracting audio…")
    if not os.path.isfile(out_wav):
        raise AppError(
            "Audio extraction produced no output file",
            "ការស្រង់សំឡេងមិនបានបង្កើតឯកសារលទ្ធផលទេ",
        )
    if progress:
        progress(1.0, "Audio extracted")
    return out_wav


_PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)%")


def _total_ram_gb() -> float:
    """The computer's memory in GB (0 = not known)."""
    if os.name != "nt":
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
        except (ValueError, OSError, AttributeError):
            return 0.0
    import ctypes

    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = _MemStatus()
    st.dwLength = ctypes.sizeof(_MemStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / 1024 ** 3
    except (OSError, AttributeError):
        pass
    return 0.0


def demucs_jobs() -> int:
    """How many pieces of the audio Demucs separates side by side (-j).
    Measured on a 32-thread CPU, 60 s of drama audio: 26.5 s one at a time,
    21.7 s with 4, 20.0 s with 8. Each piece is held in memory, so the number
    follows the CPU (a quarter of its threads, at most 8) and the memory
    (one per 4 GB)."""
    jobs = max(1, min(8, (os.cpu_count() or 4) // 4))
    ram = _total_ram_gb()
    if ram:
        jobs = max(1, min(jobs, int(ram // 4)))
    return jobs


def separate_vocals(audio: str, out_dir: str, model: str = "htdemucs",
                    device: str = "auto",
                    progress: Callable[[float, str], None] | None = None,
                    cancel: threading.Event | None = None) -> tuple[str, str]:
    """Split audio into (vocals_wav, accompaniment_wav) using Demucs, or —
    where Demucs is not installed (the .exe build) — the MDX-Net ONNX model
    (app.engines.mdx_separator)."""
    _check_cancel(cancel)
    if not os.path.isfile(audio):
        raise AppError(
            f"Audio file not found: {audio}",
            f"រកមិនឃើញឯកសារសំឡេងទេ៖ {audio}",
        )
    if importlib.util.find_spec("demucs") is None:
        from app.engines import mdx_separator
        if mdx_separator.available():
            return mdx_separator.separate(audio, out_dir, progress=progress,
                                          cancel=cancel)
        raise EngineNotAvailable(
            "No voice/music separator: download the 'MDX-Net voice separator' "
            "model (Settings → Models, or Fix in the Full Dub checklist), or "
            f"{pip_install('demucs')}.",
            "មិនមានឧបករណ៍បំបែកសំឡេង/តន្ត្រីទេ៖ សូមទាញយកម៉ូដែល 'MDX-Net voice separator' "
            f"(ការកំណត់ → Models) ឬ {pip_install('demucs')}។",
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    torch_home = models_dir() / "demucs"
    torch_home.mkdir(parents=True, exist_ok=True)
    env["TORCH_HOME"] = str(torch_home)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "-m", "demucs.separate",
           "--two-stems", "vocals", "-n", model,
           "-o", str(out_path)]
    if device in ("cpu", "cuda"):
        cmd += ["-d", device]
    jobs = demucs_jobs()
    if jobs > 1:
        cmd += ["-j", str(jobs)]
    cmd.append(audio)

    if progress:
        progress(0.01, "Separating vocals (Demucs)…")

    def on_line(line: str) -> None:
        if progress is None:
            return
        m = _PCT_RE.search(line)
        if m:
            pct = min(100.0, float(m.group(1)))
            progress(0.02 + 0.96 * (pct / 100.0), f"Separating vocals… {pct:.0f}%")

    tail: deque = deque(maxlen=15)
    rc = _stream_subprocess(cmd, label="demucs", env=env, cancel=cancel,
                            on_line=on_line, tail=tail)
    if rc != 0:
        joined = " | ".join(tail)
        if "No module named" in joined:
            raise EngineNotAvailable(
                f"Demucs is not installed. Install it with: {pip_install('demucs')}",
                f"កញ្ចប់ Demucs មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ៖ {pip_install('demucs')}",
            )
        raise AppError(
            f"Vocal separation (Demucs) failed (code {rc}): {' | '.join(list(tail)[-4:])}",
            f"ការបំបែកសំឡេង (Demucs) បរាជ័យ (កូដ {rc})។ សូមពិនិត្យកំណត់ហេតុ (log)។",
        )

    stem = Path(audio).stem
    base = out_path / model / stem
    vocals = base / "vocals.wav"
    accomp = base / "no_vocals.wav"
    if not (vocals.exists() and accomp.exists()):
        # Fall back to a recursive search (demucs versions differ slightly).
        found_v = sorted(out_path.rglob("vocals.wav"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        found_a = sorted(out_path.rglob("no_vocals.wav"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if found_v and found_a:
            vocals, accomp = found_v[0], found_a[0]
        else:
            raise AppError(
                "Demucs finished but the output files were not found",
                "Demucs បានបញ្ចប់ ប៉ុន្តែរកមិនឃើញឯកសារលទ្ធផលទេ",
            )
    if progress:
        progress(1.0, "Vocals separated")
    return str(vocals), str(accomp)


_STEMS_KEEP = 6          # separated videos kept in the cache


def separated_stems(video: str,
                    progress: Callable[[float, str], None] | None = None,
                    cancel: threading.Event | None = None) -> tuple[str, str]:
    """(vocals_16k_wav, music_flac) of a video's audio: voices for speech
    recognition, music & effects without the voices for a background bed.

    Separated once at full quality (44.1 kHz stereo, 6 dB headroom) and
    cached under app_data_dir()/cache/stems/<key>/ (key: path, size, mtime),
    so a recap's transcription and its montage share one separation run.
    Only the newest _STEMS_KEEP videos are kept."""
    import hashlib
    st = os.stat(video)
    key = hashlib.sha1(f"{os.path.abspath(video)}|{st.st_size}|{st.st_mtime}"
                       .encode("utf-8")).hexdigest()[:16]
    root = app_data_dir() / "cache" / "stems"
    d = root / key
    vocals, music = d / "vocals_16k.wav", d / "music.flac"
    if vocals.is_file() and music.is_file():
        os.utime(d)                      # recently used: keep it
        if progress:
            progress(1.0, "Voices and music already separated")
        return str(vocals), str(music)
    d.mkdir(parents=True, exist_ok=True)
    full = d / "audio.wav"
    try:
        extract_audio(video, str(full), sr=44100, mono=False, gain_db=-6.0,
                      progress=(lambda f, t: progress(0.1 * f, t)) if progress else None,
                      cancel=cancel)
        v, m = separate_vocals(str(full), str(d / "sep"),
                               progress=(lambda f, t: progress(0.1 + 0.85 * f, t))
                               if progress else None, cancel=cancel)
        _run_ffmpeg(["-i", v, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
                     str(vocals)], cancel=cancel, status_text="Saving voices…")
        _run_ffmpeg(["-i", m, "-c:a", "flac", str(music)], cancel=cancel,
                    status_text="Saving music & effects…")
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    finally:
        try:
            full.unlink()
        except OSError:
            pass
        shutil.rmtree(d / "sep", ignore_errors=True)
    try:
        old = sorted((p for p in root.iterdir() if p.is_dir()),
                     key=lambda p: p.stat().st_mtime, reverse=True)[_STEMS_KEEP:]
        for p in old:
            shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass
    if progress:
        progress(1.0, "Voices and music separated")
    return str(vocals), str(music)


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def _load_audio_16k(path: str, cancel: threading.Event | None = None) -> np.ndarray:
    """Any audio/video file -> 16 kHz mono float32 samples, decoded by the
    bundled ffmpeg. faster-whisper's own decoder goes through PyAV, whose
    newer releases dropped an option it passes ("open() got an unexpected
    keyword argument 'metadata_errors'"), so Whisper gets samples instead."""
    _check_cancel(cancel)
    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-v", "error", "-nostdin", "-i", path,
         "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True, creationflags=_CREATE_NO_WINDOW)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no audio"]
        raise AppError(f"Could not read the audio for transcription: {err[0]}",
                       f"មិនអាចអានសំឡេងសម្រាប់ចម្លងជាអក្សរបានទេ៖ {err[0]}")
    _check_cancel(cancel)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Second listen: speech the first pass left without words
# ---------------------------------------------------------------------------
#
# In fast or overlapping speech Whisper's decoder can jump over a sentence,
# and a 30 s window whose no_speech_prob is over 0.6 with a low log-prob is
# fast-forwarded whole (faster-whisper says so only at DEBUG level). Measured
# on a Chinese drama (first 20 min, large-v3, on-screen subtitles as the
# truth): 32 subtitle lines had no recognised word; a gap transcribed on its
# own gave 12 of them back — three fast sentences in a row among them —
# while gaps with little voice in them (Silero < 0.4) gave only Whisper's
# stock phrases ("谢谢大家") and loops ("喲喲喲喲…"). Hence: only voiced gaps,
# one at a time with a little context, words kept only inside the gap, stock
# phrases / loops / duplicates refused — and every refusal reported.

_VAD_THRESHOLD = 0.35     # Silero speech threshold of the first pass
_LOW_LOGPROB = -1.0       # Whisper's own log-prob limit: below it, a segment is doubtful
_GAP_VAD = 0.4            # speech probability that makes a gap worth a second listen
_GAP_MIN_S = 0.4          # shorter voiced stretches are breaths and clicks
_GAP_PAD_S = 0.25         # a recognised word this close already covers the speech
_GAP_CONTEXT_S = 1.0      # audio around a gap given to Whisper (its words not kept)
_GAP_MAX = 400            # second listens per transcript (longest gaps first)

# Whisper's stock phrases from its training subtitles — said over silence or
# music, never really spoken there
_STOCK_PHRASES = re.compile(
    r"[谢謝]{2}(大家|[观觀][看赏賞]|收看|你的[观觀]看)|感[谢謝]([观觀]看|收看|您的[观觀]看)|請?不吝|"
    r"[点點][赞讚贊]|[订訂][阅閱]|[转轉][发發]|打[赏賞]|字幕(由|[组組]|志[愿願]者|提供)|明[镜鏡]与[点點]{2}|"
    r"我的[频頻]道|[铃鈴][铛鐺]|收到最新(消息|[视視][频頻]|影片)|下期(再)?[见見]|分享[给給]朋友|"
    r"thank(s| you) (so much )?for watching|please (like and )?subscribe|subtitles? by|"
    r"(?<=[^\x00-\x7f])by [a-z]{2,6}$|"           # "…下期見by bwd", not "stand by me"
    r"ご視聴ありがとうございました|チャンネル登録|시청해\s?주셔서|구독과 좋아요|amara\.org",
    re.IGNORECASE)
_LOOP = re.compile(r"(.{1,4}?)\1{4,}")          # one sound five times or more
_NORM = re.compile(r"[\s\W_]+", re.UNICODE)
# more than anyone says in that time: the 0.8 s gap under a "嗯" came back
# as a different invented sentence each run ("能收到最新消息哦", 16 chars)
_MAX_CPS = 14.0           # characters a second (Chinese / Japanese…)
_MAX_WPS = 7.0            # words a second (spaced languages)


_WORDISH = "a-z0-9À-ɏЀ-ӿ'"
_UNIT = re.compile(rf"[{_WORDISH}]+|[^\W\d_{_WORDISH}]")


def _units(text: str) -> list[str]:
    """Speech units: a Latin / Cyrillic word or number is one, and so is each
    character of Chinese, Japanese, Korean, Thai, Khmer… — so mixed text
    ("鈴鐺我們下期見by bwd") counts right."""
    return _UNIT.findall((text or "").lower())


def _spaced(units: list[str]) -> bool:
    """Every unit a whole word (a spaced language), none a single character."""
    return bool(units) and all(re.match(rf"[{_WORDISH}]", u) for u in units)


def find_speech_gaps(samples: np.ndarray, segments: list[Segment],
                     sr: int = 16000) -> list[tuple[float, float]]:
    """Stretches of speech (Silero VAD, `_GAP_VAD`) that no recognised word
    covers (± `_GAP_PAD_S`), at least `_GAP_MIN_S` long."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    speech = get_speech_timestamps(samples, VadOptions(
        threshold=_GAP_VAD, min_speech_duration_ms=int(_GAP_MIN_S * 1000),
        min_silence_duration_ms=300, speech_pad_ms=60), sampling_rate=sr)
    covered: list[tuple[float, float]] = []
    for s in segments:
        ws = [w for w in (s.words or []) if str(w.text).strip()]
        if ws:
            covered += [(float(w.start) - _GAP_PAD_S, float(w.end) + _GAP_PAD_S) for w in ws]
        elif str(s.text).strip():
            covered.append((float(s.start) - _GAP_PAD_S, float(s.end) + _GAP_PAD_S))
    union: list[tuple[float, float]] = []
    for c in sorted(covered):                 # disjoint, sorted: the walk below needs it
        if union and c[0] <= union[-1][1]:
            union[-1] = (union[-1][0], max(union[-1][1], c[1]))
        else:
            union.append(c)
    covered = union
    gaps: list[tuple[float, float]] = []
    k = 0
    for ch in speech:
        a, b = ch["start"] / sr, ch["end"] / sr
        while k < len(covered) and covered[k][1] <= a:
            k += 1
        cur = a
        j = k
        while j < len(covered) and covered[j][0] < b:
            if covered[j][0] > cur and covered[j][0] - cur >= _GAP_MIN_S:
                gaps.append((cur, covered[j][0]))
            cur = max(cur, covered[j][1])
            j += 1
        if b - cur >= _GAP_MIN_S:
            gaps.append((cur, b))
    return gaps


def _refusal(text: str, seg_stats: tuple[float, float], near_units: list[str],
             seconds: float = 0.0) -> str:
    """Why a second-listen text is not taken ("" = take it). `seconds`: how
    long the gap it was heard in is."""
    core = _NORM.sub("", text)
    if not core:
        return "nothing recognised"
    if _STOCK_PHRASES.search(text):
        # mostly a stock phrase ("鈴鐺我們下期見by bwd" leaves only 我們)
        rest = len(_units(_STOCK_PHRASES.sub(" ", text)))
        if rest < 2 or rest < 0.4 * len(_units(text)):
            return "Whisper stock phrase"
    if _LOOP.search(core):
        return "repetition loop"
    logprob, no_speech = seg_stats
    if no_speech > 0.6 and logprob < _LOW_LOGPROB:
        return "Whisper heard no speech"
    mine = _units(text)
    if seconds > 0 and len(mine) > 2:
        if len(mine) / (seconds + 0.2) > (_MAX_WPS if _spaced(mine) else _MAX_CPS):
            return "too many words for the time"
    if mine and near_units:
        from difflib import SequenceMatcher
        same = sum(b.size for b in SequenceMatcher(None, mine, near_units,
                                                    autojunk=False).get_matching_blocks())
        if same >= 0.8 * len(mine):
            return "duplicate"
    return ""


def recover_gaps(model, samples: np.ndarray, segments: list[Segment],
                 language: str | None,
                 progress: Callable[[float, str], None] | None = None,
                 cancel: threading.Event | None = None,
                 sr: int = 16000) -> tuple[list[Segment], list[tuple]]:
    """Listen again, one at a time, to the voiced stretches no recognised
    word covers (find_speech_gaps). Returns (segments with the recovered
    ones added in time order — marked review="recovered" — and the speech
    that still has no words: (start, end, reason, text heard)). Every gap's
    outcome is logged; a duplicate of words already in the transcript is
    the only one not reported as unheard (nothing was lost there)."""
    try:
        gaps = find_speech_gaps(samples, segments, sr)
    except Exception as exc:              # no Silero in this faster-whisper
        log.warning("second listen skipped: speech gaps not measured (%s)", exc)
        return segments, []
    if not gaps:
        LOG_BUS.publish("[whisper] second listen: no speech was left without words")
        return segments, []
    if len(gaps) > _GAP_MAX:
        keep = set(sorted(gaps, key=lambda g: g[0] - g[1])[:_GAP_MAX])
        log.warning("second listen: %d gaps, only the %d longest are checked",
                    len(gaps), _GAP_MAX)
        gaps = [g for g in gaps if g in keep]
    LOG_BUS.publish(f"[whisper] second listen: {len(gaps)} stretches of speech "
                    f"({sum(b - a for a, b in gaps):.0f} s) had no words — listening again")
    found: list[Segment] = []
    unheard: list[tuple] = []
    # what the first pass heard, by time: a duplicate is only words heard
    # right around the gap (a name said elsewhere in a long segment is not)
    heard_at: list[tuple[float, str]] = []
    for s in segments:
        ws = [w for w in (s.words or []) if str(w.text).strip()]
        if ws:
            heard_at += [((float(w.start) + float(w.end)) / 2, str(w.text)) for w in ws]
        elif str(s.text).strip():
            heard_at.append(((float(s.start) + float(s.end)) / 2, str(s.text)))
    heard_at.sort()
    starts = [t for t, _ in heard_at]
    import bisect
    for n, (a, b) in enumerate(gaps):
        _check_cancel(cancel)
        if progress:
            progress(n / len(gaps), f"Listening again {n + 1}/{len(gaps)} "
                                    f"({_fmt_srt_time(a)[:8]})")
        lo = max(0.0, a - _GAP_CONTEXT_S)
        clip = samples[int(lo * sr):int((b + _GAP_CONTEXT_S) * sr)]
        try:
            seg_iter, _info = model.transcribe(
                clip, language=language, beam_size=5, word_timestamps=True,
                vad_filter=False, condition_on_previous_text=False,
                no_speech_threshold=None)          # judged below, not skipped
            heard = list(seg_iter)
        except CancelledError:
            raise
        except Exception as exc:
            log.warning("second listen %.2f-%.2f s failed: %s", a, b, exc)
            unheard.append((round(a, 2), round(b, 2), f"recogniser error: {exc}", ""))
            continue
        words, stats = [], []
        for s in heard:
            inside = [w for w in (s.words or [])
                      if a - 0.1 <= lo + (float(w.start) + float(w.end)) / 2 <= b + 0.1]
            if inside:
                words += inside
                stats.append((float(s.avg_logprob), float(s.no_speech_prob)))
        text = "".join(str(w.word) for w in words).strip()
        logprob = min((x[0] for x in stats), default=0.0)
        no_speech = max((x[1] for x in stats), default=0.0)
        i = bisect.bisect_left(starts, a - 2.0)
        near_text = []
        while i < len(heard_at) and heard_at[i][0] <= b + 2.0:
            near_text.append(heard_at[i][1])
            i += 1
        for f in found[-2:]:                 # the context of two gaps can overlap
            if f.end >= a - 2.0:
                near_text.append(" " + f.text)
        near = _units("".join(near_text))
        why = _refusal(text, (logprob, no_speech), near, b - a)
        if why == "duplicate":
            log.info("second listen %.2f-%.2f s: %r already in the transcript", a, b, text)
            continue
        if why:
            log.info("second listen %.2f-%.2f s: not taken (%s) %r", a, b, why, text)
            unheard.append((round(a, 2), round(b, 2), why, text))
            continue
        ws = [Word(start=round(lo + float(w.start), 3), end=round(lo + float(w.end), 3),
                   text=str(w.word)) for w in words]
        found.append(Segment(start=ws[0].start, end=ws[-1].end, text=text, words=ws,
                             review="recovered"))
        log.info("second listen %.2f-%.2f s: recovered %r", a, b, text)
    if found:
        segments = sorted(list(segments) + found, key=lambda s: (float(s.start), float(s.end)))
    LOG_BUS.publish(f"[whisper] second listen: {len(found)} recovered, "
                    f"{len(unheard)} stretches still without words")
    if progress:
        progress(1.0, f"Second listen: {len(found)} recovered")
    return segments, unheard


class WhisperEngine:
    """faster-whisper wrapper with CUDA→CPU fallback and word timestamps."""

    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.loaded_on = ""          # "cuda" | "cpu" once the model is loaded
        self._model = None
        self._lock = threading.Lock()

    @staticmethod
    def _cuda_runtime_ok() -> bool:
        """Check if CUDA runtime libs are actually loadable, not just the
        driver (cuBLAS from NVIDIA's pip package or the system)."""
        from app.core import gpu
        if gpu.cuda_libs_ok():
            return True
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _ensure_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel  # lazy heavy import
            except ImportError as e:
                raise EngineNotAvailable(
                    "faster-whisper is not installed. Install it with: "
                    f"{pip_install('faster-whisper')}",
                    "កញ្ចប់ faster-whisper មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ៖ "
                    f"{pip_install('faster-whisper')}",
                ) from e
            root = models_dir() / "whisper"
            root.mkdir(parents=True, exist_ok=True)
            cuda_ok = self.device != "cpu" and self._cuda_runtime_ok()
            attempts: list[tuple[str, str]] = []
            if cuda_ok:
                # fastest type the card does first; older cards (no fast
                # float16) take int8 / float32; the CPU last
                from app.core import gpu
                types = gpu.ct2_cuda_compute_types() or ["float16"]
                attempts = [("cuda", t) for t in types[:2]]
            elif self.device == "cuda":
                log.warning("CUDA requested but runtime libs not found, "
                            "falling back to CPU")
            attempts.append(("cpu", "int8"))
            last_exc: Exception | None = None
            for dev, ctype in attempts:
                try:
                    model = WhisperModel(self.model_size, device=dev,
                                         compute_type=ctype,
                                         download_root=str(root))
                    self.loaded_on = dev
                    log.info("Whisper '%s' loaded on %s (%s)",
                             self.model_size, dev, ctype)
                    LOG_BUS.publish(
                        f"[whisper] model={self.model_size} device={dev} "
                        f"compute={ctype}")
                    self._model = model
                    return model
                except Exception as e:  # cuda missing / cudnn errors etc.
                    last_exc = e
                    log.warning("Whisper load failed on %s/%s: %s", dev, ctype, e)
            raise AppError(
                f"Failed to load Whisper model '{self.model_size}': {last_exc}",
                f"មិនអាចផ្ទុកម៉ូដែល Whisper '{self.model_size}' បានទេ៖ {last_exc}",
            )

    def detect_language(self, audio: str) -> tuple[str, float]:
        """(language code, probability) from the first ~30 s of speech."""
        model = self._ensure_model()
        samples = _load_audio_16k(audio)
        lang, prob, _all = model.detect_language(samples, vad_filter=True,
                                                 language_detection_segments=3)
        return str(lang), float(prob)

    def transcribe(self, audio: str, language: str | None = None,
                   word_timestamps: bool = True,
                   progress: Callable[[float, str], None] | None = None,
                   cancel: threading.Event | None = None,
                   initial_prompt: str | None = None,
                   hotwords: str | None = None,
                   recover: bool = True) -> Transcript:
        """`recover`: listen again, one at a time, to the stretches of speech
        the first pass left without words (recover_gaps)."""
        try:
            return self._transcribe(audio, language, word_timestamps, progress, cancel,
                                    initial_prompt, hotwords, recover)
        except AppError as exc:
            if getattr(self, "loaded_on", "") != "cuda":
                raise
            # the GPU failed while working (an old card, a driver problem):
            # the same job on the CPU instead of no transcript
            log.warning("Whisper on the GPU failed (%s); transcribing on the CPU", exc.en)
            LOG_BUS.publish("[whisper] the GPU failed — transcribing on the CPU instead")
            with self._lock:
                self._model, self.device, self.loaded_on = None, "cpu", ""
            return self._transcribe(audio, language, word_timestamps, progress, cancel,
                                    initial_prompt, hotwords, recover)

    def recover(self, audio: str, transcript: Transcript, language: str | None = None,
                progress: Callable[[float, str], None] | None = None,
                cancel: threading.Event | None = None) -> Transcript:
        """recover_gaps() on another recogniser's transcript (Deepgram):
        returns it with the recovered segments added."""
        model = self._ensure_model()
        samples = _load_audio_16k(audio, cancel)
        segs, unheard = recover_gaps(model, samples, transcript.segments,
                                     language or transcript.language or None,
                                     progress=progress, cancel=cancel)
        added = len(segs) - len(transcript.segments)
        return Transcript(segments=segs, language=transcript.language,
                          duration=transcript.duration,
                          recovered=transcript.recovered + added,
                          unheard=list(transcript.unheard) + unheard)

    def _transcribe(self, audio, language, word_timestamps, progress, cancel,
                    initial_prompt, hotwords, recover=True) -> Transcript:
        _check_cancel(cancel)
        if not os.path.isfile(audio):
            raise AppError(
                f"Audio file not found: {audio}",
                f"រកមិនឃើញឯកសារសំឡេងទេ៖ {audio}",
            )
        model = self._ensure_model()
        if progress:
            progress(0.0, "Transcribing…")
        samples = _load_audio_16k(audio, cancel)
        recover = recover and word_timestamps
        span = 0.9 if recover else 0.99        # progress left for the second listen
        try:
            seg_iter, info = model.transcribe(
                samples,
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=True,
                # Silero's default 0.5 left quiet and fast speech out before
                # Whisper heard it: 0.35 found 4 of 32 missed subtitle lines
                # on a Chinese drama (20 min, large-v3) with no extra
                # unmatched text; 0.30 added doubtful text. The 2 s silence
                # default stays (1 s gained nothing).
                vad_parameters={"threshold": _VAD_THRESHOLD},
                beam_size=5,
                initial_prompt=initial_prompt or None,
                hotwords=hotwords or None,
                # stop one hallucinated phrase from echoing through later
                # segments (repeated lines in drama audio)
                condition_on_previous_text=False,
            )
        except Exception as e:
            raise AppError(
                f"Transcription failed to start: {e}",
                f"ការចម្លងសំឡេងជាអក្សរមិនអាចចាប់ផ្តើមបានទេ៖ {e}",
            ) from e

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration <= 0:
            try:
                duration = get_duration(audio)
            except AppError:
                duration = 0.0

        segments: list[Segment] = []
        try:
            for s in seg_iter:
                _check_cancel(cancel)
                words: list[Word] = []
                if word_timestamps and getattr(s, "words", None):
                    for w in s.words:
                        words.append(Word(start=float(w.start),
                                          end=float(w.end),
                                          text=str(w.word)))
                # Whisper's own doubt (its log-prob limit): kept, but marked
                # for a person to check — never dropped
                doubt = float(getattr(s, "avg_logprob", 0.0) or 0.0) < _LOW_LOGPROB
                segments.append(Segment(start=float(s.start), end=float(s.end),
                                        text=str(s.text).strip(), words=words,
                                        review="low_confidence" if doubt else ""))
                if progress and duration > 0:
                    frac = min(span, span * float(s.end) / duration)
                    progress(frac, f"Transcribing… {_fmt_srt_time(s.end)[:8]}")
        except CancelledError:
            raise
        except Exception as e:
            raise AppError(
                f"Transcription failed: {e}",
                f"ការចម្លងសំឡេងជាអក្សរបរាជ័យ៖ {e}",
            ) from e

        lang = str(getattr(info, "language", "") or language or "")
        if duration <= 0 and segments:
            duration = segments[-1].end
        first = len(segments)
        unheard: list = []
        if recover:
            sub = (lambda f, t: progress(span + (1.0 - span) * f, t)) if progress else None
            segments, unheard = recover_gaps(model, samples, segments, lang or None,
                                             progress=sub, cancel=cancel)
        if progress:
            progress(1.0, f"Transcribed {len(segments)} segments")
        return Transcript(segments=segments, language=lang, duration=duration,
                          recovered=len(segments) - first, unheard=unheard)


STT_ENGINES = ["whisper", "deepgram"]


def stt_engine(config=None) -> str:
    """The speech-to-text engine chosen in Settings → Speech services."""
    try:
        if config is None:
            from app.core.config_manager import get_default_config
            config = get_default_config()
        e = str(config.get("stt.engine") or "whisper").lower()
    except Exception:
        e = "whisper"
    return e if e in STT_ENGINES else "whisper"


def installed_whisper_size() -> str:
    """The best already-downloaded Whisper model that is quick to load, for
    a language check before Deepgram transcribes; "" when none is there."""
    from app.core.model_manager import _whisper_present
    for size in ("small", "base", "medium", "tiny", "large-v3"):
        try:
            if _whisper_present(size):
                return size
        except Exception:
            pass
    return ""


def detect_spoken_language(audio: str) -> str:
    """The language spoken in `audio` ("zh", "en", ...), found on this PC by
    a downloaded Whisper model; "" when there is none or it fails (nothing
    is downloaded for it)."""
    size = installed_whisper_size()
    if not size:
        return ""
    try:
        lang, prob = WhisperEngine(model_size=size).detect_language(audio)
    except CancelledError:
        raise
    except Exception as exc:
        log.warning("language detection failed (%s); Deepgram decides", exc)
        return ""
    log.info("spoken language %s (%.0f%%, Whisper %s)", lang, prob * 100, size)
    LOG_BUS.publish(f"[stt] spoken language: {lang} ({prob:.0%})")
    return str(lang or "")


def speech_to_text(audio: str, *, config=None, model_size: str = "small",
                   language: str | None = None, word_timestamps: bool = True,
                   progress: Callable[[float, str], None] | None = None,
                   cancel: threading.Event | None = None,
                   initial_prompt: str | None = None,
                   hotwords: str | None = None) -> Transcript:
    """Transcribe with the engine chosen in Settings: Whisper on this PC, or
    Deepgram online (which also labels speakers: Segment.speaker "D1", …).
    Deepgram cannot do every language (no Khmer): Whisper is used then, and
    when Deepgram hears no speech at all."""
    if config is None:
        from app.core.config_manager import get_default_config
        config = get_default_config()
    if stt_engine(config) == "deepgram":
        from app.engines import deepgram_stt
        if not language:
            # left to Deepgram the language can come back wrong: a Mandarin
            # clip was taken for English, then Korean, then French — "Bon,"
            # was all it heard, and the recap had no story to tell. Found
            # here first, as the dub does (TODO #22)
            if progress is not None:
                progress(0.0, "Detecting the spoken language")
            language = detect_spoken_language(audio) or None
            _check_cancel(cancel)
        if deepgram_stt.supports(language or ""):
            tr = deepgram_stt.transcribe(audio, config.get_secret("deepgram.api_key"),
                                         language=language or "", diarize=True,
                                         progress=progress, cancel=cancel)
            if any((s.text or "").strip() for s in tr.segments):
                return tr
            LOG_BUS.publish("[deepgram] heard no speech — trying Whisper")
        else:
            LOG_BUS.publish(f"[deepgram] cannot transcribe '{language}' — using Whisper")
    return WhisperEngine(model_size=model_size).transcribe(
        audio, language=language, word_timestamps=word_timestamps,
        progress=progress, cancel=cancel, initial_prompt=initial_prompt,
        hotwords=hotwords)


# ---------------------------------------------------------------------------
# Gender detection (numpy autocorrelation F0)
# ---------------------------------------------------------------------------

def _read_wav_slice(path: str, start: float, end: float,
                    max_seconds: float = 10.0) -> tuple[np.ndarray, int]:
    """Return (float32 mono samples in [-1,1], sample_rate) for the slice."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        pos = max(0, min(nframes, int(start * sr)))
        want = int(max(0.0, end - start) * sr)
        want = min(want, int(max_seconds * sr), nframes - pos)
        if want <= 0:
            return np.zeros(0, dtype=np.float32), sr
        wf.setpos(pos)
        raw = wf.readframes(want)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")
    if ch > 1:
        usable = (len(data) // ch) * ch
        data = data[:usable].reshape(-1, ch).mean(axis=1)
    return data.astype(np.float32), sr


# Hz: a median pitch below this sounds male — between typical men (~125 Hz)
# and women (~235 Hz) on a log scale; young male leads often speak at 150-180
_F0_SPLIT = 172.0


def _voiced_frames(x: np.ndarray, sr: int, max_frames: int = 500
                   ) -> tuple[list[float], list[int], int]:
    """Frame-wise autocorrelation pitch in the 60–400 Hz band: (F0 of each
    voiced frame, its start sample, frame length)."""
    flen = int(0.040 * sr)          # 40 ms frames
    hop = int(0.020 * sr)           # 20 ms hop
    lag_min = max(2, int(sr / 400.0))   # 400 Hz
    lag_max = int(sr / 60.0)            # 60 Hz
    f0s: list[float] = []
    offs: list[int] = []
    if flen <= lag_max or x.size < flen:
        return f0s, offs, flen
    peak = float(np.max(np.abs(x)))
    energy_gate = max(0.01, 0.05 * peak)
    for n_frames, off in enumerate(range(0, x.size - flen + 1, hop)):
        if n_frames >= max_frames:
            break
        frame = x[off:off + flen]
        frame = frame - float(np.mean(frame))
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < energy_gate:
            continue
        ac = np.correlate(frame, frame, mode="full")[flen - 1:]
        if ac[0] <= 1e-9:
            continue
        band = ac[lag_min:lag_max + 1] / ac[0]
        if band.size == 0:
            continue
        k = int(np.argmax(band))
        if float(band[k]) < 0.30:
            continue
        f0 = sr / float(lag_min + k)
        if 60.0 <= f0 <= 400.0:
            f0s.append(f0)
            offs.append(off)
    return f0s, offs, flen


def voice_gender(wav: str, start: float, end: float) -> tuple[str, float, float]:
    """How male or female the speech in a wav slice sounds:
    ("M" | "F" | "" when unmeasurable, confidence 0..1, median F0 in Hz).

    Confidence grows with the distance of the median pitch from the split:
    1.0 at about 120 Hz / 245 Hz and beyond, near 0 for 155–190 Hz voices
    (deep women, high men, shouting) — those are left for review."""
    try:
        x, sr = _read_wav_slice(wav, start, end)
        if sr <= 0 or x.size == 0 or float(np.max(np.abs(x))) < 1e-4:
            return "", 0.0, 0.0
        f0s, _offs, _flen = _voiced_frames(x, sr)
        if len(f0s) < 3:
            return "", 0.0, 0.0
        f0 = float(np.median(np.asarray(f0s)))
        score = max(-1.0, min(1.0, float(np.log2(f0 / _F0_SPLIT)) / 0.5))
        return ("F" if score > 0 else "M"), round(abs(score), 3), round(f0, 1)
    except Exception:
        return "", 0.0, 0.0


def detect_gender(wav: str, start: float, end: float) -> str:
    """Speaker gender ("M"/"F") of a wav slice — see voice_gender().
    Returns "F" when it cannot be measured."""
    return voice_gender(wav, start, end)[0] or "F"


def assign_voices(segments: list[Segment], lang: str = "km",
                  male_voice: str = "km-KH-PisethNeural",
                  female_voice: str = "km-KH-SreymomNeural") -> None:
    """Normalize each segment's gender to "M"/"F" (default "F") and set its
    speaker to the matching voice name."""
    for seg in segments:
        g = (seg.gender or "").strip().upper()
        if g not in ("M", "F"):
            g = "F"
        seg.gender = g
        seg.speaker = male_voice if g == "M" else female_voice


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from sync code, tolerating odd loop states."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


_EDGE_VOICES: list[dict] | None = None
_EDGE_VOICES_LOCK = threading.Lock()
_EDGE_CACHE_TTL = 7 * 86400  # one week


def _edge_cache_file() -> Path:
    return app_data_dir() / "edge_voices.json"


def list_edge_voices(lang_prefix: str = "") -> list[dict]:
    """List Microsoft Edge TTS voices, cached in-module and on disk."""
    global _EDGE_VOICES
    with _EDGE_VOICES_LOCK:
        voices = _EDGE_VOICES
        cache = _edge_cache_file()
        if voices is None:
            try:
                if cache.exists():
                    data = json.loads(cache.read_text(encoding="utf-8"))
                    if (data.get("voices")
                            and time.time() - float(data.get("fetched_at", 0)) < _EDGE_CACHE_TTL):
                        voices = list(data["voices"])
            except Exception:
                voices = None
        if voices is None:
            try:
                import edge_tts  # lazy heavy import
            except ImportError as e:
                raise EngineNotAvailable(
                    f"edge-tts is not installed. Install it with: {pip_install('edge-tts')}",
                    f"កញ្ចប់ edge-tts មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ៖ {pip_install('edge-tts')}",
                ) from e
            try:
                fetched = _run_async(edge_tts.list_voices())
                voices = [dict(v) for v in fetched]
                try:
                    cache.write_text(
                        json.dumps({"fetched_at": time.time(), "voices": voices},
                                   ensure_ascii=False),
                        encoding="utf-8")
                except Exception:
                    pass
            except Exception as e:
                # Network failed — fall back to a stale disk cache if present.
                try:
                    stale = json.loads(cache.read_text(encoding="utf-8"))
                    voices = list(stale.get("voices") or [])
                except Exception:
                    voices = []
                if not voices:
                    raise AppError(
                        f"Could not fetch the Edge TTS voice list (network required): {e}",
                        "មិនអាចទាញយកបញ្ជីសំឡេង Edge TTS បានទេ (ត្រូវការអ៊ីនធឺណិត)",
                    ) from e
        _EDGE_VOICES = voices

    if lang_prefix:
        lp = lang_prefix.lower()
        return [v for v in voices
                if str(v.get("ShortName", "")).lower().startswith(lp)
                or str(v.get("Locale", "")).lower().startswith(lp)]
    return list(voices)


def _to_24k_mono_wav(src: str, out_wav: str,
                     cancel: threading.Event | None = None) -> str:
    _run_ffmpeg(["-i", src, "-ac", "1", "-ar", str(_TTS_SR),
                 "-acodec", "pcm_s16le", out_wav],
                cancel=cancel, label="ffmpeg", status_text="Converting audio…")
    return out_wav


def _write_wav_int16(path: str, data: np.ndarray, sr: int) -> None:
    data = np.asarray(data)
    if data.dtype != np.int16:
        data = np.clip(np.asarray(data, dtype=np.float64), -1.0, 1.0)
        data = (data * 32767.0).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(data.tobytes())


def faster_rate(rate: str, speed: float) -> str:
    """Edge TTS rate for `speed` times the given rate ("+10%" x 1.2 -> "+32%").
    Edge's rate is exact (+30% speaks 1.30x faster), so a line that must be
    quicker is better re-spoken at a higher rate than time-stretched."""
    pct = round((_rate_to_speed(rate) * speed - 1.0) * 100)
    return f"{max(-50, min(100, pct)):+d}%"


def _rate_to_speed(rate: str) -> float:
    """'+10%' -> 1.10, '-20%' -> 0.80, clamped to [0.25, 4.0]."""
    try:
        m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", rate or "")
        if not m:
            return 1.0
        return max(0.25, min(4.0, 1.0 + float(m.group(1)) / 100.0))
    except Exception:
        return 1.0


def _synth_edge(text: str, out_wav: str, voice: str, rate: str, pitch: str,
                volume: str, cancel: threading.Event | None) -> str:
    try:
        import edge_tts  # lazy heavy import
    except ImportError as e:
        raise EngineNotAvailable(
            f"edge-tts is not installed. Install it with: {pip_install('edge-tts')}",
            f"កញ្ចប់ edge-tts មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ៖ {pip_install('edge-tts')}",
        ) from e
    voice = voice or _DEFAULT_EDGE_VOICE
    tmp_mp3 = out_wav + ".edge.mp3"
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate or "+0%",
                                    pitch=pitch or "+0Hz",
                                    volume=volume or "+0%")
        _run_async(comm.save(tmp_mp3))
        _check_cancel(cancel)
        if not os.path.isfile(tmp_mp3) or os.path.getsize(tmp_mp3) == 0:
            raise AppError(
                "Edge TTS returned no audio",
                "Edge TTS មិនបានបញ្ជូនសំឡេងមកវិញទេ",
            )
        return _to_24k_mono_wav(tmp_mp3, out_wav, cancel=cancel)
    except (AppError, CancelledError):
        raise
    except Exception as e:
        raise AppError(
            f"Edge TTS synthesis failed: {e}",
            f"ការបង្កើតសំឡេងតាម Edge TTS បរាជ័យ៖ {e}",
        ) from e
    finally:
        try:
            if os.path.isfile(tmp_mp3):
                os.remove(tmp_mp3)
        except OSError:
            pass


def _synth_openai(text: str, out_wav: str, voice: str, rate: str,
                  emotion: str, api_key: str, config,
                  cancel: threading.Event | None) -> str:
    key = api_key
    if not key and config is not None:
        key = config.key_pool.acquire("openai")
    if not key:
        raise ProviderError(
            "No OpenAI API key available for TTS",
            "គ្មានសោ API របស់ OpenAI សម្រាប់ TTS ទេ។ សូមបញ្ចូលសោនៅក្នុងការកំណត់",
            provider="openai", status_code=None, retryable=False,
        )
    base = "https://api.openai.com/v1"
    model = "gpt-4o-mini-tts"
    if config is not None:
        base = (config.get("provider.openai.base_url") or base).rstrip("/")
        model = config.get("tts.openai.model") or model
    body: dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice or "alloy",
        "response_format": "mp3",
        "speed": _rate_to_speed(rate),
    }
    if emotion:
        body["instructions"] = f"Speak with this emotion/style: {emotion}"
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base}/audio/speech",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json=body, timeout=120,
        )
    except requests.RequestException as e:
        if config is not None and not api_key:
            config.key_pool.report_failure("openai", key, None, str(e))
        raise ProviderError(
            f"OpenAI TTS network error: {e}",
            f"បញ្ហាបណ្តាញពេលហៅ OpenAI TTS៖ {e}",
            provider="openai", status_code=None, retryable=True,
        ) from e
    _check_cancel(cancel)
    if resp.status_code != 200:
        msg = resp.text[:300]
        if config is not None and not api_key:
            config.key_pool.report_failure("openai", key, resp.status_code, msg)
        raise ProviderError(
            f"OpenAI TTS failed (HTTP {resp.status_code}): {msg}",
            f"ការហៅ TTS ទៅ OpenAI បរាជ័យ (កូដ {resp.status_code})",
            provider="openai", status_code=resp.status_code,
            retryable=resp.status_code == 429 or resp.status_code >= 500,
        )
    if config is not None and not api_key:
        config.key_pool.report_success("openai", key, int((time.time() - t0) * 1000))
    tmp = out_wav + ".openai.mp3"
    try:
        with open(tmp, "wb") as f:
            f.write(resp.content)
        return _to_24k_mono_wav(tmp, out_wav, cancel=cancel)
    finally:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _synth_elevenlabs(text: str, out_wav: str, voice: str, api_key: str,
                      config, cancel: threading.Event | None) -> str:
    key = api_key
    if not key and config is not None:
        key = config.get_secret("elevenlabs.api_key")
    if not key:
        raise ProviderError(
            "No ElevenLabs API key configured (settings key 'elevenlabs.api_key')",
            "គ្មានសោ API របស់ ElevenLabs ទេ។ សូមបញ្ចូលសោនៅក្នុងការកំណត់",
            provider="elevenlabs", status_code=None, retryable=False,
        )
    voice_id = voice or "21m00Tcm4TlvDq8ikWAM"  # "Rachel" default
    model_id = "eleven_multilingual_v2"
    if config is not None:
        model_id = config.get("tts.elevenlabs.model") or model_id
    try:
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": key, "Content-Type": "application/json",
                     "accept": "audio/mpeg"},
            json={"text": text, "model_id": model_id},
            timeout=120,
        )
    except requests.RequestException as e:
        raise ProviderError(
            f"ElevenLabs network error: {e}",
            f"បញ្ហាបណ្តាញពេលហៅ ElevenLabs៖ {e}",
            provider="elevenlabs", status_code=None, retryable=True,
        ) from e
    _check_cancel(cancel)
    if resp.status_code != 200:
        raise ProviderError(
            f"ElevenLabs TTS failed (HTTP {resp.status_code}): {resp.text[:300]}",
            f"ការហៅ TTS ទៅ ElevenLabs បរាជ័យ (កូដ {resp.status_code})",
            provider="elevenlabs", status_code=resp.status_code,
            retryable=resp.status_code == 429 or resp.status_code >= 500,
        )
    tmp = out_wav + ".el.mp3"
    try:
        with open(tmp, "wb") as f:
            f.write(resp.content)
        return _to_24k_mono_wav(tmp, out_wav, cancel=cancel)
    finally:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass


_LOCAL_ENGINE_INFO = {          # engine: (module, pip package)
    "voxcpm": ("voxcpm", "voxcpm"),
    "f5tts": ("f5_tts", "f5-tts"),
    "cosyvoice": ("cosyvoice", "cosyvoice"),
}
_LOCAL_TTS_CACHE: dict[str, Any] = {}
_LOCAL_TTS_LOCK = threading.Lock()


def _require_local_module(engine: str) -> None:
    module, package = _LOCAL_ENGINE_INFO[engine]
    if importlib.util.find_spec(module) is None:
        hint = pip_install(package)
        raise EngineNotAvailable(
            f"TTS engine '{engine}' is not installed. Install it with: {hint}",
            f"ម៉ាស៊ីន TTS '{engine}' មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ៖ {hint}",
        )


def voxcpm_model_id(config=None) -> str:
    """VoxCPM checkpoint: setting "tts.voxcpm.model" (a HF id or a local
    folder). Default VoxCPM2: 30 languages incl. Khmer, ~8 GB VRAM."""
    mid = ""
    if config is not None:
        try:
            mid = str(config.get("tts.voxcpm.model") or "")
        except Exception:
            mid = ""
    return mid or "openbmb/VoxCPM2"


def _synth_voxcpm(text: str, out_wav: str, voice: str,
                  cancel: threading.Event | None, ref_text: str = "",
                  config=None) -> str:
    _require_local_module("voxcpm")
    try:
        os.environ.setdefault("HF_HOME", str(models_dir() / "hf"))
        from voxcpm import VoxCPM  # lazy heavy import
        mid = voxcpm_model_id(config)
        with _LOCAL_TTS_LOCK:
            model = _LOCAL_TTS_CACHE.get(("voxcpm", mid))
            if model is None:
                model = VoxCPM.from_pretrained(mid, load_denoiser=False)
                _LOCAL_TTS_CACHE[("voxcpm", mid)] = model
        _check_cancel(cancel)
        kwargs: dict[str, Any] = {"text": text, "cfg_value": 2.0,
                                  "inference_timesteps": 10}
        if voice and os.path.isfile(voice):
            # VoxCPM2: reference_wav_path clones the timbre; adding the same
            # clip + its transcript as the prompt reproduces the delivery too
            kwargs["reference_wav_path"] = voice
            if ref_text.strip():
                kwargs["prompt_wav_path"] = voice
                kwargs["prompt_text"] = ref_text.strip()
        try:
            audio = model.generate(**kwargs)
        except TypeError:
            # VoxCPM 0.5B / 1.x: no reference_wav_path, prompt needs a transcript
            kwargs.pop("reference_wav_path", None)
            if not kwargs.get("prompt_text"):
                kwargs.pop("prompt_wav_path", None)
            audio = model.generate(**kwargs)
        arr = np.asarray(audio, dtype=np.float32).flatten()
        sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", 16000) or 16000)
        tmp = out_wav + ".vox.wav"
        _write_wav_int16(tmp, arr, sr)
        try:
            return _to_24k_mono_wav(tmp, out_wav, cancel=cancel)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    except (AppError, CancelledError):
        raise
    except Exception as e:
        hint = ""
        if "out of memory" in str(e).lower():
            hint = (" — the model needs more GPU memory (VoxCPM2 ~8 GB); "
                    "set tts.voxcpm.model to a smaller checkpoint or use edge+clone")
        raise AppError(
            f"VoxCPM synthesis failed: {e}{hint}",
            f"ការបង្កើតសំឡេងតាម VoxCPM បរាជ័យ៖ {e}",
        ) from e


def _synth_f5tts(text: str, out_wav: str, voice: str,
                 cancel: threading.Event | None, ref_text: str = "") -> str:
    _require_local_module("f5tts")
    if not voice or not os.path.isfile(voice):
        raise AppError(
            "F5-TTS needs a reference audio file path passed as 'voice'",
            "F5-TTS ត្រូវការផ្លូវឯកសារសំឡេងគំរូ (reference) នៅក្នុងប៉ារ៉ាម៉ែត្រ voice",
        )
    try:
        os.environ.setdefault("HF_HOME", str(models_dir() / "hf"))
        from f5_tts.api import F5TTS  # lazy heavy import
        with _LOCAL_TTS_LOCK:
            model = _LOCAL_TTS_CACHE.get("f5tts")
            if model is None:
                model = F5TTS()
                _LOCAL_TTS_CACHE["f5tts"] = model
        _check_cancel(cancel)
        tmp = out_wav + ".f5.wav"
        # An empty ref_text makes F5 transcribe the reference itself.
        model.infer(ref_file=voice, ref_text=ref_text.strip(), gen_text=text,
                    file_wave=tmp)
        if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            raise AppError(
                "F5-TTS produced no audio",
                "F5-TTS មិនបានបង្កើតសំឡេងទេ",
            )
        try:
            return _to_24k_mono_wav(tmp, out_wav, cancel=cancel)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    except (AppError, CancelledError):
        raise
    except Exception as e:
        raise AppError(
            f"F5-TTS synthesis failed: {e}",
            f"ការបង្កើតសំឡេងតាម F5-TTS បរាជ័យ៖ {e}",
        ) from e


def _synth_cosyvoice(text: str, out_wav: str, voice: str,
                     cancel: threading.Event | None, ref_text: str = "") -> str:
    _require_local_module("cosyvoice")
    model_dir = models_dir() / "cosyvoice"
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise EngineNotAvailable(
            f"CosyVoice model files not found. Download a CosyVoice model into: {model_dir}",
            f"រកមិនឃើញឯកសារម៉ូដែល CosyVoice ទេ។ សូមទាញយកម៉ូដែលដាក់ក្នុង៖ {model_dir}",
        )
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice  # lazy heavy import
        import torch  # lazy heavy import

        with _LOCAL_TTS_LOCK:
            model = _LOCAL_TTS_CACHE.get("cosyvoice")
            if model is None:
                model = CosyVoice(str(model_dir))
                _LOCAL_TTS_CACHE["cosyvoice"] = model
        _check_cancel(cancel)
        chunks = []
        if voice and os.path.isfile(voice):
            # Zero-shot cloning from a reference clip (+ its transcript).
            from cosyvoice.utils.file_utils import load_wav  # lazy heavy import
            prompt = load_wav(voice, 16000)
            for out in model.inference_zero_shot(text, ref_text.strip(), prompt):
                _check_cancel(cancel)
                chunks.append(out["tts_speech"])
        else:
            spk = voice
            if not spk:
                spks = model.list_avaliable_spks()
                spk = spks[0] if spks else ""
            for out in model.inference_sft(text, spk):
                _check_cancel(cancel)
                chunks.append(out["tts_speech"])
        if not chunks:
            raise AppError(
                "CosyVoice produced no audio",
                "CosyVoice មិនបានបង្កើតសំឡេងទេ",
            )
        speech = torch.cat(chunks, dim=1)
        arr = speech.detach().cpu().numpy().flatten().astype(np.float32)
        sr = int(getattr(model, "sample_rate", 22050))
        tmp = out_wav + ".cosy.wav"
        _write_wav_int16(tmp, arr, sr)
        try:
            return _to_24k_mono_wav(tmp, out_wav, cancel=cancel)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    except (AppError, CancelledError):
        raise
    except Exception as e:
        raise AppError(
            f"CosyVoice synthesis failed: {e}",
            f"ការបង្កើតសំឡេងតាម CosyVoice បរាជ័យ៖ {e}",
        ) from e


def synthesize(text: str, out_wav: str, engine: str = "edge", voice: str = "",
               rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%",
               emotion: str = "", api_key: str = "", config=None,
               progress: Callable[[float, str], None] | None = None,
               cancel: threading.Event | None = None,
               ref_text: str = "") -> str:
    """Synthesize speech to a 24 kHz mono 16-bit PCM WAV file.

    For the local cloning engines `voice` may be a reference wav path and
    `ref_text` its transcript (zero-shot voice cloning)."""
    _check_cancel(cancel)
    text = (text or "").strip()
    if not text:
        raise AppError(
            "Cannot synthesize empty text",
            "មិនអាចបង្កើតសំឡេងពីអត្ថបទទទេបានទេ",
        )
    engine = (engine or "edge").strip().lower()
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.05, f"Synthesizing ({engine})…")

    if engine == "edge":
        result = _synth_edge(text, out_wav, voice, rate, pitch, volume, cancel)
    elif engine == "openai":
        result = _synth_openai(text, out_wav, voice, rate, emotion,
                               api_key, config, cancel)
    elif engine == "elevenlabs":
        result = _synth_elevenlabs(text, out_wav, voice, api_key, config, cancel)
    elif engine == "voxcpm":
        result = _synth_voxcpm(text, out_wav, voice, cancel, ref_text, config)
    elif engine == "f5tts":
        result = _synth_f5tts(text, out_wav, voice, cancel, ref_text)
    elif engine == "cosyvoice":
        result = _synth_cosyvoice(text, out_wav, voice, cancel, ref_text)
    else:
        raise AppError(
            f"Unknown TTS engine: {engine}",
            f"ម៉ាស៊ីន TTS មិនស្គាល់៖ {engine}",
        )
    if not os.path.isfile(result) or os.path.getsize(result) == 0:
        raise AppError(
            f"TTS engine '{engine}' produced no audio",
            f"ម៉ាស៊ីន TTS '{engine}' មិនបានបង្កើតសំឡេងទេ",
        )
    if progress:
        progress(1.0, "Synthesized")
    return result


# ---------------------------------------------------------------------------
# Voice clean-up and loudness
# ---------------------------------------------------------------------------

# Every finished mix is normalised to this integrated loudness (EBU R128):
# the reference level of YouTube, TikTok and Facebook, so a dub or recap
# plays as loud as other videos without being turned down.
TARGET_LUFS = -14.0
# Sample-peak ceiling of the final limiter (-1.5 dBFS): keeps the true peak
# under about -1 dBTP, so the AAC encode never clips.
PEAK_LIMIT = 0.84
# Spoken lines are levelled to this RMS (over their voiced frames).
LINE_RMS_DB = -20.0
# Above this noise floor (dBFS, the pauses inside the speech) a voice track
# has hiss worth removing — voice conversion leaves -60..-40; clean TTS
# sits near -90.
NOISY_FLOOR_DB = -60.0


def _speech_span(data: np.ndarray, sr: int, thresh: int = 300) -> tuple[int, int] | None:
    """(first, last+1) sample of the speech in a clip: the 10 ms frames whose
    RMS passes `thresh` (int16 units; 300 = -40 dBFS). Frame RMS rather than
    single samples, so the hiss of a converted voice (about -50 dBFS, with
    stray peaks above -40) is never taken for speech."""
    fr = max(1, int(0.01 * sr))
    n = data.size // fr
    if n == 0:
        return None
    x = data[:n * fr].astype(np.float64).reshape(n, fr)
    loud = np.nonzero(np.sqrt(np.mean(x ** 2, axis=1)) > thresh)[0]
    if loud.size == 0:
        return None
    return int(loud[0]) * fr, min(data.size, (int(loud[-1]) + 1) * fr)


def trim_silence(data: np.ndarray, sr: int, thresh: int = 300,
                 lead: float = 0.03, tail: float = 0.08) -> np.ndarray:
    """Drop the leading and trailing silence of a spoken clip (Edge TTS pads
    ~0.2 s before and ~0.6-0.9 s after the speech; voice conversion similar),
    keeping `lead` / `tail` seconds around the speech."""
    span = _speech_span(data, sr, thresh)
    if span is None:
        return data
    start = max(0, span[0] - int(lead * sr))
    end = min(data.size, span[1] + int(tail * sr))
    return data[start:end]


def _frame_rms(x: np.ndarray, sr: int, frame_s: float = 0.02) -> np.ndarray:
    fr = max(1, int(frame_s * sr))
    n = x.size // fr
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(np.mean(x[:n * fr].reshape(n, fr) ** 2, axis=1))


def smooth_line(data: np.ndarray, sr: int, rms_db: float = LINE_RMS_DB,
                fade_in: float = 0.01, fade_out: float = 0.04) -> np.ndarray:
    """Level one spoken line and fade its edges (int16 in, int16 out).

    The RMS of the voiced 20 ms frames is brought to `rms_db` (gain held
    within +-12 dB, peaks kept under -1 dBFS), so every line — male or
    female voice, Edge or cloned — sits at the same level; raised-cosine
    fades stop the cut edges of trimmed speech from clicking."""
    if data.size == 0:
        return data
    x = data.astype(np.float32) / 32768.0
    rms = _frame_rms(x, sr)
    voiced = rms[rms > 10 ** (-45 / 20)]
    if voiced.size:
        cur = 20 * np.log10(float(np.sqrt(np.mean(voiced ** 2))))
        gain = 10 ** (float(np.clip(rms_db - cur, -12.0, 12.0)) / 20)
        peak = float(np.max(np.abs(x))) * gain
        if peak > 0.89:                          # -1 dBFS
            gain *= 0.89 / peak
        x *= gain
    k = min(int(fade_in * sr), x.size // 2)
    if k > 1:
        x[:k] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k, dtype=np.float32))
    k = min(int(fade_out * sr), x.size // 2)
    if k > 1:
        x[-k:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, k, dtype=np.float32))
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def noise_floor_db(data: np.ndarray, sr: int, thresh: int = 300) -> float:
    """Background level (dBFS) of an UNTRIMMED spoken clip, measured in the
    padding the engine leaves before and after the words: about -90 for
    clean TTS, -60..-40 when voice conversion left hiss behind; -120 when
    the clip has no padding to measure. (Quiet frames inside the speech are
    no guide: on a short line they are soft speech, not noise.)"""
    span = _speech_span(data, sr, thresh)
    if span is None:
        return -120.0
    pad = np.concatenate([data[:max(0, span[0] - int(0.03 * sr))],
                          data[span[1] + int(0.08 * sr):]])
    if pad.size < int(0.05 * sr):
        return -120.0
    x = pad.astype(np.float64) / 32768.0
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def polish_voice(in_wav: str, out_wav: str, denoise: bool = False,
                 cancel: threading.Event | None = None) -> str:
    """Broadcast-style clean-up of a finished voice track in one ffmpeg pass:
    cut rumble below 70 Hz, remove the hiss voice conversion leaves (only
    when `denoise`: on clean TTS it would just dull the voice), tame harsh
    "s" sounds and even out the level with a gentle 2:1 compressor.
    Writes 24 kHz mono 16-bit, same length as the input."""
    chain = ["highpass=f=70:poles=2"]
    if denoise:
        chain.append("afftdn=nr=12:nf=-50:tn=1")
    chain += ["deesser=i=0.3:m=0.5:f=0.5",
              "acompressor=threshold=0.1:ratio=2:attack=10:release=150:knee=4",
              "alimiter=limit=0.89:level=0"]
    tmp = out_wav + ".polish.wav" if os.path.abspath(in_wav) == os.path.abspath(out_wav) \
        else out_wav
    _run_ffmpeg(["-i", in_wav, "-af", ",".join(chain), "-ac", "1",
                 "-ar", str(_TTS_SR), "-acodec", "pcm_s16le", tmp],
                cancel=cancel, label="ffmpeg", status_text="Cleaning up the voices…")
    if tmp != out_wav:
        os.replace(tmp, out_wav)
    return out_wav


def loudness_lufs(path: str) -> float | None:
    """Integrated loudness (EBU R128) of a file via ffmpeg, or None."""
    if not path or not os.path.isfile(path):
        return None
    try:
        r = subprocess.run([ffmpeg_path(), "-hide_banner", "-nostats", "-i", path,
                            "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                           capture_output=True, encoding="utf-8", errors="replace",
                           creationflags=_CREATE_NO_WINDOW, timeout=1200)
        m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr or "")
        v = float(m[-1]) if m else None
        return v if v is not None and v > -70 else None
    except Exception as exc:
        log.warning("loudness measurement failed for %s: %s", path, exc)
        return None


def normalize_loudness(in_wav: str, out_wav: str, target: float = TARGET_LUFS,
                       duration: float = 0.0,
                       progress: Callable[[float, str], None] | None = None,
                       cancel: threading.Event | None = None) -> str:
    """Two-pass loudness normalisation: measure the integrated loudness,
    apply one fixed gain to reach `target` LUFS (no pumping, unlike
    single-pass loudnorm), then a peak limiter at PEAK_LIMIT. Writes 48 kHz
    stereo 16-bit PCM."""
    cur = loudness_lufs(in_wav)
    gain = 0.0 if cur is None else max(-24.0, min(24.0, target - cur))
    if cur is not None:
        LOG_BUS.publish(f"[mix] loudness {cur:.1f} LUFS -> {target:.0f} LUFS "
                        f"({gain:+.1f} dB)")
    tmp = out_wav + ".norm.wav" if os.path.abspath(in_wav) == os.path.abspath(out_wav) \
        else out_wav
    _run_ffmpeg(["-i", in_wav, "-af",
                 f"volume={gain:.2f}dB,alimiter=limit={PEAK_LIMIT}:level=0",
                 "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", tmp],
                duration=duration, progress=progress, cancel=cancel,
                label="ffmpeg", status_text="Setting the loudness…")
    if tmp != out_wav:
        os.replace(tmp, out_wav)
    return out_wav


# ---------------------------------------------------------------------------
# Fitting / track assembly / mixing
# ---------------------------------------------------------------------------

def fit_to_duration(wav: str, target: float, min_tempo: float = 0.85,
                    max_tempo: float = 1.35) -> str:
    """Time-stretch a wav toward `target` seconds using chained ffmpeg atempo.
    The tempo is clamped to [min_tempo, max_tempo]; if the clip is still too
    long after clamping, it is returned as-is (caller shifts gaps)."""
    try:
        cur = get_duration(wav)
    except Exception:
        return wav
    if cur <= 0 or target <= 0:
        return wav
    tempo = cur / target
    tempo = max(min_tempo, min(max_tempo, tempo))
    if abs(tempo - 1.0) < 0.02:
        return wav

    factors: list[float] = []
    t = tempo
    while t > 2.0:
        factors.append(2.0)
        t /= 2.0
    while t < 0.5:
        factors.append(0.5)
        t /= 0.5
    factors.append(t)
    afilter = ",".join(f"atempo={f:.6f}" for f in factors)

    tmp = wav + ".fit.wav"
    try:
        _run_ffmpeg(["-i", wav, "-filter:a", afilter,
                     "-acodec", "pcm_s16le", tmp],
                    label="ffmpeg", status_text="Fitting duration…")
        os.replace(tmp, wav)
    except CancelledError:
        raise
    except Exception as e:
        log.warning("fit_to_duration failed for %s: %s", wav, e)
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
    return wav


def _read_wav_as_int16_mono(path: str, target_sr: int,
                            cancel: threading.Event | None = None) -> np.ndarray:
    """Read a wav as int16 mono at target_sr, converting via ffmpeg if the
    file is not already in that format."""
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            if sr == target_sr and ch == 1 and sw == 2:
                raw = wf.readframes(n)
                return np.frombuffer(raw, dtype=np.int16).copy()
    except Exception:
        pass  # fall through to ffmpeg conversion
    tmp = path + ".norm.wav"
    try:
        _run_ffmpeg(["-i", path, "-ac", "1", "-ar", str(target_sr),
                     "-acodec", "pcm_s16le", tmp],
                    cancel=cancel, label="ffmpeg",
                    status_text="Normalizing audio…")
        with wave.open(tmp, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).copy()
    finally:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass


def build_voice_track(lines: list[tuple[float, str]], total: float, out_wav: str,
                      voice_for_line: Callable[[int], dict],
                      progress: Callable[[float, str], None] | None = None,
                      cancel: threading.Event | None = None) -> str:
    """Synthesize each (start_seconds, text) line, fit it into its time slot
    (up to the next line's start), and place it sample-exactly into a mono
    int16 buffer at 24 kHz. `lines` is expected sorted by start time.

    Lines are only ever sped up (to at most 1.35x), never slowed down to fill
    a gap. A line still too long for its slot pushes the following lines
    later instead of being spoken over them; the delay is absorbed by the
    next pause long enough to hold it.

    voice_for_line(i) -> dict with optional keys:
      engine, voice, rate, pitch, volume, emotion, api_key, config
    """
    _check_cancel(cancel)
    if total <= 0:
        raise AppError(
            "build_voice_track: total duration must be positive",
            "រយៈពេលសរុបត្រូវតែធំជាងសូន្យ",
        )
    n_samples = int(round(total * _TTS_SR))
    buf = np.zeros(n_samples, dtype=np.int16)

    tmp_root = app_data_dir() / "temp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="voicetrack_", dir=str(tmp_root))

    n_lines = len(lines)
    synth_ok = 0
    synth_fail = 0
    pushed = 0
    busy_until = 0          # sample where the previous voiced line ends
    gap = int(0.08 * _TTS_SR)
    floors: list[float] = []
    try:
        for i, (start, text) in enumerate(lines):
            _check_cancel(cancel)
            text = (text or "").strip()
            if not text or start >= total:
                continue
            next_start = lines[i + 1][0] if i + 1 < n_lines else total
            slot = max(0.3, min(next_start, total) - start)

            try:
                vf = voice_for_line(i) or {}
            except Exception:
                vf = {}
            tmp_wav = os.path.join(tmpdir, f"line_{i:05d}.wav")
            try:
                synthesize(
                    text, tmp_wav,
                    engine=vf.get("engine", "edge"),
                    voice=vf.get("voice", ""),
                    rate=vf.get("rate", "+0%"),
                    pitch=vf.get("pitch", "+0Hz"),
                    volume=vf.get("volume", "+0%"),
                    emotion=vf.get("emotion", ""),
                    api_key=vf.get("api_key", ""),
                    config=vf.get("config"),
                    cancel=cancel,
                )
                # only the speech counts against the slot: drop the TTS
                # padding first, so less (or no) speed-up is needed
                raw = _read_wav_as_int16_mono(tmp_wav, _TTS_SR, cancel=cancel)
                floors.append(noise_floor_db(raw, _TTS_SR))
                data = trim_silence(raw, _TTS_SR)
                _write_wav_int16(tmp_wav, data, _TTS_SR)
                # leave a tiny breath gap when the slot allows it; speed up
                # only — slowing short lines to fill a pause sounds dragged
                target = slot - 0.05 if slot > 0.5 else slot
                fit_to_duration(tmp_wav, target, min_tempo=1.0)
                data = smooth_line(_read_wav_as_int16_mono(tmp_wav, _TTS_SR,
                                                           cancel=cancel), _TTS_SR)
            except CancelledError:
                raise
            except (AppError, Exception) as e:
                synth_fail += 1
                msg = getattr(e, "en", str(e))
                log.warning("Line %d TTS failed, leaving silence: %s", i, msg)
                LOG_BUS.publish(f"[tts] line {i + 1}/{n_lines} failed: {msg}")
                if progress:
                    progress(min(0.99, (i + 1) / max(1, n_lines)),
                             f"Line {i + 1}/{n_lines} failed")
                continue

            offset = int(round(start * _TTS_SR))
            if offset < busy_until:
                # the previous line is still speaking: start right after it
                offset = busy_until + gap
                pushed += 1
            n = min(data.size, n_samples - offset)
            if n > 0:
                mixed = (buf[offset:offset + n].astype(np.int32)
                         + data[:n].astype(np.int32))
                buf[offset:offset + n] = np.clip(mixed, -32768, 32767).astype(np.int16)
                busy_until = offset + n
            synth_ok += 1
            if progress:
                progress(min(0.99, (i + 1) / max(1, n_lines)),
                         f"Voiced line {i + 1}/{n_lines}")

        if synth_ok == 0 and synth_fail > 0:
            raise AppError(
                f"All {synth_fail} line(s) failed to synthesize",
                f"ការបង្កើតសំឡេងបរាជ័យទាំង {synth_fail} ជួរ",
            )

        if pushed:
            LOG_BUS.publish(f"[tts] {pushed} line(s) were too long for their slot "
                            "and start a little late — shorten them to keep sync")
        _write_wav_int16(out_wav, buf, _TTS_SR)
        noisy = bool(floors) and float(np.median(floors)) > NOISY_FLOOR_DB
        polish_voice(out_wav, out_wav, denoise=noisy, cancel=cancel)
        if progress:
            progress(1.0, f"Voice track built ({synth_ok}/{n_lines} lines)")
        return out_wav
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def mix_with_ducking(voice_wav: str, bgm_wav: str, out_wav: str,
                     duck_db: float = -18.0, fade_ms: int = 180,
                     bgm_gain_db: float = 0.0,
                     progress: Callable[[float, str], None] | None = None,
                     cancel: threading.Event | None = None) -> str:
    """Mix voice over background music with sidechain ducking. The BGM is
    compressed (ducked) whenever the voice is present — with a soft knee,
    60 ms attack and >= 0.5 s release, so it dips and comes back smoothly
    instead of pumping — then the voice is mixed on top and the result is
    normalised to TARGET_LUFS with a peak limiter."""
    _check_cancel(cancel)
    for p, what_en, what_km in ((voice_wav, "voice", "សំឡេងនិយាយ"),
                                (bgm_wav, "background music", "តន្ត្រីផ្ទៃខាងក្រោយ")):
        if not os.path.isfile(p):
            raise AppError(
                f"File not found ({what_en}): {p}",
                f"រកមិនឃើញឯកសារ ({what_km})៖ {p}",
            )
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

    ratio = max(1.5, min(20.0, abs(duck_db) / 3.0 + 1.0))
    release = max(500, int(fade_ms) * 3)
    fc = (
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,"
        f"volume={bgm_gain_db:.2f}dB[bg];"
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[sc][voice];"
        f"[bg][sc]sidechaincompress=threshold=0.02:ratio={ratio:.2f}:"
        f"attack=60:release={release}:knee=4[ducked];"
        f"[ducked][voice]amix=inputs=2:duration=longest:"
        f"dropout_transition=0:normalize=0[out]"
    )
    try:
        duration = max(get_duration(voice_wav), get_duration(bgm_wav))
    except AppError:
        duration = 0.0
    if progress:
        progress(0.0, "Mixing with ducking…")
    raw = out_wav + ".raw.wav"
    try:
        # 32-bit float in between: the sum of voice and music may pass full
        # scale; only the final limiter decides the peaks
        _run_ffmpeg(
            ["-i", bgm_wav, "-i", voice_wav,
             "-filter_complex", fc, "-map", "[out]",
             "-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", raw],
            duration=duration, progress=progress, cancel=cancel,
            label="ffmpeg", status_text="Mixing with ducking…",
        )
        normalize_loudness(raw, out_wav, duration=duration, cancel=cancel)
    finally:
        try:
            os.remove(raw)
        except OSError:
            pass
    if progress:
        progress(1.0, "Mix complete")
    return out_wav


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # to_srt formatting
    words = [Word(0.0, 0.5, "Hello"), Word(0.5, 1.0, "world")]
    seg1 = Segment(0.0, 1.0, "Hello world", words)
    seg2 = Segment(61.5, 3725.462, "Second line", [], speaker="km-KH-PisethNeural",
                   gender="M")
    t = Transcript(segments=[seg1, seg2], language="en", duration=3726.0)
    srt = t.to_srt()
    assert srt.startswith("1\n"), srt[:20]
    assert "00:00:00,000 --> 00:00:01,000" in srt, srt
    assert "00:01:01,500 --> 01:02:05,462" in srt, srt
    assert "Hello world" in srt and "Second line" in srt

    # json round-trip (via a real serialize/parse cycle)
    blob = json.dumps(t.to_json(), ensure_ascii=False)
    t2 = Transcript.from_json(json.loads(blob))
    assert t2.language == "en"
    assert abs(t2.duration - 3726.0) < 1e-9
    assert len(t2.segments) == 2
    assert t2.segments[0].text == "Hello world"
    assert len(t2.segments[0].words) == 2
    assert t2.segments[0].words[1].text == "world"
    assert abs(t2.segments[0].words[1].start - 0.5) < 1e-9
    assert t2.segments[1].speaker == "km-KH-PisethNeural"
    assert t2.segments[1].gender == "M"
    assert t2.to_srt() == srt

    # tolerant from_json
    t3 = Transcript.from_json({"segments": [{"text": "x"}]})
    assert t3.segments[0].start == 0.0 and t3.segments[0].words == []

    # assign_voices mapping
    segs = [Segment(0, 1, "a", [], gender="M"), Segment(1, 2, "b", []),
            Segment(2, 3, "c", [], gender="f")]
    assign_voices(segs)
    assert segs[0].speaker == "km-KH-PisethNeural" and segs[0].gender == "M"
    assert segs[1].speaker == "km-KH-SreymomNeural" and segs[1].gender == "F"
    assert segs[2].speaker == "km-KH-SreymomNeural" and segs[2].gender == "F"

    # voice clean-up helpers
    sr_t = _TTS_SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr_t) / sr_t) * 32767).astype(np.int16)
    pad = np.zeros(int(0.5 * sr_t), dtype=np.int16)
    clip = np.concatenate([pad, tone, pad])
    assert noise_floor_db(clip, sr_t) < -100, "digital-silence padding is clean"
    rng = np.random.default_rng(0)
    hiss = (rng.standard_normal(clip.size) * 0.003 * 32767).astype(np.int16)   # ~ -50 dBFS
    assert noise_floor_db(clip + hiss, sr_t) > NOISY_FLOOR_DB, "hissy padding is noisy"
    trimmed = trim_silence(clip, sr_t)
    assert abs(trimmed.size / sr_t - (1.0 + 0.03 + 0.08)) < 0.01, trimmed.size / sr_t
    sm = smooth_line(trimmed, sr_t)
    assert sm.dtype == np.int16 and sm[0] == 0 and abs(int(sm[-1])) < 50, "faded edges"
    lvl = 20 * np.log10(np.sqrt(np.mean((sm[sr_t // 10:-sr_t // 10] / 32768.0) ** 2)))
    assert abs(lvl - LINE_RMS_DB) < 1.0, lvl
    assert faster_rate("+0%", 1.3) == "+30%" and faster_rate("+50%", 1.1) == "+65%"
    assert faster_rate("+90%", 1.5) == "+100%", "Edge rate capped"

    # srt time edge cases
    assert _fmt_srt_time(0) == "00:00:00,000"
    assert _fmt_srt_time(3599.9995) == "01:00:00,000"
    assert _fmt_srt_time(-1) == "00:00:00,000"

    # review marks and the second listen's report survive the json round trip
    rt = Transcript([Segment(1, 2, "嗯", [Word(1, 2, "嗯")], review="recovered")], "zh", 3.0,
                    recovered=1, unheard=[(2.0, 2.6, "nothing recognised", "")])
    rt2 = Transcript.from_json(json.loads(json.dumps(rt.to_json(), ensure_ascii=False)))
    assert rt2.segments[0].review == "recovered" and rt2.recovered == 1
    assert rt2.unheard == [(2.0, 2.6, "nothing recognised", "")], rt2.unheard
    assert "review" not in Segment(0, 1, "x", []).to_json(), "no key when nothing to check"

    # speech units: a Latin word is one, each Chinese character is one
    assert _units("鈴鐺我們下期見by bwd") == list("鈴鐺我們下期見") + ["by", "bwd"]
    assert _units("Okay, fine! I'm out.") == ["okay", "fine", "i'm", "out"]
    assert _spaced(_units("Stand by me")) and not _spaced(_units("下期見by bwd"))

    # what the second listen refuses: Whisper's stock phrases and loops, more
    # words than fit the time, its own "no speech" — never a real short line
    for text, secs, refused in [
            ("谢谢大家", 1.0, True), ("能收到最新消息哦", 0.76, True), ("阅我的频道", 0.76, True),
            ("鈴鐺我們下期見by bwd", 0.76, True), ("Thank you for watching!", 1.5, True),
            ("喲喲喲喲喲喲喲喲喲", 2.4, True), ("想不想看我做的那一集 看你好吃看", 0.76, True),
            ("顾川", 0.6, False), ("嗯", 0.5, False), ("哈哈哈哈", 1.0, False),
            ("谢谢大家帮我们找回来了这么多东西", 2.5, False), ("Stand by me.", 1.0, False),
            ("气死我了他一直", 1.2, False), ("Okay, fine! I'm freaked out.", 1.6, False)]:
        why = _refusal(text, (0.0, 0.0), [], secs)
        assert bool(why) == refused, (text, why)
    assert _refusal("嗯", (-1.5, 0.9), [], 0.5) == "Whisper heard no speech"
    assert _refusal("我记得你", (0.0, 0.0), list("顾川我记得你之前"), 0.6) == "duplicate"

    # recover_gaps with a stand-in recogniser over four gaps: a missed name
    # is taken (marked for review, words timed in the video), a stock phrase
    # and a gap with no words of its own are reported as unheard, and words
    # already in the transcript are left out without a report
    from types import SimpleNamespace as _NS
    first = [Segment(3.0, 4.6, "你好", [Word(3.0, 3.5, "你"), Word(3.5, 4.6, "好")]),
             Segment(31.2, 32.0, "我记得你", [Word(31.2, 31.6, "我记得"), Word(31.6, 32.0, "你")])]

    class _FakeModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, clip, **kw):
            assert kw["vad_filter"] is False and kw["no_speech_threshold"] is None
            self.calls += 1
            # times are within the clip, which starts 1 s before each gap
            heard = {1: [("顾", 1.1, 1.4), ("川", 1.4, 1.8)],
                     2: [("谢谢", 1.0, 1.4), ("大家", 1.4, 1.7)],
                     3: [("嗯", 0.2, 0.5)],                    # in the context only
                     4: [("我记得", 1.1, 1.5), ("你", 1.5, 1.8)]}[self.calls]
            ws = [_NS(word=t, start=s, end=e) for t, s, e in heard]
            return iter([_NS(words=ws, avg_logprob=-0.3, no_speech_prob=0.05)]), None

    _real_gaps = find_speech_gaps
    find_speech_gaps = lambda x, segs, sr=16000: [(5.0, 6.0), (10.0, 10.8), (20.0, 20.8),
                                                  (30.0, 31.0)]
    fm = _FakeModel()
    got, unheard = recover_gaps(fm, np.zeros(16000 * 40, np.float32), first, "zh")
    find_speech_gaps = _real_gaps
    rec = [s for s in got if s.review == "recovered"]
    assert fm.calls == 4 and [s.text for s in rec] == ["顾川"], [s.text for s in got]
    assert abs(rec[0].start - 5.1) < 1e-6 and abs(rec[0].end - 5.8) < 1e-6, rec[0]
    assert [s.start for s in got] == sorted(s.start for s in got), "time order kept"
    assert [(u[0], u[2]) for u in unheard] == [(10.0, "Whisper stock phrase"),
                                               (20.0, "nothing recognised")], unheard

    # Deepgram with no language given (the recap): the language is found
    # on this PC first and handed over; no speech heard -> Whisper (#22)
    from unittest import mock as _mock

    from app.engines import deepgram_stt as _dg

    class _Cfg:
        def get(self, k, d=None):
            return "deepgram" if k == "stt.engine" else d

        def get_secret(self, k):
            return "dg-test-key"

    asked = {}

    def _dg_transcribe(audio, key, language="", **kw):
        asked["dg"] = language
        heard = asked.get("dg_heard", True)
        return Transcript(segments=[Segment(0.5, 1.5, "我能赚吗", [])] if heard else [],
                          language=language or "fr", duration=45.0)

    def _whisper_transcribe(self, audio, language=None, **kw):
        asked["whisper"] = (self.model_size, language)
        return Transcript(segments=[Segment(0.5, 1.5, "我能赚吗", [])], language=language or "",
                          duration=45.0)

    with _mock.patch.object(_dg, "transcribe", _dg_transcribe), \
            _mock.patch(f"{__name__}.installed_whisper_size", lambda: "small"), \
            _mock.patch.object(WhisperEngine, "detect_language", lambda self, a: ("zh", 0.98)), \
            _mock.patch.object(WhisperEngine, "transcribe", _whisper_transcribe):
        said = []
        tr = speech_to_text("x.wav", config=_Cfg(), progress=lambda f, t: said.append(t))
        assert asked["dg"] == "zh" and tr.language == "zh", asked
        assert "Detecting the spoken language" in said, said
        asked.clear()
        speech_to_text("x.wav", config=_Cfg(), language="ja")
        assert asked["dg"] == "ja", "a given language is used as it is"
        asked.clear()
        asked["dg_heard"] = False
        tr = speech_to_text("x.wav", config=_Cfg(), model_size="medium")
        assert asked["whisper"] == ("medium", "zh") and tr.segments, asked
    with _mock.patch.object(_dg, "transcribe", _dg_transcribe), \
            _mock.patch(f"{__name__}.installed_whisper_size", lambda: ""):
        asked.clear()
        speech_to_text("x.wav", config=_Cfg())
        assert asked["dg"] == "", "no Whisper model on the PC: Deepgram decides, as before"

    print("PASS")
