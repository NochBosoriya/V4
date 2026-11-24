"""Recap montage assembler: turns a recap timeline into a finished video.

For every TimelineRow with narration:
  1. synthesize the narration (TTS) and measure its real duration,
  2. cut the row's source footage [start, end] so the clip lasts exactly as
     long as the narration (+ a short breath gap) — trimming when the slot is
     longer, gently slowing the footage (up to ``max_slowdown``) and then
     extending into the following footage when the slot is shorter,
  3. place the narration sample-exactly on a new voice track.

The clips are encoded one by one (fast ``-ss`` seeking, identical stream
parameters), joined with the concat demuxer, and a final pass mixes the voice
over the (ducked) original audio and/or an optional looped BGM file. An SRT
with the new narration timings is always written next to the output and can
optionally be burned in.

No Qt imports. Only stdlib + numpy at module level; audio_pipeline and
media_exporter are imported lazily inside functions.
"""
from __future__ import annotations

import os
import re
import shutil
import statistics
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from app.core.errors import AppError, CancelledError
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import ffmpeg_path, temp_dir

log = get_logger("montage_engine")

ProgressFn = Callable[[float, str], None]

# Narration presets (MontageOptions fields). "storyteller_km" was matched to
# the reference recap in "sample video/": Edge Piseth sped up ~50% scored a
# 0.97 ECAPA speaker similarity with the same 129 Hz pitch and pace.
NARRATOR_PRESETS: dict[str, dict] = {
    "storyteller_km": {"tts_engine": "edge", "tts_voice": "km-KH-PisethNeural",
                       "tts_rate": "+50%", "tts_pitch": "+0Hz"},
}

# Progress budget for the three phases.
_P_TTS = (0.0, 0.35)
_P_CLIPS = (0.35, 0.85)
_P_FINAL = (0.85, 1.0)

_AUDIO_SR = 48000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MontageOptions:
    """All knobs for assemble_recap(). Every field has a sensible default."""

    # --- voice (passed to audio_pipeline.synthesize) ---
    tts_engine: str = "edge"
    tts_voice: str = "km-KH-SreymomNeural"
    tts_rate: str = "+0%"
    tts_pitch: str = "+0Hz"
    tts_volume: str = "+0%"

    # --- timing ---
    gap_seconds: float = 0.25          # silence after each narration line
    max_slowdown: float = 1.35         # max footage slow-motion factor
    max_tts_speedup: float = 1.15      # max narration speed-up before extending
    min_clip_seconds: float = 1.0

    # --- audio mix ---
    original_audio_db: float = -16.0   # source audio level; <= -60 mutes it
    # background without the original dialogue: the source's music & effects
    # stem (audio_pipeline.separated_stems, cached); falls back to the
    # original audio when no separator is installed
    remove_voices: bool = True
    bgm_path: str = ""                 # optional music file (looped)
    bgm_db: float = -14.0
    duck_db: float = -18.0             # extra ducking under the narration
    voice_gain_db: float = 0.0
    sfx: bool = False                  # whoosh on cuts + impact on first/last scene
    sfx_db: float = -10.0

    # --- video ---
    width: int = 0                     # 0 = keep source size
    height: int = 0
    fps: float = 0.0                   # 0 = keep source fps
    fit_mode: str = "blur"             # "blur" | "pad" | "crop" | "smart" (when aspect differs)
    snap_to_scenes: bool = True        # start clips on shot changes
    fade_seconds: float = 0.0          # 0 = hard cuts; else fade in/out per clip
    vcodec: str = "libx264"
    bitrate: str = ""                  # "" = quality-based (crf/cq)

    # --- subtitles ---
    burn_subtitles: bool = False
    subtitle_font: str = "Kantumruy Pro"
    subtitle_max_chars: int = 42
    subtitle_style: str = ""           # app.engines.subtitles preset ("" = plain)

    keep_temp: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MontageOptions":
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class MontageClip:
    """How one timeline row was placed in the output."""
    index: int
    src_start: float
    src_len: float
    slowdown: float
    out_start: float
    out_len: float
    voice_len: float
    narration: str
    crop_x: float = 0.5                # subject centre for "smart" framing


@dataclass
class MontageResult:
    out_path: str
    srt_path: str
    voice_path: str
    duration: float
    clips: list[MontageClip] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)   # rows with failed TTS


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


def _emit(progress: ProgressFn | None, frac: float, text: str) -> None:
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, frac)), text)
    except Exception:
        log.debug("progress callback raised", exc_info=True)


def _sub(progress: ProgressFn | None, lo: float, hi: float) -> ProgressFn | None:
    if progress is None:
        return None

    def fn(f: float, t: str) -> None:
        _emit(progress, lo + (hi - lo) * max(0.0, min(1.0, float(f))), t)
    return fn


def _even(v: float) -> int:
    v = int(round(v))
    return max(2, v - (v % 2))


def _fmt_srt(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?។៕！？。…])\s*")


_KM_COENG = "\u17d2"


def _km_base(ch: str) -> bool:
    """A Khmer consonant or independent vowel (starts or ends a syllable)."""
    return "\u1780" <= ch <= "\u17b3"


def _km_mark(ch: str) -> bool:
    """A Khmer dependent vowel, sign, coeng, repeat sign (ៗ) or full stop
    (។): belongs to the letter before, never starts a line."""
    return "\u17b4" <= ch <= "\u17dd"


def _syllables(word: str) -> list[str]:
    """Split a word without spaces where a line may break: Khmer between
    written syllables (a consonant with its subscript, vowel, signs and final
    consonant: ចាប់, នាង, ដំ|បូង — never inside one), other scripts between
    characters (Chinese) or not at all. Joining the pieces gives the word."""
    out: list[str] = []
    for i, ch in enumerate(word):
        prev = word[i - 1] if i else ""
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if not out:
            out.append(ch)
        elif _km_mark(ch) or prev == _KM_COENG:
            out[-1] += ch                       # vowel/sign/subscript: same syllable
        elif _km_base(ch):
            last = out[-1]
            # has a vowel or sign (a subscript alone is not one: ទ្រ|របស់)
            full = _km_base(last[0]) and any(_km_mark(c) and c != _KM_COENG for c in last)
            closed = (_km_base(last[-1]) and len(last) > 1 and last[-2] != _KM_COENG) \
                or last[-1] in "\u17cb\u17cc\u17cd\u17ce"
            nxt2 = word[i + 2] if i + 2 < len(word) else ""
            final = (nxt in ("\u17cb", "\u17cc", "\u17cd", "\u17ce") or not _km_mark(nxt)) \
                and not (_km_base(nxt) and nxt2 in ("\u17cb", "\u17cc", "\u17cd", "\u17ce"))
            # (យ in នៅយប់ is not នៅ's final: ប់ closes a new syllable យប់)
            marked = nxt in ("\u17cb", "\u17cc", "\u17cd", "\u17ce")
            # a bare consonant after a vowel is that syllable's final (នា+ង);
            # one with a final mark always is (យ+ប់)
            if (not closed and _km_base(last[0])
                    and ((full and final and nxt != _KM_COENG) or marked)):
                out[-1] += ch
            else:
                out.append(ch)
        elif "㐀" <= ch <= "鿿" or not ("\u1780" <= prev <= "\u17ff"):
            out.append(ch)                      # Chinese etc.: any character
        else:
            out.append(ch)
    # a lone consonant is ambiguous without a dictionary (អ|រគុណ, ទ្រ|របស់,
    # អ្ន|ក): never break next to it — it joins the next piece (the one
    # before at the end of the word)
    joined: list[str] = []
    carry = ""
    for piece in out:
        if len(piece) == 1 and _km_base(piece):
            carry += piece
            continue
        joined.append(carry + piece)
        carry = ""
    if carry:
        if joined:
            joined[-1] += carry
        else:
            joined.append(carry)
    out = joined
    # non-Khmer, non-CJK letters (a Latin word) stay one piece
    merged: list[str] = []
    for piece in out:
        if (merged and piece.isascii() and piece.isalnum()
                and merged[-1].isascii() and merged[-1][-1:].isalnum()):
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def _split_caption(text: str, max_chars: int) -> list[str]:
    """Split narration into caption-sized chunks: sentences first, then
    even pieces broken at spaces (or, in words longer than half a caption,
    between Khmer syllables / Chinese characters). A hard cut every
    max_chars once split ដំបូង and left "ង" alone on screen for 0.07 s."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    max_chars = max(10, int(max_chars))
    out: list[str] = []
    for sent in (s.strip() for s in _SENT_SPLIT.split(text)):
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        units: list[str] = []
        for w in re.findall(r"\S+\s*", sent):
            if len(w.strip()) > max_chars // 2:
                pieces = _syllables(w.rstrip())
                pieces[-1] += w[len(w.rstrip()):]
                units.extend(pieces)
            else:
                units.append(w)
        # even pieces: each aims at an equal share of what is left
        done = 0

        def target() -> float:
            rest = len(sent) - done
            # aim a little under the limit: pieces end on whole syllables/words
            return rest / max(1, -(-rest // int(max_chars * 0.9)))

        held: list[str] = []          # units of the piece being filled

        def cut(n_units: int) -> None:
            nonlocal done, held
            piece = "".join(held[:n_units])
            out.append(piece.strip())
            done += len(piece)
            held = held[n_units:]

        def at_space(goal: float) -> int:
            """Units up to the last space that still makes a fair piece
            (0 = none): a break between words reads better."""
            size = 0
            best = 0
            for j, h in enumerate(held):
                size += len(h)
                if h[-1:].isspace() and size >= 0.6 * goal and j + 1 < len(held):
                    best = j + 1
            return best

        for u in units:
            if held and len(("".join(held) + u).strip()) > max_chars:
                cut(at_space(target()) or len(held))
            held.append(u)
            left = len(sent) - done - len("".join(held))
            if left > 0 and len("".join(held).strip()) >= target():
                cut(at_space(target()) or len(held))
        while held:
            cut(len(held))
        out[:] = [o for o in out if o]
    return out


def build_srt(clips: list[MontageClip], max_chars: int = 42) -> str:
    """SRT for the assembled video: each narration spread over its voice span,
    chunk durations proportional to chunk length."""
    blocks: list[str] = []
    n = 0
    for c in clips:
        chunks = _split_caption(c.narration, max_chars)
        if not chunks:
            continue
        total_chars = sum(len(x) for x in chunks) or 1
        t = c.out_start
        span = max(0.3, c.voice_len)
        for ch in chunks:
            d = span * len(ch) / total_chars
            n += 1
            blocks.append(f"{n}\n{_fmt_srt(t)} --> {_fmt_srt(t + d)}\n{ch}\n")
            t += d
    return "\n".join(blocks)


def _atempo_chain(tempo: float) -> str:
    """atempo only accepts 0.5..2.0 per instance — chain when needed."""
    factors: list[float] = []
    t = tempo
    while t > 2.0:
        factors.append(2.0)
        t /= 2.0
    while t < 0.5:
        factors.append(0.5)
        t /= 0.5
    factors.append(t)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def _row_get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


# ---------------------------------------------------------------------------
# Planning (pure — unit-testable without ffmpeg)
# ---------------------------------------------------------------------------

def plan_clip(src_start: float, src_end: float, voice_len: float,
              video_duration: float, opts: MontageOptions,
              cuts: list[float] | None = None) -> tuple[float, float, float, float]:
    """Decide how to cut one row.

    Returns (src_start, src_len, slowdown, out_len):
      out_len   = voice_len + gap (>= min_clip_seconds)
      slowdown  = playback slow factor applied to the footage (>= 1.0)
      src_len   = source seconds consumed (= out_len / slowdown)
    """
    out_len = max(opts.min_clip_seconds, voice_len + max(0.0, opts.gap_seconds))
    slot = max(0.0, src_end - src_start)
    if slot >= out_len or slot <= 0.05:
        slowdown = 1.0
    else:
        slowdown = min(max(1.0, opts.max_slowdown), out_len / slot)
    src_len = out_len / slowdown
    start = max(0.0, src_start)
    if cuts:
        from app.engines.scenes import snap_start
        start = max(0.0, snap_start(start, src_len, cuts))
    if video_duration > 0:
        if src_len > video_duration:
            # Whole source is too short: slow it down as much as needed.
            src_len = max(0.1, video_duration)
            slowdown = out_len / src_len
            start = 0.0
        elif start + src_len > video_duration:
            start = max(0.0, video_duration - src_len)
    return start, src_len, slowdown, out_len


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assemble_recap(video: str, rows: list, out_path: str,
                   options: MontageOptions | None = None, config=None,
                   progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> MontageResult:
    """Build the narrated recap video from timeline rows.

    `rows` are recap_engine.TimelineRow objects (or dicts with the same keys:
    start, end, narration). `config` is the ConfigManager (needed for
    OpenAI/ElevenLabs TTS keys). Returns a MontageResult.
    """
    from app.engines import audio_pipeline as ap
    from app.engines import media_exporter as me

    opts = options or MontageOptions()
    _check_cancel(cancel)
    if not video or not os.path.isfile(video):
        raise AppError(f"Source video not found: {video}",
                       f"រកមិនឃើញវីដេអូប្រភព៖ {video}")
    if opts.bgm_path and not os.path.isfile(opts.bgm_path):
        raise AppError(f"Background music file not found: {opts.bgm_path}",
                       f"រកមិនឃើញឯកសារតន្ត្រីផ្ទៃខាងក្រោយ៖ {opts.bgm_path}")

    items = []
    for i, r in enumerate(rows or []):
        text = str(_row_get(r, "narration", "") or "").strip()
        if not text:
            continue
        try:
            s = float(_row_get(r, "start", 0.0) or 0.0)
            e = float(_row_get(r, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        items.append((i, s, max(s, e), text))
    if not items:
        raise AppError(
            "The timeline has no rows with narration to assemble.",
            "តារាងពេលវេលាគ្មានជួរដែលមានការនិទានសម្រាប់ផ្គុំទេ។")

    info = me.probe(video)
    has_audio = bool(info.acodec)
    src_w, src_h = info.width or 1280, info.height or 720
    out_w = _even(opts.width) if opts.width > 0 else _even(src_w)
    out_h = _even(opts.height) if opts.height > 0 else _even(src_h)
    fps = opts.fps if opts.fps > 0 else (info.fps if 0 < info.fps <= 120 else 30.0)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="montage_", dir=str(temp_dir()))
    log.info("Montage work dir: %s", work)

    try:
        # ---------------- Phase 1: narration TTS ------------------------
        voiced: dict[int, tuple[int, float, float, str, str, float]] = {}
        floor_of: dict[int, float] = {}
        failed: dict[int, str] = {}       # k -> why (not voiced yet)
        sr = ap._TTS_SR
        edge = (opts.tts_engine or "edge").strip().lower() == "edge"
        n = len(items)

        def voice_line(k: int) -> None:
            idx, s, e, text = items[k]
            wav = os.path.join(work, f"tts_{k:04d}.wav")

            def speak(rate: str) -> tuple[float, Any]:
                """(background level of the TTS padding, trimmed speech)"""
                ap.synthesize(text, wav, engine=opts.tts_engine,
                              voice=opts.tts_voice, rate=rate,
                              pitch=opts.tts_pitch, volume=opts.tts_volume,
                              config=config, cancel=cancel)
                raw = ap._read_wav_as_int16_mono(wav, sr, cancel=cancel)
                # only the speech: the TTS pads ~0.2 s before and ~0.8 s
                # after it, which would leave dead air in every clip
                return ap.noise_floor_db(raw, sr), ap.trim_silence(raw, sr)
            floor, data = speak(opts.tts_rate)
            vlen = data.size / sr
            # Narration far longer than the slot even with slow-mo: speed
            # the voice up a little (bounded) before eating extra footage
            # — an Edge voice re-spoken faster (its rate is exact and
            # sounds natural), other engines time-stretched.
            slot = e - s
            limit = slot * max(1.0, opts.max_slowdown)
            if slot > 0 and vlen > limit and opts.max_tts_speedup > 1.0:
                speed = min(opts.max_tts_speedup, vlen / limit * 1.03)
                if edge:
                    floor, data = speak(ap.faster_rate(opts.tts_rate, speed))
                else:
                    ap._write_wav_int16(wav, data, sr)
                    ap.fit_to_duration(wav, limit, min_tempo=1.0,
                                       max_tempo=opts.max_tts_speedup)
                    data = ap._read_wav_as_int16_mono(wav, sr, cancel=cancel)
            # one level for every line; faded edges never click
            data = ap.smooth_line(data, sr)
            ap._write_wav_int16(wav, data, sr)
            floor_of[k] = floor
            voiced[k] = (idx, s, e, text, wav, data.size / sr)

        def attempt(k: int) -> None:
            try:
                voice_line(k)
                failed.pop(k, None)
            except CancelledError:
                raise
            except Exception as exc:
                msg = getattr(exc, "en", str(exc))
                if k not in failed:
                    log.warning("TTS failed for row %d: %s", items[k][0], msg)
                failed[k] = msg

        for k in range(n):
            _check_cancel(cancel)
            _emit(progress, _P_TTS[0] + (_P_TTS[1] - _P_TTS[0]) * k / n,
                  f"Voicing line {k + 1}/{n}")
            attempt(k)
        # Edge TTS fails in bursts ("No audio was received", ~30 % of requests
        # for minutes): a failed line used to be left out, and a recap of 8
        # lines came out with 4 (no error). Failed lines get more tries over
        # ~90 s, the way the dub voices its lines; still failing = an error.
        waiter = cancel if cancel is not None else threading.Event()
        for wait in (3.0, 6.0, 12.0, 24.0, 48.0):
            if not failed:
                break
            LOG_BUS.publish(f"[montage] {len(failed)} narration line(s) not voiced yet; "
                            f"the voice service is dropping requests — trying again "
                            f"in {wait:.0f} s")
            _emit(progress, _P_TTS[1], f"Voice service busy — retrying "
                                        f"{len(failed)} line(s) in {wait:.0f} s")
            if waiter.wait(wait):
                _check_cancel(cancel)
            for k in sorted(failed):
                _check_cancel(cancel)
                attempt(k)
        if failed:
            which = ", ".join(str(items[k][0] + 1) for k in sorted(failed))
            why = failed[min(failed)]
            raise AppError(
                f"Text-to-speech failed for narration line(s) {which} of {n} even "
                f"after retrying for 90 s ({why}). The recap was not made with lines "
                "missing: try again in a few minutes, or pick another voice or engine.",
                f"ការបង្កើតសំឡេងបរាជ័យសម្រាប់ជួរ {which} នៃ {n} ទោះបីសាកម្តងទៀត "
                f"អស់ 90 វិនាទី ({why})។ សូមសាកម្តងទៀតក្នុងរយៈពេលប៉ុន្មាននាទីទៀត "
                "ឬជ្រើសសំឡេង ឬម៉ាស៊ីនផ្សេង។")
        voices = [voiced[k] for k in range(n)]
        floors = [floor_of[k] for k in range(n)]
        skipped: list[int] = []           # kept in the result: none any more

        # ---------------- Plan the timeline -----------------------------
        cuts: list[float] = []
        if opts.snap_to_scenes:
            from app.engines.scenes import detect_scenes
            try:
                _emit(progress, _P_TTS[1], "Finding scene cuts")
                cuts = detect_scenes(video, cancel=cancel)
            except CancelledError:
                raise
            except Exception as exc:
                log.warning("Scene detection failed (%s); clips not snapped", exc)
        clips: list[MontageClip] = []
        t = 0.0
        for idx, s, e, text, wav, vlen in voices:
            st, slen, slow, olen = plan_clip(s, e, vlen, info.duration, opts,
                                             cuts=cuts)
            clips.append(MontageClip(index=idx, src_start=st, src_len=slen,
                                     slowdown=slow, out_start=t, out_len=olen,
                                     voice_len=vlen, narration=text))
            t += olen
        total = t
        if (opts.fit_mode or "").lower() == "smart":
            from app.engines.scenes import subject_x
            import cv2  # lazy heavy import
            cap = cv2.VideoCapture(video)
            try:
                for c in clips:
                    _check_cancel(cancel)
                    c.crop_x = subject_x(video, c.src_start,
                                         c.src_start + c.src_len, cap=cap)
            finally:
                cap.release()

        # ---------------- Background without the original voices --------
        bed = ""
        if opts.remove_voices and has_audio and opts.original_audio_db > -60:
            try:
                _emit(progress, _P_TTS[1], "Removing the original voices from the background")
                _v, bed = ap.separated_stems(video, cancel=cancel)
                LOG_BUS.publish("[montage] background: music & effects only "
                                "(original voices removed)")
            except CancelledError:
                raise
            except Exception as exc:
                log.warning("No voice separation (%s); keeping the original audio",
                            getattr(exc, "en", exc))
                LOG_BUS.publish("[montage] voice separation unavailable — the "
                                "original audio stays under the narration")

        # ---------------- Phase 2: encode clips -------------------------
        vargs = me._vcodec_args(me.ExportPreset(vcodec=opts.vcodec or "libx264",
                                                bitrate=opts.bitrate or ""))
        clip_files: list[str] = []
        for k, c in enumerate(clips):
            _check_cancel(cancel)
            lo = _P_CLIPS[0] + (_P_CLIPS[1] - _P_CLIPS[0]) * k / len(clips)
            hi = _P_CLIPS[0] + (_P_CLIPS[1] - _P_CLIPS[0]) * (k + 1) / len(clips)
            clip_path = os.path.join(work, f"clip_{k:04d}.mp4")
            cmd = _clip_command(video, c, clip_path, has_audio, out_w, out_h,
                                fps, opts, vargs, src_w, src_h, bed=bed)
            me._run_ffmpeg(cmd, c.out_len, _sub(progress, lo, hi), cancel,
                           label="montage",
                           status_text=f"Clip {k + 1}/{len(clips)}")
            clip_files.append(clip_path)

        # ---------------- Phase 3: voice track + join + mix -------------
        _check_cancel(cancel)
        _emit(progress, _P_FINAL[0], "Building narration track")
        voice_path = os.path.join(work, "narration.wav")
        _build_narration_track(ap, voices, clips, total, voice_path, cancel)
        floor = statistics.median(floors) if floors else -120.0
        ap.polish_voice(voice_path, voice_path, denoise=floor > ap.NOISY_FLOOR_DB,
                        cancel=cancel)
        sfx_path = ""
        if opts.sfx and clips:
            from app.engines.sfx import build_sfx_track
            _emit(progress, _P_FINAL[0] + 0.01, "Adding sound effects")
            cut_times = [c.out_start for c in clips[1:]]
            hits = [0.0] + ([clips[-1].out_start] if len(clips) > 2 else [])
            sfx_path = build_sfx_track(cut_times, total, os.path.join(work, "sfx.wav"),
                                       hits=hits)

        list_file = os.path.join(work, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as fh:
            for p in clip_files:
                fh.write(f"file '{me._concat_escape(p)}'\n")

        srt_text = build_srt(clips, opts.subtitle_max_chars)
        srt_path = os.path.splitext(out_path)[0] + ".srt"
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(srt_text)
        burn_path = srt_path
        if opts.subtitle_style:
            from app.engines.subtitles import srt_to_ass
            burn_path = srt_to_ass(srt_path, os.path.splitext(out_path)[0] + ".ass",
                                   out_w, out_h, opts.subtitle_style)
        elif opts.burn_subtitles:
            # wrap plain subtitles for the real output size (9:16 too)
            burn_path = me._plain_srt_to_ass(srt_path, opts.subtitle_font, (out_w, out_h))

        # audio first (32-bit float: the sum may pass full scale), then one
        # loudness gain + peak limiter while the video is written
        mix_wav = os.path.join(work, "mix.wav")
        _emit(progress, _P_FINAL[0] + 0.02, "Mixing audio")
        me._run_ffmpeg(_mix_command(list_file, voice_path, mix_wav, opts, total,
                                    sfx_wav=sfx_path,
                                    bed_gain_db=6.0 if bed else 0.0),
                       total, _sub(progress, _P_FINAL[0] + 0.02, 0.9), cancel,
                       label="montage", status_text="Mixing audio")
        cur = ap.loudness_lufs(mix_wav)
        gain = 0.0 if cur is None else max(-24.0, min(24.0, ap.TARGET_LUFS - cur))
        if cur is not None:
            LOG_BUS.publish(f"[montage] loudness {cur:.1f} LUFS -> "
                            f"{ap.TARGET_LUFS:.0f} LUFS ({gain:+.1f} dB)")
        cmd = _final_command(list_file, mix_wav, out_path, burn_path,
                             opts, vargs, me, gain_db=gain)
        _emit(progress, 0.9, "Finalizing")
        me._run_ffmpeg(cmd, total, _sub(progress, 0.9, 0.995),
                       cancel, label="montage", status_text="Finalizing")

        kept_voice = os.path.splitext(out_path)[0] + ".narration.wav"
        try:
            shutil.copyfile(voice_path, kept_voice)
        except OSError:
            kept_voice = ""

        _emit(progress, 1.0, f"Recap video ready ({total:.0f}s, {len(clips)} clips)")
        LOG_BUS.publish(f"[montage] done: {out_path} ({total:.1f}s, "
                        f"{len(clips)} clips, {len(skipped)} skipped)")
        return MontageResult(out_path=out_path, srt_path=srt_path,
                             voice_path=kept_voice, duration=total,
                             clips=clips, skipped=skipped)
    finally:
        if not opts.keep_temp:
            shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# ffmpeg command builders
# ---------------------------------------------------------------------------

def _video_chain(c: MontageClip, out_w: int, out_h: int, fps: float,
                 opts: MontageOptions, src_w: int = 0, src_h: int = 0) -> str:
    """filter_complex for one clip; input [0:v] -> [v]."""
    pre = f"[0:v]setpts={c.slowdown:.6f}*(PTS-STARTPTS)"
    mode = (opts.fit_mode or "blur").lower()
    if mode == "smart" and src_w and src_h and src_w * out_h > src_h * out_w:
        # narrower output (e.g. 9:16 from 16:9): crop around the subject
        sw = _even(src_w * out_h / src_h)
        x = int(max(0, min(sw - out_w, c.crop_x * sw - out_w / 2)))
        body = (f"{pre},scale={sw}:{out_h},crop={out_w}:{out_h}:{x}:0,"
                f"setsar=1[fit]")
    elif mode in ("crop", "smart"):
        body = (f"{pre},scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},setsar=1[fit]")
    elif mode == "pad":
        body = (f"{pre},scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[fit]")
    else:  # blur: blurred, cropped copy of the frame behind the fitted frame
        body = (f"{pre},split=2[bga][fga];"
                f"[bga]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},gblur=sigma=24,eq=brightness=-0.08[bg];"
                f"[fga]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[fit]")
    tail = f"[fit]fps={fps:.4f},format=yuv420p"
    fd = max(0.0, float(opts.fade_seconds))
    if fd > 0 and c.out_len > fd * 2 + 0.1:
        tail += (f",fade=t=in:st=0:d={fd:.3f}"
                 f",fade=t=out:st={c.out_len - fd:.3f}:d={fd:.3f}")
    return f"{body};{tail}[v]"


def _clip_command(video: str, c: MontageClip, out: str, has_audio: bool,
                  out_w: int, out_h: int, fps: float, opts: MontageOptions,
                  vargs: list[str], src_w: int = 0, src_h: int = 0,
                  bed: str = "") -> list[str]:
    """One clip. Its audio is the source's, or `bed` (the source's music &
    effects without the voices) cut at the same times."""
    graph = _video_chain(c, out_w, out_h, fps, opts, src_w, src_h)
    cmd = [ffmpeg_path(), "-y", "-hide_banner",
           "-ss", f"{c.src_start:.3f}", "-t", f"{c.src_len + 0.05:.3f}",
           "-i", video,
           "-f", "lavfi", "-t", f"{c.out_len:.3f}",
           "-i", f"anullsrc=r={_AUDIO_SR}:cl=stereo"]
    if bed:
        cmd += ["-ss", f"{c.src_start:.3f}", "-t", f"{c.src_len + 0.05:.3f}",
                "-i", bed]
    if has_audio:
        afil = f"[{2 if bed else 0}:a]asetpts=PTS-STARTPTS"
        if abs(c.slowdown - 1.0) > 1e-3:
            afil += "," + _atempo_chain(1.0 / c.slowdown)
        # short fades at both ends: hard cuts in the background sound would
        # click at every clip boundary
        fo = max(0.0, c.out_len - 0.05)
        afil += (f",aresample={_AUDIO_SR},aformat=channel_layouts=stereo,"
                 f"apad,afade=t=in:st=0:d=0.03,afade=t=out:st={fo:.3f}:d=0.05[a]")
        graph += ";" + afil
        amap = "[a]"
    else:
        amap = "1:a"
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", amap,
            "-t", f"{c.out_len:.3f}"]
    cmd += vargs
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-ar", str(_AUDIO_SR), "-ac", "2",
            "-progress", "pipe:1", "-nostats", out]
    return cmd


def _mix_command(list_file: str, voice_wav: str, out_wav: str,
                 opts: MontageOptions, total: float, sfx_wav: str = "",
                 bed_gain_db: float = 0.0) -> list[str]:
    """Audio only: the narration over the (ducked) clip audio and/or looped
    BGM, effects in the gaps -> 48 kHz stereo 32-bit float WAV of `total`
    seconds. `bed_gain_db` undoes the headroom of a separated background."""
    cmd = [ffmpeg_path(), "-y", "-hide_banner",
           "-f", "concat", "-safe", "0", "-i", list_file,
           "-i", voice_wav]
    use_bgm = bool(opts.bgm_path)
    if use_bgm:
        cmd += ["-stream_loop", "-1", "-i", opts.bgm_path]
    if sfx_wav:
        cmd += ["-i", sfx_wav]

    parts: list[str] = [f"[1:a]volume={opts.voice_gain_db:.2f}dB,"
                        f"aresample={_AUDIO_SR},aformat=channel_layouts=stereo,"
                        f"asplit=2[sc][vo]"]
    beds: list[str] = []
    if opts.original_audio_db > -60:
        parts.append(f"[0:a]volume={opts.original_audio_db + bed_gain_db:.2f}dB[orig]")
        beds.append("[orig]")
    if use_bgm:
        parts.append(f"[2:a]volume={opts.bgm_db:.2f}dB,aresample={_AUDIO_SR},"
                     f"aformat=channel_layouts=stereo[bgm]")
        beds.append("[bgm]")
    # effects sit in the gaps between lines: mixed after ducking, never ducked
    fx = ""
    if sfx_wav:
        parts.append(f"[{3 if use_bgm else 2}:a]volume={opts.sfx_db:.2f}dB,"
                     f"aresample={_AUDIO_SR},aformat=channel_layouts=stereo[sfx]")
        fx = "[sfx]"

    if beds:
        if len(beds) == 2:
            parts.append(f"{beds[0]}{beds[1]}amix=inputs=2:duration=first:"
                         f"dropout_transition=0:normalize=0[bed]")
        else:
            parts.append(f"{beds[0]}anull[bed]")
        ratio = max(1.5, min(20.0, abs(opts.duck_db) / 3.0 + 1.0))
        # soft knee, 60 ms attack, 0.5 s release: the background dips under
        # each line and comes back smoothly instead of pumping
        parts.append(f"[bed][sc]sidechaincompress=threshold=0.02:"
                     f"ratio={ratio:.2f}:attack=60:release=500:knee=4[ducked]")
        parts.append(f"[ducked][vo]{fx}amix=inputs={3 if fx else 2}:duration=first:"
                     "dropout_transition=0:normalize=0[aout]")
    else:
        parts.append("[sc]anullsink")
        if fx:
            parts.append(f"[vo]{fx}amix=inputs=2:duration=first:dropout_transition=0:"
                         "normalize=0[aout]")
        else:
            parts.append("[vo]anull[aout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[aout]",
            "-t", f"{total:.3f}", "-ar", str(_AUDIO_SR), "-ac", "2",
            "-c:a", "pcm_f32le", "-progress", "pipe:1", "-nostats", out_wav]
    return cmd


def _final_command(list_file: str, mix_wav: str, out_path: str,
                   srt_path: str, opts: MontageOptions, vargs: list[str],
                   me, gain_db: float = 0.0) -> list[str]:
    """The joined clips' video with the mixed audio, set to the target
    loudness by `gain_db` behind a peak limiter (auto-level off: it would
    push the peaks back up to full scale)."""
    from app.engines.audio_pipeline import PEAK_LIMIT
    cmd = [ffmpeg_path(), "-y", "-hide_banner",
           "-f", "concat", "-safe", "0", "-i", list_file,
           "-i", mix_wav]
    parts = [f"[1:a]volume={gain_db:.2f}dB,alimiter=limit={PEAK_LIMIT}:level=0[aout]"]
    if opts.burn_subtitles and os.path.isfile(srt_path):
        parts.append(f"[0:v]{me._subtitles_filter(srt_path, opts.subtitle_font)}[vout]")
        cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]",
                "-map", "[aout]"]
        cmd += vargs + ["-pix_fmt", "yuv420p"]
    else:
        cmd += ["-filter_complex", ";".join(parts), "-map", "0:v:0",
                "-map", "[aout]", "-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", str(_AUDIO_SR),
            "-shortest"]
    if out_path.lower().endswith(".mp4"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", out_path]
    return cmd


def _build_narration_track(ap, voices, clips: list[MontageClip], total: float,
                           out_wav: str, cancel: threading.Event | None) -> None:
    """Place every TTS wav at its clip's output start on one mono track."""
    import numpy as np

    sr = ap._TTS_SR
    buf = np.zeros(int(round(total * sr)) + sr, dtype=np.int32)
    for (_idx, _s, _e, _text, wav, _vlen), c in zip(voices, clips):
        _check_cancel(cancel)
        data = ap._read_wav_as_int16_mono(wav, sr, cancel=cancel)
        off = int(round(c.out_start * sr))
        n = min(data.size, buf.size - off)
        if n > 0:
            buf[off:off + n] += data[:n].astype(np.int32)
    ap._write_wav_int16(out_wav, np.clip(buf, -32768, 32767).astype(np.int16), sr)


# ---------------------------------------------------------------------------
# Offline self-test (no ffmpeg needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    o = MontageOptions(gap_seconds=0.25, max_slowdown=1.35)
    # Slot longer than voice -> trimmed, no slow-mo.
    st, sl, slow, ol = plan_clip(10.0, 20.0, 3.0, 600.0, o)
    assert (st, slow) == (10.0, 1.0) and abs(ol - 3.25) < 1e-9 and abs(sl - 3.25) < 1e-9
    # Slot a bit short -> slowed down.
    st, sl, slow, ol = plan_clip(10.0, 13.0, 3.5, 600.0, o)
    assert 1.0 < slow <= 1.35 and abs(sl * slow - ol) < 1e-9
    # Slot far too short -> capped slow-mo, extends into following footage.
    st, sl, slow, ol = plan_clip(10.0, 11.0, 5.0, 600.0, o)
    assert abs(slow - 1.35) < 1e-9 and sl > 1.0
    # Near the end of the video -> start shifted back to fit.
    st, sl, slow, ol = plan_clip(598.0, 599.0, 5.0, 600.0, o)
    assert st + sl <= 600.0 + 1e-9
    # Captions
    caps = _split_caption("Hello there. This is a much longer sentence that "
                          "needs wrapping into pieces!", 20)
    assert caps[0] == "Hello there." and all(len(c) <= 20 for c in caps[1:])
    srt = build_srt([MontageClip(0, 0, 3, 1, 0.0, 3.25, 3.0, "Hi. Bye.")])
    assert "00:00:00,000 --> 00:00:01,286" in srt, srt  # 3 of 7 chars
    assert _atempo_chain(0.25).count("atempo") == 2
    # scene snapping: a cut 0.5 s after the planned start becomes the start
    st, sl, slow, ol = plan_clip(10.0, 20.0, 3.0, 600.0, o, cuts=[10.5, 30.0])
    assert abs(st - 10.54) < 1e-9, st
    # smart framing crops around the subject for 9:16 from 16:9
    c = MontageClip(0, 0, 3, 1.0, 0, 3, 3, "x", crop_x=0.8)
    g = _video_chain(c, 1080, 1920, 30, MontageOptions(fit_mode="smart"), 1920, 1080)
    assert "scale=3412:1920,crop=1080:1920:2189:0" in g, g
    g = _video_chain(c, 1920, 1080, 30, MontageOptions(fit_mode="smart"), 1920, 1080)
    assert "force_original_aspect_ratio=increase" in g, "same aspect: plain crop"
    # Khmer captions break between whole syllables into even pieces: a hard
    # cut at 42 once left "ង" (of ដំបូង) alone on screen for 0.07 s
    km = "រឿងចាប់ផ្ដើមនៅពេលដែលនាងបានជួបគាត់ជាលើកដំបូង"
    caps = _split_caption(km, 42)
    assert caps == ["រឿងចាប់ផ្ដើមនៅពេលដែលនាង", "បានជួបគាត់ជាលើកដំបូង"], caps
    assert _syllables("សូមអរគុណច្រើនសម្រាប់ការគាំទ្ររបស់អ្នក") == [
        "\u179f\u17bc\u1798", "\u17a2\u179a\u1782\u17bb\u178e", "\u1785\u17d2\u179a\u17be\u1793", "\u179f\u1798\u17d2\u179a\u17b6\u1794\u17cb", "\u1780\u17b6\u179a", "\u1782\u17b6\u17c6", "\u1791\u17d2\u179a", "\u179a\u1794\u179f\u17cb", "\u17a2\u17d2\u1793\u1780"]
    assert _syllables("នៅយប់នោះ។") == ["\u1793\u17c5", "\u1799\u1794\u17cb", "\u1793\u17c4\u17c7\u17d4"]
    long_km = ("គាត់មិនដឹងទេថានាងជានរណា ប៉ុន្តែអ្វីៗបានផ្លាស់ប្ដូរនៅយប់នោះ "
               "ហើយនាងសម្រេចចិត្តតស៊ូ")
    caps = _split_caption(long_km, 42)
    assert "".join(caps).replace(" ", "") == long_km.replace(" ", "")
    assert caps[0] == "គាត់មិនដឹងទេថានាងជានរណា", caps          # breaks at the space
    assert max(map(len, caps)) - min(map(len, caps)) <= 8, caps   # even pieces
    zh = "他恐怕还没用全力，比想象中还要强得多，大家都没有想到他会这样做真是太厉害了而且一点都不累"
    assert [len(c) for c in _split_caption(zh, 42)] == [22, 22]
    print("PASS")
