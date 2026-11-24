"""Recap engine: beat analysis and recap-script writing on top of AIEngine.

Contract surface (ARCHITECTURE.md):
    TimelineRow, RecapProject, SPEECH_CPS, estimate_speech_seconds,
    check_row_timing, RecapEngine (analyze_beats / write_recap / full_pipeline).

No Qt imports.  Heavy deps (whisper/demucs) are only touched indirectly through
app.engines.audio_pipeline, which is lazy-imported inside full_pipeline.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from app.core.errors import AppError, CancelledError
from app.core.logger import get_logger

if TYPE_CHECKING:  # type-only imports; real imports are lazy (parallel builds)
    from app.core.ai_engine import AIEngine
    from app.engines.audio_pipeline import Transcript

log = get_logger("recap_engine")

SCHEMA_VERSION = 1

ProgressFn = Callable[[float, str], None]

# ---------------------------------------------------------------------------
# Speech-rate model
# ---------------------------------------------------------------------------

# Characters-per-second of comfortable narration speech, per language.
SPEECH_CPS = {"km": 12.0, "en": 15.0, "zh": 5.5, "th": 9.0, "vi": 14.0,
              "ko": 6.5, "ja": 7.0, "fr": 15.0, "es": 15.5, "hi": 12.0}

# Languages whose characters are counted WITHOUT spaces (scripts written
# without word spacing, or where spaces are not phonetic content).
_NO_SPACE_LANGS = {"km", "zh", "th", "ja", "ko"}

_DEFAULT_CPS = 12.0


def _base_lang(lang: str) -> str:
    """'km-KH' / 'zh_CN' / 'EN' -> 'km' / 'zh' / 'en'."""
    return (lang or "").strip().lower().replace("_", "-").split("-")[0]


def estimate_speech_seconds(text: str, lang: str) -> float:
    """Estimate how many seconds it takes to speak `text` in `lang`.

    Whitespace runs are collapsed to a single space.  For CJK/Khmer-style
    languages spaces are excluded from the character count; for latin-ish
    languages spaces count (they represent real inter-word time).
    """
    if not text:
        return 0.0
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return 0.0
    code = _base_lang(lang)
    cps = SPEECH_CPS.get(code, _DEFAULT_CPS)
    if code in _NO_SPACE_LANGS:
        normalized = normalized.replace(" ", "")
    return len(normalized) / cps


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

def _to_seconds(value: Any, default: float = 0.0) -> float:
    """Tolerant numeric/timestamp coercion: 12.5, '12.5', '1:23', '0:01:23'."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            f = float(value)
            return f if f == f else default  # NaN guard
        except (TypeError, ValueError, OverflowError):
            return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", s)
    if m:
        h = int(m.group(1) or 0)
        return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return default
    return default


def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS (minutes may exceed 59 for long videos)."""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(round(seconds - m * 60))
    if s >= 60:
        m, s = m + 1, s - 60
    return f"{m:02d}:{s:02d}"


@dataclass
class TimelineRow:
    start: float
    end: float
    title: str
    narration: str
    visual_plan: str
    onscreen_text: str

    def to_json(self) -> dict:
        return {
            "start": float(self.start),
            "end": float(self.end),
            "title": self.title,
            "narration": self.narration,
            "visual_plan": self.visual_plan,
            "onscreen_text": self.onscreen_text,
        }

    @classmethod
    def from_json(cls, d: dict) -> "TimelineRow":
        d = d or {}
        return cls(
            start=_to_seconds(d.get("start"), 0.0),
            end=_to_seconds(d.get("end"), 0.0),
            title=str(d.get("title", "") or ""),
            narration=str(d.get("narration", "") or ""),
            visual_plan=str(d.get("visual_plan", "") or ""),
            onscreen_text=str(d.get("onscreen_text", "") or ""),
        )


def check_row_timing(row: TimelineRow, lang: str) -> tuple[bool, float]:
    """Return (fits, needed_seconds) for the row's narration vs its slot."""
    needed = estimate_speech_seconds(row.narration, lang)
    slot = max(0.0, float(row.end) - float(row.start))
    # Small slack: TTS tempo-fitting can absorb ~5% (min 0.3 s).
    slack = max(0.3, slot * 0.05)
    return (needed <= slot + slack, needed)


@dataclass
class RecapProject:
    source_video: str
    language: str
    rows: list[TimelineRow]
    beats: list[dict]
    transcript: "Transcript | None"

    def to_json(self) -> dict:
        transcript_json = None
        if self.transcript is not None:
            transcript_json = self.transcript.to_json()
        return {
            "schema_version": SCHEMA_VERSION,
            "source_video": self.source_video,
            "language": self.language,
            "rows": [r.to_json() for r in self.rows],
            "beats": list(self.beats or []),
            "transcript": transcript_json,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RecapProject":
        d = d or {}
        version = int(d.get("schema_version", 1) or 1)
        if version > SCHEMA_VERSION:
            log.warning("RecapProject schema_version %s is newer than "
                        "supported %s; loading best-effort", version,
                        SCHEMA_VERSION)
        transcript = None
        tj = d.get("transcript")
        if tj:
            try:
                from app.engines.audio_pipeline import Transcript
                transcript = Transcript.from_json(tj)
            except Exception as exc:  # keep project usable without transcript
                log.warning("Could not restore transcript from project "
                            "file: %s", exc)
                transcript = None
        rows = [TimelineRow.from_json(r) for r in (d.get("rows") or [])
                if isinstance(r, dict)]
        beats = [b for b in (d.get("beats") or []) if isinstance(b, dict)]
        return cls(
            source_video=str(d.get("source_video", "") or ""),
            language=str(d.get("language", "km") or "km"),
            rows=rows,
            beats=beats,
            transcript=transcript,
        )

    def save(self, path: str) -> None:
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.to_json(), fh, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        except OSError as exc:
            raise AppError(
                f"Failed to save recap project to {path}: {exc}",
                f"មិនអាចរក្សាទុកគម្រោង Recap ទៅ {path} បានទេ៖ {exc}",
            ) from exc

    @classmethod
    def load(cls, path: str) -> "RecapProject":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except OSError as exc:
            raise AppError(
                f"Failed to open recap project {path}: {exc}",
                f"មិនអាចបើកគម្រោង Recap {path} បានទេ៖ {exc}",
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise AppError(
                f"Recap project file is not valid JSON: {path} ({exc})",
                f"ឯកសារគម្រោង Recap មិនមែនជា JSON ត្រឹមត្រូវទេ៖ {path} ({exc})",
            ) from exc
        if not isinstance(data, dict):
            raise AppError(
                f"Recap project file has an unexpected structure: {path}",
                f"ឯកសារគម្រោង Recap មានរចនាសម្ព័ន្ធមិនត្រឹមត្រូវ៖ {path}",
            )
        return cls.from_json(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


def _emit(progress: ProgressFn | None, fraction: float, text: str) -> None:
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, fraction)), text)
    except Exception:  # a broken UI callback must never kill the pipeline
        log.debug("progress callback raised", exc_info=True)


def _sub_progress(progress: ProgressFn | None, lo: float,
                  hi: float) -> ProgressFn | None:
    """Map a child operation's 0..1 progress into [lo, hi] of the parent."""
    if progress is None:
        return None
    span = hi - lo

    def fn(fraction: float, text: str) -> None:
        try:
            f = float(fraction)
        except (TypeError, ValueError):
            f = 0.0
        _emit(progress, lo + span * max(0.0, min(1.0, f)), text)

    return fn


def _extract_json(text: str) -> Any:
    """Tolerant JSON extraction, delegating to ai_engine.extract_json."""
    from app.core.ai_engine import extract_json
    return extract_json(text)


_NUM_LINE_RE = re.compile(r"^\s*(\d{1,4})\s*[\.\):\-\u3002]\s*(.*)$")


def _parse_numbered_list(text: str, expected: int) -> list[str] | None:
    """Parse '1. xxx / 2. yyy' replies. Returns None if count mismatches."""
    items: dict[int, list[str]] = {}
    current: int | None = None
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        m = _NUM_LINE_RE.match(line)
        if m:
            current = int(m.group(1))
            items.setdefault(current, []).append(m.group(2).strip())
        elif current is not None and line.strip():
            items[current].append(line.strip())
    if len(items) != expected:
        return None
    if sorted(items.keys()) != list(range(1, expected + 1)):
        return None
    return [" ".join(items[i]).strip() for i in range(1, expected + 1)]


def _coerce_beat(d: Any) -> dict | None:
    """Validate/normalise one beat dict from the LLM. None if unusable."""
    if not isinstance(d, dict):
        return None
    start = _to_seconds(d.get("start"), -1.0)
    end = _to_seconds(d.get("end"), -1.0)
    summary = str(d.get("summary", "") or "").strip()
    if start < 0 or not summary:
        return None
    if end < start:
        end = start
    try:
        importance = int(float(d.get("importance", 3)))
    except (TypeError, ValueError):
        importance = 3
    importance = max(1, min(5, importance))
    cues = str(d.get("cues", "") or "").strip()
    return {"start": round(start, 3), "end": round(end, 3),
            "summary": summary, "importance": importance, "cues": cues}


def _coalesce_beats(beats: list[dict], gap: float = 2.0) -> list[dict]:
    """Sort by start and merge beats whose gap is under `gap` seconds."""
    beats = sorted(beats, key=lambda b: (b["start"], b["end"]))
    merged: list[dict] = []
    for b in beats:
        if merged and b["start"] - merged[-1]["end"] < gap:
            cur = merged[-1]
            cur["end"] = max(cur["end"], b["end"])
            if b["summary"] and b["summary"] not in cur["summary"]:
                cur["summary"] = (cur["summary"] + " " + b["summary"]).strip()
            if b["cues"] and b["cues"] not in cur["cues"]:
                cur["cues"] = "; ".join(x for x in (cur["cues"], b["cues"]) if x)
            cur["importance"] = max(cur["importance"], b["importance"])
        else:
            merged.append(dict(b))
    return merged


# Language display names for prompts.
_LANG_NAMES = {"km": "Khmer (ភាសាខ្មែរ)", "en": "English", "zh": "Chinese",
               "th": "Thai", "vi": "Vietnamese", "ko": "Korean",
               "ja": "Japanese", "fr": "French", "es": "Spanish",
               "hi": "Hindi"}


def _lang_name(lang: str) -> str:
    return _LANG_NAMES.get(_base_lang(lang), lang)


# ---------------------------------------------------------------------------
# RecapEngine
# ---------------------------------------------------------------------------

_BEATS_SYSTEM = (
    "You are a professional drama story analyst. You receive a portion of a "
    "timestamped dialogue transcript of a drama/movie in the format "
    "\"[MM:SS-MM:SS] text\". Identify the STORY BEATS: plot events, "
    "reveals, conflicts, emotional turning points.\n"
    "Rules:\n"
    "1. Respond with ONLY a single JSON object. No prose, no markdown, no "
    "code fences, no comments.\n"
    "2. Schema (exact keys): {\"beats\": [{\"start\": <number, seconds>, "
    "\"end\": <number, seconds>, \"summary\": <string>, \"importance\": "
    "<integer 1-5>, \"cues\": <string>}]}\n"
    "3. \"start\"/\"end\" are absolute seconds from the video start, derived "
    "from the [MM:SS-MM:SS] timestamps of the lines the beat covers.\n"
    "4. \"summary\" is one or two factual sentences describing what happens "
    "(who does what, and why it matters). Never quote dialogue verbatim.\n"
    "5. \"importance\": 5 = pivotal plot point, 1 = minor detail.\n"
    "6. \"cues\" lists visual/audio cues useful for editing (e.g. "
    "\"confrontation at hospital, slap, crying\").\n"
    "7. Cover the whole excerpt with 3-12 beats. Do not invent events that "
    "are not supported by the transcript."
)


class RecapEngine:
    """Turns a transcript into story beats and a timed recap script."""

    def __init__(self, ai: "AIEngine"):
        self.ai = ai

    # -- beats ------------------------------------------------------------

    def analyze_beats(self, transcript: "Transcript",
                      progress: ProgressFn | None = None,
                      cancel: threading.Event | None = None) -> list[dict]:
        _check_cancel(cancel)
        segments = [s for s in (transcript.segments or [])
                    if (s.text or "").strip()]
        if not segments:
            raise AppError(
                "Transcript has no usable segments to analyze.",
                "អត្ថបទចម្លងគ្មានផ្នែកដែលអាចវិភាគបានទេ។",
            )
        heard = sum(len(re.sub(r"\s+", "", s.text or "")) for s in segments)
        length = float(getattr(transcript, "duration", 0.0) or 0.0)
        if length >= 20.0 and heard < 12:
            # a word or two from a whole video: the speech was taken for
            # another language (Deepgram heard Mandarin as French: "Bon,"),
            # or it has no dialogue — no story for the AI to find (#22)
            sample = "; ".join((s.text or "").strip() for s in segments[:3])[:60]
            lang = getattr(transcript, "language", "") or "?"
            mm = f"{int(length // 60)}:{int(length % 60):02d}"
            raise AppError(
                f"Almost nothing was heard in {mm} of video: {len(segments)} line(s), "
                f"“{sample}” (taken as language '{lang}'). If the video has dialogue, "
                "its language was probably mistaken: transcribe again, or choose "
                "Whisper in Settings → Speech services.",
                f"ស្ទើរតែមិនឮអ្វីសោះក្នុងវីដេអូ {mm}៖ {len(segments)} ឃ្លា, “{sample}” "
                f"(យល់ថាជាភាសា '{lang}')។ បើវីដេអូមានសន្ទនា ប្រហែលជាភាសារបស់វាត្រូវបានយល់ខុស៖ "
                "សូមចម្លងជាអក្សរម្ដងទៀត ឬជ្រើស Whisper នៅ ការកំណត់ → សេវាសំឡេង។",
            )

        lines = []
        for s in segments:
            text = re.sub(r"\s+", " ", s.text).strip()
            lines.append(f"[{_fmt_ts(s.start)}-{_fmt_ts(s.end)}] {text}")

        # Chunk to ~4000 chars with 1-line (1-segment) overlap between chunks.
        max_chars = 4000
        chunks: list[list[str]] = []
        cur: list[str] = []
        cur_len = 0
        for ln in lines:
            if cur and cur_len + len(ln) + 1 > max_chars:
                chunks.append(cur)
                overlap = cur[-1]
                cur = [overlap]
                cur_len = len(overlap) + 1
            cur.append(ln)
            cur_len += len(ln) + 1
        if cur:
            chunks.append(cur)

        all_beats: list[dict] = []
        n = len(chunks)
        failed_chunks = 0
        for i, chunk_lines in enumerate(chunks):
            _check_cancel(cancel)
            _emit(progress, i / n, f"Analyzing story beats {i + 1}/{n}")
            user_msg = (
                f"Transcript excerpt {i + 1} of {n}:\n\n"
                + "\n".join(chunk_lines)
                + "\n\nReturn the JSON object now."
            )
            result = self.ai.chat(
                feature="recap",
                messages=[{"role": "user", "content": user_msg}],
                system=_BEATS_SYSTEM,
                temperature=0.3,
                max_tokens=4096,
                json_mode=True,
                cancel=cancel,
            )
            _check_cancel(cancel)
            chunk_beats: list[dict] = []
            try:
                data = _extract_json(result.text)
                raw = data.get("beats") if isinstance(data, dict) else data
                if isinstance(raw, list):
                    for item in raw:
                        beat = _coerce_beat(item)
                        if beat is not None:
                            chunk_beats.append(beat)
            except Exception as exc:
                log.warning("Beat chunk %d/%d: unparseable LLM reply (%s); "
                            "skipping chunk", i + 1, n, exc)
            if not chunk_beats:
                failed_chunks += 1
                log.warning("Beat chunk %d/%d produced no valid beats",
                            i + 1, n)
            all_beats.extend(chunk_beats)

        if not all_beats:
            raise AppError(
                "The AI did not return any valid story beats "
                f"({failed_chunks}/{n} chunks failed).",
                "AI មិនបានផ្តល់ចំណុចសាច់រឿងត្រឹមត្រូវណាមួយទេ "
                f"(បរាជ័យ {failed_chunks}/{n} ផ្នែក)។",
            )

        beats = _coalesce_beats(all_beats, gap=2.0)
        _emit(progress, 1.0, f"Found {len(beats)} story beats")
        return beats

    # -- script -----------------------------------------------------------

    def write_recap(self, beats: list[dict], transcript: "Transcript",
                    target_lang: str, target_minutes: float = 3.0,
                    style: str = "cinematic",
                    progress: ProgressFn | None = None,
                    cancel: threading.Event | None = None) -> list[TimelineRow]:
        _check_cancel(cancel)
        if not beats:
            raise AppError(
                "No story beats provided — run beat analysis first.",
                "គ្មានចំណុចសាច់រឿងទេ — សូមដំណើរការវិភាគសាច់រឿងជាមុនសិន។",
            )
        lang = _base_lang(target_lang) or "km"
        lang_name = _lang_name(lang)
        cps = SPEECH_CPS.get(lang, _DEFAULT_CPS)
        duration = float(getattr(transcript, "duration", 0.0) or 0.0)
        if duration <= 0:
            duration = max(float(b.get("end", 0.0)) for b in beats)
        target_seconds = max(30.0, float(target_minutes) * 60.0)
        char_budget = int(target_seconds * cps * 0.92)  # small safety margin

        # ---------------- Pass 1: draft script (feature="recap") ----------
        _emit(progress, 0.02, "Writing recap script (draft)")
        beat_table = "\n".join(
            f"- [{_fmt_ts(b['start'])}-{_fmt_ts(b['end'])}] "
            f"(importance {b['importance']}/5) {b['summary']}"
            + (f" | cues: {b['cues']}" if b.get("cues") else "")
            for b in beats
        )
        draft_system = (
            "You are a professional short-video recap scriptwriter for "
            "drama/movie recap channels. You write voice-over scripts that "
            "keep viewers hooked from the first second.\n"
            "Rules:\n"
            "1. Respond with ONLY one JSON object, no prose, no markdown "
            "fences.\n"
            "2. Schema (exact keys): {\"rows\": [{\"start\": <number, "
            "seconds into the SOURCE video>, \"end\": <number>, \"title\": "
            "<string>, \"narration\": <string>, \"visual_plan\": <string>, "
            "\"onscreen_text\": <string>}]}\n"
            f"3. \"narration\" must be written NATIVELY in {lang_name} — "
            "compose it directly in that language as a native storyteller "
            "would; do NOT translate word-by-word from another language.\n"
            "4. Rows must be in chronological order and must NOT overlap: "
            "each row's start >= the previous row's end. All times within "
            f"[0, {duration:.1f}] seconds of the source video.\n"
            "5. Each row's slot (end - start) marks which source footage the "
            "editor will show while that narration plays.\n"
            f"6. Speech budget: at ~{cps:g} characters/second in {lang_name}, "
            f"the TOTAL narration must be speakable within about "
            f"{target_minutes:g} minutes — keep total narration under "
            f"~{char_budget} characters, and each row's narration speakable "
            "within its own slot.\n"
            "7. \"title\" is a short scene label; \"visual_plan\" describes "
            "shots/cuts/effects for the editor (any language); "
            "\"onscreen_text\" is an optional short caption in "
            f"{lang_name} (empty string if none).\n"
            "8. Open with a strong hook row; end with a payoff/cliffhanger. "
            "Prefer high-importance beats; drop trivia. 8-25 rows total."
        )
        draft_user = (
            f"Source video duration: {duration:.1f} seconds "
            f"({_fmt_ts(duration)}).\n"
            f"Target narration language: {lang_name} (code: {lang}).\n"
            f"Target recap length: {target_minutes:g} minutes of speech.\n"
            f"Style: {style or 'cinematic'}.\n\n"
            f"Story beats (chronological):\n{beat_table}\n\n"
            "Write the recap script JSON now."
        )
        result = self.ai.chat(
            feature="recap",
            messages=[{"role": "user", "content": draft_user}],
            system=draft_system,
            temperature=0.7,
            max_tokens=8192,
            json_mode=True,
            cancel=cancel,
        )
        _check_cancel(cancel)
        try:
            data = _extract_json(result.text)
        except Exception as exc:
            raise AppError(
                f"Could not parse the recap draft from the AI: {exc}",
                f"មិនអាចញែកសេចក្តីព្រាងស្គ្រីប Recap ពី AI បានទេ៖ {exc}",
            ) from exc
        raw_rows = data.get("rows") if isinstance(data, dict) else data
        rows: list[TimelineRow] = []
        if isinstance(raw_rows, list):
            for item in raw_rows:
                if not isinstance(item, dict):
                    continue
                row = TimelineRow.from_json(item)
                if row.narration.strip():
                    rows.append(row)
        if not rows:
            raise AppError(
                "The AI recap draft contained no usable rows.",
                "សេចក្តីព្រាងស្គ្រីប Recap ពី AI គ្មានជួរដេកដែលអាចប្រើបានទេ។",
            )

        # Normalise: clamp to video, sort, remove overlaps.
        for row in rows:
            row.start = max(0.0, min(row.start, duration))
            row.end = max(0.0, min(row.end, duration))
            if row.end < row.start:
                row.start, row.end = row.end, row.start
        rows.sort(key=lambda r: (r.start, r.end))
        prev_end = 0.0
        for row in rows:
            if row.start < prev_end:
                row.start = prev_end
            if row.end <= row.start:
                needed = estimate_speech_seconds(row.narration, lang)
                row.end = min(duration, row.start + max(1.0, needed))
            prev_end = row.end
        _emit(progress, 0.5, f"Draft script ready ({len(rows)} rows)")

        # ---------------- Pass 2: polish (feature="translate_polish") -----
        _check_cancel(cancel)
        _emit(progress, 0.55, "Polishing narration")
        if lang == "km":
            style_prompt = (
                "You are a master Khmer storyteller for cinematic movie-recap "
                "voice-overs. Rewrite each numbered narration line into "
                "vivid, natural spoken Khmer:\n"
                "- Cinematic Khmer storytelling voice, as if narrating a "
                "gripping film trailer.\n"
                "- Use natural Khmer idioms and expressive phrasing "
                "(សំនួនវោហារ) where they fit; absolutely no stiff, "
                "word-by-word translationese.\n"
                "- Vary the rhythm: mix short punchy sentences with longer "
                "flowing ones.\n"
                "- Line 1 is the hook — make it grab the listener instantly.\n"
                "- Keep every fact and name unchanged; keep each line about "
                "the same length or shorter (it must fit the same time slot)."
            )
        else:
            style_prompt = (
                f"You are a master {lang_name} storyteller for cinematic "
                "movie-recap voice-overs. Rewrite each numbered narration "
                f"line into vivid, natural spoken {lang_name}:\n"
                "- Cinematic trailer-style delivery with natural idioms; no "
                "stiff or literal phrasing.\n"
                "- Vary sentence rhythm: short punches mixed with flowing "
                "lines.\n"
                "- Line 1 is the hook — make it grab the listener instantly.\n"
                "- Keep every fact and name unchanged; keep each line about "
                "the same length or shorter (it must fit the same time slot)."
            )
        numbered = "\n".join(
            "{}. {}".format(i + 1, re.sub(r"\s+", " ", r.narration).strip())
            for i, r in enumerate(rows))
        polish_user = (
            f"{numbered}\n\n"
            f"Return EXACTLY {len(rows)} numbered lines (1. to {len(rows)}.), "
            "one polished narration per line, in the same order. No extra "
            "text before or after the list."
        )
        polished: list[str] | None = None
        for attempt in range(2):
            _check_cancel(cancel)
            try:
                presult = self.ai.chat(
                    feature="translate_polish",
                    messages=[{"role": "user", "content": polish_user}],
                    system=style_prompt,
                    temperature=0.8 if attempt == 0 else 0.4,
                    max_tokens=8192,
                    cancel=cancel,
                )
            except CancelledError:
                raise
            except AppError as exc:
                log.warning("Polish pass failed (%s); keeping draft "
                            "narration", exc)
                break
            polished = _parse_numbered_list(presult.text, len(rows))
            if polished is not None:
                break
            log.warning("Polish pass returned a mismatched list "
                        "(attempt %d); retrying" % (attempt + 1))
        if polished is not None:
            for row, text in zip(rows, polished):
                if text.strip():
                    row.narration = text.strip()
            _emit(progress, 0.8, "Narration polished")
        else:
            log.warning("Polish pass unusable after retry; keeping draft "
                        "narration")
            _emit(progress, 0.8, "Polish skipped (kept draft narration)")

        # ---------------- Timing pass: shorten overlong rows --------------
        _check_cancel(cancel)
        overlong = [(i, row) for i, row in enumerate(rows)
                    if not check_row_timing(row, lang)[0]]
        if overlong:
            _emit(progress, 0.85,
                  f"Shortening {len(overlong)} overlong rows")
            entries = []
            for i, row in overlong:
                slot = max(0.5, row.end - row.start)
                max_chars = max(10, int(slot * cps))
                entries.append({
                    "index": i,
                    "max_seconds": round(slot, 2),
                    "max_chars": max_chars,
                    "narration": row.narration,
                })
            shorten_system = (
                f"You shorten voice-over narration lines in {lang_name} so "
                "they fit their time slots, preserving meaning, names and "
                "the storytelling tone.\n"
                "Rules:\n"
                "1. Respond with ONLY one JSON object, no prose, no fences.\n"
                "2. Schema: {\"rows\": [{\"index\": <same integer index you "
                "were given>, \"narration\": <shortened text>}]}\n"
                "3. Each shortened narration must be at most its given "
                "\"max_chars\" characters (speakable within \"max_seconds\" "
                f"seconds at ~{cps:g} chars/second).\n"
                "4. Keep the language natural — trim words, do not mangle "
                "grammar. Return every row you were given."
            )
            shorten_user = (
                "Shorten these narration lines:\n"
                + json.dumps({"rows": entries}, ensure_ascii=False, indent=2)
                + "\n\nReturn the JSON object now."
            )
            try:
                sresult = self.ai.chat(
                    feature="translate_polish",
                    messages=[{"role": "user", "content": shorten_user}],
                    system=shorten_system,
                    temperature=0.3,
                    max_tokens=8192,
                    json_mode=True,
                    cancel=cancel,
                )
                _check_cancel(cancel)
                sdata = _extract_json(sresult.text)
                sraw = sdata.get("rows") if isinstance(sdata, dict) else sdata
                if isinstance(sraw, list):
                    valid = {i for i, _ in overlong}
                    for item in sraw:
                        if not isinstance(item, dict):
                            continue
                        try:
                            idx = int(item.get("index"))
                        except (TypeError, ValueError):
                            continue
                        text = str(item.get("narration", "") or "").strip()
                        if idx in valid and text:
                            rows[idx].narration = text
            except CancelledError:
                raise
            except Exception as exc:
                log.warning("Shorten pass failed (%s); flagging overlong "
                            "rows", exc)
            # Re-check and flag whatever still does not fit.
            still = 0
            for i, _ in overlong:
                row = rows[i]
                fits, _needed = check_row_timing(row, lang)
                if not fits and not row.visual_plan.endswith("[OVERLONG]"):
                    row.visual_plan = (row.visual_plan + " [OVERLONG]").strip()
                    still += 1
            if still:
                log.warning("%d rows remain overlong after shorten pass",
                            still)

        _emit(progress, 1.0, f"Recap script ready ({len(rows)} rows)")
        return rows

    # -- full pipeline ------------------------------------------------------

    def full_pipeline(self, video: str, *, whisper_size: str = "small",
                      use_demucs: bool = True, target_lang: str = "km",
                      target_minutes: float = 3.0,
                      progress: ProgressFn | None = None,
                      cancel: threading.Event | None = None) -> RecapProject:
        _check_cancel(cancel)
        if not video or not os.path.isfile(video):
            raise AppError(
                f"Video file not found: {video}",
                f"រកមិនឃើញឯកសារវីដេអូ៖ {video}",
            )
        from app.core.errors import EngineNotAvailable
        from app.engines.audio_pipeline import (extract_audio, speech_to_text, stt_engine,
                                                separated_stems)

        work_dir = tempfile.mkdtemp(prefix="recap_")
        wav_path = os.path.join(work_dir, "audio_16k.wav")

        # Stage 1: extract audio (0.00 - 0.15)
        _emit(progress, 0.0, "Extracting audio")
        extract_audio(video, wav_path, sr=16000, mono=True,
                      progress=_sub_progress(progress, 0.0, 0.15),
                      cancel=cancel)
        _check_cancel(cancel)

        # Stage 2: vocal separation (0.15 - 0.35), optional
        asr_input = wav_path
        if use_demucs:
            _emit(progress, 0.15, "Separating vocals (demucs)")
            try:
                # cached at full quality: the montage reuses the music &
                # effects stem as a background without the original voices
                vocals_wav, _music = separated_stems(
                    video, progress=_sub_progress(progress, 0.15, 0.35),
                    cancel=cancel)
                if vocals_wav and os.path.isfile(vocals_wav):
                    asr_input = vocals_wav
            except CancelledError:
                raise
            except EngineNotAvailable as exc:
                log.warning("Demucs unavailable (%s); transcribing the "
                            "original audio instead", exc.en)
            except AppError as exc:
                log.warning("Vocal separation failed (%s); transcribing the "
                            "original audio instead", exc.en)
        _emit(progress, 0.35, "Audio ready for transcription")
        _check_cancel(cancel)

        # Stage 3: ASR (0.35 - 0.60)
        config = getattr(self.ai, "config", None)
        _emit(progress, 0.35, "Transcribing (Deepgram)" if stt_engine(config) == "deepgram"
              else f"Transcribing (whisper {whisper_size})")
        transcript = speech_to_text(
            asr_input, config=config, model_size=whisper_size, word_timestamps=True,
            progress=_sub_progress(progress, 0.35, 0.60), cancel=cancel)
        _check_cancel(cancel)

        # Stage 4: beats (0.60 - 0.75)
        _emit(progress, 0.60, "Analyzing story beats")
        beats = self.analyze_beats(
            transcript, progress=_sub_progress(progress, 0.60, 0.75),
            cancel=cancel)
        _check_cancel(cancel)

        # Stage 5: script (0.75 - 1.00)
        _emit(progress, 0.75, "Writing recap script")
        rows = self.write_recap(
            beats, transcript, target_lang=target_lang,
            target_minutes=target_minutes,
            progress=_sub_progress(progress, 0.75, 1.0), cancel=cancel)

        _emit(progress, 1.0, "Recap project ready")
        return RecapProject(source_video=video, language=_base_lang(target_lang)
                            or "km", rows=rows, beats=beats,
                            transcript=transcript)


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures: list[str] = []

    def expect(cond: bool, label: str) -> None:
        if not cond:
            failures.append(label)
            print(f"FAIL: {label}")

    # estimate_speech_seconds -------------------------------------------------
    # English: 30 chars incl. spaces at 15 cps -> 2.0 s
    en_text = "abcde fghij klmno pqrst uvwxy"  # 29 chars
    expect(abs(estimate_speech_seconds(en_text, "en") - 29 / 15.0) < 1e-9,
           "en cps math")
    # Khmer: spaces excluded. 16 Khmer chars at 8 cps -> 2.0 s
    km_text = "\u1780\u17d2\u1798\u17c1\u1784 \u1781\u17d2\u1798\u17c2\u179a " \
              "\u179f\u17d2\u17a1\u17b6\u1789 \u1785\u17b6\u179f\u17cb"
    km_len_no_space = len(km_text.replace(" ", ""))
    expect(abs(estimate_speech_seconds(km_text, "km")
               - km_len_no_space / SPEECH_CPS["km"]) < 1e-9, "km cps math (no spaces)")
    # Whitespace-run collapsing
    expect(abs(estimate_speech_seconds("a   b\n\tc", "en")
               - len("a b c") / 15.0) < 1e-9, "whitespace collapse")
    expect(estimate_speech_seconds("", "en") == 0.0, "empty text")
    expect(estimate_speech_seconds("abc", "xx") == 3 / _DEFAULT_CPS,
           "unknown lang default cps")
    expect(abs(estimate_speech_seconds("abc", "km-KH") - 3 / SPEECH_CPS["km"]) < 1e-9,
           "region code normalised")

    # check_row_timing --------------------------------------------------------
    fit_row = TimelineRow(start=0.0, end=10.0, title="t",
                          narration="hello world", visual_plan="",
                          onscreen_text="")
    fits, needed = check_row_timing(fit_row, "en")
    expect(fits and abs(needed - 11 / 15.0) < 1e-9, "short row fits")
    long_row = TimelineRow(start=0.0, end=2.0, title="t",
                           narration="x" * 300, visual_plan="",
                           onscreen_text="")
    fits2, needed2 = check_row_timing(long_row, "en")
    expect((not fits2) and needed2 == 20.0, "overlong row detected")

    # RecapProject save/load round-trip --------------------------------------
    rows = [
        TimelineRow(0.0, 8.5, "Hook", "\u179a\u17bf\u1784\u1785\u17b6\u1794"
                    "\u17cb\u1795\u17d2\u178a\u17be\u1798...", "fast cuts",
                    "EP.1"),
        TimelineRow(8.5, 20.0, "Reveal", "the twist", "slow zoom", ""),
    ]
    beats = [{"start": 0.0, "end": 12.0, "summary": "opening fight",
              "importance": 5, "cues": "rain, neon"}]
    proj = RecapProject(source_video="C:/videos/ep1.mp4", language="km",
                        rows=rows, beats=beats, transcript=None)
    tmp = tempfile.NamedTemporaryFile(suffix=".recap.json", delete=False)
    tmp.close()
    try:
        proj.save(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
        expect(on_disk.get("schema_version") == SCHEMA_VERSION,
               "schema_version persisted")
        loaded = RecapProject.load(tmp.name)
        expect(loaded.source_video == proj.source_video, "source_video kept")
        expect(loaded.language == "km", "language kept")
        expect(len(loaded.rows) == 2, "row count kept")
        expect(loaded.rows[0].to_json() == rows[0].to_json(),
               "row 0 round-trip")
        expect(loaded.rows[1].to_json() == rows[1].to_json(),
               "row 1 round-trip")
        expect(loaded.beats == beats, "beats round-trip")
        expect(loaded.transcript is None, "transcript None round-trip")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    # Beat coalescing ---------------------------------------------------------
    merged = _coalesce_beats([
        {"start": 10.0, "end": 12.0, "summary": "b", "importance": 2,
         "cues": "y"},
        {"start": 0.0, "end": 9.0, "summary": "a", "importance": 4,
         "cues": "x"},
        {"start": 30.0, "end": 35.0, "summary": "c", "importance": 1,
         "cues": ""},
    ])
    expect(len(merged) == 2, "coalesce <2s gap")
    expect(merged[0]["importance"] == 4 and merged[0]["end"] == 12.0,
           "coalesce keeps max importance / extends end")

    # Numbered-list parsing ---------------------------------------------------
    parsed = _parse_numbered_list("1. one\n2) two\n   continued\n3. three", 3)
    expect(parsed == ["one", "two continued", "three"], "numbered parse")
    expect(_parse_numbered_list("1. only", 2) is None,
           "numbered parse mismatch -> None")

    # Almost nothing heard in a whole video (#22: Deepgram took Mandarin
    # for French, "Bon,"): a plain message, no AI call -----------------------
    from types import SimpleNamespace as _NS

    from app.engines.audio_pipeline import Segment as _Seg
    from app.engines.audio_pipeline import Transcript as _Tr

    class _CountAI:
        def __init__(self):
            self.calls = 0

        def chat(self, **kw):
            self.calls += 1
            return _NS(text='{"beats": [{"start": 0, "end": 2, "summary": "s", '
                            '"importance": 3, "cues": ""}]}')

    cai = _CountAI()
    try:
        RecapEngine(cai).analyze_beats(_Tr([_Seg(0.08, 0.32, "Bon,", [])], "fr", 45.1))
        msg = ""
    except AppError as exc:
        msg = exc.en
    expect("Almost nothing was heard in 0:45" in msg and "'fr'" in msg and "Bon," in msg
           and cai.calls == 0, f"near-empty transcript: a plain message, no AI call ({msg[:60]})")
    beats = RecapEngine(cai).analyze_beats(_Tr([_Seg(0.0, 2.0, "我能赚吗", [])], "zh", 8.0))
    expect(len(beats) == 1 and cai.calls == 1, "a short clip with a little speech still goes on")

    if failures:
        print(f"FAILED ({len(failures)} checks)")
        raise SystemExit(1)
    print("PASS")
