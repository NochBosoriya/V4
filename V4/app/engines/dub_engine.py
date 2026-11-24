"""Full-video dubbing engine: translate a whole video with one cloned voice
per character.

Pipeline (each step is resumable and persisted in a project folder under
``projects_dir()/dub``):

  prepare     extract audio, split vocals / music+SFX (Demucs)
  transcribe  Whisper with word timestamps, split into speakable lines
  speakers    speaker clustering -> characters (who says what)
  voices      per-character reference clip + transcript for voice cloning
  glossary    AI story summary, character names, glossary of terms
  translate   dialogue-aware translation with speaker gender, glossary,
              rolling context and a per-line character budget; optional
              review pass
  fit         shorten lines that cannot be spoken inside their time slot
  synthesize  per-line TTS in each character's (cloned) voice, fitted and
              placed on the timeline
  mix         new voices + original music/SFX bed, limited
  export      dubbed video (+ soft or burned subtitles) and a .srt

No Qt imports. Heavy deps (torch, speechbrain, openvoice, faster_whisper,
demucs, edge_tts and the local TTS engines) are lazy-imported.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.ai_engine import (
    AI_ADVICE,
    _parse_numbered_list,
    ai_failure_kind,
    ai_unavailable,
    extract_json,
)
from app.core.errors import AppError, CancelledError, EngineNotAvailable
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import models_dir, pip_install, projects_dir
from app.engines import audio_pipeline as ap
from app.engines import km_order
from app.engines import numbers as nb
from app.engines.recap_engine import (SPEECH_CPS, _DEFAULT_CPS, _base_lang,
                                      _lang_name, estimate_speech_seconds)

log = get_logger("dub_engine")

ProgressFn = Callable[[float, str], None]

# (step id, English label) — order is the pipeline order.
STEPS: list[tuple[str, str]] = [
    ("prepare", "Extract audio & separate voices from music"),
    ("transcribe", "Transcribe dialogue"),
    ("speakers", "Detect speakers / characters"),
    ("glossary", "Story, names, glossary & speaker check (AI)"),
    ("voices", "Build character voice samples (cloning)"),
    ("translate", "Translate dialogue with context"),
    ("fit", "Check timing & shorten long lines"),
    ("synthesize", "Speak every line in its character's voice"),
    ("mix", "Mix new voices with original music & SFX"),
    ("export", "Export dubbed video + subtitles"),
]
STEP_IDS = [s for s, _ in STEPS]
FIRST_AI_STEP = "glossary"      # the tasks before it run without an AI key
AI_STEPS = ("glossary", "translate", "fit")      # the tasks that ask the AI
# rough share of a whole dub's time per task (progress bars)
STEP_WEIGHTS = {"prepare": 3, "transcribe": 4, "speakers": 1, "glossary": 2,
                "voices": 1, "translate": 3, "fit": 1, "synthesize": 5,
                "mix": 1, "export": 2}


SPEED_KEY = "dub.speed"   # config: {task: [seconds per minute of video, last 5 runs]}
# seconds per minute of video before a task has been timed here (a 2.6-min
# Chinese drama on 2026-11-11: GPU Whisper, Deepgram, Edge voices). Mix and
# export always take under 3 s and are never learned: they need a start
# value, or the time left could never be worked out.
DEFAULT_PACE = {"prepare": 13.0, "transcribe": 11.3, "speakers": 0.5, "glossary": 3.4,
                "voices": 0.1, "translate": 16.5, "fit": 6.1, "synthesize": 76.4,
                "mix": 0.3, "export": 0.8}


def learn_speeds(config, p: "DubProject", steps) -> dict:
    """Remember how long `steps` took on this video, per minute of video
    (the last 5 runs of each). A task done in under 3 s reused saved results
    and teaches nothing. Returns the table."""
    table = dict((config.get(SPEED_KEY) if config is not None else None) or {})
    if config is None or not p.duration:
        return table
    mins = max(0.5, p.duration / 60.0)
    changed = False
    for s in steps:
        secs = float(p.steps.get(s, {}).get("secs") or 0.0)
        if secs < 3.0 or not p.step_done(s):
            continue
        table[s] = [float(v) for v in (table.get(s) or [])][-4:] + [round(secs / mins, 2)]
        changed = True
    if changed:
        config.set(SPEED_KEY, table)
    return table


def estimate_left(config, p: "DubProject", running: str | None, within: float,
                  running_for: float, after: list[str]) -> float | None:
    """Seconds left of a run: the running task from its own pace once it
    has shown some (≥ 5 % in ≥ 10 s), else its learned pace; plus the
    learned pace of the tasks `after` it (DEFAULT_PACE until a task has
    been timed on this PC). None without the video's length."""
    table = dict((config.get(SPEED_KEY) if config is not None else None) or {})
    if not p.duration:
        return None
    mins = max(0.5, p.duration / 60.0)

    def pace(s: str) -> float | None:
        vals = sorted(float(v) for v in table.get(s) or [])
        if vals:
            return vals[len(vals) // 2] * mins                 # the median run
        return DEFAULT_PACE[s] * mins if s in DEFAULT_PACE else None

    total = 0.0
    if running:
        within = max(0.0, min(1.0, within))
        if within >= 0.05 and running_for >= 10:
            total += running_for * (1.0 - within) / within
        else:
            pc = pace(running)
            if pc is None:
                return None
            total += max(pc * (1.0 - within), pc - running_for, 0.0)
    for s in after:
        pc = pace(s)
        if pc is None:
            return None
        total += pc
    return total


def overall_progress(p: "DubProject", running: str | None = None, frac: float = 0.0) -> float:
    """How far the whole dub is (0..1): finished tasks plus `frac` of the
    `running` one, by STEP_WEIGHTS."""
    total = float(sum(STEP_WEIGHTS.values()))
    done = sum(STEP_WEIGHTS[s] for s in STEP_IDS if p.step_done(s) and s != running)
    if running in STEP_WEIGHTS:
        done += STEP_WEIGHTS[running] * max(0.0, min(1.0, frac))
    return done / total


def dict_label(step: str | None) -> str:
    """English label of a step id ("" for None/unknown)."""
    return dict(STEPS).get(step or "", step or "")


# Character voice engines. "auto" resolves to the best installed cloning
# engine for the target language; "edge+clone" = Edge TTS in the target
# language, then OpenVoice tone-color conversion to the character's voice
# (language-agnostic, so it works for Khmer).
CHAR_ENGINES = ["auto", "edge", "edge+clone", "edge+seedvc", "f5tts", "voxcpm",
                "cosyvoice", "openai", "elevenlabs"]
# "edge+seedvc" = Edge TTS, then Seed-VC zero-shot voice conversion (GPL-3,
# optional; usually closer to the reference voice than OpenVoice)
_CLONE_LANGS = {
    "f5tts": {"zh", "en"},
    # VoxCPM2 (default checkpoint): 30 languages, Khmer included
    "voxcpm": {"ar", "my", "zh", "da", "nl", "en", "fi", "fr", "de", "el", "he",
               "hi", "id", "it", "ja", "km", "ko", "lo", "ms", "no", "pl", "pt",
               "ru", "es", "sw", "sv", "tl", "th", "tr", "vi"},
    "cosyvoice": {"zh", "en", "ja", "ko"},
}
_NETWORK_ENGINES = {"edge", "openai", "elevenlabs"}

_EDGE_DEFAULTS = {  # lang -> (male, female)
    "km": ("km-KH-PisethNeural", "km-KH-SreymomNeural"),
    "en": ("en-US-GuyNeural", "en-US-JennyNeural"),
    "zh": ("zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
    "th": ("th-TH-NiwatNeural", "th-TH-PremwadeeNeural"),
    "vi": ("vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"),
    "ko": ("ko-KR-InJoonNeural", "ko-KR-SunHiNeural"),
    "ja": ("ja-JP-KeitaNeural", "ja-JP-NanamiNeural"),
    "fr": ("fr-FR-HenriNeural", "fr-FR-DeniseNeural"),
    "es": ("es-ES-AlvaroNeural", "es-ES-ElviraNeural"),
    "hi": ("hi-IN-MadhurNeural", "hi-IN-SwaraNeural"),
}
_ISO3 = {"km": "khm", "en": "eng", "zh": "chi", "th": "tha", "vi": "vie",
         "ko": "kor", "ja": "jpn", "fr": "fre", "es": "spa", "hi": "hin"}

_LINE_GAP = 0.08     # s of silence kept between two dubbed lines
_LATE_OK = 0.4       # s a line may push the next one late before its words are cut
# AI batches asked at the same time (translate, review, meaning check,
# shortening). One after the other, a 1-hour drama's 169 translate requests
# took 22 minutes; the AI services answer several at once.
_AI_WORKERS = 3
# Synthesize: from this many lines on, the lines found too long are reworded
# while later lines are still being voiced, every _REWORD_EVERY voiced lines
_OVERLAP_MIN_LINES = 150
_REWORD_EVERY = 100
_FAST_MARGIN = 0.1   # speed-up past COMFORT_TEMPO not worth rewording a line for
# Bumped when the translation prompts change: cached drafts of an older
# version are translated again instead of reused.
_TR_VERSION = 7        # 3: address words name the listener; misheard words;
#                        4: each numbered line translated on its own;
#                        5: one matching pronoun pair between two people;
#                        6: Chinese address words (师傅 is not គ្រូ);
#                        7: scene notes; Chinese drama terms (神豪, 报到, 摩的)
_NOTES_VERSION = 1     # the scene-notes prompt
_CHECK_VERSION = 11    # the meaning check's rules; 8: kinship pronouns; 9: cut fixes; 10: wrong vs missing; 11: what a brief line may leave out
_SHORTEN_VERSION = 2   # the shortening prompts; 2: the user's preferred dub as the model

# How a line is said more briefly — the model is the user's preferred dub
# (Ep2, 2026-11-11), which sounded better than one sped up to 2x: what the
# voice and the picture already carry is left out (an interjection, a
# surprised "what did he say?!" before the real point, a pronoun Khmer does
# not need, fillers), and names, titles and numbers take their short spoken
# form. Unlike that dub, every statement stays ("get out of Haishi" kept).
# (original, too long, said briefly)
_KM_BRIEF_EXAMPLES = [
    ("他说什么，10万买他赢", "គាត់និយាយស្អី? ដប់ម៉ឺនភ្នាល់គាត់ឈ្នះ?", "មួយសែនភ្នាល់គាត់ឈ្នះ?"),
    ("途！我受不了了！", "ហឺ! ខ្ញុំទ្រាំលែងបានហើយ!", "ទ្រាំលែងបានហើយ!"),
    ("原来是这么赚钱的", "អូ! វិធីរកលុយបែបនេះសោះ។", "វិធីរកលុយបែបនេះសោះ។"),
    ("也就说他想在10分钟内", "និយាយរួមគឺគាត់ចង់ក្នុងរយៈពេលដប់នាទី", "និយាយរួមគឺគាត់ចង់ក្នុង១០នាទី"),
    ("现在怎么跟瘟鸡一样", "ឥឡូវម៉េចបានដូចមាន់ឈឺអ៊ីចឹង?", "ឥឡូវម៉េចបានដូចមាន់ឈឺ?"),
    ("揍我小弟的时候不是挺猛的吗", "ពេលវាយប្អូនខ្ញុំ មិនឃើញខ្លាំងទេអី?",
     "វាយប្អូនខ្ញុំ មិនឃើញខ្លាំងទេអី?"),
    ("跃庭是在侮辱人吗，要不直接认输，滚出海市算了",
     "សាលប្រជុំយីធីង កំពុងតែមើលងាយមនុស្សមែនទេ? ចុះចាញ់ហើយចេញពីហៃស៊ឺទៅ។",
     "យីធីងមើលងាយមនុស្សមែនទេ? ចុះចាញ់ ចេញពីហៃស៊ឺទៅ។"),
]


def _brief_guide(target_lang: str) -> str:
    """How to say a dubbing line more briefly without losing a statement —
    for the shortening prompts; with the user's own approved lines as
    examples when dubbing into Khmer."""
    rule = ("Say it the way a good dubbing actor would. What the voice and the "
            "picture already carry may go: an interjection, a surprised echo "
            "question before the real point (\"what did he say?!\"), a pronoun "
            "the language does not need, fillers and softeners; names, titles and "
            "numbers take their short spoken form. Every statement, request, "
            "order, real question, name, number and negation stays.")
    if _base_lang(target_lang) != "km":
        return rule
    ex = "\n".join(f"- {s} | {long} → {short}" for s, long, short in _KM_BRIEF_EXAMPLES)
    return (rule + " In Khmer: interjections like ហឺ! / អូ!, ខ្ញុំ / គាត់ when it is "
            "clear who, fillers like អ៊ីចឹង, បន្តិច, កំពុងតែ, ពេល; short forms like "
            "យីធីង for សាលប្រជុំយីធីង, មួយសែន for ដប់ម៉ឺន, ១០នាទី for រយៈពេលដប់នាទី. "
            "Lines approved by the user (original | too long → said briefly):\n" + ex)


def _brief_ok(target_lang: str) -> str:
    """For the meaning check of shortened lines: the user's approved brief
    lines, which it must not flag (it kept asking for 他说什么 back)."""
    if _base_lang(target_lang) != "km":
        return ""
    return (" These shortened Khmer lines are correct — do not flag lines like them: "
            + "; ".join(f"{s} → {short}" for s, _l, short in _KM_BRIEF_EXAMPLES) + ".")
_REFUSED = "\x00refused\x00"   # check cache: found wrong, but no usable fix


# The speed-up a line may take to end inside its own slot (lip-sync), and
# the one translation budgets ("max N chars") are made for — so lines keep
# their natural length and only those that need it are spoken faster, up to
# DubOptions.max_tempo (2.0x by default) to end before the next line.
COMFORT_TEMPO = 1.4


def _squeeze(max_tempo: float) -> float:
    """Strongest speed-up of a dubbed line: max_tempo itself (2.0 by
    default), never beyond — a line still too long at that speed is first
    said in fewer words, and only then starts the next line a little late."""
    return max(1.0, max_tempo)


def _faster_rate(rate: str, speed: float) -> str:
    """Edge TTS rate for `speed` times the given rate ("+10%" x 1.2 -> "+32%")."""
    return ap.faster_rate(rate, speed)


# Below this gender confidence a character's voice-measured gender is shown
# as "check it" (it sounds between male and female).
GENDER_SURE = 0.4

# Pitch offsets that keep same-gender preset voices apart when no cloning
# is available (rank 0 = most talkative character of that gender).
_PITCH_SPREAD = ["+0Hz", "-8Hz", "+8Hz", "-14Hz", "+14Hz", "-20Hz", "+20Hz"]

_OPENVOICE_DIR_NAME = "openvoice"
_OPENVOICE_FILES = {
    "config.json": ("https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/"
                    "main/converter/config.json", 1),
    "checkpoint.pth": ("https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/"
                       "main/converter/checkpoint.pth", 131),
}
OPENVOICE_PIP = [
    ["--no-deps", "git+https://github.com/myshell-ai/OpenVoice.git"],
    ["librosa", "eng_to_ipa", "inflect", "unidecode", "jieba", "pypinyin",
     "cn2an"],
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


def _emit(progress: ProgressFn | None, frac: float, text: str) -> None:
    if progress is not None:
        try:
            progress(max(0.0, min(1.0, float(frac))), text)
        except Exception:
            pass


def _sub(progress: ProgressFn | None, lo: float, hi: float) -> ProgressFn | None:
    if progress is None:
        return None

    def fn(f: float, t: str) -> None:
        _emit(progress, lo + (hi - lo) * max(0.0, min(1.0, f)), t)
    return fn


def _sha1(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip(" .")
    return (name or "video")[:60]


def _from_dict(cls, d: Any):
    """Tolerant dataclass construction (unknown keys ignored)."""
    if not isinstance(d, dict):
        return cls()
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def _cps(lang: str) -> float:
    return SPEECH_CPS.get(_base_lang(lang), _DEFAULT_CPS)


# Khmer Edge voices, measured on 2,465 voiced lines (2026-11-11): about 17.5
# characters a second of running speech, plus a pause at each space, "?",
# "!" or "។" inside a line, plus a fixed onset/offset. The plain 12 chars/s
# estimate was 0.54 s off on average — too long for running text, too short
# for lines with pauses ("ហឺស! ឯង": 7 characters, 1.65 s).
_KM_S_PER_CHAR, _KM_PAUSE_S, _KM_BASE_S = 0.057, 0.166, 0.30
_KM_PAUSE = re.compile(r"[!?！？។,，…]+\s*|\s+")


def speech_seconds(text: str, lang: str) -> float:
    """How long a dub voice takes to say `text` (trimmed speech)."""
    if _base_lang(lang) != "km":
        return estimate_speech_seconds(text, lang)
    t = speakable_km(text or "").strip()
    core = re.sub(r"\s+", "", t)
    if not core:
        return 0.0
    pauses = len(_KM_PAUSE.findall(t.rstrip("។?!.…！？ ")))
    return _KM_BASE_S + _KM_S_PER_CHAR * len(core) + _KM_PAUSE_S * pauses


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class Character:
    id: str = "S1"
    name: str = ""
    gender: str = "F"            # "M" | "F"
    ref_wav: str = ""            # reference clip for cloning
    ref_text: str = ""           # its transcript (source language)
    ref_custom: bool = False     # user supplied the reference: keep it
    engine: str = "auto"         # one of CHAR_ENGINES
    voice: str = ""              # preset voice (edge / openai / elevenlabs)
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    line_count: int = 0
    talk_seconds: float = 0.0
    embedding: list[float] = field(default_factory=list)   # voice fingerprint
    embed_method: str = ""       # "ecapa" | "mfcc"
    cast_match: str = ""         # series cast member this was matched to
    age: str = ""                # "child" | "teen" | "adult" | "elder" (AI)
    gender_source: str = ""      # "voice" | "ai" | "user"
    gender_conf: float = 0.0     # how sure the voice measurement is (0..1)
    gender_recheck: bool = False  # gender set by hand: re-decide its lines

    def label(self) -> str:
        return self.name.strip() or self.id


@dataclass
class DubLine:
    start: float = 0.0
    end: float = 0.0
    source: str = ""
    text: str = ""               # translation
    speaker: str = "S1"
    status: str = ""             # "" | ok | overlong | overrun | failed
    needed: float = 0.0          # estimated speech seconds of `text`
    tts_key: str = ""            # cache key of the synthesized raw clip
    fast_key: str = ""           # cache key of the same line spoken faster
    asr_speaker: str = ""        # speaker label from the recogniser (Deepgram)
    voice_gender: str = ""       # "M" | "F" | "" measured on this line's audio
    voice_conf: float = 0.0      # how sure that measurement is (0..1)
    gender: str = ""             # this line's voice gender; "" = its character's
    gender_src: str = ""         # "" | "voice" (measured) | "user" (set by hand)
    text_src: str = ""           # "" (AI) | "user": typed in Lines, kept when re-translating
    heard: str = ""              # the source as recognised, when a misheard word was corrected
    on_screen: bool = False      # the source is the video's own hard subtitle (not misheard)
    voice_sim: float = -1.0      # voice likeness to the line before (cosine; -1 = not measured)
    voice_sure: bool = False     # its voice print clearly matches its speaker: the AI check keeps it
    print_gender: str = ""       # "M" | "F" from the voice prints' male/female split ("" = none)
    delay: float = 0.0           # s the dubbed line starts late (no overlap)
    # the whole translation, as the Translate step left it — before Fit /
    # Synthesize reworded or cut it to fit the timing; subtitles show it
    full_text: str = ""
    # why a person should check this line ("" = nothing): "recovered" (heard
    # only by the recogniser's second listen), "low_confidence" (the
    # recogniser doubted it), "maybe_prompt" (only a name the recogniser was
    # given), "foreign_script" (the dub voice cannot read part of the
    # translation)
    review: str = ""
    # left out by the user ("nobody says this"): not translated, voiced or
    # subtitled. Kept in the list, so line numbers and their voice files stay
    skip: bool = False
    # how much faster than normal the voice says it (Synthesize: the voice's
    # own faster rate × time-stretch); 1.0 = normal speed
    speed: float = 1.0

    def gender_of(self, ch: "Character | None") -> str:
        """Gender this line is voiced and translated with."""
        return self.gender or (ch.gender if ch else "") or "F"


@dataclass
class GlossaryTerm:
    source: str = ""
    target: str = ""
    note: str = ""
    origin: str = ""             # "ai" | "user" (typed in the Glossary tab) | "series"; "" = older project


@dataclass
class DubOptions:
    target_lang: str = "km"
    source_lang: str = ""        # "" = auto-detect
    whisper_size: str = "auto"       # "auto" = medium on a GPU, small on CPU
    use_demucs: bool = True
    num_speakers: int = 0        # 0 = auto
    max_speakers: int = 6
    # "pair": one male + one female voice (Khmer: Piseth / Sreymom), chosen
    # line by line from the speaker's gender; "clone": each character's own
    # voice; "preset": a preset voice per character
    voice_mode: str = "pair"
    vary_pitch: bool = False     # shift same-gender characters' pitch apart
    background: str = "keep"     # "keep" music & effects | "none" = voices only
    auto_shorten: bool = True    # AI-shorten lines whose real speech is too long
    quality: str = "high"        # "high" = translate + review, "fast" = 1 pass
    style: str = ""
    # strongest speed-up of a line too long for its time, the last resort:
    # lines are fitted at COMFORT_TEMPO (1.4x) — longer ones said more
    # briefly with the same meaning first — and only what still does not fit
    # is spoken faster, just enough to end before the next line, up to this.
    # The user asked to "speed the speech to fit the slot" rather than start
    # the next line late (2026-11-15), then preferred rewording before speed
    # (2026-11-16: the 1.4x dub of 2026-11-11 sounded better than 2.0x).
    max_tempo: float = 2.0
    music_gain_db: float = 0.0
    voice_gain_db: float = 0.0
    soft_subtitles: bool = True
    burn_subtitles: bool = False
    # subtitles show each line's whole translation, not the dub's shortened
    # wording (on a Chinese drama the dub lines had lost 9-32 % of the Khmer)
    subtitle_full: bool = True
    # a line too long for its time even spoken max_tempo faster: True — the
    # AI may cut it down to its main point (lip-sync first); False — it may
    # only be worded more tightly, every statement and question kept, and is
    # then spoken faster / a little late (nothing said is left out)
    allow_cuts: bool = False
    output_path: str = ""
    series: str = ""             # drama series: shared glossary + voice cast
    use_tm: bool = True          # reuse translation memory
    subtitle_style: str = ""     # app.engines.subtitles preset ("" = plain srt)


@dataclass
class Check:
    id: str
    label: str
    ok: bool
    detail: str = ""
    fix: str = ""
    required: bool = True
    pip: list[list[str]] = field(default_factory=list)  # pip arg-lists to fix
    download: str = ""           # model id this app can download to fix


class DubProject:
    """All state of one dubbing job; saved as ``project.json`` in work_dir."""

    def __init__(self, video: str, work_dir: str,
                 options: DubOptions | None = None):
        self.video = video
        self.work_dir = work_dir
        self.options = options or DubOptions()
        self.duration = 0.0
        self.src_lang = ""
        self.audio_wav = ""
        self.vocals_wav = ""
        self.music_wav = ""
        self.asr_wav = ""
        self.lines: list[DubLine] = []
        self.characters: list[Character] = []
        self.glossary: list[GlossaryTerm] = []
        self.story_context = ""
        self.steps: dict[str, dict] = {s: {"status": "pending", "note": ""}
                                       for s in STEP_IDS}
        self.voice_track = ""
        self.mix_wav = ""
        self.srt_path = ""
        self.output = ""
        # finished translate / review / check / shorten results, so a failed
        # run resumes and a re-run gives the same lines
        self.tr_cache: dict[str, dict[str, str]] = {"draft": {}, "review": {},
                                                    "check": {}, "shorten": {},
                                                    "misheard": {}, "notes": {}}
        self.speaker_score = 0.0     # voice-clustering silhouette (0 = unclear)

    # -- persistence ------------------------------------------------------
    @property
    def path(self) -> str:
        return os.path.join(self.work_dir, "project.json")

    def sub(self, *parts: str) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def to_json(self) -> dict:
        return {
            "version": 1,
            "video": self.video, "work_dir": self.work_dir,
            "options": asdict(self.options),
            "duration": self.duration, "src_lang": self.src_lang,
            "audio_wav": self.audio_wav, "vocals_wav": self.vocals_wav,
            "music_wav": self.music_wav, "asr_wav": self.asr_wav,
            "lines": [asdict(l) for l in self.lines],
            "characters": [asdict(c) for c in self.characters],
            "glossary": [asdict(g) for g in self.glossary],
            "story_context": self.story_context,
            "steps": self.steps,
            "voice_track": self.voice_track, "mix_wav": self.mix_wav,
            "srt_path": self.srt_path, "output": self.output,
            "tr_cache": self.tr_cache, "speaker_score": self.speaker_score,
        }

    @classmethod
    def from_json(cls, d: dict) -> "DubProject":
        p = cls(str(d.get("video", "")), str(d.get("work_dir", "")),
                _from_dict(DubOptions, d.get("options")))
        p.duration = float(d.get("duration") or 0.0)
        for k in ("src_lang", "audio_wav", "vocals_wav", "music_wav",
                  "asr_wav", "story_context", "voice_track", "mix_wav",
                  "srt_path", "output"):
            setattr(p, k, str(d.get(k) or ""))
        p.lines = [_from_dict(DubLine, x) for x in d.get("lines") or []]
        p.speaker_score = float(d.get("speaker_score") or 0.0)
        tc = d.get("tr_cache") if isinstance(d.get("tr_cache"), dict) else {}
        p.tr_cache = {"draft": dict(tc.get("draft") or {}),
                      "review": dict(tc.get("review") or {}),
                      "check": dict(tc.get("check") or {}),
                      "shorten": dict(tc.get("shorten") or {}),
                      "misheard": dict(tc.get("misheard") or {}),
                      "notes": dict(tc.get("notes") or {})}
        p.characters = [_from_dict(Character, x)
                        for x in d.get("characters") or []]
        p.glossary = [_from_dict(GlossaryTerm, x)
                      for x in d.get("glossary") or []]
        steps = d.get("steps") or {}
        for s in STEP_IDS:
            st = steps.get(s) if isinstance(steps.get(s), dict) else {}
            p.steps[s] = {"status": str(st.get("status") or "pending"),
                          "note": str(st.get("note") or "")}
            try:                       # how long the task took (the task list shows it)
                if st.get("secs") is not None:
                    p.steps[s]["secs"] = float(st["secs"])
            except (TypeError, ValueError):
                pass
            if st.get("ai"):           # why a paused task found no AI (the Dub tab's advice)
                p.steps[s]["ai"] = str(st["ai"])
            if st.get("note_km"):      # a failed task's message in Khmer
                p.steps[s]["note_km"] = str(st["note_km"])
        return p

    def save(self) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    @classmethod
    def load(cls, path: str) -> "DubProject":
        with open(path, "r", encoding="utf-8") as f:
            p = cls.from_json(json.load(f))
        p.work_dir = os.path.dirname(os.path.abspath(path))
        return p

    @classmethod
    def for_video(cls, video: str,
                  options: DubOptions | None = None) -> "DubProject":
        """Open the existing project of this video, or create a new one."""
        video = os.path.abspath(video)
        work = _project_dir(video)
        # a project moved over to this path (relink) stays in its old folder
        pj = Path(project_json_for(video) or work / "project.json")
        if pj.is_file():
            try:
                p = cls.load(str(pj))
                if options is not None:
                    p.options = options
                return p
            except Exception as exc:
                log.warning("Could not load dub project %s (%s); starting "
                            "fresh", pj, exc)
        return cls(video, str(work), options)

    @classmethod
    def relink(cls, project_json: str, video: str) -> "DubProject":
        """An earlier project (find_moved_project) for `video` at its new
        place. The project stays in its folder (every file of it is there);
        it now names the new path, so opening the video finds it again."""
        p = cls.load(project_json)
        old = p.video
        p.video = os.path.abspath(video)
        # the dubbed video and its subtitles usually moved along with it
        folder = os.path.dirname(p.video)
        for attr in ("output", "srt_path"):
            was = getattr(p, attr)
            near = os.path.join(folder, os.path.basename(was)) if was else ""
            if was and not os.path.isfile(was) and os.path.isfile(near):
                setattr(p, attr, near)
                if attr == "output" and p.options.output_path == was:
                    p.options.output_path = near
        p.save()
        log.info("dub project %s now belongs to %s (was %s)", p.work_dir, p.video, old)
        return p

    # -- step state ---------------------------------------------------------
    def step_status(self, step: str) -> str:
        return self.steps.get(step, {}).get("status", "pending")

    def step_done(self, step: str) -> bool:
        return self.step_status(step) in ("done", "warn")

    def set_step(self, step: str, status: str, note: str = "") -> None:
        self.steps[step] = {"status": status, "note": note}

    def invalidate_from(self, step: str) -> None:
        """Mark `step` and every later step as pending (their outputs are
        stale)."""
        if step not in STEP_IDS:
            return
        for s in STEP_IDS[STEP_IDS.index(step):]:
            if self.steps[s]["status"] != "pending":
                self.steps[s] = {"status": "pending", "note": ""}

    def next_step(self) -> str | None:
        for s in STEP_IDS:
            if not self.step_done(s):
                return s
        return None

    def character(self, cid: str) -> Character | None:
        for c in self.characters:
            if c.id == cid:
                return c
        return None

    def recount(self) -> None:
        for c in self.characters:
            mine = [l for l in self.lines if l.speaker == c.id]
            c.line_count = len(mine)
            c.talk_seconds = round(sum(max(0.0, l.end - l.start)
                                       for l in mine), 2)

    def slot(self, i: int) -> float:
        """Seconds line i may occupy: its own span plus up to 0.6 s of the
        silence before the next line."""
        ln = self.lines[i]
        nxt = (self.lines[i + 1].start if i + 1 < len(self.lines)
               else max(self.duration, ln.end))
        own = max(0.2, ln.end - ln.start)
        return max(own, min(nxt, ln.end + 0.6) - ln.start)

    def comfort_tempo(self) -> float:
        """The speed-up for lip-sync and translation budgets: COMFORT_TEMPO,
        never more than max_tempo."""
        return max(1.0, min(COMFORT_TEMPO, self.options.max_tempo))

    def max_chars(self, i: int) -> int:
        """Characters line i can hold when spoken at the comfortable speed-up
        (COMFORT_TEMPO): the budget translations are written for. A longer
        line is not reworded for that — it is spoken faster, up to max_tempo."""
        tempo = self.comfort_tempo()
        if _base_lang(self.options.target_lang) == "km":
            # the measured Khmer voice model, a pause or two allowed for
            secs = self.slot(i) * tempo - _KM_BASE_S
            return max(4, int(secs / _KM_S_PER_CHAR * 0.9))
        return max(4, int(self.slot(i) * _cps(self.options.target_lang) * tempo))


# ---------------------------------------------------------------------------
# transcript -> lines
# ---------------------------------------------------------------------------

_SENT_END = re.compile(r"[.!?。！？…]+[\"'”’)]*$")


def _unstretch(words: list) -> list:
    """Whisper sometimes stretches a word across a long silence or music
    (one 8-character line lasting 28 s). Cap each word's duration: the first
    word is trimmed from the left (late start), others from the right."""
    from types import SimpleNamespace as NS
    out = []
    for k, w in enumerate(words):
        s, e, t = float(w.start), float(w.end), str(w.text)
        cap = 0.8 + 0.3 * len(t.strip())
        if e - s > cap:
            if k == 0:
                s = e - cap
            else:
                e = s + cap
        out.append(NS(start=s, end=e, text=t, speaker=getattr(w, "speaker", None)))
    return out


def _is_asr_label(spk) -> bool:
    """A speaker label set by the recogniser ("D1"…), not a voice name that
    assign_voices() may have stored in Segment.speaker."""
    return bool(re.fullmatch(r"D\d+", str(spk or "")))


def quiet_spans(wav: str, min_len: float = 0.08, depth: float = 25.0) -> list[tuple[float, float]]:
    """Quiet stretches of a speech track (10 ms frames `depth` dB under its
    loud speech, at least `min_len` s): where a sentence or a speaker ends
    (25 dB), or a clause inside fast speech (18 dB, 50 ms: these matched
    Whisper's sentence breaks on a Chinese drama). Deepgram's Chinese word
    times touch end to end and show no pause at all."""
    try:
        x = ap._read_wav_as_int16_mono(wav, 16000).astype(np.float32) / 32768.0
    except Exception:
        return []
    hop = 160
    n = len(x) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    voiced = db[db > -50]
    if voiced.size == 0:
        return []
    quiet = np.append(db < np.percentile(voiced, 90) - depth, False)
    spans, s = [], None
    for i, q in enumerate(quiet):
        if q and s is None:
            s = i
        elif not q and s is not None:
            if (i - s) * 0.01 >= min_len:
                spans.append((s * 0.01, i * 0.01))
            s = None
    return spans


# Chinese sentence-final particles: at a speaker change they end the sentence
# before ("拉客了 | 你说的是…", "七十多万吧 | 开什么")
_ZH_PARTICLES = set("了吗呢吧啊呀啦嘛哦哈哇呗咯嘞")


def _refine_cjk_words(words: list, quiet: list[tuple[float, float]],
                      dips: list[tuple[float, float]] = ()) -> list:
    """Deepgram's Chinese / Japanese words → one Word per character, with its
    speaker changes put where they really are. Deepgram cuts turns inside
    words ("对啊后" | "座不是…") and its word times touch end to end, so:
    a particle at a speaker change stays with the sentence it ends; a blip
    of one or two characters of another speaker inside a sentence goes back
    to the speaker around it; a speaker change moves to the nearest real
    pause in the audio (`quiet`), and every pause becomes a gap between the
    characters (what split_segments splits at)."""
    from app.engines.audio_pipeline import Word
    ws = list(words)
    for k in range(len(ws) - 1):            # a word dated after the next one
        if float(ws[k].start) > float(ws[k + 1].start):
            prev_end = float(ws[k - 1].end) if k else float(ws[k + 1].start)
            ws[k] = Word(start=min(prev_end, float(ws[k + 1].start)),
                         end=float(ws[k + 1].start), text=ws[k].text, speaker=ws[k].speaker)
    chars = []
    for w in ws:
        t = re.sub(r"\s+", "", str(w.text))
        if not t:
            continue
        step = max(0.0, float(w.end) - float(w.start)) / len(t)
        chars += [Word(start=float(w.start) + j * step, end=float(w.start) + (j + 1) * step,
                       text=c, speaker=w.speaker) for j, c in enumerate(t)]
    if len(chars) < 2:
        return chars

    def runs() -> list[tuple[int, int]]:
        out, s = [], 0
        for k in range(1, len(chars) + 1):
            if k == len(chars) or chars[k].speaker != chars[s].speaker:
                out.append((s, k))
                s = k
        return out

    # particles at the start of a new speaker's run end the previous sentence
    for (a, _b), (c, d) in zip(runs(), runs()[1:]):
        k = c
        while k < d and chars[k].text in _ZH_PARTICLES:
            chars[k].speaker = chars[a].speaker
            k += 1
    # a blip: one or two characters of another speaker between one speaker;
    # or up to three ending the utterance with no pause before them
    # ("多少钱3 | 万多", "再贵也是 | 别人的")
    rs = runs()
    for j in range(1, len(rs) - 1):
        a, (c, d), e = rs[j - 1][0], rs[j], rs[j + 1][0]
        if d - c <= 2 and chars[a].speaker == chars[e].speaker:
            for k in range(c, d):
                chars[k].speaker = chars[a].speaker
    rs = runs()
    if len(rs) > 1:
        (a, _b), (c, d) = rs[-2], rs[-1]
        t = chars[c].start
        if d - c <= 3 and not any(s - 0.15 <= t <= e + 0.15 for s, e in quiet):
            for k in range(c, d):
                chars[k].speaker = chars[a].speaker
    # each speaker change moves to the nearest pause within 0.6 s
    for (a, _b), (c, _d) in zip(runs(), runs()[1:]):
        old, new, t = chars[a].speaker, chars[c].speaker, chars[c].start
        near = [s for s in quiet if abs((s[0] + s[1]) / 2 - t) <= 0.6]
        if not near:
            continue
        cut = sum(min(near, key=lambda s: abs((s[0] + s[1]) / 2 - t))) / 2
        lo, hi = min(cut, t) - 0.05, max(cut, t) + 0.05
        for ch in chars:
            mid = (ch.start + ch.end) / 2
            if lo <= mid <= hi and ch.speaker in (old, new):
                ch.speaker = old if mid < cut else new
    # cuts only between words: character times are estimates, and a cut by
    # time alone fell inside words ("挤公 | 交", "也买 | 了任务")
    ok = _cut_points("".join(c.text for c in chars))

    def snap(k: int) -> int:
        """The nearest allowed cut to index k (cut before chars[k])."""
        for dk in (0, 1, -1, 2, -2, 3, -3):
            if 0 < k + dk < len(chars) and ok[k + dk]:
                return k + dk
        return k

    for (a, _b), (c, _d) in zip(runs(), runs()[1:]):
        c2 = snap(c)
        for k in range(min(c, c2), max(c, c2)):
            chars[k].speaker = chars[a].speaker if k < c2 else chars[c].speaker
    # a pause (or a dip) becomes a mark before the first character after it
    # (at a word edge); split_segments splits there
    for spans, attr in ((quiet, "pause_before"), (dips, "dip_before")):
        for s, e in spans:
            mid = (s + e) / 2
            k = next((j for j, ch in enumerate(chars) if (ch.start + ch.end) / 2 > mid), 0)
            if 0 < k < len(chars) and (chars[k - 1].start + chars[k - 1].end) / 2 < mid:
                k = snap(k)
                setattr(chars[k], attr, max(getattr(chars[k], attr, 0.0), e - s))
    return chars


def _cut_points(text: str) -> list[bool]:
    """ok[k]: a line may be cut (or a speaker change put) before text[k] —
    at a word edge (jieba, when installed) and never before a particle."""
    starts = None
    try:
        import jieba
        jieba.setLogLevel(60)
        starts, pos = set(), 0
        for w in jieba.cut(text):
            starts.add(pos)
            pos += len(w)
    except Exception:
        pass
    return [k > 0 and (starts is None or k in starts) and text[k] not in _ZH_PARTICLES
            for k in range(len(text))]


def split_segments(segments: list, max_len: float = 10.0,
                   pause: float = 0.6, min_len: float = 0.8, lang: str = "",
                   quiet: list[tuple[float, float]] | None = None,
                   dips: list[tuple[float, float]] = ()) -> list[DubLine]:
    """Split Whisper segments at pauses / sentence ends so each line is one
    utterance (and usually one speaker); drop repeated and empty lines.
    Words labelled with a speaker (Deepgram) also split at speaker changes,
    and each line keeps that label in `asr_speaker`. Chinese / Japanese from
    Deepgram (`lang`, with the audio's `quiet` spans and shallower `dips`):
    see _refine_cjk_words; a line over 2.5 s also splits at a dip."""
    out: list[DubLine] = []
    cjk = _base_lang(lang) in ("zh", "ja") and quiet is not None
    for seg in segments:
        words = _unstretch([w for w in (getattr(seg, "words", None) or [])
                            if str(getattr(w, "text", "")).strip()])
        text = str(getattr(seg, "text", "") or "").strip()
        seg_spk = str(getattr(seg, "speaker", "") or "") if _is_asr_label(
            getattr(seg, "speaker", "")) else ""
        review = str(getattr(seg, "review", "") or "")
        if not text:
            continue
        if getattr(seg, "subtitle", False) and words and \
                max(float(w.end) for w in words) - min(float(w.start) for w in words) <= max_len:
            # one hard subtitle = one line, one speaker: Deepgram's turns and
            # the pauses inside it are no cuts ("苏" | "大佬请喝茶", "反正是垃圾"
            # | "话"); its label is the one heard longest in it
            talk: dict[str, float] = {}
            for w in words:
                if _is_asr_label(w.speaker):
                    talk[w.speaker] = talk.get(w.speaker, 0.0) + float(w.end) - float(w.start)
            out.append(DubLine(start=min(float(w.start) for w in words),
                               end=max(float(w.end) for w in words), source=text,
                               asr_speaker=max(talk, key=talk.get) if talk else seg_spk,
                               review=review))
            continue
        refined = cjk and bool(words) and any(getattr(w, "speaker", None) for w in words)
        if refined:
            words = _refine_cjk_words(words, quiet, dips)
        if not words:
            out.append(DubLine(start=float(seg.start), end=float(seg.end),
                               source=text, asr_speaker=seg_spk, review=review))
            continue
        pieces: list[list] = [[words[0]]]
        for prev, w in zip(words, words[1:]):
            cur = pieces[-1]
            gap = getattr(w, "pause_before", None) if refined else None
            if gap is None:
                gap = 0.0 if refined else float(w.start) - float(prev.end)
            cur_len = float(prev.end) - float(cur[0].start)
            if (gap >= (0.25 if cjk else pause) or w.speaker != prev.speaker
                    or (_SENT_END.search(str(prev.text).strip())
                        and cur_len >= 1.0)
                    or (refined and cur_len >= 2.5 and getattr(w, "dip_before", 0.0) > 0)
                    or cur_len >= max_len):
                pieces.append([w])
            else:
                cur.append(w)
        # merge fragments that are too short to voice on their own (never
        # across a speaker change)
        merged: list[list] = []
        for p in pieces:
            dur = float(p[-1].end) - float(p[0].start)
            if merged and dur < min_len and merged[-1][-1].speaker == p[0].speaker:
                merged[-1].extend(p)
            else:
                merged.append(p)
        if len(merged) > 1:
            dur0 = float(merged[0][-1].end) - float(merged[0][0].start)
            if dur0 < min_len and merged[0][-1].speaker == merged[1][0].speaker:
                merged[1] = merged[0] + merged[1]
                merged.pop(0)
        for p in merged:
            t = "".join(str(w.text) for w in p).strip()
            t = re.sub(r",\s*([.!?])", r"\1", t)          # "Roger,." -> "Roger."
            if t:
                spk = p[0].speaker if _is_asr_label(p[0].speaker) else seg_spk
                out.append(DubLine(start=float(p[0].start),
                                   end=float(p[-1].end), source=t,
                                   asr_speaker=spk or "", review=review))
    out.sort(key=lambda l: (l.start, l.end))
    return drop_repeats(_fix_turn_splits(out))


def _mark_heard(lines: list[DubLine], asr_segments: list, events: list) -> None:
    """After the hard subtitles corrected the transcript: a line under a
    subtitle is `on_screen` (its words are the script, not a guess), and one
    whose words differ from what the recogniser heard keeps that in `heard`
    (shown in the Lines tab)."""
    if not events:
        return
    words = []
    for seg in asr_segments:
        ws = [w for w in (getattr(seg, "words", None) or []) if str(w.text).strip()]
        if ws:
            words += [((float(w.start) + float(w.end)) / 2, str(w.text)) for w in ws]
        elif str(seg.text).strip():
            words.append(((float(seg.start) + float(seg.end)) / 2, str(seg.text)))
    words.sort()
    spans = [(float(e.start), float(e.end)) for e in events]
    for ln in lines:
        dur = max(0.01, ln.end - ln.start)
        cover = sum(max(0.0, min(ln.end, b + 0.15) - max(ln.start, a - 0.15)) for a, b in spans)
        ln.on_screen = cover >= 0.5 * dur
        heard = "".join(t for m, t in words if ln.start - 0.05 <= m <= ln.end + 0.05).strip()
        if ln.on_screen and heard and _norm_text(heard) != _norm_text(ln.source):
            ln.heard = heard


# Short words that open a new turn ("No. I…"): never handed back to the
# previous speaker as the end of their sentence.
_TURN_WORDS = {
    "no", "yes", "yeah", "yep", "nope", "okay", "ok", "well", "fine", "look",
    "wait", "hey", "thanks", "sorry", "right", "alright", "sure", "stop", "go",
    "what", "why", "how", "who", "where", "when", "oh", "ah", "uh", "um", "hmm",
    "god", "please", "good", "great", "really", "listen", "come", "hello", "hi",
    "bye", "exactly", "never", "maybe", "now", "here", "there", "so", "and",
    "but", "damn", "nothing", "done", "me", "you", "i", "we", "they", "he",
    "she", "it", "that", "this", "yours", "mine", "ours"}
_VOCATIVE = re.compile(r"^([A-Z][a-z'’-]+(?: [A-Z][a-z'’-]+)?)([.!?])\s+(\S.*)$")


def _fix_turn_splits(lines: list[DubLine]) -> list[DubLine]:
    """Repair two cuts recognisers make at a speaker change in English-like
    text (in place):

    - a name that ends a sentence but landed in the next line ("You're a
      jerk," | "Tom. Look, Celia…"): it goes back, with its share of the
      time (also when both pieces carry one label: word labels flip inside
      sentences);
    - a sentence that goes on in lower case after the cut ("Memory synced" |
      "and" | "locked."): those pieces keep the first speaker's label, so the
      lines can be joined later."""
    for a, b in zip(lines, lines[1:]):
        gap = b.start - a.end
        a_txt, b_txt = a.source.strip(), b.source.strip()
        if gap <= 0.35 and a_txt.endswith(","):
            m = _VOCATIVE.match(b_txt)
            if m and not any(w.lower() in _TURN_WORDS for w in m.group(1).split()):
                head, rest = m.group(1) + m.group(2), m.group(3).strip()
                dt = (b.end - b.start) * len(head) / max(1, len(b_txt))
                a.source = f"{a_txt} {head}"
                a.end = max(a.end, b.start + dt)
                b.source, b.start = rest, b.start + dt
                continue
        if (gap <= 0.5 and a.asr_speaker != b.asr_speaker
                and not _SENT_END.search(a_txt) and re.match(r"[a-z]", b_txt)):
            b.asr_speaker = a.asr_speaker
    return lines


def _norm_text(t: str) -> str:
    return re.sub(r"[\s\W_]+", "", t or "", flags=re.UNICODE).lower()


def drop_repeats(lines: list[DubLine]) -> list[DubLine]:
    """Remove a line that only echoes the previous one (a Whisper echo),
    keeping the longer copy: the same words right after it (< 1 s) that
    overlap it in time or take under half its time (8 characters in 0.2 s
    after the real 1.2 s line). A real repeat ("快走！快走！", "No! No!")
    takes about as long as the first and is kept — it used to be dropped
    like an echo. Every removal is logged."""
    out: list[DubLine] = []
    for ln in lines:
        if out:
            prev = out[-1]
            same = _norm_text(ln.source) and _norm_text(ln.source) == _norm_text(prev.source)
            if same and ln.start - prev.end < 1.0:
                d_prev, d_ln = prev.end - prev.start, ln.end - ln.start
                if ln.start < prev.end - 0.05 or min(d_prev, d_ln) < 0.5 * max(d_prev, d_ln):
                    keep, gone = (ln, prev) if d_ln > d_prev else (prev, ln)
                    log.info("echo left out: %r at %.2f s repeats the line at %.2f s",
                             gone.source, gone.start, keep.start)
                    out[-1] = keep
                    continue
        out.append(ln)
    return out


_CJK_LIKE = {"zh", "ja", "ko", "km", "th", "lo", "my"}
# CAM++ voice likeness (cosine) below which two neighbouring lines are not
# joined: different speakers next to each other measured ≤ 0.46 on a drama,
# the same speaker mostly 0.5-0.8
_SAME_VOICE = 0.5


def merge_fragments(lines: list[DubLine], lang: str, max_gap: float = 0.35,
                    short: float = 1.2, max_len: float = 6.0) -> list[DubLine]:
    """Join a very short line to its neighbour when the same speaker says
    both with (almost) no pause: whole sentences translate and time better
    than 0.4 s fragments."""
    if not lines:
        return lines
    joiner = "，" if _base_lang(lang) == "zh" else ("" if _base_lang(lang) in _CJK_LIKE else " ")
    out = [lines[0]]
    for ln in lines[1:]:
        prev = out[-1]
        gap = ln.start - prev.end
        # a sentence going on in lower case ("synced" | "and locked.") may
        # have a longer pause inside it
        goes_on = (not _SENT_END.search(prev.source.strip())
                   and re.match(r"[a-z]", ln.source.strip()) is not None)
        # a measured voice that clearly differs from the line before: two
        # people, whatever label they carry ("对" + the next speaker's
        # sentence became one line under the wrong voice)
        other_voice = 0.0 <= ln.voice_sim < _SAME_VOICE
        if (ln.speaker == prev.speaker and gap <= (0.8 if goes_on else max_gap)
                and not other_voice and not ln.skip and not prev.skip
                and (prev.end - prev.start < short or ln.end - ln.start < short)
                and ln.end - prev.start <= max_len):
            if joiner == " ":            # words keep a space after . ? !
                sep = " "
            else:
                sep = "" if re.search(r"[，,。.!?！？、…]$", prev.source.strip()) else joiner
            prev.source = (prev.source.strip() + sep + ln.source.strip()).strip()
            prev.end = ln.end
            prev.text, prev.tts_key, prev.status = "", "", ""
            prev.full_text = ""
            prev.review = prev.review or ln.review     # still worth a look
            continue
        out.append(ln)
    return out


# Instruction-style prompts ("please add punctuation") leak into drama
# transcripts as fake lines, so only known names are passed (as hotwords and
# a short prompt); any line that just repeats the prompt is dropped.
_WHISPER_PROMPTS: dict[str, str] = {}


def drop_prompt_echo(lines: list[DubLine], prompt: str) -> list[DubLine]:
    """Leave out a line that writes out the Whisper prompt (the series'
    names joined by 、): two or more of its names with nothing between
    them. A line that is one name alone ("林凡！") is someone calling that
    person — it used to be dropped too; now it stays, marked for review.
    Every removal is logged."""
    if not prompt:
        return lines
    pn = _norm_text(prompt)
    names = [n for n in (_norm_text(x) for x in re.split(r"[、,，;；\s]+", prompt)) if n]
    out = []
    for ln in lines:
        t = _norm_text(ln.source)
        if t and t in pn:
            if sum(1 for n in names if n in t) >= 2:
                log.info("prompt echo left out: %r at %.2f s", ln.source, ln.start)
                continue
            ln.review = ln.review or "maybe_prompt"
        out.append(ln)
    return out


def _installed_whisper_size() -> str:
    """The best already-downloaded Whisper model that is quick to load, for
    a language check before Deepgram transcribes; "" when none is there
    (audio_pipeline's, shared with every Deepgram transcription)."""
    return ap.installed_whisper_size()


def _recovery_whisper_size() -> str:
    """The Whisper model for the second listen after Deepgram: the one this
    PC would pick itself when it is downloaded, else the largest one there
    (small or bigger — the tiny models invent more than they recover); ""
    when none is there (nothing is downloaded for it)."""
    from app.core.model_manager import _whisper_present

    def present(size: str) -> bool:
        try:
            return _whisper_present(size)
        except Exception:
            return False
    best = _auto_whisper_size()
    if present(best):
        return best
    return next((s for s in ("large-v3", "medium", "small") if present(s)), "")


def _mmss(t: float) -> str:
    """Seconds as m:ss (h:mm:ss from an hour), for notes."""
    t = int(max(0.0, t))
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _auto_whisper_size() -> str:
    """The most accurate Whisper the PC runs well: large-v3 on an NVIDIA card
    with 6 GB or more (float16 needs ~4.5 GB; 10 s for 2 min on an RTX 5070
    Laptop), medium from 3 GB, small on the CPU or a smaller card. The GPU
    counts only when Whisper can really use it (cuBLAS present)."""
    try:
        from app.core.gpu import ct2_cuda, list_gpus
        if ct2_cuda():
            vram = max((g.vram_gb or 0 for g in list_gpus() if g.vendor == "nvidia"), default=0)
            return "large-v3" if vram >= 6 else "medium" if vram >= 3 else "small"
    except Exception:
        pass
    return "small"


# ---------------------------------------------------------------------------
# speaker features & clustering (numpy; speechbrain ECAPA when installed)
# ---------------------------------------------------------------------------

_FB_CACHE: dict[tuple, np.ndarray] = {}


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40,
                    fmin: float = 60.0, fmax: float = 7600.0) -> np.ndarray:
    key = (sr, n_fft, n_mels)
    if key in _FB_CACHE:
        return _FB_CACHE[key]
    fmax = min(fmax, sr / 2.0)

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            if b > a:
                fb[m - 1, k] = (k - a) / (b - a)
        for k in range(b, c):
            if c > b:
                fb[m - 1, k] = (c - k) / (c - b)
    _FB_CACHE[key] = fb
    return fb


def _median_f0(x: np.ndarray, sr: int) -> float:
    flen, hop = int(0.040 * sr), int(0.020 * sr)
    lag_min, lag_max = max(2, int(sr / 400.0)), int(sr / 60.0)
    if x.size < flen or flen <= lag_max:
        return float("nan")
    peak = float(np.max(np.abs(x))) or 1.0
    gate = max(0.01, 0.05 * peak)
    f0s = []
    for off in range(0, x.size - flen + 1, hop):
        fr = x[off:off + flen] - float(np.mean(x[off:off + flen]))
        if float(np.sqrt(np.mean(fr * fr))) < gate:
            continue
        ac = np.correlate(fr, fr, mode="full")[flen - 1:]
        if ac[0] <= 1e-9:
            continue
        band = ac[lag_min:lag_max + 1] / ac[0]
        k = int(np.argmax(band))
        if band[k] >= 0.30:
            f0s.append(sr / float(lag_min + k))
        if len(f0s) >= 200:
            break
    return float(np.median(f0s)) if len(f0s) >= 3 else float("nan")


def voice_features(x: np.ndarray, sr: int) -> np.ndarray | None:
    """Fixed-size voice descriptor: MFCC mean/std (19+19) + log-F0."""
    if x.size < int(0.35 * sr):
        return None
    x = x.astype(np.float32) - float(np.mean(x))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    flen, hop, n_fft = int(0.025 * sr), int(0.010 * sr), 512
    while n_fft < flen:
        n_fft *= 2
    nf = 1 + (x.size - flen) // hop
    if nf < 10:
        return None
    idx = np.arange(flen)[None, :] + hop * np.arange(nf)[:, None]
    frames = x[idx] * np.hamming(flen).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    logmel = np.log(spec @ _mel_filterbank(sr, n_fft).T + 1e-10)
    energy = logmel.mean(axis=1)
    keep = energy >= np.percentile(energy, 35)
    if int(keep.sum()) < 8:
        return None
    logmel = logmel[keep]
    n_mels = logmel.shape[1]
    n = np.arange(n_mels)
    dct = np.cos(np.pi / n_mels * (n[None, :] + 0.5) * np.arange(20)[:, None])
    mfcc = (logmel @ dct.T)[:, 1:]
    f0 = _median_f0(x, sr)
    lf0 = np.log(f0) if np.isfinite(f0) else np.nan
    return np.concatenate([mfcc.mean(0), mfcc.std(0),
                           [lf0, lf0, lf0]]).astype(np.float64)


def _normalize_features(feats: list[np.ndarray | None]) -> tuple[np.ndarray, list[int]]:
    rows = [i for i, f in enumerate(feats) if f is not None]
    if not rows:
        return np.zeros((0, 1)), []
    X = np.vstack([feats[i] for i in rows])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN F0 column
        col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X = np.where(np.isfinite(X), X, col_mean)
    std = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(std > 1e-9, std, 1.0)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return X, rows


def _kmeans_cos(X: np.ndarray, k: int, iters: int = 40, restarts: int = 5,
                seed: int = 7) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_lab, best_score = np.zeros(len(X), dtype=int), -np.inf
    for _ in range(restarts):
        # k-means++ initialisation on cosine distance
        cents = [X[rng.integers(len(X))]]
        for _c in range(1, k):
            d = 1.0 - np.max(X @ np.vstack(cents).T, axis=1)
            d = np.clip(d, 0, None) ** 2
            p = d / d.sum() if d.sum() > 0 else None
            cents.append(X[rng.choice(len(X), p=p)])
        C = np.vstack(cents)
        lab = np.zeros(len(X), dtype=int)
        for _it in range(iters):
            new = np.argmax(X @ C.T, axis=1)
            if _it and np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                m = X[lab == j]
                if len(m):
                    c = m.sum(axis=0)
                    C[j] = c / max(np.linalg.norm(c), 1e-9)
        score = float(np.sum(X * C[lab]))
        if score > best_score:
            best_lab, best_score = lab.copy(), score
    return best_lab, best_score


def _silhouette_cos(X: np.ndarray, lab: np.ndarray, sample: int = 700) -> float:
    if len(set(lab.tolist())) < 2:
        return -1.0
    idx = np.arange(len(X))
    if len(X) > sample:
        idx = np.random.default_rng(3).choice(len(X), sample, replace=False)
    Xs, ls = X[idx], lab[idx]
    D = 1.0 - Xs @ Xs.T
    s = []
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, ls == o].mean() for o in set(ls.tolist()) if o != ls[i])
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s)) if s else -1.0


def cluster_speakers(X: np.ndarray, num: int = 0, max_k: int = 6,
                     min_silhouette: float = 0.06, return_score: bool = False,
                     merge_above: float | None = None):
    """Cluster L2-normalised voice vectors. num=0 picks k by silhouette.
    With return_score, returns (labels, best silhouette) — a low score
    means the voices are hard to tell apart. `merge_above`: clusters whose
    mean voices are at least this alike are one speaker (k-means splits a
    lead who talks calmly and shouts into two)."""
    n = len(X)
    if n == 0:
        z = np.zeros(0, dtype=int)
        return (z, 0.0) if return_score else z
    if num == 1 or n < 4:
        z = np.zeros(n, dtype=int)
        return (z, 0.0) if return_score else z
    if num and num > 1:
        lab = _kmeans_cos(X, min(num, n))[0]
        return (lab, _silhouette_cos(X, lab)) if return_score else lab
    top = -1.0
    best_lab, best_s = np.zeros(n, dtype=int), min_silhouette
    for k in range(2, max(2, min(max_k, n - 1)) + 1):
        lab, _ = _kmeans_cos(X, k)
        s = _silhouette_cos(X, lab)
        top = max(top, s)
        if s > best_s + 0.01:   # prefer fewer speakers on near-ties
            best_lab, best_s = lab, s
    # fold tiny clusters (< 3 lines and < 3%) into their nearest neighbour
    while True:
        ids, counts = np.unique(best_lab, return_counts=True)
        tiny = [c for c, m in zip(ids, counts) if m < 3 and m < 0.03 * n]
        if not tiny or len(ids) <= 1:
            break
        t = tiny[0]
        cents = {c: X[best_lab == c].mean(axis=0) for c in ids if c != t}
        tc = X[best_lab == t].mean(axis=0)
        target = max(cents, key=lambda c: float(cents[c] @ tc))
        best_lab = np.where(best_lab == t, target, best_lab)
    while merge_above is not None and not num:
        ids = np.unique(best_lab)
        if len(ids) <= 1:
            break
        cents = {}
        for c in ids:
            v = X[best_lab == c].mean(axis=0)
            cents[c] = v / max(float(np.linalg.norm(v)), 1e-9)
        pair, best = None, merge_above
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1:]:
                s = float(cents[a] @ cents[b])
                if s >= best:
                    pair, best = (a, b), s
        if pair is None:
            break
        best_lab = np.where(best_lab == pair[1], pair[0], best_lab)
    _, best_lab = np.unique(best_lab, return_inverse=True)
    return (best_lab.astype(int), max(0.0, top)) if return_score else best_lab.astype(int)


# Long videos: speakers grouped by how alike their voices are. Picking the
# number of speakers by the cleanest split (cluster_speakers) fails there:
# men vs women is always the clearest split, so a 2-hour drama came out as
# 2 speakers (2,225 lines; silhouette 0.20 for 2 groups, 0.10–0.17 for 3–12)
# and 19–36 % of clear speaker changes were voiced as one person. Average-
# linkage grouping of the steady prints (lines ≥ 1.5 s) at 0.35 cosine missed
# none of them on two long dramas and split a speaker in 0–7 % of clear
# same-speaker pairs (10 and 18 speakers). Short clips have too few steady
# lines for it (2 of 7 speakers on a 2-minute scene): they keep the old rule.
_LONG_STEADY_S = 1.5        # a print this long is steady enough to group on
_LONG_MIN_STEADY = 60       # this many steady lines: use the grouping
_LONG_GROUP_SIM = 0.35      # average likeness that joins two groups
_LONG_MIN_LINES = 3         # smaller groups are noise: their lines go elsewhere
_LONG_MAX_SPEAKERS = 20


def group_long(X: np.ndarray, durations, sim: float = _LONG_GROUP_SIM,
               cap: int = _LONG_MAX_SPEAKERS,
               merge_above: float = 0.7) -> np.ndarray | None:
    """Speaker labels for L2-normalised prints of a long video, or None when
    it has too few steady lines (use cluster_speakers then). Groups the
    steady lines by average-linkage cosine likeness; every line then joins
    the group whose mean voice it is nearest. Groups whose mean voices are
    at least `merge_above` alike are one speaker, and at most `cap` stay."""
    dur = np.asarray(durations, dtype=float)
    steady = np.where(dur >= _LONG_STEADY_S)[0]
    if len(steady) < _LONG_MIN_STEADY or len(X) != len(dur):
        return None
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
    except Exception as exc:          # pragma: no cover - scipy ships with the app
        log.warning("scipy unavailable (%s); speakers grouped the old way", exc)
        return None
    lab = fcluster(linkage(X[steady], "average", metric="cosine"), 1.0 - sim, "distance")
    cents, weights = [], []
    for c in np.unique(lab):
        m = steady[lab == c]
        if len(m) >= _LONG_MIN_LINES:
            v = X[m].mean(axis=0)
            cents.append(v / max(float(np.linalg.norm(v)), 1e-9))
            weights.append(float(len(m)))
    if len(cents) < 2:
        return None
    # one speaker in two groups (calm and shouting), or more than `cap`:
    # join the two most alike
    while len(cents) > 1:
        M = np.vstack(cents)
        S = M @ M.T
        np.fill_diagonal(S, -2.0)
        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
        if S[a, b] < merge_above and len(cents) <= cap:
            break
        v = weights[a] * cents[a] + weights[b] * cents[b]
        cents[a] = v / max(float(np.linalg.norm(v)), 1e-9)
        weights[a] += weights[b]
        del cents[b], weights[b]
    M = np.vstack(cents)
    _, out = np.unique(np.argmax(X @ M.T, axis=1), return_inverse=True)
    return out.astype(int)


def _relabel_by_print(p: "DubProject", unit: dict[int, np.ndarray],
                      lab_of: dict[int, int]) -> int:
    """Move a line whose voice print is clearly another labelled speaker's
    to that speaker (`lab_of`: line → recogniser speaker, changed in place;
    the line becomes `voice_sure`). Returns how many lines moved. A
    speaker's voice is the mean print of its other lines of 1 s or more (at
    least two); a line of 0.8 s or more moves when it is at least 0.4 alike
    to another speaker's voice and 0.12 more alike than to its own: Deepgram
    filed the heroine's "听不懂" under the crowd (0.46 against 0.26) and two
    replies of a Qin drama under the wrong man; nothing moved on two English
    clips.

    A line that stays and is that clearly its own speaker's voice (at least
    0.4 alike, 0.12 more than to any other) is `voice_sure` too, as with
    the voice-print grouping: the AI speaker check moved 16 such lines of a
    drama's hero to the crowd's speaker (0.63-0.85 alike to their own)."""
    steady: dict[int, list[int]] = {}
    for i, lab in lab_of.items():
        if i in unit and p.lines[i].end - p.lines[i].start >= 1.0:
            steady.setdefault(lab, []).append(i)
    sums = {lab: np.sum([unit[i] for i in idx], axis=0) for lab, idx in steady.items()}
    members = {lab: set(idx) for lab, idx in steady.items()}

    def voice(lab: int, without: int) -> np.ndarray | None:
        inside = without in members.get(lab, ())
        if len(members.get(lab, ())) - inside < 2:
            return None
        v = sums[lab] - (unit[without] if inside else 0.0)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    moves: dict[int, int] = {}
    for i, lab in lab_of.items():
        ln = p.lines[i]
        own = voice(lab, i) if i in unit and ln.end - ln.start >= 0.8 else None
        if own is None:
            continue
        s_own = float(unit[i] @ own)
        best, s_best = None, -1.0
        for other in steady:
            v = voice(other, i) if other != lab else None
            if v is not None and float(unit[i] @ v) > s_best:
                best, s_best = other, float(unit[i] @ v)
        if best is not None and s_best >= 0.4 and s_best - s_own >= 0.12:
            moves[i] = best
            log.info("line %d (%.1f s, %r) moved to another speaker: its voice is %.2f alike "
                     "to theirs, %.2f to its own label's", i + 1, ln.start, ln.source[:20],
                     s_best, s_own)
        elif best is not None and s_own >= 0.4 and s_own - s_best >= 0.12:
            ln.voice_sure = True
    for i, lab in moves.items():
        lab_of[i] = lab
        p.lines[i].voice_sure = True
    return len(moves)


# Hz: a line's median pitch above this may be a man shouting as high as a
# woman (a drama's crowd and thug measured 270-390 Hz)
_CALM_F0 = 260.0


def _gender_split(X: np.ndarray, f0: np.ndarray, what: str, strict: bool = False
                  ) -> tuple[np.ndarray, int, dict[int, float]] | None:
    """The prints X split in two, named by pitch: (labels, the men's side,
    median pitch per side) when the two sides differ like men and women
    (else the split is between two men or two women: None).

    `strict` (calm lines only): a man's raised lines lift his side's median
    (148-153 against 194-202 Hz on one drama, so a 1.3 ratio passed or
    failed from run to run). Instead the men's side must have many lines at
    a man's pitch (30 % under 155 Hz) and the women's side almost none (at
    most 5 % under 135 Hz, 15 % under 155): a film's two scenes split
    152 / 198 Hz with 110 Hz men on the "women's" side."""
    lab, _ = _kmeans_cos(X, 2)
    f0s = {c: f0[(lab == c) & (f0 > 0)] for c in (0, 1)}
    if min(v.size for v in f0s.values()) < 2 or min(np.bincount(lab, minlength=2)) < 0.1 * len(X):
        return None
    med = {c: float(np.median(v)) for c, v in f0s.items()}
    lo, hi = sorted(med, key=lambda c: med[c])
    deep = {c: float(np.mean(f0s[c] < 155)) for c in (lo, hi)}
    very_deep = float(np.mean(f0s[hi] < 135))
    if strict:
        ok = (med[lo] < 165 and med[hi] > 185 and deep[lo] >= 0.3
              and very_deep <= 0.05 and deep[hi] <= 0.15)
    else:
        ok = med[lo] < 165 and med[hi] > 190 and med[hi] / med[lo] >= 1.3
    if not ok:
        log.info("%s split at %.0f / %.0f Hz (under 155 Hz: %.0f%% / %.0f%% of the lines, under "
                 "135 Hz on the higher side: %.0f%%): not a male/female split", what, med[lo],
                 med[hi], 100 * deep[lo], 100 * deep[hi], 100 * very_deep)
        return None
    return lab, lo, med


def _print_genders(p: "DubProject", X: np.ndarray, rows: list[int]) -> dict[int, str]:
    """{line: "M" | "F"} from the voice prints split in two, the side with
    the lower median pitch being the men — when the two sides clearly
    differ in pitch (else the split is between two men or two women, and
    {} is returned). Gender is the strongest direction in CAM++ space: on a
    drama scene this split misgendered 2 one-word replies of 60 lines,
    pitch line by line 14 (shouting men).

    A crowd shouting at 300 Hz and more can be the strongest direction
    instead (a drama split 164-167 / 318-323 Hz from run to run: the calm
    hero and heroine on one side, the shouting crowd and a thug on the
    other — passing the pitch test or not by 1 Hz). So the calm lines are
    also split alone, every loud line taking the side its voice is clearly
    nearer (the thug and the crowd: the men's); that split is used when the
    strongest one is not male/female, or when it is but calls the calm
    lines differently (it put the heroine with the men)."""
    if len(rows) < 6:
        return {}
    f0 = np.zeros(len(rows))
    for k, i in enumerate(rows):
        ln = p.lines[i]
        try:
            f0[k] = float(ap.voice_gender(_speech_source(p, ln), ln.start, ln.end)[2] or 0.0)
        except Exception:
            pass
    calm = np.where((f0 > 0) & (f0 <= _CALM_F0))[0]
    by_calm = _calm_genders(X, f0, rows, calm) if 6 <= len(calm) < len(rows) else None
    res = _gender_split(X, f0, "voice prints")
    if res is not None:
        lab, men, med = res
        full = {i: ("M" if int(lab[k]) == men else "F") for k, i in enumerate(rows)}
        both = [rows[k] for k in calm if rows[k] in (by_calm or {})]
        agree = (sum(full[i] == by_calm[i] for i in both) / len(both)) if both else 1.0
        if by_calm is None or agree >= 0.8:
            log.info("voice prints: men at %.0f Hz, women at %.0f Hz (median pitch)",
                     med[men], med[1 - men])
            return full
        log.info("voice prints split at %.0f / %.0f Hz name only %.0f%% of the calm lines as "
                 "their own split does: the calm lines' split is used", med[men], med[1 - men],
                 100 * agree)
    return by_calm or {}


def _calm_genders(X: np.ndarray, f0: np.ndarray, rows: list[int], calm: np.ndarray
                  ) -> dict[int, str] | None:
    """{line: "M" | "F"} from the split of the calm lines' prints (`calm`:
    indices into rows), each louder line by the side its print is ≥ 0.05
    nearer; None when that split is not male/female."""
    res = _gender_split(X[calm], f0[calm], "calm voice prints", strict=True)
    if res is None:
        return None
    lab, men, med = res
    cm, cf = (X[calm][lab == c].mean(axis=0) for c in (men, 1 - men))
    cm, cf = cm / max(float(np.linalg.norm(cm)), 1e-9), cf / max(float(np.linalg.norm(cf)), 1e-9)
    if float(cm @ cf) >= 0.7:            # one voice calm and raised, not two people
        return None
    # a line at a man's pitch is never named a woman's here
    out = {rows[k]: ("M" if int(lab[j]) == men else "F") for j, k in enumerate(calm)
           if int(lab[j]) == men or f0[k] >= 155}
    calm_set = set(calm.tolist())
    loud = [k for k in range(len(rows)) if k not in calm_set]
    for k in loud:
        d = float(X[k] @ cm - X[k] @ cf)
        if abs(d) >= 0.05:
            out[rows[k]] = "M" if d > 0 else "F"
    log.info("voice prints of calm lines: men at %.0f Hz, women at %.0f Hz; %d loud lines "
             "(over %.0f Hz) by likeness: %d men's, %d women's", med[men], med[1 - men],
             len(loud), _CALM_F0, sum(out.get(rows[k]) == "M" for k in loud),
             sum(out.get(rows[k]) == "F" for k in loud))
    return out


_ECAPA: Any = None
_ECAPA_LOCK = threading.Lock()


def _ecapa_embed(wav: str, spans: list[tuple[float, float]],
                 cancel: threading.Event | None) -> list[np.ndarray | None] | None:
    """Speaker embeddings via speechbrain ECAPA, or None if unavailable."""
    global _ECAPA
    if importlib.util.find_spec("speechbrain") is None:
        return None
    try:
        import torch  # lazy heavy import
        with _ECAPA_LOCK:
            if _ECAPA is None:
                try:
                    from speechbrain.inference.speaker import EncoderClassifier
                except ImportError:
                    from speechbrain.pretrained import EncoderClassifier
                kw: dict[str, Any] = {}
                try:
                    from speechbrain.utils.fetching import LocalStrategy
                    kw["local_strategy"] = LocalStrategy.COPY  # no symlinks on Windows
                except Exception:
                    pass
                dev = "cuda:0" if torch.cuda.is_available() else "cpu"
                _ECAPA = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(models_dir() / "spkrec-ecapa"),
                    run_opts={"device": dev}, **kw)
        out: list[np.ndarray | None] = []
        for s, e in spans:
            _check_cancel(cancel)
            x, sr = ap._read_wav_slice(wav, s, e, max_seconds=10.0)
            if x.size < int(0.5 * sr):
                out.append(None)
                continue
            t = torch.from_numpy(x)[None, :]
            if sr != 16000:
                # ECAPA is trained on 16 kHz; other rates (Demucs 44.1 kHz,
                # TTS 24 kHz) give embeddings that don't compare across files
                import torchaudio
                t = torchaudio.functional.resample(t, sr, 16000)
            with torch.no_grad():
                emb = _ECAPA.encode_batch(t)
            out.append(emb.squeeze().detach().cpu().numpy().astype(np.float64))
        return out
    except CancelledError:
        raise
    except Exception as exc:
        log.warning("speechbrain speaker embeddings unavailable (%s); using "
                    "built-in voice features", exc)
        return None


# ---------------------------------------------------------------------------
# voice conversion (OpenVoice tone colour) — makes any TTS voice sound like
# the character; language-agnostic, so Khmer works
# ---------------------------------------------------------------------------

_OV: dict[str, Any] = {}
_OV_LOCK = threading.Lock()


def openvoice_dir() -> Path:
    return models_dir() / _OPENVOICE_DIR_NAME / "converter"


def openvoice_installed() -> bool:
    return importlib.util.find_spec("openvoice") is not None


def openvoice_ready() -> bool:
    d = openvoice_dir()
    return (openvoice_installed()
            and all((d / n).is_file() for n in _OPENVOICE_FILES))


def download_openvoice(progress: ProgressFn | None = None,
                       cancel: threading.Event | None = None) -> str:
    """Download the OpenVoice V2 tone-colour converter checkpoint."""
    from app.core.model_manager import _download_url
    d = openvoice_dir()
    names = list(_OPENVOICE_FILES)
    for i, name in enumerate(names):
        _check_cancel(cancel)
        url, mb = _OPENVOICE_FILES[name]
        dest = d / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        _download_url(url, dest, _sub(progress, i / len(names),
                                      (i + 1) / len(names)), cancel, mb)
    _emit(progress, 1.0, "OpenVoice converter ready")
    return str(d)


def _ov_converter():
    with _OV_LOCK:
        if "tcc" in _OV:
            return _OV["tcc"]
        if not openvoice_installed():
            raise EngineNotAvailable(
                "OpenVoice is not installed (needed for 'edge+clone').",
                "OpenVoice មិនទាន់បានដំឡើងទេ (ត្រូវការសម្រាប់ 'edge+clone')។")
        d = openvoice_dir()
        if not all((d / n).is_file() for n in _OPENVOICE_FILES):
            raise EngineNotAvailable(
                f"OpenVoice converter checkpoint missing in {d}.",
                f"ខ្វះឯកសារម៉ូដែល OpenVoice នៅក្នុង {d}។")
        import torch  # lazy heavy import
        from openvoice.api import (OpenVoiceBaseClass,  # lazy heavy import
                                   ToneColorConverter)
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        # ToneColorConverter.__init__ forwards enable_watermark to a base
        # class that rejects it, and otherwise needs the wavmark package;
        # build it via the base class with the watermark turned off.
        tcc = ToneColorConverter.__new__(ToneColorConverter)
        OpenVoiceBaseClass.__init__(tcc, str(d / "config.json"), device=dev)
        tcc.watermark_model = None
        tcc.version = getattr(tcc.hps, "_version_", "v1")
        tcc.load_ckpt(str(d / "checkpoint.pth"))
        _OV["tcc"] = tcc
        return tcc


def _ov_embedding(wav: str):
    key = ("se", wav, os.path.getmtime(wav))
    with _OV_LOCK:
        if key in _OV:
            return _OV[key]
    tcc = _ov_converter()
    se = tcc.extract_se([wav])
    with _OV_LOCK:
        _OV[key] = se
    return se


def convert_voice(src_wav: str, src_sample_wav: str, ref_wav: str,
                  out_wav: str, cancel: threading.Event | None = None,
                  tau: float = 0.3) -> str:
    """Re-voice `src_wav` (spoken by the voice heard in `src_sample_wav`)
    with the timbre of `ref_wav`. Output: 24 kHz mono wav."""
    _check_cancel(cancel)
    try:
        tcc = _ov_converter()
        src_se = _ov_embedding(src_sample_wav)
        tgt_se = _ov_embedding(ref_wav)
        _check_cancel(cancel)
        tmp = out_wav + ".ov.wav"
        tcc.convert(audio_src_path=src_wav, src_se=src_se, tgt_se=tgt_se,
                    output_path=tmp, tau=tau)
        try:
            return ap._to_24k_mono_wav(tmp, out_wav, cancel=cancel)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    except (AppError, CancelledError):
        raise
    except Exception as exc:
        raise AppError(f"Voice conversion (OpenVoice) failed: {exc}",
                       f"ការបំលែងសំឡេង (OpenVoice) បរាជ័យ៖ {exc}") from exc


# ---------------------------------------------------------------------------
# voice conversion (Seed-VC) — zero-shot, language-agnostic, GPL-3 (optional
# install: pip install --no-deps seed-vc munch matplotlib)
# ---------------------------------------------------------------------------

_SVC: dict[str, Any] = {}
_SVC_LOCK = threading.Lock()
SEEDVC_SR = 22050


def seedvc_installed() -> bool:
    return _has("seed_vc")


def _svc_model():
    """Speech-only Seed-VC (22 kHz DiT + BigVGAN 22 kHz + CAMPPlus). The
    stock wrapper also loads the singing model, a 44 kHz vocoder and RMVPE,
    which speech conversion never uses."""
    with _SVC_LOCK:
        if "m" in _SVC:
            return _SVC["m"]
        if not seedvc_installed():
            raise EngineNotAvailable(
                "Seed-VC is not installed (needed for 'edge+seedvc').",
                "Seed-VC មិនទាន់បានដំឡើងទេ (ត្រូវការសម្រាប់ 'edge+seedvc')។")
        os.environ.setdefault("HF_HOME", str(models_dir() / "hf"))
        import torch  # lazy heavy import
        from huggingface_hub import hf_hub_download
        import seed_vc.hf_utils as hfu  # lazy heavy import
        import seed_vc.seed_vc_wrapper as w

        cache = str(models_dir() / "seed-vc")

        def load_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
            # upstream writes to "./checkpoints" in the current folder
            mp = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=cache)
            if config_filename is None:
                return mp
            return mp, hf_hub_download(repo_id=repo_id, filename=config_filename,
                                       cache_dir=cache)

        hfu.load_custom_model_from_hf = load_from_hf
        w.load_custom_model_from_hf = load_from_hf

        class _SpeechOnly(w.SeedVCWrapper):
            def _load_f0_model(self):
                self.model_f0 = None

            def _load_additional_modules(self):
                ck = load_from_hf("funasr/campplus", "campplus_cn_common.bin", None)
                self.campplus_model = w.CAMPPlus(feat_dim=80, embedding_size=192)
                self.campplus_model.load_state_dict(torch.load(ck, map_location="cpu"))
                self.campplus_model.eval().to(self.device)
                # called directly: newer huggingface_hub no longer passes the
                # proxies/resume_download arguments BigVGAN still requires
                self.bigvgan_model = w.bigvgan.BigVGAN._from_pretrained(
                    model_id="nvidia/bigvgan_v2_22khz_80band_256x", revision=None,
                    cache_dir=cache, force_download=False, proxies=None,
                    resume_download=False, local_files_only=False, token=None,
                    map_location="cpu", strict=False, use_cuda_kernel=False)
                self.bigvgan_model.remove_weight_norm()
                self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = _SpeechOnly(device=dev)
        _SVC["fp32"] = _svc_needs_fp32()
        if _SVC["fp32"]:
            m.whisper_model = m.whisper_model.float()
        _SVC["m"] = m
        return m


def _svc_needs_fp32() -> bool:
    """GTX 16-series cards produce NaN (silent output) in fp16; so do CPUs
    without half support. Setting tts.seedvc.fp32 forces it either way."""
    try:
        import torch  # lazy heavy import
        if not torch.cuda.is_available():
            return True
        name = torch.cuda.get_device_name(0)
        return "GTX 16" in name or "GTX16" in name.replace(" ", "")
    except Exception:
        return True


class _NoAutocast:
    """Stand-in for torch.autocast that keeps everything in fp32."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _svc_run(m, src_wav, ref_wav, diffusion_steps, cfg_rate, fp32: bool):
    import torch  # lazy heavy import
    real = torch.autocast
    if fp32:
        torch.autocast = _NoAutocast       # the wrapper hard-codes fp16 autocast
    try:
        gen = m.convert_voice(src_wav, ref_wav, diffusion_steps=diffusion_steps,
                              length_adjust=1.0, inference_cfg_rate=cfg_rate,
                              f0_condition=False, auto_f0_adjust=False,
                              pitch_shift=0, stream_output=False)
        try:            # convert_voice is a generator even without streaming
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value
    finally:
        torch.autocast = real


def convert_voice_seedvc(src_wav: str, ref_wav: str, out_wav: str,
                         cancel: threading.Event | None = None,
                         diffusion_steps: int = 10, cfg_rate: float = 0.7) -> str:
    """Re-voice `src_wav` with the timbre of `ref_wav` (Seed-VC, zero-shot).
    Output: 24 kHz mono wav, same length as the source."""
    _check_cancel(cancel)
    try:
        m = _svc_model()
        with _SVC_LOCK:
            audio = _svc_run(m, src_wav, ref_wav, diffusion_steps, cfg_rate, _SVC["fp32"])
            if audio is not None and np.isnan(np.asarray(audio)).any() and not _SVC["fp32"]:
                log.warning("Seed-VC produced NaN in fp16 on this GPU; switching to fp32")
                _SVC["fp32"] = True
                m.whisper_model = m.whisper_model.float()
                audio = _svc_run(m, src_wav, ref_wav, diffusion_steps, cfg_rate, True)
        if audio is None or not len(audio) or np.isnan(np.asarray(audio)).any():
            raise AppError("Seed-VC returned no usable audio", "Seed-VC មិនបានបង្កើតសំឡេងទេ")
        tmp = out_wav + ".svc.wav"
        a = np.asarray(audio, dtype=np.float32).flatten()
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        if peak > 0.89:                    # BigVGAN output can hit full scale
            a = a * (0.89 / peak)
        ap._write_wav_int16(tmp, a, SEEDVC_SR)
        try:
            return ap._to_24k_mono_wav(tmp, out_wav, cancel=cancel)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    except (AppError, CancelledError):
        raise
    except Exception as exc:
        raise AppError(f"Voice conversion (Seed-VC) failed: {exc}",
                       f"ការបំលែងសំឡេង (Seed-VC) បរាជ័យ៖ {exc}") from exc


# ---------------------------------------------------------------------------
# readiness ("what this needs to translate perfectly")
# ---------------------------------------------------------------------------

def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _vram_at_least(gb: float) -> bool:
    try:
        from app.core.gpu import primary_gpu
        g = primary_gpu()
        return bool(g and g.vendor == "nvidia" and (g.vram_gb or 0) >= gb)
    except Exception:
        return False


def best_clone_engine(lang: str) -> str:
    """The best installed cloning engine for a target language, or ""."""
    lang = _base_lang(lang)
    if (lang == "km" and seedvc_installed() and _has("edge_tts")
            and _vram_at_least(4.0)):
        # Khmer: Seed-VC sounded closest to the actors in a listening test
        # (Ep3). It peaks at ~3.1 GB in fp32; without such a GPU it would
        # crawl on the CPU, so OpenVoice stays the fallback below.
        return "edge+seedvc"
    for eng, mod in (("cosyvoice", "cosyvoice"), ("f5tts", "f5_tts"),
                     ("voxcpm", "voxcpm")):
        if lang in _CLONE_LANGS[eng] and _has(mod):
            if eng == "voxcpm" and not _vram_at_least(7.5):
                continue        # VoxCPM2 needs ~8 GB: pick it by hand to try
            return eng
    if openvoice_ready() and _has("edge_tts"):
        return "edge+clone"
    return ""


def readiness(config, options: DubOptions | None = None) -> list[Check]:
    """Everything the full-dub pipeline needs, with how to fix each gap."""
    from app.core import model_manager as mm
    opt = options or DubOptions()
    lang = _base_lang(opt.target_lang)
    out: list[Check] = []

    ff = False
    try:
        ff = bool(mm._ffmpeg_ok())
    except Exception:
        pass
    out.append(Check("ffmpeg", "FFmpeg (audio/video processing)", ff,
                     "found" if ff else "ffmpeg.exe not found",
                     "Put ffmpeg.exe + ffprobe.exe in the app's bin folder "
                     "or install FFmpeg on PATH."))

    deepgram = ap.stt_engine(config) == "deepgram"
    if deepgram:
        has_key = bool(config.get_secret("deepgram.api_key")) if config else False
        out.append(Check("deepgram", "Speech recognition (Deepgram, online)", has_key,
                         "API key saved" if has_key else "no API key",
                         "Settings → Speech services: paste your Deepgram key, "
                         "or switch speech-to-text back to Whisper."))
    fw = _has("faster_whisper")
    out.append(Check("whisper", "Speech recognition (faster-whisper)", fw,
                     "installed" if fw else "not installed",
                     pip_install("faster-whisper"), pip=[["faster-whisper"]],
                     required=not deepgram))
    present = False
    size = opt.whisper_size if opt.whisper_size not in ("", "auto") else _auto_whisper_size()
    try:
        present = mm._whisper_present(size)
    except Exception:
        pass
    out.append(Check("whisper_model", f"Whisper model '{size}'",
                     present,
                     "downloaded" if present else "will download on first run",
                     "Settings → Models, or it downloads automatically.",
                     required=False, download=f"whisper-{size}"))
    try:
        from app.core import gpu as G
        nvidia = any(g.vendor == "nvidia" for g in G.list_gpus())
        on_gpu = nvidia and G.ct2_cuda()
    except Exception:
        nvidia = on_gpu = False
    if nvidia and fw:
        # Whisper needs NVIDIA's cuBLAS to use the card (cuDNN comes with it)
        out.append(Check("whisper_gpu", "Speech recognition on the NVIDIA GPU", on_gpu,
                         "on — several times faster than the CPU" if on_gpu else
                         "runs on the CPU: NVIDIA's cuBLAS library is missing",
                         "Fix installs NVIDIA's cuBLAS (553 MB download): "
                         + pip_install(f"--no-deps {G.CUBLAS_PIP}"),
                         required=False, pip=[] if on_gpu else [["--no-deps", G.CUBLAS_PIP]]))

    from app.engines import mdx_separator
    dm, mdx_ok = _has("demucs"), mdx_separator.available()
    keep = opt.background != "none"
    out.append(Check("demucs", "Voice / music separation", dm or mdx_ok,
                     "Demucs installed" if dm else
                     "MDX-Net model ready" if mdx_ok else
                     ("not ready — the original voices would stay under the dub"
                      if keep else "not needed for 'voices only'"),
                     "Download the 'MDX-Net voice separator' model (about "
                     f"{mdx_separator.MODEL['size_mb']} MB), or {pip_install('demucs')}.",
                     required=keep and opt.use_demucs,
                     download="" if (dm or mdx_ok) else "mdx-vocals"))

    routes_ok, detail, keyed = True, [], set()
    try:
        keyed = set(config.providers_with_keys())
        backup = config._default_route() if keyed else None
        for feat in ("translate_draft", "translate_polish"):
            r = config.get_route(feat)
            ok = r.provider in keyed or (r.fallback_provider or "") in keyed
            routes_ok &= ok
            if ok:
                detail.append(f"{feat}: {r.provider}/{r.model}")
            elif backup is not None and backup.provider in keyed:
                # the AI engine falls back to any provider with a key: it
                # works, but the route should name a keyed provider
                detail.append(f"{feat}: {r.provider}/{r.model} has no API key — "
                              f"{backup.provider}/{backup.model} answers instead")
            else:
                detail.append(f"{feat}: {r.provider}/{r.model} (no API key)")
    except Exception as exc:
        routes_ok, detail = False, [str(exc)]
    # red (blocks the dub) only when no provider has a key at all; a route
    # without its own key is a note — a false "missing" alarm asked "Continue
    # anyway?" before every dub
    if not keyed and detail and all("(no API key)" in d for d in detail):
        detail = ["no AI key yet"]        # not the route names (translate_draft…)
    out.append(Check("ai", "AI translation provider + API key", routes_ok,
                     "; ".join(detail),
                     "Settings → Per-feature routing: pick a provider that has a key for "
                     "Draft translation / Polish" if keyed else
                     "Settings → AI Providers: add an AI key (for example an OpenRouter "
                     "key). Without one, “Transcribe only” still turns the speech into "
                     "text and finds the speakers.",
                     required=not keyed))

    et = _has("edge_tts")
    out.append(Check("edge_tts", "Edge TTS (Khmer & 300+ preset voices)", et,
                     "installed" if et else "not installed",
                     pip_install("edge-tts"), pip=[["edge-tts"]]))

    clone = best_clone_engine(lang)
    if lang in ("km", "th", "vi", "hi"):
        hint = ("Install OpenVoice and download its converter: Edge TTS "
                "speaks the language, OpenVoice gives each line the "
                "character's own voice.")
    else:
        hint = ("Install CosyVoice / F5-TTS (direct cloning), or OpenVoice "
                "for 'edge+clone'.")
    if clone == "edge+seedvc":
        extra = " (Seed-VC, ~5 s per line; set a character to 'edge+clone' for fast OpenVoice)"
    elif seedvc_installed():
        extra = " (Seed-VC also installed: pick 'edge+seedvc' per character; slower, GPL-3)"
    else:
        extra = ""
    out.append(Check("clone", "Voice cloning engine (per-character voices)",
                     bool(clone),
                     (f"using '{clone}'" + extra) if clone else "none — characters get "
                     "preset voices with pitch variation", hint,
                     required=False,
                     pip=[] if clone or openvoice_installed() else OPENVOICE_PIP,
                     download="" if openvoice_ready() or not openvoice_installed()
                     else "openvoice"))

    from app.engines import speaker_embed, subtitle_ocr
    ort_ok = _has("onnxruntime")
    cp = speaker_embed.model_present()
    out.append(Check("speakers", "Speaker recognition (CAM++ voice prints)", cp and ort_ok,
                     "ready" if cp and ort_ok else
                     "downloads on first run (28 MB)" if ort_ok else
                     "onnxruntime missing — built-in voice features (mix up similar voices)",
                     "Download the CAM++ model (28 MB), or it downloads on the first dub."
                     if ort_ok else pip_install("onnxruntime"),
                     required=False, download="" if cp else speaker_embed.MODEL_ID,
                     pip=[] if ort_ok else [["onnxruntime"]]))
    so_ok = subtitle_ocr.models_present()
    out.append(Check("subtitle_ocr", "On-screen subtitle reader (Chinese / Japanese dramas)",
                     so_ok and ort_ok,
                     "ready: hard subtitles correct misheard words" if so_ok and ort_ok else
                     "downloads on first run (21 MB)",
                     "Download the PP-OCRv5 subtitle reader (21 MB), or it downloads "
                     "on the first Chinese / Japanese dub.",
                     required=False, download="" if so_ok else subtitle_ocr.MODEL_ID))

    try:
        from app.core import gpu as G
        gpus = G.list_gpus()
        prim = G.primary_gpu(gpus)
        acc = G.accel_summary(gpus)
        hints = G.accel_hints(gpus)
    except Exception as exc:
        prim, acc, hints = None, {}, [str(exc)]
    torch_gpu = acc.get("torch") == "CUDA"
    if prim is None:
        detail = "no GPU found — everything runs on the CPU"
    else:
        detail = (f"{prim.label}: voice separation & cloning on "
                  f"{acc.get('torch', 'CPU')}, speech recognition on "
                  f"{acc.get('whisper', 'CPU')}")
    out.append(Check("gpu", "GPU acceleration", torch_gpu, detail,
                     " ".join(hints) or "These steps need an NVIDIA GPU (CUDA); "
                     "AMD/Intel GPUs speed up video export only.",
                     required=False))

    free = 0.0
    try:
        free = float(mm.disk_free_gb(projects_dir()))
    except Exception:
        pass
    out.append(Check("disk", "Free disk space (≥ 5 GB)", free >= 5.0,
                     f"{free:.1f} GB free", "Free up space on the data drive.",
                     required=False))
    return out


def install_packages(pip_args: list[list[str]],
                     progress: ProgressFn | None = None,
                     cancel: threading.Event | None = None) -> str:
    """Run `python -m pip install ...` for each arg list, streaming output."""
    if getattr(sys, "frozen", False):
        # sys.executable is RecapStudio.exe here: "-m pip" would just open
        # another copy of the app
        raise EngineNotAvailable(
            "This packaged build can't install Python packages. Use the "
            "source version (python main.py) for voice cloning / Demucs.",
            "កំណែកម្មវិធីនេះមិនអាចដំឡើងកញ្ចប់ Python បានទេ។ សូមប្រើកំណែកូដប្រភព "
            "(python main.py) សម្រាប់ការចម្លងសំឡេង / Demucs។")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    n = max(1, len(pip_args))
    for i, args in enumerate(pip_args):
        _check_cancel(cancel)
        _emit(progress, i / n, "pip install " + " ".join(args))
        cmd = [sys.executable, "-m", "pip", "install",
               "--disable-pip-version-check"] + list(args)
        rc = ap._stream_subprocess(cmd, label="pip", env=env, cancel=cancel)
        if rc != 0:
            raise AppError(f"pip install failed (code {rc}): {' '.join(args)}",
                           f"pip install បរាជ័យ (កូដ {rc})៖ {' '.join(args)}")
    importlib.invalidate_caches()
    _emit(progress, 1.0, "Installed")
    return "ok"


def _project_video(pj: Path) -> str:
    """The video a project.json belongs to: "video" is near its top, so
    only the first lines are read (every project is looked at)."""
    try:
        with open(pj, "r", encoding="utf-8") as f:
            head = f.read(8192)
        m = re.search(r'"video":\s*("(?:[^"\\]|\\.)*")', head)
        if m:
            return str(json.loads(m.group(1)))
    except Exception:
        pass
    return ""


def _dub_projects() -> list[Path]:
    root = projects_dir() / "dub"
    return sorted(root.glob("*/project.json")) if root.is_dir() else []


def _project_dir(video: str) -> Path:
    """The folder a new project of `video` gets (name + hash of the path)."""
    video = os.path.abspath(video)
    return projects_dir() / "dub" / f"{_safe_name(Path(video).stem)}_{_sha1(video.lower())[:8]}"


def project_json_for(video: str) -> str:
    """The project.json of `video`'s dub, "" = none yet: in the folder named
    after its path, or a project relinked to it (find_moved_project)."""
    pj = _project_dir(video) / "project.json"
    return str(pj) if pj.is_file() else _linked_project(video)


def _linked_project(video: str) -> str:
    """The project.json that names `video` although its folder is named
    after another path (a project relinked to a moved video); "" = none."""
    want = os.path.normcase(os.path.abspath(video))
    for pj in _dub_projects():
        v = _project_video(pj)
        if v and os.path.normcase(os.path.abspath(v)) == want:
            return str(pj)
    return ""


def find_moved_project(video: str) -> str:
    """An earlier dub of `video` made before the video was moved: a project
    whose video had the same file name, is no longer where it was, and
    lasted as long (±1 s, when known). The canonical folder, then the most
    tasks done, then the newest. Returns its project.json, or "" = none."""
    video = os.path.abspath(video)
    name = os.path.basename(video).lower()
    length: float | None = None
    best: tuple[tuple, str] | None = None
    for pj in _dub_projects():
        old = _project_video(pj)
        if not old or os.path.basename(old).lower() != name:
            continue
        if os.path.normcase(os.path.abspath(old)) == os.path.normcase(video) \
                or os.path.isfile(old):
            continue                 # its own project, or that video is still there
        try:
            with open(pj, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        was = float(d.get("duration") or 0.0)
        if was > 0:
            if length is None:
                try:
                    length = float(ap.get_duration(video))
                except Exception:
                    length = 0.0
            if length and abs(length - was) > 1.0:
                continue             # same name, another video
        steps = d.get("steps") if isinstance(d.get("steps"), dict) else {}
        done = sum(1 for s in steps.values()
                   if isinstance(s, dict) and s.get("status") in ("done", "warn"))
        canonical = pj.parent.name == _project_dir(old).name
        key = (canonical, done, pj.stat().st_mtime)
        if best is None or key > best[0]:
            best = (key, str(pj))
    return best[1] if best else ""


def ai_pause_error(step: str, exc: BaseException) -> AppError:
    """The message for a task stopped because no AI could answer: the cause
    in plain words and what to do about it (no key, refused key, no credit,
    rate limit, no answer, no connection), that the work so far is saved,
    and Continue. `.ai_kind` names the cause (ai_failure_kind)."""
    kind = ai_failure_kind(exc) or "other"
    plain = bool(getattr(exc, "plain", False))
    # a message that already says what to do ("Your OpenRouter account is
    # out of credit…") is used as it is, instead of the general advice
    adv_en, adv_km = ((exc.en, exc.km) if plain
                      else AI_ADVICE.get(kind, AI_ADVICE[""]))
    # "Continue" is said by the panel under it, once ("…Continue tries
    # again from this task; Cancel stops here.")
    en = f"{dict(STEPS).get(step, step)} paused: {adv_en} Everything finished so far is saved."
    km = f"កិច្ចការនេះបានផ្អាក៖ {adv_km} ការងារដែលធ្វើរួចត្រូវបានរក្សាទុក។"
    if kind != "no_key" and not plain:    # the service's own words, for the curious
        detail_en = getattr(exc, "en", "") or str(exc)
        en += f" (Details: {detail_en})"
        km += f" (ព័ត៌មានលម្អិត៖ {getattr(exc, 'km', '') or detail_en})"
    err = AppError(en, km)
    err.ai_kind = kind
    return err


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

class DubEngine:
    def __init__(self, ai, config=None):
        self.ai = ai
        self.config = config if config is not None else getattr(ai, "config", None)
        self.scene_notes: dict[int, str] = {}     # line → meaning note (translate step)
        self.meaning_refused: set[int] = set()
        self._late_room: dict[int, float] = {}    # line → room left after earlier overruns

    # -- orchestration ------------------------------------------------------
    def run_step(self, project: DubProject, step: str,
                 progress: ProgressFn | None = None,
                 cancel: threading.Event | None = None) -> DubProject:
        fn = getattr(self, f"_step_{step}", None)
        if fn is None:
            raise AppError(f"Unknown dub step: {step}",
                           f"មិនស្គាល់ជំហាន៖ {step}")
        idx = STEP_IDS.index(step)
        for prev in STEP_IDS[:idx]:
            if not project.step_done(prev):
                raise AppError(
                    f"Run the earlier step first: {dict(STEPS)[prev]}",
                    f"សូមដំណើរការជំហានមុនសិន៖ {dict(STEPS)[prev]}")
        project.set_step(step, "running")
        t_start = time.monotonic()
        try:
            result = fn(project, progress, cancel)
        except CancelledError:
            project.set_step(step, "pending", "cancelled")
            project.save()
            raise
        except AppError as exc:
            if ai_unavailable(exc) and getattr(exc, "provider", "") == "budget":
                # the app's own monthly limit, not the AI service
                err = AppError(
                    f"{dict(STEPS)[step]} paused: {exc.en} Everything finished so far "
                    "is saved.",
                    f"កិច្ចការនេះបានផ្អាក៖ {exc.km} ការងារដែលធ្វើរួចត្រូវបានរក្សាទុក។")
                err.ai_kind = "budget"
                self._failed(project, step, err)
                raise err from exc
            if ai_unavailable(exc):
                # no key / bad key / no credit / rate limit / no connection:
                # nothing is lost — finished batches are in tr_cache and a
                # later run goes on. The message says which, and what to do.
                err = ai_pause_error(step, exc)
                self._failed(project, step, err)
                raise err from exc
            self._failed(project, step, exc)
            raise
        except Exception as exc:
            project.set_step(step, "failed", str(exc))
            project.save()
            raise AppError(f"{dict(STEPS)[step]} failed: {exc}",
                           f"ជំហាន {step} បរាជ័យ៖ {exc}") from exc
        status, note = result if isinstance(result, tuple) else ("done", str(result or ""))
        # everything after a re-run step is stale now
        for later in STEP_IDS[idx + 1:]:
            if project.steps[later]["status"] != "pending":
                project.steps[later] = {"status": "pending", "note": ""}
        project.set_step(step, status, note)
        # how long it took: the Dub tab learns each task's pace from this
        # (seconds per minute of video) for its "time left"
        project.steps[step]["secs"] = round(time.monotonic() - t_start, 1)
        project.save()
        _emit(progress, 1.0, note or "Done")
        return project

    @staticmethod
    def _failed(project: DubProject, step: str, err: AppError) -> None:
        """Mark `step` failed and save. The note keeps the English text, and
        "note_km" the Khmer one (a reopened project shows it in Khmer);
        "ai" names why no AI answered, for a pause (the Dub tab's advice)."""
        project.set_step(step, "failed", err.en)
        if err.km and err.km != err.en:
            project.steps[step]["note_km"] = err.km
        if getattr(err, "ai_kind", ""):
            project.steps[step]["ai"] = err.ai_kind
        project.save()

    def run_all(self, project: DubProject,
                progress: ProgressFn | None = None,
                cancel: threading.Event | None = None,
                on_step: Callable[[str], None] | None = None,
                stop_before: str = "") -> DubProject:
        """Run every step that is not done yet, in order. `stop_before`: stop
        before that step (e.g. "glossary": the tasks that need no AI key)."""
        todo = [s for s in STEP_IDS if not project.step_done(s)]
        if stop_before in STEP_IDS:
            todo = [s for s in todo if STEP_IDS.index(s) < STEP_IDS.index(stop_before)]
        if not todo:
            _emit(progress, 1.0, "Everything is already done")
            return project
        weights = STEP_WEIGHTS
        total = float(sum(weights[s] for s in todo))
        done = 0.0
        for s in todo:
            _check_cancel(cancel)
            if on_step is not None:
                try:
                    on_step(s)
                except Exception:
                    pass
            lo, hi = done / total, (done + weights[s]) / total
            label = dict(STEPS)[s]
            self.run_step(project, s,
                          _sub(progress and (lambda f, t, _l=label:
                                             progress(f, f"{_l}: {t}")), lo, hi),
                          cancel)
            done += weights[s]
        _emit(progress, 1.0, "Dub complete" if project.next_step() is None else "Done")
        return project

    def run_until_ai(self, project: DubProject,
                     progress: ProgressFn | None = None,
                     cancel: threading.Event | None = None) -> DubProject:
        """The tasks that need no AI key: extract the audio, transcribe,
        find the speakers ("Transcribe only" when no key is set yet)."""
        return self.run_all(project, progress, cancel, stop_before=FIRST_AI_STEP)

    # -- step 1: prepare ----------------------------------------------------
    def _step_prepare(self, p: DubProject, progress, cancel):
        if not os.path.isfile(p.video):
            raise AppError(f"Video not found: {p.video}",
                           f"រកមិនឃើញវីដេអូ៖ {p.video}")
        p.duration = ap.get_duration(p.video)
        p.audio_wav = p.sub("audio", "original.wav")
        # 6 dB headroom: loud dramas decode above full scale and 16-bit
        # would clip them; the final mix is loudness-normalised anyway
        ap.extract_audio(p.video, p.audio_wav, sr=44100, mono=False,
                         progress=_sub(progress, 0.0, 0.15), cancel=cancel,
                         gain_db=-6.0)
        status, note = "done", ""
        p.vocals_wav, p.music_wav = p.audio_wav, ""
        if p.options.background == "none":
            status, note = "done", "voices only: no music separation needed"
        elif p.options.use_demucs:
            try:
                sep = p.sub("audio", "sep", "x")
                shutil.rmtree(os.path.dirname(sep), ignore_errors=True)
                p.vocals_wav, p.music_wav = ap.separate_vocals(
                    p.audio_wav, os.path.dirname(sep),
                    progress=_sub(progress, 0.15, 0.9), cancel=cancel)
            except CancelledError:
                raise
            except AppError as exc:
                status, note = "warn", f"Voice separation unavailable ({exc.en}); " \
                    "the original voices will remain under the dub"
                log.warning(note)
        else:
            status, note = "warn", "Vocal separation turned off"
        p.asr_wav = p.sub("audio", "vocals_16k.wav")
        ap._run_ffmpeg(["-i", p.vocals_wav, "-ac", "1", "-ar", "16000",
                        "-acodec", "pcm_s16le", p.asr_wav],
                       duration=p.duration, progress=_sub(progress, 0.9, 1.0),
                       cancel=cancel, status_text="Preparing speech track…")
        return status, f"{p.duration / 60:.1f} min audio, " + (
            note or "voices and music separated")

    # -- step 2: transcribe -------------------------------------------------
    def _step_transcribe(self, p: DubProject, progress, cancel):
        lang = _base_lang(p.options.source_lang)
        size = p.options.whisper_size
        if not size or size == "auto":
            size = _auto_whisper_size()
        use_deepgram = ap.stt_engine(self.config) == "deepgram"
        whisper = None
        if not lang:
            # the language first, on this PC: left to Deepgram, a Mandarin
            # clip was taken for English (then Korean) and nothing was heard
            det_size = _installed_whisper_size() if use_deepgram else size
            if det_size:
                _emit(progress, 0.01, "Detecting the spoken language")
                try:
                    whisper = ap.WhisperEngine(model_size=det_size)
                    lang, prob = whisper.detect_language(p.asr_wav)
                    log.info("detected language %s (%.0f%%)", lang, prob * 100)
                    LOG_BUS.publish(f"[dub] spoken language: {lang} ({prob:.0%})")
                except CancelledError:
                    raise
                except Exception as exc:
                    log.warning("language detection failed (%s); the recogniser decides", exc)
                    lang, whisper = "", None
        if use_deepgram:
            from app.engines import deepgram_stt
            if deepgram_stt.supports(lang):
                note = self._transcribe_deepgram(p, lang, progress, cancel)
                if note:
                    return note
                LOG_BUS.publish("[deepgram] heard no speech in this video — trying Whisper")
            else:
                LOG_BUS.publish(f"[deepgram] cannot transcribe '{lang}' — using Whisper")
        if whisper is None or whisper.model_size != size:
            whisper = ap.WhisperEngine(model_size=size)
        names = []
        if p.options.series.strip():
            from app.core.series_memory import SeriesMemory
            mem = SeriesMemory.load(p.options.series)
            names = [t["source"] for g in mem.glossary.values() for t in g][:40]
            names += [c.name for c in mem.cast if c.name][:20]
        prompt = _WHISPER_PROMPTS.get(lang, "")
        if names:
            prompt = (prompt + " " + "、".join(dict.fromkeys(names))).strip()
        tr = whisper.transcribe(p.asr_wav, language=lang or None,
                                word_timestamps=True, progress=_sub(progress, 0.0, 0.9),
                                cancel=cancel, initial_prompt=prompt or None,
                                hotwords=" ".join(dict.fromkeys(names)) or None)
        with open(p.sub("transcript.json"), "w", encoding="utf-8") as f:
            json.dump(tr.to_json(), f, ensure_ascii=False)
        p.src_lang = tr.language or p.options.source_lang
        if tr.duration and not p.duration:
            p.duration = tr.duration
        segs, sub_note, events = self._apply_subtitles(p, tr.segments, progress, cancel)
        p.lines = drop_prompt_echo(split_segments(segs), prompt)
        _mark_heard(p.lines, tr.segments, events)
        if not p.lines:
            raise AppError(
                f"No speech was found in this video (Whisper {size}, language "
                f"{lang or 'auto'}). If it does have dialogue, set its language or "
                "another Whisper size in the options and try again.",
                f"រកមិនឃើញការនិយាយក្នុងវីដេអូនេះទេ (Whisper {size}, ភាសា {lang or 'auto'})។ "
                "បើវាមានសន្ទនាមែន សូមកំណត់ភាសា ឬទំហំ Whisper ផ្សេង ហើយសាកម្ដងទៀត។")
        p.characters = []
        return self._transcribe_result(
            p, tr, f"{len(p.lines)} lines, language: {p.src_lang or '?'}, "
                   f"Whisper {size}" + sub_note)

    @staticmethod
    def _transcribe_result(p: DubProject, tr, note: str):
        """The transcribe step's (status, note): what the second listen
        recovered (marked for review in Lines), the lines the recogniser
        doubted, and the voiced stretches still without words — with their
        times, so a lost sentence is reported, never silently gone."""
        def n(k: int, one: str, many: str) -> str:
            return f"{k} {one if k == 1 else many}"
        parts = []
        if tr.recovered:
            parts.append(n(tr.recovered, "missed line", "missed lines")
                         + " recovered by a second listen (marked ⚑ in Lines: check them)")
        doubt = sum(1 for ln in p.lines if ln.review in ("low_confidence", "maybe_prompt"))
        if doubt:
            parts.append(n(doubt, "line", "lines") + " the recogniser was unsure of (marked ⚑)")
        if tr.unheard:
            at = ", ".join(_mmss(u[0]) for u in tr.unheard[:8])
            parts.append(n(len(tr.unheard), "short stretch of voice", "short stretches of voice")
                         + f" had no recognisable words (at {at}"
                         + f"{', …' if len(tr.unheard) > 8 else ''})")
            log.info("voice without recognisable words: %s", "; ".join(
                f"{_mmss(u[0])}-{_mmss(u[1])} {u[2]} {u[3]!r}" for u in tr.unheard))
        return ("warn" if tr.unheard or tr.recovered or doubt else "done",
                note + "".join(f"; {x}" for x in parts))

    def _apply_subtitles(self, p: DubProject, segments: list, progress, cancel
                         ) -> tuple[list, str, list]:
        """Correct the recognised words from the video's hard subtitles
        (Chinese / Japanese dramas): returns (segments, note for the step,
        subtitles read). The recogniser's timing stays; a subtitle nobody
        was heard saying becomes a line when a voice is there."""
        from app.engines import subtitle_ocr as so
        base = _base_lang(p.src_lang)
        if base not in so.LANGS or not p.video or not os.path.isfile(p.video):
            return segments, "", []
        try:
            if not so.models_present():
                _emit(progress, 0.9, "Downloading the subtitle reader (21 MB)")
                from app.core import model_manager as mm
                mm.download_model(so.MODEL_ID, progress=_sub(progress, 0.9, 0.93), cancel=cancel)
            _emit(progress, 0.93, "Reading on-screen subtitles")
            events = so.cached_read(p.video, p.sub("subtitles.json"),
                                    _sub(progress, 0.93, 0.99), cancel,
                                    duration=p.duration)
        except CancelledError:
            raise
        except Exception as exc:
            log.warning("on-screen subtitles not read (%s); transcript as heard",
                        getattr(exc, "en", exc))
            return segments, "", []
        if not events:
            return segments, "; no on-screen subtitles", []
        quiet = quiet_spans(p.asr_wav) if p.asr_wav and os.path.isfile(p.asr_wav) else []

        def voiced(s: float, e: float) -> float:
            if e <= s:
                return 0.0
            q = sum(max(0.0, min(e, b) - max(s, a)) for a, b in quiet)
            return 1.0 - q / (e - s)

        segs, st = so.apply_to_segments(segments, events, base, voiced=voiced)
        if not st.used:
            return segments, (f"; on-screen subtitles do not match the speech "
                              f"({st.coverage:.0%}), not used"), events
        LOG_BUS.publish(f"[dub] on-screen subtitles: {len(events)} read, "
                        f"{st.changed} words corrected"
                        + (f" ({', '.join(st.examples)})" if st.examples else "")
                        + (f", {st.added} missed words added" if st.added else ""))
        return segs, (f"; words checked against {len(events)} on-screen subtitles: "
                      f"{st.changed} corrected, {st.added} missed ones added"), events

    def _transcribe_deepgram(self, p: DubProject, lang: str, progress, cancel):
        """Deepgram (online): transcript plus who speaks each line. Returns
        the step's (status, note), or "" when Deepgram heard nothing (the
        caller then tries Whisper)."""
        from app.engines import deepgram_stt
        key = self.config.get_secret("deepgram.api_key") if self.config else ""
        tr = deepgram_stt.transcribe(p.asr_wav, key, language=lang, diarize=True,
                                     progress=_sub(progress, 0.0, 0.8), cancel=cancel)
        if not any(str(w.text).strip() for s in tr.segments for w in (s.words or [])) \
                and not any(str(s.text).strip() for s in tr.segments):
            return ""
        tr = self._second_listen(p, tr, lang or tr.language, _sub(progress, 0.8, 0.9), cancel)
        with open(p.sub("transcript.json"), "w", encoding="utf-8") as f:
            json.dump(tr.to_json(), f, ensure_ascii=False)
        p.src_lang = tr.language or p.options.source_lang
        if tr.duration and not p.duration:
            p.duration = tr.duration
        base = _base_lang(p.src_lang)
        cjk = base in ("zh", "ja")
        segs, sub_note, events = self._apply_subtitles(p, tr.segments, progress, cancel)
        p.lines = split_segments(segs, lang=base,
                                 quiet=quiet_spans(p.asr_wav) if cjk else None,
                                 dips=quiet_spans(p.asr_wav, 0.05, 18.0) if cjk else ())
        _mark_heard(p.lines, tr.segments, events)
        if not p.lines:
            return ""
        p.characters = []
        n_spk = len({l.asr_speaker for l in p.lines if l.asr_speaker})
        return self._transcribe_result(
            p, tr, f"{len(p.lines)} lines, language: {p.src_lang or '?'}, "
                   f"Deepgram ({n_spk} speakers)" + sub_note)

    def _second_listen(self, p: DubProject, tr, lang: str, progress, cancel):
        """Whisper listens again to the voiced stretches Deepgram left without
        words (ap.recover_gaps) — only with a Whisper model already on this PC
        (nothing is downloaded for it). In 20 min of a Chinese drama Deepgram
        missed 22 subtitle lines; Whisper, listening to each gap alone, heard
        4 of them. Any failure keeps Deepgram's transcript as it is."""
        size = _recovery_whisper_size()
        if not size:
            log.info("second listen skipped: no Whisper model on this PC")
            return tr
        try:
            _emit(progress, 0.0, f"Listening again to what Deepgram missed (Whisper {size})")
            return ap.WhisperEngine(model_size=size).recover(
                p.asr_wav, tr, language=lang or None, progress=progress, cancel=cancel)
        except CancelledError:
            raise
        except Exception as exc:
            log.warning("second listen failed (%s); Deepgram's transcript kept",
                        getattr(exc, "en", exc))
            return tr

    # -- step 3: speakers ---------------------------------------------------
    def _campplus(self, p: DubProject, spans: list[tuple[float, float]], progress,
                  cancel) -> list[np.ndarray | None] | None:
        """CAM++ voice prints (downloaded once, 28 MB), or None."""
        from app.engines import speaker_embed as se
        try:
            if not se.model_present():
                _emit(progress, 0.02, "Downloading the speaker recogniser (28 MB)")
                from app.core import model_manager as mm
                mm.download_model(se.MODEL_ID, progress=_sub(progress, 0.02, 0.05),
                                  cancel=cancel)
            _emit(progress, 0.05, "Measuring voices")
            return se.embed_spans(p.asr_wav, spans, cancel)
        except CancelledError:
            raise
        except Exception as exc:
            log.warning("CAM++ speaker recognition unavailable (%s)", getattr(exc, "en", exc))
            return None

    def _step_speakers(self, p: DubProject, progress, cancel):
        spans = []
        for i, ln in enumerate(p.lines):
            s, e = ln.start, ln.end
            if e - s < 1.0:
                # widen a short line a little for a steadier print — into the
                # silence around it only: a reply ("对") widened into the next
                # sentence took on that speaker's voice
                lo = p.lines[i - 1].end if i > 0 else 0.0
                hi = p.lines[i + 1].start if i + 1 < len(p.lines) else e + 0.25
                s, e = max(0.0, max(lo, s - 0.25)), max(e, min(hi, e + 0.25))
            spans.append((s, e))
        _emit(progress, 0.02, "Measuring voices")
        feats = self._campplus(p, spans, progress, cancel)
        method, emb_kind = "CAM++ voice prints", "campplus"
        if feats is None or sum(f is not None for f in feats) < 0.5 * len(feats):
            feats = _ecapa_embed(p.asr_wav, spans, cancel)
            method, emb_kind = "speechbrain ECAPA", "ecapa"
        if feats is None:
            method, emb_kind = "built-in voice features", "mfcc"
            feats = []
            for i, (s, e) in enumerate(spans):
                if i % 20 == 0:
                    _check_cancel(cancel)
                    _emit(progress, 0.05 + 0.6 * i / max(1, len(spans)),
                          f"Measuring voices {i + 1}/{len(spans)}")
                x, sr = ap._read_wav_slice(p.asr_wav, s, e, max_seconds=8.0)
                feats.append(voice_features(x, sr))
        neural = emb_kind in ("ecapa", "campplus")
        for ln in p.lines:
            ln.voice_sure = False
        if neural:
            # speaker embeddings are compared by plain cosine (no z-scoring)
            rows = [i for i, f in enumerate(feats) if f is not None]
            X = (np.vstack([feats[i] for i in rows]) if rows else np.zeros((0, 1)))
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
            unit = {i: X[k] for k, i in enumerate(rows)}
            for i, ln in enumerate(p.lines):
                a, b = unit.get(i - 1), unit.get(i)
                ln.voice_sim = round(float(a @ b), 3) if a is not None and b is not None else -1.0
            pg = _print_genders(p, X, rows) if emb_kind == "campplus" else {}
            for i, ln in enumerate(p.lines):
                ln.print_gender = pg.get(i, "")
        else:
            X, rows = _normalize_features(feats)
            for ln in p.lines:
                ln.voice_sim, ln.print_gender = -1.0, ""
        asr = [ln.asr_speaker for ln in p.lines]
        use_asr = (not p.options.num_speakers and p.lines
                   and sum(1 for a in asr if a) >= 0.9 * len(p.lines))
        if use_asr:
            # Deepgram already told who speaks each line
            _emit(progress, 0.7, "Using the speaker labels from Deepgram")
            ids_asr = {a: k for k, a in enumerate(dict.fromkeys(a for a in asr if a))}
            lab_of = {i: ids_asr[a] for i, a in enumerate(asr) if a}
            moved = _relabel_by_print(p, unit, lab_of) if emb_kind == "campplus" else 0
            rows = sorted(lab_of)
            p.speaker_score = 1.0
            method = "Deepgram speaker labels" + (
                f", {moved} line{'s' if moved != 1 else ''} moved by voice print" if moved else "")
        else:
            _emit(progress, 0.7, "Grouping voices into characters")
            # CAM++ tells more voices apart: a 2-minute drama scene had 7
            # speakers, and capped at 6 the best split fell to 3 (the hero,
            # a delivery man and a guard as one person)
            max_k = int(p.options.max_speakers or 6)
            if emb_kind == "campplus":
                max_k = max(max_k, 10)
            labels = None
            if emb_kind == "campplus" and not p.options.num_speakers:
                labels = group_long(X, [p.lines[r].end - p.lines[r].start for r in rows])
                if labels is not None:
                    method += " (long video: grouped by voice likeness)"
                    score = (_silhouette_cos(X, labels)
                             if len(set(labels.tolist())) > 1 else 0.0)
            if labels is None:
                labels, score = cluster_speakers(
                    X, num=int(p.options.num_speakers or 0), max_k=max_k,
                    return_score=True,
                    merge_above=0.7 if emb_kind == "campplus" else None)
            p.speaker_score = round(float(score), 3)
            lab_of = {r: int(l) for r, l in zip(rows, labels)}
            if emb_kind == "campplus" and len(set(labels.tolist())) > 1:
                # a line whose print sits clearly nearer its own group than
                # any other is that speaker: the dialogue check later moved
                # a whole scene of men's lines to a woman
                labs = np.asarray(labels)
                C = {c: X[labs == c].mean(axis=0) for c in set(labs.tolist())}
                C = {c: v / max(float(np.linalg.norm(v)), 1e-9) for c, v in C.items()}
                for k, r in enumerate(rows):
                    own = float(X[k] @ C[int(labs[k])])
                    other = max(float(X[k] @ v) for c, v in C.items() if c != int(labs[k]))
                    ln = p.lines[r]
                    ln.voice_sure = own - other >= 0.12 and ln.end - ln.start >= 0.8
        # lines without a usable print inherit their neighbour's speaker
        prev = lab_of[rows[0]] if rows else 0
        assigned = []
        for i in range(len(p.lines)):
            prev = lab_of.get(i, prev)
            assigned.append(prev)
        # S1 = most talk time
        talk: dict[int, float] = {}
        for ln, l in zip(p.lines, assigned):
            talk[l] = talk.get(l, 0.0) + (ln.end - ln.start)
        order = sorted(talk, key=lambda l: -talk[l])
        ids = {l: f"S{i + 1}" for i, l in enumerate(order)}
        old = {c.id: c for c in p.characters}
        p.characters = []
        for l in order:
            cid = ids[l]
            ch = old.get(cid) or Character(id=cid, name=f"Speaker {cid[1:]}")
            p.characters.append(ch)
        for ln, l in zip(p.lines, assigned):
            ln.speaker = ids[l]
        p.recount()
        flagged = self._detect_genders(p, progress, cancel)
        # voice fingerprint per character = mean of its lines' raw vectors
        for ch in p.characters:
            vecs = [np.asarray(feats[i], dtype=np.float64)
                    for i, ln in enumerate(p.lines)
                    if ln.speaker == ch.id and feats[i] is not None]
            emb: list[float] = []
            if vecs:
                # lines without a measurable pitch carry NaN there: average
                # around the gaps instead of dropping the whole line
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m = np.nanmean(np.vstack(vecs), axis=0)
                emb = [float(v) if np.isfinite(v) else 0.0 for v in m]
            ch.embedding = emb
            ch.embed_method = emb_kind
        for ch in p.characters:
            ch.gender_source = ch.gender_source or "voice"
        note = f"{len(p.characters)} characters ({method}, separation {p.speaker_score:.2f})"
        talk = sorted(((ch.label(), sum(1 for ln in p.lines if ln.speaker == ch.id))
                       for ch in p.characters), key=lambda t: -t[1])
        log.info("speakers: %d found by %s over %d lines, separation %.2f; lines each: %s",
                 len(p.characters), method, len(p.lines), p.speaker_score,
                 ", ".join(f"{name} {n}" for name, n in talk[:20]))
        known = self._apply_series_cast(p)
        if known:
            note += f"; recognised from the series: {', '.join(known)}"
        unsure = [c.label() for c in p.characters
                  if c.gender_source == "voice" and c.gender_conf < GENDER_SURE]
        if flagged:
            note += f"; {flagged} line(s) sound like the other gender (voiced so)"
        if unsure:
            note += f"; check gender of: {', '.join(unsure)}"
            return "warn", note
        return note

    def _detect_genders(self, p: DubProject, progress, cancel,
                        keep_characters: bool = False,
                        ai_check: bool = False,
                        only: set[str] | None = None) -> int:
        """Measure every line's voice, give each character the gender most
        of its talk time sounds like (with a confidence), and voice lines
        with the other gender when a character mixes two people (several
        lines clearly the other gender). Hand-set genders are kept; with
        `keep_characters` the characters' genders are not touched at all,
        only the lines are re-measured and re-decided against them. With
        `ai_check` a switch also needs the dialogue not to say otherwise: a
        man shouting or upset can measure as high as a woman; with both, a
        character the story gave one gender but who mostly sounds like the
        other is checked first (_ai_gender_conflicts). With `only`
        (character ids) just those characters' lines are re-decided, from
        the voices measured before.
        Returns how many lines were switched."""
        n = max(1, len(p.lines))
        for i, ln in enumerate(p.lines):
            if only is not None:
                if ln.speaker in only and ln.gender_src == "voice":
                    ln.gender, ln.gender_src, ln.tts_key = "", "", ""
                continue
            if i % 25 == 0:
                _check_cancel(cancel)
                _emit(progress, 0.8 + 0.15 * i / n, f"Checking voices {i + 1}/{n}")
            ln.voice_gender, ln.voice_conf, _f0 = ap.voice_gender(
                _speech_source(p, ln), ln.start, ln.end)
            if ln.print_gender:
                # the voice print's side of the male/female split: pitch
                # alone took a shouting young man for a woman (14 of 60
                # lines wrong on a drama; the prints 2, both one-word replies)
                agree = ln.voice_gender == ln.print_gender
                ln.voice_gender = ln.print_gender
                ln.voice_conf = max(ln.voice_conf, 0.9) if agree else 0.8
            if ln.gender_src == "voice":      # re-decided below
                ln.gender, ln.gender_src = "", ""
        if ai_check and keep_characters and only is None:
            self._ai_gender_conflicts(p, cancel)
        candidates: list[DubLine] = []
        for ch in p.characters:
            if only is not None and ch.id not in only:
                continue
            ch.gender_recheck = False
            mine = [ln for ln in p.lines if ln.speaker == ch.id and ln.voice_gender]
            weight = sum(min(6.0, ln.end - ln.start) for ln in mine)
            if not weight:
                if not keep_characters:
                    ch.gender_conf = 0.0
                continue
            if not keep_characters:
                score = sum(min(6.0, ln.end - ln.start) * ln.voice_conf
                            * (1 if ln.voice_gender == "F" else -1) for ln in mine) / weight
                if ch.gender_source != "user":
                    ch.gender = "F" if score > 0 else "M"
                ch.gender_conf = round(min(1.0, abs(score) / 0.6), 2)
            # one character, two people: some lines clearly the other gender
            other = [ln for ln in mine if ln.voice_gender != ch.gender
                     and ln.voice_conf >= 0.6 and ln.end - ln.start >= 0.8]
            other_s = sum(ln.end - ln.start for ln in other)
            if len(other) >= 2 and other_s >= 0.2 * sum(ln.end - ln.start for ln in mine):
                candidates += [ln for ln in other if ln.gender_src != "user"]
            elif ch.gender == "F":
                # a voice this low (about 125 Hz or less) is a man's even when
                # he says only a line or two that went to this character; a high
                # voice is not that sure (a man shouting), so that needs more
                candidates += [ln for ln in other if ln.voice_gender == "M"
                               and ln.voice_conf >= 0.9 and ln.gender_src != "user"]
        keep: set[int] = set()
        if ai_check and candidates:
            _emit(progress, 0.96, f"Checking {len(candidates)} lines that sound like "
                                  "the other gender")
            keep = self._ai_gender_veto(p, candidates, cancel)
        switched = 0
        for ln in candidates:
            if id(ln) in keep:
                continue
            ln.gender, ln.gender_src = ln.voice_gender, "voice"
            ln.tts_key = ""
            switched += 1
        if keep:
            log.info("%d lines keep their character's gender: the dialogue says so "
                     "(the voice only sounds like the other gender)", len(keep))
        return switched

    def _ai_gender_conflicts(self, p: DubProject, cancel) -> list[str]:
        """Characters the story calls one gender while most of their talk
        time sounds like the other: a man reborn in a woman's body, a woman
        disguised as a man — or only a man shouting, or two people under one
        label. One focused question each; the voice we hear wins only when
        the dialogue shows why. Returns the labels of characters changed."""
        changed = []
        for ch in p.characters:
            if ch.gender_source != "ai":
                continue
            mine = [ln for ln in p.lines if ln.speaker == ch.id and ln.voice_gender]
            total = sum(ln.end - ln.start for ln in mine)
            other_s = sum(ln.end - ln.start for ln in mine if ln.voice_gender != ch.gender)
            if len(mine) < 3 or total < 4.0 or other_s < 0.65 * total:
                continue
            said, heard = ("male", "female") if ch.gender == "M" else ("female", "male")
            system = (
                "You help dub a drama; the dub voice must match the body and "
                "voice we see and hear. The story calls one character "
                f"{said}, but their voice measures {heard} in most lines. "
                f"Answer \"{heard}\" only if the dialogue shows why: a body "
                f"swap, reincarnation or rebirth into a {heard} body, or a "
                f"{heard} actor playing or disguised as a {said} character. "
                f"Answer \"{said}\" if the voice only sounds {heard} for other "
                "reasons (a man shouting, crying or upset, a boy whose voice has "
                "not broken, a deep-voiced woman) or other people's lines were "
                "filed under this character. Respond with ONLY one JSON object: "
                "{\"gender\": \"male\"|\"female\", \"evidence\": <short quote "
                "from the dialogue and why>}")
            user = (f"Character: {ch.id} ({ch.label()}), the story says {said}; "
                    f"{other_s / total:.0%} of their measured talk time sounds {heard}.\n"
                    f"Story: {p.story_context[:1500]}\n\nTranscript (speaker-labelled):\n"
                    f"{_speaker_transcript(p, max_chars=8000)}\n\nReturn the JSON now.")
            try:
                res = self.ai.chat(feature="translate_draft", json_mode=True,
                                   temperature=0.1, max_tokens=600, cancel=cancel,
                                   system=system,
                                   messages=[{"role": "user", "content": user}])
                data = extract_json(res.text)
            except CancelledError:
                raise
            except Exception as exc:
                if ai_unavailable(exc):
                    raise    # stop: a later run does this pass
                log.warning("AI gender conflict check failed for %s (%s)", ch.label(), exc)
                continue
            g = str(data.get("gender") or "").lower() if isinstance(data, dict) else ""
            why = str(data.get("evidence") or "").strip() if isinstance(data, dict) else ""
            if g == heard and len(why) >= 3:
                ch.gender = "M" if heard == "male" else "F"
                changed.append(ch.label())
                log.info("%s: voiced %s as heard, not %s as the story says: %s",
                         ch.label(), heard, said, why[:160])
                LOG_BUS.publish(f"[dub] {ch.label()}: voiced {heard} as heard ({why[:90]})")
            else:
                log.info("%s stays %s although it sounds %s: %s", ch.label(), said,
                         heard, why[:160])
        return changed

    def _ai_gender_veto(self, p: DubProject, candidates: list[DubLine],
                        cancel) -> set[int]:
        """Lines (ids of the DubLine objects) that the dialogue shows are
        spoken by their character's own gender, although the voice measured
        like the other one. Any failure keeps the voice measurement."""
        pos = {id(ln): i for i, ln in enumerate(p.lines)}
        # names only: ids and the story summary carry the (maybe wrong) labels
        cast = "; ".join(f"{c.label()} ({'male' if c.gender == 'M' else 'female'})"
                         for c in p.characters if not c.label().startswith("Speaker "))
        show: set[int] = set()
        for ln in candidates:
            i = pos[id(ln)]
            show.update(range(max(0, i - 3), min(len(p.lines), i + 4)))
        ask = {pos[id(ln)] for ln in candidates}
        rows, prev = [], -2
        for i in sorted(show):
            if i != prev + 1:
                rows.append("…")
            mark = " <<CHECK" if i in ask else ""
            # no speaker labels: the AI would just repeat the label's gender
            rows.append(f"{i + 1}. {p.lines[i].source}{mark}")
            prev = i
        system = (
            "You check who speaks some lines of a drama transcript, for dubbing. "
            "Voice pitch is unsure about the lines marked <<CHECK: a man who "
            "shouts, cries or is upset can sound as high as a woman, but a woman "
            "can also be in the scene. Speaker labels are left out on purpose "
            "(voice clustering often files several people under one label). For "
            "each <<CHECK line decide from the dialogue alone whether a male or a "
            "female speaks it: who is addressed and who answers whom in an "
            "exchange between named characters, or words for the speaker "
            "themself (a self-reference like 小爷 or 老娘, being called by name, "
            "he or she). What the speaker says about other people and the tone "
            "are not evidence. Give that evidence as a short fact (e.g. "
            "\"answers Celia, who called him Tom\"); with no such evidence answer "
            "\"unknown\". Answer the gender of the body and voice we see and "
            "hear, which the dub must match: in a body-swap, reincarnation or "
            "rebirth story a man in a woman's body is \"female\". Respond with "
            "ONLY one JSON object: {\"rows\": [{\"line\": <line number>, "
            "\"gender\": \"male\"|\"female\"|\"unknown\", \"evidence\": <short "
            "quote or fact, \"\" if none>}]}")
        try:
            res = self.ai.chat(
                feature="translate_draft", json_mode=True, temperature=0.1,
                max_tokens=2000, cancel=cancel, system=system,
                messages=[{"role": "user", "content":
                           (f"Named characters: {cast}\n\n" if cast else "")
                           + "Transcript:\n" + "\n".join(rows)
                           + "\n\nReturn the JSON now."}])
            data = extract_json(res.text)
        except CancelledError:
            raise
        except Exception as exc:
            if ai_unavailable(exc):
                raise    # stop: a later run does this pass
            log.warning("AI gender check failed (%s); following the voice", exc)
            return set()
        said: dict[int, tuple[str, str]] = {}
        for r in (data.get("rows") if isinstance(data, dict) else None) or []:
            try:
                said[int(r.get("line")) - 1] = (str(r.get("gender") or "").lower(),
                                                str(r.get("evidence") or "").strip())
            except (TypeError, ValueError, AttributeError):
                continue
        keep = set()
        for ln in candidates:
            g, why = said.get(pos[id(ln)], ("", ""))
            ch = p.character(ln.speaker)
            # a "same as the label" answer without evidence from the
            # dialogue ("S2 is speaking") is no answer
            if (ch and g in ("male", "female") and len(why) >= 3
                    and not re.search(r"\b[SC]\d+\b|\blabel", why, re.I)
                    and ("M" if g == "male" else "F") == ch.gender):
                keep.add(id(ln))
                log.info("line %d stays %s: %s", pos[id(ln)] + 1, g, why[:120])
        return keep

    # -- series memory --------------------------------------------------------
    def _apply_series_cast(self, p: DubProject) -> list[str]:
        """Give characters whose voice matches a known cast member that
        member's name, gender and voice (including its cloning sample)."""
        if not p.options.series.strip():
            return []
        from app.core.series_memory import SeriesMemory, match_cast
        mem = SeriesMemory.load(p.options.series)
        if not mem.cast:
            return []
        method = p.characters[0].embed_method if p.characters else ""
        res = match_cast([c.embedding for c in p.characters], method, mem.cast)
        names = []
        for ch, m in zip(p.characters, res):
            if m is None:
                continue
            member, score = m
            ch.name, ch.gender = member.name, member.gender
            ch.engine, ch.voice = member.engine or "auto", member.voice
            ch.rate, ch.pitch, ch.volume = member.rate, member.pitch, member.volume
            if member.ref_wav and os.path.isfile(member.ref_wav):
                ch.ref_wav, ch.ref_text, ch.ref_custom = (member.ref_wav,
                                                          member.ref_text, True)
            ch.cast_match = f"{member.name} ({score:.2f})"
            names.append(member.name)
        return names

    def save_to_series(self, p: DubProject, manual: bool = False) -> str:
        """Store this episode's named characters (voice + fingerprint) and
        its glossary in the series memory. Returns a short summary."""
        name = p.options.series.strip()
        if not name:
            return ""
        from app.core.series_memory import SeriesMemory
        mem = SeriesMemory.load(name)
        saved = []
        if len(p.characters) < 2 and not manual:
            # one character for a whole episode is almost always a detection
            # failure: don't teach the series a voice made of everyone
            log.warning("save_to_series: only one character; cast not saved")
        for ch in (p.characters if (len(p.characters) >= 2 or manual) else []):
            label = ch.name.strip()
            if not label or label.startswith("Speaker ") or ch.line_count < 3:
                continue     # unnamed or bit parts: not worth remembering
            mem.remember(name=label, gender=ch.gender, engine=ch.engine,
                         voice=ch.voice, rate=ch.rate, pitch=ch.pitch,
                         volume=ch.volume, ref_wav=ch.ref_wav,
                         ref_text=ch.ref_text, embedding=ch.embedding,
                         embed_method=ch.embed_method)
            saved.append(label)
        lang = _base_lang(p.options.target_lang)
        terms = mem.merge_glossary(lang, [asdict(g) for g in p.glossary])
        if p.story_context and not mem.story:
            mem.story = p.story_context
        mem.save()
        return (f"series '{name}': {len(saved)} characters, {terms} glossary "
                "changes saved")

    # -- step 4: voices -----------------------------------------------------
    def _step_voices(self, p: DubProject, progress, cancel):
        src = p.vocals_wav if os.path.isfile(p.vocals_wav) else p.asr_wav
        built = 0
        noisy = []
        n = max(1, len(p.characters))
        for i, ch in enumerate(p.characters):
            _check_cancel(cancel)
            _emit(progress, i / n, f"Voice sample for {ch.label()}")
            if ch.ref_custom and os.path.isfile(ch.ref_wav):
                continue
            out = p.sub("voices", f"{ch.id}.wav")
            mine = [l for l in p.lines if l.speaker == ch.id]
            res = build_reference(src, mine, out)
            if res is None and p.audio_wav and p.audio_wav != src \
                    and os.path.isfile(p.audio_wav):
                # separation suppressed this voice: fall back to the full mix
                res = build_reference(p.audio_wav, mine, out)
                if res:
                    noisy.append(ch.label())
            if res:
                ch.ref_wav, ch.ref_text = res
                built += 1
            else:
                ch.ref_wav, ch.ref_text = "", ""
        clone = best_clone_engine(p.options.target_lang)
        mode = (f"cloning with '{clone}'" if clone and p.options.voice_mode == "clone"
                else "preset voices (no cloning engine installed)"
                if p.options.voice_mode == "clone" else "preset voices")
        missing = [c.label() for c in p.characters if not c.ref_wav]
        note = f"{built} voice samples built; {mode}"
        if noisy:
            note += f"; samples include background music: {', '.join(noisy)}"
        if missing:
            note += f"; too little clean speech for: {', '.join(missing)}"
        if noisy or missing:
            return "warn", note
        return "done", note

    # -- step 5: glossary ---------------------------------------------------
    def _step_glossary(self, p: DubProject, progress, cancel):
        series_terms = 0
        # the AI's terms of an earlier run are asked for again (a wrong one
        # stayed for good and the meaning check enforced it: 神豪系统 →
        # "goddess system"); terms typed in the Glossary tab and the
        # series' stay
        p.glossary = [g for g in p.glossary if g.origin in ("user", "series")]
        if p.options.series.strip():
            # start from the series' fixed spellings (user edits win)
            from app.core.series_memory import SeriesMemory
            mem = SeriesMemory.load(p.options.series)
            have = {g.source.strip().lower() for g in p.glossary}
            for t in mem.glossary_for(_base_lang(p.options.target_lang)):
                if t["source"].strip().lower() not in have:
                    p.glossary.append(GlossaryTerm(t["source"], t.get("target", ""),
                                                   t.get("note", ""), origin="series"))
                    series_terms += 1
            if mem.story and not p.story_context:
                p.story_context = mem.story
        transcript = _speaker_transcript(p, max_chars=14000)
        names = ", ".join(f"{c.id} (voice sounds {'male' if c.gender == 'M' else 'female'})"
                          for c in p.characters)
        tgt = _lang_name(p.options.target_lang)
        system = (
            "You prepare a film/drama for professional dubbing. Respond with "
            "ONLY one JSON object, no prose, no fences. Schema: "
            "{\"summary\": <3-6 sentence plot summary in English>, "
            "\"speakers\": [{\"id\": <speaker id>, \"name\": <character name "
            "as said in the dialogue, or \"\" if unknown>, \"role\": <short "
            "description, e.g. 'the boy's mother'>, \"gender\": \"male\"|"
            "\"female\"|\"unknown\", \"age\": \"child\"|\"teen\"|\"adult\"|"
            "\"elder\"}], \"relationships\": [<e.g. 'S1 is the mother of S2'>], "
            "\"terms\": [{\"source\": <name/term exactly as in the transcript>, "
            f"\"target\": <how to write it in {tgt}>, \"note\": <short note>}}]}}. "
            "Gender/age: judge from the dialogue (how others address them, "
            "what they say) — a child's voice can sound female, so a boy is "
            "\"male\" + \"child\". Gender is the body and voice we see and "
            "hear, which the dub must match: in a body-swap, reincarnation or "
            "rebirth story (a man waking up in a woman's body) give the body's "
            "gender and say so in the role. The speaker labels come from voice "
            "clustering and can be wrong on some lines; describe each label "
            "by the majority of its lines. Terms: person and place names, "
            "titles, ranks, organisations, invented words, recurring "
            f"catch-phrases. Transliterate names phonetically into {tgt} "
            "script; translate other terms by what they MEAN in the story, "
            "never word by word. Only names you can infer from the dialogue. "
            "Radio and work words (Roger, Copy, Over, Mayday) and forms of "
            "address (Sir, Ma'am, Honey, Boss, Doc) are not names: leave them "
            "out. Never a phrase or idiom whose meaning depends on the line "
            "(你别说 is \"don't tell me…\" in one scene, \"I must say…\" in "
            "another): a term is written the same way every time."
            + (" Khmer targets in Khmer script only: write brands and foreign "
               "names by their sound (川崎 / Kawasaki → កាវ៉ាសាគី, Benz → បេនស៍); "
               "a model code such as H2 may stay." if _base_lang(p.options.target_lang) == "km"
               else "")
            + "".join(f" {h}" for h in self._term_hints(p)))
        title = os.path.splitext(os.path.basename(p.video))[0]
        context_line = (f"Series: {p.options.series}\n" if p.options.series else "") + \
            f"Episode file: {title}\n"
        user = (f"{context_line}Speakers: {names}\n\nTranscript (speaker-labelled):\n"
                f"{transcript}\n\nReturn the JSON object now.")
        _emit(progress, 0.1, "Asking AI for story context and names")
        # room for a summary, every character and the terms: with seven
        # characters a 4096-token reply ended before its "terms"
        res = self.ai.chat(feature="translate_draft",
                           messages=[{"role": "user", "content": user}],
                           system=system, temperature=0.2, max_tokens=8192,
                           json_mode=True, cancel=cancel)
        try:
            data = extract_json(res.text)
        except Exception as exc:
            raise AppError(f"Could not read the AI glossary reply: {exc}",
                           f"មិនអាចអានចម្លើយសទ្ទានុក្រមពី AI បានទេ៖ {exc}") from exc
        if not isinstance(data, dict):
            data = {}
        summary = str(data.get("summary") or "").strip()
        if summary:
            p.story_context = summary
        roles = []
        for sp in data.get("speakers") or []:
            if not isinstance(sp, dict):
                continue
            ch = p.character(str(sp.get("id") or ""))
            if ch is None:
                continue
            name = str(sp.get("name") or "").strip()
            if name and (not ch.name or ch.name.startswith("Speaker ")):
                ch.name = name
            role = str(sp.get("role") or "").strip()
            g = str(sp.get("gender") or "").lower()
            age = str(sp.get("age") or "").lower()
            if age in ("child", "teen", "adult", "elder"):
                ch.age = age
            heard = self._print_majority(p, ch.id)
            if (g in ("male", "female") and heard and ("M" if g == "male" else "F") != heard
                    and age != "child" and ch.gender_source != "user"):
                # the story guessed the gender from words alone ("a rude
                # driver" became a woman); the voice prints say who we hear
                log.info("%s stays %s: the voices are a %s's (the dialogue suggested %s)",
                         ch.label(), heard, "man" if heard == "M" else "woman", g)
                ch.gender, ch.gender_source = heard, "voice"
                g = ""
            if g in ("male", "female") and ch.gender_source != "user":
                new = "M" if g == "male" else "F"
                if new != ch.gender:
                    log.info("%s: gender %s -> %s from the dialogue (%s)", ch.label(),
                             ch.gender, new, age or "?")
                ch.gender, ch.gender_source = new, "ai"
            if role:
                roles.append(f"{ch.label()} ({ch.id}, "
                             f"{'male' if ch.gender == 'M' else 'female'}"
                             f"{', ' + ch.age if ch.age else ''}): {role}")
        if roles:
            p.story_context = (p.story_context + "\nCharacters: "
                               + "; ".join(roles)).strip()
        rel = [str(r).strip() for r in data.get("relationships") or [] if str(r).strip()]
        if rel:
            p.story_context = (p.story_context + "\nRelationships: "
                               + "; ".join(rel[:12])).strip()
        have = {g.source.strip().lower() for g in p.glossary}
        added = 0
        for t in data.get("terms") or []:
            if not isinstance(t, dict):
                continue
            src = str(t.get("source") or "").strip()
            target = str(t.get("target") or "").strip()
            if target and self._wrong_script(p, target, src):
                # every line would copy it ("ម៉ូតូ Kawasaki H2" kept Latin
                # letters through two re-translations)
                log.info("glossary term %s → %s left out: not in %s script", src, target, tgt)
                continue
            if src and src.lower() not in have:
                p.glossary.append(GlossaryTerm(src, str(t.get("target") or "").strip(),
                                               str(t.get("note") or "").strip(), origin="ai"))
                have.add(src.lower())
                added += 1
        note = f"{added} glossary terms, story summary ready"
        if series_terms:
            note += f"; {series_terms} terms from the series"
        before = len(p.lines)
        # CAM++ prints group voices well even at a silhouette of 0.15
        # (a drama's clustering scored 0.16 with every long line right)
        campplus = any(c.embed_method == "campplus" for c in p.characters)
        if p.speaker_score < (0.05 if campplus else 0.15) or len(p.characters) < 2:
            assigned = self._ai_assign_speakers(p, progress, cancel)
            if assigned:
                note += (f"; voices too alike (separation {p.speaker_score:.2f}): "
                         f"AI assigned {assigned} lines to {len(p.characters)} characters")
        else:
            moved = self._ai_check_speakers(p, progress, cancel)
            if moved:
                note += f"; {moved} lines moved to the right speaker"
        p.lines = merge_fragments(p.lines, p.src_lang)
        p.recount()
        if len(p.lines) < before:
            note += f"; {before - len(p.lines)} fragments joined into sentences"
        # lines moved to other speakers, fragments joined, genders read from
        # the dialogue: the per-line voice genders measured in the speakers
        # step are stale now — measure the final lines and decide again
        # which ones get the other gender's voice (the characters' genders
        # stay as they are)
        story_g = {c.id: c.gender for c in p.characters}
        switched = self._detect_genders(p, progress, cancel, keep_characters=True,
                                        ai_check=True)
        heard = [c for c in p.characters if c.gender != story_g[c.id]]
        if heard:
            note += ("; voiced as heard, not as the story says: "
                     + ", ".join(c.label() for c in heard))
            # the translation reads the story: say how these characters speak
            p.story_context = (p.story_context + "\nVoices: " + "; ".join(
                f"{c.label()} is heard as {'a woman' if c.gender == 'F' else 'a man'} "
                f"(body swap / rebirth): translate {c.label()}'s lines with "
                f"{'female' if c.gender == 'F' else 'male'} forms" for c in heard)).strip()
        if switched:
            note += f"; {switched} lines voiced with the other gender"
        return note

    @staticmethod
    def _print_majority(p: DubProject, cid: str) -> str:
        """The gender 85% or more of a character's talk time has in the voice
        prints' male/female split, or ""."""
        mine = [ln for ln in p.lines if ln.speaker == cid and ln.print_gender]
        total = sum(ln.end - ln.start for ln in mine)
        if total < 1.0:
            return ""
        f = sum(ln.end - ln.start for ln in mine if ln.print_gender == "F") / total
        return "F" if f >= 0.85 else "M" if f <= 0.15 else ""

    def _ai_assign_speakers(self, p: DubProject, progress, cancel) -> int:
        """Dialogue-based speaker attribution for when the voices cannot be
        told apart (e.g. a child voiced by an adult woman): the AI lists the
        speaking characters and labels every line. Voice-cluster labels are
        given as weak hints. Returns the number of lines labelled."""
        system = (
            "You do speaker attribution for a drama transcript, for dubbing. "
            "Voice analysis could not tell the speakers apart, so decide from "
            "the dialogue: who is addressed, who answers, turn-taking, what "
            "each character would say. Respond with ONLY one JSON object: "
            "{\"characters\": [{\"id\": \"C1\", \"name\": <name or short "
            "description, e.g. 'Mother'>, \"gender\": \"male\"|\"female\", "
            "\"age\": \"child\"|\"teen\"|\"adult\"|\"elder\", \"role\": <short "
            "role>}], \"lines\": [{\"line\": <line number>, \"speaker\": "
            "<character id>}]}. Gender is the body and voice we hear (a man "
            "reborn in a woman's body is \"female\"). Label EVERY line. Use as "
            "few characters as the "
            "dialogue needs (minor voices like a landlord or a crowd may share "
            "one 'Other' character). Keep ids consistent with any character "
            "list you are given.")
        known: list[dict] = []
        labels: dict[int, str] = {}
        chunk = 100
        for b0 in range(0, len(p.lines), chunk):
            _check_cancel(cancel)
            part = p.lines[b0:b0 + chunk]
            body = "\n".join(f"{b0 + k + 1}. [voice {l.speaker}] {l.source}"
                             for k, l in enumerate(part))
            given = ("Characters so far (reuse these ids): "
                     + json.dumps(known, ensure_ascii=False) + "\n") if known else ""
            _emit(progress, 0.6 + 0.4 * b0 / len(p.lines), "Working out who says what")
            try:
                res = self.ai.chat(
                    feature="translate_draft", json_mode=True, temperature=0.1,
                    max_tokens=6000, cancel=cancel, system=system,
                    messages=[{"role": "user", "content":
                               f"Story: {p.story_context[:1500]}\n{given}"
                               f"\nTranscript:\n{body}\n\nReturn the JSON now."}])
                data = extract_json(res.text)
            except CancelledError:
                raise
            except Exception as exc:
                if ai_unavailable(exc):
                    raise    # stop: a later run does this pass
                log.warning("AI speaker attribution failed (%s); keeping voice labels", exc)
                return 0
            if not isinstance(data, dict):
                return 0
            for c in data.get("characters") or []:
                if isinstance(c, dict) and c.get("id") and not any(
                        k["id"] == c["id"] for k in known):
                    known.append({k: str(c.get(k) or "") for k in
                                  ("id", "name", "gender", "age", "role")})
            for item in data.get("lines") or []:
                try:
                    i = int(item.get("line")) - 1
                except (TypeError, ValueError, AttributeError):
                    continue
                if 0 <= i < len(p.lines):
                    labels[i] = str(item.get("speaker") or "")
        ids = [k["id"] for k in known]
        if len(labels) < len(p.lines) * 0.6 or not ids:
            log.warning("AI attribution covered %d/%d lines; keeping voice labels",
                        len(labels), len(p.lines))
            return 0
        # characters ordered by talk time become S1, S2, ...
        talk: dict[str, float] = {}
        for i, cid in labels.items():
            if cid in ids:
                talk[cid] = talk.get(cid, 0.0) + p.lines[i].end - p.lines[i].start
        order = sorted(talk, key=lambda c: -talk[c])
        new_id = {cid: f"S{n + 1}" for n, cid in enumerate(order)}
        info = {k["id"]: k for k in known}
        chars = []
        for cid in order:
            k = info[cid]
            g = k.get("gender", "").lower()
            ch = Character(id=new_id[cid], name=k.get("name") or new_id[cid],
                           gender="M" if g == "male" else "F",
                           age=k.get("age", "").lower(), gender_source="ai")
            chars.append(ch)
        prev = order[0]
        for i, ln in enumerate(p.lines):
            cid = labels.get(i)
            if cid not in new_id:
                cid = prev                      # unlabelled: previous speaker
            ln.speaker, ln.tts_key = new_id[cid], ""
            prev = cid
        p.characters = chars
        p.recount()
        roles = "; ".join(f"{c.label()} ({c.id}, {'male' if c.gender == 'M' else 'female'}"
                          f"{', ' + c.age if c.age else ''}): {info[cid].get('role', '')}"
                          for cid, c in zip(order, chars))
        p.story_context = re.sub(r"\nCharacters: .*", "", p.story_context)
        p.story_context = (p.story_context + "\nCharacters: " + roles).strip()
        return len(labels)

    def _ai_check_speakers(self, p: DubProject, progress, cancel) -> int:
        """Ask the AI which lines were given to the wrong character (voice
        clustering confuses similar voices, e.g. a mother and her young
        son); move only the lines it is sure about."""
        if len(p.characters) < 2 or len(p.lines) < 4:
            return 0
        cast = "; ".join(f"{c.id} = {c.label()} ({'male' if c.gender == 'M' else 'female'}"
                         f"{', ' + c.age if c.age else ''})" for c in p.characters)
        system = (
            "You check speaker labels in a drama transcript for dubbing. Labels "
            "come from voice clustering and are sometimes wrong. Using the "
            "dialogue (who is addressed, who answers, what each character "
            "would say, turn-taking), find lines whose label is clearly "
            "wrong. Respond with ONLY one JSON object: {\"changes\": "
            "[{\"line\": <line number>, \"speaker\": <correct speaker id>}]}. "
            "Only include lines you are confident about; never invent new "
            "speaker ids; return {\"changes\": []} if all look right.")
        moved = 0
        ids = {c.id for c in p.characters}
        chunk = 120
        for b0 in range(0, len(p.lines), chunk):
            _check_cancel(cancel)
            part = p.lines[b0:b0 + chunk]
            body = "\n".join(f"{b0 + k + 1}. [{l.speaker}] {l.source}"
                             for k, l in enumerate(part))
            _emit(progress, 0.6 + 0.4 * b0 / len(p.lines), "Checking who says what")
            try:
                res = self.ai.chat(
                    feature="translate_draft", json_mode=True, temperature=0.1,
                    max_tokens=3000, cancel=cancel, system=system,
                    messages=[{"role": "user", "content":
                               f"Characters: {cast}\nStory: {p.story_context[:1500]}"
                               f"\n\nTranscript:\n{body}\n\nReturn the JSON now."}])
                data = extract_json(res.text)
            except CancelledError:
                raise
            except Exception as exc:
                if ai_unavailable(exc):
                    raise    # stop: a later run does this pass
                log.warning("AI speaker check failed (%s); keeping voice labels", exc)
                return moved
            for c in (data.get("changes") if isinstance(data, dict) else None) or []:
                try:
                    i = int(c.get("line")) - 1
                except (TypeError, ValueError, AttributeError):
                    continue
                sp = str(c.get("speaker") or "")
                if not (0 <= i < len(p.lines) and sp in ids and p.lines[i].speaker != sp):
                    continue
                ln = p.lines[i]
                tgt_ch = p.character(sp)
                if ln.voice_sure:
                    log.info("line %d stays %s: its voice print says so (AI said %s)",
                             i + 1, ln.speaker, sp)
                    continue
                cur_ch = p.character(ln.speaker)
                if (tgt_ch is not None and cur_ch is not None and tgt_ch.gender
                        and tgt_ch.gender != cur_ch.gender and ln.voice_gender
                        and ln.voice_gender != tgt_ch.gender and ln.voice_conf >= 0.5):
                    # a move to the other gender against the line's own
                    # voice ("顾先生", a man's line, went to a woman); a boy
                    # voiced by a woman still moves to the boy when his
                    # voice is unclear
                    log.info("line %d stays %s: its voice does not sound %s", i + 1,
                             ln.speaker, "male" if tgt_ch.gender == "M" else "female")
                    continue
                ln.speaker, ln.tts_key = sp, ""
                moved += 1
        if moved:
            p.lines = merge_fragments(p.lines, p.src_lang)
            p.recount()
        return moved

    # -- step 6: translate --------------------------------------------------
    def _gender_of(self, p: DubProject, ln: DubLine) -> str:
        ch = p.character(ln.speaker)
        return ln.gender or (ch.gender if ch else "")

    def _fix_mishearings(self, p: DubProject, cancel) -> list[str]:
        """Speech recognition hears a wrong but similar-sounding word now and
        then ("a shield our hospital hired" for "a shill", "We're only one"
        for "We're on in one"); translated literally, the line is nonsense,
        and a checker that sees the transcript insists on the literal word.
        One AI pass over the transcript names such words from the scene; a
        correction is taken only when it sounds like what was heard, and a
        line is corrected at most once (`heard` keeps the recognised words).
        Returns "heard → meant" notes."""
        cache = p.tr_cache.setdefault("misheard", {})
        fixed: list[str] = []
        # a line heard the same as in an earlier run gets that run's
        # correction again (a new transcription used to lose it: 领配 came
        # back after it had been corrected to 顶配)
        for i, l in enumerate(p.lines):
            if l.heard or l.on_screen or self._hand_edited(l):
                continue
            meant = cache.get(_sha1(l.source))
            if meant and meant != l.source:
                l.heard, l.source = l.source, meant
                l.text, l.tts_key = "", ""
                fixed.append(f"{l.heard} → {meant}")
        # the video's own subtitles are the script: nothing misheard there
        todo = [i for i, l in enumerate(p.lines)
                if l.source.strip() and not l.heard and not l.on_screen
                and not self._hand_edited(l) and _sha1(l.source) not in cache]
        if not todo:
            return fixed
        src = _lang_name(p.src_lang) if p.src_lang else "the source language"
        cjk = _base_lang(p.src_lang) in _CJK_LIKE
        names = set(self._speaker_names(p)) | {g.source.strip().lower() for g in p.glossary}
        system = (
            f"You proofread a {src} speech-recognition transcript of a film scene. "
            "Find ONLY words the recogniser must have misheard: the heard words make "
            "no sense where they stand, and a similar-SOUNDING phrase fits the scene "
            "(\"a shield our hospital hired\" → \"a shill\"; \"We're only one\" said "
            "before \"All systems go\" → \"We're on in one\"). Leave slang, names, "
            "numbers and odd but possible lines alone; when in doubt, change nothing. "
            "Respond with ONLY one JSON object: {\"rows\": [{\"line\": <number>, "
            "\"heard\": <the exact words as written>, \"meant\": <replacement>}]} — "
            "an empty list when nothing is misheard.")
        for b0 in range(0, len(todo), 120):
            _check_cancel(cancel)
            chunk = todo[b0:b0 + 120]
            # lines already confirmed (on-screen subtitles) are shown as the
            # scene around the ones to check, and are not changed
            shown = range(max(0, chunk[0] - 3), min(len(p.lines), chunk[-1] + 4))
            body = "\n".join(f"{i + 1}. {p.lines[i].source}" for i in shown
                             if i in chunk or p.lines[i].on_screen)
            try:
                res = self.ai.chat(
                    feature="translate_draft", json_mode=True, temperature=0.1,
                    max_tokens=3000, cancel=cancel, system=system,
                    messages=[{"role": "user", "content":
                               f"Story: {p.story_context[:1500]}\n\nTranscript:\n{body}"
                               "\n\nReturn the JSON now."}])
                data = extract_json(res.text)
            except CancelledError:
                raise
            except Exception as exc:
                if ai_unavailable(exc):
                    raise    # stop: a later run does this pass
                log.warning("mishearing check failed (%s); transcript kept", exc)
                continue
            rows = data.get("rows") if isinstance(data, dict) else data
            for r in rows or []:
                if not isinstance(r, dict):
                    continue
                try:
                    i = int(r.get("line")) - 1
                except (TypeError, ValueError):
                    continue
                heard = str(r.get("heard") or "").strip()
                meant = str(r.get("meant") or "").strip()
                if i not in chunk or not heard or not meant or heard.lower() == meant.lower():
                    continue
                line = p.lines[i].source
                pos = line.lower().find(heard.lower())
                if pos < 0 or heard.lower() in names or meant.lower() in names:
                    continue
                if not sounds_alike(heard, meant, cjk, _base_lang(p.src_lang)):
                    log.info("line %d: mishearing %r → %r not taken (does not sound alike)",
                             i + 1, heard, meant)
                    continue
                p.lines[i].heard = line
                p.lines[i].source = line[:pos] + meant + line[pos + len(heard):]
                p.lines[i].text, p.lines[i].tts_key = "", ""
                fixed.append(f"{heard} → {meant}")
                log.info("line %d: misheard %r → %r", i + 1, heard, meant)
            for i in chunk:
                cache[_sha1(p.lines[i].heard or p.lines[i].source)] = p.lines[i].source
        if fixed:
            LOG_BUS.publish(f"[dub] misheard words corrected: {', '.join(fixed)}")
        return fixed

    def _scene_notes(self, p: DubProject, cancel) -> dict[int, str]:
        """One AI read of the whole scene before translating: short notes on
        the lines a literal translation gets wrong — idioms (你别说 =
        "don't tell me…"), genre words (神豪 = a tycoon), who "you" / 您 is
        (a caller talking to the hero), a sentence running over several
        lines. Translated 20 lines at a time, the model missed what the
        scene made clear (神豪 → "goddess", 你别说 → "don't speak").
        Returns {line index: note}; kept in tr_cache["notes"]."""
        cache = p.tr_cache.setdefault("notes", {})
        src = _lang_name(p.src_lang) if p.src_lang else "the source language"
        tgt = _lang_name(p.options.target_lang)
        system = (
            f"You are a {src} → {tgt} dubbing script consultant. Read the whole "
            "scene, then write a short note for the translator on each line "
            "whose meaning a literal translation would get wrong: idioms and "
            "set phrases (Chinese 你别说 = \"don't tell me…\" in disbelief, or "
            "\"I must say…\"); slang and genre words (神豪 = a super-rich "
            "tycoon in web-novel stories; 系统 = the System, a game-like voice "
            "that gives the hero tasks); whom \"you\" / 您 means when it is not "
            "obvious (a caller on the phone speaking to the hero, not to the "
            "person beside him); who \"he\" / \"she\" is; a name used to "
            "address someone; sarcasm, a joke, a threat; a sentence that runs "
            "over several lines; a word that only makes sense from an earlier "
            "line. Write what is MEANT in plain English, at most 25 words; add "
            f"the {tgt} word when an everyday one fits. Speaker labels may be "
            "wrong now and then. Leave plain lines out. Respond with ONLY one "
            "JSON object: {\"notes\": [{\"line\": <number>, \"note\": <text>}]}.")
        notes: dict[int, str] = {}
        chunk = 150
        for b0 in range(0, len(p.lines), chunk):
            _check_cancel(cancel)
            part = range(b0, min(len(p.lines), b0 + chunk))
            body = "\n".join(
                f"{i + 1}. [{self._speaker_tag(p, i).split(' |')[0].lstrip('[')}] "
                f"{p.lines[i].source}" for i in part)
            key = _sha1(_NOTES_VERSION, src, tgt, p.story_context, body)
            if key in cache:
                got = json.loads(cache[key])
            else:
                try:
                    res = self.ai.chat(
                        feature="translate_draft", json_mode=True, temperature=0.2,
                        max_tokens=4000, cancel=cancel, system=system,
                        messages=[{"role": "user", "content":
                                   f"{self._context_block(p)}\n\nScene:\n{body}\n\n"
                                   "Return the JSON now."}])
                    data = extract_json(res.text)
                except CancelledError:
                    raise
                except Exception as exc:
                    if ai_unavailable(exc):
                        raise    # stop: a later run does this pass
                    log.warning("scene notes failed (%s); translating without them", exc)
                    continue
                rows = data.get("notes") if isinstance(data, dict) else data
                got = {}
                for r in rows or []:
                    if not isinstance(r, dict):
                        continue
                    try:
                        i = int(r.get("line")) - 1
                    except (TypeError, ValueError):
                        continue
                    note = re.sub(r"\s+", " ", str(r.get("note") or "")).strip()
                    if i in part and note:
                        got[str(i)] = note[:240]
                cache[key] = json.dumps(got, ensure_ascii=False)
            notes.update({int(k): v for k, v in got.items()})
        if notes:
            log.info("scene notes on %d lines: %s", len(notes),
                     "; ".join(f"{i + 1}: {v}" for i, v in sorted(notes.items())[:12]))
        return notes

    def _notes_block(self, idx: list[int], extra: list[int] | None = None) -> str:
        """The scene notes for a numbered batch (numbers as in the batch)."""
        notes = getattr(self, "scene_notes", {}) or {}
        rows = [f"{k + 1}: {notes[i]}" for k, i in enumerate(idx) if i in notes]
        if not rows:
            return ""
        return ("\n\nNotes on what some of these lines mean (from reading the whole "
                "scene; numbers as in the list below):\n" + "\n".join(rows))

    def _step_translate(self, p: DubProject, progress, cancel):
        n = len(p.lines)
        _emit(progress, 0.0, "Checking the transcript for misheard words")
        misheard = self._fix_mishearings(p, cancel)
        self.scene_notes = {}
        if p.options.quality == "high":
            _emit(progress, 0.02, "Reading the whole scene for meaning")
            self.scene_notes = self._scene_notes(p, cancel)
        batch = 20
        two_pass = p.options.quality == "high"
        lang = _base_lang(p.options.target_lang)
        series = p.options.series.strip()
        drafts: list[str] = [""] * n
        # lines typed in the Lines tab stay as they are (later batches see
        # their text as the previous lines)
        kept = {i for i, ln in enumerate(p.lines) if self._hand_edited(ln)}
        for i in kept:
            drafts[i] = p.lines[i].text
        # lines left out in Lines ("nobody says this") stay empty
        skipped = {i for i, ln in enumerate(p.lines) if ln.skip}
        # translation memory first: only the misses cost AI calls
        tm_hits = stale = 0
        tm = None
        if p.options.use_tm:
            from app.core.translation_memory import get_tm
            tm = get_tm()
            for i, ln in enumerate(p.lines):
                if i in kept or i in skipped:
                    continue
                hit = tm.lookup(lang, ln.source, self._gender_of(p, ln), series,
                                rules=_TR_VERSION)
                if hit:
                    # older runs stored lines with a speaker name glued on ("បប: …")
                    t = strip_speaker_name(_clean_line(hit[0]), ln.source,
                                           self._speaker_names(p))
                    # an AI entry made before a glossary term was set or fixed
                    # keeps the old wording run after run ("ប្រព័ន្ធទេពធីតា",
                    # goddess system, for 神豪系统): translated again instead
                    if hit[1] != "human" and self._glossary_misses(p, ln.source, t):
                        stale += 1
                        continue
                    drafts[i] = t
                    tm_hits += 1
        todo = [i for i in range(n) if not drafts[i] and i not in skipped]
        batches = [todo[b:b + batch] for b in range(0, len(todo), batch)]
        steps = max(1, len(batches) * (2 if two_pass else 1))
        ckey = {i: _sha1(lang, p.lines[i].source, p.lines[i].speaker,
                         self._gender_of(p, p.lines[i]), p.options.style,
                         _TR_VERSION, *(("other voice",) if self._other_voice(
                             p, p.lines[i]) else ()),
                         *((self.scene_notes[i],) if i in self.scene_notes else ()))
                for i in todo}
        resumed = 0
        jobs: list[list[int]] = []
        for idx in batches:
            cached = {i: p.tr_cache["draft"].get(ckey[i]) for i in idx}
            missing = [i for i in idx if not cached[i]]
            for i in idx:
                if cached[i]:
                    drafts[i] = cached[i]
            resumed += len(idx) - len(missing)
            if missing:
                jobs.append(missing)
        lock = threading.Lock()

        def draft(missing: list[int]) -> None:
            out = self._translate_batch(p, missing, drafts, cancel)
            with lock:
                for i, t in zip(missing, out):
                    drafts[i] = t
                    p.tr_cache["draft"][ckey[i]] = t
                p.save()                      # keep finished batches

        # a few batches at a time (_AI_WORKERS); each sees the lines before it
        # translated when they are, else in the original
        k0 = len(batches) - len(jobs)
        self._parallel(jobs, draft, cancel, say=lambda d, idx: _emit(
            progress, (k0 + d) / steps,
            f"Translating lines {idx[0] + 1}-{idx[-1] + 1} of {n}"))
        k = len(batches)
        final = list(drafts)
        if two_pass:
            jobs = []
            rkeys: dict[int, str] = {}
            for idx in batches:
                rkey = {i: _sha1(ckey[i], drafts[i]) for i in idx}
                rkeys.update(rkey)
                missing = [i for i in idx if not p.tr_cache["review"].get(rkey[i])]
                for i in idx:
                    if rkey[i] in p.tr_cache["review"]:
                        final[i] = p.tr_cache["review"][rkey[i]]
                if missing:
                    jobs.append(missing)

            def review(missing: list[int]) -> None:
                out = self._review_batch(p, missing, drafts, final, cancel)
                with lock:
                    for i, t in zip(missing, out):
                        final[i] = t
                        p.tr_cache["review"][rkeys[i]] = t
                    p.save()

            k0 = k + len(batches) - len(jobs)
            self._parallel(jobs, review, cancel, say=lambda d, idx: _emit(
                progress, (k0 + d) / steps,
                f"Reviewing lines {idx[0] + 1}-{idx[-1] + 1} of {n}"))
        for i, (ln, t) in enumerate(zip(p.lines, final)):
            if i not in kept:
                ln.text = t.strip()
                # the voice key stays: it is the fingerprint of the words and
                # the voice that were spoken (_line_key), so Synthesize voices
                # again only the lines whose words end up different (a re-run
                # to put numbers right re-voiced all 1,074 lines of a drama)
                ln.status = ""
            if ln.review in ("foreign_script", "glossary", "number"):   # checked again below
                ln.review = ""
        fixed = self._fix_wrong_script(p, cancel)
        meant = 0
        if two_pass:
            _emit(progress, 0.97, "Checking that every line means what the original says")
            meant = self._check_meaning(p, list(range(n)), cancel)
        spelled = self._fix_spelling(p, range(n))
        # nouns in Khmer order (ប្រធានថ្នាក់, not the Chinese order ថ្នាក់ប្រធាន)
        ordered = self._fix_word_order(p, range(n))
        # every number of the original in its line, with its exact value
        _emit(progress, 0.985, "Checking the numbers")
        num_fixed, num_bad = self._fix_numbers(p, range(n), cancel)
        # the whole translation, before Fit / Synthesize fit it to the timing:
        # the subtitles show it, and a re-run of Fit starts from it again
        for ln in p.lines:
            ln.full_text = ln.text
        # a line that still does not use a glossary spelling: a person looks
        off_glossary = []
        for i, ln in enumerate(p.lines):
            if ln.text and not ln.review and not self._hand_edited(ln) \
                    and self._glossary_misses(p, ln.source, ln.text):
                ln.review = "glossary"
                off_glossary.append(i + 1)
        foreign = [i + 1 for i, ln in enumerate(p.lines) if ln.review == "foreign_script"]
        if tm is not None and todo:
            tm.add_many(lang, [(p.lines[i].source, p.lines[i].text,
                                self._gender_of(p, p.lines[i])) for i in todo],
                        series=series, origin="ai", src_lang=p.src_lang, rules=_TR_VERSION)
        empty = sum(1 for l in p.lines if not l.text and not l.skip)
        note = f"{n - len(skipped)} lines translated" + (" (translated + reviewed)" if two_pass else "")
        if skipped:
            note += f"; {len(skipped)} left out as you chose"
        if misheard:
            note += (f"; {len(misheard)} misheard words corrected in the transcript ("
                     + ", ".join(misheard[:3]) + (", …" if len(misheard) > 3 else "") + ")")
        if kept:
            note += f"; {len(kept)} kept as you edited them"
        if tm_hits:
            note += f"; {tm_hits} reused from translation memory"
        if stale:
            note += (f"; {stale} remembered translations not reused (they did not follow "
                     "the glossary)")
        def which(nums: list[int]) -> str:
            return (f"line{'s' if len(nums) > 1 else ''} {', '.join(map(str, nums[:10]))}"
                    f"{', …' if len(nums) > 10 else ''}: marked ⚑ in Lines")
        if off_glossary:
            note += (f"; {len(off_glossary)} "
                     + ("line does" if len(off_glossary) == 1 else "lines do")
                     + f" not use the glossary spelling ({which(off_glossary)})")
        if resumed:
            note += f"; {resumed} resumed from an earlier run"
        if fixed:
            note += f"; {fixed} lines re-translated (mixed-in foreign script)"
        if meant:
            note += f"; meaning check corrected {meant} lines"
        if spelled:
            note += f"; spelling corrected in {spelled} lines"
        if ordered:
            note += (f"; {ordered} "
                     + ("noun" if ordered == 1 else "nouns") + " put in Khmer word order")
        if num_fixed:
            note += (f"; {num_fixed} wrong "
                     + ("number" if num_fixed == 1 else "numbers") + " put right")
        if foreign:
            note += (f"; {len(foreign)} "
                     + ("line still has" if len(foreign) == 1 else "lines still have")
                     + f" words the voice cannot read ({which(foreign)})")
        if num_bad:
            nums = [i + 1 for i in num_bad]
            note += (f"; {len(nums)} "
                     + ("line has" if len(nums) == 1 else "lines have")
                     + f" a number to check ({which(nums)})")
        if empty:
            return "warn", note + f"; {empty} lines empty"
        return ("warn" if foreign or off_glossary or num_bad else "done"), note

    def _fix_word_order(self, p: DubProject, idx) -> int:
        """Known compound nouns written in the source's order put in Khmer
        order (km_order.fix: ថ្នាក់ប្រធាន for 班长 → ប្រធានថ្នាក់), in the
        lines and in the glossary (a wrong glossary entry is copied into
        every line). Khmer targets only; lines typed in Lines stay. Returns
        how many were put right in the lines."""
        if _base_lang(p.options.target_lang) != "km":
            return 0
        for g in p.glossary:
            fixed_target, k = km_order.fix(g.source, g.target)
            if k:
                log.info("glossary %s: %s -> %s (Khmer word order)", g.source, g.target,
                         fixed_target)
                g.target = fixed_target
        total = 0
        for i in idx:
            ln = p.lines[i]
            if not ln.text.strip() or ln.skip or self._hand_edited(ln):
                continue
            text, k = km_order.fix(ln.source, ln.text)
            if k:
                log.info("line %d: Khmer word order: %s -> %s", i + 1, ln.text, text)
                ln.text, ln.tts_key = text, ""
                total += k
        return total

    def _fix_numbers(self, p: DubProject, idx, cancel,
                     ask_ai: bool = True) -> tuple[int, list[int]]:
        """Every number of the original in its dub line, with its exact value
        (app.engines.numbers). The AI wrote Khmer numbers in words and got
        about one in five wrong (681 as ប្រាំរយប៉ែតសិបមួយ, 581; 692 as 962;
        "fifty" misspelled). A clear slip is put right in code (the number is
        written as digits); the other lines are translated again with their
        numbers named; what is still wrong is marked ⚑ "number" in Lines.
        Khmer lines get Khmer digits (the voice says them in words made by
        code, speakable_km). Repairs are kept in tr_cache["numbers"].
        Returns (lines put right, lines marked)."""
        src_lang, tgt = p.src_lang or "", p.options.target_lang
        km = _base_lang(tgt) == "km"
        cache = p.tr_cache.setdefault("numbers", {})
        fixed = 0
        bad: list[int] = []
        for i in idx:
            ln = p.lines[i]
            if not ln.text.strip() or ln.skip or self._hand_edited(ln):
                continue
            if km:
                ln.text = nb.km_digits(ln.text)
            if not nb.check(ln.source, ln.text, src_lang)[0]:
                if ln.review == "number":
                    ln.review = ""
                continue
            key = _sha1(ln.source, ln.text, "numbers1")
            done = cache.get(key) or nb.repair(ln.source, ln.text, src_lang, tgt)
            if done and not nb.check(ln.source, done, src_lang)[0]:
                log.info("line %d: number put right: %s -> %s", i + 1, ln.text, done)
                cache[key] = done
                ln.text, ln.tts_key = done, ""
                fixed += 1
                continue
            bad.append(i)
        if bad and ask_ai and self.ai is not None:
            keys = {i: _sha1(p.lines[i].source, p.lines[i].text, "numbers1") for i in bad}
            drafts = [ln.text for ln in p.lines]
            lock = threading.Lock()

            def again(part: list[int]) -> None:
                want = {i: nb.check(p.lines[i].source, p.lines[i].text, src_lang)[0]
                        for i in part}
                extra = ("These lines lost or changed a number of the original. Write "
                         "every number with digits, exactly as the original says it, "
                         "never in words: " + "; ".join(
                             f"line {k + 1}: " + ", ".join(str(v) for v in want[i])
                             for k, i in enumerate(part)) + ".")
                out = self._translate_batch(p, part, drafts, cancel, extra=extra)
                with lock:
                    for i, t in zip(part, out):
                        t = nb.km_digits(t.strip()) if km else t.strip()
                        if (t and not nb.check(p.lines[i].source, t, src_lang)[0]
                                and not self._wrong_script(p, t, p.lines[i].source)):
                            log.info("line %d: number put right by the AI: %s -> %s",
                                     i + 1, p.lines[i].text, t)
                            cache[keys[i]] = t
                            p.lines[i].text, p.lines[i].tts_key = t, ""

            try:
                self._parallel([bad[b:b + 20] for b in range(0, len(bad), 20)], again, cancel)
            except CancelledError:
                raise
            except Exception as exc:
                if ai_unavailable(exc):
                    raise                    # stop: a later run does this pass
                log.warning("Number re-translation failed (%s); the lines are marked", exc)
            still = [i for i in bad if nb.check(p.lines[i].source, p.lines[i].text, src_lang)[0]]
            fixed += len(bad) - len(still)
            bad = still
        for i in bad:
            p.lines[i].review = "number"
        return fixed, bad

    @staticmethod
    def _glossary_misses(p: DubProject, source: str, text: str) -> list[GlossaryTerm]:
        """Glossary terms the original line uses whose spelling its
        translation lacks (spaces ignored). Only terms of two or more
        characters on both sides — a one-letter address word (哥 → បង) is
        not checked — and Latin names as whole words ("Tom", not "Tomorrow")."""
        def squash(t: str) -> str:
            return re.sub(r"[\s\u200b]+", "", t or "").lower()
        src, tgt = squash(source), squash(text)
        out = []
        for g in p.glossary:
            s, t = squash(g.source), squash(g.target)
            if len(s) < 2 or len(t) < 2:
                continue
            if re.fullmatch(r"[a-z0-9' -]+", g.source.strip().lower()):
                found = re.search(rf"\b{re.escape(g.source.strip().lower())}\b",
                                  (source or "").lower()) is not None
            else:
                found = s in src
            if found and t not in tgt:
                out.append(g)
        return out

    @staticmethod
    def _hand_edited(ln: DubLine) -> bool:
        """Typed in the Lines tab: no AI pass rewrites it (translate, meaning
        check, spelling, shortening). Emptying it hands it back to the AI."""
        return ln.text_src == "user" and bool(ln.text.strip())

    def _fix_spelling(self, p: DubProject, idx) -> int:
        """Correct the misspellings the AI keeps making (Khmer: លឺ → ឮ) in
        lines idx, glossary spellings untouched. Returns how many changed."""
        if _base_lang(p.options.target_lang) != "km":
            return 0
        keep = [g.target for g in p.glossary]
        n = 0
        for i in idx:
            ln = p.lines[i]
            if self._hand_edited(ln):
                continue
            t = km_spelling(ln.text, keep)
            if t != ln.text:
                ln.text, ln.tts_key = t, ""
                n += 1
        return n

    def _wrong_script(self, p: DubProject, text: str, source: str = "") -> bool:
        lang = p.options.target_lang
        return foreign_script(text, lang) or latin_leftover(text, lang, source)

    def _fix_wrong_script(self, p: DubProject, cancel, tries: int = 2) -> int:
        """Re-translate lines whose translation mixes in another script
        (e.g. Thai vowels inside Khmer words, leftover Chinese, or English
        words a Khmer voice cannot read) — TTS would read them wrongly.
        Returns how many lines were fixed."""
        bad = [i for i, ln in enumerate(p.lines)
               if ln.text and not self._hand_edited(ln)
               and self._wrong_script(p, ln.text, ln.source)]
        tgt = _lang_name(p.options.target_lang)
        extra = (f"Write every word in {tgt} script only: transliterate names, "
                 "brands and English words by their sound; no Latin letters.")
        fixed = 0
        for _attempt in range(tries):
            if not bad:
                break
            _check_cancel(cancel)
            drafts = [l.text for l in p.lines]
            words = sorted({w for i in bad for w in re.findall(r"[A-Za-z]{3,}", p.lines[i].text)
                            if w.lower() not in p.lines[i].source.lower()})
            named = (f" These words must be written in {tgt} letters: "
                     + ", ".join(words) + ".") if words else ""
            out = self._translate_batch(p, bad, drafts, cancel, extra=extra + named)
            still = []
            for i, t in zip(bad, out):
                if t and not self._wrong_script(p, t, p.lines[i].source):
                    p.lines[i].text = t.strip()
                    fixed += 1
                else:
                    still.append(i)
            bad = still
        if bad:
            # the voice skips or garbles those words: a person must look
            log.warning("%d lines still contain foreign script: %s", len(bad),
                        [i + 1 for i in bad])
            for i in bad:
                p.lines[i].review = "foreign_script"
        return fixed

    def _check_meaning(self, p: DubProject, idx: list[int], cancel,
                       shortened: bool = False, cut: bool = False,
                       uncut: dict[int, str] | None = None) -> int:
        """Back-translation check ("high" quality): the AI writes in the
        source language what each dubbed line really says, compares it with
        the original and rewrites only the lines whose meaning is wrong or
        lost something important (a negation, who does what, a question, a
        name, a number). Finished batches are kept in tr_cache["check"].
        Returns how many lines it rewrote; the lines found wrong without a
        usable fix are in self.meaning_refused."""
        self.meaning_refused = set()
        idx = [i for i in idx if p.lines[i].text.strip() and p.lines[i].source.strip()
               and not self._hand_edited(p.lines[i])]
        if not idx:
            return 0
        src = _lang_name(p.src_lang) if p.src_lang else "the source language"
        tgt = _lang_name(p.options.target_lang)
        system = (
            f"You are a bilingual {src}–{tgt} dubbing quality checker. For each "
            f"numbered line you get the ORIGINAL ({src}) and the DUB ({tgt}). "
            f"First write in {src} what the dub line actually tells a viewer (a "
            "literal back-translation). Then compare it with the original: is "
            "the meaning the same? Check negation, who does what to whom "
            "(\"you\" turned into \"they\" or \"he\" is a meaning error), "
            "question vs. statement, names, numbers and the emotion. Other "
            "wording, looser or shorter phrasing and natural pronouns for the "
            "same person are fine"
            + (" (these lines were shortened on purpose to fit their timing). "
               "Judge each word by what it means in this scene: a synonym or a "
               "natural equivalent of an idiom or insult of the same strength "
               "is fine (\"you're a jerk\" = \"you're a bad / rotten person\"). "
               "Flag a shortened line only when a fact is lost or changed: a "
               "negation, an action or thing (\"synced and locked\" losing "
               "\"locked\"), who does what, question vs. statement, a name or a "
               "number, or a whole statement or question of the original left "
               "out (\"No, he thought I meant a sports car. The system is on\" "
               "cut to \"The system is on\"). What the voice and the picture "
               "carry is no lost statement: an interjection, a surprised echo "
               "question before the real point (他说什么 / \"what did he say?!\"), "
               "a pronoun the language does not need, a filler, a short form of "
               "a name or number — leaving those out is fine."
               + _brief_ok(p.options.target_lang)
               + " Your fix must fit the line's \"max N chars\""
               if shortened and not cut else "")
            + (" (these lines were too long to be spoken in their time even "
               "after rewording, so words were cut on purpose, as the last "
               "resort). A lost detail, side remark or secondary statement is "
               "expected here and fine. Flag a line only when what it still "
               "says is DIFFERENT from the original or false: a flipped "
               "negation, the wrong person doing something, a wrong name or "
               "number, a question turned into a statement, or nothing left of "
               "the line's main point. Your fix must fit the line's \"max N "
               "chars\"" if cut else "")
            + " — flag only real meaning errors or an important missing part. "
            "Words that are not in this line's original (a piece of the "
            "previous or next line) are an error too: the fix drops them. "
            "A line without a politeness word (e.g. Khmer បាទ/ចាស) is fine; "
            "never add one the original does not need. "
            "Judge the words only: a fix must say what THIS line's ORIGINAL "
            "says. A line that seems to belong to another speaker or scene is "
            "not a translation error: mark it ok. Never change an address "
            "word (bro, sis, sir…) by who you think speaks — it names the "
            "listener. Which word a line uses for \"you\" or \"I\" is not "
            "judged here: Khmer speakers say \"you\" and \"I\" with kinship "
            "words and titles (ក្មួយ, បង, អូន, មីង, ពូ, លោក, អ្នក, ឯង, ខ្ញុំ, "
            "អញ…), so ក្មួយ for 你 adds no word and is no error — never swap "
            "one for another. The ORIGINAL is a speech-recognition "
            "transcript: a dub that renders the evident meaning of a misheard "
            "word (\"shield\" for \"shill\") is right; do not force the misheard "
            "word. "
            + ("A noun phrase left in the original's word order (the "
               "describing word before the noun, as in Chinese or English: "
               "ថ្នាក់ប្រធាន for 班长 / \"class president\") says something else "
               "in Khmer: flag it, the fix puts the noun first (ប្រធានថ្នាក់). "
               if _base_lang(p.options.target_lang) == "km" else "")
            + "Numbers written differently but equal (100,000,000 = 100 "
            "million) are fine. "
            f"For a flagged line write a corrected {tgt} line that follows the "
            "rules and stays within its budget. Respond with ONLY one JSON "
            "object: {\"rows\": [{\"n\": <number>, \"back\": <back-translation>, "
            "\"ok\": true|false, \"kind\": \"wrong\"|\"missing\" (when not ok: "
            "wrong = the dub says something different or false; missing = "
            "only something left out), \"problem\": <short, when not ok>, "
            "\"fix\": <corrected line, when not ok>}]}\nRules:\n" + self._rules(p))
        cache = p.tr_cache.setdefault("check", {})
        # the key names the rules the line was checked with: a rule change
        # re-checks every line (the first version swapped good short lines
        # for longer ones, the second rewrote lines to fit a speaker label)
        kind = f"{'cut' if cut else 'short' if shortened else 'full'}{_CHECK_VERSION}"
        shortened = shortened or cut
        fixed = [0]
        lock = threading.Lock()

        def check(part: list[int]) -> None:
            key = {i: _sha1(p.lines[i].source, p.lines[i].text,
                            self._gender_of(p, p.lines[i]), kind) for i in part}
            todo = []
            for i in part:
                done = cache.get(key[i])
                if done is None:
                    todo.append(i)
                elif done.startswith(_REFUSED):
                    self.meaning_refused.add(i)
                elif done and done != p.lines[i].text:
                    p.lines[i].text, p.lines[i].tts_key = done, ""
                    with lock:
                        fixed[0] += 1
            if not todo:
                return
            # no story, characters or speaker labels: given them, the check
            # rewrote lines to fit who it thought was speaking
            header = self._context_block(p, story=False)
            prev = [i for i in range(max(0, todo[0] - 4), todo[0])]
            if prev:
                header += "\n\nPrevious lines (context only):\n" + "\n".join(
                    f"{self._speaker_tag(p, i, True)} {p.lines[i].source} => {p.lines[i].text}"
                    for i in prev)
            header += self._next_lines(p, todo, blind=True)
            body = "\n".join(f"{k + 1}. {self._speaker_tag(p, i, True)} ORIGINAL: "
                             f"{p.lines[i].source} || DUB: {p.lines[i].text}"
                             for k, i in enumerate(todo))
            try:
                res = self.ai.chat(
                    feature="translate_polish", json_mode=True, temperature=0.2,
                    max_tokens=6000, cancel=cancel, system=system,
                    messages=[{"role": "user", "content":
                               f"{header}\n\nLines:\n{body}\n\nReturn the JSON now."}])
                data = extract_json(res.text)
            except CancelledError:
                raise
            except Exception as exc:
                if ai_unavailable(exc):
                    raise    # stop: a later run does this pass
                log.warning("Meaning check failed (%s); keeping the lines", exc)
                return
            rows = data.get("rows") if isinstance(data, dict) else data
            for r in rows or []:
                if not isinstance(r, dict):
                    continue
                try:
                    k = int(r.get("n")) - 1
                except (TypeError, ValueError):
                    continue
                if not 0 <= k < len(todo):
                    continue
                i = todo[k]
                ok = r.get("ok")
                bad = ok is False or str(ok).strip().lower() == "false"
                if bad and cut and str(r.get("kind") or "").strip().lower() == "missing":
                    # a last-resort cut leaves things out by design; the check
                    # put "我还等着报到呢" back into a line twice its time
                    log.info("line %d: cut kept (only left out: %s)", i + 1,
                             str(r.get("problem") or "")[:120])
                    bad = False
                fix = strip_speaker_name(_clean_line(str(r.get("fix") or "")),
                                         p.lines[i].source, self._speaker_names(p))
                # a shortened line was made short to fit its time: a fix that
                # is longer than its budget (and than the line) would undo that
                room = max(p.max_chars(i), len(p.lines[i].text))
                # a cut that says something wrong: the checker's fix, over the
                # budget, still beats going back to the uncut line
                if cut and uncut and i in uncut:
                    room = max(room, len(uncut[i]) - 1)
                if bad and fix and shortened and len(fix) > room:
                    log.info("line %d: meaning fix refused, %d chars > %d (%s): %s",
                             i + 1, len(fix), room, str(r.get("problem") or "")[:120], fix)
                    fix = ""
                if bad and fix and _base_lang(p.options.target_lang) == "km":
                    fix = km_order.fix(p.lines[i].source, nb.km_digits(fix))[0]
                # a fix that loses or changes a number of the original is no fix
                if bad and fix and not nb.keeps(p.lines[i].source, p.lines[i].text, fix,
                                                p.src_lang or ""):
                    log.info("line %d: meaning fix refused, it changes a number: %s", i + 1, fix)
                    fix = ""
                # a copy of another line is never a translation of this one
                # (two lines after "I'm done." once became "I quit" too)
                if bad and fix and any(fix == p.lines[j].text.strip() for j in range(len(p.lines))
                                       if j != i and p.lines[j].text.strip()):
                    log.info("line %d: meaning fix refused, it repeats another line (%s): %s",
                             i + 1, str(r.get("problem") or "")[:120], fix)
                    fix = ""
                if (bad and fix and fix != p.lines[i].text
                        and not self._wrong_script(p, fix, p.lines[i].source)):
                    log.info("line %d meaning fix (%s): %s -> %s", i + 1,
                             str(r.get("problem") or "")[:120], p.lines[i].text, fix)
                    cache[key[i]] = fix
                    p.lines[i].text, p.lines[i].tts_key = fix, ""
                    # the fix itself counts as checked
                    cache[_sha1(p.lines[i].source, fix, self._gender_of(p, p.lines[i]),
                                kind)] = fix
                    with lock:
                        fixed[0] += 1
                elif bad:
                    cache[key[i]] = _REFUSED + p.lines[i].text
                    self.meaning_refused.add(i)
                else:
                    cache[key[i]] = p.lines[i].text
            with lock:
                p.save()

        # a few batches at a time (_AI_WORKERS): each line is its own
        self._parallel([idx[b0:b0 + 20] for b0 in range(0, len(idx), 20)],
                       check, cancel)
        if fixed[0]:
            LOG_BUS.publish(f"[dub] meaning check rewrote {fixed[0]} of {len(idx)} lines")
        return fixed[0]

    def _rules(self, p: DubProject) -> str:
        tgt = _lang_name(p.options.target_lang)
        rules = [
            f"Write natural SPOKEN {tgt} dialogue that a voice actor performs "
            "in sync with the picture — translate meaning, intent and "
            "emotion, never word-by-word.",
            "Each line shows its speaker and a budget \"max N chars\": stay "
            "within it so the line fits the original timing.",
            "Translate each numbered line on its own: everything that line "
            "says goes into its own line and nothing from a neighbouring "
            "line, even when one sentence runs across two lines.",
            "Match the speaker's gender and the relationship between the "
            "speakers for pronouns, forms of address and politeness.",
            "Address words (bro, sis, sir, ma'am, dude, honey…) name the "
            "LISTENER, never the speaker: a woman saying \"bro\" is talking "
            "to a man. Choose them by the listener's gender, age and "
            "closeness, whoever speaks.",
            "The original is a speech-recognition transcript, so a word can "
            "be misheard: when a word makes no sense in the scene and a "
            "similar-sounding one does (\"a shield our hospital hired\" = "
            "\"a shill\", someone paid to promote), translate the sensible "
            "word; never translate a nonsense word literally.",
            "Use the glossary spellings for every name and term.",
            "Keep exclamations, laughter and short reactions short.",
            "Keep what the line does: a question stays a question, an order "
            "an order, a joke a joke; never drop a negation, an insult or a "
            "name.",
            "Numbers: every number the original says, with exactly its value, "
            "written with the digits 0-9 as the original writes it (681, 3.5, "
            "2026, 50%; a Chinese number such as 六百八十一 as 681) — never "
            "spell a number out in words, never change, round or leave one out. "
            "The voice reads the digits. Khmer digits (៦៨១) are the same "
            "numbers as 0-9 (681).",
            "Output only the spoken words: no speaker name, label or colon "
            "before the line.",
        ]
        if _base_lang(p.src_lang) == "en":
            rules += [
                "English idioms, slang and sarcasm: translate what is meant, "
                "not the words (\"you're a rock star\" = you are amazing, "
                "\"break a leg\" = good luck, \"I'm freaked out\" = I'm scared).",
                "Radio and work talk: \"Roger\" / \"Copy\" = message received, "
                "\"Over\" = your turn to speak — they are not names.",
                "English \"you\" and \"I\" say nothing about rank or closeness: "
                "choose the pronouns from who speaks to whom (Characters and "
                "Relationships) and keep the same pair between the same two "
                "people; \"you guys\" / \"you all\" is plural.",
                "Insults and swearing: a target-language expression of the same "
                "strength — do not soften or drop it.",
            ]
        if _base_lang(p.options.target_lang) == "km":
            rules += [
                "Khmer: male speakers (boys too) answer politely with បាទ, "
                "female speakers with ចាស; choose pronouns from the "
                "Relationships in the story context: a parent calls "
                "themself ម៉ាក់/ប៉ា (or ម៉ែ/ពុក) and the child កូន, and a "
                "child answers ម៉ាក់/ប៉ា — never បង/អូន between parent and "
                "child; បង/អូន for siblings, couples and friends (also when "
                "they quarrel: an insult does not make a couple formal); "
                "លោក/អ្នកស្រី for strangers and superiors; ខ្ញុំ/ឯង/អញ in "
                "anger. Two people keep one matching pair in both directions "
                "for the whole scene: when he says បង/អូន to her, she says "
                "អូន/បង back — never លោក one way and បង the other; "
                "\"you guys\" / \"you all\" = ពួកអ្នក or អ្នកទាំងអស់គ្នា "
                "(ពួកឯង among friends) — ពួកគាត់ / ពួកគេ mean \"they\", "
                "never \"you\". Use everyday colloquial Khmer, not literary or news "
                "style; write numbers as they are spoken.",
                "Khmer script only: the Khmer voice cannot read Latin letters, "
                "so transliterate every name, brand and borrowed word into "
                "Khmer (Thom → ថម, OK → អូខេ).",
                "English address words in Khmer (by the LISTENER): Mr./Sir → "
                "លោក, Mrs./Ma'am → អ្នកស្រី, Miss → កញ្ញា; bro / brother said "
                "to a man who is not a sibling → លោកបង (also when a woman "
                "says it); sis / sister → បង or បងស្រី to an older or equal "
                "woman, អូន to a younger one, as the scene shows; little / "
                "younger brother or sister → ប្អូនប្រុស / ប្អូនស្រី; older / big "
                "brother or sister → បងប្រុស / បងស្រី; dude / man / buddy / "
                "mate → as bro, or left out when the line reads naturally "
                "without it; honey / babe between a couple → អូន (to the "
                "woman) or បង (to the man).",
                "Correct Khmer spelling (e.g. ឮ \"hear\", not លឺ).",
                km_order.rule(p.src_lang),
            ]
            rules += self._term_hints(p)
        if p.options.style.strip():
            rules.append(f"Style: {p.options.style.strip()}")
        return "\n".join(f"- {r}" for r in rules)

    @staticmethod
    def _term_hints(p: DubProject) -> list[str]:
        """How common words of the source genre are said in the target
        language (Chinese dramas → Khmer); also given to the glossary step,
        whose wrong term (神豪系统 → "goddess system") the later passes kept."""
        if _base_lang(p.options.target_lang) != "km" or _base_lang(p.src_lang) != "zh":
            return []
        return [
            "Chinese address words in Khmer (by who is addressed): 师傅 "
            "(a driver, worker or craftsman) → បង or លោក — never គ្រូ, "
            "unless they teach a skill; 老师 → លោកគ្រូ / អ្នកគ្រូ; 阿姨 → "
            "អ្នកមីង; 叔叔 → ពូ; 大哥 / 哥 → បង (បងប្រុស for an older "
            "brother); 姐 → បង (បងស្រី for an older sister); 弟弟 / 妹妹 → "
            "ប្អូនប្រុស / ប្អូនស្រី; 老板 → ថៅកែ; 小姐 → កញ្ញា; 爷爷 / 奶奶 → "
            "លោកតា / លោកយាយ (តា / យាយ for one's own grandparents); a name "
            "with 哥 / 姐 (静哥, 李姐) → បង + the name (បងជីង); a surname "
            "with 先生 / 女士 (顾先生) → លោក / អ្នកស្រី + the surname (លោកគូ).",
            "Chinese drama words in Khmer: 神豪 = a super-rich tycoon "
            "(神豪系统 → ប្រព័ន្ធមហាសេដ្ឋី — never a god or a goddess); "
            "系统 = the System, the game-like voice that sets the hero "
            "tasks (→ ប្រព័ន្ធ); 任务 → បេសកកម្ម (新手任务 → បេសកកម្មដំបូង); "
            "奖励 → រង្វាន់; 报到 said of a student → ចុះឈ្មោះចូលរៀន "
            "(arriving at school), not រាយការណ៍ខ្លួន; 摩的 (motorbike taxi) → "
            "ម៉ូតូឌុប; 机车 → ម៉ូតូ; 顶配 → ម៉ូដែលពេញលេញ / កំពូល; 你别说 = "
            "\"don't tell me…\" (disbelief) or \"I must say…\", as the "
            "scene shows — never \"don't speak\"; 呵 (a scoff) → ហឺ.",
            km_order.hints(),
        ]

    def _context_block(self, p: DubProject, story: bool = True) -> str:
        """Story, characters and glossary for a prompt; `story=False` gives
        the glossary only (for a check that must judge words, not who speaks)."""
        parts = []
        if story and p.story_context:
            parts.append(f"Story context:\n{p.story_context}")
        chars = "; ".join(
            f"{c.id} = {c.label()} ({'male' if c.gender == 'M' else 'female'})"
            for c in p.characters) if story else ""
        if chars:
            parts.append(f"Characters: {chars}")
        gl = [g for g in p.glossary if g.source.strip() and g.target.strip()]
        if gl:
            parts.append("Glossary (source → target):\n" + "\n".join(
                f"{g.source} → {g.target}" + (f" ({g.note})" if g.note else "")
                for g in gl[:150]))
        return "\n\n".join(parts)

    @staticmethod
    def _other_voice(p: DubProject, ln: DubLine) -> bool:
        """The line is voiced with the other gender than its character:
        someone else's line filed under that character."""
        ch = p.character(ln.speaker)
        return bool(ln.gender and ch and ln.gender != ch.gender)

    def _speaker_tag(self, p: DubProject, i: int, blind: bool = False) -> str:
        """The line's speaker, voice gender and budget; `blind`: the budget
        only (the meaning check rewrote lines to fit a speaker label)."""
        if blind:
            return f"[max {p.max_chars(i)} chars]"
        ln = p.lines[i]
        ch = p.character(ln.speaker)
        # "[Tom (F)]" makes the AI write Tom's male forms (បាទ)
        who = ("another speaker" if self._other_voice(p, ln)
               else ch.label() if ch else ln.speaker)
        g = ln.gender_of(ch)
        return f"[{who} ({g}) | max {p.max_chars(i)} chars]"

    def _speaker_names(self, p: DubProject) -> set[str]:
        """Lower-case ids, names and glossary spellings of the characters
        (to spot a name the AI wrote before a line)."""
        names = {n.strip().lower() for c in p.characters
                 for n in (c.id, c.name, c.label()) if n and n.strip()}
        for g in p.glossary:
            if g.source.strip().lower() in names and g.target.strip():
                names.add(g.target.strip().lower())
        return names

    def _next_lines(self, p: DubProject, idx: list[int], k: int = 3,
                    blind: bool = False) -> str:
        """The k source lines after a batch, as look-ahead context: the
        answer often decides what a line means (a question, sarcasm, who
        "you" is)."""
        last = idx[-1]
        nxt = [i for i in range(last + 1, min(len(p.lines), last + 1 + k))]
        if not nxt:
            return ""
        return ("\n\nNext lines (context only — do not translate them):\n"
                + "\n".join(f"{self._speaker_tag(p, i, blind)} {p.lines[i].source}"
                            for i in nxt))

    def _translate_batch(self, p: DubProject, idx: list[int],
                         drafts: list[str], cancel, extra: str = "") -> list[str]:
        src = _lang_name(p.src_lang) if p.src_lang else "the source language"
        tgt = _lang_name(p.options.target_lang)
        system = (f"You are an award-winning film and drama dubbing "
                  f"translator from {src} into {tgt}.\nRules:\n"
                  + self._rules(p))
        # the lines before (batches are translated a few at a time: a line
        # still being translated in another batch is shown in the original)
        prev = [i for i in range(max(0, idx[0] - 6), idx[0])
                if drafts[i] or (not p.lines[i].skip and p.lines[i].source.strip())]
        header = self._context_block(p)
        if prev and all(drafts[i] for i in prev):
            header += "\n\nPrevious lines (already translated, for context " \
                      "only — do not output them):\n" + "\n".join(
                          f"{self._speaker_tag(p, i)} {p.lines[i].source} => "
                          f"{drafts[i]}" for i in prev)
        elif prev:
            header += "\n\nPrevious lines (for context only — do not output them; " \
                      "the ones without \"=>\" are being translated elsewhere):\n" + "\n".join(
                          f"{self._speaker_tag(p, i)} {p.lines[i].source}"
                          + (f" => {drafts[i]}" if drafts[i] else "") for i in prev)
        header += self._next_lines(p, idx)
        header += self._notes_block(idx)
        items = [f"{self._speaker_tag(p, i)} {p.lines[i].source}" for i in idx]
        return self._numbered(
            "translate_draft", system,
            header + f"\n\nTranslate these lines into {tgt}. Output only the "
            "translated dialogue (no speaker tags or names, no brackets)."
            + (f"\n{extra}" if extra else ""), items,
            fallback=[p.lines[i].source for i in idx], cancel=cancel,
            sources=[p.lines[i].source for i in idx], names=self._speaker_names(p))

    def _review_batch(self, p: DubProject, idx: list[int], drafts: list[str],
                      done: list[str], cancel) -> list[str]:
        tgt = _lang_name(p.options.target_lang)
        system = (f"You are the senior {tgt} dubbing script editor. You get "
                  "each original line and a draft translation. Return the "
                  "best final version of every line: fix mistranslations and "
                  "omissions, make it sound like real people talking, keep "
                  "names/pronouns consistent, and respect each budget. If "
                  "a draft is already perfect, return it unchanged.\nRules:\n"
                  + self._rules(p))
        header = self._context_block(p)
        prev = [i for i in range(max(0, idx[0] - 4), idx[0]) if done[i]]
        if prev:
            header += "\n\nPrevious lines (final, context only):\n" + "\n".join(
                f"{self._speaker_tag(p, i)} {p.lines[i].source} => {done[i]}"
                for i in prev)
        header += self._next_lines(p, idx)
        header += self._notes_block(idx)
        items = [f"{self._speaker_tag(p, i)} ORIGINAL: {p.lines[i].source} "
                 f"|| DRAFT: {drafts[i]}" for i in idx]
        return self._numbered(
            "translate_polish", system,
            header + "\n\nReturn only the final " + tgt
            + " line for each number (no speaker names).", items,
            fallback=[drafts[i] for i in idx], cancel=cancel,
            sources=[p.lines[i].source for i in idx], names=self._speaker_names(p))

    @staticmethod
    def _parallel(items: list, fn, cancel, say=None) -> list:
        """fn(item) for every item, up to _AI_WORKERS at a time (each asks the
        AI on its own), the results in the items' order. A failure stops the
        items not started yet (the finished ones keep what they saved) and is
        raised, the first real one before a Cancel it caused. `say(done,
        item)`: progress, called from this thread only, with the number of
        items finished and the next item."""
        items = list(items)
        if not items:
            return []
        workers = min(_AI_WORKERS, len(items))
        if workers <= 1:
            out = []
            for k, item in enumerate(items):
                _check_cancel(cancel)
                if say is not None:
                    say(k, item)
                out.append(fn(item))
            return out
        results: list = [None] * len(items)
        state = {"next": 0, "done": 0}
        lock = threading.Lock()
        stop = threading.Event()
        errors: list[BaseException] = []

        def worker() -> None:
            while not stop.is_set():
                with lock:
                    if state["next"] >= len(items):
                        return
                    k = state["next"]
                    state["next"] += 1
                try:
                    _check_cancel(cancel)
                    results[k] = fn(items[k])
                except BaseException as exc:      # noqa: BLE001 - raised below
                    with lock:
                        errors.append(exc)
                    stop.set()
                    return
                with lock:
                    state["done"] += 1

        threads = [threading.Thread(target=worker, name="dub-ai", daemon=True)
                   for _ in range(workers)]
        for t in threads:
            t.start()
        shown = -1
        while True:
            alive = [t for t in threads if t.is_alive()]
            done = state["done"]
            if say is not None and done != shown and done < len(items):
                shown = done
                say(done, items[done])
            if not alive:
                break
            alive[0].join(0.2)
        if errors:
            raise next((e for e in errors if not isinstance(e, CancelledError)), errors[0])
        _check_cancel(cancel)
        return results

    def _numbered(self, feature: str, system: str, header: str,
                  items: list[str], fallback: list[str], cancel,
                  sources: list[str] | None = None,
                  names: set[str] | None = None) -> list[str]:
        """Numbered-list call; on count mismatch retries once, then splits
        the batch in halves down to single lines. A reply that numbers its
        lines twice ("1. 1. text") loses the extra number, unless the
        source line itself starts with a number."""
        n = len(items)
        body = "\n".join(f"{k + 1}. {t}" for k, t in enumerate(items))
        prompt = (f"{header}\n\nLines:\n{body}\n\nReturn EXACTLY {n} lines "
                  f"formatted \"N. text\" (1 to {n}), same order, nothing else.")

        def clean(t: str, k: int) -> str:
            t = _clean_line(t)
            src = sources[k] if sources and k < len(sources) else ""
            if not re.match(r"\s*[\d០-៩]", src):
                t = re.sub(r"^\s*\d{1,4}\s*[.)]\s+", "", t)
            return strip_speaker_name(t, src, names) if names else t

        for attempt in range(2):
            _check_cancel(cancel)
            res = self.ai.chat(feature=feature,
                               messages=[{"role": "user", "content": prompt}],
                               system=system,
                               temperature=0.4 if attempt == 0 else 0.1,
                               max_tokens=6000, cancel=cancel)
            parsed = _parse_numbered_list(res.text, n)
            if parsed is not None:
                return [clean(t, k) or f for k, (t, f) in enumerate(zip(parsed, fallback))]
            log.warning("dub %s: numbered list mismatch (%d lines, attempt %d)",
                        feature, n, attempt + 1)
        if n == 1:
            text = re.sub(r"^\s*1[.)]\s*", "", (res.text or "").strip().splitlines()[0]
                          if (res.text or "").strip() else "")
            return [clean(text, 0) or fallback[0]]
        h = n // 2
        s = sources or [""] * n
        return (self._numbered(feature, system, header, items[:h], fallback[:h], cancel,
                               s[:h], names)
                + self._numbered(feature, system, header, items[h:], fallback[h:], cancel,
                                 s[h:], names))

    # -- step 7: fit --------------------------------------------------------
    def _step_fit(self, p: DubProject, progress, cancel):
        """Make every line fit its time, in this order: spoken up to the
        comfortable speed-up (COMFORT_TEMPO, 1.4x) → said more briefly with
        the same meaning → worded more tightly, every statement kept (or cut,
        allow_cuts) → spoken faster still, only as much as needed, up to
        max_tempo (2.0x) → the next line starts a little late.

        Rewording comes before extra speed: with speed first (2026-11-15) a
        drama's dub had 12 lines at 1.6-2.0x where the dub of 2026-11-11 had
        said them more briefly at 1.4x — the user found the older one better
        (2026-11-16)."""
        lang = p.options.target_lang
        tempo = max(1.0, p.options.max_tempo)      # the last resort
        easy = p.comfort_tempo()                    # what lines are fitted for
        cuts = p.options.allow_cuts
        # every run starts from the whole translations: a second run used to
        # shorten the already shortened lines again (the caches give the same
        # answers for the same lines, so this costs nothing)
        for ln in p.lines:
            if ln.full_text and ln.text != ln.full_text and not self._hand_edited(ln):
                ln.text, ln.tts_key = ln.full_text, ""

        def measure(speed: float) -> list[int]:
            bad = []
            for i, ln in enumerate(p.lines):
                ln.needed = round(speech_seconds(ln.text, lang), 2)
                fits = ln.needed <= p.slot(i) * speed
                ln.status = "ok" if fits else "overlong"
                if not fits and ln.text:
                    bad.append(i)
            return bad

        # 1. a line fits when it can be spoken up to the comfortable speed-up
        #    (1.4x) faster: those keep every word
        bad = measure(easy)
        before ={i: len(p.lines[i].text) for i in bad}
        meant = condensed = 0
        changed: list[int] = []
        squeezed: list[int] = []
        if bad:
            # 2. too long at the comfortable speed: the same meaning in fewer words
            _emit(progress, 0.1, f"Rewording {len(bad)} long lines more briefly")
            changed = self._ai_shorten(p, {i: p.max_chars(i) for i in bad}, cancel)
            # 3. still too long (Chinese → Khmer: up to 1.7x its time): graded
            #    versions, the longest that fits is kept. allow_cuts: down to
            #    the line's main point, and the meaning check only stops a
            #    cut that turns the line false. Otherwise (the default) only
            #    tighter wording with every statement and question kept, and
            #    a version that lost one goes back to the reworded line —
            #    which is then spoken faster (up to max_tempo) to fit
            far = [i for i in bad if speech_seconds(p.lines[i].text, lang)
                   > p.slot(i) * easy * 1.05]
            if far:
                _emit(progress, 0.35, (f"Cutting words from {len(far)} lines still too long"
                                       if cuts else
                                       f"Tightening {len(far)} lines still too long"))
                kept = {i: p.lines[i].text for i in far}
                squeezed = self._ai_shorten(p, {i: p.max_chars(i) for i in far}, cancel,
                                            condense=True)
                if squeezed and p.options.quality == "high":
                    self._check_meaning(p, squeezed, cancel, shortened=True, cut=cuts,
                                        uncut=kept)
                    for i in self.meaning_refused & set(squeezed):
                        p.lines[i].text, p.lines[i].tts_key = kept[i], ""
                        squeezed.remove(i)
                condensed = len(squeezed)
            reworded_only = sorted(set(changed) - set(squeezed))
            changed = sorted(set(changed) | set(squeezed))
            if reworded_only and p.options.quality == "high":
                _emit(progress, 0.6, "Checking the meaning of the reworded lines")
                meant = self._check_meaning(p, reworded_only, cancel, shortened=True)
        # 4. what is still longer than the comfortable speed allows is spoken
        #    faster, only as much as needed, up to max_tempo
        fast = [i for i, ln in enumerate(p.lines)
                if ln.text and speech_seconds(ln.text, lang) > p.slot(i) * easy * 1.02]
        still = len(measure(tempo))
        sped = sum(1 for i, ln in enumerate(p.lines)
                   if ln.text and ln.status == "ok" and p.slot(i) * 1.02 < ln.needed
                   <= p.slot(i) * easy * 1.02)
        parts = []
        if sped:
            parts.append(f"{sped} lines fit when spoken up to {easy:.1f}x faster")
        reworded = len(set(changed) - set(squeezed))
        if reworded:
            cut = [1 - len(p.lines[i].text) / n for i, n in before.items()
                   if n and i in changed and i not in squeezed]
            parts.append(f"{reworded} reworded more briefly, meaning kept "
                         f"({100 * sum(cut) / max(1, len(cut)):.0f}% shorter on average)")
        if condensed:
            parts.append(f"{condensed} had words cut as the last resort" if cuts else
                         f"{condensed} worded more tightly, every statement kept")
        if meant:
            parts.append(f"{meant} rewritten by the meaning check")
        n_fast = sum(1 for i in fast if p.lines[i].status == "ok")
        if n_fast and tempo > easy:
            parts.append(f"{n_fast} spoken faster than {easy:.1f}x to fit (up to {tempo:.1f}x)")
        if changed and p.options.subtitle_full:
            parts.append("the subtitles keep the whole translations")
        note = "; ".join(parts) or "All lines fit their timing"
        if still:
            return "warn", note + (f"; {still} still long: spoken at {tempo:.1f}x, the next "
                                   "line starts a little later (edit them in Lines)")
        return "done", note

    # -- step 8: synthesize -------------------------------------------------
    def voice_params(self, p: DubProject, ch: Character,
                     ln: DubLine | None = None) -> dict:
        """Resolved TTS parameters for a character (for one of its lines, when
        given: a line can be voiced with the other gender)."""
        lang = _base_lang(p.options.target_lang)
        gender = ln.gender_of(ch) if ln is not None else ch.gender
        engine = (ch.engine or "auto").lower()
        if engine == "auto":
            engine = (best_clone_engine(lang)
                      if p.options.voice_mode == "clone" else "") or "edge"
        if engine in ("f5tts", "voxcpm", "cosyvoice", "edge+clone", "edge+seedvc") \
                and not os.path.isfile(ch.ref_wav):
            log.warning("%s has no voice sample; using a preset voice",
                        ch.label())
            engine = "edge"
        male, female = _EDGE_DEFAULTS.get(lang, _EDGE_DEFAULTS["en"])
        # "pair" mode: every man Piseth, every woman Sreymom (for Khmer),
        # switching line by line; a voice typed for a character still wins
        preset = ch.voice.strip() or (male if gender == "M" else female)
        pitch = ch.pitch or "+0Hz"
        vary = p.options.vary_pitch or p.options.voice_mode == "preset"
        if engine == "edge" and vary and pitch in ("+0Hz", "0Hz", ""):
            # keep same-gender characters apart (pair mode: only when asked)
            same = sorted((c for c in p.characters if c.gender == ch.gender
                           and not c.voice.strip()),
                          key=lambda c: -c.talk_seconds)
            if ch in same:
                pitch = _PITCH_SPREAD[same.index(ch) % len(_PITCH_SPREAD)]
        params = {"engine": engine, "rate": ch.rate or "+0%", "pitch": pitch,
                  "volume": ch.volume or "+0%", "ref_text": "", "clone_ref": "",
                  "converter": ""}
        if engine in ("f5tts", "voxcpm", "cosyvoice"):
            params.update(voice=ch.ref_wav, ref_text=ch.ref_text)
        elif engine == "edge+clone":
            params.update(engine="edge", voice=preset, clone_ref=ch.ref_wav,
                          converter="openvoice")
        elif engine == "edge+seedvc":
            params.update(engine="edge", voice=preset, clone_ref=ch.ref_wav,
                          converter="seedvc")
        elif engine == "edge":
            params.update(voice=preset)
        else:
            params.update(voice=ch.voice.strip())
        return params

    def _base_sample(self, p: DubProject, voice: str, cancel) -> str:
        """A ~10 s clip of an Edge voice in the target language (source
        timbre for OpenVoice)."""
        out = p.sub("voices", f"base_{_safe_name(voice)}.wav")
        if os.path.isfile(out):
            return out
        text = " ".join(l.text for l in p.lines if l.text)[:220] or "..."
        ap.synthesize(text, out, engine="edge", voice=voice, cancel=cancel)
        return out

    def synth_text(self, p: DubProject, ch: Character, text: str, out_wav: str,
                   cancel: threading.Event | None = None,
                   ln: DubLine | None = None, speed: float = 1.0) -> str:
        """Speak `text` in the character's voice (raw, not time-fitted);
        with `ln`, in the voice that line is meant to have. `speed` > 1 makes
        an Edge voice (also the base of a clone) speak that much faster."""
        prm = self.voice_params(p, ch, ln)
        if _base_lang(p.options.target_lang) == "km":
            text = speakable_km(text)          # "H2" said as អេចពីរ, not skipped
        if speed != 1.0 and prm["engine"] == "edge":
            prm["rate"] = _faster_rate(prm["rate"], speed)
        if prm["clone_ref"]:
            tmp = out_wav + ".base.wav"
            try:
                ap.synthesize(text, tmp, engine="edge", voice=prm["voice"],
                              rate=prm["rate"], pitch="+0Hz",
                              volume=prm["volume"], cancel=cancel)
                if prm.get("converter") == "seedvc":
                    return convert_voice_seedvc(tmp, prm["clone_ref"], out_wav, cancel)
                base = self._base_sample(p, prm["voice"], cancel)
                return convert_voice(tmp, base, prm["clone_ref"], out_wav, cancel)
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return ap.synthesize(text, out_wav, engine=prm["engine"],
                             voice=prm.get("voice", ""), rate=prm["rate"],
                             pitch=prm["pitch"], volume=prm["volume"],
                             config=self.config, cancel=cancel,
                             ref_text=prm["ref_text"])

    @staticmethod
    def _gap_before(p: DubProject, i: int) -> float:
        """Silence kept before dubbed line i: _LINE_GAP, or less where the
        actors answer each other with no pause (a fixed 0.08 s after every
        line pushed a fast exchange 0.5 s behind the picture)."""
        if i <= 0 or i >= len(p.lines):
            return _LINE_GAP
        return min(_LINE_GAP, max(0.02, p.lines[i].start - p.lines[i - 1].end))

    def _room(self, p: DubProject, i: int) -> float:
        """Seconds line i can speak before the next line starts."""
        ln = p.lines[i]
        nxt = p.lines[i + 1].start if i + 1 < len(p.lines) else max(p.duration, ln.end + 2.0)
        return max(0.3, nxt - ln.start - self._gap_before(p, i + 1))

    def _too_long(self, p: DubProject, speech: dict[int, np.ndarray],
                  sr: int) -> dict[int, int]:
        """{line: character target} for lines that, spoken up to the
        comfortable speed-up (1.4x), would still run into the next line —
        counting a late start left by the lines before (in a fast exchange
        small overruns add up: 22 lines once started up to 1.8 s late).
        Those are said more briefly first; only what is then still too long
        is spoken faster, up to max_tempo (_respeak_faster, _place_lines)."""
        easy = p.comfort_tempo()
        if _squeeze(p.options.max_tempo) > easy:
            # speed to fit: placement never leaves a line late while it can go
            # faster, so "a little late at 1.4x" would be spoken at 1.6x —
            # those are said more briefly instead; a small margin is not worth
            # new words (Ep2: "途！我受不了了！" at 1.61x where the user's
            # preferred dub said "ទ្រាំលែងបានហើយ!")
            squeeze, late_ok = min(_squeeze(p.options.max_tempo), easy + _FAST_MARGIN), 0.0
        else:
            squeeze, late_ok = easy, _LATE_OK
        out: dict[int, int] = {}
        self._late_room = {}
        cursor = 0.0
        for i in sorted(speech):
            ln = p.lines[i]
            real = speech[i].size / sr
            begin = max(ln.start, cursor + self._gap_before(p, i)) if cursor else ln.start
            nxt = (p.lines[i + 1].start if i + 1 < len(p.lines)
                   else max(p.duration, ln.end + 2.0))
            room = max(0.3, nxt - self._gap_before(p, i + 1) - begin)
            dur = real / min(squeeze, max(1.0, real / max(0.3, self._room(p, i))))
            # without speed to fit the next line may start a little late
            # (_LATE_OK): cutting to the last tenth of a second in a fast
            # exchange turned "my son twists the throttle and goes" into "my
            # son left"
            if dur > room + late_ok + 1e-6:
                allowed = room + late_ok / 2
                out[i] = max(4, int(len(ln.text) * allowed * squeeze / max(real, 0.1) * 0.95))
                self._late_room[i] = allowed
            cursor = begin + dur
        return out

    def _respeak_faster(self, p: DubProject, speech: dict[int, np.ndarray],
                        sr: int, workers: int, cancel,
                        applied: dict[int, float] | None = None) -> int:
        """Lines whose speech is longer than their room: let the TTS speak
        them faster itself (Edge's rate is exact: +30% = 1.30x) instead of
        time-stretching the audio afterwards — natural fast speech rather
        than the warble of a stretch. Edge voices only (also as the base of a
        voice clone); other engines keep the stretch in _place_lines.
        Replaces the entries of `speech`; returns how many lines changed."""
        squeeze = _squeeze(p.options.max_tempo)
        todo: list[tuple[int, float]] = []
        for i, data in speech.items():
            need = data.size / sr / self._room(p, i)
            if need <= 1.03:
                continue
            ln = p.lines[i]
            ch = p.character(ln.speaker) or Character(id=ln.speaker)
            if self.voice_params(p, ch, ln)["engine"] != "edge":
                continue
            # a touch faster than needed: the rate is exact, the trim is not
            todo.append((i, round(min(squeeze, need * 1.03), 2)))
        changed = [0]
        lock = threading.Lock()

        def job(item: tuple[int, float]) -> None:
            i, speed = item
            if cancel is not None and cancel.is_set():
                return
            ln = p.lines[i]
            ch = p.character(ln.speaker) or Character(id=ln.speaker)
            path = p.sub("tts", f"fast_{i:05d}.wav")
            key = f"{self._line_key(p, ln)}@{speed}"
            data = None
            # Edge drops requests in bursts ("No audio was received", at any
            # rate, 2026-11-16: 6 lines of one dub): tried again like the
            # normal voicing, else the line was time-stretched (a warble)
            for attempt, wait in enumerate((1.5, 3.0, 0.0)):
                try:
                    if not (ln.fast_key == key and os.path.isfile(path)):
                        self.synth_text(p, ch, ln.text, path, cancel, ln=ln, speed=speed)
                        ln.fast_key = key
                    data = ap.trim_silence(ap._read_wav_as_int16_mono(path, sr), sr)
                    break
                except CancelledError:
                    return
                except Exception as exc:
                    log.warning("line %d: faster re-voicing attempt %d failed (%s)", i + 1,
                                attempt + 1, getattr(exc, "en", exc))
                    ln.fast_key = ""
                    if not wait or (cancel is not None and cancel.wait(wait)):
                        return               # keep the normal-speed clip (stretched later)
                    if cancel is None:
                        time.sleep(wait)
            if data is None:
                return
            if 0 < data.size < speech[i].size:
                with lock:
                    # what the voice really gained: Edge's rate stops at
                    # +100 %, so a voice already set faster gets less than
                    # asked, and the stretch later must make up the rest
                    got = min(speed, speech[i].size / data.size)
                    speech[i] = data
                    changed[0] += 1
                    if applied is not None:
                        applied[i] = got         # the stretch later may add only the rest

        if workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(job, todo))
        else:
            for item in todo:
                job(item)
        return changed[0]

    def _ai_shorten(self, p: DubProject, targets: dict[int, int], cancel,
                    condense: bool = False,
                    rooms: dict[int, float] | None = None) -> list[int]:
        """Ask the AI to tighten lines to about targets[i] characters without
        changing their meaning; `condense`: a line still too long is said
        again with only its core (`rooms`: estimated seconds each may take,
        when not its slot at max_tempo). Answers are kept in
        tr_cache["shorten"]: a re-run gives the same short lines (it once
        dropped "locked" from a line that had been fine). Returns the lines
        whose text changed."""
        # a line typed in Lines keeps its words: spoken faster instead
        targets = {i: n for i, n in targets.items() if not self._hand_edited(p.lines[i])}
        cache = p.tr_cache.setdefault("shorten", {})
        # the condense policy is part of the key: a line cut to its main point
        # is never reused once cuts are not allowed; the prompts' version too
        # (answers of older prompts are asked again)
        policy = "condense6" if p.options.allow_cuts else "tighten1"
        key = {i: _sha1(p.lines[i].source, p.lines[i].text, targets[i],
                        self._gender_of(p, p.lines[i]), _SHORTEN_VERSION,
                        *((policy, round((rooms or {}).get(i, 0.0), 2)) if condense else ()))
               for i in targets}
        changed: list[int] = []
        for i in sorted(targets):
            t = cache.get(key[i])
            if t is not None and t != p.lines[i].text:
                p.lines[i].text, p.lines[i].tts_key = t, ""
                changed.append(i)
        todo = {i: n for i, n in targets.items() if key[i] not in cache}
        if todo:
            changed += self._ai_shorten_new(p, todo, key, cache, cancel, condense, rooms)
            p.save()
        self._fix_spelling(p, changed)
        return sorted(changed)

    def _condense_prompt(self, p: DubProject) -> str:
        """The second shortening pass. Not the full translation rules: those
        ("everything the line says…", "never drop…") made it return lines
        nearly unchanged (4% shorter on a Chinese drama).

        Without allow_cuts only two versions, both keeping every statement:
        the four graded ones went down to the line's main point, and when
        none fitted the shortest was taken — on a Chinese drama the dub lines
        had lost 32 % of their Khmer ("…能不能快点，我还等着报到呢" lost
        "I'm still waiting to register")."""
        tgt = _lang_name(p.options.target_lang)
        km = _base_lang(p.options.target_lang) == "km"
        if not p.options.allow_cuts:
            return (f"Each {tgt} dubbing line below is too long for its time slot: "
                    "only about \"keep_percent\" % of its current length fits. Write "
                    "TWO new versions of it, the second shorter than the first: (1) "
                    "about nine tenths — every word of meaning kept, only tighter "
                    "wording; (2) as short as it can be while EVERY statement, "
                    "request, order, real question, name, number and negation of the "
                    "line is still there, in the shortest everyday "
                    f"{tgt} words. Never leave out a sentence or a clause that says "
                    "something, never summarise, never change who does what. It is "
                    "fine if version (2) is still longer than what fits: the voice "
                    "will speak it faster. Keep the politeness words"
                    + (" (បាទ/ចាស), Khmer script only, colloquial Khmer" if km else "")
                    + ". " + _brief_guide(p.options.target_lang)
                    + "\nRespond with ONLY one JSON object: {\"rows\": [{\"index\": "
                    "<int>, \"versions\": [<1>, <2>]}]}")
        return (f"Each {tgt} dubbing line below is far too long for its time "
                "slot: only about \"keep_percent\" % of its current length fits. "
                "Write FOUR new versions of it, each shorter than the one "
                "before, so that the longest one that fits can be used and no "
                "more is lost than needed: (1) about nine tenths — every word "
                "of meaning kept, only tighter wording; (2) about three "
                "quarters — every statement and question still there, small "
                "details dropped; (3) about half — the main statements only; "
                "(4) only the line's MAIN point in a few words — its question, "
                "request, order or news, usually the first thing said, plus "
                "what a joke or the next line needs (\"two-seater\" in \"you "
                "said you have a two-seater car\"). Keep only "
                "what matters in the scene: the core fact, request, question or "
                "feeling. Drop details, repetitions, softeners, side remarks and "
                f"secondary clauses; use the shortest everyday {tgt} words. Never "
                "change who does what, a negation, a question, or a name or "
                "number the scene needs; keep the current line's pronouns and "
                "politeness words"
                + (" (បាទ/ចាស), Khmer script only, colloquial Khmer" if km else "")
                + ". Respond with ONLY one JSON object: {\"rows\": [{\"index\": "
                "<int>, \"versions\": [<1>, <2>, <3>, <4>]}]}")

    def _ai_shorten_new(self, p: DubProject, targets: dict[int, int], key: dict[int, str],
                        cache: dict[str, str], cancel, condense: bool = False,
                        rooms: dict[int, float] | None = None) -> list[int]:
        """The AI part of _ai_shorten; every line it answers goes into `cache`
        (unchanged ones too, so they are not asked again)."""
        tgt = _lang_name(p.options.target_lang)
        system = self._condense_prompt(p) if condense else (
                  f"You tighten {tgt} dubbing lines so they can be spoken "
                  "inside their time slot. The MEANING comes first: keep "
                  "every negation, who does what to whom, conditions "
                  "(if/then), numbers and names exactly as in \"original\". "
                  "Keep politeness words (បាទ/ចាស) and the speaker's tone. "
                  "Use shorter everyday words. " + _brief_guide(p.options.target_lang)
                  + "\nIf the line cannot get shorter without changing "
                  "its meaning, return it unchanged or only slightly "
                  "shorter — a slightly long line is better than a wrong "
                  "one. Respond with ONLY one JSON object: {\"rows\": "
                  "[{\"index\": <int>, \"text\": <line, aim for at most "
                  "max_chars characters>}]}\nThe translation rules still hold "
                  "(it once put a misheard word back into a line):\n" + self._rules(p))
        changed: list[int] = []
        idx = sorted(targets)
        lock = threading.Lock()

        def shorten(chunk: list[int]) -> None:
            entries = [{"index": i, "speaker": self._speaker_tag(p, i),
                        "original": p.lines[i].source,
                        "max_chars": targets[i],
                        **({"keep_percent": round(100 * targets[i] / max(1, len(p.lines[i].text)))}
                           if condense else {}),
                        "current": p.lines[i].text} for i in chunk]
            try:
                res = self.ai.chat(
                    feature="translate_polish",
                    messages=[{"role": "user", "content":
                               json.dumps({"rows": entries}, ensure_ascii=False)}],
                    system=system, temperature=0.3, max_tokens=6000,
                    json_mode=True, cancel=cancel)
                data = extract_json(res.text)
                rows = data.get("rows") if isinstance(data, dict) else data
                for r in rows or []:
                    if not isinstance(r, dict):
                        continue
                    try:
                        i = int(r.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if i not in chunk:
                        continue
                    if condense:
                        # graded versions: the longest that fits, else the
                        # shortest (the model cannot count Khmer characters)
                        vs = [strip_speaker_name(_clean_line(str(v or "")), p.lines[i].source,
                                                 self._speaker_names(p))
                              for v in (r.get("versions") or [r.get("text")])]
                        vs = [v for v in vs if v and not self._wrong_script(p, v, p.lines[i].source)]
                        if not p.options.allow_cuts:
                            vs = vs[:2]      # never the main-point-only versions
                        # never a version that loses or changes a number
                        vs = [v for v in vs if nb.keeps(p.lines[i].source, p.lines[i].text,
                                                        v, p.src_lang or "")]
                        # "fits" as the Fit step measures it (spoken up to
                        # the comfortable speed-up): the longest such
                        # version keeps most; what is left is spoken faster
                        room = (rooms or {}).get(i, p.slot(i) * p.comfort_tempo())
                        fit = [v for v in vs if speech_seconds(
                            v, p.options.target_lang) <= room]
                        t = max(fit, key=len) if fit else min(vs, key=len, default="")
                    else:
                        t = _clean_line(str(r.get("text") or ""))
                        t = strip_speaker_name(t, p.lines[i].source, self._speaker_names(p))
                    if _base_lang(p.options.target_lang) == "km":
                        t = km_order.fix(p.lines[i].source, nb.km_digits(t))[0]
                    shorter = not condense or len(t) < len(p.lines[i].text)
                    if (t and t != p.lines[i].text and shorter
                            and not self._wrong_script(p, t, p.lines[i].source)
                            and nb.keeps(p.lines[i].source, p.lines[i].text, t,
                                         p.src_lang or "")):
                        p.lines[i].text, p.lines[i].tts_key = t, ""
                        with lock:
                            changed.append(i)
                    elif t and not nb.keeps(p.lines[i].source, p.lines[i].text, t,
                                            p.src_lang or ""):
                        log.info("line %d: shorter version refused, it loses a number: %s",
                                 i + 1, t)
                    cache[key[i]] = p.lines[i].text
            except CancelledError:
                raise
            except Exception as exc:
                if ai_unavailable(exc):
                    raise    # stop: a later run does this pass
                log.warning("Shorten pass failed (%s)", exc)

        # a few chunks at a time (_AI_WORKERS): each line is its own
        self._parallel([idx[b0:b0 + 25] for b0 in range(0, len(idx), 25)],
                       shorten, cancel)
        return changed

    def _place_lines(self, p: DubProject, speech: dict[int, np.ndarray],
                     buf: np.ndarray, sr: int, errors: dict, cancel,
                     applied: dict[int, float] | None = None) -> tuple[int, float]:
        """Mix every voiced line into `buf` so that no two lines ever sound
        at the same time. Each line starts at its original time, or right
        after the previous line when that one is still speaking; it is sped
        up just as much as it needs to end before the next line — in all at
        most max_tempo, counting what the voice already sped it up
        (`applied`: line → speed; once two speed-ups stacked to ~2x) — and,
        up to the comfortable speed-up, to end inside its own slot
        (lip-sync). A line still too long delays the next one; the delay
        fades out at the next pause. Sets each line's `speed`. Returns
        (lines starting late, largest delay in seconds)."""
        applied = applied or {}
        cursor = 0                      # sample where the previous line ends
        late, max_late = 0, 0.0
        for i, ln in enumerate(p.lines):
            _check_cancel(cancel)
            ln.delay, ln.speed = 0.0, 1.0
            if i in errors:
                ln.status = "failed"
                continue
            raw = speech.get(i)
            if raw is None or raw.size == 0:
                continue
            fit = p.sub("tts", f"fit_{i:05d}.wav")
            ap._write_wav_int16(fit, raw, sr)
            gap = int(self._gap_before(p, i) * sr)
            begin = max(int(ln.start * sr), cursor + gap if cursor else 0)
            nxt = (int(p.lines[i + 1].start * sr) if i + 1 < len(p.lines)
                   else buf.size)
            room = max(0.3 * sr, nxt - begin - int(self._gap_before(p, i + 1) * sr))
            # what is left of max_tempo after the voice's own speed-up
            spoke = max(1.0, applied.get(i, 1.0))
            left = max(1.0, _squeeze(p.options.max_tempo) / spoke)
            if raw.size > room and left > 1.01:
                # one stretch, never faster than needed: enough to end before
                # the next line; more, up to the comfortable speed-up only, to
                # end inside its own slot (squeezing a line to 2x for lip-sync
                # alone, with silence after it, made it sound rushed)
                easy = max(1.0, p.comfort_tempo() / spoke)
                need_room = raw.size / room
                need_slot = raw.size / max(1.0, (p.slot(i) - 0.04) * sr)
                tempo = min(left, max(need_room, min(need_slot, easy)))
                ap.fit_to_duration(fit, raw.size / sr / tempo, min_tempo=1.0,
                                   max_tempo=left)
                data = ap._read_wav_as_int16_mono(fit, sr, cancel=cancel)
            else:
                data = raw
            ln.speed = round(spoke * raw.size / max(1, data.size), 2)
            m = min(data.size, buf.size - begin)
            if m > 0:
                buf[begin:begin + m] += data[:m].astype(np.int32)
            cursor = begin + max(0, m)
            ln.delay = round((begin - int(ln.start * sr)) / sr, 2)
            if ln.delay > 0.25:
                ln.status = "late"
                late += 1
                max_late = max(max_late, ln.delay)
            elif ln.status != "overlong":
                ln.status = "ok"
        return late, max_late

    def _line_key(self, p: DubProject, ln: DubLine) -> str:
        ch = p.character(ln.speaker) or Character(id=ln.speaker)
        prm = self.voice_params(p, ch, ln)
        ref = prm.get("clone_ref") or (prm.get("voice") if prm["engine"] in
                                       ("f5tts", "voxcpm", "cosyvoice") else "")
        mt = os.path.getmtime(ref) if ref and os.path.isfile(ref) else 0
        return _sha1(ln.text, json.dumps(prm, sort_keys=True), mt)

    def _original_voice(self, p: DubProject, i: int, sr: int, cancel) -> np.ndarray:
        """The original speech of line i (the separated vocals, else the
        recogniser's audio) as int16 at `sr`, trimmed: what plays when the
        line could not be voiced. Empty when there is no audio to take."""
        src = p.vocals_wav if p.vocals_wav and os.path.isfile(p.vocals_wav) else p.asr_wav
        if not src or not os.path.isfile(src):
            return np.zeros(0, dtype=np.int16)
        ln = p.lines[i]
        out = p.sub("tts", f"orig_{i:05d}.wav")
        try:
            ap._run_ffmpeg(["-ss", f"{max(0.0, ln.start):.3f}",
                            "-t", f"{max(0.1, ln.end - ln.start):.3f}", "-i", src,
                            "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", out],
                           duration=max(0.1, ln.end - ln.start), cancel=cancel)
            return _trim_silence(ap._read_wav_as_int16_mono(out, sr, cancel=cancel), sr)
        except CancelledError:
            raise
        except Exception as exc:
            log.warning("line %d: could not keep the original voice (%s)", i + 1, exc)
            return np.zeros(0, dtype=np.int16)

    def _step_synthesize(self, p: DubProject, progress, cancel):
        n = len(p.lines)
        recheck = {c.id for c in p.characters if c.gender_recheck}
        regendered = 0
        if recheck:
            # a gender set by hand: which of those characters' lines are
            # someone else's voice is decided again against the new gender
            regendered = self._detect_genders(p, progress, cancel, keep_characters=True,
                                              ai_check=True, only=recheck)
            p.save()
        engines = {self.voice_params(p, c)["engine"] for c in p.characters}
        clone_any = any(self.voice_params(p, c)["clone_ref"] for c in p.characters)
        workers = 4 if engines <= _NETWORK_ENGINES and not clone_any else 1
        done_n = [0]
        lock = threading.Lock()
        errors: dict[int, str] = {}
        waiter = cancel if cancel is not None else threading.Event()

        def raw_path(i: int) -> str:
            return p.sub("tts", f"raw_{i:05d}.wav")

        def job(i: int, count: bool = True, waits=(1.5, 3.0, 6.0)) -> None:
            if cancel is not None and cancel.is_set():
                return
            ln = p.lines[i]
            if not ln.text.strip():
                return
            key = self._line_key(p, ln)
            if ln.tts_key == key and os.path.isfile(raw_path(i)):
                pass
            else:
                ch = p.character(ln.speaker) or Character(id=ln.speaker)
                for attempt, wait in enumerate(waits):   # Edge TTS drops requests in bursts
                    try:
                        self.synth_text(p, ch, ln.text, raw_path(i), cancel, ln=ln)
                        ln.tts_key = key
                        errors.pop(i, None)
                        break
                    except CancelledError:
                        return
                    except Exception as exc:
                        errors[i] = getattr(exc, "en", str(exc))
                        ln.tts_key = ""
                        log.warning("line %d TTS attempt %d failed: %s", i + 1,
                                    attempt + 1, errors[i])
                        if waiter.wait(wait):
                            return
            if not count:
                return
            with lock:
                done_n[0] += 1
                _emit(progress, 0.85 * done_n[0] / max(1, n),
                      f"Voicing line {done_n[0]}/{n}")

        def run(indexes) -> None:
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(job, indexes))
            else:
                for i in indexes:
                    job(i)

        sr = ap._TTS_SR
        # TTS / voice conversion pad ~0.2 s before and ~0.8 s after the
        # speech: only the speech itself counts
        speech: dict[int, np.ndarray] = {}
        floor_of: dict[int, float] = {}     # background level of the padding

        def measure(indexes) -> None:
            for i in indexes:
                if i not in errors and p.lines[i].text.strip() and os.path.isfile(raw_path(i)):
                    raw = ap._read_wav_as_int16_mono(raw_path(i), sr, cancel=cancel)
                    floor_of[i] = ap.noise_floor_db(raw, sr)
                    speech[i] = _trim_silence(raw, sr)

        early: list[int] = []        # reworded while later lines were being voiced
        reworded: set[int] = set()   # every line given to the first rewording so far
        if workers > 1 and p.options.auto_shorten and n >= _OVERLAP_MIN_LINES:
            # the voice service and the AI at work at the same time: lines
            # found too long near the start are reworded while later lines
            # are voiced (a 1-hour dub waited 25 min for its rewordings after
            # its voices were done). Whether a line is too long depends only
            # on the lines before it, so these are the lines the step would
            # pick anyway.
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(job, i) for i in range(n)]
                try:
                    ready = scanned = 0
                    while True:
                        # lines voiced in a row from the start (a failed one
                        # waits for its retry: the lines after it are checked
                        # at the end)
                        while ready < n and futures[ready].done() and ready not in errors:
                            ready += 1
                        finished = all(f.done() for f in futures)
                        if ready - scanned >= _REWORD_EVERY or (finished and ready > scanned):
                            measure(range(scanned, ready))
                            scanned = ready
                            found = {i: c for i, c in self._too_long(
                                         p, {i: s for i, s in speech.items() if i < ready},
                                         sr).items() if i not in reworded}
                            if found:
                                reworded |= set(found)
                                _emit(progress, 0.85 * done_n[0] / max(1, n),
                                      f"Rewording {len(found)} lines that are too long")
                                got = self._ai_shorten(p, found, cancel)
                                if got and p.options.quality == "high":
                                    self._check_meaning(p, got, cancel, shortened=True)
                                early += got
                        if finished or waiter.wait(0.3):
                            break
                except BaseException:
                    for f in futures:
                        f.cancel()          # the voices not started yet
                    raise
        else:
            run(range(n))
        _check_cancel(cancel)
        # Edge TTS fails in bursts ("No audio was received": ~30% of requests
        # for minutes at a time, at any concurrency). Failed lines get more
        # tries spread over ~90 s: each round waits once, then tries every
        # failed line once, so the wait does not grow with their number.
        for wait in (3.0, 6.0, 12.0, 24.0, 48.0):
            if not errors:
                break
            LOG_BUS.publish(f"[dub] {len(errors)} lines not voiced yet; the voice service "
                            f"is dropping requests — trying again in {wait:.0f} s")
            if waiter.wait(wait):
                _check_cancel(cancel)
            for i in sorted(errors):
                job(i, count=False, waits=(0.0,))
            _check_cancel(cancel)
        if errors and len(errors) >= sum(1 for l in p.lines if l.text.strip()):
            first = next(iter(errors.values()))
            raise AppError(f"Every line failed to synthesize: {first}",
                           f"ការបង្កើតសំឡេងបរាជ័យគ្រប់ជួរ៖ {first}")

        # every line measured once (the ones reworded early keep the length
        # they were picked by until they are voiced again)
        measure([i for i in range(n) if i not in speech])
        shortened = 0
        if p.options.auto_shorten:
            # lines whose real speech cannot fit before the next line even at
            # full speed-up: one AI rewrite with the measured length, re-voiced
            targets = {i: c for i, c in self._too_long(p, speech, sr).items()
                       if i not in reworded}
            if targets or reworded:
                changed = list(early)
                if targets:
                    _emit(progress, 0.86, f"Rewording {len(targets)} lines that are too long")
                    later = self._ai_shorten(p, targets, cancel)
                    if later and p.options.quality == "high":
                        self._check_meaning(p, later, cancel, shortened=True)
                    changed = sorted(set(changed) | set(later))
                if changed:
                    _emit(progress, 0.87, f"Voicing {len(changed)} shortened lines")
                    run(changed)
                    _check_cancel(cancel)
                    measure(changed)
                    shortened = len(changed)
                # still too long, measured: worded more tightly (or cut,
                # allow_cuts), sized by this voice's real speed on the line at
                # the comfortable speed-up — what is left is spoken faster
                still = self._too_long(p, speech, sr)
                if still:
                    squeeze = p.comfort_tempo()
                    rooms = {}
                    for i in still:
                        # the room left (late start counted), in the
                        # estimate's seconds for this line's real speed
                        real = speech[i].size / sr
                        est = max(0.1, speech_seconds(p.lines[i].text, p.options.target_lang))
                        room = self._late_room.get(i, self._room(p, i))
                        rooms[i] = room * squeeze * est / real
                    kept_text = {i: p.lines[i].text for i in still}
                    _emit(progress, 0.875,
                          f"Cutting words from {len(still)} lines still too long"
                          if p.options.allow_cuts else
                          f"Tightening {len(still)} lines still too long")
                    cut = self._ai_shorten(p, still, cancel, condense=True, rooms=rooms)
                    if cut and p.options.quality == "high":
                        self._check_meaning(p, cut, cancel, shortened=True,
                                            cut=p.options.allow_cuts, uncut=kept_text)
                        for i in self.meaning_refused & set(cut):
                            p.lines[i].text, p.lines[i].tts_key = kept_text[i], ""
                        cut = [i for i in cut if p.lines[i].text != kept_text[i]]
                    if cut:
                        _emit(progress, 0.88, f"Voicing {len(cut)} cut lines")
                        run(cut)
                        _check_cancel(cancel)
                        measure(cut)
                        shortened += len(cut)

        # lines still too long for their room: spoken faster by the voice
        # itself where it can, so _place_lines rarely has to stretch audio
        _emit(progress, 0.89, "Speeding up long lines")
        applied: dict[int, float] = {}
        faster = self._respeak_faster(p, speech, sr, workers, cancel, applied)
        _check_cancel(cancel)
        floors = list(floor_of.values())
        for i in list(speech):
            # same level for every line and voice; fades so no edge clicks
            speech[i] = ap.smooth_line(speech[i], sr)

        # a line that still could not be voiced keeps its original voice, so
        # the dub has no silent hole; the Lines tab shows it as failed
        kept = 0
        for i in sorted(errors):
            orig = self._original_voice(p, i, sr, cancel)
            if orig.size:
                speech[i] = ap.smooth_line(orig, sr)
                kept += 1
        # place every line on the timeline: never two at once
        _emit(progress, 0.9, "Placing lines on the timeline")
        total = max(p.duration, max((l.end for l in p.lines), default=0) + 2)
        buf = np.zeros(int(total * sr) + 1, dtype=np.int32)
        late, max_late = self._place_lines(
            p, speech, buf, sr, {i: e for i, e in errors.items() if i not in speech}, cancel,
            applied)
        for i in errors:
            p.lines[i].status = "failed"
        p.voice_track = p.sub("audio", "voice_track.wav")
        ap._write_wav_int16(p.voice_track,
                            np.clip(buf, -32768, 32767).astype(np.int16), sr)
        floor = float(np.median(floors)) if floors else -120.0
        noisy = floor > ap.NOISY_FLOOR_DB
        LOG_BUS.publish(f"[dub] voice noise floor {floor:.0f} dBFS — "
                        + ("removing hiss" if noisy else "clean, no denoise needed"))
        _emit(progress, 0.97, "Cleaning up the voices")
        ap.polish_voice(p.voice_track, p.voice_track, denoise=noisy, cancel=cancel)
        note = f"{n - len(errors)}/{n} lines voiced"
        if shortened:
            note += f"; {shortened} shortened to fit"
        # how fast the voice had to go (what the user hears), not how many
        # lines the voice re-spoke
        fast = [ln.speed for ln in p.lines if ln.text.strip() and ln.speed > 1.02]
        if fast:
            easy = p.comfort_tempo()
            over = sum(1 for s in fast if s > easy + 0.01)
            note += (f"; {len(fast)} spoken faster to fit (up to {max(fast):.1f}×"
                     + (f", {over} over {easy:.1f}×" if over else "") + ")")
        elif faster:
            note += f"; {faster} spoken faster to fit"
        if noisy:
            note += "; hiss removed"
        if regendered:
            note += (f"; after your gender change {regendered} lines voiced with "
                     "the other gender")
        warn = []
        if errors:
            warn.append(f"{len(errors)} failed" + (
                f" ({kept} kept the original voice; redo Speak to try again)" if kept else ""))
        if late:
            msg = (f"{late} start a little late (up to {max_late:.1f} s) so voices "
                   "never overlap")
            if max_late >= 1.0:
                warn.append(msg + " — shorten the yellow lines")
            else:
                note += f"; {msg}"
        if warn:
            return "warn", note + "; " + "; ".join(warn)
        return "done", note

    # -- step 9: mix --------------------------------------------------------
    def _step_mix(self, p: DubProject, progress, cancel):
        if not os.path.isfile(p.voice_track):
            raise AppError("Voice track missing — run the speak step again.",
                           "ខ្វះបទសំឡេង — សូមដំណើរការជំហាននិយាយម្តងទៀត។")
        p.mix_wav = p.sub("audio", "dub_mix.wav")
        loud = f"; loudness {ap.TARGET_LUFS:.0f} LUFS"
        if not (p.options.background == "none"
                or (p.music_wav and os.path.isfile(p.music_wav))):
            ap.mix_with_ducking(p.voice_track, p.audio_wav, p.mix_wav,
                                duck_db=-30.0, bgm_gain_db=-6.0 + p.options.music_gain_db,
                                progress=progress, cancel=cancel)
            return ("warn", "No separated music: the original audio was ducked "
                    "under the new voices" + loud)
        # mixed in 32-bit float (the sum may pass full scale), then one
        # loudness gain + peak limiter for the whole mix
        raw = p.sub("audio", "dub_mix_raw.wav")
        voice = "aresample=48000,pan=stereo|c0=c0|c1=c0"
        if p.options.background == "none":
            # voices only: the new voices, no original sound
            ap._run_ffmpeg(["-i", p.voice_track, "-af", voice,
                            "-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", raw],
                           duration=p.duration, progress=_sub(progress, 0, 0.6),
                           cancel=cancel, status_text="Mixing…")
            note = "Voices only (no background sound)"
        else:
            # match the new voices to the loudness of the original dialogue
            # (TTS comes out ~10 dB quieter than drama dialogue)
            auto = 0.0
            ref, cur = loudness_lufs(p.vocals_wav), loudness_lufs(p.voice_track)
            if ref is not None and cur is not None:
                auto = max(-6.0, min(20.0, ref - cur))
            gain = auto + p.options.voice_gain_db
            # the music & effects dip ~4 dB under the new voices (soft knee,
            # slow attack / release: no pumping) — clearer speech, and what
            # the separation left of the original voices is masked
            fc = (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,"
                  f"volume={p.options.music_gain_db:.2f}dB[m];"
                  f"[1:a]{voice},volume={gain:.2f}dB,asplit=2[v][sc];"
                  "[m][sc]sidechaincompress=threshold=0.05:ratio=2:attack=80:"
                  "release=500:knee=4[md];"
                  "[md][v]amix=inputs=2:duration=first:dropout_transition=0:"
                  "normalize=0[out]")
            ap._run_ffmpeg(["-i", p.music_wav, "-i", p.voice_track,
                            "-filter_complex", fc, "-map", "[out]",
                            "-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", raw],
                           duration=p.duration, progress=_sub(progress, 0, 0.6),
                           cancel=cancel, status_text="Mixing…")
            note = ("Voices mixed over the original music & effects"
                    + (f" (voices {auto:+.1f} dB to match the original dialogue)"
                       if auto else ""))
        try:
            ap.normalize_loudness(raw, p.mix_wav, duration=p.duration,
                                  progress=_sub(progress, 0.6, 1.0), cancel=cancel)
        finally:
            try:
                os.remove(raw)
            except OSError:
                pass
        return note + loud

    # -- step 10: export ----------------------------------------------------
    def default_output(self, p: DubProject) -> str:
        v = Path(p.video)
        lang = _base_lang(p.options.target_lang) or "dub"
        return str(v.with_name(f"{v.stem}.dub.{lang}.mp4"))   # -> "x.dub.km.srt"

    def subtitle_text(self, p: DubProject, ln: DubLine) -> str:
        """What the subtitle of a line says: its whole translation (subtitle_full)
        — the dub's wording may have been shortened to fit the timing, the
        subtitle need not be. A line typed in Lines shows what was typed; an
        emptied line has no subtitle (and no dub)."""
        text = ln.text.strip()
        if not text or not p.options.subtitle_full or self._hand_edited(ln):
            return text
        return (ln.full_text or "").strip() or text

    def write_srt(self, p: DubProject, path: str) -> str:
        blocks = []
        k = 0
        for ln in p.lines:
            text = self.subtitle_text(p, ln)
            if not text:
                continue
            k += 1
            blocks.append(f"{k}\n{ap._fmt_srt_time(ln.start)} --> "
                          f"{ap._fmt_srt_time(ln.end)}\n{text}\n")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(blocks))
        return path

    def write_ass(self, p: DubProject, path: str) -> str:
        """Styled subtitles (app.engines.subtitles preset) for the dub."""
        from app.engines import media_exporter as me
        from app.engines import subtitles as st
        try:
            info = me.probe(p.video)
            w, h = info.width or 1920, info.height or 1080
        except Exception:
            w, h = 1920, 1080
        cues = [st.Cue(l.start, l.end, t) for l in p.lines
                if (t := self.subtitle_text(p, l))]
        return st.write_ass(cues, path, w, h, p.options.subtitle_style or "clean")

    def _render(self, p: DubProject, out: str, subs: str | None,
                progress, cancel) -> None:
        """Re-encode the video with the dub audio (and burned subtitles):
        the detected GPU encoder in quality mode (no fixed bitrate), and
        libx264 if that encoder fails on this video."""
        from app.engines import media_exporter as me
        enc = me.best_h264_encoder(self.config)
        # the dub's language on the audio track, as the stream-copy export does
        lang = _ISO3.get(_base_lang(p.options.target_lang), "und")
        try:
            me.render(p.video, out, me.ExportPreset(vcodec=enc, bitrate=""),
                      srt_path=subs, dub_audio=p.mix_wav, progress=progress,
                      cancel=cancel, audio_lang=lang)
        except AppError as exc:
            if enc == "libx264":
                raise
            log.warning("%s failed (%s); encoding with libx264", enc, exc.en)
            me.render(p.video, out, me.ExportPreset(bitrate=""), srt_path=subs,
                      dub_audio=p.mix_wav, progress=progress, cancel=cancel,
                      audio_lang=lang)

    @staticmethod
    def _keep_poster(p: DubProject, out: str) -> str:
        """The source video's poster (the picture a downloaded video carries as
        its thumbnail) put into the dubbed video too: writing a new file left
        it behind. A note for the task, "" when the source has no poster."""
        from app.engines import cover_art
        try:
            poster = cover_art.find_poster(p.video)
            if not poster:
                return ""
            if cover_art.set_cover(out, poster):
                return "the original's poster kept as its thumbnail"
            log.warning("could not put the poster into %s", out)
        except Exception as exc:          # never fail a finished dub over its thumbnail
            log.warning("could not keep the poster: %s", exc)
        return ""

    def _step_export(self, p: DubProject, progress, cancel):
        if not os.path.isfile(p.mix_wav):
            raise AppError("Mixed audio missing — run the mix step again.",
                           "ខ្វះសំឡេងលាយ — សូមដំណើរការជំហានលាយម្តងទៀត។")
        out = p.options.output_path.strip() or self.default_output(p)
        lang = _base_lang(p.options.target_lang)
        stem = str(Path(out).with_suffix(""))
        if not stem.endswith(f".{lang}"):
            stem += f".{lang}"           # "x.mp4" -> "x.km.srt"; "x.km.mp4" -> "x.km.srt"
        p.srt_path = self.write_srt(p, stem + ".srt")
        iso = _ISO3.get(lang, "und")
        burn_path = p.srt_path
        if p.options.subtitle_style:
            burn_path = self.write_ass(p, stem + ".ass")
        if p.options.burn_subtitles:
            self._render(p, out, burn_path, progress, cancel)
        else:
            mkv = out.lower().endswith(".mkv")
            args = ["-i", p.video, "-i", p.mix_wav]
            if p.options.soft_subtitles:
                args += ["-i", p.srt_path]
            args += ["-map", "0:v:0", "-map", "1:a:0"]
            if p.options.soft_subtitles:
                args += ["-map", "2:s:0", "-c:s", "srt" if mkv else "mov_text",
                         "-metadata:s:s:0", f"language={iso}"]
            # the video's own length — not -shortest: that counts the soft
            # subtitle stream too and cut the video off at the last line
            length = ["-t", f"{p.duration:.3f}"] if p.duration > 0 else ["-shortest"]
            args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-metadata:s:a:0", f"language={iso}"] + length
            if not mkv:
                args += ["-movflags", "+faststart"]
            try:
                ap._run_ffmpeg(args + [out], duration=p.duration,
                               progress=progress, cancel=cancel,
                               status_text="Writing dubbed video…")
            except AppError as exc:
                # e.g. a video codec the container cannot hold: re-encode
                log.warning("Stream copy failed (%s); re-encoding", exc.en)
                self._render(p, out, None, progress, cancel)
        p.output = out
        note = f"Saved {os.path.basename(out)} + subtitles"
        kept = self._keep_poster(p, out)
        if kept:
            note += f"; {kept}"
        try:
            saved = self.save_to_series(p)
            if saved:
                note += f"; {saved}"
        except Exception as exc:
            log.warning("Could not update series memory: %s", exc)
        return note


# ---------------------------------------------------------------------------
# reference clips
# ---------------------------------------------------------------------------

def build_reference(wav: str, lines: list[DubLine], out_wav: str,
                    target_s: float = 9.0, max_s: float = 12.0
                    ) -> tuple[str, str] | None:
    """Pick the cleanest lines of one speaker and join them into a cloning
    reference clip. Returns (wav_path, transcript) or None."""
    cands = []
    for ln in lines:
        d = ln.end - ln.start
        if d < 1.2 or d > 12.0 or not ln.source.strip():
            continue
        x, sr = ap._read_wav_slice(wav, ln.start, ln.end, max_seconds=12.0)
        if x.size < int(sr):
            continue
        hop = max(1, int(0.05 * sr))
        nfr = x.size // hop
        rms = np.sqrt(np.mean(x[:nfr * hop].reshape(nfr, hop) ** 2, axis=1))
        peak = float(np.max(np.abs(x)))
        if peak < 0.02:
            continue
        density = float(np.mean(rms > 0.1 * float(np.max(rms))))
        clipped = float(np.mean(np.abs(x) > 0.985))
        # prefer 2.5–8 s of dense, unclipped speech
        score = density - 4.0 * clipped - 0.05 * abs(min(d, 8.0) - 5.0)
        cands.append((score, ln, x, sr))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    chosen, total = [], 0.0
    for c in cands:
        d = c[2].size / c[3]
        if total + d > max_s and chosen:
            continue
        chosen.append(c)
        total += d
        if total >= target_s:
            break
    chosen.sort(key=lambda c: c[1].start)
    sr = chosen[0][3]
    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    parts = []
    for c in chosen:
        x = c[2] if c[3] == sr else np.interp(
            np.linspace(0, c[2].size - 1, int(c[2].size * sr / c[3])),
            np.arange(c[2].size), c[2]).astype(np.float32)
        parts += [x, gap]
    y = np.concatenate(parts[:-1])
    y = y / max(float(np.max(np.abs(y))), 1e-6) * 0.89   # ≈ -1 dBFS
    ap._write_wav_int16(out_wav, y, sr)
    return out_wav, " ".join(c[1].source.strip() for c in chosen)


def loudness_lufs(path: str) -> float | None:
    """Integrated loudness (EBU R128) of a file via ffmpeg, or None."""
    return ap.loudness_lufs(path)


def _speech_source(p: DubProject, ln: DubLine) -> str:
    """The separated vocals, unless separation left this line near-silent
    (then the original mix)."""
    try:
        x, _sr = ap._read_wav_slice(p.asr_wav, ln.start, ln.end, max_seconds=4.0)
        if x.size and float(np.max(np.abs(x))) >= 0.02:
            return p.asr_wav
    except Exception:
        pass
    return p.audio_wav if p.audio_wav and os.path.isfile(p.audio_wav) else p.asr_wav


def _trim_silence(data: np.ndarray, sr: int, thresh: int = 300) -> np.ndarray:
    """Drop leading and trailing silence (TTS engines and voice conversion
    pad ~0.2 s before and up to ~0.9 s after the speech) so lines start on
    cue and only speech counts against the slot; keep 30 ms / 80 ms."""
    return ap.trim_silence(data, sr, thresh=thresh)


_SCRIPT_RANGES = {
    "km": [(0x1780, 0x17FF), (0x19E0, 0x19FF)],
    "th": [(0x0E00, 0x0E7F)],
    "lo": [(0x0E80, 0x0EFF)],
    "my": [(0x1000, 0x109F)],
    "zh": [(0x3400, 0x9FFF)], "ja": [(0x3040, 0x30FF), (0x3400, 0x9FFF)],
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "hi": [(0x0900, 0x097F)],
}


def foreign_script(text: str, lang: str) -> bool:
    """True if `text` (meant to be in `lang`) contains letters of another
    non-Latin script from the table above (Latin, digits and punctuation
    are fine: names, numbers). For Khmer, Thai, Lao and Burmese — voices
    that read only their own script — any other letter or mark counts:
    Tibetan, Greek and Thai letters inside Khmer number words went unseen
    ("បីរយបདាយសិប", "ហ้าសិប") and the voice said nonsense."""
    base = _base_lang(lang)
    own = _SCRIPT_RANGES.get(base)
    if not own:
        return False
    strict = base in _NO_LATIN_LANGS
    for ch in text:
        o = ord(ch)
        if any(a <= o <= b for a, b in own):
            continue
        if strict and o >= 0x80 and unicodedata.category(ch)[0] in "LM":
            return True
        for ranges in _SCRIPT_RANGES.values():
            if any(a <= o <= b for a, b in ranges):
                return True
    return False


# Targets whose TTS voices cannot read Latin letters: a leftover English word
# or name is spoken wrongly or skipped, so it is transliterated instead.
_NO_LATIN_LANGS = {"km", "th", "lo", "my"}


def _pinyin(chars: str) -> list[str] | None:
    """Toneless pinyin per character, or None without pypinyin."""
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return None
    return [s.lower() for s in lazy_pinyin(chars)]


def _sounds_alike_cjk(heard: str, meant: str, lang: str) -> bool:
    """Character-script version of sounds_alike: punctuation is ignored, the
    two are aligned character by character and only the changed characters
    count — at most two, each replaced one for one (nothing added or dropped).
    Chinese: each pair must sound alike in pinyin (销/校 xiao, 亲/新 qin/xin,
    书/出 shu/chu — not 单/货 dan/huo, 什么/玩笑 shen me/wan xiao); without
    pypinyin a Chinese correction is not taken. Other such scripts: one or two
    changed characters."""
    import difflib
    strip = lambda s: re.sub(r"[\W_]+", "", s)          # noqa: E731
    a, b = strip(heard), strip(meant)
    if not a or not b:
        return False
    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or i2 - i1 != j2 - j1:
            return False
        pairs += list(zip(a[i1:i2], b[j1:j2]))
    if not 0 < len(pairs) <= 2:
        return False
    if lang != "zh":
        return True
    for x, y in pairs:
        px, py = _pinyin(x), _pinyin(y)
        if px is None or py is None:
            return False
        if difflib.SequenceMatcher(None, px[0], py[0]).ratio() < 0.6:
            return False
    return True


def sounds_alike(heard: str, meant: str, cjk: bool = False, lang: str = "") -> bool:
    """Could `meant` be what a recogniser wrote down as `heard`? The two are
    aligned word by word and only the words that differ are compared: at most
    three of them, each pair sharing most of its letters and about the same
    length ("shield" / "shill", "only" / "on in"), nothing merely added or
    dropped ("money" / "surgery" is a rewrite, not a mishearing). Chinese and
    the like: see _sounds_alike_cjk."""
    import difflib
    if cjk:
        return _sounds_alike_cjk(heard, meant, lang)
    hw = re.findall(r"[\w']+", heard.lower())
    mw = re.findall(r"[\w']+", meant.lower())
    if not hw or not mw:
        return False
    changed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, hw, mw).get_opcodes():
        if tag == "equal":
            continue
        a, b = "".join(hw[i1:i2]), "".join(mw[j1:j2])
        if not a or not b:
            return False
        changed += max(i2 - i1, j2 - j1)
        if (difflib.SequenceMatcher(None, a, b).ratio() < 0.5
                or abs(len(a) - len(b)) > max(3, len(a) // 2)):
            return False
    return 0 < changed <= 3


def strip_speaker_name(text: str, source: str, names: set[str]) -> str:
    """Drop a speaker name the AI wrote before a line ("ស៊ីលៀ: ចាស" or
    "Celia (F): …"), unless the source line itself starts that way.
    `names`: lower-case character ids, names and their spellings."""
    m = re.match(r"^\s*([^:：\n]{1,30}?)\s*[:：]\s*(\S.*)$", text or "", re.S)
    if not m:
        return text
    who = re.sub(r"\s*\((?:M|F|male|female)\)\s*$", "", m.group(1).strip(), flags=re.I)
    src = re.match(r"^\s*([^:：\n]{1,30}?)\s*[:：]", source or "")
    if src and src.group(1).strip().lower() in names:
        return text                      # the source line starts with that name too
    # a known name, or any short label when the source has no "name:" at all
    # (a Khmer spelling of a name the glossary never listed: "បប: …")
    if who.lower() in names or (not src and len(who.split()) <= 3
                                and not re.search(r"\d", who)):
        return m.group(2).strip()
    return text


def latin_leftover(text: str, lang: str, source: str = "") -> bool:
    """True if `text` (meant to be in `lang`) still has Latin letters and
    `lang` is written in a script whose voices cannot read them. Khmer: a
    short code the original line has too (H2, ICU) may stay — it is spelled
    out for the voice (speakable_km) and reads well in subtitles; a word, or
    letters the original does not have (an AI slip such as "nh"), count."""
    base = _base_lang(lang)
    if base not in _NO_LATIN_LANGS:
        return False
    runs = re.findall(r"[A-Za-z]+", text or "")
    if base != "km":
        return bool(runs)
    src = (source or "").lower()
    return any(len(r) >= 4 or r.lower() not in src for r in runs)


# how a Khmer speaker says the Latin letters of a code ("H2", "ICU", "BMW")
_KM_LETTERS = {
    "a": "អេ", "b": "ប៊ី", "c": "ស៊ី", "d": "ឌី", "e": "អ៊ី", "f": "អេហ្វ", "g": "ជី",
    "h": "អេច", "i": "អាយ", "j": "ជេ", "k": "ខេ", "l": "អែល", "m": "អឹម", "n": "អិន",
    "o": "អូ", "p": "ភី", "q": "ឃ្យូ", "r": "អា", "s": "អេស", "t": "ធី", "u": "យូ",
    "v": "វី", "w": "ដាប់ប៊លយូ", "x": "អិច", "y": "វ៉ាយ", "z": "ហ្សិត"}


def speakable_km(text: str) -> str:
    """The words a Khmer voice should say for `text`: Latin letters left in a
    line (a model name, an abbreviation) spelled out in Khmer, as people read
    them ("H2" → "អេចពីរ", "ICU" → "អាយស៊ីយូ"), and numbers in Khmer words made
    by code (៦៨១ → ប្រាំមួយរយប៉ែតសិបមួយ; numbers.km_say_numbers): the voice
    never reads digits on its own, and the AI's own number words were wrong
    one time in five. Only the voice gets this; the subtitles keep the
    letters and digits."""
    spelled = re.sub(r"[A-Za-z]+",
                     lambda m: "".join(_KM_LETTERS[c] for c in m.group(0).lower()), text or "")
    return nb.km_say_numbers(spelled)


# Khmer misspellings the AI keeps writing although the rules name them:
# លឺ for ឮ "hear" (a ល under a coeng belongs to another word: ភ្លឺ "bright",
# ពន្លឺ "light"), ហិរ for ហឹរ "spicy" (not in ហិរញ្ញវត្ថុ "finance"). The
# right spelling sounds the same, so the voice is unchanged.
_KM_SPELLING = [(re.compile("(?<!្)លឺ"), "ឮ"),
                (re.compile("ហិរ(?![ញណ])"), "ហឹរ"),
                # a vowel sign or mark typed twice ("ម៉ូូតូ"): Khmer never repeats one
                (re.compile(r"([ា-៓])\1+"), r"\1")]


def km_spelling(text: str, keep: list[str] | tuple[str, ...] = ()) -> str:
    """`text` with the common Khmer misspellings corrected; the `keep` terms
    (glossary names such as a transliterated "Tyler") stay as written."""
    keep = sorted({k for k in keep if k and any(rx.search(k) for rx, _g in _KM_SPELLING)},
                  key=len, reverse=True)
    parts = re.split("(" + "|".join(map(re.escape, keep)) + ")", text) if keep else [text]
    for j in range(0, len(parts), 2):          # the odd parts are kept terms
        for rx, good in _KM_SPELLING:
            parts[j] = rx.sub(good, parts[j])
    return "".join(parts)


def _clean_line(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^\[[^\]]*\]\s*", "", t)           # echoed speaker tag
    t = re.sub(r"^(ORIGINAL|DRAFT|FINAL)\s*:\s*", "", t, flags=re.I)
    if "||" in t:
        t = t.split("||")[-1].strip()
        t = re.sub(r"^(DRAFT|FINAL)\s*:\s*", "", t, flags=re.I)
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    return t


def _speaker_transcript(p: DubProject, max_chars: int = 14000) -> str:
    rows = [f"{l.speaker}: {l.source}" for l in p.lines if l.source.strip()]
    text = "\n".join(rows)
    if len(text) <= max_chars:
        return text
    # beginning, middle and end — names are usually introduced early
    third = max_chars // 3
    mid = len(text) // 2
    return (text[:third] + "\n...\n" + text[mid - third // 2: mid + third // 2]
            + "\n...\n" + text[-third:])


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace as NS

    failures: list[str] = []

    def expect(cond: bool, label: str) -> None:
        print(("ok:   " if cond else "FAIL: ") + label)
        if not cond:
            failures.append(label)

    # split_segments: pause + sentence-end splits, short fragment merge
    W = lambda s, e, t: NS(start=s, end=e, text=t)
    seg = NS(start=0.0, end=6.0, text="Where are you going? Home. Wait",
             words=[W(0.0, 0.4, "Where"), W(0.4, 0.6, " are"),
                    W(0.6, 0.8, " you"), W(0.8, 1.3, " going?"),
                    W(2.0, 2.6, " Home."), W(2.6, 2.9, " Wait")])
    ls = split_segments([seg])
    expect(len(ls) == 2, f"split into 2 lines (got {len(ls)})")
    expect(ls[0].source == "Where are you going?", "first line text")
    expect(ls[1].source == "Home. Wait" and abs(ls[1].start - 2.0) < 1e-9,
           "short fragment merged")
    expect(split_segments([NS(start=1, end=2, text="hi", words=[])])[0].source
           == "hi", "segment without words kept")
    # stretched first word (whisper over music) is trimmed from the left
    st = split_segments([NS(start=24.5, end=53.0, text="还是回家喝奶去吧", words=[
        W(24.5, 52.0, "还是回家"), W(52.0, 53.0, "喝奶去吧")])])
    expect(len(st) == 1 and st[0].start >= 49.9, f"stretched word trimmed ({st[0].start})")
    # immediate echo of the same line is dropped
    ec = drop_repeats([DubLine(82.8, 84.0, "我给你买了儿童餐"), DubLine(84.0, 84.2, "我给你买了儿童餐"),
                       DubLine(84.2, 85.3, "餐厅说")])
    expect([l.source for l in ec] == ["我给你买了儿童餐", "餐厅说"], "echo dropped")
    # a real repeat takes about as long as the first one: both are kept
    rp2 = drop_repeats([DubLine(10.0, 10.6, "快走！"), DubLine(10.7, 11.3, "快走！"),
                        DubLine(12.0, 12.8, "No! No!"), DubLine(12.9, 13.6, "No! No!")])
    expect(len(rp2) == 4, f"real repeats kept ({[l.source for l in rp2]})")
    # the prompt written out (two names together) is left out; one name
    # alone is someone calling that person: kept, marked for review
    pe = drop_prompt_echo([DubLine(0, 1, "林凡、苏晴"), DubLine(1, 2, "林凡！"),
                           DubLine(2, 3, "林凡和苏晴去哪了")], "林凡、苏晴、顾川")
    expect([l.source for l in pe] == ["林凡！", "林凡和苏晴去哪了"]
           and pe[0].review == "maybe_prompt" and pe[1].review == "",
           f"prompt echo: names list left out, a called name kept ({[(l.source, l.review) for l in pe]})")
    # a segment's review mark reaches its lines, and survives a merge
    rv = split_segments([NS(start=0.0, end=3.0, text="顾川。我记得你", review="recovered",
                            words=[W(0.0, 0.5, "顾川。"), W(1.6, 2.0, "我记得"), W(2.0, 3.0, "你")])])
    expect(rv and all(l.review == "recovered" for l in rv), f"review mark kept ({rv})")
    mg = merge_fragments([DubLine(0, 0.5, "嗯", speaker="S1"),
                          DubLine(0.6, 1.4, "好的", speaker="S1", review="recovered")], "zh")
    expect(len(mg) == 1 and mg[0].review == "recovered", "review mark survives a merge")
    sk_line = DubLine(2.0, 2.5, "嗯", speaker="S1", skip=True)
    expect(len(merge_fragments([DubLine(1.6, 1.9, "对", speaker="S1"), sk_line], "zh")) == 2,
           "a line left out is never merged into another")
    # the transcribe note names what the second listen did, with times
    tn_dir = tempfile.mkdtemp(prefix="rs_tn_")
    tn_p = DubProject(os.path.join(tn_dir, "tn.mp4"), os.path.join(tn_dir, "tn"))
    tn_p.lines = [DubLine(0, 1, "a"), DubLine(1, 2, "b", review="low_confidence")]
    tn_st, tn_note = DubEngine._transcribe_result(
        tn_p, ap.Transcript([], "zh", 99.0, recovered=2,
                            unheard=[(61.0, 61.8, "nothing recognised", "")]), "2 lines")
    expect(tn_st == "warn" and "2 missed lines recovered" in tn_note
           and "1 line the recogniser was unsure" in tn_note and "at 1:01" in tn_note,
           f"transcribe note reports the second listen ({tn_note!r})")
    # fragments of one speaker are joined; other speakers are not
    fr = merge_fragments([DubLine(18.6, 19.9, "比想象中还要强", speaker="S1"),
                          DubLine(19.9, 20.6, "此刻", speaker="S1"),
                          DubLine(20.6, 22.1, "他恐怕还没用全力", speaker="S1"),
                          DubLine(22.2, 22.6, "你", speaker="S2")], "zh")
    expect([l.source for l in fr] == ["比想象中还要强，此刻", "他恐怕还没用全力", "你"],
           f"fragments merged ({[l.source for l in fr]})")
    en = merge_fragments([DubLine(1.0, 2.0, "Are you ready?", speaker="S1"),
                          DubLine(2.1, 2.9, "Of course.", speaker="S1")], "en")
    expect(en[0].source == "Are you ready? Of course.", f"English join keeps a space ({en[0].source!r})")
    # speaker-change cuts: a name goes back to its sentence, a lower-case
    # continuation keeps the first speaker; a new turn ("No.") stays
    ts = _fix_turn_splits([
        DubLine(23.2, 24.1, "You're a jerk,", asr_speaker="D1"),
        DubLine(24.1, 27.3, "Tom. Look, Celia, we have to follow our passions.", asr_speaker="D2"),
        DubLine(50.0, 52.0, "Robot's memory synced", asr_speaker="D2"),
        DubLine(52.2, 52.9, "and", asr_speaker="D1"),
        DubLine(53.3, 53.7, "locked.", asr_speaker="D1"),
        DubLine(60.0, 61.0, "Well,", asr_speaker="D1"),
        DubLine(61.1, 62.5, "No. I don't think so.", asr_speaker="D2")])
    expect(ts[0].source == "You're a jerk, Tom." and ts[1].source.startswith("Look, Celia")
           and ts[0].end > 24.1 and abs(ts[1].start - ts[0].end) < 1e-9,
           f"name handed back ({ts[0].source!r} | {ts[1].source!r})")
    expect([l.asr_speaker for l in ts[2:5]] == ["D2", "D2", "D2"], "continuation keeps speaker")
    expect(ts[6].source == "No. I don't think so." and ts[6].asr_speaker == "D2",
           "a new turn is not moved")
    same = _fix_turn_splits([DubLine(23.2, 24.1, "You're a jerk,", asr_speaker="D1"),
                             DubLine(24.1, 27.3, "Tom. Look, Celia.", asr_speaker="D1")])
    expect(same[0].source == "You're a jerk, Tom.", "name handed back within one label")
    expect(latin_leftover("ថម Hello", "km", "Hello Tom") and not latin_leftover("ថម OK", "km", "OK, Tom")
           and not latin_leftover("កាវ៉ាសាគី H2", "km", "奖励顶配川崎H2")
           and latin_leftover("ភារកិច្ចដំបូងគឺ nh", "km", "新手任务")
           and latin_leftover("ថម OK", "km") and not latin_leftover("ថម ១២៣ 25", "km")
           and not latin_leftover("OK 好", "zh") and latin_leftover("ไป OK", "th"),
           "Latin in Khmer: only a short code the original has too may stay")
    expect(speakable_km("រង្វាន់គឺ កាវ៉ាសាគី H2") == "រង្វាន់គឺ កាវ៉ាសាគី អេចពីរ"
           and speakable_km("នៅ ICU") == "នៅ អាយស៊ីយូ" and speakable_km("ថម") == "ថម",
           "Latin letters spelled in Khmer for the voice")
    expect(speakable_km("ថ្នាក់ប្រធានបាន៦៨១") == "ថ្នាក់ប្រធានបានប្រាំមួយរយប៉ែតសិបមួយ"
           and speakable_km("បាន 692 ពិន្ទុ") == "បាន ប្រាំមួយរយកៅសិបពីរ ពិន្ទុ",
           "numbers said in Khmer words made by code (681 is not 581)")
    expect(foreign_script("បីរយបདាយសិប", "km") and foreign_script("ហ้าសិប", "km")
           and foreign_script("មនុស្ស άប្រយោជន៍", "km") and not foreign_script("បាន ៦៨១!", "km")
           and not foreign_script("你好 é", "zh"),
           "any other script's letter inside a Khmer line is caught (Tibetan, Thai, Greek)")
    expect(km_spelling("ម៉ូូតូឌុប") == "ម៉ូតូឌុប" and km_spelling("ស្រឡាញ់") == "ស្រឡាញ់",
           "a doubled Khmer vowel sign becomes one")
    nm = {"s1", "celia", "ស៊ីលៀ", "tom", "ថម"}
    expect(strip_speaker_name("ស៊ីលៀ: បានទទួលសារ។", "Roger, roger.", nm) == "បានទទួលសារ។"
           and strip_speaker_name("Tom (M): ទៅ", "Go.", nm) == "ទៅ"
           and strip_speaker_name("ម៉ោង 7:30 ហើយ", "It's 7:30.", nm) == "ម៉ោង 7:30 ហើយ"
           and strip_speaker_name("ថម: ទៅ", "Tom: go.", nm) == "ថម: ទៅ",
           "leaked speaker name removed, real colons kept")
    expect(strip_speaker_name("បប: បង! មិនមែនទេអី?", "Hey, bro. Are you?", nm) == "បង! មិនមែនទេអី?"
           and strip_speaker_name("នាយកស្មីត: ល្អ!", "Fine.", nm) == "ល្អ!"
           and strip_speaker_name("ចំណាំ៖ ម៉ោង ១០:៣០", "Note: 10:30", nm) == "ចំណាំ៖ ម៉ោង ១០:៣០"
           and strip_speaker_name("ម៉ោង 7:30 ហើយ", "Already.", nm) == "ម៉ោង 7:30 ហើយ",
           "an unknown short label before a line is removed when the source has no colon")

    class _StubAI:
        """Answers ai.chat with the queued replies, in order."""
        def __init__(self, *replies):
            self.replies, self.prompts, self.systems = list(replies), [], []

        def chat(self, **kw):
            self.prompts.append(kw["messages"][0]["content"])
            self.systems.append(kw.get("system", ""))
            r = self.replies.pop(0)
            if isinstance(r, BaseException):
                raise r
            return NS(text=r, provider="stub", model="stub")

    se = DubEngine(ai=_StubAI("1. 1. សួស្តី\n2. 4 3"), config=None)
    got = se._numbered("translate_draft", "sys", "hdr", ["[A] Hi", "[A] 4 3"],
                       ["Hi", "4 3"], None, sources=["Hi", "4 3"])
    expect(got == ["សួស្តី", "4 3"], f"doubled list number stripped, countdown kept ({got})")

    # clustering: two synthetic voices (100 Hz vs 220 Hz harmonics)
    sr = 16000
    rng = np.random.default_rng(1)

    def voice(f0, dur):
        t = np.arange(int(dur * sr)) / sr
        jitter = 1 + 0.02 * np.sin(2 * np.pi * 3 * t)
        x = sum(np.sin(2 * np.pi * f0 * k * t * jitter) / k for k in range(1, 8))
        return (0.3 * x / 3 + 0.01 * rng.standard_normal(t.size)).astype(np.float32)

    feats = [voice(105 + rng.uniform(-5, 5), 1.5) for _ in range(12)] + \
            [voice(225 + rng.uniform(-8, 8), 1.5) for _ in range(12)]
    X, rows = _normalize_features([voice_features(x, sr) for x in feats])
    expect(len(rows) == 24, "features for every clip")
    lab = cluster_speakers(X)
    expect(len(set(lab[:12].tolist())) == 1 and len(set(lab[12:].tolist())) == 1
           and lab[0] != lab[12], f"two voices separated (labels {lab.tolist()})")
    expect(len(set(cluster_speakers(X, num=1).tolist())) == 1, "forced 1 speaker")

    # project round-trip, slots and step state
    tmp = tempfile.mkdtemp()
    p = DubProject("C:/v.mp4", tmp, DubOptions(target_lang="km"))
    p.duration = 10.0
    p.lines = [DubLine(0.0, 2.0, "a", "ក", "S1"), DubLine(2.1, 3.0, "b", "", "S2"),
               DubLine(6.0, 7.0, "c", "គ", "S1")]
    p.characters = [Character("S1", gender="M"), Character("S2")]
    expect(abs(p.slot(0) - 2.1) < 1e-9, "slot ends at next line start")
    expect(abs(p.slot(1) - 1.5) < 1e-9, "slot borrows up to 0.6 s of silence")
    expect(p.max_chars(0) == int((2.1 * COMFORT_TEMPO - _KM_BASE_S) / _KM_S_PER_CHAR * 0.9)
           and p.options.max_tempo == 2.0 and p.comfort_tempo() == COMFORT_TEMPO,
           f"char budget from the measured Khmer voice at the comfortable speed-up "
           f"({p.max_chars(0)}), lines may go faster to fit (max_tempo 2.0)")
    expect(abs(speech_seconds("ហឺស! ឯង", "km") - (0.30 + 0.057 * 6 + 0.166)) < 1e-6
           and speech_seconds("", "km") == 0.0,
           f"Khmer speech time counts the pause after '!' ({speech_seconds('ហឺស! ឯង', 'km'):.2f})")
    p.set_step("prepare", "done")
    p.set_step("transcribe", "warn")
    expect(p.next_step() == "speakers", "next step after done + warn")
    p.invalidate_from("transcribe")
    expect(p.next_step() == "transcribe", "invalidate resets later steps")
    p.recount()
    expect(p.characters[0].line_count == 2, "recount lines per character")
    p.save()
    q = DubProject.load(p.path)
    expect(len(q.lines) == 3 and q.lines[0].text == "ក" and
           q.characters[0].gender == "M" and q.options.target_lang == "km",
           "project json round-trip")

    eng = DubEngine(ai=None, config=None)
    srt = open(eng.write_srt(p, os.path.join(tmp, "x.srt")), encoding="utf-8").read()
    expect(srt.startswith("1\n00:00:00,000 --> 00:00:02,000\nក") and "2\n" in srt
           and "b" not in srt, "srt skips empty lines and renumbers")
    prm = eng.voice_params(p, p.characters[0])
    expect(prm["engine"] == "edge" and prm["voice"] == "km-KH-PisethNeural",
           "male character without sample -> Khmer male preset")
    from unittest import mock
    with mock.patch(f"{__name__}.seedvc_installed", lambda: True),             mock.patch(f"{__name__}._has", lambda m: True),             mock.patch(f"{__name__}._vram_at_least", lambda gb: gb <= 6.0):
        expect(best_clone_engine("km") == "edge+seedvc", "Khmer prefers Seed-VC with a GPU")
    with mock.patch(f"{__name__}.seedvc_installed", lambda: True),             mock.patch(f"{__name__}._vram_at_least", lambda gb: False),             mock.patch(f"{__name__}.openvoice_ready", lambda: True),             mock.patch(f"{__name__}._has", lambda m: m == "edge_tts"):
        expect(best_clone_engine("km") == "edge+clone", "no GPU: OpenVoice fallback")

    expect(_clean_line("[Lin (F) | max 9 chars] ចាស") == "ចាស", "strip echoed tag")
    expect(foreign_script("ភาระបែបហ្នឹង", "km") and not foreign_script("ស៊ូ OK 123", "km")
           and foreign_script("ម៉ាក់苏", "km") and not foreign_script("妈妈", "zh")
           and foreign_script("ママ", "zh"), "foreign script check")
    expect(_clean_line("ORIGINAL: x || DRAFT: y") == "y", "strip review echo")
    expect(km_spelling("ខ្ញុំលឺ ពន្លឺភ្លឺ លឺទេ?") == "ខ្ញុំឮ ពន្លឺភ្លឺ ឮទេ?"
           and km_spelling("តៃលឺលឺ", ["តៃលឺ"]) == "តៃលឺឮ"
           and km_spelling("ស្លាបមាន់ហិរៗ ហិរញ្ញវត្ថុ") == "ស្លាបមាន់ហឹរៗ ហិរញ្ញវត្ថុ",
           "Khmer spelling: លឺ → ឮ, ហិរ → ហឹរ; not in ពន្លឺ/ភ្លឺ/ហិរញ្ញវត្ថុ or a glossary name")

    # misheard words: taken only when the correction sounds like what was
    # heard; a line is corrected once and the recognised words are kept
    mh = DubProject(os.path.join(tmp, "mh.mp4"), os.path.join(tmp, "mh"), DubOptions(target_lang="km"))
    mh.src_lang = "en"
    mh.characters = [Character("S1", name="Tom", gender="M")]
    mh.lines = [DubLine(0, 2, "You're not a shield our hospital hired, are you?", "", "S1"),
                DubLine(2, 4, "We're only one.", "", "S1"),
                DubLine(4, 6, "Fine, Tom.", "", "S1")]
    mh_ai = _StubAI(json.dumps({"rows": [
        {"line": 1, "heard": "a shield our hospital hired", "meant": "a shill our hospital hired"},
        {"line": 2, "heard": "only one", "meant": "on in one"},
        {"line": 1, "heard": "hospital", "meant": "bakery"},        # does not sound alike
        {"line": 3, "heard": "Tom", "meant": "Tim"},                # a name: never
        {"line": 3, "heard": "Great", "meant": "Grate"}]}),         # not in the line
        '{"rows": []}')
    expect(sounds_alike("a shield our hospital hired", "a shill our hospital hired")
           and sounds_alike("only one", "on in one") and sounds_alike("shield", "shill")
           and not sounds_alike("life saving money", "life-saving surgery")
           and not sounds_alike("Detective", "What am I to do?")
           and not sounds_alike("hospital", "bakery") and not sounds_alike("fine", "fine")
           and sounds_alike("明天", "名天", cjk=True) and not sounds_alike("明天见到你", "再见", cjk=True),
           "sounds-alike guard for misheard words")
    if _pinyin("书") is not None:
        zh_ok = [("我爸骑书去拉客", "我爸骑出去拉客"),
                 ("不是他以为我说的是跑车，神豪系统激活，亲手任务",
                  "不是，他以为我说的是跑车。神豪系统激活，新手任务"),
                 ("销门口别按喇叭", "校门口别按喇叭"), ("把整辆车都排进去", "把整辆车都拍进去")]
        zh_no = [("我爸骑书去拉客", "我爸骑车去拉客"), ("单物交付", "货物交付"),
                 ("这辆得七十多万吧，开什么", "这辆得七十多万吧，开玩笑"),
                 ("你好", "你好吗")]
        expect(all(sounds_alike(a, b, True, "zh") for a, b in zh_ok)
               and not any(sounds_alike(a, b, True, "zh") for a, b in zh_no),
               "Chinese mishearings judged by pinyin of the changed characters only")
    mh_eng = DubEngine(ai=mh_ai, config=None)
    got = mh_eng._fix_mishearings(mh, None)
    again = mh_eng._fix_mishearings(mh, None)
    expect(got == ["a shield our hospital hired → a shill our hospital hired", "only one → on in one"]
           and again == []
           and mh.lines[0].source == "You're not a shill our hospital hired, are you?"
           and mh.lines[0].heard == "You're not a shield our hospital hired, are you?"
           and mh.lines[1].source == "We're on in one." and mh.lines[2].source == "Fine, Tom."
           and len(mh_ai.prompts) == 1 and "Tom" not in mh.lines[0].source,
           f"misheard words corrected once, guarded ({got}, {again}, {len(mh_ai.prompts)} calls)")
    expect(_trim_silence(np.array([0, 0, 0, 5000, 1], np.int16), 16000)[0] == 0,
           "trim keeps a short lead-in")

    # reference clip from synthetic speech
    wav = os.path.join(tmp, "v.wav")
    ap._write_wav_int16(wav, np.concatenate([voice(120, 4.0), voice(120, 4.0)]), sr)
    ref = build_reference(wav, [DubLine(0, 4, "one"), DubLine(4, 8, "two")],
                          os.path.join(tmp, "ref.wav"))
    expect(ref is not None and os.path.isfile(ref[0]) and ref[1] == "one two",
           "reference clip built")

    # a line that could not be voiced keeps its original speech (trimmed,
    # at the voice track's rate); nothing when there is no audio to take
    ov = DubProject(os.path.join(tmp, "ov.mp4"), os.path.join(tmp, "ov"))
    ov.lines = [DubLine(1.0, 2.0, "hello", "សួស្តី", "S1")]
    ov.vocals_wav = os.path.join(tmp, "ov_vocals.wav")
    ap._write_wav_int16(ov.vocals_wav, np.concatenate(
        [np.zeros(sr, np.int16), voice(120, 1.0), np.zeros(sr, np.int16)]), sr)
    orig = DubEngine(ai=None, config=None)._original_voice(ov, 0, 24000, None)
    ov.vocals_wav, ov.asr_wav = "", ""
    none = DubEngine(ai=None, config=None)._original_voice(ov, 0, 24000, None)
    expect(0.85 * 24000 <= orig.size <= 1.1 * 24000 and np.abs(orig).max() > 1000
           and none.size == 0,
           f"original voice kept for a failed line ({orig.size / 24000:.2f} s)")

    # Deepgram is told the language found on this PC; when it still hears
    # nothing, the step uses Whisper instead of failing
    from app.engines.audio_pipeline import Segment, Transcript, Word
    from app.engines import deepgram_stt as _dgs

    class _FakeWhisper:
        def __init__(self, model_size="small", device="auto"):
            self.model_size = model_size

        def detect_language(self, audio):
            return "zh", 0.99

        def transcribe(self, audio, language=None, **kw):
            return Transcript(segments=[Segment(start=1.0, end=2.0, text="你好",
                                                words=[Word(start=1.0, end=2.0, text="你好")])],
                              language=language or "zh", duration=3.0)

    dg = DubProject(os.path.join(tmp, "dg.mp4"), os.path.join(tmp, "dg"), DubOptions())
    dg.asr_wav = os.path.join(tmp, "dg.wav")
    ap._write_wav_int16(dg.asr_wav, voice(120, 1.0), sr)
    dg_calls: list[str] = []

    def fake_deepgram(audio, key, language="", **kw):
        dg_calls.append(language)
        return Transcript(segments=[], language=language or "en", duration=3.0)

    with mock.patch.object(ap, "stt_engine", lambda cfg=None: "deepgram"), \
            mock.patch.object(ap, "WhisperEngine", _FakeWhisper), \
            mock.patch.object(_dgs, "transcribe", fake_deepgram), \
            mock.patch(f"{__name__}._installed_whisper_size", lambda: "small"):
        dg_note = DubEngine(ai=None, config=None)._step_transcribe(dg, None, None)
    expect(dg_calls == ["zh"] and [l.source for l in dg.lines] == ["你好"] and dg.src_lang == "zh"
           and "Whisper" in str(dg_note),
           f"Deepgram told the detected language; heard nothing: Whisper used ({dg_calls}, {dg_note!r})")

    # Deepgram's Chinese words: turns put at real pauses and word edges, a
    # particle stays with its sentence, a one-character blip goes back
    from app.engines.audio_pipeline import Segment as _Seg, Word as _W

    def zw(text, spk, t0, t1):
        return _W(start=t0, end=t1, text=text, speaker=spk)
    seg = _Seg(start=0.0, end=6.0, text="我爸出去拉客了你说的是摩的对啊后座不是只能坐一个人吗",
               speaker="D1", words=[
        zw("我爸", "D2", 0.0, 0.4), zw("出去", "D2", 0.4, 0.8), zw("拉客", "D2", 0.8, 1.2),
        zw("了你", "D1", 1.2, 1.8), zw("说的是", "D1", 1.8, 2.2), zw("摩的", "D1", 2.2, 2.6),
        zw("对啊", "D2", 2.6, 3.0), zw("后", "D2", 3.0, 3.2), zw("座", "D3", 3.2, 3.4),
        zw("不是", "D3", 3.4, 3.8), zw("只能", "D3", 3.8, 4.2), zw("坐", "D1", 4.2, 4.3),
        zw("一个人", "D3", 4.3, 4.9), zw("吗", "D3", 4.9, 5.2)])
    zl = split_segments([seg], lang="zh", quiet=[(1.48, 1.62), (2.55, 2.65), (2.92, 3.08)])
    expect([(l.source, l.asr_speaker) for l in zl] ==
           [("我爸出去拉客了", "D2"), ("你说的是摩的", "D1"), ("对啊", "D2"),
            ("后座不是只能坐一个人吗", "D3")],
           f"Deepgram Chinese turns at pauses and word edges ({[(l.source, l.asr_speaker) for l in zl]})")

    # placement: dubbed lines never sound at the same time, and a delay
    # caused by a long line fades out at the next pause
    q = DubProject(os.path.join(tmp, "v.mp4"), os.path.join(tmp, "place"))
    q.duration = 20.0
    q.lines = [DubLine(0.0, 1.0, "a", "x"), DubLine(1.2, 2.0, "b", "y"),
               DubLine(2.1, 3.0, "c", "z"), DubLine(12.0, 13.0, "d", "w")]
    tts_sr = ap._TTS_SR
    tone = lambda s: (8000 * np.sin(2 * np.pi * 200 * np.arange(int(s * tts_sr)) / tts_sr)
                      ).astype(np.int16)
    spoken = {0: tone(3.0), 1: tone(2.5), 2: tone(1.0), 3: tone(0.8)}   # 0 and 1 too long
    track = np.zeros(int(20 * tts_sr), np.int32)
    n_late, worst = eng._place_lines(q, spoken, track, tts_sr, {}, None)
    blk = tts_sr // 100                                  # 10 ms loudness envelope
    env = np.abs(track[:track.size // blk * blk]).reshape(-1, blk).max(axis=1) > 100
    edges = np.flatnonzero(np.diff(np.concatenate([[0], env.astype(np.int8), [0]])))
    segs = [(float(a) / 100, float(b) / 100) for a, b in zip(edges[::2], edges[1::2])]
    expect(int(np.max(np.abs(track))) <= 8000, "no two lines mixed on top of each other")
    expect(len(segs) == 4 and all(b2 > e1 for (_, e1), (b2, _) in zip(segs, segs[1:])),
           f"4 lines, one after another with a gap ({segs})")
    expect(q.lines[1].delay > 0 and q.lines[3].delay == 0.0 and n_late >= 1,
           f"long line delays the next, the pause absorbs it ({[l.delay for l in q.lines]})")

    # speed to fit (max_tempo 2.0 by default): a line too long for the time
    # before the next one is spoken just as fast as it needs, not faster, and
    # nobody starts late; only past max_tempo does the next line start late;
    # squeezing into its own slot for lip-sync stops at the comfortable 1.4x
    sf = DubProject(os.path.join(tmp, "v.mp4"), os.path.join(tmp, "speedfit"))
    sf.duration = 30.0
    sf.lines = [DubLine(0.0, 1.0, "a", "x"), DubLine(2.0, 3.0, "b", "y"),
                DubLine(10.0, 11.0, "c", "z"), DubLine(12.0, 13.0, "d", "w"),
                DubLine(20.0, 21.0, "e", "v"), DubLine(22.6, 23.0, "f", "u")]
    sf_speech = {0: tone(3.0),     # room 1.92 s: needs 1.56x
                 1: tone(1.0),     # fits
                 2: tone(5.0),     # room 1.92 s: needs 2.6x, gets 2.0x
                 3: tone(0.8),
                 4: tone(2.8),     # room 2.52 s (1.11x), own slot 1.56 s (1.8x)
                 5: tone(0.3)}
    sf_track = np.zeros(int(30 * tts_sr), np.int32)
    sf_late, _w = eng._place_lines(sf, sf_speech, sf_track, tts_sr, {}, None)
    sp = [l.speed for l in sf.lines]
    expect(abs(sp[0] - 3.0 / 1.92) < 0.06 and sf.lines[1].delay == 0.0 and sp[1] == 1.0
           and abs(sp[2] - 2.0) < 0.06 and sf.lines[3].delay > 0.3 and sf_late == 1
           and abs(sp[4] - 1.4) < 0.06 and sf.lines[5].delay == 0.0,
           f"speed to fit: as fast as needed, 2.0x at most, lip-sync only to 1.4x "
           f"(speeds {sp}, delays {[l.delay for l in sf.lines]}, late {sf_late})")
    # which measured lines are said more briefly: with speed to fit, one
    # that needs more than 1.5x (comfortable + margin) of its room — it
    # would otherwise be sped up, never left late; at max_tempo 1.4 (old
    # projects) a line may still run up to _LATE_OK late instead
    tl = DubProject(os.path.join(tmp, "v.mp4"), os.path.join(tmp, "toolong"))
    tl.duration = 20.0
    tl.lines = [DubLine(0.0, 1.0, "a", "x"), DubLine(1.58, 2.0, "b", "y"),
                DubLine(5.0, 6.0, "c", "z"), DubLine(6.58, 7.0, "d", "w")]
    tl_speech = {0: tone(2.0), 1: tone(0.3),      # room 1.5 s: needs 1.33x
                 2: tone(2.4), 3: tone(0.3)}      # room 1.5 s: needs 1.6x
    tl_new = set(eng._too_long(tl, tl_speech, tts_sr))
    tl.options.max_tempo = 1.4
    tl_old = set(eng._too_long(tl, tl_speech, tts_sr))
    expect(tl_new == {2} and tl_old == set(),
           f"reworded when speed to fit would pass 1.5x ({tl_new}); old projects keep "
           f"the late start ({tl_old})")
    # faster re-voicing is tried again when Edge drops a request ("No audio
    # was received" hit 6 lines of one dub); it used to fall back at once to
    # time-stretched audio
    rs = DubProject(os.path.join(tmp, "v.mp4"), os.path.join(tmp, "respeak"),
                    DubOptions(target_lang="km"))
    rs.duration = 10.0
    rs.characters = [Character("S1", gender="M")]
    rs.lines = [DubLine(0.0, 1.0, "a", "ក" * 10, "S1"), DubLine(1.5, 2.5, "b", "ខ", "S1")]
    rs_calls: list[float] = []

    def flaky_synth(_p, _ch, _text, out_wav, _cancel=None, ln=None, speed=1.0):
        rs_calls.append(speed)
        if len(rs_calls) == 1:
            raise AppError("Edge TTS synthesis failed: No audio was received.")
        ap._write_wav_int16(out_wav, tone(2.0 / speed), tts_sr)
        return out_wav

    rs_eng = DubEngine(ai=None, config=None)
    rs_eng.synth_text = flaky_synth
    rs_speech = {0: tone(2.0), 1: tone(0.5)}
    rs_applied: dict[int, float] = {}
    rs_n = rs_eng._respeak_faster(rs, rs_speech, tts_sr, 1, None, rs_applied)
    expect(rs_n == 1 and len(rs_calls) == 2 and rs_applied.get(0, 0) > 1.3
           and abs(rs_speech[0].size / tts_sr - 2.0 / rs_calls[1]) < 0.05,
           f"faster re-voicing tried again after a dropped request ({rs_calls}, {rs_applied})")

    # gender: lines measured as the other gender switch only if the
    # dialogue does not say otherwise
    g = DubProject(os.path.join(tmp, "g.mp4"), os.path.join(tmp, "g"), DubOptions())
    g.characters = [Character("S1", name="Thom", gender="M"), Character("S2", gender="F")]
    g.lines = [DubLine(0, 2, "I'm not scared.", speaker="S1"),
               DubLine(2, 4, "Fine, I'm scared!", speaker="S1"),
               DubLine(4, 6, "Shouldn't you be down there?", speaker="S1"),
               DubLine(6, 8, "Let's go.", speaker="S1")]
    meas = {0: ("F", 1.0, 280.0), 2: ("F", 1.0, 270.0), 4: ("F", 1.0, 260.0), 6: ("M", 1.0, 110.0)}
    g.lines.insert(2, DubLine(3, 3.9, "Women are weak!", speaker="S1"))
    meas[3] = ("F", 1.0, 290.0)        # a woman in the crowd, filed under Thom
    stub = _StubAI('{"rows": [{"line": 1, "gender": "male", "evidence": "Celia calls him Tom"},'
                   ' {"line": 2, "gender": "male", "evidence": "answers Celia, as Tom"},'
                   ' {"line": 3, "gender": "male", "evidence": "S1 is speaking"},'
                   ' {"line": 4, "gender": "unknown"}]}')
    ge = DubEngine(ai=stub, config=None)
    with mock.patch.object(ap, "voice_gender", lambda w, s, e: meas[int(s)]), \
            mock.patch(f"{__name__}._speech_source", lambda pp, ln: ""):
        n_sw = ge._detect_genders(g, None, None, keep_characters=True, ai_check=True)
    expect(n_sw == 2 and [l.gender for l in g.lines] == ["", "", "F", "F", ""],
           "dialogue keeps a shouting man male, not without evidence "
           f"({n_sw}, {[l.gender for l in g.lines]})")
    expect("<<CHECK" in stub.prompts[0] and "Fine, I'm scared!" in stub.prompts[0],
           "gender check shows the lines in context")
    # one deep male line among a woman's many lines is voiced male; one
    # high line among a man's lines is not (could be shouting)
    h = DubProject(os.path.join(tmp, "h.mp4"), os.path.join(tmp, "h"), DubOptions())
    h.characters = [Character("S1", gender="F"), Character("S2", gender="M")]
    h.lines = ([DubLine(k * 2, k * 2 + 1.5, f"line {k}", speaker="S1") for k in range(10)]
               + [DubLine(30 + k * 2, 31.5 + k * 2, f"his {k}", speaker="S2") for k in range(10)])
    mh = {int(l.start): ("F", 1.0, 260.0) for l in h.lines[:10]}
    mh.update({int(l.start): ("M", 1.0, 110.0) for l in h.lines[10:]})
    mh[6] = ("M", 1.0, 108.0)          # her 4th line: a man's voice
    mh[36] = ("F", 1.0, 280.0)         # his 4th line: high (shouting?)
    he = DubEngine(ai=_StubAI('{"rows": [{"line": 4, "gender": "unknown"}]}'), config=None)
    with mock.patch.object(ap, "voice_gender", lambda w, s, e: mh[int(s)]), \
            mock.patch(f"{__name__}._speech_source", lambda pp, ln: ""):
        n_h = he._detect_genders(h, None, None, keep_characters=True, ai_check=True)
    expect(n_h == 1 and h.lines[3].gender == "M" and h.lines[13].gender == "",
           f"deep line switched, single high line kept ({n_h})")
    # gender set by hand (a man reborn in a woman's body, voiced by a woman):
    # that character's lines are decided again from the voices measured
    # before, other characters are left alone
    r = DubProject(os.path.join(tmp, "r.mp4"), os.path.join(tmp, "r"), DubOptions())
    r.characters = [Character("S1", name="Su", gender="F", gender_source="user",
                              gender_recheck=True), Character("S2", gender="M")]
    r.lines = [DubLine(k * 2, k * 2 + 1.5, f"su {k}", speaker="S1",
                       voice_gender="F", voice_conf=0.8) for k in range(6)]
    r.lines[2].voice_gender, r.lines[2].voice_conf = "M", 0.94    # a man's lines
    r.lines[3].voice_gender, r.lines[3].voice_conf = "M", 0.81    # filed under Su
    r.lines.append(DubLine(20, 22, "crowd", speaker="S2", voice_gender="F",
                           voice_conf=1.0, gender="F", gender_src="voice"))
    re_ai = _StubAI('{"rows": [{"line": 3, "gender": "male"}, {"line": 4, "gender": "male"}]}')

    def _no_measure(*_a):
        raise AssertionError("re-check must use the voices measured before")
    with mock.patch.object(ap, "voice_gender", _no_measure):
        n_r = DubEngine(ai=re_ai, config=None)._detect_genders(
            r, None, None, keep_characters=True, ai_check=True, only={"S1"})
    expect(n_r == 2 and [l.gender for l in r.lines] == ["", "", "M", "M", "", "", "F"]
           and not r.characters[0].gender_recheck and r.characters[0].gender == "F",
           f"hand-set gender re-checks only that character ({n_r}, "
           f"{[l.gender for l in r.lines]})")
    tags = [DubEngine(ai=None, config=None)._speaker_tag(r, i) for i in (1, 2)]
    expect(tags[0].startswith("[Su (F)") and tags[1].startswith("[another speaker (M)"),
           f"a line in the other voice is not translated as its character ({tags})")
    expect("body-swap" in re_ai.systems[0] and "su 2 <<CHECK" in re_ai.prompts[0]
           and "crowd <<CHECK" not in re_ai.prompts[0],
           "gender check follows the body and voice we hear")
    # the story says male, the voice mostly female: the voice wins only
    # with a reason from the dialogue (rebirth), not for a man shouting
    b = DubProject(os.path.join(tmp, "b.mp4"), os.path.join(tmp, "b"), DubOptions())
    b.characters = [Character("S1", name="Su", gender="M", gender_source="ai"),
                    Character("S2", name="Thom", gender="M", gender_source="ai"),
                    Character("S3", name="Lin", gender="F", gender_source="ai")]
    b.lines = ([DubLine(k * 2, k * 2 + 1.5, f"su {k}", speaker="S1", voice_gender="F",
                        voice_conf=0.4) for k in range(4)]
               + [DubLine(10 + k * 2, 11.5 + k * 2, f"thom {k}", speaker="S2",
                          voice_gender="F", voice_conf=1.0) for k in range(4)]
               + [DubLine(20 + k * 2, 21.5 + k * 2, f"lin {k}", speaker="S3",
                          voice_gender="F" if k else "M", voice_conf=0.9) for k in range(4)])
    b_ai = _StubAI('{"gender": "female", "evidence": "这副身体: a man reborn in a woman\'s body"}',
                   '{"gender": "female", "evidence": ""}')
    flipped = DubEngine(ai=b_ai, config=None)._ai_gender_conflicts(b, None)
    expect(flipped == ["Su"] and [c.gender for c in b.characters] == ["F", "M", "F"]
           and len(b_ai.prompts) == 2 and "rebirth" in b_ai.systems[0],
           f"story vs voice: rebirth voiced as heard, no evidence kept ({flipped})")

    # meaning check rewrites only wrong lines and never takes Latin letters
    m = DubProject(os.path.join(tmp, "m.mp4"), os.path.join(tmp, "m"), DubOptions(target_lang="km"))
    m.src_lang, m.duration = "en", 10.0
    m.characters = [Character("S1", gender="F")]
    m.lines = [DubLine(0, 2, "You're a jerk, Tom.", "លោកជាមនុស្ស", "S1"),
               DubLine(2, 4, "We're done.", "យើងចប់ហើយ។", "S1"),
               DubLine(4, 6, "Roger.", "រ៉ូចឺរ។", "S1")]
    me = DubEngine(ai=_StubAI(
        '{"rows": [{"n": 1, "back": "You are a person", "ok": false, "problem": "insult lost",'
        ' "fix": "ថម ឯងជាមនុស្សអាក្រក់!"}, {"n": 2, "back": "We are done.", "ok": true},'
        ' {"n": 3, "back": "Roger (a name)", "ok": false, "fix": "Roger ទទួលបាន"}]}'), config=None)
    n_fix = me._check_meaning(m, [0, 1, 2], None)
    expect(n_fix == 1 and m.lines[0].text == "ថម ឯងជាមនុស្សអាក្រក់!" and m.lines[1].text == "យើងចប់ហើយ។"
           and m.lines[2].text == "រ៉ូចឺរ។", f"meaning check ({n_fix}, {[l.text for l in m.lines]})")
    expect(me._check_meaning(m, [0, 1, 2], None) == 0 and len(me.ai.prompts) == 1,
           "checked lines are not asked again")
    expect('"you" turned into "they"' in me.ai.systems[0] and "ពួកអ្នក" in me.ai.systems[0],
           "meaning check: a change of person is a meaning error")
    expect("names the listener" in me.ai.systems[0] and "misheard" in me.ai.systems[0]
           and "LISTENER" in me._rules(m) and "លោកបង" in me._rules(m)
           and "ប្អូនប្រុស / ប្អូនស្រី" in me._rules(m) and "shill" in me._rules(m),
           "rules: address words name the listener; misheard words by the scene")
    m.lines[1].text = "ខ្ញុំលឺពួកគាត់។"
    expect(me._fix_spelling(m, [0, 1, 2]) == 1 and m.lines[1].text == "ខ្ញុំឮពួកគាត់។",
           "AI lines get the spelling fix")
    q2 = DubProject.load(m.path)
    expect(len(q2.tr_cache["check"]) == 4, "meaning-check results are saved with the project")

    # an AI limit (all keys rate-limited / out of credit) stops a check so a
    # later run does it, instead of quietly skipping it; the task is marked
    # paused and finished work stays saved. An unreadable reply is still skipped.
    from app.core.errors import ProviderError
    lim = ProviderError("All providers failed. Last error: HTTP 429 rate limited",
                        provider="openrouter", status_code=429)
    m.lines.append(DubLine(6, 8, "Bye.", "លាហើយ។", "S1"))
    le = DubEngine(ai=_StubAI(lim), config=None)
    try:
        le._check_meaning(m, [3], None)
        stopped = False
    except ProviderError:
        stopped = True
    expect(stopped and len(m.tr_cache["check"]) == 4,
           "AI limit stops the meaning check; earlier results kept")
    expect(DubEngine(ai=_StubAI("not json"), config=None)._check_meaning(m, [3], None) == 0,
           "an unreadable AI reply is still skipped")
    pe = DubEngine(ai=None, config=None)

    def _limited(p, progress, cancel):
        raise lim

    pe._step_prepare = _limited
    try:
        pe.run_step(m, "prepare")
        paused = ""
    except AppError as exc:
        paused = exc.en
    note = m.steps["prepare"]["note"]
    expect("paused" in paused and "is saved" in note and "Continue" not in note
           and DubProject.load(m.path).steps["prepare"]["status"] == "failed",
           f"AI limit: the task is paused, the work saved; Continue is left to the panel "
           f"({paused[:80]})")
    expect("rate limit" in paused and "When the limit resets" not in paused
           and m.steps["prepare"].get("ai") == "busy",
           "AI limit: named as a rate limit, and the cause is saved with the task")
    from app.core.ai_engine import NO_KEY_EN, NO_KEY_KM
    nokey = ProviderError(NO_KEY_EN, NO_KEY_KM, provider="no_key")

    def _no_key(p, progress, cancel):
        raise nokey

    pe._step_prepare = _no_key
    try:
        pe.run_step(m, "prepare")
        paused = ""
    except AppError as exc:
        paused = exc.en
    reloaded = DubProject.load(m.path).steps["prepare"]
    expect("No AI key yet" in paused and "AI Providers" in paused and "limit" not in paused
           and "deepseek" not in paused and "is saved" in paused
           and reloaded.get("ai") == "no_key"
           and "មិនទាន់មាន AI key" in reloaded.get("note_km", ""),
           f"no AI key: says so, with where to add one; the Khmer text is saved too "
           f"({paused[:70]})")
    m.lines.pop()
    budget = ProviderError("Monthly AI budget reached: $2.01 of $2.00 spent. Raise the "
                           "limit in Settings → Usage to continue.", provider="budget")

    def _over(p, progress, cancel):
        raise budget

    pe._step_prepare = _over
    try:
        pe.run_step(m, "prepare")
        paused = ""
    except AppError as exc:
        paused = exc.en
    expect("budget reached" in paused and "no AI key" not in paused
           and "is saved" in paused and m.steps["prepare"].get("ai") == "budget",
           f"the app's budget limit is named as the cause ({paused[:70]})")

    # a task that failed for another reason keeps its Khmer message too (a
    # reopened project shows it in Khmer); a plain Python error has none
    def _broken(p, progress, cancel):
        raise AppError("Video not found: x.mp4", "រកមិនឃើញវីដេអូ៖ x.mp4")

    pe._step_prepare = _broken
    try:
        pe.run_step(m, "prepare")
    except AppError:
        pass
    st = DubProject.load(m.path).steps["prepare"]
    expect(st["note"] == "Video not found: x.mp4" and st.get("note_km") == "រកមិនឃើញវីដេអូ៖ x.mp4"
           and "ai" not in st, "a failed task's Khmer message is saved with it")

    def _crash(p, progress, cancel):
        raise ValueError("bad value")

    pe._step_prepare = _crash
    try:
        pe.run_step(m, "prepare")
    except AppError:
        pass
    st = DubProject.load(m.path).steps["prepare"]
    expect(st["note"] == "bad value" and "note_km" not in st,
           "a plain Python error has no Khmer text (shown in English)")

    # readiness: a route whose provider has no key is a note (another keyed
    # provider answers), not a blocker that asks "Continue anyway?" every dub
    from app.core.config_manager import ConfigManager, Route
    rc = ConfigManager(Path(tmp) / "ready.db")
    ai_check = lambda: next(c for c in readiness(rc, DubOptions()) if c.id == "ai")  # noqa: E731
    expect(ai_check().required and not ai_check().ok, "readiness: no AI key at all blocks the dub")
    c = ai_check()
    expect(c.detail == "no AI key yet" and "AI Providers" in c.fix
           and "translate_draft" not in c.detail + c.fix and "Transcribe only" in c.fix,
           "readiness: no AI key is said in plain words, not route names")
    # "Transcribe only": the tasks before the first AI task, then a stop
    ue = DubEngine(ai=None, config=None)
    ran: list[str] = []
    up = DubProject(os.path.join(tmp, "up.mp4"), os.path.join(tmp, "up"), DubOptions())
    for sid in STEP_IDS:
        setattr(ue, f"_step_{sid}", lambda p, progress, cancel, _s=sid: ran.append(_s) or "ok")
    ue.run_until_ai(up)
    expect(ran == ["prepare", "transcribe", "speakers"] and up.next_step() == FIRST_AI_STEP,
           f"Transcribe only runs the tasks without AI, then stops ({ran})")

    # a moved video finds its earlier dub (temp projects folder)
    _real_projects_dir = projects_dir
    projects_dir = lambda: Path(tmp) / "projects"           # noqa: E731
    old_v = os.path.join(tmp, "old place", "Drama EP1.mp4")   # no longer there
    new_v = os.path.join(tmp, "new place", "Drama EP1.mp4")
    os.makedirs(os.path.dirname(new_v))
    Path(new_v).write_bytes(b"x")
    mp = DubProject.for_video(old_v)
    mp.duration = 61.0
    for sid in STEP_IDS:
        mp.set_step(sid, "done")
    mp.output = mp.options.output_path = os.path.join(tmp, "old place", "Drama EP1.km.mp4")
    Path(os.path.join(tmp, "new place", "Drama EP1.km.mp4")).write_bytes(b"x")  # moved too
    mp.save()
    other = DubProject.for_video(os.path.join(tmp, "x", "Drama EP1.mp4"))   # same name,
    other.duration = 300.0                                                  # another video
    other.save()
    with mock.patch.object(ap, "get_duration", lambda v: 61.4):
        found = find_moved_project(new_v)
    expect(found == mp.path, f"moved video: its earlier dub is found by name + length ({found})")
    with mock.patch.object(ap, "get_duration", lambda v: 200.0):
        expect(find_moved_project(new_v) == "", "same name, another length: not that dub")
    expect(not os.path.isfile(DubProject.for_video(new_v).path),
           "before relinking, the new path has no project")
    rl = DubProject.relink(found, new_v)
    again = DubProject.for_video(new_v)
    expect(rl.work_dir == mp.work_dir and again.path == mp.path and again.next_step() is None
           and again.video == os.path.abspath(new_v),
           "relinked: opening the video at its new place opens the earlier dub")
    expect(again.output == os.path.join(tmp, "new place", "Drama EP1.km.mp4")
           and again.options.output_path == again.output,
           "relinked: the dubbed video that moved along is found next to it")
    with mock.patch.object(ap, "get_duration", lambda v: 61.4):
        expect(find_moved_project(new_v) == "", "a relinked project is not offered again")
    projects_dir = _real_projects_dir
    rc.add_keys("openrouter", ["or-test"])
    rc.set_route("translate_draft", Route(provider="custom", model="google/gemini-2.5-flash"))
    rc.set_route("translate_polish", Route(provider="openrouter", model="google/gemini-2.5-flash"))
    c = ai_check()
    expect(not c.required and "answers instead" in c.detail,
           f"readiness: a route without its own key is a note, not a blocker ({c.detail[:60]})")
    rc.set_route("translate_draft", Route(provider="openrouter", model="google/gemini-2.5-flash"))
    expect(ai_check().ok, "readiness: keyed routes are fine")

    # time left: learned per task, per minute of video
    sp = DubProject(os.path.join(tmp, "sp.mp4"), os.path.join(tmp, "sp"), DubOptions())
    sp.duration = 120.0
    sp.set_step("translate", "done")
    sp.steps["translate"]["secs"] = 60.0
    sp.set_step("fit", "done")
    sp.steps["fit"]["secs"] = 1.0                       # reused results: teaches nothing
    learn_speeds(rc, sp, ["translate", "fit"])
    expect(rc.get(SPEED_KEY) == {"translate": [30.0]},
           f"task timings learned per minute of video, instant reruns ignored ({rc.get(SPEED_KEY)})")
    expect(abs(estimate_left(rc, sp, "translate", 0.0, 0.0, ["fit"])
               - (30.0 * 2 + DEFAULT_PACE["fit"] * 2)) < 0.01,
           "a task never timed here (always-fast mix / export) uses the built-in pace")
    sp.steps["fit"]["secs"] = 10.0
    learn_speeds(rc, sp, ["fit"])
    expect(abs(estimate_left(rc, sp, "translate", 0.0, 0.0, ["fit"]) - 70.0) < 0.01,
           "time left = the learned pace of the running task and the ones after it")
    expect(abs(estimate_left(rc, sp, "translate", 0.5, 20.0, ["fit"]) - 30.0) < 0.01,
           "the running task's own pace counts once it has shown some")
    rc.close()

    # long videos: speakers by voice likeness, not by the cleanest split
    # (men vs women won that: a 2-hour drama came out as 2 speakers)
    rs = np.random.default_rng(7)
    g = rs.normal(size=192); g /= np.linalg.norm(g)
    who = [rs.normal(size=192) for _ in range(6)]
    pts, truth = [], []
    # 3 men, 3 women, 40 lines each; like real CAM++ prints: two lines of one
    # person ~0.60 alike, two people of the same gender ~0.32
    for k in range(6):
        for _ in range(40):
            v = (0.55 if k < 3 else -0.55) * g + 0.55 * who[k] / np.linalg.norm(who[k]) \
                + 0.63 * rs.normal(size=192) / np.sqrt(192)
            pts.append(v / np.linalg.norm(v))
            truth.append(k)
    Xl = np.vstack(pts)
    dl = np.full(len(Xl), 2.0)
    old_k = len(set(cluster_speakers(Xl, max_k=10, merge_above=0.7).tolist()))
    new = group_long(Xl, dl)
    pure = new is not None and all(len({truth[i] for i in range(len(new)) if new[i] == c}) == 1
                                   for c in set(new.tolist()))
    expect(old_k == 2 and new is not None and len(set(new.tolist())) == 6 and pure,
           f"long video: 6 voices found by likeness, none mixed (old rule: {old_k})")
    expect(group_long(Xl[:50], dl[:50]) is None,
           "short clips (few steady lines) keep the old speaker rule")

    # meaning check of shortened lines: a fix longer than the line's room is
    # refused (it would undo the shortening), a fix that fits is taken
    ms = DubProject(os.path.join(tmp, "ms.mp4"), os.path.join(tmp, "ms"), DubOptions(target_lang="km"))
    ms.src_lang = "en"
    ms.characters = [Character("S1", gender="M")]
    ms.lines = [DubLine(0, 1, "You're a jerk, Tom.", "បងអាក្រក់ណាស់ ថម។", "S1"),
                DubLine(1, 3, "Synced and locked.", "ធ្វើសមកាលកម្មហើយ។", "S1")]
    room0 = max(ms.max_chars(0), len(ms.lines[0].text))
    long_fix, short_fix = "ក" * (room0 + 4), "ធ្វើសមកាលកម្ម ចាក់សោ។"
    ms_ai = _StubAI(json.dumps({"rows": [
        {"n": 1, "back": "You are very bad", "ok": False, "problem": "word", "fix": long_fix},
        {"n": 2, "back": "Synced.", "ok": False, "problem": "lost locked", "fix": short_fix}]}))
    n_ms = DubEngine(ai=ms_ai, config=None)._check_meaning(ms, [0, 1], None, shortened=True)
    expect(n_ms == 1 and ms.lines[0].text == "បងអាក្រក់ណាស់ ថម។" and ms.lines[1].text == short_fix
           and len(short_fix) <= ms.max_chars(1) and "jerk" in ms_ai.systems[0]
           and "\"locked\"" in ms_ai.systems[0],
           f"shortened-line check: too-long fix refused, fitting fix taken ({n_ms}, "
           f"{[l.text for l in ms.lines]}, budgets {ms.max_chars(0)}/{ms.max_chars(1)})")

    # a line still far too long after shortening is said again more briefly;
    # kept when the meaning check passes, else the first shortening stays.
    # allow_cuts: down to its core, and "only something left out" is fine;
    # the default: tighter wording only — a version that left out a
    # statement goes back to the reworded line (spoken faster instead)
    for cuts in (True, False):
        for verdict in ("ok", "lost", "missing"):
            tag = f"fc{verdict}{int(cuts)}"
            fc = DubProject(os.path.join(tmp, f"{tag}.mp4"), os.path.join(tmp, tag),
                            DubOptions(target_lang="km", quality="high", allow_cuts=cuts))
            fc.src_lang, fc.duration = "en", 10.0
            fc.characters = [Character("S1", gender="M")]
            fc.lines = [DubLine(0, 1.2, "Well, you know, I really think we should go home now.",
                                "ក" * 200, "S1", full_text="ក" * 200)]
            mc = fc.max_chars(0)
            first_cut, core = "ក" * (2 * mc), "ខ" * mc
            check = {"ok": {"n": 1, "back": "Let's go home.", "ok": True},
                     "lost": {"n": 1, "back": "Go.", "ok": False, "problem": "lost 'home'",
                              "fix": "គ" * (3 * mc)},
                     "missing": {"n": 1, "back": "Let's go home.", "ok": False,
                                 "kind": "missing", "problem": "left out 'I think'",
                                 "fix": "គ" * (3 * mc)}}[verdict]
            fc_ai = _StubAI(json.dumps({"rows": [{"index": 0, "text": first_cut}]}),
                            json.dumps({"rows": [{"index": 0, "versions": [first_cut[:-1], core,
                                                                           "ឃ" * 3]}]}),
                            json.dumps({"rows": [check]}),
                            json.dumps({"rows": [{"n": 1, "back": "x", "ok": True}]}))
            _s, fc_note = DubEngine(ai=fc_ai, config=None)._step_fit(fc, None, None)
            took = verdict == "ok" or (cuts and verdict == "missing")
            want = core if took else first_cut
            expect(fc.lines[0].text == want
                   and (("words cut" in fc_note) == (took and cuts))
                   and (("every statement kept" in fc_note) == (took and not cuts))
                   and (("reworded more briefly" in fc_note) == (not took))
                   and (("TWO new versions" in fc_ai.systems[1]) == (not cuts))
                   and fc.lines[0].full_text == "ក" * 200
                   and DubEngine(ai=None).subtitle_text(fc, fc.lines[0]) == "ក" * 200,
                   f"condensing a far-too-long line (cuts {cuts}, {verdict}): "
                   f"{len(fc.lines[0].text)} chars, {fc_note!r}")
    # a second Fit run starts from the whole translation, not from the cut
    # line (it used to shorten the shortened line again)
    rf = DubProject(os.path.join(tmp, "rf.mp4"), os.path.join(tmp, "rf"),
                    DubOptions(target_lang="km", quality="fast"))
    rf.src_lang, rf.duration = "en", 10.0
    rf.characters = [Character("S1", gender="M")]
    rf.lines = [DubLine(0, 1.2, "We should go home now.", "ខ" * 9, "S1", full_text="ក" * 60)]
    rf_ai = _StubAI(json.dumps({"rows": [{"index": 0, "text": "ក" * 20}]}),
                    json.dumps({"rows": [{"index": 0, "versions": ["ក" * 19, "ក" * 18]}]}))
    DubEngine(ai=rf_ai, config=None)._step_fit(rf, None, None)
    expect("ក" * 60 in rf_ai.prompts[0] and "ខ" * 9 not in rf_ai.prompts[0],
           "a re-run of Fit shortens the whole translation, not the shortened line")

    # order of fitting (the user preferred the 2026-11-11 dub, said more
    # briefly at 1.4x, to one sped up to 2.0x): a line needing 1.75x is
    # reworded, not just sped up; a line still over 1.4x after rewording and
    # tightening is spoken faster (up to max_tempo 2.0x), not started late
    od = DubProject(os.path.join(tmp, "od.mp4"), os.path.join(tmp, "od"),
                    DubOptions(target_lang="km", quality="fast"))
    od.src_lang, od.duration = "zh", 20.0
    od.characters = [Character("S1", gender="M")]
    od.lines = [DubLine(0.0, 1.2, "我们走吧", "ក" * 50, "S1", full_text="ក" * 50),
                DubLine(5.0, 6.2, "你说什么", "ខ" * 60, "S1", full_text="ខ" * 60)]
    od_ai = _StubAI(json.dumps({"rows": [{"index": 0, "text": "ក" * 35},
                                         {"index": 1, "text": "ខ" * 50}]}),
                    json.dumps({"rows": [{"index": 1, "versions": ["ខ" * 48, "ខ" * 46]}]}))
    od_st, od_note = DubEngine(ai=od_ai, config=None)._step_fit(od, None, None)
    expect(len(od_ai.prompts) == 2 and [len(l.text) for l in od.lines] == [35, 46]
           and all(l.status == "ok" for l in od.lines) and od_st == "done"
           and "1 reworded more briefly" in od_note and "every statement kept" in od_note
           and "1 spoken faster than 1.4x to fit (up to 2.0x)" in od_note
           and "still long" not in od_note,
           f"fit order: reworded at 1.4x first, extra speed only for the rest "
           f"({[len(l.text) for l in od.lines]}, {od_note!r})")
    # the user's preferred dub is the model for saying lines briefly: its
    # approved lines reach both shortening prompts, and the meaning check
    # knows an echo question or interjection left out is no lost statement
    expect(all("មួយសែនភ្នាល់គាត់ឈ្នះ?" in s and "what did he say?!" in s
               for s in od_ai.systems[:2])
           and "他说什么" in ms_ai.systems[0],
           "model lines in the shortening prompts; the meaning check allows them")

    # shortening: a re-run gives the same short lines without asking again
    sh = DubProject(os.path.join(tmp, "s.mp4"), os.path.join(tmp, "s"), DubOptions(target_lang="km"))
    sh.characters = [Character("S1", gender="M")]
    sh.lines = [DubLine(0, 1, "Robot's memory synced and locked.", "ការចងចាំមនុស្សយន្តត្រូវបានធ្វើសមកាលកម្មហើយចាក់សោ។", "S1"),
                DubLine(1, 2, "I heard you.", "ខ្ញុំឮអ្នកហើយ។", "S1"),
                DubLine(2, 3, "Go.", "ទៅ។", "S1")]
    s_ai = _StubAI('{"rows": [{"index": 0, "text": "ការចងចាំធ្វើសមកាលកម្ម ចាក់សោហើយ។"},'
                   ' {"index": 1, "text": "ខ្ញុំឮអ្នកហើយ។"}]}')
    se2 = DubEngine(ai=s_ai, config=None)
    old = [l.text for l in sh.lines]
    first = (se2._ai_shorten(sh, {0: 30, 1: 10, 2: 2}, None), [l.text for l in sh.lines])
    for ln, t in zip(sh.lines, old):
        ln.text = t
    sh2 = DubProject.load(sh.path)
    for ln, t in zip(sh2.lines, old):
        ln.text = t
    again = (se2._ai_shorten(sh2, {0: 30, 1: 10, 2: 2}, None), [l.text for l in sh2.lines])
    # line 1: "cannot be shorter" is kept too; line 2 got no answer: asked again
    expect(first == again == ([0], ["ការចងចាំធ្វើសមកាលកម្ម ចាក់សោហើយ។", "ខ្ញុំឮអ្នកហើយ។", "ទៅ។"])
           and len(s_ai.prompts) == 2 and "Go." in s_ai.prompts[1]
           and "I heard" not in s_ai.prompts[1] and len(sh2.tr_cache["shorten"]) == 2,
           f"shortened lines are kept: same lines on a re-run ({first}, {again}, "
           f"{len(s_ai.prompts)} calls)")

    # lines typed in Lines stay as written when the translation is redone,
    # and no AI pass (check, spelling, shortening) rewrites them
    he = DubProject(os.path.join(tmp, "he.mp4"), os.path.join(tmp, "he"),
                    DubOptions(target_lang="km", quality="fast", use_tm=False))
    he.src_lang = "en"
    he.characters = [Character("S1", gender="M")]
    he.lines = [DubLine(0, 2, "Hello.", "", "S1"),
                DubLine(2, 4, "I heard you.", "ខ្ញុំលឺអ្នក", "S1", text_src="user"),
                DubLine(4, 6, "Go.", "", "S1")]
    # replies: the mishearing check, the translation, the meaning check
    he_ai = _StubAI('{"rows": []}', "1. សួស្តី\n2. ទៅ", '{"rows": []}')
    he_eng = DubEngine(ai=he_ai, config=None)
    _st, he_note = he_eng._step_translate(he, None, None)
    he_eng._check_meaning(he, [0, 1, 2], None)
    short = he_eng._ai_shorten(he, {1: 3}, None)
    expect([l.text for l in he.lines] == ["សួស្តី", "ខ្ញុំលឺអ្នក", "ទៅ"]
           and all("I heard you" not in pr for pr in he_ai.prompts)
           and "kept as you edited" in he_note and short == [] and len(he_ai.prompts) == 3,
           f"hand-edited line kept ({[l.text for l in he.lines]}, {he_note!r}, "
           f"{len(he_ai.prompts)} calls)")

    # a line left out in Lines ("nobody says this") is not translated, voiced
    # or subtitled; every other line keeps its whole translation (full_text)
    sk = DubProject(os.path.join(tmp, "sk.mp4"), os.path.join(tmp, "sk"),
                    DubOptions(target_lang="km", quality="fast", use_tm=False))
    sk.src_lang = "en"
    sk.characters = [Character("S1", gender="M")]
    sk.lines = [DubLine(0, 2, "Hello.", "", "S1"),
                DubLine(2, 4, "Thanks for watching!", "", "S1", skip=True),
                DubLine(4, 6, "Go.", "", "S1")]
    sk_ai = _StubAI('{"rows": []}', "1. សួស្តី\n2. ទៅ")
    sk_eng = DubEngine(ai=sk_ai, config=None)
    _st, sk_note = sk_eng._step_translate(sk, None, None)
    sk_srt = open(sk_eng.write_srt(sk, os.path.join(tmp, "sk.srt")), encoding="utf-8").read()
    expect([l.text for l in sk.lines] == ["សួស្តី", "", "ទៅ"]
           and [l.full_text for l in sk.lines] == ["សួស្តី", "", "ទៅ"]
           and "Thanks for watching" not in sk_ai.prompts[1]
           and "1 left out as you chose" in sk_note and "empty" not in sk_note
           and sk_srt.count("-->") == 2,
           f"a line left out stays out ({[l.text for l in sk.lines]}, {sk_note!r})")
    # subtitles: the whole translation; what the user typed; nothing for an
    # emptied line; the dub's own words when subtitle_full is off
    sub_p = DubProject(os.path.join(tmp, "sb.mp4"), os.path.join(tmp, "sb"), DubOptions())
    sb_lines = [DubLine(0, 1, "a", "ខ្លី", full_text="ពេញលេញ"),
                DubLine(1, 2, "b", "វាយ", full_text="ចាស់", text_src="user"),
                DubLine(2, 3, "c", "", full_text="ពេញ")]
    got_sub = [sk_eng.subtitle_text(sub_p, l) for l in sb_lines]
    sub_p.options.subtitle_full = False
    got_off = sk_eng.subtitle_text(sub_p, sb_lines[0])
    expect(got_sub == ["ពេញលេញ", "វាយ", ""] and got_off == "ខ្លី",
           f"subtitle text rules ({got_sub}, {got_off})")

    # glossary spellings: a remembered AI translation without them is
    # translated again (the memory kept "goddess system" for 神豪系统 after
    # the glossary said "tycoon system"); spaces do not matter, one-letter
    # address words are not checked, Latin names only as whole words
    gq = DubProject(os.path.join(tmp, "gq.mp4"), os.path.join(tmp, "gq"), DubOptions())
    gq.glossary = [GlossaryTerm("神豪系统", "ប្រព័ន្ធមហាសេដ្ឋី"), GlossaryTerm("哥", "បង"),
                   GlossaryTerm("Tom", "ថម")]
    gm = DubEngine._glossary_misses
    expect([g.source for g in gm(gq, "神豪系统激活", "ប្រព័ន្ធទេពធីតាបានដំណើរការ")] == ["神豪系统"]
           and not gm(gq, "神豪系统激活", "ប្រព័ន្ធ មហាសេដ្ឋី ដំណើរការ")
           and not gm(gq, "哥，走吧", "តោះ")
           and not gm(gq, "See you tomorrow", "ជួបគ្នាថ្ងៃស្អែក")
           and [g.source for g in gm(gq, "Tom, wait!", "ចាំបន្តិច!")] == ["Tom"],
           "glossary check: terms of 2+ characters, spaces ignored, whole Latin words")
    from app.core.translation_memory import TranslationMemory
    tmx = TranslationMemory(os.path.join(tmp, "gq_tm.db"))
    tmx.add("km", "神豪系统激活", "ប្រព័ន្ធទេពធីតាបានដំណើរការ", gender="M", rules=_TR_VERSION)
    tmx.add("km", "走吧", "តោះទៅ", gender="M", rules=_TR_VERSION)
    gt = DubProject(os.path.join(tmp, "gt.mp4"), os.path.join(tmp, "gt"),
                    DubOptions(target_lang="km", quality="fast", use_tm=True))
    gt.src_lang = "zh"
    gt.characters = [Character("S1", gender="M")]
    gt.glossary = [GlossaryTerm("神豪系统", "ប្រព័ន្ធមហាសេដ្ឋី")]
    gt.lines = [DubLine(0, 2, "神豪系统激活", "", "S1"), DubLine(2, 3, "走吧", "", "S1")]
    gt_ai = _StubAI('{"rows": []}', "1. ប្រព័ន្ធមហាសេដ្ឋីបានដំណើរការ")
    with mock.patch("app.core.translation_memory.get_tm", lambda: tmx):
        _st, gt_note = DubEngine(ai=gt_ai, config=None)._step_translate(gt, None, None)
    expect([l.text for l in gt.lines] == ["ប្រព័ន្ធមហាសេដ្ឋីបានដំណើរការ", "តោះទៅ"]
           and "神豪系统激活" in gt_ai.prompts[-1] and "Return EXACTLY 1 lines" in gt_ai.prompts[-1]
           and "1 remembered translations not reused" in gt_note
           and not any(l.review for l in gt.lines),
           f"remembered translation off the glossary is redone ({[l.text for l in gt.lines]}, "
           f"{gt_note!r})")

    # Khmer word order: a known compound in the Chinese order put right in the
    # line and in the glossary; a line typed in Lines stays as typed
    wo = DubProject(os.path.join(tmp, "wo.mp4"), os.path.join(tmp, "wo"),
                    DubOptions(target_lang="km"))
    wo.src_lang = "zh"
    wo.glossary = [GlossaryTerm("班长", "ថ្នាក់ប្រធាន")]
    wo.lines = [DubLine(0, 2, "班长681", "ថ្នាក់ប្រធានបាន៦៨១។"),
                DubLine(2, 4, "总分684", "សរុបពិន្ទុ ៦៨៤", text_src="user"),
                DubLine(4, 6, "校长来了", "នាយកសាលាមកហើយ។")]
    wo_n = DubEngine(ai=None, config=None)._fix_word_order(wo, range(3))
    expect(wo_n == 1 and [ln.text for ln in wo.lines] == ["ប្រធានថ្នាក់បាន៦៨១។", "សរុបពិន្ទុ ៦៨៤",
                                                            "នាយកសាលាមកហើយ។"]
           and wo.glossary[0].target == "ប្រធានថ្នាក់"
           and "ប្រធានថ្នាក់ (never ថ្នាក់ប្រធាន)" in DubEngine(ai=None, config=None)._rules(wo),
           f"nouns in Khmer word order ({wo_n}, {[ln.text for ln in wo.lines]})")
    # numbers: a clear slip put right in code, a broken one re-translated by
    # the AI with its number named, one still wrong marked for a person
    nu = DubProject(os.path.join(tmp, "nu.mp4"), os.path.join(tmp, "nu"),
                    DubOptions(target_lang="km"))
    nu.src_lang = "zh"
    nu.characters = [Character("S1", gender="M")]
    nu.lines = [DubLine(0, 2, "班长681", "ថ្នាក់ប្រធានបានប្រាំរយប៉ែតសិបមួយ។", "S1"),
                DubLine(2, 4, "我不想只考250分", "ខ្ញុំមិនចង់បានត្រឹមពីររយហាស៊ីនទេ!", "S1"),
                DubLine(4, 6, "对750", "ត្រូវហើយ ប្រាំពីររយហាស៊ីន!", "S1"),
                DubLine(6, 8, "他扣了我79分", "គេកាត់ពិន្ទុខ្ញុំ 79 ពិន្ទុ។", "S1"),
                DubLine(8, 10, "第二名628", "លេខពីរ ៦២៨", "S1", text_src="user")]
    nu_ai = _StubAI("1. ខ្ញុំមិនចង់បានត្រឹម 250 ពិន្ទុទេ!\n2. ត្រូវហើយ ប្រាំពីររយហាស៊ីន!")
    nu_fixed, nu_bad = DubEngine(ai=nu_ai, config=None)._fix_numbers(nu, range(5), None)
    expect([ln.text for ln in nu.lines] == ["ថ្នាក់ប្រធានបាន៦៨១។", "ខ្ញុំមិនចង់បានត្រឹម ២៥០ ពិន្ទុទេ!",
                                            "ត្រូវហើយ ប្រាំពីររយហាស៊ីន!", "គេកាត់ពិន្ទុខ្ញុំ ៧៩ ពិន្ទុ។",
                                            "លេខពីរ ៦២៨"]
           and nu_fixed == 2 and nu_bad == [2] and nu.lines[2].review == "number"
           and "line 1: 250; line 2: 750" in nu_ai.prompts[0],
           f"numbers checked ({[ln.text for ln in nu.lines]}, {nu_fixed}, {nu_bad})")

    # hard subtitles correct what was heard (homophones, near-homophones),
    # add a word the recogniser missed, and split where the subtitle changes;
    # a subtitle over silence is not a line, one over a voice is
    from app.engines import subtitle_ocr as so

    def zseg(text, t0, t1, cps=None):
        dur = (t1 - t0) / len(text)
        return _Seg(t0, t1, text, [_W(t0 + k * dur, t0 + (k + 1) * dur, c)
                                   for k, c in enumerate(text)])
    heard_segs = [zseg("你说的是摩迪", 10.0, 11.0), zseg("对啊", 11.0, 11.4),
                  zseg("后座不是只能坐一个人吗", 11.4, 12.8), zseg("他以为我说的是跑车", 14.4, 16.0),
                  zseg("领配川崎H2", 20.0, 21.1)]
    subs = [so.SubEvent(10.1, 11.1, "你说的是摩的？"), so.SubEvent(11.1, 13.1, "对啊后座不是只能坐一个人吗"),
            so.SubEvent(14.1, 16.1, "不是她以为我说的是跑车？"), so.SubEvent(17.0, 17.8, "第一集"),
            so.SubEvent(18.0, 18.8, "顾先生"), so.SubEvent(19.9, 21.1, "顶配川崎H2")]
    voice = lambda s, e: 0.0 if s < 17.9 and e > 16.9 else 0.9      # silence under "第一集"
    fixed_segs, sst = so.apply_to_segments(heard_segs, subs, "zh", voiced=voice)
    got = [s.text for s in fixed_segs]
    expect(got == ["你说的是摩的？", "对啊后座不是只能坐一个人吗", "不是她以为我说的是跑车？",
                   "顾先生", "顶配川崎H2"] and sst.used,
           f"hard subtitles correct the transcript ({got}, {sst})")
    _l, sst2 = so.apply_to_segments(heard_segs, [so.SubEvent(10.0, 16.0, "完全不同的字幕内容在这里显示")],
                                    "zh")
    expect(not sst2.used, f"subtitles that do not match the speech are not used ({sst2.coverage})")
    sl = split_segments(fixed_segs)
    _mark_heard(sl, heard_segs, subs)
    expect(sl[0].on_screen and sl[0].heard == "你说的是摩迪" and not sl[1].heard,
           f"subtitled lines are marked, with what was heard ({[(l.source, l.heard, l.on_screen) for l in sl[:2]]})")

    # Deepgram words (a short drama, 2026-11-11): one subtitle stays one line
    # with one speaker; a subtitle nobody was heard saying joins the words it
    # overlaps instead of landing between their characters ("当，呵，然");
    # a second read of a subtitle (下註) is dropped; 一 read as a dash and a
    # word stretched over 7.7 s still match their subtitle
    def dseg(*words):
        ws = [_W(t0, t1, t, spk) for t, t0, t1, spk in words]
        return _Seg(ws[0].start, ws[-1].end, "".join(w.text for w in ws), ws)
    dg = [dseg(("苏", 11.76, 12.24, "D1"), ("大佬请喝茶", 12.24, 14.08, "D2")),
          dseg(("我能", 20.02, 20.43, "D3"), ("赚", 20.43, 20.66, "D3"), ("吗", 20.66, 21.23, "D3"),
               ("当然", 21.23, 21.86, "D3")),
          dseg(("一个小时", 58.09, 58.80, "D2"), ("前", 58.80, 59.84, "D2")),
          dseg(("这副身体", 81.25, 82.44, "D3"), ("果然", 82.44, 82.84, "D3"), ("还是", 82.84, 83.17, "D3"),
               ("太", 83.17, 83.41, "D3"), ("弱", 83.41, 83.56, "D3"), ("了", 83.56, 91.25, "D1")),
          dseg(("现在怎么跟瘟鸡一样", 100.89, 102.49, "D1"), ("何必", 102.48, 103.67, "D2"),
               ("这么大的火气", 103.67, 105.11, "D2")),
          dseg(("做什么", 111.52, 112.23, "D2"), ("下", 112.23, 112.56, "D3"), ("注", 112.56, 113.19, "D3"))]
    dev = [so.SubEvent(11.5, 13.7, "苏大佬请喝茶"), so.SubEvent(20.1, 21.1, "我能赚吗"),
           so.SubEvent(21.3, 21.7, "呵"), so.SubEvent(22.1, 22.7, "当然"),
           so.SubEvent(58.1, 59.3, "-个小时前"), so.SubEvent(81.1, 83.5, "这副身体果然还是太弱了"),
           so.SubEvent(100.9, 102.1, "现在怎么跟瘟鸡一样"), so.SubEvent(102.3, 103.1, "肖少"),
           so.SubEvent(103.3, 104.9, "何必这么大的火气"), so.SubEvent(111.3, 112.1, "做什么"),
           so.SubEvent(112.1, 112.5, "下注"), so.SubEvent(112.5, 112.9, "下註")]
    dsegs, _dst = so.apply_to_segments(dg, dev, "zh", voiced=lambda s, e: 0.9)
    dl = split_segments(dsegs, lang="zh", quiet=[(12.15, 12.35), (82.4, 82.5)], dips=())
    got = [(l.source, l.asr_speaker) for l in dl]
    expect(got == [("苏大佬请喝茶", "D2"), ("我能赚吗", "D3"), ("呵，当然", "D3"),
                   ("一个小时前", "D2"), ("这副身体果然还是太弱了", "D3"),
                   ("现在怎么跟瘟鸡一样", "D1"), ("肖少，何必这么大的火气", "D2"),
                   ("做什么", "D2"), ("下注", "D3")]
           and all(a.end <= b.start + 0.05 for a, b in zip(dl, dl[1:])),
           f"one subtitle = one whole line on the Deepgram path ({got})")
    # 傻了吧 heard as 傻了罢了: the extra 了 between two subtitles is no line
    psegs, _pst = so.apply_to_segments(
        [dseg(("泡妞", 45.98, 46.38, "D2"), ("傻了", 46.61, 47.02, "D2"), ("罢了", 47.02, 47.66, "D2"),
              ("哈哈哈", 47.81, 48.38, "D3"))],
        [so.SubEvent(45.9, 47.3, "泡妞傻了吧"), so.SubEvent(47.7, 48.5, "哈哈哈")], "zh",
        voiced=lambda s, e: 0.9)
    expect([s.text for s in psegs] == ["泡妞傻了吧", "哈哈哈"],
           f"a lone particle next to a subtitle is no line ({[s.text for s in psegs]})")

    # print genders when a shouting crowd is the clearest split (the calm
    # hero and heroine on one side, 166 / 308 Hz): the calm lines are split,
    # the crowd takes the side its voices are nearer — the men's
    from unittest import mock as _mock
    grng = np.random.default_rng(5)
    gax, lax, bax = np.eye(24)[0], np.eye(24)[1], np.eye(24)[2]
    gv = np.vstack([bax + 0.6 * gax + grng.normal(0, 0.06, 24) for _ in range(10)]
                   + [bax - 0.6 * gax + grng.normal(0, 0.06, 24) for _ in range(10)]
                   + [bax + 0.3 * gax + 2.0 * lax + grng.normal(0, 0.06, 24) for _ in range(12)])
    gv /= np.linalg.norm(gv, axis=1, keepdims=True)
    gpitch = [120.0] * 10 + [215.0] * 10 + [330.0] * 12
    gp = DubProject(os.path.join(tmp, "gp.mp4"), os.path.join(tmp, "gp"), DubOptions())
    gp.lines = [DubLine(k, k + 0.9, f"l{k}") for k in range(32)]
    with _mock.patch.object(ap, "voice_gender", lambda w, s, e: ("", 0.0, gpitch[int(s)])), \
            _mock.patch(f"{__name__}._speech_source", lambda pp, ln: ""):
        pgs = _print_genders(gp, gv, list(range(32)))
    expect([pgs.get(k) for k in range(32)] == ["M"] * 10 + ["F"] * 10 + ["M"] * 12,
           f"a shouting crowd of men is not taken for women ({''.join(pgs.get(k) or '-' for k in range(32))})")
    # ...but a calm split with men's pitch on its "women's" side (a film's
    # two scenes) names nobody
    gv2 = np.vstack([bax + 0.6 * gax + grng.normal(0, 0.06, 24) for _ in range(6)]
                    + [bax - 0.6 * gax + grng.normal(0, 0.06, 24) for _ in range(14)]
                    + [bax + 0.3 * gax + 2.0 * lax + grng.normal(0, 0.06, 24) for _ in range(12)])
    gv2 /= np.linalg.norm(gv2, axis=1, keepdims=True)
    gpitch2 = [140.0] * 6 + [215.0] * 11 + [110.0, 115.0, 120.0] + [330.0] * 12
    with _mock.patch.object(ap, "voice_gender", lambda w, s, e: ("", 0.0, gpitch2[int(s)])), \
            _mock.patch(f"{__name__}._speech_source", lambda pp, ln: ""):
        pgs2 = _print_genders(gp, gv2, list(range(32)))
    expect(pgs2 == {}, f"no genders from a split that is not men / women ({len(pgs2)} named)")

    # Deepgram's label of a line whose voice print is clearly another
    # speaker's moves there; a short line and a lone label's line do not
    rp = DubProject(os.path.join(tmp, "rp.mp4"), os.path.join(tmp, "rp"), DubOptions())
    rp.lines = [DubLine(2 * k, 2 * k + 1.5, f"r{k}") for k in range(14)]
    rp.lines[11].end = rp.lines[11].start + 0.5
    rv = {k: (gax if k < 5 or k in (12,) else lax) + grng.normal(0, 0.05, 24) for k in range(13)}
    rv[10] = lax + grng.normal(0, 0.05, 24)            # said like label 1, filed under 0
    rv[11] = lax + grng.normal(0, 0.05, 24)            # the same, but only 0.5 s
    rv[13] = gax + grng.normal(0, 0.05, 24)            # the only line of label 2
    rv = {k: v / np.linalg.norm(v) for k, v in rv.items()}
    rl = {k: (0 if k < 5 or k in (10, 11, 12) else 1) for k in range(13)}
    rl[13] = 2
    n_mv = _relabel_by_print(rp, rv, rl)
    expect(n_mv == 1 and rl[10] == 1 and rl[11] == 0 and rl[13] == 2 and rp.lines[10].voice_sure,
           f"a line in another speaker's voice moves to them ({n_mv}, {rl})")
    # lines clearly in their own speaker's voice are sure too (the AI speaker
    # check leaves them); a short line and a lone label's line are not
    expect(rp.lines[0].voice_sure and rp.lines[5].voice_sure and not rp.lines[11].voice_sure
           and not rp.lines[13].voice_sure,
           f"clear voices are kept from the AI check ({[k for k in range(14) if rp.lines[k].voice_sure]})")

    # two neighbouring lines in clearly different voices are not joined,
    # whatever speaker label they carry
    vm = [DubLine(27.5, 27.8, "对", speaker="S1"),
          DubLine(27.8, 29.2, "我儿子要是继续读书", speaker="S1", voice_sim=0.14),
          DubLine(29.2, 29.9, "现在也该", speaker="S1", voice_sim=0.7)]
    vm = merge_fragments(vm, "zh")
    expect([l.source for l in vm] == ["对", "我儿子要是继续读书，现在也该"],
           f"different voices are not joined ({[l.source for l in vm]})")

    # a correction from an earlier run comes back when the line is heard the
    # same again; lines under subtitles are not sent to the mishearing check
    mh = DubProject(os.path.join(tmp, "mh.mp4"), os.path.join(tmp, "mh"), DubOptions(target_lang="km"))
    mh.src_lang = "zh"
    mh.lines = [DubLine(0, 1, "领配川崎H2", speaker="S1"), DubLine(1, 2, "你说的是摩的", speaker="S1",
                                                              on_screen=True)]
    mh.tr_cache["misheard"][_sha1("领配川崎H2")] = "顶配川崎H2"
    mh_ai = _StubAI('{"rows": []}')
    mh_fixed = DubEngine(ai=mh_ai, config=None)._fix_mishearings(mh, None)
    expect(mh.lines[0].source == "顶配川崎H2" and mh.lines[0].heard == "领配川崎H2"
           and mh_fixed and not mh_ai.prompts,
           f"remembered correction applied again ({mh.lines[0].source}, {len(mh_ai.prompts)} calls)")

    # CAM++ features: Kaldi-style 80-bin log-mel, 10 ms frames, mean removed
    from app.engines import speaker_embed as _se
    fb = _se.fbank(np.random.default_rng(1).normal(0, 0.1, 16000).astype(np.float32))
    expect(fb is not None and fb.shape == (98, 80) and abs(float(fb.mean())) < 1e-3,
           f"voice-print features ({None if fb is None else fb.shape})")
    expect(_squeeze(1.3) == 1.3, "speech is never sped up beyond max_tempo")

    shutil.rmtree(tmp, ignore_errors=True)
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)
