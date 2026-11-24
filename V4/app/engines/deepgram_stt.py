"""Deepgram speech-to-text (online): a Transcript with word timings and a
speaker label per word (diarization), as an alternative to Whisper.

Deepgram hears Chinese (Mandarin / Cantonese), Thai, Vietnamese, Korean,
Japanese, English and ~40 more languages, but not Khmer, and it does not say
whether a speaker is male or female (the app measures that itself). Billing is
per audio minute on the user's Deepgram account.

No Qt imports; requests only.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Any, Callable

import requests

from app.core.errors import AppError, CancelledError, EngineNotAvailable
from app.core.logger import LOG_BUS, get_logger

log = get_logger("deepgram")

API = "https://api.deepgram.com/v1"
_TIMEOUT = (30, 900)          # (connect, read) — upload + processing
_NOTE_EVERY_S = 20

# Languages per model (Deepgram docs, models & languages overview, 2026-10).
_NOVA3 = {
    "af", "ar", "hy", "as", "be", "bn", "bs", "bg", "ca", "zh", "zh-cn", "zh-hans",
    "zh-tw", "zh-hant", "zh-hk", "hr", "cs", "da", "nl", "en", "et", "fi", "fr",
    "ka", "de", "el", "gu", "he", "hi", "hu", "id", "it", "ja", "kn", "kk", "ko",
    "lv", "lt", "mk", "ms", "mr", "mn", "ne", "no", "ps", "fa", "pl", "pt", "pa",
    "ro", "ru", "sr", "sk", "sl", "es", "sv", "tl", "ta", "te", "th", "tr", "uk",
    "ur", "vi",
}
# languages written without spaces between words
_NO_SPACE = {"zh", "ja", "th", "lo", "my", "km"}

ProgressFn = Callable[[float, str], None]


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


# a dropped connection, a rate limit (429) or a Deepgram server error (5xx)
# is tried again after these waits — one network blip used to fail the
# whole step after uploading a two-hour episode
_RETRY_WAITS = (5, 20)


def _sleep(seconds: float, cancel: threading.Event | None) -> None:
    """Wait, ending at once on Cancel."""
    if cancel is not None:
        if cancel.wait(seconds):
            raise CancelledError("Operation cancelled")
    else:
        time.sleep(seconds)


def supports(language: str) -> bool:
    """True if Deepgram can transcribe `language` ("" = detect: True)."""
    lang = (language or "").strip().lower()
    return not lang or lang in _NOVA3 or lang.split("-")[0] in _NOVA3


def _error_text(resp: requests.Response) -> str:
    try:
        d = resp.json()
        return str(d.get("err_msg") or d.get("message") or d)[:300]
    except ValueError:
        return (resp.text or "")[:300]


def _request(method: str, url: str, cancel: threading.Event | None = None,
             label: str = "Deepgram", **kw) -> requests.Response:
    """requests on a daemon thread, so Cancel works at once and a long wait
    is reported in the console."""
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["resp"] = requests.request(method, url, timeout=_TIMEOUT, **kw)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["exc"] = exc
        finally:
            done.set()

    threading.Thread(target=run, daemon=True, name="deepgram-http").start()
    started, note_at = time.monotonic(), _NOTE_EVERY_S
    while not done.wait(0.25):
        _check_cancel(cancel)
        waited = time.monotonic() - started
        if waited >= note_at:
            LOG_BUS.publish(f"[deepgram] {label}: still working ({int(waited)} s)…")
            note_at += _NOTE_EVERY_S
    exc = box.get("exc")
    if isinstance(exc, requests.RequestException):
        raise AppError(f"Could not reach Deepgram: {exc}",
                       f"មិនអាចភ្ជាប់ទៅ Deepgram បានទេ៖ {exc}") from exc
    if exc is not None:
        raise exc
    return box["resp"]


def check_key(api_key: str) -> str:
    """Raise AppError if the key is not accepted; else a short OK message."""
    if not (api_key or "").strip():
        raise AppError("Enter a Deepgram API key first.",
                       "សូមបញ្ចូល Deepgram API key ជាមុនសិន។")
    r = _request("GET", f"{API}/projects",
                 headers={"Authorization": f"Token {api_key.strip()}"})
    if r.status_code in (401, 403):
        raise AppError(f"Deepgram rejected this key: {_error_text(r)}",
                       "Deepgram មិនទទួលយក key នេះទេ។")
    if r.status_code >= 400:
        raise AppError(f"Deepgram error {r.status_code}: {_error_text(r)}",
                       f"កំហុស Deepgram {r.status_code}")
    n = len((r.json() or {}).get("projects") or [])
    return f"Deepgram key OK ({n} project{'s' if n != 1 else ''})"


def _compress(audio: str, cancel: threading.Event | None) -> str:
    """Mono 16 kHz Opus: ~15 MB per hour instead of ~115 MB of WAV."""
    from app.engines import audio_pipeline as ap
    out = os.path.join(tempfile.mkdtemp(prefix="rs_deepgram_"), "speech.ogg")
    ap._run_ffmpeg(["-i", audio, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                    "-b:a", "32k", "-application", "voip", out],
                   cancel=cancel, status_text="Preparing audio for Deepgram…")
    return out


def transcribe(audio: str, api_key: str, language: str = "", diarize: bool = True,
               progress: ProgressFn | None = None,
               cancel: threading.Event | None = None):
    """Transcribe `audio` (any file ffmpeg reads). `language` "" = detect.
    Every Segment / Word carries speaker "D1", "D2", … when `diarize`."""
    from app.engines.audio_pipeline import Segment, Transcript, Word

    _check_cancel(cancel)
    key = (api_key or "").strip()
    if not key:
        raise EngineNotAvailable(
            "Deepgram is chosen for speech-to-text but no Deepgram key is saved "
            "(Settings → Speech services).",
            "បានជ្រើស Deepgram ប៉ុន្តែមិនទាន់មាន Deepgram key ទេ (ការកំណត់ → Speech services)។")
    lang = (language or "").strip().lower()
    if not supports(lang):
        raise AppError(
            f"Deepgram cannot transcribe '{language}'. Choose Whisper in "
            "Settings → Speech services for this language.",
            f"Deepgram មិនអាចស្តាប់ភាសា '{language}' បានទេ។ សូមជ្រើស Whisper ក្នុងការកំណត់។")

    if progress:
        progress(0.02, "Preparing audio for Deepgram…")
    clip = _compress(audio, cancel)
    params: dict[str, Any] = {"model": "nova-3", "smart_format": "true",
                              "punctuate": "true", "utterances": "true",
                              "utt_split": "0.8"}
    if lang:
        params["language"] = lang
    else:
        params["model"] = "nova-3-general"
        params["detect_language"] = "true"
    if diarize:
        params["diarize_model"] = "latest"
    try:
        mb = os.path.getsize(clip) / 1048576
        for attempt in range(len(_RETRY_WAITS) + 1):
            if progress:
                progress(0.1, f"Sending {mb:.1f} MB to Deepgram…"
                         + (f" (try {attempt + 1})" if attempt else ""))
            last = attempt == len(_RETRY_WAITS)
            try:
                with open(clip, "rb") as f:
                    r = _request("POST", f"{API}/listen", cancel=cancel, label="transcribing",
                                 params=params, data=f,
                                 headers={"Authorization": f"Token {key}",
                                          "Content-Type": "audio/ogg"})
            except AppError as exc:          # the connection failed: worth another try
                if last:
                    raise
                why = exc.en
            else:
                if not (r.status_code == 429 or r.status_code >= 500) or last:
                    break
                why = f"HTTP {r.status_code}: {_error_text(r)}"
            wait = _RETRY_WAITS[attempt]
            log.warning("Deepgram request failed (%s); trying again in %d s", why, wait)
            LOG_BUS.publish(f"[deepgram] request failed ({why}) — trying again in {wait} s")
            _sleep(wait, cancel)
    finally:
        try:
            os.remove(clip)
            os.rmdir(os.path.dirname(clip))
        except OSError:
            pass
    if r.status_code in (401, 403):
        raise AppError(f"Deepgram rejected the API key: {_error_text(r)}",
                       "Deepgram មិនទទួលយក API key ទេ។ សូមពិនិត្យក្នុងការកំណត់។")
    if r.status_code == 402:
        raise AppError("Your Deepgram account has no credit left.",
                       "គណនី Deepgram របស់អ្នកអស់ឥណទានហើយ។")
    if r.status_code >= 400:
        raise AppError(f"Deepgram error {r.status_code}: {_error_text(r)}",
                       f"កំហុស Deepgram {r.status_code}៖ {_error_text(r)}")
    data = r.json()
    if progress:
        progress(0.95, "Reading Deepgram's transcript…")

    res = data.get("results") or {}
    channel = (res.get("channels") or [{}])[0]
    detected = str(channel.get("detected_language") or lang or "")
    base = detected.split("-")[0].lower()
    spaced = base not in _NO_SPACE

    def word(w: dict, first: bool) -> Word:
        text = str(w.get("punctuated_word") or w.get("word") or "")
        if spaced and not first:
            text = " " + text            # Whisper-style, so "".join() reads
        spk = w.get("speaker")
        return Word(start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0)),
                    text=text, speaker=(f"D{int(spk) + 1}" if diarize and spk is not None
                                        else None))

    segments: list = []
    for u in res.get("utterances") or []:
        ws = [word(w, k == 0) for k, w in enumerate(u.get("words") or [])]
        text = str(u.get("transcript") or "".join(w.text for w in ws)).strip()
        if not text:
            continue
        spk = u.get("speaker")
        segments.append(Segment(
            start=float(u.get("start", 0.0)), end=float(u.get("end", 0.0)),
            text=text, words=ws,
            speaker=f"D{int(spk) + 1}" if diarize and spk is not None else None))
    if not segments:
        # no utterances block: one segment per speaker turn / pause
        alt = (channel.get("alternatives") or [{}])[0]
        cur: list = []
        for w in alt.get("words") or []:
            ww = word(w, not cur)
            if cur and (ww.speaker != cur[-1].speaker or ww.start - cur[-1].end > 0.8):
                segments.append(Segment(cur[0].start, cur[-1].end,
                                        "".join(x.text for x in cur).strip(), cur,
                                        speaker=cur[0].speaker))
                cur, ww = [], word(w, True)
            cur.append(ww)
        if cur:
            segments.append(Segment(cur[0].start, cur[-1].end,
                                    "".join(x.text for x in cur).strip(), cur,
                                    speaker=cur[0].speaker))
    duration = float((data.get("metadata") or {}).get("duration") or 0.0)
    if not duration and segments:
        duration = segments[-1].end
    n_spk = len({s.speaker for s in segments if s.speaker})
    LOG_BUS.publish(f"[deepgram] {len(segments)} utterances, language {detected or '?'}"
                    + (f", {n_spk} speakers" if diarize else ""))
    if progress:
        progress(1.0, f"Deepgram: {len(segments)} utterances")
    return Transcript(segments=segments, language=base or detected, duration=duration)


if __name__ == "__main__":
    # offline: parsing a canned response (no network, no key)
    canned = {
        "metadata": {"duration": 6.0},
        "results": {
            "channels": [{"detected_language": "zh", "alternatives": [{"words": []}]}],
            "utterances": [
                {"start": 0.5, "end": 2.0, "speaker": 0, "transcript": "你去哪里？",
                 "words": [{"word": "你", "start": 0.5, "end": 0.8, "speaker": 0,
                            "punctuated_word": "你"},
                           {"word": "去哪里", "start": 0.8, "end": 2.0, "speaker": 0,
                            "punctuated_word": "去哪里？"}]},
                {"start": 2.4, "end": 4.0, "speaker": 1, "transcript": "回家。",
                 "words": [{"word": "回家", "start": 2.4, "end": 4.0, "speaker": 1,
                            "punctuated_word": "回家。"}]},
            ]}}

    class _Resp:
        status_code = 200

        def json(self):
            return canned

    globals()["_request"] = lambda *a, **k: _Resp()
    globals()["_compress"] = lambda audio, cancel: tempfile.mkstemp(suffix=".ogg")[1]
    tr = transcribe("x.wav", "key")
    assert tr.language == "zh" and abs(tr.duration - 6.0) < 1e-9
    assert [s.speaker for s in tr.segments] == ["D1", "D2"], tr.segments
    assert "".join(w.text for w in tr.segments[0].words) == "你去哪里？"
    assert tr.segments[1].words[0].speaker == "D2"
    assert supports("zh") and supports("zh-TW") and supports("") and not supports("km")

    # a dropped connection and a server error are tried again; a refused
    # key is not, and a server that keeps failing stops after the last try
    globals()["_RETRY_WAITS"] = (0, 0)
    calls: list[int] = []

    class _Err:
        text = "busy"

        def __init__(self, code: int):
            self.status_code = code

        def json(self):
            return {"err_msg": "busy"}

    def flaky(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            raise AppError("Could not reach Deepgram: connection reset")
        return _Err(503) if len(calls) == 2 else _Resp()

    globals()["_request"] = flaky
    tr = transcribe("x.wav", "key")
    assert len(calls) == 3 and len(tr.segments) == 2, calls
    for code, tries in ((401, 1), (503, 3), (400, 1)):
        calls.clear()
        globals()["_request"] = lambda *_a, _c=code, **_k: (calls.append(1), _Err(_c))[1]
        try:
            transcribe("x.wav", "key")
            raise AssertionError(f"HTTP {code} must fail")
        except AppError:
            pass
        assert len(calls) == tries, (code, calls)
    print("PASS")
