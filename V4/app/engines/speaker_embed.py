"""Speaker fingerprints with CAM++ (3D-Speaker, trained on Chinese and
English speech) as ONNX: a 192-number vector per stretch of speech; two
stretches of the same voice point the same way (cosine).

onnxruntime + numpy only. speechbrain's ECAPA needs torchaudio, whose
library Windows Smart App Control blocks on some PCs; without a model the
dub fell back to plain voice features that mixed up two speakers on most
Chinese dramas (separation 0.2).

No Qt imports."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from app.core.errors import CancelledError
from app.core.logger import get_logger
from app.core.paths import models_dir

log = get_logger("speaker_embed")

MODEL_ID = "speaker-campplus"
_DIR = "speaker"
_NAME = "campplus_zh_en_16k.onnx"
# two known good versions: the file was re-exported upstream (PyTorch 2.14.1)
# in Nov 2026 — one byte more, another sha256, the same voice prints (44
# speech clips: identical to 5 decimals). The old fixed sha256 refused every
# download: "campplus_zh_en_16k.onnx: wrong file".
FILES: dict[str, tuple] = {
    _NAME: ((28281164, 28281165),
            ("aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
             "53cab20ba8f2eb2f6e2d89ecc18e88ef3222169b61b4419cbc13666744a4574b"), [
        "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/"
        "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
        "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"]),
}
SIZE_MB = 28
SR = 16000


def model_dir() -> Path:
    return models_dir() / _DIR


def model_present() -> bool:
    p = model_dir() / _NAME
    sizes = FILES[_NAME][0]
    return p.is_file() and p.stat().st_size in (sizes if isinstance(sizes, tuple) else (sizes,))


def available() -> bool:
    if not model_present():
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Kaldi-compatible log-mel filterbank (what the model was trained on:
# 25 ms povey windows every 10 ms, pre-emphasis 0.97, 80 bins 20 Hz–8 kHz)
# ---------------------------------------------------------------------------

_WIN, _HOP, _NFFT, _BINS = 400, 160, 512, 80
_BANK: np.ndarray | None = None
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WIN) / (_WIN - 1))) ** 0.85


def _mel_bank() -> np.ndarray:
    global _BANK
    if _BANK is None:
        def mel(f):
            return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)
        lo, hi = mel(20.0), mel(SR / 2.0)
        delta = (hi - lo) / (_BINS + 1)
        b = np.arange(_BINS)[:, None]
        left = lo + b * delta
        center = left + delta
        right = center + delta
        m = mel(np.arange(_NFFT // 2) * SR / _NFFT)[None, :]
        w = np.maximum(0.0, np.minimum((m - left) / (center - left), (right - m) / (right - center)))
        _BANK = np.pad(w, ((0, 0), (0, 1)))          # the Nyquist bin gets no weight
    return _BANK


def fbank(x: np.ndarray) -> np.ndarray | None:
    """(frames, 80) log-mel features of 16 kHz mono float audio in [-1, 1],
    each bin's mean over the clip removed; None when shorter than a frame."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < _WIN:
        return None
    n = 1 + (x.size - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)
    pre = fr.copy()
    pre[:, 1:] -= 0.97 * fr[:, :-1]
    pre[:, 0] -= 0.97 * fr[:, 0]
    spec = np.abs(np.fft.rfft(pre * _WINDOW, n=_NFFT)) ** 2
    feat = np.log(np.maximum(spec @ _mel_bank().T, np.finfo(np.float32).eps))
    return (feat - feat.mean(axis=0, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

_SESS = None
_LOCK = threading.Lock()


def _session():
    global _SESS
    with _LOCK:
        if _SESS is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
            so.log_severity_level = 3
            _SESS = ort.InferenceSession(str(model_dir() / _NAME), so,
                                         providers=["CPUExecutionProvider"])
        return _SESS


def embed(x: np.ndarray, sr: int = SR) -> np.ndarray | None:
    """Unit-length fingerprint of one stretch of speech (float audio)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr != SR and x.size:
        # linear resampling is enough for a voice fingerprint
        t = np.linspace(0.0, x.size / sr, int(round(x.size * SR / sr)), endpoint=False)
        x = np.interp(t, np.arange(x.size) / sr, x).astype(np.float32)
    if x.size < int(0.3 * SR):
        return None
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:                  # int16 scale given: back to [-1, 1]
        x = x / 32768.0
    f = fbank(x)
    if f is None or f.shape[0] < 20:
        return None
    s = _session()
    with _LOCK:
        e = s.run(None, {s.get_inputs()[0].name: f[None]})[0][0]
    e = np.asarray(e, dtype=np.float64)
    nrm = float(np.linalg.norm(e))
    return e / nrm if nrm > 0 else None


def embed_spans(wav: str, spans: list[tuple[float, float]],
                cancel: threading.Event | None = None,
                max_seconds: float = 10.0) -> list[np.ndarray | None]:
    """A fingerprint per (start, end) span of `wav`."""
    from app.engines import audio_pipeline as ap
    out: list[np.ndarray | None] = []
    for s, e in spans:
        if cancel is not None and cancel.is_set():
            raise CancelledError("cancelled")
        x, sr = ap._read_wav_slice(wav, s, e, max_seconds=max_seconds)
        try:
            out.append(embed(x, sr))
        except Exception as exc:
            log.warning("speaker fingerprint failed at %.1f s (%s)", s, exc)
            out.append(None)
    return out
