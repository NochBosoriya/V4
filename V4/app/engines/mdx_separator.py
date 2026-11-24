"""Voice / music separation without PyTorch: an MDX-Net ONNX model (from the
Ultimate Vocal Remover model set) run with onnxruntime, so the .exe build —
which has no Demucs — can keep the background music and effects without the
original voices.

The audio is processed in overlapping ~6 s windows straight from a WAV file,
so a long episode needs little memory. numpy does the STFT; onnxruntime picks
CUDA / DirectML when installed, else the CPU.

No Qt imports. onnxruntime is imported lazily.
"""
from __future__ import annotations

import os
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.errors import CancelledError, EngineNotAvailable
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import models_dir, pip_install

log = get_logger("mdx")

ProgressFn = Callable[[float, str], None]

# Kim Vocal 2 (settings from UVR's model data, hash 970b3f94…). On clean
# speech mixed over real drama music it scored voices 16.3 dB / music 19.4 dB
# SDR, against 14.4 / 17.3 for Inst HQ 3 and 9.0 / 13.9 for Demucs htdemucs.
MODEL = {
    "file": "Kim_Vocal_2.onnx",
    "url": ("https://github.com/TRvlvr/model_repo/releases/download/"
            "all_public_uvr_models/Kim_Vocal_2.onnx"),
    "size_mb": 64,
    "bytes": 66759214,
    "n_fft": 7680, "dim_f": 3072, "dim_t": 256, "hop": 1024,
    "compensate": 1.009,
    "primary": "vocals",            # the model outputs the voices; music = mix - voices
}
SR = 44100
_BATCH = 2


def model_path() -> Path:
    return models_dir() / "mdx" / MODEL["file"]


def model_present() -> bool:
    p = model_path()
    return p.is_file() and p.stat().st_size >= MODEL["bytes"] * 0.99


def _onnxruntime():
    try:
        import onnxruntime as ort  # lazy heavy dep
    except ImportError as exc:
        raise EngineNotAvailable(
            f"onnxruntime is not installed ({pip_install('onnxruntime')}).",
            f"មិនទាន់បានដំឡើង onnxruntime ទេ ({pip_install('onnxruntime')})។") from exc
    return ort


def available() -> bool:
    """True when separation can run right now (package + model)."""
    try:
        _onnxruntime()
    except EngineNotAvailable:
        return False
    return model_present()


_SESSION = None
_SESSION_LOCK = threading.Lock()


def _session():
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            ort = _onnxruntime()
            if not model_present():
                raise EngineNotAvailable(
                    "The voice/music separation model is not downloaded yet "
                    "(Settings → Models, or the Full Dub checklist).",
                    "មិនទាន់បានទាញយកម៉ូដែលបំបែកសំឡេង/តន្ត្រីទេ។")
            try:
                from app.core.gpu import onnx_providers
                providers = onnx_providers()
            except Exception:
                providers = ["CPUExecutionProvider"]
            try:
                _SESSION = ort.InferenceSession(str(model_path()), providers=providers)
            except Exception as exc:     # e.g. CUDA libraries missing
                log.warning("separation on %s failed to start (%s); using CPU",
                            providers[0], exc)
                _SESSION = ort.InferenceSession(str(model_path()),
                                                providers=["CPUExecutionProvider"])
            LOG_BUS.publish(f"[separate] MDX-Net on {_SESSION.get_providers()[0]}")
        return _SESSION


# -- STFT matching torch.stft(center=True, periodic Hann) ---------------------

_N_FFT, _HOP = MODEL["n_fft"], MODEL["hop"]
_DIM_F, _DIM_T = MODEL["dim_f"], MODEL["dim_t"]
_CHUNK = _HOP * (_DIM_T - 1)                     # samples per window
_TRIM = _N_FFT // 2
_GEN = _CHUNK - 2 * _TRIM                        # new samples per window
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_N_FFT) / _N_FFT)
_FRAME_IDX = np.arange(_N_FFT)[None, :] + _HOP * np.arange(_DIM_T)[:, None]


def _stft(x: np.ndarray) -> np.ndarray:
    """[2, CHUNK] stereo -> model input [4, DIM_F, DIM_T] (L re, L im, R re, R im)."""
    pad = _N_FFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    spec = np.fft.rfft(xp[:, _FRAME_IDX] * _WINDOW, axis=-1)     # [2, T, bins]
    spec = spec.transpose(0, 2, 1)[:, :_DIM_F, :]                  # [2, F, T]
    return np.stack([spec[0].real, spec[0].imag, spec[1].real, spec[1].imag]
                    ).astype(np.float32)


def _istft(spec: np.ndarray) -> np.ndarray:
    """Model output [4, DIM_F, DIM_T] -> [2, CHUNK] stereo."""
    n_bins = _N_FFT // 2 + 1
    full = np.zeros((4, n_bins, _DIM_T), dtype=np.float64)
    full[:, :_DIM_F] = spec
    c = full[0::2] + 1j * full[1::2]                               # [2, bins, T]
    frames = np.fft.irfft(c.transpose(0, 2, 1), n=_N_FFT, axis=-1) * _WINDOW
    length = _N_FFT + _HOP * (_DIM_T - 1)
    y = np.zeros((2, length))
    wsum = np.zeros(length)
    w2 = _WINDOW ** 2
    for t in range(_DIM_T):
        s = t * _HOP
        y[:, s:s + _N_FFT] += frames[:, t]
        wsum[s:s + _N_FFT] += w2
    y /= np.maximum(wsum, 1e-8)
    pad = _N_FFT // 2
    return y[:, pad:pad + _CHUNK]


# -- WAV helpers ----------------------------------------------------------------

def _read_window(wf: wave.Wave_read, n: int, start: int, length: int) -> np.ndarray:
    """[2, length] float32 from a 16-bit stereo WAV, zero outside [0, n)."""
    out = np.zeros((2, length), dtype=np.float32)
    a, b = max(0, start), min(n, start + length)
    if b > a:
        wf.setpos(a)
        raw = np.frombuffer(wf.readframes(b - a), dtype=np.int16)
        out[:, a - start:b - start] = raw.reshape(-1, 2).T / 32768.0
    return out


def _writer(path: str) -> wave.Wave_write:
    w = wave.open(path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    return w


def _write(w: wave.Wave_write, x: np.ndarray) -> None:
    w.writeframes((np.clip(x.T, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())


def separate(audio: str, out_dir: str, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> tuple[str, str]:
    """Split `audio` into (vocals_wav, accompaniment_wav), 44.1 kHz stereo,
    like audio_pipeline.separate_vocals() does with Demucs."""
    from app.engines import audio_pipeline as ap

    sess = _session()
    os.makedirs(out_dir, exist_ok=True)
    mix_wav = os.path.join(out_dir, "mix_44k.wav")
    vocals = os.path.join(out_dir, "vocals.wav")
    music = os.path.join(out_dir, "no_vocals.wav")
    if progress:
        progress(0.01, "Separating voices from music…")
    ap._run_ffmpeg(["-i", audio, "-vn", "-ac", "2", "-ar", str(SR),
                    "-acodec", "pcm_s16le", mix_wav],
                   cancel=cancel, status_text="Preparing audio…")
    name = sess.get_inputs()[0].name
    comp = float(MODEL["compensate"])
    try:
        with wave.open(mix_wav, "rb") as wf, \
                _writer(vocals) as wv, _writer(music) as wm:
            n = wf.getnframes()
            starts = list(range(0, n, _GEN))
            for k in range(0, len(starts), _BATCH):
                if cancel is not None and cancel.is_set():
                    raise CancelledError("Operation cancelled")
                group = starts[k:k + _BATCH]
                windows = [_read_window(wf, n, s - _TRIM, _CHUNK) for s in group]
                pred = sess.run(None, {name: np.stack([_stft(w) for w in windows])})[0]
                for s, w, p in zip(group, windows, pred):
                    take = min(_GEN, n - s)
                    mix = w[:, _TRIM:_TRIM + take]
                    prim = _istft(p)[:, _TRIM:_TRIM + take] * comp
                    other = mix - prim
                    voc, mus = (other, prim) if MODEL["primary"] == "instrumental" \
                        else (prim, other)
                    _write(wv, voc)
                    _write(wm, mus)
                if progress:
                    done = min(n, group[-1] + _GEN)
                    progress(0.02 + 0.97 * done / max(1, n),
                             f"Separating voices from music… {100 * done / max(1, n):.0f}%")
    except BaseException:
        for p in (vocals, music):
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    finally:
        try:
            os.remove(mix_wav)
        except OSError:
            pass
    if progress:
        progress(1.0, "Voices separated")
    return vocals, music


if __name__ == "__main__":
    # STFT round trip must be (near) lossless for the model's band
    rng = np.random.default_rng(0)
    t = np.arange(_CHUNK) / SR
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 330 * t)])
    x += 0.01 * rng.standard_normal(x.shape)
    y = _istft(_stft(x))
    mid = slice(_N_FFT, _CHUNK - _N_FFT)
    err = float(np.max(np.abs(y[:, mid] - x[:, mid])))
    # bins above DIM_F (≈ 17.6 kHz) are left out of the model: noise only
    assert y.shape == x.shape and err < 0.05, err
    assert _stft(x).shape == (4, _DIM_F, _DIM_T)
    print("PASS")
