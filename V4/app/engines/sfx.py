"""Procedural sound effects for recap videos: transition whooshes and
cinematic impact hits, synthesized with numpy — no sample files, no licences.

Every effect takes a seed, so each cut gets a slightly different whoosh
(length, brightness, sweep) instead of one sample repeated 40 times.

    build_sfx_track(cuts=[3.2, 7.9, ...], total=80.0, out_wav="sfx.wav",
                    hits=[0.0, 74.5])

No Qt imports. Only stdlib + numpy.
"""
from __future__ import annotations

import wave

import numpy as np

SR = 48000
WHOOSH_PEAK = 0.7          # fraction of a whoosh where it is loudest (lands on the cut)


def _svf_bandpass(x: np.ndarray, fc: np.ndarray, q: float, sr: int) -> np.ndarray:
    """Chamberlin state-variable band-pass with a per-sample cutoff."""
    f = 2.0 * np.sin(np.pi * np.clip(fc, 20.0, sr / 7.0) / sr)
    damp = 1.0 / max(0.5, q)
    low = band = 0.0
    out = np.empty_like(x)
    for i in range(x.size):
        high = x[i] - low - damp * band
        band += f[i] * high
        low += f[i] * band
        out[i] = band
    return out


def _normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.abs(x).max()) if x.size else 0.0
    return x * (peak / m) if m > 1e-9 else x


def whoosh(seed: int = 0, sr: int = SR) -> np.ndarray:
    """Air-swoosh: band-passed noise whose level and brightness swell to a
    peak at WHOOSH_PEAK of its length, then fall away quickly."""
    rng = np.random.default_rng(seed)
    dur = rng.uniform(0.45, 0.8)
    n = int(dur * sr)
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    p = WHOOSH_PEAK
    env = np.where(u < p, (u / p) ** 2.2, (1.0 - (u - p) / (1.0 - p)) ** 1.6)
    top = rng.uniform(2500.0, 5500.0)
    fc = 250.0 + top * env ** 1.5
    noise = rng.standard_normal(n)
    body = _svf_bandpass(noise, fc, q=rng.uniform(0.9, 1.6), sr=sr)
    air = _svf_bandpass(rng.standard_normal(n), fc * 2.2, q=2.5, sr=sr) * 0.35
    return _normalize((body + air) * env)


def hit(seed: int = 0, sr: int = SR) -> np.ndarray:
    """Cinematic impact: a sub 'boom' that drops in pitch, a short noise
    crack on top, and a dark reverb-like tail."""
    rng = np.random.default_rng(seed)
    dur = 1.8
    n = int(dur * sr)
    t = np.arange(n) / sr
    freq = 36.0 + rng.uniform(70.0, 95.0) * np.exp(-t / 0.07)
    boom = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    attack = np.clip(t / 0.003, 0.0, 1.0)
    boom *= attack * np.exp(-t / 0.5)
    crack = rng.standard_normal(n) * np.exp(-t / 0.015)
    crack = np.convolve(crack, np.ones(6) / 6, "same")          # soften the top end
    ir_n = int(0.9 * sr)                                         # dark noise tail
    ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / (0.25 * sr))
    ir = np.convolve(ir, np.ones(40) / 40, "same")
    m = n + ir_n
    k = 1 << (m - 1).bit_length()
    tail = np.fft.irfft(np.fft.rfft(boom + 0.5 * crack, k) * np.fft.rfft(ir, k), k)[:n]
    mix = np.tanh(1.6 * (boom + 0.45 * crack)) + 0.25 * _normalize(tail)
    return _normalize(mix)


def build_sfx_track(cuts: list[float], total: float, out_wav: str,
                    hits: list[float] | tuple = (), sr: int = SR,
                    seed: int = 0) -> str:
    """Mono 16-bit wav of `total` seconds with a whoosh peaking on every cut
    time and an impact starting at every hit time."""
    buf = np.zeros(int(round(total * sr)) + sr, dtype=np.float64)

    def place(x: np.ndarray, start: float) -> None:
        off = int(round(start * sr))
        if off < 0:
            x, off = x[-off:], 0
        n = min(x.size, buf.size - off)
        if n > 0:
            buf[off:off + n] += x[:n]

    for i, c in enumerate(cuts):
        w = whoosh(seed * 1000 + i, sr)
        place(w, c - WHOOSH_PEAK * w.size / sr)
    for j, h in enumerate(hits):
        place(hit(seed * 1000 + 500 + j, sr), h)
    buf = np.tanh(buf)                                           # overlaps never clip
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((buf * 32767).astype("<i2").tobytes())
    return out_wav


if __name__ == "__main__":
    import os
    import tempfile

    w = whoosh(1)
    assert 0.45 * SR <= w.size <= 0.8 * SR and abs(np.abs(w).max() - 0.9) < 1e-6
    peak_at = np.argmax(np.convolve(np.abs(w), np.ones(480), "same")) / w.size
    assert 0.55 < peak_at < 0.8, peak_at                         # loudest near the cut
    assert not np.allclose(whoosh(1)[:1000], whoosh(2)[:1000])   # seeds vary
    h = hit(0)
    spec = np.abs(np.fft.rfft(h[: SR // 2]))
    freqs = np.fft.rfftfreq(SR // 2, 1 / SR)
    assert spec[freqs < 150].sum() > spec[freqs > 2000].sum(), "impact is bass-heavy"
    out = os.path.join(tempfile.mkdtemp(), "sfx.wav")
    build_sfx_track([1.0, 2.5], 4.0, out, hits=[0.0])
    with wave.open(out) as wf:
        assert wf.getframerate() == SR and abs(wf.getnframes() / SR - 5.0) < 0.01
    print("PASS")
