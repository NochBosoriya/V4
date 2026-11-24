"""Per-series memory shared by every episode of a drama.

Stored as ``app_data_dir()/series/<name>/memory.json`` (+ the cast's voice
samples in the same folder):

  glossary  names / terms with their fixed spelling per target language
  cast      characters with gender, voice settings, a voice sample and a
            voice fingerprint, so a new episode's speakers are recognised
            and keep the same name and (cloned) voice
  story     running story summary

No Qt imports; numpy only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from app.core.logger import get_logger
from app.core.paths import app_data_dir

log = get_logger("series_memory")

_LOCK = threading.RLock()


def _safe(name: str) -> str:
    r"""Folder-safe name. Keeps letters *and* combining marks (Khmer, Thai,
    Devanagari vowels are marks, not \w) and adds a short hash so names that
    differ only in characters Windows cannot store never share a folder."""
    raw = (name or "").strip()
    s = "".join(ch if (ch.isalnum() or unicodedata.category(ch).startswith("M")
                       or ch in "-. _") else "_" for ch in raw).strip(" .")
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
    return f"{(s or 'series')[:70]}_{h}"


def series_root() -> Path:
    p = app_data_dir() / "series"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_series_memories() -> list[str]:
    out = []
    for d in sorted(series_root().iterdir()):
        mj = d / "memory.json"
        if mj.is_file():
            try:
                out.append(json.loads(mj.read_text(encoding="utf-8")).get("name") or d.name)
            except Exception:
                out.append(d.name)
    return out


@dataclass
class CastMember:
    name: str = ""
    gender: str = "F"
    engine: str = "auto"
    voice: str = ""
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    ref_wav: str = ""            # copied into the series folder
    ref_text: str = ""
    embedding: list[float] = field(default_factory=list)
    embed_method: str = ""       # "ecapa" | "mfcc"
    episodes: int = 0
    updated: float = 0.0


def _from(cls, d):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


class SeriesMemory:
    def __init__(self, name: str):
        self.name = (name or "").strip()
        self.glossary: dict[str, list[dict]] = {}   # lang -> [{source,target,note}]
        self.cast: list[CastMember] = []
        self.story = ""

    # -- storage ------------------------------------------------------------
    @property
    def folder(self) -> Path:
        p = series_root() / _safe(self.name)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def load(cls, name: str) -> "SeriesMemory":
        m = cls(name)
        path = series_root() / _safe(name) / "memory.json"
        if path.is_file():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                m.glossary = {str(k): list(v) for k, v in (d.get("glossary") or {}).items()}
                m.cast = [_from(CastMember, c) for c in d.get("cast") or []]
                m.story = str(d.get("story") or "")
            except Exception as exc:
                log.warning("Series memory %s unreadable (%s); starting fresh",
                            path, exc)
        return m

    def save(self) -> None:
        with _LOCK:
            path = self.folder / "memory.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "name": self.name, "updated": time.time(),
                "glossary": self.glossary,
                "cast": [asdict(c) for c in self.cast],
                "story": self.story,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)

    # -- glossary -----------------------------------------------------------
    def glossary_for(self, lang: str) -> list[dict]:
        return [dict(t) for t in self.glossary.get(lang, [])]

    def merge_glossary(self, lang: str, terms: list[dict],
                       overwrite: bool = True) -> int:
        """Add/replace terms (keyed case-insensitively by source)."""
        cur = {t["source"].strip().lower(): t for t in self.glossary.get(lang, [])
               if t.get("source")}
        changed = 0
        for t in terms:
            src = str(t.get("source") or "").strip()
            tgt = str(t.get("target") or "").strip()
            if not src:
                continue
            key = src.lower()
            new = {"source": src, "target": tgt, "note": str(t.get("note") or "").strip()}
            if key not in cur or (overwrite and tgt and cur[key] != new):
                cur[key] = new
                changed += 1
        self.glossary[lang] = sorted(cur.values(), key=lambda t: t["source"].lower())
        return changed

    # -- cast ---------------------------------------------------------------
    def member(self, name: str) -> CastMember | None:
        key = (name or "").strip().lower()
        return next((c for c in self.cast if c.name.strip().lower() == key), None)

    def remember(self, *, name: str, gender: str, engine: str, voice: str,
                 rate: str, pitch: str, volume: str, ref_wav: str,
                 ref_text: str, embedding: list[float] | None,
                 embed_method: str) -> CastMember:
        """Create or update a cast member; copies the voice sample into the
        series folder so it outlives the episode project."""
        name = name.strip()
        m = self.member(name)
        if m is None:
            m = CastMember(name=name)
            self.cast.append(m)
            m.episodes = 0
        m.gender, m.engine, m.voice = gender, engine or "auto", voice
        m.rate, m.pitch, m.volume = rate, pitch, volume
        if ref_wav and os.path.isfile(ref_wav):
            dest = self.folder / "voices" / f"{_safe(name)}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if os.path.abspath(ref_wav) != os.path.abspath(dest):
                shutil.copyfile(ref_wav, dest)
            m.ref_wav, m.ref_text = str(dest), ref_text
        if embedding:
            if m.embedding and m.embed_method == embed_method \
                    and len(m.embedding) == len(embedding):
                # running mean over episodes keeps the fingerprint stable
                k = max(1, m.episodes)
                m.embedding = list((np.asarray(m.embedding) * k
                                    + np.asarray(embedding)) / (k + 1))
            else:
                m.embedding, m.embed_method = list(map(float, embedding)), embed_method
        m.episodes += 1
        m.updated = time.time()
        return m


def match_cast(char_vecs: list[list[float] | None], method: str,
               cast: list[CastMember]) -> list[tuple[CastMember, float] | None]:
    """Greedy one-to-one match of this episode's characters to known cast
    members by voice fingerprint. Returns, per character, (member, score) or
    None. Thresholds: ECAPA cosine ≥ 0.55; MFCC stats (normalised jointly)
    ≥ 0.6."""
    known = [c for c in cast if c.embedding and c.embed_method == method]
    out: list[tuple[CastMember, float] | None] = [None] * len(char_vecs)
    idx = [i for i, v in enumerate(char_vecs) if v]
    if not known or not idx:
        return out
    dims = {len(char_vecs[i]) for i in idx} | {len(c.embedding) for c in known}
    if len(dims) != 1:
        return out
    A = np.asarray([char_vecs[i] for i in idx], dtype=np.float64)
    B = np.asarray([c.embedding for c in known], dtype=np.float64)
    if method == "mfcc":
        X = np.vstack([A, B])
        X = np.where(np.isfinite(X), X, 0.0)
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
        A, B = X[:len(A)], X[len(A):]
        thr = 0.6
    else:
        thr = 0.55
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    S = A @ B.T
    used_a, used_b = set(), set()
    for flat in np.argsort(-S, axis=None):
        a, b = divmod(int(flat), S.shape[1])
        if a in used_a or b in used_b or S[a, b] < thr:
            continue
        used_a.add(a)
        used_b.add(b)
        out[idx[a]] = (known[b], float(S[a, b]))
    return out


if __name__ == "__main__":
    import tempfile
    from unittest import mock

    tmp = Path(tempfile.mkdtemp())
    with mock.patch(f"{__name__}.app_data_dir", lambda: tmp):
        m = SeriesMemory.load("My Drama: S1")
        assert m.merge_glossary("km", [{"source": "Dara", "target": "ដារ៉ា"}]) == 1
        assert m.merge_glossary("km", [{"source": "dara", "target": "ដារ៉ា"}]) == 1  # case change
        assert m.merge_glossary("km", [{"source": "Dara", "target": "ដារ៉ា"}],
                                overwrite=False) == 0
        ref = tmp / "r.wav"
        ref.write_bytes(b"RIFF")
        rng = np.random.default_rng(0)
        e1, e2 = rng.normal(size=41), rng.normal(size=41)
        m.remember(name="Dara", gender="M", engine="edge+clone", voice="", rate="+0%",
                   pitch="+0Hz", volume="+0%", ref_wav=str(ref), ref_text="hi",
                   embedding=list(e1), embed_method="mfcc")
        m.remember(name="Lin", gender="F", engine="auto", voice="", rate="+0%",
                   pitch="+0Hz", volume="+0%", ref_wav="", ref_text="",
                   embedding=list(e2), embed_method="mfcc")
        m.save()
        m2 = SeriesMemory.load("My Drama: S1")
        assert len(m2.cast) == 2 and m2.glossary_for("km")[0]["target"] == "ដារ៉ា"
        assert os.path.isfile(m2.member("dara").ref_wav), "voice sample copied"
        # new episode: characters in swapped order with a little noise
        others = [list(rng.normal(size=41)) for _ in range(3)]
        res = match_cast([list(e2 + 0.05 * rng.normal(size=41)),
                          list(e1 + 0.05 * rng.normal(size=41))] , "mfcc",
                         m2.cast + [CastMember(name=f"x{i}", embedding=o,
                                               embed_method="mfcc")
                                    for i, o in enumerate(others)])
        assert res[0] and res[0][0].name == "Lin", res
        assert res[1] and res[1][0].name == "Dara", res
        assert match_cast([list(e1)], "ecapa", m2.cast) == [None], "method must match"
        assert any(p.name.startswith("My Drama_ S1_") for p in series_root().iterdir())
        assert _safe("អ្នកម្ដាយ") != _safe("អ្នកម្ដោយ") and "្" in _safe("អ្នកម្ដាយ")
        assert list_series_memories() == ["My Drama: S1"]
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")
