"""Translation memory: reuse earlier translations instead of paying the AI
again, and learn from the user's corrections.

SQLite at ``app_data_dir()/translation_memory.db``. An entry is keyed by
(target language, normalised source line, speaker gender, series). Lookups
prefer, in order: a human correction in this series, a human correction in
any series, an AI translation in this series, an AI translation anywhere.
Gender matters (Khmer បាទ/ចាស, pronouns), so it is part of the key; "" means
unknown and only matches "".

No Qt imports.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path

from app.core.logger import get_logger
from app.core.paths import app_data_dir

log = get_logger("translation_memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tm (
    tgt_lang  TEXT NOT NULL,
    source    TEXT NOT NULL,          -- normalised source line
    gender    TEXT NOT NULL DEFAULT '',
    series    TEXT NOT NULL DEFAULT '',
    target    TEXT NOT NULL,
    origin    TEXT NOT NULL DEFAULT 'ai',   -- 'ai' | 'human'
    src_lang  TEXT NOT NULL DEFAULT '',
    uses      INTEGER NOT NULL DEFAULT 0,
    updated   REAL NOT NULL,
    PRIMARY KEY (tgt_lang, source, gender, series, origin)
);
CREATE INDEX IF NOT EXISTS idx_tm_lookup ON tm (tgt_lang, source, gender);
"""


_LIST_NUMBER = re.compile(r"^\s*\d{1,4}\s*[.)]\s+")


def clean_target(source: str, target: str) -> str:
    """A translation without a leaked list number ("2. ចាស") — older dub
    runs stored some AI replies with their batch number. Kept when the source
    line itself starts with a number."""
    target = str(target or "").strip()
    if re.match(r"\s*[\d០-៩]", str(source or "")):
        return target
    return _LIST_NUMBER.sub("", target)


def normalize(text: str) -> str:
    """Case/space/punctuation-insensitive key ("Where are you going?!" ==
    "where are you going?")."""
    t = unicodedata.normalize("NFC", str(text or "")).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\s.!?。！？…,，;:~\-–—\"'“”‘’]+$", "", t)
    return t


class TranslationMemory:
    def __init__(self, db_path: Path | str | None = None):
        self._path = Path(db_path) if db_path else app_data_dir() / "translation_memory.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(tm)")}
            if "rules" not in cols:
                # the translation rules an AI entry was made with (0: before
                # this column existed); a human correction has no rules
                self._conn.execute("ALTER TABLE tm ADD COLUMN rules INTEGER NOT NULL DEFAULT 0")
            self._conn.commit()

    def lookup(self, tgt_lang: str, source: str, gender: str = "",
               series: str = "", rules: int = 0) -> tuple[str, str] | None:
        """(translation, origin) or None. A human correction is always
        returned; an AI translation only when it was made with the same
        `rules` version — an older one would carry the mistakes those rules
        were changed to fix ("bro" as a woman), run after run."""
        key = normalize(source)
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT target, origin, series FROM tm WHERE tgt_lang=? AND "
                "source=? AND gender=? AND (origin='human' OR rules=?) "
                "ORDER BY (origin='human') DESC, (series=?) DESC, updated DESC LIMIT 1",
                (tgt_lang, key, gender or "", int(rules), series or "")).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE tm SET uses = uses + 1 WHERE tgt_lang=? AND source=? "
                "AND gender=? AND series=? AND origin=?",
                (tgt_lang, key, gender or "", row["series"], row["origin"]))
            self._conn.commit()
        target = clean_target(source, row["target"])
        return (target, row["origin"]) if target else None

    def add(self, tgt_lang: str, source: str, target: str, *, gender: str = "",
            series: str = "", origin: str = "ai", src_lang: str = "",
            rules: int = 0) -> None:
        self.add_many(tgt_lang, [(source, target, gender)], series=series,
                      origin=origin, src_lang=src_lang, rules=rules)

    def add_many(self, tgt_lang: str, items: list[tuple[str, str, str]], *,
                 series: str = "", origin: str = "ai", src_lang: str = "",
                 rules: int = 0) -> int:
        """items: (source, target, gender). Empty sources/targets are skipped.
        `rules`: the translation rules version an AI entry was made with."""
        rows = []
        now = time.time()
        for src, tgt, gender in items:
            key, tgt = normalize(src), clean_target(src, tgt)
            if key and tgt:
                rows.append((tgt_lang, key, gender or "", series or "", tgt,
                             origin, src_lang or "", int(rules), now))
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT INTO tm (tgt_lang, source, gender, series, target, origin, "
                "src_lang, rules, updated) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT "
                "(tgt_lang, source, gender, series, origin) DO UPDATE SET "
                "target=excluded.target, rules=excluded.rules, updated=excluded.updated",
                rows)
            self._conn.commit()
        return len(rows)

    def stats(self) -> dict:
        with self._lock:
            r = self._conn.execute(
                "SELECT COUNT(*) AS n, SUM(origin='human') AS human, "
                "SUM(uses) AS reused FROM tm").fetchone()
        return {"entries": int(r["n"] or 0), "human": int(r["human"] or 0),
                "reused": int(r["reused"] or 0)}

    def clear(self, series: str | None = None) -> None:
        with self._lock:
            if series is None:
                self._conn.execute("DELETE FROM tm")
            else:
                self._conn.execute("DELETE FROM tm WHERE series=?", (series,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_DEFAULT: TranslationMemory | None = None
_DEFAULT_LOCK = threading.Lock()


def get_tm() -> TranslationMemory:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = TranslationMemory()
        return _DEFAULT


if __name__ == "__main__":
    import os
    import tempfile
    d = tempfile.mkdtemp()
    tm = TranslationMemory(os.path.join(d, "tm.db"))
    assert normalize("  Where are you   GOING?! ") == "where are you going"
    tm.add("km", "Where are you going?", "ទៅណា?", gender="M", series="A")
    assert tm.lookup("km", "where are you going", "M") == ("ទៅណា?", "ai")
    assert tm.lookup("km", "where are you going", "F") is None, "gender is part of key"
    tm.add("km", "Where are you going?", "បងទៅណា?", gender="M", series="B",
           origin="human")
    assert tm.lookup("km", "Where are you going?", "M", series="A") == ("បងទៅណា?", "human"), \
        "a human fix beats AI even from another series"
    tm.add("km", "Where are you going?", "ទៅណាហ្នឹង?", gender="M", series="A",
           origin="human")
    assert tm.lookup("km", "Where are you going?", "M", series="A")[0] == "ទៅណាហ្នឹង?", \
        "same-series human fix wins"
    assert tm.add_many("km", [("", "x", ""), ("a", "", ""), ("ok", "អូខេ", "")]) == 1
    s = tm.stats()
    assert s["entries"] == 4 and s["human"] == 2 and s["reused"] >= 3, s
    tm.clear(series="B")
    assert tm.stats()["entries"] == 3
    tm._conn.execute("INSERT INTO tm (tgt_lang, source, gender, series, target, updated) "
                     "VALUES ('km', 'watch out', '', '', '2. ប្រយ័ត្ន!', 0)")
    assert tm.lookup("km", "Watch out!")[0] == "ប្រយ័ត្ន!", "old leaked list number removed"
    tm.add("km", "3 2 1", "3. 2. 1.")
    assert tm.lookup("km", "3 2 1")[0] == "3. 2. 1.", "a countdown keeps its numbers"
    tm.add("km", "Run!", "1. រត់!")
    assert tm.lookup("km", "run")[0] == "រត់!", "stored without the number"
    # AI entries are reused only under the rules they were made with; a
    # human correction always
    tm.add("km", "Hey, bro.", "បងស្រី!", gender="F", rules=2)
    assert tm.lookup("km", "Hey, bro.", "F", rules=2) == ("បងស្រី!", "ai")
    assert tm.lookup("km", "Hey, bro.", "F", rules=3) is None, "older rules: not reused"
    tm.add("km", "Hey, bro.", "លោកបង!", gender="F", rules=3)
    assert tm.lookup("km", "Hey, bro.", "F", rules=3) == ("លោកបង!", "ai")
    tm.add("km", "Hey, sis.", "បង!", gender="M", origin="human")
    assert tm.lookup("km", "Hey, sis.", "M", rules=9) == ("បង!", "human"), "human: any rules"
    assert tm.lookup("km", "run", rules=3) is None and tm.lookup("km", "run")[0] == "រត់!", \
        "entries from before the column count as rules 0"
    tm.close()
    print("PASS")
