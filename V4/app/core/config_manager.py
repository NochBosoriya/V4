"""SQLite-backed configuration: settings, encrypted API key pools, routes, usage.

Keys are encrypted at rest with Windows DPAPI (ctypes, no pywin32). If DPAPI is
unavailable (non-Windows or failure), values are base64-obfuscated and a warning
is logged. Thread-safe via an internal RLock; SQLite runs in WAL mode.
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import AppError, ProviderError
from app.core.logger import get_logger
from app.core.paths import app_data_dir

log = get_logger("config")

PROVIDERS = ["deepseek", "gemini", "openai", "anthropic", "openrouter", "custom"]
FEATURES = ["translate_draft", "translate_polish", "recap", "poster"]

DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "",
}

# Single API keys of non-chat services, stored encrypted with
# ConfigManager.set_secret() (exported only when keys are included).
SECRET_SETTINGS = ("elevenlabs.api_key", "deepgram.api_key")

DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    # an alias Google keeps pointing at a current Flash model (2.0-flash was retired)
    "gemini": "gemini-flash-latest",
    "openai": "gpt-4o-mini",
    # claude-sonnet-4-20250514 (the old default) is deprecated
    "anthropic": "claude-opus-5-5",
    # OpenRouter ids are "<vendor>/<model>"; Fetch models lists them all
    "openrouter": "google/gemini-2.5-flash",
}

# $ per 1M tokens: (input, output) — the one price table (usage, budget,
# ai_engine.estimate_cost). Editable via ConfigManager.set("prices", {...}).
# A key also prices longer ids that start with it (dated snapshots,
# "-latest" variants), the longest matching key winning; see price_for().
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "default": (0.5, 1.5),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    # Gemini (the -latest aliases are the app's defaults)
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
    # picture models: a picture is about 1290 output tokens (~US$0.04 on
    # 2.5 Flash Image); approximate
    "gemini-2.5-flash-image": (0.30, 30.00),
    "gemini-3.1-flash-image": (0.50, 60.00),
    "gemini-3-pro-image": (2.00, 120.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    # Anthropic
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}


def price_for(model: str, prices: dict[str, tuple[float, float]] | None = None
              ) -> tuple[float, float]:
    """($ in, $ out) per 1M tokens for a model id: an exact key, else the
    longest key the id starts with ("claude-sonnet-4-20250514" ->
    "claude-sonnet-4"), else "default". Never raises on unknown models."""
    prices = prices or DEFAULT_PRICES
    full = (model or "").strip().lower()
    if full in prices:                       # e.g. OpenRouter's own price list
        return prices[full]
    m = full.split("/")[-1]                  # "models/x", "openai/gpt-4o"
    if m in prices:
        return prices[m]
    best = ""
    for k in prices:
        if k != "default" and m.startswith(k) and len(k) > len(best):
            best = k
    return prices[best] if best else prices.get("default", (0.5, 1.5))

_KEY_STATUSES = ("unknown", "valid", "invalid", "exhausted", "cooldown")

# 429 cooldown: 60s doubling per repeat, capped at 15 minutes.
_RATE_LIMIT_BASE_S = 60.0
_RATE_LIMIT_CAP_S = 15 * 60.0
# 5xx / network hiccup cooldown.
_SERVER_ERR_COOLDOWN_S = 15.0

# Exhausted keys are retried after this long (free-tier / daily quotas reset).
_EXHAUSTED_RETRY_S = 3600
# A key whose account has no credit left is tried again after this long: a
# refused request costs nothing, and the user may just have topped it up
# (waiting does not bring credit back, so no long pause).
_NO_CREDIT_RETRY_S = 60

_EXHAUSTED_MARKERS = (
    "insufficient_quota", "insufficient quota", "quota exceeded",
    "quota_exceeded", "exceeded your current quota", "balance is insufficient",
    "insufficient balance", "out of credit", "credit balance",
    "exhausted", "billing",
)
# words of a "no money left" answer, as opposed to a quota that resets
# (OpenRouter 402 "requires more credits … can only afford 5738",
# DeepSeek 402 "Insufficient Balance", Anthropic "credit balance is too low")
_CREDIT_MARKERS = ("http 402", "credit", "balance", "afford", "billing",
                   "payment", "top up", "insufficient funds")

PROVIDER_NAMES = {"deepseek": "DeepSeek", "gemini": "Gemini", "openai": "OpenAI",
                  "anthropic": "Anthropic", "openrouter": "OpenRouter"}
# where an account gets more credit (site names: English in both languages)
_TOP_UP = {
    "openrouter": "openrouter.ai → Credits",
    "deepseek": "platform.deepseek.com → Top up",
    "openai": "platform.openai.com → Billing",
    "anthropic": "the Anthropic Console → Billing",
    "gemini": "Google AI Studio → Billing",
}


def is_credit_error(status_code: int | None, message: str = "") -> bool:
    """The account behind a key has no money left (HTTP 402, "insufficient
    credits", "balance"…), as opposed to a free-tier or daily quota that
    resets by itself."""
    if status_code == 402:
        return True
    low = (message or "").lower()
    return any(m in low for m in _CREDIT_MARKERS)


def key_out_of_credit(info: "KeyInfo") -> bool:
    """`info` is marked exhausted because its account is out of credit."""
    return info.status == "exhausted" and is_credit_error(None, info.last_error)


def no_credit_message(provider: str, n_keys: int = 1) -> tuple[str, str]:
    """(English, Khmer): every key of `provider` is out of credit, and what
    to do. Several keys of one account share its balance, so another key of
    the same account does not help."""
    name = PROVIDER_NAMES.get(provider, provider)
    where_en = _TOP_UP.get(provider) or "the provider's website"
    where_km = _TOP_UP.get(provider) or "គេហទំព័ររបស់អ្នកផ្ដល់សេវា"
    if n_keys > 1:
        en = (f"Your {name} keys are out of credit "
              "(keys of one account share one balance).")
        km = (f"key {name} របស់អ្នកអស់ទឹកប្រាក់ហើយ "
              "(key នៃគណនីតែមួយប្រើសមតុល្យរួមគ្នា)។")
    else:
        en = f"Your {name} account is out of credit."
        km = f"គណនី {name} របស់អ្នកអស់ទឹកប្រាក់ហើយ។"
    en += (f" Add credit at {where_en}, or add a key of another AI provider in "
           "Settings → AI Providers, then try again.")
    km += (f" សូមបញ្ចូលទឹកប្រាក់នៅ {where_km} ឬបន្ថែម key របស់អ្នកផ្ដល់ AI ផ្សេងនៅ "
           "ការកំណត់ → អ្នកផ្ដល់ AI រួចសាកម្ដងទៀត។")
    return en, km


# ---------------------------------------------------------------------------
# DPAPI encryption (ctypes) with base64 fallback
# ---------------------------------------------------------------------------

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", ctypes.wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_char))]


_DPAPI_PREFIX = "dpapi:"
_B64_PREFIX = "b64:"
_dpapi_warned = False


def _blob_to_bytes(blob: _DATA_BLOB) -> bytes:
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        try:
            ctypes.windll.kernel32.LocalFree(blob.pbData)
        except Exception:
            pass


def _dpapi_protect(data: bytes) -> bytes | None:
    if os.name != "nt":
        return None
    try:
        crypt32 = ctypes.windll.crypt32
        in_blob = _DATA_BLOB(len(data), ctypes.cast(
            ctypes.create_string_buffer(data, len(data)),
            ctypes.POINTER(ctypes.c_char)))
        out_blob = _DATA_BLOB()
        # CRYPTPROTECT_UI_FORBIDDEN = 0x01
        if not crypt32.CryptProtectData(ctypes.byref(in_blob), None, None,
                                        None, None, 0x01,
                                        ctypes.byref(out_blob)):
            return None
        return _blob_to_bytes(out_blob)
    except Exception:
        return None


def _dpapi_unprotect(data: bytes) -> bytes | None:
    if os.name != "nt":
        return None
    try:
        crypt32 = ctypes.windll.crypt32
        in_blob = _DATA_BLOB(len(data), ctypes.cast(
            ctypes.create_string_buffer(data, len(data)),
            ctypes.POINTER(ctypes.c_char)))
        out_blob = _DATA_BLOB()
        if not crypt32.CryptUnprotectData(ctypes.byref(in_blob), None, None,
                                          None, None, 0x01,
                                          ctypes.byref(out_blob)):
            return None
        return _blob_to_bytes(out_blob)
    except Exception:
        return None


def encrypt_secret(plain: str) -> str:
    """Encrypt a secret for storage. DPAPI when available, base64 otherwise."""
    global _dpapi_warned
    raw = plain.encode("utf-8")
    enc = _dpapi_protect(raw)
    if enc is not None:
        return _DPAPI_PREFIX + base64.b64encode(enc).decode("ascii")
    if not _dpapi_warned:
        log.warning("DPAPI unavailable; storing keys base64-obfuscated only.")
        _dpapi_warned = True
    return _B64_PREFIX + base64.b64encode(raw).decode("ascii")


def decrypt_secret(stored: str) -> str:
    """Decrypt a stored secret. Tolerates both DPAPI and base64 forms."""
    try:
        if stored.startswith(_DPAPI_PREFIX):
            enc = base64.b64decode(stored[len(_DPAPI_PREFIX):])
            dec = _dpapi_unprotect(enc)
            if dec is not None:
                return dec.decode("utf-8", errors="replace")
            log.warning("DPAPI decryption failed for a stored key.")
            return ""
        if stored.startswith(_B64_PREFIX):
            return base64.b64decode(
                stored[len(_B64_PREFIX):]).decode("utf-8", errors="replace")
        # Legacy/plaintext value: return as-is.
        return stored
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KeyInfo:
    key: str
    provider: str
    status: str = "unknown"  # "unknown"|"valid"|"invalid"|"exhausted"|"cooldown"
    latency_ms: int | None = None
    last_error: str = ""
    cooldown_until: float = 0.0  # epoch seconds


@dataclass
class Route:
    provider: str
    model: str
    fallback_provider: str | None = None
    fallback_model: str | None = None


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------

class ConfigManager:
    """Thread-safe SQLite-backed configuration store (WAL mode)."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = Path(db_path) if db_path else app_data_dir() / "config.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._key_pool = KeyPool(self)

    # -- schema ------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    provider       TEXT NOT NULL,
                    key_enc        TEXT NOT NULL,
                    key_hash       TEXT NOT NULL,
                    status         TEXT NOT NULL DEFAULT 'unknown',
                    latency_ms     INTEGER,
                    last_error     TEXT NOT NULL DEFAULT '',
                    cooldown_until REAL NOT NULL DEFAULT 0,
                    fail_streak    INTEGER NOT NULL DEFAULT 0,
                    added_at       REAL NOT NULL,
                    PRIMARY KEY (provider, key_hash)
                );
                CREATE TABLE IF NOT EXISTS routes (
                    feature           TEXT PRIMARY KEY,
                    provider          TEXT NOT NULL,
                    model             TEXT NOT NULL,
                    fallback_provider TEXT,
                    fallback_model    TEXT
                );
                CREATE TABLE IF NOT EXISTS usage (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         REAL NOT NULL,
                    provider   TEXT NOT NULL,
                    model      TEXT NOT NULL,
                    in_tokens  INTEGER NOT NULL DEFAULT 0,
                    out_tokens INTEGER NOT NULL DEFAULT 0
                );
            """)
            self._conn.commit()

    @staticmethod
    def _hash_key(key: str) -> str:
        import hashlib
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # -- generic settings ----------------------------------------------------

    def get(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default

    def set(self, key: str, value) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, payload))
            self._conn.commit()

    def get_secret(self, key: str) -> str:
        """A single API key stored with set_secret() (older plain-text values
        are read as they are)."""
        stored = self.get(key) or ""
        return decrypt_secret(str(stored)) if stored else ""

    def set_secret(self, key: str, value: str) -> None:
        value = (value or "").strip()
        self.set(key, encrypt_secret(value) if value else "")

    # -- provider helpers ----------------------------------------------------

    def get_base_url(self, provider: str) -> str:
        override = self.get(f"provider.{provider}.base_url")
        if override:
            return str(override)
        return DEFAULT_BASE_URLS.get(provider, "")

    # -- key management -------------------------------------------------------

    def add_keys(self, provider: str, keys: list[str]) -> int:
        """Add keys (deduplicated). Returns the number actually added."""
        added = 0
        now = time.time()
        with self._lock:
            for raw in keys:
                key = (raw or "").strip()
                if not key:
                    continue
                h = self._hash_key(key)
                exists = self._conn.execute(
                    "SELECT 1 FROM api_keys WHERE provider=? AND key_hash=?",
                    (provider, h)).fetchone()
                if exists:
                    continue
                self._conn.execute(
                    "INSERT INTO api_keys(provider, key_enc, key_hash, status, "
                    "latency_ms, last_error, cooldown_until, fail_streak, added_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (provider, encrypt_secret(key), h, "unknown",
                     None, "", 0.0, 0, now))
                added += 1
            self._conn.commit()
        if added:
            log.info("Added %d key(s) for provider %s", added, provider)
        return added

    def remove_key(self, provider: str, key: str) -> None:
        h = self._hash_key(key.strip())
        with self._lock:
            self._conn.execute(
                "DELETE FROM api_keys WHERE provider=? AND key_hash=?",
                (provider, h))
            self._conn.commit()

    def get_keys(self, provider: str) -> list[KeyInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM api_keys WHERE provider=? ORDER BY added_at",
                (provider,)).fetchall()
        out: list[KeyInfo] = []
        now = time.time()
        for r in rows:
            plain = decrypt_secret(r["key_enc"])
            if not plain:
                continue
            status = r["status"]
            cooldown_until = float(r["cooldown_until"] or 0.0)
            # A cooldown that has elapsed presents as its underlying status.
            if status == "cooldown" and cooldown_until <= now:
                status = "unknown"
            out.append(KeyInfo(
                key=plain, provider=provider, status=status,
                latency_ms=r["latency_ms"], last_error=r["last_error"] or "",
                cooldown_until=cooldown_until))
        return out

    def import_keys_txt(self, provider: str, path: str) -> int:
        """Import keys from a text file, one key per line. Returns added count."""
        p = Path(path)
        if not p.exists():
            raise AppError(
                f"Key file not found: {path}",
                f"រកមិនឃើញឯកសារកូនសោ៖ {path}")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise AppError(
                f"Cannot read key file: {exc}",
                f"មិនអាចអានឯកសារកូនសោបានទេ៖ {exc}") from exc
        keys = [ln.strip() for ln in text.splitlines() if ln.strip()
                and not ln.strip().startswith("#")]
        return self.add_keys(provider, keys)

    def set_key_status(self, provider: str, key: str, status: str,
                       latency_ms: int | None = None, error: str = "") -> None:
        if status not in _KEY_STATUSES:
            status = "unknown"
        h = self._hash_key(key.strip())
        with self._lock:
            self._conn.execute(
                "UPDATE api_keys SET status=?, latency_ms=COALESCE(?, latency_ms), "
                "last_error=? WHERE provider=? AND key_hash=?",
                (status, latency_ms, error, provider, h))
            if status not in ("cooldown",):
                # Leaving cooldown state clears the timer; success clears streak.
                if status in ("valid", "unknown"):
                    self._conn.execute(
                        "UPDATE api_keys SET cooldown_until=0, fail_streak=0 "
                        "WHERE provider=? AND key_hash=?", (provider, h))
            self._conn.commit()

    def _set_key_cooldown(self, provider: str, key: str, until: float,
                          streak: int, error: str = "") -> None:
        h = self._hash_key(key.strip())
        with self._lock:
            self._conn.execute(
                "UPDATE api_keys SET status='cooldown', cooldown_until=?, "
                "fail_streak=?, last_error=? WHERE provider=? AND key_hash=?",
                (until, streak, error, provider, h))
            self._conn.commit()

    def _set_key_exhausted(self, provider: str, key: str, until: float,
                           error: str = "") -> None:
        h = self._hash_key(key.strip())
        with self._lock:
            self._conn.execute(
                "UPDATE api_keys SET status='exhausted', cooldown_until=?, "
                "last_error=? WHERE provider=? AND key_hash=?",
                (until, error, provider, h))
            self._conn.commit()

    def _get_key_row(self, provider: str, key: str) -> sqlite3.Row | None:
        h = self._hash_key(key.strip())
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM api_keys WHERE provider=? AND key_hash=?",
                (provider, h)).fetchone()

    # -- routes ---------------------------------------------------------------

    def _default_route(self) -> Route:
        """Pick the best default route: user-set global > first provider with keys > deepseek."""
        global_prov = self.get("ai.default_provider")
        if global_prov and global_prov in PROVIDERS:
            model = (self.get(f"provider.{global_prov}.model")
                     or DEFAULT_MODELS.get(global_prov, ""))
            if model:
                return Route(provider=global_prov, model=model)
        for p in PROVIDERS:
            if p == "custom":
                continue
            keys = self.get_keys(p)
            if keys:
                model = (self.get(f"provider.{p}.model")
                         or DEFAULT_MODELS.get(p, ""))
                if model:
                    return Route(provider=p, model=model)
        return Route(provider="deepseek", model="deepseek-chat")

    def provider_model(self, provider: str) -> str:
        """The model set for `provider` in Settings, else its built-in default."""
        return str(self.get(f"provider.{provider}.model")
                   or DEFAULT_MODELS.get(provider, "") or "")

    def get_route(self, feature: str) -> Route:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM routes WHERE feature=?", (feature,)).fetchone()
        if row is None:
            return self._default_route()
        return Route(provider=row["provider"], model=row["model"],
                     fallback_provider=row["fallback_provider"],
                     fallback_model=row["fallback_model"])

    def set_route(self, feature: str, route: Route) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO routes(feature, provider, model, fallback_provider, "
                "fallback_model) VALUES(?,?,?,?,?) "
                "ON CONFLICT(feature) DO UPDATE SET provider=excluded.provider, "
                "model=excluded.model, fallback_provider=excluded.fallback_provider, "
                "fallback_model=excluded.fallback_model",
                (feature, route.provider, route.model,
                 route.fallback_provider, route.fallback_model))
            self._conn.commit()

    def clear_routes(self) -> None:
        """Drop all per-feature overrides so every feature follows the default route."""
        with self._lock:
            self._conn.execute("DELETE FROM routes")
            self._conn.commit()

    def providers_with_keys(self) -> list[str]:
        return [p for p in PROVIDERS if self.get_keys(p)]

    # -- usage / cost ---------------------------------------------------------

    def get_prices(self) -> dict[str, tuple[float, float]]:
        stored = self.get("prices")
        prices: dict[str, tuple[float, float]] = dict(DEFAULT_PRICES)
        if isinstance(stored, dict):
            for k, v in stored.items():
                try:
                    prices[str(k)] = (float(v[0]), float(v[1]))
                except Exception:
                    continue
        return prices

    def record_usage(self, provider: str, model: str,
                     in_tokens: int, out_tokens: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage(ts, provider, model, in_tokens, out_tokens) "
                "VALUES(?,?,?,?,?)",
                (time.time(), provider, model,
                 int(in_tokens or 0), int(out_tokens or 0)))
            self._conn.commit()

    def get_usage(self) -> list[dict]:
        """Aggregated usage rows: provider, model, in_tokens, out_tokens, est_cost."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT provider, model, SUM(in_tokens) AS in_tokens, "
                "SUM(out_tokens) AS out_tokens FROM usage "
                "GROUP BY provider, model ORDER BY provider, model").fetchall()
        prices = self.get_prices()
        out: list[dict] = []
        for r in rows:
            in_t = int(r["in_tokens"] or 0)
            out_t = int(r["out_tokens"] or 0)
            price = price_for(r["model"], prices)
            est = (in_t * price[0] + out_t * price[1]) / 1_000_000.0
            out.append({
                "provider": r["provider"], "model": r["model"],
                "in_tokens": in_t, "out_tokens": out_t,
                "est_cost": round(est, 6),
            })
        return out

    # -- budget ---------------------------------------------------------------

    def month_cost(self, now: float | None = None) -> float:
        """Estimated USD spent on AI calls since the 1st of this month."""
        lt = time.localtime(now or time.time())
        start = time.mktime((lt.tm_year, lt.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, SUM(in_tokens) AS i, SUM(out_tokens) AS o FROM usage "
                "WHERE ts >= ? GROUP BY model", (start,)).fetchall()
        prices = self.get_prices()
        total = 0.0
        for r in rows:
            p = price_for(r["model"], prices)
            total += (int(r["i"] or 0) * p[0] + int(r["o"] or 0) * p[1]) / 1_000_000.0
        return round(total, 6)

    def get_budget(self) -> tuple[float, float]:
        """(monthly limit in USD, warn fraction). Limit 0 = unlimited."""
        try:
            limit = max(0.0, float(self.get("budget.monthly_usd", 0.0) or 0.0))
        except (TypeError, ValueError):
            limit = 0.0
        try:
            warn = min(1.0, max(0.1, float(self.get("budget.warn_fraction", 0.8) or 0.8)))
        except (TypeError, ValueError):
            warn = 0.8
        return limit, warn

    # -- export / import ------------------------------------------------------

    def export_settings(self, path: str, include_keys: bool = False) -> None:
        """Export settings (and optionally keys, plaintext by design) to JSON."""
        with self._lock:
            setting_rows = self._conn.execute(
                "SELECT key, value FROM settings").fetchall()
            route_rows = self._conn.execute("SELECT * FROM routes").fetchall()
        settings: dict = {}
        for r in setting_rows:
            if r["key"] in SECRET_SETTINGS:
                # plaintext like the key pools below, and only on request
                value = self.get_secret(r["key"]) if include_keys else ""
                if value:
                    settings[r["key"]] = value
                continue
            try:
                settings[r["key"]] = json.loads(r["value"])
            except Exception:
                settings[r["key"]] = r["value"]
        routes = {
            r["feature"]: {
                "provider": r["provider"], "model": r["model"],
                "fallback_provider": r["fallback_provider"],
                "fallback_model": r["fallback_model"],
            } for r in route_rows
        }
        payload: dict = {
            "app": "RecapStudio",
            "version": 1,
            "settings": settings,
            "routes": routes,
        }
        if include_keys:
            # Intentionally plaintext inside the export file: exports are
            # portable across machines/users where DPAPI blobs cannot travel.
            keys: dict[str, list[str]] = {}
            for provider in PROVIDERS:
                infos = self.get_keys(provider)
                if infos:
                    keys[provider] = [ki.key for ki in infos]
            payload["keys"] = keys
        try:
            Path(path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError as exc:
            raise AppError(
                f"Failed to write settings export: {exc}",
                f"បរាជ័យក្នុងការសរសេរឯកសារនាំចេញការកំណត់៖ {exc}") from exc
        log.info("Settings exported to %s (include_keys=%s)", path, include_keys)

    def import_settings(self, path: str) -> None:
        """Import a JSON export. Keys (if present) are re-encrypted locally."""
        p = Path(path)
        if not p.exists():
            raise AppError(
                f"Settings file not found: {path}",
                f"រកមិនឃើញឯកសារការកំណត់៖ {path}")
        try:
            payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AppError(
                f"Invalid settings file: {exc}",
                f"ឯកសារការកំណត់មិនត្រឹមត្រូវ៖ {exc}") from exc
        if not isinstance(payload, dict):
            raise AppError(
                "Invalid settings file: expected a JSON object.",
                "ឯកសារការកំណត់មិនត្រឹមត្រូវ៖ ត្រូវការទិន្នន័យ JSON ជាប្រភេទ object។")

        settings = payload.get("settings")
        if isinstance(settings, dict):
            for k, v in settings.items():
                if str(k) in SECRET_SETTINGS:
                    self.set_secret(str(k), str(v or ""))   # re-encrypt locally
                else:
                    self.set(str(k), v)

        routes = payload.get("routes")
        if isinstance(routes, dict):
            for feature, r in routes.items():
                if not isinstance(r, dict):
                    continue
                try:
                    self.set_route(str(feature), Route(
                        provider=str(r.get("provider", "deepseek")),
                        model=str(r.get("model", "deepseek-chat")),
                        fallback_provider=r.get("fallback_provider"),
                        fallback_model=r.get("fallback_model")))
                except Exception:
                    continue

        keys = payload.get("keys")
        if isinstance(keys, dict):
            for provider, key_list in keys.items():
                if isinstance(key_list, list):
                    self.add_keys(str(provider),
                                  [str(k) for k in key_list if k])
        log.info("Settings imported from %s", path)

    # -- key pool -------------------------------------------------------------

    @property
    def key_pool(self) -> "KeyPool":
        return self._key_pool

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# KeyPool
# ---------------------------------------------------------------------------

class KeyPool:
    """Round-robin key rotation per provider, skipping unusable keys.

    Failure semantics:
      - 401/403                       -> status "invalid" (permanent)
      - 429                           -> cooldown 60s, doubling per repeated
                                         429 on the same key, capped at 15 min
      - quota-exhausted message       -> status "exhausted", retried after
                                         _EXHAUSTED_RETRY_S (quotas reset)
      - 402 / out of credit           -> status "exhausted", retried after
                                         _NO_CREDIT_RETRY_S (the user may
                                         top up; a refusal costs nothing)
      - 5xx / network (None)          -> cooldown 15s
    A ProviderError for "all keys cooling down" carries .retry_after (s).
    """

    def __init__(self, config: ConfigManager):
        self._config = config
        self._lock = threading.Lock()
        self._rr_index: dict[str, int] = {}

    def _usable_keys(self, provider: str) -> list[KeyInfo]:
        now = time.time()
        usable = []
        for ki in self._config.get_keys(provider):
            if ki.status == "invalid":
                continue
            if ki.status == "exhausted" and ki.cooldown_until > now:
                continue
            if ki.status == "cooldown" and ki.cooldown_until > now:
                continue
            usable.append(ki)
        return usable

    def has_usable(self, provider: str) -> bool:
        """A key of `provider` can be used right now (not cooling down,
        out of credit or invalid)."""
        try:
            return bool(self._usable_keys(provider))
        except Exception:
            return False

    def acquire(self, provider: str) -> str:
        with self._lock:
            usable = self._usable_keys(provider)
            if not usable:
                all_keys = self._config.get_keys(provider)
                if not all_keys:
                    raise ProviderError(
                        f"No API keys configured for provider '{provider}'. "
                        f"Add keys in Settings.",
                        f"មិនទាន់មានកូនសោ API សម្រាប់ក្រុមហ៊ុនផ្តល់សេវា "
                        f"'{provider}' ទេ។ សូមបញ្ចូលកូនសោនៅក្នុងផ្នែកការកំណត់។",
                        provider=provider, status_code=None, retryable=False)
                now = time.time()
                cooldowns = [k for k in all_keys
                             if k.status == "cooldown" and k.cooldown_until > now]
                if cooldowns:
                    wait_s = int(min(k.cooldown_until for k in cooldowns) - now) + 1
                    exc = ProviderError(
                        f"All {len(all_keys)} key(s) for '{provider}' are "
                        f"unavailable (rate-limited or failed). Next key usable "
                        f"in ~{wait_s}s.",
                        f"កូនសោទាំង {len(all_keys)} របស់ '{provider}' "
                        f"មិនអាចប្រើបានទេ (ត្រូវបានកំណត់ល្បឿន ឬបរាជ័យ)។ "
                        f"កូនសោបន្ទាប់អាចប្រើបានក្នុងរយៈពេលប្រហែល {wait_s} វិនាទី។",
                        provider=provider, status_code=429, retryable=True)
                    exc.retry_after = wait_s
                    raise exc
                exhausted = [k for k in all_keys if k.status == "exhausted"]
                if exhausted and all(key_out_of_credit(k) for k in exhausted):
                    # no money left: waiting does not help, credit or
                    # another provider's key does — said in plain words
                    en, km = no_credit_message(provider, len(exhausted))
                    exc = ProviderError(en, km, provider=provider, status_code=402,
                                        retryable=False)
                    exc.plain = True        # already says what to do: shown as it is
                    exc.retry_after = int(min(k.cooldown_until for k in exhausted) - now) + 1
                    raise exc
                if exhausted:
                    wait_s = int(min(k.cooldown_until for k in exhausted) - now) + 1
                    exc = ProviderError(
                        f"All keys for '{provider}' are out of quota; they are tried "
                        f"again in ~{max(1, wait_s // 60)} min. Add keys from another "
                        f"account/project, or use another provider.",
                        f"កូនសោទាំងអស់របស់ '{provider}' អស់កូតា។ នឹងសាកម្តងទៀតក្នុង "
                        f"~{max(1, wait_s // 60)} នាទី។ សូមបន្ថែមកូនសោពីគណនី/គម្រោងផ្សេង "
                        f"ឬប្រើអ្នកផ្ដល់សេវាផ្សេង។",
                        provider=provider, status_code=429, retryable=False)
                    exc.retry_after = wait_s
                    raise exc
                raise ProviderError(
                    f"No usable API keys for '{provider}': all keys are "
                    f"invalid or exhausted. Add new keys in Settings.",
                    f"គ្មានកូនសោ API ដែលអាចប្រើបានសម្រាប់ '{provider}' ទេ៖ "
                    f"កូនសោទាំងអស់មិនត្រឹមត្រូវ ឬអស់កូតាហើយ។ "
                    f"សូមបញ្ចូលកូនសោថ្មីនៅក្នុងផ្នែកការកំណត់។",
                    provider=provider, status_code=None, retryable=False)
            idx = self._rr_index.get(provider, 0) % len(usable)
            self._rr_index[provider] = (idx + 1) % max(len(usable), 1)
            return usable[idx].key

    def report_success(self, provider: str, key: str, latency_ms: int) -> None:
        self._config.set_key_status(provider, key, "valid",
                                    latency_ms=int(latency_ms), error="")

    def report_failure(self, provider: str, key: str,
                       status_code: int | None, message: str = "") -> None:
        msg_l = (message or "").lower()

        # a flagged request (OpenRouter moderation, HTTP 403) says nothing
        # about the key: it must not be marked invalid for good
        if status_code == 403 and ("flagged" in msg_l or "moderation" in msg_l):
            self._config.set_key_status(provider, key, "valid",
                                        error=message or "request flagged")
            return

        # Gemini answers a bad key with HTTP 400 "API key not valid"
        if status_code in (401, 403) or (
                status_code == 400 and ("api key not valid" in msg_l
                                        or "api_key_invalid" in msg_l)):
            self._config.set_key_status(provider, key, "invalid",
                                        error=message or f"HTTP {status_code}")
            log.warning("Key marked invalid for %s (HTTP %s)", provider, status_code)
            return

        # 402 Payment Required: no credit left (OpenRouter, DeepSeek); or a
        # quota used up (free tier / daily), which resets by itself
        if status_code == 402 or any(m in msg_l for m in _EXHAUSTED_MARKERS):
            credit = is_credit_error(status_code, message)
            wait = _NO_CREDIT_RETRY_S if credit else _EXHAUSTED_RETRY_S
            self._config._set_key_exhausted(
                provider, key, time.time() + wait,
                error=message or ("HTTP 402: out of credit" if credit else "quota exhausted"))
            if credit:
                log.warning("Key out of credit for %s; tried again in %d s", provider, wait)
            else:
                log.warning("Key out of quota for %s; retried in %d min", provider,
                            wait // 60)
            return

        if status_code == 429:
            row = self._config._get_key_row(provider, key)
            streak = int(row["fail_streak"]) if row is not None else 0
            streak += 1
            cooldown = min(_RATE_LIMIT_BASE_S * (2 ** (streak - 1)),
                           _RATE_LIMIT_CAP_S)
            self._config._set_key_cooldown(
                provider, key, time.time() + cooldown, streak,
                error=message or "HTTP 429 rate limited")
            log.info("Key cooldown %ds for %s (429, streak=%d)",
                     int(cooldown), provider, streak)
            return

        # 5xx or network error (status_code None or >= 500). Other 4xx
        # responses are request/configuration errors, not key failures; do
        # not hide the real provider error behind a key cooldown.
        if status_code is not None and status_code < 500:
            self._config.set_key_status(
                provider, key, "unknown", error=message or f"HTTP {status_code}")
            return

        row = self._config._get_key_row(provider, key)
        streak = int(row["fail_streak"]) if row is not None else 0
        self._config._set_key_cooldown(
            provider, key, time.time() + _SERVER_ERR_COOLDOWN_S, streak,
            error=message or (f"HTTP {status_code}" if status_code
                              else "network error"))
        log.info("Key cooldown %ds for %s (server/network error)",
                 int(_SERVER_ERR_COOLDOWN_S), provider)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_config: ConfigManager | None = None
_default_lock = threading.Lock()


def get_default_config() -> ConfigManager:
    """Process-wide ConfigManager singleton (db at app_data_dir()/config.db)."""
    global _default_config
    if _default_config is None:
        with _default_lock:
            if _default_config is None:
                _default_config = ConfigManager()
    return _default_config


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "test_config.db"
    cfg = ConfigManager(tmp)
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)
            print(f"FAIL: {name}")

    # generic settings
    cfg.set("ui.language", "km")
    check("settings get/set", cfg.get("ui.language") == "km")
    check("settings default", cfg.get("nope", 42) == 42)
    cfg.set("nested", {"a": [1, 2, {"b": "ខ្មែរ"}]})
    check("settings json roundtrip",
          cfg.get("nested") == {"a": [1, 2, {"b": "ខ្មែរ"}]})

    # keys: add / dedup / remove
    n = cfg.add_keys("deepseek", ["sk-aaa", "sk-bbb", "sk-aaa", "", "  sk-ccc  "])
    check("add_keys added 3", n == 3)
    n2 = cfg.add_keys("deepseek", ["sk-aaa"])
    check("add_keys dedup", n2 == 0)
    keys = cfg.get_keys("deepseek")
    check("get_keys count", len(keys) == 3)
    check("get_keys decrypts", {k.key for k in keys} == {"sk-aaa", "sk-bbb", "sk-ccc"})
    cfg.remove_key("deepseek", "sk-bbb")
    check("remove_key", len(cfg.get_keys("deepseek")) == 2)

    # import from txt
    txt = tmp.parent / "keys.txt"
    txt.write_text("sk-ddd\n# comment\nsk-aaa\n\nsk-eee\n", encoding="utf-8")
    n3 = cfg.import_keys_txt("deepseek", str(txt))
    check("import_keys_txt", n3 == 2)

    # key pool rotation (round-robin over usable keys)
    pool = cfg.key_pool
    seen = [pool.acquire("deepseek") for _ in range(8)]
    check("rotation covers all keys", set(seen) == {k.key for k in cfg.get_keys("deepseek")})
    check("rotation round-robin", seen[0] != seen[1])

    # failure semantics
    pool.report_failure("deepseek", "sk-aaa", 401, "unauthorized")
    check("401 -> invalid",
          next(k for k in cfg.get_keys("deepseek") if k.key == "sk-aaa").status == "invalid")
    pool.report_failure("deepseek", "sk-ccc", 429, "rate limit")
    ki = next(k for k in cfg.get_keys("deepseek") if k.key == "sk-ccc")
    check("429 -> cooldown", ki.status == "cooldown" and ki.cooldown_until > time.time())
    check("429 cooldown ~60s", 55 <= ki.cooldown_until - time.time() <= 61)
    pool.report_failure("deepseek", "sk-ccc", 429, "rate limit again")
    ki = next(k for k in cfg.get_keys("deepseek") if k.key == "sk-ccc")
    check("429 doubling", 110 <= ki.cooldown_until - time.time() <= 121)
    pool.report_failure("deepseek", "sk-ddd", None,
                        "You exceeded your current quota (insufficient_quota)")
    check("quota -> exhausted",
          next(k for k in cfg.get_keys("deepseek") if k.key == "sk-ddd").status == "exhausted")
    pool.report_failure("deepseek", "sk-eee", 503, "server error")
    ki = next(k for k in cfg.get_keys("deepseek") if k.key == "sk-eee")
    check("5xx -> 15s cooldown",
          ki.status == "cooldown" and 10 <= ki.cooldown_until - time.time() <= 16)
    # all keys unusable -> ProviderError (bilingual)
    try:
        pool.acquire("deepseek")
        check("acquire raises when none usable", False)
    except ProviderError as exc:
        check("ProviderError bilingual", bool(exc.en) and bool(exc.km) and exc.km != exc.en)

    cfg.add_keys("deepseek", ["sk-fff"])
    pool.report_failure("deepseek", "sk-fff", 400, "invalid request")
    ki = next(k for k in cfg.get_keys("deepseek") if k.key == "sk-fff")
    check("4xx request error does not cooldown key", ki.status == "unknown")
    cfg.add_keys("deepseek", ["sk-ggg"])
    pool.report_failure("deepseek", "sk-ggg", 403, "Your input was flagged by moderation")
    ki = next(k for k in cfg.get_keys("deepseek") if k.key == "sk-ggg")
    check("a flagged request (403) does not mark the key invalid", ki.status == "valid")

    # recovery: success clears cooldown & streak
    pool.report_success("deepseek", "sk-ccc", 250)
    ki = next(k for k in cfg.get_keys("deepseek") if k.key == "sk-ccc")
    check("success -> valid + latency", ki.status == "valid" and ki.latency_ms == 250)
    check("acquire after recovery", pool.acquire("deepseek") == "sk-ccc")

    # no keys at all -> ProviderError
    try:
        pool.acquire("gemini")
        check("acquire empty provider raises", False)
    except ProviderError as exc:
        check("empty provider bilingual", bool(exc.km))

    # out of credit (402) vs a quota that resets by itself
    cfg.add_keys("openrouter", ["or-1", "or-2"])
    pool.report_failure("openrouter", "or-1", 402,
                        "openrouter HTTP 402: This request requires more credits, or fewer "
                        "max_tokens. You requested up to 8192 tokens, but can only afford 5738")
    pool.report_failure("openrouter", "or-2", 402, "")
    ki = next(k for k in cfg.get_keys("openrouter") if k.key == "or-1")
    check("402 -> out of credit, tried again within a minute",
          key_out_of_credit(ki) and 50 <= ki.cooldown_until - time.time() <= 61)
    check("402 without a message is still out of credit",
          key_out_of_credit(next(k for k in cfg.get_keys("openrouter") if k.key == "or-2")))
    try:
        pool.acquire("openrouter")
        check("acquire raises when out of credit", False)
    except ProviderError as exc:
        check("out of credit: a plain message naming the provider and the fix",
              exc.status_code == 402 and getattr(exc, "plain", False)
              and exc.en.startswith("Your OpenRouter keys are out of credit")
              and "openrouter.ai → Credits" in exc.en
              and "quota" not in exc.en and "tried again in" not in exc.en
              and "OpenRouter" in exc.km and exc.km != exc.en)
    ki = next(k for k in cfg.get_keys("deepseek") if k.key == "sk-ddd")
    check("a quota (not credit) is still retried after an hour",
          not key_out_of_credit(ki) and ki.cooldown_until - time.time() > 3000)
    check("one key: 'Your … account is out of credit'",
          no_credit_message("deepseek", 1)[0].startswith("Your DeepSeek account is out of credit.")
          and "platform.deepseek.com" in no_credit_message("deepseek", 1)[0])

    # routes
    r = cfg.get_route("recap")
    check("default route", r.provider == "deepseek" and r.model == "deepseek-chat")
    cfg.set_route("recap", Route("openai", "gpt-4o-mini", "deepseek", "deepseek-chat"))
    r2 = cfg.get_route("recap")
    check("route roundtrip", r2.provider == "openai" and r2.fallback_model == "deepseek-chat")
    check("other feature untouched", cfg.get_route("poster").provider == "deepseek")

    # usage + cost
    cfg.record_usage("deepseek", "deepseek-chat", 1_000_000, 1_000_000)
    cfg.record_usage("deepseek", "deepseek-chat", 0, 0)
    cfg.record_usage("openai", "mystery-model", 2_000_000, 0)
    usage = cfg.get_usage()
    ds = next(u for u in usage if u["model"] == "deepseek-chat")
    check("usage aggregation", ds["in_tokens"] == 1_000_000 and ds["out_tokens"] == 1_000_000)
    check("usage cost known model", abs(ds["est_cost"] - (0.27 + 1.10)) < 1e-6)
    unk = next(u for u in usage if u["model"] == "mystery-model")
    check("usage default price", abs(unk["est_cost"] - 1.0) < 1e-6)

    # export / import roundtrip (with keys, plaintext by design)
    exp = tmp.parent / "export.json"
    cfg.export_settings(str(exp), include_keys=True)
    data = json.loads(exp.read_text(encoding="utf-8"))
    check("export has settings", data["settings"]["ui.language"] == "km")
    check("export keys plaintext", "sk-ccc" in data["keys"]["deepseek"])

    tmp2 = tmp.parent / "test_config2.db"
    cfg2 = ConfigManager(tmp2)
    cfg2.import_settings(str(exp))
    check("import settings", cfg2.get("ui.language") == "km")
    check("import keys re-encrypted",
          {k.key for k in cfg2.get_keys("deepseek")}
          == {k.key for k in cfg.get_keys("deepseek")})
    r3 = cfg2.get_route("recap")
    check("import routes", r3.provider == "openai" and r3.model == "gpt-4o-mini")
    # verify stored value on disk is not plaintext
    row = cfg2._conn.execute("SELECT key_enc FROM api_keys LIMIT 1").fetchone()
    check("keys encrypted at rest", not row["key_enc"].startswith("sk-")
          and (row["key_enc"].startswith(_DPAPI_PREFIX)
               or row["key_enc"].startswith(_B64_PREFIX)))

    # thread-safety smoke test
    errs: list[Exception] = []

    def hammer(i: int) -> None:
        try:
            for j in range(25):
                cfg.set(f"t.{i}.{j}", j)
                cfg.get(f"t.{i}.{j}")
                cfg.record_usage("deepseek", "deepseek-chat", 10, 10)
                cfg.key_pool.acquire("deepseek")
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("thread-safety", not errs)

    cfg.close()
    cfg2.close()

    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        raise SystemExit(1)
    print("PASS")
