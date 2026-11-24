"""Unified AI engine over multiple providers using raw HTTP (requests only).

Providers:
    deepseek / openai / custom  -> OpenAI-compatible ``/chat/completions`` + ``/models``
    gemini                      -> native v1beta ``models/{m}:generateContent`` (``?key=`` auth)
    anthropic                   -> native ``/messages`` (``x-api-key`` + anthropic-version)

No provider SDKs are used. Feature routing, key rotation, retry/backoff and
usage accounting are driven by :class:`app.core.config_manager.ConfigManager`.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import requests

try:
    from app.core.config_manager import DEFAULT_PRICES, PROVIDER_NAMES, price_for
    from app.core.errors import CancelledError, ProviderError
    from app.core.logger import LOG_BUS, get_logger
except ImportError:  # pragma: no cover - direct script execution (__main__ tests)
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from app.core.config_manager import DEFAULT_PRICES, PROVIDER_NAMES, price_for
    from app.core.errors import CancelledError, ProviderError
    from app.core.logger import LOG_BUS, get_logger

if TYPE_CHECKING:  # avoid a hard import; config_manager is built in parallel
    from app.core.config_manager import ConfigManager

log = get_logger("ai_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "",
}
_OPENAI_COMPATIBLE = ("deepseek", "openai", "openrouter", "custom")
# OpenRouter's optional app attribution headers
_OPENROUTER_HEADERS = {"X-OpenRouter-Title": "RecapStudio", "X-Title": "RecapStudio"}

# $/1M input tokens, $/1M output tokens: the shared table in config_manager
# (usage and the monthly budget read the same one). Unknown models fall back
# to "default" -- never crash on an unknown model.
PRICE_TABLE: dict[str, tuple[float, float]] = DEFAULT_PRICES

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}
# Tried after a route's own model(s) when that model is retired, out of quota
# or has no usable key. Per provider; override with "provider.<p>.fallback_models".
_DEFAULT_FALLBACK_MODELS = {
    "gemini": ["gemini-flash-latest", "gemini-flash-lite-latest"],
}
_MODEL_BLOCK_S = 3600        # model out of quota / retired: skip it this long
_TIMEOUT_BLOCK_S = 600       # model left a request unanswered: skip it this long
_BUSY_BLOCK_S = 300          # model overloaded ("high demand", HTTP 503/529): skip it this long
# How long to wait for an answer while another model could take over. Over
# 1,700 answers in the user's logs came in under 36 s, except from a free
# Gemini Flash that took 2-5 minutes (or never came): with somewhere else to
# go, waiting longer only holds the job up. The last model gets _HTTP_TIMEOUT.
_FALLBACK_READ_S = 90
_BUSY_MARKERS = ("high demand", "overloaded", "currently unavailable")
_SLOW_NOTE_S = 30            # console note every this many seconds of waiting
_MAX_KEY_WAIT_S = 90         # wait out key cooldowns up to this long per call
_QUOTA_MARKERS = ("exceeded your current quota", "quota exceeded",
                  "resource_exhausted", "resource has been exhausted")
_RETIRED_MARKERS = ("no longer available", "is not found", "not found for api",
                    "is not supported for generatecontent", "has been deprecated",
                    "does not exist", "model_not_found")
# OpenAI reasoning models (o1/o3/o4…, gpt-5…) take only the default
# temperature, and their hidden reasoning counts toward the token limit.
_OPENAI_REASONING = re.compile(r"^(o\d|gpt-5)")
# Claude 4.7+ / 5.x / Fable / Mythos take only the default sampling (any
# temperature is a 400) and the 5.x ones think by default, which also counts
# toward max_tokens.
_CLAUDE_NO_SAMPLING = re.compile(
    r"^claude-(opus-4-[7-9]|(opus|sonnet|haiku)-[5-9]|fable|mythos)")
_THINKING_MIN_TOKENS = 8192   # room for reasoning on top of the answer
# A 400 naming one of these fields means "this model / server does not take
# it": the request is adjusted and resent (and remembered per model).
_ADAPTABLE_PARAMS = ("max_tokens", "max_completion_tokens", "temperature")
_PARAM_SWAP = {"max_tokens": "max_completion_tokens",
               "max_completion_tokens": "max_tokens"}
_PARAM_REJECTED = ("unsupported", "not supported", "unrecognized", "unknown",
                   "does not support", "not allowed", "not permitted",
                   "deprecated", "extra inputs", "only the default")
# A 4xx with none of these is about the request, not the key: other keys of
# the same provider would fail the same way.
_KEY_ERROR_MARKERS = ("api key", "api_key", "apikey", "unauthorized",
                      "authentication", "permission", "billing", "quota",
                      "credit", "balance", "payment")
_BACKOFF_DELAYS = (1.0, 2.0, 4.0)
_MAX_TRIES_PER_KEY = 3
_BATCH_SIZE = 30
_HTTP_TIMEOUT = (15, 300)  # (connect, read) seconds

_LANG_NAMES = {
    "km": "Khmer", "en": "English", "zh": "Chinese", "th": "Thai",
    "vi": "Vietnamese", "ko": "Korean", "ja": "Japanese", "fr": "French",
    "es": "Spanish", "hi": "Hindi", "id": "Indonesian", "lo": "Lao",
    "my": "Burmese", "pt": "Portuguese", "ru": "Russian", "de": "German",
}

_QUOTA_HINTS = (
    "insufficient_quota", "insufficient quota", "quota", "billing",
    "exceeded your current quota", "balance", "credit", "payment required",
    "resource_exhausted", "resource has been exhausted", "exhausted",
)


# ---------------------------------------------------------------------------
# Dataclasses (contract)
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    text: str
    in_tokens: int
    out_tokens: int
    provider: str
    model: str


@dataclass
class ImageResult:
    """A picture made by an image model (AIEngine.edit_image)."""
    data: bytes
    mime: str
    provider: str
    model: str
    note: str = ""          # any words the model sent with the picture


@dataclass
class KeyCheckResult:
    key: str
    valid: bool
    latency_ms: int
    message: str
    quota: str


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    """Estimated USD cost for a call; unknown models use the default rate."""
    price_in, price_out = price_for(model, PRICE_TABLE)
    return (in_tokens * price_in + out_tokens * price_out) / 1_000_000.0


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences, returning the inner content if present."""
    s = text.strip()
    m = re.search(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", s, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    # Unterminated fence: "```json\n{...."
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n?(.*)$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


def _find_balanced_end(s: str, start: int) -> int:
    """Index of the char closing the bracket at ``start`` (string-aware), or -1."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _lenient_loads(candidate: str) -> Any:
    """json.loads with a mild trailing-comma cleanup fallback."""
    try:
        return json.loads(candidate)
    except ValueError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(cleaned)


def extract_json(text: str) -> Any:
    """Tolerant JSON extractor.

    Strips markdown code fences, then tries the whole string, then scans for
    the first balanced ``{...}`` or ``[...]`` region.  Raises ``ValueError``
    when no JSON can be salvaged.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("extract_json: empty input")
    s = _strip_code_fences(text)
    try:
        return _lenient_loads(s)
    except ValueError:
        pass
    pos = 0
    while pos < len(s):
        starts = [i for i in (s.find("{", pos), s.find("[", pos)) if i != -1]
        if not starts:
            break
        start = min(starts)
        end = _find_balanced_end(s, start)
        if end != -1:
            try:
                return _lenient_loads(s[start:end + 1])
            except ValueError:
                pass
        pos = start + 1
    raise ValueError("extract_json: no valid JSON object or array found")


def _parse_numbered_list(text: str, expected: int) -> list[str] | None:
    """Parse strict "N. text" output.  Returns None on any count/number mismatch."""
    items: dict[int, str] = {}
    current: int | None = None
    for raw in _strip_code_fences(text).splitlines():
        line = raw.strip()
        if not line:
            current = None
            continue
        m = re.match(r"^(\d{1,5})\s*[\.\)\:៖។-]\s*(.*)$", line)
        if m:
            num = int(m.group(1))
            if not 1 <= num <= expected:
                return None  # spurious extra numbered line -> hard mismatch
            current = num
            body = m.group(2).strip()
            if current in items and items[current]:
                items[current] += " " + body
            else:
                items[current] = body
        elif current is not None:
            items[current] = (items[current] + " " + line).strip()
    if len(items) != expected or set(items.keys()) != set(range(1, expected + 1)):
        return None
    return [items[i].strip() for i in range(1, expected + 1)]


_URL_KEY = re.compile(r"([?&](?:key|api_key|apikey|access_token|token)=)[^&\s)'\"]+",
                      re.IGNORECASE)


def _redact_url_keys(text: str) -> str:
    """An error text with any key in a URL query replaced by ***."""
    return _URL_KEY.sub(r"\1***", text or "")


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}…{key[-4:]}"


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get((code or "").lower(), code or "the target language")


def _extract_error_message(body_text: str) -> str:
    """Pull a human error message out of a provider error body."""
    try:
        data = json.loads(body_text)
    except ValueError:
        return body_text.strip()[:500]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or err)[:500]
        if isinstance(err, str):
            return err[:500]
        if "message" in data:
            return str(data["message"])[:500]
        if "detail" in data:
            return str(data["detail"])[:500]
    return body_text.strip()[:500]


# finish_reason of a reply its model's provider blocked (OpenAI-style and
# Gemini-style names, as OpenRouter passes them on)
_BLOCKED_FINISH = {"content_filter", "safety", "prohibited_content", "blocklist",
                   "recitation", "spii", "image_safety"}
_MODERATION_MARKERS = ("flagged", "moderation")


def _reply_error(provider: str, data: Any) -> ProviderError:
    """The error for an HTTP 200 reply that has no answer. OpenRouter puts
    its model provider's failure inside such a reply ({"error": {"code": 429,
    "message": …}}); the reason used to be dropped and the request repeated
    as if the network had failed (13 times on 2026-11-11, then "All
    providers failed"). Now a rate limit cools the key down and lets another
    key or provider answer, no credit marks the key out of credit, a blocked
    request tries another model — and the log says which."""
    err = data.get("error") if isinstance(data, dict) else None
    code: int | None = None
    if isinstance(err, dict):
        msg = str(err.get("message") or err.get("type") or err)
        meta = err.get("metadata")
        if isinstance(meta, dict) and meta.get("raw"):
            msg += f" — {str(meta['raw'])[:200]}"
        try:
            code = int(err.get("code"))
        except (TypeError, ValueError):
            code = None
    elif isinstance(err, str):
        msg = err
    else:
        body = json.dumps(data, ensure_ascii=False)[:300] if data else "empty reply"
        msg = f"no answer in the reply ({body})"
    if code is not None and not 100 <= code <= 599:
        code = None
    if code == 403 and any(m in msg.lower() for m in _MODERATION_MARKERS):
        code = 400          # the request was flagged, the key is fine
    return ProviderError(
        en=f"{provider} replied without an answer: {msg[:400]}",
        km=f"{provider} បានឆ្លើយតបដោយគ្មានចម្លើយ៖ {msg[:400]}",
        provider=provider, status_code=code,
        retryable=code is not None and code in _RETRYABLE_STATUS)


def _quota_hint(message: str) -> str:
    low = (message or "").lower()
    for hint in _QUOTA_HINTS:
        if hint in low:
            return f"quota/billing: {message[:200]}"
    return ""


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("operation cancelled")


def _drop_rejected_param(exc: ProviderError, payload: dict,
                         names: tuple[str, ...] = _ADAPTABLE_PARAMS) -> str:
    """If `exc` is a 400 saying one of `names` in `payload` is not supported,
    change the payload so it can be resent (drop the field, or rename
    max_tokens <-> max_completion_tokens) and return the field name; else ""."""
    if exc.status_code not in (400, 422):
        return ""
    low = exc.en.lower()
    if not any(w in low for w in _PARAM_REJECTED):
        return ""
    for name in names:
        if name in payload and re.search(rf"\b{name}\b", low):
            value = payload.pop(name)
            if name in _PARAM_SWAP:
                payload[_PARAM_SWAP[name]] = value
            return name
    return ""


# OpenRouter reserves max_tokens × the model's price before answering and
# refuses (HTTP 402) when the balance is lower: "This request requires more
# credits, or fewer max_tokens. You requested up to 8192 tokens, but can only
# afford 5738" — though most answers are far shorter than max_tokens.
_AFFORD_RE = re.compile(r"can only afford (\d+)", re.IGNORECASE)
_MIN_AFFORDABLE_TOKENS = 512    # fewer than this: not worth asking again


def _affordable_tokens(exc: ProviderError, asked: int) -> int:
    """How many answer tokens to ask for after a 402 "can only afford N"
    (a little under N: prices are rounded), or 0 when there is no such
    number, it is no lower than `asked`, or too few to be worth it."""
    if exc.status_code != 402:
        return 0
    m = _AFFORD_RE.search(exc.en or "")
    if not m:
        return 0
    n = int(int(m.group(1)) * 0.97)
    return n if _MIN_AFFORDABLE_TOKENS <= n < asked else 0


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _last_user(msgs: list[dict]) -> dict:
    """The last user message (pictures go with it); one is added if none."""
    for m in reversed(msgs):
        if m.get("role") == "user":
            return m
    msgs.append({"role": "user", "content": ""})
    return msgs[-1]


def _is_request_error(exc: ProviderError) -> bool:
    """A 4xx about the request itself (bad model id, too long, bad field),
    not about the key: trying the provider's other keys cannot help."""
    code = exc.status_code
    low = exc.en.lower()
    if code == 403 and any(m in low for m in _MODERATION_MARKERS):
        return True       # OpenRouter: the input was flagged — not the key
    if code is None or code >= 500 or code in (401, 402, 403, 408, 429):
        return False      # 402 Payment Required: that key's account is out of credit
    return not any(m in low for m in _KEY_ERROR_MARKERS)


def ai_unavailable(exc: BaseException) -> bool:
    """chat() found no AI to answer — every key and provider rate-limited,
    out of credit, invalid, the monthly budget reached, or no connection —
    as opposed to a reply or request the AI could not handle. Work that
    needs the AI should stop (and be resumed later) rather than be skipped."""
    return isinstance(exc, ProviderError) and not _is_request_error(exc)


# chat() with no API key for any provider: one plain message instead of
# "All providers failed … No API keys configured for provider 'deepseek'"
# (the default route's provider, which the user never chose)
NO_KEY_EN = ("No AI key yet. Add one in Settings → AI Providers "
             "(for example an OpenRouter key), then try again.")
NO_KEY_KM = ("មិនទាន់មាន AI key នៅឡើយទេ។ សូមបន្ថែមមួយនៅ ការកំណត់ → អ្នកផ្ដល់ AI "
             "(ឧទាហរណ៍ key របស់ OpenRouter) រួចសាកម្ដងទៀត។")

# what to do, per reason no AI could answer (ai_failure_kind); the dub's
# paused message and its saved state use these
AI_ADVICE: dict[str, tuple[str, str]] = {
    "no_key": ("No AI key yet: add one in Settings → AI Providers "
               "(for example an OpenRouter key).",
               "មិនទាន់មាន AI key នៅឡើយទេ៖ សូមបន្ថែមមួយនៅ ការកំណត់ → អ្នកផ្ដល់ AI "
               "(ឧទាហរណ៍ key របស់ OpenRouter)។"),
    "bad_key": ("The AI service refused the API key (wrong, expired or turned off). "
                "Check it in Settings → AI Providers, or add another key.",
                "សេវា AI មិនទទួល API key ទេ (ខុស ផុតកំណត់ ឬត្រូវបានបិទ)។ "
                "សូមពិនិត្យវានៅ ការកំណត់ → អ្នកផ្ដល់ AI ឬបន្ថែម key ផ្សេងទៀត។"),
    "no_credit": ("The AI account is out of credit or quota. Top it up on the "
                  "provider's website, or add a key of another provider in "
                  "Settings → AI Providers.",
                  "គណនី AI អស់ទឹកប្រាក់ ឬអស់កូតាហើយ។ សូមបញ្ចូលទឹកប្រាក់នៅគេហទំព័ររបស់អ្នកផ្ដល់សេវា "
                  "ឬបន្ថែម key របស់អ្នកផ្ដល់សេវាផ្សេងនៅ ការកំណត់ → អ្នកផ្ដល់ AI។"),
    "busy": ("The AI service is busy or its rate limit is reached for now. "
             "Wait a few minutes, or add another key in Settings → AI Providers.",
             "សេវា AI រវល់ ឬដល់ដែនកំណត់ល្បឿនសម្រាប់ពេលនេះ។ សូមរង់ចាំពីរបីនាទី "
             "ឬបន្ថែម key ផ្សេងទៀតនៅ ការកំណត់ → អ្នកផ្ដល់ AI។"),
    "timeout": ("The AI service did not answer in time. Try again in a few minutes, "
                "or pick another model in Settings → Per-feature routing.",
                "សេវា AI មិនបានឆ្លើយទាន់ពេលទេ។ សូមសាកម្ដងទៀតក្នុងពីរបីនាទីទៀត "
                "ឬជ្រើសម៉ូដែលផ្សេងនៅ ការកំណត់ → ការកំណត់ AI តាមមុខងារ។"),
    "offline": ("The AI service could not be reached: check the internet connection.",
                "មិនអាចភ្ជាប់ទៅសេវា AI បានទេ៖ សូមពិនិត្យការតភ្ជាប់អ៊ីនធឺណិត។"),
    "": ("No AI key could answer.", "គ្មាន AI key ណាអាចឆ្លើយបានទេ។"),
}


def ai_failure_kind(exc: BaseException) -> str:
    """Why no AI could answer, for advice the user can act on: "no_key",
    "budget" (the app's own monthly limit), "bad_key", "no_credit",
    "busy" (rate limit / server busy), "timeout", "offline", or "" (other)."""
    if not isinstance(exc, ProviderError):
        return ""
    if exc.provider in ("no_key", "budget"):
        return exc.provider
    code = exc.status_code
    low = (exc.en or "").lower()
    if code in (401, 403) or any(w in low for w in (
            "invalid", "unauthori", "incorrect api key", "api key not valid")):
        return "bad_key"
    if code == 402 or any(w in low for w in (
            "out of quota", "out of credit", "insufficient", "billing")):
        return "no_credit"
    if code == 429 or any(w in low for w in ("rate-limit", "rate limit", "too many requests")):
        return "busy"        # a per-minute limit too ("check quota" in Gemini's text)
    if "credit" in low or "quota" in low:
        return "no_credit"
    if getattr(exc, "timeout", False) or code == 408 or "did not answer within" in low:
        return "timeout"
    if "network error" in low or "connection" in low or "name resolution" in low:
        return "offline"
    if code is not None and code >= 500:
        return "busy"
    return ""


# ---------------------------------------------------------------------------
# AIEngine
# ---------------------------------------------------------------------------

class AIEngine:
    """Unified chat/translation engine over deepseek/gemini/openai/anthropic/custom."""

    def __init__(self, config: "ConfigManager"):
        self.config = config

    # -- configuration helpers ---------------------------------------------

    def _base_url(self, provider: str) -> str:
        url = self.config.get(f"provider.{provider}.base_url") or DEFAULT_BASE_URLS.get(provider, "")
        url = (url or "").rstrip("/")
        if not url:
            raise ProviderError(
                en=f"No base URL configured for provider '{provider}'.",
                km=f"មិនទាន់បានកំណត់ Base URL សម្រាប់អ្នកផ្តល់សេវា '{provider}' ទេ។ សូមកំណត់នៅក្នុងផ្ទាំងការកំណត់។",
                provider=provider, status_code=None, retryable=False,
            )
        return url

    # -- low-level HTTP ------------------------------------------------------

    def _http(self, method: str, url: str, *, provider: str,
              headers: dict | None = None, params: dict | None = None,
              payload: dict | None = None,
              cancel: threading.Event | None = None, label: str = "") -> dict:
        # requests cannot be interrupted, so the call runs on its own daemon
        # thread: Cancel then works at once (the abandoned request ends at its
        # own timeout) and a slow answer is reported in the console.
        box: dict[str, Any] = {}
        done = threading.Event()
        # chat() shortens the wait while another model could answer instead
        read_s = int(getattr(self._tls, "read_s", 0) or _HTTP_TIMEOUT[1])

        def run() -> None:
            try:
                box["resp"] = requests.request(
                    method, url, headers=headers, params=params, json=payload,
                    timeout=(_HTTP_TIMEOUT[0], read_s),
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                box["exc"] = exc
            finally:
                done.set()

        threading.Thread(target=run, daemon=True, name="ai-http").start()
        started = time.monotonic()
        note_at = _SLOW_NOTE_S
        who = label or provider
        while not done.wait(0.25):
            _check_cancel(cancel)
            waited = time.monotonic() - started
            if waited >= note_at:
                LOG_BUS.publish(f"[ai] still waiting for {who}: {int(waited)} s "
                                f"(gives up at {read_s} s)…")
                note_at += _SLOW_NOTE_S
        exc = box.get("exc")
        if isinstance(exc, requests.ReadTimeout):
            err = ProviderError(
                en=f"{who} did not answer within {read_s} s.",
                km=f"{who} មិនបានឆ្លើយតបក្នុងរយៈពេល {read_s} វិនាទីទេ។",
                provider=provider, status_code=None, retryable=True,
            )
            err.timeout = True
            err.after = read_s
            raise err from exc
        if isinstance(exc, requests.RequestException):
            # requests quotes the URL, and Gemini's carries the key
            # ("…:generateContent?key=AIza…"): never into the log or the UI
            text = _redact_url_keys(str(exc))
            raise ProviderError(
                en=f"Network error calling {provider}: {text}",
                km=f"បញ្ហាបណ្តាញពេលហៅទៅ {provider}៖ {text}",
                provider=provider, status_code=None, retryable=True,
            ) from None
        if exc is not None:
            raise exc
        resp = box["resp"]
        if resp.status_code >= 400:
            body = resp.text or ""
            msg = _extract_error_message(body)
            retryable = resp.status_code in _RETRYABLE_STATUS
            exc = ProviderError(
                en=f"{provider} HTTP {resp.status_code}: {msg}",
                km=f"{provider} បានឆ្លើយតបកំហុស HTTP {resp.status_code}៖ {msg}",
                provider=provider, status_code=resp.status_code, retryable=retryable,
            )
            if resp.status_code == 429:
                # Gemini says which quota ran out (…PerMinute… / …PerDay…) and
                # how long to wait (RetryInfo.retryDelay "37s")
                exc.quota_scope = ("day" if "PerDay" in body else
                                   "minute" if "PerMinute" in body else "")
                m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
                ra = resp.headers.get("Retry-After", "")
                exc.retry_after = (float(m.group(1)) if m else
                                   float(ra) if ra.replace(".", "", 1).isdigit() else 0.0)
            raise exc
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(
                en=f"{provider} returned a non-JSON response.",
                km=f"{provider} បានឆ្លើយតបទិន្នន័យមិនមែន JSON ទេ។",
                provider=provider, status_code=resp.status_code, retryable=True,
            ) from exc

    _rejected: dict = {}     # (provider, model) -> fields that model refused

    def _post_adapting(self, provider: str, model: str, url: str,
                       payload: dict, names: tuple[str, ...] = _ADAPTABLE_PARAMS,
                       **kw) -> dict:
        """POST `payload`; when the server refuses one of its optional fields
        (`names`) for this model (e.g. temperature on o-series / Claude 4.7+,
        or max_tokens on gpt-5), adjust it, remember that, and resend."""
        kw.setdefault("label", f"{provider}/{model}")
        rejected = AIEngine._rejected.setdefault((provider, model), set())
        for name in sorted(rejected):
            if name in payload:
                value = payload.pop(name)
                if name in _PARAM_SWAP and _PARAM_SWAP[name] not in rejected:
                    payload[_PARAM_SWAP[name]] = value
        for _ in range(len(names)):
            try:
                return self._http("POST", url, provider=provider,
                                  payload=payload, **kw)
            except ProviderError as exc:
                name = _drop_rejected_param(exc, payload, names)
                if not name:
                    raise
                rejected.add(name)
                log.info("%s %s does not take '%s'; resending without it",
                         provider, model, name)
        return self._http("POST", url, provider=provider, payload=payload, **kw)

    # -- per-provider chat implementations ----------------------------------
    # `images` [(mime type, bytes)]: pictures sent with the last user message
    # (a poster to read); a model that cannot see pictures refuses the request

    @staticmethod
    def _merge_consecutive(messages: list[dict]) -> list[dict]:
        merged: list[dict] = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += "\n" + content
            else:
                merged.append({"role": role, "content": content})
        return merged

    def _chat_openai_compatible(self, provider: str, key: str, model: str,
                                messages: list[dict], system: str | None,
                                temperature: float, max_tokens: int,
                                json_mode: bool,
                                cancel: threading.Event | None = None,
                                images: list | None = None
                                ) -> tuple[str, int, int]:
        url = self._base_url(provider) + "/chat/completions"
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                    for m in messages)
        if images:
            m = _last_user(msgs)
            m["content"] = ([{"type": "text", "text": str(m.get("content") or "")}]
                            + [{"type": "image_url",
                                "image_url": {"url": f"data:{mt};base64,{_b64(data)}"}}
                               for mt, data in images])
        payload: dict[str, Any] = {"model": model, "messages": msgs, "stream": False}
        if _OPENAI_REASONING.match(model.lower().split("/")[-1]):
            # only the default temperature; reasoning tokens need room too
            max_tokens = max(max_tokens, _THINKING_MIN_TOKENS)
        else:
            payload["temperature"] = temperature
        # OpenAI itself wants max_completion_tokens (o-series / gpt-5 refuse
        # max_tokens); compatible servers (DeepSeek, custom) know max_tokens
        payload["max_completion_tokens" if provider == "openai"
                else "max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers.update(_OPENROUTER_HEADERS)
        fewer = 0
        try:
            data = self._post_adapting(provider, model, url, payload, cancel=cancel,
                                       headers=headers)
        except ProviderError as exc:
            field = next((f for f in ("max_tokens", "max_completion_tokens")
                          if f in payload), "")
            fewer = _affordable_tokens(exc, int(payload.get(field) or 0)) if field else 0
            if not fewer:
                raise
            # the balance covers a shorter answer: ask for that once (it only
            # stretches the last cents — the account needs credit soon)
            note = (f"{provider}: the account's credit covers only {fewer} answer tokens "
                    f"(asked {payload[field]}) — asking for {fewer}; add credit soon")
            log.warning(note)
            LOG_BUS.publish(f"[ai] {note}")
            payload[field] = fewer
            data = self._post_adapting(provider, model, url, payload, cancel=cancel,
                                       headers=headers)
        try:
            choice = data["choices"][0]
            text = choice.get("message", {}).get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise _reply_error(provider, data) from exc
        if fewer and str(choice.get("finish_reason") or "").lower() == "length":
            # the answer did not fit in what the credit paid for: out of credit
            raise ProviderError(
                en=(f"{provider} HTTP 402: the answer was cut short — the account's "
                    f"credit paid for only {fewer} tokens."),
                km=(f"{provider} HTTP 402៖ ចម្លើយត្រូវបានកាត់ខ្លី — ទឹកប្រាក់ក្នុងគណនី"
                    f"បង់បានតែ {fewer} token ប៉ុណ្ណោះ។"),
                provider=provider, status_code=402, retryable=False)
        if not text.strip() and str(choice.get("finish_reason") or "").lower() in _BLOCKED_FINISH:
            # the model's provider blocked the reply: the same request on
            # another key would be blocked too — another model may answer
            raise ProviderError(
                en=(f"{provider} {model} blocked the reply "
                    f"(finish_reason: {choice.get('finish_reason')})."),
                km=f"{provider} {model} បានរារាំងចម្លើយ។",
                provider=provider, status_code=400, retryable=False)
        usage = data.get("usage") or {}
        in_t = int(usage.get("prompt_tokens") or 0)
        out_t = int(usage.get("completion_tokens") or 0)
        return text, in_t, out_t

    def _chat_gemini(self, key: str, model: str, messages: list[dict],
                     system: str | None, temperature: float, max_tokens: int,
                     json_mode: bool, cancel: threading.Event | None = None,
                     images: list | None = None
                     ) -> tuple[str, int, int]:
        provider = "gemini"
        model_id = model[len("models/"):] if model.startswith("models/") else model
        url = f"{self._base_url(provider)}/models/{model_id}:generateContent"
        contents: list[dict] = []
        extra_system: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            text = str(m.get("content", ""))
            if role == "system":
                extra_system.append(text)
                continue
            g_role = "model" if role == "assistant" else "user"
            if contents and contents[-1]["role"] == g_role:
                contents[-1]["parts"].append({"text": text})
            else:
                contents.append({"role": g_role, "parts": [{"text": text}]})
        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})
        if images:
            user = next((c for c in reversed(contents) if c["role"] == "user"), None)
            if user is None:
                user = {"role": "user", "parts": []}
                contents.append(user)
            # the picture first, then what to do with it (Google's advice)
            user["parts"][:0] = [{"inline_data": {"mime_type": mt, "data": _b64(data)}}
                                 for mt, data in images]
        gen_cfg: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        thinking = self._gemini_thinking(model_id)
        if thinking:
            gen_cfg["thinkingConfig"] = thinking
        payload: dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
        sys_text = "\n".join(x for x in ([system] if system else []) + extra_system if x)
        if sys_text:
            payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
        label = f"gemini/{model_id}"
        try:
            data = self._http("POST", url, provider=provider, payload=payload,
                              params={"key": key},
                              headers={"Content-Type": "application/json"},
                              cancel=cancel, label=label)
        except ProviderError as exc:
            if not (thinking and exc.status_code == 400 and "think" in exc.en.lower()):
                raise
            # this model does not take that thinking setting: remember, resend
            AIEngine._no_thinking.add(model_id)
            log.info("gemini %s rejected thinkingConfig; sending without it", model_id)
            gen_cfg.pop("thinkingConfig", None)
            data = self._http("POST", url, provider=provider, payload=payload,
                              params={"key": key},
                              headers={"Content-Type": "application/json"},
                              cancel=cancel, label=label)
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
            raise ProviderError(
                en=f"gemini returned no candidates (blockReason: {block}).",
                km=f"gemini មិនបានផ្តល់ចម្លើយទេ (ត្រូវបានទប់ស្កាត់៖ {block})។",
                provider=provider, status_code=None, retryable=False,
            )
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts
                       if isinstance(p, dict) and not p.get("thought"))
        meta = data.get("usageMetadata") or {}
        in_t = int(meta.get("promptTokenCount") or 0)
        # thinking tokens are billed as output but reported separately
        out_t = (int(meta.get("candidatesTokenCount") or 0)
                 + int(meta.get("thoughtsTokenCount") or 0))
        return text, in_t, out_t

    _no_thinking: set = set()

    def _gemini_thinking(self, model_id: str) -> dict | None:
        """thinkingConfig for a Gemini model from setting provider.gemini.thinking
        ("low" default | "high" | "off" | "model" = the model's own default).
        Translation and JSON tasks gain little from long thinking, and Gemini 3
        defaults to a lot of it (minutes per call, billed as output)."""
        level = str(self.config.get("provider.gemini.thinking") or "low").lower()
        m = model_id.lower()
        if level == "model" or model_id in self._no_thinking:
            return None
        if m.startswith("gemini-2.5"):
            if level == "high":
                return None
            # flash / flash-lite may switch thinking off; pro needs >= 128
            if "pro" in m:
                return {"thinkingBudget": 128 if level == "off" else 1024}
            return {"thinkingBudget": 0 if level == "off" else 512}
        if m.startswith("gemini-3") or m.startswith("gemini-4") or m.endswith("-latest"):
            return {"thinkingLevel": "high" if level == "high" else "low"}
        return None

    def _chat_anthropic(self, key: str, model: str, messages: list[dict],
                        system: str | None, temperature: float, max_tokens: int,
                        json_mode: bool, cancel: threading.Event | None = None,
                        images: list | None = None
                        ) -> tuple[str, int, int]:
        provider = "anthropic"
        url = self._base_url(provider) + "/messages"
        sys_parts: list[str] = [system] if system else []
        msgs: list[dict] = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "system":
                sys_parts.append(content)
                continue
            msgs.append({"role": "assistant" if role == "assistant" else "user",
                         "content": content})
        msgs = self._merge_consecutive(msgs)
        if not msgs or msgs[0]["role"] != "user":
            msgs.insert(0, {"role": "user", "content": "(continue)"})
        if images:
            m = _last_user(msgs)
            text = str(m.get("content") or "")
            m["content"] = ([{"type": "image", "source": {"type": "base64", "media_type": mt,
                                                          "data": _b64(data)}}
                             for mt, data in images]
                            + ([{"type": "text", "text": text}] if text.strip() else []))
        if json_mode:
            # No native response_format: enforce via prompt instruction.
            sys_parts.append(
                "Respond with a single valid JSON object or array only. "
                "Do not wrap it in markdown code fences and do not add any "
                "explanation before or after the JSON."
            )
        payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                   "messages": msgs}
        if _CLAUDE_NO_SAMPLING.match(model.lower()):
            # any temperature is a 400 here, and default-on thinking counts
            # toward max_tokens
            payload["max_tokens"] = max(max_tokens, _THINKING_MIN_TOKENS)
        else:
            payload["temperature"] = temperature
        sys_text = "\n\n".join(p for p in sys_parts if p)
        if sys_text:
            payload["system"] = sys_text
        data = self._post_adapting(
            provider, model, url, payload, names=("temperature",), cancel=cancel,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        if not text and data.get("stop_reason") == "refusal":
            raise ProviderError(
                en="anthropic declined the request (refusal).",
                km="anthropic បានបដិសេធសំណើនេះ (refusal)។",
                provider=provider, status_code=None, retryable=False,
            )
        usage = data.get("usage") or {}
        in_t = int(usage.get("input_tokens") or 0)
        out_t = int(usage.get("output_tokens") or 0)
        return text, in_t, out_t

    def _dispatch_chat(self, provider: str, key: str, model: str,
                       messages: list[dict], system: str | None,
                       temperature: float, max_tokens: int,
                       json_mode: bool, cancel: threading.Event | None = None,
                       images: list | None = None
                       ) -> tuple[str, int, int]:
        pics = {"images": images} if images else {}
        if provider in _OPENAI_COMPATIBLE:
            return self._chat_openai_compatible(
                provider, key, model, messages, system, temperature,
                max_tokens, json_mode, cancel, **pics)
        if provider == "gemini":
            return self._chat_gemini(key, model, messages, system,
                                     temperature, max_tokens, json_mode, cancel, **pics)
        if provider == "anthropic":
            return self._chat_anthropic(key, model, messages, system,
                                        temperature, max_tokens, json_mode, cancel, **pics)
        raise ProviderError(
            en=f"Unknown provider '{provider}'.",
            km=f"មិនស្គាល់អ្នកផ្តល់សេវា '{provider}' ទេ។",
            provider=provider, status_code=None, retryable=False,
        )

    # -- budget ----------------------------------------------------------------

    _warned_month: str = ""

    def check_budget(self) -> None:
        """Raise ProviderError once this month's estimated spend reaches the
        limit set in Settings (budget.monthly_usd; 0 = unlimited); log a
        one-time warning when it crosses the warn fraction."""
        try:
            limit, warn = self.config.get_budget()
            if limit <= 0:
                return
            spent = self.config.month_cost()
        except Exception:
            return
        if spent >= limit:
            raise ProviderError(
                en=(f"Monthly AI budget reached: ${spent:.2f} of ${limit:.2f} spent. "
                    "Raise the limit in Settings → Usage to continue."),
                km=(f"ថវិកា AI ប្រចាំខែបានអស់ហើយ៖ បានចំណាយ ${spent:.2f} នៃ ${limit:.2f}។ "
                    "សូមបង្កើនដែនកំណត់នៅ ការកំណត់ → ការប្រើប្រាស់។"),
                provider="budget", status_code=None, retryable=False)
        month = time.strftime("%Y-%m")
        if spent >= limit * warn and self._warned_month != month:
            AIEngine._warned_month = month
            msg = (f"AI spend this month is ${spent:.2f} — "
                   f"{spent / limit:.0%} of your ${limit:.2f} budget")
            log.warning(msg)
            LOG_BUS.publish(f"[budget] {msg}")

    def _check_any_key(self) -> None:
        """No provider has an API key at all: say that, not the default
        route's provider failing ("No API keys configured for 'deepseek'")."""
        try:
            keyed = self.config.providers_with_keys()
        except Exception:
            return
        if not keyed:
            raise ProviderError(en=NO_KEY_EN, km=NO_KEY_KM, provider="no_key",
                                status_code=None, retryable=False)

    # -- public chat ---------------------------------------------------------

    def chat(self, *, feature: str | None = None, provider: str | None = None,
             model: str | None = None, messages: list[dict],
             system: str | None = None, temperature: float = 0.7,
             max_tokens: int = 4096, json_mode: bool = False,
             cancel: threading.Event | None = None,
             images: list[tuple[str, bytes]] | None = None) -> ChatResult:
        self.check_budget()
        self._check_any_key()
        attempts = self._resolve_targets(feature, provider, model)
        pool = self.config.key_pool
        last_error: ProviderError | None = None
        first_plain: ProviderError | None = None    # e.g. the chosen provider out of credit
        waited = 0.0
        waiter = cancel if cancel is not None else threading.Event()
        upstream_retried: set[tuple[str, str]] = set()

        # skip models blocked for a while (retired / out of quota / hanging),
        # unless every target is: then trying them beats failing at once
        now = time.time()
        targets = [t for t in attempts if self._blocked_models.get(t, 0.0) <= now] or attempts
        for ti, (prov, mdl) in enumerate(targets):
            _check_cancel(cancel)
            tried_keys: set[str] = set()
            timeouts = 0
            while True:
                _check_cancel(cancel)
                try:
                    key = pool.acquire(prov)
                except ProviderError as exc:
                    last_error = exc
                    if first_plain is None and getattr(exc, "plain", False):
                        first_plain = exc
                    wait = float(getattr(exc, "retry_after", 0) or 0)
                    later = next((t for t in targets[ti + 1:]
                                  if t[0] != prov and pool.has_usable(t[0])), None)
                    if later is not None and self.config.get_keys(prov):
                        # this provider's keys are all cooling down / out of
                        # credit, another provider's key works now: go on there
                        msg = (f"all {prov} keys are rate-limited or out of credit — "
                               f"switching to {later[0]}/{later[1]}")
                        log.warning(msg)
                        LOG_BUS.publish(f"[ai] {msg}")
                        break
                    if exc.retryable and wait and waited + wait <= _MAX_KEY_WAIT_S:
                        # every key is in a short cooldown: wait instead of failing
                        log.info("chat: all %s keys cooling down; waiting %.0fs", prov, wait)
                        if waiter.wait(wait):
                            raise CancelledError("operation cancelled")
                        waited += wait
                        tried_keys.clear()
                        continue
                    log.warning("chat: no usable key for %s (%s)", prov, exc.en)
                    break  # move to fallback model / provider
                if key in tried_keys:
                    break  # round-robin wrapped: all keys tried for this call
                tried_keys.add(key)
                # another model that could answer instead: don't wait minutes here
                later = next((t for t in targets[ti + 1:] if pool.has_usable(t[0])), None)
                self._tls.read_s = _FALLBACK_READ_S if later is not None else 0
                try:
                    result = self._call_with_retries(
                        prov, key, mdl, messages, system, temperature,
                        max_tokens, json_mode, cancel,
                        **({"images": images} if images else {}))
                except CancelledError:
                    raise
                except ProviderError as exc:
                    last_error = exc
                    kind = getattr(exc, "model_problem", "")
                    if kind == "timeout":
                        timeouts += 1
                        after = int(getattr(exc, "after", 0) or _HTTP_TIMEOUT[1])
                        if later is None and timeouts < 2:
                            # nothing to fall back to: one more try
                            LOG_BUS.publish(f"[ai] {prov}/{mdl} did not answer within "
                                            f"{after} s — trying once more")
                            tried_keys.discard(key)
                            continue
                        # skip it on the next calls too while it hangs
                        self._blocked_models[(prov, mdl)] = time.time() + _TIMEOUT_BLOCK_S
                        msg = (f"{prov}/{mdl} did not answer within {after} s — "
                               + (f"trying {later[0]}/{later[1]}"
                                  if later is not None else "giving up; try again later or "
                                  "pick another model in Settings → Routing"))
                        log.warning(msg)
                        LOG_BUS.publish(f"[ai] {msg}")
                        break
                    if kind == "busy":
                        if later is None:
                            continue    # nowhere else to go: the next key (it may get through)
                        # overloaded for every key alike: the next model now,
                        # and the next requests skip this one for a while
                        self._blocked_models[(prov, mdl)] = time.time() + _BUSY_BLOCK_S
                        msg = (f"{prov}/{mdl} is overloaded (high demand) — trying "
                               f"{later[0]}/{later[1]}")
                        log.warning(msg)
                        LOG_BUS.publish(f"[ai] {msg}")
                        break
                    if kind:
                        # retired or out of quota: other keys won't help
                        self._blocked_models[(prov, mdl)] = time.time() + _MODEL_BLOCK_S
                        msg = (f"{prov} model '{mdl}' is "
                               + ("retired / not available" if kind == "retired"
                                  else "out of quota")
                               + " — trying the next model. Update Settings → Routing.")
                        log.warning(msg)
                        LOG_BUS.publish(f"[ai] {msg}")
                        break
                    if ("provider returned error" in exc.en.lower()
                            and exc.status_code != 429
                            and (prov, mdl) not in upstream_retried):
                        # OpenRouter's wrapper for a hiccup at the model's own
                        # provider (arrives as HTTP 400, which otherwise means a
                        # bad request): the same call once more, a moment later
                        upstream_retried.add((prov, mdl))
                        LOG_BUS.publish(f"[ai] {prov}/{mdl}: the model's provider returned "
                                        "an error — trying once more")
                        tried_keys.discard(key)
                        if waiter.wait(2.0):
                            raise CancelledError("operation cancelled")
                        continue
                    if _is_request_error(exc):
                        # bad model id, prompt too long, unsupported field…:
                        # every key fails the same way, so try the next model
                        log.warning("chat: %s/%s refused the request (%s); "
                                    "not retrying with other keys", prov, mdl, exc.en)
                        break
                    continue  # next key
                finally:
                    self._tls.read_s = 0
                # success
                text, in_t, out_t = result
                try:
                    self.config.record_usage(prov, mdl, in_t, out_t)
                except Exception:  # accounting must never break a good call
                    log.exception("record_usage failed")
                if json_mode:
                    try:
                        parsed = extract_json(text)
                        text = json.dumps(parsed, ensure_ascii=False)
                    except ValueError:
                        log.warning("json_mode: could not salvage JSON from %s reply", prov)
                if (prov, mdl) != attempts[0]:
                    LOG_BUS.publish(f"[ai] answered by fallback {prov}/{mdl}")
                return ChatResult(text=text, in_tokens=in_t, out_tokens=out_t,
                                  provider=prov, model=mdl)

        if last_error is not None:
            if first_plain is not None:
                # it already says what happened and what to do ("Your
                # OpenRouter account is out of credit…"): no technical
                # wrapper, and it comes first even when another provider was
                # tried after it (that one's answer follows, in brackets)
                en, km = first_plain.en, first_plain.km
                if last_error.provider != first_plain.provider:
                    name = PROVIDER_NAMES.get(last_error.provider, last_error.provider)
                    en += f" (Then {name}: {last_error.en})"
                    km += f" (បន្ទាប់មក {name}៖ {last_error.km})"
                err = ProviderError(en=en, km=km, provider=first_plain.provider,
                                    status_code=first_plain.status_code, retryable=False)
                err.plain = True
                raise err
            raise ProviderError(
                en=f"All providers failed. Last error: {last_error.en}",
                km=f"អ្នកផ្តល់សេវាទាំងអស់បរាជ័យ។ កំហុសចុងក្រោយ៖ {last_error.km}",
                provider=last_error.provider,
                status_code=last_error.status_code, retryable=False,
            )
        raise ProviderError(
            en="No AI provider or usable API key is configured.",
            km="មិនទាន់មានអ្នកផ្តល់សេវា AI ឬសោ API ដែលអាចប្រើបានទេ។ សូមបន្ថែមសោនៅក្នុងផ្ទាំងការកំណត់។",
            provider=provider or "", status_code=None, retryable=False,
        )

    _blocked_models: dict = {}
    # per thread (dub batches run side by side): the answer wait of the call
    # being made, 0 = _HTTP_TIMEOUT
    _tls = threading.local()

    def _fallback_models(self, provider: str) -> list[str]:
        custom = self.config.get(f"provider.{provider}.fallback_models")
        if isinstance(custom, list):
            return [str(m) for m in custom if str(m).strip()]
        return list(_DEFAULT_FALLBACK_MODELS.get(provider, []))

    def _resolve_targets(self, feature: str | None, provider: str | None,
                         model: str | None) -> list[tuple[str, str]]:
        """Ordered (provider, model) attempts: the route, its fallback, the
        provider's fallback models, and — if the route's provider has no keys —
        the default route of a provider that has keys."""
        attempts = self._route_targets(feature, provider, model)
        if provider:            # explicit provider+model: no substitutions
            return attempts
        extra: list[tuple[str, str]] = []
        keyed = set()
        try:
            keyed = set(self.config.providers_with_keys())
        except Exception:
            pass
        for prov, _m in list(attempts):
            for fm in self._fallback_models(prov):
                extra.append((prov, fm))
        if keyed and not any(p in keyed for p, _m in attempts + extra):
            try:
                d = self.config._default_route()
                if d.provider in keyed and d.model:
                    log.warning("route for %s uses %s without API keys; using %s/%s",
                                feature, attempts[0][0] if attempts else "?",
                                d.provider, d.model)
                    extra.append((d.provider, d.model))
                    extra += [(d.provider, fm) for fm in self._fallback_models(d.provider)]
            except Exception:
                pass
        # last resort: every other provider that has an API key, on the model
        # set for it in Settings — when all of the route's keys are
        # rate-limited or out of credit the work goes on there instead of
        # stopping (the default provider first)
        listed = {p for p, _m in attempts + extra}
        try:
            first = self.config._default_route().provider
        except Exception:
            first = ""
        for prov in sorted(keyed, key=lambda q: q != first):
            if prov in listed:
                continue
            try:
                mdl = self.config.provider_model(prov)
            except Exception:
                mdl = ""
            if mdl:
                extra.append((prov, mdl))
                extra += [(prov, fm) for fm in self._fallback_models(prov)]
                listed.add(prov)
        out: list[tuple[str, str]] = []
        for t in attempts + extra:
            if t not in out:
                out.append(t)
        if not out:
            raise ProviderError(
                en="No AI route is set up for this feature and no provider has an API key.",
                km="មិនទាន់មានផ្លូវ AI សម្រាប់មុខងារនេះ ហើយគ្មានអ្នកផ្តល់សេវាណាមានសោ API ទេ។",
                provider="", status_code=None, retryable=False)
        return out

    def _route_targets(self, feature: str | None, provider: str | None,
                       model: str | None) -> list[tuple[str, str]]:
        """Build the ordered (provider, model) attempt list from the route."""
        attempts: list[tuple[str, str]] = []
        if provider:
            mdl = model or self.config.get(f"provider.{provider}.model")
            if not mdl and feature:
                try:
                    route = self.config.get_route(feature)
                    if route and route.provider == provider:
                        mdl = route.model
                except Exception:
                    mdl = None
            if not mdl:
                raise ProviderError(
                    en=f"No model specified for provider '{provider}'.",
                    km=f"មិនទាន់បានបញ្ជាក់ម៉ូដែលសម្រាប់អ្នកផ្តល់សេវា '{provider}' ទេ។",
                    provider=provider, status_code=None, retryable=False,
                )
            attempts.append((provider, mdl))
            return attempts
        if feature:
            route = self.config.get_route(feature)
            if route and route.provider and route.model:
                attempts.append((route.provider, route.model))
            if route and route.fallback_provider and route.fallback_model:
                attempts.append((route.fallback_provider, route.fallback_model))
        return attempts        # may be empty: _resolve_targets adds fallbacks

    def _call_with_retries(self, provider: str, key: str, model: str,
                           messages: list[dict], system: str | None,
                           temperature: float, max_tokens: int, json_mode: bool,
                           cancel: threading.Event | None,
                           images: list | None = None) -> tuple[str, int, int]:
        """Call one provider with one key, retrying retryable failures.

        Backoff 1s/2s/4s, max 3 tries.  Waits honor the cancel event via
        ``event.wait(delay)``.  Raises ProviderError after reporting the
        failure to the key pool; reports success on a good call.
        """
        pool = self.config.key_pool
        waiter = cancel if cancel is not None else threading.Event()
        last_exc: ProviderError | None = None
        for attempt in range(_MAX_TRIES_PER_KEY):
            _check_cancel(cancel)
            started = time.monotonic()
            try:
                result = self._dispatch_chat(provider, key, model, messages,
                                             system, temperature, max_tokens,
                                             json_mode, cancel,
                                             **({"images": images} if images else {}))
            except ProviderError as exc:
                last_exc = exc
                log.warning("chat %s/%s key=%s attempt=%d failed: %s",
                            provider, model, _mask_key(key), attempt + 1, exc.en)
                if getattr(exc, "timeout", False):
                    # the model hangs, not the key: no key cooldown and no
                    # second long wait here — chat() moves on or retries once
                    exc.model_problem = "timeout"
                    raise
                low = exc.en.lower()
                if exc.status_code == 404 and any(m in low for m in _RETIRED_MARKERS):
                    exc.model_problem = "retired"
                    raise
                if (provider == "gemini" and exc.status_code == 429
                        and any(m in low for m in _QUOTA_MARKERS)):
                    wait = float(getattr(exc, "retry_after", 0) or 0)
                    if (getattr(exc, "quota_scope", "") != "day" and 0 < wait <= 65
                            and attempt < _MAX_TRIES_PER_KEY - 1):
                        # per-minute limit: wait it out on the same key
                        log.info("gemini per-minute limit on %s; waiting %.0fs", model, wait)
                        if waiter.wait(wait + 1):
                            raise CancelledError("operation cancelled")
                        continue
                    # daily quota (per model): the key still works for other models
                    exc.model_problem = "quota"
                    raise
                if not exc.retryable:
                    pool.report_failure(provider, key, exc.status_code, exc.en)
                    raise
                if attempt < _MAX_TRIES_PER_KEY - 1:
                    delay = _BACKOFF_DELAYS[attempt]
                    if waiter.wait(delay):  # set -> cancelled
                        raise CancelledError("operation cancelled")
                    continue
                if exc.status_code in (503, 529) or any(m in low for m in _BUSY_MARKERS):
                    # the model is overloaded ("high demand"), not the key: no
                    # key cooldown, and chat() goes to another model if it can
                    # (every key used to get three tries first: minutes)
                    exc.model_problem = "busy"
                    raise
                pool.report_failure(provider, key, exc.status_code, exc.en)
                raise
            latency_ms = int((time.monotonic() - started) * 1000)
            pool.report_success(provider, key, latency_ms)
            log.debug("chat %s/%s ok in %dms (key=%s)", provider, model,
                      latency_ms, _mask_key(key))
            return result
        # unreachable, but keeps type-checkers honest
        assert last_exc is not None
        raise last_exc

    # -- picture editing (a poster's title redesigned by an image model) -------

    # Gemini's picture models, cheapest first; their names change (2.5 Flash
    # Image, 3.1 Flash Image, 3 Pro Image): the key's own model list decides
    _IMAGE_PREFERENCE = ("gemini-3.1-flash-image", "gemini-2.5-flash-image",
                         "gemini-3-flash-image", "gemini-3-pro-image")
    _IMAGE_FALLBACK = ("gemini-2.5-flash-image", "gemini-3.1-flash-image-preview",
                       "gemini-3-pro-image-preview")
    _image_model_cache: dict = {}

    @classmethod
    def _image_rank(cls, name: str) -> tuple[int, str]:
        low = name.lower().split("/")[-1]
        for i, pref in enumerate(cls._IMAGE_PREFERENCE):
            if low.startswith(pref):
                return (i, low)
        return (len(cls._IMAGE_PREFERENCE), low)

    def image_models(self) -> list[tuple[str, str]]:
        """(provider, model) that can edit a picture, cheapest first: the
        Gemini image models the Gemini key lists, then the same models
        through OpenRouter. Setting poster.image_model ("gemini/<model>" or
        "openrouter/<model>") puts that one first."""
        out: list[tuple[str, str]] = []
        for prov in ("gemini", "openrouter"):
            if not self.config.get_keys(prov):
                continue
            names = AIEngine._image_model_cache.get(prov)
            if names is None:
                try:
                    listed = self.list_models(prov)
                except Exception as exc:
                    log.info("could not list %s models for picture editing: %s", prov, exc)
                    listed = []
                if prov == "gemini":
                    names = [m for m in listed if "image" in m.lower()
                             and "imagen" not in m.lower() and "gemini" in m.lower()]
                    names = names or list(self._IMAGE_FALLBACK)
                else:
                    names = [m for m in listed if m.lower().startswith("google/gemini")
                             and "image" in m.lower()]
                    names = names or ["google/gemini-2.5-flash-image"]
                names = sorted(set(names), key=self._image_rank)
                AIEngine._image_model_cache[prov] = names
            out += [(prov, m) for m in names]
        wanted = str(self.config.get("poster.image_model") or "").strip()
        if "/" in wanted:
            prov, mdl = wanted.split("/", 1)
            out = [(prov, mdl)] + [t for t in out if t != (prov, mdl)]
        return out

    def edit_image(self, prompt: str, image: tuple[str, bytes], *, aspect_ratio: str = "",
                   cancel: threading.Event | None = None) -> ImageResult:
        """`image` (mime, bytes) edited as `prompt` says by a Gemini image
        model; tried model by model and key by key (image_models()).
        Google gives free keys no picture quota ("limit: 0"): such a key is
        not tried again for the other image models, and a refusal about
        pictures does not cool the key down for text. Raises ProviderError
        with what each model said when none made a picture."""
        self.check_budget()
        _check_cancel(cancel)
        targets = self.image_models()
        if not targets:
            raise ProviderError(
                en="Editing a picture needs a Gemini or OpenRouter key (Settings → API keys).",
                km="ការកែរូបភាពត្រូវការ Gemini ឬ OpenRouter key (ការកំណត់ → API keys)។",
                provider="gemini", status_code=401, retryable=False)
        pool = self.config.key_pool
        no_quota: set[tuple[str, str]] = set()     # (provider, key) without picture quota
        errors: list[str] = []
        last: ProviderError | None = None
        for prov, mdl in targets:
            tried: set[str] = set()
            for _ in range(max(1, len(self.config.get_keys(prov)))):
                _check_cancel(cancel)
                try:
                    key = pool.acquire(prov)
                except ProviderError as exc:
                    last = exc
                    break
                if key in tried:
                    break
                tried.add(key)
                if (prov, key) in no_quota:
                    continue
                try:
                    if prov == "gemini":
                        data, mime, note, in_t, out_t = self._edit_image_gemini(
                            key, mdl, prompt, image, aspect_ratio, cancel)
                    else:
                        data, mime, note, in_t, out_t = self._edit_image_openrouter(
                            key, mdl, prompt, image, aspect_ratio, cancel)
                except CancelledError:
                    raise
                except ProviderError as exc:
                    last = exc
                    low = exc.en.lower()
                    errors.append(f"{prov}/{mdl}: {exc.en[:160]}")
                    if exc.status_code == 429 and ("limit: 0" in low or "free_tier" in low
                                                   or "free tier" in low):
                        no_quota.add((prov, key))
                        continue            # this key's project has no picture quota
                    if exc.status_code in (401, 403) and not getattr(exc, "picture", False):
                        pool.report_failure(prov, key, exc.status_code, exc.en)
                        continue
                    if exc.status_code == 402:
                        pool.report_failure(prov, key, exc.status_code, exc.en)
                        continue
                    if exc.status_code == 429 or exc.retryable:
                        continue            # another key may get through
                    break                   # the model refused the request: next model
                try:
                    self.config.record_usage(prov, mdl, in_t, out_t)
                except Exception:
                    log.exception("record_usage failed")
                log.info("picture edited by %s/%s", prov, mdl)
                return ImageResult(data, mime, prov, mdl, note)
        if no_quota and len(no_quota) >= len(errors) // 2:
            raise ProviderError(
                en=("No key can edit pictures: Google gives free Gemini keys no picture "
                    "quota (\"limit: 0\"). It needs a Gemini key with billing turned on "
                    "(about US$0.04 a picture) or OpenRouter credit."),
                km=("គ្មាន key ណាអាចកែរូបភាពបានទេ៖ Google មិនផ្ដល់កូតាកែរូបភាពឲ្យ Gemini "
                    "key ឥតគិតថ្លៃទេ (\"limit: 0\")។ ត្រូវការ Gemini key ដែលបើកការបង់ប្រាក់ "
                    "(ប្រហែល US$0.04 ក្នុងមួយរូប) ឬទឹកប្រាក់ OpenRouter។"),
                provider="gemini", status_code=429, retryable=False)
        detail = "; ".join(errors[-3:]) or (last.en if last else "no model answered")
        raise ProviderError(
            en=f"No AI model could edit the picture: {detail}",
            km=f"គ្មានម៉ូដែល AI ណាអាចកែរូបភាពបានទេ៖ {detail}",
            provider=targets[0][0],
            status_code=getattr(last, "status_code", None), retryable=False)

    def _edit_image_gemini(self, key: str, model: str, prompt: str,
                           image: tuple[str, bytes], aspect_ratio: str,
                           cancel: threading.Event | None):
        """One generateContent call with the picture: (bytes, mime, words,
        in tokens, out tokens)."""
        mime, raw = image
        url = f"{self._base_url('gemini')}/models/{model}:generateContent"
        gen: dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
        if aspect_ratio:
            gen["imageConfig"] = {"aspectRatio": aspect_ratio}
        payload = {"contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": mime, "data": _b64(raw)}}, {"text": prompt}]}],
            "generationConfig": gen}
        kw = dict(provider="gemini", params={"key": key},
                  headers={"Content-Type": "application/json"}, cancel=cancel,
                  label=f"gemini/{model}")
        try:
            data = self._http("POST", url, payload=payload, **kw)
        except ProviderError as exc:
            if not (aspect_ratio and exc.status_code == 400
                    and ("imageconfig" in exc.en.lower() or "aspect" in exc.en.lower())):
                raise
            gen.pop("imageConfig", None)         # a model that takes no aspect ratio
            data = self._http("POST", url, payload=payload, **kw)
        cands = data.get("candidates") or []
        parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
        words = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            blob = part.get("inlineData") or part.get("inline_data")
            if isinstance(blob, dict) and blob.get("data"):
                meta = data.get("usageMetadata") or {}
                return (base64.b64decode(blob["data"]),
                        str(blob.get("mimeType") or blob.get("mime_type") or "image/png"),
                        " ".join(words).strip(), int(meta.get("promptTokenCount") or 0),
                        int(meta.get("candidatesTokenCount") or 0))
            if part.get("text") and not part.get("thought"):
                words.append(str(part["text"]))
        why = (cands[0].get("finishReason") if cands else
               (data.get("promptFeedback") or {}).get("blockReason")) or "no picture"
        exc = ProviderError(
            en=f"gemini {model} sent no picture back ({why}). {' '.join(words)[:200]}",
            km=f"gemini {model} មិនបានផ្ញើរូបភាពមកវិញទេ ({why})។",
            provider="gemini", status_code=400, retryable=False)
        exc.picture = True
        raise exc

    def _edit_image_openrouter(self, key: str, model: str, prompt: str,
                               image: tuple[str, bytes], aspect_ratio: str,
                               cancel: threading.Event | None):
        """The same through OpenRouter (modalities image + text)."""
        mime, raw = image
        url = self._base_url("openrouter") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": model, "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(raw)}"}}]}]}
        if aspect_ratio:
            payload["image_config"] = {"aspect_ratio": aspect_ratio}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        headers.update(_OPENROUTER_HEADERS)
        data = self._http("POST", url, provider="openrouter", payload=payload, headers=headers,
                          cancel=cancel, label=f"openrouter/{model}")
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _reply_error("openrouter", data) from exc
        for img in msg.get("images") or []:
            url_ = ((img or {}).get("image_url") or {}).get("url") or ""
            if url_.startswith("data:") and "," in url_:
                head, b64data = url_.split(",", 1)
                usage = data.get("usage") or {}
                return (base64.b64decode(b64data), head[5:].split(";")[0] or "image/png",
                        str(msg.get("content") or ""), int(usage.get("prompt_tokens") or 0),
                        int(usage.get("completion_tokens") or 0))
        exc = ProviderError(
            en=f"openrouter {model} sent no picture back. {str(msg.get('content') or '')[:200]}",
            km=f"openrouter {model} មិនបានផ្ញើរូបភាពមកវិញទេ។",
            provider="openrouter", status_code=400, retryable=False)
        exc.picture = True
        raise exc

    # -- model listing -------------------------------------------------------

    def list_models(self, provider: str, key: str | None = None) -> list[str]:
        if key is None:
            key = self.config.key_pool.acquire(provider)
        return self._list_models_impl(provider, key)

    def _list_models_impl(self, provider: str, key: str) -> list[str]:
        base = self._base_url(provider)
        if provider in _OPENAI_COMPATIBLE:
            data = self._http("GET", base + "/models", provider=provider,
                              headers={"Authorization": f"Bearer {key}"})
            items = data.get("data") or []
            if provider == "openrouter":
                self._save_openrouter_prices(items)
            return sorted(str(m.get("id")) for m in items
                          if isinstance(m, dict) and m.get("id"))
        if provider == "gemini":
            models: list[str] = []
            page_token: str | None = None
            for _ in range(20):  # hard page cap
                params: dict[str, Any] = {"key": key, "pageSize": 1000}
                if page_token:
                    params["pageToken"] = page_token
                data = self._http("GET", base + "/models", provider=provider,
                                  params=params)
                for m in data.get("models") or []:
                    if not isinstance(m, dict):
                        continue
                    methods = m.get("supportedGenerationMethods") or []
                    if "generateContent" not in methods:
                        continue
                    name = str(m.get("name") or "")
                    if name.startswith("models/"):
                        name = name[len("models/"):]
                    if name:
                        models.append(name)
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
            return sorted(set(models))
        if provider == "anthropic":
            data = self._http(
                "GET", base + "/models", provider=provider,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )
            items = data.get("data") or []
            return sorted(str(m.get("id")) for m in items
                          if isinstance(m, dict) and m.get("id"))
        raise ProviderError(
            en=f"Unknown provider '{provider}'.",
            km=f"មិនស្គាល់អ្នកផ្តល់សេវា '{provider}' ទេ។",
            provider=provider, status_code=None, retryable=False,
        )

    def _save_openrouter_prices(self, items: list) -> None:
        """OpenRouter lists each model's price ($ per token, as strings):
        keep them as $ per 1M tokens in the "prices" setting so usage and the
        monthly budget use OpenRouter's real prices."""
        found: dict[str, list[float]] = {}
        for m in items:
            try:
                pr = m.get("pricing") or {}
                found[str(m["id"]).lower()] = [round(float(pr["prompt"]) * 1e6, 4),
                                               round(float(pr["completion"]) * 1e6, 4)]
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        if not found:
            return
        try:
            stored = self.config.get("prices") or {}
            stored = dict(stored) if isinstance(stored, dict) else {}
            stored.update(found)
            self.config.set("prices", stored)
            log.info("saved OpenRouter prices for %d models", len(found))
        except Exception:
            log.exception("could not save OpenRouter prices")

    def _check_openrouter_key(self, key: str) -> str:
        """OpenRouter's model list needs no key, so check the key itself;
        returns a short credit summary."""
        data = self._http("GET", self._base_url("openrouter") + "/key",
                          provider="openrouter",
                          headers={"Authorization": f"Bearer {key}"})
        d = data.get("data") or {}
        left = d.get("limit_remaining")
        used = d.get("usage")
        parts = []
        if isinstance(used, (int, float)):
            parts.append(f"used ${used:.2f}")
        if isinstance(left, (int, float)):
            parts.append(f"${left:.2f} left")
        return "OK" + (f" ({', '.join(parts)})" if parts else "")

    # -- key checking --------------------------------------------------------

    def check_key(self, provider: str, key: str) -> KeyCheckResult:
        """One minimal call (model listing) measuring latency; parses quota hints."""
        started = time.monotonic()
        try:
            if provider == "openrouter":
                msg = self._check_openrouter_key(key)
                latency_ms = int((time.monotonic() - started) * 1000)
                return KeyCheckResult(key=key, valid=True, latency_ms=latency_ms,
                                      message=msg, quota="")
            models = self._list_models_impl(provider, key)
            latency_ms = int((time.monotonic() - started) * 1000)
            return KeyCheckResult(key=key, valid=True, latency_ms=latency_ms,
                                  message=f"OK ({len(models)} models)", quota="")
        except ProviderError as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            status = exc.status_code
            message = f"HTTP {status}: {exc.en}" if status else exc.en
            return KeyCheckResult(key=key, valid=False, latency_ms=latency_ms,
                                  message=message, quota=_quota_hint(exc.en))

    @staticmethod
    def _status_from_check(res: KeyCheckResult) -> str:
        if res.valid:
            return "valid"
        if res.quota:
            return "exhausted"
        low = res.message.lower()
        if "http 429" in low or "rate limit" in low or "rate_limit" in low:
            return "cooldown"
        if ("http 401" in low or "http 403" in low or "unauthorized" in low
                or "invalid api key" in low or "invalid x-api-key" in low
                or "api key not valid" in low or "authentication" in low
                or "permission" in low):
            return "invalid"
        return "unknown"

    def check_keys_live(self, provider: str,
                        progress: Callable[[float, str], None] | None = None,
                        cancel: threading.Event | None = None) -> list[KeyCheckResult]:
        infos = self.config.get_keys(provider)
        keys = [k.key for k in infos]
        results: list[KeyCheckResult] = []
        if not keys:
            if progress:
                progress(1.0, "No keys to check")
            return results
        _check_cancel(cancel)
        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self.check_key, provider, k): k for k in keys}
            try:
                for fut in as_completed(futures):
                    if cancel is not None and cancel.is_set():
                        for f in futures:
                            f.cancel()
                        raise CancelledError("key check cancelled")
                    key = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # defensive: check_key should not raise
                        res = KeyCheckResult(key=key, valid=False, latency_ms=0,
                                             message=str(exc), quota="")
                    results.append(res)
                    status = self._status_from_check(res)
                    try:
                        self.config.set_key_status(
                            provider, key, status,
                            latency_ms=res.latency_ms if res.valid else None,
                            error="" if res.valid else res.message)
                    except Exception:
                        log.exception("set_key_status failed")
                    done += 1
                    if progress:
                        progress(done / len(keys),
                                 f"{_mask_key(key)}: {status} ({res.latency_ms} ms)")
            except CancelledError:
                raise
        order = {k: i for i, k in enumerate(keys)}
        results.sort(key=lambda r: order.get(r.key, 0))
        return results

    # -- two-pass translation ------------------------------------------------

    def translate_two_pass(self, texts: list[str], target_lang: str,
                           style: str = "", context: str = "",
                           progress: Callable[[float, str], None] | None = None,
                           cancel: threading.Event | None = None,
                           kind: str = "narration") -> list[str]:
        """`kind`: "narration" — recap voice-over, polished into cinematic
        storytelling; "dialogue" — what people say (subtitles): the polish
        sees each original line too and only makes the draft natural spoken
        language, nothing summarised, added or left out."""
        if not texts:
            return []
        lang = _lang_name(target_lang)
        # Only send non-empty lines; re-insert blanks afterwards.
        idx_map = [i for i, t in enumerate(texts) if str(t).strip()]
        work = [str(texts[i]).strip() for i in idx_map]
        out = [str(t) for t in texts]
        if not work:
            return out

        batches = [work[i:i + _BATCH_SIZE] for i in range(0, len(work), _BATCH_SIZE)]
        total_steps = len(batches) * 2
        step = 0

        def report(text: str) -> None:
            if progress:
                progress(min(step / total_steps, 1.0), text)

        # ---- Pass 1: draft ----
        draft: list[str] = []
        for bi, batch in enumerate(batches):
            _check_cancel(cancel)
            report(f"Translating (draft) batch {bi + 1}/{len(batches)}")
            draft.extend(self._translate_batch(
                batch, lang, style, context, feature="translate_draft",
                cancel=cancel))
            step += 1
        report("Draft translation done")

        # ---- Pass 2: polish ----
        polished: list[str] = []
        draft_batches = [draft[i:i + _BATCH_SIZE] for i in range(0, len(draft), _BATCH_SIZE)]
        for bi, batch in enumerate(draft_batches):
            _check_cancel(cancel)
            report(f"Polishing batch {bi + 1}/{len(draft_batches)}")
            polished.extend(self._polish_batch(
                batch, target_lang, lang, style, context, cancel=cancel, kind=kind,
                sources=batches[bi]))
            step += 1
        report("Polish done")

        for pos, translated in zip(idx_map, polished):
            out[pos] = translated
        if progress:
            progress(1.0, "Translation complete")
        return out

    # translation internals ---------------------------------------------------

    @staticmethod
    def _numbered(lines: list[str]) -> str:
        return "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))

    def _translate_batch(self, batch: list[str], lang: str, style: str,
                         context: str, feature: str,
                         cancel: threading.Event | None) -> list[str]:
        n = len(batch)
        extras = []
        if context:
            extras.append(f"Story context: {context}")
        if style:
            extras.append(f"Style guidance: {style}")
        extra_txt = ("\n" + "\n".join(extras)) if extras else ""
        system = ("You are a professional film/drama subtitle translator. "
                  "You translate faithfully, naturally, and concisely.")
        prompt = (
            f"Translate the following {n} numbered lines into {lang}.\n"
            "Rules:\n"
            f"- Return EXACTLY {n} lines, each formatted as \"N. translated text\" "
            "using the same numbers 1..{n}.\n".replace("{n}", str(n)) +
            "- Never merge, split, add, omit, or reorder lines.\n"
            "- Translate every sentence of a line in full: never summarise or shorten, "
            "and keep repeated words, names and short exclamations.\n"
            "- Keep names and numbers accurate. No explanations, no notes, no markdown."
            f"{extra_txt}\n\nLines:\n{self._numbered(batch)}"
        )
        return self._numbered_call(feature, system, prompt, batch, lang, cancel)

    def _polish_batch(self, batch: list[str], target_lang: str, lang: str,
                      style: str, context: str,
                      cancel: threading.Event | None, kind: str = "narration",
                      sources: list[str] | None = None) -> list[str]:
        n = len(batch)
        if kind == "dialogue":
            return self._polish_dialogue(batch, target_lang, lang, style, context,
                                         cancel, sources)
        if (target_lang or "").lower() == "km":
            system = (
                "You are a veteran Khmer movie-recap narrator and script editor. "
                "You polish draft Khmer translations into vivid, natural spoken Khmer "
                "for cinematic narration: use authentic Khmer idioms and expressions "
                "(សំនួនវោហារខ្មែរ), smooth storytelling rhythm suitable for voice-over, "
                "natural word order, and emotionally engaging phrasing. Keep the "
                "meaning intact; never invent new plot details."
            )
            task = (f"Polish these {n} numbered draft Khmer lines into fluent, "
                    "cinematic spoken Khmer narration.")
        else:
            system = (
                f"You are an expert {lang} editor. You polish draft translations "
                f"into fluent, natural, engaging {lang} with a cinematic narration "
                "tone, preserving meaning exactly."
            )
            task = (f"Polish these {n} numbered draft {lang} lines for fluency "
                    "and cinematic tone.")
        extras = []
        if context:
            extras.append(f"Story context: {context}")
        if style:
            extras.append(f"Style guidance: {style}")
        extra_txt = ("\n" + "\n".join(extras)) if extras else ""
        prompt = (
            f"{task}\n"
            "Rules:\n"
            f"- Return EXACTLY {n} lines, each formatted as \"N. polished text\" "
            "with the same numbers.\n"
            "- Never merge, split, add, omit, or reorder lines.\n"
            "- No explanations, no notes, no markdown."
            f"{extra_txt}\n\nLines:\n{self._numbered(batch)}"
        )
        return self._numbered_call("translate_polish", system, prompt, batch,
                                   lang, cancel)

    def _polish_dialogue(self, batch: list[str], target_lang: str, lang: str,
                         style: str, context: str, cancel: threading.Event | None,
                         sources: list[str] | None) -> list[str]:
        """Polish of spoken lines (subtitles): the narration polish ("vivid
        cinematic narration, Khmer idioms") rewrote dialogue as a storyteller
        would and saw only the drafts, so a meaning lost in the draft stayed
        lost. This one compares each draft with its original."""
        n = len(batch)
        km = (target_lang or "").lower() == "km"
        system = (
            f"You are a senior {lang} subtitle editor for films and dramas. You get "
            "each original line and its draft translation and return the final "
            f"line: natural, everyday spoken {lang} that sounds like real people "
            "talking, with exactly the original's meaning, tone and register "
            "(casual stays casual, rude stays rude, a question stays a question). "
            "Fix mistranslations and anything the draft left out; if the draft is "
            "already right, return it unchanged."
            + (" Use colloquial Khmer, not literary or news style; pronouns and "
               "politeness words (បាទ/ចាស) that fit the speakers; Khmer script "
               "only; correct spelling (ឮ \"hear\", not លឺ)." if km else ""))
        extras = []
        if context:
            extras.append(f"Story context: {context}")
        if style:
            extras.append(f"Style guidance: {style}")
        extra_txt = ("\n" + "\n".join(extras)) if extras else ""
        src = list(sources or [])
        items = [f"ORIGINAL: {src[k] if k < len(src) else ''} || DRAFT: {t}"
                 for k, t in enumerate(batch)]
        prompt = (
            f"Return the final {lang} version of these {n} numbered lines.\n"
            "Rules:\n"
            f"- Return EXACTLY {n} lines, each formatted as \"N. final line\" with the "
            "same numbers — only the final line, no ORIGINAL / DRAFT labels.\n"
            "- Never merge, split, add, omit, or reorder lines.\n"
            "- Everything the original says stays: never summarise or shorten, keep "
            "repeated words, names, numbers and short exclamations; add nothing the "
            "original does not say.\n"
            "- No explanations, no notes, no markdown."
            f"{extra_txt}\n\nLines:\n{self._numbered(items)}"
        )
        out = self._numbered_call("translate_polish", system, prompt, items, lang, cancel)
        # a reply that echoed the labels keeps only the final words; an empty
        # one keeps the draft
        final = []
        for t, draft in zip(out, batch):
            t = re.sub(r"^.*\|\|\s*", "", str(t or "")).strip()
            t = re.sub(r"^\s*(?:DRAFT|FINAL)\s*:\s*", "", t, flags=re.IGNORECASE).strip()
            final.append(t or draft)
        return final

    def _numbered_call(self, feature: str, system: str, prompt: str,
                       batch: list[str], lang: str,
                       cancel: threading.Event | None) -> list[str]:
        """Chat with strict numbered-list protocol.

        Retries once on count mismatch, then falls back to per-line calls.
        """
        n = len(batch)
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            _check_cancel(cancel)
            result = self.chat(feature=feature, messages=messages, system=system,
                               temperature=0.3 if attempt == 0 else 0.1,
                               max_tokens=4096, cancel=cancel)
            parsed = _parse_numbered_list(result.text, n)
            if parsed is not None:
                # a reply numbering its lines twice ("1. 1. text") would
                # otherwise keep the extra number in the translation
                from app.core.translation_memory import clean_target
                return [clean_target(src, t) or t for src, t in zip(batch, parsed)]
            log.warning("numbered-list mismatch (%s, attempt %d, expected %d lines)",
                        feature, attempt + 1, n)
        # Per-line fallback.
        log.warning("falling back to per-line translation for %d lines (%s)",
                    n, feature)
        out: list[str] = []
        for line in batch:
            _check_cancel(cancel)
            single = (
                f"Translate or rewrite this single line into natural {lang}. "
                "Return ONLY the resulting line, with no numbering, quotes, "
                f"notes, or explanations.\n\nLine: {line}"
            )
            try:
                res = self.chat(feature=feature,
                                messages=[{"role": "user", "content": single}],
                                system=system, temperature=0.3, max_tokens=1024,
                                cancel=cancel)
                # every line of the reply but notes: a translation the model
                # broke over two lines used to lose its second half
                kept = [x.strip() for x in _strip_code_fences(res.text).strip().splitlines()
                        if x.strip() and not re.match(
                            r"\s*(?:\(|\*|#|note\b|explanation\b|translation\s*:)",
                            x, re.IGNORECASE)]
                out.append(" ".join(kept) if kept else line)
            except CancelledError:
                raise
            except ProviderError as exc:
                log.error("per-line fallback failed: %s", exc.en)
                out.append(line)  # keep the original rather than losing the row
        return out


# ---------------------------------------------------------------------------
# Offline unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0

    def check(name: str, cond: bool) -> None:
        global failures
        if cond:
            print(f"  ok   {name}")
        else:
            failures += 1
            print(f"  FAIL {name}")

    print("extract_json:")
    check("plain object", extract_json('{"a": 1}') == {"a": 1})
    check("plain array", extract_json('[1, 2, 3]') == [1, 2, 3])
    check("fenced json", extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]})
    check("bare fence", extract_json('```\n{"x": true}\n```') == {"x": True})
    check("embedded in prose",
          extract_json('Sure! Here it is: {"k": "v", "n": [1, {"m": 2}]} hope it helps')
          == {"k": "v", "n": [1, {"m": 2}]})
    check("braces inside strings",
          extract_json('noise {"s": "a { tricky ] string", "b": 2} tail')
          == {"s": "a { tricky ] string", "b": 2})
    check("escaped quotes",
          extract_json('{"s": "he said \\"hi\\""}') == {"s": 'he said "hi"'})
    check("array embedded", extract_json("answer: [\"a\", \"b\"] done") == ["a", "b"])
    check("trailing comma salvage", extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2})
    check("unterminated fence", extract_json('```json\n{"ok": 1}') == {"ok": 1})
    try:
        extract_json("no json here at all")
        check("raises on garbage", False)
    except ValueError:
        check("raises on garbage", True)

    print("numbered-list parser:")
    check("simple",
          _parse_numbered_list("1. hello\n2. world", 2) == ["hello", "world"])
    check("paren style",
          _parse_numbered_list("1) one\n2) two\n3) three", 3) == ["one", "two", "three"])
    check("colon style",
          _parse_numbered_list("1: a\n2: b", 2) == ["a", "b"])
    check("continuation lines",
          _parse_numbered_list("1. first part\ncontinues here\n\n2. second", 2)
          == ["first part continues here", "second"])
    check("fenced output",
          _parse_numbered_list("```\n1. a\n2. b\n```", 2) == ["a", "b"])
    check("khmer text",
          _parse_numbered_list("1. សួស្តី​ពិភពលោក\n2. រឿងភាគចិន", 2)
          == ["សួស្តី​ពិភពលោក", "រឿងភាគចិន"])
    check("count mismatch returns None",
          _parse_numbered_list("1. only one", 2) is None)
    check("wrong numbering returns None",
          _parse_numbered_list("1. a\n3. b", 2) is None)
    check("extra lines returns None",
          _parse_numbered_list("1. a\n2. b\n3. c", 2) is None)
    check("out-of-order accepted",
          _parse_numbered_list("2. b\n1. a", 2) == ["a", "b"])

    print("cost table:")
    check("default fallback", estimate_cost("totally-unknown-model", 1_000_000, 0) == 0.5)
    check("known model", estimate_cost("claude-opus-5", 1_000_000, 1_000_000) == 30.0)

    print("key / provider switching:")
    import pathlib
    import tempfile
    from app.core.config_manager import ConfigManager, Route
    _BACKOFF_DELAYS = (0.0, 0.0, 0.0)          # no real waits in the test
    _MAX_KEY_WAIT_S = 0
    cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "t.db")
    cfg.set_route("translate_draft", Route(provider="openrouter",
                                           model="google/gemini-2.5-flash"))
    cfg.add_keys("openrouter", ["or-key-A", "or-key-B"])
    eng = AIEngine(cfg)
    calls: list[tuple[str, str]] = []
    limited: set[str] = set()

    def fake(provider, key, model, *a, **k):
        calls.append((provider, key))
        if key in limited:
            raise ProviderError("HTTP 429: Rate limit exceeded", provider=provider,
                                status_code=429, retryable=True)
        return f"answer from {key}", 1, 1

    eng._dispatch_chat = fake
    msg = [{"role": "user", "content": "hi"}]
    limited = {"or-key-A"}
    r = eng.chat(feature="translate_draft", messages=msg)
    check("a rate-limited key hands over to the next key of the same provider",
          r.text == "answer from or-key-B")
    limited = {"or-key-A", "or-key-B"}
    try:
        eng.chat(feature="translate_draft", messages=msg)
        check("no other provider: the limit is reported", False)
    except ProviderError as exc:
        check("no other provider: the limit is reported (and stops dub passes)",
              ai_unavailable(exc))
    cfg.add_keys("gemini", ["gm-key"])
    calls.clear()
    t0 = time.monotonic()
    r = eng.chat(feature="translate_draft", messages=msg)
    check("all OpenRouter keys limited: switches to the Gemini key, with no fallback set",
          r.provider == "gemini" and r.text == "answer from gm-key")
    check("…at once, without waiting out the cooldown",
          time.monotonic() - t0 < 5 and all(p == "gemini" for p, _k in calls))
    check("a refused request is not 'AI unavailable'",
          not ai_unavailable(ProviderError("HTTP 400: prompt too long",
                                           provider="openrouter", status_code=400)))
    check("out of credit / budget / offline are 'AI unavailable'",
          ai_unavailable(ProviderError("HTTP 402", status_code=402))
          and ai_unavailable(ProviderError("budget", provider="budget"))
          and not ai_unavailable(ValueError("bad json")))
    print("replies without an answer:")
    e429 = _reply_error("openrouter", {"error": {"code": 429, "message": "Provider returned error",
                                                 "metadata": {"raw": "gemini-2.5-flash is temporarily rate-limited"}}})
    check("a rate limit inside a 200 reply is a rate limit (and says why)",
          e429.status_code == 429 and e429.retryable and "rate-limited" in e429.en)
    check("no credit inside a reply marks the key out of credit",
          _reply_error("openrouter", {"error": {"code": 402, "message": "Insufficient credits"}}).status_code == 402)
    check("a flagged request is the request's fault, not the key's",
          _is_request_error(_reply_error("openrouter", {"error": {"code": 403, "message": "Your input was flagged"}})))
    check("an unknown empty reply shows what came back",
          "no answer in the reply" in _reply_error("openrouter", {"id": "x"}).en)
    cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "t2.db")
    cfg.set_route("translate_draft", Route(provider="openrouter", model="google/gemini-2.5-flash"))
    cfg.add_keys("openrouter", ["or-key-A", "or-key-B"])
    eng = AIEngine(cfg)
    sent: list[str] = []
    replies = {}

    def post(provider, model, url, payload, cancel=None, headers=None):
        key = headers["Authorization"].split()[-1]
        sent.append(key)
        if key in replies:
            return replies[key]
        # the first key asked is rate-limited (inside a 200 reply)
        if sent.count(sent[0]) and key == sent[0]:
            return {"error": {"code": 429, "message": "Provider returned error"}}
        return {"choices": [{"message": {"content": "hello"}}]}

    eng._post_adapting = post
    r = eng.chat(feature="translate_draft", messages=msg)
    check("a key whose reply carries a rate limit hands over to the next key",
          r.text == "hello" and len(set(sent)) == 2
          and any(k.status == "cooldown" for k in cfg.get_keys("openrouter")))
    blocked = {"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}
    replies.update({"or-key-A": blocked, "or-key-B": blocked})
    cfg.add_keys("gemini", ["gm-key"])
    eng._dispatch_chat_orig = eng._dispatch_chat

    def disp(provider, key, model, *a, **k):
        if provider == "gemini":
            return "from gemini", 1, 1
        return eng._dispatch_chat_orig(provider, key, model, *a, **k)

    eng._dispatch_chat = disp
    for k in cfg.get_keys("openrouter"):
        cfg.set_key_status("openrouter", k.key, "valid")
    r = eng.chat(feature="translate_draft", messages=msg)
    check("a blocked reply tries another model, and the key stays valid",
          r.text == "from gemini"
          and all(k.status not in ("invalid", "cooldown", "exhausted")
                  for k in cfg.get_keys("openrouter")))
    cfg.close()

    print("overloaded or slow models:")
    cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "t5.db")
    cfg.set_route("translate_draft", Route(provider="gemini", model="busy-model"))
    cfg.set("provider.gemini.fallback_models", ["quick-model"])
    cfg.add_keys("gemini", ["gk-1", "gk-2", "gk-3"])
    eng = AIEngine(cfg)
    seen: list[tuple[str, str, int]] = []

    def crowded(provider, key, model, *a, **k):
        seen.append((model, key, int(getattr(eng._tls, "read_s", 0) or _HTTP_TIMEOUT[1])))
        if model in ("busy-model", "busy-alone"):
            raise ProviderError("gemini HTTP 503: This model is currently experiencing high "
                                "demand.", provider=provider, status_code=503, retryable=True)
        if model == "slow-model":
            if seen[-1][2] < 250:            # it would answer after 4 minutes
                err = ProviderError(f"gemini/{model} did not answer within {seen[-1][2]} s.",
                                    provider=provider, retryable=True)
                err.timeout, err.after = True, seen[-1][2]
                raise err
            return "slow answer", 1, 1
        return "quick answer", 1, 1

    eng._dispatch_chat = crowded
    r = eng.chat(feature="translate_draft", messages=msg)
    busy_keys = {k for m, k, _w in seen if m == "busy-model"}
    check("an overloaded model (HTTP 503 'high demand') hands over to the next model after "
          "one key's three tries (not every key's)",
          r.text == "quick answer" and len(busy_keys) == 1
          and [m for m, _k, _w in seen].count("busy-model") == 3)
    check("…and its keys are not put in cooldown",
          all(k.status != "cooldown" for k in cfg.get_keys("gemini")))
    seen.clear()
    r = eng.chat(feature="translate_draft", messages=msg)
    check("…the next request goes straight to the other model",
          r.text == "quick answer" and [m for m, _k, _w in seen] == ["quick-model"])
    cfg.set_route("translate_draft", Route(provider="gemini", model="slow-model"))
    seen.clear()
    r = eng.chat(feature="translate_draft", messages=msg)
    check(f"a slow model is waited for {_FALLBACK_READ_S} s when another model can answer, "
          "then the other one answers",
          r.text == "quick answer" and seen[0] == ("slow-model", seen[0][1], _FALLBACK_READ_S)
          and seen[-1][0] == "quick-model")
    cfg.set("provider.gemini.fallback_models", [])
    seen.clear()
    r = eng.chat(feature="translate_draft", messages=msg)
    check(f"…the only model is waited for the full {_HTTP_TIMEOUT[1]} s",
          r.text == "slow answer" and [w for _m, _k, w in seen] == [_HTTP_TIMEOUT[1]])
    cfg.set_route("translate_draft", Route(provider="gemini", model="busy-alone"))
    seen.clear()
    try:
        eng.chat(feature="translate_draft", messages=msg)
        alone = None
    except ProviderError as exc:
        alone = exc
    check("an overloaded model with nowhere else to go still gets every key",
          alone is not None and len({k for m, k, _w in seen if m == "busy-alone"}) == 3)
    cfg.close()

    print("no key / why no AI answered:")
    cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "t3.db")
    try:
        AIEngine(cfg).chat(feature="translate_draft", messages=msg)
        nk = None
    except ProviderError as exc:
        nk = exc
    check("no key anywhere: a plain message, no provider the user never chose",
          nk is not None and nk.en == NO_KEY_EN and "deepseek" not in nk.en + nk.km
          and ai_unavailable(nk) and ai_failure_kind(nk) == "no_key")
    cfg.close()

    def _kind(en, code=None, provider="openrouter"):
        return ai_failure_kind(ProviderError(en, provider=provider, status_code=code))

    check("causes are told apart",
          _kind("budget reached", provider="budget") == "budget"
          and _kind("All providers failed. Last error: openrouter HTTP 401: No auth", 401) == "bad_key"
          and _kind("No usable API keys for 'gemini': all keys are invalid or exhausted.") == "bad_key"
          and _kind("openrouter HTTP 402: Insufficient credits", 402) == "no_credit"
          and _kind("All keys for 'gemini' are out of quota; they are tried again", 429) == "no_credit"
          and _kind("gemini HTTP 429: Resource exhausted (check quota)", 429) == "busy"
          and _kind("All 2 key(s) for 'openrouter' are unavailable (rate-limited or failed).", 429) == "busy"
          and _kind("All providers failed. Last error: openrouter/x did not answer within 300 s.") == "timeout"
          and _kind("Network error calling openrouter: Max retries exceeded (NewConnectionError)") == "offline"
          and _kind("something else") == ""
          and all(k in AI_ADVICE for k in ("no_key", "bad_key", "no_credit", "busy",
                                            "timeout", "offline", "")))

    print("out of credit (OpenRouter 402):")
    cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "t4.db")
    cfg.set_route("recap", Route(provider="openrouter", model="google/gemini-2.5-flash"))
    cfg.add_keys("openrouter", ["or-key-A", "or-key-B"])
    eng = AIEngine(cfg)
    asked: list[int] = []
    afford = {"n": 5738, "finish": "stop"}

    def post402(provider, model, url, payload, cancel=None, headers=None):
        asked.append(int(payload.get("max_tokens") or 0))
        if payload["max_tokens"] > afford["n"]:
            raise ProviderError(
                "openrouter HTTP 402: This request requires more credits, or fewer "
                f"max_tokens. You requested up to {payload['max_tokens']} tokens, but can "
                f"only afford {afford['n']}. To increase, visit "
                "https://openrouter.ai/settings/credits",
                provider=provider, status_code=402, retryable=False)
        return {"choices": [{"message": {"content": '{"rows": []}'},
                             "finish_reason": afford["finish"]}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}}

    eng._post_adapting = post402
    r = eng.chat(feature="recap", messages=msg, max_tokens=8192)
    check("a balance short of max_tokens: asked once more for what it covers",
          r.text == '{"rows": []}' and len(asked) == 2 and asked[0] == 8192
          and _MIN_AFFORDABLE_TOKENS <= asked[1] <= 5738
          and all(k.status != "exhausted" for k in cfg.get_keys("openrouter")))
    asked.clear()
    afford["finish"] = "length"
    try:
        eng.chat(feature="recap", messages=msg, max_tokens=8192)
        cut = None
    except ProviderError as exc:
        cut = exc
    check("…an answer cut short by it counts as out of credit",
          cut is not None and ai_failure_kind(cut) == "no_credit" and ai_unavailable(cut)
          and all(k.status == "exhausted" for k in cfg.get_keys("openrouter")))
    check("…said plainly: no 'All providers failed', no quota, no waiting",
          cut is not None and getattr(cut, "plain", False)
          and cut.en.startswith("Your OpenRouter keys are out of credit")
          and "All providers failed" not in cut.en and "quota" not in cut.en
          and "tried again in" not in cut.en)
    afford.update(n=100, finish="stop")
    for k in cfg.get_keys("openrouter"):
        cfg.set_key_status("openrouter", k.key, "valid")
    asked.clear()
    try:
        eng.chat(feature="recap", messages=msg, max_tokens=8192)
        tiny = None
    except ProviderError as exc:
        tiny = exc
    check("a balance too small to be worth it is not asked again",
          tiny is not None and asked == [8192, 8192] and ai_failure_kind(tiny) == "no_credit")
    cfg.add_keys("gemini", ["gm-key"])
    cfg.key_pool.report_failure("gemini", "gm-key", 429,
                                "gemini HTTP 429: Quota exceeded for metric: free_tier_requests")
    try:
        eng.chat(feature="recap", messages=msg, max_tokens=8192)
        mixed = None
    except ProviderError as exc:
        mixed = exc
    check("OpenRouter out of credit, then Gemini's quota used up: the credit message first",
          mixed is not None and mixed.en.startswith("Your OpenRouter keys are out of credit")
          and "(Then Gemini: " in mixed.en and "(បន្ទាប់មក Gemini៖ " in mixed.km
          and ai_failure_kind(mixed) == "no_credit" and getattr(mixed, "plain", False))
    check("_affordable_tokens: only a 402 that names a number",
          _affordable_tokens(ProviderError("HTTP 402: Insufficient credits", status_code=402),
                             8192) == 0
          and _affordable_tokens(ProviderError("can only afford 5000", status_code=429),
                                 8192) == 0
          and _affordable_tokens(ProviderError("can only afford 9000", status_code=402),
                                 8192) == 0)
    cfg.close()

    print("translate_two_pass:")
    tp = AIEngine.__new__(AIEngine)        # only .chat is used: stubbed
    seen: list[tuple[str, str, str]] = []

    def fake_chat(*, feature=None, messages=None, system=None, **_kw):
        seen.append((feature or "", system or "", messages[0]["content"]))
        if feature == "translate_draft":
            return ChatResult("1. សួស្តី\n2. ទៅណា", 0, 0, "stub", "stub")
        # the polish echoes its labels on line 1 and leaves line 2 empty
        return ChatResult("1. ORIGINAL: Hello. || DRAFT: សួស្តីបង\n2. ", 0, 0, "stub", "stub")

    tp.chat = fake_chat
    out = tp.translate_two_pass(["Hello.", "Where to?"], "km", kind="dialogue")
    check("dialogue polish sees the originals, not the narrator prompt",
          "ORIGINAL: Hello." in seen[1][2] and "subtitle editor" in seen[1][1]
          and "narrator" not in seen[1][1] and "never summarise" in seen[0][2])
    check("dialogue polish: echoed labels stripped, an empty reply keeps the draft",
          out == ["សួស្តីបង", "ទៅណា"])
    seen.clear()
    tp.translate_two_pass(["Hello.", "Where to?"], "km")
    check("narration keeps the recap narrator polish",
          any(f == "translate_polish" and "narrator" in s for f, s, _p in seen))

    print("pictures with a chat (a poster to read):")
    pic_cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "p.db")
    pe = AIEngine(pic_cfg)
    sent: list[dict] = []
    replies = {
        "gemini": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
        "openrouter": {"choices": [{"message": {"content": "ok"}}]},
        "anthropic": {"content": [{"type": "text", "text": "ok"}]},
    }

    def fake_http(method, url, *, provider="", payload=None, **_kw):
        sent.append(payload)
        return replies[provider]

    pe._http = fake_http
    jpeg = b"\xff\xd8\xff\xe0picture"
    b64 = base64.b64encode(jpeg).decode("ascii")
    ask = [{"role": "user", "content": "Read the title."}]
    for prov in ("gemini", "openrouter", "anthropic"):
        sent.clear()
        text, _i, _o = pe._dispatch_chat(prov, "k", "m", ask, "sys", 0.3, 100, False,
                                         None, images=[("image/jpeg", jpeg)])
        p = sent[0]
        if prov == "gemini":
            parts = p["contents"][-1]["parts"]
            good = (parts[0] == {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                    and parts[1] == {"text": "Read the title."})
        elif prov == "openrouter":
            c = p["messages"][-1]["content"]
            good = (c[0] == {"type": "text", "text": "Read the title."}
                    and c[1] == {"type": "image_url",
                                 "image_url": {"url": "data:image/jpeg;base64," + b64}})
        else:
            c = p["messages"][-1]["content"]
            good = (c[0] == {"type": "image", "source": {"type": "base64",
                                                         "media_type": "image/jpeg",
                                                         "data": b64}}
                    and c[1] == {"type": "text", "text": "Read the title."})
        check(f"{prov}: the picture goes with the request, as the provider wants it",
              good and text == "ok")
        sent.clear()
        pe._dispatch_chat(prov, "k", "m", ask, "sys", 0.3, 100, False, None)
        check(f"{prov}: no picture: the request as before",
              "inline_data" not in json.dumps(sent[0]) and "image_url" not in json.dumps(sent[0])
              and '"type": "image"' not in json.dumps(sent[0]))
    pic_cfg.add_keys("gemini", ["gm-key"])
    pic_cfg.set_route("poster", Route(provider="gemini", model="gemini-flash-lite-latest"))
    got: list = []

    def fake_dispatch(provider, key, model, messages, system, temperature, max_tokens,
                      json_mode, cancel=None, images=None):
        got.append(images)
        return "{}", 1, 1

    pe._dispatch_chat = fake_dispatch
    pe.chat(feature="poster", messages=ask, images=[("image/jpeg", jpeg)])
    pe.chat(feature="poster", messages=ask)
    check("chat() hands the pictures through (and none when there are none)",
          got == [[("image/jpeg", jpeg)], None])
    pic_cfg.close()

    print("picture editing (a poster's title redesigned):")
    ed_cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "e.db")
    ed_cfg.add_keys("gemini", ["free-key", "paid-key"])
    ee = AIEngine(ed_cfg)
    AIEngine._image_model_cache.clear()
    ee.list_models = lambda prov: ["gemini-flash-lite-latest", "gemini-2.5-flash-image",
                                   "imagen-4.0-generate-001", "gemini-3.1-flash-image-preview"]
    png = b"\x89PNG\r\n\x1a\nfake"
    posted: list[tuple[str, dict, str]] = []

    def fake_edit_http(method, url, *, provider="", payload=None, params=None, headers=None,
                       **_kw):
        key = (params or {}).get("key") or (headers or {}).get("Authorization", "")
        posted.append((url, payload, key))
        if provider == "gemini" and key == "free-key":
            raise ProviderError("gemini HTTP 429: Quota exceeded for metric "
                                "generate_content_free_tier_requests, limit: 0",
                                provider="gemini", status_code=429, retryable=True)
        if provider == "gemini":
            return {"candidates": [{"content": {"parts": [
                {"text": "Here it is."},
                {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}}]}}],
                "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 1290}}
        return {"choices": [{"message": {"content": "", "images": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                + base64.b64encode(png).decode()}}]}}]}

    ee._http = fake_edit_http
    check("the image models the key lists, newest Flash first (no Imagen, no text model)",
          ee.image_models() == [("gemini", "gemini-3.1-flash-image-preview"),
                                ("gemini", "gemini-2.5-flash-image")])
    res = ee.edit_image("Replace the title.", ("image/jpeg", jpeg), aspect_ratio="16:9")
    body = [p for _u, p, k in posted if k == "paid-key"][0]
    check("a free key (no picture quota) is passed over; the paid key makes the picture",
          res.data == png and res.mime == "image/png" and res.model.startswith("gemini-3.1")
          and res.note == "Here it is.")
    check("the request: the picture, the words, picture output, the aspect ratio",
          body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/jpeg"
          and body["contents"][0]["parts"][1]["text"] == "Replace the title."
          and body["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
          and body["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"})
    free_info = [k for k in ed_cfg.get_keys("gemini") if k.key == "free-key"][0]
    check("the free key is not cooled down for text work",
          free_info.status not in ("cooldown", "exhausted", "invalid")
          and free_info.cooldown_until <= time.time())
    ed_cfg.remove_key("gemini", "paid-key")
    AIEngine._image_model_cache.clear()
    try:
        ee.edit_image("x", ("image/jpeg", jpeg))
        check("free keys only: a plain message", False)
    except ProviderError as exc:
        check("free keys only: a plain message (billing / OpenRouter credit)",
              "limit: 0" in exc.en and "billing" in exc.en)
    or_cfg = ConfigManager(pathlib.Path(tempfile.mkdtemp()) / "o.db")
    or_cfg.add_keys("openrouter", ["or-key"])
    oe = AIEngine(or_cfg)
    AIEngine._image_model_cache.clear()
    oe.list_models = lambda prov: ["google/gemini-2.5-flash", "google/gemini-2.5-flash-image",
                                   "openai/gpt-5-image"]
    oe._http = fake_edit_http
    posted.clear()
    res = oe.edit_image("Replace the title.", ("image/jpeg", jpeg))
    check("through OpenRouter: the Gemini image model, the picture from message.images",
          res.data == png and res.provider == "openrouter"
          and res.model == "google/gemini-2.5-flash-image"
          and posted[0][1]["modalities"] == ["image", "text"])
    AIEngine._image_model_cache.clear()
    ed_cfg.close()
    or_cfg.close()

    if failures:
        print(f"FAILED: {failures} test(s)")
        raise SystemExit(1)
    print("PASS")
