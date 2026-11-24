"""Tiny i18n layer: JSON locale files with graceful fallbacks (en <-> km).

Besides tr(), it can turn a text already on screen into another language
(convert()) — that is how the interface switches language without a
restart: tr() remembers which key every text it returned came from, and
catalog values with {placeholders} ("Dubbing — task {i} of {n}: {task}",
the engine's "dyn.*" progress messages "Translating lines {0}-{1} of {2}")
are matched as templates, their values converted in turn.
"""
from __future__ import annotations

import json
import re
import threading

from app.core.paths import locales_dir

_lock = threading.Lock()
_language = "en"
_catalogs: dict[str, dict[str, str]] = {}
# every text tr() returned -> (key, default): what is on screen now
_seen: dict[str, tuple[str, str | None]] = {}
_SEEN_MAX = 60000
_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_+]+)\}")
_templates: dict[str, list[tuple[re.Pattern, str, list[str]]]] = {}
_reverse: dict[str, dict[str, str]] = {}


def _load(lang: str) -> dict[str, str]:
    with _lock:
        if lang in _catalogs:
            return _catalogs[lang]
        data: dict[str, str] = {}
        path = locales_dir() / f"{lang}.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = {str(k): str(v) for k, v in raw.items()}
        except Exception:
            data = {}
        _catalogs[lang] = data
        return data


def _save_language(code: str) -> None:
    try:
        from app.core.config_manager import get_default_config
        get_default_config().set("ui.language", code)
    except Exception:
        pass


def set_language(code: str, save: bool = True) -> None:
    """Use `code` from now on; `save` also keeps it for the next start
    (tests pass save=False: a test once switched the user's setting)."""
    global _language
    if code not in ("en", "km"):
        code = "en"
    _language = code
    if save:
        _save_language(code)


def get_language() -> str:
    return _language


def pick(en: str, km: str = "") -> str:
    """One text of an English/Khmer pair (an engine or error message) for
    the UI language: the Khmer one when the UI is Khmer and there is one,
    else the English one — never both at once."""
    return km if (_language == "km" and km) else en


def reload_catalogs() -> None:
    with _lock:
        _catalogs.clear()
        _templates.clear()
        _reverse.clear()


def _value(lang: str, key: str, default: str | None = None) -> str:
    """tr() for a given language, without switching to it."""
    cat = _load(lang)
    if key in cat:
        return cat[key]
    if lang != "en":
        en = _load("en")
        if key in en:
            return en[key]
    return default if default is not None else key


def _remember(text: str, key: str, default: str | None = None) -> str:
    if text and len(_seen) < _SEEN_MAX and text not in _seen:
        _seen[text] = (key, default)
    return text


def tr(key: str, default: str | None = None) -> str:
    return _remember(_value(_language, key, default), key, default)


# ---------------------------------------------------------------------------
# converting a shown text into another language
# ---------------------------------------------------------------------------

def _reverse_index(lang: str) -> dict[str, str]:
    """value -> key of the engine's messages ("dyn.*") in one language.
    Other texts are only converted when tr() produced them (_seen): a name
    or a value that happens to equal a catalog word ("auto", "Custom") is
    data and stays as it is."""
    if lang not in _reverse:
        idx: dict[str, str] = {}
        for src in ([_load("en")] if lang != "en" else []) + [_load(lang)]:
            for k, v in src.items():
                if k.startswith("dyn.") and "{" not in v:
                    idx[v] = k
        _reverse[lang] = idx
    return _reverse[lang]


def _template_list(lang: str) -> list[tuple[re.Pattern, str, list[str]]]:
    """Catalog values with {placeholders}, as regexes over `lang`'s text
    (longest literal text first: the most specific template wins)."""
    if lang not in _templates:
        out = []
        cats = [_load(lang)] + ([_load("en")] if lang != "en" else [])
        seen_keys = set()
        for cat in cats:
            for key, val in cat.items():
                if key in seen_keys or "{" not in val:
                    continue
                parts = _PLACEHOLDER.split(val)
                literal = "".join(parts[0::2])
                if len(literal.strip()) < 3:
                    continue
                names = parts[1::2]
                rx = ""
                for i, piece in enumerate(parts):
                    rx += re.escape(piece) if i % 2 == 0 else "(.+?)"
                try:
                    out.append((re.compile(rx, re.S), key, names, len(literal)))
                    seen_keys.add(key)
                except re.error:
                    continue
        out.sort(key=lambda t: -t[3])
        _templates[lang] = [(rx, key, names) for rx, key, names, _n in out]
    return _templates[lang]


_ICON = re.compile(r"^([^\w\s«“\"'(\[]{1,3}[  ]+)(\S.*)$", re.S)
_SEPARATORS = (": ", "：", " (", " — ", "\n", " · ")


def convert(text: str, src: str, dst: str, _depth: int = 0) -> str:
    """`text` as shown in language `src`, in language `dst`. Exact texts,
    texts after an icon ("✔  Done"), templates with values ("task 6 of 10:
    Translate…", the values converted too) and a known label before ": "
    / " (" / a line break; anything else is returned unchanged."""
    if not text or src == dst or _depth > 3:
        return text
    hit = _seen.get(text)
    if hit is not None:
        return _remember(_value(dst, *hit), *hit)
    key = _reverse_index(src).get(text)
    if key is not None:
        return _remember(_value(dst, key), key)
    m = _ICON.match(text)
    if m:
        inner = convert(m.group(2), src, dst, _depth + 1)
        if inner != m.group(2):
            return m.group(1) + inner
    for rx, key, names in _template_list(src):
        m = rx.fullmatch(text)
        if not m:
            continue
        out = _value(dst, key)
        for name, val in zip(names, m.groups()):
            out = out.replace("{" + name + "}", convert(val, src, dst, _depth + 1), 1)
        return out
    for sep in _SEPARATORS:
        i = text.find(sep)
        if i > 0:
            head, tail = text[:i], text[i + len(sep):]
            new_head = _convert_part(head, src, dst, _depth) if len(head) < 160 else head
            if sep == " (":          # the bracket goes with its words: "(no keys)"
                new_tail = _convert_part("(" + tail, src, dst, _depth)[1:]
            else:
                new_tail = _convert_part(tail, src, dst, _depth)
            if new_head != head or new_tail != tail:
                return new_head + sep + new_tail
    if text[-1:] in (":", "：") and len(text) > 2:          # "Duration:"
        inner = convert(text[:-1], src, dst, _depth + 1)
        if inner != text[:-1]:
            return inner + text[-1]
    words = text.split(" ")
    for k in range(min(len(words) - 1, 6), 0, -1):                # "Cookies from" + " Firefox"
        head = " ".join(words[:k])
        if len(head) >= 4 and head in _seen:
            return _remember(_value(dst, *_seen[head]), *_seen[head]) + " " + " ".join(words[k:])
    return text


def _convert_part(part: str, src: str, dst: str, depth: int) -> str:
    """convert() a piece of a composed text, keeping its outer spaces and
    brackets: "  Translation memory: 12 (3 your fixes)", "(no keys)"."""
    core = part.strip()
    if not core:
        return part
    lead = part[:len(part) - len(part.lstrip())]
    trail = part[len(part.rstrip()):]
    new = convert(core, src, dst, depth + 1)
    if new == core and len(core) > 2 and core[0] == "(" and core[-1] == ")":
        inner = convert(core[1:-1], src, dst, depth + 1)
        if inner != core[1:-1]:
            new = "(" + inner + ")"
    return lead + new + trail


def is_ui_text(text: str) -> bool:
    """`text` came from tr() (an interface text, not a value or a name)."""
    return text in _seen


def tr_text(text: str) -> str:
    """An English message from the engine (progress, status) in the
    interface language — through the "dyn.*" templates of the catalogs."""
    if _language == "en" or not text:
        return text
    return convert(text, "en", _language)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    bad = []

    def check(name: str, got: str, want: str) -> None:
        ok = got == want
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"\n   got:  {got!r}\n   want: {want!r}"))
        if not ok:
            bad.append(name)

    saved = []
    _save_language = lambda code: saved.append(code)      # noqa: E731 (test: never the user's setting)
    set_language("en", save=False)
    done = tr("dub.run.done", "Done — the dubbed video is ready")
    running = ("⟳  " + tr("dub.run.running", "Dubbing — task {i} of {n}: {task}")
               .replace("{i}", "6").replace("{n}", "10")
               .replace("{task}", tr("dub.step.translate", "Translate dialogue with context")))
    km_done, km_task = _value("km", "dub.run.done"), _value("km", "dub.step.translate")
    check("exact text", convert(done, "en", "km"), km_done)
    check("text after an icon", convert("✔  " + done, "en", "km"), "✔  " + km_done)
    check("template with a translated value", convert(running, "en", "km"),
          "⟳  " + _value("km", "dub.run.running").replace("{i}", "6").replace("{n}", "10")
          .replace("{task}", km_task))
    tr("dub.saved_to", "Saved to")
    check("label before ': '", convert("Saved to: C:\\x\\a.mp4", "en", "km"),
          _value("km", "dub.saved_to") + ": C:\\x\\a.mp4")
    set_language("km", save=False)
    check("engine progress message", tr_text("Translating lines 41-80 of 2229"),
          "កំពុងបកប្រែជួរ 41-80 នៃ 2229")
    check("task note with a nested value", tr_text("2.0 min audio, voices and music separated"),
          "សំឡេង 2.0 នាទី, បានបំបែកសំឡេងនិយាយ និងតន្ត្រី")
    check("back to English", convert(convert(running, "en", "km"), "km", "en"), running)
    check("unknown text unchanged", tr_text("C:\\Videos\\clip.mp4"), "C:\\Videos\\clip.mp4")
    check("save=False does not touch the setting", ",".join(saved), "")
    print("PASS" if not bad else f"FAILED: {bad}")
    raise SystemExit(1 if bad else 0)
