"""Publishing: AI metadata + uploads to YouTube and Facebook Pages.

generate_metadata()  title / description / tags / hashtags in the target
                     language, from the episode context
YouTube              OAuth 2.0 for installed apps (loopback redirect, PKCE)
                     with the user's own Google Cloud OAuth client; resumable
                     upload (8 MB chunks, resumes after network errors);
                     optional custom thumbnail. Default privacy: private.
Facebook Page        Graph API resumable video upload (start / transfer /
                     finish) with a Page access token the user pastes in.

Credentials live in the ConfigManager; secrets and tokens are encrypted with
DPAPI (config_manager.encrypt_secret). Nothing is uploaded unless the caller
(the Publish tab, after the user confirms) calls upload_*().

No Qt imports; requests only.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import socket
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import requests

from app.core.config_manager import decrypt_secret, encrypt_secret
from app.core.errors import AppError, CancelledError
from app.core.logger import get_logger

log = get_logger("publisher")

ProgressFn = Callable[[float, str], None]

_YT_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
_YT_TOKEN = "https://oauth2.googleapis.com/token"
_YT_UPLOAD = "https://www.googleapis.com/upload/youtube/v3/videos"
_YT_THUMB = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
_YT_CHANNELS = "https://www.googleapis.com/youtube/v3/channels"
_YT_SCOPES = "https://www.googleapis.com/auth/youtube.upload " \
             "https://www.googleapis.com/auth/youtube.readonly"
_FB_GRAPH = "https://graph.facebook.com/v21.0"
_FB_VIDEO = "https://graph-video.facebook.com/v21.0"
_CHUNK = 8 * 1024 * 1024
PRIVACY = ["private", "unlisted", "public"]


@dataclass
class Metadata:
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)

    def description_full(self) -> str:
        tags = " ".join(h if h.startswith("#") else f"#{h}" for h in self.hashtags)
        return (self.description.strip() + ("\n\n" + tags if tags else "")).strip()


# a scheduled post must be this far ahead: Facebook takes 10 minutes to 6
# months; YouTube only needs a time in the future
SCHEDULE_MIN_S = 15 * 60
FB_SCHEDULE_MAX_S = 180 * 24 * 3600


def schedule_problem(when: datetime | None, facebook: bool = False,
                     now: datetime | None = None) -> str:
    """Why `when` cannot be a scheduled post time ("" = fine). `when` with
    no time zone is the PC's local time."""
    if when is None:
        return ""
    now = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.astimezone()
    ahead = (when - now).total_seconds()
    if ahead < SCHEDULE_MIN_S:
        return "too_soon"
    if facebook and ahead > FB_SCHEDULE_MAX_S:
        return "too_far"
    return ""


def _rfc3339_utc(when: datetime) -> str:
    if when.tzinfo is None:
        when = when.astimezone()
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def generate_metadata(ai, context: str, lang: str = "km", platform: str = "youtube",
                      title_hint: str = "", cancel: threading.Event | None = None
                      ) -> Metadata:
    from app.core.ai_engine import extract_json
    from app.engines.recap_engine import _lang_name
    lname = _lang_name(lang)
    system = (
        f"You write {platform} metadata for a drama recap / dubbed drama "
        "channel. Respond with ONLY one JSON object: {\"title\": str, "
        "\"description\": str, \"tags\": [str], \"hashtags\": [str]}. Rules: "
        f"title in {lname}, at most 90 characters, curiosity-raising but true "
        "to the story; description in "
        f"{lname}: 3-6 short paragraphs (hook, what happens without spoiling "
        "the ending, a question for comments, a follow/subscribe line); "
        "8-15 tags (mix of the language and English, include the drama name "
        "if known); 3-6 hashtags without spaces. No emojis in the title. "
        "Never claim it is an official upload.")
    user = (f"Working title: {title_hint}\n\n" if title_hint else "") + \
        f"Episode context:\n{(context or '')[:8000]}"
    res = ai.chat(feature="recap", system=system, json_mode=True, temperature=0.8,
                  max_tokens=2000, cancel=cancel,
                  messages=[{"role": "user", "content": user}])
    try:
        d = extract_json(res.text)
    except Exception as exc:
        raise AppError(f"Could not read the AI metadata: {exc}",
                       f"មិនអាចអានព័ត៌មានពី AI បានទេ៖ {exc}") from exc
    d = d if isinstance(d, dict) else {}
    tags = [str(t).strip().lstrip("#") for t in d.get("tags") or [] if str(t).strip()]
    hashtags = ["#" + str(h).strip().lstrip("#").replace(" ", "")
                for h in d.get("hashtags") or [] if str(h).strip()]
    return Metadata(title=str(d.get("title") or title_hint).strip()[:100],
                    description=str(d.get("description") or "").strip()[:4800],
                    tags=_fit_tags(tags), hashtags=hashtags[:8])


def _fit_tags(tags: list[str], limit: int = 480) -> list[str]:
    """YouTube allows ~500 characters of tags in total."""
    out, total = [], 0
    for t in tags:
        t = t.replace("<", "").replace(">", "")[:100]
        cost = len(t) + (2 if " " in t else 0) + 1
        if t and total + cost <= limit and t.lower() not in {x.lower() for x in out}:
            out.append(t)
            total += cost
    return out


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

class YouTube:
    """Upload with the user's own OAuth client (Desktop app type)."""

    def __init__(self, config):
        self.config = config

    # credentials ---------------------------------------------------------
    def set_client(self, client_id: str, client_secret: str) -> None:
        self.config.set("publish.youtube.client_id", client_id.strip())
        self.config.set("publish.youtube.client_secret", encrypt_secret(client_secret.strip()))

    def client(self) -> tuple[str, str]:
        cid = str(self.config.get("publish.youtube.client_id") or "")
        sec = self.config.get("publish.youtube.client_secret") or ""
        return cid, (decrypt_secret(sec) if sec else "")

    def _token(self) -> dict:
        raw = self.config.get("publish.youtube.token") or ""
        try:
            return json.loads(decrypt_secret(raw)) if raw else {}
        except Exception:
            return {}

    def _save_token(self, tok: dict) -> None:
        self.config.set("publish.youtube.token", encrypt_secret(json.dumps(tok)))

    def connected(self) -> bool:
        return bool(self._token().get("refresh_token"))

    def disconnect(self) -> None:
        self.config.set("publish.youtube.token", "")
        self.config.set("publish.youtube.channel", "")

    def channel_name(self) -> str:
        return str(self.config.get("publish.youtube.channel") or "")

    def connect(self, timeout: float = 300.0,
                cancel: threading.Event | None = None) -> str:
        """Browser sign-in (loopback + PKCE). Returns the channel name."""
        cid, secret = self.client()
        if not cid:
            raise AppError("Add your Google OAuth client ID first.",
                           "សូមបញ្ចូល Google OAuth client ID ជាមុនសិន។")
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        got: dict = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                got.update({k: v[0] for k, v in q.items()})
                ok = "code" in got and got.get("state") == state
                body = ("<h2>RecapStudio is connected to YouTube.</h2>You can close this tab."
                        if ok else "<h2>Sign-in failed.</h2>You can close this tab.")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *_a):
                pass

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        server.timeout = 0.5
        redirect = f"http://127.0.0.1:{port}"
        url = _YT_AUTH + "?" + urllib.parse.urlencode({
            "client_id": cid, "redirect_uri": redirect, "response_type": "code",
            "scope": _YT_SCOPES, "access_type": "offline", "prompt": "consent",
            "state": state, "code_challenge": challenge,
            "code_challenge_method": "S256"})
        webbrowser.open(url)
        t0 = time.time()
        try:
            while "code" not in got and "error" not in got:
                _check_cancel(cancel)
                if time.time() - t0 > timeout:
                    raise AppError("Timed out waiting for the Google sign-in.",
                                   "អស់ពេលរង់ចាំការចូលគណនី Google។")
                server.handle_request()
        finally:
            server.server_close()
        if got.get("state") != state or "code" not in got:
            raise AppError(f"Google sign-in failed: {got.get('error', 'unknown')}",
                           f"ការចូលគណនី Google បរាជ័យ៖ {got.get('error', '')}")
        r = requests.post(_YT_TOKEN, timeout=30, data={
            "code": got["code"], "client_id": cid, "client_secret": secret,
            "redirect_uri": redirect, "grant_type": "authorization_code",
            "code_verifier": verifier})
        if r.status_code != 200:
            raise AppError(f"Token exchange failed: {r.text[:300]}",
                           "ការប្តូរ token បរាជ័យ។")
        tok = r.json()
        tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
        self._save_token(tok)
        name = self._fetch_channel()
        self.config.set("publish.youtube.channel", name)
        return name

    def _access_token(self) -> str:
        tok = self._token()
        if not tok.get("refresh_token"):
            raise AppError("YouTube is not connected — press Connect first.",
                           "មិនទាន់ភ្ជាប់ YouTube ទេ — សូមចុច Connect ជាមុនសិន។")
        if tok.get("access_token") and tok.get("expires_at", 0) > time.time():
            return tok["access_token"]
        cid, secret = self.client()
        r = requests.post(_YT_TOKEN, timeout=30, data={
            "client_id": cid, "client_secret": secret,
            "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"})
        if r.status_code != 200:
            raise AppError(f"Could not refresh the YouTube sign-in: {r.text[:300]}",
                           "មិនអាចបន្តការចូល YouTube បានទេ — សូមភ្ជាប់ម្តងទៀត។")
        new = r.json()
        tok["access_token"] = new["access_token"]
        tok["expires_at"] = time.time() + int(new.get("expires_in", 3600)) - 60
        self._save_token(tok)
        return tok["access_token"]

    def _fetch_channel(self) -> str:
        r = requests.get(_YT_CHANNELS, timeout=30,
                         params={"part": "snippet", "mine": "true"},
                         headers={"Authorization": f"Bearer {self._access_token()}"})
        try:
            return r.json()["items"][0]["snippet"]["title"]
        except Exception:
            return "YouTube channel"

    # upload --------------------------------------------------------------
    def upload(self, video: str, meta: Metadata, privacy: str = "private",
               category_id: str = "1", thumbnail: str = "",
               made_for_kids: bool = False, language: str = "",
               progress: ProgressFn | None = None,
               cancel: threading.Event | None = None,
               publish_at: datetime | None = None) -> str:
        """Resumable upload. Returns the video URL. `publish_at`: uploaded
        private now, YouTube makes it public at that time (the PC need not
        be on)."""
        if not os.path.isfile(video):
            raise AppError(f"Video not found: {video}", f"រកមិនឃើញវីដេអូ៖ {video}")
        if schedule_problem(publish_at):
            raise AppError("The scheduled time must be at least 15 minutes from now.",
                           "ពេលវេលាកំណត់ត្រូវតែយ៉ាងហោចណាស់ 15 នាទីពីពេលនេះ។")
        size = os.path.getsize(video)
        status = {"privacyStatus": privacy if privacy in PRIVACY else "private",
                  "selfDeclaredMadeForKids": bool(made_for_kids)}
        if publish_at is not None:
            # YouTube's rule: a scheduled video is private until publishAt
            status["privacyStatus"] = "private"
            status["publishAt"] = _rfc3339_utc(publish_at)
        body = {
            "snippet": {"title": meta.title[:100], "description": meta.description_full()[:5000],
                        "tags": meta.tags, "categoryId": category_id},
            "status": status,
        }
        if language:
            body["snippet"]["defaultLanguage"] = language
            body["snippet"]["defaultAudioLanguage"] = language
        r = requests.post(
            _YT_UPLOAD, timeout=60,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={"Authorization": f"Bearer {self._access_token()}",
                     "Content-Type": "application/json; charset=UTF-8",
                     "X-Upload-Content-Length": str(size),
                     "X-Upload-Content-Type": "video/*"},
            data=json.dumps(body))
        if r.status_code not in (200, 201) or "Location" not in r.headers:
            raise AppError(f"YouTube refused the upload: {r.status_code} {r.text[:300]}",
                           f"YouTube បដិសេធការផ្ទុកឡើង៖ {r.status_code}")
        session = r.headers["Location"]
        video_id = self._send(session, video, size, progress, cancel)
        if thumbnail and os.path.isfile(thumbnail):
            try:
                with open(thumbnail, "rb") as f:
                    tr = requests.post(_YT_THUMB, params={"videoId": video_id}, timeout=120,
                                       headers={"Authorization": f"Bearer {self._access_token()}",
                                                "Content-Type": "image/jpeg"}, data=f)
                if tr.status_code != 200:
                    log.warning("thumbnail upload refused (%s): %s — custom thumbnails "
                                "need a verified channel", tr.status_code, tr.text[:200])
            except requests.RequestException as exc:
                log.warning("thumbnail upload failed: %s", exc)
        return f"https://youtu.be/{video_id}"

    def _send(self, session: str, video: str, size: int, progress, cancel) -> str:
        offset, retries = 0, 0
        with open(video, "rb") as f:
            while True:
                _check_cancel(cancel)
                f.seek(offset)
                chunk = f.read(_CHUNK)
                end = offset + len(chunk) - 1
                try:
                    r = requests.put(session, data=chunk, timeout=300, headers={
                        "Authorization": f"Bearer {self._access_token()}",
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{end}/{size}"})
                except requests.RequestException as exc:
                    retries += 1
                    if retries > 8:
                        raise AppError(f"Upload failed: {exc}", f"ការផ្ទុកឡើងបរាជ័យ៖ {exc}") from exc
                    time.sleep(min(30, 2 ** retries))
                    offset = self._resume_offset(session, size)
                    continue
                if r.status_code in (200, 201):
                    if progress:
                        progress(1.0, "Uploaded")
                    return r.json()["id"]
                if r.status_code == 308:
                    rng = r.headers.get("Range", "")
                    offset = int(rng.split("-")[1]) + 1 if "-" in rng else 0
                    retries = 0
                    if progress:
                        progress(offset / size, f"Uploading {offset / size:.0%}")
                    continue
                if r.status_code in (500, 502, 503, 504):
                    retries += 1
                    if retries > 8:
                        raise AppError(f"YouTube error {r.status_code}", f"កំហុស YouTube {r.status_code}")
                    time.sleep(min(30, 2 ** retries))
                    offset = self._resume_offset(session, size)
                    continue
                raise AppError(f"Upload failed: {r.status_code} {r.text[:300]}",
                               f"ការផ្ទុកឡើងបរាជ័យ៖ {r.status_code}")

    def _resume_offset(self, session: str, size: int) -> int:
        try:
            r = requests.put(session, timeout=60, headers={
                "Authorization": f"Bearer {self._access_token()}",
                "Content-Length": "0", "Content-Range": f"bytes */{size}"})
            rng = r.headers.get("Range", "")
            return int(rng.split("-")[1]) + 1 if "-" in rng else 0
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Facebook Page
# ---------------------------------------------------------------------------

class FacebookPage:
    def __init__(self, config):
        self.config = config

    def set_page(self, page_id: str, token: str) -> None:
        self.config.set("publish.facebook.page_id", page_id.strip())
        self.config.set("publish.facebook.token", encrypt_secret(token.strip()))

    def page(self) -> tuple[str, str]:
        pid = str(self.config.get("publish.facebook.page_id") or "")
        tok = self.config.get("publish.facebook.token") or ""
        return pid, (decrypt_secret(tok) if tok else "")

    def connected(self) -> bool:
        pid, tok = self.page()
        return bool(pid and tok)

    def page_name(self) -> str:
        pid, tok = self.page()
        r = requests.get(f"{_FB_GRAPH}/{pid}", params={"fields": "name", "access_token": tok},
                         timeout=30)
        if r.status_code != 200:
            raise AppError(f"Facebook: {r.json().get('error', {}).get('message', r.text[:200])}",
                           "Facebook មិនទទួលស្គាល់ Page ID ឬ token នេះទេ។")
        return r.json().get("name", pid)

    def upload(self, video: str, meta: Metadata, publish_now: bool = False,
               thumbnail: str = "", progress: ProgressFn | None = None,
               cancel: threading.Event | None = None,
               scheduled_at: datetime | None = None) -> str:
        """Resumable upload to the Page. Unpublished (draft) unless
        publish_now; `scheduled_at`: Facebook publishes it at that time
        (10 minutes to 6 months ahead; the PC need not be on). Returns the
        video URL."""
        pid, tok = self.page()
        if not (pid and tok):
            raise AppError("Add the Facebook Page ID and Page access token first.",
                           "សូមបញ្ចូល Facebook Page ID និង Page access token ជាមុនសិន។")
        why = schedule_problem(scheduled_at, facebook=True)
        if why:
            raise AppError("Facebook schedules a post 15 minutes to 6 months ahead."
                           if why == "too_far" else
                           "The scheduled time must be at least 15 minutes from now.",
                           "Facebook កំណត់ពេលបង្ហោះបានពី 15 នាទី ដល់ 6 ខែទៅមុខ។"
                           if why == "too_far" else
                           "ពេលវេលាកំណត់ត្រូវតែយ៉ាងហោចណាស់ 15 នាទីពីពេលនេះ។")
        size = os.path.getsize(video)
        url = f"{_FB_VIDEO}/{pid}/videos"

        def post(data, files=None):
            r = requests.post(url, data={**data, "access_token": tok}, files=files,
                              timeout=600)
            try:
                j = r.json() if r.content else {}
            except ValueError:          # an HTML error page from a proxy / 5xx
                j = {}
            if r.status_code != 200 or "error" in j:
                msg = (j.get("error") or {}).get("message") or r.text[:300]
                exc = AppError(f"Facebook upload failed: {msg}",
                               f"ការផ្ទុកឡើង Facebook បរាជ័យ៖ {msg}")
                exc.retryable = r.status_code >= 500 or bool(
                    (j.get("error") or {}).get("is_transient"))
                raise exc
            return j

        waiter = cancel or threading.Event()

        def transfer(so: int, chunk: bytes) -> dict:
            """Send one chunk, retrying network hiccups and transient errors
            (the session stays valid, so a long upload is not lost)."""
            tries = 6
            for attempt in range(tries):
                _check_cancel(cancel)
                try:
                    return post({"upload_phase": "transfer", "upload_session_id": sid,
                                 "start_offset": str(so)},
                                files={"video_file_chunk": ("chunk", chunk)})
                except AppError as exc:
                    if attempt == tries - 1 or not getattr(exc, "retryable", False):
                        raise
                    err = exc
                except requests.RequestException as exc:
                    if attempt == tries - 1:
                        raise AppError(f"Facebook upload failed: {exc}",
                                       f"ការផ្ទុកឡើង Facebook បរាជ័យ៖ {exc}") from exc
                    err = exc
                log.warning("Facebook chunk at byte %d failed (%s); retrying", so, err)
                waiter.wait(min(30, 2 ** (attempt + 1)))
            raise AssertionError("unreachable")

        start = post({"upload_phase": "start", "file_size": str(size)})
        sid = start["upload_session_id"]
        video_id = str(start.get("video_id") or "")
        so, eo = int(start["start_offset"]), int(start["end_offset"])
        with open(video, "rb") as f:
            while so < eo:
                _check_cancel(cancel)
                f.seek(so)
                chunk = f.read(eo - so)
                nxt = transfer(so, chunk)
                so, eo = int(nxt["start_offset"]), int(nxt["end_offset"])
                if progress:
                    progress(so / size, f"Uploading {so / size:.0%}")
        fin = {"upload_phase": "finish", "upload_session_id": sid,
               "title": meta.title[:250], "description": meta.description_full(),
               "published": "true" if publish_now and scheduled_at is None else "false"}
        if scheduled_at is not None:
            when = scheduled_at if scheduled_at.tzinfo else scheduled_at.astimezone()
            fin["scheduled_publish_time"] = str(int(when.timestamp()))
        files = None
        if thumbnail and os.path.isfile(thumbnail):
            files = {"thumb": (os.path.basename(thumbnail), open(thumbnail, "rb"),
                               "image/jpeg")}
        try:
            post(fin, files=files)
        finally:
            if files:
                files["thumb"][1].close()
        if progress:
            progress(1.0, "Uploaded")
        if video_id:
            return f"https://www.facebook.com/{pid}/videos/{video_id}/"
        return f"https://www.facebook.com/{pid}/videos/"


if __name__ == "__main__":
    from types import SimpleNamespace as NS
    fake = NS(chat=lambda **k: NS(text=json.dumps({
        "title": "ស្នេហ៍ក្បត់ ភាគ១", "description": "សាច់រឿង...",
        "tags": ["drama", "Drama", "khmer dub", "x" * 120],
        "hashtags": ["khmer drama", "#recap"]})))
    m = generate_metadata(fake, "ctx", "km", title_hint="t")
    assert m.title == "ស្នេហ៍ក្បត់ ភាគ១"
    assert m.tags[:2] == ["drama", "khmer dub"], m.tags      # case-insensitive dedup
    assert len(m.tags[-1]) == 100, "long tags trimmed"
    assert m.hashtags == ["#khmerdrama", "#recap"]
    assert m.description_full().endswith("#khmerdrama #recap")
    assert sum(len(t) + 1 for t in _fit_tags(["a" * 60] * 20)) <= 480
    # scheduling: at least 15 minutes ahead; Facebook at most 6 months
    from datetime import timedelta
    t0 = datetime(2026, 11, 22, 3, 0, tzinfo=timezone.utc)
    assert schedule_problem(None) == ""
    assert schedule_problem(t0 + timedelta(minutes=5), now=t0) == "too_soon"
    assert schedule_problem(t0 + timedelta(hours=2), now=t0) == ""
    assert schedule_problem(t0 + timedelta(days=200), facebook=True, now=t0) == "too_far"
    assert schedule_problem(t0 + timedelta(days=200), now=t0) == ""
    assert _rfc3339_utc(datetime(2026, 11, 22, 20, 30, tzinfo=timezone(timedelta(hours=7)))) \
        == "2026-11-22T13:30:00.000Z"
    # the requests sent: YouTube private + publishAt; Facebook unpublished + time
    import unittest.mock as um
    sent = {}

    class _R:
        def __init__(self, code, js=None, headers=None):
            self.status_code, self._js, self.headers = code, js or {}, headers or {}
            self.content = b"x"
            self.text = json.dumps(self._js)

        def json(self):
            return self._js

    def fake_post(url, **kw):
        sent.setdefault(url, []).append(kw)
        if "upload/youtube" in url:
            return _R(200, headers={"Location": "https://session"})
        if kw.get("data", {}).get("upload_phase") == "start":
            return _R(200, {"upload_session_id": "s1", "video_id": "v1",
                            "start_offset": "0", "end_offset": "0"})
        return _R(200, {"success": True})

    import tempfile
    vid = os.path.join(tempfile.mkdtemp(), "v.mp4")
    with open(vid, "wb") as fh:
        fh.write(b"0" * 10)
    later = datetime.now(timezone.utc) + timedelta(days=1)
    cfg = NS(get=lambda k, d=None: d, set=lambda *a: None)
    yt = YouTube(cfg)
    yt._access_token = lambda: "tok"
    yt._send = lambda *a, **k: "abc"
    with um.patch.object(requests, "post", fake_post):
        assert yt.upload(vid, m, privacy="public", publish_at=later) == "https://youtu.be/abc"
    yt_body = json.loads([v for k, v in sent.items() if "upload/youtube" in k][0][0]["data"])
    assert yt_body["status"]["privacyStatus"] == "private"
    assert yt_body["status"]["publishAt"] == _rfc3339_utc(later), yt_body
    fb = FacebookPage(cfg)
    fb.page = lambda: ("123", "tok")
    with um.patch.object(requests, "post", fake_post):
        fb.upload(vid, m, publish_now=True, scheduled_at=later)
    fin = [c["data"] for c in sent[f"{_FB_VIDEO}/123/videos"] if c["data"].get("upload_phase") == "finish"][0]
    assert fin["published"] == "false" and fin["scheduled_publish_time"] == str(int(later.timestamp()))
    try:
        fb.upload(vid, m, scheduled_at=datetime.now(timezone.utc) + timedelta(minutes=3))
        raise AssertionError("too soon must be refused")
    except AppError:
        pass
    print("PASS")
