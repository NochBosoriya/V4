# RecapStudio compared with a complete application

What RecapStudio already has, what a complete application has, and what is
left to do — in order. Checked on 2026-11-16 against the code in
`new update\V4\V4`. Effort figures are rough estimates for one developer.

---

## 1. What RecapStudio is today

RecapStudio is a **desktop application** for Windows. Its "frontend" and its
"backend" run in the same program on the user's PC:

```
 ┌──────────────── RecapStudio.exe / main.py (one process, the user's PC) ─────────────┐
 │  FRONTEND  PySide6 window: 9 tabs (Import, Translate & Dub, Recap,                   │
 │            Overlay, DL, Drama MG, Export, Publish, Settings),                        │
 │            English / Khmer with live switching, progress panels, error recovery     │
 │                                   │ calls (worker threads)                          │
 │  "BACKEND" app/engines + app/core: Python, no Qt — speech recognition (Whisper on   │
 │            the GPU), translation, voices, mixing, ffmpeg export, posters, OCR,      │
 │            downloads, publishing                                                    │
 │  DATA      config.db (SQLite; API keys encrypted with Windows DPAPI), project       │
 │            folders (JSON + audio) and the library, in the user's app-data folder    │
 └──────────────────────────────────────────┬───────────────────────────────────────┘
                                            │ internet
     AI (OpenRouter → Gemini, OpenAI, Anthropic, DeepSeek) · Deepgram (speech) ·
     Edge TTS (voices) · YouTube / Facebook upload · yt-dlp (downloads)
```

It is shared as a **portable folder / zip** (3.1 GB with its own Python),
started with a `.bat` file.

There are two kinds of "complete application". **A** is a complete *desktop
product*: the same app, finished for other people to download, install, trust
and pay for. **B** is a complete *online (web) application*: a website plus
servers, used in a browser or on a phone. RecapStudio is close to A. B is a
different, much bigger project. Sections 3–4 cover A; section 5 covers B.

---

## 2. Found while checking — for you to decide

1. **The code exists in two different copies, and each one is missing work.**

   | Copy | Last changed | Has | Does not have |
   |---|---|---|---|
   | `new update\V4\V4` (the one we work in) | 16 Nov | the audit fixes (lost lines, second listen, dub timing, Ep2 model) | the 12 Nov work |
   | `Backup\backup(2)\backup3\Clone voice\V4` | 12 Nov | local translation (`local_mt.py`, NLLB / MADLAD), local Khmer voice (`local_tts.py`), model manager panel, MT review, benchmarks, `redact.py` (keys scrubbed from errors and logs), Whisper cache clean-up | everything after 12 Nov |

   Both copies come from the 11 Nov version. A portable built from either one
   lacks the other one's features.

   **Recommended:** put the current copy under version control (git) first,
   then bring the 12 Nov work over as a separate change and test it.
   Effort: about 1–2 days.
2. **The portable build is out of date.** It was built on 15 Nov, so it lacks the
   16 Nov changes. Rebuild with `python build_portable.py`.
3. **The app has no version number,** so you cannot tell which build a user has.
   That matters as soon as two people use it.

---

## 3. Side by side: a complete desktop application vs. RecapStudio

| Part | A complete application has | RecapStudio now | Status |
|---|---|---|---|
| Screens (frontend) | every task doable from the window, clear progress and errors, help | 9 tabs, Khmer + English, run panel with Continue, errors kept on screen | **Done** — no built-in help or first-run guide |
| Processing (backend) | separate from the screens, can be cancelled and resumed | engines have no Qt, cancel / resume, results cached | **Done** |
| Quality checks | automatic tests on every change | 22 module self-tests (all pass), UI tests written for single tasks | **Partly** — tests are run by hand, there's no CI |
| Version control | every change recorded, one main copy, releases tagged | no git; two diverged copies (section 2) | **Missing** |
| Version and changelog | a version in the window, a list of what changed | none | **Missing** |
| Installer | install to Program Files, Start-menu entry, uninstall | portable zip; users "Unblock" the zip and start a `.bat` | **Partly** |
| Trust (code signing) | signed files, so Windows (SmartScreen, Smart App Control) lets them run | unsigned; Smart App Control blocks parts of it (e.g. torch on this PC) | **Missing** |
| Updates | the app says when a new version exists | none | **Missing** |
| Where user data lives | a visible place the user can find and back up | hidden app-data folder (with Store Python, a redirected one) | **Partly** |
| Secrets | encrypted keys, never in logs | DPAPI-encrypted keys; this copy lacks `redact.py` | **Partly** |
| Crash reports | errors from users reach the developer (opt-in) | local `app.log` only (5 × 5 MB) | **Partly** |
| Accounts and licence keys | sign-in or a licence key, if it is sold | none (a rule in ARCHITECTURE.md: "no licensing, activation") | **Missing** (only needed if you sell it) |
| Payments | a way to buy | none | **Missing** (if sold) |
| Cost control | limits on paid services | monthly AI budget | **Done** for AI; Deepgram has none |
| User guide | a manual / videos in the users' language | developer documents only (ARCHITECTURE, AUDIT_REPORT, docs/) | **Missing** |
| Legal | licence / terms, privacy note, third-party notices | ffmpeg licence file only | **Missing** (see below) |
| Support | a way for users to report problems | none | **Missing** |

**Legal points to settle before other people use it** (not legal advice — have
them checked):
- **Third-party notices.** The bundled ffmpeg is the GPL v3 build: its licence
  is included, and a source offer is needed when you give it out. PySide6 (LGPL)
  and the other packages need their notices in a `THIRD_PARTY_NOTICES.txt`.
- **Edge TTS.** It uses Microsoft Edge's read-aloud service unofficially. For a
  product you sell, Azure Speech is the licensed way to get the same voices
  (paid per character).
- **Downloads (DL tab).** yt-dlp downloads from YouTube, Bilibili and others. The
  user is responsible for copyright and the sites' terms, and your terms should
  say so.
- **Model licences.** NLLB in the 12 Nov copy is non-commercial; MADLAD is
  allowed. Voice cloning needs the voice owner's consent.
- **Privacy note.** Say what is sent where: audio to Deepgram, text and poster
  frames to the AI provider, videos to YouTube / Facebook only on Publish.
- **YouTube upload.** It uses each user's own Google OAuth client. That's fine
  for personal use; a public product needs its own verified Google app.

---

## 4. What to do to finish it as a desktop application

**P0 — before giving it to other people**

| # | Task | Why | Effort |
|---|---|---|---|
| 1 | git + a private repository (GitHub); merge the two copies (section 2) | one true copy; nothing lost; you can go back | 1–2 days |
| 2 | app version + "About" box + CHANGELOG | know which build a user has | ½ day |
| 3 | installer (Inno Setup or MSIX): Program Files, Start menu, uninstall; projects in `Documents\RecapStudio` | normal install; files easy to find and back up | 2–3 days |
| 4 | code signing: an OV/EV certificate (about $200–400/year from a reseller) or Microsoft's signing service where your country is supported; sign the installer and your own files | SmartScreen and Smart App Control trust it | 2–5 days, incl. identity checks |
| 5 | `LICENSE` / terms of use, privacy note, `THIRD_PARTY_NOTICES.txt` | legal minimum (section 3) | 1 day + review |
| 6 | run the 22 self-tests before every build; the build stops if one fails | no broken builds | ½ day |

**P1 — to make it a solid product**

| # | Task | Effort |
|---|---|---|
| 7 | update check: a small JSON file online (latest version + download link) that the app reads at start | 1–2 days |
| 8 | opt-in crash reports (e.g. Sentry's free tier): the error and the end of app.log | 1 day |
| 9 | CI: GitHub Actions on Windows runs the self-tests on every change (tests needing big models are skipped) | 1–2 days |
| 10 | user guide in Khmer and English + a first-run tour (key → video → dub → save) | 3–5 days |
| 11 | licence keys, if you sell it (Lemon Squeezy / Paddle / Keygen; ABA PayWay or KHQR for Cambodia) | 3–5 days |
| 12 | for a paid product: Edge TTS → Azure Speech; a Deepgram budget limit like the AI one | 2–3 days |

**P2 — later:** undo / history in the Lines editor, project list with
search and backup, accessibility (keyboard, screen size), a smaller
download (models fetched on first use instead of shipped).

---

## 5. If you want an online (web) application with frontend + backend servers

```
 Browser / phone ── FRONTEND (React / Next.js): upload, lines editor, poster canvas, …
        │ HTTPS (REST + live progress over WebSocket / SSE)
 API SERVER (FastAPI, Python): accounts, projects, jobs, payments, limits
        │                       │                          │
 DATABASE (Postgres)     JOB QUEUE (Redis + Celery/RQ)   FILE STORAGE (S3 / R2):
 users, projects,               │                        uploaded videos, results
 jobs, invoices          GPU WORKERS = today's app/engines
                         (Whisper, translation, voices, ffmpeg, posters)
 + sign-in (Clerk / Auth0 / Supabase) · payments (Stripe / Paddle / ABA PayWay)
 + monitoring (Sentry, logs, uptime) · admin page · abuse / copyright handling
```

**What carries over:** the engines (`app/engines`, `app/core`) are already
separate from the screens. That's the biggest asset: they become the GPU
workers almost as they are. The `en.json` / `km.json` texts and the
ARCHITECTURE contract also carry over.

**What is new:**
- every screen rebuilt for the browser (the Lines editor, timeline and poster
  canvas are the big ones)
- the API server and the job queue
- resumable upload of large videos
- per-user storage and quotas
- accounts and payments
- servers with GPUs (roughly $100–500+ a month, more as users grow)
- security, privacy and copyright procedures, monitoring, and support

**Effort:** roughly 3–6 months for one developer to reach a first public
version, plus monthly server and AI costs.

**Recommendation:** finish the desktop product (section 4, P0 then P1) first, and
find out what users pay for. A sensible middle step is a *small* backend only for
licence keys, update checks and crash reports; it is cheap and keeps all the
heavy work on users' PCs. Build the full web version only when you know people
want it.
