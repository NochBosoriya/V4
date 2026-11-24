# RecapStudio — Technical audit: foreign speech → Khmer (2026-11-15)

Scope: the whole path from an imported video to Khmer subtitles and dub —
audio extraction, separation, speech recognition (ASR), line building,
translation, timing fit, subtitles, export — with the focus on the reported
problem: **in fast speech, sentences disappear from the Khmer translation.**

Everything below was measured on your own projects
(`%LOCALAPPDATA%\…\RecapStudio\projects\dub`), not estimated. Ground truth
for speech recognition is the on-screen Chinese subtitles the app itself
reads (`subtitles.json`); "fast" = a subtitle of ≥ 7 characters a second.

---

## 1. Root causes in one page

| # | Where sentences were lost | Measured on your projects | Status |
|---|---|---|---|
| 1 | **Fit / Synthesize shortening.** Khmer needs more time than Chinese; fast speech leaves short slots. The last-resort "condense" pass rewrote lines down to their *main point* — and when no version fitted it took **the shortest one**. | Khmer removed from the dub lines: **32 %** (ah sterr), 11–17 % (Ep2), 9 % (2-h Chinese drama: 292 lines cut below 75 %, 73 below 50 %), 6 % (2-h English drama). Example: 前面那辆破摩托能不能快点，**我还等着报到呢** → dub said only "can that broken motorbike go faster?" | Fixed |
| 2 | **Subtitles were written from the shortened dub text** (`write_srt` / `write_ass` used `ln.text`). | **29 %** of the translation was missing from the exported subtitles of ah sterr; 8–16 % for Ep2. | Fixed |
| 3 | **Speech recognition skipped lines** (Whisper's decoder jumps over sentences in dense speech; Silero VAD at 0.5 cut quiet/fast speech before Whisper heard it). | Whisper large-v3 missed **32 of 448** subtitle lines in 20 min of a drama; Deepgram missed 136 of 2,489 in the full 2 h, and recalled fast lines worse (**86.7 %** vs 94.5 % overall). For zh/ja dramas with burned-in subtitles the subtitle reader filled most gaps; **for every other video those lines were simply gone.** | Improved (32 → 25) |
| 4 | **Silent deletions after recognition**: `drop_repeats` deleted real repeats ("快走！快走！"), `drop_prompt_echo` deleted any line that was one character's name, foreign-script leftovers were only logged. | Code inspection + new tests. | Fixed |
| 5 | **Translation memory bypassed the glossary**: an old AI entry kept its wrong term forever (神豪系统 → "goddess system" while the glossary says "tycoon system"; character names). | 3 such entries in ah sterr, 5 in Ep2 fixed. | Fixed |

So for your Chinese dramas the loss was mostly **downstream of recognition**
(cause 1–2): with hard subtitles, the dub lines covered 100 % of the subtitle
text, and then up to a third of the Khmer was cut away.

---

## 2. What was changed

| Change | Files | Default |
|---|---|---|
| Subtitles show each line's **whole translation** (`DubLine.full_text`); a line you typed shows what you typed; an emptied line has no subtitle | dub_engine, dub_tab | on (checkbox "Subtitles show the whole translation") |
| **Fitting order** (2026-11-16, with your 2026-11-11 Ep2 dub as the model): spoken up to 1.4× → said more briefly, same meaning → wording tightened, every statement kept → only what still does not fit is spoken faster, just enough to end before the next line, up to "Max speed-up" **2.0×** (range 1.0–2.5×) → late start. (2026-11-15 had speed first: 11 lines of Ep2 at 1.4–2.0×; you preferred the older dub.) Each line's speed is shown in Lines | dub_engine, dub_tab | on (projects made before keep their saved Max speed-up) |
| **Saying a line briefly like your model dub**: the rewording and tightening prompts carry your approved Ep2 lines as examples (drop what voice and picture carry — interjections, a surprised "what did he say?!", pronouns Khmer doesn't need, fillers; short forms of names and numbers — never a statement); the meaning check accepts those drops | dub_engine | — |
| Shortening **keeps every statement**: the condense pass may only tighten wording (2 versions, no "main point only"), and the meaning check refuses a version that left a statement out | dub_engine | on (checkbox "Allow cutting words to keep lip-sync" brings the old cuts back per project) |
| Faster re-voicing retries when Edge drops a request (6 lines of one dub had fallen back to stretched, warbly audio) | dub_engine | — |
| Fit always starts again from the whole translation (a re-run used to shorten already-shortened lines) | dub_engine | — |
| Silero VAD threshold **0.35** (was 0.5) — measured, see §4 | audio_pipeline | — |
| **Second listen** (`recover_gaps`): voiced stretches no recognised word covers are transcribed again one by one; invented text (stock phrases, loops, too many words for the time, Whisper's own "no speech", duplicates) is refused; every outcome is logged, recovered lines are marked ⚑, unrecognised voice is listed with times in the step note and saved in transcript.json | audio_pipeline, dub_engine | on (Whisper and, when a Whisper model is installed, after Deepgram) |
| Lines Whisper doubted (log-prob < −1) are kept and marked ⚑ | audio_pipeline | — |
| `drop_repeats` only removes a real echo (overlapping, or under half the time); `drop_prompt_echo` only two+ names together; both log | dub_engine | — |
| Translation-memory hits that break the glossary are translated again; lines still off the glossary are marked ⚑ | dub_engine | — |
| Lines tab: ⚑ marks with the reason, shortened lines tinted with the whole translation in the tooltip, "Only lines to check" filter, right-click **Checked** / **Nobody says this: leave the line out** / **Put the line back** | dub_tab | — |
| Deepgram: retries on dropped connection, 429, 5xx (5 s, 20 s) | deepgram_stt | — |
| Translate tab: SRT dialogue is polished as dialogue against its originals (was "cinematic narration" from the drafts alone); "never summarise" rule; per-line fallback keeps the whole reply | ai_engine, translate_tab | — |
| API keys in URLs are redacted from network-error messages (Gemini sends `?key=`) | ai_engine | — |
| New UI texts and engine notes in Khmer (`km.json`, `dyn.*` templates) | locales | — |

Verification: all 21 module self-tests pass (`python -m app.…`; dub_engine
92 → 112 checks, new tests in audio_pipeline, deepgram_stt, ai_engine);
the Dub tab was driven offscreen on a copy of a real project (12 checks);
all 10 tabs build. The tests used no API credit; only the comparison
below did (a few AI calls, approved).

**Dub audio, measured (2026-11-15, with your approval to use AI credit):**
a copy of "ah sterr" re-dubbed from Translate on with the new defaults
(output: `Video for test/Test results 2026-11-15/ah sterr.dub.km.mp4`):

| ah sterr | old dub (2026-11-11) | new dub |
|---|---|---|
| Khmer the voice says | 623 chars (68 % of the translation) | **870 chars (98 %)** |
| lines starting late | 1 (0.39 s) | 2 (0.40 s at most) |
| speed-up | up to 1.3× | median 1.53×; 13 lines normal speed, 4 up to 1.4×, 10 at 1.4–1.6×, 8 at 1.6–1.8×, 10 at 1.8–2.0× |

That was speed first. You compared Ep2 with the 2026-11-11 dub and found
the older one better — the fast lines were the difference. With the
corrected fitting order and your dub as the model (re-run of your Ep2
project from Fit, `Video for test/Test results 2026-11-16/Ep2.dub.km.mp4`):

| Ep2 | Khmer said | lines > 1.4× | lines ≥ 1.6× | late |
|---|---|---|---|---|
| model dub (2026-11-11) | 670 chars | 0 | 0 | 1 |
| speed first (2026-11-16 00:15) | 693 chars | 11 | 7 | 0 |
| corrected order + model | 678 chars | 6 | 3 | 0 |

"他说什么，10万买他赢" now comes out as the model said it,
"មួយសែនភ្នាល់គាត់ឈ្នះ?", at normal speed (it was 2.0×). AI answers vary
a little between runs (6–8 lines over 1.4× in three runs).

**To ship:** `dist/RecapStudio-portable` is a copy of `app/` and does not
have these changes until you run `python build_portable.py`.

---

## 3. Audit findings (Part 1)

Priority: Critical = loses content users see; High = loses content in common
cases or breaks a promise; Medium = quality / reliability; Low = hygiene.

### 3.1 Architecture (as found)

- **Frontend:** PySide6 desktop app, 10 tabs built lazily, long work on
  `FnWorker` threads with progress/cancel, log console fed by `LOG_BUS`,
  English/Khmer catalogs with runtime switching. Sound design.
- **Backend:** in-process engines (no server). `DubProject` keeps every step's
  state and caches (`tr_cache`, translation memory), so steps resume and
  re-runs are cheap. `AIEngine` rotates keys, cools down rate-limited keys,
  falls back across providers, enforces a budget. Settings in SQLite, API
  keys encrypted with Windows DPAPI.
- **"Upload":** local file import (drag & drop) and yt-dlp downloads — no
  network upload of user video (Deepgram receives a 16 kHz Opus copy of the
  speech track only when chosen).

### 3.2 Findings

| # | Area | Current implementation | Problem | Why it matters | Solution | Priority | Status |
|---|---|---|---|---|---|---|---|
| 1 | Timing fit | Fit + Synthesize "condense" pass: 4 graded versions down to the main point; shortest taken when none fits; meaning check accepts "only something left out" | Whole clauses and sentences removed | The #1 cause of "missing sentences" (§1) | Two versions that keep every statement; refuse versions that drop one; speed up / late start instead; old behaviour as an option | Critical | Fixed |
| 2 | Subtitles | `write_srt`/`write_ass` from the dub text | Subtitles lose what the dub cut | Subtitles have no lip-sync limit; users read them | `full_text` + `subtitle_text()` | Critical | Fixed |
| 3 | Fit re-run | Fit shortens `ln.text`, which it overwrote last time | Cuts compound on every re-run | Quality decays silently | Restart from `full_text` | High | Fixed |
| 4 | ASR / VAD | faster-whisper `vad_filter` with Silero default 0.5 | Quiet / fast speech removed before Whisper | 4 of 32 missed lines came back at 0.35 with no extra false text | `vad_parameters={"threshold": 0.35}` | High | Fixed |
| 5 | ASR completeness | Transcript accepted as is; a 30 s window with no_speech_prob > 0.6 and low log-prob is skipped, logged only at DEBUG inside faster-whisper | Lost speech leaves no trace | Violates "never silently discard" | Coverage check vs VAD + second listen + report | High | Fixed |
| 6 | Line building | `drop_repeats`: same text within 1 s → dropped | Real repeats deleted | Common in drama dialogue | Drop only true echoes; log | High | Fixed |
| 7 | Line building | `drop_prompt_echo`: any line contained in the name prompt → dropped | A line that is one name (someone calling) deleted | Series users lose short lines | Only 2+ names; one name kept ⚑ | High | Fixed |
| 8 | Translation memory | TM hit reused as is | Old wrong terms / names reused forever | Glossary fixes never reach older lines | Check hits against the glossary | High | Fixed |
| 9 | Process | `dist/` holds a copy of `app/` | Fixes not in the portable build until rebuilt | Users run the old code | Rebuild after changes | High | Open (run `build_portable.py`) |
| 10 | Process | Git repository with **no commits**; videos, build and dist folders untracked inside it | No history, no diff, no rollback | Hard to review or undo changes like these | Commit `app/`, docs and scripts; ignore `build/`, `dist/`, videos | Medium | Open |
| 11 | Translation | Foreign-script leftovers only logged | Voice skips those words silently | Missing words in the dub | ⚑ + step note | Medium | Fixed |
| 12 | Translate tab | Dialogue polished as "cinematic narration" from drafts only | Embellishment, meaning drift | Subtitles should be faithful | `kind="dialogue"` polish with originals | Medium | Fixed |
| 13 | Reliability | Deepgram request not retried | One network blip fails a 2-h upload | Lost time, user frustration | Retry 429 / 5xx / connection | Medium | Fixed |
| 14 | Security | Gemini key sent as `?key=` query; `requests` errors quote the URL into the log and UI | Key could leak into `app.log` / screenshots (none found in your logs) | Key exposure | Redact URL keys (done); better: send the `x-goog-api-key` header | Medium | Mitigated |
| 15 | Review | No way to remove a line nobody says (emptied lines are re-translated) | Invented lines stay in the dub | Needed once recovery adds lines | "Nobody says this" (`DubLine.skip`, indices kept) | Medium | Fixed |
| 16 | Overlapping speech | One line per time span; Deepgram diarization has no overlap; the second listen skips covered time | Two people talking at once → one is lost | Shouting matches, crowd scenes | Overlap detection (pyannote, needs PyTorch + HF token) only if common in your content | Medium | Open |
| 17 | Subtitle timing | Cue = line start/end | A whole translation can be long for a short cue | Hard to read in fast exchanges | Extend a cue into the following pause, or split long cues | Medium | Open |
| 18 | CPU users | Auto picks Whisper `small` on CPU | More wrong characters (30 partial lines vs 22 for medium) | Translation of misheard words | Offer `medium` (int8) or `large-v3-turbo` on CPU; measure speed first | Medium | Open |
| 19 | Lines tab | Edited line timed with the generic 12 chars/s estimate | Status disagreed with Fit's Khmer model | Confusing ⚠ / ✔ | Use `speech_seconds` | Low | Fixed |
| 20 | Translate fallback | Per-line fallback kept only the reply's first line | Second half of a two-line reply lost | Rare | Join reply lines, skip notes | Low | Fixed |
| 21 | Memory | Whole speech track as float32 in RAM, plus faster-whisper's VAD copy | ~1 GB peak for a 2-h episode | Fine on 8 GB+, tight below | Stream / memory-map for very long videos | Low | Open |
| 22 | Security | Base64 fallback when DPAPI fails (logged) | Keys only obfuscated | Rare on Windows | Refuse to save or warn in the UI | Low | Open |
| 23 | Maintainability | `dub_engine.py` ≈ 6,000 lines, `dub_tab.py` ≈ 2,300 | Hard to change safely | Self-tests help a lot | Split into steps / modules over time | Low | Open |

No `shell=True`, no `eval`, no disabled TLS checks were found; subprocesses
use argument lists and `CREATE_NO_WINDOW`.

---

## 4. Part 2 — Missing sentences in fast speech: the checklist

| Question | Answer (measured / read in code) | Action |
|---|---|---|
| Does the ASR fail on fast speech? | Partly. 2-h drama, Deepgram: fast lines 86.7 % vs 94.5 % overall. 20 min, Whisper large-v3: 93.3 % (fast) | VAD 0.35 + second listen: 94.2 % |
| Is audio divided into chunks incorrectly? | The app does not chunk. faster-whisper merges VAD speech into ≤ 30 s windows and continues from the last timestamp; no words lost at window edges were found | Coverage check catches any loss anyway |
| Sentences cut between chunks / overlap sufficient? | No fixed chunks, so no overlap setting exists; decoding is sequential (seek), not chunk-parallel | — |
| Short segments filtered out? | Yes: `drop_repeats`, `drop_prompt_echo` (subtitle OCR also leaves out a lone particle beside a subtitle — correct, logged) | Fixed (#6, #7) |
| Confidence thresholds too aggressive? | VAD 0.5: yes. Whisper's no-speech window skip did not fire in the tests (small and large-v3 on separated vocals), but it is silent when it does | VAD 0.35; second listen covers skipped windows |
| Timestamps causing discards? | No discard by time. `_unstretch` trims stretched words; a word stretched over a gap can hide that gap from the second listen (limitation) | — |
| Async tasks failing? | Translation batches retry, split on count mismatch, fall back to the source text; TTS retried in rounds; AI outages pause the task. Deepgram had no retry | Fixed (#13) |
| Translation API receives every segment? | Yes, every line is sent or taken from memory — the loss happened after translation (Fit) | Fixed (#1–3) |
| Sample rate / channels? | 44.1 kHz stereo with 6 dB headroom → voice separation → 16 kHz mono for ASR: correct for Whisper and Silero | — |

Second-listen results (20 min, 448 subtitle lines, large-v3):

| | char recall | lines missed | fast-line recall | time (RTX 5070 Laptop) |
|---|---|---|---|---|
| before | 93.9 % | 32 | 93.3 % | 75 s |
| VAD 0.35 | 94.5 % | 28 | 94.2 % | 72 s |
| VAD 0.35 + second listen | **94.6 %** | **25** | 94.2 % | 91 s |

Of the 25 still "missed": 9 are on-screen text nobody says (numbers,
signs, game-system cards), 11 are interjections (嗯 / 哦 / 哎 / 嘿) that
Whisper tends to leave out, and 5 are real lines (顾川, 不…不会吧, 快快快,
亲自做的, one under the "system" voice) whose time the recogniser gave to
neighbouring words, so no gap shows. Short stretches that still have no
words are now listed in the transcribe note ("at 1:18, …").

Guarding against invented text was the hard part: on a 0.8 s gap under a
"嗯", Whisper wrote a different YouTube outro each run ("能收到最新消息哦",
"阅我的频道", "鈴鐺我們下期見by bwd"). The refusal rules were tuned until
three runs in a row gave the same result with none of those accepted; finer
gap settings (0.1–0.15 s padding) recovered nothing more and let more
outros through, so they were rejected.

---

## 5. Part 3 — Audio processing

| Setting | Now | Assessment | Recommendation |
|---|---|---|---|
| Extraction | ffmpeg → 44.1 kHz stereo PCM, −6 dB | Headroom prevents clipping of loud masters | Keep |
| Separation | Demucs htdemucs (or MDX-Net ONNX) | Big help for VAD/ASR under music | Keep on for dramas; off only for voice-only sources |
| ASR input | 16 kHz mono from the separated vocals | Native rate for Whisper / Silero | Keep |
| Normalisation | None before ASR | Whisper's features are level-independent; the missed spoken lines measured −14 to −27 dB RMS, so level was not the cause | Do not add without a measured gain |
| VAD | Silero, threshold **0.35**, 2 s silence, 400 ms pad | 0.30 added doubtful text; 1 s silence gained nothing | As set |
| Gap check | VAD 0.4, ≥ 0.4 s, 0.25 s word padding | Tighter settings recovered nothing more | As set |
| Chunking | faster-whisper ≤ 30 s windows over VAD speech | No loss at edges observed | Keep |

---

## 6. Part 4 — Speech recognition engines

Measured on the same 20 minutes (448 subtitle lines):

| Engine | char recall | lines missed | fast-line recall | runs on | cost |
|---|---|---|---|---|---|
| Deepgram nova-3 (as used) | 94.7 % | 22 | 93.9 % | cloud | per audio minute (check current price) + upload time |
| Whisper large-v3, new settings | 94.6 % | 25 | 94.2 % | NVIDIA GPU, ~4.5 GB VRAM fp16 | free |
| Whisper medium, new settings | 94.7 % | 24 | 93.9 % | GPU ≥ 3 GB or slow CPU | free |
| Whisper small, old settings | 93.2 % | 33 | 92.3 % | CPU default | free |
| Whisper small, new settings | 94.0 % | 24 | 93.3 % | CPU default | free |

- **Keep both engines.** Deepgram had the fewest misses and gives speaker
  labels; local Whisper is free, private and now nearly as complete. The
  second listen also runs after Deepgram when a Whisper model is installed
  (it recovered 4 of Deepgram's 22 misses in the probe).
- **WhisperX** adds wav2vec2 word alignment and pyannote diarization on top
  of the same Whisper. It would sharpen word times (useful for lip-sync and
  for the gap check) but not recognise more speech; it needs PyTorch,
  transformers and a Hugging Face token (~2–3 GB). Not recommended now.
- **large-v3-turbo** (4 decoder layers) decodes several times faster than
  large-v3 at slightly lower accuracy — the most promising upgrade for CPU
  users over `small`. Not measured (1.6 GB download); worth a test run.
- Model size mattered less than expected for *missing* lines; it matters
  for *wrong* characters (small: 30 partially wrong lines, medium 22).

---

## 7. Part 5 — Khmer translation quality

Already strong (kept): context of 6 previous and 3 next lines, story
context, glossary, speaker gender and relationships, Khmer pronoun and
politeness rules, Chinese-drama term hints, two passes (translate + review),
a back-translation meaning check, spelling fixes, script checks, translation
memory with human corrections first.

Fixed in this round: shortening that summarised (#1), subtitles from cut
text (#2), memory bypassing the glossary (#8), narration-style polish of
dialogue (#12), leftovers in foreign script (#11).

Recommended next:
1. For lines marked ⚑ glossary, one targeted re-translation call (like the
   foreign-script fix) instead of only flagging.
2. Keep a small reference set (50–100 lines from different dramas with a
   human Khmer translation) and run it after prompt changes: the meaning
   check's fix rate and human ratings are the only reliable quality signal.
3. Pass "uncertain" (⚑) to the translator so it prefers the evident meaning
   over a doubtful word.
4. Cantonese and other dialects: set the spoken language explicitly
   (large-v3 knows `yue`) rather than auto-detect.

---

## 8. How the numbers were produced

Scripts (not part of the app) compared, for each project: the raw
transcript with the on-screen subtitles (pinyin per character, ±1 s; a
subtitle with < 34 % found counts as missed), each final dub line with its
full translation in the translation memory (read-only), and the exported
SRT with the SRT the new code writes. Whisper runs used the app's own
`WhisperEngine` on the saved speech tracks with large-v3, medium and small
on the RTX 5070 Laptop GPU.
