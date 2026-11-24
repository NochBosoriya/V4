# RecapStudio — Architecture Contract (v1)

Every module MUST conform to this contract exactly. Other modules are built in
parallel against these signatures — do not rename, re-scope, or "improve" the
public API. Internals are yours; the surface is fixed.

## Product

Standalone Windows 10/11 desktop app: automated drama/movie recaps, multilingual
AI dubbing (Khmer-first), poster translation/design. **No licensing, activation,
trials, telemetry, or DRM anywhere — every feature unlocked forever.**

## Global rules

- Python 3.13. App is a package rooted at the repo (the folder holding `main.py`).
  Entry point `main.py` (repo root) → `from app.ui_main import run; run()`.
  All internal imports absolute: `from app.core.config_manager import ConfigManager`.
  Every package dir has `__init__.py` (may be empty).
- **Core/engine modules contain NO Qt imports.** Only `app/ui*` may import PySide6.
- **Heavy deps are lazy-imported inside functions/methods**, never at module top:
  torch, faster_whisper, demucs, paddleocr, cv2, onnxruntime, librosa, yt_dlp,
  edge_tts, uharfbuzz. Top-level imports allowed: stdlib, requests, PIL, numpy.
  A missing heavy dep raises `EngineNotAvailable(bilingual message)` — never a
  bare ImportError at app start.
- **Long operations** accept `progress: Callable[[float, str], None] | None`
  (fraction 0..1, status text) and `cancel: threading.Event | None`, poll cancel
  frequently, and raise `CancelledError(RuntimeError)` (defined in
  `app.core.errors`) when set. They are plain blocking functions — the UI wraps
  them in workers.
- **Errors:** `app/core/errors.py` defines:
  ```python
  class AppError(Exception):
      def __init__(self, en: str, km: str = ""): ...  # .en, .km, str() -> en
  class CancelledError(RuntimeError): ...
  class EngineNotAvailable(AppError): ...
  class ProviderError(AppError):  # + .provider, .status_code, .retryable
  ```
- **Logging:** `app/core/logger.py`:
  ```python
  get_logger(name: str) -> logging.Logger   # rotating file + memory handlers
  LOG_BUS: "LogBus"                          # singleton
  class LogBus:  # thread-safe pub/sub of log lines for the UI console dock
      def subscribe(self, fn: Callable[[str], None]) -> None
      def unsubscribe(self, fn) -> None
      def publish(self, line: str) -> None   # also called by a logging.Handler
      def tail(self, n: int = 500) -> list[str]
  ```
  Rotating file: `paths.app_data_dir()/logs/app.log`, 5 MB × 5 backups.
  Subprocess runners (ffmpeg/demucs/yt-dlp) must stream each stdout/stderr line
  to `LOG_BUS.publish()`.
- **Paths:** `app/core/paths.py`:
  ```python
  APP_NAME = "RecapStudio"
  def app_root() -> Path        # repo root in dev; exe dir when frozen (sys.frozen)
  def app_data_dir() -> Path    # %LOCALAPPDATA%/RecapStudio (created)
  def models_dir() -> Path      # app_data_dir()/models (created)
  def bin_dir() -> Path         # app_root()/bin
  def ffmpeg_path() -> str      # bin/ffmpeg.exe if exists else "ffmpeg"
  def ffprobe_path() -> str
  def fonts_dir() -> Path       # app_root()/app/assets/fonts
  def locales_dir() -> Path
  def projects_dir() -> Path    # app_data_dir()/projects (created)
  ```
- **Subprocesses on Windows:** always `creationflags=subprocess.CREATE_NO_WINDOW`,
  `encoding="utf-8", errors="replace"`. Kill process tree on cancel.
- **i18n:** `app/core/i18n.py`:
  ```python
  def set_language(code: str) -> None    # "en" | "km", persisted via ConfigManager
  def get_language() -> str
  def tr(key: str, default: str | None = None) -> str
  # Looks up key in app/locales/{lang}.json; falls back to en.json, then
  # `default`, then the key itself. UI code calls tr("settings.check_keys",
  # "Check Keys (Live)") — so the app works even before km.json is filled.
  ```
- No feature may block the UI thread; no `time.sleep` loops in UI code.
- Write COMPLETE production code. No placeholder bodies, no `TODO`, no
  `NotImplementedError` except where this contract explicitly allows a fallback.

## Module surfaces

### app/core/config_manager.py
SQLite at `app_data_dir()/config.db`. API keys encrypted with Windows DPAPI via
ctypes `CryptProtectData`/`CryptUnprotectData` (no pywin32); if DPAPI fails,
base64-obfuscate and log a warning.

```python
PROVIDERS = ["deepseek", "gemini", "openai", "anthropic", "custom"]
FEATURES  = ["translate_draft", "translate_polish", "recap", "poster"]

@dataclass
class KeyInfo:  key: str; provider: str; status: str  # "unknown"|"valid"|"invalid"|"exhausted"|"cooldown"
                latency_ms: int | None; last_error: str; cooldown_until: float  # epoch
@dataclass
class Route:    provider: str; model: str; fallback_provider: str | None; fallback_model: str | None

class ConfigManager:                      # thread-safe (internal lock)
    def __init__(self, db_path: Path | None = None): ...
    def get(self, key: str, default=None); def set(self, key: str, value) -> None
    # generic JSON-serialized settings, incl. "ui.language", "provider.<p>.base_url",
    # "provider.<p>.enabled", "duck.db", "export.presets", etc.
    def add_keys(self, provider: str, keys: list[str]) -> int      # dedup, returns added
    def remove_key(self, provider: str, key: str) -> None
    def get_keys(self, provider: str) -> list[KeyInfo]
    def import_keys_txt(self, provider: str, path: str) -> int     # one key per line
    def set_key_status(self, provider, key, status, latency_ms=None, error="") -> None
    def get_route(self, feature: str) -> Route
    def set_route(self, feature: str, route: Route) -> None
    def record_usage(self, provider, model, in_tokens: int, out_tokens: int) -> None
    def get_usage(self) -> list[dict]     # rows: provider, model, in_tokens, out_tokens, est_cost
    def export_settings(self, path: str, include_keys: bool = False) -> None
    def import_settings(self, path: str) -> None
    @property
    def key_pool(self) -> "KeyPool"

class KeyPool:   # round-robin per provider, skips cooldown/invalid/exhausted
    def acquire(self, provider: str) -> str          # raises ProviderError if none usable
    def report_success(self, provider: str, key: str, latency_ms: int) -> None
    def report_failure(self, provider: str, key: str, status_code: int | None,
                       message: str = "") -> None
    # 401/403 -> invalid; 429 -> cooldown 60s (doubling per repeat, cap 15 min);
    # insufficient_quota/exhausted message -> exhausted (retried after 1 h);
    # 402 / credit message (is_credit_error) -> exhausted, retried after 60 s
    # (a refusal costs nothing; the user may have topped up); 5xx/network ->
    # cooldown 15s. All keys out of credit: acquire() raises a plain message
    # (no_credit_message: "Your OpenRouter keys are out of credit … Add credit
    # at openrouter.ai → Credits, or add a key of another AI provider…"),
    # status 402, exc.plain = True.
```

Default base URLs (user-overridable via settings):
deepseek `https://api.deepseek.com`, openai `https://api.openai.com/v1`,
gemini `https://generativelanguage.googleapis.com/v1beta`,
anthropic `https://api.anthropic.com/v1`, custom user-provided.

### app/core/ai_engine.py
`requests` only, no provider SDKs. deepseek/openai/custom speak
OpenAI-compatible `/chat/completions` + `/models`; gemini native REST
(`models/{m}:generateContent`, `?key=`); anthropic native (`/messages`,
`x-api-key`, `anthropic-version: 2023-06-01`, `/models` for listing).

```python
@dataclass
class ChatResult: text: str; in_tokens: int; out_tokens: int; provider: str; model: str
@dataclass
class KeyCheckResult: key: str; valid: bool; latency_ms: int; message: str; quota: str

class AIEngine:
    def __init__(self, config: ConfigManager): ...
    def chat(self, *, feature: str | None = None, provider: str | None = None,
             model: str | None = None, messages: list[dict], system: str | None = None,
             temperature: float = 0.7, max_tokens: int = 4096, json_mode: bool = False,
             cancel: threading.Event | None = None,
             images: list[tuple[str, bytes]] | None = None) -> ChatResult
    # edit_image(prompt, (mime, bytes), aspect_ratio="", cancel) -> ImageResult
    # (data, mime, provider, model, note): a picture edited by a Gemini image
    # model (2026-11-21, Replace title — same style). image_models(): the
    # Gemini key's own list (names with "image", not Imagen; 3.1 Flash Image,
    # 2.5 Flash Image, 3 Pro Image — cheapest first; a fixed list when listing
    # fails), then google/gemini-*-image through OpenRouter; setting
    # poster.image_model ("provider/model") goes first. Gemini:
    # generateContent with inline_data + text, generationConfig
    # responseModalities ["TEXT","IMAGE"] + imageConfig.aspectRatio (dropped
    # on a 400 about it), the picture from parts[].inlineData. OpenRouter:
    # modalities ["image","text"], message.images[].image_url data URL. Free
    # Gemini keys have NO picture quota (429 "limit: 0" / free_tier): such a
    # key is skipped for the other models and NOT cooled down (text work goes
    # on); all keys like that -> a plain "needs billing (about US$0.04 a
    # picture) or OpenRouter credit" ProviderError. Prices: approximate image
    # model rows in DEFAULT_PRICES.
    # images [(mime, bytes)]: pictures sent with the last user message
    # (Gemini inline_data, OpenAI-style image_url data URL, Anthropic image
    # block; 2026-11-21, the poster title). A model that cannot see pictures
    # refuses: a request error (the caller falls back), no key cooldown.
    # feature -> Route from config (primary then fallback). Key rotation via
    # KeyPool; on retryable failure: backoff 1s,2s,4s (max 3 tries per key),
    # then next key; exhausts all keys of primary -> fallback provider ->
    # the provider's fallback models -> every other provider that has a key
    # (its Settings model, default provider first) -> ProviderError(bilingual).
    # When all keys of a provider are cooling down / out of credit and a later
    # provider has a usable key, it switches at once (no 90 s cooldown wait).
    # ai_unavailable(exc): no AI could answer (limit, credit, budget, offline);
    # dub passes then stop the task ("paused", resumable) instead of skipping.
    # No key for any provider: ProviderError(NO_KEY_EN, provider="no_key")
    # before any route is tried. ai_failure_kind(exc) names the cause (no_key,
    # budget, bad_key, no_credit, busy, timeout, offline); AI_ADVICE says what
    # to do about each (en, km). A pool error with .plain (out of credit) is
    # raised as it is, without the "All providers failed" wrapper; the first
    # one comes first, a later provider's reason follows in brackets. The dub's
    # ai_pause_error uses a plain message instead of AI_ADVICE.
    # OpenRouter 402 "can only afford N" (it reserves max_tokens × price):
    # resent once with ~0.97·N max_tokens when that is ≥ 512; an answer then
    # cut short (finish_reason "length") is a 402 (out of credit).
    # Records usage into config. json_mode: request
    # JSON output (response_format where supported, else prompt instruction)
    # and salvage-parse the reply (strip code fences).
    def list_models(self, provider: str, key: str | None = None) -> list[str]
    def check_key(self, provider: str, key: str) -> KeyCheckResult   # 1 cheap call
    def check_keys_live(self, provider: str, progress=None, cancel=None) -> list[KeyCheckResult]
    # threads, updates config key statuses
    def translate_two_pass(self, texts: list[str], target_lang: str,
                           style: str = "", context: str = "",
                           progress=None, cancel=None) -> list[str]
    # pass1 feature="translate_draft", pass2 "translate_polish"; batches ~30
    # lines per call with numbered-list protocol; parse strictly, retry on
    # count mismatch.
def extract_json(text: str) -> Any   # module-level tolerant JSON extractor
```
Cost table: keep a small editable dict of $/1M token prices with a
`"default": (0.5, 1.5)` fallback; unknown models use default — never crash.

### app/engines/audio_pipeline.py
```python
@dataclass
class Word:    start: float; end: float; text: str
@dataclass
class Segment: start: float; end: float; text: str; words: list[Word]
               speaker: str | None = None; gender: str | None = None  # "M"|"F"
@dataclass
class Transcript: segments: list[Segment]; language: str; duration: float
                  def to_srt(self) -> str;  def to_json(self)/from_json(d) classmethod

def extract_audio(video: str, out_wav: str, sr: int = 16000, mono=True,
                  progress=None, cancel=None) -> str          # ffmpeg
def separate_vocals(audio: str, out_dir: str, model: str = "htdemucs",
                    device: str = "auto", progress=None, cancel=None) -> tuple[str, str]
# returns (vocals_wav, accompaniment_wav). Runs `sys.executable -m demucs.separate
# --two-stems vocals` as subprocess with env TORCH_HOME=models_dir()/demucs,
# streams stderr lines to LOG_BUS and parses N% progress. `-j demucs_jobs()`
# (2026-11-20): pieces separated side by side, a quarter of the CPU threads,
# at most 8, one per 4 GB of RAM (60 s on 32 threads: 26.5 s → 20.0 s with 8).
class WhisperEngine:
    def __init__(self, model_size: str = "small", device: str = "auto"): ...
    # faster_whisper, download_root=models_dir()/whisper, CUDA→CPU fallback
    def transcribe(self, audio: str, language: str | None = None,
                   word_timestamps: bool = True, progress=None, cancel=None) -> Transcript
def detect_gender(wav: str, start: float, end: float) -> str   # "M"/"F" via
# numpy autocorrelation f0 on the slice (median f0 < 165 Hz -> M)
def assign_voices(segments: list[Segment], lang: str = "km",
                  male_voice: str = "km-KH-PisethNeural",
                  female_voice: str = "km-KH-SreymomNeural") -> None  # sets .gender/.speaker

TTS_ENGINES = ["edge", "voxcpm", "f5tts", "cosyvoice", "openai", "elevenlabs"]
def list_edge_voices(lang_prefix: str = "") -> list[dict]      # cached edge-tts list
def synthesize(text: str, out_wav: str, engine: str = "edge", voice: str = "",
               rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%",
               emotion: str = "", api_key: str = "", config=None,
               progress=None, cancel=None) -> str
# edge via edge_tts asyncio (mp3->wav via ffmpeg); openai/elevenlabs via
# requests using keys from config; local engines raise EngineNotAvailable
# with install hint if package missing.
def fit_to_duration(wav: str, target: float, min_tempo=0.85, max_tempo=1.35) -> str
# ffmpeg atempo (chained); if still long, return as-is (caller shifts gaps)
def build_voice_track(lines: list[tuple[float, str]], total: float, out_wav: str,
                      voice_for_line: Callable[[int], dict], progress=None,
                      cancel=None) -> str   # synth each line, fit, place at offsets
def mix_with_ducking(voice_wav: str, bgm_wav: str, out_wav: str,
                     duck_db: float = -18.0, fade_ms: int = 180,
                     bgm_gain_db: float = 0.0, progress=None, cancel=None) -> str
# ffmpeg sidechaincompress (threshold/ratio/attack/release mapped from args)
def get_duration(path: str) -> float   # ffprobe
```

### app/engines/recap_engine.py
```python
@dataclass
class TimelineRow: start: float; end: float; title: str; narration: str
                   visual_plan: str; onscreen_text: str
@dataclass
class RecapProject: source_video: str; language: str; rows: list[TimelineRow]
                    beats: list[dict]; transcript: Transcript | None
                    def to_json(self)/from_json(d); save(path); load(path) classmethod

SPEECH_CPS = {"km": 12.0, "en": 15.0, "zh": 5.5, "th": 9.0, "vi": 14.0,
              "ko": 6.5, "ja": 7.0, "fr": 15.0, "es": 15.5, "hi": 12.0}
def estimate_speech_seconds(text: str, lang: str) -> float
def check_row_timing(row: TimelineRow, lang: str) -> tuple[bool, float]  # (fits, needed_s)

class RecapEngine:
    def __init__(self, ai: AIEngine): ...
    def analyze_beats(self, transcript: Transcript, progress=None, cancel=None) -> list[dict]
    # chunk transcript (~4000 chars, timestamped lines), feature="recap",
    # json_mode; merge chunk beats. Beat: {start,end,summary,importance 1-5,cues}
    def write_recap(self, beats, transcript, target_lang: str, target_minutes: float = 3.0,
                    style: str = "cinematic", progress=None, cancel=None) -> list[TimelineRow]
    # Pass 1 draft (feature="recap"), Pass 2 polish (feature="translate_polish",
    # prompt tuned for cinematic Khmer storytelling rhythm/idioms when km).
    # Enforce speech-timing per row after polish (ask model to shorten overlong
    # rows once; then flag).
    def full_pipeline(self, video: str, *, whisper_size="small", use_demucs=True,
                      target_lang="km", target_minutes=3.0, progress=None,
                      cancel=None) -> RecapProject
```

### app/engines/poster_designer.py
```python
@dataclass
class OcrBox: x: int; y: int; w: int; h: int; text: str; conf: float
    alts: tuple = ()   # other readings of the same text (read at other sizes)
@dataclass
class StyleSpec:  # all fields have defaults; from_llm_json(d) classmethod tolerant
    font: str = "Moul"; size_ratio: float = 0.12       # of image width
    fill: str = "#ffffff"; gradient: list[str] | None = None   # 2-4 hex stops, top to bottom of each line
    stroke_color: str = "#000000"; stroke_width_ratio: float = 0.04  # of font size
    shadow_color: str = "#000000cc"; shadow_offset: tuple[int,int] = (4,4); shadow_blur: int = 8
    glow_color: str | None = None; glow_radius: int = 0
    bevel: bool = False; extrude_depth: int = 0; extrude_color: str = "#222222"
    pos: tuple[float, float] = (0.5, 0.82)  # center, fraction of W/H
    letter_spacing: int = 0; rotation: float = 0.0
    backdrop: bool = False; backdrop_color: str = "#0b0805e8"  # ink-brush band behind each line
    sparkles: int = 0                                        # small glints (max 12 drawn)
    line_break: str = ""   # "" as typed ("|" = new line) | "colon" | "auto" (colon, else middle space)
    line_scales: list[float] | None = None                   # size of each line, 1.0 = largest
    align: str = "center"  # the lines: "center" | "left" | "right" (pos stays the block's centre)

def detect_title_boxes(image_path: str, progress=None, cancel=None) -> list[OcrBox]
# PaddleOCR lazy (lang="ch" handles zh+en); sorted by area*conf desc.
# When PaddleOCR cannot load (blocked by Smart App Control / missing) or
# cannot read (PaddlePaddle 3 CPU: "(Unimplemented) Convert…"), the
# built-in reader does it: subtitle_ocr.read_image (PP-OCRv5 ONNX). The
# verdict (usable or not) is kept in models/paddleocr/unusable.json with
# PaddleOCR's version: a new version is tried again. While it is not known
# the reader answers at once and PaddleOCR is tried in a background thread
# (_start_paddle_probe). 3.x runs without document unwarping / page
# orientation. AppError only if both fail.
# The reader looks at copies of 1280, 640 and 352 px (_reader_sides): a big
# decorative title (神豪人生, DATA ANALYSIS) is missed on the full picture
# and found whole on the small copies; _merge_reads keeps one box per text
# (the surer, bigger one; the other readings in .alts) and drops a box
# over two stacked lines in favour of the lines (2026-11-21).
def inpaint_region(image_path: str, boxes: list[OcrBox], out_path: str,
                   method: str = "auto", progress=None, cancel=None) -> str
# "auto": LaMa onnx (models_dir()/lama/lama_fp32.onnx via onnxruntime) if
# present else cv2.inpaint TELEA; mask = boxes dilated 6px. The usual LaMa
# export takes only 512x512 (fed a whole poster it refused and OpenCV's fill
# was used silently until 2026-11-20): each part of the mask is filled from a
# square around it with room on every side, shrunk to 512 when larger
# (_inpaint_lama); the session is loaded once (_lama_session). 2026-11-21:
# the masked pixels are blanked before LaMa (as it was trained), the mask
# is shrunk with INTER_AREA (any covered pixel stays masked; NEAREST
# skipped bits of thin strokes) and mirrored with the picture past its
# edge (the mirror copy of a title at the edge was drawn back as a faint
# ghost). Sharper (the user: the erased area looked smudged): a hole is
# filled at the picture's own resolution in overlapping squares
# (_lama_square), shrunk only as far as its thickness needs (scale =
# 0.12 * 512 / thickest distance into it; one square when it fits), then
# _restore_detail sharpens the filled area (unsharp mask) until its edges
# are 0.6 as crisp as the picture's around it (Laplacian energy; at most
# 2.5x). Grain copied from around was tried and dropped: it drew scribbles.
def replace_title(image_path, text, out_path, boxes=None, fonts_dir=None,
                  as_title=False, progress=None, cancel=None, grouped=False,
                  layout=False) -> dict
# {"found", "clean", "spec", "boxes", "area", "text"}: group_title_boxes (the most prominent
# text >= 3.5% of the height + lines next to it of a similar size) ->
# _title_layer (the letters by colour against what surrounds the title, an
# ink band when the gaps and the area a little away from the letters are
# dark, the erase mask: letters + outline + glow + band; the scene inside
# the boxes is kept) -> _title_style (fill / 2-3 stop gradient top to bottom,
# outline colour and thickness, band, glow beyond the outline, sparkles,
# angle, line sizes, place and size via fit_size_ratio; the font:
# _title_face — Moul for strokes that swell and thin (brush), Koulen for
# heavy even strokes (block letters), Kantumruy Pro for light even ones) ->
# _inpaint_array. No title: gold_brush_spec() at the lower middle. as_title
# (boxes the user drew): those boxes are the title and are erased whole;
# grouped (read_headline): the boxes are the title, only its letters go.
# layout: the text laid out by _best_layout ("text" has its lines).
# title_text(image_path): the title as the reader reads it (rows joined).
# The letters are checked by reading them (_letters_read_checked): drawn
# black on white they must read as the title; when the colour mask reads
# under 0.35 (white letters by a white hoodie: it held the navy jacket) the
# colour groups of the boxes that read best are taken (not masks reaching
# the box edges inside the picture — the space around letters reads too;
# an outline's inside when the best is an outline). On a plain background
# (_smooth_on_plain: detail < 6 grey levels) each line's erase is one area.
# The seed takes ">=" the 80th percentile (pure white letters tie there).
def gold_brush_spec() -> StyleSpec    # metallic gold, dark outline, warm glow
# (gold_glow_radius: a tenth of the letter size), ink band, sparkles, -3°
def translate_title(title: str, ai: AIEngine, target_lang: str = "km",
                    genre: str = "", cancel=None) -> list[str]        # exactly 3 options
def read_headline(image_path, ai, target_lang="km", boxes=None, progress=None,
                  cancel=None) -> dict
# The main title read and translated in one AI request (feature "poster",
# json): the poster as JPEG (ai.chat images=) + the reader's texts with
# places, sizes and alternative readings. Reply: status ok | none | already
# | unreadable, headline (lines as printed), language, parts (the reader's
# texts that make the title -> _pick_title_boxes, else the AI's box
# 0-1000 [ymin, xmin, ymax, xmax], else group_title_boxes), meaning
# (English), translation (lines). A model that cannot see pictures (request
# error) gets the reader's texts only ("seen": False); no AI at all
# (ai_unavailable) raises. Khmer: km_order.rule in the prompt, km_order.fix
# after, Khmer digits, one more request when CJK letters are left.
def auto_translate_title(image_path, out_path, ai, target_lang="km",
                         fonts_dir=None, progress=None, cancel=None) -> dict
# read_headline -> replace_title(grouped=True) -> _best_layout (the AI's
# lines, one line or two balanced ones: the biggest letters in the old
# title's place, 1.1x wide / 1.35x tall at most; line sizes matched by how
# tall the letters look (_matched_scales); flush left / right like the old
# lines (_title_align); kept on the picture) -> render_title (PNG, same
# size; out_path "" = not drawn) -> _title_checks: removed (the reader finds
# nothing of the old title there), letters (no CJK, none the font lacks),
# numbers (numbers.check), shaping (RAQM or uharfbuzz), size.
def auto_translate_titles(paths, out_dir, ai, target_lang="km", fonts_dir=None,
                          overwrite=True, progress=None, cancel=None) -> list[dict]
# each picture on its own -> <out_dir>/<name>_KH.png (output_name); errors
# per picture; no AI at all stops the batch ("stopped").
def ai_replace_title(image_path, text, out_path, ai, source="", boxes=None,
                     target_lang="km", verify=True, progress=None, cancel=None) -> dict
# The title lettered by the AI picture model (ai.edit_image) in the original
# title's art style — no fonts (the user's Doubao example). The prompt
# (_edit_prompt): the text letter for letter (Khmer subscripts and vowel
# signs exactly), the original's art style, the old title gone, nothing
# else changed; the poster as JPEG <= 1536 px + _nearest_aspect. The
# picture back (~1K) is lined up (_align_edit: resize, ORB + RANSAC
# similarity when it agrees) and only the title's part goes in (_merge_edit:
# the old title's zone + what changed next to it, colours matched on a ring,
# feathered; the rest stays the poster's own pixels at full size). Checks:
# spelling (_read_title_back: the AI reads the new title off the result;
# _same_letters), removed (the reader), size. {"out", "provider", "model",
# "note", "read", "checks", "region", "text"}.
def analyze_image_style(image_path: str) -> dict   # dominant colors (k-means, numpy),
# brightness, contrast — no LLM
def design_spec(image_stats: dict, title: str, genre: str, ai: AIEngine,
                cancel=None) -> StyleSpec          # feature="poster", json_mode
def render_title(image_path: str, text: str, spec: StyleSpec, out_path: str,
                 fonts_dir: Path | None = None) -> str
# Pillow; shape with uharfbuzz (required for Khmer subscripts ជើងអក្សរ):
# hb shape -> glyph ids/advances/offsets -> FreeType render via PIL ImageFont
# .getmask on a per-cluster basis is NOT sufficient; implement hb-buffer shaping
# and draw with font.render using features; fallback: PIL with
# layout_engine=ImageFont.Layout.RAQM if uharfbuzz missing; last resort BASIC
# with logged warning. Effects order: ink band -> shadow -> glow -> extrude ->
# stroke -> fill (solid or 2-4 stop gradient per line) -> bevel highlight ->
# sparkles. Lines: title_lines() + _line_scales() (a lead-in ending in ":" is
# 0.62 of the main line). fit_size_ratio(text, spec, box_w, box_h, img_w).
def export_variants(image_path: str, out_dir: str, base_name: str) -> dict
# {"original": p, "9x16": p, "16x9": p} — smart-crop center, blurred-pad extend
def find_font_file(name: str, fonts_dir: Path | None = None) -> str | None
# match bundled fonts by family name, then Windows fonts dir
```

### app/engines/media_exporter.py
```python
@dataclass
class MediaInfo: duration: float; width: int; height: int; fps: float
                 vcodec: str; acodec: str; bitrate: int
def probe(path: str) -> MediaInfo                     # ffprobe -print_format json
def detect_encoders() -> list[str]                    # test-encode 1 frame for
# h264_nvenc/hevc_nvenc/h264_qsv; always include libx264/libx265; cached
@dataclass
class OverlayConfig:
    watermark: str | None = None; watermark_opacity: float = 0.6   # logo picture
    watermark_pos: str = "top_right"      # GRID_POSITIONS (3x3) or "tile"
    watermark_scale: float = 0.15         # logo width / frame width (text: height x 6)
    watermark_text: str = ""; watermark_color: str = "auto"   # "auto" = white + shadow
    watermark_font: str = "Kantumruy Pro"
    watermark_full: bool = False          # internal: `watermark` is the finished layer
    # from_dict(): {"watermark_on": False} (the Overlay tab's switch) = no watermark
    banner: str | None = None; banner_height_ratio: float = 0.12
    chroma_video: str | None = None; chroma_color: str = "0x00ff00"
    chroma_similarity: float = 0.15; chroma_pos: str = "bottom_right"
    blur_regions: list[tuple[int,int,int,int]] = field(default_factory=list)
    blur_strength: int = 20
    ad_video: str | None = None; ad_start: float = 0.0; ad_pos: str = "top_left"
    ad_scale: float = 0.3
@dataclass
class ExportPreset: name: str = "Default"; container: str = "mp4"
    vcodec: str = "libx264"; bitrate: str = "6M"; width: int = 0; height: int = 0
    fps: float = 0.0; audio_bitrate: str = "192k"   # 0 = keep source
def build_filtergraph(src_info: MediaInfo, overlays: OverlayConfig,
                      srt_path: str | None, font_name: str = "Kantumruy Pro") -> str
# subtitles filter: filename escaped (\: and \' rules), fontsdir=fonts_dir(),
# force_style FontName
def render(video: str, out_path: str, preset: ExportPreset,
           overlays: OverlayConfig | None = None, srt_path: str | None = None,
           dub_audio: str | None = None, progress=None, cancel=None,
           subtitle_style: str = "", audio_lang: str = "",
           start: float = 0.0, duration: float = 0.0, speed: float = 1.0,
           poster: str = "", intro_image: str = "", intro_seconds: float = 0.0) -> str
# single ffmpeg pass; -progress pipe parsing out_time_us / duration for progress.
# The watermark (logo and/or text, a 3x3 place or tiled at 30°) is drawn once
# with PIL as a full-frame layer of the output size (prepare_watermark →
# watermark_layer, the same picture the Overlay tab's live preview shows via
# preview_frame) and laid over the video with overlay=0:0.
# poster: put into the result as cover art afterwards (cover_art.set_cover).
# intro_image/intro_seconds: that long a still of the picture (fitted over a
# blurred copy of itself, silence under it) before the video (_intro_graph;
# the speed change stays on the video's own part). The file comment says what
# poster the result carries (a copied "RecapStudio poster" comment is replaced).
# start/duration (source s, 0 = to the end): one part (an episode cut from a
# long video): -ss/-t on the video and the dub audio; subtitles copied and
# moved to the part's clock (shift_subtitles, .srt/.ass); the smart-crop
# track made for the part's shots only. speed (SPEED_MIN 0.25 – SPEED_MAX 4):
# setpts + fps (frame rate kept) at the end of the video graph, atempo chain
# (pitch kept) on the audio.
def plan_parts(duration, part_seconds, speed=1.0) -> tuple[int, float]
# (whole episodes, leftover s of finished video); within 0.5 s of a whole one
# counts as one, under LEFTOVER_MIN (1 s) as none
def part_spans(duration, part_seconds, speed=1.0, leftover="new"|"join")
    -> list[tuple[float, float]]    # (start, length) in source s
def cut_segments(video: str, segments: list[tuple[float,float]], out_path: str,
                 reencode: bool = True, progress=None, cancel=None) -> str

@dataclass
class RenderJob: id: str; label: str; kwargs: dict; status: str = "pending"
                 progress: float = 0.0; error: str = ""
class RenderQueue:   # persists state to app_data_dir()/render_queue.json
    def add(self, job: RenderJob) -> None
    def run(self, on_job_progress: Callable[[str, float, str], None] | None = None,
            cancel: threading.Event | None = None) -> None   # sequential, resumable
    def load(self) / save(self); jobs: list[RenderJob]
def save_preset(config, preset) / load_presets(config) -> list[ExportPreset]
```

### app/engines/dub_engine.py  (Full Dub tab)
Whole-video dubbing with one voice per character. State lives in
`DubProject` (`projects_dir()/dub/<stem>_<hash>/project.json`), so every step
is resumable; re-running a step marks all later steps pending.
```python
STEPS = [("prepare",…), ("transcribe",…), ("speakers",…), ("glossary",…),
         ("voices",…), ("translate",…), ("fit",…), ("synthesize",…),
         ("mix",…), ("export",…)]
# glossary runs before voices: the AI fixes speaker labels first (or, when
# voices are too alike — speaker_score < 0.15 — assigns every line from the
# dialogue), then fragments are merged and voice samples cut.
CHAR_ENGINES = ["auto", "edge", "edge+clone", "f5tts", "voxcpm", "cosyvoice",
                "openai", "elevenlabs"]
# "edge+clone" = Edge TTS in the target language -> OpenVoice tone-colour
# conversion to the character's reference clip (works for Khmer).
# "edge+seedvc" = same, with Seed-VC (optional, GPL-3, ~3 GB VRAM in fp32 on
# GTX 16xx, ~5 s per line at 10 steps) — convert_voice_seedvc().
# "auto" for Khmer = edge+seedvc when Seed-VC is installed and an NVIDIA GPU
# has >= 4 GB; otherwise edge+clone (OpenVoice).
@dataclass Character: id, name, gender, ref_wav, ref_text, ref_custom, engine,
                      voice, rate, pitch, volume, line_count, talk_seconds
@dataclass DubLine:   start, end, source, text, speaker, status, needed, tts_key
@dataclass GlossaryTerm: source, target, note
@dataclass DubOptions: target_lang, source_lang, whisper_size, use_demucs,
                       num_speakers, voice_mode, quality, max_tempo, ...
class DubProject: for_video(video, options) / load(path) / save();
                  steps, invalidate_from(step), next_step(), slot(i), max_chars(i)
                  relink(project_json, video): an earlier project for a moved
                  video (stays in its folder; names the new path)
def find_moved_project(video) -> str   # same file name + length (±1 s), old
# place empty; for_video also finds a relinked project by its "video" field
class DubEngine:
    def __init__(self, ai: AIEngine, config=None)
    def run_step(self, project, step, progress=None, cancel=None) -> DubProject
    def run_all(self, project, progress=None, cancel=None, on_step=None,
                stop_before="") -> DubProject
    def run_until_ai(self, project, progress=None, cancel=None)  # "Transcribe only":
    # the tasks before FIRST_AI_STEP (prepare, transcribe, speakers)
    # A task paused for want of AI (ai_pause_error) keeps the cause in
    # project.steps[step]["ai"]: the Dub tab shows it amber ("Paused") with
    # Open Settings. Every failed task keeps the engine's Khmer text in
    # steps[step]["note_km"] (a reopened project shows it in Khmer).
def project_json_for(video) -> str  # its project: own folder or relinked
    def synth_text(self, project, character, text, out_wav, cancel=None) -> str
def readiness(config, options=None) -> list[Check]   # what the pipeline needs,
# each with a fix hint and optional pip args / model download
def install_packages(pip_args, progress=None, cancel=None) -> str
def split_segments(segments) -> list[DubLine]; def cluster_speakers(X, num=0) -> labels
def build_reference(wav, lines, out_wav) -> (path, transcript) | None
def convert_voice(src_wav, src_sample_wav, ref_wav, out_wav, cancel=None) -> str
```
Speaker vectors: CAM++ (`app/engines/speaker_embed.py`, ONNX, model id
`speaker-campplus`, downloaded on first use), else speechbrain ECAPA, else
numpy MFCC/F0 statistics; spherical k-means with silhouette-chosen k, then
clusters whose mean voices are ≥ 0.7 alike merged. Long videos (≥ 60 lines of
≥ 1.5 s, CAM++, automatic count) use `group_long()` instead: average-linkage
grouping of those steady prints at 0.35 cosine, groups < 3 lines dropped,
every line to the nearest group mean, ≥ 0.7-alike means merged, at most 20
(silhouette picked 2 = men/women on 2-hour dramas). The speakers step logs
the count and lines per speaker. `DubLine.voice_sim` (likeness
to the line before) stops `merge_fragments` joining two voices.
With Deepgram's labels (≥ 90 % of lines labelled) `_relabel_by_print()` moves
a line ≥ 0.8 s to another label when its print is ≥ 0.4 alike to that
label's mean voice and 0.12 more alike than to its own (both voices from ≥ 2
other lines of ≥ 1 s); a moved line is `voice_sure`, and so is one that is
≥ 0.4 alike to its own label's voice and 0.12 more than to any other (the AI
speaker check leaves `voice_sure` lines where they are, as with CAM++ groups).
Print genders (`_print_genders`): the prints split in two (k = 2), the side
with the lower median pitch the men's, used when the sides differ like men
and women (< 165 / > 190 Hz, ratio ≥ 1.3). The lines up to 260 Hz
(`_CALM_F0`) are also split alone (`_calm_genders`; men's side < 165 Hz with
≥ 30 % of lines under 155 Hz, women's side > 185 Hz with ≤ 5 % under 135 Hz
and ≤ 15 % under 155 Hz, sides < 0.7 alike), each louder line taking the side
its print is ≥ 0.05 nearer. That split is used when the full one fails, or
names < 80 % of the calm lines the same: a shouting crowd can be the clearest
split (164-167 / 318-323 Hz on a drama, hero and heroine on one side).
Hard subtitles (`app/engines/subtitle_ocr.py`, PP-OCRv5 mobile ONNX, model id
`subtitle-ocr`): for zh / ja sources the transcribe step reads the subtitle
row and takes the words from it, keeping the recogniser's timing
(`apply_to_segments`); lines get `on_screen` and, when changed, `heard`.
A video of 4 min or more is read by `_readers()` readers at once (a quarter
of the CPU threads, at most 8, 2 ONNX threads each; more threads in one
reader were slower): the first `_CALIB_S` (90 s) by one reader, which finds
the subtitle row, then the rest in parts on the frame grid (`_part_spans`,
`_scan`); a 1-hour drama: 15.5 → 5.8 min, the same 1,148 subtitles.
One subtitle = one segment (`Segment.subtitle`) = one line: `split_segments`
does not cut it again at Deepgram's turns or pauses, and gives it the label
heard longest in it. Recognised words stretched over silence are capped
first (0.8 s + 0.3 s a character). A subtitle nobody was heard saying keeps
its on-screen time; a short one (≤ 4 characters) with no free time there
joins the words it overlaps with "，" ("肖少，何必…"). A second
read of a neighbouring subtitle (same sound, ≤ 0.3 s apart; 下注 / 下註) is
dropped, and a leading dash is the numeral 一 when the speech has one there.
Recognised words outside the subtitles that are only a particle (了吗呢吧嘛…)
within 0.5 s of a subtitle are dropped (傻了吧 heard as 傻了罢了).
Fit order (2026-11-16): spoken up to `COMFORT_TEMPO` 1.4× faster (also the
speed of translation budgets `max_chars()` and of lip-sync) → reworded more
briefly, meaning kept → worded more tightly, every statement kept
(`allow_cuts`: words cut as the last resort) → spoken faster still, just as
much as needed to end before the next line, up to `max_tempo` (voice rate
and stretch together; 2.0× by default, the Dub tab's "Max speed-up",
1.0-2.5×) → the next line starts a little late. Fit, `_too_long` and the
condense rooms decide at 1.4×; only `_respeak_faster` / `_place_lines` go
past it. (Speed first, 2026-11-15, put 12 lines of a drama at 1.6-2.0× that
the 2026-11-11 dub had said more briefly at 1.4× — the user preferred the
older one.) `DubLine.speed` records the speed-up each line got (Lines tab,
Fit column). Fit overwrites the
line's text, so a changed speed-up redoes the dub from Translate (the full
translations come back from the translation memory); "Dub whole video" on a
finished project first applies changed settings, then runs what they
invalidated.
`synthesize()` in audio_pipeline takes an optional `ref_text` for zero-shot
cloning engines.

### Added modules (v1.1)
- `app/core/gpu.py` — vendor-neutral adapters via DXGI (PowerShell fallback):
  `list_gpus()`, `primary_gpu()`, `accel_summary()` (what runs on CUDA /
  DirectML / CPU), `accel_hints()`, `onnx_providers()` (CUDA > DirectML > CPU).
  `model_manager.startup_report()` adds `gpu_vendor`, `gpus`, `accel`,
  `accel_hints`.
- `app/core/series_memory.py` — per-series glossary (per language), voice
  cast (name, gender, voice settings, copied voice sample, voice
  fingerprint) and story; `match_cast()` recognises characters by voice.
- `app/core/translation_memory.py` — SQLite TM keyed by (target lang,
  normalised source, gender, series); human fixes beat AI output.
- `app/core/publisher.py` — `generate_metadata()`, `YouTube` (OAuth loopback
  + PKCE, resumable upload, default private), `FacebookPage` (resumable).
  When to post (Publish tab, 2026-11-22): Post immediately (YouTube with the
  visibility chosen, Facebook published at once) or Post on schedule (a
  local date-time; uploaded now, YouTube `status.publishAt` RFC 3339 UTC with
  privacyStatus private, Facebook finish `published=false` +
  `scheduled_publish_time`; the platforms post it, the PC can be off).
  `schedule_problem()`: at least 15 minutes ahead, Facebook at most 6 months.
- `app/engines/scenes.py` — `detect_scenes()` (cached), `snap_start()`,
  `subject_x()` / `crop_track()` / `crop_x_expr()` for smart 9:16 crops.
- `app/engines/subtitles.py` — `.ass` presets (Khmer fonts, karaoke,
  Shorts safe zone), cluster-safe Khmer word splitting, `srt_to_ass()`.
- `app/engines/thumbnail.py` — `find_frames()`, `compose()`,
  `suggest_titles()`. Frames are written with `cv2.imencode` + Python file
  I/O (`cv2.imwrite` silently writes nothing into a folder with non-ASCII
  letters on Windows).
- `app/engines/numbers.py` (2026-11-21) — numbers in dub lines.
  `source_numbers(text, lang)` (digits, "100万", a ≥ 7-digit run as a digit
  string; Chinese numerals that are exact numbers: not 万一 / 一下 / 三四个 /
  六七十 / 一天两天 / 说两句, a lone digit only before a measure word or
  after 第), `dub_numbers(text)` (digits 0-9 / ០-៩, "100 ម៉ឺន", runs of
  Khmer number words split where two digit words meet: មួយពីរ = 1, 2),
  `check(source, dub, lang)` → (missing, unmatched), `repair()` (one missing
  number, one cleanly written number of about its size → digits),
  `keeps(source, old, new)` (a rewording loses no number), `km_words(n)`
  (681 → ប្រាំមួយរយប៉ែតសិបមួយ; ម៉ឺន / សែន below a million),
  `km_say_numbers(text)` (the voice: digits → words, 3.5, 50 %, 8:30, codes
  digit by digit), `km_digits(text)` (subtitles: ០-៩, not inside H2).
  Why: on a 1-hour drama the AI's Khmer number words were wrong in 22 of
  98 lines with a number (681 → 581: ប្រាំមួយ "five-one" lost its មួយ).
  Dub: the rules ask for digits; `DubEngine._fix_numbers` (end of Translate;
  cache `tr_cache["numbers"]`) repairs in code, re-translates the rest with
  the numbers named, marks what is left `review = "number"` (⚑ in Lines);
  shortening and meaning-check fixes that lose a number are refused;
  `speakable_km` says numbers with `km_say_numbers`. `foreign_script` for
  km / th / lo / my: any non-ASCII letter or mark outside the own script
  (Tibetan, Hebrew, Greek letters were inside Khmer number words). Translate
  keeps each line's `tts_key` (the fingerprint of the spoken words and
  voice): a re-run voices again only the lines whose words changed.
- `app/engines/km_order.py` (2026-11-21) — Khmer word order for nouns (the
  noun first, then what describes it: ប្រធានថ្នាក់, not the Chinese order
  ថ្នាក់ប្រធាន for 班长). `rule(src_lang)` (the formula with Chinese or English
  examples; titles before names; front prefixes អនុ / អគ្គ / អតីត / មហា, រង
  after) is in `_rules` for Khmer targets; `COMPOUNDS` (common school / work
  nouns: Chinese, Khmer noun, describer) go to the glossary and translation
  as `hints()`; `fix(source, text)` puts a known compound the original says
  back in Khmer order (whole words only). Dub: `_fix_word_order` at the end
  of Translate (lines and glossary), and on shortened lines and meaning-check
  fixes; the meaning check (Khmer targets) flags a noun phrase left in the
  original's order. Scan of 13 projects (~6,000 lines): one such slip.
- `app/engines/cover_art.py` (2026-11-20) — a video file's poster (MP4 cover
  art, the thumbnail Explorer / phones show): `read_cover()`, `has_cover()`,
  `find_poster(video, sources)` (the file's cover → the DL tab's poster cache
  for the file → the same for `sources` → for the source of a dub project
  whose output is the file), `set_cover(video, image, intro=None)` (stream
  copy, in place via a temp file beside it, comment "RecapStudio poster"),
  `intro_seconds()` (the DL tab's 1 s poster intro, from the comment
  "RecapStudio poster + 1s intro"). The dub's export step keeps the source's
  poster (`DubEngine._keep_poster`); the Export tab's split can keep it.
- `app/engines/auto_pipeline.py` — `AutoPipeline.run(episodes, AutoOptions)`:
  recap or full dub per episode + overlays + 9:16 + thumbnail, resumable.
- AI reliability: per-provider fallback models (`provider.<p>.fallback_models`,
  Gemini default `gemini-flash-latest` → `gemini-flash-lite-latest`); a route
  whose provider has no keys falls back to a keyed default route; 404
  "retired" and Gemini daily-quota errors block only that model (1 h);
  per-minute limits are waited out; keys out of quota are retried after 1 h,
  keys out of credit after 1 min;
  short key cooldowns are waited (≤ 90 s); Gemini `thinkingConfig` from
  `provider.gemini.thinking` ("low" default). Translate batches checkpoint
  into `DubProject.tr_cache`.
  Since 2026-11-20: while another target with a usable key remains, a call
  waits `_FALLBACK_READ_S` (90 s) for its answer instead of `_HTTP_TIMEOUT`
  (300 s; per thread, `AIEngine._tls.read_s`), then the model is skipped for
  `_TIMEOUT_BLOCK_S`; an overloaded model (HTTP 503/529, "high demand") after
  one key's three tries is `model_problem = "busy"`: no key cooldown, skipped
  for `_BUSY_BLOCK_S` (5 min) when another model can answer.
  The dub sends AI batches `_AI_WORKERS` (3) at a time
  (`DubEngine._parallel`): translate, review, meaning check, shortening; a
  batch whose earlier lines are not translated yet sees them in the original.
  Synthesize (≥ `_OVERLAP_MIN_LINES` lines, network voices): the lines found
  too long are reworded every `_REWORD_EVERY` voiced lines while the later
  lines are voiced (`_too_long` depends only on the lines before, so the
  same lines are picked).
- Budget: `ConfigManager.month_cost()/get_budget()`; `AIEngine.check_budget()`
  runs before every chat (settings `budget.monthly_usd`, `budget.warn_fraction`).
  A dub task stopped by it is "paused" and names the budget.
- Replies without an answer (`_reply_error`): OpenRouter's in-body errors keep
  their code (429 cooldown + next key/provider, 402 out of credit); a blocked
  reply (`finish_reason` content_filter/safety…) or a flagged request (403
  moderation) is a request error — next model, the key stays usable.
- Recap voicing (`assemble_recap`): failed lines are retried in rounds
  (3/6/12/24/48 s, like the dub); a line still failing stops the recap with
  its number (no recap with lines missing). Captions (`_split_caption`) break
  at spaces or between whole Khmer syllables (`_syllables`) into even pieces.
- Self-tests (`python -m app.…`) log to `%TEMP%/RecapStudio-selftest/`.
- `ExportPreset.fit_mode` ("" | blur | pad | crop | smart) and
  `render(..., subtitle_style=)`; `MontageOptions.snap_to_scenes`,
  `fit_mode="smart"`, `subtitle_style`; encoders now include AMF and HEVC QSV;
  `known_encoders(config)` reads the persisted list.
- UI: Publish tab, Overlay → Thumbnail sub-tab (merged into the Poster side
  on 2026-11-20), Library "Process series…" (`auto_dialog.AutoDialog`).

### Added in v1.2 (2026-11-15) — no line is lost without a trace
Measured on the user's projects (on-screen subtitles as the truth): the Fit /
Synthesize shortening had removed 9-32 % of the Khmer of Chinese dramas, and
the subtitles were written from that shortened text; Whisper missed 32 of
448 subtitle lines in 20 min of a drama. All additions are optional (old
projects load unchanged).
- Transcription (`audio_pipeline`): Silero VAD threshold 0.35 (`_VAD_THRESHOLD`;
  0.5 left fast/quiet speech out). `WhisperEngine.transcribe(..., recover=True)`
  then runs `recover_gaps()`: voiced stretches (VAD ≥ 0.4, ≥ 0.4 s) that no
  recognised word covers are transcribed again one by one (1 s context, words
  kept only inside the gap); stock phrases, loops, more words than fit the
  time, Whisper's own "no speech" and duplicates are refused. Recovered
  segments get `Segment.review = "recovered"`; a segment under Whisper's
  log-prob limit `"low_confidence"`. `Transcript.recovered` / `.unheard`
  (start, end, reason, text) are saved in transcript.json and named in the
  transcribe step's note. `WhisperEngine.recover(audio, transcript)` does the
  same for Deepgram's transcript when a Whisper model is already installed.
- `deepgram_stt.transcribe` retries a dropped connection, 429 and 5xx (5 s,
  20 s); 4xx fail at once.
- `audio_pipeline.speech_to_text` (the Recap tab and `RecapEngine.
  full_pipeline`; the dub has its own copy of this in `_step_transcribe`):
  with Deepgram and no language given it finds the language first with an
  installed Whisper model (`detect_spoken_language`, `installed_whisper_size`)
  — left to Deepgram, Mandarin came back as English, Korean, French ("Bon,");
  Deepgram hearing no speech at all falls back to Whisper. `RecapEngine.
  analyze_beats` stops with a plain message when 20 s or more of video gave
  under 12 characters of text (no AI call).
- `DubLine.full_text` (the whole translation, set by Translate; Fit restarts
  from it), `DubLine.review` ("recovered" | "low_confidence" | "maybe_prompt" |
  "foreign_script" | "glossary": shown ⚑ in Lines), `DubLine.skip` (left out by
  the user: not translated, voiced or subtitled; the line keeps its index, so
  the index-named tts files stay valid).
- `DubOptions.subtitle_full = True` (subtitles show `full_text`; a typed line
  shows what was typed; `DubEngine.subtitle_text()`), `DubOptions.allow_cuts =
  False` (the condense pass may only tighten wording with every statement and
  question kept — two versions, policy in the cache key — and the meaning
  check refuses a version that left one out; True = the old main-point cuts).
- `drop_repeats` drops only an echo (overlapping, or under half the time of
  the line it repeats); `drop_prompt_echo` only two or more prompt names
  together (one name alone is kept, `review = "maybe_prompt"`); both log.
- Translation memory hits that lack a glossary spelling the source uses
  (`DubEngine._glossary_misses`, terms of 2+ characters) are translated again;
  lines still off the glossary get `review = "glossary"`.
- `AIEngine.translate_two_pass(..., kind="narration" | "dialogue")`: dialogue
  is polished against its originals, never as narration.
  Network errors are reported with URL keys redacted (`_redact_url_keys`).

### app/core/downloader.py  (+ Library in the same file)
```python
@dataclass
class DownloadItem: id: str; url: str; title: str = ""; status: str = "queued"
    # queued|fetching_meta|downloading|paused|done|error
    progress: float = 0.0; speed: str = ""; eta: str = ""; filepath: str = ""
    thumbnail: str = ""; error: str = ""; series: str = ""
class DownloadQueue:
    def __init__(self, library: "Library", out_dir: str | None = None): ...
    def add(self, url: str, series: str = "") -> DownloadItem      # also batch: add_many
    def start(self) / pause(item_id) / resume(item_id) / retry(item_id) / cancel(item_id)
    def set_callback(self, fn: Callable[[DownloadItem], None]) -> None  # any change
    # yt_dlp Python API in a worker thread pool (2 concurrent); progress_hooks
    # update items; pause = abort keeping .part, resume = re-run (continuedl);
    # fetch metadata + thumbnail first, register into Library on completion.
class Library:      # SQLite app_data_dir()/library.db, thread-safe
    def add_series(self, name: str, poster: str = "", meta: dict | None = None) -> int
    def add_episode(self, series_id: int, title: str, filepath: str,
                    duration: float = 0, width: int = 0, height: int = 0) -> int
    def set_status(self, episode_id: int, **flags) -> None
    # transcribed/translated/dubbed/rendered: bool columns
    def get_series(self) -> list[dict];  def get_episodes(self, series_id) -> list[dict]
    def delete_episode(self, episode_id) / delete_series(series_id)
    def find_episode_by_path(self, path: str) -> dict | None
```

### app/core/model_manager.py
```python
MODELS = {  # id -> {label, kind, size_mb, check(), fetch strategy}
  "whisper-tiny|base|small|medium|large-v3": via faster_whisper download_root,
  "demucs-htdemucs": torch-hub file into models_dir()/demucs,
  "lama": direct URL -> models_dir()/lama/lama_fp32.onnx,
  "paddleocr-ch": paddle default fetch with HOME redirect,
}
def check_missing() -> list[str]
def disk_free_gb(path=None) -> float
def download_model(model_id: str, progress=None, cancel=None) -> None
# direct URLs: requests streaming with Range resume (.part file); others invoke
# the owning lib's downloader inside a thread while polling cancel
def startup_report() -> dict  # {gpu: str|None, vram_gb, cuda: bool, disk_free_gb,
# missing_models: [...], ffmpeg_ok: bool}  (GPU via nvidia-smi subprocess; never crash)
```

### UI (app/ui_main.py + app/ui/…)
- `app/version.py`: `__version__` (1.4.0), `BUILD_DATE`: window title,
  the status bar's "v1.4.0" button (About box) and the log's first line.
- `app/ui/widgets.scroll_page(page)`: a page body that scrolls instead of
  squeezing (Export, Publish). The Dub tab folds its settings by itself
  below SHORT_TAB_H (640 px). Export asks where to save every job.
- `app/ui_main.py`: `def run() -> int` — QApplication, dark palette + QSS from
  `app/ui/theme.py`, ConfigManager/AIEngine singletons, MainWindow, splash-less
  fast start; startup env check dialog (model_manager.startup_report) offering
  downloads. On import it sets `OPENCV_OPENCL_RUNTIME=disabled` (setdefault):
  OpenCV's OpenCL is never used (numpy arrays only), and with CUDA initialised
  by the startup probe in one worker thread, OpenCV loading the NVIDIA OpenCL
  driver in another left the process hung in the driver at exit (~1 GB until a
  reboot; measured 2026-11-16).
- `app/ui/theme.py`: `QSS: str` — bg `#0d1117`, cards `#161b22`, accent
  `#58a6ff`, borders `#30363d`, 6px radii; `apply(app)`. The arrows of
  drop-down lists and number boxes are painted PNG triangles (1x and @2x,
  normal / hover / disabled) written once to `<data>/cache/ui` (else the
  temp folder) and added by `_arrow_qss()`: the CSS border triangle used
  before drew as a grey bar.
- Error texts in the UI: `i18n.pick(en, km)` gives one language (the UI's);
  the Recap / Import / Overlay `_msg` and `ProgressRow` use it.
- `app/ui/workers.py`:
  ```python
  class FnWorker(QThread):
      progress = Signal(float, str); finished_ok = Signal(object); failed = Signal(str, str)  # en, km
      def __init__(self, fn, *args, **kwargs)  # injects progress=/cancel= if fn accepts
      def cancel(self)   # sets the threading.Event
  class WorkerPool:  # keeps refs so threads aren't GC'd; cancel_all()
  ```
- `app/ui/widgets.py`: `ConsoleDock` (QPlainTextEdit, subscribes LOG_BUS via
  queued signal, max 5000 lines), `VideoPreview` (QMediaPlayer+QVideoWidget,
  open(path), position slider, play/pause), `TimelineGrid` (QTableWidget
  wrapper: set_rows(list[TimelineRow]), rows(), inline edit, per-row timing
  warning highlight `#f8514966`), `DropArea` mixin (file drag&drop),
  `ProgressRow` (label+bar+cancel button).
- `app/ui/tabs/` one file per tab, class names fixed:
  `import_tab.ImportTab`, `dub_tab.DubTab`,
  `recap_tab.RecapTab`,
  `overlay_tab.OverlayTab`, `dl_tab.DlTab`, `library_tab.LibraryTab`,
  `export_tab.ExportTab`, `publish_tab.PublishTab`, `settings_tab.SettingsTab`
  (9 tabs). The Translate tab (SRT / narration -> TTS voice track -> mix ->
  export) was removed on 2026-11-19: the series work never used it (backup in
  Ai Tool\Backup\translate-tab-2026-11-19). The language list it held is
  `app/ui/langs.py`; the Recap tab shares its narration as `ctx.recap_rows`
  (the Poster side's "AI title ideas").
  Export tab: "Speed" (0.25x–4x, every export) and "Split into episodes":
  episodes of a set length of finished video (QTimeEdit, remembered as
  `export.split_seconds`); the time left over is added to the last episode or
  made an episode of its own (asked); one render-queue job per episode,
  `<name>_ep1.<ext>`, `_ep2`… in a folder asked each time (replace asked).
  "Keep the original video's poster" (`export.split_keep_poster`, on by
  default): every episode gets the poster as cover art (`render(poster=)`)
  and, when the video starts with the DL tab's poster second, starts with it
  too (`intro_image`; episodes are planned on the video after that second);
  off: no poster, episode 1 starts after the poster second.
  Overlay tab = two sides (2026-11-20; the old Video Overlays / Poster Editor
  / Thumbnail sub-tabs): **Poster** — a video row (its own thumbnail from
  `cover_art.find_poster` opens at once; "Find best frames" adds frames to a
  picture strip) feeding the poster editor: "Replace title — same style"
  (poster_designer.replace_title; boxes found or drawn already are used, drawn
  ones are the title), the "Gold brush style" preset + the four thumbnail
  styles, "Ink band" and "Sparkles", a tag line, thumbnail sizes (cover-crop
  around faces), "AI title ideas", the title drawn while typing, "Put into
  the video as its thumbnail" (`cover_art.set_cover`). Detect title (OCR) and
  Remove title (inpaint) were removed (2026-11-21, the user). "Replace title
  — same style" (2026-11-21 evening): the AI picture model redesigns the
  title (ai_replace_title; the editor then draws no font title over it,
  `_ai_designed`; Export gives that picture). When it cannot (free keys: no
  picture quota) the user is told why and asked whether to draw it with the
  app's fonts instead (`_replace_with_font`, below). The font way always works from the picture as loaded (`_image_path`),
  with the title the automatic translation found (`_auto_boxes`, grouped) or
  a box drawn by hand, and layout=True: an AI idea or a typed title gets the
  ORIGINAL title's art style (before, after the automatic translation it
  looked on the cleaned picture, found nothing and used the gold brush).
  "Translate title (3 options)" translates `_auto_source` (the title the AI
  read), a title typed in another language, or reads the title off the
  picture (title_text). At the top of Steps
  (2026-11-21): "Translate into:" + "Translate the title automatically"
  (auto_translate_title: the editor gets the clean picture, the spec and
  the translation; Export PNG/JPG offers <name>_KH.png beside the picture)
  and "Translate the titles of many pictures…" (auto_translate_titles into
  a chosen folder; a list of results — click one to open it in the
  editor; files translated before: replace or keep). **Overlay** — the
  overlays drawn live on a frame of the video (slider; `preview_frame`), the
  watermark as text and/or logo with opacity, size, a 3x3 place or Tile,
  blur boxes drawn on the preview, banner / chroma / ad / subtitles, Apply
  config, a 10 s render from the frame shown. `overlay.last` keeps a
  switched-off watermark's text and logo (`watermark_on`).
  Recap tab: "Full Pipeline" runs steps 1-3 (transcribe, write, voice + video;
  `_full_pipeline_fn`), asking the video settings first (`_AssembleDialog`);
  "Batch all episodes…" makes one video per episode in a chosen folder
  (`_AssembleDialog(folder=True)`, `recap.batch_dir`) and does not start the
  rest when no AI can answer. The recap JSON still goes to the projects folder;
  a video that fails keeps the written recap in the timeline.
  Each: `def __init__(self, ctx: AppContext)`.
- `app/ui/context.py`:
  ```python
  @dataclass
  class AppContext: config: ConfigManager; ai: AIEngine; library: Library
      downloads: DownloadQueue; pool: WorkerPool; main: "MainWindow | None" = None
      current_video: str | None = None
      def open_in_preview(self, path: str)  # routes to main window preview
  ```
- `app/ui/main_window.py`: `MainWindow(ctx)` — header tab bar (Import |
  Translate | Full Dub | Recap | Overlay | DL | Drama MG | Export | Publish | Settings), stacked
  pages, bottom `ConsoleDock`, EN/ខ្មែរ language switcher (rebuild-on-restart is
  acceptable: prompt to restart), status bar with GPU/queue info.
- All user-visible strings through `tr(key, "English default")`. Key style:
  `"tab.recap"`, `"settings.fetch_models"`, `"err.no_keys"`.
- Every long action runs in FnWorker with a visible ProgressRow and Cancel.
- Language switches live (no restart): `MainWindow.retranslate_ui(old, new)`
  walks every top-level window with `app/ui/retranslate.retranslate_tree()`
  (labels, buttons, group titles, tabs, tooltips, placeholders, combo items
  that have item data or came from `tr()`, table / tree headers, docks,
  actions; each widget remembers its texts per language for exact round
  trips), then calls each tab's optional `retranslate()` (Dub: task list,
  readiness, run panel; Settings: provider list, month line).
  `i18n.convert(text, src, dst)`: texts `tr()` returned (or a conversion
  produced), catalog values with `{placeholders}` as templates (values
  converted recursively), a label before ": " / " (" / " · " / a line
  break, "Label:", "Known label + word". `i18n.tr_text()` turns the
  engine's English messages into the UI language through the `dyn.*`
  catalog templates (progress, task notes; ProgressRow and the run panel
  use it). `set_language(code, save=True)` — tests pass `save=False`.
- `app/ui/run_panel.RunPanel` (also `dub_tab.DubRunPanel`) is shared by
  the Dub tab and the Library batch dialog (`auto_dialog.AutoDialog`:
  episode i of n, failed episodes listed with Continue = run again,
  finished episodes skipped). `set_error(..., settings, paused)`: an Open
  Settings button; amber "Paused" when no AI could answer.
  `set_partial(..., title, detail, settings)`. Both windows share
  `ask_no_key()` (Open Settings / Transcribe only / Cancel), `no_ai_key()`,
  `open_settings()` and `FIX_IN_SETTINGS`. The batch asks once about
  moved episodes (relink) and runs `AutoOptions.transcribe_only` with no key.
- Time left: `run_step` stores each task's seconds in `steps[task]["secs"]`;
  the Dub tab learns seconds per minute of video (`dub_engine.learn_speeds`,
  config `dub.speed`, last 5 runs, runs under 3 s ignored) and
  `estimate_left()` adds the running task (its own pace once ≥ 5 %) to the
  learned pace of the tasks after it.
- Dub tab runs use `DubRunPanel` (dub_tab.py) instead, which stays on screen:
  running (overall progress by `dub_engine.STEP_WEIGHTS` /
  `overall_progress()`, task i of 10, elapsed + time left, Cancel), done
  (saved file, size, length, Play / Show in folder) until another video is
  opened, error (bar kept, reason, Continue / Cancel — no pop-up), stopped.
  Continue re-runs the last action (finished tasks are skipped). The save
  location is asked once per video (`_ask_output`: writable folder, free
  space ≥ 1.1 × the video, not the source file; last folder in
  `dub.out_dir`). While dubbing Windows is kept awake
  (SetThreadExecutionState) and the taskbar flashes at the end. A tab with
  `busy_reason()` makes MainWindow ask before closing; a task left
  "running" by a closed app opens as stopped ("interrupted").
- Dub tab layout (the user's design, 2026-11-11; parts in
  `app/ui/dub_widgets.py`, icons painted with QPainter — no icon font):
  "⚙ Dubbing settings" folds away (remembered in `ui.dub.settings_open`;
  folded by itself while a dub runs, since the settings are locked then).
  Below the action buttons a splitter (≈ 60 / 40): a glowing
  "Task Tracking & Problem Detection" card (`dw.GlowCard`, contents in a
  scroll area so nothing is squeezed on a short window) with a live /
  idle pill, the run panel (round state icon, % beside the bar, buttons on
  the title row, the save path shortened in the middle — `path_text()`
  has it whole) or an idle hint (pick a video / ready / paused before task
  i), "System Readiness Check" (summary, four hardware tiles from
  `_hardware_fn`: GPU, CUDA for Whisper with the cuBLAS version, VRAM,
  Python; every check as a row, two columns, click = details + fix,
  double-click = fix; opens by itself when a required check fails) and
  "Task List" (running / pending / completed / notes / failed counters;
  rows drawn by `dw.TaskDelegate`: icon, name, the note dimmed below,
  state with the task's time, a bar on the running task). The right card
  holds Characters / Lines / Glossary / Output. The console dock starts
  120 px high.

## Packaging
- `requirements.txt` pinned versions (PySide6, faster-whisper, edge-tts,
  yt-dlp, demucs, paddleocr+paddlepaddle, onnxruntime, opencv-python-headless,
  pillow, uharfbuzz, numpy, requests, keyring, torch installed separately —
  document CUDA wheel line in README).
- `build_exe.py` → PyInstaller one-dir via `recapstudio.spec` (collect PySide6,
  app/assets, app/locales, bin/); `installer.iss` Inno Setup. Model weights NOT
  bundled — first-run downloader handles them.
