"""Recap tab: transcribe, analyze beats, write recap timeline, batch pipeline."""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.errors import AppError
from app.core.i18n import pick, tr
from app.core.logger import get_logger
from app.core.paths import projects_dir
from app.ui.workers import FnWorker

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.recap_tab")

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
LANGS = [
    ("km", "ខ្មែរ (Khmer)"),
    ("en", "English"),
    ("zh", "中文 (Chinese)"),
    ("th", "ไทย (Thai)"),
    ("vi", "Tiếng Việt (Vietnamese)"),
    ("ko", "한국어 (Korean)"),
    ("ja", "日本語 (Japanese)"),
    ("fr", "Français (French)"),
    ("es", "Español (Spanish)"),
    ("hi", "हिन्दी (Hindi)"),
]
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi")


def _msg(en: str, km: str = "") -> str:
    """The message in the UI language (English when there is no Khmer)."""
    return pick(en, km)


def _safe_name(text: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", text).strip() or "recap"


def _show_in_folder(path: str) -> None:
    """Open Explorer at `path`: a folder itself, a file selected."""
    try:
        if os.path.isdir(path):
            os.startfile(path)  # noqa: S606 - Windows desktop app
        else:
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    except Exception:
        log.exception("open folder failed")


# --- defensive ProgressRow adapters (widget built in parallel) --------------

def _make_progress_row(label: str) -> QWidget:
    from app.ui import widgets as W
    try:
        row = W.ProgressRow(label)
    except TypeError:
        row = W.ProgressRow()
        lab = row.findChild(QLabel)
        if lab is not None:
            lab.setText(label)
    row.setVisible(False)
    return row


def _row_progress(row: QWidget, frac: float, text: str) -> None:
    fn = getattr(row, "set_progress", None)
    if callable(fn):
        try:
            fn(frac, text)
            return
        except Exception:
            pass
    bar = row.findChild(QProgressBar)
    if bar is not None:
        bar.setRange(0, 1000)
        bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
    if text:
        lab = row.findChild(QLabel)
        if lab is not None:
            lab.setText(text)


def _wire_cancel(row: QWidget, worker) -> None:
    row._rs_worker = worker
    if getattr(row, "_rs_cancel_wired", False):
        return
    sig = None
    for name in ("cancel_requested", "cancelled", "canceled", "cancel_clicked"):
        cand = getattr(row, name, None)
        if cand is not None and hasattr(cand, "connect"):
            sig = cand
            break
    if sig is None:
        btn = row.findChild(QToolButton) or row.findChild(QPushButton)
        if btn is not None:
            sig = btn.clicked

    def _do_cancel(*_a):
        w = getattr(row, "_rs_worker", None)
        if w is not None:
            try:
                w.cancel()
            except Exception:
                pass

    if sig is not None:
        try:
            sig.connect(_do_cancel)
            row._rs_cancel_wired = True
        except Exception:
            pass


# --- worker functions (engine imports stay lazy inside) ---------------------

def _transcribe_fn(video: str, model_size: str, use_demucs: bool,
                   progress=None, cancel=None):
    import tempfile
    from app.engines import audio_pipeline as ap

    tmp = tempfile.mkdtemp(prefix="recap_tx_")
    wav = os.path.join(tmp, "audio.wav")

    def sub(base, span):
        def _p(f, t):
            if progress:
                progress(base + span * max(0.0, min(1.0, f)), t)
        return _p

    ap.extract_audio(video, wav, progress=sub(0.0, 0.15), cancel=cancel)
    audio = wav
    if use_demucs:
        try:
            # cached: the montage reuses the music & effects stem as a
            # background without the original voices
            vocals, _music = ap.separated_stems(
                video, progress=sub(0.15, 0.35), cancel=cancel)
            audio = vocals
        except AppError as e:
            log.warning("demucs unavailable, using raw audio: %s", e.en)
    # Whisper or Deepgram, as chosen in Settings → Speech services
    return ap.speech_to_text(audio, model_size=model_size,
                             progress=sub(0.5 if use_demucs else 0.15,
                                          0.5 if use_demucs else 0.85),
                             cancel=cancel)


def _analyze_write_fn(ai, transcript, target_lang: str, target_minutes: float,
                      progress=None, cancel=None):
    from app.engines.recap_engine import RecapEngine

    engine = RecapEngine(ai)

    def sub(base, span):
        def _p(f, t):
            if progress:
                progress(base + span * max(0.0, min(1.0, f)), t)
        return _p

    beats = engine.analyze_beats(transcript, progress=sub(0.0, 0.45), cancel=cancel)
    rows = engine.write_recap(beats, transcript, target_lang,
                              target_minutes=target_minutes,
                              progress=sub(0.45, 0.55), cancel=cancel)
    return beats, rows


_RECAP_PART = 0.55      # share of the progress bar for steps 1-2; the video gets the rest


def _part(progress, lo: float, hi: float, prefix: str = ""):
    """A progress callback for the [lo, hi] slice of `progress`."""
    def _p(f, t):
        if progress:
            progress(lo + (hi - lo) * max(0.0, min(1.0, f)), prefix + str(t))
    return _p


def _recap_json_path(video: str, lang: str) -> str:
    """Where a recap written for `video` is kept (Load JSON opens it again)."""
    return os.path.join(str(projects_dir()),
                        _safe_name(os.path.splitext(os.path.basename(video))[0])
                        + f".recap.{lang}.json")


def _spoken_rows(rows) -> list:
    return [r for r in rows or [] if str(getattr(r, "narration", "") or "").strip()]


def _full_pipeline_fn(ai, video: str, whisper_size: str, use_demucs: bool,
                      target_lang: str, target_minutes: float, out_path: str,
                      options: dict, config, progress=None, cancel=None) -> dict:
    """Steps 1-3 in one go: transcribe, write the recap (kept as JSON in the
    projects folder), then voice it and make the video. A failure or Cancel
    while making the video keeps the written recap: it comes back with the
    reason instead of being lost."""
    from app.core.errors import CancelledError
    from app.engines.recap_engine import RecapEngine

    project = RecapEngine(ai).full_pipeline(
        video, whisper_size=whisper_size, use_demucs=use_demucs,
        target_lang=target_lang, target_minutes=target_minutes,
        progress=_part(progress, 0.0, _RECAP_PART), cancel=cancel)
    out = {"project": project, "json": "", "result": None, "en": "", "km": "",
           "stopped": False}
    try:
        out["json"] = _recap_json_path(video, target_lang)
        project.save(out["json"])
    except Exception:
        log.exception("auto-save of recap project failed")
        out["json"] = ""
    try:
        out["result"] = _assemble_fn(video, _spoken_rows(project.rows), out_path, options,
                                     config, progress=_part(progress, _RECAP_PART, 1.0),
                                     cancel=cancel)
    except CancelledError:
        out["stopped"] = True
    except AppError as e:
        out["en"], out["km"] = e.en, e.km
    except Exception as e:  # noqa: BLE001 - the recap text is kept either way
        log.exception("recap video failed")
        out["en"] = out["km"] = str(e)
    return out


def _batch_pipeline_fn(ai, videos: list, whisper_size: str, use_demucs: bool,
                       target_lang: str, target_minutes: float, out_dir: str,
                       options: dict, config, progress=None, cancel=None) -> list:
    """Steps 1-3 for each episode: its recap video goes to `out_dir` (the
    recap JSON to the projects folder). One episode's failure does not stop
    the others — except when no AI could answer (no key, no credit…): the
    next episodes would fail the same way, so they are not started."""
    from app.core.ai_engine import ai_unavailable
    from app.core.errors import CancelledError
    from app.engines.recap_engine import RecapEngine

    engine = RecapEngine(ai)
    results: list[dict] = []
    n = max(1, len(videos))
    for i, video in enumerate(videos):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        name = os.path.basename(video)
        item = {"video": video, "out": "", "en": "", "km": "", "not_started": False}
        results.append(item)
        if results[:-1] and any(r.get("ai") for r in results[:-1]):
            item["not_started"] = True
            continue
        lo, hi = i / n, (i + 1) / n
        mid = lo + (hi - lo) * _RECAP_PART
        try:
            project = engine.full_pipeline(
                video, whisper_size=whisper_size, use_demucs=use_demucs,
                target_lang=target_lang, target_minutes=target_minutes,
                progress=_part(progress, lo, mid, f"[{name}] "), cancel=cancel)
            project.save(_recap_json_path(video, target_lang))
            base = _safe_name(os.path.splitext(name)[0])
            res = _assemble_fn(video, _spoken_rows(project.rows),
                               os.path.join(out_dir, f"{base}.recap.{target_lang}.mp4"),
                               options, config, progress=_part(progress, mid, hi, f"[{name}] "),
                               cancel=cancel)
            item["out"] = res.out_path
        except CancelledError:
            raise
        except AppError as e:
            item["en"], item["km"] = e.en, e.km
            item["ai"] = ai_unavailable(e)
        except Exception as e:  # keep going per-episode
            log.exception("batch recap failed for %s", name)
            item["en"] = item["km"] = str(e)
    return results


def _assemble_fn(video: str, rows: list, out_path: str, options: dict, config,
                 progress=None, cancel=None):
    from app.engines.montage_engine import MontageOptions, assemble_recap
    return assemble_recap(video, rows, out_path,
                          MontageOptions.from_dict(options), config=config,
                          progress=progress, cancel=cancel)


MONTAGE_KEY = "montage.last"
BATCH_DIR_KEY = "recap.batch_dir"      # where the last batch put its videos

# Suggested Edge voices per target language (the combo stays editable).
_VOICE_SUGGESTIONS = {
    "km": ["km-KH-SreymomNeural", "km-KH-PisethNeural"],
    "en": ["en-US-AndrewNeural", "en-US-GuyNeural", "en-US-JennyNeural",
           "en-US-AriaNeural"],
    "zh": ["zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural"],
    "th": ["th-TH-NiwatNeural", "th-TH-PremwadeeNeural"],
    "vi": ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
    "ko": ["ko-KR-InJoonNeural", "ko-KR-SunHiNeural"],
    "ja": ["ja-JP-KeitaNeural", "ja-JP-NanamiNeural"],
    "fr": ["fr-FR-HenriNeural", "fr-FR-DeniseNeural"],
    "es": ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
    "hi": ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"],
}

# (key, label, width, height, fit_mode)
_OUTPUT_FORMATS = [
    ("keep", "Keep source size", 0, 0, "pad"),
    ("1080p", "1080p 16:9 (1920x1080)", 1920, 1080, "pad"),
    ("720p", "720p 16:9 (1280x720)", 1280, 720, "pad"),
    ("9x16", "Vertical 9:16 (1080x1920) — blurred background", 1080, 1920, "blur"),
    ("9x16s", "Vertical 9:16 (1080x1920) — smart crop on faces", 1080, 1920, "smart"),
    ("1x1", "Square 1:1 (1080x1080) — blurred background", 1080, 1080, "blur"),
]


class _AssembleDialog(QDialog):
    """Options for montage_engine.assemble_recap; remembers the last values.
    `folder`: ask for a folder (the batch: one video per episode) instead of
    a file. `title` / `start` name the window and its OK button."""

    def __init__(self, parent, config, lang: str, default_out: str,
                 folder: bool = False, title: str = "", start: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title or tr("recap.asm.title", "Assemble recap video"))
        self.resize(560, 0)
        self._config = config
        self._folder = folder
        try:
            saved = dict(config.get(MONTAGE_KEY) or {})
        except Exception:
            saved = {}
        lay = QVBoxLayout(self)

        # --- voice ---
        vbox = QGroupBox(tr("recap.asm.voice", "Narration voice"))
        vf = QFormLayout(vbox)
        self.preset = QComboBox()
        self.preset.addItem(tr("recap.asm.preset.custom", "Custom"), "")
        if lang == "km":
            self.preset.addItem(tr("recap.asm.preset.storyteller_km",
                                   "Fast storyteller — Piseth +50% (like the sample video)"),
                                "storyteller_km")
        vf.addRow(tr("recap.asm.preset", "Narrator preset:"), self.preset)
        self.engine = QComboBox()
        try:
            from app.engines.audio_pipeline import TTS_ENGINES
            self.engine.addItems(list(TTS_ENGINES))
        except Exception:
            self.engine.addItems(["edge", "openai", "elevenlabs"])
        self.engine.setCurrentText(saved.get("tts_engine", "edge"))
        vf.addRow(tr("recap.asm.engine", "Engine:"), self.engine)
        self.voice = QComboBox()
        self.voice.setEditable(True)
        suggestions = _VOICE_SUGGESTIONS.get(lang, [])
        self.voice.addItems(suggestions)
        saved_voice = saved.get("tts_voice", "")
        # Only reuse the saved voice if it matches the target language.
        if saved_voice and (not suggestions or saved_voice.lower().startswith(lang)):
            self.voice.setCurrentText(saved_voice)
        vf.addRow(tr("recap.asm.voice_name", "Voice:"), self.voice)
        self.rate = QSpinBox()
        self.rate.setRange(-50, 100)
        self.rate.setSuffix(" %")
        self.rate.setValue(int(str(saved.get("tts_rate", "+0%")).rstrip("%") or 0))
        vf.addRow(tr("recap.asm.rate", "Speaking rate:"), self.rate)
        lay.addWidget(vbox)
        i = self.preset.findData(saved.get("_narrator", ""))
        self.preset.setCurrentIndex(max(0, i))
        self.preset.currentIndexChanged.connect(self._apply_preset)
        for sig in (self.engine.currentTextChanged, self.voice.currentTextChanged,
                    self.rate.valueChanged):
            sig.connect(self._sync_preset)

        # --- audio ---
        abox = QGroupBox(tr("recap.asm.audio", "Audio mix"))
        af = QFormLayout(abox)
        self.orig_db = QDoubleSpinBox()
        self.orig_db.setRange(-80.0, 6.0)
        self.orig_db.setSuffix(" dB")
        self.orig_db.setValue(float(saved.get("original_audio_db", -16.0)))
        self.orig_db.setToolTip(tr("recap.asm.orig_tip",
                                   "Level of the source video's own audio. -60 or lower mutes it."))
        af.addRow(tr("recap.asm.orig", "Original audio:"), self.orig_db)
        self.remove_voices = QCheckBox(tr("recap.asm.remove_voices",
                                          "Remove original voices (keep music && effects)"))
        self.remove_voices.setChecked(bool(saved.get("remove_voices", True)))
        self.remove_voices.setToolTip(tr(
            "recap.asm.remove_voices_tip",
            "The original dialogue is taken out of the background (Demucs), so "
            "only the narrator speaks. Uses the separation already done for the "
            "transcript when 'Use Demucs separation' was on; otherwise it runs "
            "once now (a few minutes for a long episode)."))
        af.addRow("", self.remove_voices)
        bgm_row = QHBoxLayout()
        self.bgm = QLineEdit(saved.get("bgm_path", ""))
        self.bgm.setPlaceholderText(tr("recap.asm.bgm_ph", "Optional music file (looped)"))
        bgm_btn = QPushButton(tr("common.browse", "Browse…"))
        bgm_btn.clicked.connect(self._pick_bgm)
        bgm_row.addWidget(self.bgm, 1)
        bgm_row.addWidget(bgm_btn)
        bgm_holder = QWidget()
        bgm_holder.setLayout(bgm_row)
        af.addRow(tr("recap.asm.bgm", "Background music:"), bgm_holder)
        self.bgm_db = QDoubleSpinBox()
        self.bgm_db.setRange(-60.0, 6.0)
        self.bgm_db.setSuffix(" dB")
        self.bgm_db.setValue(float(saved.get("bgm_db", -14.0)))
        af.addRow(tr("recap.asm.bgm_db", "Music level:"), self.bgm_db)
        self.sfx = QCheckBox(tr("recap.asm.sfx", "Sound effects on scene cuts"))
        self.sfx.setChecked(bool(saved.get("sfx", False)))
        self.sfx.setToolTip(tr("recap.asm.sfx_tip",
                               "A whoosh on every cut between clips and an impact hit on the "
                               "opening and the final scene. Generated by the app — no sound "
                               "files or licences needed."))
        af.addRow("", self.sfx)
        self.sfx_db = QDoubleSpinBox()
        self.sfx_db.setRange(-40.0, 6.0)
        self.sfx_db.setSuffix(" dB")
        self.sfx_db.setValue(float(saved.get("sfx_db", -10.0)))
        self.sfx_db.setEnabled(self.sfx.isChecked())
        self.sfx.toggled.connect(self.sfx_db.setEnabled)
        af.addRow(tr("recap.asm.sfx_db", "Effects level:"), self.sfx_db)
        lay.addWidget(abox)

        # --- video ---
        xbox = QGroupBox(tr("recap.asm.video", "Video"))
        xf = QFormLayout(xbox)
        self.fmt = QComboBox()
        for key, label, _w, _h, _m in _OUTPUT_FORMATS:
            self.fmt.addItem(tr(f"recap.asm.fmt.{key}", label), key)
        i = self.fmt.findData(saved.get("_format", "keep"))
        self.fmt.setCurrentIndex(max(0, i))
        xf.addRow(tr("recap.asm.format", "Output format:"), self.fmt)
        self.slow = QDoubleSpinBox()
        self.slow.setRange(1.0, 2.0)
        self.slow.setSingleStep(0.05)
        self.slow.setPrefix("x")
        self.slow.setValue(float(saved.get("max_slowdown", 1.35)))
        self.slow.setToolTip(tr("recap.asm.slow_tip",
                                "When narration is longer than its scene, footage is slowed "
                                "down up to this factor, then extended into the next footage."))
        xf.addRow(tr("recap.asm.slow", "Max slow-motion:"), self.slow)
        self.fade = QDoubleSpinBox()
        self.fade.setRange(0.0, 1.0)
        self.fade.setSingleStep(0.05)
        self.fade.setSuffix(" s")
        self.fade.setValue(float(saved.get("fade_seconds", 0.0)))
        xf.addRow(tr("recap.asm.fade", "Fade between clips:"), self.fade)
        self.encoder = QComboBox()
        try:
            from app.engines.media_exporter import known_encoders
            encs = known_encoders(config)
        except Exception:
            encs = []
        for e in (encs or ["libx264", "libx265"]):
            self.encoder.addItem(e)
        self.encoder.setCurrentText(saved.get("vcodec", "libx264"))
        xf.addRow(tr("recap.asm.encoder", "Encoder:"), self.encoder)
        self.snap = QCheckBox(tr("recap.asm.snap", "Start clips on scene cuts"))
        self.snap.setChecked(bool(saved.get("snap_to_scenes", True)))
        self.snap.setToolTip(tr("recap.asm.snap_tip",
                                "Detects shot changes so clips never open mid-shot "
                                "or flash the next shot at the end."))
        xf.addRow("", self.snap)
        self.burn = QCheckBox(tr("recap.asm.burn", "Burn subtitles into the video"))
        self.burn.setChecked(bool(saved.get("burn_subtitles", False)))
        xf.addRow("", self.burn)
        self.sub_style = QComboBox()
        self.sub_style.addItem(tr("recap.asm.style_plain", "Plain"), "")
        from app.engines.subtitles import PRESETS
        for sid, st in PRESETS.items():
            self.sub_style.addItem(st.name, sid)
        i = self.sub_style.findData(saved.get("subtitle_style", ""))
        self.sub_style.setCurrentIndex(max(0, i))
        xf.addRow(tr("recap.asm.sub_style", "Subtitle style:"), self.sub_style)
        lay.addWidget(xbox)

        # --- output ---
        out_row = QHBoxLayout()
        self.out = QLineEdit(default_out)
        out_btn = QPushButton(tr("common.browse", "Browse…"))
        out_btn.clicked.connect(self._pick_out)
        out_row.addWidget(QLabel(tr("recap.asm.output_dir", "Save videos in:") if folder
                                 else tr("recap.asm.output", "Save as:")))
        out_row.addWidget(self.out, 1)
        out_row.addWidget(out_btn)
        lay.addLayout(out_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText(
            start or tr("recap.asm.start", "Assemble"))
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _apply_preset(self):
        from app.engines.montage_engine import NARRATOR_PRESETS
        p = NARRATOR_PRESETS.get(self.preset.currentData() or "")
        if not p:
            return
        self._applying = True       # the three setters below would trip _sync_preset
        try:
            self.engine.setCurrentText(p["tts_engine"])
            self.voice.setCurrentText(p["tts_voice"])
            self.rate.setValue(int(p["tts_rate"].rstrip("%")))
        finally:
            self._applying = False

    def _sync_preset(self, *_):
        """Back to "Custom" once the voice no longer matches the preset."""
        from app.engines.montage_engine import NARRATOR_PRESETS
        if getattr(self, "_applying", False):
            return
        p = NARRATOR_PRESETS.get(self.preset.currentData() or "")
        if p and (self.engine.currentText() != p["tts_engine"]
                  or self.voice.currentText().strip() != p["tts_voice"]
                  or f"{self.rate.value():+d}%" != p["tts_rate"]):
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(0)
            self.preset.blockSignals(False)

    def _pick_bgm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("recap.asm.bgm", "Background music:"), "",
            "Audio (*.mp3 *.wav *.m4a *.aac *.flac *.ogg)")
        if path:
            self.bgm.setText(path)

    def _pick_out(self):
        if self._folder:
            path = QFileDialog.getExistingDirectory(
                self, tr("recap.asm.output_dir", "Save videos in:"), self.out.text())
            if path:
                self.out.setText(os.path.normpath(path))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("recap.asm.output", "Save as:"), self.out.text(), "MP4 (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.out.setText(path)

    def _accept(self):
        if not self.voice.currentText().strip() and self.engine.currentText() == "edge":
            QMessageBox.warning(self, tr("err.title", "Error"),
                                _msg("Choose a narration voice.", "សូមជ្រើសរើសសំឡេងនិទាន។"))
            return
        if not self.out.text().strip():
            QMessageBox.warning(self, tr("err.title", "Error"),
                                tr("recap.asm.no_folder", "Choose a folder for the videos.")
                                if self._folder else
                                _msg("Choose an output file.", "សូមជ្រើសរើសឯកសារលទ្ធផល។"))
            return
        if self._folder:
            try:
                os.makedirs(self.out.text().strip(), exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, tr("err.title", "Error"),
                                    tr("recap.asm.bad_folder",
                                       "This folder cannot be used:") + f"\n{e}")
                return
        try:
            self._config.set(MONTAGE_KEY, self.options())
        except Exception:
            log.warning("could not persist montage options", exc_info=True)
        self.accept()

    def output_path(self) -> str:
        p = self.out.text().strip()
        return p if p.lower().endswith(".mp4") else p + ".mp4"

    def output_folder(self) -> str:
        """The folder chosen in folder mode (the batch)."""
        return self.out.text().strip()

    def options(self) -> dict:
        from app.engines.montage_engine import NARRATOR_PRESETS
        key = self.fmt.currentData()
        _k, _l, w, h, mode = next(f for f in _OUTPUT_FORMATS if f[0] == key)
        narrator = self.preset.currentData() or ""
        return {
            "_format": key,  # UI-only; ignored by MontageOptions.from_dict
            "_narrator": narrator,  # UI-only
            "tts_engine": self.engine.currentText(),
            "tts_voice": self.voice.currentText().strip(),
            "tts_rate": f"{self.rate.value():+d}%",
            "tts_pitch": NARRATOR_PRESETS.get(narrator, {}).get("tts_pitch", "+0Hz"),
            "original_audio_db": float(self.orig_db.value()),
            "remove_voices": self.remove_voices.isChecked(),
            "bgm_path": self.bgm.text().strip(),
            "bgm_db": float(self.bgm_db.value()),
            "sfx": self.sfx.isChecked(),
            "sfx_db": float(self.sfx_db.value()),
            "width": w, "height": h, "fit_mode": mode,
            "max_slowdown": float(self.slow.value()),
            "fade_seconds": float(self.fade.value()),
            "vcodec": self.encoder.currentText(),
            "burn_subtitles": self.burn.isChecked(),
            "snap_to_scenes": self.snap.isChecked(),
            "subtitle_style": self.sub_style.currentData() or "",
        }


class _BatchDialog(QDialog):
    """Pick library episodes for a batch full-pipeline run."""

    def __init__(self, parent, library):
        super().__init__(parent)
        self.setWindowTitle(tr("recap.batch_title", "Batch all episodes"))
        self.resize(520, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(tr(
            "recap.batch_hint",
            "Select the episodes to run through the full recap pipeline:")))
        self.listw = QListWidget()
        lay.addWidget(self.listw, 1)
        try:
            for s in library.get_series():
                sid = s.get("id", s.get("series_id"))
                sname = s.get("name", s.get("title", str(sid)))
                for ep in library.get_episodes(sid):
                    path = ep.get("filepath", ep.get("path", ""))
                    if not path:
                        continue
                    it = QListWidgetItem(f"{sname} — {ep.get('title', os.path.basename(path))}")
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked
                                     if os.path.exists(path)
                                     else Qt.CheckState.Unchecked)
                    it.setData(Qt.ItemDataRole.UserRole, path)
                    it.setToolTip(path)
                    self.listw.addItem(it)
        except Exception:
            log.exception("could not list library episodes")
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def selected_videos(self) -> list:
        out = []
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                p = str(it.data(Qt.ItemDataRole.UserRole) or "")
                if p:
                    out.append(p)
        return out


class RecapTab(QWidget):
    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self._workers: list = []
        self._project = None       # current RecapProject or None
        self._beats: list = []

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ------------------------------------------------ left: controls card
        left = QVBoxLayout()
        controls = QGroupBox(tr("recap.controls", "Recap Controls"))
        form = QFormLayout(controls)

        src_row = QHBoxLayout()
        self.source_label = QLabel(tr("recap.no_video", "No video selected"))
        self.source_label.setWordWrap(True)
        src_row.addWidget(self.source_label, 1)
        pick_btn = QPushButton(tr("recap.pick", "Pick…"))
        pick_btn.clicked.connect(self._on_pick_video)
        src_row.addWidget(pick_btn)
        src_holder = QWidget()
        src_holder.setLayout(src_row)
        form.addRow(tr("recap.source", "Source:"), src_holder)

        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_SIZES)
        self.model_combo.setCurrentText("small")
        form.addRow(tr("recap.whisper_model", "Whisper model:"), self.model_combo)

        self.demucs_check = QCheckBox(tr("recap.use_demucs", "Use Demucs separation"))
        self.demucs_check.setChecked(True)
        form.addRow("", self.demucs_check)

        self.lang_combo = QComboBox()
        for code, native in LANGS:
            self.lang_combo.addItem(native, code)
        form.addRow(tr("recap.target_lang", "Target language:"), self.lang_combo)

        self.minutes_spin = QDoubleSpinBox()
        self.minutes_spin.setRange(0.5, 30.0)
        self.minutes_spin.setSingleStep(0.5)
        self.minutes_spin.setValue(3.0)
        self.minutes_spin.setSuffix(" " + tr("recap.min", "min"))
        form.addRow(tr("recap.target_minutes", "Target length:"), self.minutes_spin)

        left.addWidget(controls)

        self.transcribe_btn = QPushButton(tr("recap.transcribe", "1. Transcribe"))
        self.transcribe_btn.clicked.connect(self._on_transcribe)
        left.addWidget(self.transcribe_btn)
        self.transcribe_row = _make_progress_row(tr("recap.transcribing", "Transcribing…"))
        left.addWidget(self.transcribe_row)

        self.analyze_btn = QPushButton(tr("recap.analyze", "2. Analyze && Write Recap"))
        self.analyze_btn.clicked.connect(self._on_analyze)
        left.addWidget(self.analyze_btn)
        self.analyze_row = _make_progress_row(tr("recap.analyzing", "Analyzing…"))
        left.addWidget(self.analyze_row)

        self.assemble_btn = QPushButton(
            tr("recap.assemble", "3. Assemble Recap Video"))
        self.assemble_btn.setToolTip(tr(
            "recap.assemble_tip",
            "Voice every narration row, cut each row's footage to fit its "
            "narration, and render one finished recap video."))
        self.assemble_btn.clicked.connect(self._on_assemble)
        left.addWidget(self.assemble_btn)
        self.assemble_row = _make_progress_row(
            tr("recap.assembling", "Assembling recap video…"))
        left.addWidget(self.assemble_row)

        self.full_btn = QPushButton(tr("recap.full_pipeline", "Full Pipeline"))
        self.full_btn.setToolTip(tr(
            "recap.full_tip",
            "Steps 1, 2 and 3 in one go: transcribe, write the recap, then voice it "
            "and make the video. Asks for the video settings first."))
        self.full_btn.clicked.connect(self._on_full_pipeline)
        left.addWidget(self.full_btn)
        self.full_row = _make_progress_row(tr("recap.pipeline_running", "Running pipeline…"))
        left.addWidget(self.full_row)

        self.batch_btn = QPushButton(tr("recap.batch", "Batch all episodes…"))
        self.batch_btn.setToolTip(tr(
            "recap.batch_tip",
            "A recap video for each episode you pick (steps 1, 2 and 3 for each). "
            "Asks for the video settings and a folder first."))
        self.batch_btn.clicked.connect(self._on_batch)
        left.addWidget(self.batch_btn)
        self.batch_row = _make_progress_row(tr("recap.batch_running", "Batch running…"))
        left.addWidget(self.batch_row)

        self.batch_list = QListWidget()
        self.batch_list.setVisible(False)
        self.batch_list.setMaximumHeight(140)
        left.addWidget(self.batch_list)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8b949e;")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch(1)

        left_holder = QWidget()
        left_holder.setLayout(left)
        left_holder.setMaximumWidth(380)
        root.addWidget(left_holder)

        # ------------------------------------------------ right: timeline
        right = QVBoxLayout()
        toolbar = QHBoxLayout()
        save_btn = QPushButton(tr("recap.save_json", "Save JSON"))
        save_btn.clicked.connect(self._on_save_json)
        toolbar.addWidget(save_btn)
        load_btn = QPushButton(tr("recap.load_json", "Load JSON"))
        load_btn.clicked.connect(self._on_load_json)
        toolbar.addWidget(load_btn)
        toolbar.addStretch(1)
        right.addLayout(toolbar)

        from app.ui.widgets import TimelineGrid, VideoPreview
        self.preview = VideoPreview()
        self.timeline = TimelineGrid()
        self.timeline.currentCellChanged.connect(self._on_timeline_row)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.preview)
        splitter.addWidget(self.timeline)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        right.addWidget(splitter, 1)
        root.addLayout(right, 1)

        self._refresh_source_label()

    # ----------------------------------------------------------------- utils

    def _launch(self, worker) -> None:
        self._workers = [w for w in self._workers if w.isRunning()]
        self._workers.append(worker)
        pool = getattr(self.ctx, "pool", None)
        for name in ("add", "start", "submit", "track"):
            fn = getattr(pool, name, None)
            if callable(fn):
                try:
                    fn(worker)
                    break
                except Exception:
                    continue
        try:
            if not worker.isRunning():
                worker.start()
        except Exception:
            log.exception("could not start worker")

    def _run_task(self, row, fn, *args, on_ok=None, running_text="", **kwargs):
        # One recap job at a time: they share the transcript / timeline, and
        # two at once run Whisper and the AI twice (double time, double cost).
        if any(w.isRunning() for w in self._workers):
            QMessageBox.information(
                self, tr("recap.title", "Recap"),
                _msg("A recap job is already running. Wait for it to finish, "
                     "or press Cancel on it first.",
                     "ការងារសង្ខេបមួយកំពុងដំណើរការ។ សូមរង់ចាំឱ្យវាចប់ "
                     "ឬចុចបោះបង់វាជាមុនសិន។"))
            return None
        worker = FnWorker(fn, *args, **kwargs)
        row.setVisible(True)
        _row_progress(row, 0.0, running_text or tr("common.working", "Working…"))
        _wire_cancel(row, worker)
        worker.progress.connect(lambda f, t, r=row: _row_progress(r, f, t))
        worker.cancelled.connect(
            lambda r=row: _row_progress(r, 0.0, tr("common.cancelled", "Cancelled")))

        def _ok(result):
            _row_progress(row, 1.0, tr("common.done", "Done"))
            if on_ok:
                try:
                    on_ok(result)
                except Exception as e:
                    log.exception("post-task handler failed")
                    QMessageBox.warning(self, tr("err.title", "Error"), str(e))

        def _fail(en, km):
            _row_progress(row, 0.0, tr("common.failed", "Failed"))
            QMessageBox.warning(self, tr("err.title", "Error"), _msg(en, km))

        worker.finished_ok.connect(_ok)
        worker.failed.connect(_fail)
        self._launch(worker)
        return worker

    def _refresh_source_label(self):
        v = self.ctx.current_video
        if v:
            self.source_label.setText(os.path.basename(v))
            self.source_label.setToolTip(v)
            self._ensure_preview(v)
        else:
            self.source_label.setText(tr("recap.no_video", "No video selected"))
            self.source_label.setToolTip("")

    def showEvent(self, ev):  # noqa: N802 - refresh when tab becomes visible
        self._refresh_source_label()
        super().showEvent(ev)

    def _require_video(self) -> str | None:
        v = self.ctx.current_video
        if not v or not os.path.exists(v):
            QMessageBox.information(
                self, tr("recap.title", "Recap"),
                _msg("Select a source video first (Import tab or Pick…).",
                     "សូមជ្រើសរើសវីដេអូប្រភពជាមុនសិន (ផ្ទាំង Import ឬប៊ូតុង Pick…)។"))
            return None
        return v

    def _target_lang(self) -> str:
        return str(self.lang_combo.currentData() or "km")

    def _ensure_preview(self, path: str) -> None:
        try:
            if path and os.path.exists(path) and \
                    os.path.normcase(self.preview.current_path() or "") != os.path.normcase(path):
                self.preview.open(path)
        except Exception:
            log.exception("recap preview open failed")

    def _on_timeline_row(self, row: int, _col: int, prev_row: int, _pc: int) -> None:
        """Jump the preview to the selected row's source timestamp."""
        if row < 0 or row == prev_row:
            return
        rows = self._current_rows()
        if row >= len(rows):
            return
        v = self.ctx.current_video
        if not v:
            return
        self._ensure_preview(v)
        try:
            self.preview.seek(float(getattr(rows[row], "start", 0.0)))
        except Exception:
            log.exception("preview seek failed")

    # ---------------------------------------------------------------- actions

    def _on_assemble(self):
        video = self._require_video()
        if not video:
            return
        rows = [r for r in self._current_rows()
                if str(getattr(r, "narration", "") or "").strip()]
        if not rows:
            QMessageBox.information(
                self, tr("recap.title", "Recap"),
                _msg("The timeline has no narration yet — run step 2 or load a "
                     "recap JSON first.",
                     "តារាងពេលវេលាមិនទាន់មានការនិទានទេ — សូមដំណើរការជំហានទី 2 "
                     "ឬបើកឯកសារ JSON ជាមុនសិន។"))
            return
        lang = self._target_lang()
        base = _safe_name(os.path.splitext(os.path.basename(video))[0])
        default_out = os.path.join(os.path.dirname(video), f"{base}.recap.{lang}.mp4")
        dlg = _AssembleDialog(self, self.ctx.config, lang, default_out)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        out_path = dlg.output_path()
        if os.path.abspath(out_path) == os.path.abspath(video):
            QMessageBox.warning(self, tr("err.title", "Error"),
                                _msg("The output must not overwrite the source video.",
                                     "ឯកសារលទ្ធផលមិនត្រូវជាន់លើវីដេអូប្រភពទេ។"))
            return

        self._run_task(
            self.assemble_row, _assemble_fn, video, rows, out_path,
            dlg.options(), self.ctx.config,
            on_ok=lambda result: self._show_video_done(video, result),
            running_text=tr("recap.assembling", "Assembling recap video…"))

    def _show_video_done(self, video: str, result) -> None:
        """A recap video is made (step 3 or Full Pipeline): say so, show it,
        offer its folder."""
        msg = (tr("recap.asm.done", "Recap video ready") +
               f": {os.path.basename(result.out_path)} "
               f"({result.duration / 60:.1f} min, {len(result.clips)} clips)")
        if result.skipped:
            msg += " — " + tr("recap.asm.skipped", "rows without voice") + \
                   f": {', '.join(str(i + 1) for i in result.skipped)}"
        self.status_label.setText(msg)
        self._mark_rendered(video)
        try:
            self.preview.open(result.out_path)
        except Exception:
            log.exception("could not open assembled video")
        box = QMessageBox(self)
        box.setWindowTitle(tr("recap.title", "Recap"))
        box.setText(msg + "\n\n" + result.out_path +
                    "\n" + tr("recap.asm.srt_saved", "Subtitles") + f": {result.srt_path}")
        open_btn = box.addButton(tr("recap.asm.open_folder", "Open folder"),
                                 QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            _show_in_folder(result.out_path)

    def _mark_rendered(self, video: str) -> None:
        try:
            ep = self.ctx.library.find_episode_by_path(video)
            if ep:
                self.ctx.library.set_status(ep.get("id"), rendered=True)
        except Exception:
            log.debug("library status update skipped", exc_info=True)

    def _on_pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("recap.pick_video", "Choose source video"), "",
            tr("import.filter_video", "Videos") + " (*.mp4 *.mkv *.mov *.webm *.avi)")
        if path:
            self.ctx.current_video = path
            self._refresh_source_label()
            try:
                self.ctx.open_in_preview(path)
            except Exception:
                log.exception("open_in_preview failed")

    def _on_transcribe(self):
        video = self._require_video()
        if not video:
            return

        def done(transcript):
            self.ctx.transcript = transcript
            n = len(getattr(transcript, "segments", []) or [])
            self.status_label.setText(
                tr("recap.transcribed", "Transcribed") +
                f": {n} " + tr("recap.segments", "segments") +
                f" ({getattr(transcript, 'language', '?')})")

        self._run_task(
            self.transcribe_row, _transcribe_fn, video,
            self.model_combo.currentText(), self.demucs_check.isChecked(),
            on_ok=done,
            running_text=tr("recap.transcribing", "Transcribing…"))

    def _on_analyze(self):
        transcript = getattr(self.ctx, "transcript", None)
        if transcript is None:
            QMessageBox.information(
                self, tr("recap.title", "Recap"),
                _msg("Run '1. Transcribe' first — no transcript is cached yet.",
                     "សូមដំណើរការ '1. Transcribe' ជាមុនសិន — មិនទាន់មានអត្ថបទចម្លងទេ។"))
            return
        lang = self._target_lang()
        minutes = float(self.minutes_spin.value())

        def done(result):
            beats, rows = result
            self._beats = beats
            self._set_rows(rows, lang)

        self._run_task(
            self.analyze_row, _analyze_write_fn, self.ctx.ai, transcript,
            lang, minutes,
            on_ok=done,
            running_text=tr("recap.analyzing", "Analyzing…"))

    def _on_full_pipeline(self):
        """Steps 1-3: the video settings are asked first (as in step 3), so
        the whole run needs nobody at the computer."""
        video = self._require_video()
        if not video:
            return
        lang = self._target_lang()
        base = _safe_name(os.path.splitext(os.path.basename(video))[0])
        dlg = _AssembleDialog(
            self, self.ctx.config, lang,
            os.path.join(os.path.dirname(video), f"{base}.recap.{lang}.mp4"),
            title=tr("recap.full_title", "Full Pipeline: the video"),
            start=tr("recap.full_start", "Start"))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        out_path = dlg.output_path()
        if os.path.abspath(out_path) == os.path.abspath(video):
            QMessageBox.warning(self, tr("err.title", "Error"),
                                _msg("The output must not overwrite the source video.",
                                     "ឯកសារលទ្ធផលមិនត្រូវជាន់លើវីដេអូប្រភពទេ។"))
            return

        def done(out):
            project = out["project"]
            self._project = project
            self._beats = list(getattr(project, "beats", []) or [])
            if getattr(project, "transcript", None) is not None:
                self.ctx.transcript = project.transcript
            self._set_rows(list(project.rows or []), lang, project=project)
            if out["result"] is not None:
                self._show_video_done(video, out["result"])
                return
            # the recap is written and kept; only the video is missing
            _row_progress(self.full_row, 0.0,
                          tr("common.cancelled", "Cancelled") if out["stopped"]
                          else tr("common.failed", "Failed"))
            again = tr("recap.full_try_again", "Press “{button}” to make it.").replace(
                "{button}", tr("recap.assemble", "3. Assemble Recap Video"))
            if out["stopped"]:
                self.status_label.setText(
                    tr("recap.full_stopped", "Stopped while making the video. The recap "
                                             "is written (it is in the timeline).")
                    + " " + again)
                return
            QMessageBox.warning(
                self, tr("recap.title", "Recap"),
                tr("recap.full_no_video", "The recap is written (it is in the timeline), "
                                          "but the video could not be made:")
                + "\n" + pick(out["en"], out["km"]) + "\n\n" + again)

        self._run_task(
            self.full_row, _full_pipeline_fn, self.ctx.ai, video,
            self.model_combo.currentText(), self.demucs_check.isChecked(),
            lang, float(self.minutes_spin.value()), out_path, dlg.options(),
            self.ctx.config,
            on_ok=done,
            running_text=tr("recap.pipeline_running", "Running pipeline…"))

    def _on_batch(self):
        dlg = _BatchDialog(self, self.ctx.library)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        videos = dlg.selected_videos()
        if not videos:
            QMessageBox.information(
                self, tr("recap.title", "Recap"),
                _msg("No episodes selected.", "មិនបានជ្រើសរើសភាគណាមួយទេ។"))
            return
        lang = self._target_lang()
        try:
            last = str(self.ctx.config.get(BATCH_DIR_KEY) or "")
        except Exception:
            last = ""
        vdlg = _AssembleDialog(
            self, self.ctx.config, lang,
            last if last and os.path.isdir(last) else os.path.dirname(videos[0]),
            folder=True, title=tr("recap.batch_video_title", "Batch: the videos"),
            start=tr("recap.full_start", "Start"))
        if vdlg.exec() != QDialog.DialogCode.Accepted:
            return
        out_dir = vdlg.output_folder()
        try:
            self.ctx.config.set(BATCH_DIR_KEY, out_dir)
        except Exception:
            log.warning("could not remember the batch folder", exc_info=True)
        self.batch_list.clear()
        self.batch_list.setVisible(True)
        for v in videos:
            self.batch_list.addItem("⏳ " + os.path.basename(v))

        def done(results):
            self.batch_list.clear()
            ok = 0
            for r in results:
                name = os.path.basename(r["video"])
                if r["out"]:
                    ok += 1
                    self._mark_rendered(r["video"])
                    it = QListWidgetItem("✓ " + name)
                    it.setToolTip(r["out"])
                elif r["not_started"]:
                    it = QListWidgetItem("– " + name + " — " + tr(
                        "recap.batch_not_started",
                        "not started: no AI could answer (see the first error)"))
                else:
                    err = pick(r["en"], r["km"])
                    it = QListWidgetItem("✗ " + name + " — " + err)
                    it.setToolTip(err)
                self.batch_list.addItem(it)
            self.status_label.setText(
                tr("recap.batch_done", "Batch finished") +
                f": {ok}/{len(results)} " + tr("recap.ok", "ok") + f" → {out_dir}")

        self._run_task(
            self.batch_row, _batch_pipeline_fn, self.ctx.ai, videos,
            self.model_combo.currentText(), self.demucs_check.isChecked(),
            lang, float(self.minutes_spin.value()), out_dir, vdlg.options(),
            self.ctx.config,
            on_ok=done,
            running_text=tr("recap.batch_running", "Batch running…"))

    # ------------------------------------------------------------ timeline io

    def _set_rows(self, rows: list, lang: str, project=None):
        if project is None:
            from app.engines.recap_engine import RecapProject
            project = RecapProject(
                source_video=self.ctx.current_video or "",
                language=lang, rows=list(rows), beats=list(self._beats),
                transcript=getattr(self.ctx, "transcript", None))
        self._project = project
        try:
            self.timeline.set_rows(list(rows))
        except Exception:
            log.exception("TimelineGrid.set_rows failed")
        self._share_rows(rows)
        self.status_label.setText(
            tr("recap.rows_ready", "Timeline rows ready") + f": {len(rows)}")

    def _current_rows(self) -> list:
        try:
            return list(self.timeline.rows())
        except Exception:
            log.exception("TimelineGrid.rows failed")
            return list(getattr(self._project, "rows", []) or [])

    def _on_save_json(self):
        rows = self._current_rows()
        if not rows and self._project is None:
            QMessageBox.information(
                self, tr("recap.title", "Recap"),
                _msg("Nothing to save yet.", "មិនទាន់មានអ្វីត្រូវរក្សាទុកទេ។"))
            return
        default_name = _safe_name(os.path.splitext(os.path.basename(
            self.ctx.current_video or "recap"))[0]) + ".recap.json"
        path, _ = QFileDialog.getSaveFileName(
            self, tr("recap.save_json", "Save JSON"),
            os.path.join(str(projects_dir()), default_name),
            "JSON (*.json)")
        if not path:
            return
        try:
            if self._project is None:
                from app.engines.recap_engine import RecapProject
                self._project = RecapProject(
                    source_video=self.ctx.current_video or "",
                    language=self._target_lang(), rows=rows,
                    beats=list(self._beats),
                    transcript=getattr(self.ctx, "transcript", None))
            else:
                self._project.rows = rows
            self._project.save(path)
            self.status_label.setText(tr("recap.saved_to", "Saved to") + f": {path}")
        except AppError as e:
            QMessageBox.warning(self, tr("err.title", "Error"), _msg(e.en, e.km))
        except Exception as e:
            log.exception("save recap json failed")
            QMessageBox.warning(self, tr("err.title", "Error"), str(e))

    def _on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("recap.load_json", "Load JSON"),
            str(projects_dir()), "JSON (*.json)")
        if not path:
            return
        try:
            from app.engines.recap_engine import RecapProject
            project = RecapProject.load(path)
        except AppError as e:
            QMessageBox.warning(self, tr("err.title", "Error"), _msg(e.en, e.km))
            return
        except Exception as e:
            log.exception("load recap json failed")
            QMessageBox.warning(self, tr("err.title", "Error"), str(e))
            return
        self._project = project
        self._beats = list(getattr(project, "beats", []) or [])
        if getattr(project, "transcript", None) is not None:
            self.ctx.transcript = project.transcript
        if getattr(project, "source_video", "") and os.path.exists(project.source_video):
            self.ctx.current_video = project.source_video
            self._refresh_source_label()
        idx = self.lang_combo.findData(getattr(project, "language", "km"))
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        try:
            self.timeline.set_rows(list(project.rows or []))
        except Exception:
            log.exception("TimelineGrid.set_rows failed")
        self._share_rows(list(project.rows or []))
        self.status_label.setText(
            tr("recap.loaded", "Loaded") + f": {os.path.basename(path)} "
            f"({len(project.rows or [])} " + tr("recap.rows", "rows") + ")")

    def _share_rows(self, rows: list) -> None:
        """The recap's narration for other tabs: the Overlay's thumbnail
        panel writes its AI title ideas from it."""
        setattr(self.ctx, "recap_rows", list(rows))
        setattr(self.ctx, "recap_lang", self._target_lang())
