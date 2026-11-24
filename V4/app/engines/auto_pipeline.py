"""One-click series pipeline: episode -> finished video(s), for many
episodes in a row.

Per episode, depending on the mode:
  recap  transcribe -> story beats -> recap script -> narrated montage
  dub    Full Dub (all 10 steps, resumable, uses the series memory)
then optionally: saved overlays (watermark / banner / ...), a 9:16 version
with smart crop and Shorts captions, and a thumbnail with an AI title.
Library flags are updated as each episode finishes.

Every stage is skipped when its output already exists (unless overwrite),
so a stopped run resumes where it left off. Settings saved by the tabs are
reused: "montage.last" (Recap -> Assemble) and "overlay.last" (Overlay).

No Qt imports.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from app.core.ai_engine import ai_failure_kind, ai_unavailable
from app.core.errors import AppError, CancelledError
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import projects_dir

log = get_logger("auto_pipeline")

ProgressFn = Callable[[float, str], None]
MODES = ["recap", "dub"]


@dataclass
class AutoOptions:
    mode: str = "dub"                 # "recap" | "dub"
    target_lang: str = "km"
    series: str = ""                  # for the series glossary / voice cast
    whisper_size: str = "small"
    recap_minutes: float = 3.0
    use_overlays: bool = False        # apply the Overlay tab's saved settings
    vertical: bool = False            # also export 1080x1920 (smart crop)
    vertical_style: str = "shorts"    # subtitle preset burned into 9:16
    thumbnail: bool = True
    thumb_style: str = "bold"
    out_dir: str = ""                 # "" = "<video folder>/RecapStudio out"
    vcodec: str = "libx264"
    overwrite: bool = False
    # dub: only the tasks that need no AI key (audio, transcription,
    # speakers) — no AI key yet; translated and dubbed by a later run
    transcribe_only: bool = False


@dataclass
class EpisodeResult:
    episode_id: int
    title: str
    video: str
    status: str = "pending"           # pending | running | done | failed | skipped
    outputs: list[str] = field(default_factory=list)
    message: str = ""
    seconds: float = 0.0
    ai: str = ""                      # failed because no AI answered: why (no_key, budget…)
    message_km: str = ""              # the failure in Khmer, when the engine gave one


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


def _sub(progress: ProgressFn | None, lo: float, hi: float, prefix: str = ""):
    if progress is None:
        return None

    def fn(f: float, t: str) -> None:
        progress(lo + (hi - lo) * max(0.0, min(1.0, f)), prefix + t)
    return fn


def _stem(video: str) -> str:
    from app.engines.dub_engine import _safe_name
    return _safe_name(Path(video).stem)


def out_folder(video: str, opts: AutoOptions) -> str:
    d = opts.out_dir.strip() or os.path.join(os.path.dirname(video), "RecapStudio out")
    os.makedirs(d, exist_ok=True)
    return d


def _fresh(path: str, opts: AutoOptions) -> bool:
    """True if `path` exists and may be reused."""
    return (not opts.overwrite) and os.path.isfile(path) and os.path.getsize(path) > 0


def _overlays(config):
    from app.engines.media_exporter import OverlayConfig
    d = dict(config.get("overlay.last") or {})
    for k in ("subtitle_srt", "subtitle_font"):
        d.pop(k, None)
    ov = OverlayConfig.from_dict(d)
    used = any([ov.watermark, (ov.watermark_text or "").strip(), ov.banner,
                ov.chroma_video, ov.ad_video, ov.blur_regions])
    return ov if used else None


class AutoPipeline:
    def __init__(self, ai, config, library=None):
        self.ai, self.config, self.library = ai, config, library

    # -- one episode --------------------------------------------------------
    def _recap(self, video: str, opts: AutoOptions, base: str, progress, cancel):
        from app.engines.montage_engine import MontageOptions, assemble_recap
        from app.engines.recap_engine import RecapEngine, RecapProject
        proj = str(projects_dir() / f"{_stem(video)}.recap.{opts.target_lang}.json")
        if _fresh(proj, opts):
            project = RecapProject.load(proj)
        else:
            project = RecapEngine(self.ai).full_pipeline(
                video, whisper_size=opts.whisper_size, target_lang=opts.target_lang,
                target_minutes=opts.recap_minutes,
                progress=_sub(progress, 0.0, 0.5, "Recap: "), cancel=cancel)
            project.save(proj)
        _check_cancel(cancel)
        mo = MontageOptions.from_dict(dict(self.config.get("montage.last") or {}))
        mo.vcodec = opts.vcodec or mo.vcodec
        if not _fresh(base, opts):
            res = assemble_recap(video, project.rows, base, mo, self.config,
                                 progress=_sub(progress, 0.5, 0.85, "Assemble: "),
                                 cancel=cancel)
            srt = res.srt_path
        else:
            srt = os.path.splitext(base)[0] + ".srt"
        context = " ".join(r.narration for r in project.rows)[:4000]
        return base, srt, context

    @staticmethod
    def _dub_project(video: str, opts: AutoOptions, base: str):
        """The episode's dub project (a moved video's earlier one too, once
        relinked) with this run's settings."""
        from app.engines.dub_engine import DubOptions, DubProject
        p = DubProject.for_video(video)
        o = p.options if os.path.isfile(p.path) else DubOptions()
        o.target_lang, o.whisper_size = opts.target_lang, opts.whisper_size
        o.series, o.output_path = opts.series or o.series, base
        p.options = o
        return p

    def _transcribe(self, video: str, opts: AutoOptions, base: str, progress, cancel):
        """No AI key yet: extract the audio, transcribe, find the speakers."""
        from app.engines.dub_engine import DubEngine
        p = self._dub_project(video, opts, base)
        DubEngine(self.ai, self.config).run_until_ai(
            p, progress=_sub(progress, 0.0, 1.0, "Dub: "), cancel=cancel)
        return p

    def _dub(self, video: str, opts: AutoOptions, base: str, progress, cancel):
        from app.engines.dub_engine import DubEngine
        p = self._dub_project(video, opts, base)
        if opts.overwrite:
            p.invalidate_from("export")
        if not (_fresh(base, opts) and p.step_done("export")):
            DubEngine(self.ai, self.config).run_all(
                p, progress=_sub(progress, 0.0, 0.85, "Dub: "), cancel=cancel)
        context = (p.story_context or "") + "\n" + " ".join(
            (l.full_text or l.text) for l in p.lines[:80])   # whole, not timing-shortened
        return p.output or base, p.srt_path, context

    def run_episode(self, ep: dict, opts: AutoOptions,
                    progress: ProgressFn | None = None,
                    cancel: threading.Event | None = None) -> EpisodeResult:
        from app.engines import media_exporter as me
        video = ep.get("filepath", "")
        r = EpisodeResult(int(ep.get("id") or 0), str(ep.get("title") or
                                                      Path(video).stem), video)
        t0 = time.time()
        if not video or not os.path.isfile(video):
            r.status, r.message = "failed", "video file missing"
            return r
        folder = out_folder(video, opts)
        stem, lang = _stem(video), opts.target_lang
        tag = "recap" if opts.mode == "recap" else "dub"
        final = os.path.join(folder, f"{stem}.{tag}.{lang}.mp4")
        overlays = _overlays(self.config) if opts.use_overlays else None
        base = final if overlays is None else os.path.join(folder, f"{stem}.{tag}.{lang}.base.mp4")

        if opts.mode == "dub" and opts.transcribe_only:
            p = self._transcribe(video, opts, base, progress, cancel)
            self._mark(ep, opts, transcribed_only=True)
            r.status, r.seconds = "done", round(time.time() - t0, 1)
            r.message = f"transcribed: {len(p.lines)} lines, {len(p.characters)} speakers"
            return r
        if opts.mode == "recap":
            made, srt, context = self._recap(video, opts, base, progress, cancel)
        else:
            made, srt, context = self._dub(video, opts, base, progress, cancel)
        _check_cancel(cancel)
        if overlays is not None and not _fresh(final, opts):
            me.render(made, final, me.ExportPreset(vcodec=opts.vcodec, bitrate=""),
                      overlays=overlays, progress=_sub(progress, 0.85, 0.9, "Overlays: "),
                      cancel=cancel)
        r.outputs.append(final if os.path.isfile(final) else made)
        if srt and os.path.isfile(srt):
            r.outputs.append(srt)

        if opts.vertical:
            vpath = os.path.join(folder, f"{stem}.{tag}.{lang}.9x16.mp4")
            if not _fresh(vpath, opts):
                me.render(r.outputs[0], vpath,
                          me.ExportPreset(width=1080, height=1920, fit_mode="smart",
                                          vcodec=opts.vcodec, bitrate="8M"),
                          srt_path=srt if srt and os.path.isfile(srt) else None,
                          subtitle_style=opts.vertical_style,
                          progress=_sub(progress, 0.9, 0.96, "9:16: "), cancel=cancel)
            r.outputs.append(vpath)

        if opts.thumbnail:
            tpath = os.path.join(folder, f"{stem}.{tag}.{lang}.thumb.jpg")
            if not _fresh(tpath, opts):
                r.outputs.append(self._thumbnail(video, context, r.title, opts,
                                                 tpath, folder, cancel))
            else:
                r.outputs.append(tpath)

        self._mark(ep, opts)
        r.status, r.seconds = "done", round(time.time() - t0, 1)
        r.message = ", ".join(os.path.basename(o) for o in r.outputs)
        return r

    def _thumbnail(self, video, context, title, opts, tpath, folder, cancel) -> str:
        from app.engines import thumbnail as th
        frames = th.find_frames(video, os.path.join(folder, ".frames", _stem(video)),
                                count=4, cancel=cancel)
        text = title
        try:
            ideas = th.suggest_titles(self.ai, context or title, opts.target_lang,
                                      n=3, cancel=cancel)
            text = ideas[0] if ideas else title
        except CancelledError:
            raise
        except Exception as exc:
            log.warning("AI title ideas failed (%s); using the episode title", exc)
        best = frames[0]
        th.compose(best.path, text, tpath, style=opts.thumb_style,
                   face_x=best.face_x, face_y=best.face_y)
        return tpath

    def _mark(self, ep: dict, opts: AutoOptions, transcribed_only: bool = False) -> None:
        if self.library is None or not ep.get("id"):
            return
        flags = {"transcribed": True, "translated": True, "rendered": True}
        if transcribed_only:
            flags = {"transcribed": True}
        elif opts.mode == "dub":
            flags["dubbed"] = True
        try:
            self.library.set_status(int(ep["id"]), **flags)
        except Exception as exc:
            log.warning("library status update failed: %s", exc)

    # -- many episodes ------------------------------------------------------
    def run(self, episodes: list[dict], opts: AutoOptions,
            progress: ProgressFn | None = None,
            cancel: threading.Event | None = None,
            on_result: Callable[[EpisodeResult], None] | None = None
            ) -> list[EpisodeResult]:
        if opts.mode not in MODES:
            raise AppError(f"Unknown mode: {opts.mode}", f"មិនស្គាល់របៀប៖ {opts.mode}")
        n = max(1, len(episodes))
        results = []
        for i, ep in enumerate(episodes):
            _check_cancel(cancel)
            label = f"[{i + 1}/{n}] {ep.get('title') or Path(ep.get('filepath', '')).stem}: "
            try:
                res = self.run_episode(ep, opts, _sub(progress, i / n, (i + 1) / n, label),
                                       cancel)
            except CancelledError:
                raise
            except Exception as exc:
                msg = getattr(exc, "en", str(exc))
                log.warning("episode %s failed: %s", ep.get("title"), msg)
                LOG_BUS.publish(f"[auto] {label}FAILED — {msg}")
                # no AI answered: the dub says why (ai_kind); a recap's AI
                # error comes straight from the AI engine
                kind = str(getattr(exc, "ai_kind", "") or "")
                if not kind and ai_unavailable(exc):
                    kind = ai_failure_kind(exc) or "other"
                km = str(getattr(exc, "km", "") or "")
                res = EpisodeResult(int(ep.get("id") or 0), str(ep.get("title") or ""),
                                    ep.get("filepath", ""), "failed", message=msg,
                                    ai=kind, message_km=km if km != msg else "")
            results.append(res)
            if on_result is not None:
                try:
                    on_result(res)
                except Exception:
                    pass
        done = sum(1 for r in results if r.status == "done")
        if progress:
            progress(1.0, f"{done}/{len(results)} episodes finished")
        return results


def summary(results: list[EpisodeResult]) -> str:
    lines = []
    for r in results:
        mark = {"done": "✔", "failed": "✖"}.get(r.status, "•")
        lines.append(f"{mark} {r.title} — {r.message}"
                     + (f" ({r.seconds / 60:.1f} min)" if r.seconds else ""))
    return "\n".join(lines)


def options_dict(o: AutoOptions) -> dict:
    return asdict(o)
