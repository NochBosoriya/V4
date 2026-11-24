"""Model registry, presence checks, downloads, and startup environment report.

Contract: app/core/model_manager.py — MODELS, check_missing, disk_free_gb,
download_model, startup_report. No Qt imports; heavy deps lazy-imported.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from app.core.errors import AppError, CancelledError, EngineNotAvailable
from app.core.logger import LOG_BUS, get_logger
from app.core.paths import app_data_dir, ffmpeg_path, models_dir, pip_install

log = get_logger("model_manager")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_DEMUCS_URL = ("https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
               "955717e8-8726e21a.th")
_DEMUCS_FILE = "955717e8-8726e21a.th"
_LAMA_URL = ("https://huggingface.co/Carve/LaMa-ONNX/resolve/main/"
             "lama_fp32.onnx")

_WHISPER_SIZES = {
    "tiny": 39, "base": 74, "small": 244, "medium": 769, "large-v3": 1550,
}


# --------------------------------------------------------------------------
# Presence checks
# --------------------------------------------------------------------------

def _whisper_dir(size: str) -> Path:
    return models_dir() / "whisper"


def _whisper_present(size: str) -> bool:
    """A converted faster-whisper model dir contains model.bin."""
    root = _whisper_dir(size)
    if not root.exists():
        return False
    needles = (f"faster-whisper-{size}", f"whisper-{size}", size)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if any(n in name for n in needles):
            for f in child.rglob("model.bin"):
                if f.stat().st_size > 1024 * 1024:
                    return True
    return False


def _demucs_ckpt_path() -> Path:
    return models_dir() / "demucs" / "checkpoints" / _DEMUCS_FILE


def _demucs_present() -> bool:
    p = _demucs_ckpt_path()
    # checkpoint is ~319 MB; anything under 10 MB is a broken partial
    return p.exists() and p.stat().st_size > 10 * 1024 * 1024


def _lama_path() -> Path:
    return models_dir() / "lama" / "lama_fp32.onnx"


def _lama_present() -> bool:
    p = _lama_path()
    return p.exists() and p.stat().st_size > 10 * 1024 * 1024


def _paddle_root() -> Path:
    return models_dir() / "paddleocr"


def _paddle_present() -> bool:
    # paddleocr 2.x: models inside our models/paddleocr dir
    root = _paddle_root()
    if root.exists():
        for pat in ("*.pdiparams", "*.pdmodel", "*.onnx", "inference.json"):
            for f in root.rglob(pat):
                if f.stat().st_size > 0:
                    return True
    # paddleocr 3.x: models in ~/.paddlex/official_models/
    paddlex_root = Path.home() / ".paddlex" / "official_models"
    if paddlex_root.exists():
        has_det = any(
            d.is_dir() and d.name.endswith("_det")
            and (d / "inference.pdiparams").exists()
            for d in paddlex_root.iterdir())
        has_rec = any(
            d.is_dir() and d.name.endswith("_rec")
            and (d / "inference.pdiparams").exists()
            for d in paddlex_root.iterdir())
        if has_det and has_rec:
            return True
    return False


def _mdx():
    from app.engines import mdx_separator  # numpy only; no Qt
    return mdx_separator


def _subocr():
    from app.engines import subtitle_ocr  # numpy only; no Qt
    return subtitle_ocr


def _spk():
    from app.engines import speaker_embed  # numpy only; no Qt
    return speaker_embed


def _make_whisper_entry(size: str) -> dict:
    return {
        "label": f"Whisper {size} (speech-to-text)",
        "kind": "whisper",
        "size_mb": _WHISPER_SIZES[size],
        "check": (lambda s=size: _whisper_present(s)),
        "arg": size,
    }


MODELS: dict[str, dict] = {
    **{f"whisper-{s}": _make_whisper_entry(s) for s in _WHISPER_SIZES},
    "demucs-htdemucs": {
        "label": "Demucs htdemucs (vocal separation)",
        "kind": "url-torchhub",
        "size_mb": 319,
        "check": _demucs_present,
        "url": _DEMUCS_URL,
        "dest": _demucs_ckpt_path,
    },
    "mdx-vocals": {
        "label": "MDX-Net voice separator (keeps music without the original voices)",
        "kind": "url",
        "size_mb": 64,
        "check": lambda: _mdx().model_present(),
        "url": ("https://github.com/TRvlvr/model_repo/releases/download/"
                "all_public_uvr_models/Kim_Vocal_2.onnx"),
        "dest": lambda: _mdx().model_path(),
    },
    "lama": {
        "label": "LaMa ONNX (poster inpainting)",
        "kind": "url",
        "size_mb": 208,
        "check": _lama_present,
        "url": _LAMA_URL,
        "dest": _lama_path,
    },
    "paddleocr-ch": {
        "label": "PaddleOCR ch (poster text detection)",
        "kind": "paddleocr",
        "size_mb": 25,
        "check": _paddle_present,
    },
    "subtitle-ocr": {
        "label": "PP-OCRv5 subtitle reader (corrects speech recognition from hard subtitles)",
        "kind": "files",
        "size_mb": 21,
        "check": lambda: _subocr().models_present(),
        "dir": lambda: _subocr().model_dir(),
        "files": lambda: _subocr().FILES,
    },
    "speaker-campplus": {
        "label": "CAM++ speaker recognition (tells the characters' voices apart)",
        "kind": "files",
        "size_mb": 28,
        "check": lambda: _spk().model_present(),
        "dir": lambda: _spk().model_dir(),
        "files": lambda: _spk().FILES,
    },
}


def check_missing() -> list[str]:
    """Model ids whose files are not present on disk. Never raises."""
    missing = []
    for mid, entry in MODELS.items():
        try:
            present = bool(entry["check"]())
        except Exception:
            present = False
        if not present:
            missing.append(mid)
    return missing


def disk_free_gb(path: str | Path | None = None) -> float:
    try:
        usage = shutil.disk_usage(str(path or app_data_dir()))
        return round(usage.free / (1024 ** 3), 2)
    except OSError:
        return 0.0


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------

def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("cancelled")


def _download_url(url: str, dest: Path,
                  progress: Callable[[float, str], None] | None,
                  cancel: threading.Event | None,
                  expected_mb: int) -> None:
    """requests streaming download with Range resume via a .part file."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": "RecapStudio/1.0"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        r = requests.get(url, headers=headers, stream=True, timeout=30,
                         allow_redirects=True)
    except requests.RequestException as exc:
        raise AppError(
            f"Network error downloading model: {exc}",
            f"បញ្ហាបណ្ដាញពេលទាញយកម៉ូដែល: {exc}") from exc

    if resume_from > 0 and r.status_code == 416:
        # Range starts past the end: either the .part is already complete
        # (the app stopped between the last write and the rename) or it is
        # bigger than the file and broken
        size = r.headers.get("Content-Range", "").rpartition("/")[2]
        r.close()
        if size.isdigit() and int(size) == resume_from:
            dest.unlink(missing_ok=True)
            part.rename(dest)
            if progress is not None:
                progress(1.0, "Done / រួចរាល់")
            return
        part.unlink(missing_ok=True)
        return _download_url(url, dest, progress, cancel, expected_mb)
    if resume_from > 0 and r.status_code == 200:
        # server ignored Range — restart from scratch
        resume_from = 0
        part.unlink(missing_ok=True)
    elif r.status_code not in (200, 206):
        r.close()
        raise AppError(
            f"Model download failed (HTTP {r.status_code}): {url}",
            f"ការទាញយកម៉ូដែលបានបរាជ័យ (HTTP {r.status_code}): {url}")

    total = None
    cl = r.headers.get("Content-Length")
    if cl and cl.isdigit():
        total = int(cl) + resume_from
    exact = total is not None      # a known size the result must match
    if not total:
        total = expected_mb * 1024 * 1024

    done = resume_from
    mode = "ab" if resume_from > 0 else "wb"
    last = 0.0
    try:
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                _check_cancel(cancel)
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if progress is not None and (now - last) >= 0.2:
                    last = now
                    frac = min(0.999, done / total) if total else 0.0
                    progress(frac, f"{done / 1048576:.1f} / "
                                   f"{total / 1048576:.0f} MB")
    except requests.RequestException as exc:
        raise AppError(
            f"Connection lost while downloading model: {exc}",
            f"បាត់ការតភ្ជាប់ពេលទាញយកម៉ូដែល: {exc}") from exc
    finally:
        r.close()

    if exact and done != total:
        # the connection ended early without an error: keep the .part so the
        # next attempt resumes, and don't let a cut-off file pass as the model
        raise AppError(
            f"Model download stopped early ({done / 1048576:.1f} of "
            f"{total / 1048576:.1f} MB). Press Download again to resume.",
            f"ការទាញយកម៉ូដែលបានឈប់មុនពេល ({done / 1048576:.1f} នៃ "
            f"{total / 1048576:.1f} MB)។ សូមចុចទាញយកម្តងទៀតដើម្បីបន្ត។")
    dest.unlink(missing_ok=True)
    part.rename(dest)
    if progress is not None:
        progress(1.0, "Done / រួចរាល់")


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_versions(size, sha) -> list[tuple[int, str]]:
    """The known good (bytes, sha256) of a model file. `size` / `sha` are one
    value each, or tuples of the same length: a file re-published upstream
    with other bytes but the same model has more than one."""
    sizes = size if isinstance(size, (tuple, list)) else (size,)
    shas = sha if isinstance(sha, (tuple, list)) else (sha,)
    return list(zip(sizes, shas))


def _download_files(entry: dict,
                    progress: Callable[[float, str], None] | None,
                    cancel: threading.Event | None) -> None:
    """A model made of several files {name: (bytes, sha256, [mirror urls])}
    (bytes / sha256 may be tuples: every known good version, file_versions):
    each missing one from the first mirror that delivers a known version."""
    folder: Path = entry["dir"]()
    files: dict = entry["files"]()
    total = sum(max(s for s, _h in file_versions(size, sha))
                for size, sha, _u in files.values()) or 1
    done = 0
    for name, (sizes, shas, urls) in files.items():
        versions = file_versions(sizes, shas)
        size = max(s for s, _h in versions)
        dest = folder / name
        if dest.is_file() and dest.stat().st_size in {s for s, _h in versions}:
            done += size
            continue
        last_err: Exception | None = None
        for url in urls:
            _check_cancel(cancel)
            base = done

            def sub(frac: float, text: str, base=base, size=size) -> None:
                if progress is not None:
                    progress(min(0.999, (base + frac * size) / total), f"{name}: {text}")
            try:
                _download_url(url, dest, sub, cancel, max(1, size // 1048576))
            except CancelledError:
                raise
            except Exception as exc:          # next mirror
                last_err = exc
                log.warning("download of %s from %s failed: %s", name, url, exc)
                continue
            if (dest.stat().st_size, _sha256(dest)) in versions:
                last_err = None
                break
            log.warning("%s from %s is not the expected file; trying another mirror", name, url)
            dest.unlink(missing_ok=True)
            last_err = AppError(f"{name}: wrong file", f"{name}: ឯកសារមិនត្រឹមត្រូវ")
        if last_err is not None:
            raise AppError(f"Could not download {name}: {last_err}",
                           f"មិនអាចទាញយក {name} បានទេ៖ {last_err}")
        done += size
    if progress is not None:
        progress(1.0, "Done / រួចរាល់")


def _download_whisper(size: str,
                      progress: Callable[[float, str], None] | None,
                      cancel: threading.Event | None) -> None:
    try:
        import faster_whisper  # lazy heavy dep
    except ImportError as exc:
        raise EngineNotAvailable(
            f"faster-whisper is not installed. Run: {pip_install('faster-whisper')}",
            "faster-whisper មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ: "
            f"{pip_install('faster-whisper')}") from exc

    root = _whisper_dir(size)
    root.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(0.02, f"Downloading Whisper {size}… / "
                       f"កំពុងទាញយក Whisper {size}…")

    result: dict = {}

    def work() -> None:
        try:
            fn = getattr(faster_whisper, "download_model", None)
            if fn is not None:
                fn(size, output_dir=None, cache_dir=str(root))
            else:
                # older versions: instantiating triggers the download
                faster_whisper.WhisperModel(
                    size, device="cpu", compute_type="int8",
                    download_root=str(root))
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc

    t = threading.Thread(target=work, daemon=True, name=f"whisper-dl-{size}")
    t.start()
    frac = 0.05
    while t.is_alive():
        _check_cancel(cancel)
        t.join(0.5)
        if progress is not None and t.is_alive():
            frac = min(0.95, frac + 0.005)
            progress(frac, f"Downloading Whisper {size}… / "
                           f"កំពុងទាញយក Whisper {size}…")
    if "err" in result:
        exc = result["err"]
        raise AppError(
            f"Whisper {size} download failed: {exc}",
            f"ការទាញយក Whisper {size} បានបរាជ័យ: {exc}")
    if progress is not None:
        progress(1.0, "Done / រួចរាល់")


def _download_paddleocr(progress: Callable[[float, str], None] | None,
                        cancel: threading.Event | None) -> None:
    root = _paddle_root()
    root.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(0.02, "Downloading PaddleOCR models… / "
                       "កំពុងទាញយកម៉ូដែល PaddleOCR…")

    result: dict = {}

    def work() -> None:
        old_home = os.environ.get("HOME")
        old_base = os.environ.get("PADDLE_OCR_BASE_DIR")
        os.environ["HOME"] = str(root)
        os.environ["PADDLE_OCR_BASE_DIR"] = str(root)
        try:
            from paddleocr import PaddleOCR  # lazy heavy dep
            PaddleOCR(lang="ch")
            result["ok"] = True
        except ImportError:
            result["err"] = EngineNotAvailable(
                f"PaddleOCR is not installed. Run: {pip_install('paddleocr paddlepaddle')}",
                "PaddleOCR មិនទាន់បានដំឡើងទេ។ សូមដំឡើងដោយ: "
                f"{pip_install('paddleocr paddlepaddle')}")
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            if old_base is None:
                os.environ.pop("PADDLE_OCR_BASE_DIR", None)
            else:
                os.environ["PADDLE_OCR_BASE_DIR"] = old_base

    t = threading.Thread(target=work, daemon=True, name="paddleocr-dl")
    t.start()
    frac = 0.05
    while t.is_alive():
        _check_cancel(cancel)
        t.join(0.5)
        if progress is not None and t.is_alive():
            frac = min(0.95, frac + 0.01)
            progress(frac, "Downloading PaddleOCR models… / "
                           "កំពុងទាញយកម៉ូដែល PaddleOCR…")
    if "err" in result:
        err = result["err"]
        if isinstance(err, AppError):
            raise err
        raise AppError(
            f"PaddleOCR model fetch failed: {err}",
            f"ការទាញយកម៉ូដែល PaddleOCR បានបរាជ័យ: {err}")
    if progress is not None:
        progress(1.0, "Done / រួចរាល់")


def download_model(model_id: str,
                   progress: Callable[[float, str], None] | None = None,
                   cancel: threading.Event | None = None) -> None:
    """Download one model by registry id. Blocking; raises AppError on failure,
    CancelledError when cancel is set."""
    entry = MODELS.get(model_id)
    if entry is None:
        raise AppError(
            f"Unknown model id: {model_id}",
            f"មិនស្គាល់លេខសម្គាល់ម៉ូដែល: {model_id}")

    LOG_BUS.publish(f"[models] downloading {model_id} ({entry['label']})")
    _check_cancel(cancel)

    kind = entry["kind"]
    if kind == "whisper":
        _download_whisper(entry["arg"], progress, cancel)
    elif kind in ("url", "url-torchhub"):
        dest: Path = entry["dest"]()
        # demucs: prefer torch.hub's downloader (hash-checked) when torch is
        # importable; fall back to plain streaming otherwise.
        if kind == "url-torchhub":
            try:
                import torch  # lazy heavy dep
                dest.parent.mkdir(parents=True, exist_ok=True)
                if progress is not None:
                    progress(0.02, "Downloading htdemucs… / "
                                   "កំពុងទាញយក htdemucs…")
                result: dict = {}

                def work() -> None:
                    try:
                        torch.hub.download_url_to_file(
                            entry["url"], str(dest), progress=False)
                        result["ok"] = True
                    except Exception as exc:  # noqa: BLE001
                        result["err"] = exc

                t = threading.Thread(target=work, daemon=True,
                                     name="demucs-dl")
                t.start()
                frac = 0.05
                while t.is_alive():
                    _check_cancel(cancel)
                    t.join(0.5)
                    if progress is not None and t.is_alive():
                        if dest.exists():
                            frac = min(
                                0.98,
                                dest.stat().st_size /
                                (entry["size_mb"] * 1048576))
                        else:
                            frac = min(0.95, frac + 0.002)
                        progress(frac,
                                 "Downloading htdemucs… / "
                                 "កំពុងទាញយក htdemucs…")
                if "err" in result:
                    raise AppError(
                        f"htdemucs download failed: {result['err']}",
                        f"ការទាញយក htdemucs បានបរាជ័យ: {result['err']}")
                if progress is not None:
                    progress(1.0, "Done / រួចរាល់")
            except ImportError:
                _download_url(entry["url"], dest, progress, cancel,
                              entry["size_mb"])
        else:
            _download_url(entry["url"], dest, progress, cancel,
                          entry["size_mb"])
    elif kind == "paddleocr":
        _download_paddleocr(progress, cancel)
    elif kind == "files":
        _download_files(entry, progress, cancel)
    else:  # defensive; registry is static
        raise AppError(
            f"Unsupported model kind: {kind}",
            f"ប្រភេទម៉ូដែលមិនគាំទ្រ: {kind}")

    try:
        ok = bool(entry["check"]())
    except Exception:
        ok = False
    if not ok:
        raise AppError(
            f"Model {model_id} downloaded but verification failed.",
            f"បានទាញយកម៉ូដែល {model_id} ប៉ុន្តែការផ្ទៀងផ្ទាត់បានបរាជ័យ។")
    LOG_BUS.publish(f"[models] {model_id} ready")


# --------------------------------------------------------------------------
# Startup report
# --------------------------------------------------------------------------

def _gpu_info() -> tuple[str | None, float | None]:
    """(gpu name, vram GB) of the primary adapter, any vendor (DXGI), with
    nvidia-smi as a fallback; (None, None) on any failure."""
    try:
        from app.core.gpu import primary_gpu
        g = primary_gpu()
        if g is not None:
            return g.name, g.vram_gb
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, timeout=3,
            encoding="utf-8", errors="replace",
            creationflags=_CREATE_NO_WINDOW)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None, None
        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if parts and parts[0] else None
        vram = None
        if len(parts) > 1:
            m = parts[1].split()
            if m and m[0].replace(".", "", 1).isdigit():
                val = float(m[0])
                unit = m[1].upper() if len(m) > 1 else "MIB"
                if unit.startswith("MIB") or unit.startswith("MB"):
                    vram = round(val / 1024, 2)
                elif unit.startswith("GIB") or unit.startswith("GB"):
                    vram = round(val, 2)
                else:
                    vram = round(val / 1024, 2)
        return name, vram
    except Exception:
        return None, None


def _cuda_available() -> bool:
    try:
        import torch  # lazy heavy dep
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ffmpeg_ok() -> bool:
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-version"],
            capture_output=True, timeout=10,
            encoding="utf-8", errors="replace",
            creationflags=_CREATE_NO_WINDOW)
        return proc.returncode == 0
    except Exception:
        return False


def startup_report() -> dict:
    """Environment summary for the startup check dialog. Never raises."""
    report = {
        "gpu": None,
        "vram_gb": None,
        "cuda": False,
        "disk_free_gb": 0.0,
        "missing_models": [],
        "ffmpeg_ok": False,
        "gpu_vendor": None,     # nvidia | amd | intel | other
        "gpus": [],             # every adapter: {name, vendor, vram_gb, integrated}
        "accel": {},            # gpu.accel_summary(): what runs where
        "accel_hints": [],      # what to install to use the GPUs present
    }
    try:
        report["gpu"], report["vram_gb"] = _gpu_info()
    except Exception:
        pass
    try:
        from app.core import gpu
        gpus = gpu.list_gpus()
        report["gpus"] = gpu.to_json(gpus)
        p = gpu.primary_gpu(gpus)
        report["gpu_vendor"] = p.vendor if p else None
        report["accel"] = gpu.accel_summary(gpus)
        report["accel_hints"] = gpu.accel_hints(gpus)
    except Exception:
        pass
    try:
        report["cuda"] = _cuda_available()
    except Exception:
        pass
    try:
        report["disk_free_gb"] = disk_free_gb()
    except Exception:
        pass
    try:
        report["missing_models"] = check_missing()
    except Exception:
        pass
    try:
        report["ffmpeg_ok"] = _ffmpeg_ok()
    except Exception:
        pass
    return report


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    missing = check_missing()
    assert isinstance(missing, list)
    assert all(isinstance(m, str) and m in MODELS for m in missing)

    free = disk_free_gb()
    assert isinstance(free, float) and free >= 0.0
    assert disk_free_gb(".") >= 0.0

    for mid, entry in MODELS.items():
        assert set(entry) >= {"label", "kind", "size_mb", "check"}, mid
        assert callable(entry["check"]), mid
    assert len([m for m in MODELS if m.startswith("whisper-")]) == 5

    rep = startup_report()
    assert set(rep) == {"gpu", "vram_gb", "cuda", "disk_free_gb",
                        "missing_models", "ffmpeg_ok", "gpu_vendor", "gpus",
                        "accel", "accel_hints"}
    assert isinstance(rep["cuda"], bool)
    assert isinstance(rep["ffmpeg_ok"], bool)
    assert isinstance(rep["missing_models"], list)

    try:
        download_model("no-such-model")
        raise SystemExit("expected AppError")
    except AppError:
        pass

    ev = threading.Event()
    ev.set()
    try:
        download_model("lama", cancel=ev)
        raise SystemExit("expected CancelledError")
    except CancelledError:
        pass

    print("PASS")
