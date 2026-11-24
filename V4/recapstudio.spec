# -*- mode: python ; coding: utf-8 -*-
# PyInstaller one-dir build of RecapStudio. Run through build_exe.py.
#
# Layout matches app/core/paths.py when frozen (app_root() = the exe's folder):
#   dist/RecapStudio/RecapStudio.exe
#   dist/RecapStudio/app/assets/fonts/*.ttf
#   dist/RecapStudio/app/locales/{en,km}.json
#   dist/RecapStudio/bin/ffmpeg.exe, ffprobe.exe
# contents_directory="." keeps those beside the exe instead of in _internal/.
#
# The PyTorch stack (Demucs, speechbrain, OpenVoice, Seed-VC, F5/Cosy/VoxCPM)
# and PaddleOCR are excluded: they are several GB, and the app launches Demucs
# and pip as `sys.executable -m ...`, which a frozen exe cannot do. The frozen
# build covers downloads, Whisper (CTranslate2), Edge TTS, translation, recap,
# posters (OpenCV inpaint), overlays, export and publish.
#
# CUDA runtime DLLs are dropped too (PyInstaller finds cuBLAS/cuFFT in torch\lib
# and onnxruntime-gpu's 325 MB CUDA provider). Without the full cuDNN set,
# bundling cuBLAS alone would make faster-whisper pick CUDA and then crash
# mid-transcription; with none of them it falls back to CPU cleanly. Video
# encoding still uses NVENC/QSV/AMF through ffmpeg.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH)
# RECAP_CONSOLE=1 builds with a console window, to see startup tracebacks.
CONSOLE = os.environ.get("RECAP_CONSOLE") == "1"

datas = [
    (str(ROOT / "app" / "assets"), "app/assets"),
    (str(ROOT / "app" / "locales"), "app/locales"),
]
binaries = [(str(p), "bin") for p in (ROOT / "bin").glob("*.exe")]
datas += [(str(p), "bin") for p in (ROOT / "bin").glob("*.txt")]

excludes = [
    "torch", "torchaudio", "torchvision", "demucs", "speechbrain",
    "openvoice", "seed_vc", "f5_tts", "cosyvoice", "voxcpm",
    "paddle", "paddleocr", "tensorboard", "matplotlib",
    # pulled in by other packages' hooks; the app never imports them
    "transformers", "IPython", "jedi", "pandas", "librosa", "numba",
    "llvmlite", "sklearn", "scipy", "sympy", "onnxruntime.transformers",
    # a stale vendored pkg_resources crashes its runtime hook (needs appdirs);
    # only onnxruntime's benchmark scripts import it
    "pkg_resources",
    "tkinter", "PySide6.Qt3DCore", "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets", "PySide6.QtQuick3D",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    # tabs are imported by name (main_window._TAB_SPECS), so list all of app.*
    hiddenimports=collect_submodules("app") + [
        "edge_tts", "faster_whisper", "yt_dlp", "uharfbuzz", "freetype"],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

_CUDA_DLLS = ("cublas", "cufft", "cudart", "curand", "cusparse", "cusolver",
              "nvrtc", "nvjitlink", "nvinfer", "nvonnxparser",
              "onnxruntime_providers_cuda", "onnxruntime_providers_tensorrt")


def _keep(entry) -> bool:
    dest = entry[0].replace("\\", "/").lower()
    return not (dest.startswith("torch/")
                or any(k in dest.rsplit("/", 1)[-1] for k in _CUDA_DLLS))


a.binaries = [b for b in a.binaries if _keep(b)]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RecapStudio",
    console=CONSOLE,
    upx=False,
    contents_directory=".",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="RecapStudio",
)
