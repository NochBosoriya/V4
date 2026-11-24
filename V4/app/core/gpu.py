"""Vendor-neutral GPU detection and compute-backend selection.

Adapters are enumerated through DXGI (works for NVIDIA, AMD and Intel on
Windows 10/11, exact dedicated VRAM, no subprocess); PowerShell
Win32_VideoController is the fallback. Compute backends are probed lazily:

  cuda      PyTorch CUDA (Demucs, OpenVoice, speechbrain) — NVIDIA only
  ct2_cuda  CTranslate2 CUDA (faster-whisper) — NVIDIA only
  onnx      onnxruntime execution providers (CUDA / DirectML / CPU) — LaMa

No Qt imports; heavy deps (torch, ctranslate2, onnxruntime) are lazy.
"""
from __future__ import annotations

import ctypes
import json
import re
import subprocess
import threading
from dataclasses import asdict, dataclass

from app.core.logger import get_logger
from app.core.paths import pip_install

log = get_logger("gpu")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel",
           0x1414: "microsoft"}
VENDOR_LABEL = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel",
                "microsoft": "Microsoft", "other": "Other"}

# Hardware video encoders per vendor, in preference order.
VENDOR_ENCODERS = {
    "nvidia": ["h264_nvenc", "hevc_nvenc"],
    "amd": ["h264_amf", "hevc_amf"],
    "intel": ["h264_qsv", "hevc_qsv"],
}


@dataclass
class GpuInfo:
    name: str
    vendor: str              # nvidia | amd | intel | microsoft | other
    vram_gb: float | None    # dedicated video memory
    integrated: bool         # iGPU / APU (little or no dedicated memory)
    device_id: int = 0

    @property
    def label(self) -> str:
        return self.name + (f" ({self.vram_gb:.0f} GB)" if self.vram_gb else "")


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------

class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", _LUID), ("Flags", ctypes.c_uint)]


# IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_FACTORY1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA,
                      (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C,
                                           0x83, 0xD1, 0xB3, 0x87))
_DXGI_ERROR_NOT_FOUND = 0x887A0002
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
# vtable slots: IUnknown(3) + IDXGIObject(4) + IDXGIFactory(5) -> EnumAdapters1
_SLOT_RELEASE, _SLOT_ENUM_ADAPTERS1, _SLOT_GET_DESC1 = 2, 12, 10


def _com_call(obj: ctypes.c_void_p, slot: int, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(vtbl[slot])


def _vendor(vid: int) -> str:
    return VENDORS.get(int(vid), "other")


def _is_integrated(vendor: str, vram_gb: float | None, name: str) -> bool:
    if vendor == "intel" and "arc" not in name.lower():
        return True
    # APUs / iGPUs only get a small carve-out of dedicated memory
    return vram_gb is not None and vram_gb < 1.0


def _enum_dxgi() -> list[GpuInfo]:
    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    create = dxgi.CreateDXGIFactory1
    create.restype = ctypes.c_long
    create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    hr = create(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory))
    if hr != 0 or not factory:
        raise OSError(f"CreateDXGIFactory1 failed: 0x{hr & 0xFFFFFFFF:08X}")
    out: list[GpuInfo] = []
    try:
        enum = _com_call(factory, _SLOT_ENUM_ADAPTERS1, ctypes.c_long,
                         ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while i < 16:
            adapter = ctypes.c_void_p()
            hr = enum(factory, i, ctypes.byref(adapter))
            if (hr & 0xFFFFFFFF) == _DXGI_ERROR_NOT_FOUND or not adapter:
                break
            i += 1
            try:
                desc = _DXGI_ADAPTER_DESC1()
                get = _com_call(adapter, _SLOT_GET_DESC1, ctypes.c_long,
                                ctypes.POINTER(_DXGI_ADAPTER_DESC1))
                if get(adapter, ctypes.byref(desc)) != 0:
                    continue
                if desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE:
                    continue
                vendor = _vendor(desc.VendorId)
                if vendor == "microsoft":       # Basic Render Driver
                    continue
                vram = round(desc.DedicatedVideoMemory / 1024 ** 3, 2) or None
                name = desc.Description.strip()
                out.append(GpuInfo(name, vendor, vram,
                                   _is_integrated(vendor, vram, name),
                                   int(desc.DeviceId)))
            finally:
                _com_call(adapter, _SLOT_RELEASE, ctypes.c_ulong)(adapter)
    finally:
        _com_call(factory, _SLOT_RELEASE, ctypes.c_ulong)(factory)
    # the same adapter can appear once per output; de-duplicate
    seen, uniq = set(), []
    for g in out:
        key = (g.vendor, g.device_id, g.name)
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    return uniq


def _enum_powershell() -> list[GpuInfo]:
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           "Get-CimInstance Win32_VideoController | "
           "Select-Object Name,PNPDeviceID,AdapterRAM | ConvertTo-Json -Compress"]
    cp = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                        errors="replace", timeout=15,
                        creationflags=_CREATE_NO_WINDOW)
    data = json.loads(cp.stdout or "[]")
    if isinstance(data, dict):
        data = [data]
    out = []
    for d in data:
        pnp = str(d.get("PNPDeviceID") or "").upper()
        vid = 0
        if "VEN_" in pnp:
            try:
                vid = int(pnp.split("VEN_")[1][:4], 16)
            except ValueError:
                vid = 0
        vendor = _vendor(vid)
        if vendor == "microsoft":
            continue
        ram = d.get("AdapterRAM")
        # AdapterRAM is a uint32: it caps at 4 GB, so treat 4 GB as "≥ 4"
        vram = round(int(ram) / 1024 ** 3, 2) if ram else None
        name = str(d.get("Name") or "GPU").strip()
        out.append(GpuInfo(name, vendor, vram, _is_integrated(vendor, vram, name)))
    return out


_GPUS: list[GpuInfo] | None = None
_LOCK = threading.Lock()


def list_gpus(refresh: bool = False) -> list[GpuInfo]:
    """All hardware display adapters (cached). Never raises."""
    global _GPUS
    with _LOCK:
        if _GPUS is not None and not refresh:
            return list(_GPUS)
        gpus: list[GpuInfo] = []
        for fn in (_enum_dxgi, _enum_powershell):
            try:
                gpus = fn()
                if gpus:
                    break
            except Exception as exc:
                log.info("GPU enumeration via %s failed: %s", fn.__name__, exc)
        _GPUS = gpus
        return list(gpus)


_RANK = {"nvidia": 0, "amd": 1, "intel": 2, "other": 3}


def primary_gpu(gpus: list[GpuInfo] | None = None) -> GpuInfo | None:
    """The adapter heavy work should go to: discrete before integrated,
    NVIDIA (CUDA) first, then the most VRAM."""
    gpus = list_gpus() if gpus is None else gpus
    if not gpus:
        return None
    return sorted(gpus, key=lambda g: (g.integrated, _RANK.get(g.vendor, 9),
                                       -(g.vram_gb or 0.0)))[0]


def vendors(gpus: list[GpuInfo] | None = None) -> set[str]:
    return {g.vendor for g in (list_gpus() if gpus is None else gpus)}


# ---------------------------------------------------------------------------
# compute backends
# ---------------------------------------------------------------------------

def torch_cuda() -> bool:
    try:
        import torch  # lazy heavy import
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# NVIDIA's cuBLAS for faster-whisper (CTranslate2 ships cuDNN itself): the
# newest release with a Windows wheel, checked 2026-11-10 (553 MB). CUDA 12
# runs on GTX 900-series (Maxwell) cards and newer, RTX 50-series included.
CUBLAS_PIP = "nvidia-cublas-cu12==12.9.2.10"
_DLL_DIRS: set[str] = set()


def add_cuda_dll_dirs() -> list[str]:
    """Put the bin folders of NVIDIA's pip packages (nvidia-cublas-cu12 …)
    on the DLL search path and PATH, so CTranslate2 finds cuBLAS. Safe to
    call again (after an install, new folders are added)."""
    import importlib.util
    import os
    from pathlib import Path
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    roots = list(spec.submodule_search_locations or []) if spec else []
    found = []
    for root in roots:
        for sub in sorted(Path(root).glob("*/bin")):
            d = str(sub)
            found.append(d)
            if d not in _DLL_DIRS:
                _DLL_DIRS.add(d)
                try:
                    os.add_dll_directory(d)
                except (OSError, AttributeError):
                    pass
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return found


def cuda_libs_ok() -> bool:
    """cuBLAS can be loaded. The driver alone lets CTranslate2 count the GPU,
    but a Whisper model then cannot run on it."""
    import os
    if os.name != "nt":
        return True
    add_cuda_dll_dirs()
    for name in ("cublas64_12.dll", "cublas64_11.dll"):
        try:
            ctypes.WinDLL(name)
            return True
        except OSError:
            pass
    return False


def ct2_cuda() -> bool:
    """faster-whisper (CTranslate2) can really run on an NVIDIA GPU here."""
    try:
        import ctranslate2  # lazy heavy import
        return ctranslate2.get_cuda_device_count() > 0 and cuda_libs_ok()
    except Exception:
        return False


def ct2_cuda_compute_types() -> list[str]:
    """The CUDA compute types to try for Whisper, fastest first: float16 on
    cards that do it well, int8 / float32 on older ones."""
    try:
        import ctranslate2
        have = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return []
    return [t for t in ("float16", "int8_float16", "int8_float32", "float32") if t in have]


def onnx_providers() -> list[str]:
    """onnxruntime providers to use, best first (CUDA > DirectML > CPU)."""
    try:
        import onnxruntime as ort  # lazy heavy import
        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    order = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider")
             if p in avail]
    return order + ["CPUExecutionProvider"]


def accel_summary(gpus: list[GpuInfo] | None = None) -> dict:
    """What each kind of work will run on."""
    gpus = list_gpus() if gpus is None else gpus
    p = primary_gpu(gpus)
    tc, wc = torch_cuda(), ct2_cuda()
    onnx = onnx_providers()
    onnx_dev = ("CUDA" if onnx[0] == "CUDAExecutionProvider" else
                "DirectML" if onnx[0] == "DmlExecutionProvider" else "CPU")
    return {
        "gpu": p.name if p else None,
        "vendor": p.vendor if p else None,
        "torch": "CUDA" if tc else "CPU",      # Demucs, OpenVoice, speechbrain
        "whisper": "CUDA" if wc else "CPU",
        "onnx": onnx_dev,                      # LaMa inpainting
        "encoders": [e for v in sorted(vendors(gpus), key=lambda v: _RANK.get(v, 9))
                     for e in VENDOR_ENCODERS.get(v, [])],
    }


# NVIDIA cards CUDA 13 dropped (Maxwell, Pascal, Volta): GTX 9xx / 10xx,
# MX 1xx-3xx, GT 10xx, TITAN X / Xp / V, Quadro M / P / GV, Tesla M / P / V
_PRE_TURING = re.compile(r"GTX\s*(9|10)\d\d|\bMX\s*[1-3]\d\d\b|\bGT\s*10\d\d|"
                         r"TITAN\s*(X|Xp|V)\b|Quadro\s*(M|P|GV)\d|Tesla\s*[MPV]\d", re.I)


def torch_cuda_index(gpus: list[GpuInfo]) -> str:
    """PyTorch wheel index for the NVIDIA GPUs present (download.pytorch.org,
    checked 2026-11-03): CUDA 13.2 builds (torch 2.15) for RTX 20xx / GTX
    16xx and newer (RTX 50xx needs CUDA 12.8+); CUDA 12.6 (torch 2.14) for
    older cards, which CUDA 13 no longer supports."""
    names = [g.name for g in gpus if g.vendor == "nvidia"]
    old = bool(names) and all(_PRE_TURING.search(n) for n in names)
    return "https://download.pytorch.org/whl/" + ("cu126" if old else "cu132")


def accel_hints(gpus: list[GpuInfo] | None = None) -> list[str]:
    """Plain-English notes on what to install to use the GPUs present."""
    gpus = list_gpus() if gpus is None else gpus
    vs = vendors(gpus)
    hints = []
    if "nvidia" in vs and not torch_cuda():
        # --force-reinstall: a plain install keeps the CPU build ("already
        # satisfied"); --no-deps: the rest stays as it is (CUDA is inside torch)
        hints.append("NVIDIA GPU found but PyTorch is CPU-only: install the CUDA "
                     "build of PyTorch ("
                     + pip_install("--force-reinstall --no-deps torch --index-url "
                                   + torch_cuda_index(gpus))
                     + ").")
    if vs & {"amd", "intel"} and "nvidia" not in vs:
        prov = onnx_providers()
        if "DmlExecutionProvider" not in prov:
            hints.append("AMD/Intel GPU: install onnxruntime-directml (instead "
                         "of onnxruntime-gpu) so poster inpainting uses the GPU.")
        hints.append("AMD/Intel GPU: speech recognition, voice separation and "
                     "voice cloning run on the CPU (they need NVIDIA CUDA); "
                     "video export uses the GPU encoder.")
    return hints


def to_json(gpus: list[GpuInfo] | None = None) -> list[dict]:
    return [asdict(g) for g in (list_gpus() if gpus is None else gpus)]


if __name__ == "__main__":
    failures = []

    def check(name: str, cond: bool) -> None:
        print(("ok:   " if cond else "FAIL: ") + name)
        if not cond:
            failures.append(name)

    gs = [GpuInfo("Intel(R) UHD Graphics 630", "intel", 0.12, True),
          GpuInfo("AMD Radeon RX 6600", "amd", 7.98, False),
          GpuInfo("NVIDIA GeForce GTX 1660 Ti", "nvidia", 5.85, False)]
    check("primary prefers discrete NVIDIA", primary_gpu(gs).vendor == "nvidia")
    check("primary prefers discrete AMD over iGPU",
          primary_gpu(gs[:2]).vendor == "amd")
    check("iGPU alone is still primary", primary_gpu(gs[:1]).vendor == "intel")
    check("no GPUs -> None", primary_gpu([]) is None)
    check("integrated heuristic", _is_integrated("intel", 0.1, "Intel UHD")
          and not _is_integrated("intel", 16.0, "Intel(R) Arc(TM) A770")
          and _is_integrated("amd", 0.5, "AMD Radeon(TM) Graphics")
          and not _is_integrated("nvidia", 6.0, "RTX"))
    check("vendor ids", _vendor(0x10DE) == "nvidia" and _vendor(0x1002) == "amd"
          and _vendor(0x8086) == "intel" and _vendor(0x1234) == "other")
    s = accel_summary(gs)
    check("encoders ordered by vendor",
          s["encoders"] == ["h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf",
                            "h264_qsv", "hevc_qsv"])
    check("onnx providers end with CPU", onnx_providers()[-1] == "CPUExecutionProvider")

    def nv(name: str) -> GpuInfo:
        return GpuInfo(name, "nvidia", 8.0, False)
    check("PyTorch CUDA build fits the card",
          torch_cuda_index([nv("NVIDIA GeForce RTX 5070 Laptop GPU")]).endswith("cu132")
          and torch_cuda_index([nv("NVIDIA GeForce GTX 1660 Ti")]).endswith("cu132")
          and torch_cuda_index([nv("NVIDIA GeForce GTX 1060 6GB")]).endswith("cu126")
          and torch_cuda_index([nv("NVIDIA GeForce MX150")]).endswith("cu126")
          and torch_cuda_index([nv("NVIDIA GeForce MX450")]).endswith("cu132")
          and torch_cuda_index([nv("NVIDIA GeForce GTX 1080"), nv("NVIDIA GeForce RTX 3060")])
          .endswith("cu132"))
    real = list_gpus()
    print("this PC:", [g.label + f" [{g.vendor}{', integrated' if g.integrated else ''}]"
                       for g in real])
    print("accel:", accel_summary())
    print("FAIL" if failures else "PASS")
    raise SystemExit(1 if failures else 0)
