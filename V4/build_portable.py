"""Build dist/RecapStudio-portable/ and dist/RecapStudio-portable.zip:
RecapStudio with its own Python, to hand to other people. They unzip it and
double-click RecapStudio.exe (or "Start RecapStudio.bat"); nothing is installed.

RecapStudio.exe is python.org's own pythonw.exe under the app's name: its
Python Software Foundation signature stays valid when renamed, so Smart App
Control lets it run (an exe built for the app — PyInstaller — is unsigned
and gets blocked). Python's core DLLs sit next to it, RecapStudio._pth points
it at the Python in python\\, and sitecustomize.py starts main.py.

    python build_portable.py            # folder + zip
    python build_portable.py --no-zip   # folder only

Inside: python.org's embeddable Python of this Python's version (its files
are signed by the Python Software Foundation, checked here), the packages
this Python has installed except build / dev tools, Microsoft's C++ runtime,
the app (code, fonts, translations) and ffmpeg. Never inside: settings, API
keys, logins, projects or models. They live in each user's
%LOCALAPPDATA%\\RecapStudio, so every person adds their own key.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
OUT = DIST / "RecapStudio-portable"
CACHE = ROOT / "build" / "portable"
MARKER = ".recapstudio-portable"     # only a folder holding this is ever deleted

# build / dev tools, and PySide6's web browser (349 MB: the app never uses it,
# the exe leaves it out too)
SKIP_DISTS = {"pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile",
              "pyflakes", "virtualenv", "distlib", "python-discovery",
              "pyside6-webengine",
              # NVIDIA's cuBLAS (553 MB) for Whisper on the GPU: only NVIDIA PCs
              # need it, and the Full Dub checklist installs it there in one click
              "nvidia-cublas-cu12"}
# Microsoft's C++ runtime: torch, ctranslate2, onnxruntime and OpenCV use it
# without shipping it, and a freshly installed Windows may not have it
CPP_RUNTIME = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
               "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
               "concrt140.dll", "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
# packages loaded from the user's own Python folder would mix in: off for the
# app and everything it starts (pip, Demucs)
NO_USER_SITE = "set PYTHONNOUSERSITE=1"

START_BAT = rf"""@echo off
rem RecapStudio (portable): runs with the Python in this folder; nothing is installed.
{NO_USER_SITE}
start "" "%~dp0python\pythonw.exe" "%~dp0main.py"
"""
DEBUG_BAT = rf"""@echo off
rem Same as "Start RecapStudio.bat", in a window that shows any error message.
{NO_USER_SITE}
"%~dp0python\python.exe" "%~dp0main.py"
echo.
pause
"""
# imported at start-up by every Python that has this folder on its path; it
# only acts for RecapStudio.exe (python\python.exe and the app's helper
# processes pass straight through)
LAUNCHER = r'''"""Opens RecapStudio when RecapStudio.exe is started (portable copy).

RecapStudio.exe is python.org's own pythonw.exe under the app's name: its
Python Software Foundation signature stays valid, so Smart App Control lets
it run. Python reads RecapStudio._pth next to it and imports this file at
start-up; here it starts main.py. Any other Python start passes through."""
import os
import sys


def _launch() -> None:
    root = os.path.dirname(os.path.abspath(sys.executable))
    py = os.path.join(root, "python")
    first = (sys.argv[0] if getattr(sys, "argv", None) else "") or ""
    if first.lower().endswith((".py", ".pyw", ".pyz")) or first in ("-c", "-m"):
        return                      # Python itself asked to run something
    import site
    # the packages' own start files (.pth) and the DLLs beside them
    site.addsitedir(os.path.join(py, "Lib", "site-packages"))
    os.add_dll_directory(py)
    # packages in the user's own Python folder stay out, for the helpers too
    os.environ["PYTHONNOUSERSITE"] = "1"
    user = os.path.normcase(getattr(site, "USER_SITE", "") or "?")
    sys.path[:] = [d for d in sys.path if not os.path.normcase(d).startswith(user)]
    # the helpers the app starts (Demucs, pip) run on the folder's pythonw
    for name in ("executable", "_base_executable"):
        setattr(sys, name, os.path.join(py, "pythonw.exe"))
    code = 0
    probe = os.environ.pop("RECAPSTUDIO_LAUNCH_PROBE", "")
    try:
        if probe:                   # the build's check: it starts, nothing shown
            import importlib
            for mod in ("PySide6.QtWidgets", "numpy", "onnxruntime", "cv2", "app.core.paths"):
                importlib.import_module(mod)
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write(f"ok|{sys.version.split()[0]}|{sys.executable}|{root}")
        else:
            import runpy
            main = os.path.join(root, "main.py")
            sys.argv = [main]
            runpy.run_path(main, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        code = 1
        text = traceback.format_exc()
        try:
            logs = os.path.join(os.environ.get("LOCALAPPDATA", root), "RecapStudio", "logs")
            os.makedirs(logs, exist_ok=True)
            with open(os.path.join(logs, "start-error.txt"), "w", encoding="utf-8") as fh:
                fh.write(text)
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, "RecapStudio could not start:\n\n" + text[-1500:]
                + "\n\nDouble-click \"Start RecapStudio (show errors).bat\" for more.",
                "RecapStudio", 0x10)
        except Exception:
            pass
    try:
        import atexit
        atexit._run_exitfuncs()     # logging and the app's own clean-up
    except Exception:
        pass
    os._exit(code)


if os.path.basename(sys.executable).lower() == "recapstudio.exe":
    _launch()
'''

HOW_TO = r"""RecapStudio — portable version (nothing to install)

Start
1. If your Windows has Smart App Control on: BEFORE unzipping, right-click the
   zip > Properties > tick "Unblock" > OK. Do this only for a zip from someone
   you trust. Without it, Windows blocks the start file.
2. Unzip (Extract) the whole folder anywhere you like. It needs about 4 GB.
3. Double-click RecapStudio.exe (or "Start RecapStudio.bat"). RecapStudio.exe
   is Python's own signed program under the app's name: Smart App Control
   lets it run. It has Python's icon.
4. Open Settings and add your own AI key (API key). Speech-recognition
   models download the first time you use them.

If it does not open
Double-click "Start RecapStudio (show errors).bat": a black window shows what
went wrong.

To remove RecapStudio, delete this folder. Your settings and projects are kept
in %LOCALAPPDATA%\RecapStudio.

────────────────────────────────────────────

RecapStudio — កំណែ portable (មិនចាំបាច់ដំឡើង)

ការចាប់ផ្ដើម
1. បើ Windows របស់អ្នកបើក Smart App Control៖ មុនពេលពន្លា (Extract) zip សូមចុច
   កណ្ដុរស្ដាំលើឯកសារ zip > Properties > គូសធីក "Unblock" > OK។ ធ្វើដូច្នេះតែ
   ចំពោះ zip ពីអ្នកដែលអ្នកទុកចិត្តប៉ុណ្ណោះ។ បើមិនធ្វើ Windows នឹងរារាំងឯកសារចាប់ផ្ដើម។
2. ពន្លាថតទាំងមូលទៅកន្លែងណាក៏បាន។ ត្រូវការទំហំទំនេរប្រហែល 4 GB។
3. ចុចពីរដងលើ RecapStudio.exe (ឬ "Start RecapStudio.bat")។ RecapStudio.exe
   គឺជាកម្មវិធី Python ដែលមានហត្ថលេខាឌីជីថលផ្ទាល់ ដាក់ឈ្មោះកម្មវិធីនេះ៖
   Smart App Control អនុញ្ញាតឲ្យវាដំណើរការ។ វាមានរូបតំណាងរបស់ Python។
4. បើក Settings ហើយបញ្ចូលសោ AI (API key) របស់អ្នកផ្ទាល់។ ម៉ូដែលស្គាល់សំឡេង
   នឹងទាញយកនៅពេលប្រើលើកដំបូង។

បើកម្មវិធីមិនបើក
ចុចពីរដងលើ "Start RecapStudio (show errors).bat"៖ ផ្ទាំងខ្មៅនឹងបង្ហាញបញ្ហា។

ដើម្បីលុប RecapStudio គ្រាន់តែលុបថតនេះ។ ការកំណត់ និងគម្រោងរបស់អ្នក
ត្រូវបានរក្សាទុកនៅ %LOCALAPPDATA%\RecapStudio។
"""


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _say(text: str) -> None:
    print(text, flush=True)


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def packages_dir() -> Path:
    """Where this Python's packages are (Store Python: its LocalCache folder)."""
    return Path(md.distribution("PySide6").locate_file("")).resolve()


def left_out(sp: Path) -> tuple[set[str], set[str], list[str]]:
    """What to leave out of site-packages (paths relative to it, lower-case):
    the files only SKIP_DISTS own (PySide6_WebEngine also lists PySide6's
    shared __init__.py, which must stay), the top-level folders and modules
    only they own (so no empty package or .dist-info folder is left behind),
    and the names of those packages. A SKIP_DIST a kept package needs stays."""
    dists = {_norm(d.metadata["Name"]): d for d in md.distributions()}
    needed: dict[str, set[str]] = {}
    for name, d in dists.items():
        for req in d.requires or []:
            if "extra ==" not in req:
                dep = _norm(re.split(r"[ ;<>=!~\[(]", req, maxsplit=1)[0])
                needed.setdefault(dep, set()).add(name)
    names = []
    for name in sorted(SKIP_DISTS & set(dists)):
        users = needed.get(name, set()) - SKIP_DISTS - {"pyside6"}   # PySide6 only lists it
        if users:
            _say(f"  keeping {name}: needed by {', '.join(sorted(users))}")
        else:
            names.append(name)
    file_owners: dict[str, set[str]] = {}     # file -> packages listing it
    top_owners: dict[str, set[str]] = {}      # top-level entry -> packages with files in it
    for name, d in dists.items():
        for f in d.files or []:
            p = Path(os.path.normpath(sp / f))
            if not p.is_relative_to(sp):
                continue                      # a script in Scripts\: not copied anyway
            rel = p.relative_to(sp)
            file_owners.setdefault(rel.as_posix().lower(), set()).add(name)
            top_owners.setdefault(rel.parts[0].lower(), set()).add(name)
    gone = set(names)
    files = {f for f, who in file_owners.items() if who <= gone}
    tops = {t for t, who in top_owners.items() if who <= gone}
    return files, tops, names


def embeddable_zip(version: str) -> Path:
    """python.org's embeddable Python of this version (downloaded once)."""
    name = f"python-{version}-embed-amd64.zip"
    path = CACHE / name
    if not path.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        url = f"https://www.python.org/ftp/python/{version}/{name}"
        _say(f"  downloading {url}")
        part = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as fh:
            shutil.copyfileobj(r, fh)
        part.replace(path)
    return path


def check_signed(files: list[Path]) -> None:
    """Every file must carry a valid Python Software Foundation signature."""
    paths = ",".join("'" + str(f).replace("'", "''") + "'" for f in files)
    cmd = (f"foreach ($p in @({paths})) {{ $s = Get-AuthenticodeSignature -LiteralPath $p; "
           "'{0}|{1}' -f $s.Status, $s.SignerCertificate.Subject }")
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                         capture_output=True, text=True, timeout=120).stdout.splitlines()
    for f, line in zip(files, out):
        status, _, subject = line.partition("|")
        if status != "Valid" or "Python Software Foundation" not in subject:
            raise SystemExit(f"{f.name} is not signed by the Python Software Foundation "
                             f"({line}) — not building with it")
    if len(out) != len(files):
        raise SystemExit("could not check the Python signatures (PowerShell)")


def prepare_out() -> None:
    if OUT.exists():
        if not (OUT / MARKER).is_file():
            raise SystemExit(f"{OUT} exists and was not made by this script — move it away first")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / MARKER).write_text("made by build_portable.py\n", encoding="utf-8")


def put_python(version: str) -> Path:
    py = OUT / "python"
    with zipfile.ZipFile(embeddable_zip(version)) as z:
        z.extractall(py)
    # every program file of it (vcruntime140*.dll are Microsoft's)
    check_signed(sorted(f for f in py.iterdir() if f.suffix.lower() in (".exe", ".dll", ".pyd")
                        and not f.name.lower().startswith("vcruntime")))
    pth = next(py.glob("python*._pth"))
    stdlib = next(py.glob("python3*.zip")).name
    # the standard library, this folder, the packages, and the app one level up
    pth.write_text(f"{stdlib}\n.\nLib\\site-packages\n..\nimport site\n", encoding="utf-8")
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for dll in CPP_RUNTIME:
        if not (py / dll).exists() and (system32 / dll).is_file():
            shutil.copy2(system32 / dll, py / dll)
    return py


def put_packages(py: Path) -> None:
    sp = packages_dir()
    files, tops, names = left_out(sp)
    _say(f"  from {sp}")
    _say(f"  left out: {', '.join(names) or 'nothing'}")

    def ignore(folder: str, entries: list[str]) -> set[str]:
        rel = Path(folder).resolve().relative_to(sp).as_posix().lower()
        if rel == ".":
            return {e for e in entries if e.lower() in tops or e.lower() in files}
        return {e for e in entries if f"{rel}/{e.lower()}" in files}

    shutil.copytree(sp, py / "Lib" / "site-packages", ignore=ignore, dirs_exist_ok=True)


def put_app() -> None:
    no_cache = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "app", OUT / "app", ignore=no_cache)
    shutil.copy2(ROOT / "main.py", OUT / "main.py")
    (OUT / "bin").mkdir()
    for f in list((ROOT / "bin").glob("*.exe")) + list((ROOT / "bin").glob("*.txt")):
        shutil.copy2(f, OUT / "bin" / f.name)
    # .bat files need Windows line ends; the note opens right in Notepad
    (OUT / "Start RecapStudio.bat").write_text(START_BAT, encoding="ascii", newline="\r\n")
    (OUT / "Start RecapStudio (show errors).bat").write_text(DEBUG_BAT, encoding="ascii",
                                                             newline="\r\n")
    (OUT / "How to start.txt").write_text(HOW_TO, encoding="utf-8-sig", newline="\r\n")


def put_launcher(py: Path) -> None:
    """RecapStudio.exe: python.org's PSF-signed pythonw.exe under the app's
    name (renaming keeps the signature: Smart App Control lets it run),
    Python's core DLLs beside it (it loads them from its own folder), a
    ._pth pointing at python\\ and the launcher."""
    shutil.copy2(py / "pythonw.exe", OUT / "RecapStudio.exe")
    core = [f.name for f in py.glob("python3*.dll")] + [
        f.name for f in py.glob("vcruntime140*.dll")]
    for name in core:
        shutil.copy2(py / name, OUT / name)
    stdlib = next(py.glob("python3*.zip")).name
    # this folder first: its sitecustomize.py is the launcher
    (OUT / "RecapStudio._pth").write_text(
        f".\npython\\{stdlib}\npython\npython\\Lib\\site-packages\nimport site\n",
        encoding="utf-8")
    (OUT / "sitecustomize.py").write_text(LAUNCHER, encoding="utf-8")
    check_signed([OUT / "RecapStudio.exe"] + sorted(
        OUT / n for n in core if not n.lower().startswith("vcruntime")))


def check_launcher() -> None:
    """RecapStudio.exe starts (Smart App Control lets it run on this PC), finds
    its Python and packages and the app, and hands helpers to python\\pythonw."""
    import tempfile
    report = Path(tempfile.mkdtemp(prefix="rs_launch_")) / "probe.txt"
    env = dict(os.environ, RECAPSTUDIO_LAUNCH_PROBE=str(report))
    env.pop("PYTHONPATH", None)
    try:
        rc = subprocess.run([str(OUT / "RecapStudio.exe")], cwd=OUT, env=env, timeout=180).returncode
    except OSError as e:
        raise SystemExit(f"RecapStudio.exe could not start: {e}")
    got = report.read_text(encoding="utf-8") if report.is_file() else ""
    shutil.rmtree(report.parent, ignore_errors=True)
    ok, _, rest = got.partition("|")
    if rc != 0 or ok != "ok":
        raise SystemExit(f"RecapStudio.exe did not start the app (code {rc}): {got or 'no report'}")
    version, helper, root = (rest.split("|") + ["", "", ""])[:3]
    if Path(helper).resolve() != (OUT / "python" / "pythonw.exe").resolve():
        raise SystemExit(f"RecapStudio.exe hands helpers to {helper}, not python\\pythonw.exe")
    _say(f"  RecapStudio.exe starts: Python {version}, helpers on python\\pythonw.exe")


def check_python(py: Path) -> None:
    """Compile the app with the portable Python and import the main packages:
    proves it runs on its own (no user folder, nothing from this Python)."""
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    exe = str(py / "python.exe")
    subprocess.run([exe, "-m", "compileall", "-q", "app", "main.py"], cwd=OUT, env=env, check=True)
    probe = ("import sys, PySide6.QtWidgets, numpy, ctranslate2, faster_whisper, onnxruntime, "
             "cv2, edge_tts, yt_dlp, pypinyin, jieba, app.core.paths as p; "
             "assert sys.prefix.lower().startswith(%r), sys.prefix; "
             "print('  Python', sys.version.split()[0], '| app root', p.app_root())" % str(py).lower())
    subprocess.run([exe, "-c", probe], cwd=OUT, env=env, check=True)
    # PyTorch only serves Demucs / speechbrain / voice cloning. Smart App Control
    # on the build PC may block its unsigned torch\_C.pyd in a fresh copy (the
    # same file loads from the source Python): that is this PC's policy, not a
    # broken copy, so it is a warning. Any other failure stops the build.
    r = subprocess.run([exe, "-c", "import torch, demucs; print('  torch', torch.__version__)"],
                       cwd=OUT, env=env, capture_output=True, text=True)
    if r.returncode == 0:
        print(r.stdout.rstrip(), flush=True)
    elif "Application Control policy has blocked" in r.stderr:
        _say("   warning: Smart App Control on this PC blocks PyTorch's unsigned torch\\_C.pyd in the copy.\n"
             "   PyTorch is kept: it works where Smart App Control is off; elsewhere only Demucs and\n"
             "   voice cloning fail.")
    else:
        sys.stderr.write(r.stderr)
        raise SystemExit("torch / demucs failed to import in the portable Python")


def make_zip(target: Path) -> None:
    tmp = target.with_suffix(".zip.part")
    files = sorted(f for f in OUT.rglob("*") if f.is_file() and f.name != MARKER)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for k, f in enumerate(files, 1):
            z.write(f, f"{OUT.name}/{f.relative_to(OUT).as_posix()}")
            if k % 5000 == 0:
                _say(f"  {k} / {len(files)} files")
    tmp.replace(target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-zip", action="store_true", help="build the folder only")
    args = ap.parse_args()
    if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise SystemExit("build_portable.py makes a 64-bit Windows copy: run it on 64-bit Windows")
    from build_exe import check_assets
    problems = check_assets()
    if problems:
        raise SystemExit("Missing assets — run  python fetch_assets.py  first: " + "; ".join(problems))
    t0 = time.time()
    version = platform.python_version()
    _say(f"1/5 Python {version} (python.org embeddable)")
    prepare_out()
    py = put_python(version)
    _say("2/5 packages")
    put_packages(py)
    _say("3/5 the app, ffmpeg, start files, RecapStudio.exe")
    put_app()
    put_launcher(py)
    _say("4/5 checking that the portable Python runs on its own")
    check_python(py)
    check_launcher()
    size = _size(OUT)
    _say(f"   folder: {OUT}  ({size / 1e9:.2f} GB)")
    if not args.no_zip:
        _say("5/5 zip")
        target = DIST / f"{OUT.name}.zip"
        make_zip(target)
        _say(f"   zip: {target}  ({target.stat().st_size / 1e9:.2f} GB)")
    _say(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
