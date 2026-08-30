# PyInstaller spec — standalone one-dir, no console (GUI). Preferred distribution.
# Build:  .venv/Scripts/pyinstaller packaging/vocalis.spec   -> dist/Vocalis/Vocalis(.exe)
#   or:   bash packaging/build.sh  /  powershell packaging/build.ps1
# Requires: pip install pyinstaller (install.ps1 --with-binary does it)
# CPU-only torch must be installed before building (see install scripts).

# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# PyInstaller exec()s the spec without __file__; use SPECPATH if available, else CWD
try:
    _spec_path = Path(__file__)  # type: ignore[name-defined]
except NameError:
    _spec_path = Path(SPECPATH) / "vocalis.spec" if "SPECPATH" in globals() else Path.cwd() / "packaging" / "vocalis.spec"  # type: ignore[name-defined]
ROOT = _spec_path.resolve().parent.parent
ICON_ICO = str(ROOT / "packaging" / "icon.ico")
ICON_PNG = str(ROOT / "packaging" / "icon.png")
icon = ICON_ICO if Path(ICON_ICO).exists() else (ICON_PNG if Path(ICON_PNG).exists() else None)

block_cipher = None

hidden = []
# Keep hiddenimports lean — collecting all of scipy/tensorflow OOMs Linux (was crashing).
# Only pull what we actually need; PyInstaller will discover transitive deps.
hidden += collect_submodules("faster_whisper")
hidden += collect_submodules("openwakeword")
hidden += collect_submodules("kokoro")
hidden += collect_submodules("piper")
hidden += collect_submodules("sounddevice")
# scipy: only the bits openwakeword actually uses (signal/special), not the whole 200-module tree
for _m in ("scipy.signal", "scipy.special", "scipy.linalg", "scipy.spatial"):
    try:
        hidden += collect_submodules(_m)
    except Exception:
        pass
# onnxruntime for wake word — onnx only, no tflite required
hidden += collect_submodules("onnxruntime")
# Do NOT collect tensorflow/tflite_runtime — onnx-only build per user request
# (previously collected tflite_runtime if present, now explicitly excluded)
# openai + platformdirs are lightweight but ensure hooks
hidden += collect_submodules("openai")

datas = []
datas += collect_data_files("faster_whisper", include_py_files=False)
# kokoro ships voice configs; include if needed (harmless if missing)
try:
    datas += collect_data_files("kokoro", include_py_files=False)
except Exception:
    pass
# openwakeword: bundle pretrained models + feature models (melspectrogram, embedding, VAD)
# so custom wake words work in frozen builds without extra downloads
try:
    datas += collect_data_files("openwakeword", include_py_files=False)
except Exception:
    pass
# Explicitly include openwakeword resources/models folder (collect_data_files may miss some)
try:
    import openwakeword
    ow_resources = Path(openwakeword.__file__).parent / "resources"
    if ow_resources.exists():
        datas.append((str(ow_resources), "openwakeword/resources"))
except Exception:
    pass
# bundle sound assets (wake-up.ogg, finished-listening.ogg, Lithium.mp3)
try:
    _assets_src = str(ROOT / "assets")
    if Path(_assets_src).exists():
        datas.append((_assets_src, "assets"))
    # also handle frozen _MEIPASS case: sounds.py looks up assets/ relative to exe
    # so we also add as datas for PyInstaller
except Exception:
    pass
# soundfile/miniaudio data (if any)
try:
    datas += collect_data_files("soundfile", include_py_files=False)
except Exception:
    pass
# piper espeak-ng data and voices (needed for Piper TTS)
try:
    datas += collect_data_files("piper", include_py_files=False)
except Exception:
    pass

# Linux OOM guard: on low-RAM machines PyInstaller can get killed. Keep excludes aggressive.
_excludes = [
    "torch.cuda", "torch.backends.cuda", "torch.backends.cudnn",
    "torch.distributed", "triton",
    "matplotlib", "matplotlib.tests", "mpl_toolkits",
    "numpy.tests", "numpy.distutils", "scipy.tests",
    "PIL", "cv2", "tkinter", "IPython", "jupyter",
    "tensorflow", "tensorflow.python",  # not needed — we use onnx
    "tflite_runtime", "tflite",  # onnx-only: explicitly exclude tflite
    # DO NOT exclude scipy.signal/special — openwakeword/custom_verifier_model.py requires it
]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# UPX on Linux is RAM-hungry; disable it there to avoid OOM crashes.
import sys as _sys
_is_linux = _sys.platform.startswith("linux")
_use_upx = not _is_linux  # Windows keeps UPX, Linux skips it

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Vocalis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True if _is_linux else False,
    upx=_use_upx,
    console=False,
    icon=icon,
    # Windows: no console, still handles --check/--headless via console=False?
    # For CLI modes we rely on `Vocalis --check` still working headless (PyInstaller console=False still allows stdout when launched from terminal).
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True if _is_linux else False,
    upx=_use_upx,
    name="Vocalis",
)
