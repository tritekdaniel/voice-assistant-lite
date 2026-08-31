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
# App modules — ensure new features are bundled even if dynamically imported
hidden += [
    "vocalis.watchdog", "vocalis.alarms", "vocalis.calibration", "vocalis.updater",
    "vocalis.sounds", "vocalis.timer", "vocalis.session", "vocalis.audio_io",
    "vocalis.stt", "vocalis.tts", "vocalis.llm", "vocalis.wakeword", "vocalis.bootstrap",
    "vocalis.config", "vocalis.logger", "vocalis.textsplit", "vocalis.runner",
]
# http stack for llm.py/updater.py (dynamically imported inside functions, missed by static analysis)
hidden += ["httpx", "httpcore", "anyio", "h11", "idna", "sniffio"]
# Keep hiddenimports lean — collecting all of torch/scipy/faster_whisper OOMs Linux even on 32GB.
# Let PyInstaller discover via entry.py; only add what it misses. On Linux, be extra lean.
import sys as _sys_hidden
_is_linux_hidden = _sys_hidden.platform.startswith("linux")
if _is_linux_hidden:
    # Linux: minimal hiddenimports; PyInstaller will walk entry.py imports. Avoid collecting
    # heavy trees (faster_whisper/openwakeword/kokoro) which peak 6-8GB and OOM-kill.
    hidden += ["faster_whisper", "openwakeword", "kokoro", "sounddevice"]
else:
    try:
        hidden += collect_submodules("faster_whisper", filter=lambda name: "test" not in name and "model" not in name)
    except Exception:
        hidden += ["faster_whisper"]
    try:
        hidden += collect_submodules("openwakeword", filter=lambda name: "test" not in name)
    except Exception:
        hidden += ["openwakeword"]
    # kokoro/piper/sounddevice: explicit, not full tree (kokoro pulls torch; piper pulls espeak)
    hidden += ["kokoro", "kokoro.pipeline", "kokoro.model"]
    hidden += ["sounddevice"]
# piper is optional (pip install -e .[piper]); keep binary lean — never include piper on Linux
# (Linux piper install often needs --only-binary and can OOM during Analysis; venv mode handles piper fine)
_is_linux_spec = __import__("sys").platform.startswith("linux")
try:
    import importlib.util
    if not _is_linux_spec and importlib.util.find_spec("piper") is not None:
        hidden += ["piper", "piper.voice", "piper.config", "piper.phonemize_espeak"]
        try:
            hidden += collect_submodules("piper", filter=lambda n: n in ("piper.voice", "piper.config"))
        except Exception:
            pass
    elif _is_linux_spec and importlib.util.find_spec("piper") is not None:
        # On Linux, piper stays venv-only; binary uses kokoro to avoid 32GB OOM
        pass
except Exception:
    pass
# scipy: only the bits openwakeword actually uses (signal/special), not the whole 200-module tree
if _is_linux_hidden:
    hidden += ["scipy.signal", "scipy.special"]
else:
    for _m in ("scipy.signal", "scipy.special", "scipy.linalg", "scipy.spatial"):
        try:
            hidden += collect_submodules(_m)
        except Exception:
            pass
# onnxruntime for wake word — onnx only, no tflite required
if _is_linux_hidden:
    hidden += ["onnxruntime"]
else:
    hidden += collect_submodules("onnxruntime", filter=lambda n: "test" not in n and "tools" not in n)
# openai is lightweight
hidden += ["openai", "openai.resources", "openai.types"]

datas = []
if _is_linux_hidden:
    # Linux lean: models download at runtime (~700MB), don't bundle to keep binary small and avoid OOM
    # Only bundle what is needed for offline wake word if already present
    pass
else:
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
# piper espeak-ng data and voices (needed for Piper TTS) — never in Linux binary (venv-only)
try:
    import importlib.util
    _is_linux_data = __import__("sys").platform.startswith("linux")
    if not _is_linux_data and importlib.util.find_spec("piper") is not None:
        datas += collect_data_files("piper", include_py_files=False)
except Exception:
    pass

# Linux OOM guard: on low-RAM machines PyInstaller can get killed. Keep excludes aggressive.
_excludes = [
    "torch.cuda", "torch.backends.cuda", "torch.backends.cudnn",
    "torch.distributed", "triton", "triton.language",
    "matplotlib", "matplotlib.tests", "mpl_toolkits",
    "numpy.tests", "numpy.distutils", "scipy.tests",
    "PIL", "cv2", "tkinter", "IPython", "jupyter", "pytest",
    "tensorflow", "tensorflow.python",  # not needed — we use onnx
    "tflite_runtime", "tflite",  # onnx-only: explicitly exclude tflite
    "xformers", "transformers.tests", "ctranslate2.tests",
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
    strip=False,  # never strip on Linux (can crash display/coredump); Windows ok but keep False for safety
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
    strip=False,
    upx=_use_upx,
    name="Vocalis",
)
