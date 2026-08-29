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
hidden += collect_submodules("faster_whisper")
hidden += collect_submodules("openwakeword")
hidden += collect_submodules("kokoro")
hidden += collect_submodules("sounddevice")
# openai + platformdirs are lightweight but ensure hooks
hidden += collect_submodules("openai")

datas = []
datas += collect_data_files("faster_whisper", include_py_files=False)
# kokoro ships voice configs; include if needed (harmless if missing)
try:
    datas += collect_data_files("kokoro", include_py_files=False)
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

a = Analysis(
    [str(ROOT / "src" / "vocalis" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "torch.cuda", "torch.backends.cuda", "torch.backends.cudnn",
        "torch.distributed", "triton",
        "matplotlib", "scipy",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Vocalis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    name="Vocalis",
)
