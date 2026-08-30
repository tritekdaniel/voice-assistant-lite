#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
NO_CHECK=0
NO_BINARY=0
for a in "$@"; do case "$a" in --no-check) NO_CHECK=1;; --no-binary) NO_BINARY=1;; --with-binary) NO_BINARY=0;; esac; done

echo "Vocalis installer — Linux/macOS"
echo "Project: $ROOT"

# 1. Find python3.11
PY=""
for c in python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    ver="$("$c" --version 2>&1 || true)"
    if echo "$ver" | grep -q "3\.11\."; then PY="$c"; echo "Using $c ($ver)"; break; fi
    echo "Skipping $c ($ver) — need 3.11 for CPU torch"
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11 not found. Install it (e.g. sudo apt install python3.11 python3.11-venv) then re-run."
  exit 1
fi

# 2. System deps — auto-install PortAudio/scipy prereqs on Debian/Ubuntu
# Never auto-run apt on Linux without --with-apt (it can restart audio/display and kick the user to login)
_with_apt=0
for a in "$@"; do case "$a" in --with-apt) _with_apt=1;; esac; done
if [ "$_with_apt" -eq 1 ] && command -v apt-get >/dev/null 2>&1; then
  if ! dpkg -l | grep -q "portaudio19-dev" 2>/dev/null; then
    echo "Installing system deps: portaudio19-dev, libportaudio2, python3.11-venv (needs sudo)..."
    sudo apt-get update && sudo apt-get install -y portaudio19-dev libportaudio2 python3.11-venv || echo "apt install failed — please run manually: sudo apt install portaudio19-dev libportaudio2"
  fi
elif command -v apt-get >/dev/null 2>&1; then
  if ! dpkg -l | grep -q "portaudio19-dev" 2>/dev/null; then
    echo "Note: portaudio19-dev not found — needed for mic. Install with: sudo apt install portaudio19-dev libportaudio2  or  bash scripts/install.sh --with-apt"
  fi
  if ! python3 -c "import scipy" 2>/dev/null; then
    echo "Note: scipy will be pip-installed with vocalis (openwakeword needs it)"
  fi
fi

# 3. venv
if [ ! -d "$VENV" ]; then
  echo "Creating venv at $VENV ..."
  "$PY" -m venv "$VENV"
else
  echo "venv already exists at $VENV — reusing (rm -rf .venv to force fresh)"
fi
PIP="$VENV/bin/pip"
PYV="$VENV/bin/python"
# Use --no-cache-dir and single process to keep RAM low on 32GB (pip cache can spike 2-3GB with torch)
"$PYV" -m pip install --upgrade pip --no-cache-dir || "$PYV" -m pip install --upgrade pip

# 4. torch CPU first (lean, no cache, handle OOM)
echo "Installing torch (CPU-only) ..."
if ! "$PIP" install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch --upgrade; then
  echo "torch install failed (possible OOM or network). Retrying without cache and with --no-deps fallback..."
  "$PIP" install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch --no-deps --upgrade || {
    echo "ERROR: torch CPU install failed. See logs. Continuing to try vocalis install (may fail)..."
  }
fi

# 5. vocalis (base, no piper by default to keep RAM lean)
echo "Installing vocalis (pip install -e .) ..."
if ! "$PIP" install --no-cache-dir -e "$ROOT" --upgrade; then
  echo "pip install -e . failed — retrying with --no-build-isolation (saves RAM)..."
  "$PIP" install --no-cache-dir --no-build-isolation -e "$ROOT" --upgrade || {
    echo "ERROR: vocalis install failed. Check $ROOT/pip.log or run with --no-binary"
    exit 1
  }
fi
# Piper voices (34MB, optional) — try to install by default, but never build from source on Linux (kicks).
# On Linux, piper-tts often has no wheel for this Python/manylinux -> source build OOMs and kicks to login, so use binary-only.
# This block never crashes the host (set -e is bypassed).
_should_try_piper=1
for a in "$@"; do case "$a" in --no-piper) _should_try_piper=0; break;; --with-piper) _should_try_piper=1; break;; esac; done
# Also respect --no-piper to skip entirely
if [ "$_should_try_piper" -eq 1 ]; then
  if ! "$PYV" -c "import piper" 2>/dev/null; then
    echo "Installing piper-tts (for GLaDOS/custom voices) ..."
    if grep -q "Linux" /proc/version 2>/dev/null; then
      echo "Linux: piper install binary-only (source build kicks to login)..."
      if "$PIP" install --no-cache-dir --only-binary=:all: -e "$ROOT[piper]" --upgrade 2>&1 | tail -30; then
        echo "piper installed (binary)"
      else
        echo "piper binary not available for this Linux/Python — piper will stay unavailable (kokoro will still work)."
        echo "To try source build (may kick): .venv/bin/pip install --no-cache-dir piper-tts  or  bash scripts/install.sh --with-piper"
      fi
    else
      "$PIP" install --no-cache-dir -e "$ROOT[piper]" --upgrade 2>&1 | tail -20 || echo "piper install failed — kokoro will still work"
    fi
  else
    echo "piper already installed — skipping"
  fi
else
  echo "Skipping piper (--no-piper)"
fi
# Explicit --with-piper forces install (even from source on Linux, may kick to login — user asked for it)
for a in "$@"; do case "$a" in --with-piper)
  if "$PYV" -c "import piper" 2>/dev/null; then echo "piper already installed"; break; fi
  echo "Installing piper voices (--with-piper, may build from source on Linux) ..."
  if grep -q "Linux" /proc/version 2>/dev/null; then
    if ! "$PIP" install --no-cache-dir --only-binary=:all: -e "$ROOT[piper]" --upgrade 2>&1 | tail -30; then
      echo "Binary not available — trying source build with nice (may be heavy, not OOM)..."
      if ! nice -n 19 "$PIP" install --no-cache-dir --no-build-isolation -e "$ROOT[piper]" --upgrade 2>&1 | tail -30; then
        echo "piper source build failed — kokoro will still work. Try: pip install piper-tts --no-cache-dir"
      fi
    fi
  else
    "$PIP" install --no-cache-dir -e "$ROOT[piper]" --upgrade 2>&1 | tail -20 || echo "piper install failed"
  fi
  break;;
esac; done

# 6. standalone binary — on Linux, never build by default (it previously kicked the user to login).
# PyInstaller with kokoro+whisper can still take the desktop down even on 32GB (strip/UPX/fork).
# Use venv mode by default on Linux; build only if --with-binary is explicitly passed.
BINARY="$ROOT/dist/Vocalis/Vocalis"
_should_build=1
# On Linux, default to venv-only to avoid desktop kick (user reported back-to-login on 32GB)
_is_linux=0
if grep -q "Linux" /proc/version 2>/dev/null; then _is_linux=1; fi
if [ "$_is_linux" -eq 1 ] && [ "$NO_BINARY" -eq 0 ]; then
  _has_with_binary=0
  for a in "$@"; do case "$a" in --with-binary) _has_with_binary=1;; esac; done
  if [ "$_has_with_binary" -eq 0 ]; then
    echo ""
    echo "Linux: skipping PyInstaller binary by default (previous builds kicked desktop to login on 32GB)."
    echo "Using venv mode (same features, no build, no kick). To build binary: bash scripts/install.sh --with-binary"
    _should_build=0
  fi
fi
if [ "$NO_BINARY" -ne 0 ]; then
  _should_build=0
else
  _force=0
  for a in "$@"; do case "$a" in --with-binary) _force=1;; esac; done
  if [ "$_force" -eq 0 ] && [ -f /proc/meminfo ]; then
    _avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    _avail_mb=$((_avail_kb / 1024))
    if [ "$_avail_mb" -gt 0 ] && [ "$_avail_mb" -lt 8000 ]; then
      echo ""
      echo "Low RAM detected (${_avail_mb} MB available) — PyInstaller binary is RAM-heavy (kokoro+piper+whisper)"
      echo "and can OOM even on 32GB if hiddenimports are too broad. Skipping binary build; venv mode is leaner."
      echo "To force: bash scripts/install.sh --with-binary  or use --no-binary then build manually"
      _should_build=0
    fi
    _swap_kb=$(awk '/SwapTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    if [ "$_swap_kb" -eq 0 ] && [ "$_avail_mb" -lt 16000 ]; then
      echo "No swap and ${_avail_mb}MB RAM — binary build may still OOM-kill. Will try but fallback to venv on failure."
    fi
  fi
fi
if [ "$_should_build" -eq 1 ]; then
  echo ""
  echo "Building standalone binary (PyInstaller, ~1.5 GB binary, a few minutes) ..."
  echo "If this crashes/hangs/OOMs, re-run with --no-binary (venv mode is same features, no build needed)."
  echo "Lean spec: piper excluded from binary by default (use venv for Piper voices: .venv/bin/pip install -e .[piper])"
  export PYINSTALLER_COMPILE_BOOTLOADER=0
  # Keep pip/pyinstaller low-memory
  export PYTHONHASHSEED=0
  "$PIP" install --no-cache-dir --upgrade pyinstaller >/dev/null 2>&1 || true
  # Limit Python GC and use single process; timeout after 15 min to avoid hanging host
  if timeout 900 "$VENV/bin/pyinstaller" packaging/vocalis.spec --noconfirm --clean --log-level=WARN; then
    echo "Binary built: $BINARY"
  else
    rc=$?
    echo "Binary build failed (exit $rc) — falling back to venv script (this is expected on low-RAM or OOM)."
    echo "Tip: use --no-binary for venv-only install: bash scripts/install.sh --no-binary"
    echo "For Piper voices in venv: .venv/bin/pip install -e .[piper]"
    BINARY=""
  fi
else
  if [ "$NO_BINARY" -ne 0 ]; then echo "Skipping binary build (--no-binary)"; fi
  BINARY=""
fi

# 7. check — prefers binary if built
if [ "$NO_CHECK" -eq 0 ]; then
  echo ""
  echo "Running vocalis --check (first run downloads ~700 MB) ..."
  TO_CHECK=""
  if [ -n "$BINARY" ] && [ -x "$BINARY" ]; then TO_CHECK="$BINARY"
  elif [ -x "$VENV/bin/vocalis" ]; then TO_CHECK="$VENV/bin/vocalis"
  fi
  if [ -n "$TO_CHECK" ]; then "$TO_CHECK" --check || echo "check reported failures — GUI will still run (models lazy-load)"
  else "$PYV" -m vocalis --check || true; fi
fi

# 8. desktop entry (Linux) — prefers binary
ICON="$ROOT/packaging/icon.png"
if [ -d "$HOME/.local/share/applications" ]; then
  mkdir -p "$HOME/.local/share/applications"
  EXEC="$VENV/bin/vocalis"
  if [ -n "$BINARY" ] && [ -x "$BINARY" ]; then EXEC="$BINARY"; fi
  cat > "$HOME/.local/share/applications/vocalis.desktop" <<EOF
[Desktop Entry]
Name=Vocalis
Comment=Voice assistant for local LLMs
Exec=$EXEC
Icon=$ICON
Terminal=false
Type=Application
Categories=AudioVideo;Utility;
Keywords=voice;assistant;llm;whisper;tts;
EOF
  echo "Desktop entry: ~/.local/share/applications/vocalis.desktop -> $EXEC"
fi

echo ""
echo "Done. Launch with:"
if [ -n "$BINARY" ] && [ -x "$BINARY" ]; then echo "  $BINARY        # standalone binary (preferred, double-click)"; fi
echo "  .venv/bin/vocalis         # GUI (venv)"
echo "  .venv/bin/vocalis --check"
echo "  .venv/bin/vocalis --headless"
echo "Uninstall: Settings → Danger Zone  or  bash scripts/uninstall.sh  (add --clean)"
echo "Tip: Settings → Danger Zone has one-click Uninstall buttons."
