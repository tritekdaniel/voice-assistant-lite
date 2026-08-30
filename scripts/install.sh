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
if command -v apt-get >/dev/null 2>&1; then
  if ! dpkg -l | grep -q "portaudio19-dev" 2>/dev/null; then
    echo "Installing system deps: portaudio19-dev, libportaudio2, python3.11-venv (needs sudo)..."
    sudo apt-get update && sudo apt-get install -y portaudio19-dev libportaudio2 python3.11-venv || echo "apt install failed — please run manually: sudo apt install portaudio19-dev libportaudio2"
  fi
  # Also ensure scipy system deps are present (openwakeword needs it)
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
# Optional: piper voices (heavy, 34MB + build). Install only if requested or already present.
for a in "$@"; do case "$a" in --with-piper) echo "Installing piper voices (--with-piper) ..."; "$PIP" install --no-cache-dir -e "$ROOT[piper]" --upgrade || echo "piper install failed — kokoro will still work"; break;; esac; done
# If pip extra piper was already requested via existing venv, keep it; otherwise skip to save RAM

# 6. standalone binary (preferred) — skip with --no-binary
# Linux OOM guard: PyInstaller Analysis with kokoro+whisper+piper can peak >8GB and OOM-kill even on 32GB.
# Check RAM and be lean; always fall back to venv on failure (don't crash host).
BINARY="$ROOT/dist/Vocalis/Vocalis"
_should_build=1
if [ "$NO_BINARY" -ne 0 ]; then
  _should_build=0
else
  _force=0
  for a in "$@"; do case "$a" in --with-binary) _force=1;; esac; done
  if [ "$_force" -eq 0 ] && [ -f /proc/meminfo ]; then
    _avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    _avail_mb=$((_avail_kb / 1024))
    # Even 32GB can OOM with full hiddenimports (kokoro+piper+whisper). Warn if <8GB available, and skip piper in binary.
    if [ "$_avail_mb" -gt 0 ] && [ "$_avail_mb" -lt 8000 ]; then
      echo ""
      echo "Low RAM detected (${_avail_mb} MB available) — PyInstaller binary is RAM-heavy (kokoro+piper+whisper)"
      echo "and can OOM even on 32GB if hiddenimports are too broad. Skipping binary build; venv mode is leaner."
      echo "To force: bash scripts/install.sh --with-binary  or use --no-binary then build manually"
      _should_build=0
    fi
    # Also check swap
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
