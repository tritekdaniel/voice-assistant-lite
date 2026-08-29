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

# 2. System deps hint
if command -v apt-get >/dev/null 2>&1; then
  echo "If pip install fails on audio, run: sudo apt install portaudio19-dev python3.11-venv"
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
"$PYV" -m pip install --upgrade pip

# 4. torch CPU first
echo "Installing torch (CPU-only) ..."
"$PIP" install --index-url https://download.pytorch.org/whl/cpu torch --upgrade

# 5. vocalis
echo "Installing vocalis (pip install -e .) ..."
"$PIP" install -e "$ROOT" --upgrade

# 6. standalone binary (preferred) — skip with --no-binary
BINARY="$ROOT/dist/Vocalis/Vocalis"
if [ "$NO_BINARY" -eq 0 ]; then
  echo ""
  echo "Building standalone binary (PyInstaller, ~1.5 GB, a few minutes) ..."
  "$PIP" install --upgrade pyinstaller >/dev/null 2>&1 || true
  if "$VENV/bin/pyinstaller" packaging/vocalis.spec --noconfirm --clean; then
    echo "Binary built: $BINARY"
  else
    echo "Binary build failed — falling back to venv script"
    BINARY=""
  fi
else
  echo "Skipping binary build (--no-binary)"
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
