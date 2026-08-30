#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then PY="python3.11"; fi
echo "Building standalone Vocalis binary (PyInstaller)..."
if [[ "${1:-}" == "--clean" ]]; then rm -rf "$ROOT/build" "$ROOT/dist"; fi
# Linux desktop-kick guard: PyInstaller even lean can take the session down (strip, fork, pulse).
# Default to venv-only on Linux; require --force to build.
if grep -q "Linux" /proc/version 2>/dev/null; then
  if [[ "${1:-}" != "--force" && "${2:-}" != "--force" ]]; then
    echo "Linux: not building PyInstaller binary by default (previous builds kicked desktop to login on 32GB)."
    echo "Use venv: .venv/bin/vocalis  — or force: bash packaging/build.sh --force"
    exit 0
  fi
fi
# OOM guard — lean spec can still peak 8GB with kokoro+whisper
_avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
_avail_mb=$((_avail_kb / 1024))
if [ "$_avail_mb" -gt 0 ] && [ "$_avail_mb" -lt 8000 ]; then
  echo "Low RAM: ${_avail_mb} MB available — build may OOM (lean spec needs ~6-8GB). Add swap or use venv: .venv/bin/vocalis"
  if [[ "${2:-}" != "--force" && "${1:-}" != "--force" ]]; then
    echo "  To force: bash packaging/build.sh --force  (or --no-binary via install.sh)"
    echo "  Continuing anyway in 3s… (Ctrl+C to cancel)"
    sleep 3
  fi
fi
_swap_kb=$(awk '/SwapTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$_swap_kb" -eq 0 ] && [ "$_avail_mb" -gt 0 ] && [ "$_avail_mb" -lt 16000 ]; then
  echo "No swap — OOM-killer may freeze host. Consider: sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
fi
"$PY" -m pip install --no-cache-dir --upgrade pyinstaller
if ! "$PY" -m pip show torch >/dev/null 2>&1; then
  echo "Installing torch CPU first..."
  "$VENV/bin/pip" install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch --upgrade
fi
cd "$ROOT"
export PYINSTALLER_COMPILE_BOOTLOADER=0
export PYTHONHASHSEED=0
# timeout + nice to avoid host crash; lean spec excludes piper by default
if ! timeout 900 "$VENV/bin/pyinstaller" packaging/vocalis.spec --noconfirm --clean --log-level=WARN; then
  echo "Build failed/OOM — fallback to venv mode (.venv/bin/vocalis). Not a bug, just RAM."
  echo "For Piper in venv: .venv/bin/pip install -e .[piper]"
  exit 1
fi
echo "Done: dist/Vocalis/Vocalis"
echo "Run: dist/Vocalis/Vocalis  or  dist/Vocalis/Vocalis --check"
echo "If the machine crashed, use venv mode: .venv/bin/vocalis --check"
