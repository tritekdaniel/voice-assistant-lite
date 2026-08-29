#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then PY="python3.11"; fi
echo "Building standalone Vocalis binary (PyInstaller)..."
if [[ "${1:-}" == "--clean" ]]; then rm -rf "$ROOT/build" "$ROOT/dist"; fi
# OOM guard — same as install.sh
_avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
_avail_mb=$((_avail_kb / 1024))
if [ "$_avail_mb" -gt 0 ] && [ "$_avail_mb" -lt 3500 ]; then
  echo "Low RAM: ${_avail_mb} MB available — build may OOM. Consider --no-binary or add swap."
  echo "  Fallback: .venv/bin/vocalis (same features, no binary needed)"
  if [[ "${2:-}" != "--force" && "${1:-}" != "--force" ]]; then
    echo "  To force: bash packaging/build.sh --force"
    echo "  Continuing anyway in 3s… (Ctrl+C to cancel)"
    sleep 3
  fi
fi
"$PY" -m pip install --upgrade pyinstaller
if ! "$PY" -m pip show torch >/dev/null 2>&1; then
  echo "Installing torch CPU first..."
  "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch --upgrade
fi
cd "$ROOT"
export PYINSTALLER_COMPILE_BOOTLOADER=0
"$VENV/bin/pyinstaller" packaging/vocalis.spec --noconfirm --clean --log-level=WARN
echo "Done: dist/Vocalis/Vocalis"
echo "Run: dist/Vocalis/Vocalis  or  dist/Vocalis/Vocalis --check"
echo "If the machine crashed, use venv mode: .venv/bin/vocalis --check"
