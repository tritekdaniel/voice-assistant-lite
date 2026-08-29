#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then PY="python3.11"; fi
echo "Building standalone Vocalis binary (PyInstaller)..."
if [[ "${1:-}" == "--clean" ]]; then rm -rf "$ROOT/build" "$ROOT/dist"; fi
"$PY" -m pip install --upgrade pyinstaller
if ! "$PY" -m pip show torch >/dev/null 2>&1; then
  echo "Installing torch CPU first..."
  "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch --upgrade
fi
cd "$ROOT"
"$VENV/bin/pyinstaller" packaging/vocalis.spec --noconfirm --clean
echo "Done: dist/Vocalis/Vocalis"
echo "Run: dist/Vocalis/Vocalis  or  dist/Vocalis/Vocalis --check"
