#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLEAN=0
KEEP_MODELS=0
for a in "$@"; do
  case "$a" in --clean) CLEAN=1;; --keep-models) KEEP_MODELS=1;; esac
done

echo "Vocalis uninstaller — Linux/macOS"
echo "Project: $ROOT"

# 1. pip uninstall if venv exists
if [ -x "$ROOT/.venv/bin/pip" ]; then
  echo "Running pip uninstall vocalis ..."
  "$ROOT/.venv/bin/pip" uninstall -y vocalis 2>/dev/null || true
fi

# 2. venv / build artefacts
for p in .venv build dist __pycache__ "src/vocalis/__pycache__" .pytest_cache; do
  if [ -e "$ROOT/$p" ]; then echo "Removing $p ..."; rm -rf "$ROOT/$p"; fi
done
rm -rf "$ROOT"/*.egg-info 2>/dev/null || true

# 3. desktop entry
if [ -f "$HOME/.local/share/applications/vocalis.desktop" ]; then
  echo "Removing desktop entry ..."
  rm -f "$HOME/.local/share/applications/vocalis.desktop"
fi

# 4. config / data
if [ "$CLEAN" -eq 1 ]; then
  # platformdirs: ~/.config/vocalis and ~/.local/share/vocalis (XDG)
  CFG="${XDG_CONFIG_HOME:-$HOME/.config}/vocalis"
  DATA="${XDG_DATA_HOME:-$HOME/.local/share}/vocalis"
  if [ -d "$CFG" ]; then echo "Removing config at $CFG ..."; rm -rf "$CFG"; else echo "No config at $CFG"; fi
  if [ "$KEEP_MODELS" -eq 1 ]; then
    echo "--keep-models: leaving $DATA (keeps ~700 MB models and offline alarms.json)"
  else
    if [ -d "$DATA" ]; then echo "Removing data/models/alarms at $DATA (includes alarms.json) ..."; rm -rf "$DATA"; else echo "No data at $DATA"; fi
  fi
else
  echo "Keeping user config/models. Re-run with --clean to remove them."
  echo "  bash scripts/uninstall.sh --clean               # remove everything"
  echo "  bash scripts/uninstall.sh --clean --keep-models # keep models"
fi

echo ""
echo "Uninstall done. To reinstall: bash scripts/install.sh"
