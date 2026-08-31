from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .config import data_dir
from .logger import get_logger

log = get_logger(__name__)

def _version_current() -> str:
    try:
        from importlib.metadata import version
        return version("vocalis")
    except Exception:
        return "0.1.0"

def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        parts = v.lstrip("v").split(".")
        return tuple(int(p) for p in parts[:3] if p.isdigit() or p.split("-")[0].isdigit())
    except Exception:
        return (0,)

def check_for_update(repo: str) -> tuple[bool, str, str, str]:
    """Check GitHub releases/latest. Returns (has_update, latest_version, url, notes)."""
    if not repo or "/" not in repo:
        return (False, "", "", "No update repo configured")
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        import httpx
        with httpx.Client(timeout=10.0, headers={"Accept": "application/vnd.github+json"}) as c:
            r = c.get(api)
            if r.status_code != 200:
                return (False, "", "", f"GitHub {r.status_code}")
            data = r.json()
            latest = (data.get("tag_name") or data.get("name") or "").strip()
            url = data.get("html_url") or f"https://github.com/{repo}/releases"
            notes = data.get("body") or ""
            cur = _version_current()
            has = _version_tuple(latest) > _version_tuple(cur)
            log.info("Update check %s vs %s -> has_update=%s", cur, latest, has)
            return (has, latest, url, notes[:500])
    except Exception as e:
        log.debug("Update check failed: %s", e)
        return (False, "", "", str(e))

def spawn_update(repo: str, branch: str = "main"):
    """Spawn detached update: git pull or reinstall. Best-effort."""
    # Try to find git repo root
    try:
        root = Path(__file__).resolve().parents[2]
        if (root / ".git").exists():
            # git pull
            if sys.platform.startswith("win"):
                cmd = f'powershell -ExecutionPolicy Bypass -Command "cd \'{root}\'; git pull"'
                subprocess.Popen(cmd, shell=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore
            else:
                subprocess.Popen(["bash", "-lc", f"cd '{root}' && git pull"], start_new_session=True)
            log.info("Spawned git pull update")
            return
        # fallback: pip install -e . via install scripts
        if sys.platform.startswith("win"):
            scr = root / "install.bat"
            if scr.exists():
                subprocess.Popen(f'cmd /c \"{scr}\"', shell=True, cwd=str(root))
                return
        else:
            scr = root / "scripts" / "install.sh"
            if scr.exists():
                subprocess.Popen(["bash", str(scr)], cwd=str(root), start_new_session=True)
                return
        # fallback open url
        import webbrowser
        webbrowser.open(f"https://github.com/{repo}/releases")
    except Exception as e:
        log.exception("Spawn update failed: %s", e)
        try:
            import webbrowser
            webbrowser.open(f"https://github.com/{repo}/releases")
        except Exception:
            pass
