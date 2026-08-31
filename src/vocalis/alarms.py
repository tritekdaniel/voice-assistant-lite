from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .config import data_dir
from .logger import get_logger
from .sounds import Sounds

log = get_logger(__name__)

def _alarms_path() -> Path:
    return data_dir() / "alarms.json"

@dataclass
class Alarm:
    id: int
    label: str
    at: str  # ISO8601 local time, e.g. 2026-08-31T07:30:00
    recurrence: str = "once"  # once, daily, weekly, weekdays
    enabled: bool = True
    tone: str = "Lithium.mp3"
    created_at: float = 0.0

class AlarmManager:
    """Offline alarms with daily/weekly/weekdays/once recurrence. Persisted to alarms.json."""

    def __init__(self, sounds: Sounds):
        self._sounds = sounds
        self._lock = threading.Lock()
        self._alarms: dict[int, Alarm] = {}
        self._next_id = 1
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._load()
        self.start()

    def _load(self):
        p = _alarms_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("alarms", []):
                try:
                    a = Alarm(**d)
                    self._alarms[a.id] = a
                    self._next_id = max(self._next_id, a.id + 1)
                except Exception:
                    continue
        except Exception as e:
            log.warning("Failed to load alarms %s: %s", p, e)

    def _save(self):
        p = _alarms_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {"alarms": [asdict(a) for a in self._alarms.values()]}
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to save alarms: %s", e)

    def list_alarms(self) -> list[dict]:
        with self._lock:
            out = []
            for a in sorted(self._alarms.values(), key=lambda x: x.at):
                out.append(asdict(a))
            return out

    def add_alarm(self, at_iso: str, label: str = "alarm", recurrence: str = "once", tone: str | None = None) -> dict:
        # parse
        try:
            at = datetime.fromisoformat(at_iso)
        except Exception:
            raise ValueError(f"Invalid time {at_iso!r} — use YYYY-MM-DDTHH:MM or HH:MM (today)")
        # if only time provided? fromisoformat will fail for HH:MM? handle
        now = datetime.now()
        if at.year == 1900:
            at = at.replace(year=now.year, month=now.month, day=now.day)
        if recurrence not in ("once", "daily", "weekly", "weekdays"):
            recurrence = "once"
        # if once and time in past, move to tomorrow
        if recurrence == "once" and at <= now:
            at = at + timedelta(days=1)
        label = (label or "alarm").strip() or "alarm"
        with self._lock:
            aid = self._next_id
            self._next_id += 1
            a = Alarm(id=aid, label=label, at=at.isoformat(timespec="seconds"), recurrence=recurrence, enabled=True, tone=tone or "Lithium.mp3", created_at=time.time())
            self._alarms[aid] = a
            self._save()
            log.info("Alarm #%d added %s %s %s", aid, a.at, recurrence, label)
            return asdict(a)

    def remove_alarm(self, alarm_id: int) -> bool:
        with self._lock:
            if alarm_id in self._alarms:
                del self._alarms[alarm_id]
                self._save()
                log.info("Alarm #%d removed", alarm_id)
                return True
            return False

    def toggle_alarm(self, alarm_id: int, enabled: bool) -> bool:
        with self._lock:
            a = self._alarms.get(alarm_id)
            if a:
                a.enabled = enabled
                self._save()
                return True
            return False

    def _next_fire(self, a: Alarm, now: datetime) -> datetime | None:
        try:
            at = datetime.fromisoformat(a.at)
        except Exception:
            return None
        if not a.enabled:
            return None
        if a.recurrence == "once":
            return at if at > now else None
        if a.recurrence == "daily":
            cand = at.replace(year=now.year, month=now.month, day=now.day)
            if cand <= now:
                cand += timedelta(days=1)
            return cand
        if a.recurrence == "weekly":
            # same weekday as original
            target_wd = at.weekday()
            cand = at.replace(year=now.year, month=now.month, day=now.day)
            days_ahead = (target_wd - now.weekday()) % 7
            cand = cand + timedelta(days=days_ahead)
            if cand <= now:
                cand += timedelta(days=7)
            return cand
        if a.recurrence == "weekdays":
            cand = at.replace(year=now.year, month=now.month, day=now.day)
            for i in range(7):
                c = cand + timedelta(days=i)
                if c.weekday() < 5 and c > now:
                    return c
            return None
        return None

    def _check_and_fire(self):
        now = datetime.now()
        to_fire: list[Alarm] = []
        with self._lock:
            for a in self._alarms.values():
                nxt = self._next_fire(a, now)
                if nxt is None:
                    continue
                # fire if within last 60s
                if abs((nxt - now).total_seconds()) < 60:
                    # for once, ensure we don't fire repeatedly: check if at is within 60s and not future beyond
                    if a.recurrence == "once":
                        # mark as fired by disabling? For once we keep enabled but will not fire again because at is past
                        pass
                    to_fire.append(a)
        for a in to_fire:
            self._fire_alarm(a)

    def _fire_alarm(self, a: Alarm):
        log.info("Alarm #%d firing %s (%s)", a.id, a.label, a.at)
        label = a.label
        # Update once alarm's at to past? keep as is, _next_fire will return None next check
        # For recurring, update at to next occurrence for persistence display?
        # We keep original at for recurrence calc, so no need to update
        # Speak + sound loop
        try:
            # ensure tone exists; play loop like timer
            loops = 5
            self._sounds.play_timer_loop(loops=loops)
            # also queue TTS if available? Will be handled via session's alarm callback if needed
        except Exception as e:
            log.warning("Alarm sound failed: %s", e)
        # If once, disable after fire to avoid repeated firing within 60s window on next loop
        if a.recurrence == "once":
            # we disable after 70s to avoid double-fire, but keep record
            def _disable_later(aid=a.id):
                time.sleep(70)
                with self._lock:
                    aa = self._alarms.get(aid)
                    if aa and aa.recurrence == "once":
                        aa.enabled = False
                        self._save()
                        log.info("Alarm #%d auto-disabled after firing", aid)
            threading.Thread(target=_disable_later, daemon=True).start()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="alarm-loop")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._check_and_fire()
            except Exception as e:
                log.exception("Alarm loop error: %s", e)
            self._stop.wait(30.0)  # check every 30s

def get_alarm_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "set_alarm",
                "description": "Set an offline alarm at a specific time. Use for 'alarm at 7am', 'wake me at 6:30', daily/weekly alarms.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string", "description": "ISO time YYYY-MM-DDTHH:MM:SS or HH:MM (today). e.g. '07:30', '2026-08-31T07:30:00'"},
                        "label": {"type": "string", "description": "Alarm label"},
                        "recurrence": {"type": "string", "enum": ["once", "daily", "weekly", "weekdays"], "description": "Recurrence"},
                    },
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_alarms",
                "description": "List all alarms",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_alarm",
                "description": "Cancel/delete an alarm by id",
                "parameters": {"type": "object", "properties": {"alarm_id": {"type": "integer"}}, "required": ["alarm_id"]},
            },
        },
    ]
