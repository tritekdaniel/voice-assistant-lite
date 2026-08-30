from __future__ import annotations

import re
import threading
import time

from .logger import get_logger
from .sounds import Sounds

log = get_logger(__name__)

# Regex for direct voice command fallback (if LLM doesn't call tool)
_TIMER_RE = re.compile(
    r"\b(?:set|start|create)\s+(?:a\s+)?timer\s+(?:for\s+)?(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?\b",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(r"\b(?:stop|cancel|clear|end)\s+(?:the\s+)?timer\b|\bstop\s+timer\b", re.IGNORECASE)


class _TimerEntry:
    def __init__(self, tid: int, label: str, duration: int, start: float, timer: threading.Timer):
        self.tid = tid
        self.label = label
        self.duration = duration
        self.start = start
        self.timer = timer


class TimerManager:
    """Manages multiple concurrent countdown timers. Each firing plays the alarm loop; setting plays timer-set."""

    def __init__(self, sounds: Sounds):
        self._sounds = sounds
        self._lock = threading.Lock()
        self._timers: dict[int, _TimerEntry] = {}
        self._next_id: int = 1
        # backwards compat single-timer aliases
        self._active_label: str | None = None

    @property
    def is_active(self) -> bool:
        with self._lock:
            return any(t.timer.is_alive() for t in self._timers.values())

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._timers.values() if t.timer.is_alive())

    def remaining_seconds(self) -> int | None:
        """Return remaining seconds for the soonest timer (backwards compat)."""
        with self._lock:
            if not self._timers:
                return None
            soonest: int | None = None
            now = time.monotonic()
            for e in self._timers.values():
                if not e.timer.is_alive():
                    continue
                rem = int(e.duration - (now - e.start))
                rem = max(0, rem)
                if soonest is None or rem < soonest:
                    soonest = rem
            return soonest

    def list_timers(self) -> list[dict]:
        """Return list of active timers for UI: [{id, label, remaining}]."""
        with self._lock:
            now = time.monotonic()
            out = []
            for tid, e in sorted(self._timers.items()):
                if not e.timer.is_alive():
                    continue
                rem = max(0, int(e.duration - (now - e.start)))
                out.append({"id": tid, "label": e.label, "seconds": e.duration, "remaining": rem})
            return out

    def set_timer(self, seconds: int, label: str = "timer") -> str:
        if seconds <= 0:
            return "Timer duration must be positive."
        if seconds > 24 * 3600:
            return "Timer too long (max 24h)."
        label = (label or "timer").strip() or "timer"
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            self._active_label = label

        log.info("Timer #%d set for %ds (%s) — %d active now", tid, seconds, label, self.active_count() + 1)

        def _fire(tid_inner=tid, secs=seconds, lbl=label):
            log.info("Timer #%d fired after %ds (%s)", tid_inner, secs, lbl)
            with self._lock:
                self._timers.pop(tid_inner, None)
                if not self._timers:
                    self._active_label = None
            # Play alarm 5 times — cancellable via stop_timer_loop (shared event, all alarms stop together)
            self._sounds.play_timer_loop(loops=5)

        t = threading.Timer(seconds, _fire)
        t.daemon = True
        entry = _TimerEntry(tid, label, seconds, time.monotonic(), t)
        with self._lock:
            self._timers[tid] = entry
        t.start()
        # Play timer-set confirmation sound immediately (assets/timer-set.mp3)
        try:
            self._sounds.play_timer_set()
        except Exception as e:
            log.debug("timer-set sound failed: %s", e)
        count = self.active_count()
        if count == 1:
            return f"Timer set for {seconds} seconds. A sound will play when it rings. Say 'stop timer' to cancel."
        else:
            return f"Timer #{tid} set for {seconds} seconds ({label}). {count} timers active. Say 'stop timer' to cancel all."

    def cancel(self, label: str | None = None, timer_id: int | None = None) -> str:
        """Cancel timers. No args -> cancel all. With label/id -> cancel matching."""
        to_cancel: list[_TimerEntry] = []
        with self._lock:
            if timer_id is not None:
                e = self._timers.get(timer_id)
                if e:
                    to_cancel = [e]
            elif label:
                # cancel by label substring (case-insensitive)
                low = label.lower()
                to_cancel = [e for e in self._timers.values() if low in e.label.lower() or e.label.lower() in low]
                if not to_cancel:
                    # try exact id string
                    try:
                        tid = int(label)
                        e = self._timers.get(tid)
                        if e:
                            to_cancel = [e]
                    except Exception:
                        pass
            else:
                # cancel all
                to_cancel = list(self._timers.values())
                self._timers.clear()
                self._active_label = None
            # remove selected from dict
            for e in to_cancel:
                self._timers.pop(e.tid, None)
            if not self._timers:
                self._active_label = None

        stopped = False
        for e in to_cancel:
            try:
                e.timer.cancel()
                stopped = True
            except Exception:
                pass
        # also stop any currently playing alarm loop
        try:
            self._sounds.stop_timer_loop()
            stopped = True
        except Exception:
            pass
        if to_cancel:
            log.info("Timer cancelled: %s", ", ".join(f"#{e.tid} {e.label} {e.duration}s" for e in to_cancel))
            if len(to_cancel) == 1:
                return f"Timer #{to_cancel[0].tid} ({to_cancel[0].label}) cancelled."
            return f"Cancelled {len(to_cancel)} timers."
        if stopped:
            # stopped the alarm loop even though no timer entry existed
            return "Timer cancelled."
        return "No active timer."

    def cancel_all(self) -> str:
        return self.cancel()

    # --- voice command helpers ---

    def parse_voice_set(self, text: str) -> int | None:
        """Return seconds if text contains 'set timer for X ...', else None."""
        m = _TIMER_RE.search(text)
        if not m:
            return None
        try:
            val = int(m.group(1))
            unit = (m.group(2) or "s").lower()
            if unit.startswith("m"):  # minute(s), min, m
                return val * 60
            if unit.startswith("h"):
                return val * 3600
            return val
        except Exception:
            return None

    def is_cancel_intent(self, text: str) -> bool:
        return bool(_CANCEL_RE.search(text))

    def status_message(self) -> str:
        lst = self.list_timers()
        if not lst:
            return "No active timer."
        if len(lst) == 1:
            e = lst[0]
            return f"Timer #{e['id']} ({e['label']}) — {e['remaining']}s of {e['seconds']}s remaining. Say 'stop timer' to cancel."
        parts = ", ".join(f"#{e['id']} {e['label']} {e['remaining']}s/{e['seconds']}s" for e in lst)
        return f"{len(lst)} timers active: {parts}. Say 'stop timer' to cancel all."

def get_timer_tools() -> list[dict]:
    """OpenAI tool definitions for set/stop timer. Model can set timer via tool call."""
    return [
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Set a countdown timer. When it fires, a sound will play in a loop until the user says 'stop timer'. Use this when the user asks to set a timer, reminder, or alarm.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "integer", "description": "Duration in seconds (e.g., 60 for 1 minute, 300 for 5 minutes)"},
                        "label": {"type": "string", "description": "Optional label for the timer"},
                    },
                    "required": ["seconds"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_timer",
                "description": "Stop/cancel the active timer and its alarm sound. Use when user says stop timer, cancel timer, or similar.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "timer_status",
                "description": "Check remaining time on active timer.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]
