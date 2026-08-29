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


class TimerManager:
    """Manages a single countdown timer that plays a sound 5× when fired."""

    def __init__(self, sounds: Sounds):
        self._sounds = sounds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._active_label: str | None = None
        self._duration: int | None = None
        self._start_monotonic: float | None = None

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._timer is not None and self._timer.is_alive()

    def remaining_seconds(self) -> int | None:
        with self._lock:
            if self._timer is None or self._start_monotonic is None or self._duration is None:
                return None
            elapsed = time.monotonic() - self._start_monotonic
            rem = int(self._duration - elapsed)
            return max(0, rem)

    def set_timer(self, seconds: int, label: str = "timer") -> str:
        if seconds <= 0:
            return "Timer duration must be positive."
        if seconds > 24 * 3600:
            return "Timer too long (max 24h)."
        self.cancel()
        log.info("Timer set for %ds (%s)", seconds, label)
        def _fire():
            log.info("Timer fired after %ds", seconds)
            with self._lock:
                self._active_label = None
                self._timer = None
                self._start_monotonic = None
            # Play alarm 5 times — cancellable via stop_timer_loop
            self._sounds.play_timer_loop(loops=5)
        with self._lock:
            self._active_label = label
            self._duration = seconds
            self._start_monotonic = time.monotonic()
            self._timer = threading.Timer(seconds, _fire)
            self._timer.daemon = True
            self._timer.start()
        return f"Timer set for {seconds} seconds. A sound will play when it rings. Say 'stop timer' to cancel."

    def cancel(self) -> str:
        with self._lock:
            t = self._timer
            self._timer = None
            self._active_label = None
            self._duration = None
            self._start_monotonic = None
        stopped = False
        if t is not None:
            try:
                t.cancel()
                stopped = True
            except Exception:
                pass
        # also stop any currently playing Lithium loop
        try:
            self._sounds.stop_timer_loop()
            stopped = True
        except Exception:
            pass
        if stopped:
            log.info("Timer cancelled")
            return "Timer cancelled."
        return "No active timer."

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
        rem = self.remaining_seconds()
        if rem is None:
            return "No active timer."
        return f"Timer active — {rem} seconds remaining. Say 'stop timer' to cancel."

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
