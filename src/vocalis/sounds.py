from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .audio_io import OUT_RATE, Playback
from .logger import get_logger

log = get_logger(__name__)

def _assets_dir() -> Path:
    # Try dev layout: <root>/assets
    dev = Path(__file__).resolve().parents[2] / "assets"
    if dev.exists():
        return dev
    # Frozen (PyInstaller): assets bundled as datas -> _MEIPASS/assets
    try:
        import sys
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            frozen = Path(sys._MEIPASS) / "assets"  # type: ignore[attr-defined]
            if frozen.exists():
                return frozen
    except Exception:
        pass
    # Fallback: src/vocalis/assets (if copied)
    alt = Path(__file__).parent / "assets"
    if alt.exists():
        return alt
    return dev

ASSETS = _assets_dir()

def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        # stereo -> mono average
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)

def _resample_linear(data: np.ndarray, sr_in: int, sr_out: int = OUT_RATE) -> np.ndarray:
    if sr_in == sr_out:
        return data
    # linear interpolation via np.interp
    ratio = sr_out / sr_in
    new_len = int(len(data) * ratio)
    if new_len == 0:
        return np.zeros(0, dtype=np.float32)
    old_idx = np.arange(len(data))
    new_idx = np.linspace(0, len(data) - 1, new_len)
    res = np.interp(new_idx, old_idx, data).astype(np.float32)
    return res

def _load_asset(name: str) -> np.ndarray:
    path = ASSETS / name
    if not path.exists():
        # Try alternative locations (frozen)
        alt = Path(__file__).parent / "assets" / name
        if alt.exists():
            path = alt
        else:
            log.warning("Sound asset missing: %s", path)
            return np.zeros(0, dtype=np.float32)
    # Try soundfile first (handles ogg, mp3 via libsndfile)
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        data = _to_mono_float32(data)
        data = _resample_linear(data, sr, OUT_RATE)
        # normalize to 0.7 peak to avoid clipping with TTS
        peak = float(np.max(np.abs(data))) if len(data) else 0
        if peak > 0.01:
            data = (data / peak * 0.7).astype(np.float32)
        log.debug("Loaded sound %s: %d samples @%d -> %d @%d", name, len(data), sr, len(data), OUT_RATE)
        return data
    except Exception as e:
        log.debug("soundfile failed for %s: %s, trying miniaudio", name, e)
    # Fallback: miniaudio
    try:
        import miniaudio
        d = miniaudio.decode_file(str(path))
        # d.samples is array('h') or bytes int16
        import array
        if isinstance(d.samples, array.array):
            arr = np.frombuffer(d.samples.tobytes(), dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(d.samples, bytes):
            arr = np.frombuffer(d.samples, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            arr = np.asarray(d.samples, dtype=np.float32)
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
        # d.nchannels may be 2, need to deinterleave and to mono
        if d.nchannels == 2:
            # interleaved stereo -> mono
            arr = arr.reshape(-1, 2).mean(axis=1)
        arr = _resample_linear(arr, d.sample_rate, OUT_RATE)
        peak = float(np.max(np.abs(arr))) if len(arr) else 0
        if peak > 0.01:
            arr = (arr / peak * 0.7).astype(np.float32)
        return arr
    except Exception as e:
        log.warning("Failed to load sound %s via miniaudio: %s", name, e)
        return np.zeros(0, dtype=np.float32)

class Sounds:
    """Preloads wake, finished, and timer sounds and plays them via Playback."""

    def __init__(self, playback: Playback):
        self._playback = playback
        self._wake: np.ndarray = np.zeros(0, dtype=np.float32)
        self._finished: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lithium: np.ndarray = np.zeros(0, dtype=np.float32)
        self._timer_set: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._timer_stop = threading.Event()
        self._timer_thread: threading.Thread | None = None

    def preload(self) -> None:
        # Load in background thread to not block UI, but also allow sync preload
        def _load():
            try:
                self._wake = _load_asset("wake-up.ogg")
                self._finished = _load_asset("finished-listening.ogg")
                self._lithium = _load_asset("Lithium.mp3")
                self._timer_set = _load_asset("timer-set.mp3")
                log.info("Sounds preloaded: wake %d, finished %d, lithium %d, timer_set %d samples", len(self._wake), len(self._finished), len(self._lithium), len(self._timer_set))
            except Exception as e:
                log.exception("Sounds preload failed: %s", e)
        t = threading.Thread(target=_load, daemon=True, name="sounds-preload")
        t.start()
        # also try sync for immediate use if needed
        if len(self._wake) == 0:
            # quick sync fallback for first wake
            try:
                self._wake = _load_asset("wake-up.ogg")
            except Exception:
                pass
        if len(self._timer_set) == 0:
            try:
                self._timer_set = _load_asset("timer-set.mp3")
            except Exception:
                pass

    def ensure_loaded(self) -> None:
        if len(self._wake) == 0:
            self._wake = _load_asset("wake-up.ogg")
        if len(self._finished) == 0:
            self._finished = _load_asset("finished-listening.ogg")
        if len(self._lithium) == 0:
            self._lithium = _load_asset("Lithium.mp3")
        if len(self._timer_set) == 0:
            self._timer_set = _load_asset("timer-set.mp3")

    def play_wake(self) -> None:
        self.ensure_loaded()
        if len(self._wake):
            log.debug("Playing wake sound %d samples", len(self._wake))
            # wake should be crisp — don't cancel ongoing TTS? But we do want to be heard
            self._playback.put(self._wake)

    def play_finished(self) -> None:
        self.ensure_loaded()
        if len(self._finished):
            log.debug("Playing finished-listening %d samples", len(self._finished))
            self._playback.put(self._finished)

    def play_lithium_once(self) -> None:
        self.ensure_loaded()
        if len(self._lithium):
            self._playback.put(self._lithium)

    def play_timer_set(self) -> None:
        self.ensure_loaded()
        if len(self._timer_set):
            log.debug("Playing timer-set %d samples", len(self._timer_set))
            self._playback.put(self._timer_set)
        else:
            log.warning("timer-set.mp3 not loaded, skipping")

    def play_timer_loop(self, loops: int = 5, gap_ms: int = 200) -> None:
        """Play timer alarm `loops` times with small gaps, cancellable via stop_timer_loop()."""
        self._timer_stop.clear()
        def _run():
            log.info("Timer loop start: %d loops", loops)
            for i in range(loops):
                if self._timer_stop.is_set():
                    log.info("Timer loop cancelled at %d/%d", i, loops)
                    break
                self.play_lithium_once()
                # wait for sound duration + gap, but cancellable
                dur = len(self._lithium) / OUT_RATE if len(self._lithium) else 1.0
                wait = dur + gap_ms / 1000.0
                # sleep in small increments to allow quick cancel
                end = __import__("time").monotonic() + wait
                while __import__("time").monotonic() < end:
                    if self._timer_stop.is_set():
                        break
                    __import__("time").sleep(0.05)
            log.info("Timer loop done")
        self._timer_thread = threading.Thread(target=_run, daemon=True, name="timer-loop")
        self._timer_thread.start()

    def stop_timer_loop(self) -> None:
        self._timer_stop.set()
        # also cancel any queued timer audio — but don't cancel all TTS if possible
        # For now, cancel all playback (timer is distinctive, user expects stop to be immediate)
        try:
            self._playback.cancel()
        except Exception:
            pass
        log.info("Timer loop stop requested")

    @property
    def is_timer_active(self) -> bool:
        t = self._timer_thread
        return t is not None and t.is_alive() and not self._timer_stop.is_set()
