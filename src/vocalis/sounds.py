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

def _resolve_sound_path(name_or_path: str) -> Path | None:
    """If name_or_path is an absolute/custom path that exists, use it; else resolve asset name."""
    if name_or_path and Path(name_or_path).expanduser().exists():
        return Path(name_or_path).expanduser()
    if name_or_path and (Path(name_or_path).exists()):
        return Path(name_or_path)
    # bare filename -> asset
    base = Path(name_or_path).name if name_or_path else name_or_path
    name = base if base else name_or_path
    p = ASSETS / name if name else None
    if p is not None and p.exists():
        return p
    alt = Path(__file__).parent / "assets" / name if name else None
    if alt is not None and alt.exists():
        return alt
    return p

def _load_asset(name: str) -> np.ndarray:
    # name may be custom path or asset filename
    resolved = _resolve_sound_path(name)
    path = resolved if resolved is not None else (ASSETS / name)
    if not path.exists():
        log.warning("Sound asset missing: %s (tried %s)", name, path)
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

    def __init__(self, playback: Playback, cfg=None):
        self._playback = playback
        self._cfg = cfg
        self._wake: np.ndarray = np.zeros(0, dtype=np.float32)
        self._finished: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lithium: np.ndarray = np.zeros(0, dtype=np.float32)
        self._timer_set: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._timer_stop = threading.Event()
        self._timer_thread: threading.Thread | None = None

    def _vol(self) -> float:
        try:
            if self._cfg and hasattr(self._cfg, "sound_volume"):
                return float(max(0.05, min(1.0, float(self._cfg.sound_volume))))  # type: ignore
        except Exception:
            pass
        return 0.7

    def _enabled(self) -> bool:
        try:
            if self._cfg and hasattr(self._cfg, "sound_enabled"):
                return bool(self._cfg.sound_enabled)  # type: ignore
        except Exception:
            pass
        return True

    def _path(self, kind: str) -> str:
        """Resolve configured custom path or default asset name."""
        try:
            if self._cfg:
                mapping = {
                    "wake": getattr(self._cfg, "sound_wake_path", ""),
                    "finished": getattr(self._cfg, "sound_finished_path", ""),
                    "timer_set": getattr(self._cfg, "sound_timer_set_path", ""),
                    "alarm": getattr(self._cfg, "sound_alarm_path", "") or getattr(self._cfg, "alarm_tone", ""),
                }
                v = (mapping.get(kind) or "").strip() if kind in mapping else ""
                if v:
                    return v
        except Exception:
            pass
        defaults = {"wake": "wake-up.ogg", "finished": "finished-listening.ogg", "timer_set": "timer-set.mp3", "alarm": "Lithium.mp3"}
        return defaults.get(kind, kind)

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return arr
        vol = self._vol()
        if abs(vol - 0.7) < 0.01:
            return arr
        # _load_asset already normalized to 0.7 peak → rescale
        peak = float(np.max(np.abs(arr))) if len(arr) else 0
        if peak < 0.01:
            return arr
        return (arr / peak * vol).astype(np.float32)

    def reload(self) -> None:
        """Force reload from current cfg paths (call after Save)."""
        try:
            w = _load_asset(self._path("wake"))
            f = _load_asset(self._path("finished"))
            l = _load_asset(self._path("alarm"))
            ts = _load_asset(self._path("timer_set"))
            with self._lock:
                if len(w):
                    self._wake = self._scale(w)
                else:
                    self._wake = w
                if len(f):
                    self._finished = self._scale(f)
                if len(l):
                    self._lithium = self._scale(l)
                if len(ts):
                    self._timer_set = self._scale(ts)
            log.info("Sounds reloaded: wake %d, finished %d, alarm %d, timer_set %d", len(self._wake), len(self._finished), len(self._lithium), len(self._timer_set))
        except Exception as e:
            log.exception("Sounds reload failed: %s", e)

    def preload(self) -> None:
        # Load synchronously first for immediate use, then background refresh
        if len(self._wake) == 0:
            try:
                self._wake = self._scale(_load_asset(self._path("wake")))
            except Exception:
                pass
        if len(self._timer_set) == 0:
            try:
                self._timer_set = self._scale(_load_asset(self._path("timer_set")))
            except Exception:
                pass
        def _load():
            try:
                w = self._scale(_load_asset(self._path("wake")))
                f = self._scale(_load_asset(self._path("finished")))
                l = self._scale(_load_asset(self._path("alarm")))
                ts = self._scale(_load_asset(self._path("timer_set")))
                with self._lock:
                    if len(w):
                        self._wake = w
                    if len(f):
                        self._finished = f
                    if len(l):
                        self._lithium = l
                    if len(ts):
                        self._timer_set = ts
                log.info("Sounds preloaded: wake %d, finished %d, alarm %d, timer_set %d samples", len(self._wake), len(self._finished), len(self._lithium), len(self._timer_set))
            except Exception as e:
                log.exception("Sounds preload failed: %s", e)
        t = threading.Thread(target=_load, daemon=True, name="sounds-preload")
        t.start()

    def ensure_loaded(self) -> None:
        if len(self._wake) == 0:
            self._wake = _load_asset(self._path("wake"))
        if len(self._finished) == 0:
            self._finished = _load_asset(self._path("finished"))
        if len(self._lithium) == 0:
            self._lithium = _load_asset(self._path("alarm"))
        if len(self._timer_set) == 0:
            self._timer_set = _load_asset(self._path("timer_set"))

    def play_wake(self) -> None:
        if not self._enabled():
            return
        self.ensure_loaded()
        if len(self._wake):
            log.debug("Playing wake sound %d samples", len(self._wake))
            self._playback.put(self._wake)

    def play_finished(self) -> None:
        if not self._enabled():
            return
        self.ensure_loaded()
        if len(self._finished):
            log.debug("Playing finished-listening %d samples", len(self._finished))
            self._playback.put(self._finished)

    def play_lithium_once(self) -> None:
        if not self._enabled():
            return
        self.ensure_loaded()
        if len(self._lithium):
            self._playback.put(self._lithium)

    def play_timer_set(self) -> None:
        if not self._enabled():
            return
        self.ensure_loaded()
        if len(self._timer_set):
            log.debug("Playing timer-set %d samples", len(self._timer_set))
            self._playback.put(self._timer_set)
        else:
            log.warning("timer sound not loaded, skipping")

    def preview(self, kind: str) -> bool:
        """Preview a sound kind: wake/finished/timer_set/alarm — returns True if played."""
        if not self._enabled():
            return False
        try:
            arr = self._scale(_load_asset(self._path(kind)))
            if len(arr) == 0:
                return False
            self._playback.put(arr)
            return True
        except Exception:
            return False

    def preview_file(self, path: str) -> bool:
        """Preview arbitrary file path (for Browse before Save)."""
        if not self._enabled():
            return False
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return False
            arr = self._scale(_load_asset(str(p)))
            if len(arr) == 0:
                return False
            self._playback.put(arr)
            return True
        except Exception:
            return False

    def play_timer_loop(self, loops: int = 5, gap_ms: int = 200) -> None:
        """Play timer alarm `loops` times with small gaps, cancellable via stop_timer_loop()."""
        with self._lock:
            # stop any existing loop before starting a new one
            if self._timer_thread is not None and self._timer_thread.is_alive():
                self._timer_stop.set()
                # don't join while holding lock to avoid deadlock; release briefly
                t_old = self._timer_thread
            else:
                t_old = None
            self._timer_stop.clear()
            stop_evt = self._timer_stop
        if t_old is not None:
            try:
                t_old.join(timeout=0.5)
            except Exception:
                pass
        def _run(stop=stop_evt):
            log.info("Timer loop start: %d loops", loops)
            for i in range(loops):
                if stop.is_set():
                    log.info("Timer loop cancelled at %d/%d", i, loops)
                    break
                self.play_lithium_once()
                dur = len(self._lithium) / OUT_RATE if len(self._lithium) else 1.0
                wait = dur + gap_ms / 1000.0
                end = __import__("time").monotonic() + wait
                while __import__("time").monotonic() < end:
                    if stop.is_set():
                        break
                    __import__("time").sleep(0.05)
            log.info("Timer loop done")
        with self._lock:
            self._timer_thread = threading.Thread(target=_run, daemon=True, name="timer-loop")
            self._timer_thread.start()

    def stop_timer_loop(self) -> None:
        with self._lock:
            t = self._timer_thread
        self._timer_stop.set()
        # cancel queued timer audio — but wake new non-timer TTS will still play after
        try:
            self._playback.cancel()
        except Exception:
            pass
        if t is not None and t.is_alive():
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        log.info("Timer loop stop requested")

    @property
    def is_timer_active(self) -> bool:
        t = self._timer_thread
        return t is not None and t.is_alive() and not self._timer_stop.is_set()
