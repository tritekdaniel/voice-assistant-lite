from __future__ import annotations

import queue
import threading

import numpy as np

from .logger import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz, the openwakeword window
OUT_RATE = 24000  # Kokoro output rate
PLAY_CHUNK_SAMPLES = int(OUT_RATE * 0.25)


class AudioIn:
    """Microphone input as fixed-size int16 frames fanned out to subscriber queues."""

    def __init__(self, device: int | None = None):
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()
        self._stream = None
        self._device = device

    def add_subscriber(self, q: queue.Queue) -> None:
        with self._subs_lock:
            self._subs.append(q)

    def start(self) -> None:
        import sounddevice as sd

        if status := sd.query_devices(self._device) if self._device is not None else None:
            log.debug("AudioIn using device %s: %s", self._device, status)

        def _cb(*args):  # robust to 4-arg (InputStream) vs 5-arg (RawStream) signatures
            try:
                if len(args) == 4:
                    indata, frames, time_info, status = args
                elif len(args) == 5:
                    indata, _outdata, frames, time_info, status = args  # RawStream
                else:
                    log.warning("AudioIn callback unexpected argc=%s", len(args))
                    return
                if status:
                    log.warning("AudioIn status: %s", status)
                # indata: RawStream -> bytes, InputStream -> np.ndarray (frames,1)
                if isinstance(indata, bytes):
                    data = np.frombuffer(indata, dtype=np.int16).copy()
                elif isinstance(indata, np.ndarray):
                    # squeeze (1280,1) -> (1280,)
                    data = np.squeeze(indata).copy()
                    if data.ndim != 1:
                        data = data.reshape(-1)
                    if data.dtype != np.int16:
                        data = data.astype(np.int16)
                else:
                    # fallback for buffer protocol
                    data = np.frombuffer(bytes(indata), dtype=np.int16).copy()
                # ensure expected size (pad/truncate if driver gave different block)
                if data.shape[0] != FRAME_SAMPLES:
                    orig = data.shape[0]
                    if data.shape[0] > FRAME_SAMPLES:
                        data = data[:FRAME_SAMPLES].copy()
                    else:
                        data = np.pad(data, (0, FRAME_SAMPLES - data.shape[0])).astype(np.int16, copy=False)
                    log.debug("AudioIn callback normalized %s -> %s samples", orig, FRAME_SAMPLES)
            except BaseException as e:
                # Never raise from callback — would become CFFI "Exception ignored"
                log.exception("AudioIn callback failed: %s", e)
                return
            with self._subs_lock:
                subs = list(self._subs)
            for q in subs:
                try:
                    q.put(data.copy())
                except Exception:
                    pass

        try:
            # Use InputStream (not RawStream) — callback is (indata frames time status)
            # InputStream gives us np.ndarray directly, no bytes conversion, and avoids the 5-arg RawStream mismatch
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                device=self._device,
                callback=_cb,
            )
            self._stream.start()
            log.info("AudioIn started device=%s block=%s (InputStream)", self._device, FRAME_SAMPLES)
        except BaseException as e:
            # Provide actionable hint for Linux binary missing system lib
            msg = str(e)
            if "PortAudio" in msg and "not found" in msg:
                e = OSError(f"{e} — on Linux install system PortAudio: sudo apt install portaudio19-dev libportaudio2, then rebuild or run .venv/bin/python -m vocalis")
            log.exception("AudioIn start failed device=%s: %s", self._device, e)
            raise

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
                log.debug("AudioIn stopped")
            except Exception as e:
                log.warning("AudioIn stop error: %s", e)
            finally:
                self._stream = None


class Playback:
    """Sequential float32 24 kHz mono playback on its own thread.

    Audio is queued in <=250 ms pieces so an interrupt cuts off within a
    quarter second."""

    def __init__(self, device: int | None = None):
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._device = device
        self._stream = None
        self._closed = threading.Event()
        self._cancel = threading.Event()
        self._writing = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        # Thread objects are one-shot; recreate if already used. Check ident (set after first start) instead of private _started.
        if self._thread.ident is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
        try:
            self._thread.start()
        except RuntimeError:
            pass

    def put(self, audio: np.ndarray) -> None:
        a = np.asarray(audio, dtype=np.float32).ravel()
        for i in range(0, len(a), PLAY_CHUNK_SAMPLES):
            if self._closed.is_set():
                return
            self._q.put(a[i:i + PLAY_CHUNK_SAMPLES].copy())

    def cancel(self) -> None:
        self._cancel.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def idle(self) -> bool:
        # Use lock-free but consistent check: _writing is only written by _run thread
        # and read here; GIL ensures atomic bool read/write.
        return self._q.empty() and not self._writing

    def stop(self) -> None:
        self.cancel()
        self._closed.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._close_stream()

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._cancel.is_set():
                self._cancel.clear()
                self._drop_stream()
                while True:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        break
                continue
            self._ensure_stream()
            if self._stream is None:
                # stream failed — re-queue chunk with backoff instead of dropping
                try:
                    self._q.put(chunk)
                except Exception:
                    pass
                import time as _time
                _time.sleep(0.2)
                continue
            self._writing = True
            try:
                self._stream.write(chunk)
            except Exception:
                # on write failure, re-queue and retry
                try:
                    if not self._closed.is_set() and not self._cancel.is_set():
                        self._q.put(chunk)
                except Exception:
                    pass
            finally:
                self._writing = False

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=OUT_RATE,
                channels=1,
                dtype="float32",
                device=self._device,
            )
            self._stream.start()
            log.debug("Playback stream opened device=%s (OutputStream)", self._device)
        except BaseException as e:
            log.warning("Playback _ensure_stream failed device=%s: %s", self._device, e)
            self._stream = None

    def _drop_stream(self) -> None:
        if self._writing:
            return
        self._close_stream()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None


def input_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    out = []
    for d in sd.query_devices():
        if d.get("max_input_channels", 0) > 0:
            out.append((d["index"], d["name"]))
    return out


def output_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    out = []
    for d in sd.query_devices():
        if d.get("max_output_channels", 0) > 0:
            out.append((d["index"], d["name"]))
    return out
