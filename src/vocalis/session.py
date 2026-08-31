from __future__ import annotations

import collections
import enum
import json
import math
import queue
import threading
import time

import numpy as np

from .audio_io import FRAME_SAMPLES, AudioIn, Playback
from .alarms import AlarmManager, get_alarm_tools
from .llm import History, LLMClient
from .logger import get_logger
from .sounds import Sounds
from .stt import Transcriber
from .textsplit import SentenceBuffer
from .timer import TimerManager, get_timer_tools
from .tts import Speaker, create_speaker
from .wakeword import WakeWord

log = get_logger(__name__)

FRAME_MS = 1000.0 * FRAME_SAMPLES / 16000
_PREROLL_FRAMES = 4


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Listener:
    def state_changed(self, state: State) -> None: ...
    def heard_text(self, text: str) -> None: ...
    def reply_delta(self, delta: str) -> None: ...
    def reply_complete(self, full: str) -> None: ...
    def error(self, message: str) -> None: ...


def rms_dbfs(frame_int16: np.ndarray) -> float:
    if frame_int16.size == 0:
        return -200.0
    x = frame_int16.astype(np.float32) / 32768.0
    r = math.sqrt(float(np.mean(x * x)))
    return 20.0 * math.log10(r + 1e-9)


class UtteranceBuffer:
    """Collects frames into one utterance; endpointing on trailing silence."""

    def __init__(self, speech_dbfs: float, silence_seconds: float, max_utterance_seconds: float):
        self._thr = speech_dbfs
        self._limit_ms = silence_seconds * 1000.0
        self._max_ms = max_utterance_seconds * 1000.0
        self._preroll: collections.deque[np.ndarray] = collections.deque(maxlen=_PREROLL_FRAMES)
        self.reset()

    def reset(self) -> None:
        self._parts: list[np.ndarray] = []
        self._started = False
        self.silence_ms = 0.0
        self.active_ms = 0.0
        self.last_speech_at: float | None = None
        self._preroll.clear()

    def feed_frame(self, frame_int16: np.ndarray) -> bool:
        speech = rms_dbfs(frame_int16) > self._thr
        if not self._started:
            if speech:
                preroll = list(self._preroll)
                self._parts.extend(preroll)
                self._parts.append(frame_int16)
                self._started = True
                self.silence_ms = 0.0
                # count preroll + first speech frame in active_ms
                self.active_ms = len(preroll) * FRAME_MS + FRAME_MS
            else:
                self._preroll.append(frame_int16)
            if speech:
                self.last_speech_at = time.monotonic()
            return speech
        if speech:
            self.silence_ms = 0.0
            self.last_speech_at = time.monotonic()
        else:
            self.silence_ms += FRAME_MS
        self._parts.append(frame_int16)
        self.active_ms += FRAME_MS
        return speech

    @property
    def endpoint(self) -> bool:
        return self._started and (self.silence_ms >= self._limit_ms or self.active_ms >= self._max_ms)

    @property
    def started(self) -> bool:
        return self._started

    def audio_f32(self) -> np.ndarray:
        if not self._parts:
            return np.zeros(0, dtype=np.float32)
        raw = np.concatenate(self._parts).astype(np.float32) / 32768.0
        return raw


class Session:
    """Owns the conversation loop: wake word -> STT -> LLM -> TTS -> repeat."""

    def __init__(self, cfg, audio_in: AudioIn, playback: Playback, wakeword: WakeWord,
                 stt: Transcriber, llm: LLMClient, history: History, tts: Speaker,
                 listener: Listener, sounds: Sounds | None = None, timer: TimerManager | None = None, alarms: AlarmManager | None = None):
        self.cfg = cfg
        self._audio_in = audio_in
        self._playback = playback
        self._wakeword = wakeword
        self._stt = stt
        self._llm = llm
        self._history = history
        self._tts = tts
        self.listener = listener
        self._sounds = sounds
        self._timer = timer
        self._alarms = alarms

        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._wake_frames: queue.Queue[np.ndarray] = queue.Queue()
        self._lock = threading.RLock()
        self._state = State.IDLE
        self._wake_evt = threading.Event()
        self._cancel = threading.Event()
        self._stop = threading.Event()
        self._utt = UtteranceBuffer(
            cfg.vad_rms_dbfs, cfg.vad_silence_seconds, cfg.max_utterance_seconds
        )
        self._listen_entered_at: float | None = None
        self._wake_thread: threading.Thread | None = None
        self._grace_until: float | None = None  # continuous listening grace after speak

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        log.info("Session start: state=%s", self._state.value)
        self._audio_in.add_subscriber(self._frames)
        self._audio_in.add_subscriber(self._wake_frames)
        try:
            self._playback.start()
            log.debug("Playback started")
        except Exception as e:
            log.exception("Playback start failed: %s", e)
            self.listener.error(f"Playback failed: {e}")
        # preload sounds
        if self._sounds is not None:
            try:
                self._sounds.preload()
            except Exception as e:
                log.warning("Sounds preload failed: %s", e)
        # Try selected device, then fallback to default (fixes PortAudio Device unavailable on Linux after device change)
        try:
            self._audio_in.start()
            log.info("AudioIn started on device=%s", getattr(self._audio_in, "_device", None))
        except Exception as e:
            if getattr(self._audio_in, "_device", None) is not None and "Device unavailable" in str(e):
                log.warning("Selected input device %s unavailable, retrying with default", self._audio_in._device)
                try:
                    self._audio_in._device = None  # type: ignore[attr-defined]
                    # need to re-create stream with None - stop first if partially started
                    try:
                        self._audio_in.stop()
                    except Exception:
                        pass
                    self._audio_in.start()
                    log.info("AudioIn started on fallback default device")
                except Exception as e2:
                    log.exception("AudioIn fallback also failed: %s", e2)
                    self.listener.error(f"Microphone failed: {e2} — check Settings -> audio device")
            else:
                log.exception("AudioIn start failed: %s", e)
                self.listener.error(f"Microphone failed: {e} — check Settings -> audio device")
        self._wake_thread = threading.Thread(target=self._wake_loop, daemon=True, name="wake")
        self._wake_thread.start()
        log.debug("Wake thread started")

    def run(self) -> None:
        log.info("Session run loop entered")
        try:
            while not self._stop.is_set():
                st = self.state
                if st is State.IDLE:
                    fired = self._wake_evt.wait(timeout=0.25)
                    self._wake_evt.clear()
                    if fired and not self._stop.is_set():
                        log.info("Wake word fired -> LISTENING")
                        self._enter_listening()
                elif st is State.LISTENING:
                    self._listen_step()
                elif st is State.THINKING:
                    self._think_step()
                else:
                    self._speak_step()
        except BaseException as e:
            log.critical("Session run crashed: %s", e, exc_info=True)
            self.listener.error(f"Session crashed: {e} — see log {__import__('pathlib').Path(__import__('vocalis.config', fromlist=['data_dir']).data_dir()) / 'logs' / 'vocalis.log'}")
            self._stop.set()
            raise
        finally:
            self._stop.set()
            log.info("Session run loop exited")

    def stop(self) -> None:
        log.info("Session stop requested (state=%s)", self.state.value)
        self._stop.set()
        self._cancel.set()
        self._wake_evt.set()
        if self._wake_thread is not None:
            self._wake_thread.join(timeout=2.0)
            log.debug("Wake thread joined")
        # stop timer as well
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass
        if self._sounds is not None:
            try:
                self._sounds.stop_timer_loop()
            except Exception:
                pass
        if self._alarms is not None:
            try:
                self._alarms.stop()
            except Exception:
                pass
        try:
            self._playback.stop()
            log.debug("Playback stopped")
        except Exception as e:
            log.warning("Playback stop error: %s", e)
        try:
            self._audio_in.stop()
            log.debug("AudioIn stopped")
        except Exception as e:
            log.warning("AudioIn stop error: %s", e)

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def update_wakeword(self, threshold: float | None = None, cooldown_ms: int | None = None):
        try:
            if threshold is not None and hasattr(self._wakeword, "set_threshold"):
                self._wakeword.set_threshold(float(threshold))
            elif threshold is not None:
                self._wakeword.threshold = float(threshold)
            if cooldown_ms is not None and hasattr(self._wakeword, "set_cooldown"):
                self._wakeword.set_cooldown(int(cooldown_ms))
            log.info("Live wakeword updated threshold=%s cooldown=%s", threshold, cooldown_ms)
        except Exception as e:
            log.warning("Live wakeword update failed: %s", e)

    # -- wake word thread ---------------------------------------------------

    def _wake_loop(self) -> None:
        log.debug("Wake loop started (onnx-only)")
        warned = False
        consecutive = 0
        while not self._stop.is_set():
            try:
                frame = self._wake_frames.get(timeout=0.25)
            except queue.Empty:
                continue
            # Defensive: ensure 1280 samples (audio_io already pads, but be safe)
            try:
                if frame.shape[0] != FRAME_SAMPLES:
                    log.warning("Wake frame size %s != %s, normalizing", frame.shape[0], FRAME_SAMPLES)
                    if frame.shape[0] > FRAME_SAMPLES:
                        frame = frame[:FRAME_SAMPLES].copy()
                    else:
                        import numpy as np
                        frame = np.pad(frame, (0, FRAME_SAMPLES - frame.shape[0])).astype(np.int16, copy=False)
            except Exception:
                pass
            try:
                fired = self._wakeword.trigger(frame)
                consecutive = 0  # reset on success
            except BaseException as e:  # noqa: BLE001
                consecutive += 1
                # Log first error at exception, then every 50th to avoid spam
                if consecutive == 1:
                    # Include shape/threshold for diagnostics, don't leak huge traceback every frame
                    try:
                        shape = getattr(frame, "shape", "?")
                    except Exception:
                        shape = "?"
                    log.exception("Wake word trigger failed (shape=%s thr=%.2f): %s", shape, getattr(self._wakeword, "threshold", 0), e)
                    self.listener.error(f"Wake word error: {e} — check Settings -> Wake word (log: {__import__('pathlib').Path(__import__('vocalis.config', fromlist=['data_dir']).data_dir()) / 'logs' / 'vocalis.log'})")
                elif consecutive % 50 == 0:
                    log.warning("Wake word still failing (%d times): %s", consecutive, e)
                # Shorter backoff now that onnx is stable; old 0.5-1s made wake appear dead
                time.sleep(0.05 if consecutive < 10 else 0.2)
                continue
            if fired:
                log.info("Wake word detected (state=%s)", self.state.value)
                with self._lock:
                    st = self._state
                if st is State.IDLE:
                    self._wake_evt.set()
                else:
                    log.info("Barge-in wake word -> cancel current turn + timer")
                    self._cancel.set()
                    # also stop timer loop if active
                    if self._timer is not None:
                        try:
                            self._timer.cancel()
                        except Exception:
                            pass
                    if self._sounds is not None:
                        try:
                            self._sounds.stop_timer_loop()
                        except Exception:
                            pass
        log.debug("Wake loop exiting")

    # -- state helpers -------------------------------------------------------

    def _set_state(self, st: State) -> None:
        with self._lock:
            self._state = st
        log.info("State -> %s", st.value)
        self.listener.state_changed(st)

    def _drain_frames(self) -> list[np.ndarray]:
        out = []
        while True:
            try:
                out.append(self._frames.get_nowait())
            except queue.Empty:
                return out

    def _enter_listening(self) -> None:
        self._cancel.clear()
        for f in self._drain_frames():
            pass
        self._utt.reset()
        self._listen_entered_at = time.monotonic()
        # set grace period for continuous listening after speak
        if getattr(self.cfg, "continuous_listening", True):
            grace = float(getattr(self.cfg, "listen_grace_seconds", 10.0))
            self._grace_until = time.monotonic() + grace
        else:
            self._grace_until = None
        self._set_state(State.LISTENING)
        # wake sound — play without blocking
        if self._sounds is not None:
            try:
                self._sounds.play_wake()
            except Exception as e:
                log.debug("Wake sound failed: %s", e)

    def _enter_idle(self) -> None:
        self._cancel.clear()
        for f in self._drain_frames():
            pass
        self._utt.reset()
        self._listen_entered_at = None
        self._grace_until = None
        self._set_state(State.IDLE)

    # -- LISTENING ------------------------------------------------------------

    def _listen_step(self) -> None:
        try:
            frame = self._frames.get(timeout=0.25)
        except queue.Empty:
            if self._idle_expired():
                self._enter_idle()
            return
        for f in [frame] + self._drain_frames():
            self._process_frame(f)
            if self.state is not State.LISTENING:
                break

    def _process_frame(self, frame: np.ndarray) -> None:
        if self._cancel.is_set():
            self._cancel.clear()
            self._utt.reset()
            return
        was_started = self._utt.started
        self._utt.feed_frame(frame)
        # finished listening sound when we detect endpoint after speech
        if self.state is State.LISTENING and self._idle_expired():
            self._enter_idle()
            return
        if self._utt.endpoint:
            # play finished sound before STT
            if not was_started or self._utt.started:
                if self._sounds is not None:
                    try:
                        self._sounds.play_finished()
                    except Exception as e:
                        log.debug("Finished sound failed: %s", e)
            self._flush_utterance()

    def _idle_expired(self) -> bool:
        if self._utt.started or self._listen_entered_at is None:
            return False
        # during grace period after speak, don't expire to IDLE — stay in LISTENING for follow-up
        if self._grace_until is not None and time.monotonic() < self._grace_until:
            return False
        return (time.monotonic() - self._listen_entered_at) > self.cfg.idle_timeout_seconds

    def _flush_utterance(self) -> None:
        audio = self._utt.audio_f32()
        self._utt.reset()
        if len(audio) < int(16000 * 0.2):
            log.debug("Utterance too short (%.2fs), ignoring", len(audio)/16000)
            # keep grace window open for follow-up attempts
            return
        # reset grace once we got a valid utterance
        self._grace_until = None
        log.info("STT transcribe %.2fs audio", len(audio)/16000)
        try:
            text = self._stt.transcribe(audio)
            log.info("STT result: %r", text)
        except BaseException as e:  # noqa: BLE001
            log.exception("STT failed: %s", e)
            self.listener.error(f"STT failed: {e}")
            return
        if self._cancel.is_set():
            log.info("STT result discarded due to cancel")
            self._cancel.clear()
            return
        text = text.strip()
        if not text:
            log.debug("STT empty, ignoring")
            return
        # timer direct voice commands — handle without LLM if needed
        if self._timer is not None:
            low = text.lower()
            if self._timer.is_cancel_intent(low):
                msg = self._timer.cancel()
                log.info("Heard timer cancel: %r -> %s", text, msg)
                self._history.add_user(text)
                self._history.add_assistant(msg)
                self.listener.heard_text(text)
                self.listener.reply_complete(msg)
                # speak confirmation and go straight back to listening (no LLM)
                self._speak_sentence(msg)
                self._set_state(State.SPEAKING)
                return
            secs = self._timer.parse_voice_set(low)
            if secs is not None:
                msg = self._timer.set_timer(secs)
                log.info("Heard timer set: %r -> %s", text, msg)
                self._history.add_user(text)
                self._history.add_assistant(msg)
                self.listener.heard_text(text)
                self.listener.reply_complete(msg)
                self._speak_sentence(msg)
                self._set_state(State.SPEAKING)
                return
        log.info("Heard: %s", text)
        self._history.add_user(text)
        self.listener.heard_text(text)
        self._set_state(State.THINKING)

    # -- THINKING --------------------------------------------------------------

    def _think_step(self) -> None:
        log.info("THINKING: calling LLM %s (temp %.2f, %d history msgs, forget=%s)", self.cfg.llm_model, self.cfg.temperature, len(self._history.messages()), getattr(self.cfg, "forget_history", False))
        # include timer + alarm tools if available
        tools = None
        all_tools: list[dict] = []
        if self._timer is not None:
            try:
                all_tools.extend(get_timer_tools())
            except Exception:
                pass
        if self._alarms is not None and getattr(self.cfg, "alarms_enabled", True):
            try:
                all_tools.extend(get_alarm_tools())
            except Exception:
                pass
        if all_tools:
            tools = all_tools
        splitter = SentenceBuffer()
        reply_parts: list[str] = []
        gen = None
        cancelled = False
        tool_calls_handled = False
        try:
            gen = self._llm.stream_reply(self._history.messages(), tools=tools)
            for delta in gen:
                if self._stop.is_set() or self._cancel.is_set():
                    cancelled = True
                    log.info("LLM stream cancelled mid-reply")
                    break
                reply_parts.append(delta)
                self.listener.reply_delta(delta)
                for sent in splitter.feed(delta):
                    if self._stop.is_set() or self._cancel.is_set():
                        cancelled = True
                        break
                    if cancelled:
                        break
                    self._speak_sentence(sent)
                if cancelled:
                    break
        except BaseException as e:  # noqa: BLE001
            log.exception("LLM failed: %s", e)
            self.listener.error(f"LLM failed: {e} — is the server at {self.cfg.llm_base_url} running?")
        finally:
            if gen is not None:
                try:
                    gen.close()
                except Exception:  # noqa: BLE001
                    pass

        # handle tool calls (timer) even if content was empty
        pending = getattr(self._llm, "pending_tool_calls", [])
        if pending and not cancelled and not self._stop.is_set():
            log.info("Handling %d tool calls", len(pending))
            for tc in pending:
                try:
                    name = tc.get("function", {}).get("name", "")
                    args_raw = tc.get("function", {}).get("arguments", "{}")
                    args = json.loads(args_raw) if args_raw else {}
                    log.info("Tool call %s %s", name, args)
                    if name == "set_timer":
                        secs = int(args.get("seconds", 0))
                        res = self._timer.set_timer(secs, args.get("label", "timer")) if self._timer else "Timer not available"
                        # add tool result to history as assistant message for context
                        self._history.add_user(f"[tool set_timer {secs}s -> {res}]")
                        self._speak_sentence(res)
                        tool_calls_handled = True
                    elif name == "stop_timer":
                        res = self._timer.cancel() if self._timer else "No timer"
                        self._history.add_user("[tool stop_timer]")
                        self._speak_sentence(res)
                        tool_calls_handled = True
                    elif name == "timer_status":
                        res = self._timer.status_message() if self._timer else "No timer"
                        self._speak_sentence(res)
                        tool_calls_handled = True
                    elif name == "set_alarm":
                        t = args.get("time") or args.get("at") or ""
                        # handle HH:MM short form -> today
                        if t and len(t) <= 5 and ":" in t and "T" not in t:
                            from datetime import datetime as _dt
                            t = _dt.now().strftime("%Y-%m-%dT") + t + ":00"
                        label = args.get("label", "alarm")
                        rec = args.get("recurrence", "once")
                        try:
                            res_d = self._alarms.add_alarm(t, label, rec) if self._alarms else {"error": "no alarms"}
                            res = f"Alarm set for {res_d.get('at')} ({rec}) {label}"
                        except Exception as e:
                            res = f"Alarm failed: {e}"
                        self._history.add_user(f"[tool set_alarm {t} -> {res}]")
                        self._speak_sentence(res)
                        tool_calls_handled = True
                    elif name == "list_alarms":
                        lst = self._alarms.list_alarms() if self._alarms else []
                        if not lst:
                            res = "No alarms."
                        else:
                            parts = [f"#{a['id']} {a['label']} at {a['at']} ({a['recurrence']}) {'on' if a['enabled'] else 'off'}" for a in lst]
                            res = "; ".join(parts)
                        self._speak_sentence(res)
                        tool_calls_handled = True
                    elif name == "cancel_alarm":
                        aid = int(args.get("alarm_id", 0))
                        ok = self._alarms.remove_alarm(aid) if self._alarms else False
                        res = f"Alarm #{aid} cancelled." if ok else f"No alarm #{aid}"
                        self._speak_sentence(res)
                        tool_calls_handled = True
                except Exception as e:
                    log.exception("Tool handling failed: %s", e)

        tail = splitter.flush()
        full = "".join(reply_parts).strip()
        log.info("LLM reply done: %r (cancelled=%s, len=%d, tools=%s)", full[:200], cancelled, len(full), tool_calls_handled)
        if self._cancel.is_set():
            cancelled = True
        if not cancelled and tail and not self._stop.is_set():
            log.debug("Speaking tail: %r", tail)
            self._speak_sentence(tail)
        # Decide what constitutes a reply: either content or tool handling
        has_reply = bool(full) or tool_calls_handled
        if has_reply and full:
            self._history.add_assistant(full)
            self.listener.reply_complete(full)
        elif tool_calls_handled and not full:
            # tool-only turn: still complete with tool result already spoken
            self.listener.reply_complete("Timer updated.")
        if self._stop.is_set():
            return
        if cancelled or self._cancel.is_set():
            log.info("THINKING cancelled -> LISTENING")
            self._playback.cancel()
            self._enter_listening()
            return
        # forget history if requested (stateless) — but keep system prompt
        # History handling: forget vs preserve+compact
        forget = bool(getattr(self.cfg, "forget_history", False))
        preserve = bool(getattr(self.cfg, "preserve_history", False))
        # preserve overrides forget when both are set
        if preserve:
            forget = False
        if forget:
            log.debug("Forgetting history (forget_history=True)")
            self._history.clear()
        elif preserve:
            # Compact when we exceed compact_after (default 30)
            try:
                ca = int(getattr(self.cfg, "compact_after", 30))
            except Exception:
                ca = 30
            if len(self._history._turns) > ca:  # type: ignore[attr-defined]
                self._history.compact(keep=ca)  # type: ignore[attr-defined]
        self._set_state(State.SPEAKING)

    def _speak_sentence(self, sent: str) -> None:
        log.debug("TTS synthesize: %r", sent[:80])
        try:
            audio = self._tts.synthesize(sent)
            log.debug("TTS produced %d samples (%.2fs)", len(audio), len(audio)/24000 if len(audio) else 0)
        except BaseException as e:  # noqa: BLE001
            log.exception("TTS failed for %r: %s", sent[:80], e)
            self.listener.error(f"TTS failed: {e}")
            return
        if len(audio) > 0 and not self._cancel.is_set() and not self._stop.is_set():
            self._playback.put(audio)
            log.debug("Queued %d TTS samples for playback", len(audio))

    # -- SPEAKING ---------------------------------------------------------------

    def _speak_step(self) -> None:
        # Barge-in has priority — go straight to LISTENING to capture new utterance
        if self._cancel.is_set():
            log.info("SPEAKING cancelled (pre-check) -> cancel playback")
            self._playback.cancel()
            log.debug("SPEAKING cancelled -> LISTENING")
            self._enter_listening()
            return
        # After speaking, return to IDLE (requires wake word) — not directly to LISTENING/activated.
        # This fixes "goes straight into activated mode" bug. Continuous listening is opt-in via config.
        if self._playback.idle():
            log.debug("SPEAKING: nothing queued, → IDLE")
            self._enter_idle()
            return
        start = time.monotonic()
        max_wait = 30.0  # don't hang forever if playback stream never becomes idle
        while not self._stop.is_set() and not self._cancel.is_set():
            if self._playback.idle():
                break
            if time.monotonic() - start > max_wait:
                log.warning("SPEAKING: playback not idle after %.1fs, forcing → IDLE", max_wait)
                break
            time.sleep(0.05)
        if self._cancel.is_set():
            log.info("SPEAKING cancelled -> cancel playback")
            self._playback.cancel()
            log.debug("SPEAKING cancelled -> LISTENING")
            self._enter_listening()
            return
        # Normal finish: respect continuous_listening flag
        if getattr(self.cfg, "continuous_listening", False):
            log.debug("SPEAKING done -> LISTENING (continuous mode)")
            self._enter_listening()
        else:
            log.debug("SPEAKING done -> IDLE")
            self._enter_idle()


class SessionFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, audio_in: AudioIn, playback: Playback, listener: Listener) -> Session:
        # sounds and timer and alarms are created per session and share playback
        sounds = Sounds(playback, self.cfg)
        # preload will be called in Session.start(), but we can also preload now
        from .sounds import Sounds as _S  # noqa: F811
        timer = TimerManager(sounds)
        alarms = None
        if getattr(self.cfg, "alarms_enabled", True):
            try:
                alarms = AlarmManager(sounds)
            except Exception as e:
                from .logger import get_logger
                get_logger(__name__).warning("Alarm init failed: %s", e)
        # TTS: kokoro or piper via factory
        try:
            tts = create_speaker(self.cfg)  # type: ignore[assignment]
        except Exception as e:
            # fallback to kokoro if piper not available / misconfigured
            from .logger import get_logger
            get_logger(__name__).warning("TTS factory failed (%s), falling back to kokoro", e)
            tts = Speaker(getattr(self.cfg, "tts_voice", "af_heart"), float(getattr(self.cfg, "tts_speed", 1.0)))
        return Session(
            self.cfg,
            audio_in,
            playback,
            WakeWord(self.cfg.wake_word, self.cfg.wakeword_threshold,
                     self.cfg.wakeword_cooldown_ms, self.cfg.wakeword_embeddings),
            Transcriber(self.cfg.whisper_model),
            LLMClient(self.cfg.llm_base_url, self.cfg.llm_model,
                      self.cfg.llm_api_key, self.cfg.temperature),
            History(self.cfg.system_prompt, self.cfg.max_history_messages,
                    compact_after=getattr(self.cfg, "compact_after", 30)),
            tts,
            listener,
            sounds=sounds,
            timer=timer,
            alarms=alarms,
        )
