from __future__ import annotations

import threading
import time

from .audio_io import AudioIn, Playback
from .config import Config
from .logger import get_logger
from .session import Listener, Session, State

log = get_logger(__name__)


class ConsoleListener(Listener):
    def state_changed(self, state: State) -> None:
        print(f"[state] {state.value}")
        log.info("Headless state: %s", state.value)

    def heard_text(self, text: str) -> None:
        print(f"[heard] {text}")
        log.info("Headless heard: %s", text)

    def reply_delta(self, delta: str) -> None:
        pass

    def reply_complete(self, full: str) -> None:
        print(f"[reply] {full}")
        log.info("Headless reply: %s", full)

    def error(self, message: str) -> None:
        print(f"[error] {message}")
        log.error("Headless error: %s", message)


def start_session(cfg: Config, listener: Listener) -> Session:
    from .session import SessionFactory

    audio_in = AudioIn(device=cfg.input_device)
    playback = Playback(device=cfg.output_device)
    session = SessionFactory(cfg).build(audio_in, playback, listener)
    session.start()
    return session


def run_headless(cfg: Config) -> None:
    listener = ConsoleListener()
    log.info("Headless start: base_url=%s model=%s", cfg.llm_base_url, cfg.llm_model)
    print("vocalis (headless) - say the wake word to start. Ctrl+C to quit.")
    print(f"Log: {__import__('vocalis.logger', fromlist=['log_path']).log_path()}  (--logs to print path)")
    session = start_session(cfg, listener)
    t = threading.Thread(target=session.run, daemon=True, name="session")
    t.start()
    try:
        while not session.stopped:
            if not t.is_alive():
                log.critical("Session thread died unexpectedly — exiting headless loop")
                print("[error] Session crashed — see log for details")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        log.info("Headless KeyboardInterrupt")
    except BaseException as e:
        log.critical("Headless died: %s", e, exc_info=True)
        raise
    finally:
        log.info("Headless stopping")
        session.stop()
        log.info("Headless stopped")
        from .logger import flush
        flush()
