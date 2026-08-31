from __future__ import annotations

import threading
import time

from .logger import get_logger

log = get_logger(__name__)

class Watchdog:
    """Monitors Session + its thread, auto-restarts on crash."""

    def __init__(self, cfg, window_getter, session_factory_fn, max_restarts: int = 5, window_seconds: int = 300):
        """
        window_getter: callable returning MainWindow or None (GUI) / dummy for headless
        session_factory_fn: callable(cfg, listener) -> Session
        """
        self.cfg = cfg
        self._window_getter = window_getter
        self._factory = session_factory_fn
        self._max_restarts = max_restarts
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._restarts: list[float] = []
        self._session = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def attach(self, session, thread: threading.Thread):
        with self._lock:
            self._session = session
            self._thread = thread

    def start(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop.clear()
        self._monitor_thread = threading.Thread(target=self._loop, daemon=True, name="watchdog")
        self._monitor_thread.start()
        log.info("Watchdog started (max %d restarts per %ds)", self._max_restarts, self._window_seconds)

    def stop(self):
        self._stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _can_restart(self) -> bool:
        now = time.monotonic()
        with self._lock:
            # prune old
            self._restarts = [t for t in self._restarts if now - t < self._window_seconds]
            if len(self._restarts) >= self._max_restarts:
                return False
            self._restarts.append(now)
            return True

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(1.0)
            try:
                sess = self._session
                thr = self._thread
                if sess is None or thr is None:
                    continue
                # intentional stop — don't restart
                if sess.stopped:
                    continue
                if not thr.is_alive():
                    log.critical("Watchdog: session thread died (stopped=%s) — restarting", sess.stopped)
                    self._restart()
                    continue
                # also check wake thread if available
                try:
                    wt = getattr(sess, "_wake_thread", None)
                    if wt is not None and not wt.is_alive() and not sess.stopped:
                        # wake thread should never die (it catches exceptions); if it did, restart
                        log.critical("Watchdog: wake thread died — restarting session")
                        self._restart()
                except Exception:
                    pass
            except Exception as e:
                log.exception("Watchdog loop error: %s", e)

    def _restart(self):
        if not self._can_restart():
            log.critical("Watchdog: too many restarts (%d in %ds) — giving up, needs manual restart", self._max_restarts, self._window_seconds)
            try:
                win = self._window_getter()
                if win is not None:
                    try:
                        # use signal to show error on GUI thread
                        win.sig_error.emit(f"Session crashed repeatedly — check log {__import__('vocalis.logger', fromlist=['log_path']).log_path()}")  # type: ignore
                    except Exception:
                        pass
            except Exception:
                pass
            return
        try:
            win = self._window_getter()
            if win is not None:
                # GUI restart via MainWindow._restart_session if available
                if hasattr(win, "_restart_session"):
                    log.info("Watchdog: invoking GUI _restart_session")
                    # must run on GUI thread via QTimer
                    try:
                        from PySide6.QtCore import QTimer
                        # capture
                        def do():
                            try:
                                win._restart_session()  # type: ignore
                                # re-attach new session/thread
                                try:
                                    new_sess = getattr(win, "_session", None)
                                    # find thread by name? MainWindow doesn't store thread; hunt?
                                    # watchdog will be re-attached by launch_gui after restart; for now just set
                                    pass
                                except Exception:
                                    pass
                            except Exception as e:
                                log.exception("Watchdog GUI restart failed: %s", e)
                        QTimer.singleShot(0, do)
                    except Exception as e:
                        log.exception("Watchdog QTimer restart failed: %s", e)
                    # give it a moment then re-attach will happen via poll; don't try to attach now
                    time.sleep(1.0)
                    return
            # headless fallback: restart via factory
            log.info("Watchdog: headless restart")
            old = self._session
            if old is not None:
                try:
                    old.stop()
                except Exception:
                    pass
            # need listener — for headless we need to recreate
            # try to get listener from window or create ConsoleListener
            listener = None
            try:
                win = self._window_getter()
                if win is not None and hasattr(win, "listener"):
                    listener = win.listener()  # type: ignore
            except Exception:
                pass
            if listener is None:
                from .runner import ConsoleListener
                listener = ConsoleListener()
            new_sess = self._factory(self.cfg, listener)
            new_thr = threading.Thread(target=new_sess.run, daemon=True, name="session")
            new_thr.start()
            self.attach(new_sess, new_thr)
            # update window if GUI
            try:
                win = self._window_getter()
                if win is not None and hasattr(win, "attach_session"):
                    win.attach_session(new_sess)  # type: ignore
            except Exception:
                pass
            log.info("Watchdog: restart complete")
        except Exception as e:
            log.exception("Watchdog restart failed: %s", e)

    def note_thread_error(self, thread_name: str):
        # called from threading.excepthook
        if thread_name in ("session", "wake"):
            log.warning("Watchdog noted thread error %s", thread_name)
