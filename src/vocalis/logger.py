from __future__ import annotations

import logging
import logging.handlers
import sys
import traceback
from pathlib import Path

from .config import data_dir

LOG_NAME = "vocalis"
_logger: logging.Logger | None = None
_log_path: Path | None = None
_file_handler: logging.Handler | None = None


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_path() -> Path:
    return logs_dir() / "vocalis.log"


def setup_logging(level: str = "INFO", also_console: bool = False) -> logging.Logger:
    global _logger, _log_path, _file_handler

    if _logger is not None:
        return _logger

    # Normalize level
    lvl = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(lvl)
    # Don't propagate to root (avoid duplicate)
    logger.propagate = False
    # Clear old handlers if any
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — rotated, always on. Flush immediately so crash logs survive.
    lp = log_path()
    _log_path = lp
    fh = logging.handlers.RotatingFileHandler(
        lp, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    _file_handler = fh

    # Also mirror warnings
    logging.captureWarnings(True)

    # Console handler — for headless or when explicitly requested
    if also_console or _is_headless_env():
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    else:
        # In GUI mode still log errors to stderr so `vocalis --check` sees them
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Install global hooks so "closed before I could check" never happens again
    _install_hooks(logger)

    # Log startup banner
    logger.info("Vocalis %s starting - log: %s (level %s)", _version(), lp, level.upper())

    _logger = logger
    return logger


def get_logger(name: str = LOG_NAME) -> logging.Logger:
    if _logger is not None and name == LOG_NAME:
        return _logger
    # ensure base logger exists
    if _logger is None:
        setup_logging()
    return logging.getLogger(name)


def flush() -> None:
    if _file_handler is not None:
        try:
            _file_handler.flush()
        except Exception:
            pass
    for h in logging.getLogger(LOG_NAME).handlers:
        try:
            h.flush()
        except Exception:
            pass


def reveal_in_file_manager(path: Path | None = None) -> None:
    """Open the logs folder in the OS file manager (best-effort)."""
    import subprocess
    import sys

    target = Path(path) if path is not None else logs_dir()
    if not target.exists():
        target = logs_dir()
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(f'explorer "{target}"')  # type: ignore
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception:
        pass


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("vocalis")
    except Exception:
        try:
            from . import __version__
            return __version__
        except Exception:
            return "unknown"


def _is_headless_env() -> bool:
    return any(a in sys.argv for a in ("--headless", "--check")) or not sys.stdout.isatty()


def _install_hooks(logger: logging.Logger) -> None:
    import threading

    # sys.excepthook — catches unhandled exceptions on main thread, logs + dialog
    orig_excepthook = sys.excepthook

    def _hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        try:
            msg = "".join(traceback.format_exception(exc_type, exc, tb))
            logger.critical("Unhandled exception - will keep window open if GUI is running:\n%s", msg)
            flush()
            # Try to show a dialog if GUI is up
            _try_show_crash_dialog(msg)
        except Exception:
            pass
        # Still call original (prints to stderr)
        try:
            orig_excepthook(exc_type, exc, tb)
        except Exception:
            pass

    sys.excepthook = _hook

    # threading.excepthook — catches background thread crashes (wake/tts/stt threads)
    orig_thread_hook = getattr(threading, "excepthook", None)

    def _thread_hook(args):  # type: ignore[no-untyped-def]
        try:
            msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
            logger.error("Unhandled exception in thread %s: %s", args.thread.name if args.thread else "?", msg)
            flush()
        except Exception:
            pass
        if orig_thread_hook is not None:
            try:
                orig_thread_hook(args)
            except Exception:
                pass

    threading.excepthook = _thread_hook  # type: ignore[attr-defined]


def _try_show_crash_dialog(msg: str) -> None:
    # Only if a QApplication already exists (don't create one just to show dialog)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        if app is None:
            return
        # truncate
        short = msg[-4000:] if len(msg) > 4000 else msg
        # Must run on GUI thread — use singleShot if not on main thread
        from PySide6.QtCore import QTimer

        def _show():
            try:
                from pathlib import Path as _P

                lp = _log_path or log_path()
                QMessageBox.critical(
                    None,
                    "Vocalis — unexpected error",
                    f"Vocalis hit an unexpected error. It has been logged and the window will stay open.\n\n"
                    f"Log: {lp}\n\n"
                    f"Details (last lines):\n{short}\n\n"
                    f"Tip: Settings -> Open Logs Folder, or run vocalis --check --log-level DEBUG",
                )
            except Exception:
                pass

        # If we're on the GUI thread, show directly; otherwise post to it
        try:
            _show()
        except Exception:
            QTimer.singleShot(0, _show)
    except Exception:
        pass
