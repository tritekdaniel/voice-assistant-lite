from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocalis",
        description="Voice-first assistant for local LLMs (OpenAI-compatible APIs).",
    )
    p.add_argument("--base-url", help="LLM OpenAI-compatible base URL")
    p.add_argument("--model", help="LLM model name")
    p.add_argument("--api-key", help="LLM API key (any non-empty string for local servers)")
    p.add_argument("--system-prompt", help="System prompt")
    p.add_argument("--temperature", type=float, help="Sampling temperature 0-2")
    p.add_argument("--whisper-model", help="faster-whisper model, e.g. base.en small.en medium.en")
    p.add_argument("--voice", help="Kokoro voice id, e.g. af_heart am_adam")
    p.add_argument("--speed", type=float, help="TTS speed 0.5-2.0")
    p.add_argument("--tts-engine", choices=["kokoro", "piper"], help="TTS engine")
    p.add_argument("--piper-model", help="Piper voice .onnx path")
    p.add_argument("--wake-word", help="Wake word model name (hey_jarvis) or path to a custom .onnx")
    p.add_argument("--wakeword-threshold", type=float, help="Wake trigger threshold 0.1-0.9")
    p.add_argument("--headless", action="store_true", help="Run without the GUI window")
    p.add_argument("--check", action="store_true", help="Self-test models and audio, then exit")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level (default INFO)")
    p.add_argument("--logs", action="store_true", help="Print log file location and exit")
    return p


def _apply_flags(args, cfg) -> None:
    from dataclasses import fields

    names = {f.name for f in fields(type(cfg))}
    # Normalize model ids like config does (strip leading /\, handle Windows paths)
    def _norm(v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip().lstrip("/\\").strip()
        # keep namespace like unsloth/ but basename gguf
        if "\\" in s or ("/" in s and s.count("/") > 1 and ":" not in s):
            if s.lower().endswith((".gguf", ".bin", ".onnx")):
                s = s.replace("\\", "/").split("/")[-1]
        return s
    mapping = {
        "llm_base_url": args.base_url,
        "llm_model": _norm(args.model) if args.model else None,
        "llm_api_key": args.api_key,
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "whisper_model": args.whisper_model.strip() if args.whisper_model else None,
        "tts_voice": args.voice.strip().lstrip("/\\") if args.voice else None,
        "tts_speed": args.speed,
        "tts_engine": args.tts_engine.strip().lower() if args.tts_engine else None,
        "piper_model": args.piper_model.strip() if args.piper_model else None,
        "wake_word": args.wake_word.strip().lstrip("/\\") if args.wake_word else None,
        "wakeword_threshold": args.wakeword_threshold,
    }
    changed = False
    for name, value in mapping.items():
        if value is not None and name in names:
            # Don't overwrite with empty string
            if isinstance(value, str) and not value.strip():
                continue
            setattr(cfg, name, value)
            changed = True
    return changed


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # --logs is special: print path without polluting stdout with log banner
    if args.logs:
        # Use config directly to avoid initializing logger's banner on stdout
        from .config import data_dir
        lp = data_dir() / "logs" / "vocalis.log"
        # Also try logger's path (handles platformdirs correctly)
        try:
            from .logger import log_path as _lp
            lp = _lp()
        except Exception:
            pass
        print(str(lp))
        return 0

    # Logging — must be first, before any vocalis import that may log
    from .logger import setup_logging, get_logger, log_path

    also_console = args.headless or args.check
    log = setup_logging(args.log_level, also_console=also_console)

    from .config import apply_model_env, load_config, save_config

    log.debug("CLI args: %s", args)
    cfg = load_config()
    tts_info = cfg.piper_model if getattr(cfg, "tts_engine", "kokoro") == "piper" else cfg.tts_voice
    log.info("Config loaded from %s: base_url=%s model=%s whisper=%s tts=%s voice=%s wake=%s",
             "config.toml", cfg.llm_base_url, cfg.llm_model, cfg.whisper_model, getattr(cfg, "tts_engine", "kokoro"), tts_info, cfg.wake_word)
    if _apply_flags(args, cfg):
        save_config(cfg)
        log.info("Config saved via CLI flags")
    apply_model_env()
    log.debug("HF_HOME=%s", __import__("os").environ.get("HF_HOME", ""))

    if args.check:
        log.info("Running --check")
        rc = _run_check(cfg)
        log.info("--check finished with code %s", rc)
        from .logger import flush as _flush
        _flush()
        return rc
    if args.headless:
        log.info("Starting headless mode")
        from .runner import run_headless

        try:
            run_headless(cfg)
        except BaseException as e:
            log.critical("Headless crashed: %s", e, exc_info=True)
            raise
        return 0
    log.info("Starting GUI mode")
    from .app import launch_gui

    try:
        launch_gui(cfg)
    except BaseException as e:
        # Never let the window vanish silently — log + keep message visible
        log.critical("GUI crashed: %s", e, exc_info=True)
        try:
            from .logger import flush
            flush()
        except Exception:
            pass
        # Show fallback dialog if QApplication not yet handling it
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is not None:
                QMessageBox.critical(None, "Vocalis — crashed",
                    f"Vocalis crashed and was logged to:\n{log_path()}\n\n{e}\n\n"
                    "Run: vocalis --logs  to find the log, or Settings -> Open Logs Folder.\n"
                    "The window will stay open — check the log before closing.")
                app.exec()
        except Exception:
            pass
        raise
    return 0


def _run_check(cfg) -> int:
    import numpy as np

    ok = True

    def line(label, good, detail=""):
        nonlocal ok
        mark = "ok" if good else "FAIL"
        print(f"[{mark}] {label}" + (f" - {detail}" if detail else ""))
        ok = ok and good

    try:
        from .audio_io import input_devices, output_devices

        ins, outs = input_devices(), output_devices()
        line("audio devices", bool(ins) and bool(outs),
             f"{len(ins)} in / {len(outs)} out")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[fail] audio devices - {e}")

    try:
        from .wakeword import WakeWord

        ww = WakeWord(cfg.wake_word, cfg.wakeword_threshold,
                      cfg.wakeword_cooldown_ms, cfg.wakeword_embeddings)
        score = ww.score(np.zeros(1280, dtype=np.int16))
        line("wake word", True, f"model '{cfg.wake_word}' loaded (silent score {score:.3f})")
    except KeyboardInterrupt:
        ok = False
        print("[fail] wake word - scipy import hang detected (known issue on some systems)")
        print("       Try: export SCIPY_USE_PROPAGATE=1  or  use venv mode without --check")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[fail] wake word - {e}")

    try:
        from .stt import Transcriber

        stt = Transcriber(cfg.whisper_model)
        text = stt.transcribe(np.zeros(16000, dtype=np.float32))
        line("whisper", True, f"'{cfg.whisper_model}' loaded (silence -> '{text}')")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[fail] whisper - {e}")

    # Check active TTS engine (kokoro or piper)
    try:
        from .tts import create_speaker

        spk = create_speaker(cfg)
        audio = spk.synthesize("Check.")
        eng = getattr(cfg, "tts_engine", "kokoro")
        label = getattr(cfg, "piper_model", "") if eng == "piper" else getattr(cfg, "tts_voice", "")
        line(eng, len(audio) > 0, f"'{label}' synthesized {len(audio) / 24000:.1f}s")
    except Exception as e:  # noqa: BLE001
        ok = False
        eng = getattr(cfg, "tts_engine", "kokoro")
        print(f"[fail] {eng} - {e}")

    try:
        from .llm import LLMClient

        reply = LLMClient(cfg.llm_base_url, cfg.llm_model, cfg.llm_api_key).check()
        line("llm", True, f"{cfg.llm_base_url} model '{cfg.llm_model}' replied '{reply[:40]}'")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[fail] llm - is the server running at {cfg.llm_base_url}? ({e})")

    print("check passed" if ok else "check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
