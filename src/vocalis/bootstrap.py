from __future__ import annotations

import time

from .logger import get_logger

log = get_logger(__name__)


def ensure_models(cfg, status) -> None:
    """Load (and on first run download) every model. Components are cached so a
    later Session rebuild is instant."""
    from .stt import Transcriber
    from .tts import create_speaker
    from .wakeword import WakeWord

    t0 = time.monotonic()
    eng = getattr(cfg, "tts_engine", "kokoro")
    voice_info = getattr(cfg, "piper_model", "") if eng == "piper" else getattr(cfg, "tts_voice", "af_heart")
    log.info("Bootstrap start: wake=%s whisper=%s tts=%s voice=%s", cfg.wake_word, cfg.whisper_model, eng, voice_info)
    status("wake", "Loading wake word model...")
    ww = WakeWord(cfg.wake_word, cfg.wakeword_threshold,
                 cfg.wakeword_cooldown_ms, cfg.wakeword_embeddings)
    try:
        ww.ensure_loaded()
        log.info("Bootstrap wake OK")
    except BaseException as e:  # noqa: BLE001
        log.exception("Bootstrap wake failed: %s", e)
        raise RuntimeError(f"Wake word model '{cfg.wake_word}' could not be loaded: {e}") from e

    status("stt", f"Loading Whisper ({cfg.whisper_model})...")
    stt = Transcriber(cfg.whisper_model)
    try:
        stt.ensure_loaded()
        log.info("Bootstrap whisper OK")
    except BaseException as e:  # noqa: BLE001
        log.exception("Bootstrap whisper failed: %s", e)
        raise RuntimeError(f"Whisper model '{cfg.whisper_model}' could not be loaded: {e}") from e

    eng = getattr(cfg, "tts_engine", "kokoro")
    if eng == "piper":
        status("tts", "Loading Piper...")
    else:
        status("tts", "Loading Kokoro-82M...")
    try:
        from .tts import create_speaker as _cs
        spk = _cs(cfg)
    except Exception:
        from .tts import Speaker as _Sp
        spk = _Sp(cfg.tts_voice, cfg.tts_speed)
    try:
        spk.ensure_loaded()
        log.info("Bootstrap tts OK engine=%s", eng)
    except BaseException as e:  # noqa: BLE001
        log.exception("Bootstrap tts failed: %s", e)
        label = "Piper" if eng == "piper" else "Kokoro-82M"
        raise RuntimeError(f"{label} could not be loaded: {e}") from e

    elapsed = time.monotonic() - t0
    log.info("Bootstrap done in %.1fs", elapsed)
    status("done", f"Ready in {elapsed:.0f}s")
