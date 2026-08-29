from __future__ import annotations

import numpy as np

from .logger import get_logger

log = get_logger(__name__)

_cache: dict[str, object] = {}


class Transcriber:
    """faster-whisper speech-to-text, CPU int8."""

    def __init__(self, model_name: str = "base.en"):
        self._name = model_name
        self._model = _cache.get(model_name)

    @property
    def model_name(self) -> str:
        return self._name

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Loading Whisper model %r (cpu int8)", self._name)
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(self._name, device="cpu", compute_type="int8")
            _cache[self._name] = self._model
            log.info("Whisper %r ready", self._name)
        except BaseException as e:
            log.exception("Whisper load failed %r: %s", self._name, e)
            raise

    def transcribe(self, audio_f32_16k: np.ndarray) -> str:
        self.ensure_loaded()
        log.debug("STT transcribe %.2fs", len(audio_f32_16k)/16000 if len(audio_f32_16k) else 0)
        try:
            segments, _info = self._model.transcribe(
                audio_f32_16k, language="en", beam_size=1
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            log.debug("STT -> %r", text)
            return text
        except BaseException as e:
            log.exception("STT transcribe failed: %s", e)
            raise
