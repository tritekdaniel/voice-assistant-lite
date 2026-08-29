from __future__ import annotations

import numpy as np

from .logger import get_logger

log = get_logger(__name__)

_cache: dict[str, object] = {}


class Speaker:
    """Kokoro-82M text-to-speech (v1.0 weights), 24 kHz mono float32 out."""

    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._pipe = _cache.get("kokoro")

    def reconfigure(self, voice: str, speed: float) -> None:
        self.voice = voice
        self.speed = speed

    def _lang_for_voice(self, voice: str) -> str:
        # Kokoro KPipeline lang_code: 'a' = American English, 'b' = British, 'j' = Japanese, etc.
        # Our preset voices are all 'a' (af_*, am_*) but derive from prefix for future.
        v = (voice or "af_heart").lower()
        if v.startswith("b"):
            return "b"
        if v.startswith("j"):
            return "j"
        if v.startswith("z"):
            return "z"
        return "a"

    def ensure_loaded(self) -> None:
        if self._pipe is not None:
            lang = self._lang_for_voice(self.voice)
            cached_lang = _cache.get("kokoro_lang")
            if cached_lang == lang:
                return
        from kokoro import KPipeline

        lang = self._lang_for_voice(self.voice)
        log.info("Loading Kokoro pipeline lang=%r voice=%r", lang, self.voice)
        try:
            self._pipe = KPipeline(lang_code=lang)
            log.info("Kokoro ready lang=%r", lang)
        except BaseException as e:
            log.exception("Kokoro load failed lang=%r: %s", lang, e)
            raise
        _cache["kokoro"] = self._pipe
        _cache["kokoro_lang"] = lang

    def synthesize(self, text: str) -> np.ndarray:
        self.ensure_loaded()
        log.debug("TTS synthesize voice=%s speed=%.2f text=%r", self.voice, self.speed, text[:80])
        parts: list[np.ndarray] = []
        try:
            gen = self._pipe(text, voice=self.voice, speed=self.speed)
        except TypeError:
            gen = self._pipe(text, voice=self.voice)
        try:
            for chunk in gen:
                if hasattr(chunk, "audio"):
                    audio = chunk.audio
                elif isinstance(chunk, (tuple, list)):
                    audio = chunk[0]
                else:
                    audio = chunk
                try:
                    a = audio.detach().cpu().numpy().ravel()  # type: ignore[union-attr]
                except AttributeError:
                    a = np.asarray(audio, dtype=np.float32).ravel()
                if len(a):
                    parts.append(np.ascontiguousarray(a, dtype=np.float32))
        except BaseException as e:
            log.exception("TTS synthesize failed %r: %s", text[:80], e)
            raise
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        log.debug("TTS produced %d samples (%.2fs) for %r", len(out), len(out)/24000 if len(out) else 0, text[:40])
        return out
