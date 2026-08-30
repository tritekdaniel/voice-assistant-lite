from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .logger import get_logger

log = get_logger(__name__)

_cache: dict[str, object] = {}


def _resample_to_out_rate(data: np.ndarray, sr_in: int, sr_out: int = 24000) -> np.ndarray:
    if sr_in == sr_out or len(data) == 0:
        return data.astype(np.float32, copy=False)
    ratio = sr_out / sr_in
    new_len = int(len(data) * ratio)
    if new_len == 0:
        return np.zeros(0, dtype=np.float32)
    old_idx = np.arange(len(data))
    new_idx = np.linspace(0, len(data) - 1, new_len)
    return np.interp(new_idx, old_idx, data).astype(np.float32)


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


# ---------- Piper ----------

_PIPER_CACHE: dict[str, object] = {}


class PiperSpeaker:
    """Piper TTS (onnx + json). Supports any Piper voice with custom .onnx.

    Sample rate is taken from the voice config (typically 22050) and resampled to 24 kHz
    so it can share the same Playback (OUT_RATE=24000).
    """

    def __init__(
        self,
        model_path: str,
        config_path: str = "",
        speaker_id: int = 0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        self.model_path = (model_path or "").strip()
        self.config_path = (config_path or "").strip()
        self.speaker_id = int(speaker_id)
        self.length_scale = float(length_scale)
        self.noise_scale = float(noise_scale)
        self.noise_w = float(noise_w)
        self._voice = None
        self._sample_rate = 22050

    def reconfigure(
        self,
        model_path: str = "",
        speaker_id: int | None = None,
        length_scale: float | None = None,
    ) -> None:
        if model_path:
            self.model_path = model_path.strip()
        if speaker_id is not None:
            self.speaker_id = int(speaker_id)
        if length_scale is not None:
            self.length_scale = float(length_scale)

    def _is_valid_piper_config(self, p: Path) -> bool:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # valid piper configs have num_symbols + audio.sample_rate + phoneme_id_map
            return isinstance(data, dict) and "num_symbols" in data and "audio" in data
        except Exception:
            return False

    def _resolve_paths(self) -> tuple[Path, Path | None]:
        if not self.model_path:
            raise ValueError("Piper model path is not set — choose a .onnx file in Settings")
        m = Path(self.model_path).expanduser()
        if not m.exists():
            raise FileNotFoundError(f"Piper model not found: {m}")
        if m.suffix.lower() != ".onnx":
            raise ValueError(f"Piper model must be a .onnx file, got: {m}")
        c: Path | None = None
        # sibling candidates (always valid piper configs if they exist)
        cj = Path(str(m) + ".json")
        alt = m.with_suffix(".json")
        sibling: Path | None = cj if cj.exists() else (alt if alt.exists() else None)
        if self.config_path:
            cand = Path(self.config_path).expanduser()
            if not cand.exists():
                raise FileNotFoundError(f"Piper config not found: {cand}")
            if self._is_valid_piper_config(cand):
                c = cand
            else:
                # Provided config is not a valid piper config (e.g. glados.json without num_symbols)
                # Fall back to sibling if available, otherwise keep cand and let load fail with clear message
                if sibling is not None and self._is_valid_piper_config(sibling):
                    log.warning("Piper config %s is not a valid piper voice config (missing num_symbols) — using sibling %s", cand, sibling)
                    c = sibling
                else:
                    log.warning("Piper config %s looks invalid (missing num_symbols) — will try it anyway", cand)
                    c = cand
        else:
            # auto .onnx.json
            if sibling is not None:
                c = sibling
        return m, c

    def ensure_loaded(self) -> None:
        m, c = self._resolve_paths()
        key = f"{m}|{c}|{self.speaker_id}"
        cached = _PIPER_CACHE.get(key)
        if cached is not None and self._voice is not None:
            return
        if cached is not None:
            self._voice = cached
            try:
                self._sample_rate = int(getattr(getattr(cached, "config", None), "sample_rate", 22050))
            except Exception:
                pass
            return
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise RuntimeError("piper-tts is not installed. Run: pip install piper-tts") from e

        log.info("Loading Piper voice model=%s config=%s speaker=%s", m, c, self.speaker_id)
        last_exc: BaseException | None = None
        # Try primary config first, then sibling fallback if KeyError num_symbols or similar
        candidates: list[Path | None] = [c]
        # add sibling as fallback if different from primary
        try:
            sib = Path(str(m) + ".json")
            if sib.exists() and sib != c:
                candidates.append(sib)
            alt2 = m.with_suffix(".json")
            if alt2.exists() and alt2 not in candidates:
                candidates.append(alt2)
        except Exception:
            pass
        for cand in candidates:
            try:
                self._voice = PiperVoice.load(str(m), config_path=str(cand) if cand else None, use_cuda=False)
                self._sample_rate = int(getattr(getattr(self._voice, "config", None), "sample_rate", 22050))
                if cand != c:
                    log.info("Piper voice ready with fallback config %s sr=%s speakers=%s", cand, self._sample_rate, getattr(self._voice.config, "num_speakers", "?"))
                else:
                    log.info("Piper voice ready sr=%s speakers=%s", self._sample_rate, getattr(self._voice.config, "num_speakers", "?"))
                _PIPER_CACHE[key] = self._voice
                return
            except (KeyError, ValueError, FileNotFoundError) as e:
                last_exc = e
                # Config mismatch like 'num_symbols' — try next candidate
                if cand is not None and "num_symbols" in str(e) or isinstance(e, KeyError):
                    log.warning("Piper load with config %s failed (%s) — trying fallback", cand, e)
                    continue
                # For other errors, don't retry
                log.exception("Piper load failed %s: %s", m, e)
                raise
            except BaseException as e:
                last_exc = e
                log.exception("Piper load failed %s: %s", m, e)
                raise
        # All candidates failed — raise last
        if last_exc is not None:
            raise RuntimeError(f"Piper voice {m.name} failed to load with config {c} (tried {candidates}): {last_exc}. Ensure the .onnx and its matching .onnx.json are a valid piper voice pair (download from rhasspy/piper-voices).") from last_exc
        raise RuntimeError(f"Piper voice {m} failed to load — no valid config found")

    def synthesize(self, text: str) -> np.ndarray:
        self.ensure_loaded()
        assert self._voice is not None
        log.debug("Piper synthesize model=%s speaker=%s len=%.2f text=%r", Path(self.model_path).name, self.speaker_id, self.length_scale, text[:80])
        try:
            from piper.config import SynthesisConfig
        except ImportError:
            SynthesisConfig = None  # type: ignore

        # Build synthesis config
        syn_cfg = None
        if SynthesisConfig is not None:
            try:
                syn_cfg = SynthesisConfig(
                    speaker_id=self.speaker_id if getattr(self._voice.config, "num_speakers", 1) > 1 else None,
                    length_scale=self.length_scale if self.length_scale != 1.0 else None,
                    noise_scale=self.noise_scale if self.noise_scale != 0.667 else None,
                    noise_w_scale=self.noise_w if self.noise_w != 0.8 else None,
                )
            except Exception:
                syn_cfg = None

        parts: list[np.ndarray] = []
        sr = self._sample_rate
        try:
            gen = self._voice.synthesize(text, syn_config=syn_cfg)  # type: ignore[arg-type]
            for chunk in gen:
                # chunk is AudioChunk
                arr = getattr(chunk, "audio_float_array", None)
                if arr is None:
                    # fallback: try int16
                    arr = getattr(chunk, "audio_int16_array", None)
                    if arr is not None:
                        arr = np.asarray(arr, dtype=np.float32) / 32768.0
                    else:
                        arr = np.asarray(chunk, dtype=np.float32)
                else:
                    arr = np.asarray(arr, dtype=np.float32)
                if len(arr) == 0:
                    continue
                # Piper chunk sr -> OUT_RATE
                if sr != 24000:
                    arr = _resample_to_out_rate(arr, sr, 24000)
                parts.append(np.ascontiguousarray(arr, dtype=np.float32))
        except BaseException as e:
            log.exception("Piper synthesize failed %r: %s", text[:80], e)
            raise
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        log.debug("Piper produced %d samples (%.2fs) for %r", len(out), len(out)/24000 if len(out) else 0, text[:40])
        return out


def create_speaker(cfg) -> Speaker | PiperSpeaker:
    """Factory: return Kokoro Speaker or PiperSpeaker based on cfg.tts_engine.

    On Linux, Piper is venv-only and may not be installed; if piper model missing or
    piper-tts not installed, fall back to Kokoro so install/check never crashes.
    """
    engine = str(getattr(cfg, "tts_engine", "kokoro")).lower()
    if engine == "piper":
        model = (getattr(cfg, "piper_model", "") or "").strip()
        # If no model configured, fall back to kokoro (don't crash)
        if not model:
            log.warning("TTS engine piper selected but piper_model is empty — falling back to kokoro")
            engine = "kokoro"
        else:
            # Check piper is importable and model exists before returning PiperSpeaker
            try:
                import importlib.util
                if importlib.util.find_spec("piper") is None:
                    raise ImportError("piper not installed")
                from pathlib import Path
                if not Path(model).expanduser().exists():
                    raise FileNotFoundError(f"piper model not found: {model}")
            except Exception as e:
                log.warning("Piper not available (%s) — falling back to kokoro", e)
                engine = "kokoro"
    if engine == "piper":
        return PiperSpeaker(
            model_path=getattr(cfg, "piper_model", ""),
            config_path=getattr(cfg, "piper_config", ""),
            speaker_id=int(getattr(cfg, "piper_speaker", 0) or 0),
            length_scale=float(getattr(cfg, "piper_length_scale", 1.0) or 1.0),
            noise_scale=float(getattr(cfg, "piper_noise_scale", 0.667) or 0.667),
            noise_w=float(getattr(cfg, "piper_noise_w", 0.8) or 0.8),
        )
    # default kokoro
    return Speaker(
        voice=getattr(cfg, "tts_voice", "af_heart"),
        speed=float(getattr(cfg, "tts_speed", 1.0) or 1.0),
    )
