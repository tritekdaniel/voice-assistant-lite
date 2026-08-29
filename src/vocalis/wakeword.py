from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .logger import get_logger

log = get_logger(__name__)


class WakeWord:
    """openwakeword wrapper with trigger threshold + cooldown policy."""

    def __init__(self, model_spec: str, threshold: float = 0.5, cooldown_ms: int = 800,
                 embeddings_path: str = ""):
        self._spec = model_spec
        self.threshold = threshold
        self._cooldown_s = max(0.1, cooldown_ms / 1000.0)
        self._embeddings = embeddings_path
        self._model = None
        self._last_trigger = -1e9
        self.error: str | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Loading wake word model %r (threshold %.2f, cooldown %dms)", self._spec, self.threshold, int(self._cooldown_s*1000))
        import openwakeword.model as om

        p = Path(self._spec).expanduser()
        is_file = p.suffix.lower() == ".onnx" and p.exists()
        model_ref = str(p) if is_file else self._spec

        # New openwakeword (0.6+) uses Model(wakeword_models=[...], inference_framework="onnx")
        # Old versions used WakewordModel(models=[...]). Support both.
        ModelCls = getattr(om, "Model", None) or getattr(om, "WakewordModel", None)
        if ModelCls is None:
            raise ImportError("openwakeword.model has neither Model nor WakewordModel")

        # Try new API first, auto-downloading if needed
        tried = []
        for kwargs in [
            {"wakeword_models": [model_ref], "inference_framework": "onnx"},
            {"models": [model_ref], "inference_framework": "onnx"},
        ]:
            try:
                emb = Path(self._embeddings).expanduser() if self._embeddings else None
                if emb is not None and emb.exists():
                    for ek in ("user_embeddings", "custom_verifier_models"):
                        try:
                            k2 = dict(kwargs)
                            key = p.stem if is_file else model_ref
                            k2[ek] = {key: str(emb)}
                            self._model = ModelCls(**k2)
                            return
                        except TypeError:
                            continue
                    self.error = f"could not load wake embeddings from {emb}; using model as-is"
                try:
                    self._model = ModelCls(**kwargs)
                    return
                except Exception as e:
                    # If model file missing, try downloading then retry once
                    msg = str(e)
                    if "NoSuchFile" in msg or "Load model" in msg or "doesn't exist" in msg or "Failed" in msg:
                        try:
                            from openwakeword.utils import download_models
                            # For pretrained names, download needs base name like hey_jarvis
                            # For custom .onnx files, download all (to get embedding/melspectrogram)
                            if is_file:
                                download_models()
                            else:
                                # try specific model name, fallback to all
                                try:
                                    download_models(model_names=[model_ref])
                                except Exception:
                                    download_models()
                            self._model = ModelCls(**kwargs)
                            return
                        except Exception as de:
                            tried.append(f"{kwargs} -> download retry failed: {de} (orig {e})")
                            raise
                    raise
            except TypeError as e:
                tried.append(f"{kwargs} -> {e}")
                continue
            except Exception as e:
                # Surface the last tried error after loop instead of immediate raise
                # to allow second kwargs variant to be tried
                tried.append(f"{kwargs} -> {e}")
                continue
        raise RuntimeError(f"Could not init wake word model with {tried}")

    def score(self, chunk_int16: np.ndarray) -> float:
        self.ensure_loaded()
        preds = self._model.predict(chunk_int16.astype(np.int16, copy=False))
        if isinstance(preds, dict):
            preds = [preds]
        best = 0.0
        for d in preds:
            if not d:
                continue
            best = max(best, max(float(v) for v in d.values()))
        return best

    def trigger(self, chunk_int16: np.ndarray) -> bool:
        return self.should_trigger(self.score(chunk_int16))

    def should_trigger(self, score: float) -> bool:
        now = time.monotonic()
        if score >= self.threshold and (now - self._last_trigger) >= self._cooldown_s:
            self._last_trigger = now
            return True
        return False

    def reset(self) -> None:
        self.error = None
