from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .logger import get_logger

log = get_logger(__name__)

# Known pretrained names -> filename stem (openwakeword.MODELS uses <name>_v0.1.tflite)
_KNOWN = {"hey_jarvis", "hey_mycroft", "alexa", "hey_rhasspy", "timer", "weather"}


def _writable_model_dir() -> Path:
    """Writable dir for openwakeword models — works in frozen (PyInstaller) on Linux."""
    # Use vocalis data_dir/models/openwakeword so downloads persist and are writable
    try:
        from .config import models_dir
        d = models_dir() / "openwakeword"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        # Fallback to package resources if config not available
        import openwakeword
        return Path(openwakeword.__file__).parent / "resources" / "models"


def _resolve_pretrained_path(name: str, prefer_onnx: bool = True) -> Path | None:
    """Resolve a pretrained name (e.g. hey_jarvis) to an explicit file in writable dir, downloading if needed."""
    name = (name or "").strip()
    if not name:
        return None
    # If it's already a file path that exists, return it
    p = Path(name).expanduser()
    if p.exists() and p.suffix.lower() in (".onnx", ".tflite"):
        return p
    # If name is "custom" without a path -> invalid
    if name == "custom":
        raise ValueError("Wake word is 'custom' but no .onnx/.tflite file was selected — choose a file via Browse.")
    # Map bare name to actual model file; openwakeword uses <name>_v0.1.*
    # Known names have a suffix, unknown may already be the full stem
    base = name
    # If user passed hey_jarvis_v0.1, keep it; otherwise try bare name
    candidates: list[str] = []
    if base in _KNOWN:
        candidates = [f"{base}_v0.1.onnx", f"{base}_v0.1.tflite"] if prefer_onnx else [f"{base}_v0.1.tflite", f"{base}_v0.1.onnx"]
    else:
        # Try as-is with both extensions
        stem = Path(base).stem
        candidates = [f"{stem}.onnx", f"{stem}.tflite"] if prefer_onnx else [f"{stem}.tflite", f"{stem}.onnx"]
        # Also try bare name as key substring
        candidates.append(base)
    wdir = _writable_model_dir()
    # Check writable dir first
    for cand in candidates:
        pp = wdir / cand
        if pp.exists():
            return pp
    # Check default package dir (dev case)
    try:
        import openwakeword
        pkg_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        for cand in candidates:
            pp = pkg_dir / cand
            if pp.exists():
                # Copy to writable for consistency? Just return it
                return pp
    except Exception:
        pass
    # Not found — try to download to writable dir
    try:
        from openwakeword.utils import download_models
        # Try specific name first, then all (also pulls melspectrogram/embedding)
        tried_download = False
        for dl_name in ([base] if base in _KNOWN else [base, ""]):
            try:
                if dl_name:
                    download_models(model_names=[dl_name], target_directory=str(wdir))
                else:
                    download_models(target_directory=str(wdir))
                tried_download = True
                break
            except Exception as e:
                log.debug("download_models(%r) -> %s", dl_name, e)
                continue
        if not tried_download:
            # Last resort: download all to default loc then copy? Just try default download
            download_models(target_directory=str(wdir))
    except Exception as e:
        log.warning("Failed to download wake word model %r to %s: %s", name, wdir, e)
    # Re-check after download
    for cand in candidates:
        pp = wdir / cand
        if pp.exists():
            return pp
    # Fallback: let openwakeword try to resolve by name (may use tflite default)
    return None


class WakeWord:
    """openwakeword wrapper with trigger threshold + cooldown policy. Handles Linux frozen + onnx/tflite."""

    def __init__(self, model_spec: str, threshold: float = 0.5, cooldown_ms: int = 800,
                 embeddings_path: str = ""):
        self._spec = (model_spec or "hey_jarvis").strip()
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

        # Resolve to explicit file if possible (writable dir, handles frozen)
        p = Path(self._spec).expanduser()
        is_file = p.exists() and p.suffix.lower() in (".onnx", ".tflite")
        if is_file:
            model_ref = str(p)
            # Ensure melspectrogram/embedding are available in writable dir as well (needed for custom)
            try:
                wdir = _writable_model_dir()
                # download_models ensures feature models exist
                from openwakeword.utils import download_models
                # If feature models missing, download
                if not (wdir / "melspectrogram.tflite").exists() or not (wdir / "embedding_model.tflite").exists():
                    download_models(target_directory=str(wdir))
            except Exception:
                pass
        else:
            # Pretrained name — resolve to file in writable dir
            resolved = _resolve_pretrained_path(self._spec, prefer_onnx=True)
            if resolved is not None and resolved.exists():
                model_ref = str(resolved)
                is_file = True
                log.info("Resolved wake word %r -> %s", self._spec, model_ref)
            else:
                model_ref = self._spec
                log.warning("Wake word %r not resolved to file, will let openwakeword search (tried %s)", self._spec, resolved)

        ModelCls = getattr(om, "Model", None) or getattr(om, "WakewordModel", None)
        if ModelCls is None:
            raise ImportError("openwakeword.model has neither Model nor WakewordModel")

        # Try onnx first, then tflite — Linux standalone often lacks tflite runtime issues and vice versa
        tried: list[str] = []
        # Build kwargs variants: prefer explicit file paths + onnx, then tflite, then name-only
        variants: list[dict] = []
        for fw in ("onnx", "tflite"):
            for key in ("wakeword_models", "models"):
                variants.append({key: [model_ref], "inference_framework": fw})
        # If we have a file, also try without explicit framework (let Model pick)
        if is_file:
            variants.append({"wakeword_models": [model_ref]})
            variants.append({"models": [model_ref]})

        for kwargs in variants:
            try:
                emb = Path(self._embeddings).expanduser() if self._embeddings else None
                if emb is not None and emb.exists():
                    for ek in ("user_embeddings", "custom_verifier_models"):
                        try:
                            k2 = dict(kwargs)
                            key = p.stem if is_file else model_ref
                            k2[ek] = {key: str(emb)}
                            self._model = ModelCls(**k2)
                            log.info("Wake word loaded with %s (%s)", ek, kwargs)
                            return
                        except TypeError:
                            continue
                    self.error = f"could not load wake embeddings from {emb}; using model as-is"
                self._model = ModelCls(**kwargs)
                log.info("Wake word loaded: %s framework=%s", model_ref, kwargs.get("inference_framework", "default"))
                return
            except Exception as e:
                msg = str(e)
                # If model file missing, ensure download to writable dir and retry once per variant
                if any(s in msg for s in ("NoSuchFile", "Load model", "doesn't exist", "Failed", "Could not find pretrained")):
                    try:
                        wdir = _writable_model_dir()
                        from openwakeword.utils import download_models
                        if is_file:
                            download_models(target_directory=str(wdir))
                        else:
                            try:
                                download_models(model_names=[model_ref], target_directory=str(wdir))
                            except Exception:
                                download_models(target_directory=str(wdir))
                        # Re-resolve after download
                        if not is_file:
                            rr = _resolve_pretrained_path(self._spec, prefer_onnx=(kwargs.get("inference_framework") == "onnx"))
                            if rr is not None and rr.exists():
                                kwargs = dict(kwargs)
                                # patch the path in kwargs
                                for k in ("wakeword_models", "models"):
                                    if k in kwargs:
                                        kwargs[k] = [str(rr)]
                        self._model = ModelCls(**kwargs)
                        log.info("Wake word loaded after download: %s", kwargs)
                        return
                    except Exception as de:
                        tried.append(f"{kwargs} -> download retry failed: {de} (orig {e})")
                        continue
                tried.append(f"{kwargs} -> {e}")
                continue
        raise RuntimeError(f"Could not init wake word model {self._spec!r} (resolved {model_ref!r}) with {tried}")

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
