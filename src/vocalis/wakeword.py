from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# --- SciPy 1.13+ workaround: applied lazily in ensure_loaded() ---
# Do NOT import scipy at module load time — that import itself can hang on some
# systems (entropy calc in _distribution_infrastructure). We patch on demand
# just before importing openwakeword/sklearn. See _apply_scipy_workaround().

def _apply_scipy_workaround() -> None:
    try:
        import scipy.stats._distribution_infrastructure as _di  # type: ignore
        orig = getattr(_di, "_generate_example", None)
        if orig is not None and not getattr(orig, "_vocalis_patched", False):
            def _safe_generate_example(self):  # type: ignore[no-untyped-def]
                try:
                    return orig(self)
                except Exception:
                    return "Example unavailable (scipy workaround)"
            _safe_generate_example._vocalis_patched = True  # type: ignore[attr-defined]
            _di._generate_example = _safe_generate_example  # type: ignore[attr-defined]
    except Exception:
        pass

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


def _download_onnx_file(url: str, target_dir: Path) -> None:
    """Download a single .onnx URL to target_dir (no tflite)."""
    try:
        from openwakeword.utils import download_file
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = url.split("/")[-1]
        if fname.endswith(".tflite"):
            url = url.replace(".tflite", ".onnx")
            fname = fname.replace(".tflite", ".onnx")
        dest = target_dir / fname
        if dest.exists():
            return
        download_file(url, str(target_dir))
    except Exception as e:
        log.debug("onnx download failed %s -> %s: %s", url, target_dir, e)
        raise


def _download_onnx_for_spec(spec: str, target_dir: Path) -> None:
    """Download onnx model for a spec name (hey_jarvis -> hey_jarvis_v0.1.onnx)."""
    try:
        import openwakeword
        base = (spec or "").strip()
        if not base or base == "custom":
            return
        # Normalize: strip path, take stem
        key = base
        if key not in openwakeword.MODELS:
            # try without _v0.1 suffix or with path
            stem = Path(base).stem
            if stem in openwakeword.MODELS:
                key = stem
            elif stem.replace("_v0.1", "") in openwakeword.MODELS:
                key = stem.replace("_v0.1", "")
            else:
                # substring match
                for k in openwakeword.MODELS:
                    if k in base:
                        key = k
                        break
                else:
                    return
        url = openwakeword.MODELS[key]["download_url"]
        _download_onnx_file(url, target_dir)
        # Also ensure feature models (onnx only) are present
        for fm in openwakeword.FEATURE_MODELS.values():
            furl = fm["download_url"]
            _download_onnx_file(furl, target_dir)
        # silero vad is already onnx
        for vm in openwakeword.VAD_MODELS.values():
            vurl = vm["download_url"]
            _download_onnx_file(vurl, target_dir)
    except Exception as e:
        log.debug("download for spec %r failed: %s", spec, e)
        raise


def _resolve_pretrained_path(name: str, prefer_onnx: bool = True) -> Path | None:  # noqa: ARG001
    """Resolve a pretrained name (e.g. hey_jarvis) to an explicit .onnx file in writable dir, downloading if needed.

    ONNX-only: we never require tflite. If a .tflite path is given we look for the .onnx sibling.
    """
    _ = prefer_onnx  # kept for compat, always onnx
    name = (name or "").strip()
    if not name:
        return None
    # If it's already a file path that exists, return it (prefer onnx sibling if tflite)
    p = Path(name).expanduser()
    if p.exists():
        if p.suffix.lower() == ".onnx":
            return p
        if p.suffix.lower() == ".tflite":
            onnx_sibling = p.with_suffix(".onnx")
            if onnx_sibling.exists():
                log.info("Wake word %r is tflite, using onnx sibling %s", name, onnx_sibling)
                return onnx_sibling
            # keep tflite path but caller will handle conversion; return it so we can warn
            return p
    # If name is "custom" without a path -> invalid
    if name == "custom":
        raise ValueError("Wake word is 'custom' but no .onnx file was selected — choose a file via Browse.")
    base = name
    # ONNX-only candidates
    candidates: list[str] = []
    if base in _KNOWN:
        candidates = [f"{base}_v0.1.onnx"]
    else:
        stem = Path(base).stem
        # bare stem + versioned fallback
        candidates = [f"{stem}.onnx", f"{stem}_v0.1.onnx"]
        candidates.append(base if base.endswith(".onnx") else f"{base}.onnx")
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
                return pp
    except Exception:
        pass
    # Not found — try to download specific model (onnx only)
    try:
        import openwakeword
        wdir.mkdir(parents=True, exist_ok=True)
        # Map base to download URL via openwakeword.MODELS
        model_key = base if base in openwakeword.MODELS else None
        if model_key is None:
            # Try stem match (e.g. hey_jarvis_v0.1 -> hey_jarvis)
            for k in openwakeword.MODELS:
                if k in base:
                    model_key = k
                    break
        if model_key and model_key in openwakeword.MODELS:
            url = openwakeword.MODELS[model_key]["download_url"]
            try:
                _download_onnx_file(url, wdir)
            except Exception as e:
                log.debug("download onnx for %r failed: %s", model_key, e)
        else:
            # Fallback: try generic download_models for this name (it will download tflite+onnx, we ignore tflite)
            try:
                from openwakeword.utils import download_models
                download_models(model_names=[base] if base in _KNOWN else [], target_directory=str(wdir))
            except Exception as e:
                log.debug("download_models(%r) -> %s", base, e)
    except Exception as e:
        log.warning("Failed to download wake word model %r to %s: %s", name, wdir, e)
    # Re-check after download
    for cand in candidates:
        pp = wdir / cand
        if pp.exists():
            return pp
    # Also check if tflite was downloaded and onnx sibling now exists due to download_models' double-download
    for cand in candidates:
        tflite_cand = cand.replace(".onnx", ".tflite")
        pp = wdir / tflite_cand
        if pp.exists():
            onnx_pp = pp.with_suffix(".onnx")
            if onnx_pp.exists():
                return onnx_pp
    return None


def _patch_openwakeword_resources(writable_dir: Path) -> None:
    """Monkey-patch openwakeword to use writable_dir for feature models (onnx-only)."""
    try:
        import openwakeword
        import openwakeword.utils as ow_utils
        orig_get_paths = openwakeword.get_pretrained_model_paths

        def patched_get_paths(framework: str = "onnx"):  # default onnx
            # Force onnx — tflite is not required. Map any tflite request to onnx.
            req_fw = "onnx" if framework not in ("onnx", "tflite") else framework
            # Always fetch onnx paths for feature models, even if caller asked tflite
            try:
                paths = orig_get_paths("onnx")
            except Exception:
                paths = orig_get_paths(req_fw)
            feature_names = ["melspectrogram", "embedding_model", "silero_vad"]
            patched = []
            for path in paths:
                name = Path(path).stem
                if any(fn in name for fn in feature_names):
                    base_name = name.split("_v")[0] if "_v" in name else name
                    found = False
                    for check_name in (name, base_name):
                        # onnx only
                        for ext in (".onnx",):
                            wp = writable_dir / f"{check_name}{ext}"
                            if wp.exists():
                                patched.append(str(wp))
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        # if writable doesn't have it, keep original onnx path (package resources)
                        patched.append(path)
                else:
                    # For wake-word models, also prefer writable onnx if present
                    stem = Path(path).stem
                    wp_onnx = writable_dir / f"{stem}.onnx"
                    if wp_onnx.exists():
                        patched.append(str(wp_onnx))
                    else:
                        patched.append(path)
            return patched

        openwakeword.get_pretrained_model_paths = patched_get_paths
        if hasattr(ow_utils, "get_pretrained_model_paths"):
            ow_utils.get_pretrained_model_paths = patched_get_paths
        log.debug("Patched openwakeword.get_pretrained_model_paths to use writable dir %s (onnx-only)", writable_dir)
    except Exception as e:
        log.debug("Failed to patch openwakeword resources: %s", e)


class WakeWord:
    """openwakeword wrapper with trigger threshold + cooldown policy. ONNX-only (no tflite required)."""

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
        # Apply scipy workaround before importing anything that pulls in scipy/sklearn
        _apply_scipy_workaround()
        # Ensure feature models are available BEFORE importing openwakeword.model
        wdir = _writable_model_dir()
        self._ensure_feature_models(wdir)

        # Fail fast if onnxruntime not installed (we are onnx-only since user requested no tflite)
        try:
            import onnxruntime  # noqa: F401
        except ImportError as e:
            raise RuntimeError("onnxruntime is required for wake word (ONNX). Install with: pip install onnxruntime") from e

        import openwakeword.model as om
        import openwakeword

        # Patch resource paths for ALL builds (feature models must be in writable dir, onnx-only)
        _patch_openwakeword_resources(wdir)

        # Resolve to explicit .onnx file if possible (writable dir, handles frozen)
        p = Path(self._spec).expanduser()
        is_file = False
        model_ref: str
        if p.exists():
            if p.suffix.lower() == ".onnx":
                model_ref = str(p)
                is_file = True
            elif p.suffix.lower() == ".tflite":
                # tflite given but we are onnx-only — look for onnx sibling
                onnx_sibling = p.with_suffix(".onnx")
                if onnx_sibling.exists():
                    model_ref = str(onnx_sibling)
                    is_file = True
                    log.info("Wake word %r is tflite, using onnx sibling %s", self._spec, model_ref)
                else:
                    # Keep original but will fail with clear message — try to download onnx version
                    model_ref = str(p)
                    is_file = True
                    log.warning("Wake word file %s is tflite but onnx is required — looking for onnx sibling %s", p, onnx_sibling)
            else:
                model_ref = str(p)
                is_file = True
        else:
            resolved = _resolve_pretrained_path(self._spec, prefer_onnx=True)
            if resolved is not None and resolved.exists():
                model_ref = str(resolved)
                is_file = True
                log.info("Resolved wake word %r -> %s", self._spec, model_ref)
                # If resolved is tflite (legacy cache), swap to onnx if exists
                if model_ref.endswith(".tflite"):
                    onnx_try = model_ref[:-7] + ".onnx"
                    if Path(onnx_try).exists():
                        model_ref = onnx_try
                        log.info("Resolved tflite %r -> onnx %s", resolved, model_ref)
            else:
                model_ref = self._spec
                log.warning("Wake word %r not resolved to file, will try name lookup (tried %s)", self._spec, resolved)

        ModelCls = getattr(om, "Model", None) or getattr(om, "WakewordModel", None)
        if ModelCls is None:
            raise ImportError("openwakeword.model has neither Model nor WakewordModel")

        tried: list[str] = []
        # ONNX-only variants. Try explicit onnx file with both kwarg names.
        variants: list[dict] = [
            {"wakeword_models": [model_ref], "inference_framework": "onnx"},
            {"models": [model_ref], "inference_framework": "onnx"},
        ]
        # If we have a file, also try without explicit framework (Model defaults, but we force onnx via patch)
        if is_file:
            variants.append({"wakeword_models": [model_ref]})
            variants.append({"models": [model_ref]})

        for kwargs in variants:
            # Force onnx if framework not set or is tflite
            if kwargs.get("inference_framework") not in (None, "onnx"):
                kwargs = dict(kwargs)
                kwargs["inference_framework"] = "onnx"
            # If model_ref is tflite, reject early with clear message (will be caught and retried)
            if any(str(v).endswith(".tflite") for v in kwargs.get("wakeword_models", []) + kwargs.get("models", [])):
                tried.append(f"{kwargs} -> tflite not supported (onnx-only build)")
                continue
            try:
                emb = Path(self._embeddings).expanduser() if self._embeddings else None
                if emb is not None and emb.exists():
                    for ek in ("user_embeddings", "custom_verifier_models"):
                        try:
                            k2 = dict(kwargs)
                            key = p.stem if is_file else Path(model_ref).stem
                            # key must match model's stem without version suffix? Use full stem.
                            k2[ek] = {key: str(emb)}
                            self._model = ModelCls(**k2)
                            log.info("Wake word loaded with %s (%s)", ek, kwargs)
                            return
                        except TypeError:
                            continue
                    self.error = f"could not load wake embeddings from {emb}; using model as-is"
                # Ensure onnx file exists before calling Model (gives clearer error than Model's ValueError)
                if is_file and not Path(model_ref).exists():
                    raise FileNotFoundError(f"Wake word model file not found: {model_ref}")
                # If not is_file, ensure resolved onnx exists; otherwise Model will try to find pretrained name
                self._model = ModelCls(**kwargs)
                log.info("Wake word loaded: %s framework=%s", model_ref, kwargs.get("inference_framework", "onnx"))
                return
            except Exception as e:
                msg = str(e)
                # If model file missing, try to download onnx version once and retry
                if any(s in msg for s in ("NoSuchFile", "Load model", "doesn't exist", "Failed", "Could not find pretrained", "not found")):
                    try:
                        # Try to download missing onnx
                        try:
                            _download_onnx_for_spec(self._spec, wdir)
                        except Exception:
                            pass
                        if not is_file:
                            rr = _resolve_pretrained_path(self._spec, prefer_onnx=True)
                            if rr is not None and rr.exists():
                                kwargs = dict(kwargs)
                                for k in ("wakeword_models", "models"):
                                    if k in kwargs:
                                        kwargs[k] = [str(rr)]
                                model_ref = str(rr)
                        # If is_file and tflite, check for onnx sibling after download
                        if is_file and model_ref.endswith(".tflite"):
                            onnx_try = model_ref[:-7] + ".onnx"
                            if Path(onnx_try).exists():
                                model_ref = onnx_try
                                for k in ("wakeword_models", "models"):
                                    if k in kwargs:
                                        kwargs[k] = [model_ref]
                        self._model = ModelCls(**kwargs)
                        log.info("Wake word loaded after download: %s", kwargs)
                        return
                    except Exception as de:
                        tried.append(f"{kwargs} -> download retry failed: {de} (orig {e})")
                        continue
                tried.append(f"{kwargs} -> {e}")
                continue
        raise RuntimeError(f"Could not init wake word model {self._spec!r} (resolved {model_ref!r}) with {tried} — onnxruntime {__import__('onnxruntime').__version__ if 'onnxruntime' in __import__('sys').modules else '?'}; ensure .onnx model exists in {wdir}")

    def _ensure_feature_models(self, wdir: Path) -> None:
        """Ensure onnx feature models (melspectrogram, embedding, VAD) are in writable dir (no tflite)."""
        feature_names = ["melspectrogram", "embedding_model", "silero_vad"]
        # Try onnx-only download via helper (avoids downloading tflite bloat)
        try:
            import openwakeword
            for key in ("melspectrogram", "embedding"):
                if key in openwakeword.FEATURE_MODELS:
                    url = openwakeword.FEATURE_MODELS[key]["download_url"]
                    try:
                        _download_onnx_file(url, wdir)
                    except Exception as e:
                        log.debug("Feature %s download failed: %s", key, e)
            for key in openwakeword.VAD_MODELS:
                url = openwakeword.VAD_MODELS[key]["download_url"]
                try:
                    _download_onnx_file(url, wdir)
                except Exception as e:
                    log.debug("VAD %s download failed: %s", key, e)
            log.debug("Feature models (onnx) download attempted in %s", wdir)
        except Exception as e:
            log.warning("Could not download feature models to %s: %s", wdir, e)
        # Fallback: copy missing .onnx from package resources
        try:
            import openwakeword
            pkg_dir = Path(openwakeword.__file__).parent / "resources" / "models"
            for name in feature_names:
                dst = wdir / f"{name}.onnx"
                if not dst.exists():
                    src = pkg_dir / f"{name}.onnx"
                    if src.exists():
                        import shutil
                        shutil.copy2(src, dst)
                        log.debug("Copied %s from package resources to %s", src.name, dst)
                    else:
                        # Legacy tflite source? Try to copy then convert? Just warn
                        src_tflite = pkg_dir / f"{name}.tflite"
                        if src_tflite.exists():
                            log.debug("Package has only %s.tflite, onnx missing — will download", name)
        except Exception as e2:
            log.debug("Fallback copy also failed: %s", e2)
        missing = [f"{n}.onnx" for n in feature_names if not (wdir / f"{n}.onnx").exists()]
        if missing:
            log.warning("Feature models still missing in %s: %s (onxruntime will fail without them)", wdir, missing)
        # Last resort: ensure package resources has onnx too (some code paths load from there directly)
        try:
            import openwakeword
            pkg_dir = Path(openwakeword.__file__).parent / "resources" / "models"
            pkg_dir.mkdir(parents=True, exist_ok=True)
            for name in feature_names:
                dst = pkg_dir / f"{name}.onnx"
                if not dst.exists():
                    src = wdir / f"{name}.onnx"
                    if src.exists():
                        import shutil
                        shutil.copy2(src, dst)
                        log.debug("Copied %s to package resources", dst.name)
        except Exception as e3:
            log.debug("Package resources copy failed: %s", e3)

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