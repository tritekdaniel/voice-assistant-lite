from __future__ import annotations

import dataclasses
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import platformdirs
import tomllib
from tomli_w import dump as toml_dump

APP_NAME = "vocalis"

DEFAULT_SYSTEM_PROMPT = (
    "You are Vocalis, a concise voice assistant. Be brief, direct, and conversational. "
    "Keep replies short — 1-3 sentences, no lists, no markdown, no preamble. "
    "Speak naturally as if in a voice chat. Never mention internal implementation details, "
    "model names, or sound file names — if a timer rings, just say the timer is done. "
    "You can set timers: when the user asks for a timer, call set_timer with seconds (e.g., 60 for 1 min). "
    "After calling set_timer, say 'Timer set' and nothing more about the sound. "
    "A sound will loop when the timer fires; the user can say 'stop timer' to cancel it. "
    "If asked what sound plays, say 'a chime' and do not name files."
)


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME))


def data_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME))


def models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_file() -> Path:
    return logs_dir() / "vocalis.log"


def apply_model_env() -> None:
    """Point Hugging Face caches at the app data dir. Call before importing any
    library that talks to huggingface_hub."""
    hf = models_dir() / "hf"
    os.environ["HF_HOME"] = str(hf)
    os.environ["HF_HUB_CACHE"] = str(hf / "hub")


@dataclass
class Config:
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"
    llm_api_key: str = ""  # optional — leave empty for local Ollama/LM Studio/Unsloth
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_history_messages: int = 40
    forget_history: bool = True  # if true, forget previous user/assistant turns after each reply
    preserve_history: bool = False  # if true, keep history and compact after 30 messages
    compact_after: int = 30  # when preserve_history is true, compact when turns exceed this

    whisper_model: str = "base.en"

    # TTS engine: "kokoro" or "piper"
    tts_engine: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    # Piper-specific (onnx + json pair). piper_model is path to .onnx; json is derived or explicit.
    piper_model: str = ""
    piper_config: str = ""  # optional explicit .onnx.json path; auto-derived if empty
    piper_speaker: int = 0  # speaker id for multi-speaker models
    piper_length_scale: float = 1.0  # 1.0 normal, <1 faster, >1 slower
    piper_noise_scale: float = 0.667
    piper_noise_w: float = 0.8

    wake_word: str = "hey_jarvis"
    wakeword_embeddings: str = ""
    wakeword_threshold: float = 0.5
    wakeword_cooldown_ms: int = 800

    vad_rms_dbfs: float = -42.0
    vad_silence_seconds: float = 0.8
    max_utterance_seconds: float = 30.0
    idle_timeout_seconds: float = 30.0
    # if True, after speaking stay in LISTENING for follow-up without wake word; if False, go back to IDLE (requires wake word)
    continuous_listening: bool = False
    listen_grace_seconds: float = 10.0

    input_device: int | None = None
    output_device: int | None = None


def config_path() -> Path:
    return config_dir() / "config.toml"


def _norm_model_id(s: str) -> str:
    return (s or "").strip().lstrip("/\\").strip()


def load_config() -> Config:
    p = config_path()
    if not p.exists():
        return Config()
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return Config()
    known = {f.name for f in dataclasses.fields(Config)}
    # normalize model id that may have been saved with leading slash from old provider paths
    if "llm_model" in raw and isinstance(raw["llm_model"], str):
        raw["llm_model"] = _norm_model_id(raw["llm_model"])
    if "whisper_model" in raw and isinstance(raw["whisper_model"], str):
        raw["whisper_model"] = raw["whisper_model"].strip()
    if "tts_voice" in raw and isinstance(raw["tts_voice"], str):
        v = raw["tts_voice"].strip()
        if " —" in v:
            v = v.split(" —")[0].strip()
        raw["tts_voice"] = v
    # treat old placeholder tokens as empty (now optional)
    if "llm_api_key" in raw and isinstance(raw["llm_api_key"], str):
        if raw["llm_api_key"].strip() in ("vocalis-local", "ollama", "lm-studio", "unsloth", "none"):
            raw["llm_api_key"] = ""
    # migrate old verbose system prompt to new concise one
    _old_prompt = (
        "You are a helpful voice assistant. Keep answers concise and conversational, in plain "
        "spoken language with no lists or markdown. Address the user directly."
    )
    _old_with_lithium = "The Lithium sound will loop 5 times when it rings"
    if "system_prompt" in raw and isinstance(raw["system_prompt"], str):
        sp = raw["system_prompt"].strip()
        if sp == _old_prompt.strip() or _old_with_lithium in sp:
            raw["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.llm_model = _norm_model_id(cfg.llm_model)
    if cfg.llm_api_key.strip() in ("vocalis-local", "ollama", "lm-studio", "unsloth", "none"):
        cfg.llm_api_key = ""
    if cfg.system_prompt.strip() == _old_prompt.strip() or _old_with_lithium in cfg.system_prompt:
        cfg.system_prompt = DEFAULT_SYSTEM_PROMPT
    # preserve_history is new — if user had forget_history=False, keep preserved history by default
    if "preserve_history" not in raw and cfg.forget_history is False:
        cfg.preserve_history = True
    # sanity: compact_after
    if cfg.compact_after < 10:
        cfg.compact_after = 30
    if cfg.compact_after > cfg.max_history_messages:
        cfg.compact_after = max(10, cfg.max_history_messages - 10)
    # One-time migration: old bug caused continuous listening to leave mic hot after speaking.
    # If the file still has True from that era and the version hasn't opted-in explicitly,
    # force back to False once. User can re-enable via Settings.
    if "continuous_listening" in raw and raw["continuous_listening"] is True:
        # Keep user's explicit True only if they've saved after 0.1.0 — for now, respect file but log
        # We previously forced False; now we respect the file. Only log migration note.
        log = __import__("logging").getLogger("vocalis")
        log.info("Config has continuous_listening=True — respecting user choice (say 'hey jarvis' not needed if true)")
    # TTS engine sanity + piper defaults
    if cfg.tts_engine not in ("kokoro", "piper"):
        cfg.tts_engine = "kokoro"
    cfg.tts_engine = cfg.tts_engine.strip().lower()
    # normalize piper paths
    if cfg.piper_model:
        cfg.piper_model = cfg.piper_model.strip()
    if cfg.piper_config:
        cfg.piper_config = cfg.piper_config.strip()
    try:
        cfg.piper_speaker = int(cfg.piper_speaker)
    except Exception:
        cfg.piper_speaker = 0
    try:
        cfg.piper_length_scale = float(cfg.piper_length_scale)
        if not 0.3 <= cfg.piper_length_scale <= 3.0:
            cfg.piper_length_scale = 1.0
    except Exception:
        cfg.piper_length_scale = 1.0
    return cfg


def save_config(cfg: Config) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # tomli_w writes bytes and can't handle None, so filter Nones
    data = {k: v for k, v in asdict(cfg).items() if v is not None}
    with open(p, "wb") as f:
        toml_dump(data, f)
