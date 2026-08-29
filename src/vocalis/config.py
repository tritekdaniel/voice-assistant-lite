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
    "You are a concise voice assistant. Be brief, direct, and conversational. "
    "Keep replies short — 1-3 sentences, no lists, no markdown, no preamble. "
    "Speak naturally as if in a voice chat. "
    "You can set timers: when the user asks for a timer, call set_timer with seconds (e.g., 60 for 1 min). "
    "Say 'Timer set' after. The Lithium sound will loop 5 times when it rings; user can say 'stop timer' to cancel."
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

    whisper_model: str = "base.en"

    tts_voice: str = "af_heart"
    tts_speed: float = 1.0

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
    if "system_prompt" in raw and isinstance(raw["system_prompt"], str):
        if raw["system_prompt"].strip() == _old_prompt.strip():
            raw["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.llm_model = _norm_model_id(cfg.llm_model)
    if cfg.llm_api_key.strip() in ("vocalis-local", "ollama", "lm-studio", "unsloth", "none"):
        cfg.llm_api_key = ""
    if cfg.system_prompt.strip() == _old_prompt.strip():
        cfg.system_prompt = DEFAULT_SYSTEM_PROMPT
    # fix previous bug where continuous listening left mic activated after speaking
    if "continuous_listening" in raw and raw["continuous_listening"] is True:
        # user reported "goes straight into activated mode" — force back to idle-wait
        cfg.continuous_listening = False
    return cfg


def save_config(cfg: Config) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # tomli_w writes bytes and can't handle None, so filter Nones
    data = {k: v for k, v in asdict(cfg).items() if v is not None}
    with open(p, "wb") as f:
        toml_dump(data, f)
