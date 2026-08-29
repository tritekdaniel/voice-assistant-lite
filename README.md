# Vocalis — voice-first desktop assistant for local LLMs

Talk to any local LLM that exposes an **OpenAI-compatible** `chat/completions` endpoint (Ollama, LM Studio, Unsloth Desktop). No screen or keyboard needed after setup — wake word → listen → think → speak, with sentence-streamed TTS so the first words play before the full reply is done. Linux + Windows first.

![Vocalis](https://img.shields.io/badge/python-3.11%20only-blue) ![license](https://img.shields.io/badge/license-MIT-green)

## One-click install (standalone binary preferred)

**Windows — double-click:**
```
install.bat
```
This does a full one-click: creates `.venv` (Python 3.11), installs CPU-only `torch` (~300 MB not 2–3 GB CUDA) + deps, `pip install -e .`, **then builds `dist/Vocalis/Vocalis.exe` (standalone, no Python needed)** via PyInstaller, runs `Vocalis.exe --check`, and drops a Desktop + Start Menu shortcut to the **binary** (fallback to `.venv\Scripts\vocalis.exe` if the build fails). No admin needed. Add `-NoBinary` to skip the build, `-NoCheck` to skip the download.

**Linux / macOS — one command:**
```bash
bash scripts/install.sh          # builds dist/Vocalis/Vocalis by default
bash scripts/install.sh --no-binary   # venv only
```
After it finishes:
```bash
dist/Vocalis/Vocalis              # standalone binary (preferred, double-click)
.venv/bin/vocalis                 # venv fallback
dist/Vocalis/Vocalis --check      # or .venv/bin/vocalis --check
```

Both installers are idempotent — run again to repair/upgrade. The binary is ~1.5 GB (bundled torch) but runs without Python.

## Uninstall (one click — now also in the GUI)

**In the app:** `Settings → Danger Zone` at the bottom has two red buttons:
- **Uninstall — keep models** — deletes config (`%APPDATA%\vocalis` / `~/.config/vocalis`) but keeps `~700 MB` models so you can reinstall without re-downloading.
- **Uninstall — erase everything** — deletes config **and** models (`%LOCALAPPDATA%\vocalis` / `~/.local/share/vocalis`). The app then quits — delete the app folder / `dist/` manually if you used the binary.

**Or outside the app:**
- **Windows:** double-click `uninstall.bat` or `powershell -ExecutionPolicy Bypass -File scripts/uninstall.ps1 -Clean`
- **Linux/macOS:** `bash scripts/uninstall.sh --clean`

Removes `.venv`, `build/`, `dist/`, Desktop/Start Menu shortcuts. With `-Clean` / `--clean` also removes config+models (add `--keep-models` / `-KeepModels` to keep models).

```powershell
# Windows — keep models, remove config (same as GUI button)
.\scripts\uninstall.ps1 -Clean -KeepModels
# Windows — nuke everything
.\scripts\uninstall.ps1 -Clean
```
```bash
bash scripts/uninstall.sh --clean --keep-models
bash scripts/uninstall.sh --clean
```

## Use

- **GUI (default):** `vocalis` or `python -m vocalis` — tiny window + tray icon. The window shows the current state dot (idle/listening/thinking/speaking), the last thing you said, the streaming reply, and any error. Closing the window hides to tray; **Quit** in the tray menu exits.
- **Headless:** `vocalis --headless` — same voice loop, console logs only (useful for servers).
- **Self-test:** `vocalis --check` — probes audio devices, wake word, Whisper, Kokoro, and LLM connectivity; exits 0/1.
- **Barge-in:** say the wake word while the assistant is speaking to cut in.

## Configure — GUI for everything

`Settings` in the main window covers every option (no TOML editing required, but `config.toml` is still there if you want it):

- **LLM:** preset `Ollama` / `LM Studio` / `Unsloth Desktop` / `Custom` (fills base URL + key), base URL, model, API key, temperature, system prompt. `Test LLM` button does a live `check()`.
- **Voice:** Kokoro voice (`af_heart`, `am_adam`, …) + speed, `Speak test` plays a sentence on the chosen output device.
- **Wake word:** `hey_jarvis` (default), `hey_mycroft`, `alexa`, or a custom `.onnx` (Browse…) + optional embeddings file, threshold, cooldown. `Listen 8s` shows live score/trigger count.
- **STT:** Whisper model `tiny.en`/`base.en`/`small.en`/`medium.en` (English-only, CPU `int8`).
- **Audio:** input/output device pickers.
- **Timing:** VAD threshold (dBFS), trailing silence, max utterance, idle timeout.

Settings save to `config.toml` (`%APPDATA%\vocalis\config.toml` on Windows, `~/.config/vocalis/config.toml` on Linux) and restart the session live. All flags are also available as CLI overrides (`vocalis --base-url … --model … --wake-word …`).

## How it works

- **Audio:** `sounddevice` RawStream, 16 kHz in / 24 kHz out, 80 ms frames (1280 samples — the openwakeword window). Mic is fanned out to two queues (VAD + wake word). Playback is chunked in 250 ms pieces so an interrupt stops within a quarter second.
- **Wake word:** `openwakeword` `hey_jarvis` by default, custom `.onnx` supported, threshold 0.5 + 800 ms cooldown. Wake word in any state cancels the current utterance/reply/TTS and (re)enters listening.
- **VAD:** RMS dBFS on int16 frames, 4-frame preroll, endpoint on `silence_seconds` (0.8 s) or `max_utterance_seconds` (30 s), `idle_timeout_seconds` (30 s) returns to idle if nothing said.
- **STT → LLM → TTS:** `faster-whisper` → streaming `openai` client (`stream=True`) → `SentenceBuffer` (cut on `.!?`+space, hard cap 160 chars) → `Kokoro-82M` sentence-by-sentence synthesis; playback and synthesis overlap for low latency.
- **State machine:** `IDLE → LISTENING → THINKING → SPEAKING → LISTENING`, with `IDLE` entered on timeout or explicit cancel.

## Manual install (without the scripts)

Requires **Python 3.11** (`py -V:Astral\CPython3.11.15` on Windows, `python3.11` on Linux). Python 3.14 pulls CUDA wheels and is blocked by `requires-python = ">=3.10,<3.14"` for a reason.

```powershell
py -V:Astral\CPython3.11.15 -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\vocalis --check
.venv\Scripts\vocalis
```

## Build a standalone binary (manual)

```powershell
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller packaging/vocalis.spec  # → dist/Vocalis/Vocalis.exe
# or: powershell packaging/build.ps1  /  bash packaging/build.sh
```

## Logs — it closed before I could check? Never again

Every run is logged to a rotating file (5 MB × 3, flush on every error/crash):

- **Find it:** `vocalis --logs` prints the path, or **GUI → Settings → Logs → Open Logs Folder / View Log**, or main window footer **Open Logs / View**. Tray menu → **Open Logs Folder / View Log**.
- **Path:** `vocalis --logs` → e.g. `C:\Users\You\AppData\Local\vocalis\vocalis\logs\vocalis.log` (Windows) or `~/.local/share/vocalis/logs/vocalis.log` (Linux). Also `vocalis --log-level DEBUG --check` for verbose.
- **Never vanishes:** unhandled exceptions (main thread + background threads) are caught by `sys.excepthook`/`threading.excepthook`, logged with full traceback, flushed, and shown in a dialog that keeps the window open. The red error label in the main window now persists + tooltip shows the log path. Headless also logs to `logs/vocalis.log` + console.
- **View:** Settings → Logs → **View Log** shows last 300 lines in-app with **Copy** and **Open folder**. Main window footer has the same.

```bash
vocalis --logs                  # print log file path
vocalis --log-level DEBUG --check   # verbose check
vocalis --log-level DEBUG       # verbose GUI/headless
# then:
cat "$(vocalis --logs)" | tail -n 100
```

## Troubleshooting

- **Window closed instantly:** open `vocalis --logs`, run `vocalis --log-level DEBUG`, then **View Log**. Most crashes are audio device or missing model — the log shows the full traceback now, and the GUI stays open with the error.
- **No audio devices:** install system audio (Windows: check privacy → microphone; Linux: `sudo apt install portaudio19-dev` before `pip install`). Check `vocalis --check` and the log.
- **LLM fails:** is the server running? `Ollama` → `http://localhost:11434/v1`, `LM Studio` → `http://localhost:1234/v1`, `Unsloth Desktop` → `http://127.0.0.1:8080/v1`. Use `Test LLM` in Settings. Log shows `LLM check failed: Connection error` with traceback.
- **Wake word never fires:** lower threshold to 0.35–0.45 or run `Listen 8s` to see live scores. Log shows `Wake word detected` and scores.
- **First run slow:** first launch downloads ~700 MB (Whisper + Kokoro + wake word) via `bootstrap.ensure_models` — the GUI shows progress; you can Skip and it will lazy-load later. Log shows `Bootstrap start/done` and download progress.
