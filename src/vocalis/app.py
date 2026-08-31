from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .config import Config, logs_dir, models_dir, save_config
from .logger import get_logger, log_path, reveal_in_file_manager
from .session import Listener, State

log = get_logger(__name__)

STATE_LABELS = {
    State.IDLE: "Waiting — say \u201cHey Jarvis\u201d",
    State.LISTENING: "Listening\u2026",
    State.THINKING: "Thinking\u2026",
    State.SPEAKING: "Speaking\u2026",
}
STATE_COLORS = {
    State.IDLE: "#64748b",
    State.LISTENING: "#22c55e",
    State.THINKING: "#f59e0b",
    State.SPEAKING: "#38bdf8",
}
PRESETS: dict[str, tuple[str, str]] = {
    "Ollama": ("http://localhost:11434/v1", ""),
    "LM Studio": ("http://localhost:1234/v1", ""),
    "Unsloth Desktop": ("http://127.0.0.1:8080/v1", ""),
    "Custom": ("", ""),
}
WAKE_CHOICES = ["hey_jarvis", "hey_mycroft", "alexa", "custom"]
# Kokoro voices — grouped by accent/gender. All are 82M single model; speed param controls pacing.
# “faster” = higher speed values; smaller/faster STT is Whisper tiny.en.
VOICES = [
    # American Female (most natural for en)
    "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky",
    "af_alloy", "af_aoede", "af_jessica", "af_kore", "af_nova", "af_river",
    # American Male
    "am_adam", "am_michael", "am_liam", "am_onyx", "am_puck",
    # British Female/Male (use lang_code b)
    "bf_alice", "bf_emma", "bf_isabella", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
VOICE_INFO: dict[str, str] = {
    "af_heart": "American F — warm, default",
    "af_bella": "American F — bright",
    "af_sarah": "American F — soft",
    "af_nicole": "American F — clear",
    "af_sky": "American F — light",
    "af_alloy": "American F — neutral",
    "af_aoede": "American F — expressive",
    "af_jessica": "American F — friendly",
    "af_kore": "American F — crisp",
    "af_nova": "American F — youthful",
    "af_river": "American F — calm",
    "am_adam": "American M — deep, default male",
    "am_michael": "American M — steady",
    "am_liam": "American M — young",
    "am_onyx": "American M — rich",
    "am_puck": "American M — playful",
    "bf_alice": "British F — soft (b)",
    "bf_emma": "British F — clear (b)",
    "bf_isabella": "British F — warm (b)",
    "bm_daniel": "British M — mature (b)",
    "bm_fable": "British M — narrative (b)",
    "bm_george": "British M — crisp (b)",
    "bm_lewis": "British M — gentle (b)",
}
# Whisper (faster-whisper, CPU int8, English-only). Smaller = faster, larger = more accurate.
WHISPER_CHOICES = ["tiny.en", "base.en", "small.en", "medium.en"]
WHISPER_INFO: dict[str, str] = {
    "tiny.en":   "39 MB • fastest • ~6x real-time • lowest accuracy",
    "base.en":   "74 MB • fast • ~3x real-time • balanced (default)",
    "small.en":  "244 MB • 1x real-time • better accuracy",
    "medium.en": "769 MB • ~0.5x real-time • best accuracy, slower",
}
# No hardcoded LLM list — models are discovered via /v1/models from the provider.
# Hints are intentionally empty; Refresh populates the dropdown from the live server.

def _norm_model_app(s: str) -> str:
    """Normalize model id like config._norm_model_id / llm._norm_id (gguf -> basename)."""
    s = (s or "").strip().lstrip("/\\").strip()
    low = s.lower()
    if low.endswith((".gguf", ".bin", ".onnx")):
        if "\\" in s:
            s = s.replace("\\", "/").split("/")[-1]
        elif "/" in s and ":" not in s:
            s = s.split("/")[-1]
    return s.strip()


def _make_icon() -> QIcon:
    pm = QPixmap(128, 128)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    # background
    p.setBrush(QColor("#0f172a"))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, 128, 128, 26, 26)
    # mic capsule
    p.setBrush(QColor("#38bdf8"))
    p.drawRoundedRect(54, 22, 20, 44, 10, 10)
    # mic arc
    p.setPen(QColor("#38bdf8"))
    p.setBrush(Qt.NoBrush)
    # small stand
    p.setPen(QColor("#94a3b8"))
    p.drawLine(64, 78, 64, 96)
    p.drawLine(48, 96, 80, 96)
    # waveform bars
    for x, h, col in [(28, 18, "#22c55e"), (36, 28, "#22c55e"), (84, 22, "#f59e0b"), (92, 14, "#f59e0b")]:
        p.fillRect(x, 64 - h // 2, 6, h, QColor(col))
    p.end()
    return QIcon(pm)


class _Signals(QObject):
    sig_state = Signal(object)
    sig_heard = Signal(str)
    sig_delta = Signal(str)
    sig_complete = Signal(str)
    sig_error = Signal(str)


class GuiListener(Listener):
    def __init__(self, sig: _Signals):
        self._sig = sig

    def state_changed(self, state: State) -> None:
        self._sig.sig_state.emit(state)

    def heard_text(self, text: str) -> None:
        self._sig.sig_heard.emit(text)

    def reply_delta(self, delta: str) -> None:
        self._sig.sig_delta.emit(delta)

    def reply_complete(self, full: str) -> None:
        self._sig.sig_complete.emit(full)

    def error(self, message: str) -> None:
        self._sig.sig_error.emit(message)


class MainWindow(QMainWindow):
    sig_state = Signal(object)
    sig_heard = Signal(str)
    sig_delta = Signal(str)
    sig_complete = Signal(str)
    sig_error = Signal(str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._session = None
        self._reply_buf = ""
        self._signals = _Signals()
        # bridge internal signals to our own (so GuiListener can emit via _signals)
        self._signals.sig_state.connect(self.sig_state.emit)
        self._signals.sig_heard.connect(self.sig_heard.emit)
        self._signals.sig_delta.connect(self.sig_delta.emit)
        self._signals.sig_complete.connect(self.sig_complete.emit)
        self._signals.sig_error.connect(self.sig_error.emit)

        self.setWindowTitle("Vocalis")
        self.setFixedSize(560, 260)
        self.setWindowIcon(_make_icon())

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(8)

        # top row: dot + state + spacer + settings
        top = QHBoxLayout()
        self._dot = QLabel()
        self._dot.setFixedSize(14, 14)
        self._dot.setStyleSheet(f"background:{STATE_COLORS[State.IDLE]}; border-radius:7px;")
        self._state_label = QLabel(STATE_LABELS[State.IDLE])
        self._state_label.setStyleSheet("font-size:15px; font-weight:600; color:#e2e8f0;")
        top.addWidget(self._dot)
        top.addWidget(self._state_label)
        top.addStretch(1)
        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setFixedHeight(28)
        self._btn_settings.setStyleSheet("padding:0 12px;")
        self._btn_settings.clicked.connect(self._open_settings)
        top.addWidget(self._btn_settings)
        v.addLayout(top)

        self._heard = QLabel("\u2014")
        self._heard.setStyleSheet("color:#94a3b8; font-style:italic;")
        self._heard.setWordWrap(True)
        self._heard.setMaximumHeight(40)
        v.addWidget(self._heard)

        self._reply = QPlainTextEdit()
        self._reply.setReadOnly(True)
        self._reply.setPlaceholderText("Assistant reply will appear here\u2026")
        self._reply.setMaximumHeight(96)
        self._reply.setStyleSheet("background:#0b1220; color:#e2e8f0; border:1px solid #1e293b; border-radius:8px; padding:6px;")
        v.addWidget(self._reply)

        self._error = QLabel("")
        self._error.setStyleSheet("color:#f87171; font-size:11px;")
        self._error.setWordWrap(True)
        self._error.setVisible(False)
        v.addWidget(self._error)

        # bottom hint
        hint = QLabel("Wake word to start \u00b7 wake word again to interrupt")
        hint.setStyleSheet("color:#475569; font-size:11px;")
        v.addWidget(hint)

        # log row — persistent so “closed before I could check” never happens again
        log_row = QHBoxLayout()
        try:
            lp = str(log_path())
        except Exception:
            lp = "<log unavailable>"
        self._log_label = QLabel(lp)
        self._log_label.setStyleSheet("color:#475569; font-size:10px;")
        self._log_label.setToolTip(lp)
        self._log_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._log_label.setWordWrap(False)
        log_row.addWidget(self._log_label, 1)
        self._btn_logs = QPushButton("Open Logs")
        self._btn_logs.setFixedHeight(22)
        self._btn_logs.setToolTip("Open logs folder (also: vocalis --logs)")
        self._btn_logs.setStyleSheet("font-size:11px; padding:0 8px;")
        self._btn_logs.clicked.connect(self._open_logs)
        log_row.addWidget(self._btn_logs)
        self._btn_view = QPushButton("View")
        self._btn_view.setFixedHeight(22)
        self._btn_view.setToolTip("View last 300 lines of log")
        self._btn_view.setStyleSheet("font-size:11px; padding:0 8px;")
        self._btn_view.clicked.connect(self._view_log)
        log_row.addWidget(self._btn_view)
        v.addLayout(log_row)

        self.setStyleSheet("QMainWindow{background:#0f172a;} QWidget{color:#e2e8f0;}")

        self.sig_state.connect(self._on_state)
        self.sig_heard.connect(self._on_heard)
        self.sig_delta.connect(self._on_delta)
        self.sig_complete.connect(self._on_complete)
        self.sig_error.connect(self._on_error)

        self._tray: QSystemTrayIcon | None = None

    def listener(self) -> GuiListener:
        return GuiListener(self._signals)

    def attach_session(self, session) -> None:
        self._session = session

    def _on_state(self, state: State) -> None:
        col = STATE_COLORS.get(state, "#64748b")
        self._dot.setStyleSheet(f"background:{col}; border-radius:7px;")
        self._state_label.setText(STATE_LABELS.get(state, state.value))
        log.debug("UI state -> %s", state.value)
        if state is State.LISTENING:
            self._reply_buf = ""
            self._reply.setPlainText("")
        # keep error visible — don't auto-hide so crash reason survives

    def _on_heard(self, text: str) -> None:
        self._heard.setText(f"You: {text}")
        self._reply_buf = ""
        self._reply.setPlainText("")
        log.debug("UI heard: %s", text)

    def _on_delta(self, delta: str) -> None:
        self._reply_buf += delta
        self._reply.setPlainText(self._reply_buf)
        self._reply.verticalScrollBar().setValue(self._reply.verticalScrollBar().maximum())

    def _on_complete(self, full: str) -> None:
        self._reply_buf = full
        self._reply.setPlainText(full)
        log.debug("UI reply complete: %s", full[:120])

    def _on_error(self, msg: str) -> None:
        log.error("UI error: %s", msg)
        try:
            from .logger import flush
            flush()
        except Exception:
            pass
        # keep full message in tooltip, truncated in label
        self._error.setText(msg)
        self._error.setToolTip(msg + f"\n\nLog: {log_path()}")
        self._error.setVisible(True)

    def _open_logs(self) -> None:
        try:
            reveal_in_file_manager(logs_dir())
            log.info("Open logs folder requested")
        except Exception as e:
            log.warning("Open logs failed: %s", e)
            QMessageBox.information(self, "Logs", f"Log folder:\n{logs_dir()}\n\nLog file:\n{log_path()}\n\nError opening: {e}")

    def _view_log(self) -> None:
        try:
            p = log_path()
            text = ""
            if p.exists():
                # last ~300 lines, ~100KB
                data = p.read_text(encoding="utf-8", errors="replace")
                lines = data.splitlines()
                text = "\n".join(lines[-300:])
                if len(lines) > 300:
                    text = f"... ({len(lines)-300} earlier lines hidden) ...\n" + text
            else:
                text = f"No log yet at:\n{p}\n\nRun once to create it, or check --log-level"
            dlg = QDialog(self)
            dlg.setWindowTitle("Vocalis — Log (last 300 lines)")
            dlg.resize(760, 480)
            lay = QVBoxLayout(dlg)
            hint = QLabel(f"File: {p}  —  also: vocalis --logs  or  vocalis --log-level DEBUG")
            hint.setStyleSheet("color:#94a3b8; font-size:11px;")
            hint.setWordWrap(True)
            lay.addWidget(hint)
            view = QPlainTextEdit()
            view.setReadOnly(True)
            view.setPlainText(text or "(empty)")
            view.setStyleSheet("background:#0b1220; color:#e2e8f0; font-family: Consolas, monospace; font-size:11px;")
            lay.addWidget(view, 1)
            row = QHBoxLayout()
            btn_copy = QPushButton("Copy to clipboard")
            def _copy():
                view.selectAll()
                view.copy()
            btn_copy.clicked.connect(_copy)
            btn_open = QPushButton("Open folder")
            btn_open.clicked.connect(lambda: reveal_in_file_manager(logs_dir()))
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dlg.accept)
            row.addWidget(btn_copy)
            row.addWidget(btn_open)
            row.addStretch(1)
            row.addWidget(btn_close)
            lay.addLayout(row)
            dlg.exec()
        except Exception as e:
            log.exception("View log failed: %s", e)
            QMessageBox.warning(self, "Logs", f"Could not view log: {e}\n\nPath: {log_path()}")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted:
            # save already done inside dialog; restart session
            save_config(self.cfg)
            self._restart_session()
            QMessageBox.information(self, "Saved", "Settings saved. Session restarted.")

    def _restart_session(self) -> None:
        if self._session is not None:
            try:
                self._session.stop()
            except Exception:
                pass
        from .runner import start_session
        try:
            s = start_session(self.cfg, self.listener())
            t = threading.Thread(target=s.run, daemon=True, name="session")
            t.start()
            self.attach_session(s)
            self._session_thread = t  # type: ignore
            # re-attach watchdog if present
            try:
                wd = getattr(self, "_watchdog", None)
                if wd is not None:
                    wd.attach(s, t)
            except Exception:
                pass
            self._on_state(State.IDLE)
        except Exception as e:
            self._on_error(f"Could not start audio: {e}")

    def closeEvent(self, event):  # type: ignore[override]
        if self._tray is not None and self._tray.isVisible():
            event.ignore()
            self.hide()
            self._tray.showMessage("Vocalis", "Still listening in the background. Use the tray icon to show or quit.", QSystemTrayIcon.Information, 2500)
        else:
            if self._session is not None:
                try:
                    self._session.stop()
                except Exception:
                    pass
            event.accept()


class _TestSignals(QObject):
    sig_llm = Signal(str)
    sig_tts = Signal(str)
    sig_wake = Signal(str)
    sig_wake_done = Signal(str)
    sig_models_ok = Signal(object)  # list[str]
    sig_models_err = Signal(str)


class SettingsDialog(QDialog):
    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Vocalis — Settings")
        self.setMinimumWidth(540)
        self.setWindowIcon(_make_icon())
        self._test_signals = _TestSignals()
        self._workers: list[threading.Thread] = []

        outer = QVBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(0, 0, 6, 0)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        # Preset
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self._preset = QComboBox()
        self._preset.addItems(["Ollama", "LM Studio", "Unsloth Desktop", "Custom"])
        # guess preset from base_url
        cur = cfg.llm_base_url
        guess = "Custom"
        for k, (u, _) in PRESETS.items():
            if k != "Custom" and u == cur:
                guess = k
                break
        self._preset.setCurrentText(guess)
        form.addRow("LLM preset", self._preset)

        self._base_url = QLineEdit(cfg.llm_base_url)
        self._base_url.setPlaceholderText("http://localhost:11434/v1")
        form.addRow("Base URL", self._base_url)

        # Model as selectable list (fetch from provider) + editable fallback
        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setInsertPolicy(QComboBox.NoInsert)
        self._model.setMinimumWidth(260)
        cur_model = _norm_model_app(cfg.llm_model or "")
        # Store last fetched ids to validate selection on save
        self._last_fetched_ids: list[str] = []
        if cur_model:
            self._model.addItem(cur_model, cur_model)
            self._model.setCurrentIndex(0)
        # ensure lineEdit shows it even if editable
        self._model.setCurrentText(cur_model)
        self._model.setToolTip("Click Refresh to load models from the provider via /v1/models — selected model will be auto-loaded on prompt (LM Studio TTL)")
        self._model.lineEdit().setPlaceholderText("select or type model id — click Refresh to fetch from provider")
        model_row = QHBoxLayout()
        model_row.addWidget(self._model, 1)
        self._btn_refresh_models = QPushButton("Refresh")
        self._btn_refresh_models.setFixedHeight(26)
        self._btn_refresh_models.setToolTip("Fetch /v1/models from the base URL (Ollama/LM Studio/Unsloth)")
        self._btn_refresh_models.clicked.connect(self._refresh_models)
        model_row.addWidget(self._btn_refresh_models)
        form.addRow("Model", model_row)
        self._lbl_models = QLabel("Click Refresh to load available models from the provider. Leave API key empty for local providers.")
        self._lbl_models.setStyleSheet("color:#64748b; font-size:11px;")
        self._lbl_models.setWordWrap(True)
        form.addRow("", self._lbl_models)

        self._api_key = QLineEdit(cfg.llm_api_key)
        self._api_key.setPlaceholderText("Optional — leave empty for local providers (Ollama / LM Studio / Unsloth)")
        self._api_key.setToolTip("Leave empty for local providers. Only needed for cloud APIs (OpenAI, etc.).")
        self._api_key.setEchoMode(QLineEdit.Password)
        form.addRow("API key (optional)", self._api_key)

        self._temp = QDoubleSpinBox()
        self._temp.setRange(0.0, 2.0)
        self._temp.setSingleStep(0.1)
        self._temp.setValue(float(cfg.temperature))
        form.addRow("Temperature", self._temp)

        self._sys = QPlainTextEdit(cfg.system_prompt)
        self._sys.setMaximumHeight(70)
        self._sys.setPlaceholderText("System prompt")
        form.addRow("System prompt", self._sys)

        # History: forget vs preserve+compact
        from PySide6.QtWidgets import QCheckBox
        self._preserve = QCheckBox("Preserve conversation history (compact after 30)")
        self._preserve.setChecked(bool(getattr(cfg, "preserve_history", False)))
        self._preserve.setToolTip("If off, each turn is forgotten (stateless). If on, last 30 messages are kept and older ones are compacted into a summary.")
        self._preserve.stateChanged.connect(self._on_preserve_changed)
        form.addRow("", self._preserve)
        self._compact_spin = QSpinBox()
        self._compact_spin.setRange(10, 100)
        self._compact_spin.setValue(int(getattr(cfg, "compact_after", 30)))
        self._compact_spin.setSuffix(" msgs")
        self._compact_spin.setToolTip("When preserving, compact when history exceeds this many messages")
        self._compact_spin.setEnabled(self._preserve.isChecked())
        form.addRow("Compact after", self._compact_spin)
        self._lbl_history = QLabel("Off: each reply forgets prior chat. On: keeps last 30 and compacts older into a summary (no LLM call).")
        self._lbl_history.setStyleSheet("color:#64748b; font-size:11px;")
        self._lbl_history.setWordWrap(True)
        form.addRow("", self._lbl_history)

        lay.addLayout(form)

        # test LLM row
        row_llm = QHBoxLayout()
        self._btn_test_llm = QPushButton("Test LLM")
        self._btn_test_llm.clicked.connect(self._do_test_llm)
        self._lbl_llm = QLabel("")
        self._lbl_llm.setStyleSheet("color:#94a3b8;")
        self._lbl_llm.setWordWrap(True)
        row_llm.addWidget(self._btn_test_llm)
        row_llm.addWidget(self._lbl_llm, 1)
        lay.addLayout(row_llm)
        self._test_signals.sig_llm.connect(self._on_llm_result)
        self._test_signals.sig_tts.connect(self._on_tts_result)
        self._test_signals.sig_models_ok.connect(self._on_models_ok)
        self._test_signals.sig_models_err.connect(self._on_models_err)

        # TTS Engine — Kokoro vs Piper (custom voices)
        from PySide6.QtWidgets import QCheckBox  # local for type checkers
        engine_form = QFormLayout()
        engine_form.setLabelAlignment(Qt.AlignRight)
        self._tts_engine = QComboBox()
        self._tts_engine.addItems(["kokoro", "piper"])
        cur_eng = getattr(cfg, "tts_engine", "kokoro")
        if cur_eng not in ("kokoro", "piper"):
            cur_eng = "kokoro"
        self._tts_engine.setCurrentText(cur_eng)
        self._tts_engine.setToolTip("kokoro = 82M built-in voices; piper = local .onnx custom voices (e.g. GLaDOS)")
        engine_form.addRow("TTS engine", self._tts_engine)
        lay.addLayout(engine_form)

        # Kokoro controls (shown when engine==kokoro)
        self._kokoro_frame = QFrame()
        self._kokoro_frame.setStyleSheet("QFrame{border:none;}")
        kf_lay = QFormLayout(self._kokoro_frame)
        kf_lay.setContentsMargins(0, 0, 0, 0)
        self._voice = QComboBox()
        self._voice.setEditable(True)
        for v in VOICES:
            info = VOICE_INFO.get(v, "")
            self._voice.addItem(f"{v} — {info}" if info else v, v)
        idx = self._voice.findData(cfg.tts_voice)
        if idx >= 0:
            self._voice.setCurrentIndex(idx)
        else:
            self._voice.addItem(cfg.tts_voice, cfg.tts_voice)
            self._voice.setCurrentIndex(self._voice.count() - 1)
        self._voice.setToolTip("Kokoro 82M — voice doesn't change model size; use Speed for faster/slower. 'a'=American, 'b'=British.")
        kf_lay.addRow("Voice", self._voice)
        self._lbl_voice_info = QLabel(VOICE_INFO.get(cfg.tts_voice, "Tip: speed 1.5–2.0 is faster; 0.8–1.0 is more natural."))
        self._lbl_voice_info.setStyleSheet("color:#64748b; font-size:11px;")
        self._lbl_voice_info.setWordWrap(True)
        kf_lay.addRow("", self._lbl_voice_info)
        self._voice.currentIndexChanged.connect(self._on_voice_changed)
        if self._voice.lineEdit():
            self._voice.lineEdit().textChanged.connect(lambda t: self._on_voice_changed(-1))
        self._speed = QDoubleSpinBox()
        self._speed.setRange(0.5, 2.0)
        self._speed.setSingleStep(0.05)
        self._speed.setValue(float(cfg.tts_speed))
        self._speed.setToolTip("1.0 = normal, 1.5–1.8 = noticeably faster, 0.8 = slower/more natural")
        kf_lay.addRow("Speed", self._speed)
        lay.addWidget(self._kokoro_frame)

        # Piper controls (shown when engine==piper)
        self._piper_frame = QFrame()
        self._piper_frame.setStyleSheet("QFrame{border:none;}")
        pf_lay = QFormLayout(self._piper_frame)
        pf_lay.setContentsMargins(0, 0, 0, 0)
        # Model path
        self._piper_model = QLineEdit(getattr(cfg, "piper_model", "") or "")
        self._piper_model.setPlaceholderText("Path to .onnx voice (e.g. en_US-lessac-medium.onnx)")
        self._piper_model.setToolTip("Piper voice model .onnx — comes with a .onnx.json config (auto-found)")
        pf_lay.addRow("Piper model (.onnx)", self._piper_model)
        self._btn_piper_browse = QPushButton("Browse .onnx…")
        self._btn_piper_browse.setToolTip("Choose a Piper voice .onnx (its .onnx.json will be auto-loaded)")
        self._btn_piper_browse.clicked.connect(self._browse_piper_model)
        pf_lay.addRow("", self._btn_piper_browse)
        # Config override (optional)
        self._piper_config = QLineEdit(getattr(cfg, "piper_config", "") or "")
        self._piper_config.setPlaceholderText("Optional .onnx.json (auto if empty)")
        self._piper_config.setToolTip("Leave empty to use <model>.json next to the .onnx")
        pf_lay.addRow("Piper config (.json)", self._piper_config)
        self._btn_piper_cfg_browse = QPushButton("Browse .json…")
        self._btn_piper_cfg_browse.clicked.connect(self._browse_piper_config)
        pf_lay.addRow("", self._btn_piper_cfg_browse)
        # Speaker & prosody
        self._piper_speaker = QSpinBox()
        self._piper_speaker.setRange(0, 32)
        self._piper_speaker.setValue(int(getattr(cfg, "piper_speaker", 0) or 0))
        self._piper_speaker.setToolTip("Speaker id for multi-speaker models (0 = default)")
        pf_lay.addRow("Speaker id", self._piper_speaker)
        self._piper_length = QDoubleSpinBox()
        self._piper_length.setRange(0.3, 3.0)
        self._piper_length.setSingleStep(0.05)
        self._piper_length.setValue(float(getattr(cfg, "piper_length_scale", 1.0) or 1.0))
        self._piper_length.setToolTip("Length scale: <1 faster, >1 slower (Piper)")
        pf_lay.addRow("Length scale", self._piper_length)
        self._lbl_piper_info = QLabel("Piper voices are local .onnx files — drop any voice (e.g. GLaDOS, Darth Maul via community voices) into a folder and browse to it. No download needed.")
        self._lbl_piper_info.setStyleSheet("color:#64748b; font-size:11px;")
        self._lbl_piper_info.setWordWrap(True)
        pf_lay.addRow("", self._lbl_piper_info)
        lay.addWidget(self._piper_frame)

        # Toggle frames by engine
        def _update_tts_frames():
            is_piper = self._tts_engine.currentText() == "piper"
            self._kokoro_frame.setVisible(not is_piper)
            self._piper_frame.setVisible(is_piper)
        self._tts_engine.currentTextChanged.connect(lambda _: _update_tts_frames())
        _update_tts_frames()

        row_tts = QHBoxLayout()
        self._btn_tts = QPushButton("Speak test")
        self._btn_tts.clicked.connect(self._do_tts_test)
        self._lbl_tts = QLabel("")
        self._lbl_tts.setStyleSheet("color:#94a3b8;")
        row_tts.addWidget(self._btn_tts)
        row_tts.addWidget(self._lbl_tts, 1)
        lay.addLayout(row_tts)

        # Wake word
        form3 = QFormLayout()
        self._wake = QComboBox()
        self._wake.setEditable(True)
        self._wake.addItems(WAKE_CHOICES)
        # cfg.wake_word may be a path — show basename or full path; handle file vs preset
        cur_wake = (cfg.wake_word or "hey_jarvis").strip()
        if cur_wake in WAKE_CHOICES:
            self._wake.setCurrentText(cur_wake)
        else:
            # Custom path — ensure it's visible even though editable combo only has presets
            if self._wake.findText(cur_wake) == -1:
                self._wake.addItem(cur_wake)
            self._wake.setCurrentText(cur_wake)
        form3.addRow("Wake word", self._wake)
        self._btn_wake_browse = QPushButton("Browse .onnx / .tflite…")
        self._btn_wake_browse.clicked.connect(self._browse_wake)
        form3.addRow("", self._btn_wake_browse)

        self._wake_emb = QLineEdit(cfg.wakeword_embeddings)
        self._wake_emb.setPlaceholderText("Optional embeddings file")
        form3.addRow("Embeddings", self._wake_emb)
        self._btn_emb_browse = QPushButton("Browse…")
        self._btn_emb_browse.clicked.connect(self._browse_emb)
        form3.addRow("", self._btn_emb_browse)

        self._wake_thr = QDoubleSpinBox()
        self._wake_thr.setRange(0.1, 0.9)
        self._wake_thr.setSingleStep(0.05)
        self._wake_thr.setValue(float(cfg.wakeword_threshold))
        form3.addRow("Threshold", self._wake_thr)

        self._wake_cd = QSpinBox()
        self._wake_cd.setRange(100, 5000)
        self._wake_cd.setSingleStep(100)
        self._wake_cd.setValue(int(cfg.wakeword_cooldown_ms))
        self._wake_cd.setSuffix(" ms")
        form3.addRow("Cooldown", self._wake_cd)

        lay.addLayout(form3)
        row_wake = QHBoxLayout()
        self._btn_wake_test = QPushButton("Listen 8s — test wake word")
        self._btn_wake_test.clicked.connect(self._do_wake_test)
        self._btn_wake_cal = QPushButton("Auto-calibrate 30s")
        self._btn_wake_cal.setToolTip("30s wizard: 12s silence + 18s say wake word 3-5 times, auto-suggests threshold/cooldown")
        self._btn_wake_cal.clicked.connect(self._do_wake_calibrate)
        self._lbl_wake = QLabel("")
        self._lbl_wake.setStyleSheet("color:#94a3b8;")
        row_wake.addWidget(self._btn_wake_test)
        row_wake.addWidget(self._btn_wake_cal)
        row_wake.addWidget(self._lbl_wake, 1)
        lay.addLayout(row_wake)
        self._test_signals.sig_wake.connect(self._lbl_wake.setText)
        self._test_signals.sig_wake_done.connect(lambda t: (self._lbl_wake.setText(t), self._btn_wake_test.setEnabled(True), self._btn_wake_cal.setEnabled(True)))

        # --- Sounds (swappable earcons) ---
        from PySide6.QtWidgets import QCheckBox as _QCB
        sounds_form = QFormLayout()
        sounds_form.setLabelAlignment(Qt.AlignRight)
        self._sounds_enabled = _QCB("Enable earcons (wake/finished/timer/alarm sounds)")
        self._sounds_enabled.setChecked(bool(getattr(cfg, "sound_enabled", True)))
        self._sounds_enabled.setToolTip("Master mute for all earcons. Uncheck for silent operation.")
        sounds_form.addRow("", self._sounds_enabled)
        self._sound_volume = QDoubleSpinBox()
        self._sound_volume.setRange(0.05, 1.0)
        self._sound_volume.setSingleStep(0.05)
        self._sound_volume.setValue(float(getattr(cfg, "sound_volume", 0.7)))
        self._sound_volume.setSuffix(" vol")
        sounds_form.addRow("Volume", self._sound_volume)
        # helper to make sound row
        def _make_sound_row(label: str, attr: str, default_name: str, tooltip: str):
            le = QLineEdit(getattr(cfg, attr, "") or "")
            le.setPlaceholderText(f"Custom .wav/.mp3/.ogg or empty = {default_name}")
            le.setToolTip(tooltip)
            btn_browse = QPushButton("Browse…")
            btn_preview = QPushButton("Preview")
            btn_reset = QPushButton("Default")
            row = QHBoxLayout()
            row.addWidget(le, 1)
            row.addWidget(btn_browse)
            row.addWidget(btn_preview)
            row.addWidget(btn_reset)
            # closures
            def _browse(le=le):
                path, _ = QFileDialog.getOpenFileName(self, f"Choose {label} sound", "", "Audio (*.wav *.mp3 *.ogg *.flac);;All files (*.*)")
                if path:
                    le.setText(path)
            def _preview(le=le):
                p = le.text().strip()
                if not p:
                    # preview default
                    try:
                        from .sounds import Sounds
                        from .audio_io import Playback
                        s = Sounds(Playback(), cfg)
                        # temporarily set cfg for preview? just preview file directly
                        ok = s.preview_file(p) if p else s.preview({"Wake": "wake", "Finished": "finished", "Timer set": "timer_set", "Alarm": "alarm"}.get(label, "wake"))
                        if not ok:
                            QMessageBox.information(self, "Preview", f"No audio produced for {label}. Check path or output device.")
                    except Exception as e:
                        QMessageBox.warning(self, "Preview", f"Failed: {e}")
                    return
                try:
                    from .sounds import Sounds
                    from .audio_io import Playback
                    s = Sounds(Playback(), cfg)
                    # use custom path directly
                    ok = s.preview_file(p)
                    if not ok:
                        # try default preview
                        ok2 = s.preview({"Wake": "wake", "Finished": "finished", "Timer set": "timer_set", "Alarm": "alarm"}.get(label, "wake"))
                        if not ok2:
                            QMessageBox.information(self, "Preview", f"No audio for {label}")
                except Exception as e:
                    QMessageBox.warning(self, "Preview", f"Failed: {e}")
            def _reset(le=le):
                le.clear()
            btn_browse.clicked.connect(_browse)
            btn_preview.clicked.connect(_preview)
            btn_reset.clicked.connect(_reset)
            return le, btn_browse, btn_preview, btn_reset, row
        self._sound_wake_le, _, _, _, row_wake_snd = _make_sound_row("Wake", "sound_wake_path", "wake-up.ogg", "Played after wake word. Empty uses bundled wake-up.ogg")
        sounds_form.addRow("Wake sound", row_wake_snd)
        self._sound_finished_le, _, _, _, row_fin = _make_sound_row("Finished", "sound_finished_path", "finished-listening.ogg", "Played when listening ends (endpoint).")
        sounds_form.addRow("Finished sound", row_fin)
        self._sound_timer_le, _, _, _, row_tim = _make_sound_row("Timer set", "sound_timer_set_path", "timer-set.mp3", "Played when timer is set.")
        sounds_form.addRow("Timer set sound", row_tim)
        self._sound_alarm_le, _, _, _, row_alm = _make_sound_row("Alarm", "sound_alarm_path", "Lithium.mp3 (or alarm_tone)", "Looped alarm tone for timer & alarms. Also used for Alarms.")
        sounds_form.addRow("Alarm sound", row_alm)
        lay.addLayout(sounds_form)
        # test all sounds
        row_snd_test = QHBoxLayout()
        self._btn_snd_test_all = QPushButton("Test All Sounds")
        self._btn_snd_test_all.setToolTip("Play wake → finished → timer-set → alarm in sequence")
        self._btn_snd_test_all.clicked.connect(self._preview_all_sounds)
        self._lbl_snd_test = QLabel("")
        self._lbl_snd_test.setStyleSheet("color:#94a3b8;")
        row_snd_test.addWidget(self._btn_snd_test_all)
        row_snd_test.addWidget(self._lbl_snd_test, 1)
        lay.addLayout(row_snd_test)

        # STT — smaller/faster vs larger/more accurate
        form4 = QFormLayout()
        self._whisper = QComboBox()
        for w in WHISPER_CHOICES:
            info = WHISPER_INFO.get(w, "")
            self._whisper.addItem(f"{w} — {info}" if info else w, w)
        # select current
        w_idx = self._whisper.findData(cfg.whisper_model)
        if w_idx >= 0:
            self._whisper.setCurrentIndex(w_idx)
        else:
            self._whisper.addItem(cfg.whisper_model, cfg.whisper_model)
            self._whisper.setCurrentIndex(self._whisper.count() - 1)
        self._whisper.setToolTip("Smaller = faster, larger = more accurate. tiny.en fastest (~6x real-time), medium.en slowest but best.")
        form4.addRow("Whisper model", self._whisper)
        self._lbl_whisper = QLabel(WHISPER_INFO.get(cfg.whisper_model, WHISPER_INFO.get("base.en", "")))
        self._lbl_whisper.setStyleSheet("color:#64748b; font-size:11px;")
        self._lbl_whisper.setWordWrap(True)
        form4.addRow("", self._lbl_whisper)
        self._whisper.currentIndexChanged.connect(self._on_whisper_changed)
        lay.addLayout(form4)

        # Audio devices
        form5 = QFormLayout()
        self._in_dev = QComboBox()
        self._out_dev = QComboBox()
        self._populate_devices()
        form5.addRow("Input device", self._in_dev)
        form5.addRow("Output device", self._out_dev)
        lay.addLayout(form5)

        # Timing
        form6 = QFormLayout()
        self._vad_db = QDoubleSpinBox()
        self._vad_db.setRange(-60, -20)
        self._vad_db.setSingleStep(1)
        self._vad_db.setValue(float(cfg.vad_rms_dbfs))
        self._vad_db.setSuffix(" dBFS")
        form6.addRow("VAD threshold", self._vad_db)

        self._silence = QDoubleSpinBox()
        self._silence.setRange(0.3, 3.0)
        self._silence.setSingleStep(0.1)
        self._silence.setValue(float(cfg.vad_silence_seconds))
        self._silence.setSuffix(" s")
        form6.addRow("Silence", self._silence)

        self._max_utt = QSpinBox()
        self._max_utt.setRange(5, 120)
        self._max_utt.setValue(int(cfg.max_utterance_seconds))
        self._max_utt.setSuffix(" s")
        form6.addRow("Max utterance", self._max_utt)

        self._idle = QSpinBox()
        self._idle.setRange(5, 600)
        self._idle.setValue(int(cfg.idle_timeout_seconds))
        self._idle.setSuffix(" s")
        form6.addRow("Idle timeout", self._idle)

        lay.addLayout(form6)

        # Behavior (continuous listening etc — previously hidden)
        beh_form = QFormLayout()
        beh_form.setLabelAlignment(Qt.AlignRight)
        from PySide6.QtWidgets import QCheckBox as _QCB2
        self._cont_listen = _QCB2("Continuous listening (no wake word after reply)")
        self._cont_listen.setChecked(bool(getattr(cfg, "continuous_listening", False)))
        self._cont_listen.setToolTip("If on, after speaking returns to LISTENING for follow-up without saying Hey Jarvis again. Off = requires wake word each time.")
        beh_form.addRow("", self._cont_listen)
        self._grace = QDoubleSpinBox()
        self._grace.setRange(2.0, 60.0)
        self._grace.setSingleStep(1.0)
        self._grace.setValue(float(getattr(cfg, "listen_grace_seconds", 10.0)))
        self._grace.setSuffix(" s")
        self._grace.setToolTip("Grace window after speaking where idle timeout is suppressed (only when continuous listening).")
        beh_form.addRow("Listen grace", self._grace)
        self._cont_listen.toggled.connect(lambda on: self._grace.setEnabled(on))
        self._grace.setEnabled(self._cont_listen.isChecked())
        lay.addLayout(beh_form)

        # Preset -> fill handling
        self._preset.currentTextChanged.connect(self._apply_preset)
        self._preset.currentTextChanged.connect(lambda _: self._refresh_models())
        self._base_url.editingFinished.connect(lambda: self._lbl_models.setText("Base URL changed — click Refresh to reload models"))
        # auto-fetch once on open (provider dropdown)
        if self._base_url.text().strip():
            QTimer.singleShot(350, self._refresh_models)

        # --- Logs ---
        logs_frame = QFrame()
        logs_frame.setStyleSheet("QFrame { border:1px solid #1e293b; border-radius:8px; background:#0b1220; }")
        lf_lay = QVBoxLayout(logs_frame)
        lf_lay.setContentsMargins(10, 8, 10, 8)
        lf_lay.setSpacing(6)
        lf_title = QLabel("Logs")
        lf_title.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:12px; border:none; background:transparent;")
        lf_lay.addWidget(lf_title)
        lf_desc = QLabel(f"All runs are logged to:\n{log_path()}\nIf the window closes unexpectedly, check this file.")
        lf_desc.setStyleSheet("color:#94a3b8; font-size:11px; border:none; background:transparent;")
        lf_desc.setWordWrap(True)
        lf_desc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lf_lay.addWidget(lf_desc)
        lf_btns = QHBoxLayout()
        self._btn_open_logs_s = QPushButton("Open Logs Folder")
        self._btn_open_logs_s.clicked.connect(lambda: reveal_in_file_manager(logs_dir()))
        self._btn_view_logs_s = QPushButton("View Log")
        self._btn_view_logs_s.clicked.connect(self._view_log_settings)
        lf_btns.addWidget(self._btn_open_logs_s)
        lf_btns.addWidget(self._btn_view_logs_s)
        lf_btns.addStretch(1)
        lf_lay.addLayout(lf_btns)
        lay.addWidget(logs_frame)

        # --- Alarms (offline) ---
        alarms_frame = QFrame()
        alarms_frame.setStyleSheet("QFrame { border:1px solid #1e293b; border-radius:8px; background:#0b1220; }")
        af_lay = QVBoxLayout(alarms_frame)
        af_lay.setContentsMargins(10, 8, 10, 8)
        af_lay.setSpacing(6)
        af_title = QLabel("Alarms (offline)")
        af_title.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:12px; border:none; background:transparent;")
        af_lay.addWidget(af_title)
        af_desc = QLabel("Voice: 'set alarm for 07:30 daily' or use LLM tool. Stored in data_dir()/alarms.json — no cloud.")
        af_desc.setStyleSheet("color:#94a3b8; font-size:11px; border:none; background:transparent;")
        af_desc.setWordWrap(True)
        af_lay.addWidget(af_desc)
        self._alarms_list = QLabel("No alarms loaded")
        self._alarms_list.setStyleSheet("color:#94a3b8; font-size:11px; border:none; background:transparent;")
        self._alarms_list.setWordWrap(True)
        self._alarms_list.setTextInteractionFlags(Qt.TextSelectableByMouse)
        af_lay.addWidget(self._alarms_list)
        af_row = QHBoxLayout()
        self._alarm_time = QLineEdit()
        self._alarm_time.setPlaceholderText("HH:MM or YYYY-MM-DDTHH:MM (e.g. 07:30)")
        self._alarm_label = QLineEdit()
        self._alarm_label.setPlaceholderText("label")
        self._alarm_rec = QComboBox()
        self._alarm_rec.addItems(["once", "daily", "weekly", "weekdays"])
        self._btn_alarm_add = QPushButton("Add Alarm")
        self._btn_alarm_add.clicked.connect(self._add_alarm)
        self._btn_alarm_refresh = QPushButton("Refresh")
        self._btn_alarm_refresh.clicked.connect(self._refresh_alarms)
        af_row.addWidget(self._alarm_time, 1)
        af_row.addWidget(self._alarm_label, 1)
        af_row.addWidget(self._alarm_rec)
        af_row.addWidget(self._btn_alarm_add)
        af_row.addWidget(self._btn_alarm_refresh)
        af_lay.addLayout(af_row)
        lay.addWidget(alarms_frame)
        QTimer.singleShot(400, self._refresh_alarms)

        # --- Updates ---
        upd_frame = QFrame()
        upd_frame.setStyleSheet("QFrame { border:1px solid #1e293b; border-radius:8px; background:#0b1220; }")
        uf_lay = QVBoxLayout(upd_frame)
        uf_lay.setContentsMargins(10, 8, 10, 8)
        uf_lay.setSpacing(6)
        uf_title = QLabel("Updates")
        uf_title.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:12px; border:none; background:transparent;")
        uf_lay.addWidget(uf_title)
        self._upd_repo = QLineEdit(getattr(cfg, "update_repo", "") or "")
        self._upd_repo.setPlaceholderText("owner/repo (e.g. anomalyco/vocalis) — empty = disabled")
        uf_lay.addWidget(self._upd_repo)
        upd_row = QHBoxLayout()
        self._btn_upd_check = QPushButton("Check for update")
        self._btn_upd_check.clicked.connect(self._check_update)
        self._btn_upd_pull = QPushButton("Update (git pull)")
        self._btn_upd_pull.clicked.connect(self._do_update)
        self._lbl_upd = QLabel("")
        self._lbl_upd.setStyleSheet("color:#94a3b8; font-size:11px; border:none; background:transparent;")
        self._lbl_upd.setWordWrap(True)
        upd_row.addWidget(self._btn_upd_check)
        upd_row.addWidget(self._btn_upd_pull)
        upd_row.addWidget(self._lbl_upd, 1)
        uf_lay.addLayout(upd_row)
        lay.addWidget(upd_frame)

        # --- Danger Zone ---
        danger = QFrame()
        danger.setObjectName("danger")
        danger.setStyleSheet(
            "QFrame#danger { border:1px solid #7f1d1d; border-radius:8px; background:#1a0f0f; }"
            "QLabel#dangerTitle { color:#f87171; font-weight:700; font-size:12px; }"
            "QLabel#dangerDesc { color:#fca5a5; font-size:11px; }"
        )
        d_lay = QVBoxLayout(danger)
        d_lay.setContentsMargins(10, 8, 10, 8)
        d_lay.setSpacing(6)
        title = QLabel("Danger Zone")
        title.setObjectName("dangerTitle")
        d_lay.addWidget(title)
        desc = QLabel("Uninstall removes Vocalis data. Keep models to avoid re-downloading ~700 MB.")
        desc.setObjectName("dangerDesc")
        desc.setWordWrap(True)
        d_lay.addWidget(desc)
        d_btns = QHBoxLayout()
        self._btn_uninstall_keep = QPushButton("Uninstall — keep models")
        self._btn_uninstall_keep.setStyleSheet("background:#7f1d1d; color:#fee2e2; padding:6px 10px; border-radius:6px;")
        self._btn_uninstall_keep.clicked.connect(lambda: self._confirm_uninstall(keep_models=True))
        self._btn_uninstall_nuke = QPushButton("Uninstall — erase everything")
        self._btn_uninstall_nuke.setStyleSheet("background:#dc2626; color:white; padding:6px 10px; border-radius:6px; font-weight:600;")
        self._btn_uninstall_nuke.clicked.connect(lambda: self._confirm_uninstall(keep_models=False))
        d_btns.addWidget(self._btn_uninstall_keep)
        d_btns.addWidget(self._btn_uninstall_nuke)
        d_lay.addLayout(d_btns)
        lay.addWidget(danger)

        # buttons (outside scroll so always visible)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_save = QPushButton("Save")
        self._btn_save.setDefault(True)
        self._btn_save.clicked.connect(self._save)
        btns.addWidget(self._btn_cancel)
        btns.addWidget(self._btn_save)
        outer.addLayout(btns)

    def _populate_devices(self) -> None:
        try:
            from .audio_io import input_devices, output_devices
            ins = input_devices()
            outs = output_devices()
        except Exception:
            ins, outs = [], []
        self._in_dev.addItem("Default", None)
        for idx, name in ins:
            self._in_dev.addItem(f"{idx}: {name}", idx)
        self._out_dev.addItem("Default", None)
        for idx, name in outs:
            self._out_dev.addItem(f"{idx}: {name}", idx)
        # select current
        for cb, val in [(self._in_dev, self.cfg.input_device), (self._out_dev, self.cfg.output_device)]:
            if val is None:
                cb.setCurrentIndex(0)
            else:
                for i in range(cb.count()):
                    if cb.itemData(i) == val:
                        cb.setCurrentIndex(i)
                        break

    def _model_id(self) -> str:
        # For editable combo, visible text is ground truth; data may be stale if index mismatch
        txt = self._model.currentText().strip()
        if " —" in txt:
            txt = txt.split(" —")[0].strip()
        txt = txt.lstrip("/\\").strip()
        # normalize like config/llm (handle gguf file paths -> basename)
        def _norm(s: str) -> str:
            s = (s or "").strip().lstrip("/\\").strip()
            low = s.lower()
            if low.endswith((".gguf", ".bin", ".onnx")):
                if "\\" in s:
                    s = s.replace("\\", "/").split("/")[-1]
                elif "/" in s and ":" not in s:
                    s = s.split("/")[-1]
            return s.strip()
        txt = _norm(txt)
        data = self._model.currentData()
        if isinstance(data, str) and data.strip():
            d = _norm(data.strip())
            if d == txt:
                return d
            if d and txt and d != txt:
                log.debug("Model combo data/text mismatch data=%r text=%r -> using text", d, txt)
        return txt

    def _apply_preset(self, name: str) -> None:
        if name == "Custom":
            return
        url, key = PRESETS.get(name, ("", ""))
        if url:
            self._base_url.setText(url)
        # API key is optional for local providers — clear old placeholders
        if name in ("Ollama", "LM Studio", "Unsloth Desktop"):
            if self._api_key.text().strip().lower() in ("", "vocalis-local", "ollama", "lm-studio", "unsloth", "none", "not-needed"):
                self._api_key.clear()
        elif key and self._api_key.text().strip().lower() in ("", "vocalis-local", "ollama", "lm-studio", "unsloth", "none"):
            self._api_key.setText(key)

    def _browse_wake(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose wake-word model", "", "Wake word (*.onnx *.tflite);;ONNX (*.onnx);;TFLite (*.tflite);;All files (*.*)")
        if path:
            # If combo doesn't contain this path, add it so it persists
            if self._wake.findText(path) == -1:
                self._wake.addItem(path)
            self._wake.setCurrentText(path)

    def _browse_emb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose embeddings file", "", "All files (*.*)")
        if path:
            self._wake_emb.setText(path)

    def _safe_emit(self, signal, *args) -> None:
        try:
            import sip
            if sip.isdeleted(self):
                return
        except Exception:
            pass
        try:
            if not self.isVisible() and signal in (self._test_signals.sig_llm, self._test_signals.sig_tts, self._test_signals.sig_models_ok, self._test_signals.sig_models_err, self._test_signals.sig_wake, self._test_signals.sig_wake_done):
                # dialog already closed — don't emit to deleted QObject
                from PySide6.QtWidgets import QApplication
                if not self.isVisible() and QApplication.instance() is None:
                    return
            signal.emit(*args)
        except RuntimeError:
            pass

    def _prune_workers(self) -> None:
        self._workers = [t for t in self._workers if t.is_alive()]

    def _do_test_llm(self) -> None:
        self._btn_test_llm.setEnabled(False)
        self._lbl_llm.setText("Testing…")
        url = self._base_url.text().strip()
        model = self._model_id()
        key = self._api_key.text().strip()
        temp = float(self._temp.value())

        def worker():
            try:
                from .llm import LLMClient
                c = LLMClient(url, model, key, temp)
                r = c.check()
                self._safe_emit(self._test_signals.sig_llm, f"OK — replied: {r[:60]}")
            except Exception as e:
                self._safe_emit(self._test_signals.sig_llm, f"Failed: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._prune_workers()
        self._workers.append(t)

    def _on_llm_result(self, text: str) -> None:
        self._lbl_llm.setText(text)
        self._btn_test_llm.setEnabled(True)

    def _on_tts_result(self, text: str) -> None:
        self._lbl_tts.setText(text)
        self._btn_tts.setEnabled(True)

    def _refresh_models(self) -> None:
        url = self._base_url.text().strip()
        key = self._api_key.text().strip()
        if not url:
            self._lbl_models.setText("Set Base URL first (e.g. http://localhost:11434/v1)")
            self._lbl_models.setStyleSheet("color:#f87171; font-size:11px;")
            return
        self._btn_refresh_models.setEnabled(False)
        self._btn_refresh_models.setText("...")
        self._lbl_models.setText("Fetching /v1/models ...")
        self._lbl_models.setStyleSheet("color:#94a3b8; font-size:11px;")
        # use currentData if available, else currentText stripped of hint suffix
        cur_data = self._model.currentData()
        cur = (cur_data if isinstance(cur_data, str) and cur_data else self._model.currentText()).strip()
        if " — hint" in cur:
            cur = cur.split(" —")[0].strip()
        cur = _norm_model_app(cur)

        def worker():
            try:
                from .llm import LLMClient
                c = LLMClient(url, cur or "test", key)
                ids = c.list_models()
                self._safe_emit(self._test_signals.sig_models_ok, ids)
            except Exception as e:
                self._safe_emit(self._test_signals.sig_models_err, str(e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._prune_workers()
        self._workers.append(t)

    def _on_models_ok(self, models: object) -> None:
        self._btn_refresh_models.setEnabled(True)
        self._btn_refresh_models.setText("Refresh")
        try:
            ids = list(models)  # type: ignore[arg-type]
        except Exception:
            ids = []
        if not ids:
            self._lbl_models.setText("No models returned — check Base URL / API key, or type a model id manually.")
            self._lbl_models.setStyleSheet("color:#f59e0b; font-size:11px;")
            self._last_fetched_ids = []
            return
        self._last_fetched_ids = list(ids)
        # normalize current like config/llm (gguf -> basename)
        _cur_raw = self._model.currentData() if isinstance(self._model.currentData(), str) and self._model.currentData() else self._model.currentText()
        cur = _cur_raw.strip()
        if " —" in cur:
            cur = cur.split(" —")[0].strip()
        cur = _norm_model_app(cur)
        # preserve current typed value
        self._model.blockSignals(True)
        self._model.clear()
        # add current first if not in list (normalized) — keep it so custom file paths like C:\models\foo.gguf aren't lost
        if cur and cur not in ids:
            self._model.addItem(cur, cur)
        for mid in ids:
            self._model.addItem(mid, mid)
        # Validate selection: if cur is known and in list, keep it; otherwise keep typed but warn
        if cur and cur not in ids:
            self._lbl_models.setText(f"Loaded {len(ids)} models — current '{cur}' not in provider list, but will be auto-loaded on prompt (LM Studio) if valid.")
            self._lbl_models.setStyleSheet("color:#f59e0b; font-size:11px;")
        else:
            self._lbl_models.setText(f"Loaded {len(ids)} models — smaller (1B–3B) are fastest. Select one.")
            self._lbl_models.setStyleSheet("color:#22c55e; font-size:11px;")
        # Fix: editable combo setCurrentText alone may leave currentIndex at 0 (data mismatch) -> use index
        idx = self._model.findText(cur) if cur else -1
        if idx >= 0:
            self._model.setCurrentIndex(idx)
        elif cur:
            # custom text not in list (should already be added) — ensure lineEdit shows it
            self._model.setCurrentText(cur)
        else:
            self._model.setCurrentIndex(-1)
        self._model.blockSignals(False)
        log.info("Fetched %d LLM models from %s (current %r)", len(ids), self._base_url.text().strip(), cur)

    def _on_models_err(self, err: str) -> None:
        self._btn_refresh_models.setEnabled(True)
        self._btn_refresh_models.setText("Refresh")
        self._lbl_models.setText(f"Could not fetch models: {err[:120]} — check Base URL / provider is running, or type a model id manually.")
        self._lbl_models.setStyleSheet("color:#f87171; font-size:11px;")
        log.warning("Fetch models failed: %s", err)

    def _on_voice_changed(self, _idx: int) -> None:
        # data holds pure voice id, display may be "af_heart — warm..."
        data = self._voice.currentData()
        voice = data if isinstance(data, str) and data else self._voice.currentText().split(" —")[0].strip()
        info = VOICE_INFO.get(voice, "")
        if info:
            self._lbl_voice_info.setText(info)
        else:
            self._lbl_voice_info.setText("Custom voice — ensure it matches a Kokoro voice id.")

    def _on_whisper_changed(self, _idx: int) -> None:
        data = self._whisper.currentData()
        w = data if isinstance(data, str) and data else self._whisper.currentText().split(" —")[0].strip()
        info = WHISPER_INFO.get(w, "")
        self._lbl_whisper.setText(info or "Custom Whisper model — tiny.en is fastest.")

    def _do_tts_test(self) -> None:
        self._btn_tts.setEnabled(False)
        self._lbl_tts.setText("Synthesizing…")
        engine = self._tts_engine.currentText().strip() or "kokoro"
        out_dev = self._out_dev.currentData()
        # capture UI values for worker (avoid cross-thread UI access)
        if engine == "piper":
            piper_model = self._piper_model.text().strip()
            piper_config = self._piper_config.text().strip()
            piper_speaker = int(self._piper_speaker.value())
            piper_len = float(self._piper_length.value())
            voice_label = Path(piper_model).name if piper_model else "piper"
            def _make_spk():
                from .tts import PiperSpeaker
                return PiperSpeaker(piper_model, piper_config, piper_speaker, piper_len)
        else:
            v_data = self._voice.currentData()
            voice = (v_data if isinstance(v_data, str) and v_data else self._voice.currentText().split(" —")[0].strip()) or "af_heart"
            speed = float(self._speed.value())
            voice_label = voice
            def _make_spk():
                from .tts import Speaker
                return Speaker(voice, speed)

        def worker():
            try:
                from .audio_io import Playback
                spk = _make_spk()
                audio = spk.synthesize("This is how I sound.")
                if len(audio) == 0:
                    self._safe_emit(self._test_signals.sig_tts, "No audio produced")
                    return
                pb = Playback(device=out_dev)
                pb.start()
                pb.put(audio)
                for _ in range(100):
                    if pb.idle():
                        break
                    time.sleep(0.1)
                pb.stop()
                self._safe_emit(self._test_signals.sig_tts, f"Played {len(audio)/24000:.1f}s on {voice_label} ({engine})")
            except Exception as e:
                self._safe_emit(self._test_signals.sig_tts, f"Failed: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._prune_workers()
        self._workers.append(t)

    def _do_wake_test(self) -> None:
        self._btn_wake_test.setEnabled(False)
        self._lbl_wake.setText("Listening 8s — say the wake word…")
        spec = self._wake.currentText().strip()
        thr = float(self._wake_thr.value())
        cd = int(self._wake_cd.value())
        emb = self._wake_emb.text().strip()
        in_dev = self._in_dev.currentData()

        def worker():
            try:
                from .audio_io import AudioIn
                from .wakeword import WakeWord
                ww = WakeWord(spec, thr, cd, emb)
                # warm load so timing is in the 8s window
                ww.ensure_loaded()
                q: queue.Queue = queue.Queue()
                ai = AudioIn(device=in_dev)
                ai.add_subscriber(q)
                ai.start()
                end = time.monotonic() + 8.0
                peak = 0.0
                triggers = 0
                while time.monotonic() < end:
                    try:
                        frame = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    s = ww.score(frame)
                    if s > peak:
                        peak = s
                    if ww.should_trigger(s):
                        triggers += 1
                    # throttle UI a bit
                    self._safe_emit(self._test_signals.sig_wake, f"score {s:.2f} (peak {peak:.2f}) — triggers {triggers}")
                ai.stop()
                self._safe_emit(self._test_signals.sig_wake_done, f"Done — peak {peak:.2f}, triggers {triggers} in 8s")
            except Exception as e:
                self._safe_emit(self._test_signals.sig_wake_done, f"Wake test failed: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._prune_workers()
        self._workers.append(t)

    def _do_wake_calibrate(self) -> None:
        self._btn_wake_test.setEnabled(False)
        self._btn_wake_cal.setEnabled(False)
        self._lbl_wake.setText("Calibrating 30s: 12s silence, then say wake word 3-5 times loudly…")
        spec = self._wake.currentText().strip()
        thr = float(self._wake_thr.value())
        cd = int(self._wake_cd.value())
        emb = self._wake_emb.text().strip()
        in_dev = self._in_dev.currentData()

        def worker():
            try:
                from .calibration import calibrate_wake_word
                res = calibrate_wake_word(spec, thr, cd, emb, in_dev, duration_total=30.0)
                # live-apply suggestion
                sug = res.get("suggested_threshold")
                sug_cd = res.get("suggested_cooldown_ms")
                # emit to UI thread via safe emit
                self._safe_emit(self._test_signals.sig_wake, f"Calib: noise {res.get('peak_noise'):.2f} wake {res.get('peak_wake'):.2f} gap {res.get('gap'):.2f} → thr {sug:.2f} cd {sug_cd}ms")
                # auto-apply to spinners
                try:
                    from PySide6.QtCore import QTimer
                    def apply():
                        try:
                            self._wake_thr.setValue(float(sug))
                            self._wake_cd.setValue(int(sug_cd))
                            if res.get("warning"):
                                self._lbl_wake.setToolTip(res["warning"])
                        except Exception:
                            pass
                    QTimer.singleShot(0, apply)
                except Exception:
                    pass
                self._safe_emit(self._test_signals.sig_wake_done, f"Done — suggested threshold {sug:.2f} cooldown {sug_cd}ms | {res.get('warning') or 'say Save to persist'}")
            except Exception as e:
                self._safe_emit(self._test_signals.sig_wake_done, f"Calibrate failed: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._prune_workers()
        self._workers.append(t)

    def _preview_all_sounds(self) -> None:
        self._btn_snd_test_all.setEnabled(False)
        self._lbl_snd_test.setText("Playing…")
        # capture paths
        paths = {
            "wake": self._sound_wake_le.text().strip(),
            "finished": self._sound_finished_le.text().strip(),
            "timer_set": self._sound_timer_le.text().strip(),
            "alarm": self._sound_alarm_le.text().strip(),
        }
        vol = float(self._sound_volume.value())
        enabled = bool(self._sounds_enabled.isChecked())
        out_dev = self._out_dev.currentData()
        def worker():
            try:
                from .sounds import Sounds
                from .audio_io import Playback
                from .config import Config as _Cfg
                # make temp cfg for preview
                tmp = _Cfg()
                tmp.sound_wake_path = paths["wake"]
                tmp.sound_finished_path = paths["finished"]
                tmp.sound_timer_set_path = paths["timer_set"]
                tmp.sound_alarm_path = paths["alarm"]
                tmp.sound_volume = vol
                tmp.sound_enabled = enabled
                if not enabled:
                    self._safe_emit(self._test_signals.sig_tts, "Sounds disabled — enable to preview")
                    return
                pb = Playback(device=out_dev)
                pb.start()
                s = Sounds(pb, tmp)
                for kind in ["wake", "finished", "timer_set", "alarm"]:
                    ok = s.preview(kind)
                    if not ok:
                        # try file directly if custom path failed
                        p = paths[kind]
                        if p:
                            s.preview_file(p)
                    time.sleep(0.7)
                pb.stop()
                self._safe_emit(self._test_signals.sig_tts, "Preview done")
                # update label on GUI thread
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: (self._lbl_snd_test.setText("Played wake → finished → timer → alarm"), self._btn_snd_test_all.setEnabled(True)))
            except Exception as e:
                self._safe_emit(self._test_signals.sig_tts, f"Preview failed: {e}")
                try:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self._btn_snd_test_all.setEnabled(True))
                except Exception:
                    pass
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._prune_workers()
        self._workers.append(t)

    def _refresh_alarms(self) -> None:
        try:
            from .alarms import _alarms_path
            import json
            p = _alarms_path()
            if not p.exists():
                self._alarms_list.setText("No alarms")
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            alarms = data.get("alarms", [])
            if not alarms:
                self._alarms_list.setText("No alarms")
                return
            lines = []
            for a in alarms:
                en = "on" if a.get("enabled", True) else "off"
                lines.append(f"#{a.get('id')} {a.get('label')} at {a.get('at')} ({a.get('recurrence')}) [{en}]")
            self._alarms_list.setText("\n".join(lines))
        except Exception as e:
            self._alarms_list.setText(f"Alarms error: {e}")

    def _add_alarm(self) -> None:
        t = self._alarm_time.text().strip()
        label = self._alarm_label.text().strip() or "alarm"
        rec = self._alarm_rec.currentText().strip()
        if not t:
            QMessageBox.warning(self, "Alarm", "Enter HH:MM or YYYY-MM-DDTHH:MM")
            return
        if len(t) <= 5 and ":" in t and "T" not in t:
            from datetime import datetime as _dt
            t = _dt.now().strftime("%Y-%m-%dT") + t + ":00"
        try:
            from .alarms import AlarmManager
            from .sounds import Sounds
            from .audio_io import Playback
            # ephemeral manager to add (will persist to file)
            am = AlarmManager(Sounds(Playback()))
            am.add_alarm(t, label, rec)
            am.stop()
            self._refresh_alarms()
            self._alarm_time.clear()
        except Exception as e:
            QMessageBox.warning(self, "Alarm", f"Failed: {e}")

    def _check_update(self) -> None:
        repo = self._upd_repo.text().strip()
        if not repo:
            self._lbl_upd.setText("Set owner/repo first")
            return
        self._btn_upd_check.setEnabled(False)
        self._lbl_upd.setText("Checking…")
        def worker():
            try:
                from .updater import check_for_update
                has, latest, url, notes = check_for_update(repo)
                if has:
                    self._safe_emit(self._test_signals.sig_llm, f"Update available {latest}: {url}")
                    # reuse sig_llm for label? use direct QTimer
                    from PySide6.QtCore import QTimer
                    def set_txt():
                        self._lbl_upd.setText(f"Update {latest} available: {url}")
                        self._btn_upd_check.setEnabled(True)
                    QTimer.singleShot(0, set_txt)
                else:
                    from PySide6.QtCore import QTimer
                    def set_txt2():
                        self._lbl_upd.setText(f"No update (latest {latest})" if latest else "No update / repo not found")
                        self._btn_upd_check.setEnabled(True)
                    QTimer.singleShot(0, set_txt2)
            except Exception as e:
                from PySide6.QtCore import QTimer
                def set_err():
                    self._lbl_upd.setText(f"Check failed: {e}")
                    self._btn_upd_check.setEnabled(True)
                QTimer.singleShot(0, set_err)
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._prune_workers()
        self._workers.append(t)

    def _do_update(self) -> None:
        repo = self._upd_repo.text().strip()
        if not repo:
            QMessageBox.warning(self, "Update", "Set owner/repo first (e.g. anomalyco/vocalis)")
            return
        if QMessageBox.question(self, "Update", f"Pull latest from {repo}? This runs git pull / install scripts.") != QMessageBox.Yes:
            return
        try:
            from .updater import spawn_update
            spawn_update(repo)
            QMessageBox.information(self, "Update", "Update started in background — check log and restart after it finishes.")
        except Exception as e:
            QMessageBox.warning(self, "Update", f"Failed: {e}")

    def _view_log_settings(self) -> None:
        try:
            p = log_path()
            text = ""
            if p.exists():
                data = p.read_text(encoding="utf-8", errors="replace")
                lines = data.splitlines()
                text = "\n".join(lines[-300:])
                if len(lines) > 300:
                    text = f"... ({len(lines)-300} earlier lines hidden) ...\n" + text
            else:
                text = f"No log yet at:\n{p}\n\nRun once to create it."
            dlg = QDialog(self)
            dlg.setWindowTitle("Vocalis — Log (last 300 lines)")
            dlg.resize(760, 480)
            lay = QVBoxLayout(dlg)
            hint = QLabel(f"File: {p}")
            hint.setStyleSheet("color:#94a3b8; font-size:11px;")
            hint.setWordWrap(True)
            hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lay.addWidget(hint)
            view = QPlainTextEdit()
            view.setReadOnly(True)
            view.setPlainText(text or "(empty)")
            view.setStyleSheet("background:#0b1220; color:#e2e8f0; font-family: Consolas, monospace; font-size:11px;")
            lay.addWidget(view, 1)
            row = QHBoxLayout()
            btn_copy = QPushButton("Copy")
            btn_copy.clicked.connect(lambda: (view.selectAll(), view.copy()))
            btn_open = QPushButton("Open folder")
            btn_open.clicked.connect(lambda: reveal_in_file_manager(logs_dir()))
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dlg.accept)
            row.addWidget(btn_copy)
            row.addWidget(btn_open)
            row.addStretch(1)
            row.addWidget(btn_close)
            lay.addLayout(row)
            dlg.exec()
        except Exception as e:
            QMessageBox.warning(self, "Logs", f"Could not view log: {e}\n\nPath: {log_path()}")

    def _browse_piper_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Piper voice", "", "Piper voice (*.onnx);;All files (*.*)")
        if path:
            self._piper_model.setText(path)
            # auto-fill config if sibling exists
            cfg_candidate = path + ".json"
            if Path(cfg_candidate).exists() and not self._piper_config.text().strip():
                self._piper_config.setText(cfg_candidate)

    def _browse_piper_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Piper config", "", "JSON (*.json);;All files (*.*)")
        if path:
            self._piper_config.setText(path)

    def _save(self) -> None:
        c = self.cfg
        c.llm_base_url = self._base_url.text().strip() or c.llm_base_url
        mid = self._model_id()
        if not mid:
            QMessageBox.warning(self, "Model", "Model id is empty — please select or type a model (e.g. ling-3.0-tiny-apex).")
            return
        # Normalize and keep the typed id even if not in fetched list (LM Studio may have it but not yet listed)
        c.llm_model = mid
        # Allow clearing API key (empty is valid for local providers)
        c.llm_api_key = self._api_key.text().strip()
        c.temperature = float(self._temp.value())
        c.system_prompt = self._sys.toPlainText().strip() or c.system_prompt
        # TTS engine
        c.tts_engine = self._tts_engine.currentText().strip() or "kokoro"
        # voice/whisper store pure id (from data or split display)
        _v = self._voice.currentData()
        c.tts_voice = (_v if isinstance(_v, str) and _v else self._voice.currentText().split(" —")[0].strip()) or c.tts_voice
        c.tts_speed = float(self._speed.value())
        # Piper
        c.piper_model = self._piper_model.text().strip()
        c.piper_config = self._piper_config.text().strip()
        try:
            c.piper_speaker = int(self._piper_speaker.value())
        except Exception:
            c.piper_speaker = 0
        try:
            c.piper_length_scale = float(self._piper_length.value())
        except Exception:
            c.piper_length_scale = 1.0
        c.wake_word = self._wake.currentText().strip() or c.wake_word
        c.wakeword_embeddings = self._wake_emb.text().strip()
        c.wakeword_threshold = float(self._wake_thr.value())
        c.wakeword_cooldown_ms = int(self._wake_cd.value())
        _w = self._whisper.currentData()
        c.whisper_model = (_w if isinstance(_w, str) and _w else self._whisper.currentText().split(" —")[0].strip()) or c.whisper_model
        c.input_device = self._in_dev.currentData()
        c.output_device = self._out_dev.currentData()
        c.vad_rms_dbfs = float(self._vad_db.value())
        c.vad_silence_seconds = float(self._silence.value())
        c.max_utterance_seconds = float(self._max_utt.value())
        c.idle_timeout_seconds = float(self._idle.value())
        c.preserve_history = bool(self._preserve.isChecked())
        c.compact_after = int(self._compact_spin.value())
        # keep forget_history in sync for back-compat (preserve overrides)
        c.forget_history = not c.preserve_history
        # sounds (swappable earcons)
        c.sound_wake_path = self._sound_wake_le.text().strip()
        c.sound_finished_path = self._sound_finished_le.text().strip()
        c.sound_timer_set_path = self._sound_timer_le.text().strip()
        c.sound_alarm_path = self._sound_alarm_le.text().strip()
        c.sound_volume = float(self._sound_volume.value())
        c.sound_enabled = bool(self._sounds_enabled.isChecked())
        c.alarm_tone = c.sound_alarm_path or c.alarm_tone
        # behavior
        c.continuous_listening = bool(self._cont_listen.isChecked())
        c.listen_grace_seconds = float(self._grace.value())
        # alarms / update
        c.update_repo = self._upd_repo.text().strip()
        self.accept()
        # live-apply wakeword + sounds without full restart if possible
        try:
            parent = self.parent()
            if parent and hasattr(parent, "_session") and getattr(parent, "_session", None):
                sess = getattr(parent, "_session")
                try:
                    sess.update_wakeword(threshold=c.wakeword_threshold, cooldown_ms=c.wakeword_cooldown_ms)
                except Exception:
                    pass
                try:
                    if hasattr(sess, "_sounds") and sess._sounds:
                        sess._sounds._cfg = c
                        sess._sounds.reload()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_preserve_changed(self, _state: int) -> None:
        on = self._preserve.isChecked()
        self._compact_spin.setEnabled(on)
        if on:
            self._lbl_history.setText("On: keeps last N messages; older ones are compacted into a summary so context is preserved without growing forever.")
        else:
            self._lbl_history.setText("Off: each reply forgets prior chat. On: keeps last 30 and compacts older into a summary (no LLM call).")

    def _confirm_uninstall(self, keep_models: bool) -> None:
        if keep_models:
            text = (
                "Uninstall Vocalis settings?\n\n"
                "This will delete your config (LLM URL, voice, wake word, devices) "
                "but KEEP downloaded models (~700 MB) so you can reinstall without re-downloading.\n\n"
                "The app will quit. Delete the app folder manually if you installed as portable/bundled."
            )
            title = "Uninstall — keep models"
        else:
            text = (
                "Erase EVERYTHING?\n\n"
                "This will delete config AND all downloaded models (~700 MB) from:\n"
                "  • config dir and\n"
                "  • data/models dir\n\n"
                "This cannot be undone. The app will quit."
            )
            title = "Uninstall — erase everything"
        ret = QMessageBox.warning(
            self, title, text,
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if ret != QMessageBox.Yes:
            return
        self._do_uninstall(keep_models)

    def _do_uninstall(self, keep_models: bool) -> None:
        import shutil
        from .config import config_dir, data_dir, models_dir

        # save so _save not needed — uninstall should not keep unsaved edits as config
        err = None
        try:
            cfg_p = config_dir()
            data_p = data_dir()
            mods = models_dir()
            if keep_models:
                if cfg_p.exists():
                    shutil.rmtree(cfg_p, ignore_errors=True)
                marker = mods / ".bootstrap_ok"
                try:
                    marker.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                if cfg_p.exists():
                    shutil.rmtree(cfg_p, ignore_errors=True)
                if data_p.exists():
                    shutil.rmtree(data_p, ignore_errors=True)
        except Exception as e:
            err = str(e)

        # try to launch external uninstall script detached to remove .venv/dist
        self._spawn_external_uninstall(keep_models)

        if err:
            QMessageBox.critical(self, "Uninstall error", f"Partial uninstall: {err}\nYou may need to delete folders manually.")
        else:
            QMessageBox.information(
                self, "Uninstalled",
                "Vocalis settings" + (" (models kept)" if keep_models else " and models") + " removed.\n\n"
                "The app will now quit. If you used the standalone binary, delete the Vocalis folder. "
                "If you used pip/venv, run the uninstall script or delete .venv.",
            )
        # quit app and close dialog
        self.accept()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(200, app.quit)

    def _spawn_external_uninstall(self, keep_models: bool) -> None:
        """Try to run scripts/uninstall.* detached so .venv/dist can be deleted after exit."""
        import subprocess
        import sys
        try:
            root = Path(__file__).resolve().parents[2]
            # For frozen (PyInstaller) executable, scripts are not bundled — skip
            if getattr(sys, "frozen", False):
                return
            if sys.platform.startswith("win"):
                ps1 = root / "scripts" / "uninstall.ps1"
                if ps1.exists():
                    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)]
                    if not keep_models:
                        args.append("-Clean")
                    else:
                        args.extend(["-Clean", "-KeepModels"])
                    # detached, don't wait
                    subprocess.Popen(args, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore[attr-defined]
            else:
                sh = root / "scripts" / "uninstall.sh"
                if sh.exists():
                    args = ["bash", str(sh), "--clean"]
                    if keep_models:
                        args.append("--keep-models")
                    subprocess.Popen(args, start_new_session=True)
        except Exception:
            pass

    def closeEvent(self, event):  # type: ignore[override]
        # let worker threads be daemons; just accept
        super().closeEvent(event)


class BootstrapDialog(QDialog):
    sig_stage = Signal(str, str)
    sig_done = Signal(bool, str)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.success = False
        self.setWindowTitle("Vocalis — First run")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setWindowIcon(_make_icon())
        v = QVBoxLayout(self)
        v.setSpacing(10)
        title = QLabel("Downloading models — one time, ~700 MB")
        title.setStyleSheet("font-weight:600; font-size:14px;")
        v.addWidget(title)
        hint = QLabel("This may take a few minutes on first launch. You can Skip and models will load lazily later.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8;")
        v.addWidget(hint)

        self._labels: dict[str, QLabel] = {}
        for key, name in [("wake", "Wake word"), ("stt", "Whisper (STT)"), ("tts", "Kokoro (TTS)")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            lbl = QLabel("\u2014 waiting")
            lbl.setStyleSheet("color:#64748b;")
            row.addStretch(1)
            row.addWidget(lbl)
            v.addLayout(row)
            self._labels[key] = lbl

        self._status = QLabel("")
        self._status.setStyleSheet("color:#94a3b8; font-size:11px;")
        self._status.setWordWrap(True)
        v.addWidget(self._status)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self._btn_skip = QPushButton("Skip for now")
        self._btn_skip.clicked.connect(self.reject)
        self._btn_start = QPushButton("Download now")
        self._btn_start.setDefault(True)
        self._btn_start.clicked.connect(self._start)
        btns.addWidget(self._btn_skip)
        btns.addWidget(self._btn_start)
        v.addLayout(btns)

        self.sig_stage.connect(self._on_stage)
        self.sig_done.connect(self._on_done)

    def _start(self) -> None:
        self._btn_start.setEnabled(False)
        self._btn_skip.setEnabled(False)
        self._status.setText("Starting…")

        def worker():
            try:
                from .bootstrap import ensure_models

                def cb(stage: str, detail: str):
                    self.sig_stage.emit(stage, detail)

                ensure_models(self.cfg, cb)
                self.sig_done.emit(True, "")
            except Exception as e:
                self.sig_done.emit(False, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_stage(self, stage: str, detail: str) -> None:
        if stage in self._labels:
            self._labels[stage].setText(detail)
            self._labels[stage].setStyleSheet("color:#e2e8f0;")
        if stage == "done":
            self._status.setText(detail)

    def _on_done(self, ok: bool, err: str) -> None:
        if ok:
            self.success = True
            self._status.setText("All models ready.")
            QTimer.singleShot(400, self.accept)
        else:
            self._status.setText(f"Failed: {err}")
            self._status.setStyleSheet("color:#f87171;")
            self._btn_skip.setEnabled(True)
            self._btn_skip.setText("Continue anyway")
            self._btn_start.setEnabled(True)
            self._btn_start.setText("Retry")


def launch_gui(cfg: Config) -> None:
    log.info("launch_gui entered")
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Vocalis")
    app.setWindowIcon(_make_icon())
    log.info("Log file: %s", log_path())

    win = MainWindow(cfg)
    # Keep a reference to log label update on error via signal already done in MainWindow

    # bootstrap on first run
    marker = models_dir() / ".bootstrap_ok"
    if not marker.exists():
        log.info("First run — showing BootstrapDialog")
        dlg = BootstrapDialog(cfg, win)
        try:
            dlg.exec()
        except BaseException as e:
            log.exception("BootstrapDialog crashed: %s", e)
        if dlg.success:
            try:
                marker.write_text("ok", encoding="utf-8")
                log.info("Bootstrap marker written")
            except Exception as e:
                log.warning("Could not write bootstrap marker: %s", e)
        else:
            log.info("Bootstrap skipped or failed — continuing (models will lazy-load)")

    # tray
    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(_make_icon(), app)
        menu = QMenu()
        act_show = QAction("Show / Hide", menu)
        act_settings = QAction("Settings…", menu)
        act_logs = QAction("Open Logs Folder", menu)
        act_view = QAction("View Log", menu)
        act_quit = QAction("Quit", menu)

        def toggle():
            if win.isVisible():
                win.hide()
            else:
                win.show()
                win.raise_()
                win.activateWindow()

        act_show.triggered.connect(toggle)
        act_settings.triggered.connect(win._open_settings)
        act_logs.triggered.connect(lambda: reveal_in_file_manager(logs_dir()))
        act_view.triggered.connect(win._view_log)
        act_quit.triggered.connect(lambda: (log.info("Quit via tray"), win._session.stop() if win._session else None, app.quit()))

        menu.addAction(act_show)
        menu.addAction(act_settings)
        menu.addSeparator()
        menu.addAction(act_logs)
        menu.addAction(act_view)
        menu.addSeparator()
        menu.addAction(act_quit)
        tray.setContextMenu(menu)
        tray.setToolTip("Vocalis — voice assistant")
        tray.activated.connect(lambda reason: toggle() if reason == QSystemTrayIcon.DoubleClick else None)
        try:
            tray.show()
            log.info("Tray shown")
        except Exception as e:
            log.warning("Tray show failed: %s", e)
        win._tray = tray
    else:
        log.warning("System tray not available — window close will quit")

    # start session after UI is up
    def start_session_deferred():
        from .runner import start_session
        from .watchdog import Watchdog

        try:
            log.info("Starting session deferred")
            s = start_session(cfg, win.listener())
            t = threading.Thread(target=s.run, daemon=True, name="session")
            t.start()
            win.attach_session(s)
            # keep references for watchdog
            win._session_thread = t  # type: ignore
            # watchdog auto-restarts without tray intervention
            wd = Watchdog(cfg, lambda: win, start_session)
            wd.attach(s, t)
            wd.start()
            win._watchdog = wd  # type: ignore
            # keep watchdog's session ref fresh after restart
            def _sync_watchdog():
                try:
                    if hasattr(win, "_watchdog") and hasattr(win, "_session"):
                        wd.attach(win._session, getattr(win, "_session_thread", t))  # type: ignore
                except Exception:
                    pass
            sync_timer = QTimer()
            sync_timer.timeout.connect(_sync_watchdog)
            sync_timer.start(2000)
            win._wd_sync_timer = sync_timer  # type: ignore
            log.info("Session thread started with watchdog")
            # background update check (offline-only otherwise, but update needs net)
            try:
                if getattr(cfg, "auto_update_check", False) and getattr(cfg, "update_repo", ""):
                    def _bg_upd():
                        try:
                            from .updater import check_for_update
                            has, latest, url, _ = check_for_update(cfg.update_repo)
                            if has:
                                log.info("Update available %s at %s", latest, url)
                                try:
                                    QTimer.singleShot(0, lambda: win._on_error(f"Update {latest} available: {url} — Settings → Updates"))
                                    if tray:
                                        tray.showMessage("Vocalis update", f"{latest} available", QSystemTrayIcon.Information, 5000)
                                except Exception:
                                    pass
                        except Exception as e:
                            log.debug("Background update check failed: %s", e)
                    threading.Thread(target=_bg_upd, daemon=True).start()
            except Exception:
                pass
        except BaseException as e:
            log.exception("Audio start failed: %s", e)
            # Keep window open with error + log path, don't quit
            win._on_error(f"Audio start failed: {e}. Check Settings -> audio devices. Log: {log_path()}")
            try:
                tray.showMessage("Vocalis", f"Audio failed: {e} — see log {log_path()}", QSystemTrayIcon.Critical, 6000)  # type: ignore[attr-defined]
            except Exception:
                pass

    QTimer.singleShot(120, start_session_deferred)

    log.info("Showing main window — ready")
    win.show()
    try:
        rc = app.exec()
        log.info("GUI loop exited with %s", rc)
    except BaseException as e:
        log.critical("GUI loop crashed: %s", e, exc_info=True)
        raise
    finally:
        try:
            from .logger import flush
            flush()
        except Exception:
            pass
