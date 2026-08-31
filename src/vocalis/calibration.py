from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .logger import get_logger

log = get_logger(__name__)

def calibrate_wake_word(spec: str, threshold: float, cooldown_ms: int, embeddings: str, device: int | None, duration_total: float = 30.0) -> dict:
    """Run 2-phase calibration: noise floor + wake phrase.

    Returns dict with suggested_threshold, suggested_cooldown, stats.
    Caller is responsible for persisting to config.
    """
    from .audio_io import AudioIn
    from .wakeword import WakeWord

    ww = WakeWord(spec, threshold, cooldown_ms, embeddings)
    ww.ensure_loaded()

    q: queue.Queue = queue.Queue()
    ai = AudioIn(device=device)
    ai.add_subscriber(q)
    ai.start()

    try:
        # Phase 1: silence / background 12s
        t0 = time.monotonic()
        noise_scores: list[float] = []
        while time.monotonic() - t0 < 12.0:
            try:
                frame = q.get(timeout=0.3)
            except queue.Empty:
                continue
            s = ww.score(frame)
            noise_scores.append(s)

        peak_noise = max(noise_scores) if noise_scores else 0.0
        p95_noise = float(np.percentile(noise_scores, 95)) if noise_scores else 0.0
        median_noise = float(np.median(noise_scores)) if noise_scores else 0.0

        # Phase 2: user says wake word repeatedly for ~18s
        # We collect scores and count how many times it would have triggered at current threshold
        wake_scores: list[float] = []
        peak_wake = peak_noise
        # drain
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        t1 = time.monotonic()
        while time.monotonic() - t1 < 18.0:
            try:
                frame = q.get(timeout=0.3)
            except queue.Empty:
                continue
            s = ww.score(frame)
            wake_scores.append(s)
            if s > peak_wake:
                peak_wake = s

        # Combine
        all_scores = noise_scores + wake_scores
        p99 = float(np.percentile(all_scores, 99)) if all_scores else 0.0

        # Suggestion: halfway between peaks, clamped, with margin above noise
        # If user never said wake word, peak_wake == peak_noise => suggest slightly above noise
        if peak_wake <= peak_noise + 0.05:
            # no clear wake signal
            suggested = min(0.85, max(0.25, peak_noise + 0.25))
            warning = "No clear wake word detected — say 'Hey Jarvis' loudly 3-5 times during phase 2. Suggestion is conservative."
            gap = 0.0
        else:
            mid = (peak_noise + peak_wake) / 2
            # ensure at least 0.12 above p95 noise to avoid false wakes
            margin_above_noise = p95_noise + 0.12
            suggested = max(mid, margin_above_noise)
            suggested = float(max(0.2, min(0.85, suggested)))
            gap = float(peak_wake - peak_noise)
            warning = None
            if gap < 0.2:
                warning = f"Small gap ({gap:.2f}) between noise and wake — room may be noisy. Try quieter room."

        # Cooldown suggestion: if triggers were frequent, increase; else keep
        # estimate triggers at suggested threshold
        triggers_at_suggested = sum(1 for s in wake_scores if s >= suggested)
        if triggers_at_suggested > 8:
            suggested_cd = min(1500, cooldown_ms + 200)
        elif triggers_at_suggested == 0 and gap > 0.3:
            suggested_cd = max(400, cooldown_ms - 100)
        else:
            suggested_cd = cooldown_ms

        result = {
            "peak_noise": round(float(peak_noise), 3),
            "p95_noise": round(float(p95_noise), 3),
            "median_noise": round(float(median_noise), 3),
            "peak_wake": round(float(peak_wake), 3),
            "p99": round(float(p99), 3),
            "gap": round(float(gap), 3),
            "samples_noise": len(noise_scores),
            "samples_wake": len(wake_scores),
            "suggested_threshold": round(float(suggested), 3),
            "suggested_cooldown_ms": int(suggested_cd),
            "warning": warning,
            "triggers_at_suggested": int(triggers_at_suggested),
        }
        log.info("Calibration result: %s", result)
        return result
    finally:
        try:
            ai.stop()
        except Exception:
            pass
