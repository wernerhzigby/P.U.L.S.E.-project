import os
import time
import math
from dataclasses import dataclass, field
from collections import deque, defaultdict

CARDIAC_EVENTS = {
    "Bradycardia",
    "Tachycardia",
    "Ventricular Tachycardia",
    "Supraventricular Tachycardia (possible)",
    "Asystole / Flatline",
    "Pause / Sinus Arrest (possible)",
    "Irregular Rhythm",
    "Sinus Arrhythmia (possible)",
    "Atrial Fibrillation (possible)",
    "Sinus Node Dysfunction",
    "First-Degree AV Block (possible)",
    "Bundle Branch Block (possible)",
    "Long QT (possible)",
    "Short QT (possible)",
    "Early Repolarization / ST Elevation (possible)",
    "Premature Ventricular Contraction (PVC) (possible)",
    "Premature Atrial Contraction (PAC) (possible)",
    "Bigeminy (possible)",
    "Trigeminy (possible)",
    "Frequent Ectopy (possible)",
    "Myocarditis (possible)",
}

SIGNAL_EVENTS = {
    "Low Signal Amplitude",
    "High Noise / Motion Artifact",
    "Baseline Wander",
    "Signal Saturation / Clipping",
    "Lead Off (possible)",
}


@dataclass
class ECGConfig:
    sample_rate: int = int(os.getenv("ECG_SAMPLE_RATE", "250"))
    r_threshold: int = int(os.getenv("ECG_R_THRESHOLD", "15000"))
    brady_bpm: int = int(os.getenv("ECG_BRADY_BPM", "50"))
    tachy_bpm: int = int(os.getenv("ECG_TACHY_BPM", "100"))
    vtach_bpm: int = int(os.getenv("ECG_VTACH_BPM", "150"))
    asystole_sec: float = float(os.getenv("ECG_ASYSTOLE_SEC", "3.5"))
    buffer_sec: int = int(os.getenv("ECG_BUFFER_SEC", "120"))
    bpm_maxlen: int = int(os.getenv("ECG_BPM_MAXLEN", "1200"))
    rr_maxlen: int = int(os.getenv("ECG_RR_MAXLEN", "120"))
    qrs_maxlen: int = int(os.getenv("ECG_QRS_MAXLEN", "60"))
    qt_maxlen: int = int(os.getenv("ECG_QT_MAXLEN", "60"))
    baseline_window_sec: float = float(os.getenv("ECG_BASELINE_WINDOW_SEC", "1.0"))
    noise_window_sec: float = float(os.getenv("ECG_NOISE_WINDOW_SEC", "2.0"))
    min_rr_sec: float = float(os.getenv("ECG_MIN_RR_SEC", "0.25"))
    premature_ratio_short: float = float(os.getenv("ECG_PREMATURE_SHORT", "0.8"))
    premature_ratio_long: float = float(os.getenv("ECG_PREMATURE_LONG", "1.2"))
    low_amp_threshold: int = int(os.getenv("ECG_LOW_AMP_THRESHOLD", "800"))
    noise_derivative_threshold: int = int(os.getenv("ECG_NOISE_DERIV_THRESHOLD", "1200"))
    baseline_wander_threshold: int = int(os.getenv("ECG_BASELINE_WANDER_THRESHOLD", "1500"))
    clip_low: int = int(os.getenv("ECG_CLIP_LOW", "0"))
    clip_high: int = int(os.getenv("ECG_CLIP_HIGH", "32767"))

    @property
    def ecg_maxlen(self) -> int:
        return max(1000, self.sample_rate * self.buffer_sec)

    @property
    def baseline_window_len(self) -> int:
        return max(5, int(self.sample_rate * self.baseline_window_sec))

    @property
    def noise_window_len(self) -> int:
        return max(10, int(self.sample_rate * self.noise_window_sec))


@dataclass
class ECGState:
    config: ECGConfig
    ecg_data: deque = field(init=False)
    timestamps: deque = field(init=False)
    bpm_history: deque = field(init=False)
    bpm_timestamps: deque = field(init=False)
    rr_intervals: deque = field(init=False)
    qrs_widths: deque = field(init=False)
    qt_intervals: deque = field(init=False)
    filtered_data: deque = field(init=False)
    baseline_window: deque = field(init=False)
    derivative_window: deque = field(init=False)
    premature_flags: deque = field(init=False)
    event_state: dict = field(default_factory=dict)
    event_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    event_timeline: deque = field(init=False)
    current_bpm: int = 0
    last_peak_time: float | None = None
    last_peak_value: int | None = None
    last_signal_time: float = field(default_factory=time.time)
    last_filtered: float = 0.0

    def __post_init__(self) -> None:
        self.ecg_data = deque(maxlen=self.config.ecg_maxlen)
        self.timestamps = deque(maxlen=self.config.ecg_maxlen)
        self.bpm_history = deque(maxlen=self.config.bpm_maxlen)
        self.bpm_timestamps = deque(maxlen=self.config.bpm_maxlen)
        self.rr_intervals = deque(maxlen=self.config.rr_maxlen)
        self.qrs_widths = deque(maxlen=self.config.qrs_maxlen)
        self.qt_intervals = deque(maxlen=self.config.qt_maxlen)
        self.filtered_data = deque(maxlen=self.config.ecg_maxlen)
        self.baseline_window = deque(maxlen=self.config.baseline_window_len)
        self.derivative_window = deque(maxlen=self.config.noise_window_len)
        self.premature_flags = deque(maxlen=30)
        self.event_timeline = deque(maxlen=self.config.ecg_maxlen)

    def reset(self) -> None:
        self.ecg_data.clear()
        self.timestamps.clear()
        self.bpm_history.clear()
        self.bpm_timestamps.clear()
        self.rr_intervals.clear()
        self.qrs_widths.clear()
        self.qt_intervals.clear()
        self.filtered_data.clear()
        self.baseline_window.clear()
        self.derivative_window.clear()
        self.premature_flags.clear()
        self.event_state.clear()
        self.event_counts.clear()
        self.event_timeline.clear()
        self.current_bpm = 0
        self.last_peak_time = None
        self.last_peak_value = None
        self.last_signal_time = time.time()
        self.last_filtered = 0.0

    def set_event(self, name: str, condition: bool) -> None:
        if condition:
            self.event_state[name] = True
            self.event_counts[name] += 1
        else:
            self.event_state.pop(name, None)

    def active_flags(self) -> list[str]:
        return list(self.event_state.keys())

    def _baseline(self) -> float:
        if not self.baseline_window:
            return 0.0
        return sum(self.baseline_window) / len(self.baseline_window)

    def _compute_rr_stats(self) -> dict:
        if len(self.rr_intervals) < 2:
            return {"mean": None, "variance": None, "sdnn": None, "rmssd": None, "pnn50": None}
        mean_rr = sum(self.rr_intervals) / len(self.rr_intervals)
        variance = sum((r - mean_rr) ** 2 for r in self.rr_intervals) / len(self.rr_intervals)
        sdnn = math.sqrt(variance)
        diffs = [abs(self.rr_intervals[i] - self.rr_intervals[i - 1]) for i in range(1, len(self.rr_intervals))]
        rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
        pnn50 = sum(1 for d in diffs if d > 0.05) / len(diffs) if diffs else 0.0
        return {"mean": mean_rr, "variance": variance, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50}

    def _compute_signal_metrics(self) -> dict:
        window = list(self.filtered_data)[-self.config.noise_window_len :]
        if len(window) < 5:
            return {"range": 0.0, "stdev": 0.0, "deriv": 0.0, "baseline_range": 0.0}
        mean = sum(window) / len(window)
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        stdev = math.sqrt(variance)
        amp_range = max(window) - min(window)
        deriv = sum(self.derivative_window) / len(self.derivative_window) if self.derivative_window else 0.0
        baseline_range = 0.0
        if self.baseline_window:
            baseline_range = max(self.baseline_window) - min(self.baseline_window)
        return {"range": amp_range, "stdev": stdev, "deriv": deriv, "baseline_range": baseline_range}

    def add_sample(self, value: int, t: float) -> None:
        self.ecg_data.append(value)
        self.timestamps.append(t)

        self.baseline_window.append(value)
        baseline = self._baseline()
        filtered = value - baseline
        self.filtered_data.append(filtered)

        deriv = abs(filtered - self.last_filtered)
        self.derivative_window.append(deriv)
        self.last_filtered = filtered

        if value > self.config.r_threshold:
            if self.last_peak_time:
                rr = t - self.last_peak_time
                if rr > self.config.min_rr_sec:
                    self.rr_intervals.append(rr)
                    bpm = 60 / rr
                    self.current_bpm = int(bpm)
                    self.bpm_history.append(self.current_bpm)
                    self.bpm_timestamps.append(t)

                    self.qt_intervals.append(rr * 0.45)
                    amplitude = abs(value - self.config.r_threshold)
                    self.qrs_widths.append(0.08 + amplitude / 100000)

                    stats = self._compute_rr_stats()
                    mean_rr = stats["mean"] or rr
                    is_premature = rr < mean_rr * self.config.premature_ratio_short
                    self.premature_flags.append(is_premature)

                    if is_premature:
                        self.last_peak_value = value
            self.last_peak_time = t
            self.last_signal_time = t

        self.detect_events(value, t)
        self.event_timeline.append(",".join(self.active_flags()))

    def detect_events(self, value: int, now: float) -> None:
        self.set_event("Bradycardia", self.current_bpm and self.current_bpm < self.config.brady_bpm)
        self.set_event("Tachycardia", self.current_bpm and self.current_bpm > self.config.tachy_bpm)
        self.set_event(
            "Ventricular Tachycardia",
            self.current_bpm and self.current_bpm > self.config.vtach_bpm,
        )
        self.set_event(
            "Supraventricular Tachycardia (possible)",
            self.current_bpm and self.current_bpm > 160,
        )

        self.set_event("Asystole / Flatline", now - self.last_signal_time > self.config.asystole_sec)

        rr_stats = self._compute_rr_stats()
        mean_rr = rr_stats["mean"]
        variance = rr_stats["variance"]
        rmssd = rr_stats["rmssd"]
        pnn50 = rr_stats["pnn50"]
        sdnn = rr_stats["sdnn"]

        if mean_rr and variance is not None:
            self.set_event("Irregular Rhythm", variance > 0.02)
            self.set_event("Sinus Node Dysfunction", variance > 0.03 and mean_rr > 1.2)
            self.set_event("First-Degree AV Block (possible)", mean_rr > 1.0 and variance < 0.005)

        if mean_rr and rmssd is not None and sdnn is not None:
            self.set_event(
                "Sinus Arrhythmia (possible)",
                rmssd > 0.08 and 0.6 < mean_rr < 1.2 and sdnn > 0.05,
            )
            self.set_event(
                "Atrial Fibrillation (possible)",
                rmssd > 0.12 and pnn50 is not None and pnn50 > 0.4 and self.current_bpm < 160,
            )

        if len(self.rr_intervals) > 3:
            self.set_event(
                "Pause / Sinus Arrest (possible)",
                max(self.rr_intervals) > 2.5,
            )

        if len(self.qrs_widths) > 5:
            self.set_event(
                "Bundle Branch Block (possible)",
                sum(self.qrs_widths) / len(self.qrs_widths) > 0.14,
            )

        if len(self.qt_intervals) > 5:
            avg_qt = sum(self.qt_intervals) / len(self.qt_intervals)
            self.set_event("Long QT (possible)", avg_qt > 0.48)
            self.set_event("Short QT (possible)", avg_qt < 0.32)

        self.set_event(
            "Early Repolarization / ST Elevation (possible)",
            value > self.config.r_threshold * 1.25 and self.current_bpm < 100,
        )

        if len(self.rr_intervals) > 6 and mean_rr:
            recent_rr = list(self.rr_intervals)[-6:]
            premature = [rr < mean_rr * self.config.premature_ratio_short for rr in recent_rr]
            compensatory = [rr > mean_rr * self.config.premature_ratio_long for rr in recent_rr]
            pvc_like = any(premature[i] and compensatory[i + 1] for i in range(len(premature) - 1))
            pac_like = any(premature[i] and not compensatory[i + 1] for i in range(len(premature) - 1))
            self.set_event("Premature Ventricular Contraction (PVC) (possible)", pvc_like)
            self.set_event("Premature Atrial Contraction (PAC) (possible)", pac_like)

            if len(premature) >= 6:
                self.set_event(
                    "Bigeminy (possible)",
                    all(premature[i] != premature[i + 1] for i in range(5)),
                )
                self.set_event(
                    "Trigeminy (possible)",
                    premature[0] and not premature[1] and not premature[2] and premature[3],
                )

            ectopy_rate = sum(premature) / len(premature) if premature else 0.0
            self.set_event("Frequent Ectopy (possible)", ectopy_rate > 0.2)

        myocarditis_score = 0
        if "Tachycardia" in self.event_state:
            myocarditis_score += 1
        if "Irregular Rhythm" in self.event_state:
            myocarditis_score += 1
        if "Early Repolarization / ST Elevation (possible)" in self.event_state:
            myocarditis_score += 1
        self.set_event("Myocarditis (possible)", myocarditis_score >= 2)

        metrics = self._compute_signal_metrics()
        self.set_event("Low Signal Amplitude", metrics["range"] < self.config.low_amp_threshold)
        self.set_event("High Noise / Motion Artifact", metrics["deriv"] > self.config.noise_derivative_threshold)
        self.set_event("Baseline Wander", metrics["baseline_range"] > self.config.baseline_wander_threshold)
        self.set_event("Signal Saturation / Clipping", value <= self.config.clip_low or value >= self.config.clip_high)
        self.set_event(
            "Lead Off (possible)",
            metrics["range"] < self.config.low_amp_threshold and "Asystole / Flatline" in self.event_state,
        )
