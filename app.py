import io
import os
import csv
import json
import math
import time
import random
import zipfile
import threading
import subprocess
import re

from flask import Flask, render_template, jsonify, send_file, request
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from ecg_core import ECGConfig, ECGState, CARDIAC_EVENTS, SIGNAL_EVENTS

try:
    import board
    import busio
    from adafruit_ads1x15.ads1115 import ADS1115
    from adafruit_ads1x15.analog_in import AnalogIn
    HARDWARE_AVAILABLE = True
except Exception:
    HARDWARE_AVAILABLE = False

SAMPLE_WINDOW = 5
RESET_LOCK = threading.Lock()
REPORT_CACHE = {"timestamp": 0.0, "signature": None, "payload": None, "public_until": 0.0}

# Mailgun configuration (set directly here)
MAILGUN_API_KEY = "YOUR_MAILGUN_API_KEY"
MAILGUN_DOMAIN = "YOUR_MAILGUN_DOMAIN"
MAILGUN_FROM = "ECG Monitor <postmaster@YOUR_MAILGUN_DOMAIN>"

app = Flask(__name__)

config = ECGConfig()
state = ECGState(config)


# ================= HARDWARE / SIM =================

def init_adc():
    if not HARDWARE_AVAILABLE:
        return None
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS1115(i2c)
        return AnalogIn(ads, 0)
    except Exception:
        return None


def simulate_sample(t: float) -> int:
    # Simple ECG-like waveform: baseline + sinusoid + periodic spikes
    base = 10000 + 1000 * math.sin(2 * math.pi * 1.2 * t)
    noise = random.uniform(-200, 200)
    spike = 7000 if (t % 0.8) < 0.02 else 0
    return int(base + noise + spike)


chan = init_adc()
HARDWARE_READY = chan is not None
SIMULATE = os.getenv("ECG_SIMULATE", "0") == "1" or not HARDWARE_READY


# ================= ECG LOOP =================

def ecg_loop():
    while True:
        with RESET_LOCK:
            t = time.time()
            if SIMULATE:
                val = simulate_sample(t)
            else:
                val = chan.value
            state.add_sample(val, t)

        time.sleep(1 / config.sample_rate)


# ================= HELPERS =================

def smooth_series(values: list[int], window: int) -> list[float]:
    if not values:
        return []
    smoothed = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
            smoothed.append(running / window)
        else:
            smoothed.append(running / (i + 1))
    return smoothed


def shutdown_allowed(req) -> bool:
    token = os.getenv("ECG_SHUTDOWN_TOKEN")
    if not token:
        return False
    header = req.headers.get("X-ECG-Token")
    query = req.args.get("token")
    return header == token or query == token


def compute_signal_metrics(state: ECGState) -> dict:
    window = list(state.filtered_data)[-state.config.noise_window_len :]
    if len(window) < 5:
        return {"range": 0.0, "stdev": 0.0, "deriv": 0.0, "baseline_range": 0.0}
    mean = sum(window) / len(window)
    variance = sum((v - mean) ** 2 for v in window) / len(window)
    stdev = math.sqrt(variance)
    amp_range = max(window) - min(window)
    deriv = sum(state.derivative_window) / len(state.derivative_window) if state.derivative_window else 0.0
    baseline_range = 0.0
    if state.baseline_window:
        baseline_range = max(state.baseline_window) - min(state.baseline_window)
    return {"range": amp_range, "stdev": stdev, "deriv": deriv, "baseline_range": baseline_range}


def signal_quality_summary(state: ECGState) -> dict:
    metrics = compute_signal_metrics(state)
    cfg = state.config
    last_value = state.ecg_data[-1] if state.ecg_data else 0
    low_amp = metrics["range"] < cfg.low_amp_threshold
    noise = metrics["deriv"] > cfg.noise_derivative_threshold
    wander = metrics["baseline_range"] > cfg.baseline_wander_threshold
    clipping = last_value <= cfg.clip_low or last_value >= cfg.clip_high
    if not state.ecg_data:
        level = "No Data"
    elif clipping or low_amp:
        level = "Poor"
    elif noise or wander:
        level = "Fair"
    else:
        level = "Good"
    return {
        "level": level,
        "low_amplitude": low_amp,
        "noise": noise,
        "baseline_wander": wander,
        "clipping": clipping,
        "metrics": metrics,
    }


def session_summary(state: ECGState) -> dict:
    bpm_values = list(state.bpm_history)
    timestamps = list(state.timestamps)
    duration = 0.0
    if len(timestamps) >= 2:
        duration = max(0.0, timestamps[-1] - timestamps[0])
    return {
        "bpm_avg": round(sum(bpm_values) / len(bpm_values), 1) if bpm_values else None,
        "bpm_min": min(bpm_values) if bpm_values else None,
        "bpm_max": max(bpm_values) if bpm_values else None,
        "samples": len(timestamps),
        "duration_sec": round(duration, 1),
        "active_events": len(state.event_state),
        "total_events": int(sum(state.event_counts.values())),
    }


def explain_event(event_name: str, cfg: ECGConfig) -> str:
    explanations = {
        "Bradycardia": (
            f"Heart rate fell below {cfg.brady_bpm} BPM, indicating an unusually slow rhythm."
        ),
        "Tachycardia": (
            f"Heart rate exceeded {cfg.tachy_bpm} BPM, indicating a faster than normal rhythm."
        ),
        "Ventricular Tachycardia": (
            f"Heart rate exceeded {cfg.vtach_bpm} BPM, flagged as a possible ventricular-origin fast rhythm."
        ),
        "Supraventricular Tachycardia (possible)": (
            "Sustained heart rate above ~160 BPM suggests a possible supraventricular rapid rhythm."
        ),
        "Asystole / Flatline": (
            f"No detected peaks for > {cfg.asystole_sec:.1f} seconds, indicating a possible pause or loss of signal."
        ),
        "Pause / Sinus Arrest (possible)": (
            "Longest RR interval exceeded 2.5 seconds, suggesting a prolonged pause between beats."
        ),
        "Irregular Rhythm": (
            "Beat-to-beat RR interval variance was elevated, indicating irregular timing between beats."
        ),
        "Sinus Arrhythmia (possible)": (
            "High short-term variability with normal mean rate suggests respiratory-linked sinus arrhythmia."
        ),
        "Atrial Fibrillation (possible)": (
            "High short-term variability (RMSSD) and frequent large RR changes (pNN50) indicate an irregularly irregular rhythm."
        ),
        "Sinus Node Dysfunction": (
            "Slow mean RR interval with high variability suggests inconsistent sinus node pacing."
        ),
        "First-Degree AV Block (possible)": (
            "Long mean RR interval with low variability suggests delayed conduction with otherwise regular rhythm."
        ),
        "Bundle Branch Block (possible)": (
            "Average QRS width exceeded 0.14s, indicating widened ventricular depolarization."
        ),
        "Long QT (possible)": (
            "Average QT interval exceeded 0.48s, indicating prolonged repolarization."
        ),
        "Short QT (possible)": (
            "Average QT interval below 0.32s, indicating shortened repolarization."
        ),
        "Early Repolarization / ST Elevation (possible)": (
            "R-peak amplitude exceeded 1.25x threshold at normal rates, suggesting elevated ST morphology."
        ),
        "Premature Ventricular Contraction (PVC) (possible)": (
            "Premature beat followed by a compensatory longer interval suggests a ventricular ectopic beat."
        ),
        "Premature Atrial Contraction (PAC) (possible)": (
            "Premature beat without a compensatory pause suggests a supraventricular ectopic beat."
        ),
        "Bigeminy (possible)": (
            "Alternating premature and normal beats detected in the recent rhythm pattern."
        ),
        "Trigeminy (possible)": (
            "Recurring pattern of one premature beat every three beats detected."
        ),
        "Frequent Ectopy (possible)": (
            "More than 20% of recent beats were premature, indicating frequent ectopic activity."
        ),
        "Myocarditis (possible)": (
            "Heuristic combination of tachycardia, irregular rhythm, and ST elevation flags suggests myocarditis."
        ),
        "Low Signal Amplitude": (
            f"Signal range below {cfg.low_amp_threshold} units indicates weak or attenuated signal."
        ),
        "High Noise / Motion Artifact": (
            f"Rapid signal changes above {cfg.noise_derivative_threshold} units indicate motion or electrical noise."
        ),
        "Baseline Wander": (
            f"Baseline drift exceeded {cfg.baseline_wander_threshold} units, suggesting electrode or motion drift."
        ),
        "Signal Saturation / Clipping": (
            f"Signal hit ADC limits ({cfg.clip_low} to {cfg.clip_high}), indicating clipping or saturation."
        ),
        "Lead Off (possible)": (
            "Very low amplitude combined with flatline suggests an electrode lead may be disconnected."
        ),
    }
    return explanations.get(event_name, "No explanation available for this event.")


def event_detection_logic(event_name: str, cfg: ECGConfig) -> str:
    logic = {
        "Bradycardia": f"BPM < {cfg.brady_bpm}",
        "Tachycardia": f"BPM > {cfg.tachy_bpm}",
        "Ventricular Tachycardia": f"BPM > {cfg.vtach_bpm}",
        "Supraventricular Tachycardia (possible)": "BPM > 160",
        "Asystole / Flatline": f"No peaks for > {cfg.asystole_sec:.1f}s",
        "Pause / Sinus Arrest (possible)": "Max RR interval > 2.5s",
        "Irregular Rhythm": "RR variance > 0.02",
        "Sinus Arrhythmia (possible)": "RMSSD > 0.08 and SDNN > 0.05 with normal mean RR",
        "Atrial Fibrillation (possible)": "RMSSD > 0.12 and pNN50 > 0.4 with BPM < 160",
        "Sinus Node Dysfunction": "RR variance > 0.03 and mean RR > 1.2s",
        "First-Degree AV Block (possible)": "Mean RR > 1.0s and variance < 0.005",
        "Bundle Branch Block (possible)": "Avg QRS width > 0.14s",
        "Long QT (possible)": "Avg QT > 0.48s",
        "Short QT (possible)": "Avg QT < 0.32s",
        "Early Repolarization / ST Elevation (possible)": "Peak > 1.25x threshold and BPM < 100",
        "Premature Ventricular Contraction (PVC) (possible)": "Premature beat followed by compensatory pause",
        "Premature Atrial Contraction (PAC) (possible)": "Premature beat without compensatory pause",
        "Bigeminy (possible)": "Alternating premature/normal beats over recent window",
        "Trigeminy (possible)": "Premature every third beat in recent window",
        "Frequent Ectopy (possible)": "Premature beats > 20% of recent beats",
        "Myocarditis (possible)": "Combination of tachycardia, irregular rhythm, and ST elevation",
        "Low Signal Amplitude": f"Range < {cfg.low_amp_threshold}",
        "High Noise / Motion Artifact": f"Derivative > {cfg.noise_derivative_threshold}",
        "Baseline Wander": f"Baseline drift > {cfg.baseline_wander_threshold}",
        "Signal Saturation / Clipping": f"Signal <= {cfg.clip_low} or >= {cfg.clip_high}",
        "Lead Off (possible)": "Low amplitude + flatline present",
    }
    return logic.get(event_name, "Heuristic detection rule.")


def build_report_zip() -> bytes:
    with RESET_LOCK:
        ecg_data = list(state.ecg_data)
        timestamps = list(state.timestamps)
        bpm_history = list(state.bpm_history)
        bpm_timestamps = list(state.bpm_timestamps)
        event_timeline = list(state.event_timeline)
        event_counts = dict(state.event_counts)
        quality = signal_quality_summary(state)
        summary = session_summary(state)

    signature = (
        len(ecg_data),
        len(bpm_history),
        int(sum(event_counts.values())),
        timestamps[-1] if timestamps else 0,
    )
    now = time.time()
    if (
        REPORT_CACHE["payload"]
        and REPORT_CACHE["signature"] == signature
        and (now - REPORT_CACHE["timestamp"]) < 5
    ):
        return REPORT_CACHE["payload"]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        ecg_csv = io.StringIO()
        writer = csv.writer(ecg_csv)
        writer.writerow(["timestamp", "ecg_value", "cardiac_flags"])
        for t, v, f in zip(timestamps, ecg_data, event_timeline):
            writer.writerow([t, v, f])
        zipf.writestr("ecg_data_with_flags.csv", ecg_csv.getvalue())

        bpm_csv = io.StringIO()
        writer = csv.writer(bpm_csv)
        writer.writerow(["timestamp", "bpm"])
        for t, b in zip(bpm_timestamps, bpm_history):
            writer.writerow([t, b])
        zipf.writestr("bpm_data.csv", bpm_csv.getvalue())

        if ecg_data:
            plt.figure(figsize=(6, 3))
            plt.plot(ecg_data[-1000:])
            plt.title("ECG Snapshot")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            zipf.writestr("ecg_snapshot.png", buf.read())

        if bpm_history:
            plt.figure(figsize=(6, 2))
            plt.plot(bpm_history[-300:])
            plt.title("BPM Over Time")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            zipf.writestr("bpm_snapshot.png", buf.read())

        report_payload = {
            "summary": summary,
            "signal_quality": quality,
            "thresholds": {
                "sample_rate": config.sample_rate,
                "r_threshold": config.r_threshold,
                "brady_bpm": config.brady_bpm,
                "tachy_bpm": config.tachy_bpm,
                "vtach_bpm": config.vtach_bpm,
                "asystole_sec": config.asystole_sec,
                "low_amp_threshold": config.low_amp_threshold,
                "noise_derivative_threshold": config.noise_derivative_threshold,
                "baseline_wander_threshold": config.baseline_wander_threshold,
                "clip_low": config.clip_low,
                "clip_high": config.clip_high,
            },
            "events": [],
        }

        total = max(sum(event_counts.values()), 1)
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        for event_name, count in sorted_events:
            if event_name in CARDIAC_EVENTS or event_name in SIGNAL_EVENTS:
                category = "Cardiac" if event_name in CARDIAC_EVENTS else "Signal"
                report_payload["events"].append(
                    {
                        "name": event_name,
                        "category": category,
                        "count": count,
                        "percent": round((count / total) * 100, 1),
                        "explanation": explain_event(event_name, config),
                        "logic": event_detection_logic(event_name, config),
                    }
                )

        zipf.writestr("report.json", json.dumps(report_payload, indent=2))

        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("ECG Monitoring Summary", styles["Title"]))
        elements.append(Spacer(1, 10))
        elements.append(
            Paragraph(
                "Event detection is heuristic and for educational use only. "
                "It is not a clinical diagnosis.",
                styles["Italic"],
            )
        )
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Session Summary", styles["Heading2"]))
        duration_txt = f"{summary['duration_sec']}s" if summary["duration_sec"] else "n/a"
        elements.append(
            Paragraph(
                f"Duration: {duration_txt}. Samples: {summary['samples']}. "
                f"Avg BPM: {summary['bpm_avg'] or 'n/a'}, "
                f"Min BPM: {summary['bpm_min'] or 'n/a'}, "
                f"Max BPM: {summary['bpm_max'] or 'n/a'}.",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 8))

        elements.append(Paragraph("Signal Quality", styles["Heading2"]))
        elements.append(
            Paragraph(
                f"Quality: {quality['level']}. "
                f"Low amplitude: {quality['low_amplitude']}, "
                f"Noise: {quality['noise']}, "
                f"Baseline wander: {quality['baseline_wander']}, "
                f"Clipping: {quality['clipping']}.",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 8))

        elements.append(Paragraph("Event Breakdown", styles["Heading2"]))
        signal_events = [e for e in report_payload["events"] if e["category"] == "Signal"]
        cardiac_events = [e for e in report_payload["events"] if e["category"] == "Cardiac"]

        elements.append(Paragraph("Signal Quality Flags", styles["Heading3"]))
        if not signal_events:
            elements.append(Paragraph("No signal quality flags detected.", styles["Normal"]))
        for event in signal_events:
            pct = event["percent"]
            concern = "Normal"
            if pct > 20:
                concern = "Elevated"
            if pct > 40:
                concern = "High"
            elements.append(Paragraph(f"{event['name']}: {pct:.1f}% - {concern}", styles["Normal"]))
            elements.append(
                Paragraph(
                    "Explanation: "
                    f"{event['explanation']} "
                    f"How detected: {event['logic']}. "
                    "Higher percentages suggest this pattern appeared more often during the session.",
                    styles["Italic"],
                )
            )
            elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 4))
        elements.append(Paragraph("Cardiac Rhythm Flags", styles["Heading3"]))
        if not cardiac_events:
            elements.append(Paragraph("No cardiac rhythm flags detected.", styles["Normal"]))
        for event in cardiac_events:
            pct = event["percent"]
            concern = "Normal"
            if pct > 20:
                concern = "Elevated"
            if pct > 40:
                concern = "High"
            elements.append(Paragraph(f"{event['name']}: {pct:.1f}% - {concern}", styles["Normal"]))
            elements.append(
                Paragraph(
                    "Explanation: "
                    f"{event['explanation']} "
                    f"How detected: {event['logic']}. "
                    "Higher percentages suggest this pattern appeared more often during the session.",
                    styles["Italic"],
                )
            )
            elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 8))
        elements.append(Paragraph("What To Do Next", styles["Heading2"]))
        elements.append(
            Paragraph(
                "If symptoms such as chest pain, shortness of breath, dizziness, or fainting are present, "
                "seek medical attention. For persistent or concerning trends, consider sharing this report "
                "with a healthcare professional. This report is not a medical diagnosis.",
                styles["Normal"],
            )
        )

        doc.build(elements)
        pdf_buf.seek(0)
        zipf.writestr("report.pdf", pdf_buf.read())

        if os.path.isdir("software"):
            for root, _, files in os.walk("software"):
                for filename in files:
                    path = os.path.join(root, filename)
                    zipf.write(path, arcname=path)

    zip_buffer.seek(0)
    payload = zip_buffer.read()
    REPORT_CACHE["payload"] = payload
    REPORT_CACHE["signature"] = signature
    REPORT_CACHE["timestamp"] = now
    REPORT_CACHE["public_until"] = now + 300
    return payload


def build_report_pdf() -> bytes:
    with RESET_LOCK:
        ecg_data = list(state.ecg_data)
        bpm_history = list(state.bpm_history)
        event_counts = dict(state.event_counts)
        quality = signal_quality_summary(state)
        summary = session_summary(state)

    report_payload = {
        "summary": summary,
        "signal_quality": quality,
        "events": [],
    }

    total = max(sum(event_counts.values()), 1)
    sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
    for event_name, count in sorted_events:
        if event_name in CARDIAC_EVENTS or event_name in SIGNAL_EVENTS:
            category = "Cardiac" if event_name in CARDIAC_EVENTS else "Signal"
            report_payload["events"].append(
                {
                    "name": event_name,
                    "category": category,
                    "count": count,
                    "percent": round((count / total) * 100, 1),
                    "explanation": explain_event(event_name, config),
                    "logic": event_detection_logic(event_name, config),
                }
            )

    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("ECG Monitoring Summary", styles["Title"]))
    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph(
            "This report summarizes ECG signal quality, heart rate trends, and detected rhythm flags from the "
            "P.U.L.S.E. ECG monitor. It is intended for clinical context only and is not a diagnosis.",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 8))
    elements.append(
        Paragraph(
            "Event detection is heuristic and for educational use only. "
            "It is not a clinical diagnosis.",
            styles["Italic"],
        )
    )
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Session Summary", styles["Heading2"]))
    duration_txt = f"{summary['duration_sec']}s" if summary["duration_sec"] else "n/a"
    elements.append(
        Paragraph(
            f"Duration: {duration_txt}. Samples: {summary['samples']}. "
            f"Avg BPM: {summary['bpm_avg'] or 'n/a'}, "
            f"Min BPM: {summary['bpm_min'] or 'n/a'}, "
            f"Max BPM: {summary['bpm_max'] or 'n/a'}.",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Signal Quality", styles["Heading2"]))
    elements.append(
        Paragraph(
            f"Quality: {quality['level']}. "
            f"Low amplitude: {quality['low_amplitude']}, "
            f"Noise: {quality['noise']}, "
            f"Baseline wander: {quality['baseline_wander']}, "
            f"Clipping: {quality['clipping']}.",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 8))

    if ecg_data:
        plt.figure(figsize=(6, 2.5))
        plt.plot(ecg_data[-1000:])
        plt.title("ECG Snapshot")
        plt.tight_layout()
        ecg_buf = io.BytesIO()
        plt.savefig(ecg_buf, format="png")
        plt.close()
        ecg_buf.seek(0)
        elements.append(Paragraph("ECG Snapshot (most recent window)", styles["Heading2"]))
        elements.append(Image(ecg_buf, width=420, height=180))
        elements.append(Spacer(1, 8))

    if bpm_history:
        plt.figure(figsize=(6, 2))
        plt.plot(bpm_history[-300:])
        plt.title("BPM Over Time")
        plt.tight_layout()
        bpm_buf = io.BytesIO()
        plt.savefig(bpm_buf, format="png")
        plt.close()
        bpm_buf.seek(0)
        elements.append(Paragraph("Heart Rate Trend (BPM)", styles["Heading2"]))
        elements.append(Image(bpm_buf, width=420, height=150))
        elements.append(Spacer(1, 8))

    elements.append(Paragraph("Event Breakdown", styles["Heading2"]))
    signal_events = [e for e in report_payload["events"] if e["category"] == "Signal"]
    cardiac_events = [e for e in report_payload["events"] if e["category"] == "Cardiac"]

    elements.append(Paragraph("Signal Quality Flags", styles["Heading3"]))
    if not signal_events:
        elements.append(Paragraph("No signal quality flags detected.", styles["Normal"]))
    for event in signal_events:
        pct = event["percent"]
        concern = "Normal"
        if pct > 20:
            concern = "Elevated"
        if pct > 40:
            concern = "High"
        elements.append(Paragraph(f"{event['name']}: {pct:.1f}% - {concern}", styles["Normal"]))
        elements.append(
            Paragraph(
                "Explanation: "
                f"{event['explanation']} "
                f"How detected: {event['logic']}. "
                "Higher percentages suggest this pattern appeared more often during the session.",
                styles["Italic"],
            )
        )
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 4))
    elements.append(Paragraph("Cardiac Rhythm Flags", styles["Heading3"]))
    if not cardiac_events:
        elements.append(Paragraph("No cardiac rhythm flags detected.", styles["Normal"]))
    for event in cardiac_events:
        pct = event["percent"]
        concern = "Normal"
        if pct > 20:
            concern = "Elevated"
        if pct > 40:
            concern = "High"
        elements.append(Paragraph(f"{event['name']}: {pct:.1f}% - {concern}", styles["Normal"]))
        elements.append(
            Paragraph(
                "Explanation: "
                f"{event['explanation']} "
                f"How detected: {event['logic']}. "
                "Higher percentages suggest this pattern appeared more often during the session.",
                styles["Italic"],
            )
        )
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("What To Do Next", styles["Heading2"]))
    elements.append(
        Paragraph(
            "If symptoms such as chest pain, shortness of breath, dizziness, or fainting are present, "
            "seek medical attention. For persistent or concerning trends, consider sharing this report "
            "with a healthcare professional. This report is not a medical diagnosis.",
            styles["Normal"],
        )
    )

    doc.build(elements)
    pdf_buf.seek(0)
    return pdf_buf.read()


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mobile")
def mobile():
    return render_template("mobile.html")


@app.route("/health")
def health():
    with RESET_LOCK:
        quality = signal_quality_summary(state)
        summary = session_summary(state)
        return jsonify({
            "ok": True,
            "simulate": SIMULATE,
            "hardware": HARDWARE_READY,
            "bpm": state.current_bpm,
            "last_signal_time": state.last_signal_time,
            "signal_quality": quality,
            "summary": summary,
        })


@app.route("/data")
def data():
    with RESET_LOCK:
        ecg_slice = list(state.ecg_data)[-1200:]
        smoothed = smooth_series(ecg_slice, SAMPLE_WINDOW)
        quality = signal_quality_summary(state)

        return jsonify({
            "ecg": smoothed,
            "bpm": state.current_bpm,
            "bpm_history": list(state.bpm_history)[-300:],
            "events": list(state.event_state.keys()),
            "signal": {"filtered": list(state.filtered_data)[-1000:]},
            "signal_quality": quality,
        })


@app.route("/reset", methods=["POST"])
def reset():
    with RESET_LOCK:
        state.reset()
    return ("", 204)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    if not shutdown_allowed(request):
        return ("Forbidden", 403)
    subprocess.Popen(["sudo", "shutdown", "now"])
    return ("", 204)


@app.route("/snapshot")
def snapshot():
    with RESET_LOCK:
        ecg_data = list(state.ecg_data)
        timestamps = list(state.timestamps)
        sample_rate = config.sample_rate
    if not ecg_data or not timestamps:
        return ("No data", 404)
    window_samples = int(sample_rate * 30)
    ecg_slice = ecg_data[-window_samples:]
    time_slice = timestamps[-window_samples:]
    if not ecg_slice:
        return ("No data", 404)
    if time_slice and len(time_slice) == len(ecg_slice):
        t0 = time_slice[0]
        x_vals = [t - t0 for t in time_slice]
    else:
        x_vals = list(range(len(ecg_slice)))
    plt.figure(figsize=(14, 3.2))
    plt.plot(x_vals, ecg_slice)
    plt.title("ECG Snapshot (Last 30 Seconds)")
    plt.xlabel("Seconds")
    plt.ylabel("Signal")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return send_file(buf, download_name="ecg_snapshot_30s.png", as_attachment=True)


# ================= REPORT ZIP =================
@app.route("/report")
def report():
    payload = build_report_zip()
    return send_file(io.BytesIO(payload), download_name="ecg_report_bundle.zip", as_attachment=True)


@app.route("/report/latest")
def report_latest():
    now = time.time()
    if REPORT_CACHE["payload"] and now <= REPORT_CACHE["public_until"]:
        return send_file(
            io.BytesIO(REPORT_CACHE["payload"]),
            download_name="ecg_report_bundle.zip",
            as_attachment=True,
        )
    return ("No recent report available", 404)


@app.route("/send_report_email", methods=["POST"])
def send_report_email():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"ok": False, "error": "Invalid email address."}), 400

    api_key = MAILGUN_API_KEY
    domain = MAILGUN_DOMAIN
    from_addr = MAILGUN_FROM if MAILGUN_FROM else (f"ECG Monitor <postmaster@{domain}>" if domain else None)
    if not api_key or api_key == "YOUR_MAILGUN_API_KEY" or not domain or domain == "YOUR_MAILGUN_DOMAIN":
        return jsonify({"ok": False, "error": "Mailgun is not configured."}), 500

    report_bytes = build_report_pdf()
    subject = "ECG Report (PDF)"
    text = "Attached is the latest ECG report PDF."

    resp = requests.post(
        f"https://api.mailgun.net/v3/{domain}/messages",
        auth=("api", api_key),
        files=[("attachment", ("ecg_report.pdf", report_bytes, "application/pdf"))],
        data={
            "from": from_addr,
            "to": email,
            "subject": subject,
            "text": text,
        },
        timeout=10,
    )

    if resp.status_code >= 400:
        return jsonify({"ok": False, "error": "Failed to send email."}), 502

    return jsonify({"ok": True})


# ================= START =================
if os.getenv("ECG_AUTOSTART", "1") == "1":
    threading.Thread(target=ecg_loop, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
