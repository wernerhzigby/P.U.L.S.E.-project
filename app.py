import io
import os
import csv
import math
import time
import random
import zipfile
import threading
import subprocess

from flask import Flask, render_template, jsonify, send_file, request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
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


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    with RESET_LOCK:
        return jsonify({
            "ok": True,
            "simulate": SIMULATE,
            "hardware": HARDWARE_READY,
            "bpm": state.current_bpm,
        })


@app.route("/data")
def data():
    with RESET_LOCK:
        ecg_slice = list(state.ecg_data)[-1000:]
        smoothed = smooth_series(ecg_slice, SAMPLE_WINDOW)

        return jsonify({
            "ecg": smoothed,
            "bpm": state.current_bpm,
            "bpm_history": list(state.bpm_history)[-300:],
            "events": list(state.event_state.keys()),
            "signal": {"filtered": list(state.filtered_data)[-1000:]},
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


# ================= REPORT ZIP =================
@app.route("/report")
def report():
    with RESET_LOCK:
        ecg_data = list(state.ecg_data)
        timestamps = list(state.timestamps)
        bpm_history = list(state.bpm_history)
        bpm_timestamps = list(state.bpm_timestamps)
        event_timeline = list(state.event_timeline)
        event_counts = dict(state.event_counts)

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

        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("ECG Monitoring Summary", styles["Title"]))
        elements.append(Spacer(1, 12))

        total = max(sum(event_counts.values()), 1)
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)

        for event_name, count in sorted_events:
            if event_name in CARDIAC_EVENTS or event_name in SIGNAL_EVENTS:
                pct = (count / total) * 100
                if pct > 0:
                    concern = "Normal"
                    if pct > 20:
                        concern = "Elevated"
                    if pct > 40:
                        concern = "High"
                    elements.append(Paragraph(f"{event_name}: {pct:.1f}% - {concern}", styles["Normal"]))
                    elements.append(
                        Paragraph(
                            f"Explanation: This flag indicates {event_name.lower()} was detected in the session. "
                            "Higher percentages suggest more frequent abnormality.",
                            styles["Italic"],
                        )
                    )
                    elements.append(Spacer(1, 6))

        doc.build(elements)
        pdf_buf.seek(0)
        zipf.writestr("report.pdf", pdf_buf.read())

        if os.path.isdir("software"):
            for root, _, files in os.walk("software"):
                for filename in files:
                    path = os.path.join(root, filename)
                    zipf.write(path, arcname=path)

    zip_buffer.seek(0)
    return send_file(zip_buffer, download_name="ecg_report_bundle.zip", as_attachment=True)


# ================= START =================
if os.getenv("ECG_AUTOSTART", "1") == "1":
    threading.Thread(target=ecg_loop, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
