import time

from ecg_core import ECGConfig, ECGState


def make_state():
    config = ECGConfig(sample_rate=250, r_threshold=15000)
    return ECGState(config)


def test_bradycardia_detection():
    state = make_state()
    state.current_bpm = 40
    state.detect_events(value=0, now=time.time())
    assert "Bradycardia" in state.event_state


def test_tachycardia_detection():
    state = make_state()
    state.current_bpm = 120
    state.detect_events(value=0, now=time.time())
    assert "Tachycardia" in state.event_state


def test_asystole_detection():
    state = make_state()
    state.last_signal_time = 0.0
    state.detect_events(value=0, now=10.0)
    assert "Asystole / Flatline" in state.event_state


def test_repolarization_detection():
    state = make_state()
    state.current_bpm = 80
    state.detect_events(value=int(state.config.r_threshold * 1.3), now=time.time())
    assert "Early Repolarization / ST Elevation (possible)" in state.event_state


def test_low_signal_amplitude():
    state = make_state()
    t0 = time.time()
    for i in range(state.config.noise_window_len):
        state.add_sample(10000, t0 + i / state.config.sample_rate)
    assert "Low Signal Amplitude" in state.event_state


def test_pvc_possible():
    state = make_state()
    t0 = time.time()
    # Create a short-long RR pattern
    rr_intervals = [0.8, 0.8, 0.5, 1.1, 0.8, 0.8]
    t = t0
    for rr in rr_intervals:
        t += rr
        state.last_peak_time = t - rr
        state.add_sample(state.config.r_threshold + 5000, t)
    assert "Premature Ventricular Contraction (PVC) (possible)" in state.event_state
