"""Configuration constants for the PlutoSDR backscatter receiver (FSK).

Binary FSK: '1' -> FSK_F1_HZ (1 kHz), '0' -> FSK_F0_HZ (1.7 kHz).
Non-harmonic tones -> independent power measurement per frequency, giving a
self-normalising decision: m_f1 > m_f0 -> '1', else '0'.
"""

# ---- SDR / RF setup ---------------------------------------------------------
RX_URI = "ip:192.168.2.1"
FREQ_HZ = 2.48e9
SAMPLE_RATE = 1e6
RX_GAIN_MODE: str = "manual"
RX_GAIN_DB = 40.0
# Per-buffer realtime budget = RX_BUFFER_SIZE / SAMPLE_RATE.
#   65536 @ 1 MS/s -> 65.5 ms (recommended)
#  131072 @ 1 MS/s -> 131 ms  (4x headroom, ~2x FFT/frame)
RX_BUFFER_SIZE = 65536
ADC_FULL_SCALE = 2048.0

# ---- GUI loop ---------------------------------------------------------------
# Update callback cadence for the decoupled rx_monitor.
ANIMATION_INTERVAL_MS = 10

# ---- Lightweight receiver architecture --------------------------------------
# Real-time loop is ingest/slicing only; heavy packet search runs offline.
RX_CAPTURE_NDJSON = "receiver/captures/chips_capture.ndjson"
# Re-run FFT peak tracking every N frames in RX-only mode. Frame counts here
# are tuned for RX_BUFFER_SIZE = 65536; scale inversely if buffer size changes.
RX_ONLY_PEAK_TRACK_EVERY_FRAMES = 4
# Monitor snapshots written by the RX-only loop for the rx_monitor UI.
RX_STATUS_JSON = "receiver/captures/rx_status.json"
RX_STATUS_EVERY_FRAMES = 3
RX_MONITOR_SPECTRUM_BINS = 192
RX_TERMINAL_STATUS_EVERY_FRAMES = 15
# AES-CMAC packet search cadence. Chip history (PHASE_HISTORY_BITS) holds
# packets for several seconds, so cadences up to ~15 still catch every packet.
LIVE_DECODE_EVERY_FRAMES = 4
LIVE_DECODE_RECENT_BITS = 160
# Try the bit-inverted stream too (tolerates FSK polarity flips).
LIVE_DECODE_TRY_INVERTED = True
# Acceptance gates for live status/monitor candidates.
LIVE_DECODE_MAX_HEADER_ERRORS = 0
LIVE_DECODE_MAX_WEAK_BITS = 2
# Payload search window cap (first byte is length, rest is message).
LIVE_DECODE_MAX_PAYLOAD_BYTES = 24

# ---- Display / spectrum -----------------------------------------------------
SPECTRUM_SPAN_HZ = 150000.0
FFT_AVG_ALPHA = 0.1
# Reject DC/LO leakage; only track carriers comfortably away from DC.
EXCITER_SEARCH_MIN_HZ = 5000.0
EXCITER_SEARCH_MAX_HZ = 50000.0
# Expected exciter offset (matches exciter/pluto_exciter.py TONE_HZ); restricts
# lock to this neighborhood to avoid DC sidebands / interferers winning.
EXCITER_EXPECTED_HZ = 15625.0
EXCITER_EXPECTED_TOL_HZ = 30000.0
EXCITER_STRICT_EXPECTED_BAND = True
# Anti-jump hysteresis: prevent large peak hops unless meaningfully stronger.
EXCITER_MAX_STEP_HZ = 300.0
EXCITER_SWITCH_MARGIN_DB = 4.0
CENTERED_SPAN_HZ = 8000.0
WATERFALL_ROWS = 240
# Display-only oversampling for the rx_monitor waterfall (off the hot path);
# monitor np.interps each transported row onto factor*bins. 1 disables.
WATERFALL_DISPLAY_OVERSAMPLE = 4
WATERFALL_DYN_RANGE_DB = 28.0
DC_BLOCK_ALPHA = 0.9998
CENTERED_FREQ_SMOOTH_BINS = 3
WATERFALL_ROW_BLEND = 0.75
# Recompute waterfall clim every N frames; noise floor smoothed by EMA between.
WATERFALL_AUTOSCALE_EVERY_N = 15
WATERFALL_NF_EMA_ALPHA = 0.25

# ---- FSK modulation parameters ----------------------------------------------
FSK_F1_HZ = 1000.0           # subcarrier frequency for '1' bits
FSK_F0_HZ = 1700.0           # subcarrier frequency for '0' bits
SIDEBAND_OFFSET_KHZ = FSK_F1_HZ / 1000.0  # primary sideband shown on plots
SIDEBAND_OFFSET_F0_KHZ = FSK_F0_HZ / 1000.0  # secondary sideband shown on plots
SIDEBAND_WINDOW_HZ = 30.0
SNR_LOCK_THRESHOLD_DB = 20.0
# Sideband SNR hysteresis: enter at threshold, exit at threshold - margin.
SNR_LOCK_EXIT_MARGIN_DB = 3.0

# ---- Bit timing -------------------------------------------------------------
# Firmware Timer_A0 (OUTMOD_4 toggle, SMCLK=1 MHz):
#   '1' bit: CCR0=499, 100 ticks -> 50.00 ms
#   '0' bit: CCR0=293, 170 ticks -> 49.98 ms
# Receiver chip slices must match the firmware bit duration.
BIT_DURATION_MS = 50.0
SAMPLES_PER_CHIP = int(SAMPLE_RATE * (BIT_DURATION_MS / 1000.0))

# ---- FSK decision parameters ------------------------------------------------
# Dead zone: |m_f1 - m_f0| < this -> chip distrusted, decoded as 0.
# Higher rejects more noise (drops weak bits); lower accepts marginal chips.
FSK_DECISION_DEAD_ZONE = 0.005

# Presence floor on max(m_f1, m_f0): below this, chip is treated as silence.
FSK_PRESENCE_FLOOR = 0.01

# ---- Repetition coding ------------------------------------------------------
# 1 = each chip is one logical bit. Set >1 only with matching firmware.
REPETITION_CHIPS = 1
MAJORITY_ONES_THRESHOLD = (REPETITION_CHIPS // 2) + 1

# ---- Packet structure -------------------------------------------------------
# Msp_FSK_secure.c: 0xAA preamble, 0x7E sync, then 16 payload bytes:
#   COUNTER(4 BE) || AES-CTR ciphertext(4) || AES-CMAC tag(8)
# Bits MSB-first, 50 ms/bit, 2 s inter-packet gap.
PREAMBLE_BYTES = b"\xAA"
SYNC_BYTES = b"\x7E"

# ---- Security (AES-128-CTR + AES-CMAC + monotonic replay counter) -----------
# Must match SHARED_KEY[] in Msp_FSK_secure.c byte-for-byte.
# Default is the RFC 4493 demo key — change and re-flash before deployment.
SHARED_KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"
# Persisted highest-accepted counter; delete to reset replay state.
SECURE_RX_STATE_PATH = "receiver/captures/secure_rx_state.json"
# Secure payload size: COUNTER(4) + CIPHERTEXT(4) + TAG(8).
LIVE_DECODE_PAYLOAD_BYTES = 16

# ---- Lock detection ---------------------------------------------------------
# SNR-driven hysteresis (enter/exit thresholds defined above).
LOCK_ENTER_FRAMES = 2
LOCK_EXIT_FRAMES = 8
# How many frames an AUTHENTICATED summary stays pinned in the status line.
PACKET_STATUS_HOLD_FRAMES = 15

# Suppress repeat status lines for the same AUTHENTICATED counter; only the
# first acceptance of each new counter prints, cached re-verifications stay quiet.
AUTHENTICATED_STATUS_SUPPRESS_REPEATS = True

# ---- Bit clock --------------------------------------------------------------
PHASE_HISTORY_BITS = 256
# Phase hypotheses spread uniformly across one bit period; live decoder picks
# the best. 4 phases -> ±12.5% max misalignment, reliable at SNR > 6 dB.
PHASE_COUNT = 4

# ---- Runtime jitter telemetry -----------------------------------------------
# Late frame: total processing exceeds this share of buffer time.
JITTER_LATE_FACTOR = 1.20
# Gap slip: update-to-update gap exceeds this share of buffer time.
JITTER_GAP_FACTOR = 1.50

# ---- CFO correction (Satori-inspired two-step adaptation for FSK) ----------
# Coarse: slow average residual. Fine: faster fluctuation tracking.
CFO_CORRECTION_ENABLED = True
CFO_COARSE_ALPHA = 0.03
CFO_FINE_ALPHA = 0.20
CFO_MAX_ABS_HZ = 100.0   # residual after peak tracking is at most a few FFT bins
CFO_SNR_GATE_DB = 0.0    # only update CFO EMA above this DC-carrier SNR
# Kay phase-advance decimation: pseudo-Nyquist = SAMPLE_RATE/(2*CFO_DECIMATE).
# Factor 8 keeps accuracy and cuts per-frame mix cost ~8x.
CFO_DECIMATE = 8
