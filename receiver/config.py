"""Configuration constants for the PlutoSDR backscatter receiver — FSK version.

Binary FSK modulation:
  '1' bit  -> subcarrier at FSK_F1_HZ = 1000 Hz
  '0' bit  -> subcarrier at FSK_F0_HZ = 1700 Hz

The two frequencies are chosen so neither is a harmonic of the other:
  1 kHz square wave: spectral lines at 1, 3, 5, 7, 9 kHz
  1.7 kHz square wave: spectral lines at 1.7, 5.1, 8.5 kHz
No overlap -> the receiver can independently measure power at each
frequency without cross-contamination, which gives a clean self-
normalising decision: m_f1 > m_f0 -> '1', m_f1 < m_f0 -> '0'.
"""

# ---- SDR / RF setup ---------------------------------------------------------
RX_URI = "ip:192.168.2.1"
FREQ_HZ = 2.48e9
SAMPLE_RATE = 1e6
RX_GAIN_MODE: str = "manual"
RX_GAIN_DB = 40.0
RX_BUFFER_SIZE = 65536       # 3.8 Hz/bin — sub-bin resolution near sidebands
ADC_FULL_SCALE = 2048.0

# ---- GUI loop ---------------------------------------------------------------
# Update callback cadence for the decoupled rx_monitor.
ANIMATION_INTERVAL_MS = 10

# ---- Lightweight receiver architecture --------------------------------------
# Real-time loop is ingest/slicing only; heavy packet search runs offline.
RX_CAPTURE_NDJSON = "receiver/captures/chips_capture.ndjson"
# Re-run expensive FFT peak tracking every N frames in RX-only mode.
RX_ONLY_PEAK_TRACK_EVERY_FRAMES = 8
# Lightweight monitor snapshots written by the RX-only loop for a separate UI.
RX_STATUS_JSON = "receiver/captures/rx_status.json"
RX_STATUS_EVERY_FRAMES = 5
# Number of bins exported in the RX-only monitor spectrum snapshot.
RX_MONITOR_SPECTRUM_BINS = 192
# How often the RX-only loop prints a concise radio/decode status line.
RX_TERMINAL_STATUS_EVERY_FRAMES = 30
# Only search this many most-recent logical bits for live packet reporting.
LIVE_DECODE_RECENT_BITS = 160
# Also evaluate bit-inverted stream to tolerate FSK polarity flips.
LIVE_DECODE_TRY_INVERTED = True
# Suppress live status/monitor candidates unless they beat these quality gates.
# Allow a single header bit error to recover valid packets under light corruption.
LIVE_DECODE_MAX_HEADER_ERRORS = 0
# Maximum weak decisions allowed across header+payload window.
# Lower is stricter (better BER, fewer/late detections).
LIVE_DECODE_MAX_WEAK_BITS = 2
# Number of payload bytes to extract/display after preamble+sync.
# The first payload byte is now a length field; the receiver reads that many
# additional bytes as the actual message.  This constant caps the search window.
LIVE_DECODE_MAX_PAYLOAD_BYTES = 24

# ---- Display / spectrum -----------------------------------------------------
TIME_SAMPLES = 1200
SPECTRUM_SPAN_HZ = 150000.0
FFT_AVG_ALPHA = 0.1
EXCITER_SEARCH_MIN_HZ = 500.0
EXCITER_SEARCH_MAX_HZ = 40000.0
CENTERED_SPAN_HZ = 8000.0
WATERFALL_ROWS = 120
WATERFALL_BINS = 512
WATERFALL_DYN_RANGE_DB = 28.0
DC_BLOCK_ALPHA = 0.9998
RX_RF_BANDWIDTH = 200_000
ENABLE_FOCUSED_FILTER = False
FOCUSED_PASSBAND_HZ = 2500.0
CENTERED_FREQ_SMOOTH_BINS = 3
WATERFALL_ROW_BLEND = 0.75

# ---- FSK modulation parameters ----------------------------------------------
FSK_F1_HZ = 1000.0           # subcarrier frequency for '1' bits
FSK_F0_HZ = 1700.0           # subcarrier frequency for '0' bits
SUBCARRIER_HZ = FSK_F1_HZ    # used by spectrum markers (shows the '1' tone)
SIDEBAND_OFFSET_KHZ = FSK_F1_HZ / 1000.0  # primary sideband shown on plots
SIDEBAND_OFFSET_F0_KHZ = FSK_F0_HZ / 1000.0  # secondary sideband shown on plots
SIDEBAND_WINDOW_HZ = 30.0
SNR_LOCK_THRESHOLD_DB = 20.0

# ---- Bit timing -------------------------------------------------------------
# MSP firmware: each bit transmits at its frequency for exactly 50 ms.
BIT_DURATION_MS = 50.0
SAMPLES_PER_CHIP = int(SAMPLE_RATE * (BIT_DURATION_MS / 1000.0))

# ---- FSK decision parameters ------------------------------------------------
# When |m_f1 - m_f0| < this, we don't trust the chip; default it to 0.
# Tune higher to reject noise more aggressively (at the cost of dropping
# weak-but-real bits), lower to accept marginal chips.
FSK_DECISION_DEAD_ZONE = 0.005

# Optional fixed threshold on max(m_f1, m_f0) to ensure SOME signal is
# present at one of the two frequencies. Below this, a chip is assumed
# to be noise / silence and decoded as 0.
FSK_PRESENCE_FLOOR = 0.01

# ---- Repetition coding ------------------------------------------------------
# Single-bit mode: each transmitted chip is one logical bit.
# Set >1 only if firmware repeats each bit and you want majority voting.
REPETITION_CHIPS = 1
LOGICAL_BIT_DURATION_MS = BIT_DURATION_MS * REPETITION_CHIPS
SAMPLES_PER_LOGICAL_BIT = SAMPLES_PER_CHIP * REPETITION_CHIPS
MAJORITY_ONES_THRESHOLD = (REPETITION_CHIPS // 2) + 1

# ---- Demod display ----------------------------------------------------------
DEMOD_FILTER_PASSBAND_HZ = 3500.0  # wide enough to pass both 1 kHz and 1.7 kHz
DEMOD_ENV_SMOOTH_SAMPLES = 48
CHIP_VIEW_HISTORY = 96             # chips shown in the demod panel
ENV_Y_SMOOTH_ALPHA = 0.15
ENV_Y_MIN_SPAN = 0.01

# ---- Packet structure -------------------------------------------------------
# Msp_FSK_secure.c: preamble 0xAA, sync 0x7E, then 16 payload bytes:
#   COUNTER(4 BE) || AES-CTR ciphertext(4) || AES-CMAC tag(8)
# Bits sent MSB-first, 50 ms/bit, 2-second gap between packets.
PREAMBLE_BYTES = b"\xAA"
SYNC_BYTES = b"\x7E"
PAYLOAD_BYTES = b"OPEN"              # expected plaintext after decryption
ALLOW_INVERTED_PAYLOAD_MATCH = False  # not applicable in secure mode (MAC is the check)
PACKET_DECODE_ENABLED = True
RENDER_PLOTS = False

# ---- Security (AES-128-CTR + AES-CMAC + monotonic replay counter) -----------
# SHARED_KEY_HEX must match SHARED_KEY[] in Msp_FSK_secure.c byte-for-byte.
# The default is the RFC 4493 demo key — change before any deployment and
# re-flash the tag firmware with the matching bytes.
SECURE_MODE = True
SHARED_KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"
# JSON file that persists the highest accepted counter across receiver runs.
# Delete this file to reset replay state when re-flashing the tag.
SECURE_RX_STATE_PATH = "receiver/captures/secure_rx_state.json"
# Fixed payload size in secure mode: COUNTER(4) + CIPHERTEXT(4) + TAG(8).
LIVE_DECODE_PAYLOAD_BYTES = 16

# ---- NCC lock detection (used for status only in FSK mode) ------------------
NCC_DISPLAY_ALPHA = 0.2
NCC_ENTER_THRESHOLD = 0.13
NCC_EXIT_THRESHOLD = 0.07
NCC_ENTER_FRAMES = 2
NCC_EXIT_FRAMES = 8
# Keep authenticated status visible between packet bursts.
PACKET_STATUS_HOLD_FRAMES = 180

# ---- Debugging --------------------------------------------------------------
TERMINAL_DEBUG_BITS_EVERY = 240
TERMINAL_DEBUG_BIT_TAIL = 64
# 4 ms phase step reduces per-frame phase count to ease realtime pressure.
BIT_PHASE_STEP_SAMPLES = 4000
PHASE_HISTORY_BITS = 256

# ---- Runtime jitter telemetry -----------------------------------------------
# Report host-side timing so we can distinguish decode issues from scheduling jitter.
# Count a frame as "late" when total processing exceeds this share of buffer time.
JITTER_LATE_FACTOR = 1.20
# Count a frame-gap slip when update-to-update gap exceeds this share of buffer time.
JITTER_GAP_FACTOR = 1.50

# ---- Header / payload tolerance ---------------------------------------------
# AA (8 bits) + 7E (8 bits) = 16 header bits; allow up to 5 errors for diagnosis.
HEADER_MAX_BIT_ERRORS = 5
# 4 bytes = 32 bits; allow up to 4 errors for diagnosis.
PAYLOAD_MAX_BIT_ERRORS = 4
# Limit noisy header candidate logging per animation frame to avoid terminal I/O stalls.
HEADER_LOGS_PER_FRAME = 4
# Limit header candidates processed per phase/offset per frame.
HEADER_CANDIDATES_PER_SCAN = 2

# ---- CFO correction (Satori-inspired two-step adaptation for FSK) ----------
# Step 1: slow average residual CFO estimate.
# Step 2: faster residual fluctuation tracking around the average.
CFO_CORRECTION_ENABLED = True
CFO_COARSE_ALPHA = 0.03
CFO_FINE_ALPHA = 0.20
CFO_MAX_ABS_HZ = 3000.0
