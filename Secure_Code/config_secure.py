"""Configuration constants for the SECURE FSK backscatter receiver.

This is the encrypted/authenticated variant of config_fsk.py. Differences
from the plaintext version:

  1. REPETITION_CHIPS = 1. The secure firmware (Msp_FSK_secure.c) drops
     3-chip repetition coding because AES-CMAC provides cryptographic
     integrity, which is strictly stronger than majority voting.

  2. LIVE_DECODE_PAYLOAD_BYTES = 16. The on-air payload after the AA/7E
     header is now COUNTER(4 BE) || CIPHERTEXT(4) || CMAC_TAG(8).

  3. SHARED_KEY_HEX must match SHARED_KEY[] in Msp_FSK_secure.c.

  4. SECURE_RX_STATE_PATH points to a JSON file that the SecureReceiver
     uses to remember the highest counter ever accepted across runs.
     Delete this file to reset the replay window (e.g., when re-flashing
     the tag with a fresh counter).

  5. Capture/status paths live under Receiver_FSK_secure/ so the secure
     and plaintext receivers can coexist without overwriting each other.

Binary FSK modulation (unchanged):
  '1' bit  -> subcarrier at FSK_F1_HZ = 1000 Hz
  '0' bit  -> subcarrier at FSK_F0_HZ = 1700 Hz
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
ANIMATION_INTERVAL_MS = 10
RENDER_PLOTS = False

# ---- Lightweight receiver architecture --------------------------------------
RX_ONLY_MODE = True
RX_CAPTURE_NDJSON = "Receiver_FSK_secure/captures/chips_capture.ndjson"
RX_ONLY_PEAK_TRACK_EVERY_FRAMES = 8
RX_STATUS_JSON = "Receiver_FSK_secure/captures/rx_status.json"
RX_STATUS_EVERY_FRAMES = 5
RX_MONITOR_SPECTRUM_BINS = 192
RX_TERMINAL_STATUS_EVERY_FRAMES = 30
LIVE_DECODE_RECENT_BITS = 96
LIVE_DECODE_MAX_HEADER_ERRORS = 2
LIVE_DECODE_MAX_PAYLOAD_ERRORS = 4
LIVE_DECODE_MAX_WEAK_BITS = 6
# Secure mode: payload is per-packet random ciphertext, so we never compare it
# bitwise against an "expected" pattern. The MAC is the integrity check.
LIVE_DECODE_REQUIRE_KNOWN_PAYLOAD = False
# Secure payload = counter(4) + ciphertext(4) + tag(8) = 16 bytes.
LIVE_DECODE_PAYLOAD_BYTES = 16

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
FSK_F1_HZ = 1000.0
FSK_F0_HZ = 1700.0
SUBCARRIER_HZ = FSK_F1_HZ
SIDEBAND_OFFSET_KHZ = FSK_F1_HZ / 1000.0
SIDEBAND_OFFSET_F0_KHZ = FSK_F0_HZ / 1000.0
SIDEBAND_WINDOW_HZ = 30.0
SNR_LOCK_THRESHOLD_DB = 20.0

# ---- Bit timing -------------------------------------------------------------
BIT_DURATION_MS = 50.0
SAMPLES_PER_CHIP = int(SAMPLE_RATE * (BIT_DURATION_MS / 1000.0))

# ---- FSK decision parameters ------------------------------------------------
FSK_DECISION_DEAD_ZONE = 0.005
FSK_PRESENCE_FLOOR = 0.01

# ---- Repetition coding ------------------------------------------------------
# Secure mode: NO repetition coding. The CMAC tag is the integrity check.
# REPETITION_CHIPS = 1 makes majority_decode_triplets a passthrough and
# all the weak-triplet checks no-ops, which is the desired behavior.
REPETITION_CHIPS = 1
LOGICAL_BIT_DURATION_MS = BIT_DURATION_MS * REPETITION_CHIPS
SAMPLES_PER_LOGICAL_BIT = SAMPLES_PER_CHIP * REPETITION_CHIPS
MAJORITY_ONES_THRESHOLD = (REPETITION_CHIPS // 2) + 1

# ---- Demod display ----------------------------------------------------------
DEMOD_FILTER_PASSBAND_HZ = 3500.0
DEMOD_ENV_SMOOTH_SAMPLES = 48
CHIP_VIEW_HISTORY = 96
ENV_Y_SMOOTH_ALPHA = 0.15
ENV_Y_MIN_SPAN = 0.01

# ---- Packet structure -------------------------------------------------------
# Secure on-air format (after the AA/7E header):
#   COUNTER(4 BE) || CIPHERTEXT(4) || CMAC_TAG(8)   = 16 bytes
# PAYLOAD_BYTES is preserved (= b"OPEN") as the EXPECTED PLAINTEXT after
# decryption — it is no longer compared bitwise on the air.
PREAMBLE_BYTES = b"\xAA"
SYNC_BYTES = b"\x7E"
PAYLOAD_BYTES = b"OPEN"
PACKET_DECODE_ENABLED = not RX_ONLY_MODE

# ---- Cryptography (NEW for secure mode) -------------------------------------
# This 16-byte hex string MUST match SHARED_KEY[] in Msp_FSK_secure.c byte
# for byte. The default below is the RFC 4493 demo key — change it for any
# real deployment, and re-flash the firmware with the matching bytes.
SHARED_KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"

# JSON file in which the SecureReceiver persists "highest accepted counter".
# Delete this file to reset the replay window (e.g., when re-flashing the
# tag with a fresh counter that is below the last accepted value).
SECURE_RX_STATE_PATH = "Receiver_FSK_secure/captures/secure_rx_state.json"

# ---- NCC lock detection -----------------------------------------------------
NCC_DISPLAY_ALPHA = 0.2
NCC_ENTER_THRESHOLD = 0.13
NCC_EXIT_THRESHOLD = 0.07
NCC_ENTER_FRAMES = 2
NCC_EXIT_FRAMES = 8
PACKET_STATUS_HOLD_FRAMES = 20

# ---- Debugging --------------------------------------------------------------
TERMINAL_DEBUG_BITS_EVERY = 240
TERMINAL_DEBUG_BIT_TAIL = 64
BIT_PHASE_STEP_SAMPLES = 4000
PHASE_HISTORY_BITS = 128

# ---- Runtime jitter telemetry -----------------------------------------------
JITTER_LATE_FACTOR = 1.20
JITTER_GAP_FACTOR = 1.50

# ---- Header / payload tolerance ---------------------------------------------
HEADER_MAX_BIT_ERRORS = 5
PAYLOAD_MAX_BIT_ERRORS = 4
HEADER_LOGS_PER_FRAME = 4
HEADER_CANDIDATES_PER_SCAN = 2
ALLOW_INVERTED_PAYLOAD_MATCH = True

# ---- CFO correction ---------------------------------------------------------
CFO_CORRECTION_ENABLED = True
CFO_COARSE_ALPHA = 0.03
CFO_FINE_ALPHA = 0.20
CFO_MAX_ABS_HZ = 3000.0
