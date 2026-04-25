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
FSK_DECISION_DEAD_ZONE = 0.02

# Optional fixed threshold on max(m_f1, m_f0) to ensure SOME signal is
# present at one of the two frequencies. Below this, a chip is assumed
# to be noise / silence and decoded as 0.
FSK_PRESENCE_FLOOR = 0.06

# ---- Repetition coding ------------------------------------------------------
# Set to 1 to disable repetition coding (recommended starting point — FSK's
# self-normalising decisions don't suffer from threshold drift the way OOK
# does, so repetition's "stuck high" failure mode is much less of an issue).
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
PREAMBLE_BYTES = b"\xAA\xAA"
SYNC_BYTES = b"\xD3\x91"
PAYLOAD_BYTES = b"OPEN"
PACKET_DECODE_ENABLED = True

# ---- NCC lock detection (used for status only in FSK mode) ------------------
NCC_DISPLAY_ALPHA = 0.2
NCC_ENTER_THRESHOLD = 0.13
NCC_EXIT_THRESHOLD = 0.07
NCC_ENTER_FRAMES = 2
NCC_EXIT_FRAMES = 8
PACKET_STATUS_HOLD_FRAMES = 20

# ---- Debugging --------------------------------------------------------------
TERMINAL_DEBUG_BITS_EVERY = 20
TERMINAL_DEBUG_BIT_TAIL = 64
BIT_PHASE_STEP_SAMPLES = 5000
PHASE_HISTORY_BITS = 128

# ---- Header / payload tolerance ---------------------------------------------
HEADER_MAX_BIT_ERRORS = 4
PAYLOAD_MAX_BIT_ERRORS = 1
