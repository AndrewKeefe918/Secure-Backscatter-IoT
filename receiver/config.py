"""Configuration constants for the PlutoSDR backscatter receiver."""

RX_URI = "ip:192.168.2.1"
FREQ_HZ = 2.48e9
SAMPLE_RATE = 1e6
RX_GAIN_MODE: str = "manual"
RX_GAIN_DB = 40.0
RX_BUFFER_SIZE = 65536       # 3.8 Hz/bin — 263 bins between carrier and ±1 kHz sideband
ADC_FULL_SCALE = 2048.0
TIME_SAMPLES = 1200
SPECTRUM_SPAN_HZ = 150000.0
FFT_AVG_ALPHA = 0.1           # ~10-frame memory — ~1s response, good noise rejection
EXCITER_SEARCH_MIN_HZ = 500.0   # ignore near-DC bins
EXCITER_SEARCH_MAX_HZ = 40000.0 # prevent false lock on far-out noise peaks
CENTERED_SPAN_HZ = 8000.0
WATERFALL_ROWS = 120
WATERFALL_BINS = 512          # lower interpolation cost per frame
WATERFALL_DYN_RANGE_DB = 28.0 # wider range avoids center saturation and preserves discrete lines
DC_BLOCK_ALPHA = 0.9998       # cutoff ≈ 32 Hz — well below 1 kHz signal
RX_RF_BANDWIDTH = 200_000     # AD9361 on-chip analog LPF — rejects noise outside 200 kHz
ENABLE_FOCUSED_FILTER = False # default off: avoids filter transients/blueout artifacts
FOCUSED_PASSBAND_HZ = 2500.0
SIDEBAND_OFFSET_KHZ = 1.0     # 1 kHz subcarrier offset
SIDEBAND_WINDOW_HZ = 30.0     # ±30 Hz at 3.8 Hz/bin = ±8 bins — tight enough to avoid noise peaks
SNR_LOCK_THRESHOLD_DB = 20.0  # minimum SNR for a reliable backscatter decode
ENV_Y_SMOOTH_ALPHA = 0.15     # envelope y-limit smoothing (higher = more responsive)
ENV_Y_MIN_SPAN = 0.01         # minimum y-span for envelope plot stability
CENTERED_FREQ_SMOOTH_BINS = 3 # lighter smoothing to keep sidebands sharp
WATERFALL_ROW_BLEND = 0.75    # favor current frame; less temporal blur
BIT_DURATION_MS = 50.0
BIT_NCC_THRESHOLD = 0.10
PREAMBLE_BYTES = b"\xAA\xAA"
SYNC_BYTES = b"\xD3\x91"
PAYLOAD_BYTES = b"OPEN"
REPETITION_CHIPS = 3
NCC_DISPLAY_ALPHA = 0.2
NCC_ENTER_THRESHOLD = 0.1
NCC_EXIT_THRESHOLD = 0.05
NCC_ENTER_FRAMES = 3
NCC_EXIT_FRAMES = 8
PACKET_STATUS_HOLD_FRAMES = 20
TERMINAL_DEBUG_BITS_EVERY = 20
TERMINAL_DEBUG_BIT_TAIL = 64
BIT_PHASE_STEP_SAMPLES = 5000
PHASE_HISTORY_BITS = 128
HEADER_MAX_BIT_ERRORS = 4
