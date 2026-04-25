"""Chip-threshold and slicing pipeline helpers."""

from typing import Any

import numpy as np

from . import config
from .dsp import coherent_chip_metric, ema_scalar
from .packet import majority_decode_triplets


def update_phase_threshold(st: Any, abs_ncc: float, locked: bool) -> None:
    """Update adaptive chip threshold state for one phase."""
    if config.CONTINUOUS_ON_TEST:
        # In continuous-ON mode there is no OFF-symbol cluster, so the standard
        # low/high threshold model is ill-posed. Track one signal cluster instead.
        if st.bit_ncc_high_ema <= config.EMA_INIT_THRESHOLD:
            st.bit_ncc_high_ema = abs_ncc
        else:
            st.bit_ncc_high_ema = ema_scalar(
                st.bit_ncc_high_ema, abs_ncc, config.CONTINUOUS_ON_THRESHOLD_ALPHA
            )
        st.bit_ncc_noise_ema = st.bit_ncc_high_ema
        if not config.ADAPTIVE_BIT_NCC_THRESHOLD:
            st.bit_ncc_threshold = float(config.BIT_NCC_THRESHOLD)
            return
        cand = config.CONTINUOUS_ON_THRESHOLD_SCALE * st.bit_ncc_high_ema
        st.bit_ncc_threshold = float(
            np.clip(cand, config.BIT_NCC_THRESHOLD_MIN, config.CONTINUOUS_ON_THRESHOLD_MAX)
        )
        return

    # Two-cluster (low/high) EMA tracker; threshold blends between them.
    locked_headroom = config.LOCKED_THRESHOLD_HEADROOM

    if st.bit_ncc_low_ema <= config.EMA_INIT_THRESHOLD:
        st.bit_ncc_low_ema = abs_ncc
        st.bit_ncc_high_ema = max(abs_ncc, config.BIT_NCC_THRESHOLD_MIN + config.BIT_NCC_MIN_HIGH_GAP)
    elif abs_ncc <= st.bit_ncc_threshold:
        st.bit_ncc_low_ema = ema_scalar(st.bit_ncc_low_ema, abs_ncc, config.BIT_NCC_NOISE_ALPHA)
    elif locked:
        # Reject impulsive outliers that would pin threshold at max and erase real "1" chips.
        if abs_ncc <= st.bit_ncc_low_ema + config.HIGH_SPIKE_REJECT_HEADROOM:
            capped = min(abs_ncc, st.bit_ncc_low_ema + locked_headroom)
            st.bit_ncc_high_ema = ema_scalar(
                st.bit_ncc_high_ema, capped, max(0.5 * config.BIT_NCC_NOISE_ALPHA, config.HIGH_RELAX_ALPHA)
            )
        else:
            st.bit_ncc_high_ema = ema_scalar(
                st.bit_ncc_high_ema,
                st.bit_ncc_low_ema + config.UNLOCK_HIGH_FLOOR_MARGIN,
                config.HIGH_RELAX_ALPHA,
            )
    else:
        st.bit_ncc_high_ema = ema_scalar(
            st.bit_ncc_high_ema,
            st.bit_ncc_low_ema + config.UNLOCK_HIGH_FLOOR_MARGIN,
            config.UNLOCK_HIGH_RELAX_ALPHA,
        )

    if st.bit_ncc_high_ema < st.bit_ncc_low_ema + config.BIT_NCC_MIN_HIGH_GAP:
        st.bit_ncc_high_ema = st.bit_ncc_low_ema + config.BIT_NCC_MIN_HIGH_GAP
    st.bit_ncc_noise_ema = st.bit_ncc_low_ema

    if not config.ADAPTIVE_BIT_NCC_THRESHOLD:
        st.bit_ncc_threshold = float(config.BIT_NCC_THRESHOLD)
        return
    blend = config.LOCKED_BLEND if locked else config.UNLOCK_BLEND
    cand = st.bit_ncc_low_ema + blend * (st.bit_ncc_high_ema - st.bit_ncc_low_ema)
    if locked:
        cand = min(cand, st.bit_ncc_low_ema + locked_headroom)
    else:
        cand = min(cand, st.bit_ncc_low_ema + config.UNLOCK_THRESHOLD_HEADROOM)
    st.bit_ncc_threshold = float(np.clip(cand, config.BIT_NCC_THRESHOLD_MIN, config.BIT_NCC_THRESHOLD_MAX))


def decide_chip_bistable(st: Any, abs_ncc: float) -> int:
    """Apply hysteretic chip decision logic for one phase."""
    thr = st.bit_ncc_threshold
    if not config.CHIP_BISTABLE_ENABLE:
        return 1 if abs_ncc >= thr else 0

    hi = min(config.BIT_NCC_THRESHOLD_MAX, thr + config.CHIP_DECISION_HYSTERESIS)
    lo = max(config.BIT_NCC_THRESHOLD_MIN, thr - config.CHIP_DECISION_HYSTERESIS)

    if st.chip_state < 0:
        st.chip_state = 1 if abs_ncc >= thr else 0
        st.chip_rise_streak = 0
        st.chip_fall_streak = 0
        return st.chip_state

    rise_confirm = config.CHIP_RISE_CONFIRM_CHIPS
    fall_confirm = config.CHIP_FALL_CONFIRM_CHIPS
    if config.CONTINUOUS_ON_TEST:
        fall_confirm = max(fall_confirm, config.CONTINUOUS_ON_FALL_CONFIRM_CHIPS)

    if st.chip_state == 1:
        if abs_ncc <= lo:
            st.chip_fall_streak += 1
        else:
            st.chip_fall_streak = 0
        if st.chip_fall_streak >= fall_confirm:
            st.chip_state = 0
            st.chip_fall_streak = 0
            st.chip_rise_streak = 0
    else:
        if abs_ncc >= hi:
            st.chip_rise_streak += 1
        else:
            st.chip_rise_streak = 0
        if st.chip_rise_streak >= rise_confirm:
            st.chip_state = 1
            st.chip_rise_streak = 0
            st.chip_fall_streak = 0
    return st.chip_state


def slice_chips_for_phase(
    *,
    phase: int,
    st: Any,
    peak_hz: float,
    phase_sample_buffer: np.ndarray,
    bit_samples: int,
    ncc_lock: bool,
    env_chip_metric_history_by_phase: dict[int, np.ndarray],
    env_chip_decision_history_by_phase: dict[int, np.ndarray],
) -> int:
    """Slice chips for one phase, updating phase state and view histories."""
    added = 0
    metric_hist = env_chip_metric_history_by_phase[phase]
    decision_hist = env_chip_decision_history_by_phase[phase]

    while True:
        chip_len = max(1, int(round(st.chip_stride_samples)))
        if st.next_sample + chip_len > len(phase_sample_buffer):
            break

        start = st.next_sample
        center_chunk = phase_sample_buffer[start : start + chip_len]
        center_metric = coherent_chip_metric(
            center_chunk, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE
        )
        abs_ncc = center_metric
        step_advance = chip_len

        if config.CHIP_TIMING_TRACK_ENABLE:
            best_metric = center_metric
            best_shift = 0
            search = int(max(0, config.CHIP_TIMING_SEARCH_SAMPLES))
            step = int(max(1, config.CHIP_TIMING_STEP_SAMPLES))
            for shift in range(-search, search + 1, step):
                if shift == 0:
                    continue
                cand_start = start + shift
                if cand_start < 0 or cand_start + chip_len > len(phase_sample_buffer):
                    continue
                cand_chunk = phase_sample_buffer[cand_start : cand_start + chip_len]
                cand_metric = coherent_chip_metric(
                    cand_chunk, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE
                )
                if cand_metric > best_metric:
                    best_metric = cand_metric
                    best_shift = shift

            # Apply timing correction only when this chip likely carries tone energy
            # and the early/late gain is meaningful; avoid random walk on OFF chips.
            if (
                best_shift != 0
                and best_metric >= st.bit_ncc_threshold + config.CHIP_TIMING_TRACK_MARGIN
                and (best_metric - center_metric) >= config.CHIP_TIMING_MIN_ADVANTAGE
            ):
                abs_ncc = best_metric
                st.timing_error_ema = ema_scalar(
                    st.timing_error_ema, float(best_shift), config.CHIP_TIMING_DRIFT_ALPHA
                )
            else:
                st.timing_error_ema = ema_scalar(st.timing_error_ema, 0.0, config.CHIP_TIMING_IDLE_ALPHA)

            min_stride = bit_samples * config.CHIP_TIMING_STRIDE_MIN_SCALE
            max_stride = bit_samples * config.CHIP_TIMING_STRIDE_MAX_SCALE
            st.chip_stride_samples = float(
                np.clip(bit_samples + st.timing_error_ema, min_stride, max_stride)
            )
            step_advance = max(1, int(round(st.chip_stride_samples)))

        update_phase_threshold(st, abs_ncc, ncc_lock)
        chip = decide_chip_bistable(st, abs_ncc)
        st.chips.append(chip)
        st.next_sample += step_advance
        st.chips_seen += 1
        added += 1

        metric_hist = np.roll(metric_hist, -1)
        metric_hist[-1] = float(np.clip(abs_ncc, config.CHIP_METRIC_CLIP_MIN, config.CHIP_METRIC_CLIP_MAX))
        decision_hist = np.roll(decision_hist, -1)
        decision_hist[-1] = float(chip)

    env_chip_metric_history_by_phase[phase] = metric_hist
    env_chip_decision_history_by_phase[phase] = decision_hist

    max_chips = config.PHASE_HISTORY_BITS * config.REPETITION_CHIPS
    if len(st.chips) > max_chips:
        drop = len(st.chips) - max_chips
        del st.chips[:drop]
        st.base_chip_index += drop

    return added


def phase_structure_stats(chips: list[int]) -> tuple[float, int, int, bool, bool]:
    """Return (one_ratio, chip_transitions, logical_transitions, structured, ready)."""
    min_needed = max(config.REPETITION_CHIPS * 2, config.LOCK_QUALITY_CHIP_WINDOW // 4)
    if len(chips) < min_needed:
        return 0.5, 0, 0, False, False

    w = min(len(chips), config.LOCK_QUALITY_CHIP_WINDOW)
    chip_window = chips[-w:]
    one_ratio = float(sum(chip_window)) / float(w)
    chip_transitions = sum(left != right for left, right in zip(chip_window, chip_window[1:]))

    logical_transitions = 0
    for off in range(config.REPETITION_CHIPS):
        logical = majority_decode_triplets(chip_window, off)
        if len(logical) < 2:
            continue
        logical_transitions = max(
            logical_transitions,
            sum(left != right for left, right in zip(logical, logical[1:])),
        )

    structured = (
        config.LOCK_QUALITY_ONE_RATIO_MIN <= one_ratio <= config.LOCK_QUALITY_ONE_RATIO_MAX
        and chip_transitions >= config.LOCK_QUALITY_MIN_CHIP_TRANSITIONS
        and logical_transitions >= config.LOCK_QUALITY_MIN_LOGICAL_TRANSITIONS
    )
    return one_ratio, chip_transitions, logical_transitions, structured, True


def rank_phases(
    *,
    phase_offsets: list[int],
    phase_state: dict[int, Any],
    env_chip_metric_history_by_phase: dict[int, np.ndarray],
    phase_score_window_chips: int,
    env_plot_len: int,
) -> list[tuple[int, float]]:
    """Rank candidate phases by recent metric energy and transition richness."""
    n = min(phase_score_window_chips, env_plot_len)
    scores: list[tuple[int, float]] = []
    for p in phase_offsets:
        metric_score = float(np.mean(env_chip_metric_history_by_phase[p][-n:]))
        st = phase_state[p]
        w = min(len(st.chips), config.PHASE_SCORE_TRANSITION_CHIPS)
        trans = 0
        if w >= config.REPETITION_CHIPS * 2:
            chip_window = st.chips[-w:]
            for off in range(config.REPETITION_CHIPS):
                logical = majority_decode_triplets(chip_window, off)
                trans = max(
                    trans,
                    sum(left != right for left, right in zip(logical, logical[1:])),
                )
        if trans < config.PHASE_SCORE_MIN_TRANSITIONS:
            metric_score *= config.PHASE_SCORE_LOW_TRANSITION_PENALTY
        scores.append((p, metric_score))
    scores.sort(key=lambda it: it[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# NCC lock hysteresis (per-frame, single shared state)
# ---------------------------------------------------------------------------

def update_lock_hysteresis(
    *,
    ncc_abs_ema: float,
    ncc_lock: bool,
    ncc_enter_count: int,
    ncc_exit_count: int,
    abs_ncc: float,
) -> tuple[float, bool, int, int]:
    """Advance lock/unlock hysteresis state from one NCC sample."""
    ncc_abs_ema = ema_scalar(ncc_abs_ema, abs_ncc, config.NCC_DISPLAY_ALPHA)
    if not ncc_lock:
        if ncc_abs_ema >= config.NCC_ENTER_THRESHOLD:
            ncc_enter_count = min(config.NCC_ENTER_FRAMES, ncc_enter_count + 1)
        else:
            ncc_enter_count = max(0, ncc_enter_count - config.NCC_HYST_COUNTER_DECAY)
        if ncc_enter_count >= config.NCC_ENTER_FRAMES:
            ncc_lock = True
            ncc_exit_count = 0
    else:
        if ncc_abs_ema <= config.NCC_EXIT_THRESHOLD:
            ncc_exit_count = min(config.NCC_EXIT_FRAMES, ncc_exit_count + 1)
        else:
            ncc_exit_count = max(0, ncc_exit_count - config.NCC_HYST_COUNTER_DECAY)
        if ncc_exit_count >= config.NCC_EXIT_FRAMES:
            ncc_lock = False
            ncc_enter_count = 0
    return ncc_abs_ema, ncc_lock, ncc_enter_count, ncc_exit_count
