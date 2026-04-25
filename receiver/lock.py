"""Lock policy helpers: watchdog and structure-aware lock quality gating."""

from dataclasses import dataclass
from typing import Any

from . import config
from .chips import phase_structure_stats
from .packet import majority_decode_triplets


@dataclass
class LockPolicyState:
    lock_watchdog_streak: int = 0
    lock_quality_bad_streak: int = 0
    lock_quality_summary: str = "lockq=warming"


def _watchdog_unlock(
    *,
    ranked: list[tuple[int, float]],
    phase_state: dict[int, Any],
    ncc_lock: bool,
    ncc_abs_ema: float,
    chips_added: int,
    streak: int,
) -> tuple[bool, int]:
    if (
        not config.LOCK_WATCHDOG_ENABLE
        or chips_added <= 0
        or not ncc_lock
        or not ranked
    ):
        return False, 0

    best_phase = ranked[0][0]
    st = phase_state[best_phase]
    if len(st.chips) < max(config.LOCK_WATCHDOG_MIN_CHIPS, config.REPETITION_CHIPS):
        return False, 0

    w = min(len(st.chips), config.LOCK_WATCHDOG_CHIP_WINDOW)
    chip_window = st.chips[-w:]
    one_ratio = float(sum(chip_window)) / float(w)
    chip_transitions = sum(left != right for left, right in zip(chip_window, chip_window[1:]))

    best_transitions = 0
    for off in range(config.REPETITION_CHIPS):
        logical = majority_decode_triplets(chip_window, off)
        if len(logical) < 2:
            continue
        transitions = sum(left != right for left, right in zip(logical, logical[1:]))
        if transitions > best_transitions:
            best_transitions = transitions

    stuck = (
        one_ratio >= config.LOCK_WATCHDOG_ONE_RATIO
        and best_transitions <= config.LOCK_WATCHDOG_MAX_TRANSITIONS
        and chip_transitions <= config.LOCK_WATCHDOG_MAX_CHIP_TRANSITIONS
        and ncc_abs_ema <= config.LOCK_WATCHDOG_MAX_NCC_EMA
    )
    streak = (streak + 1) if stuck else 0

    if streak >= config.LOCK_WATCHDOG_TRIGGER_UPDATES:
        print(
            f"[RX LOCK] watchdog unlock: phase={best_phase} "
            f"one_ratio={one_ratio:.3f} transitions={best_transitions} chip_trans={chip_transitions} "
            f"ncc_ema={ncc_abs_ema:.3f}",
            flush=True,
        )
        return True, 0
    return False, streak


def apply_lock_policy(
    *,
    ranked: list[tuple[int, float]],
    phase_state: dict[int, Any],
    ncc_lock: bool,
    ncc_abs_ema: float,
    lock_before_update: bool,
    ncc_enter_count: int,
    ncc_exit_count: int,
    decode_lock_grace_chips: int,
    chips_added: int,
    state: LockPolicyState,
) -> tuple[bool, int, int, int, bool, LockPolicyState]:
    """Apply watchdog + structure gates and update decode grace policy."""
    should_unlock, state.lock_watchdog_streak = _watchdog_unlock(
        ranked=ranked,
        phase_state=phase_state,
        ncc_lock=ncc_lock,
        ncc_abs_ema=ncc_abs_ema,
        chips_added=chips_added,
        streak=state.lock_watchdog_streak,
    )
    if should_unlock:
        ncc_lock = False
        ncc_enter_count = 0
        ncc_exit_count = 0
        decode_lock_grace_chips = 0

    best_phase_structured = False
    best_phase_ready = False
    if ranked:
        best_phase = ranked[0][0]
        one_ratio, chip_trans, logical_trans, best_phase_structured, best_phase_ready = phase_structure_stats(
            phase_state[best_phase].chips
        )
        if best_phase_ready:
            state.lock_quality_summary = (
                f"lockq={'ok' if best_phase_structured else 'bad'} "
                f"ones={one_ratio:.2f} cT={chip_trans} lT={logical_trans}"
            )
        else:
            state.lock_quality_summary = "lockq=warming"
    else:
        state.lock_quality_summary = "lockq=warming"

    if ncc_lock and config.LOCK_ENTER_REQUIRE_PHASE_STRUCTURE:
        if not best_phase_ready or best_phase_structured:
            state.lock_quality_bad_streak = 0
        else:
            state.lock_quality_bad_streak += 1

        if best_phase_ready and (not lock_before_update) and not best_phase_structured:
            print(
                f"[RX LOCK] enter veto: ncc_ema={ncc_abs_ema:.3f} {state.lock_quality_summary}",
                flush=True,
            )
            ncc_lock = False
            ncc_enter_count = 0
            ncc_exit_count = 0
            decode_lock_grace_chips = 0
            state.lock_quality_bad_streak = 0
        elif state.lock_quality_bad_streak >= config.LOCK_QUALITY_VETO_FRAMES:
            print(
                f"[RX LOCK] quality unlock: ncc_ema={ncc_abs_ema:.3f} "
                f"{state.lock_quality_summary} bad_frames={state.lock_quality_bad_streak}",
                flush=True,
            )
            ncc_lock = False
            ncc_enter_count = 0
            ncc_exit_count = 0
            decode_lock_grace_chips = 0
            state.lock_quality_bad_streak = 0
    elif not ncc_lock:
        state.lock_quality_bad_streak = 0

    if ncc_lock:
        decode_lock_grace_chips = config.DECODE_UNLOCK_GRACE_CHIPS
    elif chips_added > 0:
        decode_lock_grace_chips = max(0, decode_lock_grace_chips - chips_added)

    return (
        ncc_lock,
        ncc_enter_count,
        ncc_exit_count,
        decode_lock_grace_chips,
        best_phase_structured,
        state,
    )
