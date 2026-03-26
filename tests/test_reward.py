"""
Tests for the frozen reward function.
Owner: B (RL Training Lead)
Ticket: RL-04
Run: pytest tests/test_reward.py -v
"""

from __future__ import annotations

import pytest
from agents.reward import (
    compute_rewards,
    CAPTURE_REWARD,
    TEAM_CAPTURE_BONUS,
    STEP_PENALTY,
    DISTANCE_SHAPING,
)


def _state(positions: dict) -> dict:
    return {"positions": positions}


# ---------------------------------------------------------------------------
# Capture step — reward totals
# ---------------------------------------------------------------------------

def test_capture_reward_solo_seeker():
    """Single seeker at hider's cell receives CAPTURE + TEAM_BONUS + STEP_PENALTY."""
    prev = _state({"seeker_0": (4, 4), "hider": (5, 5)})
    curr = _state({"seeker_0": (5, 5), "hider": (5, 5)})

    rewards = compute_rewards(curr, {}, prev, n_seekers=1)

    # Distance reduced by 2 (was |4-5|+|4-5|=2, now 0)
    expected = CAPTURE_REWARD + TEAM_CAPTURE_BONUS + STEP_PENALTY + DISTANCE_SHAPING * 2
    assert rewards["seeker_0"] == pytest.approx(expected)
    assert rewards["hider"] == 0.0


def test_capture_total_reward_two_seekers():
    """On capture with 2 seekers, total reward across all seekers equals
    CAPTURE_REWARD + TEAM_CAPTURE_BONUS + 2×STEP_PENALTY + shaping."""
    prev = _state({"seeker_0": (4, 4), "seeker_1": (6, 6), "hider": (5, 5)})
    curr = _state({"seeker_0": (5, 5), "seeker_1": (6, 6), "hider": (5, 5)})

    rewards = compute_rewards(curr, {}, prev, n_seekers=2)

    # seeker_0 captured: moved from (4,4) to (5,5) — distance reduced from 2 to 0
    # seeker_1 didn't move — distance from (6,6) to (5,5) was 2, still 2
    total = rewards["seeker_0"] + rewards["seeker_1"]
    expected_total = (
        CAPTURE_REWARD          # seeker_0 individual capture
        + TEAM_CAPTURE_BONUS    # shared across both (full amount)
        + 2 * STEP_PENALTY      # both seekers pay step penalty
        + DISTANCE_SHAPING * 2  # seeker_0 closed 2 steps; seeker_1 shaping = 0
    )
    assert total == pytest.approx(expected_total)
    assert rewards["hider"] == 0.0


def test_team_bonus_split_equally():
    """TEAM_CAPTURE_BONUS is divided equally among n_seekers."""
    n = 4
    prev = _state({**{f"seeker_{i}": (0, i) for i in range(n)}, "hider": (0, 0)})
    curr = _state({**{f"seeker_{i}": (0, i) for i in range(n)}, "hider": (0, 0)})

    rewards = compute_rewards(curr, {}, prev, n_seekers=n)

    # seeker_0 is at hider position (0,0) — it captures
    for i in range(n):
        assert rewards[f"seeker_{i}"] == pytest.approx(
            (CAPTURE_REWARD if i == 0 else 0.0)
            + TEAM_CAPTURE_BONUS / n
            + STEP_PENALTY
            # no distance shaping — no movement
        )


# ---------------------------------------------------------------------------
# Step penalty accumulation
# ---------------------------------------------------------------------------

def test_step_penalty_accumulates_over_10_steps():
    """Summing rewards over 10 no-capture steps gives 10 × STEP_PENALTY per seeker."""
    # Positions fixed — no capture, no movement, no shaping
    state = _state({"seeker_0": (0, 0), "hider": (9, 9)})

    total = 0.0
    for _ in range(10):
        rewards = compute_rewards(state, {}, state, n_seekers=1)
        total += rewards["seeker_0"]

    assert total == pytest.approx(10 * STEP_PENALTY)


# ---------------------------------------------------------------------------
# Distance shaping
# ---------------------------------------------------------------------------

def test_positive_shaping_when_approaching():
    """Moving one step closer to hider gives +DISTANCE_SHAPING reward."""
    prev = _state({"seeker_0": (0, 0), "hider": (5, 5)})
    curr = _state({"seeker_0": (1, 0), "hider": (5, 5)})  # 1 step closer (row)

    rewards = compute_rewards(curr, {}, prev, n_seekers=1)
    # prev_dist = 10, curr_dist = 9 → shaping = 0.1 × 1 = 0.1
    assert rewards["seeker_0"] == pytest.approx(STEP_PENALTY + DISTANCE_SHAPING * 1)


def test_negative_shaping_when_retreating():
    """Moving one step further from hider gives -DISTANCE_SHAPING reward."""
    prev = _state({"seeker_0": (1, 0), "hider": (5, 5)})
    curr = _state({"seeker_0": (0, 0), "hider": (5, 5)})  # 1 step further

    rewards = compute_rewards(curr, {}, prev, n_seekers=1)
    # prev_dist = 9, curr_dist = 10 → shaping = 0.1 × -1 = -0.1
    assert rewards["seeker_0"] == pytest.approx(STEP_PENALTY - DISTANCE_SHAPING * 1)


def test_no_shaping_when_stationary():
    """No movement means zero distance-shaping contribution."""
    state = _state({"seeker_0": (3, 3), "hider": (7, 7)})
    rewards = compute_rewards(state, {}, state, n_seekers=1)
    assert rewards["seeker_0"] == pytest.approx(STEP_PENALTY)


# ---------------------------------------------------------------------------
# Hider always gets zero
# ---------------------------------------------------------------------------

def test_hider_always_zero():
    """Hider reward is always 0.0 regardless of game state."""
    state = _state({"seeker_0": (5, 5), "hider": (5, 5)})  # capture step
    rewards = compute_rewards(state, {}, state, n_seekers=1)
    assert rewards["hider"] == 0.0
