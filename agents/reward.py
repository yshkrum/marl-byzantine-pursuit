"""
RL-04: Frozen reward function for Byzantine Pursuit.

FROZEN AFTER TEAM SIGN-OFF 
Do NOT modify reward constants after team agreement.
Any change invalidates comparisons between experiments run before and after.

Owner: B (RL Training Lead)
Imported by: env/pursuit_env.py (step()), agents/ppo/ippo.py, agents/mappo/mappo.py

Reward components (all seekers; hider receives 0.0):
  CAPTURE_REWARD       +10.0   for the seeker that tagged the hider
  TEAM_CAPTURE_BONUS   + 5.0   split equally among ALL seekers on capture
  STEP_PENALTY         - 0.01  per step per agent (encourages fast capture)
  DISTANCE_SHAPING     ± 0.3   per seeker per step:
                                 +0.3 × reduction in Manhattan distance to hider
                                 −0.3 × increase in Manhattan distance to hider

v1.0.0 → v2.0.0 change: DISTANCE_SHAPING raised 0.1 → 0.3.
Reason: v1.0.0 made passive trapping (sit still, random hider walks in) more
rewarding than active pursuit. v2.0.0 tips the balance toward chasing.
All iPPO and MAPPO runs must use v2.0.0 for valid comparisons.

Paper citation: reward_version = "2.0.0"
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Frozen constants — cite these directly in §3.2 of the paper
# ---------------------------------------------------------------------------

reward_version: str = "2.0.0"

CAPTURE_REWARD: float = 10.0
"""Individual reward for the seeker that occupies the hider's cell."""

TEAM_CAPTURE_BONUS: float = 5.0
"""Shared bonus split equally among all n_seekers on a capture step.
Encourages cooperative behaviour even when only one agent tags the hider."""

STEP_PENALTY: float = -0.01
"""Per-step cost applied to every seeker. Penalises slow capture and
prevents agents from idling to avoid negative distance-shaping signals."""

DISTANCE_SHAPING: float = 0.3
"""Coefficient for potential-based distance shaping per seeker.
Reward = DISTANCE_SHAPING × (prev_manhattan − curr_manhattan).
Positive when the seeker moves closer; negative when it moves away.
Applied identically across iPPO, MAPPO, and all Byzantine conditions.
Raised from 0.1 (v1.0.0) to 0.3 (v2.0.0) to incentivise active pursuit
over passive trapping against a random hider."""


# ---------------------------------------------------------------------------
# State type alias
# ---------------------------------------------------------------------------

# state / prev_state dicts must contain:
#   "positions": {agent_id: (row, col)}  — matches env.positions directly
EnvState = Dict[str, Dict[str, Tuple[int, int]]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rewards(
    state: Dict[str, Dict[str, Tuple[int, int]]],
    actions: Dict[str, int],
    prev_state: Dict[str, Dict[str, Tuple[int, int]]],
    n_seekers: int,
) -> Dict[str, float]:
    """Compute per-agent rewards for one environment step.

    Parameters
    ----------
    state:
        Current env state after actions are applied.
        Must contain ``"positions": {agent_id: (row, col)}``.
    actions:
        ``{agent_id: action_int}`` — recorded for future extensions
        (e.g. communication-cost penalties). Not used by v1.0.0 components.
    prev_state:
        Env state before actions were applied.
        Must contain ``"positions": {agent_id: (row, col)}``.
    n_seekers:
        Total number of seeker agents (including Byzantine ones).
        Used to split TEAM_CAPTURE_BONUS equally.

    Returns
    -------
    dict[str, float]
        ``{agent_id: reward}`` for every agent in ``state["positions"]``.
        Hider always receives 0.0.
    """
    positions: Dict[str, Tuple[int, int]] = state["positions"]
    prev_positions: Dict[str, Tuple[int, int]] = prev_state["positions"]

    hider_pos = positions["hider"]
    prev_hider_pos = prev_positions.get("hider", hider_pos)

    # Identify which seekers captured (same cell as hider after step)
    capturers = [
        aid for aid, pos in positions.items()
        if aid.startswith("seeker_") and pos == hider_pos
    ]
    captured = len(capturers) > 0
    team_share = TEAM_CAPTURE_BONUS / n_seekers if captured else 0.0

    rewards: Dict[str, float] = {}

    for agent_id, pos in positions.items():
        if agent_id == "hider":
            rewards[agent_id] = 0.0
            continue

        r: float = STEP_PENALTY

        # Capture reward — only for the seeker(s) that tagged the hider
        if agent_id in capturers:
            r += CAPTURE_REWARD

        # Team bonus — shared equally on any capture step
        if captured:
            r += team_share

        # Distance shaping — based on Manhattan distance change to hider
        prev_pos = prev_positions.get(agent_id, pos)
        prev_dist = abs(prev_pos[0] - prev_hider_pos[0]) + abs(prev_pos[1] - prev_hider_pos[1])
        curr_dist = abs(pos[0] - hider_pos[0]) + abs(pos[1] - hider_pos[1])
        r += DISTANCE_SHAPING * (prev_dist - curr_dist)

        rewards[agent_id] = r

    return rewards
