"""
Tests for the greedy BFS pursuer baseline.
Owner: B (RL Training Lead)
Ticket: RL-01
Run: pytest tests/test_greedy.py -v

Observation convention (schema.py ENV-04 locked):
  obs[0] = agent row, normalised to [0, 1]  (agent_x in schema)
  obs[1] = agent col, normalised to [0, 1]  (agent_y in schema)
  obs[2] = hider row, normalised; -1.0 if unknown  (hider_x)
  obs[3] = hider col, normalised; -1.0 if unknown  (hider_y)

All positions in tests are (row, col) tuples. obstacle_map indexed [row, col].
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.greedy.greedy_agent import GreedyAgent

GRID_SIZE = 10
_NORM = float(GRID_SIZE - 1)   # 9.0 for a 10×10 grid
_SENTINEL = -1.0

# Local action constants matching schema.py ACTION_MAP
_ACTION_NOOP  = 0
_ACTION_UP    = 1   # row - 1
_ACTION_DOWN  = 2   # row + 1
_ACTION_LEFT  = 3   # col - 1
_ACTION_RIGHT = 4   # col + 1

# (row_delta, col_delta) per schema.py ACTION_MAP comments
_ACTION_DELTAS = {
    _ACTION_NOOP:  ( 0,  0),
    _ACTION_UP:    (-1,  0),
    _ACTION_DOWN:  ( 1,  0),
    _ACTION_LEFT:  ( 0, -1),
    _ACTION_RIGHT: ( 0,  1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    self_row: int,
    self_col: int,
    hider_row: float,
    hider_col: float,
) -> np.ndarray:
    """Build a minimal 4-element observation vector with normalised values.

    Positions are in (row, col) form. Pass ``_SENTINEL`` for hider_row /
    hider_col to signal that the hider has never been observed.
    The greedy agent only reads indices 0–3, so the obstacle-map patch and
    message slots are omitted here.
    """
    obs = np.full(4, _SENTINEL, dtype=np.float32)
    obs[0] = float(self_row) / _NORM
    obs[1] = float(self_col) / _NORM
    if hider_row != _SENTINEL:
        obs[2] = float(hider_row) / _NORM
    if hider_col != _SENTINEL:
        obs[3] = float(hider_col) / _NORM
    return obs


def _apply_action(pos: list, action: int) -> list:
    """Return new [row, col] after applying *action* (no bounds check)."""
    dr, dc = _ACTION_DELTAS[action]
    return [pos[0] + dr, pos[1] + dc]


def _empty_obstacle_map() -> np.ndarray:
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)


# ---------------------------------------------------------------------------
# Core BFS correctness test (per ticket spec)
# ---------------------------------------------------------------------------

def test_greedy_moves_optimally_toward_hider_5_steps():
    """Agent at (row=0, col=0), hider at (row=5, col=5): Manhattan distance must
    decrease by 1 each step for 5 consecutive steps on a 10×10 empty grid.

    Verifies BFS returns an optimal first action and that simulating the
    resulting path makes steady progress toward the hider.
    """
    agent = GreedyAgent("seeker_0", grid_size=GRID_SIZE, seed=0)
    obstacle_map = _empty_obstacle_map()

    pos = [0, 0]    # [row, col]
    hider = (5, 5)  # (row, col)

    for step in range(5):
        prev_dist = abs(pos[0] - hider[0]) + abs(pos[1] - hider[1])
        obs = _make_obs(pos[0], pos[1], hider[0], hider[1])
        action = agent.act(obs, obstacle_map)
        pos = _apply_action(pos, action)
        curr_dist = abs(pos[0] - hider[0]) + abs(pos[1] - hider[1])
        assert curr_dist == prev_dist - 1, (
            f"Step {step + 1}: expected distance {prev_dist - 1}, got {curr_dist} "
            f"(action={action}, pos={pos})"
        )

    assert abs(pos[0] - hider[0]) + abs(pos[1] - hider[1]) == 5


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_greedy_import():
    """GreedyAgent is importable."""
    assert GreedyAgent is not None


def test_greedy_noop_when_already_at_target():
    """Agent already at the hider's position returns NOOP."""
    agent = GreedyAgent("seeker_0", grid_size=GRID_SIZE)
    obs = _make_obs(3, 3, 3, 3)
    action = agent.act(obs, _empty_obstacle_map())
    assert action == _ACTION_NOOP


def test_greedy_explores_when_hider_unknown():
    """When the hider is unseen the agent explores: never returns NOOP and
    eventually changes direction (persistent walk, not frozen).

    The agent holds one direction for _EXPLORE_PERSIST steps then re-samples,
    so variation is guaranteed across a longer run.
    """
    agent = GreedyAgent("seeker_0", grid_size=GRID_SIZE, seed=7)
    obs = _make_obs(5, 5, _SENTINEL, _SENTINEL)   # centre of grid — all 4 dirs passable
    obstacle_map = _empty_obstacle_map()

    actions = [agent.act(obs, obstacle_map) for _ in range(50)]

    assert _ACTION_NOOP not in actions, "Exploration should never return NOOP"
    assert len(set(actions)) > 1, "Agent must change direction at least once in 50 steps"


def test_greedy_noop_when_path_fully_blocked():
    """Agent surrounded by obstacles on all four sides returns NOOP."""
    agent = GreedyAgent("seeker_0", grid_size=GRID_SIZE)
    obstacle_map = _empty_obstacle_map()
    # Surround agent at (row=5, col=5) on all four sides
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        obstacle_map[5 + dr, 5 + dc] = True
    obs = _make_obs(5, 5, 9, 9)
    action = agent.act(obs, obstacle_map)
    assert action == _ACTION_NOOP


def test_greedy_navigates_around_wall():
    """Agent finds a detour when a vertical wall blocks the direct path.

    Setup: agent at (row=0, col=0), hider at (row=0, col=9).
    Wall: col=3 is blocked for rows 0–7; gap exists at rows 8–9.
    Agent must go down to row≥8, cross col=3, then navigate to the hider.
    30 steps is sufficient (optimal path ≈ 25 steps).
    """
    agent = GreedyAgent("seeker_0", grid_size=GRID_SIZE)
    obstacle_map = _empty_obstacle_map()
    # Vertical wall at col=3, rows 0–7 (gap at rows 8–9)
    for row in range(8):
        obstacle_map[row, 3] = True

    pos = [0, 0]       # [row, col]
    hider = (0, 9)     # (row, col)

    captured = False
    for _ in range(30):
        obs = _make_obs(pos[0], pos[1], hider[0], hider[1])
        action = agent.act(obs, obstacle_map)
        pos = _apply_action(pos, action)
        if pos == list(hider):
            captured = True
            break

    assert captured, f"Agent failed to reach hider around wall; final pos={pos}"


def test_greedy_action_stays_in_bounds():
    """BFS never returns an action that would move the agent off the grid."""
    agent = GreedyAgent("seeker_0", grid_size=GRID_SIZE)
    obstacle_map = _empty_obstacle_map()
    # Agent at top-left corner, hider at bottom-right
    obs = _make_obs(0, 0, 9, 9)
    action = agent.act(obs, obstacle_map)
    pos = _apply_action([0, 0], action)
    assert 0 <= pos[0] < GRID_SIZE
    assert 0 <= pos[1] < GRID_SIZE
