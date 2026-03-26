"""
Greedy heuristic pursuer using BFS shortest-path navigation.
Owner: B (RL Training Lead)
Ticket: RL-01

The GreedyAgent requires no training and serves as the performance floor
baseline. It always moves toward the hider's last known position via BFS
on the obstacle-free grid. Behaviour summary:

  - Hider visible or last known position available → BFS to that position,
    return first-step action.
  - No known position (never observed hider) → uniform random directional
    action (no NOOP).
  - Path blocked by obstacles with no route → NOOP.

Observation indices and normalisation follow env/schema.py (ENV-04 locked):
  obs[0] = agent_x = row position, normalised to [0, 1]
  obs[1] = agent_y = col position, normalised to [0, 1]
  obs[2] = hider_x = hider row, normalised; SENTINEL (-1.0) if occluded
  obs[3] = hider_y = hider col, normalised; SENTINEL (-1.0) if occluded
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np

from env.schema import SENTINEL

# ---------------------------------------------------------------------------
# Local action constants — match env/schema.py ACTION_MAP and env._ACTION_DELTAS
# ---------------------------------------------------------------------------
_ACTION_NOOP  = 0
_ACTION_UP    = 1   # row - 1
_ACTION_DOWN  = 2   # row + 1
_ACTION_LEFT  = 3   # col - 1
_ACTION_RIGHT = 4   # col + 1
_N_ACTIONS    = 5   # Discrete(5)

# (row_delta, col_delta) — matches env._ACTION_DELTAS and schema.py ACTION_MAP comments.
# Positions are always (row, col) tuples; obstacle_map is indexed [row, col].
_ACTION_DELTAS: "Dict[int, Tuple[int, int]]" = {
    _ACTION_NOOP:  ( 0,  0),
    _ACTION_UP:    (-1,  0),
    _ACTION_DOWN:  ( 1,  0),
    _ACTION_LEFT:  ( 0, -1),
    _ACTION_RIGHT: ( 0,  1),
}

# Observation field indices (schema.py OBS_FIELDS, ENV-04 locked)
_OBS_AGENT_ROW = 0   # agent_x = row, normalised [0, 1]
_OBS_AGENT_COL = 1   # agent_y = col, normalised [0, 1]
_OBS_HIDER_ROW = 2   # hider_x = row, normalised; SENTINEL if occluded
_OBS_HIDER_COL = 3   # hider_y = col, normalised; SENTINEL if occluded


class GreedyAgent:
    """BFS-based greedy pursuer for the Byzantine Pursuit gridworld.

    Parameters
    ----------
    agent_id:
        Identifier string, e.g. ``"seeker_0"``.
    grid_size:
        Side length of the square grid (used for denormalisation and BFS bounds).
    seed:
        Optional RNG seed for reproducible random fallback behaviour.
    """

    # How many steps to persist in one exploration direction before
    # randomly reconsidering. Higher = straighter corridors, less backtracking.
    _EXPLORE_PERSIST: int = 6

    def __init__(self, agent_id: str, grid_size: int, seed: Optional[int] = None) -> None:
        self.agent_id = agent_id
        self.grid_size = grid_size
        self._rng = np.random.default_rng(seed)
        self._explore_action: Optional[int] = None  # current exploration direction
        self._explore_steps: int = 0                # steps since last direction change

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray, obstacle_map: np.ndarray) -> int:
        """Choose the next action via BFS toward the hider's last known position.

        Parameters
        ----------
        observation:
            Flat float32 observation vector. Indices 0–3 are read:
              [0] agent row, normalised to [0, 1]
              [1] agent col, normalised to [0, 1]
              [2] hider row, normalised; SENTINEL if unknown
              [3] hider col, normalised; SENTINEL if unknown
        obstacle_map:
            Boolean array of shape ``(grid_size, grid_size)``.
            ``True`` marks an impassable cell. Indexed as ``[row, col]``.

        Returns
        -------
        int
            Action in ``{0, 1, 2, 3, 4}`` per schema.py ACTION_MAP.
        """
        gs = self.grid_size
        norm = float(gs - 1)

        self_row = int(round(float(observation[_OBS_AGENT_ROW]) * norm))
        self_col = int(round(float(observation[_OBS_AGENT_COL]) * norm))

        hider_row_n = float(observation[_OBS_HIDER_ROW])
        hider_col_n = float(observation[_OBS_HIDER_COL])

        # Fallback: hider not visible — persistent random walk to explore
        if hider_row_n == SENTINEL and hider_col_n == SENTINEL:
            return self._explore(self_row, self_col, obstacle_map)

        target = (int(round(hider_row_n * norm)), int(round(hider_col_n * norm)))
        start = (self_row, self_col)

        if start == target:
            return _ACTION_NOOP

        action = self._bfs(start, target, obstacle_map)
        return action if action is not None else _ACTION_NOOP

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _explore(self, row: int, col: int, obstacle_map: np.ndarray) -> int:
        """Return an action for systematic exploration when the hider is unseen.

        Uses a persistent random walk: the agent holds one direction for
        ``_EXPLORE_PERSIST`` steps, then re-samples. If the chosen direction
        is immediately blocked, a new direction is picked on the spot.
        This avoids the back-and-forth oscillation of a memoryless random walk
        and covers the grid far more efficiently.
        """
        gs = self.grid_size

        def _passable(action: int) -> bool:
            dr, dc = _ACTION_DELTAS[action]
            nr, nc = row + dr, col + dc
            return (0 <= nr < gs and 0 <= nc < gs and not obstacle_map[nr, nc])

        # Reconsider direction if: no direction yet, persistence expired, or blocked
        if (
            self._explore_action is None
            or self._explore_steps >= self._EXPLORE_PERSIST
            or not _passable(self._explore_action)
        ):
            passable = [a for a in range(1, _N_ACTIONS) if _passable(a)]
            if not passable:
                return _ACTION_NOOP
            self._explore_action = int(self._rng.choice(passable))
            self._explore_steps = 0

        self._explore_steps += 1
        return self._explore_action

    def _bfs(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_map: np.ndarray,
    ) -> Optional[int]:
        """Run BFS from *start* to *goal* and return the first action taken.

        Parameters
        ----------
        start:
            ``(row, col)`` grid coordinate of this agent.
        goal:
            ``(row, col)`` target coordinate (hider's last known position).
        obstacle_map:
            Boolean grid; ``obstacle_map[row, col]`` is ``True`` if impassable.

        Returns
        -------
        int or None
            Action ID for the first step along the shortest path, or ``None``
            if no path exists (fully blocked).
        """
        gs = self.grid_size
        visited: Set[Tuple[int, int]] = {start}
        # Queue entries: (current_pos, first_action_taken)
        queue: Deque[Tuple[Tuple[int, int], int]] = deque()

        # Seed queue with immediate neighbours
        for action, (dr, dc) in _ACTION_DELTAS.items():
            if action == _ACTION_NOOP:
                continue
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < gs and 0 <= nc < gs and not obstacle_map[nr, nc]:
                if (nr, nc) == goal:
                    return action
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), action))

        while queue:
            (cr, cc), first_action = queue.popleft()
            for action, (dr, dc) in _ACTION_DELTAS.items():
                if action == _ACTION_NOOP:
                    continue
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < gs and 0 <= nc < gs and not obstacle_map[nr, nc]:
                    if (nr, nc) == goal:
                        return first_action
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), first_action))

        return None  # no path found
