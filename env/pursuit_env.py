"""
ENV-01 / ENV-02 / ENV-03: ByzantinePursuitEnv implementation.

Implements the PettingZoo AEC API for a 2-D gridworld pursuit-evasion game
with Byzantine-corrupted inter-agent communication.

Owner: Role A – Environment Engineer
Paper section: §3.1 Environment Design

Observation vector layout (locked in ENV-04 / schema.py):
    [0]         agent_x            – row of observing agent, normalised to [0,1]
    [1]         agent_y            – col of observing agent, normalised to [0,1]
    [2]         hider_x            – row of hider, normalised; sentinel -1 if occluded
    [3]         hider_y            – col of hider, normalised; sentinel -1 if occluded
    [4 : 4+M]   local_obstacle_map – flattened obstacle patch of shape (2r+1)²
                                     when obs_radius=r; full grid (grid_size²) when
                                     obs_radius=None. 1.0=obstacle, 0.0=passable.
                                     Cells outside grid boundary padded with 1.0.
    [4+M : end] received_messages  – 2*(n_seekers-1) floats: per-peer
                                     (believed_hider_x, believed_hider_y);
                                     sentinel -1 for unknown / not yet received.

    Sentinel value: -1.0 for any unknown or occluded field.

Message struct schema (to be finalised with Role C / BYZ-01):
    Message(
        sender_id:        str   – agent id of the sender
        believed_hider_x: float – sender's belief of hider row (normalised 0-1)
        believed_hider_y: float – sender's belief of hider col (normalised 0-1)
    )
    Byzantine agents corrupt (believed_hider_x, believed_hider_y).
    Dataclass lives in env/schema.py once ENV-04 is locked.

Action space (Discrete 5):
    0 = NOOP
    1 = UP    (row - 1)
    2 = DOWN  (row + 1)
    3 = LEFT  (col - 1)
    4 = RIGHT (col + 1)

Episode termination:
    - Capture:  any seeker occupies the same cell as the hider
    - Timeout:  step_count >= max_steps
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import numpy.random


# ---------------------------------------------------------------------------
# PettingZoo imports – graceful stub if not yet installed so tests can import
# ---------------------------------------------------------------------------
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils.agent_selector import agent_selector
    import gymnasium.spaces as spaces
except ImportError:  # pragma: no cover
    class AECEnv:  # type: ignore[no-redef]
        """Minimal stub when pettingzoo is not installed."""

    class _FakeSelector:
        def __init__(self, agents: list[str]) -> None:
            self._agents = agents
            self._idx = 0

        def next(self) -> str:
            agent = self._agents[self._idx % len(self._agents)]
            self._idx += 1
            return agent

        def is_last(self) -> bool:
            return self._idx % len(self._agents) == 0

    class _FakeSpaces:
        @staticmethod
        def Box(*args: Any, **kwargs: Any) -> Any:
            return None

        @staticmethod
        def Discrete(*args: Any, **kwargs: Any) -> Any:
            return None

    agent_selector = _FakeSelector  # type: ignore[misc]
    spaces = _FakeSpaces()  # type: ignore[assignment]


# Sentinel for unknown / occluded fields in the observation vector
_SENTINEL: float = -1.0

# Movement deltas: (row_delta, col_delta) indexed by action integer
_ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (0,  0),   # NOOP
    1: (-1, 0),   # UP
    2: (1,  0),   # DOWN
    3: (0, -1),   # LEFT
    4: (0,  1),   # RIGHT
}


class ByzantinePursuitEnv(AECEnv):
    """
    Multi-agent pursuit-evasion environment with Byzantine communication.

    N seeker agents cooperate to capture 1 hider on a 2-D gridworld maze.
    A fraction *f* of seekers are Byzantine: they move honestly but corrupt
    their outgoing communication messages.

    Parameters
    ----------
    n_seekers : int
        Number of seeker agents (default 4).
    grid_size : int
        Side length of the square grid in cells (default 20).
    obs_radius : int or None
        Half-side of the square field-of-view patch around each agent.
        ``None`` → full observability (entire grid visible, no occlusion).
    obstacle_density : float
        Target fraction of non-border interior cells that become obstacles
        (default 0.2). Actual density may be slightly lower after the
        connectivity-repair pass.
    byzantine_fraction : float
        Fraction of seekers that are Byzantine (default 0.0).
        ``n_byzantine = floor(n_seekers * byzantine_fraction)``.
    max_steps : int
        Maximum steps per episode before timeout (default 500).
    seed : int
        Master seed for all stochastic operations (default 42).
    fixed_maze : bool
        If ``True``, generate the maze only on the first ``reset()`` call
        and reuse it for all subsequent episodes. Required for reproducible
        cross-condition comparisons (Exp 1 holds maze constant across
        Byzantine fractions). Default ``False``.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "name": "byzantine_pursuit_v0",
        "is_parallelizable": True,
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_seekers: int = 4,
        grid_size: int = 20,
        obs_radius: int | None = None,
        obstacle_density: float = 0.2,
        byzantine_fraction: float = 0.0,
        max_steps: int = 500,
        seed: int = 42,
        fixed_maze: bool = False,
        protocol: Any | None = None,
        byzantine_agents: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # --- Core parameters -------------------------------------------
        self.n_seekers: int = n_seekers
        self.grid_size: int = grid_size
        self.obs_radius: int | None = obs_radius
        self.obstacle_density: float = obstacle_density
        self.byzantine_fraction: float = byzantine_fraction
        self.max_steps: int = max_steps
        self.fixed_maze: bool = fixed_maze

        # --- Byzantine agent count (ENV-01) ----------------------------
        self.n_byzantine: int = math.floor(n_seekers * byzantine_fraction)

        # --- Agent registry (PettingZoo convention) --------------------
        self.possible_agents: list[str] = (
            [f"seeker_{i}" for i in range(n_seekers)] + ["hider"]
        )
        self.agents: list[str] = []

        # --- Deterministic RNG -----------------------------------------
        # Single source of randomness for maze generation, agent spawning,
        # and episode dynamics. Must never be replaced mid-episode.
        self.rng: numpy.random.Generator = np.random.default_rng(seed)
        self._master_seed: int = seed

        # --- Observation space dimension (ENV-02/03) -------------------
        # local obstacle map size depends on obs_radius:
        #   full obs  → entire grid_size × grid_size
        #   partial   → (2*obs_radius+1) × (2*obs_radius+1) patch
        if obs_radius is None:
            _local_map_size: int = grid_size * grid_size
        else:
            _patch_side: int = 2 * obs_radius + 1
            _local_map_size = _patch_side * _patch_side

        # 4 scalars + local map + message slots
        _msg_size: int = 2 * (n_seekers - 1)
        _obs_dim: int = 4 + _local_map_size + _msg_size

        # TODO (ENV-04): replace _obs_dim with schema.OBS_DIM once schema.py
        # is locked with Role C.

        self.observation_spaces: dict[str, Any] = {
            agent: spaces.Box(
                low=_SENTINEL,
                high=float(grid_size),
                shape=(_obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self.action_spaces: dict[str, Any] = {
            agent: spaces.Discrete(5)
            for agent in self.possible_agents
        }

        # --- Internal state (uninitialised until reset()) --------------
        self.grid: np.ndarray | None = None     # (grid_size, grid_size) bool
        self.positions: dict[str, tuple[int, int]] = {}  # agent → (row, col)
        self._step_count: int = 0
        self._agent_selector: Any = None

        # Message buffer: seeker_id → (believed_hider_x, believed_hider_y)
        # Populated with sentinels at reset; Role C overwrites with real msgs.
        self._message_buffer: dict[str, tuple[float, float]] = {}

        # BYZ-01/02: communication protocol and Byzantine agent registry.
        # protocol=None → no communication (message slots stay sentinel).
        # byzantine_agents maps seeker_id → ByzantineAgent for corrupt seekers.
        self._protocol: Any | None = protocol
        self._byzantine_agents: dict[str, Any] = byzantine_agents or {}

        # PettingZoo AEC bookkeeping dicts – populated in reset()
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._cumulative_rewards: dict[str, float] = {}

    # ------------------------------------------------------------------
    # PettingZoo AEC API – spaces
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> Any:
        """Return the observation space for *agent*."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Any:
        """Return the action space for *agent*."""
        return self.action_spaces[agent]

    # ------------------------------------------------------------------
    # ENV-02: Maze generation
    # ------------------------------------------------------------------

    def _is_connected(self, grid: np.ndarray) -> bool:
        """
        BFS flood-fill connectivity check (4-connectivity).

        Returns ``True`` if all passable cells (grid == False) form a single
        connected component. Deterministic: always starts from the first
        open cell in row-major order, so it does NOT consume ``self.rng``.

        Parameters
        ----------
        grid : np.ndarray
            Boolean obstacle map; True = obstacle, False = passable.

        Returns
        -------
        bool
            ``True`` when fully connected (or when there are zero open cells).
        """
        rows, cols = np.where(~grid)
        if len(rows) == 0:
            return True  # vacuously connected

        total_open = len(rows)
        start = (int(rows[0]), int(cols[0]))
        gs = grid.shape[0]

        visited: set[tuple[int, int]] = {start}
        queue: deque[tuple[int, int]] = deque([start])

        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if (
                    nb not in visited
                    and 0 <= nb[0] < gs
                    and 0 <= nb[1] < gs
                    and not grid[nb[0], nb[1]]
                ):
                    visited.add(nb)
                    queue.append(nb)

        return len(visited) == total_open

    def _generate_maze(self) -> None:
        """
        ENV-02: Procedural obstacle placement with flood-fill connectivity guarantee.

        Algorithm
        ---------
        1. Initialise grid with all border cells as obstacles, interior open.
        2. Randomly designate ``round(obstacle_density * n_interior)`` interior
           cells as obstacles (sampled without replacement via ``self.rng``).
        3. BFS flood-fill to test connectivity of all open cells.
        4. While disconnected: pick a random interior obstacle and remove it,
           repeat until the open-cell graph is fully connected.

        All randomness flows through ``self.rng`` for reproducibility.

        Sets
        ----
        self.grid : np.ndarray
            Boolean array of shape ``(grid_size, grid_size)``.
            ``True`` = obstacle, ``False`` = passable.
        """
        gs = self.grid_size

        # Step 1: open interior, walled border
        grid = np.zeros((gs, gs), dtype=bool)
        grid[0, :]  = True   # top border
        grid[-1, :] = True   # bottom border
        grid[:, 0]  = True   # left border
        grid[:, -1] = True   # right border

        # Step 2: greedily place interior obstacles while preserving connectivity.
        #
        # Strategy: shuffle the interior cell list, then try to mark each cell
        # as an obstacle. Keep it only if the grid remains fully connected after
        # the addition (checked via BFS). Stop once we have placed the target
        # number of obstacles.
        #
        # Why greedy instead of place-all-then-repair:
        #   Random batch placement can create many isolated pockets that require
        #   removing O(n_target) obstacles to reconnect — destroying the target
        #   density. The greedy approach never needs to remove obstacles, so the
        #   actual density equals the target exactly (or within ≤1 cell of it).
        interior_rc = [
            (r, c)
            for r in range(1, gs - 1)
            for c in range(1, gs - 1)
        ]
        n_interior = len(interior_rc)
        n_target = round(self.obstacle_density * n_interior)

        # Shuffle candidates with the seeded rng for reproducibility
        perm = self.rng.permutation(n_interior)

        placed = 0
        for idx in perm:
            if placed == n_target:
                break
            r, c = interior_rc[int(idx)]
            grid[r, c] = True              # tentatively add obstacle
            if self._is_connected(grid):
                placed += 1                # connectivity preserved → keep it
            else:
                grid[r, c] = False         # would disconnect → revert

        self.grid = grid

    # ------------------------------------------------------------------
    # ENV-03: Line-of-sight helpers
    # ------------------------------------------------------------------

    def _bresenham_cells(
        self, r0: int, c0: int, r1: int, c1: int
    ) -> list[tuple[int, int]]:
        """
        Return all grid cells along the Bresenham line from (r0,c0) to (r1,c1),
        inclusive of both endpoints.

        Uses the standard integer-arithmetic Bresenham algorithm so that the
        result is deterministic and symmetric (same cells regardless of direction).
        """
        cells: list[tuple[int, int]] = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r, c = r0, c0
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1
        err = dr - dc

        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

        return cells

    def _has_line_of_sight(
        self, r0: int, c0: int, r1: int, c1: int
    ) -> bool:
        """
        ENV-03: Bresenham line-of-sight check.

        Returns ``True`` if no obstacle cell lies strictly between
        (r0, c0) and (r1, c1) on the Bresenham path. The endpoints
        themselves are not checked (agents may stand on open cells).

        Parameters
        ----------
        r0, c0 : int
            Observer cell (row, col).
        r1, c1 : int
            Target cell (row, col).

        Returns
        -------
        bool
        """
        cells = self._bresenham_cells(r0, c0, r1, c1)
        # Intermediate cells only (exclude start and end)
        for r, c in cells[1:-1]:
            if self.grid[r, c]:  # type: ignore[index]
                return False
        return True

    # ------------------------------------------------------------------
    # ENV-03: Observation construction
    # ------------------------------------------------------------------

    def _get_observation(self, agent: str) -> np.ndarray:
        """
        ENV-03: Build the flat float32 observation vector for *agent*.

        Observation vector layout
        -------------------------
        Index  Field
        -----  ---------------------------------------------------------------
        0      agent_x  – agent's row position, normalised to [0, 1]
        1      agent_y  – agent's col position, normalised to [0, 1]
        2      hider_x  – hider's row, normalised; -1.0 if outside FoV or occluded
        3      hider_y  – hider's col, normalised; -1.0 if outside FoV or occluded
        4:4+M  local_obstacle_map – flattened float32 obstacle patch
                   full obs  → entire grid, row-major, shape (grid_size²,)
                   partial   → (2*obs_radius+1)² patch centred on agent;
                               out-of-bounds cells padded with 1.0 (wall)
        4+M:   received_messages – 2*(n_seekers-1) floats, per peer seeker:
                   (believed_hider_x, believed_hider_y) from self._message_buffer;
                   sentinel -1.0 until Role C populates the buffer (BYZ-01).

        Visibility rule (partial obs only):
            The hider is visible if:
              (a) Manhattan distance to agent ≤ obs_radius, AND
              (b) no obstacle lies on the Bresenham path between them.

        Parameters
        ----------
        agent : str
            Agent id, e.g. ``"seeker_0"`` or ``"hider"``.

        Returns
        -------
        np.ndarray
            dtype=float32, shape matching ``self.observation_spaces[agent]``.
        """
        assert self.grid is not None, "Call reset() before observe()."
        gs = self.grid_size
        r_a, c_a = self.positions[agent]
        r_h, c_h = self.positions["hider"]

        # --- Agent and hider positions (normalised) --------------------
        norm = float(gs - 1)
        agent_x = r_a / norm
        agent_y = c_a / norm

        if self.obs_radius is None:
            # Full observability: hider always visible
            hider_x = float(r_h) / norm
            hider_y = float(c_h) / norm
        else:
            # Partial: Manhattan-distance gate + Bresenham LoS check
            manhattan = abs(r_h - r_a) + abs(c_h - c_a)
            if (
                manhattan <= self.obs_radius
                and self._has_line_of_sight(r_a, c_a, r_h, c_h)
            ):
                hider_x = float(r_h) / norm
                hider_y = float(c_h) / norm
            else:
                hider_x = _SENTINEL
                hider_y = _SENTINEL

        # --- Local obstacle map ----------------------------------------
        if self.obs_radius is None:
            # Full grid in row-major order
            obs_patch = self.grid.astype(np.float32).flatten()
        else:
            rad = self.obs_radius
            side = 2 * rad + 1
            patch = np.ones((side, side), dtype=np.float32)  # default = wall
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    nr, nc = r_a + dr, c_a + dc
                    if 0 <= nr < gs and 0 <= nc < gs:
                        patch[dr + rad, dc + rad] = float(self.grid[nr, nc])
            obs_patch = patch.flatten()

        # --- Received message slots ------------------------------------
        # Always exactly n_seekers-1 slots so the obs vector shape is identical
        # for every agent (seekers and hider alike).
        # Ordered by seeker index for a stable layout that Role C can rely on.
        # Role C (BYZ-01) overwrites self._message_buffer before each call.
        msg_slots: list[float] = []
        slots_added = 0
        for peer in self.possible_agents:
            if slots_added == self.n_seekers - 1:
                break                                   # cap at n_seekers-1 slots
            if peer == "hider" or peer == agent:
                continue                                # skip non-seekers and self
            bx, by = self._message_buffer.get(peer, (_SENTINEL, _SENTINEL))
            msg_slots.extend([bx, by])
            slots_added += 1
        # Safety pad — triggers only if n_seekers == 1 (no peers)
        while len(msg_slots) < 2 * (self.n_seekers - 1):
            msg_slots.extend([_SENTINEL, _SENTINEL])

        # --- Assemble and return ---------------------------------------
        obs = np.concatenate([
            np.array([agent_x, agent_y, hider_x, hider_y], dtype=np.float32),
            obs_patch,
            np.array(msg_slots, dtype=np.float32),
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Steps
        -----
        1. Re-seed RNG if *seed* given.
        2. Regenerate maze (ENV-02) unless ``fixed_maze=True`` and a maze
           already exists.
        3. Spawn all agents at random, non-overlapping open cells.
        4. Reset message buffer to sentinel values.
        5. Build initial observations for every agent (ENV-03).
        6. Reset PettingZoo bookkeeping.

        Parameters
        ----------
        seed : int or None
            If provided, replaces the internal RNG for this episode.
        options : dict or None
            Reserved for future use (e.g. forced spawn positions).

        Returns
        -------
        observations : dict[str, np.ndarray]
            ``{agent_id: obs_vector}`` for every agent in ``self.agents``.
        infos : dict[str, dict]
            Auxiliary info dict (empty at reset).
        """
        # Step 1: optional re-seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Restore live agent list
        self.agents = list(self.possible_agents)
        self._step_count = 0

        # PettingZoo bookkeeping
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Step 2: maze generation (ENV-02)
        # Skip if fixed_maze is True and maze already exists from a prior reset.
        if not (self.fixed_maze and self.grid is not None):
            self._generate_maze()

        assert self.grid is not None  # guaranteed by _generate_maze

        # Step 3: spawn agents on distinct open (non-obstacle) cells
        open_cells = list(zip(*np.where(~self.grid)))
        assert len(open_cells) >= len(self.agents), (
            f"Not enough open cells ({len(open_cells)}) to place "
            f"{len(self.agents)} agents. Reduce obstacle_density."
        )
        chosen = self.rng.choice(
            len(open_cells), size=len(self.agents), replace=False
        )
        self.positions = {
            agent: (int(open_cells[int(idx)][0]), int(open_cells[int(idx)][1]))
            for agent, idx in zip(self.agents, chosen)
        }

        # Step 4: reset message buffer – all sentinels until Role C fills it
        self._message_buffer = {
            s: (_SENTINEL, _SENTINEL)
            for s in self.possible_agents
            if s != "hider"
        }

        # BYZ-01/02: reset any per-episode protocol state (e.g. reputation scores)
        if self._protocol is not None:
            self._protocol.reset()

        # Step 5: initial observations (ENV-03)
        observations: dict[str, np.ndarray] = {
            agent: self._get_observation(agent) for agent in self.agents
        }

        return observations, dict(self.infos)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int) -> None:
        """
        Advance the environment by one agent step (AEC loop).

        The action applies to ``self.agent_selection``. After resolving
        movement the agent pointer rotates to the next live agent.

        Parameters
        ----------
        action : int
            Action for the current agent: 0=NOOP, 1=UP, 2=DOWN, 3=LEFT,
            4=RIGHT.

        Raises
        ------
        NotImplementedError
            Until reward.py (RL-04, Role B) is locked and integrated.
        AssertionError
            If called before ``reset()``.
        """
        assert self.grid is not None, "Call env.reset() before env.step()."

        agent = self.agent_selection

        # Skip already-dead agents (standard PettingZoo AEC convention)
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            self._cumulative_rewards[agent] = 0
            self.agent_selection = self._agent_selector.next()
            return

        # 1. Save positions before this agent moves (needed for distance shaping)
        prev_positions = dict(self.positions)

        # 2. Resolve movement — reject if obstacle or out of bounds
        row, col = self.positions[agent]
        dr, dc = _ACTION_DELTAS[action]
        nr, nc = row + dr, col + dc
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and not self.grid[nr, nc]:
            self.positions[agent] = (nr, nc)

        # 2b. BYZ-01/02: communication hook — update this seeker's message
        #     buffer entry after movement so obs[2:4] reflects new position.
        #     SilentByzantine → reset slot to sentinel; others → write buffer.
        if agent.startswith("seeker_") and self._protocol is not None:
            from comms.interface import EnvState
            _r_h, _c_h = self.positions["hider"]
            _norm = float(self.grid_size - 1)
            _state = EnvState(
                obs=self._get_observation(agent),
                step=self._step_count,
                grid_size=self.grid_size,
                true_hider_pos=(_r_h / _norm, _c_h / _norm),
            )
            _msg = self._protocol.send(agent, _state)
            if agent in self._byzantine_agents:
                _msg = self._byzantine_agents[agent].corrupt_message(_msg)
            if _msg is None:
                self._message_buffer[agent] = (_SENTINEL, _SENTINEL)
            else:
                self._message_buffer.update(self._protocol.receive([_msg]))

        # 3. Detect capture: any seeker occupies the hider's cell
        hider_pos = self.positions["hider"]
        captured = any(
            self.positions[a] == hider_pos
            for a in self.agents
            if a.startswith("seeker_")
        )

        # 4. Increment step count and detect timeout
        self._step_count += 1
        timeout = self._step_count >= self.max_steps

        # 5. Compute rewards via the frozen reward function (RL-04)
        from agents.reward import compute_rewards
        step_rewards = compute_rewards(
            state={"positions": self.positions},
            actions={agent: action},
            prev_state={"positions": prev_positions},
            n_seekers=self.n_seekers,
        )

        # 6. Set terminations / truncations for all agents
        if captured:
            for a in self.agents:
                self.terminations[a] = True
        if timeout:
            for a in self.agents:
                self.truncations[a] = True

        # 7. Update rewards and accumulate into cumulative totals
        self._cumulative_rewards[agent] = 0  # clear acting agent's slate first
        for a, r in step_rewards.items():
            if a in self.rewards:
                self.rewards[a] = r
                self._cumulative_rewards[a] += r

        # 8. Rotate selector; remove terminated/truncated agents from live list
        self.agent_selection = self._agent_selector.next()
        self.agents = [a for a in self.agents
                       if not (self.terminations[a] or self.truncations[a])]

    # ------------------------------------------------------------------
    # observe()
    # ------------------------------------------------------------------

    def observe(self, agent: str) -> np.ndarray:
        """
        Return the current observation vector for *agent* (ENV-03).

        Delegates to ``_get_observation(agent)``. See that method's
        docstring for the full vector layout.

        Parameters
        ----------
        agent : str
            Agent identifier.

        Returns
        -------
        np.ndarray
            Flat float32 observation of length matching the agent's
            observation space.
        """
        assert self.grid is not None, "Call reset() before observe()."
        return self._get_observation(agent)

    # ------------------------------------------------------------------
    # render() / close()
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | None:
        """
        Render the current environment state.

        Delegates to ``_save_frame()`` so Role E (VIZ-03) can call
        ``_save_frame(path)`` directly without duplicating render logic.

        Returns
        -------
        np.ndarray or None
            RGB array H×W×3 uint8 when render_mode == "rgb_array", else None.
        """
        # TODO (ENV-01): implement render once grid and positions are set.
        raise NotImplementedError(
            "ENV-01: render() not yet implemented. "
            "Implement via _save_frame() for VIZ-03 compatibility."
        )

    def _save_frame(self, path: str) -> None:
        """
        Save the current frame as an image file.

        Kept separate from ``render()`` so Role E can call it directly
        during rollout recording (VIZ-03).

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"rollouts/frame_0042.png"``).
        """
        # TODO (ENV-01): implement with matplotlib imshow or pygame surface.
        raise NotImplementedError("ENV-01: _save_frame() not yet implemented.")

    def close(self) -> None:
        """Release any renderer resources."""
        pass

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ByzantinePursuitEnv("
            f"n_seekers={self.n_seekers}, "
            f"grid_size={self.grid_size}, "
            f"obs_radius={self.obs_radius}, "
            f"n_byzantine={self.n_byzantine}, "
            f"byzantine_fraction={self.byzantine_fraction}, "
            f"fixed_maze={self.fixed_maze}, "
            f"max_steps={self.max_steps})"
        )
