"""
ENV-01: ByzantinePursuitEnv scaffold.

Implements the PettingZoo AEC API for a 2-D gridworld pursuit-evasion game
with Byzantine-corrupted inter-agent communication.

Owner: Role A – Environment Engineer
Paper section: §3.1 Environment Design

Observation vector layout (to be finalised in ENV-04 / schema.py):
    [0]       agent_x            – row of the observing agent (normalised 0-1)
    [1]       agent_y            – col of the observing agent (normalised 0-1)
    [2]       hider_x            – row of hider; -1 if not in FoV   (ENV-03)
    [3]       hider_y            – col of hider; -1 if not in FoV   (ENV-03)
    [4:4+R²]  local_obstacle_map – flattened (2*obs_radius+1)² binary patch  (ENV-03)
    [4+R²:]   received_messages  – 2*(n_seekers-1) floats: per-peer
                                   (believed_hider_x, believed_hider_y);
                                   sentinel -1 for unknown / not received

    Sentinel value: -1.0 for any unknown or occluded field.

Message struct schema (to be finalised with Role C / BYZ-01):
    Message(
        sender_id:   str   – agent id of the sender
        believed_hider_x: float  – sender's belief of hider row (normalised 0-1)
        believed_hider_y: float  – sender's belief of hider col (normalised 0-1)
    )
    Byzantine agents send corrupted (believed_hider_x, believed_hider_y) values.
    The message struct dataclass will live in env/schema.py once ENV-04 is locked.

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
    from pettingzoo.utils import agent_selector
    import gymnasium.spaces as spaces
except ImportError:  # pragma: no cover
    # Minimal stubs so the module is importable before deps are installed.
    class AECEnv:  # type: ignore[no-redef]
        """Minimal stub when pettingzoo is not installed."""

    class _FakeSelector:
        def __init__(self, agents: list[str]) -> None:
            self._agents = agents
            self._idx = 0

        def reset(self) -> None:
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


# ---------------------------------------------------------------------------
# Sentinel used for unknown / occluded values in the observation vector
# ---------------------------------------------------------------------------
_SENTINEL: float = -1.0


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
        Half-width of the square field-of-view window.  ``None`` means
        full observability (to be refined in ENV-03).
    obstacle_density : float
        Fraction of non-border cells that become obstacles (default 0.2).
    byzantine_fraction : float
        Fraction of seekers that are Byzantine (default 0.0).
        ``n_byzantine = floor(n_seekers * byzantine_fraction)``.
    max_steps : int
        Maximum steps per episode before timeout (default 500).
    seed : int
        Master seed for all stochastic operations (default 42).
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "name": "byzantine_pursuit_v0",
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
    ) -> None:
        super().__init__()

        # --- Core parameters -------------------------------------------
        self.n_seekers: int = n_seekers
        self.grid_size: int = grid_size
        self.obs_radius: int | None = obs_radius
        self.obstacle_density: float = obstacle_density
        self.byzantine_fraction: float = byzantine_fraction
        self.max_steps: int = max_steps

        # --- Byzantine agent count (ENV-01) ----------------------------
        self.n_byzantine: int = math.floor(n_seekers * byzantine_fraction)

        # --- Agent registry (PettingZoo convention) --------------------
        # Seekers are indexed 0 … n_seekers-1; hider is a separate agent.
        self.possible_agents: list[str] = (
            [f"seeker_{i}" for i in range(n_seekers)] + ["hider"]
        )
        # `agents` is the *live* subset; populated in reset().
        self.agents: list[str] = []

        # --- Deterministic RNG (seed everything through this) ----------
        # All randomness – maze generation, agent spawning, episode
        # dynamics – must flow through self.rng so that setting the same
        # seed reproduces the same episode sequence.
        self.rng: numpy.random.Generator = np.random.default_rng(seed)
        self._master_seed: int = seed  # kept for re-seeding on reset

        # --- Observation / action spaces (sizes computed here) ---------
        # obs_radius defaults to full grid if None (full observability)
        _r: int = obs_radius if obs_radius is not None else grid_size
        _patch_side: int = 2 * _r + 1
        _local_map_size: int = _patch_side * _patch_side

        # Flat obs dim: (agent_x, agent_y, hider_x, hider_y)
        #             + local_obstacle_map
        #             + received_messages  (2 floats × (n_seekers-1) peers)
        _msg_size: int = 2 * (n_seekers - 1)
        _obs_dim: int = 4 + _local_map_size + _msg_size

        # TODO (ENV-04): replace hand-rolled dim with schema.OBS_DIM once
        # schema.py is locked with Role C.

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
        # TODO (ENV-02): grid populated by maze generation algorithm
        self.grid: np.ndarray | None = None          # shape (grid_size, grid_size), dtype bool
        self.positions: dict[str, tuple[int, int]] = {}  # agent_id → (row, col)
        self._step_count: int = 0
        self._agent_selector: Any = None

        # PettingZoo AEC bookkeeping dicts – populated in reset()
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._cumulative_rewards: dict[str, float] = {}

    # ------------------------------------------------------------------
    # PettingZoo AEC API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> Any:
        """Return the observation space for *agent*."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Any:
        """Return the action space for *agent*."""
        return self.action_spaces[agent]

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

        Re-seeds the RNG if *seed* is provided, then:
        1. Generates a new maze (ENV-02).
        2. Spawns agents at random non-obstacle, non-overlapping cells.
        3. Constructs initial observations for all agents (ENV-03).
        4. Resets PettingZoo bookkeeping (agents, rewards, terminations …).

        Parameters
        ----------
        seed : int or None
            If given, re-seeds the internal RNG for this episode.
        options : dict or None
            Reserved for future use (e.g. forced spawn positions).

        Returns
        -------
        observations : dict[str, np.ndarray]
            ``{agent_id: obs_vector}`` for every agent in ``self.agents``.
        infos : dict[str, dict]
            Auxiliary info dict (empty at reset).
        """
        # Re-seed if caller requests it
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Restore live agent list
        self.agents = list(self.possible_agents)

        # Reset episode counters
        self._step_count = 0

        # Initialise PettingZoo bookkeeping
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Agent selector drives the AEC turn order
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # TODO (ENV-02): replace stub grid with procedural maze generation.
        # Requirements:
        #   - 2-D numpy bool array, shape (grid_size, grid_size)
        #   - True = obstacle, False = passable
        #   - Border cells must all be obstacles
        #   - Flood-fill guarantee: every open cell reachable from every other
        #   - Density of interior obstacles ≈ self.obstacle_density
        #   - All randomness via self.rng
        raise NotImplementedError(
            "ENV-02: Maze generation not yet implemented. "
            "Replace this stub in env/pursuit_env.py with a procedural "
            "maze generator that satisfies the flood-fill connectivity guarantee."
        )

        # TODO (ENV-01): spawn agents on random open, non-overlapping cells.
        # Use self.rng to sample positions; store in self.positions.

        # TODO (ENV-03): build initial observations for each agent using
        # field-of-view masking (obs_radius) and fill message slots with
        # sentinel -1.0 (no messages received yet at episode start).

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int) -> None:
        """
        Advance the environment by one *agent* step (AEC loop).

        In the AEC API ``step()`` is called with the action for
        ``self.agent_selection``; it advances *that* agent and then
        rotates ``self.agent_selection`` to the next live agent.

        Processing order per call:
        1. Validate action; apply movement to ``self.agent_selection``.
        2. Check termination (capture) and truncation (timeout).
        3. Compute rewards via ``agents.reward.compute_rewards()``
           (ENV-04 dependency: reward.py must be locked first).
        4. Build updated observations for the agent that just moved
           plus any peer whose message slots changed.
        5. Rotate ``self.agent_selection``.

        Parameters
        ----------
        action : int
            Integer action in ``{0, 1, 2, 3, 4}`` for the current agent.
            0=NOOP, 1=UP (row-1), 2=DOWN (row+1), 3=LEFT (col-1),
            4=RIGHT (col+1).

        Raises
        ------
        NotImplementedError
            Until ENV-01 / ENV-02 / ENV-03 are implemented.
        AssertionError
            If the environment has not been reset before calling step().
        """
        assert self.agents, (
            "step() called on an environment with no live agents. "
            "Call env.reset() first."
        )

        # TODO (ENV-01): implement full step logic.
        # Sub-tasks:
        #   - Resolve movement: clip to grid bounds, ignore if obstacle
        #   - Update self.positions[self.agent_selection]
        #   - Detect capture: any seeker pos == hider pos → termination
        #   - Detect timeout: self._step_count >= self.max_steps → truncation
        #   - Call compute_rewards() from agents/reward.py (dependency: RL-04)
        #   - Update self.rewards, self.terminations, self.truncations
        #   - Build new obs for current agent (ENV-03)
        #   - Increment self._step_count
        #   - Rotate self.agent_selection via self._agent_selector.next()
        raise NotImplementedError(
            "ENV-01: step() not yet implemented. "
            "Implement movement, capture detection, reward computation, "
            "and observation construction."
        )

    # ------------------------------------------------------------------
    # observe()
    # ------------------------------------------------------------------

    def observe(self, agent: str) -> np.ndarray:
        """
        Return the current observation vector for *agent*.

        Observation layout:
            [agent_x, agent_y, hider_x, hider_y,
             local_obstacle_map (flattened),
             received_messages (flattened)]

        Sentinel -1.0 is used for any unknown or occluded field.

        Parameters
        ----------
        agent : str
            Agent identifier (e.g. ``"seeker_0"`` or ``"hider"``).

        Returns
        -------
        np.ndarray
            Flat float32 observation vector of length ``OBS_DIM``.
        """
        # TODO (ENV-03): implement FoV masking and message aggregation.
        raise NotImplementedError(
            "ENV-03: observe() not yet implemented. "
            "Implement field-of-view masking (obs_radius) and "
            "message-slot population."
        )

    # ------------------------------------------------------------------
    # render() / close()
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | None:
        """
        Render the current environment state.

        Delegates to ``_save_frame()`` so that VIZ-03 (Role E) can call
        ``save_frame(path)`` directly without duplicating render logic.

        Returns
        -------
        np.ndarray or None
            RGB array (H×W×3 uint8) when render_mode == "rgb_array",
            else None.
        """
        # TODO (ENV-01): implement render once grid and positions are set.
        raise NotImplementedError(
            "ENV-01: render() not yet implemented. "
            "Implement via _save_frame() helper for VIZ-03 compatibility."
        )

    def _save_frame(self, path: str) -> None:
        """
        Save the current frame as an image to *path*.

        Thin wrapper kept separate from render() so Role E can call it
        directly in rollout recording (VIZ-03).

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"rollouts/frame_0042.png"``).
        """
        # TODO (ENV-01): implement using matplotlib imshow or pygame surface.
        raise NotImplementedError(
            "ENV-01: _save_frame() not yet implemented."
        )

    def close(self) -> None:
        """Release any resources held by the renderer."""
        # TODO (ENV-01): close pygame window / matplotlib figure if open.
        pass  # No resources to release in the scaffold

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ByzantinePursuitEnv("
            f"n_seekers={self.n_seekers}, "
            f"grid_size={self.grid_size}, "
            f"n_byzantine={self.n_byzantine}, "
            f"byzantine_fraction={self.byzantine_fraction}, "
            f"max_steps={self.max_steps})"
        )
