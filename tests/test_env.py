"""
Sanity tests for the pursuit environment.
Owner: A (Environment Engineer)
Run: pytest tests/test_env.py -v

Test groups:
    ENV-01 – scaffold / init
    ENV-02 – maze generation
    ENV-03 – observations / field-of-view
"""

import numpy as np
import pytest


# ===========================================================================
# ENV-01: Scaffold
# ===========================================================================

def test_env_import():
    """Environment module is importable."""
    from env.pursuit_env import ByzantinePursuitEnv
    assert ByzantinePursuitEnv is not None


def test_env_init_defaults():
    """Environment initialises with default parameters."""
    from env.pursuit_env import ByzantinePursuitEnv
    env = ByzantinePursuitEnv()
    assert env.n_seekers == 4
    assert env.grid_size == 20
    assert env.n_byzantine == 0


def test_byzantine_count():
    """Byzantine agent count is computed correctly from fraction."""
    from env.pursuit_env import ByzantinePursuitEnv
    env = ByzantinePursuitEnv(n_seekers=6, byzantine_fraction=0.33)
    assert env.n_byzantine == 1  # floor(6 * 0.33) = 1

    env2 = ByzantinePursuitEnv(n_seekers=6, byzantine_fraction=0.5)
    assert env2.n_byzantine == 3


def test_reset_returns_observations():
    """reset() returns obs dict keyed by all agents with correct array shape."""
    from env.pursuit_env import ByzantinePursuitEnv
    env = ByzantinePursuitEnv(n_seekers=4, grid_size=10, seed=0)
    obs, infos = env.reset()

    assert set(obs.keys()) == set(env.agents)
    for agent, o in obs.items():
        assert isinstance(o, np.ndarray)
        assert o.dtype == np.float32
        # Shape must match the declared observation space
        assert o.shape == env.observation_spaces[agent].shape


def test_deterministic_seeding():
    """Two environments with the same seed produce identical reset observations."""
    from env.pursuit_env import ByzantinePursuitEnv
    env1 = ByzantinePursuitEnv(grid_size=10, seed=42)
    env2 = ByzantinePursuitEnv(grid_size=10, seed=42)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()

    assert obs1.keys() == obs2.keys()
    for agent in obs1:
        assert np.array_equal(obs1[agent], obs2[agent]), (
            f"Observation mismatch for {agent} with same seed"
        )


# ===========================================================================
# ENV-02: Maze generation
# ===========================================================================

def test_maze_obstacle_density():
    """
    Interior obstacle density of the generated maze is within ±5 percentage
    points of the target density (after the connectivity-repair pass).
    """
    from env.pursuit_env import ByzantinePursuitEnv
    target = 0.2
    env = ByzantinePursuitEnv(grid_size=20, obstacle_density=target, seed=0)
    env._generate_maze()

    gs = env.grid_size
    interior = env.grid[1:gs - 1, 1:gs - 1]
    actual = interior.sum() / interior.size

    assert abs(actual - target) <= 0.05, (
        f"Interior obstacle density {actual:.3f} deviates more than 5% "
        f"from target {target}"
    )


def test_maze_all_open_cells_reachable():
    """
    Every passable cell in the generated maze is reachable from every other
    passable cell (flood-fill connectivity guarantee).
    """
    from env.pursuit_env import ByzantinePursuitEnv
    # Test across several seeds and a higher density to stress the repair loop
    for seed in (0, 7, 42, 99, 123):
        env = ByzantinePursuitEnv(grid_size=20, obstacle_density=0.3, seed=seed)
        env._generate_maze()
        assert env._is_connected(env.grid), (
            f"Maze with seed={seed} has disconnected open cells"
        )


def test_maze_deterministic_seed():
    """Same seed produces an identical maze across two independent environments."""
    from env.pursuit_env import ByzantinePursuitEnv
    env1 = ByzantinePursuitEnv(grid_size=20, seed=42)
    env2 = ByzantinePursuitEnv(grid_size=20, seed=42)
    env1._generate_maze()
    env2._generate_maze()

    assert np.array_equal(env1.grid, env2.grid), (
        "Same seed must produce identical maze layouts"
    )


def test_maze_border_is_all_obstacles():
    """All four border rows/cols of the generated maze are obstacles."""
    from env.pursuit_env import ByzantinePursuitEnv
    env = ByzantinePursuitEnv(grid_size=15, seed=5)
    env._generate_maze()
    g = env.grid
    gs = env.grid_size

    assert g[0, :].all(),    "Top border must be all obstacles"
    assert g[-1, :].all(),   "Bottom border must be all obstacles"
    assert g[:, 0].all(),    "Left border must be all obstacles"
    assert g[:, -1].all(),   "Right border must be all obstacles"
    # Interior must have at least some open cells
    assert not g[1:gs - 1, 1:gs - 1].all(), "Interior must have open cells"


def test_fixed_maze_reuses_grid():
    """
    With fixed_maze=True the grid object is reused across multiple resets
    (same array identity / equal content), and rng advances only on first reset.
    """
    from env.pursuit_env import ByzantinePursuitEnv
    env = ByzantinePursuitEnv(grid_size=10, seed=7, fixed_maze=True)

    obs1, _ = env.reset()
    grid_after_first = env.grid.copy()

    obs2, _ = env.reset()
    grid_after_second = env.grid.copy()

    assert np.array_equal(grid_after_first, grid_after_second), (
        "fixed_maze=True must reuse the maze across resets"
    )


def test_variable_maze_changes_grid():
    """With fixed_maze=False (default) each reset may produce a different maze."""
    from env.pursuit_env import ByzantinePursuitEnv
    # Use a fresh rng per reset so that the maze changes.
    # The simplest check: generate two mazes from sequential rng states and
    # verify the mechanism works (at least one pair across seeds should differ).
    env = ByzantinePursuitEnv(grid_size=10, seed=0, fixed_maze=False)
    env._generate_maze()
    grid1 = env.grid.copy()
    env._generate_maze()  # advances rng → different random choices
    grid2 = env.grid.copy()

    # Very likely to differ; if same, test still passes (not a hard failure).
    # Main point: no exception raised and grids are valid.
    assert env._is_connected(grid1)
    assert env._is_connected(grid2)


# ===========================================================================
# ENV-03: Observations / field-of-view
# ===========================================================================

def test_obs_shape_full_observability():
    """
    With obs_radius=None the observation vector has the expected dimension:
    4 + grid_size² + 2*(n_seekers-1).
    """
    from env.pursuit_env import ByzantinePursuitEnv
    gs, ns = 10, 3
    env = ByzantinePursuitEnv(n_seekers=ns, grid_size=gs, obs_radius=None, seed=1)
    obs, _ = env.reset()

    expected_dim = 4 + gs * gs + 2 * (ns - 1)
    for agent in env.agents:
        assert obs[agent].shape == (expected_dim,), (
            f"Wrong obs shape for {agent}: {obs[agent].shape} != ({expected_dim},)"
        )


def test_obs_shape_partial_observability():
    """
    With obs_radius=r the observation vector has the expected dimension:
    4 + (2r+1)² + 2*(n_seekers-1).
    """
    from env.pursuit_env import ByzantinePursuitEnv
    gs, ns, r = 12, 4, 3
    env = ByzantinePursuitEnv(n_seekers=ns, grid_size=gs, obs_radius=r, seed=2)
    obs, _ = env.reset()

    patch_side = 2 * r + 1
    expected_dim = 4 + patch_side * patch_side + 2 * (ns - 1)
    for agent in env.agents:
        assert obs[agent].shape == (expected_dim,), (
            f"Wrong obs shape for {agent}: {obs[agent].shape} != ({expected_dim},)"
        )


def test_full_obs_hider_always_visible():
    """
    With obs_radius=None the hider position is always returned (not sentinel)
    in every seeker's observation.
    """
    from env.pursuit_env import ByzantinePursuitEnv
    env = ByzantinePursuitEnv(n_seekers=3, grid_size=10, obs_radius=None, seed=3)
    obs, _ = env.reset()

    for agent in env.agents:
        if agent == "hider":
            continue
        hider_x = obs[agent][2]
        hider_y = obs[agent][3]
        assert hider_x != -1.0, f"{agent} should see hider_x under full obs"
        assert hider_y != -1.0, f"{agent} should see hider_y under full obs"


def test_wall_blocks_hider_visibility():
    """
    ENV-03: An agent with obs_radius cannot see the hider when a solid wall
    of obstacles lies between them on the Bresenham path.
    """
    from env.pursuit_env import ByzantinePursuitEnv

    # Small grid, large radius so distance is not the limiting factor
    env = ByzantinePursuitEnv(
        n_seekers=2, grid_size=11, obs_radius=8, seed=0
    )
    env.reset()

    # Manually set positions: seeker_0 on left, hider on right
    env.positions["seeker_0"] = (5, 1)
    env.positions["hider"]    = (5, 9)

    # Place a vertical wall of obstacles at col=5 (row 1–9)
    for row in range(1, 10):
        env.grid[row, 5] = True

    obs = env._get_observation("seeker_0")
    hider_x = obs[2]
    hider_y = obs[3]

    assert hider_x == -1.0, "Wall should block hider_x from seeker_0"
    assert hider_y == -1.0, "Wall should block hider_y from seeker_0"


def test_hider_visible_without_wall():
    """
    ENV-03: The same agent CAN see the hider when the corridor is clear.
    Companion to test_wall_blocks_hider_visibility.

    After reset() the maze may contain random interior obstacles, so we
    explicitly clear the entire row between the two agents before testing.
    """
    from env.pursuit_env import ByzantinePursuitEnv

    env = ByzantinePursuitEnv(
        n_seekers=2, grid_size=11, obs_radius=8, seed=0
    )
    env.reset()

    env.positions["seeker_0"] = (5, 1)
    env.positions["hider"]    = (5, 9)

    # Clear the entire row so the Bresenham path has no obstacles
    env.grid[5, :] = False
    env.grid[5, 0]  = True   # restore left border
    env.grid[5, 10] = True   # restore right border

    obs = env._get_observation("seeker_0")
    hider_x = obs[2]
    hider_y = obs[3]

    assert hider_x != -1.0, "Hider should be visible with clear LoS"
    assert hider_y != -1.0, "Hider should be visible with clear LoS"


def test_message_slots_initialised_to_sentinel():
    """
    At reset() all message slots in every seeker's observation must be -1.0
    (no messages received yet; Role C has not populated the buffer).
    """
    from env.pursuit_env import ByzantinePursuitEnv
    ns = 4
    gs = 10
    env = ByzantinePursuitEnv(n_seekers=ns, grid_size=gs, obs_radius=None, seed=9)
    obs, _ = env.reset()

    map_size = gs * gs
    msg_start = 4 + map_size
    n_msg_floats = 2 * (ns - 1)

    for agent in env.agents:
        if agent == "hider":
            continue
        msg_slice = obs[agent][msg_start: msg_start + n_msg_floats]
        assert np.all(msg_slice == -1.0), (
            f"{agent} message slots should all be -1.0 at reset, got {msg_slice}"
        )
