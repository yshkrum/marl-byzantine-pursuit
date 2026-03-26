"""
Sanity tests for the pursuit environment.
Owner: A (Environment Engineer)
Run: pytest tests/test_env.py -v
"""

import pytest


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


@pytest.mark.skip(reason="ENV-01 not yet implemented")
def test_reset_returns_observations():
    from env.pursuit_env import ByzantinePursuitEnv
    env = ByzantinePursuitEnv(n_seekers=4, seed=0)
    obs, infos = env.reset()
    assert set(obs.keys()) == set(env.agents)


@pytest.mark.skip(reason="ENV-01 not yet implemented")
def test_deterministic_seeding():
    from env.pursuit_env import ByzantinePursuitEnv
    env1 = ByzantinePursuitEnv(seed=42)
    env2 = ByzantinePursuitEnv(seed=42)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    assert obs1 == obs2
