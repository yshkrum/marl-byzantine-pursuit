"""
Tests for MAPPO — Multi-Agent PPO with Centralised Training / Decentralised Execution.
Owner: B (RL Training Lead)
Ticket: RL-03
Run: pytest tests/test_mappo.py -v

Three required tests (handoff §8):
  1. Import test       — all public symbols are importable
  2. Smoke test        — exact 5-episode run from handoff §5; CSV must be written
  3. Critic shape test — critic accepts correctly shaped global_obs and returns scalar
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Constants matching canonical config
# ---------------------------------------------------------------------------

_N_SEEKERS   = 4
_GRID_SIZE   = 10
_OBS_RADIUS  = None   # full observability

# OBS_DIM(4, 10, None) = 4 + 10*10 + 2*(4-1) = 4 + 100 + 6 = 110
_OBS_DIM        = 110
_GLOBAL_OBS_DIM = _N_SEEKERS * _OBS_DIM   # 440


# ---------------------------------------------------------------------------
# Test 1 — Import test
# ---------------------------------------------------------------------------

def test_mappo_imports():
    """All public symbols in agents/mappo/mappo.py are importable."""
    from agents.mappo.mappo import (   # noqa: F401
        train,
        load_mappo,
    )
    # Internal classes needed by other tests — also verify they're accessible
    from agents.mappo.mappo import _SharedActor, _CentralisedCritic  # noqa: F401
    assert train is not None
    assert load_mappo is not None
    assert _SharedActor is not None
    assert _CentralisedCritic is not None


# ---------------------------------------------------------------------------
# Test 2 — Smoke test (exact snippet from handoff §5)
# ---------------------------------------------------------------------------

def test_smoke_5_episodes(tmp_path, monkeypatch):
    """5-episode training run completes without error and writes a CSV.

    Uses the exact code from handoff §5, with the output directory
    redirected to tmp_path so no files are left in the repo tree.
    """
    # Redirect CSV output to pytest's tmp directory
    monkeypatch.chdir(tmp_path)

    from env.pursuit_env import ByzantinePursuitEnv
    from agents.mappo.mappo import train
    from scripts.logger import EpisodeLogger

    env = ByzantinePursuitEnv(
        n_seekers=4, grid_size=10, obs_radius=None,
        obstacle_density=0.15, byzantine_fraction=0.0,
        max_steps=150, seed=0,
    )
    logger = EpisodeLogger("smoke_mappo", str(tmp_path))
    actor, critic = train(env, n_episodes=5, seed=0, logger=logger)
    logger.close()

    # Verify return types
    from agents.mappo.mappo import _SharedActor, _CentralisedCritic
    assert isinstance(actor,  _SharedActor),        "train() must return a _SharedActor"
    assert isinstance(critic, _CentralisedCritic),  "train() must return a _CentralisedCritic"

    # Verify CSV was written with the expected number of rows
    csv_path = tmp_path / "smoke_mappo.csv"
    assert csv_path.exists(), f"Expected CSV at {csv_path}"
    rows = csv_path.read_text().strip().splitlines()
    # 1 header + 5 data rows
    assert len(rows) == 6, f"Expected 6 lines (header + 5 eps), got {len(rows)}"

    # Verify CSV columns match the frozen EpisodeLogger schema
    header = rows[0].split(",")
    for required_col in ("episode", "capture_time", "capture_success",
                         "n_seekers", "byzantine_fraction", "protocol",
                         "seed", "policy_entropy"):
        assert required_col in header, f"Missing column: {required_col}"

    # Verify protocol field is "mappo" throughout
    import csv, io
    reader = csv.DictReader(io.StringIO(csv_path.read_text()))
    for row in reader:
        assert row["protocol"] == "mappo", (
            f"Expected protocol='mappo', got '{row['protocol']}'"
        )


# ---------------------------------------------------------------------------
# Test 3 — Critic input shape test
# ---------------------------------------------------------------------------

def test_critic_input_shape():
    """Centralised critic accepts global_obs of shape (n_seekers * obs_dim,)
    and returns a scalar value for a single observation.

    Also verifies:
    - Batch input of shape (B, global_obs_dim) returns shape (B,)
    - Actor and critic are distinct nn.Module instances (handoff §3 assertion)
    - Actor accepts single-agent obs of shape (obs_dim,) and returns logits (5,)
    """
    from agents.mappo.mappo import _SharedActor, _CentralisedCritic

    actor  = _SharedActor(_OBS_DIM)
    critic = _CentralisedCritic(_GLOBAL_OBS_DIM)

    # Assertion from spec: separate instances
    assert actor is not critic, "actor and critic must be distinct nn.Module instances"
    assert isinstance(actor,  torch.nn.Module)
    assert isinstance(critic, torch.nn.Module)

    # --- Single observation ---
    single_global = torch.zeros(1, _GLOBAL_OBS_DIM)
    value_single  = critic(single_global)
    assert value_single.shape == (1,), (
        f"Expected critic output shape (1,), got {value_single.shape}"
    )

    # --- Batch of observations ---
    batch_size   = 8
    batch_global = torch.randn(batch_size, _GLOBAL_OBS_DIM)
    value_batch  = critic(batch_global)
    assert value_batch.shape == (batch_size,), (
        f"Expected critic output shape ({batch_size},), got {value_batch.shape}"
    )

    # --- Actor single obs ---
    single_obs = torch.zeros(1, _OBS_DIM)
    logits     = actor(single_obs)
    assert logits.shape == (1, 5), (
        f"Expected actor output shape (1, 5), got {logits.shape}"
    )

    # --- Critic input is exactly n_seekers * obs_dim wide ---
    assert _GLOBAL_OBS_DIM == _N_SEEKERS * _OBS_DIM, (
        f"global_obs_dim mismatch: {_GLOBAL_OBS_DIM} != {_N_SEEKERS} * {_OBS_DIM}"
    )

    # --- Critic rejects wrong input size ---
    with pytest.raises((RuntimeError, Exception)):
        critic(torch.zeros(1, _OBS_DIM))   # too small — single-agent obs