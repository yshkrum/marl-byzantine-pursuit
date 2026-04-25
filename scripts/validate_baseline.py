"""
Baseline sanity checker — greedy agents, no Byzantine corruption.
Owner: D (Experiment Runner)
Ticket: EXP-03

Runs the GreedyAgent on a clean environment (f=0.0, NoneProtocol) and asserts
that capture_rate > 0.3.  Hard-stops (exit 1) if the criterion is not met —
this blocks all downstream experiments from running on a broken environment.

Usage
-----
    python scripts/validate_baseline.py
    python scripts/validate_baseline.py --n_seekers 4 --n_episodes 50 --seed 0
    python scripts/validate_baseline.py --grid_size 10 --obs_radius 3

Exit codes
----------
    0  — capture_rate > 0.3 (baseline PASS)
    1  — capture_rate ≤ 0.3 (baseline FAIL — investigate env/reward before proceeding)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Add project root to path so 'env', 'agents', 'comms', 'scripts' are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Per-episode result
# ---------------------------------------------------------------------------

@dataclass
class _EpisodeResult:
    episode: int
    steps: int
    captured: bool


# ---------------------------------------------------------------------------
# Run one episode with greedy agents on the AEC env
# ---------------------------------------------------------------------------

def _run_episode(env, greedy_agents: dict, episode_idx: int, seed: int) -> _EpisodeResult:
    """Step through one episode using the PettingZoo AEC API.

    Parameters
    ----------
    env:
        ``ByzantinePursuitEnv`` instance — already constructed, will be reset
        inside this function.
    greedy_agents:
        ``{agent_id: GreedyAgent}`` for every seeker.
    episode_idx:
        Episode number (used for per-episode seed offset).
    seed:
        Master seed; each episode gets ``seed + episode_idx`` to stay
        reproducible across different ``--n_episodes`` values.
    """
    env.reset(seed=seed + episode_idx)

    steps = 0
    rng = np.random.default_rng(seed + episode_idx + 10_000)  # hider random walk

    # Use the AEC agent_selection loop (matches validate_byzantine.py pattern)
    while env.agents:
        agent = env.agent_selection
        obs   = env.observe(agent)

        if agent.startswith("seeker_"):
            action = greedy_agents[agent].act(obs, env.grid)
        else:
            # Hider: uniform random directional action (never NOOP=0)
            action = int(rng.integers(1, 5))

        env.step(action)
        steps += 1

    # Capture detection: any seeker terminated (not just truncated)
    captured = any(
        env.terminations.get(a, False)
        for a in env.possible_agents
        if a.startswith("seeker_")
    )

    return _EpisodeResult(episode=episode_idx, steps=steps, captured=captured)


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def validate(
    n_seekers: int,
    grid_size: int,
    obs_radius: int | None,
    obstacle_density: float,
    max_steps: int,
    n_episodes: int,
    seed: int,
) -> bool:
    """Run the greedy baseline and return True if capture_rate > 0.3."""

    from env.pursuit_env import ByzantinePursuitEnv
    from comms.interface import NoneProtocol
    from agents.greedy.greedy_agent import GreedyAgent

    print("=" * 60)
    print("Baseline Validation — GreedyAgent, f=0.0, NoneProtocol")
    print("=" * 60)
    print(f"  n_seekers       : {n_seekers}")
    print(f"  grid_size       : {grid_size}")
    print(f"  obs_radius      : {obs_radius}")
    print(f"  obstacle_density: {obstacle_density}")
    print(f"  max_steps       : {max_steps}")
    print(f"  n_episodes      : {n_episodes}")
    print(f"  seed            : {seed}")
    print()

    env = ByzantinePursuitEnv(
        n_seekers         = n_seekers,
        grid_size         = grid_size,
        obs_radius        = obs_radius,
        obstacle_density  = obstacle_density,
        byzantine_fraction = 0.0,
        max_steps         = max_steps,
        seed              = seed,
        fixed_maze        = False,   # vary maze per episode for a general sanity check
        protocol          = NoneProtocol(),
        byzantine_agents  = {},
    )

    seeker_ids = [a for a in env.possible_agents if a.startswith("seeker_")]
    greedy_agents = {
        sid: GreedyAgent(agent_id=sid, grid_size=grid_size, seed=seed + i)
        for i, sid in enumerate(seeker_ids)
    }

    results: List[_EpisodeResult] = []

    print(f"{'Ep':>4}  {'Steps':>6}  {'Captured':>8}")
    print("-" * 25)

    for ep in range(n_episodes):
        result = _run_episode(env, greedy_agents, episode_idx=ep, seed=seed)
        results.append(result)
        captured_str = "YES" if result.captured else "no"
        print(f"{ep + 1:>4}  {result.steps:>6}  {captured_str:>8}")

    # --- Summary statistics ------------------------------------------------
    capture_rate   = sum(r.captured for r in results) / n_episodes
    mean_steps     = np.mean([r.steps for r in results])
    captured_steps = [r.steps for r in results if r.captured]
    mean_cap_steps = np.mean(captured_steps) if captured_steps else float("nan")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Episodes run        : {n_episodes}")
    print(f"  Captures            : {sum(r.captured for r in results)}")
    print(f"  Capture rate        : {capture_rate:.3f}  (threshold > 0.30)")
    print(f"  Mean steps/episode  : {mean_steps:.1f}")
    print(f"  Mean steps (success): {mean_cap_steps:.1f}")
    print()

    # --- Pass / Fail -------------------------------------------------------
    THRESHOLD = 0.30
    if capture_rate > THRESHOLD:
        print(f"  PASS  capture_rate={capture_rate:.3f} > {THRESHOLD}")
        print("=" * 60)
        return True
    else:
        print(f"  FAIL  capture_rate={capture_rate:.3f} <= {THRESHOLD}")
        print()
        print("  Action required:")
        print("    1. Check env/pursuit_env.py reset() and step() for correctness.")
        print("    2. Check agents/reward.py — DISTANCE_SHAPING may be too low.")
        print("    3. Reduce obstacle_density or grid_size for a simpler smoke test.")
        print("    4. Do NOT run Exp 1–4 until this check passes.")
        print("=" * 60)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy baseline sanity check (EXP-03 / Role D)"
    )
    parser.add_argument("--n_seekers",        type=int,   default=4,    help="Number of seeker agents (default 4)")
    parser.add_argument("--grid_size",        type=int,   default=10,   help="Grid side length (default 10)")
    parser.add_argument("--obs_radius",       type=int,   default=None, help="Observation radius; None = full obs (default None)")
    parser.add_argument("--obstacle_density", type=float, default=0.15, help="Obstacle density (default 0.15)")
    parser.add_argument("--max_steps",        type=int,   default=200,  help="Max steps per episode (default 200)")
    parser.add_argument("--n_episodes",       type=int,   default=30,   help="Episodes to evaluate (default 30)")
    parser.add_argument("--seed",             type=int,   default=42,   help="Master RNG seed (default 42)")
    args = parser.parse_args()

    passed = validate(
        n_seekers        = args.n_seekers,
        grid_size        = args.grid_size,
        obs_radius       = args.obs_radius,
        obstacle_density = args.obstacle_density,
        max_steps        = args.max_steps,
        n_episodes       = args.n_episodes,
        seed             = args.seed,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
