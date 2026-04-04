"""
BYZ-04: Byzantine degradation sanity check — go/no-go gate for Experiment 1.

Usage
-----
    python scripts/validate_byzantine.py
    python scripts/validate_byzantine.py --n_episodes 50 --seed 0

What this script does
---------------------
Runs N_EPISODES episodes under two conditions and compares capture performance:

  Condition 1 - f=0.0 : N=6 honest seekers, broadcast protocol
  Condition 2 - f=0.5 : 3 RandomNoiseByzantine + 3 honest, broadcast protocol

Both conditions use:
  - Open arena  (obstacle_density=0.0)
  - Partial observability  (obs_radius=2, 5x5 FoV per seeker)
  - Message-aware greedy agent (BFS + message fallback)
  - grid_size=10, max_steps=500, 30 episodes

Why partial observability?
--------------------------
With full observability every seeker always sees the hider directly — messages
are never consulted and Byzantine corruption has zero effect.  With obs_radius=2,
seekers only see within a 5x5 patch.  When the hider is outside a seeker's FoV
the agent RELIES on teammate messages to navigate, making message quality the
bottleneck that Byzantine agents exploit.

Why a message-aware greedy agent?
----------------------------------
No communication-aware RL policy exists yet (MAPPO is a later ticket).  The
standard GreedyAgent ignores message slots entirely.  The message-aware wrapper
(_CommAwareGreedy) reads message slots when obs[2:4] == SENTINEL, feeds the
message consensus back to the BFS as an estimated hider target, and tracks
exactly when and how much the fallback was triggered per episode.

Degradation metrics collected per episode
-----------------------------------------
  fallback_pct  : % of seeker steps where direct sight was lost and messages
                  were consulted as a fallback.
  blind_pct     : % of seeker steps with no direct sight AND no messages.
  mean_msg_err  : mean L2 error (normalised) between each message in the buffer
                  and the true hider position.  Zero for honest agents; random
                  for Byzantine agents.

Pass/fail criterion
-------------------
PASS if EITHER:
  mean_capture_steps(f=0.5) >= mean_capture_steps(f=0.0) x 1.15
  OR
  capture_rate(f=0.5) <= capture_rate(f=0.0) - 0.10

Output saved to:
  experiments/results/validation/byzantine_sanity_check.txt

Owner : Role C (Byzantine & Comms)
Ticket: BYZ-04
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.pursuit_env import ByzantinePursuitEnv
from env.schema import SENTINEL
from agents.greedy.greedy_agent import GreedyAgent
from agents.byzantine import RandomNoiseByzantine
from comms.broadcast import BroadcastProtocol

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

N_SEEKERS: int = 6
GRID_SIZE: int = 10
OBS_RADIUS: int = 2           # 5x5 FoV — partial observability
OBSTACLE_DENSITY: float = 0.0 # open arena
MAX_STEPS: int = 500
N_EPISODES: int = 30
F_BYZANTINE: float = 0.5
SEED_BASE: int = 42

CAPTURE_TIME_THRESHOLD: float = 1.15  # f=0.5 must be >=15% slower
CAPTURE_RATE_THRESHOLD: float = 0.10  # f=0.5 capture rate must drop >=10 pp

OUTPUT_PATH = Path("experiments/results/validation/byzantine_sanity_check.txt")


# ---------------------------------------------------------------------------
# Per-episode diagnostics container
# ---------------------------------------------------------------------------

@dataclass
class _EpDiag:
    """Diagnostic data collected for one episode."""
    steps: int = 0
    captured: bool = False
    seeker_steps: int = 0       # total seeker-agent steps
    fallback_steps: int = 0     # steps where message fallback was triggered
    blind_steps: int = 0        # steps: no direct sight AND no messages
    msg_errors: list[float] = field(default_factory=list)  # per-message L2 error

    @property
    def fallback_pct(self) -> float:
        return 100.0 * self.fallback_steps / self.seeker_steps if self.seeker_steps else 0.0

    @property
    def blind_pct(self) -> float:
        return 100.0 * self.blind_steps / self.seeker_steps if self.seeker_steps else 0.0

    @property
    def mean_msg_err(self) -> float:
        return float(np.mean(self.msg_errors)) if self.msg_errors else 0.0


# ---------------------------------------------------------------------------
# Message-aware greedy wrapper
# ---------------------------------------------------------------------------

class _CommAwareGreedy:
    """
    Greedy BFS seeker augmented with message-slot fallback.

    Decision logic each step:
      1. obs[2:4] != SENTINEL  -> direct sight; BFS to true position.
      2. obs[2:4] == SENTINEL, messages available
                               -> replace obs[2:4] with message mean; BFS.
                                  Increments fallback_steps on the _EpDiag.
      3. No direct sight, no messages
                               -> GreedyAgent random exploration.
                                  Increments blind_steps on the _EpDiag.
    """

    def __init__(self, agent_id: str, grid_size: int, seed: Optional[int] = None) -> None:
        self._greedy = GreedyAgent(agent_id=agent_id, grid_size=grid_size, seed=seed)

    def reset(self) -> None:
        self._greedy._explore_action = None
        self._greedy._explore_steps = 0

    def act(
        self,
        obs: np.ndarray,
        obstacle_map: np.ndarray,
        n_seekers: int,
        diag: _EpDiag,
    ) -> int:
        hider_x = float(obs[2])
        hider_y = float(obs[3])
        diag.seeker_steps += 1

        if hider_x != SENTINEL and hider_y != SENTINEL:
            # Direct observation — use as-is
            return self._greedy.act(obs, obstacle_map)

        # Hider outside FoV — scan message slots
        msg_start = len(obs) - 2 * (n_seekers - 1)
        slots = obs[msg_start:]
        xs: list[float] = []
        ys: list[float] = []
        for i in range(0, len(slots) - 1, 2):
            mx, my = float(slots[i]), float(slots[i + 1])
            if mx != SENTINEL and my != SENTINEL:
                xs.append(mx)
                ys.append(my)

        if xs:
            diag.fallback_steps += 1
            aug = obs.copy()
            aug[2] = float(np.mean(xs))
            aug[3] = float(np.mean(ys))
            return self._greedy.act(aug, obstacle_map)
        else:
            diag.blind_steps += 1
            return self._greedy.act(obs, obstacle_map)  # random exploration


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_condition(
    byzantine_fraction: float,
    n_episodes: int,
    seed_base: int,
) -> list[_EpDiag]:
    """
    Run *n_episodes* and return per-episode diagnostics.
    """
    n_byzantine = math.floor(N_SEEKERS * byzantine_fraction)
    seeker_ids = [f"seeker_{i}" for i in range(N_SEEKERS)]
    norm = float(GRID_SIZE - 1)

    byzantine_agents: dict = {}
    for i in range(n_byzantine):
        sid = seeker_ids[i]
        byzantine_agents[sid] = RandomNoiseByzantine(
            agent_id=sid, grid_size=GRID_SIZE, seed=seed_base + i,
        )

    env = ByzantinePursuitEnv(
        n_seekers=N_SEEKERS,
        grid_size=GRID_SIZE,
        obs_radius=OBS_RADIUS,
        obstacle_density=OBSTACLE_DENSITY,
        byzantine_fraction=byzantine_fraction,
        max_steps=MAX_STEPS,
        seed=seed_base,
        fixed_maze=False,
        protocol=BroadcastProtocol(),
        byzantine_agents=byzantine_agents,
    )

    greedy_agents = {
        sid: _CommAwareGreedy(agent_id=sid, grid_size=GRID_SIZE, seed=seed_base + idx)
        for idx, sid in enumerate(seeker_ids)
    }

    rng = np.random.default_rng(seed_base)
    results: list[_EpDiag] = []

    for ep in range(n_episodes):
        for g in greedy_agents.values():
            g.reset()

        env.reset(seed=seed_base + ep)
        diag = _EpDiag()

        while env.agents:
            agent = env.agent_selection
            obs = env.observe(agent)

            if agent.startswith("seeker_"):
                action = greedy_agents[agent].act(obs, env.grid, N_SEEKERS, diag)
            else:
                action = int(rng.integers(0, 5))

            env.step(action)
            diag.steps += 1

            # After step: measure error of every message in buffer vs true hider
            if agent.startswith("seeker_") and env.positions:
                true_hr = env.positions["hider"][0] / norm
                true_hc = env.positions["hider"][1] / norm
                for bx, by in env._message_buffer.values():
                    if bx != SENTINEL and by != SENTINEL:
                        err = math.sqrt((bx - true_hr) ** 2 + (by - true_hc) ** 2)
                        diag.msg_errors.append(err)

        diag.captured = any(
            env.terminations.get(a, False)
            for a in env.possible_agents
            if a.startswith("seeker_")
        )
        results.append(diag)

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(value: float, max_value: float, width: int = 20) -> str:
    """ASCII bar proportional to value / max_value."""
    if max_value == 0:
        return " " * width
    filled = int(round(value / max_value * width))
    filled = max(0, min(filled, width))
    return "#" * filled + "-" * (width - filled)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_episodes: int, seed: int) -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 72)
    out("BYZ-04: BYZANTINE DEGRADATION SANITY CHECK")
    out(f"  N_seekers={N_SEEKERS}  grid={GRID_SIZE}x{GRID_SIZE}  "
        f"obs_radius={OBS_RADIUS}  episodes={n_episodes}")
    out(f"  obstacle_density={OBSTACLE_DENSITY}  max_steps={MAX_STEPS}  seed={seed}")
    out(f"  Byzantine subtype : RandomNoiseByzantine  f={F_BYZANTINE}")
    out(f"  Policy            : message-aware greedy (BFS + message fallback)")
    out("=" * 72)

    out("\nRunning condition 1 - f=0.0 (all honest) ...")
    diags_f0 = run_condition(0.0, n_episodes, seed)

    out(f"Running condition 2 - f={F_BYZANTINE} (RandomNoiseByzantine) ...")
    diags_f5 = run_condition(F_BYZANTINE, n_episodes, seed)

    # -----------------------------------------------------------------------
    # Per-episode comparison table
    # -----------------------------------------------------------------------
    out("")
    out("-" * 72)
    out("PER-EPISODE RESULTS")
    out("-" * 72)
    out(
        f"{'Ep':>3}  "
        f"{'f=0.0':>7}  "
        f"{'f=0.5':>7}  "
        f"{'delta':>7}  "
        f"{'delta%':>7}  "
        f"{'fallback%':>10}  "
        f"{'msg_err':>8}"
    )
    out(
        f"{'':>3}  "
        f"{'steps':>7}  "
        f"{'steps':>7}  "
        f"{'steps':>7}  "
        f"{'':>7}  "
        f"{'(f=0.5)':>10}  "
        f"{'(f=0.5)':>8}"
    )
    out("-" * 72)

    for i, (d0, d5) in enumerate(zip(diags_f0, diags_f5), start=1):
        delta = d5.steps - d0.steps
        pct   = delta / d0.steps * 100.0 if d0.steps else 0.0
        cap0  = "C" if d0.captured else "T"   # C=captured T=timeout
        cap5  = "C" if d5.captured else "T"
        out(
            f"{i:>3}  "
            f"{d0.steps:>5}{cap0}  "
            f"{d5.steps:>5}{cap5}  "
            f"{delta:>+7}  "
            f"{pct:>+6.1f}%  "
            f"{d5.fallback_pct:>9.1f}%  "
            f"{d5.mean_msg_err:>8.4f}"
        )

    out("-" * 72)
    out("  C=captured within max_steps  T=timeout  "
        "fallback%=steps using message fallback  "
        "msg_err=mean L2 error of messages vs true hider pos")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    steps_f0 = [d.steps for d in diags_f0]
    steps_f5 = [d.steps for d in diags_f5]

    mean_f0, std_f0, rate_f0 = np.mean(steps_f0), np.std(steps_f0), np.mean([d.captured for d in diags_f0])
    mean_f5, std_f5, rate_f5 = np.mean(steps_f5), np.std(steps_f5), np.mean([d.captured for d in diags_f5])

    time_degradation_pct = (mean_f5 - mean_f0) / mean_f0 * 100.0
    rate_drop_pp         = (float(rate_f0) - float(rate_f5)) * 100.0

    out("")
    out("-" * 72)
    out("SUMMARY")
    out("-" * 72)
    out(f"{'Condition':<24} {'capture_rate':>12} {'mean_steps':>11} {'std_steps':>10} {'median':>8}")
    out("-" * 72)
    out(
        f"{'f=0.0 (honest)':<24} "
        f"{float(rate_f0)*100:>11.1f}% "
        f"{float(mean_f0):>11.1f} "
        f"{float(std_f0):>10.1f} "
        f"{float(np.median(steps_f0)):>8.1f}"
    )
    out(
        f"{'f=0.5 (RandomNoise)':<24} "
        f"{float(rate_f5)*100:>11.1f}% "
        f"{float(mean_f5):>11.1f} "
        f"{float(std_f5):>10.1f} "
        f"{float(np.median(steps_f5)):>8.1f}"
    )
    out("-" * 72)
    out(f"  Capture-time degradation : {time_degradation_pct:+.1f}%   "
        f"(threshold >= +{(CAPTURE_TIME_THRESHOLD - 1) * 100:.0f}%)")
    out(f"  Capture-rate drop        : {rate_drop_pp:+.1f} pp  "
        f"(threshold >= +{CAPTURE_RATE_THRESHOLD * 100:.0f} pp)")

    # -----------------------------------------------------------------------
    # Degradation analysis
    # -----------------------------------------------------------------------
    avg_fallback_f0 = float(np.mean([d.fallback_pct for d in diags_f0]))
    avg_fallback_f5 = float(np.mean([d.fallback_pct for d in diags_f5]))
    avg_blind_f0    = float(np.mean([d.blind_pct    for d in diags_f0]))
    avg_blind_f5    = float(np.mean([d.blind_pct    for d in diags_f5]))
    avg_err_f0      = float(np.mean([d.mean_msg_err for d in diags_f0]))
    avg_err_f5      = float(np.mean([d.mean_msg_err for d in diags_f5]))

    out("")
    out("-" * 72)
    out("DEGRADATION ANALYSIS")
    out("-" * 72)
    out("  Why does f=0.5 take longer? Three compounding factors:")
    out("")
    out(f"  [1] Message fallback triggered")
    out(f"      f=0.0  {avg_fallback_f0:5.1f}% of seeker steps consulted messages")
    out(f"      f=0.5  {avg_fallback_f5:5.1f}% of seeker steps consulted messages")
    out(f"      -> When fallback triggers, f=0.5 seekers receive corrupted")
    out(f"         guidance from {math.floor(N_SEEKERS * F_BYZANTINE)} Byzantine senders.")
    out("")
    out(f"  [2] Message quality (mean L2 error vs true hider position)")
    out(f"      f=0.0  avg error = {avg_err_f0:.4f}  (honest messages cluster near hider)")
    out(f"      f=0.5  avg error = {avg_err_f5:.4f}  (Byzantine noise pulls estimate away)")
    if avg_err_f0 > 0:
        out(f"      -> Message error is {avg_err_f5 / avg_err_f0:.1f}x higher under f=0.5")
    else:
        out(f"      -> Honest error ~0 (hider always visible to senders)")
    out("")
    out(f"  [3] Blind steps (no direct sight, no messages)")
    out(f"      f=0.0  {avg_blind_f0:5.1f}% of seeker steps were fully blind")
    out(f"      f=0.5  {avg_blind_f5:5.1f}% of seeker steps were fully blind")
    out(f"      -> Blind steps trigger random exploration, wasting time.")

    # ASCII step-count distribution
    out("")
    out(f"  Step-count distribution (each # = 1 episode):")
    max_steps_seen = max(max(steps_f0), max(steps_f5))
    bucket_size = max(1, max_steps_seen // 10)
    buckets = range(0, max_steps_seen + bucket_size, bucket_size)
    out(f"  {'steps':>8}  {'f=0.0':^12}  {'f=0.5':^12}")
    for lo in buckets:
        hi = lo + bucket_size
        c0 = sum(1 for s in steps_f0 if lo <= s < hi)
        c5 = sum(1 for s in steps_f5 if lo <= s < hi)
        if c0 == 0 and c5 == 0:
            continue
        bar0 = ("#" * c0).ljust(12)
        bar5 = ("#" * c5).ljust(12)
        out(f"  {lo:>4}-{hi:<4}  {bar0}  {bar5}")

    # -----------------------------------------------------------------------
    # Pass / fail
    # -----------------------------------------------------------------------
    time_criterion = mean_f5 >= mean_f0 * CAPTURE_TIME_THRESHOLD
    rate_criterion = float(rate_f5) <= float(rate_f0) - CAPTURE_RATE_THRESHOLD
    passed = time_criterion or rate_criterion

    out("")
    out("-" * 72)
    if passed:
        out("RESULT: PASS")
        out(f"  {'[OK]' if time_criterion else '[ ]'} capture-time +{time_degradation_pct:.1f}% >= threshold +{(CAPTURE_TIME_THRESHOLD-1)*100:.0f}%")
        out(f"  {'[OK]' if rate_criterion  else '[ ]'} capture-rate drop {rate_drop_pp:.1f} pp >= threshold {CAPTURE_RATE_THRESHOLD*100:.0f} pp")
        out("")
        out("  Byzantine implementation is correctly degrading coordination.")
        out("  Experiment 1 sweep is cleared to run.")
    else:
        out("RESULT: FAIL")
        out("")
        out("  Byzantine agents are NOT sufficiently degrading capture performance.")
        out("  Investigate before running Experiment 1:")
        out("")
        out("  [1] Byzantine agents not injected")
        out(f"      Expected {math.floor(N_SEEKERS * F_BYZANTINE)} Byzantine agents in env._byzantine_agents.")
        out("      Verify: print(env._byzantine_agents)")
        out("")
        out("  [2] Messages not reaching policy input")
        out("      obs_radius=None means full observability — messages never consulted.")
        out("      Ensure OBS_RADIUS > 0 in this script.")
        out("      Verify env._message_buffer is non-sentinel after first seeker step.")
        out("")
        out("  [3] Message error too low (corruption invisible)")
        out(f"      Current avg msg_err f=0.5: {avg_err_f5:.4f}")
        out("      RandomNoiseByzantine should produce errors ~0.3-0.5 on a 10x10 grid.")
        out("      Check RandomNoiseByzantine.corrupt_message() is being called.")
        out("")
        out("  [4] Reward not dependent on hider position (iPPO only)")
        out("      Not applicable here (greedy used). If switching to iPPO, verify")
        out("      DISTANCE_SHAPING > 0 in reward.py and message slots are NOT zeroed.")

    out("")
    out("=" * 72)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOutput saved -> {OUTPUT_PATH}")

    sys.exit(0 if passed else 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BYZ-04 Byzantine degradation sanity check.")
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES,
                        help=f"Episodes per condition (default {N_EPISODES})")
    parser.add_argument("--seed", type=int, default=SEED_BASE,
                        help=f"Master RNG seed (default {SEED_BASE})")
    args = parser.parse_args()
    main(n_episodes=args.n_episodes, seed=args.seed)
