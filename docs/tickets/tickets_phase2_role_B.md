# Phase 2 Tickets — Role B: RL Training Lead
*Paper ownership: §3.2 Learning Setup*
*Contact: Kiryl*

---

### RL-06 · Retrain iPPO on experiment-canonical config
**Priority:** Critical · **Blocks:** EXP-01 (Experiment 1) · **Deadline:** ASAP

**Background**

All existing iPPO training (v4, seeds 42/43/44) used `n_seekers=4, grid_size=10, obs_radius=None`.
The experiment configs are now locked to the **PettingZoo pursuit benchmark defaults**:

```
n_seekers=8, grid_size=16, obs_radius=7, obstacle_density=0.15, max_steps=500
```

OBS_DIM(8, 16, obs_radius=7) = 243. OBS_DIM(4, 10, None) = 110. These are incompatible —
existing checkpoints cannot be used for any Byzantine experiment. iPPO at the canonical
config is the no-comms (NoneProtocol) baseline that MAPPO must beat.

**Why these values?** N=8 with f ∈ {0, 0.125, 0.25, 0.375, 0.5} gives exactly {0,1,2,3,4}
Byzantine agents — one per step, no duplicate conditions. N=8, 16×16, obs_radius=7 matches
the PettingZoo pursuit default, making results directly comparable to the literature.

**Acceptance criteria**

- [ ] Training config matches exactly: `n_seekers=8, grid_size=16, obs_radius=7, obstacle_density=0.15, max_steps=500`
- [ ] Seeds 42, 43, 44 all trained for ≥500 episodes
- [ ] CSVs saved to `runs/ippo_exp_seed{42,43,44}.csv` (preserves v4 results)
- [ ] Checkpoints at `checkpoints/ippo_exp_seed{N}/ep{K}/`
- [ ] Capture rate reported; expect 45–60% (partial obs makes task harder than v4 full-obs)

**How to run**

```python
from env.pursuit_env import ByzantinePursuitEnv
from agents.ppo.ippo import train
from scripts.logger import EpisodeLogger

for seed in [42, 43, 44]:
    env = ByzantinePursuitEnv(
        n_seekers=8, grid_size=16, obs_radius=7,
        obstacle_density=0.15, byzantine_fraction=0.0,
        max_steps=500, seed=seed,
    )
    logger = EpisodeLogger(f"ippo_exp_seed{seed}", "runs/")
    train(env, n_episodes=500, seed=seed, logger=logger)
    logger.close()
```

Expected runtime: ~40–60 min per seed on CPU.

---

### RL-07 · Retrain MAPPO on experiment-canonical config
**Priority:** Critical · **Blocks:** EXP-01, EXP-02 · **Deadline:** ASAP

**Background**

MAPPO was only trained on `grid_size=30` (50 episodes, smoke test only). It needs to be trained
on the canonical config to serve as the honest-comms baseline in Experiment 1. At `f=0.0` with
broadcast protocol, MAPPO should outperform iPPO by exploiting peer messages. This gap is the
central result of the paper.

**Acceptance criteria**

- [ ] Training config matches exactly: `n_seekers=8, grid_size=16, obs_radius=7, obstacle_density=0.15, max_steps=500, protocol=BroadcastProtocol`
- [ ] Seeds 42, 43, 44 trained for ≥500 episodes
- [ ] CSVs saved to `runs/mappo_exp_seed{42,43,44}.csv`
- [ ] Checkpoints at `checkpoints/mappo_exp_seed{N}/ep{K}/`
- [ ] MAPPO capture rate at f=0.0 is meaningfully higher than iPPO (target: +10pp or more)
- [ ] `protocol="mappo_broadcast"` in logger call to distinguish from no-comms runs

**How to wire BroadcastProtocol into training**

```python
from comms.broadcast import BroadcastProtocol

protocol = BroadcastProtocol()
env = ByzantinePursuitEnv(
    n_seekers=8, grid_size=16, obs_radius=7,
    obstacle_density=0.15, byzantine_fraction=0.0,
    max_steps=500, seed=seed,
    protocol=protocol,
    byzantine_agents={},   # empty for honest baseline
)
```

**Important:** `_zero_message_slots()` in `mappo.py` currently zeros ALL message slots for
the actor. For the honest-comms run the actor should receive real messages — **remove or
bypass `_zero_message_slots` when `protocol` is not None and `byzantine_fraction == 0.0`**.
Message slots should only be zeroed when running iPPO (no comms) or during Byzantine
conditions where you want the actor to ignore corrupted messages.

Expected runtime: ~50–70 min per seed on CPU (larger obs_dim = 243 vs 110).

---

### RL-08 · Update §3.2 Learning Setup paper section
**Priority:** High · **Deadline:** After RL-06/07 complete

**Background**

Paper §3.2 needs to be written (ticket RL-05, previously deferred). It must cite the
final canonical training config, not the 10×10 exploratory runs.

**Content to include (~350–400 words)**

1. **Algorithm family:** Independent PPO (iPPO) as the no-comms baseline; MAPPO with
   centralised critic (CTDE) as the comms-enabled baseline. Cite Schulman et al. 2017
   (PPO) and Yu et al. 2022 (MAPPO).
2. **Canonical config table** (matches PettingZoo pursuit benchmark defaults):
   n_seekers=8, grid_size=16, obs_radius=7, obstacle_density=0.15, max_steps=500,
   seeds {42,43,44}, n_episodes=500. Note: obs_radius=7 gives OBS_DIM=243 per agent.
3. **Byzantine fractions:** f ∈ {0.0, 0.125, 0.25, 0.375, 0.5} → {0,1,2,3,4} Byzantine
   agents. Chosen so each increment adds exactly one Byzantine agent, enabling a clean
   monotonic degradation curve.
4. **Frozen hyperparams table:** LR=3e-4, γ=0.99, λ=0.95, ε=0.2, H=0.01, 2×64 ReLU.
5. **Reward function:** CAPTURE_REWARD=10, TEAM_CAPTURE_BONUS=5, STEP_PENALTY=-0.01,
   DISTANCE_SHAPING=0.3 (v2.0.0). One sentence on why DS=0.3 was chosen (passive trapping
   diagnosis — degenerate strategy with DS=0.1, mean capture time 47 steps vs 10 with DS=0.3).
6. **Observation layout:** cite §3.1 / schema.py; note message slots zeroed for iPPO,
   populated for MAPPO. Explain that partial observability (obs_radius=7) is why comms
   matters — agents cannot see the full 16×16 grid.

**File:** `analysis/paper/sec3_2_learning_setup.md`
