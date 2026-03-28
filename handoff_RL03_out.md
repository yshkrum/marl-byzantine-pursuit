# RL-03 MAPPO — Outbound Handoff

> **From:** Role B (RL Training Lead) — ticket RL-03 complete
> **To:** Role C (Byzantine & Comms) → Section 1 · Role D (Experiment Runner) → Section 2
> **Branch:** `b/RL-03-MAPPO-implementation` (open PR — merge before branching off)
> **Files delivered:**
> - `agents/mappo/__init__.py`
> - `agents/mappo/mappo.py`
> - `tests/test_mappo.py` — 3/3 passing
> **Smoke test:** 5-episode run passes, CSV written, `(actor, critic)` returned correctly

---

## 1. Byzantine Node Implementation — For Role C (BYZ-01)

### 1.1 What MAPPO Delivers to You

MAPPO is implemented as **Centralised Training with Decentralised Execution (CTDE)**. This
means there are two distinct data pathways through the system, each of which you need to
understand before injecting Byzantine corruption.

| Pathway | Input | Used for | Message slots |
|---------|-------|----------|---------------|
| **Actor** | Single seeker's local obs `(obs_dim,)` — message slots **zeroed** | Selecting actions at every step (train + test) | Zeroed — your injection point |
| **Centralised Critic** | All seekers' raw obs concatenated `(n_seekers × obs_dim,)` | Value estimation during training only | Raw (currently also zeroed in practice since you haven't filled them yet — but pathway is intentionally separate) |

The critical design decision for you: **actor and critic pathways are kept explicitly separate**
so that you can populate message slots in the critic's input independently of the actor, and
vice versa, without touching shared code.

---

### 1.2 Where Message Slots Live in the Observation Vector

From `env/schema.py` (frozen):

```
obs_dim = OBS_DIM(n_seekers, grid_size, obs_radius)

obs[0]          agent_x  — agent ROW, normalised [0, 1]
obs[1]          agent_y  — agent COL, normalised [0, 1]
obs[2]          hider_x  — hider ROW, normalised; SENTINEL=-1.0 if occluded
obs[3]          hider_y  — hider COL, normalised; SENTINEL=-1.0 if occluded
obs[4 : 4+M]    local_obstacle_map  (M = grid_size² when obs_radius=None)
obs[4+M :]      received_messages   ← YOUR DOMAIN
                2*(n_seekers-1) floats: per-peer (believed_hider_x, believed_hider_y)
```

For the canonical config (N=4, grid_size=10, obs_radius=None):

```
obs_dim          = 4 + 100 + 6  = 110
message slots    = obs[104 : 110]   (6 floats, 2 per peer)
global_obs_dim   = 4 × 110       = 440
```

---

### 1.3 Where to Inject — Actor Pathway

**File:** `agents/mappo/mappo.py`
**Function:** `_zero_message_slots()` — lines 161–169

```python
def _zero_message_slots(obs: np.ndarray, n_seekers: int) -> np.ndarray:
    out = obs.copy()
    msg_start = len(obs) - 2 * (n_seekers - 1)   # = obs_dim - 2*(n_seekers-1)
    out[msg_start:] = 0.0                          # ← REPLACE THIS with your logic
    return out
```

**Called at:** `train()` line 452, once per seeker per step, before the obs is passed to the
shared actor:

```python
obs_clean = _zero_message_slots(obs_dict[agent_id], n_seekers)   # line 452
# obs_clean is what the actor sees — fill message slots here instead of zeroing
```

**Your task:** Replace the `out[msg_start:] = 0.0` zero-fill with honest or Byzantine message
content depending on the agent's role. The `env.byzantine_fraction` and the agent's assigned
role (honest vs Byzantine) are available from the environment.

The function signature and return type must not change — `obs_clean` is passed directly to
`torch.tensor()` on the next line and must remain `np.ndarray` of dtype `float32`.

---

### 1.4 Where the Critic Sees Messages — Critic Pathway

**File:** `agents/mappo/mappo.py`
**Lines:** 434–438 (inside `train()` episode loop)

```python
# Build global obs (raw, not zeroed) in fixed sorted order
global_obs = np.concatenate([
    obs_dict.get(sid, np.zeros(obs_dim, dtype=np.float32))
    for sid in sorted_seeker_ids                             # deterministic order
])
global_obs_list.append(global_obs)                           # line 438
```

`obs_dict` is populated by `parallel_env.step()` — meaning it reflects whatever the
environment has already written into the message slots of each seeker's observation vector.
**If the environment populates received messages before returning obs, the critic will see
them automatically here.** No change to `mappo.py` is needed for the critic pathway — you
only need to ensure `pursuit_env.py` writes Byzantine-corrupted values into the correct
`received_messages` slice of each obs vector before returning from `step()`.

The critic input at each step has shape `(440,)` for N=4:

```
global_obs = [ obs_seeker_0 (110,) | obs_seeker_1 (110,) | obs_seeker_2 (110,) | obs_seeker_3 (110,) ]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              Each seeker's slice contains its own message slots at positions [104:110]
              Byzantine seekers have corrupted values in their slice — critic sees all of them
```

---

### 1.5 Separation Invariant — Do Not Break

```
Actor  sees: obs_clean = _zero_message_slots(obs_dict[agent_id], n_seekers)
Critic sees: global_obs built from raw obs_dict (env-populated)
```

This separation is **intentional**. It means:
- You can test message corruption effects on the critic's value estimates without affecting
  actor behaviour, and vice versa.
- The actor path is the single injection point for honest/Byzantine message content at test
  time when the critic is not used.
- Do **not** bypass `_zero_message_slots` for the actor — replace its body instead.

---

### 1.6 Logger Field for Your Experiments

`scripts/logger.py` EpisodeLogger already accepts `byzantine_fraction` and `protocol`. For
Byzantine MAPPO sweeps, pass:

```python
logger.log(
    ...,
    byzantine_fraction=env.byzantine_fraction,  # e.g. 0.25
    protocol="mappo",                           # keep as "mappo" for all MAPPO variants
    ...
)
```

Role D will use `byzantine_fraction` to separate sweep results in the analysis CSVs.

---

### 1.7 Files You Own (Do Not Modify These Without Team Agreement)

| File | You own | Do not touch |
|------|---------|--------------|
| `agents/byzantine/` | ✓ | |
| `comms/` | ✓ | |
| `env/pursuit_env.py` — message injection in `step()` | ✓ (BYZ-01) | All other methods |
| `agents/mappo/mappo.py` — `_zero_message_slots()` body | ✓ (replace body only) | All other functions |
| `env/schema.py` | Frozen — read only | |
| `scripts/logger.py` | Frozen — read only | |
| `agents/reward.py` | Frozen — read only | |

---

## 2. Execution Guide — For Role D (EXP-03 Baselines)

### 2.1 Prerequisites

Verify the environment is ready before starting any run:

```bash
python -c "from env.pursuit_env import ByzantinePursuitEnv; print('env OK')"
python -c "from agents.mappo.mappo import train; print('MAPPO OK')"
python -c "from agents.ppo.ippo import train; print('iPPO OK')"
```

All three must print `OK`. If MAPPO fails, check that `b/RL-03-MAPPO-implementation` is
merged into `main` and you have pulled the latest.

Install dependencies if needed:

```bash
pip install pettingzoo gymnasium torch stable-baselines3 pytest
```

---

### 2.2 Canonical Configuration — Do Not Change

All MAPPO runs must use exactly these values. Any deviation invalidates comparison against
the locked iPPO baseline (`runs/ippo_v4_seed{42,43,44}.csv`).

| Parameter | Value | Source |
|-----------|-------|--------|
| `n_seekers` | **4** | canonical config |
| `grid_size` | 10 | `env/pursuit_env.py` |
| `obs_radius` | `None` (full obs) | baseline condition |
| `obstacle_density` | 0.15 | `env/pursuit_env.py` |
| `byzantine_fraction` | 0.0 (baseline) | varied in BYZ sweep |
| `max_steps` | 150 | `env/pursuit_env.py` |
| `DISTANCE_SHAPING` | 0.3 | `agents/reward.py` v2.0.0 — frozen |
| Seeds | **42, 43, 44** | frozen across all experiments |
| `n_episodes` | **1000** | determined by iPPO convergence analysis |
| LR | 3e-4 | frozen hyperparams |
| γ | 0.99 | frozen hyperparams |
| λ_GAE | 0.95 | frozen hyperparams |
| ε (clip) | 0.2 | frozen hyperparams |
| ENT_COEF | 0.01 | frozen hyperparams |
| Network | 2×64 ReLU MLP | frozen hyperparams |

---

### 2.3 Step 1 — MAPPO Baseline Runs (No Byzantine Agents)

Run this once `b/RL-03-MAPPO-implementation` is merged:

```python
from env.pursuit_env import ByzantinePursuitEnv
from agents.mappo.mappo import train
from scripts.logger import EpisodeLogger

for seed in [42, 43, 44]:
    env = ByzantinePursuitEnv(
        n_seekers=4, grid_size=10, obs_radius=None,
        obstacle_density=0.15, byzantine_fraction=0.0,
        max_steps=150, seed=seed,
    )
    logger = EpisodeLogger(f"mappo_seed{seed}", "runs/")
    actor, critic = train(env, n_episodes=1000, seed=seed, logger=logger)
    logger.close()
    print(f"Seed {seed} complete")
```

**Expected output files:** `runs/mappo_seed42.csv`, `runs/mappo_seed43.csv`,
`runs/mappo_seed44.csv` — each 1001 lines (header + 1000 episodes).

**Expected runtime:** ~20–30 min per seed on CPU (60–90 min total). GPU gives no speedup
at this scale — the env step is the bottleneck, not the network.

**Expected performance target:** >70% overall capture rate, mean capture time ≤10 steps.
If these targets are not met, do not proceed to the Byzantine sweep — notify Role B.

**Checkpoints** are saved automatically every 50 episodes to
`checkpoints/mappo_seed{seed}/ep{N}/actor.pt` and `critic.pt`.

---

### 2.4 Step 2 — Byzantine Sweep (Pending Role C / BYZ-01)

Once Role C delivers `agents/byzantine/` and the message injection is merged, run the full
Byzantine degradation sweep over both algorithms:

```python
from env.pursuit_env import ByzantinePursuitEnv
from agents.mappo.mappo import train as mappo_train
from agents.ppo.ippo import train as ippo_train
from scripts.logger import EpisodeLogger

for byz_frac in [0.0, 0.25, 0.5]:
    for seed in [42, 43, 44]:
        tag = f"byz{int(byz_frac * 100):02d}_seed{seed}"

        # MAPPO
        env = ByzantinePursuitEnv(
            n_seekers=4, grid_size=10, obs_radius=None,
            obstacle_density=0.15, byzantine_fraction=byz_frac,
            max_steps=150, seed=seed,
        )
        logger = EpisodeLogger(f"mappo_{tag}", "runs/")
        mappo_train(env, n_episodes=1000, seed=seed, logger=logger)
        logger.close()

        # iPPO (same config for direct comparison)
        env = ByzantinePursuitEnv(
            n_seekers=4, grid_size=10, obs_radius=None,
            obstacle_density=0.15, byzantine_fraction=byz_frac,
            max_steps=150, seed=seed,
        )
        logger = EpisodeLogger(f"ippo_{tag}", "runs/")
        ippo_train(env, n_episodes=1000, seed=seed, logger=logger)
        logger.close()

        print(f"byz={byz_frac} seed={seed} done")
```

**Output files:** `runs/mappo_byz{00,25,50}_seed{42,43,44}.csv` and
`runs/ippo_byz{00,25,50}_seed{42,43,44}.csv` — 18 CSVs total.

---

### 2.5 Paper-Level Summary Statistics

After all training runs are complete, use this script to print the numbers for §4 (Results):

```python
import csv
from pathlib import Path
import numpy as np

def summarise(glob_pattern, label):
    files = sorted(Path("runs").glob(glob_pattern))
    if not files:
        print(f"{label}: no files found"); return
    all_rates, all_last100, all_times = [], [], []
    for p in files:
        rows = list(csv.DictReader(open(p)))
        if not rows: continue
        s = [r for r in rows if r["capture_success"].lower() == "true"]
        all_rates.append(len(s) / len(rows))
        all_last100.append(
            sum(1 for r in rows[-100:] if r["capture_success"].lower() == "true")
            / min(100, len(rows))
        )
        times = [int(r["capture_time"]) for r in s]
        if times: all_times.extend(times)
    print(f"{label}")
    print(f"  Overall capture rate : {np.mean(all_rates):.1%} ± {np.std(all_rates):.1%}")
    print(f"  Last-100 capture rate: {np.mean(all_last100):.1%} ± {np.std(all_last100):.1%}")
    print(f"  Mean capture time    : {np.mean(all_times):.1f} steps ± {np.std(all_times):.1f}")
    print()

# Baselines
summarise("ippo_v4_seed*.csv",      "iPPO  N=4  byz=0.0  (locked baseline)")
summarise("mappo_seed*.csv",         "MAPPO N=4  byz=0.0  (target: >70%)")

# Byzantine degradation
for pct in ["25", "50"]:
    summarise(f"ippo_byz{pct}_seed*.csv",   f"iPPO  N=4  byz=0.{pct}")
    summarise(f"mappo_byz{pct}_seed*.csv",  f"MAPPO N=4  byz=0.{pct}")
```

Key metrics for the paper (§4):
- **Capture rate** — overall and last-100 as convergence proxy
- **Mean capture time** — active pursuit (≤15 steps) vs passive trapping (>30 steps)
- **Degradation %** = `(baseline − byzantine) / baseline × 100`

---

### 2.6 Output Files Reference

| File | Contents |
|------|----------|
| `runs/mappo_seed{42,43,44}.csv` | MAPPO baseline training logs |
| `runs/mappo_byz{00,25,50}_seed{N}.csv` | MAPPO Byzantine sweep logs |
| `runs/ippo_byz{00,25,50}_seed{N}.csv` | iPPO Byzantine sweep logs (for comparison) |
| `checkpoints/mappo_seed{N}/ep{K}/actor.pt` | Shared actor weights at checkpoint K |
| `checkpoints/mappo_seed{N}/ep{K}/critic.pt` | Centralised critic weights at checkpoint K |

CSV columns (frozen by `scripts/logger.py`):
```
episode, capture_time, capture_success, n_seekers, byzantine_fraction,
protocol, seed, policy_entropy
```

To load a trained MAPPO policy for evaluation or the Byzantine sweep:

```python
from agents.mappo.mappo import load_mappo
from env.schema import OBS_DIM

obs_dim = OBS_DIM(n_seekers=4, grid_size=10, obs_radius=None)  # = 110
actor, critic = load_mappo("checkpoints/mappo_seed42/ep1000", obs_dim, n_seekers=4)
# actor and critic are in eval() mode — critic not needed at test time
```

---

### 2.7 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `NotImplementedError` on `env.reset()` | Branch not merged | `git pull` and check `main` |
| `ModuleNotFoundError: pettingzoo` | Missing dependency | `pip install pettingzoo gymnasium` |
| `TypeError: 'module' object is not callable` | PettingZoo ≥1.25 compat | Ensure `env/pursuit_env.py` uses `from pettingzoo.utils.agent_selector import agent_selector` |
| Capture rate flat at ~20–30% after 1000 eps | Byzantine injection not zeroed | Confirm `byzantine_fraction=0.0` for baseline runs |
| Capture rate well below iPPO (~60%) | Hyperparameter drift | Verify all constants in `mappo.py` match the frozen values in §2.2 |
| `FileNotFoundError: actor.pt` in `load_mappo` | Checkpoint not yet written | Confirm training completed; checkpoints land every 50 eps |
| Shape mismatch in critic | N or obs_dim wrong in `load_mappo` call | Use `OBS_DIM(4, 10, None) = 110`; global = `4 × 110 = 440` |

---

### 2.8 Files Not to Touch During Runs

| File | Reason |
|------|--------|
| `agents/reward.py` | Frozen v2.0.0 — any change invalidates all comparisons |
| `env/schema.py` | Frozen — obs layout must be stable across all agents |
| `scripts/logger.py` | Frozen — CSV columns must be consistent for analysis |
| `agents/mappo/mappo.py` | Do not modify hyperparameters or `train()` signature |
| `checkpoints/mappo_seed*/ep1000/` | Final checkpoints — back up before any retraining |

---

## 3. Quick Reference

| Field | Value |
|-------|-------|
| **Delivered branch** | `b/RL-03-MAPPO-implementation` |
| **PR target** | `main` |
| **Run tests** | `pytest tests/test_mappo.py -v` (3/3 passing) |
| **Frozen files** | `env/schema.py`, `agents/reward.py`, `scripts/logger.py` |
| **Never commit** | `*.pt`, `*.pth`, `runs/`, `wandb/` |
| **Paper section** | §3.2 Learning Setup (Role B), §4 Results (Role D + E) |
| **iPPO baseline** | 59.5% capture, 10.1 steps mean — in `runs/ippo_v4_seed*.csv` |
| **MAPPO target** | >70% capture, ≤10 steps mean |