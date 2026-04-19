# Experiment Runner Guide — Byzantine Pursuit MARL

> **Audience:** Whoever runs the full evaluation suite for the paper.
> **Goal:** Reproduce all results, generate paper-level CSVs and figures.
> **Platform:** Windows 11, Python 3.10+, GPU optional (CPU runs fine for this scale).

---

## 1. Environment Setup

```bash
pip install pettingzoo stable-baselines3 torch gymnasium matplotlib opencv-python
```

Verify with:

```bash
python -c "from env.pursuit_env import ByzantinePursuitEnv; print('env OK')"
python -c "from agents.ppo.ippo import train; print('iPPO OK')"
python -c "from agents.greedy.greedy_agent import GreedyAgent; print('greedy OK')"
```

All three must print `OK` before starting any training run.

---

## 2. Canonical Configuration

All paper results use these values. Do not change them.

| Parameter | Value | Source |
|-----------|-------|--------|
| `n_seekers` | **4** | spec RL-02 target |
| `grid_size` | 10 | `env/pursuit_env.py` |
| `obs_radius` | None (full obs) | baseline condition |
| `obstacle_density` | 0.15 | `env/pursuit_env.py` |
| `byzantine_fraction` | 0.0 (baseline) | varied in BYZ sweep |
| `max_steps` | 150 | `env/pursuit_env.py` |
| `DISTANCE_SHAPING` | 0.3 | `agents/reward.py` v2.0.0 |
| Seeds | 42, 43, 44 | frozen across all experiments |
| Training episodes | 1000 | determined by convergence analysis |

---

## 3. What to Run and In What Order

### Step 1 — iPPO Baseline (already done, results in `runs/`)

**Status:** Complete. CSVs: `runs/ippo_v4_seed{42,43,44}.csv`. Checkpoints: `checkpoints/ippo_seed{42,43,44}/ep1000/`.

If you need to reproduce from scratch:

```bash
python scripts/retrain_ippo.py --seeds 42 43 44 --n_episodes 1000 --run_tag v4
```

Expected time: ~15–25 min per seed on CPU (45–75 min total). GPU gives no speedup at this scale (env is the bottleneck, not the network).

---

### Step 2 — MAPPO Baseline (RL-03, pending)

Once `agents/mappo/mappo.py` is implemented and merged:

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
    train(env, n_episodes=1000, seed=seed, logger=logger)
    logger.close()
```

Expected time: ~20–30 min per seed (shared actor is smaller than 4×iPPO policies, but centralised critic adds overhead).

Target: >70% overall capture rate, mean capture time ≤10 steps.

---

### Step 3 — Byzantine Sweep (BYZ-01, pending Role C)

Once byzantine agents are implemented, run over `byzantine_fraction` ∈ {0.0, 0.25, 0.5}:

```python
for byz_frac in [0.0, 0.25, 0.5]:
    for seed in [42, 43, 44]:
        tag = f"byz{int(byz_frac*100)}_seed{seed}"
        # run iPPO and MAPPO at each fraction
        # log to runs/ippo_{tag}.csv and runs/mappo_{tag}.csv
```

This is the core paper experiment — degradation of capture rate as Byzantine fraction increases.

---

### Step 4 — Greedy Reference (already runnable)

Greedy gives a non-learning upper bound on coordination without comms:

```bash
python scripts/visualize_greedy.py --n_seekers 4 --seed 42
```

For bulk stats, call `GreedyAgent` in a loop and record capture outcomes manually (no built-in script yet).

---

## 4. Output Files Reference

| File | Contents |
|------|----------|
| `runs/ippo_v4_seed{N}.csv` | iPPO training log, 1000 rows per seed |
| `runs/mappo_seed{N}.csv` | MAPPO training log, 1000 rows per seed |
| `runs/ippo_byz*_seed{N}.csv` | Byzantine sweep logs |
| `checkpoints/ippo_seed{N}/ep{K}/` | SB3 `.zip` files, one per seeker agent |
| `checkpoints/mappo_seed{N}/ep{K}/` | `actor.pt` + `critic.pt` |

CSV columns (frozen by `scripts/logger.py`):

```
episode, capture_time, capture_success, n_seekers, byzantine_fraction, protocol, seed, policy_entropy
```

---

## 5. Paper-Level Evaluation

Run this after all training is complete to generate the numbers cited in the paper:

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
            sum(1 for r in rows[-100:] if r["capture_success"].lower() == "true") / min(100, len(rows))
        )
        times = [int(r["capture_time"]) for r in s]
        if times: all_times.extend(times)
    print(f"{label}")
    print(f"  Overall capture rate : {np.mean(all_rates):.1%} ± {np.std(all_rates):.1%}")
    print(f"  Last-100 capture rate: {np.mean(all_last100):.1%} ± {np.std(all_last100):.1%}")
    print(f"  Mean capture time    : {np.mean(all_times):.1f} steps ± {np.std(all_times):.1f}")
    print()

summarise("ippo_v4_seed*.csv",  "iPPO  (N=4, no-comms baseline)")
summarise("mappo_seed*.csv",     "MAPPO (N=4, CTDE baseline)")
summarise("ippo_byz25_*.csv",   "iPPO  + 25% Byzantine")
summarise("mappo_byz25_*.csv",  "MAPPO + 25% Byzantine")
summarise("ippo_byz50_*.csv",   "iPPO  + 50% Byzantine")
summarise("mappo_byz50_*.csv",  "MAPPO + 50% Byzantine")
```

Key metrics for §4 (Results):
- **Capture rate** (overall and last-100 as convergence proxy)
- **Mean capture time** (active pursuit vs passive trapping diagnostic)
- **Degradation %** = (baseline − byzantine) / baseline × 100

---

## 6. Visualisation

### Live episode window

```bash
# Trained iPPO policy
python scripts/visualize_ippo.py --policy_seed 42 --checkpoint ep1000 --n_seekers 4

# Greedy agent
python scripts/visualize_greedy.py --n_seekers 4 --seed 42
```

### Save to MP4 (for paper supplementary / demo)

```bash
python scripts/visualize_ippo.py --policy_seed 42 --checkpoint ep1000 --n_seekers 4 --save_mp4 outputs/chase.mp4
python scripts/visualize_greedy.py --n_seekers 4 --seed 42 --save_mp4 outputs/greedy_chase.mp4
```

MP4s are saved to `outputs/ippo/` and `outputs/greedy/` with timestamped filenames. Requires `opencv-python`.

---

## 7. Known Results and Baselines

| Condition | Capture rate | Mean cap time | Notes |
|-----------|-------------|---------------|-------|
| iPPO N=2 v1 (passive trapping) | 76.8% | 47.0 steps | Inflated — hider walks into stationary seeker |
| iPPO N=2 v3 (active pursuit) | 51.1% | 17.4 steps | Honest but below spec |
| **iPPO N=4 v4 (paper baseline)** | **59.5%** | **10.1 steps** | Active pursuit, coordination ceiling without comms |
| MAPPO N=4 | TBD | TBD | Target: >70% capture |

The step from v1 → v4 reflects two fixes:
1. `DISTANCE_SHAPING` raised 0.1 → 0.3 (reward v2.0.0) to penalise passive trapping
2. `n_seekers` corrected 2 → 4 to match the spec target

The ~60% iPPO floor is scientifically meaningful — it sets the no-communication coordination ceiling that MAPPO's centralised critic is designed to overcome.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `NotImplementedError` on `env.reset()` | ENV-01 not merged | Pull latest `main` |
| `ImportError: pettingzoo` | Missing dep | `pip install pettingzoo` |
| Checkpoint not found in viz script | Wrong seed or episode number | Check `checkpoints/` directory |
| CUDA/CPU tensor mismatch in viz | Policy trained on GPU, loaded without device | `PPO.load(..., device="cpu")` |
| Capture rate ~50% with N=4 | Normal for iPPO — coordination limit | Expected; MAPPO should beat this |
| Seekers not moving in viz | Loaded wrong checkpoint (N=2 policy with N=4 env) | Match `--n_seekers` to training config |

---

## 9. Files Not to Touch During Experiments

| File | Reason |
|------|--------|
| `agents/reward.py` | Frozen after team sign-off; any change invalidates all comparisons |
| `env/schema.py` | Frozen; obs layout must be stable across all agents |
| `scripts/logger.py` | Frozen; CSV columns must be consistent for analysis |
| `checkpoints/ippo_seed*/ep1000/` | Final iPPO checkpoints; back up before any retraining |
