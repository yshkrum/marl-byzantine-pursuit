# Byzantine Experiments Handoff — RL Training Update

> **From:** Role B (RL Training Lead)
> **To:** Role C (Byzantine & Comms)
> **Relevant tickets:** EXP-05 (Byzantine degradation), EXP-06 (Protocol comparison)
> **Date:** 2026-04-17

---

## TL;DR

Both iPPO and MAPPO have been retrained from scratch with a fixed codebase. Use the new
checkpoints below. Honest baselines are ~60% for both — MAPPO captures faster (15.7s vs 17.7s).
The critical thing for your evaluation scripts: **do not zero MAPPO's message slots at eval time**
— Byzantine message corruption is the signal you are measuring.

---

## 1. New Checkpoints (use these, ignore older ones)

```
iPPO  seed42: checkpoints/ippo_exp_seed42/ep1000/   (8 × seeker_N.zip)
iPPO  seed43: checkpoints/ippo_exp_seed43/ep1000/   (8 × seeker_N.zip)
iPPO  seed44: checkpoints/ippo_exp_seed44/ep1000/   (8 × seeker_N.zip)

MAPPO seed42: checkpoints/mappo_exp_seed42/ep1000/  (actor.pt + critic.pt)
MAPPO seed43: checkpoints/mappo_exp_seed43/ep1000/  (actor.pt + critic.pt)
MAPPO seed44: checkpoints/mappo_exp_seed44/ep1000/  (actor.pt + critic.pt)
```

**WARNING — old checkpoint directories to ignore:**
- `checkpoints/ippo_seed{42,43,44}/` — mixed directory, contains weights from two
  different configs (old N=4 run up to ep1000, new N=8 run up to ep500). Do not use.
- `checkpoints/mappo_seed{42,43,44}/` — old run before comms bugs were fixed.

---

## 2. Canonical Environment Config

All training and all experiments must use this exact config:

```python
env = ByzantinePursuitEnv(
    n_seekers        = 8,
    grid_size        = 16,
    obs_radius       = 7,
    obstacle_density = 0.15,
    max_steps        = 500,
    seed             = seed,
    # Byzantine experiments: add these two:
    byzantine_fraction = f,               # 0.0, 0.125, 0.25, 0.375, 0.5
    protocol           = BroadcastProtocol(),
    byzantine_agents   = <set of agent IDs>,
)
```

OBS_DIM at this config: `4 + (2*7+1)² + 2*(8-1) = 4 + 225 + 14 = 243`

```python
from env.schema import OBS_DIM
assert OBS_DIM(8, 16, 7) == 243
```

---

## 3. Honest Baselines (your f=0.0 anchor points)

These are the numbers to anchor your degradation curves. Report capture rate drop
relative to these honest baselines.

| Algorithm | Full-run rate | Converged rate (last 200/seed) | Mean capture time |
|-----------|--------------|-------------------------------|-------------------|
| iPPO      | 61.5%        | **59.5%**                     | 17.7s             |
| MAPPO     | 55.9%        | **60.3%**                     | 15.7s             |

> Converged rate (last 200 episodes per seed) is the number to cite in the paper.
> Full-run rate is lower because MAPPO spends its first ~500 episodes still learning.

**CSVs for reference:** `runs/ippo_exp_seed{42,43,44}.csv`, `runs/mappo_exp_seed{42,43,44}.csv`

---

## 4. How to Load Checkpoints

### iPPO

```python
from agents.ppo.ippo import load_policies

seeker_ids = ["seeker_%d" % i for i in range(8)]
policies = load_policies("checkpoints/ippo_exp_seed42/ep1000", seeker_ids)
# policies: dict[str, PPO]  — one SB3 PPO per seeker
```

### MAPPO

```python
from agents.mappo.mappo import load_mappo
from env.schema import OBS_DIM

obs_dim = OBS_DIM(8, 16, 7)   # 243
actor, critic = load_mappo("checkpoints/mappo_exp_seed42/ep1000", obs_dim, n_seekers=8)
actor.eval()
# critic not needed at eval time (CTDE — centralised training only)
```

---

## 5. Critical: Message Slot Behaviour at Evaluation

This is the most important thing to get right. The two algorithms behave differently
and this difference IS the experiment.

### iPPO — always zero message slots

iPPO was trained with zeroed message slots and must be evaluated the same way.
Byzantine messages have **zero effect** on iPPO — this is intentional and is your
control condition.

```python
from agents.ppo.ippo import _zero_message_slots

obs_clean = _zero_message_slots(obs_dict[agent_id], n_seekers=8)
# pass obs_clean to the iPPO policy
```

### MAPPO — pass raw obs (including Byzantine-corrupted messages)

MAPPO was trained **with real messages** from honest peers (`_use_comms=True` during
training because `protocol=BroadcastProtocol()` and `f=0.0`). At evaluation with
Byzantine agents (f>0), the message slots in `obs_dict[agent_id]` will contain a mix
of honest and corrupted positions. Pass them through **without zeroing** — the
performance drop you observe IS the Byzantine degradation signal.

```python
# MAPPO eval — do NOT zero message slots
obs_raw = obs_dict[agent_id]   # message slots already populated by env protocol
with torch.no_grad():
    obs_t  = torch.tensor(obs_raw[np.newaxis], dtype=torch.float32)
    logits = actor(obs_t)
    action = torch.distributions.Categorical(logits=logits).sample().item()
```

**Summary:**

| | f=0.0 | f>0 |
|---|---|---|
| iPPO | zero slots (same as training) | zero slots (immune) |
| MAPPO | raw obs with honest messages | raw obs with corrupted messages ← signal |

---

## 6. What Changed in the Implementation (bugs fixed since last handoff)

Four bugs were fixed in `agents/ppo/ippo.py` and `agents/mappo/mappo.py`. You do not
need to change anything, but understanding these explains why the honest baselines
shifted slightly from earlier reported numbers.

| # | Fix | Effect |
|---|-----|--------|
| 1 | **Truncation GAE bootstrap** — timeout episodes now bootstrap V(s_T) from the critic instead of 0.0 | More accurate returns for the ~40% of episodes that hit max_steps without capture |
| 2 | **Advantage normalisation** — iPPO now normalises per mini-batch (matching MAPPO) | Consistent gradient scale across both algorithms; makes comparison fair |
| 3 | **RNG seeding** — `np.random.seed(seed)` added so mini-batch shuffles are deterministic | Reproducibility: same seed → identical training run |
| 4 | **Double forward pass** (iPPO only) — removed redundant actor pass during rollout | Efficiency only, no effect on trained weights |

Also fixed in an earlier session (before these four):
- MAPPO comms bug: message slots were incorrectly zeroed for the actor even in honest training
- MAPPO critic bottleneck: widened from 1944→64 to 1944→256→128→1
- BroadcastProtocol wired into `retrain_mappo.py`

---

## 7. Experiment Configs (already updated)

These files are already correct — do not change them:

- `experiments/configs/exp1_byzantine_degradation.yaml`
  - `byzantine_fraction: [0.0, 0.125, 0.25, 0.375, 0.5]` → {0, 1, 2, 3, 4} Byzantine agents
- `experiments/configs/exp2_protocol_comparison.yaml`
  - `byzantine_fraction: [0.25]` → exactly 2 of 8 agents Byzantine

---

## 8. Quick Eval Sanity Check

Before running full sweeps, verify the loaded checkpoint produces the expected
honest performance. Run 100 episodes at f=0.0 and confirm capture rate is in the
55–65% range:

```python
# Expected: ~60% capture at f=0.0 for both algorithms
# If you see <30% something is wrong (wrong checkpoint, wrong config, zeroed MAPPO obs)
# If you see >80% something is wrong (full observability accidentally set, wrong env)
```

If MAPPO at f=0.0 matches iPPO exactly, you have accidentally zeroed MAPPO's message
slots — check step 5 above.
