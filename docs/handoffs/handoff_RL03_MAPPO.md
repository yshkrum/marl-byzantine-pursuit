# RL-03 MAPPO — Handoff Notes for Implementer

> **Owner:** Role B (RL Training Lead) → handoff to RL-03 assignee
> **Branch:** `b/RL-01-02-04-training-pipeline` (open PR — merge before branching off)
> **File to create:** `agents/mappo/mappo.py`
> **Deadline:** Friday EOD

---

## 1. What Is Already Done

| Ticket | File | Status |
|--------|------|--------|
| RL-01 | `agents/greedy/greedy_agent.py` | Done — BFS pursuer, tests pass |
| RL-02 | `agents/ppo/ippo.py` | Done — iPPO baseline, 1000-ep training verified |
| RL-04 | `agents/reward.py` | Done — frozen reward v2.0.0, tests pass |
| ENV step() | `env/pursuit_env.py` | Done — AEC API implemented |
| Schema | `env/schema.py` | Frozen — do not modify |
| Logger | `scripts/logger.py` | Frozen — do not modify |

### iPPO Baseline Results — your performance floor to beat

**Configuration:** N=4 seekers, 10×10 grid, 15% obstacles, full observability, 1000 episodes, seeds 42/43/44.
**Reward:** `DISTANCE_SHAPING=0.3`, `reward_version="2.0.0"` (see `agents/reward.py`).

| Seed | Overall capture | Last-100 capture | Mean capture time |
|------|----------------|-----------------|-------------------|
| 42 | 59.7% | 58.0% | 10.6 steps |
| 43 | 60.0% | 58.0% | 10.2 steps |
| 44 | 58.7% | 59.0% | 9.6 steps |
| **avg** | **59.5%** | **58.3%** | **10.1 steps** |

These seekers are actively chasing (10-step captures) but hitting a coordination ceiling without communication. MAPPO with a centralised critic should meaningfully improve capture rate — target >70%. Mean capture time should stay fast (~10 steps or better).

**CSVs for comparison:** `runs/ippo_v4_seed{42,43,44}.csv`
**Checkpoints:** `checkpoints/ippo_seed{42,43,44}/ep1000/`

---

## 2. Critical Connectivity Caveats

Read these before writing a single line of code.

### 2.1 Observation layout — do not guess, use `OBS_DIM()`

```python
from env.schema import OBS_DIM, SENTINEL

obs_dim = OBS_DIM(n_seekers, env.grid_size, env.obs_radius)
# [0]       agent_x  = agent ROW, normalised to [0, 1]
# [1]       agent_y  = agent COL, normalised to [0, 1]
# [2]       hider_x  = hider ROW, normalised; SENTINEL (-1.0) if occluded
# [3]       hider_y  = hider COL, normalised; SENTINEL (-1.0) if occluded
# [4 : 4+M] local_obstacle_map  (M = grid_size² or (2*obs_radius+1)²)
# [4+M :]   received_messages   (2*(n_seekers-1) floats)
```

**WARNING — x means row, y means col.** This is the opposite of the usual screen convention. `agent_x = obs[0] = ROW`. Every place you denormalise a position:

```python
row = int(round(float(obs[0]) * (grid_size - 1)))
col = int(round(float(obs[1]) * (grid_size - 1)))
```

If you flip row/col, the centralised critic feeds garbage into the value function.

### 2.2 Centralised critic input

The critic sees the **concatenation of all seekers' observations**. Order must be fixed and consistent across every call:

```python
# Good — deterministic order
sorted_seeker_ids = sorted(seeker_ids)   # ["seeker_0", "seeker_1", "seeker_2", "seeker_3"]
global_obs = np.concatenate([obs_dict[sid] for sid in sorted_seeker_ids])
# global_obs.shape = (n_seekers * obs_dim,)  = (4 * obs_dim,)
```

Never use dict iteration order. Never include the hider's observation in the critic input.

### 2.3 Message slots must be zeroed for the actor

Message slots (indices `[4+M:]`) are populated by Role C (BYZ-01), which is **not yet implemented**. Zero them before passing to the actor, exactly as iPPO does:

```python
# Copy from ippo.py — identical logic required
def _zero_message_slots(obs: np.ndarray, n_seekers: int) -> np.ndarray:
    out = obs.copy()
    msg_start = len(obs) - 2 * (n_seekers - 1)
    out[msg_start:] = 0.0
    return out
```

Do NOT zero for the critic — the critic should see raw obs (still zeroed in practice since Role C hasn't filled them, but the critic's pathway must be separate so it can be upgraded later without touching actor logic).

### 2.4 Frozen hyperparameters — must match iPPO exactly

These are locked by the team. Do not change them or comparisons between iPPO and MAPPO will be invalid:

```python
LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.01
N_EPOCHS      = 10
BATCH_SIZE    = 64
MAX_GRAD_NORM = 0.5
POLICY_KWARGS = {"net_arch": [64, 64], "activation_fn": torch.nn.ReLU}
```

### 2.5 Reward function — import, never copy

```python
from agents.reward import compute_rewards
# reward is computed inside env/pursuit_env.py step()
# you read rewards back from parallel_env.step() — do NOT call compute_rewards yourself
```

The reward uses `DISTANCE_SHAPING=0.3` (v2.0.0). Do not change `agents/reward.py`.

### 2.6 Parallel env wrapping — same as iPPO

```python
from pettingzoo.utils.conversions import aec_to_parallel
parallel_env = aec_to_parallel(env)
# env must have metadata["is_parallelizable"] = True  ← already set in pursuit_env.py
```

### 2.7 EpisodeLogger — call signature is frozen

```python
logger.log(
    episode=episode,
    capture_time=episode_steps,
    capture_success=capture_success,
    n_seekers=n_seekers,
    byzantine_fraction=env.byzantine_fraction,
    protocol="mappo",          # ← change from "none" in iPPO
    seed=seed,
    policy_entropy=mean_entropy,
)
```

---

## 3. Architecture Spec

### Actor (shared MLP — parameter sharing across all seekers)

- Input: single seeker's observation, shape `(obs_dim,)` — message slots zeroed
- Hidden: 2 × 64, ReLU
- Output: `Discrete(5)` policy head
- **One set of weights shared by all seekers** — each seeker feeds its own obs forward independently

### Centralised critic (CTDE)

- Input: concatenation of **all seekers'** observations, shape `(n_seekers * obs_dim,)` = `(4 * obs_dim,)` for N=4
- Hidden: 2 × 64, ReLU
- Output: scalar state value V(s)
- Updated using **returns from all seekers' trajectories** jointly

### Why shared actor?

With parameter sharing, every seeker's experience is a training signal for the one shared policy. This increases effective sample size by `n_seekers` and is standard in CTDE MAPPO literature. It also means MAPPO has no more parameters than a single iPPO policy.

---

## 4. Implementation Steps

### Step 1 — Directory scaffold

```
agents/mappo/
    __init__.py       (empty)
    mappo.py          (your file)
```

### Step 2 — Define the shared actor and centralised critic as `nn.Module`

```python
import torch
import torch.nn as nn

class _SharedActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs):               # obs: (B, obs_dim)
        return self.net(obs)              # logits: (B, n_actions)

    def get_action_and_logprob(self, obs_t):
        logits = self.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class _CentralisedCritic(nn.Module):
    def __init__(self, global_obs_dim: int):   # n_seekers * obs_dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),             nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, global_obs):    # (B, global_obs_dim)
        return self.net(global_obs).squeeze(-1)   # (B,)
```

### Step 3 — Rollout buffer

Same `_AgentRollout` dataclass as in `ippo.py` — copy it verbatim. Add a shared `global_obs_list` alongside the per-agent buffers to accumulate centralised critic inputs at each step:

```python
# At each step, build global observation
global_obs = np.concatenate([obs_dict[sid] for sid in sorted_seeker_ids])
# Append to a shared global_obs_list
```

### Step 4 — GAE with the centralised critic

Replace the per-agent value estimates with the centralised critic's values. The critic sees `global_obs` at each step:

```python
value = critic(torch.tensor(global_obs).unsqueeze(0)).item()
# store in each seeker's rollout.values — all seekers share the same V(s)
```

GAE computation is otherwise identical to `_compute_gae()` in `ippo.py` — copy it without changes.

### Step 5 — PPO update

Two separate optimisers — one for the shared actor, one for the critic:

```python
actor_optim  = torch.optim.Adam(actor.parameters(),  lr=LR)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR)
```

Actor update: same clipped PPO surrogate as `_ppo_update()` in `ippo.py`. The difference is you accumulate trajectories from **all seekers** into one batch (they share weights, so their gradients all flow into the same actor).

Critic update: MSE loss against the shared returns.

### Step 6 — `train()` signature

Must match the iPPO interface so `scripts/run_sweep.py` can call either:

```python
def train(
    env,
    n_episodes: int,
    seed: int,
    logger: EpisodeLogger,
) -> tuple[_SharedActor, _CentralisedCritic]:
    ...
```

### Step 7 — Checkpointing

```python
CHECKPOINT_INTERVAL = 50  # same as iPPO

# Save:
torch.save(actor.state_dict(),  ckpt_dir / "actor.pt")
torch.save(critic.state_dict(), ckpt_dir / "critic.pt")

# Load helper:
def load_mappo(checkpoint_dir, obs_dim, n_seekers):
    actor  = _SharedActor(obs_dim)
    critic = _CentralisedCritic(obs_dim * n_seekers)
    actor.load_state_dict(torch.load(Path(checkpoint_dir) / "actor.pt"))
    critic.load_state_dict(torch.load(Path(checkpoint_dir) / "critic.pt"))
    return actor, critic
```

---

## 5. Smoke Test (run this before committing)

```python
from env.pursuit_env import ByzantinePursuitEnv
from agents.mappo.mappo import train
from scripts.logger import EpisodeLogger

env = ByzantinePursuitEnv(
    n_seekers=4, grid_size=10, obs_radius=None,
    obstacle_density=0.15, byzantine_fraction=0.0,
    max_steps=150, seed=0,
)
logger = EpisodeLogger("smoke_mappo", "runs/")
actor, critic = train(env, n_episodes=5, seed=0, logger=logger)
logger.close()
print("Smoke test passed")
```

Expected: 5 episodes run without error, CSV written to `runs/smoke_mappo.csv`.

---

## 6. Full Training Run

Must use identical config to iPPO v4 for valid comparison:

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
```

CSVs land in `runs/mappo_seed{42,43,44}.csv`. Compare `capture_success` rate and `capture_time` to `runs/ippo_v4_seed{42,43,44}.csv`.

---

## 7. Prompt Skeletons (copy into your AI assistant if needed)

### For the initial implementation

```
Implement agents/mappo/mappo.py for the Byzantine Pursuit MARL project.

Architecture:
- Shared actor MLP (2×64, ReLU) — one set of weights used by ALL seekers
- Centralised critic MLP (2×64, ReLU) — input is concatenation of all seekers' observations
- PPO clipped objective ε=0.2, GAE λ=0.95, γ=0.99, entropy coef=0.01
- Two separate Adam optimisers (lr=3e-4), one for actor, one for critic
- N=4 seekers, 10×10 grid

Observation schema (env/schema.py, FROZEN):
- OBS_DIM(n_seekers, grid_size, obs_radius) gives obs vector length
- obs[0] = agent ROW normalised [0,1]  (agent_x)
- obs[1] = agent COL normalised [0,1]  (agent_y)
- obs[2] = hider ROW normalised; SENTINEL=-1.0 if unseen
- obs[3] = hider COL normalised; SENTINEL=-1.0 if unseen
- obs[4:4+M] = local obstacle map
- obs[4+M:] = message slots (2*(n_seekers-1) floats) — zero these before actor

Centralised critic input:
- np.concatenate([obs_dict[sid] for sid in sorted(seeker_ids)])  shape=(n_seekers*obs_dim,)
- Use the same global_obs at every step for both the value estimate and the critic update

train() signature must be:
  def train(env, n_episodes, seed, logger) -> (actor, critic)

Logger call (FROZEN interface from scripts/logger.py):
  logger.log(episode=, capture_time=, capture_success=, n_seekers=,
             byzantine_fraction=, protocol="mappo", seed=, policy_entropy=)

Env: PettingZoo AEC wrapped with aec_to_parallel (pettingzoo.utils.conversions).
Hider takes uniform random actions.
Checkpoint actor.pt / critic.pt every 50 episodes under checkpoints/mappo_seed{seed}/ep{N}/.
```

### For the GAE / update section specifically

```
In agents/mappo/mappo.py:
- The centralised critic produces one value estimate V(global_obs) per step
- Store this value in every seeker's rollout buffer (they all get the same V)
- After the episode, run GAE on each seeker's (rewards, values, dones) identically
  to _compute_gae() in agents/ppo/ippo.py
- For the actor update: collect all seekers' obs/actions/advantages into one batch
  (they share weights — this is N_seekers × more samples per update)
- For the critic update: collect global_obs from all steps, MSE against returns
- Normalise advantages per mini-batch (mean=0, std=1)
```

---

## 8. Files You Will Touch

| File | Action |
|------|--------|
| `agents/mappo/__init__.py` | Create (empty) |
| `agents/mappo/mappo.py` | Create — your main work |
| `tests/test_mappo.py` | Create — import test, smoke 5-ep test, critic input shape test |

**Do not modify:**
- `env/schema.py` — frozen
- `env/pursuit_env.py` — Role A owns it
- `agents/reward.py` — frozen (reward_version=2.0.0, DISTANCE_SHAPING=0.3)
- `scripts/logger.py` — frozen
- `agents/ppo/ippo.py` — Role B owns it

---

## 9. Branch Convention

Branch off the RL-01/02/04 PR after it merges:

```bash
git checkout main
git pull
git checkout -b b/RL-03-mappo
```

Push your branch and open a PR into `main`. Tag Role B for review.
