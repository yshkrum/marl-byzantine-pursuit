# MARL Byzantine-Resilient Pursuit
## AI Context Descriptor — Role B: RL Training Lead

> *You define what good performance looks like. Your baselines are the yardstick every Byzantine experiment is measured against.*

---

## 1. Project Overview

| Field | Value |
|-------|-------|
| **Name** | MARL Byzantine-Resilient Pursuit |
| **Repo** | github.com/yshkrum/marl-byzantine-pursuit |
| **Research Q** | Does Byzantine communication corruption degrade cooperative pursuit performance predictably, and can learned or engineered resilience protocols recover it — and at what team size does coordination itself become the bottleneck? |
| **Tech stack** | Python 3.11, PettingZoo, PyTorch, stable-baselines3, numpy, pandas, matplotlib, seaborn, wandb, pytest, YAML configs |

A 5-person MSc group project building a multi-agent reinforcement learning testbed for pursuit-evasion with Byzantine-corrupted communication. N seeker agents cooperate to capture 1 hider in a 2D gridworld maze. A fraction f of seekers are Byzantine: they execute honest movement but corrupt their outgoing communication messages. The project measures capture performance degradation, tests resilience protocols, and identifies scalability limits.

---

## 2. Experiments

- Exp 1: Byzantine degradation — capture time vs f ∈ {0, 0.17, 0.33, 0.5}, N=6, broadcast protocol
- Exp 2: Protocol resilience — 5 protocols at f=0.33, N=6
- Exp 3: Scalability — N ∈ {2,4,6,8,10,14}, f=0
- Exp 4: Phase diagram — f × N sweep, capture probability heatmap

---

## 3. Team Structure

| Role | Owner |
|------|-------|
| **A** | Environment Engineer — owns env/pursuit_env.py, maze generation, observation space |
| **B** | **RL Training Lead — owns agents/ppo/, agents/mappo/, agents/greedy/, reward function** |
| **C** | Byzantine & Communication Systems — owns agents/byzantine/, comms/ |
| **D** | Experiment Runner & Data Pipeline — owns scripts/run_sweep.py, scripts/logger.py, experiments/configs/ |
| **E** | Analysis & Visualisation — owns analysis/plots/, paper §2 Related Work |

---

## 4. Frozen Contracts (never modify unilaterally)

- `env/schema.py` — observation vector, message struct, action space (locked Tue Week 1)
- `agents/reward.py` — reward function with magnitudes (locked Tue Week 1)
- `experiments/configs/*.yaml` — experiment parameters (reviewed before running)
- `scripts/logger.py` — EpisodeMetrics fields (locked Wed Week 1)

---

## 5. File & Output Conventions

- Output CSVs: `experiments/results/{exp_name}_f{f}_N{n}_s{seed}.csv`
- Checkpoints: `checkpoints/{algorithm}_seed{seed}/` (gitignored)
- Figures: `analysis/plots/fig{N}_{description}.pdf` + `.png`
- Never commit: `*.pt`, `*.pth`, `experiments/results/`, `wandb/`

---

## YOUR ROLE: RL Training Lead
*Paper section: §3.2 Learning Setup*

---

## 6. Files You Own

- `agents/ppo/ippo.py` — Independent PPO training loop
- `agents/mappo/mappo.py` — MAPPO with centralised critic
- `agents/greedy/greedy_agent.py` — BFS heuristic pursuer
- `agents/reward.py` — frozen reward function (shared by all agents)
- `tests/test_agents.py` — training sanity tests

---

## 7. Dependencies & Blockers

**Depends on:**
- A (ENV-01) for stable environment to train against
- A (ENV-04) for observation vector layout
- D (EXP-01) for EpisodeLogger to log training metrics

**You block:**
- D cannot run baseline sweeps (EXP-03)
- E cannot produce baseline plots (VIZ-02)

---

## 8. Week 1 Priority Order

> RL-04 (reward function) by Tue, RL-01 (greedy) by Tue on stub grid, RL-02 (iPPO) by Thu, RL-03 (MAPPO) by Fri.

---

## 9. Key Technical Decisions (already made)

- Reward: +10 capture (individual), +5 team bonus (split), -0.01 step penalty, +0.1×delta_distance shaping. All in `reward.py` as constants.
- iPPO: each seeker gets independent PPO instance. No shared weights. Message slots in obs zeroed out.
- MAPPO: shared actor weights across seekers. Centralised critic input = concatenated all-agent obs. Decentralised at execution.
- Network: MLP 2×64 hidden, ReLU. PPO epsilon=0.2, GAE lambda=0.95, entropy coeff=0.01, lr=3e-4.
- Episode budget: 300 episodes per run, 5 seeds. Do not tune hyperparameters this week.
- Target: iPPO achieves >70% capture rate on open arena N=4 within 300 episodes. If not, debug env/reward before proceeding.

---

## 10. Technical Context

Use stable-baselines3 PPO as base. For iPPO: wrap PettingZoo env with SuperSuit `ss.pettingzoo_env_to_vec_env_v1` for each agent independently. For MAPPO: implement custom centralised critic using `nn.Module`, feed concatenated obs from all seekers. Parameter sharing: all seekers share one actor `nn.Module` instance but call it with their own obs. Checkpoint via `torch.save(policy.state_dict(), path)`. Load via `policy.load_state_dict(torch.load(path))`. Log via D's EpisodeLogger — import from `scripts.logger`.

---

## 11. Best Practices for Fast, Efficient Delivery

- Build greedy agent FIRST — it needs no env, confirm capture logic works before touching RL.
- Time-box MAPPO to 2 days. Working but imperfect > perfect but late.
- Add a convergence check: if capture rate over last 50 episodes < 0.3 after 200 episodes, raise a warning and log it.
- Separate training loop from policy class — `train(env, n_episodes)` should be callable from `run_sweep.py`.
- Never tune hyperparameters in week 1. Write them as module constants, move on.
- Log policy entropy every episode — it is a required metric for the paper.
- Test `reward.py` independently before integrating — unit test that capture gives +15 total, step gives -0.01.

---

## 12. AI Model Context Prompt

Paste this at the start of every Claude Code / Cowork session for this role. Replace `{TASK_DESCRIPTION}` with your specific ticket.

```
You are working on the reinforcement learning agents for a multi-agent research project
called MARL Byzantine-Resilient Pursuit.

PROJECT: A 5-person MSc group project building a multi-agent reinforcement learning testbed
for pursuit-evasion with Byzantine-corrupted communication. N seeker agents cooperate to
capture 1 hider in a 2D gridworld maze. A fraction f of seekers are Byzantine: they execute
honest movement but corrupt their outgoing communication messages. The project measures
capture performance degradation, tests resilience protocols, and identifies scalability limits.

YOUR ROLE: RL Training Lead. You own the training loops for Independent PPO, MAPPO, and the
greedy heuristic baseline.

FROZEN CONTRACTS (do not modify):
- env/schema.py: OBS_FIELDS defines observation vector layout. Message slots are the last
  2*(n_seekers-1) elements.
- agents/reward.py: compute_rewards(state, actions, prev_state, n_seekers) — do not
  reimplement reward logic elsewhere.
- scripts/logger.py: EpisodeLogger — use this for all metric logging, do not write custom
  CSV code.

CURRENT TASK: {TASK_DESCRIPTION}

CONSTRAINTS:
- stable-baselines3 PPO as base implementation
- Network: MLP 2×64 hidden, ReLU, PPO epsilon=0.2, GAE lambda=0.95, entropy coeff=0.01,
  lr=3e-4
- For iPPO: zero out message slots in obs (indices OBS_DIM - 2*(n_seekers-1) onward) before
  passing to policy
- For MAPPO: centralised critic receives torch.cat([obs_i for all seekers], dim=-1)
- Checkpoints: save to checkpoints/{algorithm}_seed{seed}/ every 50 episodes
- All training functions must accept (env, n_episodes:int, seed:int, logger:EpisodeLogger)
  signature

OUTPUT: Production-ready Python with type hints. Training loop must be importable and
callable from scripts/run_sweep.py.
```

---

## 13. Quick Reference Card

| Field | Value |
|-------|-------|
| **Repo** | github.com/yshkrum/marl-byzantine-pursuit |
| **Branch prefix** | `b/` (e.g. `b/RL-01-greedy-baseline`) |
| **Commit format** | `TICKET-ID: description` (e.g. `RL-01: add greedy BFS agent`) |
| **PR target** | `dev` (never `main`) |
| **Run tests** | `pytest tests/ -v` |
| **Frozen files** | `env/schema.py`, `agents/reward.py` — never edit without team agreement |
| **Never commit** | `*.pt`, `*.pth`, `experiments/results/`, `wandb/` |
| **Paper section** | §3.2 Learning Setup |
