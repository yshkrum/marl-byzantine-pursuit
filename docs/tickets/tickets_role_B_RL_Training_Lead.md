# Week 1 Sprint Tickets
## Role B: RL Training Lead
*Paper ownership: §3.2 Learning Setup*

---

### RL-01 · Implement greedy heuristic pursuer
**Priority:** High · **Deadline:** Tue EOD · **Tags:** `implementation`

**Description**

Implement a non-learning greedy pursuer that moves toward the hider's last known position using BFS shortest-path on the grid. If the hider is not currently visible, the agent continues toward its last known position. If no position has ever been observed, the agent moves randomly.

This baseline requires no training and can be developed against a stub grid before ENV-01 is complete — build a minimal 10x10 test grid to develop against. The greedy baseline serves two purposes: (1) it confirms that the capture logic and episode termination work correctly before RL is introduced, and (2) it provides the absolute performance floor that all learning algorithms must beat to justify the RL component of the paper.

**Claude Code / Cowork prompt**

```
Implement agents/greedy/greedy_agent.py for a MARL pursuit-evasion gridworld. The GreedyAgent
class takes agent_id and grid_size as constructor args. Implement act(observation, obstacle_map)
method: extract agent position and last-known hider position from the observation vector (using
indices from env/schema.py). Run BFS on the obstacle-free grid from agent position to hider's
last known position. Return the action (0–4) corresponding to the first step of the shortest
path. If no hider position is known, return a random action. If path is blocked (no route
exists), return NOOP. Include a test in tests/test_greedy.py: place agent at (0,0) and hider
at (5,5) on a 10x10 empty grid and verify the agent moves diagonally-optimally toward the
hider over 5 steps.
```

---

### RL-02 · Implement Independent PPO (no communication)
**Priority:** High · **Deadline:** Thu EOD · **Tags:** `implementation`

**Description**

Implement Independent PPO where each seeker trains its own policy with no message passing between agents. Each agent observes only its local observation vector (no received messages component). Use CleanRL or stable-baselines3 as the base implementation — do not write a PPO implementation from scratch.

Reward structure (from RL-04): +10 on capture, −0.01 per step, +1 for reducing distance to hider (shaped reward optional, document if used). Target: capture rate above 70% on open arena (N=4, no obstacles) within 200 training episodes across 3 seeds. If this target is not met, do not proceed to MAPPO — debug the environment and reward first.

This is the performance floor for all subsequent experiments. Every Byzantine degradation result in the paper is measured as percentage degradation from this baseline.

**Claude Code / Cowork prompt**

```
Implement Independent PPO for agents/ppo/ippo.py for a PettingZoo multi-agent pursuit
environment. Use stable-baselines3 PPO as the base. Each seeker agent gets its own independent
PPO instance with no shared parameters. Observation: the local obs vector from env/schema.py
with the message fields zeroed out. Policy network: MLP with two 64-unit hidden layers, ReLU
activations. Reward: +10 on capture, -0.01 per step. Implement a train(env, n_episodes, seed)
function that trains all seeker policies independently and logs per-episode capture_time,
capture_success, and mean policy entropy to the EpisodeLogger from scripts/logger.py. Save
checkpoints every 50 episodes to checkpoints/ippo_seed{seed}/. Return a trained policy dict
keyed by agent_id.
```

---

### RL-03 · Implement MAPPO with shared critic (CTDE)
**Priority:** Med · **Deadline:** Fri EOD · **Tags:** `implementation`

**Description**

Implement MAPPO (Multi-Agent PPO) using Centralised Training with Decentralised Execution. The shared critic observes the concatenated global state of all agents during training. At execution time, each actor uses only its local observation. Use parameter sharing across all seeker actors — they share weights but receive different local observations.

Time-box this to 2 days maximum. The goal this week is a working training loop that produces convergence curves, not optimal hyperparameters. Hyperparameter tuning happens in week 2 if results are poor. A partially converged MAPPO that shows improvement over iPPO on the baseline is sufficient to validate the CTDE implementation.

**Claude Code / Cowork prompt**

```
Implement MAPPO in agents/mappo/mappo.py for the Byzantine Pursuit environment. Architecture:
shared actor MLP (two 64-unit layers, parameter sharing across all seekers) and a centralised
critic MLP that takes concatenated observations of all seekers as input. Use PPO clipped
objective with epsilon=0.2, GAE lambda=0.95, entropy coefficient=0.01. At execution, each
actor uses only its local observation. Implement train(env, n_episodes, seed) with the same
logging interface as ippo.py (EpisodeLogger). The centralised critic is only used during
training — during rollout collection and at test time, actors run independently. Add an
assertion that verifies the actor and critic are separate nn.Module instances.
```

---

### RL-04 · Define and document reward function
**Priority:** High · **Deadline:** Tue EOD · **Tags:** `design` `blocks all`

**Description**

Lock the reward function shared across all experiments and document it in a constants file. Proposed structure: individual capture reward (+10 for the seeker that tags the hider), shared team reward component (+5 split across all seekers on capture), step penalty (−0.01 per step per agent), optional distance shaping reward (+0.1 for reducing distance to hider, −0.1 for increasing — document clearly if included as it affects sample efficiency comparisons).

Circulate to all 5 team members for agreement by Tuesday EOD. The reward function must not change after this point — any change invalidates comparisons between experiments run before and after the change. If distance shaping is used, it must be identical across iPPO, MAPPO, greedy, and all Byzantine conditions.

**Claude Code / Cowork prompt**

```
Create agents/reward.py defining the frozen reward function for the Byzantine Pursuit project.
Define a compute_rewards(state, actions, prev_state, n_seekers) function returning a dict of
{agent_id: float} rewards. Reward components: CAPTURE_REWARD = 10.0 for the capturing seeker,
TEAM_CAPTURE_BONUS = 5.0 divided equally among all seekers on capture, STEP_PENALTY = -0.01
per step per agent, DISTANCE_SHAPING = 0.1 * (prev_distance - curr_distance) per seeker
(positive if approaching hider). Define all magnitudes as module-level constants so they can
be cited in the paper. Write a docstring explaining the rationale for each component. Add a
reward_version string. Include unit tests: verify that total reward on a capture step sums
correctly, and that step penalty accumulates correctly over 10 steps.
```

---

### RL-05 · Write §3.2 Learning Setup (paper section)
**Priority:** Low · **Deadline:** Fri EOD · **Tags:** `paper`

**Description**

Draft the learning setup section for the paper. Cover: Independent PPO description (independent policies, no communication, performance floor), MAPPO description (centralised critic, decentralised execution, parameter sharing), greedy heuristic (BFS, sanity check baseline), reward function (all components with magnitudes), and key hyperparameters (network architecture, learning rate, PPO epsilon, GAE lambda, episode budget). Approximately 350–400 words in continuous prose.

**Claude Code / Cowork prompt**

```
Write the Learning Setup subsection (§3.2) for a MARL research paper on Byzantine-resilient
cooperative pursuit. Cover three agent types: (1) Greedy Heuristic — BFS shortest-path
pursuer, no learning, serves as lower-bound baseline; (2) Independent PPO — each seeker
trains an independent MLP policy (2×64 hidden units, ReLU) with no inter-agent communication,
trained with PPO (epsilon=0.2, GAE lambda=0.95); (3) MAPPO with CTDE — shared actor weights
across seekers, centralised critic observing global state during training, decentralised
execution. State the reward function: capture reward, team bonus, step penalty, distance
shaping. State training budget: 300 episodes, 5 independent seeds per condition. Write in
formal academic style, third person, past tense. 350–400 words, continuous prose.
```
