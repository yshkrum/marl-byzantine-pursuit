"""
Independent PPO (iPPO) — one policy per seeker, no shared parameters.
Owner: B (RL Training Lead)
Ticket: RL-02

All N seekers collect experience simultaneously each episode via the
PettingZoo parallel API. Each seeker's policy is then updated
independently from its own trajectory. Message slots in the observation
are zeroed — seekers act with no inter-agent communication.

This establishes the performance floor for all Byzantine experiments.
Every degradation result is reported as % drop from this baseline.

Architecture  : MLP 2×64 hidden layers, ReLU activations
Hyperparams   : lr=3e-4, ε=0.2, λ_GAE=0.95, H=0.01, γ=0.99
Training      : 300 episodes, 5 seeds (§9 of context)
Target        : >70% capture rate on open arena N=4 within 300 episodes

Dependencies
------------
- ENV-01 : ByzantinePursuitEnv.reset() / step()  [blocks training]
- RL-04  : agents/reward.py reward constants      [reward computed in env]
- EXP-01 : scripts/logger.py EpisodeLogger        [locked Wed Week 1]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

import gymnasium as gym
from gymnasium import spaces as gym_spaces

from env.schema import OBS_DIM
from scripts.logger import EpisodeLogger

# ---------------------------------------------------------------------------
# Frozen hyperparameters — do not change after Tue Week 1 (RL-04 deadline).
# All magnitudes must be identical across iPPO, MAPPO, greedy, and all
# Byzantine conditions so that comparisons are valid.
# ---------------------------------------------------------------------------

LR: float = 3e-4
N_EPOCHS: int = 10
BATCH_SIZE: int = 64
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
ENT_COEF: float = 0.01
MAX_GRAD_NORM: float = 0.5

CHECKPOINT_INTERVAL: int = 50   # save every N episodes
CONVERGENCE_WINDOW: int = 50    # rolling window for convergence check
CONVERGENCE_THRESHOLD: float = 0.30  # warn if capture rate < this after 200 eps

# SB3 policy_kwargs: 2×64 MLP with ReLU (overrides SB3's default Tanh)
POLICY_KWARGS: dict = {
    "net_arch": [64, 64],
    "activation_fn": torch.nn.ReLU,
}


# ---------------------------------------------------------------------------
# Per-agent trajectory buffer
# ---------------------------------------------------------------------------

@dataclass
class _AgentRollout:
    """Accumulates one episode of experience for a single seeker."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)

    def clear(self) -> None:
        for lst in (self.obs, self.actions, self.rewards,
                    self.dones, self.values, self.log_probs):
            lst.clear()

    def __len__(self) -> int:
        return len(self.obs)


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def _zero_message_slots(obs: np.ndarray, n_seekers: int) -> np.ndarray:
    """Return a copy of *obs* with all message slots set to 0.0.

    iPPO agents receive no peer communication. Zeroing the message slots
    (the last ``2*(n_seekers-1)`` elements) ensures the policy never
    conditions on message content, keeping it strictly independent.
    """
    out = obs.copy()
    msg_start = len(obs) - 2 * (n_seekers - 1)
    out[msg_start:] = 0.0
    return out


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalised Advantage Estimation and discounted returns.

    Parameters
    ----------
    rewards, values, dones:
        Per-step lists of length T from one episode.

    Returns
    -------
    advantages : np.ndarray, shape (T,)
    returns    : np.ndarray, shape (T,)  — used as value targets
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_adv: float = 0.0

    for t in reversed(range(n)):
        non_terminal = 1.0 - float(dones[t])
        next_val = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + GAMMA * next_val * non_terminal - values[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * non_terminal * last_adv
        advantages[t] = last_adv

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO mini-batch update
# ---------------------------------------------------------------------------

def _ppo_update(ppo: PPO, rollout: _AgentRollout) -> None:
    """Run N_EPOCHS mini-batch PPO updates from *rollout*.

    Uses the SB3 policy's existing Adam optimiser and
    ``evaluate_actions()`` to recompute log-probs under the updated policy.
    Clipped PPO objective + value loss + entropy bonus, matching SB3 defaults.
    """
    if len(rollout) == 0:
        return

    advantages, returns = _compute_gae(rollout.rewards, rollout.values, rollout.dones)
    # Normalise advantages within this episode
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_arr = np.array(rollout.obs, dtype=np.float32)           # (T, obs_dim)
    act_arr = np.array(rollout.actions, dtype=np.int64)          # (T,)
    old_lp_arr = np.array(rollout.log_probs, dtype=np.float32)  # (T,)

    adv_t = torch.tensor(advantages, device=ppo.device)
    ret_t = torch.tensor(returns, device=ppo.device)
    old_lp_t = torch.tensor(old_lp_arr, device=ppo.device)

    T = len(rollout)
    indices = np.arange(T)

    for _ in range(N_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, T, BATCH_SIZE):
            idx = indices[start: start + BATCH_SIZE]

            obs_t = obs_as_tensor(obs_arr[idx], ppo.device)
            act_t = torch.tensor(act_arr[idx], dtype=torch.long, device=ppo.device)

            # Recompute values, log-probs, and entropy under current policy
            values_t, log_probs_t, entropy_t = ppo.policy.evaluate_actions(obs_t, act_t)
            values_t = values_t.flatten()

            # PPO clipped surrogate loss
            ratio = torch.exp(log_probs_t - old_lp_t[idx])
            adv_b = adv_t[idx]
            pg_loss = torch.max(
                -adv_b * ratio,
                -adv_b * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE),
            ).mean()

            vf_loss = F.mse_loss(values_t, ret_t[idx])
            ent_loss = (-entropy_t.mean()
                        if entropy_t is not None else torch.tensor(0.0, device=ppo.device))

            loss = pg_loss + 0.5 * vf_loss + ENT_COEF * ent_loss

            ppo.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.policy.parameters(), MAX_GRAD_NORM)
            ppo.policy.optimizer.step()


# ---------------------------------------------------------------------------
# Dummy gym env — used only to initialise SB3 PPO with correct spaces
# ---------------------------------------------------------------------------

class _DummyEnv(gym.Env):
    """Provides observation and action spaces to SB3 PPO at init time.

    Never stepped during training. The actual training loop drives the
    PettingZoo parallel env directly.
    """

    def __init__(self, obs_dim: int, n_actions: int = 5) -> None:
        super().__init__()
        self.observation_space = gym_spaces.Box(
            low=-1.0, high=float(obs_dim), shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym_spaces.Discrete(n_actions)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train(
    env,                  # ByzantinePursuitEnv (PettingZoo AEC)
    n_episodes: int,
    seed: int,
    logger: EpisodeLogger,
) -> Dict[str, PPO]:
    """Train one independent PPO policy per seeker agent.

    All seekers collect experience simultaneously each episode via the
    PettingZoo parallel API. After each episode, every seeker's policy is
    updated independently from its own trajectory (no gradient sharing).

    Parameters
    ----------
    env:
        ``ByzantinePursuitEnv`` instance (PettingZoo AEC).
        Requires ENV-01 to be implemented; raises ``NotImplementedError``
        on the first ``env.reset()`` call until then.
    n_episodes:
        Number of training episodes (target 300, §9 of context).
    seed:
        Master RNG seed. Same seed → same training trajectory.
    logger:
        ``EpisodeLogger`` from ``scripts/logger.py`` (frozen Wed Week 1).

    Returns
    -------
    dict[str, PPO]
        ``{agent_id: trained_ppo}`` for every seeker.
        Importable by ``scripts/run_sweep.py`` via ``from agents.ppo.ippo import train``.

    Raises
    ------
    NotImplementedError
        Propagated from ``env.reset()`` until ENV-01 is merged.
    ImportError
        If ``pettingzoo`` is not installed.
    """
    try:
        from pettingzoo.utils.conversions import aec_to_parallel
    except ImportError as exc:
        raise ImportError(
            "pettingzoo not installed — run: pip install pettingzoo"
        ) from exc

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    seeker_ids: List[str] = [
        a for a in env.possible_agents if a.startswith("seeker_")
    ]
    n_seekers = len(seeker_ids)
    _obs_dim: int = OBS_DIM(n_seekers, env.grid_size, env.obs_radius)

    checkpoint_root = Path(f"checkpoints/ippo_seed{seed}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Convert AEC → parallel for simultaneous rollout collection
    parallel_env = aec_to_parallel(env)

    # One PPO instance per seeker — no shared parameters
    dummy = _DummyEnv(obs_dim=_obs_dim)
    policies: Dict[str, PPO] = {}
    for agent_id in seeker_ids:
        policies[agent_id] = PPO(
            "MlpPolicy",
            dummy,
            policy_kwargs=POLICY_KWARGS,
            learning_rate=LR,
            n_steps=env.max_steps,   # sizes the internal rollout buffer
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            seed=seed,
            verbose=0,
        )

    rollouts: Dict[str, _AgentRollout] = {s: _AgentRollout() for s in seeker_ids}
    recent_successes: List[bool] = []
    low_capture_warned = False

    for episode in range(1, n_episodes + 1):
        # ENV-01 required: raises NotImplementedError until env is implemented
        obs_dict, _ = parallel_env.reset(seed=(seed + episode))

        episode_steps = 0
        capture_success = False
        episode_entropies: List[float] = []
        cumulative_rewards: Dict[str, float] = {s: 0.0 for s in seeker_ids}

        # ----------------------------------------------------------------
        # Episode rollout — all agents act simultaneously each step
        # ----------------------------------------------------------------
        while parallel_env.agents:
            actions: dict = {}

            # Seekers: stochastic policy forward pass
            for agent_id in seeker_ids:
                if agent_id not in parallel_env.agents:
                    continue

                obs_clean = _zero_message_slots(obs_dict[agent_id], n_seekers)
                obs_t = obs_as_tensor(obs_clean[np.newaxis], policies[agent_id].device)

                with torch.no_grad():
                    action_t, value_t, log_prob_t = policies[agent_id].policy.forward(obs_t)
                    dist = policies[agent_id].policy.get_distribution(obs_t)
                    entropy = float(dist.entropy().mean().cpu().item())

                action = int(action_t.cpu().numpy()[0])
                value = float(value_t.cpu().numpy().flatten()[0])
                log_prob = float(log_prob_t.cpu().numpy()[0])

                rollouts[agent_id].obs.append(obs_clean)
                rollouts[agent_id].actions.append(action)
                rollouts[agent_id].values.append(value)
                rollouts[agent_id].log_probs.append(log_prob)

                actions[agent_id] = action
                episode_entropies.append(entropy)

            # Hider: uniform random action (no hider policy in this baseline)
            if "hider" in parallel_env.agents:
                actions["hider"] = int(rng.integers(0, 5))

            next_obs_dict, reward_dict, term_dict, trunc_dict, _ = parallel_env.step(actions)

            # Record per-step rewards and terminal signals for seekers
            for agent_id in seeker_ids:
                if agent_id not in actions:
                    continue
                r = float(reward_dict.get(agent_id, 0.0))
                done = (term_dict.get(agent_id, False)
                        or trunc_dict.get(agent_id, False))
                rollouts[agent_id].rewards.append(r)
                rollouts[agent_id].dones.append(done)
                cumulative_rewards[agent_id] += r

            episode_steps += 1
            obs_dict = next_obs_dict

            # Capture: any seeker terminates (not truncates) → task completed
            if any(term_dict.get(s, False) for s in seeker_ids):
                capture_success = True

        # ----------------------------------------------------------------
        # Independent PPO updates — one per seeker from its own trajectory
        # ----------------------------------------------------------------
        for agent_id in seeker_ids:
            _ppo_update(policies[agent_id], rollouts[agent_id])
            rollouts[agent_id].clear()

        # ----------------------------------------------------------------
        # Logging (EpisodeLogger fields locked Wed Week 1)
        # ----------------------------------------------------------------
        mean_entropy = float(np.mean(episode_entropies)) if episode_entropies else 0.0
        logger.log(
            episode=episode,
            capture_time=episode_steps,
            capture_success=capture_success,
            n_seekers=n_seekers,
            byzantine_fraction=env.byzantine_fraction,
            protocol="none",   # iPPO: no communication
            seed=seed,
            policy_entropy=mean_entropy,
        )

        # ----------------------------------------------------------------
        # Convergence check — warn if capture rate is too low (§11)
        # ----------------------------------------------------------------
        recent_successes.append(capture_success)
        if episode >= 200 and not low_capture_warned:
            window = recent_successes[-CONVERGENCE_WINDOW:]
            rate = sum(window) / len(window)
            if rate < CONVERGENCE_THRESHOLD:
                warnings.warn(
                    f"[iPPO seed={seed}] Capture rate over last {CONVERGENCE_WINDOW} "
                    f"episodes = {rate:.2f} < {CONVERGENCE_THRESHOLD} at episode {episode}. "
                    "Debug environment and reward before proceeding to MAPPO.",
                    stacklevel=2,
                )
                low_capture_warned = True

        # ----------------------------------------------------------------
        # Checkpoint every CHECKPOINT_INTERVAL episodes
        # ----------------------------------------------------------------
        if episode % CHECKPOINT_INTERVAL == 0:
            ckpt_dir = checkpoint_root / f"ep{episode}"
            ckpt_dir.mkdir(exist_ok=True)
            for agent_id, ppo in policies.items():
                ppo.save(str(ckpt_dir / agent_id))

    return policies


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_policies(checkpoint_dir: str, seeker_ids: List[str]) -> Dict[str, PPO]:
    """Load saved iPPO policies from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir:
        Path produced by ``train()``, e.g. ``"checkpoints/ippo_seed0/ep300"``.
    seeker_ids:
        Agent IDs to load, e.g. ``["seeker_0", "seeker_1", ...]``.

    Returns
    -------
    dict[str, PPO]
        Loaded PPO instances keyed by agent_id. Missing files are silently
        skipped — caller should verify completeness.
    """
    root = Path(checkpoint_dir)
    return {
        agent_id: PPO.load(str(root / agent_id))
        for agent_id in seeker_ids
        if (root / f"{agent_id}.zip").exists()
    }
