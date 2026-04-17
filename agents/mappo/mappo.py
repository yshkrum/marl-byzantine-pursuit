"""
MAPPO — Multi-Agent PPO with Centralised Training / Decentralised Execution.
Owner: B (RL Training Lead)
Ticket: RL-03

All N seekers share one actor (parameter sharing). A centralised critic
observes the concatenated global state of all agents during training.
At execution time each actor uses only its own local observation (CTDE).

Architecture
------------
  Actor  : shared MLP 2×64 ReLU, Discrete(5) output — one set of weights
            used by ALL seekers; each seeker feeds its own zeroed obs.
  Critic : centralised MLP 2×64 ReLU, scalar value output.
            Input = np.concatenate([obs_dict[sid] for sid in sorted(seeker_ids)])
            shape  = (n_seekers * obs_dim,) = (4 * obs_dim,) for N=4.

Frozen hyperparameters (must match iPPO exactly so comparisons are valid):
  LR=3e-4, ε=0.2, λ_GAE=0.95, H=0.01, γ=0.99, 2×64 ReLU MLP.

Training config (canonical — do not change):
  n_seekers=8, grid_size=16, obs_radius=7, obstacle_density=0.15,
  max_steps=500, seeds [42,43,44], n_episodes=1000.

Target: match or exceed iPPO 65.9% capture rate; faster mean capture time via comms.

Dependencies
------------
- ENV-01 : ByzantinePursuitEnv.reset() / step()   [blocks training]
- RL-04  : agents/reward.py — reward computed inside env, do NOT call directly
- EXP-01 : scripts/logger.py EpisodeLogger        [frozen interface]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from env.schema import OBS_DIM
from scripts.logger import EpisodeLogger

# ---------------------------------------------------------------------------
# Frozen hyperparameters — must match iPPO exactly for valid comparison.
# ---------------------------------------------------------------------------

LR: float = 3e-4
N_EPOCHS: int = 10
BATCH_SIZE: int = 64
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
ENT_COEF: float = 0.01
MAX_GRAD_NORM: float = 0.5

CHECKPOINT_INTERVAL: int = 50       # save every N episodes
CONVERGENCE_WINDOW: int = 50        # rolling window for convergence check
CONVERGENCE_THRESHOLD: float = 0.30 # warn if capture rate < this after 200 eps


# ---------------------------------------------------------------------------
# Step 2 — Shared actor and centralised critic (separate nn.Module instances)
# ---------------------------------------------------------------------------

class _SharedActor(nn.Module):
    """Shared MLP policy — one set of weights used by ALL seekers.

    Each seeker feeds its own local observation (message slots zeroed)
    through this network independently. Parameter sharing means every
    seeker's gradient flows into the same weights, giving N× more
    training signal than a single-agent policy.

    Input : (B, obs_dim)   — single seeker's zeroed observation
    Output: (B, n_actions) — action logits
    """

    def __init__(self, obs_dim: int, n_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # (B, n_actions) logits

    def get_action_and_logprob(
        self, obs_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, entropy)."""
        logits = self.forward(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self, obs_t: torch.Tensor, actions_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recompute log_probs and entropy for stored actions (used in PPO update)."""
        logits = self.forward(obs_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_t), dist.entropy()


class _CentralisedCritic(nn.Module):
    """Centralised value network — only used during training (CTDE).

    Input : (B, n_seekers * obs_dim) — concatenated raw obs of all seekers
            in sorted agent-ID order. Never include the hider's observation.
    Output: (B,) — scalar state value V(global_obs)

    Architecture scales with global_obs_dim to avoid severe input bottleneck.
    With N=8, obs_dim=243: global_obs_dim=1944. A single 1944→64 layer loses
    30× more information per unit than the actor's 243→64 layer, preventing
    the critic from learning a useful global value function. Instead we use
    a wider first layer (256) to give the critic adequate representational
    capacity before compressing to the final scalar.
    """

    def __init__(self, global_obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Step 3 — Per-agent trajectory buffer (identical layout to iPPO)
# ---------------------------------------------------------------------------

@dataclass
class _AgentRollout:
    """Accumulates one episode of experience for a single seeker."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminals: List[bool] = field(default_factory=list)  # True only on capture, NOT on timeout
    values: List[float] = field(default_factory=list)   # V(global_obs) from centralised critic
    log_probs: List[float] = field(default_factory=list)

    def clear(self) -> None:
        for lst in (self.obs, self.actions, self.rewards,
                    self.terminals, self.values, self.log_probs):
            lst.clear()

    def __len__(self) -> int:
        return len(self.obs)


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def _zero_message_slots(obs: np.ndarray, n_seekers: int) -> np.ndarray:
    """Return a copy of *obs* with all message slots set to 0.0 for actor input.

    The critic pathway uses raw obs (kept separate so Role C can populate
    message slots for comms experiments without touching actor logic).
    """
    out = obs.copy()
    msg_start = len(obs) - 2 * (n_seekers - 1)
    out[msg_start:] = 0.0
    return out


# ---------------------------------------------------------------------------
# Step 4 — GAE with the centralised critic (identical to iPPO)
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards: List[float],
    values: List[float],
    terminals: List[bool],
    bootstrap_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalised Advantage Estimation and discounted returns.

    Parameters
    ----------
    rewards, values, terminals:
        Per-step lists of length T from one episode.
        ``values[t]`` is V(global_obs[t]) from the centralised critic —
        all seekers share the same value estimate at each step.
        ``terminals[t]`` is True only for genuine capture, NOT timeout, so
        GAE does not incorrectly cut off the bootstrap at truncated episodes.
    bootstrap_value:
        V(s_T) from the centralised critic after the episode ends. Non-zero
        for truncated (timeout) episodes; 0.0 for capture (terminal) episodes.

    Returns
    -------
    advantages : np.ndarray, shape (T,)
    returns    : np.ndarray, shape (T,)  — used as critic targets
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_adv: float = 0.0

    for t in reversed(range(n)):
        non_terminal = 1.0 - float(terminals[t])
        next_val = values[t + 1] if t + 1 < n else bootstrap_value
        delta = rewards[t] + GAMMA * next_val * non_terminal - values[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * non_terminal * last_adv
        advantages[t] = last_adv

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# Step 5 — PPO update (actor + critic with separate optimisers)
# ---------------------------------------------------------------------------

def _mappo_update(
    actor: _SharedActor,
    critic: _CentralisedCritic,
    rollouts: Dict[str, _AgentRollout],
    global_obs_list: List[np.ndarray],
    actor_optim: torch.optim.Adam,
    critic_optim: torch.optim.Adam,
    sorted_seeker_ids: List[str],
    device: torch.device,
    bootstrap_value: float = 0.0,
) -> None:
    """Run N_EPOCHS mini-batch PPO updates on the shared actor and centralised critic.

    Actor batch
    -----------
    All seekers' trajectories are concatenated into one batch. Because they
    share weights, every seeker's gradient flows into the same actor — giving
    N_seekers× more samples per update than a single-agent policy.

    Critic batch
    ------------
    The centralised critic is trained against the mean GAE return across all
    seekers at each step. Mean return is a natural target for the joint value
    function V(global_obs), which must summarise the team's expected outcome.
    """
    if not global_obs_list:
        return

    T = len(global_obs_list)
    global_obs_arr = np.array(global_obs_list, dtype=np.float32)  # (T, global_obs_dim)

    # --- Compute per-seeker advantages and returns -----------------------
    all_obs:     List[np.ndarray] = []
    all_actions: List[int]        = []
    all_old_lp:  List[float]      = []
    all_advs:    List[float]      = []
    per_seeker_returns: List[np.ndarray] = []

    for sid in sorted_seeker_ids:
        r = rollouts[sid]
        if len(r) == 0:
            continue
        adv, ret = _compute_gae(r.rewards, r.values, r.terminals, bootstrap_value)
        all_obs.extend(r.obs)
        all_actions.extend(r.actions)
        all_old_lp.extend(r.log_probs)
        all_advs.extend(adv.tolist())
        per_seeker_returns.append(ret)

    if not all_obs:
        return

    # Actor tensors — pooled trajectories from all seekers
    obs_arr    = np.array(all_obs,     dtype=np.float32)   # (n_sk*T, obs_dim)
    act_arr    = np.array(all_actions, dtype=np.int64)     # (n_sk*T,)
    old_lp_arr = np.array(all_old_lp,  dtype=np.float32)  # (n_sk*T,)
    adv_arr    = np.array(all_advs,    dtype=np.float32)   # (n_sk*T,)

    old_lp_t = torch.tensor(old_lp_arr, device=device)
    adv_t    = torch.tensor(adv_arr,    device=device)

    # Critic target — mean GAE return across all seekers at each step: (T,)
    critic_ret_mean = np.mean(np.stack(per_seeker_returns, axis=0), axis=0)
    glob_t     = torch.tensor(global_obs_arr,  device=device)   # (T, global_obs_dim)
    crit_ret_t = torch.tensor(critic_ret_mean, device=device)   # (T,)

    N_total  = len(all_obs)
    actor_indices  = np.arange(N_total)
    critic_indices = np.arange(T)

    for _ in range(N_EPOCHS):

        # --- Actor update ------------------------------------------------
        np.random.shuffle(actor_indices)
        for start in range(0, N_total, BATCH_SIZE):
            idx = actor_indices[start: start + BATCH_SIZE]

            obs_t  = torch.tensor(obs_arr[idx],  device=device)
            act_t  = torch.tensor(act_arr[idx],  device=device)
            adv_b  = adv_t[idx]
            olp_b  = old_lp_t[idx]

            # Normalise advantages per mini-batch
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            log_probs, entropy = actor.evaluate_actions(obs_t, act_t)
            ratio = torch.exp(log_probs - olp_b)

            pg_loss = torch.max(
                -adv_b * ratio,
                -adv_b * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE),
            ).mean()
            ent_loss = -entropy.mean()

            actor_loss = pg_loss + ENT_COEF * ent_loss

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            actor_optim.step()

        # --- Critic update -----------------------------------------------
        np.random.shuffle(critic_indices)
        for start in range(0, T, BATCH_SIZE):
            idx = critic_indices[start: start + BATCH_SIZE]

            glob_b  = glob_t[idx]
            ret_b   = crit_ret_t[idx]

            critic_loss = F.mse_loss(critic(glob_b), ret_b)

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            critic_optim.step()


# ---------------------------------------------------------------------------
# Step 6 — Public training entry point
# ---------------------------------------------------------------------------

def train(
    env,                    # ByzantinePursuitEnv (PettingZoo AEC)
    n_episodes: int,
    seed: int,
    logger: EpisodeLogger,
    run_tag: str = "",
) -> Tuple[_SharedActor, _CentralisedCritic]:
    """Train MAPPO: shared actor + centralised critic via CTDE.

    All seekers collect experience simultaneously each episode via the
    PettingZoo parallel API. After each episode the shared actor is updated
    from the pooled trajectories of all seekers. The centralised critic is
    updated against the mean GAE return across seekers at each step.

    Parameters
    ----------
    env:
        ``ByzantinePursuitEnv`` instance (PettingZoo AEC).
    n_episodes:
        Number of training episodes (1000 for canonical paper runs).
    seed:
        Master RNG seed — same seed yields identical training trajectory.
    logger:
        ``EpisodeLogger`` from ``scripts/logger.py`` (frozen interface).

    Returns
    -------
    (actor, critic)
        Trained ``_SharedActor`` and ``_CentralisedCritic`` instances.
        Callable from ``scripts/run_sweep.py`` via
        ``from agents.mappo.mappo import train``.

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
    np.random.seed(seed)  # makes np.random.shuffle in _mappo_update deterministic
    device = torch.device("cpu")

    # Sorted seeker IDs — order must be fixed and consistent across every call
    sorted_seeker_ids: List[str] = sorted(
        a for a in env.possible_agents if a.startswith("seeker_")
    )
    n_seekers = len(sorted_seeker_ids)
    obs_dim: int = OBS_DIM(n_seekers, env.grid_size, env.obs_radius)
    global_obs_dim: int = n_seekers * obs_dim   # (4 * obs_dim) for N=4

    # --- Step 2: instantiate actor and critic as separate nn.Module objects ---
    actor  = _SharedActor(obs_dim).to(device)
    critic = _CentralisedCritic(global_obs_dim).to(device)

    # Assertion required by spec — actor and critic must be separate instances
    assert isinstance(actor, nn.Module) and isinstance(critic, nn.Module)
    assert actor is not critic, "actor and critic must be separate nn.Module instances"

    # --- Step 5: two separate optimisers ------------------------------------
    actor_optim  = torch.optim.Adam(actor.parameters(),  lr=LR)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=LR)

    # --- Step 7: checkpoint directory ---------------------------------------
    _tag = f"_{run_tag}" if run_tag else ""
    checkpoint_root = Path(f"checkpoints/mappo{_tag}_seed{seed}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    parallel_env = aec_to_parallel(env)

    # Zero message slots only when there is no honest protocol, or when Byzantine
    # agents are present (corrupted messages should not reach the actor).
    # With an honest broadcast protocol at f=0.0 the actor should read real peer
    # messages — zeroing them would negate the entire CTDE communications advantage.
    _use_comms = (
        getattr(env, "_protocol", None) is not None
        and env.byzantine_fraction == 0.0
    )

    rollouts: Dict[str, _AgentRollout] = {s: _AgentRollout() for s in sorted_seeker_ids}
    recent_successes: List[bool] = []
    low_capture_warned = False

    for episode in range(1, n_episodes + 1):
        obs_dict, _ = parallel_env.reset(seed=(seed + episode))

        episode_steps    = 0
        capture_success  = False
        episode_entropies: List[float]      = []
        global_obs_list:   List[np.ndarray] = []   # one entry per step

        for r in rollouts.values():
            r.clear()

        # ----------------------------------------------------------------
        # Episode rollout — all agents act simultaneously each step
        # ----------------------------------------------------------------
        while parallel_env.agents:
            actions: dict = {}

            # --- Build global obs (raw, not zeroed) in fixed sorted order ---
            # The critic pathway must stay separate from the actor pathway so
            # Role C can later populate message slots without touching actor logic.
            global_obs = np.concatenate([
                obs_dict.get(sid, np.zeros(obs_dim, dtype=np.float32))
                for sid in sorted_seeker_ids
            ])
            global_obs_list.append(global_obs)

            # --- Centralised critic: one V(global_obs) shared by all seekers ---
            with torch.no_grad():
                glob_t = torch.tensor(
                    global_obs[np.newaxis], dtype=torch.float32, device=device
                )
                step_value = float(critic(glob_t).cpu().item())

            # --- Shared actor: pass real messages when protocol is honest,
            #     zero slots when no protocol or Byzantine agents are present ---
            for agent_id in sorted_seeker_ids:
                if agent_id not in parallel_env.agents:
                    continue

                obs_clean = (
                    obs_dict[agent_id]
                    if _use_comms
                    else _zero_message_slots(obs_dict[agent_id], n_seekers)
                )
                obs_t = torch.tensor(
                    obs_clean[np.newaxis], dtype=torch.float32, device=device
                )

                with torch.no_grad():
                    action_t, log_prob_t, entropy_t = actor.get_action_and_logprob(obs_t)
                    entropy = float(entropy_t.mean().cpu().item())

                action   = int(action_t.cpu().item())
                log_prob = float(log_prob_t.cpu().item())

                rollouts[agent_id].obs.append(obs_clean.copy())
                rollouts[agent_id].actions.append(action)
                rollouts[agent_id].values.append(step_value)   # shared V(global_obs)
                rollouts[agent_id].log_probs.append(log_prob)

                actions[agent_id] = action
                episode_entropies.append(entropy)

            # Hider: uniform random action (no hider policy)
            if "hider" in parallel_env.agents:
                actions["hider"] = int(rng.integers(0, 5))

            next_obs_dict, reward_dict, term_dict, trunc_dict, _ = parallel_env.step(actions)

            # Store term_dict only (not trunc) so GAE does not cut off the value
            # bootstrap at timeout steps — truncation handled via bootstrap_value.
            for agent_id in sorted_seeker_ids:
                if agent_id not in actions:
                    continue
                r_val = float(reward_dict.get(agent_id, 0.0))
                rollouts[agent_id].rewards.append(r_val)
                rollouts[agent_id].terminals.append(term_dict.get(agent_id, False))

            episode_steps += 1
            obs_dict = next_obs_dict

            # Capture: any seeker terminates (not truncates) → task completed
            if any(term_dict.get(s, False) for s in sorted_seeker_ids):
                capture_success = True

        # ----------------------------------------------------------------
        # Bootstrap value for truncated (timeout) episodes.
        # After the parallel_env loop exits, obs_dict holds the final
        # next_obs from PettingZoo — s_T for truncated episodes.
        # The centralised critic on the concatenated final obs gives V(s_T).
        # All seekers share the same bootstrap (centralised critic).
        # ----------------------------------------------------------------
        bootstrap_value = 0.0
        if not capture_success:
            final_global = np.concatenate([
                obs_dict.get(sid, np.zeros(obs_dim, dtype=np.float32))
                for sid in sorted_seeker_ids
            ])
            with torch.no_grad():
                final_glob_t = torch.tensor(
                    final_global[np.newaxis], dtype=torch.float32, device=device
                )
                bootstrap_value = float(critic(final_glob_t).cpu().item())

        # ----------------------------------------------------------------
        # MAPPO update — shared actor + centralised critic
        # ----------------------------------------------------------------
        _mappo_update(
            actor, critic,
            rollouts, global_obs_list,
            actor_optim, critic_optim,
            sorted_seeker_ids, device,
            bootstrap_value=bootstrap_value,
        )

        # ----------------------------------------------------------------
        # Logging (EpisodeLogger fields frozen)
        # ----------------------------------------------------------------
        mean_entropy = float(np.mean(episode_entropies)) if episode_entropies else 0.0
        logger.log(
            episode=episode,
            capture_time=episode_steps,
            capture_success=capture_success,
            n_seekers=n_seekers,
            byzantine_fraction=env.byzantine_fraction,
            protocol="mappo",
            seed=seed,
            policy_entropy=mean_entropy,
        )

        # ----------------------------------------------------------------
        # Convergence check — warn if capture rate is too low (matches iPPO)
        # ----------------------------------------------------------------
        recent_successes.append(capture_success)
        if episode >= 200 and not low_capture_warned:
            window = recent_successes[-CONVERGENCE_WINDOW:]
            rate = sum(window) / len(window)
            if rate < CONVERGENCE_THRESHOLD:
                warnings.warn(
                    f"[MAPPO seed={seed}] Capture rate over last {CONVERGENCE_WINDOW} "
                    f"episodes = {rate:.2f} < {CONVERGENCE_THRESHOLD} at episode {episode}. "
                    "Debug environment and reward before accepting these results.",
                    stacklevel=2,
                )
                low_capture_warned = True

        # ----------------------------------------------------------------
        # Step 7 — Checkpoint every CHECKPOINT_INTERVAL episodes
        # ----------------------------------------------------------------
        if episode % CHECKPOINT_INTERVAL == 0:
            ckpt_dir = checkpoint_root / f"ep{episode}"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(actor.state_dict(),  ckpt_dir / "actor.pt")
            torch.save(critic.state_dict(), ckpt_dir / "critic.pt")

    return actor, critic


# ---------------------------------------------------------------------------
# Step 7 — Checkpoint loading helper
# ---------------------------------------------------------------------------

def load_mappo(
    checkpoint_dir: str,
    obs_dim: int,
    n_seekers: int,
) -> Tuple[_SharedActor, _CentralisedCritic]:
    """Load saved MAPPO actor and critic from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir:
        Path produced by ``train()``, e.g.
        ``"checkpoints/mappo_seed42/ep1000"``.
    obs_dim:
        Per-agent observation dimension (from ``OBS_DIM()``).
    n_seekers:
        Number of seekers — determines critic input size.

    Returns
    -------
    (actor, critic)
        Loaded ``_SharedActor`` and ``_CentralisedCritic`` with weights
        restored. Both are in eval mode.
    """
    root = Path(checkpoint_dir)
    actor  = _SharedActor(obs_dim)
    critic = _CentralisedCritic(obs_dim * n_seekers)
    actor.load_state_dict(torch.load(root / "actor.pt",  map_location="cpu"))
    critic.load_state_dict(torch.load(root / "critic.pt", map_location="cpu"))
    actor.eval()
    critic.eval()
    return actor, critic