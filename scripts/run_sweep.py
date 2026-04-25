"""
Experiment sweep runner — evaluates trained MAPPO/iPPO checkpoints.
Owner: D (Experiment Runner)
Ticket: EXP-02

Expands a YAML config into a Cartesian product of sweep parameters × seeds,
then calls the appropriate training function for each condition.

Usage
-----
    # List all conditions without running anything
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml --list-conditions

    # Dry-run: print every planned run without executing
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml --dry-run

    # Full sweep
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml

    # Resume after a crash (skips CSVs that already exist)
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml --resume

    # Run a single condition (useful for parallelising across machines)
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml --condition-id 2

Constraints
-----------
- pathlib.Path throughout — no os.path
- W&B optional: only initialised if wandb is importable AND WANDB_PROJECT env var is set
- config_hash (MD5 of YAML) written as a sidecar .meta file — EpisodeMetrics is frozen
Evaluates the canonical trained checkpoints across all sweep conditions defined
in an experiment YAML config.  Does NOT retrain — loads ep1000 checkpoints.

Usage:
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml
    python scripts/run_sweep.py --config experiments/configs/exp2_protocol_comparison.yaml
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml --algo ippo
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml --dry-run

Checkpoints expected at:
    checkpoints/mappo_exp_seed{42,43,44}/ep1000/   (actor.pt + critic.pt)
    checkpoints/ippo_exp_seed{42,43,44}/ep1000/    (seeker_N.zip)

Results written to:
    experiments/results/exp1/<run_name>.csv   (one CSV per condition × seed)

See handoff_byz_experiments.md for the message-slot eval contract.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import sys
import yaml
import argparse
import csv
import itertools
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.pursuit_env import ByzantinePursuitEnv
from env.schema import OBS_DIM
from agents.byzantine import RandomNoiseByzantine
from comms import BroadcastProtocol, GossipProtocol, TrimmedMeanProtocol, ReputationProtocol

# ---------------------------------------------------------------------------
# Defaults — override via CLI flags
# ---------------------------------------------------------------------------

TRAINED_SEEDS: List[int] = [42, 43, 44]
CHECKPOINT_TAG: str = "exp"
CHECKPOINT_EPISODES: int = 1000
N_EVAL_EPISODES: int = 100

_PROTOCOL_MAP = {
    "broadcast":    BroadcastProtocol,
    "gossip":       GossipProtocol,
    "trimmed_mean": TrimmedMeanProtocol,
    "reputation":   ReputationProtocol,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_protocol(name: Optional[str]):
    """Return a protocol instance, or None for the no-comms baseline."""
    if name is None or str(name).lower() == "none":
        return None
    cls = _PROTOCOL_MAP.get(str(name).lower())
    if cls is None:
        raise ValueError(f"Unknown protocol '{name}'. Valid: {list(_PROTOCOL_MAP)}")
    return cls()


def _make_byzantine_agents(n_seekers: int, byz_fraction: float, grid_size: int, seed: int) -> dict:
    """Deterministic Byzantine assignment: agents 0..floor(N*f)-1 are Byzantine."""
    n_byz = math.floor(n_seekers * byz_fraction)
    return {
        f"seeker_{i}": RandomNoiseByzantine(
            agent_id=f"seeker_{i}", grid_size=grid_size, seed=seed + i
        )
        for i in range(n_byz)
    }


# ---------------------------------------------------------------------------
# Per-algorithm eval loops
# ---------------------------------------------------------------------------

def _eval_mappo(
    checkpoint_dir: str,
    env: ByzantinePursuitEnv,
    n_episodes: int,
    n_seekers: int,
    obs_dim: int,
) -> List[dict]:
    """
    Evaluate MAPPO actor — raw observations including Byzantine-corrupted
    message slots (do NOT zero them; the performance drop IS the signal).
    """
    from agents.mappo.mappo import load_mappo

    actor, _ = load_mappo(checkpoint_dir, obs_dim, n_seekers)
    actor.eval()
    rng = np.random.default_rng(0)
    results: List[dict] = []

    for ep in range(n_episodes):
        env.reset(seed=ep)
        step = 0
        while env.agents:
            agent = env.agent_selection
            obs = env.observe(agent)
            if agent.startswith("seeker_"):
                obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32)
                with torch.no_grad():
                    action = torch.distributions.Categorical(
                        logits=actor(obs_t)
                    ).sample().item()
            else:
                action = int(rng.integers(0, 5))
            env.step(action)
            step += 1

        captured = any(
            env.terminations.get(a, False)
            for a in env.possible_agents
            if a.startswith("seeker_")
        )
        results.append({"captured": captured, "steps": step})

    return results


def _eval_ippo(
    checkpoint_dir: str,
    env: ByzantinePursuitEnv,
    n_episodes: int,
    n_seekers: int,
    obs_dim: int,
) -> List[dict]:
    """
    Evaluate iPPO policies — message slots always zeroed (immune to Byzantine
    corruption; this is the control condition).
    """
    from agents.ppo.ippo import load_policies, _zero_message_slots

    seeker_ids = [f"seeker_{i}" for i in range(n_seekers)]
    policies = load_policies(checkpoint_dir, seeker_ids)
    rng = np.random.default_rng(0)
    results: List[dict] = []

# Add project root to path so 'env', 'agents', 'comms', 'scripts' are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Config loading & hashing
    for ep in range(n_episodes):
        env.reset(seed=ep)
        step = 0
        while env.agents:
            agent = env.agent_selection
            obs = env.observe(agent)
            if agent.startswith("seeker_"):
                obs_clean = _zero_message_slots(obs, n_seekers=n_seekers)
                action, _ = policies[agent].predict(obs_clean, deterministic=True)
            else:
                action = int(rng.integers(0, 5))
            env.step(action)
            step += 1

        captured = any(
            env.terminations.get(a, False)
            for a in env.possible_agents
            if a.startswith("seeker_")
        )
        results.append({"captured": captured, "steps": step})

    return results


# ---------------------------------------------------------------------------
# Config / condition helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(path: str | Path) -> str:
    """First 8 chars of MD5 of the raw YAML bytes — links CSVs to their config."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Condition expansion
# ---------------------------------------------------------------------------

def build_conditions(config: dict) -> list[dict]:
    """Expand sweep parameters into a flat list of run conditions.

    Each condition dict contains all swept keys plus a ``_id`` field (0-based
    index) so individual conditions can be targeted via ``--condition-id``.
    """
    sweep = config.get("sweep", {})
    keys = list(sweep.keys())
    values = [v if isinstance(v, list) else [v] for v in sweep.values()]
    conditions = []
    for idx, combo in enumerate(itertools.product(*values)):
        condition = dict(zip(keys, combo))
        condition["_id"] = idx
        conditions.append(condition)
def build_conditions(config: dict) -> List[dict]:
    """Expand sweep parameters into a flat list of run conditions."""
    sweep = config.get("sweep", {})
    keys = list(sweep.keys())
    values = list(sweep.values())
    conditions: List[dict] = []
    for combo in itertools.product(*values):
        conditions.append(dict(zip(keys, combo)))
    return conditions


# ---------------------------------------------------------------------------
# CSV path helper  (must match EpisodeLogger filename convention)
# ---------------------------------------------------------------------------

def _csv_path(config: dict, condition: dict, seed: int) -> Path:
    exp_name   = config["experiment"]["name"]
    algorithm  = config["training"]["algorithm"]
    n_seekers  = condition.get("n_seekers", config["env"].get("n_seekers", 4))
    f          = condition.get("byzantine_fraction", 0.0)
    output_dir = Path(config["logging"]["output_dir"])
    filename   = f"{exp_name}_f{f}_N{n_seekers}_{algorithm}_s{seed}.csv"
    return output_dir / filename


# ---------------------------------------------------------------------------
# Protocol factory
# ---------------------------------------------------------------------------

def _make_protocol(protocol_name: str):
    if protocol_name == "none":
        from comms.interface import NoneProtocol
        return NoneProtocol()
    elif protocol_name == "broadcast":
        from comms.broadcast import BroadcastProtocol
        return BroadcastProtocol()
    else:
        raise NotImplementedError(
            f"Protocol '{protocol_name}' is not yet implemented. "
            "Expected from Role C — see comms/ directory."
        )


# ---------------------------------------------------------------------------
# Byzantine agent factory
# ---------------------------------------------------------------------------

def _make_byzantine_agents(
    n_seekers: int,
    byzantine_fraction: float,
    grid_size: int,
    seed: int,
) -> dict:
    """Return {agent_id: RandomNoiseByzantine} for the Byzantine seekers.

    Agents seeker_0 … seeker_{n_byz-1} are Byzantine (deterministic assignment).
    Uses RandomNoiseByzantine as the default attack type for sweep runs.
    """
    n_byz = math.floor(n_seekers * byzantine_fraction)
    if n_byz == 0:
        return {}
    from agents.byzantine.subtypes import RandomNoiseByzantine
    return {
        f"seeker_{i}": RandomNoiseByzantine(
            agent_id=f"seeker_{i}",
            grid_size=grid_size,
            seed=seed + i,
        )
        for i in range(n_byz)
    }


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def run_experiment(
    config: dict,
    condition: dict,
    seed: int,
    cfg_hash: str,
    dry_run: bool = False,
    resume: bool = False,
) -> None:
    """Instantiate env + logger, call the training function, close logger.

    Parameters
    ----------
    config:
        Parsed YAML config dict.
    condition:
        One expanded sweep condition (keys from ``sweep:`` block + ``_id``).
    seed:
        Integer seed for this run.
    cfg_hash:
        8-char MD5 of the config YAML — written to a sidecar .meta file.
    dry_run:
        If True, print the planned run and return without training.
    resume:
        If True, skip runs whose output CSV already exists.
    """
    exp_name      = config["experiment"]["name"]
    algorithm     = config["training"]["algorithm"]
    n_episodes    = config["training"]["n_episodes"]
    # n_seekers may be swept (exp3/4) or fixed in env block (exp1/2)
    n_seekers     = condition.get("n_seekers", config["env"].get("n_seekers", 4))
    grid_size     = config["env"]["grid_size"]
    f             = condition.get("byzantine_fraction", 0.0)
    protocol_name = condition.get(
        "protocol",
        config.get("comms", {}).get("protocol", "broadcast"),
    )

    run_name = f"{exp_name}_f{f}_N{n_seekers}_{algorithm}_s{seed}"
    csv_path = _csv_path(config, condition, seed)

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"  {prefix}Run: {run_name}")

    if dry_run:
        return

    if resume and csv_path.exists():
        print(f"    [SKIP] CSV already exists: {csv_path}")
        return

    # --- Environment -------------------------------------------------------
    from env.pursuit_env import ByzantinePursuitEnv

    protocol        = _make_protocol(protocol_name)
    byzantine_agents = _make_byzantine_agents(n_seekers, f, grid_size, seed)

    env = ByzantinePursuitEnv(
        n_seekers        = n_seekers,
        grid_size        = grid_size,
        obs_radius       = config["env"].get("obs_radius"),
        obstacle_density = config["env"].get("obstacle_density", 0.2),
        byzantine_fraction = f,
        max_steps        = config["env"].get("max_steps", 500),
        seed             = seed,
        fixed_maze       = True,   # hold maze constant across Byzantine fractions
        protocol         = protocol,
        byzantine_agents = byzantine_agents,
    )

    # --- W&B (optional) ----------------------------------------------------
    use_wandb = False
    wandb_project = config.get("logging", {}).get("wandb_project")
    if wandb_project:
        import os
        if os.environ.get("WANDB_PROJECT"):
            try:
                import wandb as _w  # noqa: F401
                use_wandb = True
            except ImportError:
                pass

    # --- Logger ------------------------------------------------------------
    from scripts.logger import EpisodeLogger

    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(
        run_name   = run_name,
        output_dir = str(output_dir),
        use_wandb  = use_wandb,
    )

    # Write config hash as a sidecar file (EpisodeMetrics is frozen — can't
    # add fields to CSV rows without breaking the locked schema).
    meta_path = output_dir / f"{run_name}.meta"
    meta_path.write_text(
        f"config_hash={cfg_hash}\n"
        f"algorithm={algorithm}\n"
        f"protocol={protocol_name}\n"
        f"byzantine_fraction={f}\n"
        f"n_seekers={n_seekers}\n"
        f"seed={seed}\n"
    )

    # --- Training ----------------------------------------------------------
    try:
        if algorithm == "ippo":
            from agents.ppo.ippo import train
        elif algorithm == "mappo":
            from agents.mappo.mappo import train
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm!r}. "
                "Expected 'ippo' or 'mappo'. "
                "For greedy runs use validate_baseline.py."
            )
        train(env=env, n_episodes=n_episodes, seed=seed, logger=logger)
    finally:
        logger.close()


# ---------------------------------------------------------------------------
# --list-conditions helper
# ---------------------------------------------------------------------------

def list_conditions(config: dict, conditions: list[dict], n_seeds: int) -> None:
    print(f"\nExperiment : {config['experiment']['name']}")
    print(f"Description: {config['experiment'].get('description', '')}")
    header = f"{'ID':<5} {'Condition':<60} Seeds"
    print(header)
    print("-" * len(header))
    for c in conditions:
        cid  = c["_id"]
        desc = "  ".join(f"{k}={v}" for k, v in c.items() if k != "_id")
        print(f"{cid:<5} {desc:<60} 0..{n_seeds - 1}")
    total = len(conditions) * n_seeds
    print(f"\nTotal: {len(conditions)} conditions × {n_seeds} seeds = {total} runs")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="MARL Byzantine-Pursuit experiment sweep runner (Role D / EXP-02)"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print every planned run without executing training",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs whose output CSV already exists (safe restart after crash)",
    )
    parser.add_argument(
        "--condition-id", type=int, default=None, metavar="N",
        help="Run only the condition with this ID (see --list-conditions). "
             "Useful for distributing runs across machines.",
    )
    parser.add_argument(
        "--list-conditions", action="store_true",
        help="Print all conditions with their IDs and exit",
    )
    args = parser.parse_args()

    config    = load_config(args.config)
    cfg_hash  = config_hash(args.config)
    conditions = build_conditions(config)
    n_seeds   = config["training"]["n_seeds"]

    # --list-conditions: print and exit
    if args.list_conditions:
        list_conditions(config, conditions, n_seeds)
        return

    # --condition-id: filter to a single condition
    if args.condition_id is not None:
        matching = [c for c in conditions if c["_id"] == args.condition_id]
        if not matching:
            print(
                f"ERROR: No condition with ID {args.condition_id}. "
                "Run --list-conditions to see valid IDs.",
                file=sys.stderr,
            )
            sys.exit(1)
        conditions = matching

    # Pre-flight summary
    total = len(conditions) * n_seeds
    print(f"Experiment  : {config['experiment']['name']}")
    print(f"Config hash : {cfg_hash}")
    print(f"Conditions  : {len(conditions)}  |  Seeds: {n_seeds}  |  Total runs: {total}")
    print(f"Output dir  : {config['logging']['output_dir']}")
    if args.dry_run:
        print("Mode        : DRY RUN — no training will execute")
    if args.resume:
        print("Mode        : RESUME — existing CSVs will be skipped")
    print()

    for condition in conditions:
        for seed in range(n_seeds):
            run_experiment(
                config    = config,
                condition = condition,
                seed      = seed,
                cfg_hash  = cfg_hash,
                dry_run   = args.dry_run,
                resume    = args.resume,
            )

    if args.dry_run:
        print(f"\n[DRY RUN complete] {total} runs would execute.")
    else:
        print(f"\nDone. Results → {config['logging']['output_dir']}")
    algo: str,
    n_eval_episodes: int,
    checkpoint_episodes: int,
    checkpoint_tag: str = CHECKPOINT_TAG,
    dry_run: bool = False,
) -> None:
    byz_frac = float(condition.get("byzantine_fraction", 0.0))
    protocol_name = str(condition.get("protocol", "broadcast"))

    env_cfg = config["env"]
    n_seekers        = int(env_cfg["n_seekers"])
    grid_size        = int(env_cfg["grid_size"])
    obs_radius       = int(env_cfg.get("obs_radius", 7))
    obstacle_density = float(env_cfg.get("obstacle_density", 0.15))
    max_steps        = int(env_cfg.get("max_steps", 500))

    exp_name  = config["experiment"]["name"]
    run_name  = f"{exp_name}_f{byz_frac}_p{protocol_name}_s{seed}"
    output_dir = Path(config["logging"]["output_dir"])
    output_path = output_dir / f"{run_name}.csv"

    print(f"  {'[DRY RUN] ' if dry_run else ''}Launching: {run_name}")
    if dry_run:
        return

    ckpt_dir = f"checkpoints/{algo}_{checkpoint_tag}_seed{seed}/ep{checkpoint_episodes}"
    if not Path(ckpt_dir).exists():
        print(f"    WARNING: checkpoint not found at {ckpt_dir} — skipping.")
        return

    protocol        = _make_protocol(protocol_name)
    byzantine_agents = _make_byzantine_agents(n_seekers, byz_frac, grid_size, seed)

    env = ByzantinePursuitEnv(
        n_seekers=n_seekers,
        grid_size=grid_size,
        obs_radius=obs_radius,
        obstacle_density=obstacle_density,
        byzantine_fraction=byz_frac,
        max_steps=max_steps,
        seed=seed,
        protocol=protocol,
        byzantine_agents=byzantine_agents,
    )

    obs_dim  = OBS_DIM(n_seekers, grid_size, obs_radius)
    eval_fn  = _eval_mappo if algo == "mappo" else _eval_ippo
    results  = eval_fn(ckpt_dir, env, n_eval_episodes, n_seekers, obs_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "capture_success", "capture_time",
            "byzantine_fraction", "protocol", "seed", "algo",
        ])
        writer.writeheader()
        for ep_idx, r in enumerate(results):
            cap_time = r["steps"] / n_seekers if r["captured"] else float("nan")
            writer.writerow({
                "episode":            ep_idx,
                "capture_success":    r["captured"],
                "capture_time":       cap_time,
                "byzantine_fraction": byz_frac,
                "protocol":           protocol_name,
                "seed":               seed,
                "algo":               algo,
            })

    n_cap = sum(1 for r in results if r["captured"])
    cap_times = [r["steps"] / n_seekers for r in results if r["captured"]]
    mean_t = statistics.mean(cap_times) if cap_times else float("nan")
    print(
        f"    -> capture={n_cap/len(results)*100:.1f}%  "
        f"mean_time={mean_t:.1f}s  "
        f"-> {output_path}"
    )


# ---------------------------------------------------------------------------
# Post-sweep summary
# ---------------------------------------------------------------------------

def _print_summary(
    config: dict,
    conditions: List[dict],
    seeds: List[int],
    algo: str,
) -> None:
    output_dir = Path(config["logging"]["output_dir"])
    exp_name   = config["experiment"]["name"]

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — {exp_name}  (algo={algo})")
    print(f"{'='*60}")

    for condition in conditions:
        byz_frac      = float(condition.get("byzantine_fraction", 0.0))
        protocol_name = str(condition.get("protocol", "broadcast"))

        all_rows: List[dict] = []
        for seed in seeds:
            run_name = f"{exp_name}_f{byz_frac}_p{protocol_name}_s{seed}"
            p = output_dir / f"{run_name}.csv"
            if p.exists():
                with open(p, newline="") as f:
                    all_rows += list(csv.DictReader(f))

        if not all_rows:
            continue

        n     = len(all_rows)
        n_cap = sum(1 for r in all_rows if r["capture_success"] == "True")
        cap_times = [
            float(r["capture_time"])
            for r in all_rows
            if r["capture_success"] == "True" and r["capture_time"] not in ("nan", "")
        ]
        mean_t = statistics.mean(cap_times) if cap_times else float("nan")

        label = f"f={byz_frac}"
        if "protocol" in condition:
            label += f"  proto={protocol_name}"
        print(
            f"  {label:<36}  capture={n_cap/n*100:5.1f}%  "
            f"mean_time={mean_t:6.1f}s  (n={n})"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained checkpoints across Byzantine sweep conditions."
    )
    parser.add_argument("--config",               required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--algo",                 default="mappo",
                        choices=["mappo", "ippo"],
                        help="Which trained checkpoint to load (default: mappo)")
    parser.add_argument("--seeds",                type=int, nargs="+",
                        default=TRAINED_SEEDS,
                        help="Seeds to evaluate (default: 42 43 44)")
    parser.add_argument("--n_eval_episodes",      type=int,
                        default=N_EVAL_EPISODES,
                        help=f"Eval episodes per condition×seed (default {N_EVAL_EPISODES})")
    parser.add_argument("--checkpoint_episodes",  type=int,
                        default=CHECKPOINT_EPISODES,
                        help=f"Episode suffix on checkpoint dir (default {CHECKPOINT_EPISODES})")
    parser.add_argument("--checkpoint_tag",       type=str,
                        default=CHECKPOINT_TAG,
                        help=f"Tag used when training, e.g. 'exp' or 'obs3' (default {CHECKPOINT_TAG})")
    parser.add_argument("--dry-run",              action="store_true",
                        help="Print runs without executing")
    args = parser.parse_args()

    config     = load_config(args.config)
    conditions = build_conditions(config)
    total      = len(conditions) * len(args.seeds)

    print(f"Experiment : {config['experiment']['name']}")
    print(f"Algorithm  : {args.algo}")
    print(f"Conditions : {len(conditions)}  |  Seeds: {len(args.seeds)}  |  Total runs: {total}")
    print(
        f"Checkpoints: checkpoints/{args.algo}_{args.checkpoint_tag}_seed*"
        f"/ep{args.checkpoint_episodes}/"
    )
    print()

    for condition in conditions:
        for seed in args.seeds:
            run_experiment(
                config, condition, seed,
                algo=args.algo,
                n_eval_episodes=args.n_eval_episodes,
                checkpoint_episodes=args.checkpoint_episodes,
                checkpoint_tag=args.checkpoint_tag,
                dry_run=args.dry_run,
            )

    if not args.dry_run:
        _print_summary(config, conditions, args.seeds, args.algo)

    print(f"\nDone. Results -> {config['logging']['output_dir']}")


if __name__ == "__main__":
    main()
