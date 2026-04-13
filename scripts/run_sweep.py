"""
Experiment sweep runner.
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
"""

from __future__ import annotations

import hashlib
import itertools
import math
import sys
import yaml
from pathlib import Path

# Add project root to path so 'env', 'agents', 'comms', 'scripts' are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Config loading & hashing
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


if __name__ == "__main__":
    main()
