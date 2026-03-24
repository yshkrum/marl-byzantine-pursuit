"""
Experiment sweep runner.
Owner: D (Experiment Runner)
Ticket: EXP-02

Usage:
    python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml
    python scripts/run_sweep.py --config experiments/configs/exp2_protocol_comparison.yaml --dry-run
"""

import argparse
import itertools
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_conditions(config: dict) -> list[dict]:
    """Expand sweep parameters into a flat list of run conditions."""
    sweep = config.get("sweep", {})
    keys = list(sweep.keys())
    values = list(sweep.values())
    conditions = []
    for combo in itertools.product(*values):
        condition = dict(zip(keys, combo))
        condition["seed"] = None  # filled in per-run below
        conditions.append(condition)
    return conditions


def run_experiment(config: dict, condition: dict, seed: int, dry_run: bool = False):
    """Launch a single training run for a given condition and seed."""
    run_name = f"{config['experiment']['name']}_f{condition.get('byzantine_fraction', 0)}_s{seed}"
    print(f"  {'[DRY RUN] ' if dry_run else ''}Launching: {run_name}")
    if dry_run:
        return
    # TODO (EXP-02): import and call training loop here
    raise NotImplementedError("Training loop not yet implemented — see RL-02")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    conditions = build_conditions(config)
    n_seeds = config["training"]["n_seeds"]

    total = len(conditions) * n_seeds
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Conditions: {len(conditions)}  |  Seeds: {n_seeds}  |  Total runs: {total}\n")

    for condition in conditions:
        for seed in range(n_seeds):
            run_experiment(config, condition, seed, dry_run=args.dry_run)

    print(f"\nDone. Results → {config['logging']['output_dir']}")


if __name__ == "__main__":
    main()
