"""
Episode logger — writes per-episode metrics to CSV and optionally W&B.
Owner: D (Experiment Runner)
Ticket: EXP-01

Usage:
    logger = EpisodeLogger(run_name="exp1_f0.33_s0", output_dir="experiments/results/exp1/")
    logger.log(episode=1, capture_time=47, capture_success=True, ...)
    logger.close()
"""

import csv
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class EpisodeMetrics:
    episode: int
    capture_time: int         # steps to capture, or max_steps if not captured
    capture_success: bool     # did seekers capture hider?
    n_seekers: int
    byzantine_fraction: float
    protocol: str
    seed: int
    policy_entropy: float = 0.0
    message_divergence: float = 0.0
    gradient_variance: float = 0.0


class EpisodeLogger:
    def __init__(self, run_name: str, output_dir: str, use_wandb: bool = False):
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        csv_path = self.output_dir / f"{run_name}.csv"
        self._f = open(csv_path, "w", newline="")
        self._writer = None  # initialised on first log call

        if use_wandb:
            import wandb
            wandb.init(project="marl-byzantine-pursuit", name=run_name)

    def log(self, **kwargs):
        metrics = EpisodeMetrics(**kwargs)
        row = asdict(metrics)
        if self._writer is None:
            self._writer = csv.DictWriter(self._f, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        if self.use_wandb:
            import wandb
            wandb.log(row)

    def close(self):
        self._f.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
