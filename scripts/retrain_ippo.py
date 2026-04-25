"""
Retrain iPPO — canonical experiment config (PettingZoo pursuit benchmark defaults).

Canonical config: n_seekers=8, grid_size=16, obs_radius=7, obstacle_density=0.15, max_steps=500
This matches exp1_byzantine_degradation.yaml and exp2_protocol_comparison.yaml exactly.

Usage:
    python scripts/retrain_ippo.py --seeds 42 43 44 --run_tag exp
    python scripts/retrain_ippo.py --seeds 42 --n_episodes 500 --run_tag exp
    # Legacy exploratory runs (10x10, full obs):
    python scripts/retrain_ippo.py --grid_size 10 --n_seekers 4 --obs_radius 0 --run_tag v4
"""
import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.pursuit_env import ByzantinePursuitEnv
from agents.ppo.ippo import train, ENT_COEF
from agents.reward import reward_version, DISTANCE_SHAPING
from scripts.logger import EpisodeLogger

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",            type=int,   nargs="+", default=[42])
parser.add_argument("--n_episodes",       type=int,   default=500)
parser.add_argument("--run_tag",          type=str,   default="exp")
parser.add_argument("--n_seekers",        type=int,   default=8)
parser.add_argument("--grid_size",        type=int,   default=16)
parser.add_argument("--obs_radius",       type=int,   default=7,
                    help="FoV half-side. Pass 0 for full observability (obs_radius=None).")
parser.add_argument("--obstacle_density", type=float, default=0.15)
parser.add_argument("--max_steps",        type=int,   default=500)
args = parser.parse_args()

obs_radius = None if args.obs_radius == 0 else args.obs_radius

print(
    f"reward_version={reward_version}  DISTANCE_SHAPING={DISTANCE_SHAPING}  ENT_COEF={ENT_COEF}\n"
    f"n_seekers={args.n_seekers}  grid={args.grid_size}x{args.grid_size}  "
    f"obs_radius={obs_radius}  obstacle_density={args.obstacle_density}  "
    f"max_steps={args.max_steps}  n_episodes={args.n_episodes}  seeds={args.seeds}"
)

for seed in args.seeds:
    print(f"\n--- seed={seed} ---")
    env = ByzantinePursuitEnv(
        n_seekers=args.n_seekers,
        grid_size=args.grid_size,
        obs_radius=obs_radius,
        obstacle_density=args.obstacle_density,
        byzantine_fraction=0.0,
        max_steps=args.max_steps,
        seed=seed,
    )
    run_name = f"ippo_{args.run_tag}_seed{seed}"
    logger = EpisodeLogger(run_name, "runs/")
    train(env, n_episodes=args.n_episodes, seed=seed, logger=logger, run_tag=args.run_tag)
    logger.close()
    print(f"seed={seed} done — logs: runs/{run_name}.csv  checkpoint: checkpoints/ippo_seed{seed}/ep{args.n_episodes}/")
