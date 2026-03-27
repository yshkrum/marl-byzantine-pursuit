"""
Retrain iPPO with updated reward (DISTANCE_SHAPING=0.3, reward_version=2.0.0).
Usage: python scripts/retrain_ippo.py
       python scripts/retrain_ippo.py --seeds 42 43 44 --n_episodes 500
       python scripts/retrain_ippo.py --run_tag v4
       python scripts/retrain_ippo.py --n_seekers 4 --run_tag v4
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
parser.add_argument("--seeds",       type=int, nargs="+", default=[42])
parser.add_argument("--n_episodes",  type=int, default=500)
parser.add_argument("--run_tag",     type=str, default="v4")
parser.add_argument("--n_seekers",   type=int, default=4)
args = parser.parse_args()

print(f"reward_version={reward_version}  DISTANCE_SHAPING={DISTANCE_SHAPING}  ENT_COEF={ENT_COEF}  n_seekers={args.n_seekers}  n_episodes={args.n_episodes}  seeds={args.seeds}")

for seed in args.seeds:
    print(f"\n--- seed={seed} ---")
    env = ByzantinePursuitEnv(
        n_seekers=args.n_seekers, grid_size=10, obs_radius=None,
        obstacle_density=0.15, byzantine_fraction=0.0,
        max_steps=150, seed=seed,
    )
    run_name = f"ippo_{args.run_tag}_seed{seed}"
    logger = EpisodeLogger(run_name, "runs/")
    train(env, n_episodes=args.n_episodes, seed=seed, logger=logger)
    logger.close()
    print(f"seed={seed} done — logs: runs/{run_name}.csv  checkpoint: checkpoints/ippo_seed{seed}/ep{args.n_episodes}/")
