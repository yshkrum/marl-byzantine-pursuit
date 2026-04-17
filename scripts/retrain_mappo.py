"""
Retrain MAPPO — canonical experiment config (PettingZoo pursuit benchmark defaults).

Canonical config: n_seekers=8, grid_size=16, obs_radius=7, obstacle_density=0.15, max_steps=500
Matches exp1_byzantine_degradation.yaml and exp2_protocol_comparison.yaml exactly.

Usage:
    python scripts/retrain_mappo.py --seeds 42 43 44 --run_tag exp
    python scripts/retrain_mappo.py --seeds 42 --n_episodes 500 --run_tag exp
"""
import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.pursuit_env import ByzantinePursuitEnv
from agents.mappo.mappo import train, ENT_COEF
from agents.reward import reward_version, DISTANCE_SHAPING
from comms.broadcast import BroadcastProtocol
from scripts.logger import EpisodeLogger

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",            type=int,   nargs="+", default=[42])
parser.add_argument("--n_episodes",       type=int,   default=500)
parser.add_argument("--run_tag",          type=str,   default="exp")
parser.add_argument("--n_seekers",        type=int,   default=8)
parser.add_argument("--grid_size",        type=int,   default=16)
parser.add_argument("--obs_radius",       type=int,   default=7,
                    help="FoV half-side. Pass 0 for full observability.")
parser.add_argument("--obstacle_density", type=float, default=0.15)
parser.add_argument("--max_steps",        type=int,   default=500)
args = parser.parse_args()

obs_radius = None if args.obs_radius == 0 else args.obs_radius

print(
    "reward_version=%s  DISTANCE_SHAPING=%.1f  ENT_COEF=%.3f\n"
    "n_seekers=%d  grid=%dx%d  obs_radius=%s  obstacle_density=%.2f  "
    "max_steps=%d  n_episodes=%d  seeds=%s" % (
        reward_version, DISTANCE_SHAPING, ENT_COEF,
        args.n_seekers, args.grid_size, args.grid_size,
        obs_radius, args.obstacle_density,
        args.max_steps, args.n_episodes, args.seeds,
    )
)

for seed in args.seeds:
    print("\n--- seed=%d ---" % seed)
    env = ByzantinePursuitEnv(
        n_seekers=args.n_seekers,
        grid_size=args.grid_size,
        obs_radius=obs_radius,
        obstacle_density=args.obstacle_density,
        byzantine_fraction=0.0,
        max_steps=args.max_steps,
        seed=seed,
        protocol=BroadcastProtocol(),
        byzantine_agents={},
    )
    run_name = "mappo_%s_seed%d" % (args.run_tag, seed)
    logger = EpisodeLogger(run_name, "runs/")
    train(env, n_episodes=args.n_episodes, seed=seed, logger=logger, run_tag=args.run_tag)
    logger.close()
    print("seed=%d done — logs: runs/%s.csv  checkpoint: checkpoints/mappo_seed%d/ep%d/" % (
        seed, run_name, seed, args.n_episodes))
