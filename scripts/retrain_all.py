"""
Retrain iPPO then MAPPO in one command — canonical experiment config.

Runs iPPO for all seeds, then MAPPO for all seeds, then prints a side-by-side
comparison summary.  All defaults match the canonical paper config exactly.

Usage:
    python scripts/retrain_all.py                              # both, 3 seeds, 500 eps
    python scripts/retrain_all.py --n_episodes 1000            # longer run
    python scripts/retrain_all.py --seeds 42                   # single seed (fast check)
    python scripts/retrain_all.py --skip_ippo                  # MAPPO only
    python scripts/retrain_all.py --skip_mappo                 # iPPO only
    python scripts/retrain_all.py --run_tag v2                 # custom checkpoint tag
"""
import sys
import argparse
import csv
import statistics
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.pursuit_env import ByzantinePursuitEnv
from comms.broadcast import BroadcastProtocol
from scripts.logger import EpisodeLogger

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seeds",            type=int,   nargs="+", default=[42, 43, 44])
parser.add_argument("--n_episodes",       type=int,   default=500)
parser.add_argument("--run_tag",          type=str,   default="exp")
parser.add_argument("--n_seekers",        type=int,   default=8)
parser.add_argument("--grid_size",        type=int,   default=16)
parser.add_argument("--obs_radius",       type=int,   default=7,
                    help="FoV half-side. Pass 0 for full observability.")
parser.add_argument("--obstacle_density", type=float, default=0.15)
parser.add_argument("--max_steps",        type=int,   default=500)
parser.add_argument("--skip_ippo",        action="store_true")
parser.add_argument("--skip_mappo",       action="store_true")
args = parser.parse_args()

obs_radius = None if args.obs_radius == 0 else args.obs_radius

print("=" * 60)
print("retrain_all — iPPO + MAPPO")
print("n_seekers=%d  grid=%dx%d  obs_radius=%s  density=%.2f" % (
    args.n_seekers, args.grid_size, args.grid_size, obs_radius, args.obstacle_density))
print("n_episodes=%d  seeds=%s  run_tag=%s" % (
    args.n_episodes, args.seeds, args.run_tag))
print("=" * 60)

wall_start = time.time()


# ---------------------------------------------------------------------------
# iPPO
# ---------------------------------------------------------------------------
if not args.skip_ippo:
    from agents.ppo.ippo import train as ippo_train, ENT_COEF as IPPO_ENT
    from agents.reward import reward_version, DISTANCE_SHAPING

    print("\n[iPPO] reward_version=%s  DS=%.1f  ENT_COEF=%.3f" % (
        reward_version, DISTANCE_SHAPING, IPPO_ENT))

    t0 = time.time()
    for seed in args.seeds:
        print("  seed=%d ..." % seed, flush=True)
        env = ByzantinePursuitEnv(
            n_seekers=args.n_seekers,
            grid_size=args.grid_size,
            obs_radius=obs_radius,
            obstacle_density=args.obstacle_density,
            byzantine_fraction=0.0,
            max_steps=args.max_steps,
            seed=seed,
        )
        run_name = "ippo_%s_seed%d" % (args.run_tag, seed)
        logger = EpisodeLogger(run_name, "runs/")
        ippo_train(env, n_episodes=args.n_episodes, seed=seed,
                   logger=logger, run_tag=args.run_tag)
        logger.close()
        print("    done -> runs/%s.csv" % run_name, flush=True)

    print("[iPPO] finished in %.1fs" % (time.time() - t0))
else:
    print("\n[iPPO] skipped (--skip_ippo)")


# ---------------------------------------------------------------------------
# MAPPO
# ---------------------------------------------------------------------------
if not args.skip_mappo:
    from agents.mappo.mappo import train as mappo_train, ENT_COEF as MAPPO_ENT

    print("\n[MAPPO] ENT_COEF=%.3f" % MAPPO_ENT, flush=True)

    t0 = time.time()
    for seed in args.seeds:
        print("  seed=%d ..." % seed, flush=True)
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
        mappo_train(env, n_episodes=args.n_episodes, seed=seed,
                    logger=logger, run_tag=args.run_tag)
        logger.close()
        print("    done -> runs/%s.csv" % run_name, flush=True)

    print("[MAPPO] finished in %.1fs" % (time.time() - t0))
else:
    print("\n[MAPPO] skipped (--skip_mappo)")


# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
def _summarise(algo, tag, seeds, n_eps):
    rows = []
    for seed in seeds:
        p = Path("runs/%s_%s_seed%d.csv" % (algo, tag, seed))
        if p.exists():
            rows += list(csv.DictReader(open(p)))
    if not rows:
        return None
    succ = [r for r in rows if r["capture_success"] == "True"]
    rate = len(succ) / len(rows) * 100
    times = [float(r["capture_time"]) for r in succ]
    mean_t = statistics.mean(times) if times else float("nan")
    window = max(1, n_eps // 5)
    last = []
    for seed in seeds:
        p = Path("runs/%s_%s_seed%d.csv" % (algo, tag, seed))
        if p.exists():
            last += list(csv.DictReader(open(p)))[-window:]
    last_succ = [r for r in last if r["capture_success"] == "True"]
    last_rate = len(last_succ) / len(last) * 100 if last else float("nan")
    return dict(rate=rate, mean_t=mean_t, last_rate=last_rate, n=len(rows))

print("\n" + "=" * 60)
print("RESULTS — %s  (tag=%s)" % (
    "obs_radius=%s" % obs_radius, args.run_tag))
print("=" * 60)

ippo_s  = _summarise("ippo",  args.run_tag, args.seeds, args.n_episodes)
mappo_s = _summarise("mappo", args.run_tag, args.seeds, args.n_episodes)

def _fmt(s):
    if s is None:
        return "not run"
    return "rate=%.1f%%  time=%.1fs  last%d%%=%.1f%%  (n=%d)" % (
        s["rate"], s["mean_t"], 100 // 5, s["last_rate"], s["n"])

print("iPPO : %s" % _fmt(ippo_s))
print("MAPPO: %s" % _fmt(mappo_s))

if ippo_s and mappo_s:
    print()
    print("MAPPO - iPPO (full):      %+.1fpp" % (mappo_s["rate"]  - ippo_s["rate"]))
    print("MAPPO - iPPO (converged): %+.1fpp" % (mappo_s["last_rate"] - ippo_s["last_rate"]))
    print("Capture time delta:       %+.1fs"  % (mappo_s["mean_t"] - ippo_s["mean_t"]))

print()
print("Checkpoints:")
for seed in args.seeds:
    print("  iPPO  seed%d: checkpoints/ippo_%s_seed%d/ep%d/" % (
        seed, args.run_tag, seed, args.n_episodes))
for seed in args.seeds:
    print("  MAPPO seed%d: checkpoints/mappo_%s_seed%d/ep%d/" % (
        seed, args.run_tag, seed, args.n_episodes))

print()
print("Total wall time: %.1fs" % (time.time() - wall_start))
