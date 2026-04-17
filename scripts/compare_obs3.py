"""
Compare iPPO vs MAPPO under partial observability (obs_radius=3).

Hypothesis: reducing obs_radius from 7→3 forces agents to rely on comms,
widening the MAPPO advantage over iPPO beyond the ~1pp seen at obs_radius=7.

Config: n_seekers=8, grid_size=16, obs_radius=3, obstacle_density=0.15
       (all canonical except obs_radius)

Usage:
    python scripts/compare_obs3.py                          # 500 eps, seeds 42 43 44
    python scripts/compare_obs3.py --n_episodes 1000
    python scripts/compare_obs3.py --seeds 42 --n_episodes 200   # quick smoke test
    python scripts/compare_obs3.py --skip_ippo               # MAPPO only (if iPPO already done)
    python scripts/compare_obs3.py --skip_mappo              # iPPO only
"""
import sys
import argparse
import csv
import statistics
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.pursuit_env import ByzantinePursuitEnv
from comms.broadcast import BroadcastProtocol
from scripts.logger import EpisodeLogger

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",            type=int,   nargs="+", default=[42, 43, 44])
parser.add_argument("--n_episodes",       type=int,   default=500)
parser.add_argument("--n_seekers",        type=int,   default=8)
parser.add_argument("--grid_size",        type=int,   default=16)
parser.add_argument("--obs_radius",       type=int,   default=3)
parser.add_argument("--obstacle_density", type=float, default=0.15)
parser.add_argument("--max_steps",        type=int,   default=500)
parser.add_argument("--skip_ippo",        action="store_true", help="Skip iPPO training")
parser.add_argument("--skip_mappo",       action="store_true", help="Skip MAPPO training")
args = parser.parse_args()

TAG = "obs%d" % args.obs_radius  # e.g. obs3 — keeps runs separate from canonical exp checkpoints

print("=" * 60)
print("obs_radius comparison: iPPO vs MAPPO")
print("obs_radius=%d  n_seekers=%d  grid=%dx%d  obs_density=%.2f" % (
    args.obs_radius, args.n_seekers, args.grid_size, args.grid_size, args.obstacle_density))
print("n_episodes=%d  seeds=%s  tag=%s" % (args.n_episodes, args.seeds, TAG))
print("=" * 60)


def make_env(seed, protocol=None):
    kwargs = dict(
        n_seekers=args.n_seekers,
        grid_size=args.grid_size,
        obs_radius=args.obs_radius,
        obstacle_density=args.obstacle_density,
        byzantine_fraction=0.0,
        max_steps=args.max_steps,
        seed=seed,
    )
    if protocol is not None:
        kwargs["protocol"] = protocol
        kwargs["byzantine_agents"] = {}
    return ByzantinePursuitEnv(**kwargs)


# ── iPPO ──────────────────────────────────────────────────────────────────────
if not args.skip_ippo:
    from agents.ppo.ippo import train as ippo_train, ENT_COEF as IPPO_ENT
    print("\n[iPPO] ENT_COEF=%.3f" % IPPO_ENT)
    for seed in args.seeds:
        print("  seed=%d ..." % seed)
        env = make_env(seed)
        run_name = "ippo_%s_seed%d" % (TAG, seed)
        logger = EpisodeLogger(run_name, "runs/")
        ippo_train(env, n_episodes=args.n_episodes, seed=seed, logger=logger, run_tag=TAG)
        logger.close()
        print("  seed=%d done -> runs/%s.csv" % (seed, run_name))
else:
    print("\n[iPPO] skipped (--skip_ippo)")

# ── MAPPO ─────────────────────────────────────────────────────────────────────
if not args.skip_mappo:
    from agents.mappo.mappo import train as mappo_train, ENT_COEF as MAPPO_ENT
    print("\n[MAPPO] ENT_COEF=%.3f" % MAPPO_ENT)
    for seed in args.seeds:
        print("  seed=%d ..." % seed)
        env = make_env(seed, protocol=BroadcastProtocol())
        run_name = "mappo_%s_seed%d" % (TAG, seed)
        logger = EpisodeLogger(run_name, "runs/")
        mappo_train(env, n_episodes=args.n_episodes, seed=seed, logger=logger, run_tag=TAG)
        logger.close()
        print("  seed=%d done -> runs/%s.csv" % (seed, run_name))
else:
    print("\n[MAPPO] skipped (--skip_mappo)")


# ── Analysis ──────────────────────────────────────────────────────────────────
def summarise(algo, tag, seeds, n_eps):
    all_rows = []
    missing = []
    for seed in seeds:
        path = Path("runs/%s_%s_seed%d.csv" % (algo, tag, seed))
        if not path.exists():
            missing.append(seed)
            continue
        all_rows += list(csv.DictReader(open(path)))

    if missing:
        print("  WARNING: missing seeds %s for %s_%s" % (missing, algo, tag))
    if not all_rows:
        return None

    succ = [r for r in all_rows if r["capture_success"] == "True"]
    rate = len(succ) / len(all_rows) * 100
    times = [float(r["capture_time"]) for r in succ]
    mean_t = statistics.mean(times) if times else float("nan")

    # last 20% of episodes per seed as "converged" estimate
    last_rows = []
    window = max(1, n_eps // 5)
    for seed in seeds:
        path = Path("runs/%s_%s_seed%d.csv" % (algo, tag, seed))
        if path.exists():
            rows = list(csv.DictReader(open(path)))
            last_rows += rows[-window:]
    last_succ = [r for r in last_rows if r["capture_success"] == "True"]
    last_rate = len(last_succ) / len(last_rows) * 100 if last_rows else float("nan")
    last_times = [float(r["capture_time"]) for r in last_succ]
    last_t = statistics.mean(last_times) if last_times else float("nan")

    return dict(rate=rate, mean_t=mean_t, last_rate=last_rate, last_t=last_t, n=len(all_rows))


print("\n" + "=" * 60)
print("RESULTS — obs_radius=%d" % args.obs_radius)
print("=" * 60)

ippo_s  = summarise("ippo",  TAG, args.seeds, args.n_episodes)
mappo_s = summarise("mappo", TAG, args.seeds, args.n_episodes)

# Canonical baseline for comparison (obs_radius=7)
canon_ippo  = dict(rate=65.9, mean_t=18.5, last_rate=60.7, last_t=17.4)
canon_mappo = dict(rate=58.5, mean_t=15.2, last_rate=59.7, last_t=15.2)

def fmt(s):
    if s is None:
        return "N/A"
    return "rate=%.1f%%  time=%.1fs  last%d%%=%.1f%%" % (
        s["rate"], s["mean_t"], 100 // 5, s["last_rate"])

print("iPPO  obs%d:  %s" % (args.obs_radius, fmt(ippo_s)))
print("MAPPO obs%d:  %s" % (args.obs_radius, fmt(mappo_s)))
print()
print("iPPO  obs7 (canonical):  rate=%.1f%%  time=%.1fs  last20%%=%.1f%%" % (
    canon_ippo["rate"], canon_ippo["mean_t"], canon_ippo["last_rate"]))
print("MAPPO obs7 (canonical):  rate=%.1f%%  time=%.1fs  last20%%=%.1f%%" % (
    canon_mappo["rate"], canon_mappo["mean_t"], canon_mappo["last_rate"]))

if ippo_s and mappo_s:
    print()
    print("--- Gap at obs_radius=%d ---" % args.obs_radius)
    print("  MAPPO - iPPO (full):      %+.1fpp" % (mappo_s["rate"]  - ippo_s["rate"]))
    print("  MAPPO - iPPO (converged): %+.1fpp" % (mappo_s["last_rate"] - ippo_s["last_rate"]))
    print("  Capture time delta:       %+.1fs" % (mappo_s["last_t"] - ippo_s["last_t"]))
    print()
    print("--- obs3 vs obs7 effect ---")
    print("  iPPO:  obs7=%.1f%%  obs3=%.1f%%  delta=%+.1fpp" % (
        canon_ippo["last_rate"], ippo_s["last_rate"],
        ippo_s["last_rate"] - canon_ippo["last_rate"]))
    print("  MAPPO: obs7=%.1f%%  obs3=%.1f%%  delta=%+.1fpp" % (
        canon_mappo["last_rate"], mappo_s["last_rate"],
        mappo_s["last_rate"] - canon_mappo["last_rate"]))
    print()
    print("Hypothesis: MAPPO should degrade LESS than iPPO when obs drops 7->3")
    print("(comms compensate for smaller FoV; iPPO has no comms to fall back on)")
