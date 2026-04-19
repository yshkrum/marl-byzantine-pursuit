# Phase 2 Tickets — Role D: Experiment Runner
*Paper ownership: §4 Results (experiment execution)*
*Contact: Divya*

---

> **Status note:** EXP-02/03/04 configs and sweep skeleton are merged (`f4c1b0f`).
> The two experiment YAML configs are correct. The sweep runner parses and expands
> conditions correctly. The remaining work is wiring the training loop into the runner
> and executing the actual experiments once Role B's retraining (RL-06, RL-07) is done.

---

### EXP-05 · Wire training loop into run_sweep.py
**Priority:** Critical · **Blocks:** All experiments · **Deadline:** ASAP

**Background**

`scripts/run_sweep.py` line 42 raises `NotImplementedError`. The config parsing and
condition expansion (`build_conditions()`) work correctly — only `run_experiment()` needs
to be filled in. This is the single change that unblocks all experiment execution.

There is also a Python 3.8 compatibility bug: `list[dict]` type annotations require
Python 3.9+. The team is on Python 3.8.

**Acceptance criteria**

- [ ] Fix type annotations: `list[dict]` → `List[dict]` (import `from typing import List, Dict`)
- [ ] `run_experiment()` instantiates the correct env from `config["env"]` fields
- [ ] Wires in the correct algorithm: `config["training"]["algorithm"]` selects `ippo.train`
  or `mappo.train`
- [ ] Wires in the correct protocol: `config["comms"]["protocol"]` selects `NoneProtocol`,
  `BroadcastProtocol`, etc.
- [ ] Wires in Byzantine agents: `condition["byzantine_fraction"]` assigns the first
  `floor(n_seekers * f)` seekers as `RandomNoiseByzantine` (Exp1 uses random noise subtype)
- [ ] Saves CSV to `config["logging"]["output_dir"]` with filename
  `{experiment_name}_f{byzantine_fraction}_p{protocol}_s{seed}.csv`
- [ ] Dry-run (`--dry-run`) still prints without executing (already works, preserve this)

**Implementation sketch**

```python
from typing import List, Dict
import math
from pathlib import Path
from env.pursuit_env import ByzantinePursuitEnv
from comms.interface import NoneProtocol
from comms.broadcast import BroadcastProtocol
from agents.ppo.ippo import train as ippo_train
from agents.mappo.mappo import train as mappo_train
from agents.byzantine.subtypes import RandomNoiseByzantine
from scripts.logger import EpisodeLogger

PROTOCOL_MAP = {
    "none":       NoneProtocol,
    "broadcast":  BroadcastProtocol,
    # "gossip":   GossipProtocol,       # add once BYZ-06 done
    # "trimmed_mean": TrimmedMeanProtocol,  # add once BYZ-07 done
}

def run_experiment(config: dict, condition: dict, seed: int, dry_run: bool = False):
    run_name = (
        f"{config['experiment']['name']}"
        f"_f{condition.get('byzantine_fraction', 0)}"
        f"_p{condition.get('protocol', 'none')}"
        f"_s{seed}"
    )
    print(f"  {'[DRY RUN] ' if dry_run else ''}Launching: {run_name}")
    if dry_run:
        return

    ecfg = config["env"]
    n_seekers = ecfg["n_seekers"]    # canonical: 8
    byz_frac  = condition.get("byzantine_fraction", 0.0)
    n_byz     = math.floor(n_seekers * byz_frac)
    # With N=8, f in {0.0,0.125,0.25,0.375,0.5} → n_byz in {0,1,2,3,4} — one per step, no duplicates
    protocol_key = condition.get("protocol", "none")

    protocol = PROTOCOL_MAP[protocol_key]()

    # Assign first n_byz seekers as Byzantine (deterministic, matches §3.3)
    byz_agents = {
        "seeker_%d" % i: RandomNoiseByzantine(grid_size=ecfg["grid_size"], seed=seed + i)
        for i in range(n_byz)
    }

    env = ByzantinePursuitEnv(
        n_seekers=n_seekers,
        grid_size=ecfg["grid_size"],       # canonical: 16
        obs_radius=ecfg.get("obs_radius"), # canonical: 7  (OBS_DIM=243)
        obstacle_density=ecfg["obstacle_density"],  # canonical: 0.15
        byzantine_fraction=byz_frac,
        max_steps=ecfg["max_steps"],
        seed=seed,
        protocol=protocol,
        byzantine_agents=byz_agents,
    )

    out_dir = Path(config["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(run_name, str(out_dir))

    algo = config["training"]["algorithm"]
    n_ep = config["training"]["n_episodes"]

    if algo == "mappo":
        mappo_train(env, n_episodes=n_ep, seed=seed, logger=logger)
    elif algo == "ippo":
        ippo_train(env, n_episodes=n_ep, seed=seed, logger=logger)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    logger.close()
```

---

### EXP-06 · Execute Experiment 1 — Byzantine Degradation Curve
**Priority:** High · **Blocks:** Paper §4.1 · **Deadline:** After RL-06, RL-07, EXP-05 done

**Background**

Experiment 1 measures how capture performance degrades as Byzantine fraction f increases,
using the broadcast protocol and RandomNoiseByzantine agents. This is the primary result
of the paper.

**Config:** `experiments/configs/exp1_byzantine_degradation.yaml`
- **N=8, 16×16, obs_radius=7, obstacle_density=0.15** (PettingZoo benchmark defaults)
- `byzantine_fraction` ∈ {0.0, 0.125, 0.25, 0.375, 0.5} → {0,1,2,3,4} Byzantine agents
- 500 episodes per run, 5 seeds → **25 total runs**
- Expected runtime: ~4–5 hours total on CPU (run seeds in parallel if possible)

**How to run**

```bash
python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml
```

**Acceptance criteria**

- [ ] 20 CSV files in `experiments/results/exp1/`
- [ ] Capture rate at f=0.0 matches Role B's iPPO/MAPPO standalone training (sanity check)
- [ ] Monotonic degradation trend: capture rate decreases as f increases (or capture time increases)
- [ ] At f=0.5 (worst case): capture rate ≥15% lower than f=0.0, OR mean capture time ≥15% higher
  (this is the BYZ-04 validation criterion from `validate_byzantine.py`)
- [ ] Pass results to Role E for plotting

---

### EXP-07 · Execute Experiment 2 — Protocol Comparison
**Priority:** Medium · **Blocks:** Paper §4.2 · **Deadline:** After EXP-06, BYZ-06, BYZ-07

**Background**

Experiment 2 asks: at fixed f=0.33, which communication protocol best maintains capture
performance? This is the protocol comparison section of the paper.

**Config:** `experiments/configs/exp2_protocol_comparison.yaml`
- **N=8, 16×16, obs_radius=7, obstacle_density=0.15** (same canonical config as Exp1)
- Fixed `byzantine_fraction=0.25` (= 2/8 Byzantine — clean label, no ambiguity)
- `protocol` ∈ {none, broadcast, gossip, trimmed_mean} (reputation optional — check BYZ-08)
- 500 episodes, 5 seeds, 4 protocols → **20 total runs** (25 if reputation included)

**Dependency:** BYZ-06 (gossip) and BYZ-07 (trimmed_mean) must be merged first.

**How to run**

```bash
python scripts/run_sweep.py --config experiments/configs/exp2_protocol_comparison.yaml
```

**Acceptance criteria**

- [ ] CSV files in `experiments/results/exp2/`
- [ ] `none` protocol should match or be close to iPPO standalone result (no comms → same)
- [ ] `trimmed_mean` should outperform `broadcast` at f=0.33 (Byzantine robustness point)
- [ ] Pass results to Role E for plotting

---

### EXP-08 · Run Byzantine sanity check (can do NOW — no retraining needed)
**Priority:** Medium · **No blockers**

**Background**

`scripts/validate_byzantine.py` uses a message-aware greedy agent (not RL), so it can
run immediately without waiting for retraining. It verifies that Byzantine agents actually
degrade capture performance — the go/no-go gate before committing compute to Exp1.

**How to run**

```bash
python scripts/validate_byzantine.py
```

Output: `experiments/results/validation/byzantine_sanity_check.txt`

**Acceptance criteria**

- [ ] Script completes without errors
- [ ] At f=0.5: capture time ≥15% higher than f=0.0, OR capture rate ≥10pp lower
- [ ] Report pass/fail result to the team Slack/group chat
