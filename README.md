# Byzantine-Resilient Cooperative Pursuit

Empirical study of Byzantine robustness in cooperative multi-agent reinforcement learning (MARL). N=8 seekers on a 16×16 grid attempt to capture a hider under four Byzantine attack subtypes, two observability regimes, and four aggregation protocols.

**Algorithms:** Independent PPO (no-comms control) vs MAPPO (centralised training, decentralised execution)  
**Paper:** *Byzantine-Resilient Cooperative Pursuit: Degradation Analysis of Multi-Agent PPO Under Adversarial Communication* — SCC452, 2025/26

---

## Project Structure

```
marl-byzantine-pursuit/
├── env/
│   ├── pursuit_env.py          # ByzantinePursuitEnv (PettingZoo AEC)
│   └── schema.py               # OBS_DIM helper, observation layout constants
├── agents/
│   ├── mappo/mappo.py          # Shared actor + centralised critic, load_mappo()
│   ├── ppo/ippo.py             # Independent PPO, load_policies(), _zero_message_slots()
│   ├── byzantine/subtypes.py   # RandomNoiseByzantine, MisdirectionByzantine,
│   │                           #   SpoofingByzantine, SilentByzantine
│   └── greedy/greedy_agent.py  # Greedy heuristic (pilot validation only)
├── comms/
│   ├── interface.py            # BaseProtocol + EnvState dataclass
│   ├── broadcast.py            # BroadcastProtocol (unfiltered baseline)
│   ├── gossip.py               # GossipProtocol (fanout k=2)
│   ├── trimmed_mean.py         # TrimmedMeanProtocol (α=0.2, coordinate-wise)
│   └── reputation.py           # ReputationProtocol (online trust scoring)
├── experiments/
│   ├── configs/                # YAML sweep configs (exp1/exp2 × obs7/obs3)
│   └── results/                # Per-cell CSVs written by run_sweep.py (gitignored)
├── scripts/
│   ├── run_sweep.py            # Main eval script — loads ep1000 checkpoints, writes CSVs
│   ├── summarize_subtypes.py   # Aggregates CSVs → LaTeX-ready summary tables
│   ├── plot_results.py         # Generates all paper figures
│   ├── retrain_mappo.py        # Train MAPPO from scratch
│   └── retrain_ippo.py         # Train iPPO from scratch
├── tests/                      # pytest unit tests
├── analysis/paper/             # LaTeX drafts for paper sections
└── requirements.txt
```

---

## Setup

```bash
cd marl-byzantine-pursuit
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.10+ recommended.** All training runs on CPU; no GPU required.

---

## Reproducing Experiments

All experiments load frozen `ep1000` checkpoints — no retraining needed.

### Experiment 1 — Byzantine Degradation (sweeps f, fixed Broadcast)

```bash
# r=7, all 4 subtypes
for subtype in random misdirection spoof silent; do
  python scripts/run_sweep.py \
    --config experiments/configs/exp1_byzantine_degradation.yaml \
    --byzantine_subtype $subtype
done

# r=3, all 4 subtypes
for subtype in random misdirection spoof silent; do
  python scripts/run_sweep.py \
    --config experiments/configs/exp1_byzantine_degradation_obs3.yaml \
    --byzantine_subtype $subtype
done
```

### Experiment 2 — Protocol Comparison (fixed f=0.25, sweeps protocol)

```bash
# r=7
for subtype in random misdirection spoof silent; do
  python scripts/run_sweep.py \
    --config experiments/configs/exp2_protocol_comparison.yaml \
    --byzantine_subtype $subtype
done

# r=3
for subtype in random misdirection spoof silent; do
  python scripts/run_sweep.py \
    --config experiments/configs/exp2_protocol_comparison_obs3.yaml \
    --byzantine_subtype $subtype
done
```

### iPPO Control (run once per regime)

```bash
python scripts/run_sweep.py \
  --config experiments/configs/exp1_byzantine_degradation.yaml \
  --algo ippo

python scripts/run_sweep.py \
  --config experiments/configs/exp1_byzantine_degradation_obs3.yaml \
  --algo ippo
```

**Total:** 240 MAPPO + 60 iPPO condition×seed cells × 100 episodes = 30,000 episodes.

### Dry-run (print conditions without executing)

```bash
python scripts/run_sweep.py \
  --config experiments/configs/exp1_byzantine_degradation.yaml \
  --dry-run
```

---

## Viewing Results

### Summary tables (LaTeX-ready)

```bash
python scripts/summarize_subtypes.py
```

Prints mean±std capture rates for all subtypes, both regimes, both algorithms.

### Generate all figures

```bash
python scripts/plot_results.py
```

Writes to `experiments/results/figures/`:

| File | Content |
|------|---------|
| `fig1_degradation.png` | Degradation curves (obs3 vs obs7) |
| `fig2_protocol.png` | Protocol comparison bar chart |
| `fig2a_protocol_obs3.png` | Protocol bars, r=3 |
| `fig2b_protocol_obs7.png` | Protocol bars, r=7 |
| `fig6_subtype_heatmap.png` | Subtype×protocol heatmap |
| `fig5_overview.png` | 6-panel full-project overview |

---

## Byzantine Subtypes

| Subtype | Flag | Description |
|---------|------|-------------|
| Random Noise | `random` | Uniform random coordinate each step |
| Misdirection | `misdirection` | Reflects hider position through agent (omniscient) |
| Spoofing | `spoof` | Honest content, forged `sender_id` (Sybil-style) |
| Silent | `silent` | Sends no message; leaves slot at sentinel −1.0 |

Spoofing is a **null attack** against slot-indexed actors — the policy reads messages by slot index, not sender ID, so identity forgery has no effect.

---

## Aggregation Protocols

| Protocol | Description | Best against |
|----------|-------------|--------------|
| `none` | Message slots zeroed (iPPO control) | — |
| `broadcast` | All-to-all, no filtering | Misdirection |
| `gossip` | Random fanout k=2 per step | — |
| `trimmed_mean` | Coordinate-wise trim α=0.2 | Random Noise, Spoofing |
| `reputation` | Online per-sender trust scoring | Misdirection, Silent |

---

## Key Results (ep1000 checkpoints, n=300 per cell)

### Honest baseline (f=0)

| Regime | MAPPO eval | iPPO eval | Cooperative gain |
|--------|-----------|-----------|-----------------|
| r=7 | 68.3±2.3% | 52.0±7.5% | +16.3 pp |
| r=3 | 97.3±3.8% | 47.0±1.0% | +50.3 pp |

### Worst-case degradation at f=0.5 (Broadcast)

| Subtype | r=7 Δf | r=3 Δf |
|---------|--------|--------|
| Misdirection | −19.6 pp | −33.6 pp |
| Silent | −9.4 pp | −48.3 pp |
| Random Noise | −9.3 pp | −10.6 pp |
| Spoofing | ≈ 0 pp | ≈ 0 pp |

MAPPO remains **net-positive** vs iPPO across all subtypes and all f ∈ [0, 0.5].

---

## Training From Scratch

Only needed if checkpoints are unavailable.

```bash
# MAPPO, r=7 (wide FoV)
python scripts/retrain_mappo.py --seeds 42 43 44 --run_tag exp

# MAPPO, r=3 (narrow FoV)
python scripts/retrain_mappo.py --seeds 42 43 44 --obs_radius 3 --run_tag obs3

# iPPO, r=7
python scripts/retrain_ippo.py --seeds 42 43 44 --run_tag exp

# iPPO, r=3
python scripts/retrain_ippo.py --seeds 42 43 44 --obs_radius 3 --run_tag obs3
```

Checkpoints saved every 50 episodes to `checkpoints/{algo}_{tag}_seed{s}/ep{e}/`.

---

## Tests

```bash
pytest tests/ -v
```

| File | Covers |
|------|--------|
| `tests/test_env.py` | Environment reset, step, termination |
| `tests/test_byzantine.py` | All 4 corruption operators |
| `tests/test_comms.py` | Protocol send/receive contracts |
| `tests/test_mappo.py` | Actor forward pass, checkpoint load |

---

## Observation Space

```
oi = [self_x, self_y, hider_x, hider_y | obstacle patch | peer messages]
      ←──── 4 ────→                       ←── (2r+1)² ──→  ←── 2(N−1) ──→
```

| Regime | r | Obs dim D |
|--------|---|-----------|
| Wide FoV | 7 | 4 + 225 + 14 = **243** |
| Narrow FoV | 3 | 4 + 49 + 14 = **67** |

Hider position fields are set to sentinel −1.0 when outside FoV or occluded by an obstacle.

---

## Environment Configuration

| Parameter | Value |
|-----------|-------|
| Grid size | 16×16 |
| Seekers (N) | 8 |
| Obstacle density (ρ) | 0.15 (r=7) / 0.25 (r=3) |
| Max steps | 500 |
| Training episodes | 1000 |
| Seeds | {42, 43, 44} |
| Hider policy | Uniform random over passable neighbours |
